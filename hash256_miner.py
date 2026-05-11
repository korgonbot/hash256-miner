#!/usr/bin/env python3
"""
hash256_miner.py — Fleet OS miner for $HASH (hash256.org)

Protocol: keccak256 PoW on Ethereum mainnet
Contract: 0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc
Chain ID: 1

Puzzle:
    challenge = keccak256(abi.encodePacked(chainId, contract, miner, epoch))
    valid     = keccak256(abi.encodePacked(challenge, nonce)) < currentDifficulty

Rules:
    - Challenges are address-bound (can't steal from mempool)
    - Epoch rotates every 100 blocks (~20 min). Pre-computed solutions expire.
    - Each (miner, nonce, epoch) tuple mints once. No replay.
    - 10 mints per block hard cap.
    - Base reward: 100 HASH, halves every 100,000 mints.

Usage:
    python hash256_miner.py                  # mine with all wallets in WALLET_FILE
    python hash256_miner.py --dry-run        # compute challenges, do not submit
    python hash256_miner.py --wallet 0x...   # mine with single address (key from env)
    python hash256_miner.py --workers 8      # override CPU worker count

Requirements:
    pip install web3 eth-abi eth-hash pysha3
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ─── dependency check ────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    for pkg in ["web3", "eth_abi", "eth_hash"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[FATAL] Missing packages: {', '.join(missing)}")
        print("  Install: pip install web3 eth-abi eth-hash pysha3")
        sys.exit(1)

_check_deps()

from eth_abi.packed import encode_packed          # noqa: E402
from eth_hash.auto import keccak                  # noqa: E402
from web3 import Web3                             # noqa: E402

# ─── constants ────────────────────────────────────────────────────────────────

CONTRACT_ADDRESS  = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc"
CHAIN_ID          = 1            # Ethereum mainnet
EPOCH_BLOCKS      = 100          # epoch = block.number // 100
GAS_BUFFER        = 1.20         # +20% on estimateGas (Fleet OS rule)
RETRY_BASE        = 1.0          # seconds
RETRY_MULT        = 2.0
RETRY_MAX         = 5
NONCE_BATCH       = 50_000       # nonces per worker batch before checking stop flag

# Minimal ABI — update once contract source is verified on Etherscan
CONTRACT_ABI = [
    {
        "name": "mine",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "nonce", "type": "uint256"}],
        "outputs": [],
    },
    {
        "name": "currentDifficulty",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "currentEpoch",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    # Some contracts expose challenge() as a view; try it and fall back to local compute
    {
        "name": "challenge",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "miner", "type": "address"}],
        "outputs": [{"name": "", "type": "bytes32"}],
    },
    {
        "name": "totalMints",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "miningOpen",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "bool"}],
    },
]

# ─── logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hash256")

# ─── env validation (Fleet OS rule 7) ─────────────────────────────────────────

REQUIRED_VARS = ["RPC_URL", "WALLET_FILE"]
OPTIONAL_VARS = {
    "LOG_LEVEL":     "INFO",
    "MAX_WORKERS":   str(max(1, mp.cpu_count() - 1)),
    "PROXY_FILE":    "",
    "GAS_PRICE_CAP": "0",    # gwei; 0 = no cap
    "DRY_RUN":       "false",
}

def validate_env() -> dict:
    """Validate .env at startup. Fail fast listing ALL missing vars."""
    from dotenv import load_dotenv
    load_dotenv()

    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        log.error("Missing required environment variables:")
        for v in missing:
            log.error(f"  {v}")
        log.error("Copy .env.example to .env and fill in the values.")
        sys.exit(1)

    cfg = {k: os.getenv(k, default) for k, default in OPTIONAL_VARS.items()}
    cfg.update({k: os.environ[k] for k in REQUIRED_VARS})

    logging.getLogger().setLevel(cfg["LOG_LEVEL"].upper())
    return cfg


# ─── state management (Fleet OS rule 4, 12) ───────────────────────────────────

STATE_DIR  = Path("state")
STATE_FILE = STATE_DIR / "hash256_state.json"

def load_state() -> dict:
    STATE_DIR.mkdir(exist_ok=True)
    if not STATE_FILE.exists():
        return {"minted": {}, "last_epoch": None, "total_minted": 0}
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("State file corrupt — starting fresh")
        return {"minted": {}, "last_epoch": None, "total_minted": 0}

def save_state(state: dict) -> None:
    """Atomic write: .tmp → rename (Fleet OS rule 12)."""
    STATE_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(STATE_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, str(STATE_FILE))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def already_minted(state: dict, wallet: str, epoch: int) -> bool:
    return str(epoch) in state["minted"].get(wallet.lower(), {})

def record_mint(state: dict, wallet: str, epoch: int, nonce: int, tx_hash: str) -> None:
    w = wallet.lower()
    state["minted"].setdefault(w, {})[str(epoch)] = {
        "nonce": nonce,
        "tx": tx_hash,
        "ts": int(time.time()),
    }
    state["total_minted"] += 1
    state["last_epoch"] = epoch
    save_state(state)


# ─── wallet loading ───────────────────────────────────────────────────────────

def load_wallets(wallet_file: str) -> list[dict]:
    """Load wallets.json [{address, private_key}, ...]"""
    path = Path(wallet_file)
    if not path.exists():
        log.error(f"Wallet file not found: {path}")
        sys.exit(1)
    try:
        wallets = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        log.error(f"wallets.json parse error: {e}")
        sys.exit(1)
    if not isinstance(wallets, list) or not wallets:
        log.error("wallets.json must be a non-empty list")
        sys.exit(1)
    log.info(f"Loaded {len(wallets)} wallets")
    return wallets


# ─── error categorization (Fleet OS rule 14) ──────────────────────────────────

ERROR_CATEGORIES = {
    "NETWORK":  ["timeout", "connection", "unreachable", "dns", "eof", "reset"],
    "GAS":      ["insufficient funds", "gas", "intrinsic"],
    "NONCE":    ["nonce", "replacement", "underpriced"],
    "REVERT":   ["revert", "require", "execution reverted"],
}

def categorize_error(err: Exception) -> str:
    msg = str(err).lower()
    for cat, keywords in ERROR_CATEGORIES.items():
        if any(kw in msg for kw in keywords):
            return cat
    return "UNKNOWN"


# ─── retry with exponential backoff (Fleet OS rule 13) ────────────────────────

def retry(fn, *args, label="", **kwargs):
    delay = RETRY_BASE
    for attempt in range(RETRY_MAX):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            cat = categorize_error(e)
            if cat == "REVERT":
                raise   # no retry on revert
            jitter = random.uniform(0, 0.5)
            log.warning(f"[{cat}] {label} attempt {attempt+1}/{RETRY_MAX}: {e} — retry in {delay:.1f}s")
            time.sleep(delay + jitter)
            delay *= RETRY_MULT
    raise RuntimeError(f"All retries exhausted for: {label}")


# ─── PoW challenge computation ────────────────────────────────────────────────

def compute_challenge(chain_id: int, contract: str, miner: str, epoch: int) -> bytes:
    """
    challenge = keccak256(abi.encodePacked(chainId, contract, miner, epoch))
    Types: uint256, address, address, uint256
    """
    packed = encode_packed(
        ["uint256", "address", "address", "uint256"],
        [chain_id, Web3.to_checksum_address(contract), Web3.to_checksum_address(miner), epoch],
    )
    return keccak(packed)


def check_solution(challenge: bytes, nonce: int, difficulty: int) -> bool:
    """
    valid = keccak256(abi.encodePacked(challenge, nonce)) < currentDifficulty
    Types: bytes32, uint256
    """
    packed = encode_packed(["bytes32", "uint256"], [challenge, nonce])
    h = keccak(packed)
    return int.from_bytes(h, "big") < difficulty


# ─── worker process: grinds nonces in a given range ───────────────────────────

def _grind_worker(
    challenge: bytes,
    difficulty: int,
    start_nonce: int,
    stop_flag: mp.Value,
    result_queue: mp.Queue,
    worker_id: int,
) -> None:
    """
    Runs in a subprocess. Increments nonce from start_nonce upward,
    checking every NONCE_BATCH iterations whether stop_flag is set.
    """
    nonce = start_nonce
    while True:
        if stop_flag.value:
            return
        # Batch of nonces before checking stop flag
        end = nonce + NONCE_BATCH
        while nonce < end:
            if check_solution(challenge, nonce, difficulty):
                result_queue.put(nonce)
                stop_flag.value = 1
                return
            nonce += 1


def grind(challenge: bytes, difficulty: int, num_workers: int) -> int:
    """
    Launch num_workers processes, each starting at a different nonce offset.
    Returns the first valid nonce found.
    """
    stop_flag   = mp.Value("i", 0)
    result_queue = mp.Queue()

    # Each worker starts at a random 64-bit offset to avoid collisions across wallets
    base = random.getrandbits(64)
    stride = (2**64) // num_workers

    workers = []
    for i in range(num_workers):
        start = (base + i * stride) % (2**64)
        p = mp.Process(
            target=_grind_worker,
            args=(challenge, difficulty, start, stop_flag, result_queue, i),
            daemon=True,
        )
        p.start()
        workers.append(p)

    nonce = result_queue.get()  # blocks until any worker finds a solution

    # Signal all workers to stop
    stop_flag.value = 1
    for p in workers:
        p.join(timeout=2)
        if p.is_alive():
            p.kill()

    return nonce


# ─── on-chain reads ───────────────────────────────────────────────────────────

def get_difficulty(contract) -> int:
    return retry(contract.functions.currentDifficulty().call, label="currentDifficulty")

def get_epoch_from_contract(contract) -> Optional[int]:
    try:
        return contract.functions.currentEpoch().call()
    except Exception:
        return None

def get_epoch_from_block(w3: Web3) -> int:
    block = retry(w3.eth.get_block, "latest", label="eth_blockNumber")
    return block["number"] // EPOCH_BLOCKS

def is_mining_open(contract) -> bool:
    """Check if mining gate is open. Falls back to True if function doesn't exist."""
    try:
        return contract.functions.miningOpen().call()
    except Exception:
        return True   # assume open if we can't read the flag

def get_challenge_from_contract(contract, miner: str) -> Optional[bytes]:
    """Some contracts expose challenge(address) as a view. Fall back to local compute."""
    try:
        result = contract.functions.challenge(Web3.to_checksum_address(miner)).call()
        return bytes(result)
    except Exception:
        return None


# ─── transaction submission ───────────────────────────────────────────────────

def submit_mine(w3: Web3, contract, wallet: dict, nonce: int, gas_price_cap_gwei: int, local_nonce: int) -> str:
    """
    Build and send the mine(nonce) transaction.
    Returns tx hash string on success.
    Fleet OS: gas +20%, local nonce tracking, never use pending RPC nonce for parallel.
    """
    address = Web3.to_checksum_address(wallet["address"])
    private_key = wallet["private_key"]

    # Gas price (use EIP-1559 if supported)
    try:
        base_fee = w3.eth.get_block("latest")["baseFeePerGas"]
        priority = w3.to_wei(1, "gwei")   # 1 gwei tip
        max_fee  = int(base_fee * 2 + priority)
        if gas_price_cap_gwei and max_fee > w3.to_wei(gas_price_cap_gwei, "gwei"):
            log.warning(f"Gas price {w3.from_wei(max_fee,'gwei'):.1f} gwei exceeds cap {gas_price_cap_gwei} gwei — skipping")
            return ""
        gas_params = {
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": priority,
        }
    except Exception:
        # Legacy gas price
        gas_price = w3.eth.gas_price
        gas_params = {"gasPrice": gas_price}

    # Estimate gas with +20% buffer (Fleet OS rule)
    try:
        tx_for_estimate = contract.functions.mine(nonce).build_transaction({
            "from": address,
            "nonce": local_nonce,
            "chainId": CHAIN_ID,
            **gas_params,
        })
        estimated = w3.eth.estimate_gas(tx_for_estimate)
        gas_limit = int(estimated * GAS_BUFFER)
    except Exception as e:
        cat = categorize_error(e)
        if cat == "REVERT":
            log.error(f"[REVERT] Gas estimate reverted for {address[:8]}: {e}")
            raise
        log.warning(f"[{cat}] Gas estimate failed, using 200_000 fallback: {e}")
        gas_limit = 200_000

    tx = contract.functions.mine(nonce).build_transaction({
        "from":    address,
        "nonce":   local_nonce,
        "chainId": CHAIN_ID,
        "gas":     gas_limit,
        **gas_params,
    })

    signed  = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt["status"] == 0:
        raise RuntimeError(f"TX reverted: {tx_hash.hex()}")

    return tx_hash.hex()


# ─── single wallet mining loop ────────────────────────────────────────────────

def mine_wallet(wallet: dict, cfg: dict, state: dict, dry_run: bool, num_workers: int) -> None:
    address = Web3.to_checksum_address(wallet["address"])
    w3 = Web3(Web3.HTTPProvider(cfg["RPC_URL"]))

    if not w3.is_connected():
        log.error(f"[{address[:8]}] RPC not connected: {cfg['RPC_URL']}")
        return

    contract = w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=CONTRACT_ABI,
    )

    # Check mining gate
    if not is_mining_open(contract):
        log.warning(
            f"[{address[:8]}] Mining is not open yet. "
            "Genesis must complete and pool must be seeded first. "
            "Run again after genesis closes."
        )
        return

    # Local nonce tracking (Fleet OS rule 10 — never use pending RPC nonce)
    local_nonce = w3.eth.get_transaction_count(address, "latest")

    gas_price_cap = int(cfg["GAS_PRICE_CAP"])
    last_epoch    = None

    log.info(f"[{address[:8]}] Starting mining loop — {num_workers} workers")

    while True:
        try:
            # ── 1. Get current epoch ──────────────────────────────────────
            epoch = get_epoch_from_contract(contract) or get_epoch_from_block(w3)

            if already_minted(state, address, epoch):
                time_to_next = _seconds_to_next_epoch(w3, epoch)
                log.info(f"[{address[:8]}] Epoch {epoch} already minted. Next epoch in ~{time_to_next}s")
                time.sleep(min(time_to_next, 30))
                continue

            if epoch != last_epoch:
                log.info(f"[{address[:8]}] New epoch: {epoch}")
                last_epoch = epoch

            # ── 2. Get difficulty ─────────────────────────────────────────
            difficulty = get_difficulty(contract)
            log.info(f"[{address[:8]}] Epoch {epoch} | Difficulty: {difficulty:#066x}")

            # ── 3. Compute challenge ──────────────────────────────────────
            challenge = get_challenge_from_contract(contract, address)
            if challenge is None:
                challenge = compute_challenge(CHAIN_ID, CONTRACT_ADDRESS, address, epoch)
            log.debug(f"[{address[:8]}] Challenge: {challenge.hex()}")

            # ── 4. Grind nonces ───────────────────────────────────────────
            log.info(f"[{address[:8]}] Grinding nonces...")
            t0    = time.time()
            nonce = grind(challenge, difficulty, num_workers)
            dt    = time.time() - t0
            log.info(f"[{address[:8]}] Found nonce {nonce} in {dt:.1f}s")

            if dry_run:
                log.info(f"[{address[:8]}] DRY RUN — would submit nonce {nonce}")
                record_mint(state, address, epoch, nonce, "dry-run")
                continue

            # ── 5. Verify epoch hasn't rotated while we were grinding ─────
            current_epoch_now = get_epoch_from_contract(contract) or get_epoch_from_block(w3)
            if current_epoch_now != epoch:
                log.warning(f"[{address[:8]}] Epoch rotated during grind ({epoch} → {current_epoch_now}). Restarting.")
                continue

            # ── 6. Submit ─────────────────────────────────────────────────
            log.info(f"[{address[:8]}] Submitting mine(nonce={nonce}) ...")
            tx_hash = retry(
                submit_mine,
                w3, contract, wallet, nonce, gas_price_cap, local_nonce,
                label=f"mine {address[:8]}",
            )
            local_nonce += 1  # local nonce tracking (Fleet OS rule 10)
            log.info(f"[{address[:8]}] ✓ Minted! TX: {tx_hash}")
            record_mint(state, address, epoch, nonce, tx_hash)

        except KeyboardInterrupt:
            log.info(f"[{address[:8]}] Stopped by user")
            break
        except Exception as e:
            cat = categorize_error(e)
            log.error(f"[{cat}] [{address[:8]}] Error: {e}")
            time.sleep(5)


def _seconds_to_next_epoch(w3: Web3, current_epoch: int) -> int:
    """Estimate seconds until next epoch boundary."""
    try:
        block = w3.eth.get_block("latest")
        blocks_done = block["number"] % EPOCH_BLOCKS
        blocks_left = EPOCH_BLOCKS - blocks_done
        return blocks_left * 12  # ~12s per mainnet block
    except Exception:
        return 60


# ─── multi-wallet orchestration ───────────────────────────────────────────────

def run_fleet(wallets: list[dict], cfg: dict, dry_run: bool, num_workers: int) -> None:
    """
    Run one mining process per wallet in parallel.
    Each wallet gets its own address-bound challenge — they don't share nonces.
    """
    state = load_state()
    log.info(f"Loaded state: {state['total_minted']} total mints recorded")

    if len(wallets) == 1:
        # Single wallet: run inline
        mine_wallet(wallets[0], cfg, state, dry_run, num_workers)
        return

    # Multi-wallet: one process per wallet
    # Divide CPU workers evenly across wallets
    workers_per_wallet = max(1, num_workers // len(wallets))
    log.info(f"Fleet: {len(wallets)} wallets × {workers_per_wallet} CPU workers each")

    processes = []
    for wallet in wallets:
        p = mp.Process(
            target=mine_wallet,
            args=(wallet, cfg, state, dry_run, workers_per_wallet),
            daemon=True,
        )
        p.start()
        processes.append(p)
        time.sleep(0.5)  # stagger startup slightly

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        log.info("Stopping all miners...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=3)


# ─── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fleet OS miner for $HASH (hash256.org)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hash256_miner.py                      Mine with all wallets in WALLET_FILE
  python hash256_miner.py --dry-run            Grind nonces but do not submit
  python hash256_miner.py --wallet 0x...       Mine with one wallet (key from .env PRIVATE_KEY)
  python hash256_miner.py --workers 4          Use 4 CPU workers for nonce grinding
        """,
    )
    parser.add_argument("--dry-run",  action="store_true", help="Grind but don't submit")
    parser.add_argument("--wallet",   type=str,            help="Single wallet address (PRIVATE_KEY in .env)")
    parser.add_argument("--workers",  type=int,            default=0, help="CPU workers (default: cpu_count - 1)")
    args = parser.parse_args()

    cfg = validate_env()

    num_workers = args.workers or int(cfg["MAX_WORKERS"])
    log.info(f"$HASH miner starting | workers={num_workers} | dry_run={args.dry_run}")

    if args.wallet:
        # Single wallet from args, private key from env
        pk = os.getenv("PRIVATE_KEY")
        if not pk:
            log.error("PRIVATE_KEY not set in .env")
            sys.exit(1)
        wallets = [{"address": args.wallet, "private_key": pk}]
    else:
        wallets = load_wallets(cfg["WALLET_FILE"])

    run_fleet(wallets, cfg, args.dry_run, num_workers)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # Safe on all platforms
    main()
