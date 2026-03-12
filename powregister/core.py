"""
Bittensor Subnet PoW Registration Implementation
Based on subtensor, btcli, and bittensor projects analysis

Supports both CPU and CUDA (GPU) solving.
For CUDA: pip install cubit (https://github.com/opentensor/cubit)
"""

import binascii
import hashlib
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import List, Optional, Tuple, Union

from Crypto.Hash import keccak

# CUDA availability check
CUDA_AVAILABLE = False
try:
    import cubit

    CUDA_AVAILABLE = True
except ImportError:
    cubit = None

try:
    import torch

    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_CUDA_AVAILABLE = False


class Network(Enum):
    """Bittensor network endpoints"""

    MAINNET = "wss://entrypoint-finney.opentensor.ai:443"
    TESTNET = "wss://test.finney.opentensor.ai:443"
    LOCAL = "ws://127.0.0.1:9944"

    @classmethod
    def from_string(cls, network: str) -> "Network":
        """Get network from string name"""
        network_lower = network.lower()
        if network_lower in ("mainnet", "finney"):
            return cls.MAINNET
        elif network_lower in ("testnet", "test"):
            return cls.TESTNET
        elif network_lower in ("local", "localhost"):
            return cls.LOCAL
        else:
            raise ValueError(f"Unknown network: {network}. Use: mainnet, testnet, or local")


def rpc_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Retry decorator for RPC calls with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, OSError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        print(f"RPC error: {e}. Retrying in {delay:.0f}s... ({attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        # Reset substrate connection
                        if args and hasattr(args[0], "_substrate"):
                            args[0]._substrate = None
                except Exception as e:
                    # Non-network errors: check if it looks like a connection issue
                    error_str = str(e).lower()
                    if any(kw in error_str for kw in ("connection", "timeout", "websocket", "broken pipe", "eof")):
                        last_error = e
                        if attempt < max_retries - 1:
                            delay = base_delay * (2**attempt)
                            print(f"RPC error: {e}. Retrying in {delay:.0f}s... ({attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            if args and hasattr(args[0], "_substrate"):
                                args[0]._substrate = None
                    else:
                        raise
            raise last_error

        return wrapper

    return decorator


@dataclass
class POWSolution:
    """
    PoW solution.

    Attributes:
        nonce: Found nonce value
        block_number: Block at which solution was found
        difficulty: Difficulty at solution time
        seal: 32-byte seal hash
    """

    nonce: int
    block_number: int
    difficulty: int
    seal: bytes

    def is_stale(self, current_block: int) -> bool:
        """
        Is solution older than 3 blocks?

        Chain only accepts solutions from last 3 blocks.
        This check should be done before submit.
        """
        return self.block_number < current_block - 3

    async def is_stale_async(self, subtensor) -> bool:
        """Async version - gets current block from subtensor"""
        current_block = await subtensor.substrate.get_block_number(None)
        return self.is_stale(current_block)

    def __repr__(self) -> str:
        return (
            f"POWSolution(nonce={self.nonce}, block={self.block_number}, "
            f"difficulty={self.difficulty:,}, seal={self.seal.hex()[:16]}...)"
        )


def hex_bytes_to_u8_list(hex_bytes: bytes) -> list[int]:
    """Convert hex string to u8 list"""
    return [int(hex_bytes[i : i + 2], 16) for i in range(0, len(hex_bytes), 2)]


def hash_block_with_hotkey(block_bytes: bytes, hotkey_bytes: bytes) -> bytes:
    """Hash Block + Hotkey with Keccak256"""
    kec = keccak.new(digest_bits=256)
    kec.update(bytearray(block_bytes + hotkey_bytes))
    return kec.digest()


def create_seal_hash(block_and_hotkey_hash: bytes, nonce: int) -> bytes:
    """
    Create seal hash:
    1. Concatenate nonce (8 byte LE hex) + block_and_hotkey_hash (32 byte hex)
    2. SHA256 hash
    3. Keccak256 hash
    """
    # Convert nonce to little-endian hex
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, "little"))

    # First 64 hex characters of block+hotkey hash (32 bytes)
    hash_hex = binascii.hexlify(block_and_hotkey_hash)[:64]

    # Concatenate
    pre_seal = nonce_bytes + hash_hex

    # SHA256 → Keccak256
    sha256_result = hashlib.sha256(bytearray(hex_bytes_to_u8_list(pre_seal))).digest()

    kec = keccak.new(digest_bits=256)
    seal = kec.update(sha256_result).digest()

    return seal


def seal_meets_difficulty(seal: bytes, difficulty: int) -> bool:
    """Does seal meet difficulty? Uses little-endian to match chain's U256::from_little_endian."""
    seal_number = int.from_bytes(seal, "little")
    product = seal_number * difficulty
    return product < (2**256)


def verify_pow_solution(
    block_hash: bytes,
    hotkey_bytes: bytes,
    nonce: int,
    difficulty: int,
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """
    Verify PoW solution before submitting.

    Performs the same check as the chain.

    Args:
        block_hash: Block hash (32 bytes)
        hotkey_bytes: Hotkey public key (32 bytes)
        nonce: Found nonce value
        difficulty: Subnet difficulty
        verbose: Detailed output

    Returns:
        (is_valid, details) tuple
    """
    # Step 1: Block + Hotkey hash
    block_and_hotkey_hash = hash_block_with_hotkey(block_hash, hotkey_bytes)

    # Step 2: Create seal hash
    seal = create_seal_hash(block_and_hotkey_hash, nonce)

    # Step 3: Difficulty check (little-endian to match chain's U256::from_little_endian)
    seal_int = int.from_bytes(seal, "little")
    limit = 2**256
    product = seal_int * difficulty
    is_valid = product < limit

    details = {
        "block_and_hotkey_hash": block_and_hotkey_hash.hex(),
        "seal": seal.hex(),
        "seal_int": seal_int,
        "difficulty": difficulty,
        "product": product,
        "limit": limit,
        "is_valid": is_valid,
        "margin": limit - product if is_valid else product - limit,
    }

    if verbose:
        print("=== PoW Solution Verification ===")
        print(f"Nonce: {nonce}")
        print(f"Seal: {seal.hex()[:32]}...")
        print(f"Difficulty: {difficulty:,}")
        print("Check: seal x difficulty < 2^256")
        print(f"  {seal_int:.2e} x {difficulty:,} = {product:.2e}")
        print(f"  Limit: {limit:.2e}")
        print(f"  Valid: {'YES' if is_valid else 'NO'}")
        if is_valid:
            margin_pct = (details["margin"] / limit) * 100
            print(f"  Margin: {margin_pct:.2f}% under limit")

    return is_valid, details


def verify_solution_object(
    solution: POWSolution,
    block_hash: bytes,
    hotkey_bytes: bytes,
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """Verify POWSolution object"""
    return verify_pow_solution(
        block_hash=block_hash,
        hotkey_bytes=hotkey_bytes,
        nonce=solution.nonce,
        difficulty=solution.difficulty,
        verbose=verbose,
    )


def solve_pow_single_thread(
    block_and_hotkey_hash: bytes,
    difficulty: int,
    block_number: int,
    nonce_start: int = 0,
    nonce_range: int = 1_000_000,
) -> Optional[POWSolution]:
    """Solve PoW with single thread"""
    for nonce in range(nonce_start, nonce_start + nonce_range):
        seal = create_seal_hash(block_and_hotkey_hash, nonce)

        if seal_meets_difficulty(seal, difficulty):
            return POWSolution(
                nonce=nonce,
                block_number=block_number,
                difficulty=difficulty,
                seal=seal,
            )

    return None


# =============================================================================
# CUDA Solver Functions
# =============================================================================


def solve_cuda(
    nonce_start: int,
    update_interval: int,
    tpb: int,
    block_and_hotkey_hash: bytes,
    difficulty: int,
    dev_id: int = 0,
) -> Tuple[int, Optional[bytes]]:
    """
    Solve PoW with CUDA.

    Args:
        nonce_start: Starting nonce value
        update_interval: Number of nonces to process
        tpb: Threads per block (GPU parallelism)
        block_and_hotkey_hash: Block + hotkey hash (32 bytes)
        difficulty: Target difficulty
        dev_id: CUDA device ID

    Returns:
        (nonce, seal) tuple. Returns (-1, None) if not found
    """
    if not CUDA_AVAILABLE:
        raise ImportError("cubit not installed. Run: pip install cubit")

    limit = 2**256 - 1
    upper = limit // difficulty
    upper_bytes = upper.to_bytes(32, byteorder="little", signed=False)

    # cubit expects hex string
    block_and_hotkey_hash_hex = binascii.hexlify(block_and_hotkey_hash)[:64]

    # Call CUDA solver
    solution_nonce = cubit.solve_cuda(
        tpb,  # threads per block
        nonce_start,  # starting nonce
        update_interval,  # nonces to process
        upper_bytes,  # difficulty upper bound
        block_and_hotkey_hash_hex,  # block+hotkey hash hex
        dev_id,  # CUDA device
    )

    if solution_nonce != -1:
        seal = create_seal_hash(block_and_hotkey_hash, solution_nonce)
        if seal_meets_difficulty(seal, difficulty):
            return solution_nonce, seal

    return -1, None


def _cuda_worker_solve(
    block_and_hotkey_hash: bytes,
    difficulty: int,
    block_number: int,
    nonce_start: int,
    update_interval: int,
    tpb: int,
    dev_id: int,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    progress_counter: mp.Value = None,
):
    """CUDA worker process"""
    nonce_limit = 2**64 - 1
    current_nonce = nonce_start
    nonces_per_batch = update_interval * tpb

    while not stop_event.is_set():
        try:
            solution_nonce, seal = solve_cuda(
                nonce_start=current_nonce,
                update_interval=update_interval,
                tpb=tpb,
                block_and_hotkey_hash=block_and_hotkey_hash,
                difficulty=difficulty,
                dev_id=dev_id,
            )

            if solution_nonce != -1 and seal is not None:
                result_queue.put(
                    POWSolution(
                        nonce=solution_nonce,
                        block_number=block_number,
                        difficulty=difficulty,
                        seal=seal,
                    )
                )
                return

            # Report progress
            if progress_counter is not None:
                with progress_counter.get_lock():
                    progress_counter.value += nonces_per_batch

            # Move to next nonce block
            current_nonce = (current_nonce + nonces_per_batch) % nonce_limit

        except Exception as e:
            print(f"CUDA worker error: {e}")
            return


@dataclass
class RegistrationStatistics:
    """
    PoW solving statistics.

    Attributes:
        time_spent_total: Total elapsed time (seconds)
        rounds_total: Total nonce blocks processed
        time_average: Average time per block
        hash_rate_perpetual: Overall hash rate
        hash_rate: Instant hash rate (EWMA)
        difficulty: Current difficulty
        block_number: Current block
        block_hash: Current block hash
    """

    time_spent_total: float = 0.0
    rounds_total: int = 0
    time_average: float = 0.0
    hash_rate_perpetual: float = 0.0
    hash_rate: float = 0.0
    difficulty: int = 0
    block_number: int = 0
    block_hash: str = ""

    def update(self, time_spent: float, nonces_processed: int):
        """Update statistics"""
        self.time_spent_total += time_spent
        self.rounds_total += 1
        self.time_average = self.time_spent_total / self.rounds_total
        self.hash_rate_perpetual = nonces_processed / max(time_spent, 0.001)
        # EWMA for hash rate (alpha = 0.2)
        alpha = 0.2
        self.hash_rate = alpha * self.hash_rate_perpetual + (1 - alpha) * self.hash_rate


def get_cpu_count() -> int:
    """Return available CPU count"""
    return mp.cpu_count()


def validate_tpb(tpb: int) -> int:
    """
    Validate threads per block value.
    TPB should be multiple of 32 (CUDA warp size).
    """
    if tpb % 32 != 0:
        new_tpb = ((tpb // 32) + 1) * 32
        print(f"Warning: tpb should be multiple of 32. Adjusting {tpb} -> {new_tpb}")
        return new_tpb
    return tpb


def solve_pow_cuda(
    block_and_hotkey_hash: bytes,
    difficulty: int,
    block_number: int,
    dev_id: Union[int, List[int]] = 0,
    tpb: int = 256,
    update_interval: int = 100_000,
    timeout: int = 3600,
) -> Optional[POWSolution]:
    """
    Solve PoW with CUDA (GPU) in parallel.

    Args:
        block_and_hotkey_hash: Block + hotkey hash (32 bytes)
        difficulty: Target difficulty
        block_number: Block number
        dev_id: CUDA device ID(s). Can be single int or list.
        tpb: Threads per block (should be multiple of 32, usually 256 or 512)
        update_interval: Nonce block to process per iteration
        timeout: Maximum wait time (seconds)

    Returns:
        POWSolution or None
    """
    if not CUDA_AVAILABLE:
        raise ImportError("cubit not installed. Run: pip install cubit")

    # TPB validation
    tpb = validate_tpb(tpb)

    # Convert dev_id to list
    if isinstance(dev_id, int):
        dev_ids = [dev_id]
    else:
        dev_ids = list(dev_id)

    result_queue = mp.Queue()
    stop_event = mp.Event()
    progress_counter = mp.Value("L", 0)
    processes = []
    nonce_limit = 2**64 - 1

    # Start a worker for each GPU
    for device in dev_ids:
        # Each GPU starts from different nonce range
        nonce_start = random.randint(0, nonce_limit)

        p = mp.Process(
            target=_cuda_worker_solve,
            args=(
                block_and_hotkey_hash,
                difficulty,
                block_number,
                nonce_start,
                update_interval,
                tpb,
                device,
                result_queue,
                stop_event,
                progress_counter,
            ),
            daemon=True,
        )
        processes.append(p)
        p.start()

    # Wait for result with progress reporting
    start_time = time.time()
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                print(f"\nCUDA timeout after {elapsed:.0f}s")
                stop_event.set()
                return None

            try:
                solution = result_queue.get(timeout=3)
                stop_event.set()
                total_hashes = progress_counter.value
                print(
                    f"\rCUDA solved! {total_hashes:,} hashes in {elapsed:.1f}s "
                    f"({_format_hashrate(total_hashes / max(elapsed, 0.001))})"
                )
                return solution
            except Exception:
                total_hashes = progress_counter.value
                if total_hashes > 0:
                    rate = total_hashes / max(elapsed, 0.001)
                    print(
                        f"\r  [{elapsed:.0f}s] CUDA {total_hashes:,} hashes | "
                        f"{_format_hashrate(rate)} | "
                        f"difficulty: {difficulty:,}",
                        end="",
                        flush=True,
                    )
    finally:
        stop_event.set()
        for p in processes:
            p.terminate()


# =============================================================================
# CPU Solver Functions
# =============================================================================


def _worker_solve(
    block_and_hotkey_hash: bytes,
    difficulty: int,
    block_number: int,
    nonce_start: int,
    nonce_range: int,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    progress_counter: mp.Value = None,
):
    """CPU worker process for parallel solving"""
    nonce_limit = 2**64 - 1
    current_nonce = nonce_start
    local_count = 0

    while not stop_event.is_set():
        for nonce in range(current_nonce, min(current_nonce + nonce_range, nonce_limit)):
            if stop_event.is_set():
                return

            seal = create_seal_hash(block_and_hotkey_hash, nonce)
            local_count += 1

            if seal_meets_difficulty(seal, difficulty):
                if progress_counter is not None:
                    with progress_counter.get_lock():
                        progress_counter.value += local_count
                result_queue.put(
                    POWSolution(
                        nonce=nonce,
                        block_number=block_number,
                        difficulty=difficulty,
                        seal=seal,
                    )
                )
                return

        # Report progress after each batch
        if progress_counter is not None:
            with progress_counter.get_lock():
                progress_counter.value += local_count
            local_count = 0

        current_nonce = (current_nonce + nonce_range) % nonce_limit


def _format_hashrate(h: float) -> str:
    """Format hash rate for display"""
    if h >= 1_000_000:
        return f"{h / 1_000_000:.2f} MH/s"
    elif h >= 1_000:
        return f"{h / 1_000:.2f} KH/s"
    return f"{h:.0f} H/s"


def solve_pow_parallel(
    block_and_hotkey_hash: bytes,
    difficulty: int,
    block_number: int,
    num_processes: int = None,
    update_interval: int = 50_000,
    timeout: int = 3600,
) -> Optional[POWSolution]:
    """Solve PoW with parallel CPU"""
    if num_processes is None:
        num_processes = mp.cpu_count()

    result_queue = mp.Queue()
    stop_event = mp.Event()
    progress_counter = mp.Value("L", 0)  # unsigned long shared counter

    # Each worker starts from different nonce range
    processes = []
    nonce_limit = 2**64 - 1

    for _ in range(num_processes):
        nonce_start = random.randint(0, nonce_limit)

        p = mp.Process(
            target=_worker_solve,
            args=(
                block_and_hotkey_hash,
                difficulty,
                block_number,
                nonce_start,
                update_interval,
                result_queue,
                stop_event,
                progress_counter,
            ),
            daemon=True,
        )
        processes.append(p)
        p.start()

    # Wait for result with progress reporting
    start_time = time.time()
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                print(f"\nTimeout after {elapsed:.0f}s")
                stop_event.set()
                return None

            try:
                solution = result_queue.get(timeout=5)
                stop_event.set()
                total_hashes = progress_counter.value
                print(
                    f"\rSolved! {total_hashes:,} hashes in {elapsed:.1f}s "
                    f"({_format_hashrate(total_hashes / max(elapsed, 0.001))})"
                )
                return solution
            except Exception:
                # No result yet, print progress
                total_hashes = progress_counter.value
                if total_hashes > 0:
                    rate = total_hashes / max(elapsed, 0.001)
                    print(
                        f"\r  [{elapsed:.0f}s] {total_hashes:,} hashes | "
                        f"{_format_hashrate(rate)} | "
                        f"difficulty: {difficulty:,}",
                        end="",
                        flush=True,
                    )
    finally:
        stop_event.set()
        for p in processes:
            p.terminate()


class SubtensorPowRegistration:
    """Subtensor PoW Registration Client"""

    def __init__(self, network: Union[str, Network] = Network.MAINNET):
        """
        Initialize registration client.

        Args:
            network: Network to connect to. Can be:
                - Network enum (Network.MAINNET, Network.TESTNET, Network.LOCAL)
                - String: "mainnet", "testnet", "local"
                - Full URL: "wss://custom-endpoint.com:443"
        """
        if isinstance(network, Network):
            self.substrate_url = network.value
            self.network_name = network.name
        elif network.startswith("ws://") or network.startswith("wss://"):
            self.substrate_url = network
            self.network_name = "CUSTOM"
        else:
            net = Network.from_string(network)
            self.substrate_url = net.value
            self.network_name = net.name

        self._substrate = None

    @property
    def substrate(self):
        if self._substrate is None:
            from substrateinterface import SubstrateInterface

            self._substrate = SubstrateInterface(url=self.substrate_url)
        return self._substrate

    def _reset_substrate(self):
        """Reset substrate connection (e.g. after bt.Subtensor used the same endpoint)"""
        self._substrate = None

    @rpc_retry()
    def get_current_block(self) -> int:
        """Get current block number"""
        return self.substrate.get_block_number(None)

    @rpc_retry()
    def get_block_hash(self, block_number: int) -> str:
        """Get block hash"""
        return self.substrate.get_block_hash(block_number)

    @rpc_retry()
    def get_slot_info(self, netuid: int) -> dict:
        """Analyze slot availability on a full subnet.

        When a subnet is full, new PoW/burn registrations replace the
        neuron with the lowest performance score that is outside its
        immunity period. Even if all neurons are immune, registration
        is still possible — the lowest pruning score gets replaced.
        """
        current_block = self.get_current_block()
        immunity_period = self.get_hyperparameter("ImmunityPeriod", netuid) or 4096
        adjustment_interval = self.get_hyperparameter("AdjustmentInterval", netuid) or 112
        target_regs = self.get_hyperparameter("TargetRegistrationsPerInterval", netuid) or 1

        # Get all registration blocks
        reg_blocks: dict[int, int] = {}
        for uid, block_at_reg in self.substrate.query_map(
            module="SubtensorModule",
            storage_function="BlockAtRegistration",
            params=[netuid],
        ):
            reg_blocks[int(uid.value)] = int(block_at_reg.value)

        # Immunity count
        immune = 0
        outside_immunity = 0
        for _uid, reg_block in reg_blocks.items():
            if current_block - reg_block >= immunity_period:
                outside_immunity += 1
            else:
                immune += 1

        # Regs in current adjustment interval
        interval_start = current_block - (current_block % adjustment_interval)
        regs_this_interval = sum(1 for b in reg_blocks.values() if b >= interval_start)

        # Rate limit is 3x target — chain rejects registrations beyond this
        max_regs_per_interval = target_regs * 3

        return {
            "outside_immunity": outside_immunity,
            "immune_neurons": immune,
            "regs_this_interval": regs_this_interval,
            "max_regs_per_interval": max_regs_per_interval,
        }

    @rpc_retry()
    def get_difficulty(self, netuid: int) -> int:
        """Get subnet difficulty"""
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="Difficulty",
            params=[netuid],
        )
        return int(result.value) if result else 10_000_000

    @rpc_retry()
    def get_burn_cost(self, netuid: int) -> int:
        """Get subnet burn cost (in RAO)"""
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function="Burn",
            params=[netuid],
        )
        return int(result.value) if result else 0

    @rpc_retry()
    def get_hyperparameter(self, param_name: str, netuid: int) -> Optional[int]:
        """Get subnet hyperparameter"""
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function=param_name,
            params=[netuid],
        )
        return int(result.value) if result and result.value is not None else None

    def get_registration_info(self, netuid: int) -> dict:
        """Get subnet registration info including validator requirements"""
        difficulty = self.get_difficulty(netuid)
        burn_rao = self.get_burn_cost(netuid)
        burn_tao = burn_rao / 1e9  # RAO to TAO

        # Registration type check
        pow_allowed = self.substrate.query(
            module="SubtensorModule",
            storage_function="NetworkPowRegistrationAllowed",
            params=[netuid],
        )
        registration_allowed = self.substrate.query(
            module="SubtensorModule",
            storage_function="NetworkRegistrationAllowed",
            params=[netuid],
        )

        # Validator parameters
        max_validators = self.get_hyperparameter("MaxAllowedValidators", netuid)
        current_neurons = self.get_hyperparameter("SubnetworkN", netuid) or 0
        max_neurons = self.get_hyperparameter("MaxAllowedUids", netuid) or 256

        # Global minimum stake for nominators
        nominator_min_stake = self.substrate.query(
            module="SubtensorModule",
            storage_function="NominatorMinRequiredStake",
        )

        return {
            "netuid": netuid,
            "difficulty": difficulty,
            "burn_cost_rao": burn_rao,
            "burn_cost_tao": burn_tao,
            "pow_registration_allowed": bool(pow_allowed.value) if pow_allowed else True,
            "registration_allowed": bool(registration_allowed.value) if registration_allowed else True,
            "max_validators": max_validators,
            "max_neurons": max_neurons,
            "current_neurons": current_neurons,
            "slots_available": max_neurons - current_neurons,
            "nominator_min_stake_tao": (int(nominator_min_stake.value) / 1e9) if nominator_min_stake else 0,
        }

    def get_validator_params(self, netuid: int) -> dict:
        """
        Get validator-specific parameters for a subnet.

        Returns validator permit requirements, stake limits, and validator counts.
        """
        # Max validators allowed
        max_validators = self.get_hyperparameter("MaxAllowedValidators", netuid)

        # Current validator count (neurons with validator permit)
        current_neurons = self.get_hyperparameter("SubnetworkN", netuid) or 0

        # Minimum stake to be a validator (global parameter)
        nominator_min_stake = self.substrate.query(
            module="SubtensorModule",
            storage_function="NominatorMinRequiredStake",
        )

        # Validator trust and other params
        validator_trust = self.get_hyperparameter("ValidatorTrust", netuid)

        return {
            "netuid": netuid,
            "max_validators": max_validators,
            "current_neurons": current_neurons,
            "nominator_min_stake_rao": int(nominator_min_stake.value) if nominator_min_stake else 0,
            "nominator_min_stake_tao": (int(nominator_min_stake.value) / 1e9) if nominator_min_stake else 0,
            "validator_trust": validator_trust,
        }

    def get_subnet_params(self, netuid: int) -> dict:
        """
        Get all important subnet parameters.

        These parameters affect slot opening time and registration.
        """
        BLOCK_TIME_SECONDS = 12  # Bittensor block time

        # Basic parameters
        tempo = self.get_hyperparameter("Tempo", netuid) or 360
        immunity_period = self.get_hyperparameter("ImmunityPeriod", netuid) or 4096
        max_neurons = self.get_hyperparameter("MaxAllowedUids", netuid) or 256
        min_difficulty = self.get_hyperparameter("MinDifficulty", netuid) or 10_000_000
        max_difficulty = self.get_hyperparameter("MaxDifficulty", netuid) or 10_000_000_000
        adjustment_interval = self.get_hyperparameter("AdjustmentInterval", netuid) or 112
        target_regs_per_interval = self.get_hyperparameter("TargetRegistrationsPerInterval", netuid) or 1

        # Get current state
        difficulty = self.get_difficulty(netuid)
        burn_rao = self.get_burn_cost(netuid)
        current_neurons = self.get_hyperparameter("SubnetworkN", netuid) or 0

        # Validator parameters
        max_validators = self.get_hyperparameter("MaxAllowedValidators", netuid)

        # Global minimum stake
        nominator_min_stake = self.substrate.query(
            module="SubtensorModule",
            storage_function="NominatorMinRequiredStake",
        )

        # Time calculations
        immunity_period_minutes = (immunity_period * BLOCK_TIME_SECONDS) / 60
        tempo_minutes = (tempo * BLOCK_TIME_SECONDS) / 60
        adjustment_interval_minutes = (adjustment_interval * BLOCK_TIME_SECONDS) / 60

        return {
            "netuid": netuid,
            # Registration
            "difficulty": difficulty,
            "burn_cost_tao": burn_rao / 1e9,
            "min_difficulty": min_difficulty,
            "max_difficulty": max_difficulty,
            # Timing (blocks)
            "tempo": tempo,
            "immunity_period": immunity_period,
            "adjustment_interval": adjustment_interval,
            "target_regs_per_interval": target_regs_per_interval,
            # Timing (human readable)
            "tempo_minutes": round(tempo_minutes, 1),
            "immunity_period_minutes": round(immunity_period_minutes, 1),
            "adjustment_interval_minutes": round(adjustment_interval_minutes, 1),
            # Capacity
            "max_neurons": max_neurons,
            "current_neurons": current_neurons,
            "slots_available": max_neurons - current_neurons,
            "open_slots": None,  # populated if subnet is full
            "immune_neurons": None,
            # Validator
            "max_validators": max_validators,
            "nominator_min_stake_tao": (int(nominator_min_stake.value) / 1e9) if nominator_min_stake else 0,
        }

    def get_subnet_params_with_slots(self, netuid: int) -> dict:
        """Get subnet params and analyze slot availability if subnet is full."""
        params = self.get_subnet_params(netuid)
        if params["slots_available"] <= 0:
            print("  Querying slot availability...")
            slot_info = self.get_slot_info(netuid)
            params.update(slot_info)
        return params

    def print_subnet_params(self, netuid: int):
        """Print subnet parameters in nice format"""
        params = self.get_subnet_params_with_slots(netuid)

        print(f"\n{'=' * 60}")
        print(f"Subnet {netuid} Parameters ({self.network_name})")
        print(f"{'=' * 60}")

        print("\nRegistration:")
        print(f"  Difficulty: {params['difficulty']:,}")
        print(f"  Burn Cost: {params['burn_cost_tao']:.4f} TAO")
        print(f"  Difficulty Range: {params['min_difficulty']:,} - {params['max_difficulty']:,}")

        print("\nValidator:")
        print(f"  Max Validators: {params['max_validators'] or 'N/A'}")
        print(f"  Min Nominator Stake: {params['nominator_min_stake_tao']:.4f} TAO")

        print("\nTiming:")
        print(f"  Tempo: {params['tempo']} blocks ({params['tempo_minutes']} min)")
        print(f"  Immunity Period: {params['immunity_period']} blocks ({params['immunity_period_minutes']} min)")
        adj = params["adjustment_interval"]
        adj_min = params["adjustment_interval_minutes"]
        print(f"  Difficulty Adjustment: every {adj} blocks ({adj_min} min)")
        print(f"  Target Registrations/Interval: {params['target_regs_per_interval']}")

        print("\nCapacity:")
        print(f"  Max Neurons: {params['max_neurons']}")
        print(f"  Current Neurons: {params['current_neurons']}")

        if params["slots_available"] > 0:
            print(f"  Slots Available: {params['slots_available']}")
        elif params.get("outside_immunity") is not None:
            outside = params["outside_immunity"]
            imm = params["immune_neurons"]
            regs = params["regs_this_interval"]
            max_regs = params["max_regs_per_interval"]
            print(f"  Regs This Interval: {regs}/{max_regs}")
            print(f"  Outside Immunity: {outside}")
            print(f"  Immune Neurons: {imm}")
            if outside > 0:
                print("\n[*] Subnet is full. New registrations replace the lowest-scoring neuron outside immunity.")
            else:
                print(
                    "\n[*] Subnet is full and all neurons are immune."
                    "\n    Registration is still possible — the neuron with"
                    " the lowest pruning score will be replaced."
                )

    def get_all_subnet_registration_costs(self, netuids: List[int] = None) -> List[dict]:
        """Get registration costs for subnets with subnet names via bt SDK.

        Args:
            netuids: List of subnet IDs to query. If None, queries all.
        """
        import bittensor as bt

        subtensor = bt.Subtensor(network=self.substrate_url)

        if netuids is None:
            total = subtensor.get_total_subnets()
            netuids = list(range(total))

        results = []
        for netuid in netuids:
            try:
                dynamic = subtensor.subnet(netuid=netuid)
                reg_info = self.get_registration_info(netuid)
                results.append(
                    {
                        **reg_info,
                        "subnet_name": dynamic.subnet_name or "",
                        "price_tao": float(dynamic.price) if hasattr(dynamic, "price") else 0,
                    }
                )
            except Exception as e:
                print(f"Error fetching netuid {netuid}: {e}")

        return results

    def print_registration_costs(
        self,
        netuids: List[int] = None,
        min_burn_tao: float = None,
        pow_only: bool = False,
        max_difficulty: int = None,
    ):
        """Print registration costs for subnets with name and cost analysis"""
        print(f"\n{self.network_name} - Subnet Registration Costs")
        print("=" * 115)
        print(f"{'NetUID':<7} {'Name':<16} {'Difficulty':<14} {'Burn (TAO)':<12} {'Slots':<12} {'PoW':<5} {'Reg':<5}")
        print("-" * 105)

        costs = self.get_all_subnet_registration_costs(netuids=netuids)
        # Filter first, then query slot info only for displayed subnets
        filtered = []
        for info in costs:
            if pow_only and not info["pow_registration_allowed"]:
                continue
            if min_burn_tao is not None and info["burn_cost_tao"] <= min_burn_tao:
                continue
            if max_difficulty is not None and max_difficulty > 0 and info["difficulty"] > max_difficulty:
                continue
            filtered.append(info)

        for info in filtered:
            pow_ok = "Y" if info["pow_registration_allowed"] else "N"
            reg_ok = "Y" if info["registration_allowed"] else "N"
            if info["slots_available"] > 0:
                slots = f"{info['slots_available']}/{info['max_neurons']}"
            else:
                try:
                    slot_info = self.get_slot_info(info["netuid"])
                    regs = slot_info["regs_this_interval"]
                    max_regs = slot_info["max_regs_per_interval"]
                    remaining = max_regs - regs
                    if remaining > 0:
                        slots = f"{remaining}/{max_regs} avail"
                    else:
                        slots = f"FULL ({regs}/{max_regs})"
                except Exception:
                    slots = f"0/{info['max_neurons']}"
            name = info.get("subnet_name", "")[:15]
            print(
                f"{info['netuid']:<7} {name:<16} {info['difficulty']:<14,} {info['burn_cost_tao']:<12.4f} "
                f"{slots:<12} {pow_ok:<5} {reg_ok:<5}"
            )

        print("=" * 105)
        print("PoW = PoW Registration, Reg = Registration")

    @rpc_retry()
    def get_balance(self, ss58_address: str) -> float:
        """Get account balance in TAO"""
        result = self.substrate.query(
            module="System",
            storage_function="Account",
            params=[ss58_address],
        )
        if result and result.value:
            free = result.value.get("data", {}).get("free", 0)
            return int(free) / 1e9  # RAO to TAO
        return 0.0

    def _ensure_hotkey_funded(self, wallet, min_balance: float = 0.01):
        """Ensure hotkey has enough TAO for registration tx fees.

        Chain requires hotkey as signer for PoW register (Pays::Yes),
        so hotkey must have balance. Transfers from coldkey if needed.
        """
        import bittensor as bt

        subtensor = bt.Subtensor(network=self.substrate_url)

        hotkey_bal = float(subtensor.get_balance(wallet.hotkey.ss58_address))
        coldkey_bal = float(subtensor.get_balance(wallet.coldkeypub.ss58_address))
        print(f"Coldkey balance: {coldkey_bal:.4f} TAO")
        print(f"Hotkey balance:  {hotkey_bal:.4f} TAO")

        if hotkey_bal >= min_balance:
            return

        # Hotkey has some balance but below threshold — ask user
        if hotkey_bal > 0:
            answer = input(
                f"Hotkey balance ({hotkey_bal:.4f} TAO) is below {min_balance} TAO. Transfer from coldkey? [y/N]: "
            )
            if answer.strip().lower() != "y":
                print(f"Continuing with hotkey balance: {hotkey_bal:.4f} TAO")
                return

        if coldkey_bal < min_balance + 0.01:
            raise ValueError(f"Coldkey balance too low ({coldkey_bal:.4f} TAO)")

        print(f"Transferring {min_balance} TAO coldkey → hotkey for tx fees...")
        result = subtensor.transfer(
            wallet=wallet,
            destination_ss58=wallet.hotkey.ss58_address,
            amount=bt.Balance.from_tao(min_balance),
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        if not result.success:
            raise RuntimeError(f"Transfer failed: {result.message}")

        new_bal = float(subtensor.get_balance(wallet.hotkey.ss58_address))
        print(f"Transfer OK. Hotkey balance: {new_bal:.4f} TAO")

    def prepare_pow_input(
        self,
        netuid: int,
        hotkey_ss58: str,
    ) -> Tuple[bytes, int, int]:
        """
        Prepare required data for PoW
        Returns: (block_and_hotkey_hash, block_number, difficulty)
        """
        from substrateinterface import Keypair

        # Get block info
        block_number = self.get_current_block()
        block_hash = self.get_block_hash(block_number)
        difficulty = self.get_difficulty(netuid)

        # Convert block hash to bytes (remove 0x prefix)
        block_bytes = bytes.fromhex(block_hash[2:])

        # Hotkey public key bytes
        hotkey_keypair = Keypair(ss58_address=hotkey_ss58)
        hotkey_bytes = hotkey_keypair.public_key

        # Block + Hotkey hash
        block_and_hotkey_hash = hash_block_with_hotkey(block_bytes, hotkey_bytes)

        return block_and_hotkey_hash, block_number, difficulty

    def solve_and_register(
        self,
        netuid: int,
        wallet,  # bittensor.Wallet
        use_cuda: bool = False,
        dev_id: Union[int, List[int]] = 0,
        tpb: int = 256,
        num_processes: int = None,
        update_interval: int = 50_000,
        max_attempts: int = 3,
        timeout: int = 3600,
    ) -> bool:
        """
        Solve PoW and register.

        Pre-creates the bt.Subtensor connection so submission is instant
        after finding a solution (avoids InvalidWorkBlock from slow connect).
        """
        BLOCK_REFRESH_SECONDS = 20  # Refresh well before 3-block window (36s)

        if use_cuda and not CUDA_AVAILABLE:
            print("CUDA not available, falling back to CPU...")
            use_cuda = False

        # Pre-create bt.Subtensor + pallet so submission is instant
        import bittensor as bt
        from bittensor.core.extrinsics.pallets.subtensor_module import SubtensorModule

        submit_subtensor = bt.Subtensor(network=self.substrate_url)
        submit_pallet = SubtensorModule(submit_subtensor)
        print("Submission connection ready.")
        self._reset_substrate()

        for attempt in range(max_attempts):
            print(f"\n=== Attempt {attempt + 1}/{max_attempts} ===")

            # Prepare PoW inputs
            block_and_hotkey_hash, block_number, difficulty = self.prepare_pow_input(
                netuid=netuid,
                hotkey_ss58=wallet.hotkey.ss58_address,
            )

            print(f"Block: {block_number}, Difficulty: {difficulty:,}")

            # Solve PoW in short rounds, refreshing block to avoid staleness
            solution = None
            attempt_start = time.time()
            round_num = 0

            while time.time() - attempt_start < timeout:
                round_num += 1
                round_timeout = BLOCK_REFRESH_SECONDS

                if use_cuda:
                    dev_list = [dev_id] if isinstance(dev_id, int) else dev_id
                    if round_num == 1:
                        print(f"Solving with CUDA on device(s): {dev_list}, TPB: {tpb}")
                    solution = solve_pow_cuda(
                        block_and_hotkey_hash=block_and_hotkey_hash,
                        difficulty=difficulty,
                        block_number=block_number,
                        dev_id=dev_id,
                        tpb=tpb,
                        update_interval=update_interval,
                        timeout=round_timeout,
                    )
                else:
                    proc_count = num_processes or mp.cpu_count()
                    if round_num == 1:
                        print(f"Solving with CPU ({proc_count} processes)...")
                    solution = solve_pow_parallel(
                        block_and_hotkey_hash=block_and_hotkey_hash,
                        difficulty=difficulty,
                        block_number=block_number,
                        num_processes=num_processes,
                        update_interval=update_interval,
                        timeout=round_timeout,
                    )

                if solution is not None:
                    break

                # No solution yet, refresh block and continue
                new_block = self.get_current_block()
                if new_block != block_number:
                    block_and_hotkey_hash, block_number, difficulty = self.prepare_pow_input(
                        netuid=netuid,
                        hotkey_ss58=wallet.hotkey.ss58_address,
                    )
                    elapsed = time.time() - attempt_start
                    print(f"\n  Block refreshed: {block_number} ({elapsed:.0f}s elapsed, round {round_num})")

            if solution is None:
                print(f"\nPoW solution not found within {timeout}s!")
                continue

            print(f"\nSolution found! Nonce: {solution.nonce}, Block: {solution.block_number}")

            # Submit immediately — no staleness check, every second counts
            # Chain validates staleness anyway (InvalidWorkBlock if too old)
            success = self._submit_registration(
                netuid=netuid,
                solution=solution,
                wallet=wallet,
                subtensor=submit_subtensor,
                pallet=submit_pallet,
            )

            if success:
                return True

        print(f"Failed after {max_attempts} attempts")
        return False

    def _submit_registration(
        self,
        netuid: int,
        solution: POWSolution,
        wallet,
        subtensor=None,
        pallet=None,
    ) -> bool:
        """Submit registration extrinsic using pre-created bt.Subtensor for speed."""
        try:
            if subtensor is None or pallet is None:
                import bittensor as bt
                from bittensor.core.extrinsics.pallets.subtensor_module import SubtensorModule

                subtensor = bt.Subtensor(network=self.substrate_url)
                pallet = SubtensorModule(subtensor)

            submit_start = time.time()

            call = pallet.register(
                netuid=netuid,
                coldkey=wallet.coldkeypub.ss58_address,
                hotkey=wallet.hotkey.ss58_address,
                block_number=solution.block_number,
                nonce=solution.nonce,
                work=[int(b) for b in solution.seal],
            )

            response = subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                sign_with="hotkey",
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )

            submit_elapsed = time.time() - submit_start
            print(f"Submission took {submit_elapsed:.1f}s")

            if response.success:
                print("Registration successful!")
                return True
            else:
                print(f"Registration failed: {response.message}")
                return False

        except Exception as e:
            print(f"Registration error: {e}")
            return False
        finally:
            self._reset_substrate()


# =============================================================================
# Utility Functions
# =============================================================================


def check_cuda_available() -> dict:
    """Check CUDA status"""
    return {
        "cubit_available": CUDA_AVAILABLE,
        "torch_cuda_available": TORCH_CUDA_AVAILABLE,
        "recommended": "CUDA" if CUDA_AVAILABLE else "CPU",
    }


def reset_cuda():
    """Reset CUDA environment"""
    if CUDA_AVAILABLE:
        cubit.reset_cuda()


def get_cuda_device_count() -> int:
    """Get available CUDA device count"""
    if TORCH_CUDA_AVAILABLE:
        import torch

        return torch.cuda.device_count()
    return 0


def get_validator_stake_threshold(network: str = "mainnet", netuids: List[int] = None) -> List[dict]:
    """
    Get the minimum stake required to become a validator for each subnet.

    This finds the stake of the lowest-ranked validator (the Nth validator,
    where N = MaxAllowedValidators). To become a validator, you need more
    stake than this amount.

    Args:
        network: Network to query (mainnet, testnet)
        netuids: List of subnet IDs to check. If None, checks common subnets.

    Returns:
        List of dicts with validator threshold info for each subnet
    """
    try:
        import bittensor as bt
    except ImportError:
        raise ImportError("bittensor package required. Run: pip install bittensor")

    # Connect to network
    if network.lower() in ("mainnet", "finney"):
        subtensor = bt.Subtensor(network="finney")
    elif network.lower() in ("testnet", "test"):
        subtensor = bt.Subtensor(network="test")
    else:
        subtensor = bt.Subtensor(network=network)

    if netuids is None:
        netuids = list(range(1, 20))  # Default: check subnets 1-19

    results = []

    for netuid in netuids:
        try:
            # Get metagraph for this subnet
            metagraph = subtensor.metagraph(netuid=netuid)

            # Get max validators for this subnet
            max_validators = subtensor.get_subnet_hyperparameters(netuid=netuid).max_validators

            if max_validators is None or max_validators == 0:
                continue

            # Get all stakes and sort descending
            # metagraph.S is already a numpy array or tensor
            stakes_array = metagraph.S
            if hasattr(stakes_array, "numpy"):
                stakes = list(stakes_array.numpy())
            else:
                stakes = list(stakes_array)
            stakes_sorted = sorted(stakes, reverse=True)

            # The threshold is the stake of the Nth validator (index N-1)
            if len(stakes_sorted) >= max_validators:
                threshold_stake = stakes_sorted[max_validators - 1]
            else:
                threshold_stake = 0  # Not enough neurons, any stake works

            # Get alpha (subnet-specific token) if available
            # Alpha is the subnet's native token stake
            threshold_alpha = None
            try:
                if hasattr(metagraph, "alpha_stake"):
                    alpha_array = metagraph.alpha_stake
                    if hasattr(alpha_array, "numpy"):
                        alpha_stakes = list(alpha_array.numpy())
                    else:
                        alpha_stakes = list(alpha_array)
                    alpha_sorted = sorted(alpha_stakes, reverse=True)
                    if len(alpha_sorted) >= max_validators:
                        threshold_alpha = alpha_sorted[max_validators - 1]
                    else:
                        threshold_alpha = 0
            except Exception:
                threshold_alpha = None

            results.append(
                {
                    "netuid": netuid,
                    "max_validators": max_validators,
                    "total_neurons": len(stakes),
                    "min_validator_stake_tao": round(threshold_stake, 4),
                    "min_validator_stake_alpha": round(threshold_alpha, 4) if threshold_alpha is not None else None,
                    "top_validator_stake_tao": round(stakes_sorted[0], 4) if stakes_sorted else 0,
                }
            )

        except Exception as e:
            print(f"Error fetching netuid {netuid}: {e}")

    return results


def print_validator_thresholds(network: str = "mainnet", netuids: List[int] = None):
    """
    Print validator stake thresholds for subnets.

    Shows the minimum TAO/Alpha stake required to become a validator.
    """
    print(f"\nFetching validator thresholds from {network}...")
    print("This may take a moment as it loads metagraph data...\n")

    results = get_validator_stake_threshold(network=network, netuids=netuids)

    if not results:
        print("No data found.")
        return

    # Check if any subnet has alpha
    has_alpha = any(r.get("min_validator_stake_alpha") is not None for r in results)

    print(f"{network.upper()} - Minimum Stake to Become Validator")
    print("=" * 90)

    if has_alpha:
        print(f"{'NetUID':<8} {'Max Val':<10} {'Neurons':<10} {'Min TAO':<15} {'Min Alpha':<15} {'Top TAO':<15}")
    else:
        print(f"{'NetUID':<8} {'Max Val':<10} {'Neurons':<10} {'Min TAO Stake':<18} {'Top Validator':<18}")
    print("-" * 90)

    for r in results:
        if has_alpha:
            alpha_str = f"{r['min_validator_stake_alpha']:.2f}" if r["min_validator_stake_alpha"] is not None else "N/A"
            print(
                f"{r['netuid']:<8} {r['max_validators']:<10} {r['total_neurons']:<10} "
                f"{r['min_validator_stake_tao']:<15.2f} {alpha_str:<15} {r['top_validator_stake_tao']:<15.2f}"
            )
        else:
            print(
                f"{r['netuid']:<8} {r['max_validators']:<10} {r['total_neurons']:<10} "
                f"{r['min_validator_stake_tao']:<18.2f} {r['top_validator_stake_tao']:<18.2f}"
            )

    print("=" * 90)
    print("Min TAO Stake = Minimum stake needed to enter top N validators")
    print("Top Validator = Stake of the highest-staked validator")
    print("\nNote: You need MORE than the 'Min TAO Stake' to replace the lowest validator.")
