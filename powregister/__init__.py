"""
Bittensor Subnet PoW Registration Tool

Supports both CPU and CUDA (GPU) solving for subnet registration.
"""

from powregister.core import (
    POWSolution,
    SubtensorPowRegistration,
    check_cuda_available,
    create_seal_hash,
    hash_block_with_hotkey,
    seal_meets_difficulty,
    solve_pow_cuda,
    solve_pow_parallel,
    solve_pow_single_thread,
    verify_pow_solution,
)

__version__ = "0.1.0"
__all__ = [
    "POWSolution",
    "SubtensorPowRegistration",
    "check_cuda_available",
    "create_seal_hash",
    "hash_block_with_hotkey",
    "seal_meets_difficulty",
    "solve_pow_cuda",
    "solve_pow_parallel",
    "solve_pow_single_thread",
    "verify_pow_solution",
]
