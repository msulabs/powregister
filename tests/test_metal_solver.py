"""Tests for the Metal (Apple GPU) PoW solver.

The kernel reimplements `core.create_seal_hash` in Metal Shading Language, so the tests that
matter are equivalence tests: whatever the GPU reports must pass the exact CPU check the
chain performs. Everything that touches the GPU is skipped when Metal is unavailable, so the
suite still runs on Linux/CI.
"""

import os

import pytest

from powregister.core import create_seal_hash, seal_meets_difficulty
from powregister.metal_solver import MetalSolver, available, solve_pow_metal_round

metal_only = pytest.mark.skipif(not available(), reason="Metal (Apple GPU) not available")

# Difficulty and range are chosen so a healthy solver finds ~16 solutions per search: at
# difficulty == range the expected count is 1 and a correct solver still comes up empty 37%
# of the time, which would make these tests flaky rather than meaningful.
EASY = 1 << 16
RANGE = 1 << 20


def test_threshold_is_a_safe_screen():
    """The kernel screens on the seal's high 64 bits. That screen must never be tighter than
    the chain's own bound, or valid solutions would be silently discarded."""
    for difficulty in (1 << 16, 10**9, 943_295_763_109):
        limit = (1 << 256) // difficulty
        assert MetalSolver.threshold(difficulty) == limit >> 192


def test_threshold_shrinks_as_difficulty_grows():
    assert MetalSolver.threshold(1 << 16) > MetalSolver.threshold(943_295_763_109)


@metal_only
def test_kernel_candidates_pass_the_chain_check():
    """Every nonce the GPU reports must satisfy the reference seal test."""
    solver = MetalSolver()
    block_hash = os.urandom(32)
    candidates = solver.search(block_hash, EASY, 0, RANGE)
    assert candidates, "expected candidates at difficulty 2**16 over 2**20 nonces"
    for nonce in candidates:
        assert seal_meets_difficulty(create_seal_hash(block_hash, nonce), EASY)


@metal_only
def test_kernel_agrees_with_reference_on_the_screened_word():
    """The kernel compares the seal's little-endian bytes 24..31. Recompute that word on the
    CPU for a reported candidate and confirm it really is below the screen."""
    solver = MetalSolver()
    block_hash = os.urandom(32)
    candidates = solver.search(block_hash, EASY, 0, RANGE)
    assert candidates
    threshold = MetalSolver.threshold(EASY)
    for nonce in candidates:
        seal = create_seal_hash(block_hash, nonce)
        top = int.from_bytes(seal[24:32], "little")
        assert top < threshold


@metal_only
def test_search_reports_the_same_solution_set_for_a_fixed_range():
    """Re-searching a fixed range must surface the same solutions: the kernel keeps no state
    between dispatches.

    Compared as sets, not lists. Reporting order follows an atomic counter, so it tracks the
    order threads happen to finish in and is not stable — and once more than MAX_HITS
    solutions exist, *which* ones win the slots varies too. The difficulty here is chosen to
    keep the expected count (~4) well under that cap so the sets are directly comparable.
    """
    solver = MetalSolver()
    block_hash = os.urandom(32)
    difficulty = 1 << 18
    first = solver.search(block_hash, difficulty, 12345, RANGE)
    second = solver.search(block_hash, difficulty, 12345, RANGE)
    assert len(first) < MetalSolver.MAX_HITS, "test needs a range that cannot overflow the slots"
    assert set(first) == set(second)


@metal_only
def test_no_candidates_when_range_is_tiny_and_difficulty_high():
    """A handful of nonces at production difficulty must not produce hits — a solver that
    reports anything here would be screening on the wrong bytes."""
    solver = MetalSolver()
    assert solver.search(os.urandom(32), 943_295_763_109, 0, 1024) == []


@metal_only
def test_round_returns_a_verified_solution():
    """solve_pow_metal_round hands back a POWSolution whose seal the chain check accepts."""
    block_hash = os.urandom(32)
    solution = solve_pow_metal_round(block_hash, EASY, block_number=999, timeout=20)
    assert solution is not None
    assert solution.block_number == 999
    assert solution.difficulty == EASY
    assert solution.seal == create_seal_hash(block_hash, solution.nonce)
    assert seal_meets_difficulty(solution.seal, EASY)


@metal_only
def test_round_gives_up_within_its_timeout():
    """An impossible difficulty must return None promptly instead of running forever: the
    caller relies on this to refresh the block before the 3-block window closes."""
    import time

    start = time.time()
    solution = solve_pow_metal_round(os.urandom(32), 1 << 62, block_number=1, timeout=3)
    elapsed = time.time() - start
    assert solution is None
    assert elapsed < 20, f"round overran its timeout: {elapsed:.1f}s"
