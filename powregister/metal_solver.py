"""Apple Silicon (Metal) PoW solver — the GPU backend for Macs.

cubit, the CUDA solver powregister already supports, is NVIDIA-only, which left Apple
machines on the pure-Python CPU path (~110 kH/s per core). On an M4 Max that is ~1.3 MH/s,
i.e. roughly a week of wall clock for a subnet at difficulty ~9.4e11. This module runs the
identical seal computation as `core.create_seal_hash` inside a Metal compute kernel, one
nonce per GPU thread, which brings the same machine into the minutes range.

Seal (verified against core.create_seal_hash):

    seal = keccak256( sha256( nonce_le64 || block_and_hotkey_hash ) )

and the chain's test is `int.from_bytes(seal, "little") * difficulty < 2**256`, i.e.
seal_le < 2**256 // difficulty. Since that limit is astronomically larger than 2**192 for
any realistic difficulty, the kernel only screens on the top 64 bits of the little-endian
seal (bytes 24..31) and reports candidates; the caller re-checks each candidate exactly in
Python, so a screening false positive can never produce an invalid submission.
"""

from __future__ import annotations

import struct
from typing import Optional

METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint K256[64] = {
0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

constant ulong RC[24] = {
0x0000000000000001UL,0x0000000000008082UL,0x800000000000808aUL,0x8000000080008000UL,
0x000000000000808bUL,0x0000000080000001UL,0x8000000080008081UL,0x8000000000008009UL,
0x000000000000008aUL,0x0000000000000088UL,0x0000000080008009UL,0x000000008000000aUL,
0x000000008000808bUL,0x800000000000008bUL,0x8000000000008089UL,0x8000000000008003UL,
0x8000000000008002UL,0x8000000000000080UL,0x000000000000800aUL,0x800000008000000aUL,
0x8000000080008081UL,0x8000000000008080UL,0x0000000080000001UL,0x8000000080008008UL};

// rho offsets, lane index = x + 5*y
constant uint ROT[25] = {
 0, 1,62,28,27,
36,44, 6,55,20,
 3,10,43,25,39,
41,45,15,21, 8,
18, 2,61,56,14};

inline uint rotr32(uint x, uint n) { return (x >> n) | (x << (32u - n)); }
inline ulong rotl64(ulong x, uint n) { return n == 0 ? x : ((x << n) | (x >> (64u - n))); }

// SHA-256 of exactly 40 bytes -> 32-byte digest (single padded block)
inline uint bswap32(uint x) {
    return ((x >> 24) & 0xffu) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}

// SHA-256 of nonce_le64 || bh (exactly 40 bytes). The message words are built straight from
// the inputs: no thread-local byte array, which is what was throttling the first version.
inline void sha256_msg(ulong nonce, thread const uint *bh, thread uint *H) {
    uint w[64];
    w[0] = bswap32((uint)(nonce & 0xffffffffUL));
    w[1] = bswap32((uint)(nonce >> 32));
    for (uint i = 0; i < 8; i++) w[2+i] = bswap32(bh[i]);
    w[10] = 0x80000000u;                 // padding byte right after the 40-byte message
    w[11] = 0; w[12] = 0; w[13] = 0; w[14] = 0;
    w[15] = 320u;                        // bit length
    for (uint i = 16; i < 64; i++) {
        uint s0 = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^ (w[i-15] >> 3);
        uint s1 = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint a=0x6a09e667u,b=0xbb67ae85u,c=0x3c6ef372u,d=0xa54ff53au;
    uint e=0x510e527fu,f=0x9b05688cu,g=0x1f83d9abu,h=0x5be0cd19u;
    for (uint i = 0; i < 64; i++) {
        uint S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        uint ch = (e & f) ^ ((~e) & g);
        uint t1 = h + S1 + ch + K256[i] + w[i];
        uint S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        uint mj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]=0x6a09e667u+a; H[1]=0xbb67ae85u+b; H[2]=0x3c6ef372u+c; H[3]=0xa54ff53au+d;
    H[4]=0x510e527fu+e; H[5]=0x9b05688cu+f; H[6]=0x1f83d9abu+g; H[7]=0x5be0cd19u+h;
}

inline void keccakf(thread ulong *A) {
    for (uint r = 0; r < 24; r++) {
        ulong C[5], D[5], B[25];
        for (uint x = 0; x < 5; x++)
            C[x] = A[x] ^ A[x+5] ^ A[x+10] ^ A[x+15] ^ A[x+20];
        for (uint x = 0; x < 5; x++)
            D[x] = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
        for (uint y = 0; y < 5; y++)
            for (uint x = 0; x < 5; x++)
                A[x+5*y] ^= D[x];
        for (uint y = 0; y < 5; y++)
            for (uint x = 0; x < 5; x++)
                B[y + 5*((2*x+3*y)%5)] = rotl64(A[x+5*y], ROT[x+5*y]);
        for (uint y = 0; y < 5; y++)
            for (uint x = 0; x < 5; x++)
                A[x+5*y] = B[x+5*y] ^ ((~B[(x+1)%5 + 5*y]) & B[(x+2)%5 + 5*y]);
        A[0] ^= RC[r];
    }
}

// Keccak-256 (original 0x01 padding, NOT SHA3) of the 32-byte SHA-256 digest.
// Returns only lane 3: the seal's little-endian bytes 24..31, which is exactly the word the
// difficulty screen needs — the other three lanes never have to leave the GPU.
inline ulong keccak256_top(thread const uint *H) {
    ulong A[25];
    for (uint i = 0; i < 25; i++) A[i] = 0;
    // digest bytes are big-endian per H word; lane j packs H[2j], H[2j+1] little-endian
    for (uint j = 0; j < 4; j++)
        A[j] = (ulong)bswap32(H[2*j]) | ((ulong)bswap32(H[2*j+1]) << 32);
    A[4]  = 0x01UL;                      // padding start, right after 32 message bytes
    A[16] = 0x8000000000000000UL;        // rate-block terminator (byte 135 |= 0x80)
    keccakf(A);
    return A[3];
}

struct Params {
    uint  bh[8];        // block_and_hotkey_hash as 8 LE-packed u32 words
    ulong nonce_start;
    ulong threshold;    // top 64 bits (LE bytes 24..31) must be < this
};

kernel void solve(const device Params &p       [[buffer(0)]],
                  device atomic_uint *found    [[buffer(1)]],
                  device ulong *out_nonce      [[buffer(2)]],
                  uint tid                     [[thread_position_in_grid]])
{
    ulong nonce = p.nonce_start + (ulong)tid;

    uint bh[8];
    for (uint i = 0; i < 8; i++) bh[i] = p.bh[i];

    uint H[8];
    sha256_msg(nonce, bh, H);
    ulong top = keccak256_top(H);

    if (top < p.threshold) {
        uint slot = atomic_fetch_add_explicit(found, 1u, memory_order_relaxed);
        if (slot < 16u) out_nonce[slot] = nonce;
    }
}
"""


class MetalSolver:
    """One Metal device, one compiled kernel, reused across batches."""

    MAX_HITS = 16

    def __init__(self):
        import metalcompute as mc

        self.mc = mc
        self.dev = mc.Device()
        self.fn = self.dev.kernel(METAL_SOURCE).function("solve")

    @staticmethod
    def threshold(difficulty: int) -> int:
        """Screen bound for the seal's high word: seal_le < 2**256 // difficulty implies
        (seal_le >> 192) < 2**64 // difficulty. Screening on the latter never misses a
        valid seal; false positives are re-checked exactly by the caller."""
        return (1 << 64) // int(difficulty)

    def search(self, block_and_hotkey_hash: bytes, difficulty: int, nonce_start: int, count: int) -> list[int]:
        """Hash `count` consecutive nonces on the GPU; return candidate nonces."""
        assert len(block_and_hotkey_hash) == 32
        words = struct.unpack("<8I", block_and_hotkey_hash)
        params = struct.pack("<8I", *words) + struct.pack("<QQ", nonce_start, self.threshold(difficulty))
        pbuf = self.dev.buffer(params)
        found = self.dev.buffer(4)
        memoryview(found).cast("I")[0] = 0
        out = self.dev.buffer(8 * self.MAX_HITS)
        self.fn(count, pbuf, found, out)
        n = memoryview(found).cast("I")[0]
        if not n:
            return []
        nonces = memoryview(out).cast("Q")
        return [int(nonces[i]) for i in range(min(n, self.MAX_HITS))]


def available() -> bool:
    try:
        import metalcompute  # noqa: F401

        return True
    except Exception:
        return False


def solve_pow_metal(
    block_and_hotkey_hash: bytes,
    difficulty: int,
    block_number: int,
    nonce_start: int = 0,
    batch: int = 1 << 22,
    max_batches: Optional[int] = None,
    verify=None,
    log=None,
):
    """Search for a valid seal on the GPU, verifying every candidate on the CPU."""
    solver = MetalSolver()
    nonce = nonce_start
    batches = 0
    while max_batches is None or batches < max_batches:
        for cand in solver.search(block_and_hotkey_hash, difficulty, nonce, batch):
            if verify is None or verify(cand):
                return cand
        nonce += batch
        batches += 1
        if log:
            log(nonce - nonce_start)
    return None


def solve_pow_metal_round(
    block_and_hotkey_hash: bytes,
    difficulty: int,
    block_number: int,
    timeout: float = 20.0,
    batch: int = 1 << 24,
    solver: Optional["MetalSolver"] = None,
    progress=None,
):
    """One block-bounded search round, shaped like solve_pow_cuda/solve_pow_parallel.

    The caller (core.solve_and_register) re-targets to a fresh block between rounds, so this
    only has to hash as many nonces as `timeout` allows and hand back a POWSolution or None.
    Every GPU candidate is re-verified with the reference CPU seal before it is returned —
    the kernel screens on 64 bits, the chain checks 256.
    """
    import random
    import time as _time

    from .core import POWSolution, create_seal_hash, seal_meets_difficulty

    solver = solver or MetalSolver()
    nonce = random.randint(0, (1 << 64) - 1 - (1 << 40))
    start = _time.time()
    hashed = 0
    while _time.time() - start < timeout:
        for cand in solver.search(block_and_hotkey_hash, difficulty, nonce, batch):
            seal = create_seal_hash(block_and_hotkey_hash, cand)
            if seal_meets_difficulty(seal, difficulty):
                return POWSolution(nonce=cand, block_number=block_number, difficulty=difficulty, seal=seal)
        nonce += batch
        hashed += batch
        if progress:
            progress(hashed, _time.time() - start)
    return None
