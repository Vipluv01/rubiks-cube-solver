"""Coordinate encoding for the corner subproblem, used both to build the
pattern database and to look a state up in it.

A corner state is (permutation of 8, orientation of 8 with values 0..2,
constrained so the 8 orientations sum to 0 mod 3). It is encoded as a single
integer in [0, 8! * 3^7) = [0, 88_179_840):

    index = perm_rank(cp) * 2187 + ori_rank(co)

where perm_rank is the Lehmer-code rank of the permutation (0..40319) and
ori_rank is the base-3 value of the first 7 orientations (the 8th is
redundant, fixed by the sum-to-0-mod-3 constraint).
"""
import numpy as np

N_CORNERS = 8
N_PERM = 40320       # 8!
N_ORI = 2187         # 3**7
N_CORNER_STATES = N_PERM * N_ORI  # 88_179_840

_FACT = [1] * (N_CORNERS + 1)
for _i in range(1, N_CORNERS + 1):
    _FACT[_i] = _FACT[_i - 1] * _i


def perm_rank(perm) -> int:
    # Plain nested loop, no generator/sum() call overhead -- this runs on
    # every node IDA* visits, so its constant factor matters.
    n = len(perm)
    rank = 0
    for i in range(n):
        pi = perm[i]
        smaller = 0
        for j in range(i + 1, n):
            if perm[j] < pi:
                smaller += 1
        rank += smaller * _FACT[n - 1 - i]
    return rank


def perm_unrank(rank: int, n: int = N_CORNERS):
    items = list(range(n))
    perm = []
    for i in range(n):
        f = _FACT[n - 1 - i]
        idx, rank = divmod(rank, f)
        perm.append(items.pop(idx))
    return perm


def ori_rank(co) -> int:
    r = 0
    for v in co[:7]:
        r = r * 3 + int(v)
    return r


def ori_unrank(rank: int):
    digits = [0] * 7
    for i in range(6, -1, -1):
        digits[i] = rank % 3
        rank //= 3
    last = (-sum(digits)) % 3
    return digits + [last]


def encode_corner(cp, co) -> int:
    return perm_rank(cp) * N_ORI + ori_rank(co)


def decode_corner(index: int):
    p_rank, o_rank = divmod(index, N_ORI)
    cp = np.array(perm_unrank(p_rank), dtype=np.int64)
    co = np.array(ori_unrank(o_rank), dtype=np.int64)
    return cp, co


def corner_coord(cube) -> int:
    return encode_corner(cube.cp, cube.co)
