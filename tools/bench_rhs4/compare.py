#!/usr/bin/env python3
"""Compare two `bench_rhs4 check` outputs: per case, the number of differing values and the
largest difference relative to the case's largest magnitude.

    compare.py old.bin new.bin
"""
import struct
import sys

NI, NJ, NK = 16, 15, 21          # the check grid in bench_rhs4.C
N = 3 * NI * NJ * NK
CASES = [(kern, os) for kern in ("rhs4th3fort", "rhs4th3fortsgstr") for os in ((0, 0), (1, 0), (0, 1), (1, 1))]


def load(path):
    data = open(path, "rb").read()
    return struct.unpack("%df" % (len(data) // 4), data)


a, b = load(sys.argv[1]), load(sys.argv[2])
worst = 0.0
for c, (kern, os) in enumerate(CASES):
    x, y = a[c * N:(c + 1) * N], b[c * N:(c + 1) * N]
    scale = max(abs(v) for v in x) or 1.0
    ndiff = sum(1 for p, q in zip(x, y) if p != q)
    rel = max(abs(p - q) for p, q in zip(x, y)) / scale
    worst = max(worst, rel)
    print(f"{kern:17s} onesided {os}: {ndiff:5d} of {N} values differ, max |diff|/max|value| = {rel:.2e}")
print(f"worst: {worst:.2e}")
