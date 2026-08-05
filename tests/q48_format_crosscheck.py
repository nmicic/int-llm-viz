#!/usr/bin/env python3
# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
"""q48_format_crosscheck.py — verify mgw.js exact decimal formatting.

Ground truth: Python fractions.Fraction, which renders raw/2^48 (and
raw/2^96) exactly. The JS side is web/shared/mgw.js Q48.toDecimalString /
Q48.fixedToDecimalString, executed via node. Every case must match
character-for-character.

Case selection: structured edge cases plus a fixed-seed pseudo-random
sample, plus real extremes from the committed model.mgw.
"""

import json
import random
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def exact_decimal(raw: int, frac_bits: int) -> str:
    """Exact terminating decimal of raw / 2**frac_bits, trailing zeros trimmed."""
    fr = Fraction(raw, 1 << frac_bits)
    sign = "-" if fr < 0 else ""
    fr = abs(fr)
    ip = fr.numerator // fr.denominator
    rem = fr - ip
    if rem == 0:
        return f"{sign}{ip}"
    # rem's denominator divides 2^frac_bits, and 2^k | 10^k, so this
    # division is exact (no truncation): the full terminating expansion.
    scaled = rem.numerator * 10 ** frac_bits // rem.denominator
    digits = str(scaled).rjust(frac_bits, "0").rstrip("0")
    return f"{sign}{ip}.{digits}"


def model_extremes():
    mgw = (ROOT / "third_party/int-llm/model.mgw").read_bytes()
    ntens, = struct.unpack_from("<I", mgw, 12)
    idx_off, = struct.unpack_from("<Q", mgw, 16)
    vals = []
    for i in range(ntens):
        off = idx_off + i * 96
        name = mgw[off:off + 64].split(b"\0")[0].decode()
        if name in ("tokenizer.uchars", "rng.state"):
            continue
        ne, doff = struct.unpack_from("<QQ", mgw, off + 64)
        t = struct.unpack_from(f"<{ne}q", mgw, doff)
        vals.append(min(t))
        vals.append(max(t))
        vals.append(t[0])
    return vals


def main() -> int:
    fp_one = 1 << 48
    cases48 = [
        0, 1, -1, 2, -2, fp_one, -fp_one, fp_one + 1, fp_one - 1,
        fp_one // 2, -(fp_one // 2), 3 * fp_one // 4,
        (1 << 53) - 1, -(1 << 53), (1 << 62), -(1 << 63),
        99516432383215,            # ATTN_SCALE = fp_inv_sqrt(8)
        562949953421312,           # inv_temperature for T = 0.5
        281474976710640,           # the 16-ULP-under-1 probs total
    ] + model_extremes()
    rng = random.Random(20260707)
    cases48 += [rng.randrange(-(1 << 62), 1 << 62) for _ in range(64)]

    cases96 = [0, 1, -1, (1 << 96), -(1 << 96), (1 << 96) + 12345,
               rng.randrange(-(1 << 120), 1 << 120),
               rng.randrange(-(1 << 120), 1 << 120)]

    js = f"""
    "use strict";
    const {{ Q48 }} = require({json.dumps(str(ROOT / "web/shared/mgw.js"))});
    const c48 = {json.dumps([str(v) for v in cases48])};
    const c96 = {json.dumps([str(v) for v in cases96])};
    const out = {{
      d48: c48.map(s => Q48.toDecimalString(BigInt(s))),
      d96: c96.map(s => Q48.fixedToDecimalString(BigInt(s), 96)),
      exact: c48.map(s => Q48.isExactAsNumber(BigInt(s))),
    }};
    process.stdout.write(JSON.stringify(out));
    """
    got = json.loads(subprocess.run(
        ["node", "-e", js], capture_output=True, check=True, text=True).stdout)

    bad = 0
    for v, s in zip(cases48, got["d48"]):
        want = exact_decimal(v, 48)
        if s != want:
            print(f"FAIL Q16.48 {v}: js={s!r} py={want!r}")
            bad += 1
    for v, s in zip(cases96, got["d96"]):
        want = exact_decimal(v, 96)
        if s != want:
            print(f"FAIL Q32.96 {v}: js={s!r} py={want!r}")
            bad += 1
    for v, e in zip(cases48, got["exact"]):
        want = abs(v) < (1 << 53)
        if e != want:
            print(f"FAIL isExactAsNumber {v}: js={e} py={want}")
            bad += 1

    if bad:
        print(f"q48_format_crosscheck: {bad} mismatches")
        return 1
    n = len(cases48) * 2 + len(cases96)
    print(f"q48_format_crosscheck: {n} comparisons, all exact matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
