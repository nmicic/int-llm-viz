#!/usr/bin/env python3
# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
"""derive_geometry.py — derived display geometry for the space page.

Reads the committed model.mgw, and for the 27 wte rows (27 tokens x 32
residual-stream coordinates) computes:

  1. An EXACT integer scatter matrix of the centered rows. To stay in
     integers, rows are scaled by V=27 before centering:
         c_i = 27*x_i - S        (S = column sum over the 27 raw rows)
         G   = sum_i c_i c_i^T   (exact Python big-int arithmetic)
     G equals 27^2 * (27 * covariance); constant positive scaling does
     not change eigenvectors or explained-variance ratios.

  2. A floating-point eigendecomposition of G via a cyclic Jacobi sweep
     (pure Python, fixed iteration order, no external dependencies).
     THIS PART IS FLOAT: the eigenvectors are display geometry, not
     model data. Claim boundary: the scatter matrix G is exact and
     portable; the Jacobi output is float display geometry with a fixed
     operation order — deterministic for a given Python build, and
     verified byte-stable in the tested environment (CPython, IEEE-754
     doubles). No stronger cross-platform guarantee is claimed; a
     platform that rounds differently would shift display coordinates
     only, never model data or any exact check.

Sign convention: each eigenvector's largest-|component| entry is made
positive. Components are sorted by eigenvalue, descending.

Output (data/geometry.json): top NUM_COMPONENTS eigenvectors, eigenvalue
ratios, and provenance. The page projects raw Q16.48 values (converted
exactly to float64) onto these float axes at render time and labels the
result as derived geometry.
"""

import json
import math
import struct
import sys
import hashlib
from pathlib import Path

NUM_COMPONENTS = 6
JACOBI_SWEEPS = 30


def parse_mgw(data: bytes):
    magic, version, endian, ntens, idx_off, data_off = struct.unpack_from(
        "<4sIIIQQ", data, 0)
    assert magic == b"MGW\x00" and version == 1 and endian == 0x01020304, \
        "not a little-endian MGW v1 file"
    tensors = {}
    for i in range(ntens):
        off = idx_off + i * 96
        name = data[off:off + 64].split(b"\0")[0].decode()
        ne, doff = struct.unpack_from("<QQ", data, off + 64)
        ndims, s0, s1 = struct.unpack_from("<III", data, off + 80)
        vals = struct.unpack_from(f"<{ne}q", data, doff)
        rows, cols = (s0, s1) if ndims == 2 else (s0, 1)
        tensors[name] = (rows, cols, vals)
    return tensors


def exact_scatter(rows_vals, n_rows, n_cols):
    """G = sum_i (V*x_i - S)(V*x_i - S)^T in exact big-int arithmetic."""
    V = n_rows
    S = [0] * n_cols
    for r in range(n_rows):
        for c in range(n_cols):
            S[c] += rows_vals[r * n_cols + c]
    centered = [[V * rows_vals[r * n_cols + c] - S[c] for c in range(n_cols)]
                for r in range(n_rows)]
    G = [[0] * n_cols for _ in range(n_cols)]
    for row in centered:
        for a in range(n_cols):
            ra = row[a]
            if ra == 0:
                continue
            Ga = G[a]
            for b in range(a, n_cols):
                Ga[b] += ra * row[b]
    for a in range(n_cols):
        for b in range(a):
            G[a][b] = G[b][a]
    return G


def jacobi_eigh(A, sweeps=JACOBI_SWEEPS):
    """Cyclic Jacobi eigendecomposition of a symmetric matrix (floats).

    Fixed (p,q) sweep order => deterministic output for identical input.
    Returns (eigenvalues, eigenvectors) with eigenvectors[k] the k-th
    eigenvector, sorted by eigenvalue descending.
    """
    n = len(A)
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += a[p][q] * a[p][q]
        if off == 0.0:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if apq == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0.0 else -1.0) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    eigs = [(a[i][i], [v[k][i] for k in range(n)]) for i in range(n)]
    eigs.sort(key=lambda e: -e[0])
    return [e[0] for e in eigs], [e[1] for e in eigs]


def main():
    root = Path(__file__).resolve().parent.parent
    mgw_path = root / "third_party" / "int-llm" / "model.mgw"
    out_path = root / "data" / "geometry.json"
    data = mgw_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    tensors = parse_mgw(data)

    rows, cols, vals = tensors["wte"]
    G = exact_scatter(vals, rows, cols)
    # Feed float64(G) to Jacobi. G entries are far below 2^63 here, but
    # they exceed 2^53, so float64(G) rounds. That is acceptable: the
    # output is display geometry. The EXACT G rows are also emitted so
    # the page (or a reviewer) can re-derive everything from integers.
    Gf = [[float(x) for x in row] for row in G]
    eigvals, eigvecs = jacobi_eigh(Gf)

    for vec in eigvecs:
        mx = max(range(len(vec)), key=lambda i: abs(vec[i]))
        if vec[mx] < 0.0:
            for i in range(len(vec)):
                vec[i] = -vec[i]

    total = sum(e for e in eigvals if e > 0.0)
    out = {
        "format": "int-llm-viz-geometry-v1",
        "provenance": {
            "generator": "tools/derive_geometry.py",
            "model_sha256": sha,
            "basis_source": ("wte (27 rows x 32 cols), exact-integer "
                            "centered scatter G = sum (27*x_i - S)(27*x_i - S)^T, "
                            "Jacobi eigendecomposition of float64(G)"),
            "exactness": ("G is exact integer arithmetic; eigenvectors are "
                          "IEEE-754 float64 display geometry (fixed sweep "
                          "order; deterministic in the tested environment, "
                          "not claimed bit-identical across platforms)"),
        },
        "n_rows": rows,
        "n_cols": cols,
        "eigenvalues": eigvals[:NUM_COMPONENTS],
        "explained_ratio": [
            (e / total if total > 0.0 else 0.0) for e in eigvals[:NUM_COMPONENTS]],
        "components": [eigvecs[k] for k in range(NUM_COMPONENTS)],
        "scatter_exact": [[str(x) for x in row] for row in G],
    }
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"geometry: wrote {out_path} "
          f"(top-{NUM_COMPONENTS} explained ratios: "
          f"{[round(r, 4) for r in out['explained_ratio']]})")


if __name__ == "__main__":
    sys.exit(main())
