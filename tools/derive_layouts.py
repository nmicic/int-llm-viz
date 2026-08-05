#!/usr/bin/env python3
# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
"""derive_layouts.py — verify the vendored MLP layout set, emit data/layouts.json.

The layout set (third_party/layout-set/) was produced by the sibling
coordinate-permutation repo's systematic-MLP-layout tool: three alternative
physical orderings of this checkpoint's 128 MLP neurons (smooth / spread /
random) plus the identity (original), each described by a small order
document. A variant checkpoint is a pure byte permutation of the committed
model.mgw — mlp_fc1 rows and mlp_fc2 column lanes move as whole 8-byte
units; no stored value changes.

Everything the set claims is re-proved inside this repo:

  1. Order documents validate structurally: exact key sets, format tags,
     source_sha256 == the committed model, a complete 0..127 bijection.
  2. Each variant's full model.mgw is reconstructed from the committed
     original in pure Python; its sha256 must equal the generator's own
     record in the vendored summary.json.
  3. Adjacency scores are recomputed exactly (Python ints, no floats):
     profile(n) = fc1 row n ++ fc2 column n (64 signed Q16.48 integers);
     score(order) = sum of L1 distances between physically adjacent
     profiles. Scores, ratios and moved-neuron counts must match the
     vendored records.
  4. This repo's own C oracle (build/trace_harness) replays every
     reconstructed variant: sampled text, logits, probabilities and RNG
     draws must be byte-identical to the original's data/trace.json, and
     each step's 128-wide h_pre vector must be exactly the original's
     permuted by new_to_old — the layout moves coordinates, nothing else.
  5. data/layouts.json is emitted for page embedding: the four orders,
     exact scores, and per-variant FNV-1a64 checksums of the permuted MLP
     tensors (cross-checked against the C loader's own checksums), so the
     atlas can re-verify the permutation it applies in-page.

Deterministic: same inputs -> byte-identical data/layouts.json.
"""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "third_party/int-llm/model.mgw"
SET_DIR = ROOT / "third_party/layout-set"
TRACE = ROOT / "data/trace.json"
HARNESS = ROOT / "build/trace_harness"
OUT = ROOT / "data/layouts.json"
WORK = ROOT / "build/layouts"

VARIANTS = ("original", "smooth", "spread", "random")
FC1, FC2 = "layers.0.mlp_fc1", "layers.0.mlp_fc2"
N_NEURONS = 128

failures = []


def check(cond, msg):
    if cond:
        print(f"  ok: {msg}")
    else:
        print(f"  FAIL: {msg}")
        failures.append(msg)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fnv1a64(values) -> int:
    """FNV-1a 64 over the little-endian 8-byte image of each int64, in
    index order — bit-compatible with tools/trace_harness.c and mgw.js
    (which seed with 1469598103934665603, not the textbook basis)."""
    h = 1469598103934665603
    for v in values:
        u = v & 0xFFFFFFFFFFFFFFFF
        for _ in range(8):
            h = ((h ^ (u & 0xFF)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            u >>= 8
    return h


def parse_tensors(mgw: bytes) -> dict:
    """Minimal independent MGW parse: name -> (rows, cols, data_offset)."""
    ntens, = struct.unpack_from("<I", mgw, 12)
    idx_off, = struct.unpack_from("<Q", mgw, 16)
    out = {}
    for i in range(ntens):
        off = idx_off + i * 96
        name = mgw[off:off + 64].split(b"\0")[0].decode()
        ne, doff = struct.unpack_from("<QQ", mgw, off + 64)
        ndims, s0, s1 = struct.unpack_from("<III", mgw, off + 80)
        rows, cols = s0, (s1 if ndims == 2 else 1)
        if rows * cols != ne:
            raise SystemExit(f"{name}: shape/count mismatch")
        out[name] = (rows, cols, doff)
    return out


def tensor_ints(mgw: bytes, rows: int, cols: int, doff: int) -> list:
    return list(struct.unpack_from(f"<{rows * cols}q", mgw, doff))


def reconstruct(mgw: bytes, tensors: dict, order: list) -> bytes:
    """Apply the neuron order as the generator defines it: physical slot j
    of the variant holds original neuron order[j] (fc1 row / fc2 column)."""
    out = bytearray(mgw)
    r1, c1, d1 = tensors[FC1]          # 128 x 32, row = neuron
    row_bytes = c1 * 8
    for j, src in enumerate(order):
        out[d1 + j * row_bytes: d1 + (j + 1) * row_bytes] = \
            mgw[d1 + src * row_bytes: d1 + (src + 1) * row_bytes]
    r2, c2, d2 = tensors[FC2]          # 32 x 128, column = neuron
    for r in range(r2):
        base = d2 + r * c2 * 8
        for j, src in enumerate(order):
            out[base + j * 8: base + (j + 1) * 8] = \
                mgw[base + src * 8: base + (src + 1) * 8]
    return bytes(out)


def ratio_12(sel: int, orig: int) -> str:
    """selected/original as a 12-decimal string, round half up — matches
    the generator's rendering of the exact integer quotient."""
    scaled, rem = divmod(sel * 10 ** 12, orig)
    if 2 * rem >= orig:
        scaled += 1
    return f"{scaled // 10 ** 12}.{scaled % 10 ** 12:012d}"


def load_order_doc(name: str, model_sha: str) -> dict:
    path = SET_DIR / name / "mlp-order.json"
    doc = json.loads(path.read_text())
    check(set(doc) == {"format", "source_sha256", "layers", "metadata"},
          f"{name}: order doc has exactly the expected top-level keys")
    check(doc["format"] == "int-llm-mlp-order-v1",
          f"{name}: format int-llm-mlp-order-v1")
    check(doc["source_sha256"] == model_sha,
          f"{name}: source_sha256 matches the committed model.mgw")
    check(len(doc["layers"]) == 1, f"{name}: exactly one layer entry")
    layer = doc["layers"][0]
    check(set(layer) == {"family", "layer", "intermediate_dim", "new_to_old"},
          f"{name}: layer entry has exactly the expected keys")
    check(layer["family"] == "microgpt" and layer["layer"] == 0
          and layer["intermediate_dim"] == N_NEURONS,
          f"{name}: layer entry is microgpt layer 0, {N_NEURONS} neurons")
    order = layer["new_to_old"]
    check(sorted(order) == list(range(N_NEURONS)),
          f"{name}: new_to_old is a complete 0..{N_NEURONS - 1} bijection")
    return doc


def compare_variant_trace(name: str, order: list, base_trace: dict,
                          var_trace: dict) -> None:
    """The permuted checkpoint must behave identically except that h_pre
    (fc1 pre-activations) is reindexed by the neuron order."""
    bm, vm = base_trace["meta"], var_trace["meta"]
    same_meta = all(bm[k] == vm[k] for k in bm if k != "tensor_fnv1a64")
    check(same_meta, f"{name}: trace meta identical (config, rng, params)")
    bf, vf = bm["tensor_fnv1a64"], vm["tensor_fnv1a64"]
    check(all(bf[t] == vf[t] for t in bf if t not in (FC1, FC2)),
          f"{name}: all non-MLP tensor checksums identical")
    v = var_trace["verification"]
    check(v["pass"] is True, f"{name}: variant harness verification passes")
    check(v["pristine_samples"] == base_trace["verification"]
          ["pristine_samples"], f"{name}: identical 20 sampled strings")

    steps_equal = h_pre_perm = 0
    total = 0
    for bs, vs in zip(base_trace["samples"], var_trace["samples"]):
        if (bs["text"], bs["pristine_text"]) != (vs["text"],
                                                 vs["pristine_text"]):
            break
        for b, w in zip(bs["steps"], vs["steps"]):
            total += 1
            if all(b[k] == w[k] for k in b if k != "h_pre"):
                steps_equal += 1
            if all(w["h_pre"][j] == b["h_pre"][src]
                   for j, src in enumerate(order)):
                h_pre_perm += 1
    check(steps_equal == total == 122,
          f"{name}: all {total} steps identical outside h_pre "
          "(logits, probs, RNG draws, residuals)")
    check(h_pre_perm == total,
          f"{name}: every step's h_pre is exactly the original's, "
          "reindexed by new_to_old")


def main() -> int:
    mgw = MODEL.read_bytes()
    model_sha = sha256(mgw)
    tensors = parse_tensors(mgw)
    r1, c1, d1 = tensors[FC1]
    r2, c2, d2 = tensors[FC2]

    print("[model]")
    check((r1, c1) == (128, 32), f"{FC1} is 128x32 (row = neuron)")
    check((r2, c2) == (32, 128), f"{FC2} is 32x128 (column = neuron)")

    print("[summary.json]")
    summary = json.loads((SET_DIR / "summary.json").read_text())
    check(summary["format"] == "int-llm-mlp-layout-set-v1",
          "format int-llm-mlp-layout-set-v1")
    check(summary["source_sha256"] == model_sha,
          "summary source_sha256 matches the committed model.mgw")
    check([v["name"] for v in summary["variants"]] == list(VARIANTS),
          "the four expected variants, in order")

    fc1_ints = tensor_ints(mgw, r1, c1, d1)
    fc2_ints = tensor_ints(mgw, r2, c2, d2)
    profiles = [fc1_ints[n * c1:(n + 1) * c1]
                + [fc2_ints[r * c2 + n] for r in range(r2)]
                for n in range(N_NEURONS)]

    def path_score(order):
        return sum(sum(abs(a - b) for a, b in zip(profiles[order[j]],
                                                  profiles[order[j + 1]]))
                   for j in range(N_NEURONS - 1))

    original_score = path_score(list(range(N_NEURONS)))

    if not HARNESS.exists():
        print(f"FAIL: {HARNESS} missing — run `make build/trace_harness`")
        return 1
    base_trace = json.loads(TRACE.read_text())
    WORK.mkdir(parents=True, exist_ok=True)

    out_variants = []
    for spec in summary["variants"]:
        name = spec["name"]
        print(f"[{name}]")
        doc = load_order_doc(name, model_sha)
        order = doc["layers"][0]["new_to_old"]

        rebuilt = reconstruct(mgw, tensors, order)
        check(sha256(rebuilt) == spec["model_sha256"],
              f"{name}: reconstructed model.mgw sha256 matches the "
              "generator's record")

        moved = sum(1 for j, src in enumerate(order) if j != src)
        sel = path_score(order)
        rec = spec["layer_scores"][0]
        doc_rec = doc["metadata"]["layer_scores"][0]
        check(rec == doc_rec,
              f"{name}: summary and order-doc score records agree")
        check(moved == rec["moved_neurons"],
              f"{name}: moved neurons recomputed = {moved}")
        check(str(original_score) == rec["original_score"],
              f"{name}: original adjacency score recomputed exactly")
        check(str(sel) == rec["selected_score"],
              f"{name}: selected adjacency score recomputed exactly")
        check(ratio_12(sel, original_score) == rec["selected_over_original"],
              f"{name}: score ratio recomputed "
              f"({rec['selected_over_original']})")

        fc1_perm = [fc1_ints[src * c1 + k] for src in order for k in range(c1)]
        fc2_perm = [fc2_ints[r * c2 + src] for r in range(r2) for src in order]
        fnv_fc1, fnv_fc2 = fnv1a64(fc1_perm), fnv1a64(fc2_perm)

        if name == "original":
            check(rebuilt == mgw, "original: identity order reproduces the "
                  "committed bytes")
            var_trace = base_trace
        else:
            vdir = WORK / name
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "model.mgw").write_bytes(rebuilt)
            res = subprocess.run(
                [str(HARNESS), str(vdir / "model.mgw"),
                 str(vdir / "trace.json")],
                capture_output=True, text=True)
            check(res.returncode == 0,
                  f"{name}: C oracle replay exits 0 (self-verifying)")
            var_trace = json.loads((vdir / "trace.json").read_text())
        vf = var_trace["meta"]["tensor_fnv1a64"]
        check(vf[FC1] == str(fnv_fc1) and vf[FC2] == str(fnv_fc2),
              f"{name}: Python-permuted tensor FNVs match the C loader's")
        compare_variant_trace(name, order, base_trace, var_trace)

        out_variants.append({
            "name": name,
            "model_sha256": spec["model_sha256"],
            "new_to_old": order,
            "moved_neurons": moved,
            "original_score": str(original_score),
            "selected_score": str(sel),
            "selected_over_original": rec["selected_over_original"],
            "fnv1a64": {FC1: str(fnv_fc1), FC2: str(fnv_fc2)},
        })

    print()
    if failures:
        print(f"LAYOUT CHECK FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1

    smooth_meta = json.loads(
        (SET_DIR / "smooth/mlp-order.json").read_text())["metadata"]
    out = {
        "format": "int-llm-viz-layouts-v1",
        "provenance": {
            "generator": ("systematic_mlp_layout/mlp_layout.py build-set "
                          "(sibling int-llm-coordinate-permutation repo)"),
            "seed": "int-llm-layout-spike-2026-08-02",
            "optimizer": smooth_meta["optimizer"],
            "verified_by": "tools/derive_layouts.py",
            "source_sha256": model_sha,
        },
        "metric": smooth_meta["metric"],
        "claim_boundary": [
            summary["claim_boundary"],
            smooth_meta["claim_boundary"],
        ],
        "microgpt_oracle": summary["microgpt_oracle"],
        "oracle_replay": ("this repo's C trace harness replayed every "
                          "reconstructed variant: 20 samples, 122 steps — "
                          "text, logits, probs and RNG draws byte-identical; "
                          "h_pre exactly reindexed by new_to_old"),
        "variants": out_variants,
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"layouts: wrote {OUT} ({OUT.stat().st_size:,} bytes; "
          f"4 variants verified against the vendored set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
