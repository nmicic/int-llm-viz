#!/usr/bin/env python3
# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
"""derive_hidden.py — verify the vendored hidden-coordinate swap, emit data/hidden.json.

The sibling coordinate-permutation repo's hidden-basis experiment shows the
smallest legal whole-model move on this checkpoint: one global transposition
of residual coordinates 0 and 31. Two arbitrary scalar weights cannot be
exchanged without changing the model function; one residual-basis
transposition can — provided the SAME two coordinate lanes move together
through every learned matrix that touches the residual stream:

    hidden COLUMN lanes: wte, wpe, lm_head, attn_wq/wk/wv, mlp_fc1   (7)
    hidden ROW lanes:    attn_wo, mlp_fc2                            (2)

Physical index j holds original coordinate new_to_old[j]; attention-head
coordinates and MLP neuron order stay canonical. Everything the vendored
order document (third_party/hidden-swap/) claims is re-proved here:

  1. The order document validates structurally: exact key set, format tag,
     source_sha256 == the committed model, a complete 0..31 bijection that
     is exactly the (0 31) transposition.
  2. The variant model.mgw is reconstructed from the committed original in
     pure Python (whole 8-byte lanes move; no value changes); its sha256
     must equal the vendored manifest's recorded value, and every
     byte outside the nine learned-tensor regions must be identical.
  3. Changed scalar positions are counted exactly per tensor (the swap
     touches 9/9 learned matrices) and FNV-1a64 checksums of each permuted
     tensor are computed for the atlas's in-page re-verification.
  4. This repo's own C oracle (build/trace_harness) replays the variant:
     sampled text, logits, probabilities, RNG draws, per-head attention,
     q/k/v, and h_pre must be byte-identical to data/trace.json, while the
     six residual-basis vectors recorded per step (emb, x0, xn_attn,
     x_mid, xn_mlp, x_out) must each be exactly the original's reindexed
     by new_to_old — the swap relabels residual coordinates, nothing else.
  5. data/hidden.json is emitted for page embedding.

Deterministic: same inputs -> byte-identical data/hidden.json.
"""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "third_party/int-llm/model.mgw"
ORDER = ROOT / "third_party/hidden-swap/hidden-order.json"
VIZ_MANIFEST = ROOT / "third_party/hidden-swap/viz-manifest.json"
TRACE = ROOT / "data/trace.json"
HARNESS = ROOT / "build/trace_harness"
OUT = ROOT / "data/hidden.json"
WORK = ROOT / "build/hidden"

HIDDEN_DIM = 32
COLUMN_TENSORS = ("wte", "wpe", "lm_head", "layers.0.attn_wq",
                  "layers.0.attn_wk", "layers.0.attn_wv", "layers.0.mlp_fc1")
ROW_TENSORS = ("layers.0.attn_wo", "layers.0.mlp_fc2")
IDENTICAL_STEP_FIELDS = (
    "pos", "token_in", "scale_init", "scale_attn", "scale_mlp", "q", "k",
    "v", "attn", "ao", "h_pre", "logits", "logits_scaled", "probs",
    "prob_total", "rng_before", "r", "chosen", "checks",
)
PERMUTED_STEP_FIELDS = ("emb", "x0", "xn_attn", "x_mid", "xn_mlp", "x_out")

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


def reconstruct(mgw: bytes, tensors: dict, n2o: list) -> bytes:
    """Physical hidden index j holds original coordinate n2o[j]: column
    tensors permute column lanes, row tensors permute whole rows."""
    out = bytearray(mgw)
    for name in COLUMN_TENSORS:
        rows, cols, doff = tensors[name]
        for r in range(rows):
            base = doff + r * cols * 8
            for j, src in enumerate(n2o):
                out[base + j * 8: base + (j + 1) * 8] = \
                    mgw[base + src * 8: base + (src + 1) * 8]
    for name in ROW_TENSORS:
        rows, cols, doff = tensors[name]
        row_bytes = cols * 8
        for j, src in enumerate(n2o):
            out[doff + j * row_bytes: doff + (j + 1) * row_bytes] = \
                mgw[doff + src * row_bytes: doff + (src + 1) * row_bytes]
    return bytes(out)


def main() -> int:
    mgw = MODEL.read_bytes()
    model_sha = sha256(mgw)
    tensors = parse_tensors(mgw)

    print("[order document]")
    doc = json.loads(ORDER.read_text())
    check(set(doc) == {"format", "source_sha256", "family", "hidden_dim",
                       "new_to_old", "metadata"},
          "order doc has exactly the expected top-level keys")
    check(doc["format"] == "int-llm-hidden-order-v1",
          "format int-llm-hidden-order-v1")
    check(doc["source_sha256"] == model_sha,
          "source_sha256 matches the committed model.mgw")
    check(doc["family"] == "microgpt" and doc["hidden_dim"] == HIDDEN_DIM,
          f"family microgpt, hidden_dim {HIDDEN_DIM}")
    n2o = doc["new_to_old"]
    check(sorted(n2o) == list(range(HIDDEN_DIM)),
          f"new_to_old is a complete 0..{HIDDEN_DIM - 1} bijection")
    meta = doc["metadata"]
    left, right = meta.get("coordinates", (None, None))
    check(meta.get("operation") == "transposition" and (left, right) == (0, 31),
          "declared operation: transposition of coordinates 0 and 31")
    expected = list(range(HIDDEN_DIM))
    expected[left], expected[right] = expected[right], expected[left]
    check(n2o == expected,
          "new_to_old is exactly that transposition (self-inverse)")

    print("[tensor plan]")
    for name in COLUMN_TENSORS:
        rows, cols, _ = tensors[name]
        check(cols == HIDDEN_DIM,
              f"{name}: {rows}x{cols}, hidden axis = columns")
    for name in ROW_TENSORS:
        rows, cols, _ = tensors[name]
        check(rows == HIDDEN_DIM,
              f"{name}: {rows}x{cols}, hidden axis = rows")
    plan_names = set(COLUMN_TENSORS) | set(ROW_TENSORS)
    check(set(tensors) - plan_names == {"tokenizer.uchars", "rng.state"},
          "the plan covers every learned tensor (9/9); only the two "
          "non-learned tensors remain")

    print("[reconstruction]")
    manifest = json.loads(VIZ_MANIFEST.read_text())
    check(manifest["format"] == "int-llm-hidden-swap-detail-manifest-v1"
          and manifest["original_sha256"] == model_sha,
          "viz manifest format + original sha256 match")
    variant_sha = manifest["variant_sha256"]
    rebuilt = reconstruct(mgw, tensors, n2o)
    check(sha256(rebuilt) == variant_sha,
          "reconstructed variant sha256 matches the vendored manifest's "
          f"recorded value ({variant_sha[:12]}…)")
    check(reconstruct(rebuilt, tensors, n2o) == mgw,
          "applying the transposition again restores the original bytes "
          "(byte-exact inverse)")
    spans = [(doff, doff + rows * cols * 8)
             for name, (rows, cols, doff) in tensors.items()
             if name in plan_names]
    outside_same = all(
        mgw[i] == rebuilt[i]
        for i in range(len(mgw))
        if not any(a <= i < b for a, b in spans))
    check(outside_same, "every byte outside the nine learned-tensor data "
          "regions is identical (header, index, tokenizer.uchars, rng.state)")

    per_tensor = []
    total_changed = 0
    total_positions = 0
    for name in COLUMN_TENSORS + ROW_TENSORS:
        rows, cols, doff = tensors[name]
        orig = tensor_ints(mgw, rows, cols, doff)
        perm = tensor_ints(rebuilt, rows, cols, doff)
        changed = sum(1 for a, b in zip(orig, perm) if a != b)
        total_changed += changed
        total_positions += rows * cols
        check(sorted(orig) == sorted(perm),
              f"{name}: identical value multiset ({changed}/{rows * cols} "
              "positions changed)")
        per_tensor.append({
            "name": name,
            "rows": rows,
            "cols": cols,
            "axis": "column" if name in COLUMN_TENSORS else "row",
            "changed_positions": changed,
            "elements": rows * cols,
            "fnv1a64_original": str(fnv1a64(orig)),
            "fnv1a64_variant": str(fnv1a64(perm)),
        })
    check(total_changed == 892 and total_positions == 14272,
          f"changed scalar positions total {total_changed}/{total_positions} "
          "(the sibling experiment's published count)")

    print("[oracle replay]")
    if not HARNESS.exists():
        print(f"FAIL: {HARNESS} missing — run `make build/trace_harness`")
        return 1
    base_trace = json.loads(TRACE.read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "model.mgw").write_bytes(rebuilt)
    res = subprocess.run(
        [str(HARNESS), str(WORK / "model.mgw"), str(WORK / "trace.json")],
        capture_output=True, text=True)
    check(res.returncode == 0, "C oracle replay exits 0 (self-verifying)")
    var_trace = json.loads((WORK / "trace.json").read_text())

    bm, vm = base_trace["meta"], var_trace["meta"]
    check(all(bm[k] == vm[k] for k in bm if k != "tensor_fnv1a64"),
          "trace meta identical (config, rng, params)")
    vf = vm["tensor_fnv1a64"]
    check(all(vf[t["name"]] == t["fnv1a64_variant"] for t in per_tensor),
          "all nine Python-permuted tensor FNVs match the C loader's")
    check(all(bm["tensor_fnv1a64"][t["name"]] == t["fnv1a64_original"]
              for t in per_tensor),
          "all nine original tensor FNVs match the C loader's")
    check(var_trace["verification"]["pass"] is True,
          "variant harness verification passes")
    check(var_trace["verification"]["pristine_samples"]
          == base_trace["verification"]["pristine_samples"],
          "identical 20 sampled strings")

    steps_equal = perm_ok = total = 0
    for bs, vs in zip(base_trace["samples"], var_trace["samples"]):
        if (bs["text"], bs["pristine_text"]) != (vs["text"],
                                                 vs["pristine_text"]):
            break
        for b, w in zip(bs["steps"], vs["steps"]):
            total += 1
            if all(b[k] == w[k] for k in IDENTICAL_STEP_FIELDS):
                steps_equal += 1
            if all(w[f][j] == b[f][src]
                   for f in PERMUTED_STEP_FIELDS
                   for j, src in enumerate(n2o)):
                perm_ok += 1
    check(steps_equal == total == 122,
          f"all {total} steps identical in every head-basis and output "
          "field (q/k/v, attention, h_pre, logits, probs, RNG draws)")
    check(perm_ok == total,
          "every step's six residual-basis vectors (emb, x0, xn_attn, "
          "x_mid, xn_mlp, x_out) are exactly the original's, reindexed "
          "by new_to_old")

    print()
    if failures:
        print(f"HIDDEN-SWAP CHECK FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1

    out = {
        "format": "int-llm-viz-hidden-v1",
        "provenance": {
            "generator": ("experiments/hidden_channel_permutation/"
                          "make_hidden_swap.py --left 0 --right 31 "
                          "(sibling int-llm-coordinate-permutation repo)"),
            "verified_by": "tools/derive_hidden.py",
            "source_sha256": model_sha,
        },
        "swap": {
            "left": left,
            "right": right,
            "new_to_old": n2o,
        },
        "variant_sha256": variant_sha,
        "tensors": per_tensor,
        "totals": {
            "changed_positions": total_changed,
            "learned_positions": total_positions,
            "matrices_changed": len(per_tensor),
        },
        "nonlearned_same": ["tokenizer.uchars", "rng.state"],
        "claim_boundary": [
            "One global residual-basis transposition, not two isolated "
            "weight edits: the same two coordinate lanes move together "
            "through all nine learned matrices, and the model function "
            "is preserved exactly.",
            "The heatmaps show physical checkpoint layout; no acceleration "
            "or performance claim is made.",
        ],
        "oracle_replay": ("this repo's C trace harness replayed the "
                          "reconstructed variant: 20 samples, 122 steps — "
                          "text, logits, probs, RNG draws, q/k/v, attention "
                          "and h_pre byte-identical; the six residual-basis "
                          "vectors per step exactly reindexed by new_to_old"),
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"hidden: wrote {OUT} ({OUT.stat().st_size:,} bytes; swap 0<->31 "
          "verified against the vendored order)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
