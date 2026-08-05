#!/usr/bin/env python3
# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
"""check_site.py — validate the built site against the project's claims.

Checks (all must pass; exit 1 otherwise):
  1. site/*.html exists for every web/*.html template and is byte-identical
     to a fresh in-process substitution of the current inputs (this also
     proves no leftover __TOKEN__ placeholders and no stale outputs).
  2. Offline self-containment: no <script src>, no external <link>/<img>,
     no @import/url(http...), no fetch/XHR/WebSocket/EventSource/sendBeacon,
     no http(s) resource loads of any kind. (Plain <a href> hyperlinks to
     public repos are allowed: the page renders fully without following them.)
  3. Structural HTML hygiene: <title>, lang attribute, viewport meta,
     SPDX license header; no U+2028/2029 (would break inline JSON-as-JS).
  4. Pages that embed the model contain the exact base64 of model.mgw.
  5. data/trace.json: verification.pass is true, 20 samples, final RNG
     states match, and the embedded copy in built pages is exact.
  6. data/geometry.json: provenance sha256 matches the committed model.
  7. data/layouts.json: source sha256 matches the committed model, four
     variants, each order a complete bijection (deep verification —
     reconstruction, scores, oracle replay — lives in derive_layouts.py).
  8. data/hidden.json: source sha256 matches the committed model, the swap
     order is a complete 0..31 bijection, totals are consistent (deep
     verification lives in derive_hidden.py).
  9. Every weight in model.mgw satisfies |raw| < 2^53 (the float64-display-
     is-exact claim made by the pages).
 10. Repo-owned sources (tools/, web/, tests/) carry SPDX headers.
"""

import base64
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import build_site  # noqa: E402  (reuse the exact substitution logic)

FORBIDDEN_PATTERNS = [
    (r"<script[^>]*\bsrc\s*=", "external <script src>"),
    (r"<img[^>]*\bsrc\s*=\s*[\"'](?!data:)", "non-data: <img src>"),
    (r"<link[^>]*\bhref\s*=\s*[\"'](?!data:)", "non-data: <link href>"),
    (r"@import", "CSS @import"),
    (r"url\(\s*[\"']?https?:", "CSS url(http...)"),
    (r"\bfetch\s*\(", "fetch()"),
    (r"XMLHttpRequest", "XMLHttpRequest"),
    (r"new\s+WebSocket", "WebSocket"),
    (r"EventSource", "EventSource"),
    (r"sendBeacon", "sendBeacon"),
    (r"import\s*\(", "dynamic import()"),
    (r"<iframe", "<iframe>"),
]

failures = []


def check(cond, msg):
    if cond:
        print(f"  ok: {msg}")
    else:
        print(f"  FAIL: {msg}")
        failures.append(msg)


def weights_all_exact_as_float64(mgw: bytes) -> tuple:
    """Re-parse tensor data independently; return (all_exact, n_weights)."""
    ntens, = struct.unpack_from("<I", mgw, 12)
    idx_off, = struct.unpack_from("<Q", mgw, 16)
    limit = 1 << 53
    n = 0
    ok = True
    for i in range(ntens):
        off = idx_off + i * 96
        name = mgw[off:off + 64].split(b"\0")[0].decode()
        if name in ("tokenizer.uchars", "rng.state"):
            continue
        ne, doff = struct.unpack_from("<QQ", mgw, off + 64)
        for v in struct.unpack_from(f"<{ne}q", mgw, doff):
            n += 1
            if abs(v) >= limit:
                ok = False
    return ok, n


def main() -> int:
    mgw = (ROOT / "third_party/int-llm/model.mgw").read_bytes()
    mgw_sha = hashlib.sha256(mgw).hexdigest()
    mgw_b64 = base64.b64encode(mgw).decode("ascii")
    trace_text = (ROOT / "data/trace.json").read_text()
    trace = json.loads(trace_text)
    geometry = json.loads((ROOT / "data/geometry.json").read_text())
    layouts_text = (ROOT / "data/layouts.json").read_text()
    layouts = json.loads(layouts_text)
    hidden_text = (ROOT / "data/hidden.json").read_text()
    hidden = json.loads(hidden_text)

    print("[trace.json]")
    v = trace["verification"]
    check(v["pass"] is True, "verification.pass is true")
    check(v["logit_mismatches"] == 0 and v["kv_mismatches"] == 0
          and v["sample_mismatches"] == 0, "zero mismatches recorded")
    check(v["rng_final_match"] is True, "final RNG states match")
    check(len(v["pristine_samples"]) == 20, "20 pristine samples")
    check(len(trace["samples"]) == 20, "20 traced samples")
    check(all(s["match_pristine"] for s in trace["samples"]),
          "every traced sample text matches the pristine runtime")
    n_steps = sum(len(s["steps"]) for s in trace["samples"])
    check(n_steps == v["steps_compared"],
          f"step count consistent ({n_steps})")

    print("[geometry.json]")
    check(geometry["provenance"]["model_sha256"] == mgw_sha,
          "geometry provenance sha256 matches model.mgw")
    check(len(geometry["components"]) == 6
          and all(len(c) == geometry["n_cols"] for c in geometry["components"]),
          "6 components of n_cols entries each")
    ratios = geometry["explained_ratio"]
    check(all(math.isfinite(r) and r >= 0.0 for r in ratios),
          "every explained ratio finite and non-negative")
    check(0.0 <= sum(ratios) <= 1.000001,
          "explained ratios sum within [0, 1.000001]")

    print("[layouts.json]")
    check(layouts["format"] == "int-llm-viz-layouts-v1",
          "layouts format tag")
    check(layouts["provenance"]["source_sha256"] == mgw_sha,
          "layouts provenance sha256 matches model.mgw")
    lv = layouts["variants"]
    check([v["name"] for v in lv] == ["original", "smooth", "spread",
                                      "random"], "four layout variants")
    check(all(sorted(v["new_to_old"]) == list(range(128)) for v in lv),
          "every new_to_old is a complete 0..127 bijection")
    check(lv[0]["model_sha256"] == mgw_sha,
          "original variant is the committed checkpoint itself")

    print("[hidden.json]")
    check(hidden["format"] == "int-llm-viz-hidden-v1", "hidden format tag")
    check(hidden["provenance"]["source_sha256"] == mgw_sha,
          "hidden provenance sha256 matches model.mgw")
    check(sorted(hidden["swap"]["new_to_old"]) == list(range(32)),
          "swap new_to_old is a complete 0..31 bijection")
    check(re.fullmatch(r"[0-9a-f]{64}", hidden["variant_sha256"]) is not None
          and hidden["variant_sha256"] != mgw_sha,
          "variant sha256 well-formed and distinct from the original")
    check(len(hidden["tensors"]) == 9
          and sum(t["changed_positions"] for t in hidden["tensors"])
          == hidden["totals"]["changed_positions"]
          and sum(t["elements"] for t in hidden["tensors"])
          == hidden["totals"]["learned_positions"],
          "nine tensors; per-tensor counts sum to the recorded totals")

    print("[model.mgw]")
    all_exact, n_weights = weights_all_exact_as_float64(mgw)
    check(all_exact, f"all {n_weights} weights |raw| < 2^53 (float64-exact)")
    check(n_weights == trace["meta"]["num_params"],
          "weight count matches trace meta")

    print("[site pages]")
    trace_strip = trace_text.strip()
    geometry_strip = (ROOT / "data/geometry.json").read_text().strip()
    layouts_strip = layouts_text.strip()
    hidden_strip = hidden_text.strip()
    summary = json.dumps(build_site.build_summary(mgw, trace), sort_keys=True)
    subst = {
        "THEME_CSS": (ROOT / "web/shared/theme.css").read_text(),
        "MGW_JS": (ROOT / "web/shared/mgw.js").read_text(),
        "MGW_B64": mgw_b64,
        "TRACE_JSON": trace_strip,
        "GEOMETRY_JSON": geometry_strip,
        "LAYOUTS_JSON": layouts_strip,
        "HIDDEN_JSON": hidden_strip,
        "SUMMARY_JSON": summary,
    }
    templates = sorted((ROOT / "web").glob("*.html"))
    check(len(templates) >= 4, f"{len(templates)} templates present (>= 4)")
    for t in templates:
        name = t.name
        raw = t.read_text()
        check("<!--UNBUILT-NOTICE-->" in raw,
              f"{name}: template carries the unbuilt-source notice")
        expect = build_site.substitute(raw, subst)
        built_path = ROOT / "site" / name
        if not built_path.exists():
            check(False, f"{name}: built page exists")
            continue
        built = built_path.read_text()
        check(built == expect, f"{name}: byte-identical to fresh substitution")
        check(not re.search(r"__(THEME_CSS|MGW_JS|MGW_B64|TRACE_JSON|"
                            r"GEOMETRY_JSON|LAYOUTS_JSON|HIDDEN_JSON|"
                            r"SUMMARY_JSON)__",
                            built),
              f"{name}: no leftover tokens")
        check("UNBUILT-NOTICE" not in built,
              f"{name}: unbuilt-source notice stripped from built page")
        for pat, label in FORBIDDEN_PATTERNS:
            if re.search(pat, built):
                check(False, f"{name}: forbidden {label}")
        check("\u2028" not in built and "\u2029" not in built,
              f"{name}: no U+2028/U+2029")
        check("<title>" in built, f"{name}: has <title>")
        check(re.search(r"<html[^>]*\blang\s*=", built) is not None,
              f"{name}: html lang attribute")
        check('name="viewport"' in built, f"{name}: viewport meta")
        check("SPDX-License-Identifier: Apache-2.0" in built,
              f"{name}: SPDX header")
        if "__MGW_B64__" in raw:
            check(mgw_b64 in built, f"{name}: embeds exact model.mgw base64")
        if "__TRACE_JSON__" in raw:
            check(trace_strip in built, f"{name}: embeds exact trace.json")
        if "__GEOMETRY_JSON__" in raw:
            check(geometry_strip in built, f"{name}: embeds exact geometry.json")
        if "__LAYOUTS_JSON__" in raw:
            check(layouts_strip in built, f"{name}: embeds exact layouts.json")
        if "__HIDDEN_JSON__" in raw:
            check(hidden_strip in built, f"{name}: embeds exact hidden.json")
        check(len(built) < 8 * 1024 * 1024, f"{name}: under 8 MB")

    print("[source headers]")
    owned = []
    for pat in ("tools/*.py", "tools/*.c", "web/*.html", "web/shared/*",
                "tests/*.py", "tests/*.js", "tests/*.sh"):
        owned += sorted(ROOT.glob(pat))
    for p in owned:
        text = p.read_text()
        check("SPDX-License-Identifier: Apache-2.0" in text
              and "Copyright 2026 Nenad" in text,
              f"{p.relative_to(ROOT)}: SPDX + copyright header")

    print()
    if failures:
        print(f"CHECK FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("CHECK PASSED: site is self-contained, exact, and consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
