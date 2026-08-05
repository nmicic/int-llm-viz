#!/usr/bin/env python3
# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
"""build_site.py — assemble self-contained pages from web/ templates.

Each template in web/*.html may contain these substitution tokens:

  __THEME_CSS__      contents of web/shared/theme.css
  __MGW_JS__         contents of web/shared/mgw.js
  __MGW_B64__        base64 of third_party/int-llm/model.mgw
  __TRACE_JSON__     contents of data/trace.json (a JSON object literal)
  __GEOMETRY_JSON__  contents of data/geometry.json
  __LAYOUTS_JSON__   contents of data/layouts.json (MLP layout set)
  __HIDDEN_JSON__    contents of data/hidden.json (hidden-coordinate swap)
  __SUMMARY_JSON__   small build-provenance object (see below)

Templates also carry a visible "unbuilt source template" notice between
<!--UNBUILT-NOTICE--> ... <!--/UNBUILT-NOTICE--> markers so that opening
web/*.html directly explains itself; the build strips that block.

Output: site/<name>.html, byte-for-byte deterministic for identical
inputs (no timestamps, stable ordering). Optionally build one page:
    python3 tools/build_site.py --page atlas
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENS = ("THEME_CSS", "MGW_JS", "MGW_B64", "TRACE_JSON",
          "GEOMETRY_JSON", "LAYOUTS_JSON", "HIDDEN_JSON", "SUMMARY_JSON")
NOTICE_RE = re.compile(r"<!--UNBUILT-NOTICE-->.*?<!--/UNBUILT-NOTICE-->\n?",
                       re.S)


def substitute(text: str, subst: dict) -> str:
    """The one substitution used by both the build and check_site.py."""
    text = NOTICE_RE.sub("", text)
    for name in TOKENS:
        token = f"__{name}__"
        if token in text:
            text = text.replace(token, subst[name])
    return text


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_summary(mgw_bytes: bytes, trace: dict) -> dict:
    v = trace["verification"]
    return {
        "format": "int-llm-viz-summary-v1",
        "model_file": "third_party/int-llm/model.mgw",
        "model_sha256": sha256(mgw_bytes),
        "model_bytes": len(mgw_bytes),
        "num_params": trace["meta"]["num_params"],
        "config": trace["meta"]["config"],
        "uchars": trace["meta"]["uchars"],
        "tensor_fnv1a64": trace["meta"]["tensor_fnv1a64"],
        "trace_pass": v["pass"],
        "trace_steps": v["steps_compared"],
        "samples": v["pristine_samples"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", help="build only this page (basename, no .html)")
    args = ap.parse_args()

    mgw_bytes = (ROOT / "third_party/int-llm/model.mgw").read_bytes()
    trace_text = (ROOT / "data/trace.json").read_text()
    trace = json.loads(trace_text)
    geometry_text = (ROOT / "data/geometry.json").read_text()
    layouts_text = (ROOT / "data/layouts.json").read_text()
    hidden_text = (ROOT / "data/hidden.json").read_text()
    summary = build_summary(mgw_bytes, trace)

    subst = {
        "THEME_CSS": (ROOT / "web/shared/theme.css").read_text(),
        "MGW_JS": (ROOT / "web/shared/mgw.js").read_text(),
        "MGW_B64": base64.b64encode(mgw_bytes).decode("ascii"),
        "TRACE_JSON": trace_text.strip(),
        "GEOMETRY_JSON": geometry_text.strip(),
        "LAYOUTS_JSON": layouts_text.strip(),
        "HIDDEN_JSON": hidden_text.strip(),
        "SUMMARY_JSON": json.dumps(summary, sort_keys=True),
    }

    (ROOT / "site").mkdir(exist_ok=True)
    templates = sorted(p for p in (ROOT / "web").glob("*.html"))
    if args.page:
        templates = [t for t in templates if t.stem == args.page]
        if not templates:
            print(f"no template web/{args.page}.html", file=sys.stderr)
            return 1
    built = []
    for t in templates:
        text = substitute(t.read_text(), subst)
        out = ROOT / "site" / t.name
        out.write_text(text)
        built.append((t.name, len(text)))
    for name, size in built:
        print(f"built site/{name} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
