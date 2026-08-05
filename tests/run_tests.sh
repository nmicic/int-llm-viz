#!/usr/bin/env bash
# Copyright 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
#
# run_tests.sh — the full deterministic-regeneration and validation suite.
#
#   1. Rebuild the C trace harness and regenerate data/trace.json; the
#      harness itself exits nonzero unless the instrumented replay matches
#      the ORIGINAL runtime bit-for-bit on every step.
#   2. Regenerate data/geometry.json.
#   3. Verify the vendored MLP layout set and regenerate data/layouts.json:
#      order docs validate, variant checkpoints reconstruct to the recorded
#      sha256s, adjacency scores recompute exactly, and the C oracle replays
#      every variant bit-identically (h_pre exactly reindexed). Then verify
#      the vendored hidden-coordinate swap the same way and regenerate
#      data/hidden.json (variant reconstructs to the manifest sha256, the
#      C oracle replays it, six residual-basis vectors exactly reindexed).
#   4. Determinism: regenerate all data products again; bytes must be
#      identical.
#   5. Rebuild site pages; rebuild again; bytes must be identical.
#   6. tools/check_site.py (self-containment, exactness, provenance).
#   7. tools/page_check.js (execute every page's scripts + self-test, no DOM).
#   8. tests/parser_crosscheck.js (JS parser === C loader, FNV-1a 64).
#   9. tests/q48_format_crosscheck.py (exact decimal formatting).
#  10. tests/browser_check.js (live weights page in headless Chrome:
#      URL-state restore, renderAt determinism across reloads, linked
#      4-D cameras, pause semantics under trusted input, reduced motion.
#      Skips cleanly when no Chrome/Chromium or Node < 22 is available;
#      set REQUIRE_BROWSER=1 to make that a failure).

set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n=== %s ===\n' "$*"; }

step "1. C harness: build + regenerate trace.json (self-verifying)"
mkdir -p build
${CC:-cc} -O2 -std=c11 -Wall -Wextra -o build/trace_harness tools/trace_harness.c -lm
./build/trace_harness third_party/int-llm/model.mgw data/trace.json

step "2. regenerate geometry.json"
python3 tools/derive_geometry.py

step "3. verify layout set + regenerate layouts.json"
python3 tools/derive_layouts.py

step "3b. verify hidden-coordinate swap + regenerate hidden.json"
python3 tools/derive_hidden.py

step "4. determinism: regenerate data products, compare bytes"
mkdir -p build/det
./build/trace_harness third_party/int-llm/model.mgw build/det/trace.json
cmp data/trace.json build/det/trace.json
echo "trace.json deterministic"
cp data/geometry.json build/det/geometry.first
python3 tools/derive_geometry.py
cmp data/geometry.json build/det/geometry.first
echo "geometry.json deterministic"
cp data/layouts.json build/det/layouts.first
python3 tools/derive_layouts.py > /dev/null
cmp data/layouts.json build/det/layouts.first
echo "layouts.json deterministic"
cp data/hidden.json build/det/hidden.first
python3 tools/derive_hidden.py > /dev/null
cmp data/hidden.json build/det/hidden.first
echo "hidden.json deterministic"

step "5. build site twice, compare bytes"
python3 tools/build_site.py
for f in site/*.html; do shasum -a 256 "$f"; done > build/det/site.sha.1
python3 tools/build_site.py
for f in site/*.html; do shasum -a 256 "$f"; done > build/det/site.sha.2
cmp build/det/site.sha.1 build/det/site.sha.2
echo "site build deterministic"

step "6. check_site.py"
python3 tools/check_site.py

step "7. page_check.js (run every page's script blocks + self-test in node)"
node tools/page_check.js site/*.html

step "8. parser_crosscheck.js (JS parser vs C loader)"
node tests/parser_crosscheck.js

step "9. q48_format_crosscheck.py (exact decimal rendering)"
python3 tests/q48_format_crosscheck.py

step "10. browser_check.js (live weights page; skips without a browser)"
node tests/browser_check.js

printf '\nALL TESTS PASSED\n'
