# int-llm-viz

An independent, exploratory visualization of the complete small
[int-llm](https://github.com/nmicic/int-llm) checkpoint `model.mgw` —
a GPT trained and run entirely in Q16.48 int64 fixed-point arithmetic:
14,272 weights, 32-dim residual stream, 4 attention heads, 1 layer,
27-token vocabulary (`a…z` + BOS), 8-position context.

Four self-contained offline pages, one exactness contract: every displayed
number is either bit-exact (BigInt end-to-end, exact terminating decimals)
or explicitly labeled as float display geometry.

## The int-llm series

This is the fourth repository in the int-llm series:

1. [int-llm](https://github.com/nmicic/int-llm) — the source: a GPT
   trained and run entirely in Q16.48 int64 arithmetic
   ([blog post](https://huggingface.co/blog/nmicic/int-llm)).
2. [int-llm-precision-ladder](https://github.com/nmicic/int-llm-precision-ladder)
   — descends the fraction-bit ladder from Q16.48
   ([blog post](https://huggingface.co/blog/nmicic/int-llm-precision-ladder)).
3. [int-llm-coordinate-permutation](https://github.com/nmicic/int-llm-coordinate-permutation)
   — permutes internal coordinates coherently across all tensors
   ([blog post](https://huggingface.co/blog/nmicic/int-llm-coordinate-permutation)).
4. **int-llm-viz** — this repository: that checkpoint, its recorded
   trace, and the permutation results made visible, every number exact.

## Viewing

To see the visualization, go to
**<https://nmicic.github.io/int-llm-viz/site/index.html>** — or open
`site/index.html` locally in a browser. Either way, the pages are fully
self-contained single files (checkpoint and trace data embedded) that
need no server, no network, no external resources. Keyboard: `?` on any
page shows its help.

(`web/` holds the *unbuilt* source templates; opening those directly shows
a notice pointing here instead of a working page.)

| Page | What it shows |
| --- | --- |
| `site/atlas.html` | All nine weight tensors as heatmaps in **literal matrix coordinates** — every parameter at its exact row/column, head boundaries marked, per-tensor histograms, bit occupancy, exact round-to-F-bits preview. Two labeled **derived** sections at the bottom: the four verified MLP neuron orders side by side (deep links: `atlas.html#order=smooth` / `spread` / `random`), and the verified residual-coordinate swap 0 ↔ 31 traced through all nine matrices (deep link: `atlas.html#hidden=fc2`). |
| `site/trace.html` | The model's complete recorded output (20 sampled names) replayed **step by step** — residual stream, per-head attention, ReLU firing, logits → probabilities → the exact roulette draw. Auto-plays through the recording one step per second (reduced-motion-respecting; picking a step or sample by hand pauses it). The browser re-derives each step in exact BigInt arithmetic and verifies it against the recorded C-oracle values. |
| `site/space.html` | The 62 vectors sharing the 32-dim residual space (27 wte + 27 lm_head + 8 wpe) in a hand-rolled 3-D view — **literal coordinate axes** vs a clearly-labeled **derived PCA** mode, plus a four-panel PC1–4 treatment. Every camera auto-rotates gently on one shared clock (reduced-motion-respecting; grabbing a camera pauses it, content switches do not). |
| `site/weights.html` | Any weight matrix as a **3-D point cloud**, a heatmap with exact readouts, and the checkpoint's literal **byte layout**; an exact **contribution mode** (each stored weight × the recorded input it met at a chosen trace step, `fp_mul`-exact, every row re-summed in BigInt against the C-oracle output), and a **4-D shadows** view — four projections of (column, row, weight, contribution) under one shared camera. Deep links carry the whole scene (`weights.html?tensor=mlp_fc1&view=4d&sample=0&step=3&yaw=0.7&pitch=-0.2`), gentle auto-rotation respects reduced motion, a captioned tour advances only by hand, and `window.VIZ.renderAt(ms)` draws reproducible frames for scripted capture. |

## Exactness architecture

- **The original C runtime is the oracle.** `tools/trace_harness.c`
  `#include`s the unmodified `third_party/int-llm/microgpt_int.c`, generates
  the 20 samples with the pristine generation loop, re-derives every
  intermediate with an instrumented replica, and compares bit-for-bit:
  122 steps, 0 logit/KV/sampling mismatches, identical final RNG state.
  The harness exits nonzero on any mismatch; its output is `data/trace.json`
  (all int64 values as decimal strings).
- **Pages parse the checkpoint themselves.** Each page embeds `model.mgw`
  as base64 and decodes it with `web/shared/mgw.js` into BigInt64Arrays.
  FNV-1a-64 checksums computed in the page are compared against the C
  loader's checksums, proving the page read the same bytes.
- **In-browser recomputation.** The trace page ports the integer inference
  path (`fp_mul`, `fp_div`, `fp_sqrt`, `fp_exp`, RMSNorm, attention,
  softmax, xorshift64 sampling) to BigInt and re-derives every recorded
  field of every step; mismatches would flip the page's self-check badge
  to FAIL.
- **Exact display.** Q16.48 values have terminating decimal expansions
  (≤ 48 fraction digits); inspectors print them in full, alongside the raw
  integer and two's-complement hex. Float64 appears only in pixel/color
  math — a conversion that is itself exact here because every weight
  satisfies |raw| < 2^53 (asserted by the build).
- **Derived geometry is provenanced.** The PCA basis comes from
  `tools/derive_geometry.py`: an exact-integer scatter matrix of the wte
  rows (emitted in full in `data/geometry.json`) eigendecomposed by a
  deterministic fixed-order Jacobi sweep. The space page re-derives the
  exact scatter from the embedded checkpoint at load time and compares.
- **The MLP layout set is re-proved, not trusted.** `third_party/layout-set/`
  vendors four physical orderings of the 128 MLP neurons (original /
  smooth / spread / random) produced by the sibling
  coordinate-permutation repo — each a pure byte permutation of
  `mlp_fc1` rows and `mlp_fc2` column lanes. `tools/derive_layouts.py`
  reconstructs every variant checkpoint from the committed original and
  asserts the generator's sha256s, recomputes the adjacency scores in
  exact integer arithmetic, and replays each variant through this repo's
  C oracle: text, logits, probabilities and RNG draws are byte-identical,
  and each step's 128-wide `h_pre` vector is exactly the original's
  reindexed by the order — proof that the layout moves coordinates and
  nothing else. The result is `data/layouts.json`, embedded by the atlas.
- **The residual-coordinate swap is re-proved the same way.**
  `third_party/hidden-swap/` vendors one transposition of residual
  coordinates 0 ↔ 31, copied from the compared visualization's committed
  records and regenerated byte-identically with the same pinned sibling
  generator ([`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md) has both
  sets of hashes) — a single global
  re-indexing applied to the residual axis of **all nine** learned
  matrices at once (columns of wte, wpe, lm_head, wq, wk, wv, fc1; rows
  of wo, fc2; 892 of 14,272 stored positions move, none changes value).
  `tools/derive_hidden.py` reconstructs the swapped checkpoint, asserts
  the generator's sha256, proves the swap is its own inverse, and replays
  it through the C oracle: everything is byte-identical except the six
  residual-basis vectors per step, which are exactly the originals
  reindexed by the swap. The result is `data/hidden.json`, embedded by
  the atlas.

## Regenerating and testing

Requires: a C compiler, Python 3, Node.

```
make               # harness → trace/geometry/layouts/hidden JSON, site pages, checks
make test          # full suite, including double-build byte-determinism
make release-check # suite with the browser step required + no-drift check
```

`tests/run_tests.sh` proves, in order: the C harness's bit-exact replay;
the layout-set verification (reconstruction to recorded sha256s, exact
score recompute, C-oracle replay of every variant); the same for the
residual-coordinate swap (`tools/derive_hidden.py`);
deterministic regeneration of `data/*.json` (byte-identical on rerun);
deterministic site builds; site self-containment and embedded-data
exactness (`tools/check_site.py`); every page's script layer executed
DOM-less in Node with its self-assertions (`tools/page_check.js` — this
includes the trace page's full 122-step BigInt recompute); the JS-parser ↔
C-loader checksum equality (`tests/parser_crosscheck.js`); exact
decimal formatting against Python `Fraction` ground truth
(`tests/q48_format_crosscheck.py`); and, when a Chrome/Chromium binary
is available, the weights, space and trace pages' live-browser contract
(`tests/browser_check.js` — URL-state restoration, `renderAt`
determinism across reloads, linked 4-D cameras, pause semantics under
trusted input events, the trace page's default step autoplay, and
`prefers-reduced-motion` on all three animated pages; the step skips
cleanly on machines without a browser).

`make release-check` is the gate before publishing: the same suite with
the browser step made mandatory (`REQUIRE_BROWSER=1` — a missing Chrome
fails instead of skipping), followed by `git diff --exit-code` over
`data/` and `site/`, so it also fails if regeneration does not
reproduce the committed tree byte-for-byte.

## Design notes

The visual dialect (dark instrument panel, serif-italic titles, mono
numerals, hand-rolled Canvas-2D 3-D) is adopted from the author's
`four-square-balance` / `three-square-balance` viewers.

Color grammar: diverging blue `#4dabf7` ↔ neutral ↔ amber `#ffa94d` for
signed values (zero = no ink); a single-hue surface→gold ramp for
magnitudes in [0, 1]; fixed categorical colors for heads and vector
families, always doubled with text labels and (for families) mark shapes.
The palette was checked with the dataviz validator against the
`#0b0e14` surface: color-vision-deficiency separation and ≥ 3:1 contrast
pass with wide margins; the OKLCH *lightness-band* check for large fills
is deliberately exceeded — these are small bright marks on near-black,
which need the brightness, and identity is never carried by color alone
(direct labels, shapes, small multiples throughout).

## Scope and limits

This is a polished spike for this one checkpoint, not a general framework:

- Page code assumes this model's shape (27 vocab, 1 layer, 4×8 heads,
  8 positions) wherever that made the pages clearer; the assumptions are
  asserted, not hidden.
- Everything-embedded works at this scale (the largest page carries a
  1.3 MB trace); it would not scale to large models — a real system would
  stream slices instead.
- The PCA basis is derived from the 27 wte rows only; lm_head/wpe are
  projected onto it (stated on the page).
- The in-browser recomputation replicates this runtime's exact operation
  order; it is a per-step verifier for this checkpoint, not a general
  fixed-point inference engine.
- No semantic claims are made anywhere: the pages state coordinates,
  values, and verified equalities, and stop there.

## License and provenance

Apache-2.0 (`LICENSE`). Repo-owned files carry
`Copyright 2026 Nenad Mićić` SPDX headers; vendored upstream files under
`third_party/int-llm/` retain their original bytes and license. Origins,
hashes, the upstream commit pin, and the independence protocol (this repo
was built blind to the compared visualization) are recorded in
[`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md).

---

*Fun extra: a song to browse by —
[“One Degree Per Frame”](https://suno.com/song/4e2c6eca-5e22-4b20-9b81-d264af0495d6).*
