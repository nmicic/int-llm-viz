# Source provenance

Throughout this document, *the compared implementation* refers to a
separate, independently developed visualization of the same checkpoint,
built in parallel with this repository and kept private; the
independence protocol at the end scopes this repository's blind-build
claim against it. Sibling repositories are referred to by name; during
construction they were local checkouts side by side with this one.

## Vendored inputs (`third_party/int-llm/`)

Copied 2026-08-02 from the sibling `int-llm-coordinate-permutation`
checkout — its committed `source/` snapshot of the int-llm runtime and
small checkpoint. Files are byte-identical to their origin; no headers
were injected into vendored files.

| File | SHA-256 |
| --- | --- |
| `model.mgw` | `466cfe9dba7b888cdaa23dedf4b10351826795793448c8e95dcb0f7a61ed33eb` |
| `microgpt_int.c` | `4374894cdf41569768101307fb609e123984da5034a88d19fa8c7ca26cfcaf73` |
| `fp_math.h` | `38520b39b86f5fe90f7d92cd3a3026877e16370f15a85c9b5f22079e291abf6c` |
| `LICENSE` | `169ae9d19b243330c66349945aa0df6db10dd910471dee1e3762eaafccb2bfb5` |

## Vendored MLP layout set (`third_party/layout-set/`)

Generated 2026-08-02 by the sibling repo's layout tool, run from this repo
against the vendored checkpoint:

```
python3 ../int-llm-coordinate-permutation/\
experiments/systematic_mlp_layout/mlp_layout.py build-set \
  --input third_party/int-llm/model.mgw \
  --output-dir <scratch> \
  --seed int-llm-layout-spike-2026-08-02 --microgpt-oracle
```

Generator repo commit at generation time:
`0a58bd7fab8d473789b8099e7cf3020b508daaac` (clean tree). Only the four
`mlp-order.json` documents and `summary.json` are vendored; the variant
checkpoints they describe are *derived* artifacts, reconstructed and
verified on every test run by `tools/derive_layouts.py`.

| Variant | model sha256 (recorded + reconstructed) |
| --- | --- |
| `original` | `466cfe9dba7b888cdaa23dedf4b10351826795793448c8e95dcb0f7a61ed33eb` |
| `smooth` | `46e9e50a18b007dbb580da6c83a91072983e03f91b80856724c6749ffd37a878` |
| `spread` | `e81eefb8c5be51c1cc39bed7eeb7e17394c989fb11459db304d428acbef9cb9e` |
| `random` | `4bf80788791780306569d2c1cab20bdfb32bbf8e81c32b3709ff748014199e26` |

The same layout set is committed in the compared implementation's
`layout-comparison/` directory (its `summary.json`
pins generator commit `181547436502b4e92358c950fdf7b720e6c8c295`): the
regeneration above reproduced all four variant checkpoints and all three
non-trivial order documents byte-identically — the two visualizations
therefore describe the same objects. The only difference between the two
`summary.json` copies is the recorded `source` input path.

Independent verification performed here (`tools/derive_layouts.py`, run
by `make test`): order documents validate structurally against the
committed model; each variant's full checkpoint is reconstructed in pure
Python by permuting `mlp_fc1` rows and `mlp_fc2` column lanes and must
hash to the recorded sha256; adjacency scores are recomputed exactly in
integer arithmetic; and this repo's own C trace harness replays every
reconstructed variant — sampled text, logits, probabilities and RNG
draws byte-identical to `data/trace.json`, with each step's 128-wide
`h_pre` vector exactly reindexed by the variant's `new_to_old`.

## Vendored hidden-coordinate swap (`third_party/hidden-swap/`)

Copied 2026-08-02 from the compared implementation's committed
`layout-comparison/hidden-swap-0-31/`
(`hidden-order.json`, and its `manifest.json` vendored here as
`viz-manifest.json`):

| File | SHA-256 |
| --- | --- |
| `hidden-order.json` | `006a380e57b950166e0f35a99286dd27bca7b4ef55c9af03dcde9183a4518a2c` |
| `viz-manifest.json` | `dd692d8ba15256e5531e6b0a2ef4cc7054e95d83866140ca0ee236da1c05b6ab` |

The order document was regenerated from this repo's vendored checkpoint
with the sibling generator (repo commit
`0a58bd7fab8d473789b8099e7cf3020b508daaac`, clean tree — the same pin as
the layout set):

```
python3 ../int-llm-coordinate-permutation/\
experiments/hidden_channel_permutation/make_hidden_swap.py \
  --input third_party/int-llm/model.mgw \
  --output <scratch>/hidden-order.json --left 0 --right 31
```

and reproduced `hidden-order.json` byte-identically — the two
visualizations therefore describe the same transposition. The swapped
checkpoint itself is a *derived* artifact, reconstructed and verified on
every test run by `tools/derive_hidden.py`; its sha256
(`8e755c30c6b0d87fa5b36af0be282a4bd6e862211587361c502d81659379228b`)
must match the value recorded in `viz-manifest.json`.

Independent verification performed here (`tools/derive_hidden.py`, run by
`make test`): the order document validates structurally (a 0…31
bijection that is exactly the 0 ↔ 31 transposition, hence its own
inverse); the swapped checkpoint is reconstructed in pure Python by
permuting the residual axis of all nine learned tensors (columns of wte,
wpe, lm_head, wq, wk, wv, fc1; rows of wo, fc2) and must hash to the
recorded sha256; applying the swap twice restores the original bytes;
bytes outside the nine learned tensors are asserted identical
(`tokenizer.uchars`, `rng.state` untouched); per-tensor value multisets
are preserved with exactly 892 of 14,272 positions changed; and this
repo's C trace harness replays the reconstructed variant — sampled text,
logits, probabilities, RNG draws, q/k/v, attention and `h_pre`
byte-identical to `data/trace.json`, with the six residual-basis vectors
per step exactly reindexed by `new_to_old`.

## Upstream pin

The origin repo's own `SOURCE_PROVENANCE.md` pins its snapshot to the public
[`nmicic/int-llm`](https://github.com/nmicic/int-llm) repository at commit
`1b80e024c5fadea955e3892578a36dbc80a8a0b5`, and its 2026-07-31 tree audit
lists exactly four files as modified relative to that commit — none of which
are `microgpt_int.c` or `fp_math.h`. The two source files vendored here are
therefore byte-identical to the pinned upstream commit.

## Cross-repo checkpoint identity

Verified 2026-08-02 during construction: the `model.mgw` vendored here is
byte-identical (same SHA-256 as above) to the committed copies in the local
checkouts of all three sibling repos —

- `int-llm`
- `int-llm-coordinate-permutation` (the copy vendored here)
- `int-llm-precision-ladder`

and `fp_math.h` carries the same SHA-256 in the siblings that vendor it.
Any result cited from a sibling repo therefore refers to the same checkpoint
bytes this repo visualizes.

## Inputs consulted during construction

- The `int-llm-coordinate-permutation` checkout's `source/{model.mgw,
  microgpt_int.c, fp_math.h, README.md}` and its
  `SOURCE_PROVENANCE.md` — the allowed model/runtime inputs.
- Sibling repos' documentation (`int-llm` validation records,
  `int-llm-precision-ladder` results, `int-llm-coordinate-permutation`
  results) — read only to *cite* their documented findings on the landing
  page; nothing from them is re-verified or re-implemented here beyond the
  checkpoint hashes above.
- The author's `four-square-balance/viz/power-balance` and
  `three-square-balance/viz` viewers — visual reference for the
  design dialect (dark instrument-panel styling, hand-rolled Canvas-2D 3-D).
  These are not model visualizations; no code was copied from them.
- The compared implementation's `index.html` (its weight explorer) —
  consulted 2026-08-02, after the blind constraint was lifted,
  as a *feature catalog* before writing `web/weights.html`: which views to
  cover (3-D weight cloud, four linked projections of column/row/weight/
  contribution, heatmap, checkpoint byte layout) and the definition of the
  contribution quantity (the exact local product fp_mul(w[row,col],
  input[col]); undefined for the wte/wpe lookup tables). No code was
  copied: the page here is implemented on this repo's parser (mgw.js),
  build tokens, theme, and Node self-check contract, and its
  contribution-input mapping was re-derived from this repo's own
  `data/trace.json` step fields (wq/wk/wv ← xn_attn, wo ← ao,
  fc1 ← xn_mlp, fc2 ← ReLU(h_pre), lm_head ← x_out), with each row of
  products asserted equal to the recorded C-oracle output vector.

## Independence protocol

This repository was built blind to the compared implementation: at no point
were the compared implementation's checkout, its staging copy, or any
`viz/` subdirectory inside the sibling `int-llm*` repos opened, searched,
or read — including their Git history and rendered output. The automated
authoring passes that produced the visualization pages carried the same
prohibition in their instructions.

**Scope of the blind claim.** The claim above covers the initial build:
commits `e94256b` (root) and `8eb16a1`. On 2026-08-02 the author explicitly
lifted the constraint ("that initial prompt is not valid anymore"), and
the compared implementation's `index.html` was consulted for the first
time — to study its "4D shadows" interaction (a single yaw/pitch/zoom
camera shared by all four panels) before adding the space page's
linked-tetrad-camera mode. Feature work from that commit onward may
reference the compared implementation; the independence claim is not
extended to it.

## License

Vendored upstream files are licensed Apache-2.0 (see
`third_party/int-llm/LICENSE`, retained verbatim). Repository-owned files
are Apache-2.0, `Copyright 2026 Nenad Mićić`, marked with SPDX headers.

The vendored derived data sets under `third_party/layout-set/` and
`third_party/hidden-swap/` (order documents, manifests, summary JSON)
are likewise redistributed under Apache-2.0: they are outputs of the
Apache-2.0 `int-llm-coordinate-permutation` generator (pinned above at
commit `0a58bd7fab8d473789b8099e7cf3020b508daaac`) applied to the
Apache-2.0 int-llm checkpoint, both by the same author/licensor.
