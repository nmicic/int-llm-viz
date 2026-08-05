/*
 * Copyright 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * parser_crosscheck.js — prove the browser-side MGW parser (web/shared/mgw.js)
 * reads the committed model.mgw bit-for-bit identically to the C loader.
 *
 * The C side of the comparison is data/trace.json, whose meta.tensor_fnv1a64
 * checksums were computed by tools/trace_harness.c over tensors loaded by the
 * ORIGINAL load_model_mgw() from microgpt_int.c. If every FNV-1a 64 checksum
 * matches here, the JS parser and the C loader agree on every stored byte.
 *
 * Also exercises: structural validation (corrupted images must throw),
 * the |raw| < 2^53 float64-exactness invariant, and exact BigInt dot products.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const assert = require("assert");

const ROOT = path.join(__dirname, "..");
const { Q48, fnv1a64, dotExact, parseMGW } =
  require(path.join(ROOT, "web/shared/mgw.js"));

let checks = 0;
function ok(cond, msg) {
  checks++;
  if (!cond) { console.error("FAIL: " + msg); process.exit(1); }
  console.log("  ok: " + msg);
}

const bytes = new Uint8Array(fs.readFileSync(path.join(ROOT, "third_party/int-llm/model.mgw")));
const trace = JSON.parse(fs.readFileSync(path.join(ROOT, "data/trace.json"), "utf8"));

/* ---- parse + config ---- */
const m = parseMGW(bytes);
ok(m.tensors.length === 11, "11 tensors parsed");
const c = m.config;
ok(c.hidden_dim === 32 && c.num_heads === 4 && c.num_kv_heads === 4 &&
   c.head_dim === 8 && c.num_layers === 1 && c.intermediate_dim === 128 &&
   c.vocab_size === 27 && c.max_seq_len === 8,
   "config matches the documented small-model shape");
ok(m.uchars === "abcdefghijklmnopqrstuvwxyz", "tokenizer.uchars = a..z");
ok(m.uchars === trace.meta.uchars, "uchars agrees with C harness");
ok(m.rngState === BigInt(trace.meta.rng_state_initial),
   "stored rng.state agrees with C harness (" + m.rngState + ")");
ok(m.numWeights === trace.meta.num_params,
   "weight count " + m.numWeights + " agrees with C harness");

/* ---- FNV-1a 64 per tensor: JS parse vs C load ---- */
for (const [name, want] of Object.entries(trace.meta.tensor_fnv1a64)) {
  const t = m.byName.get(name);
  ok(t !== undefined, "tensor present: " + name);
  const got = fnv1a64(t.raw).toString();
  ok(got === want, "fnv1a64(" + name + ") = " + got + " matches C");
}

/* ---- float64-exactness invariant ---- */
let allExact = true;
for (const t of m.weights)
  for (let i = 0; i < t.raw.length; i++)
    if (!Q48.isExactAsNumber(t.raw[i])) allExact = false;
ok(allExact, "every weight satisfies |raw| < 2^53 (float64-exact display)");

/* ---- exact decimal formatting spot checks ---- */
ok(Q48.toDecimalString(0n) === "0", "toDecimalString(0)");
ok(Q48.toDecimalString(Q48.FP_ONE) === "1", "toDecimalString(FP_ONE)");
ok(Q48.toDecimalString(-Q48.FP_ONE / 2n) === "-0.5", "toDecimalString(-FP_ONE/2)");
ok(Q48.toDecimalString(1n) ===
   "0.000000000000003552713678800500929355621337890625",
   "toDecimalString(1) = exact 2^-48");

/* ---- exact dot product sanity ---- */
const wte = m.byName.get("wte");
const selfDot = dotExact(wte.raw, wte.raw, 0, 0, 32);
ok(selfDot > 0n, "dotExact(wte row 0, itself) positive");
let refDot = 0n;
for (let i = 0; i < 32; i++) refDot += wte.raw[i] * wte.raw[i];
ok(selfDot === refDot, "dotExact matches direct BigInt summation");

/* ---- structural rejection: corrupt images must throw ---- */
function throws(mutate, label) {
  const img = new Uint8Array(bytes);
  mutate(img);
  let threw = false;
  try { parseMGW(img); } catch (e) { threw = true; }
  ok(threw, "rejects " + label);
}
throws(img => { img[0] = 0x58; }, "bad magic");
throws(img => { img[8] ^= 0xFF; }, "flipped endian tag");
throws(img => { img[4] = 2; }, "unsupported version");
let threwTrunc = false;
try { parseMGW(bytes.subarray(0, bytes.length - 8)); }
catch (e) { threwTrunc = true; }
ok(threwTrunc, "rejects truncated file");
{
  const bigger = new Uint8Array(bytes.length + 8);
  bigger.set(bytes);
  let threw = false;
  try { parseMGW(bigger); } catch (e) { threw = true; }
  ok(threw, "rejects trailing bytes");
}

console.log("parser_crosscheck: " + checks + " checks passed");
