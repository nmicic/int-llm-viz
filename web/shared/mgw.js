/*
 * Copyright 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * mgw.js — exact in-browser parser for the MGW v1 checkpoint container
 * and exact Q16.48 formatting helpers.
 *
 * Exactness contract:
 *   - Every stored scalar is decoded into a BigInt64Array; no value ever
 *     passes through floating point on the way in.
 *   - Q48.toDecimalString() renders the exact decimal value of a Q16.48
 *     integer (finite: at most 48 fraction digits, since 2^-48 has a
 *     terminating decimal expansion).
 *   - Q48.toNumber() converts to IEEE-754 float64 for display geometry
 *     only. The conversion is exact if and only if |raw| < 2^53;
 *     Q48.isExactAsNumber() reports that. (Every weight in the committed
 *     model.mgw satisfies it; the build pipeline asserts this.)
 *   - fnv1a64() mirrors the C harness's checksum over the little-endian
 *     byte image of each tensor, so the page can prove it parsed the
 *     same bytes the C loader used.
 *
 * The MGW v1 layout implemented here follows microgpt_int.c:
 *   64-byte header  { "MGW\0", u32 version=1, u32 endian_tag=0x01020304,
 *                     u32 num_tensors, u64 index_offset, u64 data_offset }
 *   64-byte config  { 10 x i32: hidden_dim, num_heads, num_kv_heads,
 *                     head_dim, num_layers, intermediate_dim, vocab_size,
 *                     max_seq_len, rope_theta, lm_head_tied }
 *   num_tensors x 96-byte index entries
 *                   { char name[64], u64 num_elements, u64 data_offset,
 *                     u32 ndims, u32 shape[2], u32 reserved }
 *   packed int64 tensor data.
 *
 * The file format is host-native endian with a rejection tag; the
 * committed model.mgw was written on a little-endian host, and this
 * parser reads all fields explicitly little-endian, verifying the tag.
 */
"use strict";

const Q48 = (() => {
  const FRAC_BITS = 48n;
  const FP_ONE = 1n << FRAC_BITS;
  const MASK = FP_ONE - 1n;
  const EXACT_LIMIT = 1n << 53n;
  const POW5_48 = 5n ** 48n;

  /* Exact decimal string of raw/2^48 (terminating, trailing zeros trimmed). */
  function toDecimalString(raw) {
    const neg = raw < 0n;
    const a = neg ? -raw : raw;
    const ip = a >> FRAC_BITS;
    const fr = a & MASK;
    if (fr === 0n) return (neg ? "-" : "") + ip.toString();
    const fs = (fr * POW5_48).toString().padStart(48, "0").replace(/0+$/, "");
    return (neg ? "-" : "") + ip.toString() + "." + fs;
  }

  /* Generic exact decimal for other fixed-point scales (e.g. Q32.96 dot
   * products, where value = raw / 2^96). */
  function fixedToDecimalString(raw, fracBits) {
    const fb = BigInt(fracBits);
    const one = 1n << fb;
    const neg = raw < 0n;
    const a = neg ? -raw : raw;
    const ip = a / one;
    const fr = a % one;
    if (fr === 0n) return (neg ? "-" : "") + ip.toString();
    const fs = (fr * 5n ** fb).toString()
      .padStart(Number(fb), "0").replace(/0+$/, "");
    return (neg ? "-" : "") + ip.toString() + "." + fs;
  }

  /* float64 for display geometry. Exact iff |raw| < 2^53. */
  function toNumber(raw) { return Number(raw) / 2 ** 48; }
  function isExactAsNumber(raw) {
    return (raw < 0n ? -raw : raw) < EXACT_LIMIT;
  }

  /* 16-hex-digit two's-complement image (what the bytes on disk mean). */
  function toHex(raw) {
    return "0x" + BigInt.asUintN(64, raw).toString(16).padStart(16, "0");
  }

  return { FRAC_BITS, FP_ONE, toDecimalString, fixedToDecimalString,
           toNumber, isExactAsNumber, toHex };
})();

/* FNV-1a 64 over the little-endian 8-byte image of each element, in index
 * order — bit-compatible with fnv1a64() in tools/trace_harness.c. */
function fnv1a64(arr /* BigInt64Array */) {
  const PRIME = 1099511628211n, M = (1n << 64n) - 1n;
  let h = 1469598103934665603n;
  for (let i = 0; i < arr.length; i++) {
    let u = BigInt.asUintN(64, arr[i]);
    for (let b = 0; b < 8; b++) {
      h ^= u & 0xFFn;
      h = (h * PRIME) & M;
      u >>= 8n;
    }
  }
  return h; // BigInt, unsigned 64-bit
}

/* Exact integer dot product of two BigInt64Array slices. The result is a
 * plain BigInt at Q32.96 scale: value = result / 2^96. Summation of
 * arbitrary-precision integers is exact; nothing is rounded. */
function dotExact(a, b, aStart, bStart, len) {
  let s = 0n;
  for (let i = 0; i < len; i++)
    s += a[aStart + i] * b[bStart + i];
  return s;
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/* Parse and validate an MGW v1 image. Throws Error on any structural
 * problem. Returns:
 *   { config, tensors: [{name, ndims, rows, cols, numElements,
 *                        dataOffset, raw: BigInt64Array}],
 *     byName: Map, weights: [tensors minus tokenizer/rng],
 *     uchars: "abc...", bos, vocabSize, rngState: BigInt(u64),
 *     numWeights, fileSize }
 */
function parseMGW(bytes /* Uint8Array */) {
  const HEADER_SIZE = 64, CONFIG_SIZE = 64, INDEX_ENTRY_SIZE = 96;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const fail = (msg) => { throw new Error("MGW parse: " + msg); };

  if (bytes.length < HEADER_SIZE + CONFIG_SIZE) fail("file too small");
  if (!(bytes[0] === 0x4D && bytes[1] === 0x47 && bytes[2] === 0x57 &&
        bytes[3] === 0x00)) fail("bad magic (want \"MGW\\0\")");
  const version = dv.getUint32(4, true);
  if (version !== 1) fail("unsupported version " + version);
  const endianTag = dv.getUint32(8, true);
  if (endianTag !== 0x01020304) fail("endian tag mismatch (not little-endian)");
  const numTensors = dv.getUint32(12, true);
  const indexOffset = Number(dv.getBigUint64(16, true));
  const dataOffset = Number(dv.getBigUint64(24, true));
  for (let i = 32; i < 64; i++)
    if (bytes[i] !== 0) fail("nonzero reserved header byte at " + i);

  const cfgNames = ["hidden_dim", "num_heads", "num_kv_heads", "head_dim",
                    "num_layers", "intermediate_dim", "vocab_size",
                    "max_seq_len", "rope_theta", "lm_head_tied"];
  const config = {};
  cfgNames.forEach((n, i) => { config[n] = dv.getInt32(HEADER_SIZE + 4 * i, true); });

  if (indexOffset !== HEADER_SIZE + CONFIG_SIZE) fail("unexpected index offset");
  if (indexOffset + numTensors * INDEX_ENTRY_SIZE > bytes.length)
    fail("index exceeds file");
  if (dataOffset !== indexOffset + numTensors * INDEX_ENTRY_SIZE)
    fail("unexpected data offset");

  const tensors = [], byName = new Map();
  let end = dataOffset;
  for (let i = 0; i < numTensors; i++) {
    const off = indexOffset + i * INDEX_ENTRY_SIZE;
    let name = "";
    for (let j = 0; j < 64; j++) {
      const c = bytes[off + j];
      if (c === 0) break;
      name += String.fromCharCode(c);
    }
    const numElements = Number(dv.getBigUint64(off + 64, true));
    const tOff = Number(dv.getBigUint64(off + 72, true));
    const ndims = dv.getUint32(off + 80, true);
    const s0 = dv.getUint32(off + 84, true);
    const s1 = dv.getUint32(off + 88, true);
    if (ndims !== 1 && ndims !== 2) fail(name + ": bad ndims " + ndims);
    const rows = s0, cols = ndims === 2 ? s1 : 1;
    if (rows * cols !== numElements) fail(name + ": shape/count mismatch");
    if (tOff % 8 !== 0) fail(name + ": unaligned data");
    if (tOff + numElements * 8 > bytes.length) fail(name + ": data exceeds file");
    const raw = new BigInt64Array(numElements);
    for (let j = 0; j < numElements; j++)
      raw[j] = dv.getBigInt64(tOff + 8 * j, true);
    const t = { name, ndims, rows, cols, numElements, dataOffset: tOff, raw };
    tensors.push(t);
    byName.set(name, t);
    end = Math.max(end, tOff + numElements * 8);
  }
  if (end !== bytes.length) fail("trailing bytes after tensor data");

  const uc = byName.get("tokenizer.uchars");
  const rng = byName.get("rng.state");
  if (!uc || !rng) fail("missing tokenizer.uchars or rng.state");
  if (uc.numElements !== config.vocab_size - 1)
    fail("tokenizer.uchars length disagrees with config.vocab_size");
  let uchars = "";
  for (let i = 0; i < uc.numElements; i++)
    uchars += String.fromCharCode(Number(uc.raw[i]));

  const weights = tensors.filter(
    (t) => t.name !== "tokenizer.uchars" && t.name !== "rng.state");
  let numWeights = 0;
  for (const t of weights) numWeights += t.numElements;

  return {
    config, tensors, byName, weights, uchars,
    bos: config.vocab_size - 1, vocabSize: config.vocab_size,
    rngState: BigInt.asUintN(64, rng.raw[0]),
    numWeights, fileSize: bytes.length,
  };
}

/* Export for Node-based cross-checks; browsers just use the globals. */
if (typeof module !== "undefined" && module.exports)
  module.exports = { Q48, fnv1a64, dotExact, b64ToBytes, parseMGW };
