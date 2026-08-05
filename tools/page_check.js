/*
 * Copyright 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * page_check.js — execute a built page's inline scripts in a DOM-less VM.
 *
 * Every site/*.html page is structured so that its computation layer is
 * pure JavaScript with no DOM dependency; when `document` is absent the
 * page runs nodeSelfTest() instead of initializing UI, performing real
 * exactness assertions (BigInt recomputation, checksum comparison, ...)
 * and finally setting:
 *
 *     globalThis.__PAGE_CHECK__ = { page, pass: true, checks: N }
 *
 * This tool extracts the page's <script> blocks, concatenates them in
 * document order (the page contract requires top-level bindings to be
 * unique across blocks, so concatenation preserves browser semantics),
 * runs them, and fails unless __PAGE_CHECK__.pass === true.
 *
 * Usage: node tools/page_check.js site/atlas.html [more.html ...]
 */
"use strict";
const fs = require("fs");
const vm = require("vm");

const files = process.argv.slice(2);
if (files.length === 0) {
  console.error("usage: node tools/page_check.js <built-page.html> ...");
  process.exit(2);
}

let failed = 0;
for (const file of files) {
  const html = fs.readFileSync(file, "utf8");
  const blocks = [];
  const re = /<script(\s[^>]*)?>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = re.exec(html)) !== null) {
    const attrs = m[1] || "";
    if (/\btype\s*=/.test(attrs) &&
        !/\btype\s*=\s*["']?text\/javascript/.test(attrs)) continue;
    if (/\bsrc\s*=/.test(attrs)) {
      console.error(`${file}: external <script src> is forbidden`);
      failed++;
      continue;
    }
    blocks.push(m[2]);
  }
  if (blocks.length === 0) {
    console.error(`${file}: no inline script blocks found`);
    failed++;
    continue;
  }
  const sandbox = {
    console,
    atob,
    btoa,
    TextEncoder,
    TextDecoder,
    performance: { now: () => 0 },
    setTimeout: (fn) => { throw new Error("setTimeout used outside browser guard"); },
    structuredClone,
  };
  vm.createContext(sandbox);
  const code = blocks.join("\n;\n");
  try {
    new vm.Script(code, { filename: file });          // syntax
    vm.runInContext(code, sandbox, { filename: file, timeout: 600000 });
  } catch (e) {
    console.error(`${file}: script execution FAILED: ${e && e.stack || e}`);
    failed++;
    continue;
  }
  const pc = sandbox.__PAGE_CHECK__;
  if (!pc || pc.pass !== true || !(pc.checks > 0)) {
    console.error(`${file}: __PAGE_CHECK__ missing or failing: ` +
                  JSON.stringify(pc));
    failed++;
    continue;
  }
  console.log(`${file}: PASS (${pc.page}, ${pc.checks} checks, ` +
              `${blocks.length} script blocks)`);
}
process.exit(failed ? 1 : 0);
