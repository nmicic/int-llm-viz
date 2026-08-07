#!/usr/bin/env node
// Copyright 2026 Nenad Mićić
// SPDX-License-Identifier: Apache-2.0
//
// browser_check.js — live-browser verification of the animated pages
// (site/weights.html, site/space.html, site/trace.html).
//
// The Node self-checks (tools/page_check.js) prove the pure layer without a
// DOM; this test proves the browser-only contracts that need a real renderer
// and *trusted* input events:
//
//   1. The page boots clean: self-check badge ✓, no uncaught exceptions or
//      console errors, window.VIZ.ready.
//   2. Auto-rotation runs by default (no reduced-motion emulated) and the
//      virtual clock advances.
//   3. URL-state restoration matrix: the documented deep link restores the
//      exact scene; junk parameters clamp; lookup tensors coerce away from
//      4-D/contribution; the heatmap never rotates; the address bar always
//      normalizes to the canonical serialization.
//   4. renderAt determinism ACROSS RELOADS: the frame hash at a given
//      virtual time is identical within a load and between two fresh loads
//      of the same URL (frames are pure functions of (scene, ms)).
//   5. Linked cameras: one virtual-time change moves all four 4-D quadrant
//      projections at once, deterministically per quadrant.
//   6. Pause semantics with trusted CDP input (real pointer ids, so
//      setPointerCapture is exercised): dragging, wheeling and orbit keys
//      pause the rotation and fold the offset; switching to the heatmap
//      force-pauses and disables the control; content switches (view among
//      cloud/4-D, quantity, tensor) leave the play state alone.
//   7. prefers-reduced-motion (CDP media emulation): autoplay is suppressed
//      at boot even with rot=1, while manual play still works as an
//      explicit user action.
//   8. The space page carries the same rotation contract across its five
//      cameras: default autoplay, drag/wheel/orbit-key pause with the
//      offset folded, mode/family switches and the matrix's cell keys
//      keep playing, all four tetrad panels turn (linked and unlinked)
//      on the one shared clock, and reduced motion gates autoplay only.
//   9. The trace page autoplays its recording by default (1 step/s on
//      the rAF clock), any manual step or sample navigation pauses it
//      (the chosen step holds still), the space bar resumes, and
//      reduced motion suppresses the autoplay while manual play still
//      works.
//
// Requires a Chrome/Chromium binary and Node >= 22 (built-in WebSocket).
// Without either the test SKIPS with exit 0 so the core suite stays
// runnable everywhere; set REQUIRE_BROWSER=1 to turn a skip into a failure.
// Set CHROME=/path/to/chrome to point at a specific binary.
"use strict";
const { spawn, spawnSync } = require("child_process");
const http = require("http");
const fs = require("fs");
const os = require("os");
const path = require("path");

const PAGE_BASE = "file://" +
  path.resolve(__dirname, "..", "site", "weights.html");
const SPACE_PAGE = "file://" +
  path.resolve(__dirname, "..", "site", "space.html");
const TRACE_PAGE = "file://" +
  path.resolve(__dirname, "..", "site", "trace.html");
const EXAMPLE =
  "?tensor=mlp_fc1&view=4d&sample=0&step=3&yaw=0.7&pitch=-0.2&rot=0";

let chromeProc = null;   // set once spawned, so skip() never orphans it
function skip(msg) {
  if (chromeProc) { try { chromeProc.kill("SIGKILL"); } catch (e) {} }
  if (process.env.REQUIRE_BROWSER === "1") {
    console.error(`browser_check: FAIL (REQUIRE_BROWSER=1): ${msg}`);
    process.exit(1);
  }
  console.log(`browser_check: SKIPPED — ${msg}`);
  process.exit(0);
}

function findChrome() {
  const cands = [];
  if (process.env.CHROME) cands.push(process.env.CHROME);
  cands.push(
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge");
  for (const c of cands) if (fs.existsSync(c)) return c;
  for (const name of ["google-chrome", "google-chrome-stable",
                      "chromium", "chromium-browser"]) {
    const r = spawnSync("which", [name], { encoding: "utf8" });
    if (r.status === 0 && r.stdout.trim()) return r.stdout.trim();
  }
  return null;
}

function get(port, p) {
  return new Promise((resolve, reject) => {
    http.get({ host: "127.0.0.1", port, path: p }, (res) => {
      let d = "";
      res.on("data", (c) => (d += c));
      res.on("end", () => { try { resolve(JSON.parse(d)); } catch (e) { reject(e); } });
    }).on("error", reject);
  });
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cdp(wsURL, events) {
  const sock = new WebSocket(wsURL);
  await new Promise((res, rej) => { sock.onopen = res; sock.onerror = rej; });
  let id = 0;
  const pending = new Map();
  sock.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
    else if (m.method === "Runtime.exceptionThrown")
      events.push("exception: " +
        JSON.stringify(m.params.exceptionDetails).slice(0, 300));
    else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error")
      events.push("console.error: " + JSON.stringify(m.params.args).slice(0, 300));
  };
  const send = (method, params = {}) => new Promise((res) => {
    const i = ++id;
    pending.set(i, res);
    sock.send(JSON.stringify({ id: i, method, params }));
  });
  return { send, close: () => sock.close() };
}

let failures = 0;
function report(name, pass, extra) {
  console.log(`  ${pass ? "ok" : "FAIL"}: ${name}${extra ? "  " + extra : ""}`);
  if (!pass) failures++;
}

(async () => {
  if (typeof WebSocket === "undefined")
    skip("Node >= 22 needed for the built-in WebSocket client");
  const chromeBin = findChrome();
  if (!chromeBin)
    skip("no Chrome/Chromium binary found (set CHROME=/path/to/chrome)");
  if (!fs.existsSync(PAGE_BASE.replace("file://", "")) ||
      !fs.existsSync(SPACE_PAGE.replace("file://", "")) ||
      !fs.existsSync(TRACE_PAGE.replace("file://", "")))
    skip("site pages not built (run tools/build_site.py first)");

  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "viz-browser-check-"));
  const chrome = chromeProc = spawn(chromeBin, [
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--remote-debugging-port=0", `--user-data-dir=${profile}`,
    "--no-first-run", "--no-default-browser-check", "--disable-extensions",
    "--window-size=1440,1300", "about:blank",
  ], { stdio: "ignore" });
  const watchdog = setTimeout(() => {
    console.error("browser_check: TIMEOUT (240 s) — killing browser");
    try { chrome.kill("SIGKILL"); } catch (e) {}
    process.exit(1);
  }, 240000);

  try {
    // The browser picks a free port and records it in the profile dir.
    let port = 0;
    for (let i = 0; i < 100 && !port; i++) {
      await sleep(150);
      try {
        const t = fs.readFileSync(
          path.join(profile, "DevToolsActivePort"), "utf8");
        port = parseInt(t.split("\n")[0], 10) || 0;
      } catch (e) {}
    }
    if (!port) skip(`browser did not expose a DevTools port (${chromeBin})`);
    let targets = null;
    for (let i = 0; i < 50 && !targets; i++) {
      await sleep(150);
      try { targets = await get(port, "/json/list"); } catch (e) {}
    }
    const page = targets && targets.find((t) => t.type === "page");
    if (!page) skip("no page target on the DevTools endpoint");

    const errors = [];
    const c = await cdp(page.webSocketDebuggerUrl, errors);
    await c.send("Page.enable");
    await c.send("Runtime.enable");
    await c.send("Emulation.setDeviceMetricsOverride",
      { width: 1440, height: 1300, deviceScaleFactor: 1, mobile: false });

    const evl = async (expr) => {
      const r = await c.send("Runtime.evaluate",
        { expression: expr, returnByValue: true, awaitPromise: true });
      if (r.result.exceptionDetails)
        throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 400));
      return r.result.result.value;
    };
    const nav = async (params) => {
      await c.send("Page.navigate", { url: PAGE_BASE + params });
      let ready = false;
      for (let i = 0; i < 100 && !ready; i++) {
        await sleep(150);
        ready = await evl(
          "document.readyState === 'complete' && !!window.VIZ && " +
          "window.VIZ.ready === true").catch(() => false);
      }
      if (!ready) throw new Error("page did not become VIZ-ready");
      await sleep(300);
    };
    // FNV-1a-32 of the stage's PNG data URL: equal hash <=> equal pixels
    // (same browser build encodes identical pixels identically).
    const frameHash = (ms) => evl(`(() => {
      window.VIZ.renderAt(${ms});
      const s = document.getElementById("stage").toDataURL();
      let a = 0x811c9dc5;
      for (let i = 0; i < s.length; i++) {
        a ^= s.charCodeAt(i); a = Math.imul(a, 0x01000193) >>> 0;
      }
      return a >>> 0;
    })()`);
    const quadHashes = (ms) => evl(`(() => {
      window.VIZ.renderAt(${ms});
      const cv = document.getElementById("stage");
      const ctx = cv.getContext("2d");
      const w = cv.width >> 1, h = cv.height >> 1;
      return [[0, 0], [w, 0], [0, h], [w, h]].map(([x, y]) => {
        const d = ctx.getImageData(x, y, w, h).data;
        let a = 0x811c9dc5;
        for (let i = 0; i < d.length; i++) {
          a ^= d[i]; a = Math.imul(a, 0x01000193) >>> 0;
        }
        return a >>> 0;
      });
    })()`);
    const centerOf = async (id) => {
      await evl(`document.getElementById(${JSON.stringify(id)})` +
        ".scrollIntoView({ block: 'center' }); true");
      await sleep(120);
      return evl(`(() => {
        const r = document.getElementById(${JSON.stringify(id)})
          .getBoundingClientRect();
        return { x: Math.round(r.x + r.width / 2),
                 y: Math.round(r.y + r.height / 2) };
      })()`);
    };
    // hash a canvas's current pixels (no renderAt — live sampling)
    const cvHash = (id) => evl(`(() => {
      const s = document.getElementById(${JSON.stringify(id)}).toDataURL();
      let a = 0x811c9dc5;
      for (let i = 0; i < s.length; i++) {
        a ^= s.charCodeAt(i); a = Math.imul(a, 0x01000193) >>> 0;
      }
      return a >>> 0;
    })()`);
    const dispatchDrag = async (pos) => {
      await c.send("Input.dispatchMouseEvent",
        { type: "mousePressed", x: pos.x, y: pos.y, button: "left",
          buttons: 1, clickCount: 1 });
      for (let i = 1; i <= 3; i++)
        await c.send("Input.dispatchMouseEvent",
          { type: "mouseMoved", x: pos.x + 20 * i, y: pos.y, buttons: 1 });
      await c.send("Input.dispatchMouseEvent",
        { type: "mouseReleased", x: pos.x + 60, y: pos.y, button: "left",
          buttons: 0, clickCount: 1 });
      await sleep(120);
    };
    const dispatchKey = async (key, code, vk) => {
      for (const type of ["rawKeyDown", "keyUp"])
        await c.send("Input.dispatchKeyEvent",
          { type, key, code,
            windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk });
      await sleep(120);
    };
    const ensurePlaying = async () => {
      await evl("if (!UI.rot.playing) document.getElementById('btnRot').click(); true");
      await sleep(150);
      return evl("UI.rot.playing === true");
    };
    const rotState = () => evl(
      "({ playing: UI.rot.playing, vms: UI.rot.virtualMs, " +
      "yaw: UI.view.yaw, pitch: UI.view.pitch, zoom: UI.view.zoom })");

    // -- space-page helpers (no VIZ contract there; UI.stages marks boot) --
    const navSpace = async () => {
      await c.send("Page.navigate", { url: SPACE_PAGE });
      let ready = false;
      for (let i = 0; i < 100 && !ready; i++) {
        await sleep(150);
        ready = await evl(
          "document.readyState === 'complete' && typeof UI === 'object' && " +
          "UI !== null && Array.isArray(UI.stages) && UI.stages.length === 5")
          .catch(() => false);
      }
      if (!ready) throw new Error("space page did not become ready");
      await sleep(300);
    };
    const spRot = () => evl(
      "({ playing: UI.rot.playing, vms: UI.rot.virtualMs })");
    const spEnsurePlaying = async () => {
      await evl(
        "if (!UI.rot.playing) document.getElementById('rotBtn').click(); true");
      await sleep(150);
      return evl("UI.rot.playing === true");
    };

    // -- trace-page helper (same shape: UI is the boot marker) --
    const navTrace = async () => {
      await c.send("Page.navigate", { url: TRACE_PAGE });
      let ready = false;
      for (let i = 0; i < 100 && !ready; i++) {
        await sleep(150);
        ready = await evl(
          "document.readyState === 'complete' && typeof UI === 'object' && " +
          "UI !== null && typeof UI.pi === 'number' && " +
          "!!document.getElementById('btnPlay')").catch(() => false);
      }
      if (!ready) throw new Error("trace page did not become ready");
      await sleep(300);
    };

    /* ---- 1. clean boot ---- */
    console.log("boot");
    await nav("");
    report("self-check badge is green",
      /^self-check: \d+ ✓$/.test(
        await evl("document.getElementById('selfBadge').textContent")),
      await evl("document.getElementById('selfBadge').textContent"));
    report("window.VIZ contract exposed",
      await evl("VIZ.ready === true && typeof VIZ.getScene === 'function' && " +
        "typeof VIZ.setScene === 'function' && typeof VIZ.renderAt === 'function' && " +
        "typeof VIZ.sceneURL === 'function'"));

    /* ---- 2. default motion ---- */
    console.log("default motion");
    report("auto-rotation plays by default (no reduced-motion here)",
      await evl("UI.rot.playing === true && UI.reducedMotion === false"));
    report("virtual clock advances",
      await evl("(async () => { const v0 = UI.rot.virtualMs; " +
        "await new Promise(r => setTimeout(r, 300)); " +
        "return UI.rot.virtualMs > v0; })()"));
    report("address bar records rot=1",
      await evl("location.search.includes('rot=1')"));

    /* ---- 3. URL-state restoration matrix ---- */
    console.log("URL-state restoration");
    await nav(EXAMPLE);
    const deep = await evl("VIZ.getScene()");
    report("documented example restores exactly",
      deep.tensor === "mlp_fc1" && deep.view === "4d" && deep.sample === 0 &&
      deep.step === 3 && deep.yaw === 0.7 && deep.pitch === -0.2 &&
      deep.zoom === 1 && deep.rot === false, JSON.stringify(deep));
    report("address bar normalized to the canonical serialization",
      await evl("location.search === '?' + serializeScene(VIZ.getScene())"));

    await nav("?tensor=nope&sample=99&step=99&yaw=9.9&pitch=-9&zoom=99&rot=0");
    report("junk parameters clamp to legal values",
      await evl("(() => { const s = VIZ.getScene(); " +
        "return s.tensor === SCENE_DEFAULT.tensor && " +
        "s.sample === TRACE.samples.length - 1 && " +
        "s.step === TRACE.samples[s.sample].steps.length - 1 && " +
        "s.pitch === -CAM.pitchLim && s.zoom === CAM.zoomHi && " +
        "s.yaw > -Math.PI && s.yaw <= Math.PI; })()"),
      JSON.stringify(await evl("VIZ.getScene()")));

    await nav("?tensor=wte&view=4d&q=contrib&rot=0");
    report("lookup tensor coerces away from 4-D and contribution",
      await evl("(() => { const s = VIZ.getScene(); " +
        "return s.tensor === 'wte' && s.view === 'cloud' && " +
        "s.quantity === 'weight'; })()"),
      JSON.stringify(await evl("VIZ.getScene()")));

    await nav("?view=heat&rot=1");
    report("the heatmap never rotates, even when the link asks",
      await evl("UI.state.view === 'heat' && UI.rot.playing === false && " +
        "document.getElementById('btnRot').disabled === true"));

    /* ---- 4. renderAt determinism across reloads ---- */
    console.log("renderAt determinism across reloads");
    await nav(EXAMPLE);
    const a0 = await frameHash(0), a9 = await frameHash(9000);
    const a0b = await frameHash(0);
    report("frame hash repeats within one load", a0 === a0b, `${a0} vs ${a0b}`);
    report("virtual times draw distinct frames", a0 !== a9);
    await nav(EXAMPLE);   // fresh load of the identical URL
    const b0 = await frameHash(0), b9 = await frameHash(9000);
    report("frame at t=0 identical across reloads", a0 === b0, `${a0} vs ${b0}`);
    report("frame at t=9000 identical across reloads", a9 === b9, `${a9} vs ${b9}`);

    /* ---- 5. linked cameras: one clock moves all four quadrants ---- */
    console.log("linked cameras");
    const q0 = await quadHashes(0), q9 = await quadHashes(9000);
    const q0b = await quadHashes(0);
    report("every 4-D quadrant responds to the one shared camera",
      q0.every((h, i) => h !== q9[i]), `${q0} vs ${q9}`);
    report("every quadrant is deterministic at a fixed time",
      q0.every((h, i) => h === q0b[i]), `${q0} vs ${q0b}`);

    /* ---- 6. pause semantics with trusted input ---- */
    console.log("pause on camera interactions (trusted CDP input)");
    await nav("?tensor=mlp_fc1&view=cloud&rot=1");

    report("playing before the drag", await ensurePlaying());
    let pos = await centerOf("stage");
    await dispatchDrag(pos);
    let st = await rotState();
    report("a real pointer drag pauses and folds (virtualMs back to 0)",
      st.playing === false && st.vms === 0, JSON.stringify(st));

    report("playing before the wheel", await ensurePlaying());
    const zoomBefore = (await rotState()).zoom;
    await c.send("Input.dispatchMouseEvent",
      { type: "mouseWheel", x: pos.x, y: pos.y, deltaX: 0, deltaY: -120 });
    await sleep(120);
    st = await rotState();
    report("a real wheel zoom pauses and changes zoom",
      st.playing === false && st.vms === 0 && st.zoom !== zoomBefore,
      JSON.stringify(st));

    report("playing before the orbit key", await ensurePlaying());
    await evl("document.getElementById('stage').focus(); true");
    await dispatchKey("ArrowRight", "ArrowRight", 39);
    st = await rotState();
    report("a real orbit key pauses and folds",
      st.playing === false && st.vms === 0, JSON.stringify(st));

    report("playing before the heatmap switch", await ensurePlaying());
    await evl("document.getElementById('btnHeat').click(); true");
    st = await rotState();
    const heatDisabled = await evl(
      "document.getElementById('btnRot').disabled === true");
    report("switching to the heatmap force-pauses and disables the control",
      st.playing === false && heatDisabled);
    await evl("document.getElementById('btnCloud').click(); true");

    console.log("content switches keep the play state");
    report("playing before the content switches", await ensurePlaying());
    await evl("document.getElementById('btnContrib').click(); true");
    report("quantity switch keeps playing", await evl("UI.rot.playing === true"));
    await evl("document.getElementById('btn4D').click(); true");
    report("cloud→4-D switch keeps playing", await evl("UI.rot.playing === true"));
    await evl("(() => { const s = document.getElementById('tensorSel'); " +
      "s.value = 'attn_wo'; s.dispatchEvent(new Event('change')); })()");
    report("tensor switch keeps playing",
      await evl("UI.state.tensor === 'attn_wo' && UI.rot.playing === true"));
    report("the clock still advances after the switches",
      await evl("(async () => { const v0 = UI.rot.virtualMs; " +
        "await new Promise(r => setTimeout(r, 300)); " +
        "return UI.rot.virtualMs > v0; })()"));

    /* ---- 7. prefers-reduced-motion suppresses autoplay ---- */
    console.log("reduced motion");
    await c.send("Emulation.setEmulatedMedia",
      { media: "", features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
    await nav("?tensor=mlp_fc1&view=cloud&rot=1");
    report("rot=1 does not autoplay under prefers-reduced-motion",
      await evl("UI.reducedMotion === true && UI.rot.playing === false"));
    report("the control shows the paused state",
      await evl("document.getElementById('btnRot')" +
        ".getAttribute('aria-pressed') === 'false'"));
    await evl("document.getElementById('btnRot').click(); true");
    report("manual play still works (explicit user action)",
      await evl("UI.rot.playing === true"));
    await c.send("Emulation.setEmulatedMedia", { media: "", features: [] });

    /* ---- 8. the space page: one rotation contract, five cameras ---- */
    console.log("space page: boot and default motion");
    await navSpace();
    report("space self-check badge is green",
      /^self-check: \d+ ✓$/.test(
        await evl("document.getElementById('checkBadge').textContent")),
      await evl("document.getElementById('checkBadge').textContent"));
    report("auto-rotation plays by default (no reduced-motion here)",
      await evl("UI.rot.playing === true && UI.reducedMotion === false"));
    report("the shared virtual clock advances",
      await evl("(async () => { const v0 = UI.rot.virtualMs; " +
        "await new Promise(r => setTimeout(r, 300)); " +
        "return UI.rot.virtualMs > v0; })()"));

    console.log("space page: camera grabs pause, content switches do not");
    pos = await centerOf("stage");
    await dispatchDrag(pos);
    let sp = await spRot();
    report("a real pointer drag pauses and folds (virtualMs back to 0)",
      sp.playing === false && sp.vms === 0, JSON.stringify(sp));
    report("the control shows the paused state",
      await evl("document.getElementById('rotBtn')" +
        ".getAttribute('aria-pressed') === 'false'"));

    report("playing before the wheel", await spEnsurePlaying());
    await c.send("Input.dispatchMouseEvent",
      { type: "mouseWheel", x: pos.x, y: pos.y, deltaX: 0, deltaY: -120 });
    await sleep(120);
    sp = await spRot();
    report("a real wheel zoom pauses and folds",
      sp.playing === false && sp.vms === 0, JSON.stringify(sp));

    report("playing before the orbit key", await spEnsurePlaying());
    await evl("document.getElementById('stage').focus(); true");
    await dispatchKey("ArrowRight", "ArrowRight", 39);
    sp = await spRot();
    report("a real orbit key pauses and folds",
      sp.playing === false && sp.vms === 0, JSON.stringify(sp));

    report("playing before the content switches", await spEnsurePlaying());
    await evl("document.getElementById('modeB').click(); true");
    report("axes→PCA mode switch keeps playing",
      await evl("UI.rot.playing === true"));
    await evl("document.getElementById('famWte').click(); true");
    const famKept = await evl("UI.rot.playing === true");
    await evl("document.getElementById('famWte').click(); true");
    report("family toggle keeps playing", famKept);
    await evl("document.getElementById('matrix').focus(); true");
    await dispatchKey("ArrowRight", "ArrowRight", 39);
    report("matrix cell keys move a cell, not a camera — still playing",
      await evl("UI.rot.playing === true"));

    console.log("space page: all four tetrad panels turn on the one clock");
    await evl("document.getElementById('modeC').click(); true");
    report("PCA→tetrad mode switch keeps playing",
      await evl("UI.rot.playing === true"));
    await evl(
      "document.getElementById('tp0').scrollIntoView({ block: 'center' }); true");
    await sleep(150);
    const TPS = ["tp0", "tp1", "tp2", "tp3"];
    const t0 = [], t1 = [];
    for (const id of TPS) t0.push(await cvHash(id));
    await sleep(450);   /* ≈ 0.06 rad of orbit — a visible pixel shift */
    for (const id of TPS) t1.push(await cvHash(id));
    report("linked: every tetrad panel keeps turning",
      t0.every((h, i) => h !== t1[i]), `${t0} vs ${t1}`);
    await evl("document.getElementById('linkBtn').click(); true");
    report("unlinking the cameras keeps playing",
      await evl("UI.rot.playing === true"));
    const u0 = [], u1 = [];
    for (const id of TPS) u0.push(await cvHash(id));
    await sleep(450);
    for (const id of TPS) u1.push(await cvHash(id));
    report("unlinked: every panel still turns at the shared rate",
      u0.every((h, i) => h !== u1[i]), `${u0} vs ${u1}`);
    await evl("document.getElementById('linkBtn').click(); true");

    console.log("space page: the A shortcut and reduced motion");
    await evl("document.activeElement && document.activeElement.blur(); true");
    await dispatchKey("a", "KeyA", 65);
    sp = await spRot();
    report("the A shortcut pauses (and the button shows it)",
      sp.playing === false && await evl("document.getElementById('rotBtn')" +
        ".getAttribute('aria-pressed') === 'false'"), JSON.stringify(sp));
    await dispatchKey("a", "KeyA", 65);
    report("the A shortcut resumes", await evl("UI.rot.playing === true"));

    await c.send("Emulation.setEmulatedMedia",
      { media: "", features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
    await navSpace();
    report("no autoplay under prefers-reduced-motion",
      await evl("UI.reducedMotion === true && UI.rot.playing === false"));
    await evl("document.getElementById('rotBtn').click(); true");
    report("manual play still works (explicit user action)",
      await evl("UI.rot.playing === true"));
    await c.send("Emulation.setEmulatedMedia", { media: "", features: [] });

    /* ---- 9. the trace page: default autoplay, hands pause it ---- */
    console.log("trace page: boot and default autoplay");
    await navTrace();
    report("oracle badge is green",
      await evl("document.getElementById('oracleBadge').className" +
        ".includes('pass')"),
      await evl("document.getElementById('oracleBadge').textContent"));
    report("autoplay is on by default (button shows the playing state)",
      await evl("UI.playing === true && UI.reducedMotion === false && " +
        "document.getElementById('btnPlay')" +
        ".getAttribute('aria-pressed') === 'true'"));
    const tr0 = await evl("({ si: UI.si, pi: UI.pi })");
    await sleep(1250);   /* the clock ticks every 1000 ms — one tick fits */
    const tr1 = await evl("({ si: UI.si, pi: UI.pi })");
    report("the 1 step/s clock advances the timeline",
      tr1.si !== tr0.si || tr1.pi !== tr0.pi,
      JSON.stringify(tr0) + " -> " + JSON.stringify(tr1));

    console.log("trace page: manual navigation pauses");
    await evl("document.getElementById('btnNext').click(); true");
    const tp = await evl("({ playing: UI.playing, pi: UI.pi })");
    report("the next-step button pauses autoplay",
      tp.playing === false && await evl("document.getElementById('btnPlay')" +
        ".getAttribute('aria-pressed') === 'false'"), JSON.stringify(tp));
    const held = await evl("UI.pi");
    await sleep(1250);
    report("the chosen step holds still while paused",
      await evl("UI.pi") === held);

    await evl("document.activeElement && document.activeElement.blur(); true");
    await dispatchKey(" ", "Space", 32);
    report("the space bar resumes autoplay", await evl("UI.playing === true"));
    await dispatchKey("ArrowRight", "ArrowRight", 39);
    report("a step arrow key pauses again", await evl("UI.playing === false"));
    await dispatchKey(" ", "Space", 32);
    report("resumed once more", await evl("UI.playing === true"));
    await evl("(() => { const b = document.querySelectorAll('#sampleStrip button');" +
      " b[(UI.si + 1) % b.length].click(); return true; })()");
    report("picking a sample by hand pauses too",
      await evl("UI.playing === false"));

    await c.send("Emulation.setEmulatedMedia",
      { media: "", features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
    await navTrace();
    report("trace: no autoplay under prefers-reduced-motion",
      await evl("UI.reducedMotion === true && UI.playing === false"));
    await evl("document.getElementById('btnPlay').click(); true");
    report("trace: manual play still works (explicit user action)",
      await evl("UI.playing === true"));
    await c.send("Emulation.setEmulatedMedia", { media: "", features: [] });

    /* ---- final ---- */
    report("no uncaught exceptions / console errors in the whole run",
      errors.length === 0, errors.join(" | "));

    console.log(failures
      ? `browser_check: ${failures} FAILURE(S)`
      : "browser_check: all live checks passed");
    process.exitCode = failures ? 1 : 0;
  } catch (e) {
    console.error("browser_check: ERROR:", e.message || e);
    process.exitCode = 1;
  } finally {
    clearTimeout(watchdog);
    try { chrome.kill(); } catch (e) {}
    await sleep(300);
    try { chrome.kill("SIGKILL"); } catch (e) {}
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) {}
  }
})();
