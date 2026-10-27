// Deterministic 30 fps capture of the replay viewer via Chrome DevTools Protocol.
// requestAnimationFrame is replaced by a fake clock, so each captured frame is exactly 1/30 s of viewer time.
// usage: node scripts/capture-viewer.mjs URL OUTDIR NFRAMES
import { spawn } from 'node:child_process';
import fs from 'node:fs';
const [, , URL_, OUT, NF] = process.argv;
const N = Number(NF || 600);
fs.mkdirSync(OUT, { recursive: true });
const CHROME = `${process.env.HOME}/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome`;
const port = 9345;
const prof = `${OUT}/../.cdp-prof`;
const proc = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, '--no-sandbox', '--hide-scrollbars',
  '--force-device-scale-factor=1', `--user-data-dir=${prof}`, '--window-size=1920,1080', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let list;
for (let i = 0; i < 80; i++) { try { list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); break; } catch { await sleep(200); } }
const target = list.find((t) => t.type === 'page');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0; const pend = new Map();
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') console.error('EXC', JSON.stringify(m.params.exceptionDetails).slice(0, 300));
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send('Page.enable'); await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send('Page.addScriptToEvaluateOnNewDocument', { source: `
  (() => {
    let q = []; let now = 0; let fid = 0;
    window.requestAnimationFrame = (cb) => { q.push([++fid, cb]); return fid; };
    window.cancelAnimationFrame = (h) => { q = q.filter(([i]) => i !== h); };
    window.__tick = (ms) => { now += ms; const run = q; q = []; for (const [, cb] of run) { try { cb(now); } catch (e) { console.error(e); } } };
  })();` });
await send('Page.navigate', { url: URL_ });
await sleep(4000); // data + fonts load; layout rAFs run as we tick below
const ev = (expr) => send('Runtime.evaluate', { expression: expr, awaitPromise: true });
for (let i = 0; i < 5; i++) { await ev('window.__tick(1)'); await sleep(100); }
for (let f = 0; f < N; f++) {
  await ev('window.__tick(1000/30)');
  const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 93, captureBeyondViewport: false });
  fs.writeFileSync(`${OUT}/f${String(f).padStart(5, '0')}.jpg`, Buffer.from(r.result.data, 'base64'));
}
ws.close(); proc.kill();
console.log('captured', N);
