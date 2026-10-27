/* Lost Together replay viewer. Vanilla JS, no build step, no external requests.
   Data (all optional except index.json and the episodes):
     data/index.json, data/<episode>.json            lost_together/export.py
     data/frames/<episode>/manifest.json, <t>_<h>.jpg, zoom/<t>_<h>.jpg   lost_together/camera_tiles.py
     data/agent/index.json (list of file names) or a directory listing, data/agent/*.json   agent/loop.py write_trace
     data/results.json  {headline: [{label, value, note}], figure: "relative/path.png"}
   No inline style attributes are written, so the page runs under CSP default-src 'self'. */
'use strict';
(() => {
  // ---------- constants ----------
  const HIKER = ['#0072B2', '#CC79A7', '#009E73', '#56B4E9', '#E69F00', '#F0E442', '#000000', '#999999'];
  const C = {
    ink: '#131a16', ink2: '#414c45', ink3: '#5c6861', paper: '#fbfcf9', ground: '#f1f3ef', rule: '#c3cac0',
    predator: '#D55E00', report: '#E69F00', reportDark: '#9a6700', fake: '#6f7a73', truth: '#F0E442', well: '#0d120f',
    title: 'rgba(13,18,15,0.82)', casing: 'rgba(8,11,9,0.5)',
  };
  // terrain RGB per state: [not yet seen, explored, just explored, living map off]
  const TERRAIN = {
    ground: [[214, 219, 211], [251, 252, 249], [255, 241, 168], [246, 247, 243]],
    trees: [[184, 194, 180], [163, 204, 152], [196, 226, 140], [181, 210, 171]],
  };
  const CAP_H = 66;                // camera caption: up to three lines (detections, evaluator, honest note)
  const FRESH_STEPS = 10;
  const TOAST_STEPS = 28;
  const DET_MIN = 0.25;            // camera_tiles.py runs YoloxDetector(conf=0.25)
  const MODE_LABEL = { enjoy: 'Walking', escape: 'Escaping', home: 'Heading home', escaped: 'Exited', caught: 'Intercepted', lost: 'Still lost' };
  const PRED_LABEL = { patrol: 'patrolling', pursue: 'pursuing', handling: 'paused' };
  const INF = 0x7fffffff;
  const FONT = '"National Park", ui-sans-serif, system-ui, sans-serif';
  const ICON_CHECK = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M2.5 7.5l3 3 6-7"/></svg>';
  const ICON_CAM = '<svg class="cam-ic" viewBox="0 0 16 16" aria-hidden="true"><rect x="1.5" y="4" width="13" height="9"/><circle cx="8" cy="8.5" r="2.4"/><path d="M5.5 4l1-2h3l1 2"/></svg>';

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const params = new URLSearchParams(location.search);
  const state = {
    index: null, cache: new Map(), L: null, R: null,
    pt: 0, T: 1, playing: false, speed: 10, sel: 0, S: 0,
    reveal: params.get('reveal') === 'end' ? 'end' : 'always',
    opts: { map: true, risk: false, ring: true, trail: true, eval: false, sightLog: false, ff: params.get('ff') !== '0' },
    quiet: new Uint8Array(1), lastT: -1, dirty: true, fast: false,
  };
  const panes = ['L', 'R'].map(side => {
    const fig = $(`#pane${side}`);
    const canvas = $('canvas.map', fig), cam = $('canvas.cam-cv', fig);
    return {
      side, fig, canvas, ctx: canvas.getContext('2d'), cam, camCtx: cam.getContext('2d'),
      head: $('.run-head', fig), label: $('.pane-label', fig), how: $('.run-how', fig), outcome: $('.outcome', fig),
      badges: $('.map-badges', fig), toasts: $('.toasts', fig), camCap: $('.cam-cap', fig),
      base: Object.assign(document.createElement('canvas'), { width: 1, height: 1 }),
      risk: Object.assign(document.createElement('canvas'), { width: 1, height: 1 }),
      px: 0, pad: 16, map: 500, cell: 6, dpr: 1, baseKey: '', riskKey: '', camKey: '',
    };
  });

  // ---------- helpers ----------
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, k) => a + (b - a) * k;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const hName = i => `Hiker ${i + 1}`;
  const hDot = i => `<span class="dot hk-${i % 8}" aria-hidden="true"></span>`;
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
  const f2 = v => Number(v).toFixed(2);
  const lower = s => String(s).toLowerCase();
  function clock(t, stepS) {
    const s = Math.round(t * stepS);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }
  function decodeMask(b64, n) {
    const bin = atob(b64), out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
    return out;
  }
  function decodeU16(b64, n) {
    const bin = atob(b64), out = new Uint16Array(n);
    for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    return out;
  }
  function fatal(html) {
    const el = $('#fatal');
    el.innerHTML = `<div>${html}</div>`;
    el.hidden = false;
  }
  async function getJSON(url) {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return res.json();
  }

  // ---------- episode preprocessing ----------
  function prep(raw, entry) {
    const n = raw.world.size, N = n * n, frames = raw.frames, cfg = raw.config;
    const pass = decodeMask(raw.world.passable, N);
    const firstSeen = new Int32Array(N).fill(INF);
    const tEnd = frames[frames.length - 1].t;
    const fi = new Int32Array(tEnd + 1);
    for (let t = 0, k = 0; t <= tEnd; t++) {
      while (k + 1 < frames.length && frames[k + 1].t <= t) k++;
      fi[t] = k;
    }
    const nh = frames[0].h.length;
    const path = Array.from({ length: nh }, () => new Int16Array(frames.length * 2));
    const sight = [], bySid = new Map(), events = [];
    let prevMode = frames[0].h.map(h => h[3]);
    let predPrev = frames[0].p ? frames[0].p[2] : null;
    frames.forEach((f, k) => {
      for (const sid of f.n) if (firstSeen[sid] === INF) firstSeen[sid] = f.t;
      f.h.forEach((h, i) => { path[i][2 * k] = h[0]; path[i][2 * k + 1] = h[1]; });
      for (const s of f.s) {
        const o = { sid: s[0], r: s[1], c: s[2], conf: s[3], h: s[4], real: s[5], t: f.t, changes: [] };
        sight.push(o); bySid.set(o.sid, o);
        events.push({ t: f.t, kind: 'sighting', h: o.h, s: o });
      }
      const decided = new Set();
      for (const e of f.e) {
        const [h, kind] = e;
        if (kind === 'relook') {
          const s = bySid.get(e[2]);
          if (s) s.changes.push([f.t, e[4]]);
          events.push({ t: f.t, kind, h, before: e[3], after: e[4], s });
        } else if (kind === 'scan') {
          events.push({ t: f.t, kind, h, found: !!e[2] });
        } else if (kind === 'escape') {
          decided.add(h); events.push({ t: f.t, kind, h, alarm: e[2] });
        } else if (kind === 'home') {
          decided.add(h); events.push({ t: f.t, kind, h });
        } else if (kind === 'escaped' || kind === 'caught') {
          events.push({ t: f.t, kind, h });
        } else if (kind === 'predator') {
          events.push({ t: f.t, kind, h: -1, state: e[2], from: predPrev });
          predPrev = e[2];
        }
      }
      f.h.forEach((h, i) => {
        if (h[2] === 'active' && h[3] !== prevMode[i] && !decided.has(i)) events.push({ t: f.t, kind: 'mode', h: i, mode: h[3] });
      });
      prevMode = f.h.map(h => h[3]);
    });
    const ownSeen = cfg.private_map && raw.world.own_seen ? raw.world.own_seen.map(b => decodeU16(b, N)) : null;
    const ended = raw.final.every(x => x.status !== 'active') || tEnd >= cfg.t_max - 1;
    const final = raw.final.map(x => x.status);
    const m = raw.metrics || {};
    return {
      raw, entry, id: raw.id, label: raw.label, cfg, metrics: m, n, N, pass, firstSeen, ownSeen, frames, fi, tEnd, nh, path, sight, events,
      exits: raw.world.exits, trailhead: raw.world.trailhead, ended, final,
      nOut: final.filter(s => s === 'escaped').length, nCaught: final.filter(s => s === 'caught').length,
      nLost: final.filter(s => s === 'active').length,
      beta: Number(m.beta ?? cfg.beta ?? 0) || 0,
      views: null, manifest: null,
    };
  }
  const mapOf = ep => (ep.ownSeen ? ep.ownSeen[clamp(state.sel, 0, ep.nh - 1)] : ep.firstSeen);
  const visOf = ep => (ep.cfg.vis ?? ep.cfg.view_radius * (1 - 0.75 * ep.cfg.fog));
  const frameAt = (ep, t) => ep.frames[ep.fi[clamp(t, 0, ep.tEnd)]];
  const frameIdx = (ep, t) => ep.fi[clamp(t, 0, ep.tEnd)];
  function hikerStatus(ep, i, t) {
    const h = frameAt(ep, t).h[i];
    if (h[2] !== 'active') return h[2];
    if (t >= ep.tEnd && ep.ended) return 'lost';
    return h[3];
  }
  function confAt(s, t) {
    let c = s.conf;
    for (const [tc, v] of s.changes) if (tc <= t) c = v;
    return c;
  }
  function howShared(cfg) {
    if (cfg.private_map && cfg.private_sightings) return 'Private maps';
    if (cfg.private_map) return 'Private terrain maps, shared sightings';
    if (cfg.private_sightings) return 'Shared terrain map, private sightings';
    return 'One shared map';
  }
  const fogVis = beta => (beta > 0 ? Math.round(3.912 / beta) : null);   // Koschmieder: visibility = 3.912 / beta

  // ---------- data loading ----------
  async function loadEpisode(id) {
    if (state.cache.has(id)) return state.cache.get(id);
    const entry = state.index.episodes.find(e => e.id === id);
    if (!entry) throw new Error(`unknown episode ${id}`);
    const ep = prep(await getJSON(`data/${entry.file}`), entry);
    await loadViews(ep);
    state.cache.set(id, ep);
    return ep;
  }
  async function loadViews(ep) {
    const byH = new Map();
    const push = v => { if (!byH.has(v.hiker)) byH.set(v.hiker, []); byH.get(v.hiker).push(v); };
    try {
      const m = await getJSON(`data/frames/${encodeURIComponent(ep.id)}/manifest.json`);
      ep.manifest = m;
      for (const v of m.views || []) push(v);
    } catch (_) {
      const cam = ep.entry && ep.entry.camera;           // older exports: frame list only, no detections
      if (cam) for (const [h, ts] of Object.entries(cam)) for (const t of ts) push({ t, hiker: Number(h), kind: null, detections: null, bare: true });
    }
    for (const list of byH.values()) list.sort((a, b) => a.t - b.t);
    ep.views = byH;
  }
  function viewAt(ep, h, t) {
    const list = ep.views && ep.views.get(h);
    if (!list) return null;
    let best = null;
    for (const v of list) { if (v.t <= t) best = v; else break; }
    return best;
  }
  const frameUrl = (ep, v) => `data/frames/${encodeURIComponent(ep.id)}/${v.t}_${v.hiker}.jpg`;
  const zoomUrl = (ep, v) => `data/frames/${encodeURIComponent(ep.id)}/zoom/${v.t}_${v.hiker}.jpg`;
  const imgCache = new Map();
  function getImg(url) {
    let o = imgCache.get(url);
    if (!o) {
      o = { img: new Image(), ok: false, err: false };
      o.img.decoding = 'async';
      o.img.onload = () => { o.ok = true; invalidateCams(); };
      o.img.onerror = () => { o.err = true; invalidateCams(); };
      o.img.src = url;
      imgCache.set(url, o);
    }
    return o;
  }
  function invalidateCams() { for (const p of panes) p.camKey = ''; state.dirty = true; }

  async function init() {
    let index;
    try {
      index = await getJSON('data/index.json');
    } catch (err) {
      fatal(`<strong>No replay data found</strong>Run <code>uv run python -m lost_together.export</code>, then serve this folder with <code>python -m http.server -d viewer 8000</code>.<br>(${esc(err.message)})`);
      return;
    }
    if (!index.episodes || index.episodes.length === 0) { fatal('<strong>No episodes in index.json</strong>'); return; }
    state.index = index;
    for (const sel of [$('#pickL'), $('#pickR')]) {
      sel.innerHTML = index.episodes.map(e => {
        const fog = e.fog ? ` · fog ${e.fog}` : '';
        return `<option value="${esc(e.id)}">${esc(e.label)} · seed ${esc(e.seed)}${esc(fog)}</option>`;
      }).join('');
    }
    const ids = index.episodes.map(e => e.id);
    const pick = (p, d) => (ids.includes(params.get(p)) ? params.get(p) : d);
    const dL = index.default?.left && ids.includes(index.default.left) ? index.default.left : ids[0];
    const dR = index.default?.right && ids.includes(index.default.right) ? index.default.right : ids[ids.length - 1];
    $('#pickL').value = pick('left', dL);
    $('#pickR').value = pick('right', dR);
    if (params.has('speed')) setSpeed(Number(params.get('speed')) || 10);
    if (params.get('eval') === '1') state.opts.eval = true;
    if (params.get('risk') === '1') state.opts.risk = true;
    if (params.get('ring') === '0') state.opts.ring = false;
    if (params.get('trail') === '0') state.opts.trail = false;
    if (params.get('map') === '0') state.opts.map = false;
    if (params.get('capture') === '1') document.body.classList.add('capture');
    if (params.has('hiker')) state.sel = clamp((Number(params.get('hiker')) || 1) - 1, 0, 7);
    syncToggles();
    buildKeys();
    try { await Promise.race([document.fonts.load(`700 16px ${FONT}`), new Promise(r => setTimeout(r, 1500))]); } catch (_) { /* fonts optional */ }
    buildLegend();
    await selectEpisodes();
    if (params.has('t')) seek(Number(params.get('t')) || 0);
    if (params.get('play') === '1') setPlaying(true);
    requestAnimationFrame(loop);
    loadFindings();
    loadAgent();
  }
  async function selectEpisodes() {
    try {
      const [L, R] = await Promise.all([loadEpisode($('#pickL').value), loadEpisode($('#pickR').value)]);
      state.L = L; state.R = R;
    } catch (err) {
      fatal(`<strong>Episode failed to load</strong>${esc(err.message)}`);
      return;
    }
    state.T = Math.max(state.L.tEnd, state.R.tEnd);
    state.pt = clamp(state.pt, 0, state.T);
    $('#scrub').max = state.T;
    panes[0].ep = state.L; panes[1].ep = state.R;
    for (const p of panes) {
      p.label.textContent = p.ep.label;
      p.how.textContent = howShared(p.ep.cfg);
      p.fig.setAttribute('aria-label', `${p.side === 'L' ? 'Left' : 'Right'} run: ${p.ep.label}`);
      p.baseKey = ''; p.riskKey = ''; p.camKey = '';
    }
    $('#rosterL').textContent = state.L.label;
    $('#rosterR').textContent = state.R.label;
    describeRun();
    buildQuiet();
    buildTicks();
    buildRoster();
    layout(true);
    state.lastT = -1; state.dirty = true;
  }
  function describeRun() {
    const { L, R } = state, c = L.cfg, m = c.cell_metres;
    const same = L.raw.seed === R.raw.seed;
    const vis = fogVis(L.beta);
    const parts = [
      '<b>Simulation</b>',
      same ? `seed ${esc(L.raw.seed)}` : `seeds ${esc(L.raw.seed)} and ${esc(R.raw.seed)}`,
      `${esc(c.perception)} perception`,
      vis ? `fog β ${L.beta} (visibility ≈${vis} m)` : 'clear air',
      `${c.n_hikers} hikers`, `${L.n * m} m forest`,
      `predator speed ${c.speed_ratio}× hiker`, `sensing radius ${c.sense_radius * m} m`,
    ];
    $('#runline').innerHTML = parts.join(' · ');
  }

  // ---------- layout: both runs get the same map size ----------
  function layout(force) {
    const board = $('#board'), cs = getComputedStyle(board);
    const W = window.innerWidth;
    const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
    const gap = parseFloat(cs.columnGap) || 24;
    const inner = board.clientWidth - padX;
    // S: run column width; the camera strip (frame + 3x tile) spans it at S/2 high and comes first.
    // M: map side. On a wide screen the map takes the height that is left, and the room beside it
    // becomes a rail for the run's latest events instead of covering the map.
    let S, M;
    // Lock the board to one viewport only when it is tall enough to film (1080p); shorter laptop
    // screens scroll a little instead of shrinking the camera frames to thumbnails.
    const fit = W >= 1100 && window.innerHeight >= 900;
    document.body.classList.toggle('fit', fit);
    if (fit) {
      const boardH = board.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
      const head = Math.max(panes[0].head.offsetHeight, panes[1].head.offsetHeight);
      const capH = CAP_H, stageGaps = 2 * 10;
      const side = clamp(inner * 0.24, 380, 460);
      S = (inner - 2 * gap - side) / 2;
      const avail = boardH - head - capH - stageGaps;
      if (avail - S / 2 < 0.62 * S) S = avail / 1.12;     // keep the map at least ~0.6 of the strip
      M = Math.min(S, avail - S / 2);
    } else if (W >= 1100) {
      S = (inner - 2 * gap - clamp(inner * 0.27, 360, 460)) / 2; M = S;
    } else if (W > 720) {
      S = (inner - gap) / 2; M = S;
    } else {
      S = inner; M = S;
    }
    S = Math.max(240, Math.floor(S / 2) * 2);
    M = Math.max(220, Math.floor(Math.min(M, S)));
    const dpr = window.devicePixelRatio || 1;
    if (!force && S === state.S && M === state.M && panes.every(p => p.dpr === dpr)) return;
    state.S = S; state.M = M;
    document.documentElement.style.setProperty('--S', `${S}px`);
    document.documentElement.style.setProperty('--M', `${M}px`);
    document.body.classList.toggle('railed', S - M >= 190);
    // The map plate is exactly M x M and lines up with the camera strip's left edge; the canvas adds a
    // transparent margin (pad) on every side that overhangs into the gutter for exit marks and labels.
    const pad = Math.round(clamp(M * 0.028, 10, 18));
    document.documentElement.style.setProperty('--pad', `${pad}px`);
    for (const p of panes) {
      p.canvas.width = p.canvas.height = Math.round((M + 2 * pad) * dpr);
      p.cam.width = Math.round(S * dpr); p.cam.height = Math.round((S / 2) * dpr);
      p.px = M + 2 * pad; p.cw = S; p.dpr = dpr;
      p.pad = pad;
      p.map = M;
      p.cell = p.map / (p.ep ? p.ep.n : 80);
      p.camKey = '';
    }
    state.dirty = true;
  }
  let layoutQueued = false;
  const queueLayout = () => {
    if (layoutQueued) return;
    layoutQueued = true;
    requestAnimationFrame(() => { layoutQueued = false; layout(false); });
  };
  const ro = new ResizeObserver(queueLayout);
  ro.observe($('#board'));
  for (const p of panes) ro.observe(p.head);       // the outcome text arrives after the first layout
  window.addEventListener('resize', queueLayout);

  // ---------- rendering: map ----------
  function renderBase(p, ep, t) {
    const seenAt = mapOf(ep);
    const key = `${ep.id}|${t}|${state.opts.map}|${ep.ownSeen ? state.sel : ''}`;
    if (p.baseKey === key) return;
    p.baseKey = key;
    const n = ep.n;
    if (p.base.width !== n) { p.base.width = p.base.height = n; p.baseImg = null; }
    const bctx = p.base.getContext('2d');
    if (!p.baseImg) p.baseImg = bctx.createImageData(n, n);
    const d = p.baseImg.data;
    for (let i = 0; i < ep.N; i++) {
      const pal = ep.pass[i] ? TERRAIN.ground : TERRAIN.trees;
      let rgb;
      if (!state.opts.map) rgb = pal[3];
      else {
        const fs = seenAt[i];
        if (fs > t) rgb = pal[0];
        else if (t - fs < FRESH_STEPS) {
          const k = (t - fs) / FRESH_STEPS, a = pal[2], b = pal[1];
          rgb = [lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k)];
        } else rgb = pal[1];
      }
      d[4 * i] = rgb[0]; d[4 * i + 1] = rgb[1]; d[4 * i + 2] = rgb[2]; d[4 * i + 3] = 255;
    }
    bctx.putImageData(p.baseImg, 0, 0);
  }
  function renderRisk(p, ep, t) {
    const own = ep.cfg.private_sightings ? clamp(state.sel, 0, ep.nh - 1) : -1;
    const key = `${ep.id}|${t}|${own}`;
    if (p.riskKey === key) return;
    p.riskKey = key;
    const n = ep.n, c = ep.cfg, sig = c.risk_sigma, R = Math.floor(3 * sig);
    const risk = new Float32Array(ep.N);
    for (const s of ep.sight) {
      if (s.t > t) break;
      if (own >= 0 && s.h !== own) continue;
      const w = confAt(s, t) * Math.pow(0.5, (t - s.t) / c.evidence_halflife);
      if (w < 0.02) continue;
      const r0 = Math.round(s.r), c0 = Math.round(s.c);
      for (let r = Math.max(0, r0 - R); r <= Math.min(n - 1, r0 + R); r++) {
        for (let q = Math.max(0, c0 - R); q <= Math.min(n - 1, c0 + R); q++) {
          risk[r * n + q] += w * Math.exp(-((r - s.r) ** 2 + (q - s.c) ** 2) / (2 * sig * sig));
        }
      }
    }
    if (p.risk.width !== n) { p.risk.width = p.risk.height = n; }
    const rctx = p.risk.getContext('2d');
    const img = rctx.createImageData(n, n), d = img.data;
    for (let i = 0; i < ep.N; i++) {
      const v = clamp(risk[i] / 1.2, 0, 1);
      d[4 * i] = 230; d[4 * i + 1] = Math.round(lerp(110, 170, 1 - v)); d[4 * i + 2] = 0; d[4 * i + 3] = Math.round(175 * Math.sqrt(v));
    }
    rctx.putImageData(img, 0, 0);
  }

  function drawPane(p, pt) {
    const ep = p.ep;
    if (!ep || !p.px) return;
    const ctx = p.ctx, cs = p.cell, t = Math.floor(pt), a = pt - t, tt = Math.min(t, ep.tEnd);
    const ae = t >= ep.tEnd ? 0 : a;
    ctx.setTransform(p.dpr, 0, 0, p.dpr, 0, 0);
    ctx.clearRect(0, 0, p.px, p.px);    // the margin stays transparent; the neatline frames the plate
    ctx.translate(p.pad, p.pad);
    renderBase(p, ep, tt);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(p.base, 0, 0, ep.n, ep.n, 0, 0, p.map, p.map);
    if (state.opts.risk) {
      renderRisk(p, ep, tt);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(p.risk, 0, 0, ep.n, ep.n, 0, 0, p.map, p.map);
    }
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1;
    ctx.strokeRect(-0.5, -0.5, p.map + 1, p.map + 1);
    const X = c => (c + 0.5) * cs, Y = r => (r + 0.5) * cs;
    const f0 = frameAt(ep, tt), f1 = frameAt(ep, tt + 1);

    drawTrailhead(ctx, X(ep.trailhead[1]), Y(ep.trailhead[0]), cs);
    for (const [r, c] of ep.exits) {
      const known = !state.opts.map || mapOf(ep)[r * ep.n + c] <= tt;
      drawExit(ctx, X(c), Y(r), cs, known, r, c, ep.n, p.map);
    }

    if (state.opts.trail) {
      // Routes: others as fine lines, the followed hiker last, on a paper casing so it reads over any terrain.
      const k = frameIdx(ep, tt);
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      const route = i => {
        const P = ep.path[i];
        ctx.beginPath();
        ctx.moveTo(X(P[1]), Y(P[0]));
        for (let j = 1; j <= k; j++) ctx.lineTo(X(P[2 * j + 1]), Y(P[2 * j]));
      };
      const wSel = Math.max(2, cs * 0.34), wOther = Math.max(1.25, cs * 0.2);
      for (let i = 0; i < ep.nh; i++) {
        if (i === state.sel) continue;
        route(i);
        ctx.strokeStyle = HIKER[i % HIKER.length]; ctx.globalAlpha = 0.6; ctx.lineWidth = wOther; ctx.stroke();
      }
      if (state.sel < ep.nh) {
        route(state.sel);
        ctx.globalAlpha = 0.9; ctx.strokeStyle = C.paper; ctx.lineWidth = wSel + 2.5; ctx.stroke();
        ctx.globalAlpha = 1; ctx.strokeStyle = HIKER[state.sel % HIKER.length]; ctx.lineWidth = wSel; ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    const vis = visOf(ep);
    const hpos = f0.h.map((h, i) => [lerp(h[0], f1.h[i][0], ae), lerp(h[1], f1.h[i][1], ae)]);
    f0.h.forEach((h, i) => {
      if (h[2] !== 'active' || (t >= ep.tEnd && ep.ended)) return;
      ctx.beginPath();
      ctx.arc(X(hpos[i][1]), Y(hpos[i][0]), vis * cs, 0, 2 * Math.PI);
      ctx.fillStyle = i === state.sel ? 'rgba(19,26,22,0.07)' : 'rgba(19,26,22,0.035)'; ctx.fill();
      ctx.strokeStyle = i === state.sel ? 'rgba(19,26,22,0.45)' : 'rgba(19,26,22,0.18)'; ctx.lineWidth = 1; ctx.stroke();
    });

    for (const s of ep.sight) {
      if (s.t > tt) break;
      const conf = confAt(s, tt), age = tt - s.t;
      const w = conf * Math.pow(0.5, age / ep.cfg.evidence_halflife);
      drawSighting(ctx, X(s.c), Y(s.r), cs, s, w, age + ae, state.opts.eval);
    }

    if (f0.p) {
      const pr = lerp(f0.p[0], f1.p ? f1.p[0] : f0.p[0], ae), pc = lerp(f0.p[1], f1.p ? f1.p[1] : f0.p[1], ae);
      if (state.opts.ring) {
        ctx.beginPath();
        ctx.arc(X(pc), Y(pr), ep.cfg.sense_radius * cs, 0, 2 * Math.PI);
        ctx.fillStyle = f0.p[2] === 'pursue' ? 'rgba(213,94,0,0.10)' : 'rgba(213,94,0,0.04)';
        ctx.fill();
        ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(213,94,0,0.75)'; ctx.stroke(); ctx.setLineDash([]);
      }
      drawPredator(ctx, X(pc), Y(pr), cs, f0.p[2], pt, p.map);
    }

    const order = f0.h.map((h, i) => i).sort((i, j) => {
      const ai = f0.h[i][2] === 'active', aj = f0.h[j][2] === 'active';
      return (ai - aj) || ((i === state.sel) - (j === state.sel));
    });
    for (const i of order) drawHiker(ctx, X(hpos[i][1]), Y(hpos[i][0]), cs, i, hikerStatus(ep, i, tt), i === state.sel, pt);
  }

  function drawTrailhead(ctx, x, y, cs) {
    const s = Math.max(6, cs * 0.9);
    ctx.beginPath();
    ctx.moveTo(x, y - s); ctx.lineTo(x + s, y + s * 0.8); ctx.lineTo(x - s, y + s * 0.8); ctx.closePath();
    ctx.fillStyle = C.paper; ctx.fill();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.stroke();
  }
  function drawExit(ctx, x, y, cs, known, r, c, n, mapW) {
    const s = Math.max(8, cs * 1.25);
    ctx.save();
    ctx.lineWidth = known ? 2.5 : 1.5;
    ctx.strokeStyle = known ? C.ink : C.ink3;
    ctx.fillStyle = known ? C.ink : 'rgba(251,252,249,0.85)';
    if (!known) ctx.setLineDash([3, 3]);
    ctx.fillRect(x - s, y - s, 2 * s, 2 * s);
    ctx.strokeRect(x - s, y - s, 2 * s, 2 * s);
    ctx.setLineDash([]);
    if (known) {      // a white door mark inside the exit square
      ctx.fillStyle = C.paper;
      ctx.fillRect(x - s * 0.35, y - s * 0.5, s * 0.7, s);
    }
    const fs = Math.max(11, Math.round(cs * 1.3)), lh = fs + 6;
    ctx.font = `700 ${fs}px ${FONT}`;
    if ('letterSpacing' in ctx) ctx.letterSpacing = '0.04em';
    ctx.textBaseline = 'middle';
    let tx = x, ty = y, align = 'center';
    if (r <= 1) ty = y + s + 4 + lh / 2; else if (r >= n - 2) ty = y - s - 4 - lh / 2;
    else if (c <= 1) { tx = x + s + 6; align = 'left'; } else { tx = x - s - 6; align = 'right'; }
    const text = known ? 'EXIT' : 'EXIT (unseen)';
    const w = ctx.measureText(text).width;
    let bx = align === 'center' ? tx - w / 2 : align === 'left' ? tx : tx - w;
    bx = Math.round(clamp(bx, 4, mapW - w - 4));
    const by = Math.round(ty - lh / 2);
    ctx.textAlign = 'left';
    ctx.fillStyle = known ? C.ink : 'rgba(251,252,249,0.94)';
    ctx.fillRect(bx - 4, by, w + 8, lh);
    if (!known) { ctx.lineWidth = 1; ctx.strokeStyle = C.rule; ctx.strokeRect(bx - 3.5, by + 0.5, w + 7, lh - 1); }
    ctx.fillStyle = known ? C.paper : C.ink2;
    ctx.fillText(text, bx, by + lh / 2 + 0.5);
    ctx.restore();
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function drawSighting(ctx, x, y, cs, s, w, age, evalView) {
    const rad = Math.max(5, cs * 0.75);
    if (evalView) {
      ctx.globalAlpha = clamp(Math.max(w, 0.25), 0, 1);
      if (s.real) {
        ctx.beginPath(); ctx.arc(x, y, rad * 0.85, 0, 2 * Math.PI);
        ctx.fillStyle = C.predator; ctx.fill();
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.stroke();
      } else {
        ctx.strokeStyle = C.fake; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI); ctx.stroke();
        const k = rad * 0.55;
        ctx.beginPath(); ctx.moveTo(x - k, y - k); ctx.lineTo(x + k, y + k); ctx.moveTo(x + k, y - k); ctx.lineTo(x - k, y + k); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      return;
    }
    if (w < 0.04) return;
    ctx.globalAlpha = clamp(w + 0.1, 0.2, 1);
    ctx.lineWidth = 2.6; ctx.strokeStyle = C.report;
    ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = 1; ctx.strokeStyle = C.reportDark;
    ctx.beginPath(); ctx.arc(x, y, rad + 1.6, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillStyle = C.reportDark;
    ctx.beginPath(); ctx.arc(x, y, Math.max(1.6, rad * 0.28), 0, 2 * Math.PI); ctx.fill();
    if (age < 8) {
      const k = age / 8;
      ctx.globalAlpha = (1 - k) * clamp(w + 0.35, 0, 1);
      ctx.lineWidth = 2; ctx.strokeStyle = C.report;
      ctx.beginPath(); ctx.arc(x, y, rad * (1 + 2.2 * k), 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  function drawPredator(ctx, x, y, cs, st, pt, mapW) {
    const s = Math.max(9, cs * 1.25);
    ctx.save();
    if (st === 'pursue') {
      const k = (pt * 0.9) % 1;
      ctx.globalAlpha = 0.65 * (1 - k);
      ctx.strokeStyle = C.predator; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, s * (1.1 + 1.2 * k), 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.globalAlpha = st === 'handling' ? 0.6 : 1;
    ctx.beginPath();
    ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath();
    ctx.fillStyle = C.predator; ctx.fill();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.globalAlpha = 1;
    tag(ctx, `predator, ${PRED_LABEL[st] || st}`, x, y - s - 13, st === 'pursue' ? C.predator : C.ink, '#ffffff', cs, mapW);
    ctx.restore();
  }
  function tag(ctx, text, x, y, bg, fg, cs, mapW) {
    const fs = Math.max(11, Math.round(cs * 1.25));
    ctx.font = `700 ${fs}px ${FONT}`;
    const w = Math.ceil(ctx.measureText(text).width) + 12, h = fs + 8;
    x = clamp(x, w / 2 + 2, Math.max(w / 2 + 2, mapW - w / 2 - 2));
    y = Math.max(y, h / 2 + 2);
    const bx = Math.round(x - w / 2), by = Math.round(y - h / 2);
    ctx.fillStyle = bg;
    ctx.fillRect(bx, by, w, h);
    ctx.fillStyle = fg; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(text, bx + 6, by + h / 2 + 0.5);
  }
  function drawHiker(ctx, x, y, cs, i, st, selected, pt) {
    const R = Math.max(7, cs * 1.05), col = HIKER[i % HIKER.length];
    ctx.save();
    if (selected && st !== 'escaped' && st !== 'caught') {
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, R + 8, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath();
      for (const a of [0, 0.5, 1, 1.5]) {       // four ticks: the camera this viewer follows
        const ca = Math.cos(a * Math.PI), sa = Math.sin(a * Math.PI);
        ctx.moveTo(x + ca * (R + 8), y + sa * (R + 8)); ctx.lineTo(x + ca * (R + 13), y + sa * (R + 13));
      }
      ctx.stroke();
    }
    if (st === 'caught') {
      ctx.beginPath(); ctx.arc(x, y, R, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(251,252,249,0.92)'; ctx.fill();
      ctx.setLineDash([3, 3]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 2;
      ctx.stroke(); ctx.setLineDash([]);
      numberText(ctx, i, x, y, R, C.ink2);
      ctx.restore(); return;
    }
    if (st === 'escaped') {
      ctx.beginPath(); ctx.arc(x, y, R * 0.95, 0, 2 * Math.PI);
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.stroke();
      ctx.strokeStyle = C.ink; ctx.lineWidth = Math.max(2, R * 0.28); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath(); ctx.moveTo(x - R * 0.45, y + R * 0.02); ctx.lineTo(x - R * 0.1, y + R * 0.38); ctx.lineTo(x + R * 0.5, y - R * 0.35); ctx.stroke();
      ctx.restore(); return;
    }
    if (st === 'escape') {
      const k = (pt * 1.2) % 1;
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(x, y, R + 3.5, 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 0.5 * (1 - k); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, R + 3.5 + 6 * k, 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 1;
    } else if (st === 'home') {
      ctx.setLineDash([4, 3]); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.arc(x, y, R + 3.5, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.globalAlpha = st === 'lost' ? 0.5 : 1;
    ctx.beginPath(); ctx.arc(x, y, R, 0, 2 * Math.PI);
    ctx.fillStyle = col; ctx.fill();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.stroke();
    numberText(ctx, i, x, y, R, i === 0 || i === 6 ? '#ffffff' : C.ink);
    ctx.restore();
  }
  function numberText(ctx, i, x, y, R, color) {
    ctx.font = `700 ${Math.round(R * 1.2)}px ${FONT}`;
    ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), x, y + 1);
  }

  // ---------- rendering: camera strip (full frame + 3x look-closer tile) ----------
  function detStyle(d, view) {
    const lab = [[String(d.cls), 700], [` ${f2(d.score)}`, 400]];
    if (!state.opts.eval) return { col: C.report, fg: C.ink, dash: null, label: lab };
    if (d.matches_truth && view.kind === 'real') return { col: C.predator, fg: '#ffffff', dash: null, label: [...lab, [' · predator', 700]] };
    if (d.matches_truth && view.kind === 'decoy') return { col: '#ffffff', fg: C.ink, dash: [5, 4], label: [...lab, [' · decoy', 700]] };
    return { col: '#ffffff', fg: C.ink, dash: [5, 4], label: [...lab, [' · false positive', 700]] };
  }
  // A label on a camera tile. bounds.placed collects earlier labels so new ones never cover them:
  // a colliding label steps down (then up) until it is clear.
  // text is a string (bold) or a list of [segment, weight] pairs, e.g. a class name in bold and its score
  // in regular weight. Every edge lands on a whole pixel so labels stay crisp.
  function chip(ctx, text, x, y, bg, fg, bounds, fs) {
    const segs = Array.isArray(text) ? text : [[text, 700]];
    const padX = 6;
    let tw = 0;
    for (const [s, wt] of segs) { ctx.font = `${wt} ${fs}px ${FONT}`; tw += ctx.measureText(s).width; }
    const w = Math.ceil(tw) + 2 * padX, h = fs + 8;
    x = Math.round(clamp(x, bounds.x, bounds.x + bounds.w - w));
    y = Math.round(clamp(y, bounds.y, bounds.y + bounds.h - h));
    const placed = bounds.placed || (bounds.placed = []);
    const hits = yy => placed.find(r => x < r.x + r.w && x + w > r.x && yy < r.y + r.h && yy + h > r.y);
    let tries = 0, dir = 1, y0 = y;
    for (let r = hits(y); r && tries < 12; r = hits(y), tries++) {
      y = dir > 0 ? r.y + r.h + 2 : r.y - h - 2;
      if (y + h > bounds.y + bounds.h || y < bounds.y) { if (dir < 0) break; dir = -1; y = y0; }
    }
    y = Math.round(clamp(y, bounds.y, bounds.y + bounds.h - h));
    placed.push({ x, y, w, h });
    ctx.fillStyle = bg; ctx.fillRect(x, y, w, h);
    ctx.fillStyle = fg; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    let cx = x + padX;
    for (const [s, wt] of segs) {
      ctx.font = `${wt} ${fs}px ${FONT}`;
      ctx.fillText(s, cx, y + h / 2 + 0.5);
      cx += ctx.measureText(s).width;
    }
    return { x, y, w, h };
  }
  function drawBoxes(ctx, dets, view, ox, oy, sc, bounds, fs) {
    const sorted = [...dets].sort((a, b) => a.score - b.score);
    for (const d of sorted) {
      const st = detStyle(d, view);
      const [bx, by, bw, bh] = d.box;
      // whole-pixel box with a 2px stroke on a dark casing, so the edge is crisp on bright foliage and sky
      const x = Math.round(ox + bx * sc), y = Math.round(oy + by * sc);
      const w = Math.max(4, Math.round(bw * sc)), h = Math.max(4, Math.round(bh * sc));
      ctx.save();
      ctx.beginPath(); ctx.rect(bounds.x, bounds.y, bounds.w, bounds.h); ctx.clip();
      ctx.lineWidth = 4; ctx.strokeStyle = C.casing;
      ctx.strokeRect(x, y, w, h);
      ctx.lineWidth = 2; ctx.strokeStyle = st.col;
      if (st.dash) ctx.setLineDash(st.dash);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      ctx.restore();
      // the label tab sits flush on the box's top edge, aligned to its outer left edge (below it when there is no room)
      const lh = fs + 8, above = y - 1 - lh >= bounds.y;
      chip(ctx, st.label, x - 1, above ? y - 1 - lh : y + h + 1, st.col, st.fg, bounds, fs);
    }
  }
  function drawTruth(ctx, box, view, ox, oy, sc, bounds, fs) {
    if (!box) return;
    const [bx, by, bw, bh] = box;
    let x = ox + bx * sc, y = oy + by * sc, w = bw * sc, h = bh * sc;
    const minS = 14;
    if (w < minS) { x -= (minS - w) / 2; w = minS; }
    if (h < minS) { y -= (minS - h) / 2; h = minS; }
    x = Math.round(x) - 3; y = Math.round(y) - 3; w = Math.round(w) + 6; h = Math.round(h) + 6;   // outside any detection box
    const k = Math.round(Math.max(5, Math.min(w, h) * 0.3));
    ctx.save();
    ctx.lineCap = 'square';
    const corners = () => {
      ctx.beginPath();
      for (const [cx, cy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
        ctx.moveTo(cx + dx * k, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * k);
      }
    };
    corners(); ctx.lineWidth = 4; ctx.strokeStyle = C.casing; ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = C.truth; ctx.stroke();
    ctx.restore();
    const what = view.kind === 'real' ? 'predator' : 'decoy';
    const dist = view.distance_m != null ? `, ${Math.round(view.distance_m)} m` : '';
    chip(ctx, `Truth: ${what}${dist}`, x, y + h + 3, C.truth, C.ink, bounds, fs);
  }
  function drawCamera(p, ep, t) {
    const ctx = p.camCtx, S = p.cw;
    if (!S) return;
    const sel = clamp(state.sel, 0, ep.nh - 1);
    const view = viewAt(ep, sel, t);
    const gap = 8, half = Math.floor((S - gap) / 2), H = S / 2, oy = Math.floor((H - half) / 2);
    const fs = S >= 560 ? 13 : 12;
    const L = { x: 0, y: oy, w: half, h: half }, R = { x: half + gap, y: oy, w: half, h: half };
    const fr = view && !view.bare ? getImg(frameUrl(ep, view)) : view ? getImg(frameUrl(ep, view)) : null;
    const zm = view && view.zoom ? getImg(zoomUrl(ep, view)) : null;
    const key = [ep.id, sel, view ? view.t : -1, state.opts.eval, S, p.dpr, fr && fr.ok, fr && fr.err, zm && zm.ok, zm && zm.err].join('|');
    if (key !== p.camKey) {
      p.camKey = key;
      ctx.setTransform(p.dpr, 0, 0, p.dpr, 0, 0);
      ctx.fillStyle = C.ground; ctx.fillRect(0, 0, S, H);
      ctx.fillStyle = C.well; ctx.fillRect(L.x, L.y, L.w, L.h); ctx.fillRect(R.x, R.y, R.w, R.h);
      // empty-state text, word-wrapped to the tile: a bold first line, quieter lines after it
      const say = (b, lines) => {
        const out = [];
        lines.forEach((ln, i) => {
          ctx.font = `${i === 0 ? 700 : 400} ${i === 0 ? fs + 2 : fs}px ${FONT}`;
          let cur = '';
          for (const w of ln.split(' ')) {
            const next = cur ? `${cur} ${w}` : w;
            if (cur && ctx.measureText(next).width > b.w - 28) { out.push([cur, i]); cur = w; } else cur = next;
          }
          out.push([cur, i]);
        });
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        out.forEach(([ln, i], k) => {
          ctx.font = `${i === 0 ? 700 : 400} ${i === 0 ? fs + 2 : fs}px ${FONT}`;
          ctx.fillStyle = i === 0 ? '#e7ebe5' : '#a9b3ac';
          ctx.fillText(ln, b.x + b.w / 2, b.y + b.h / 2 + (k - (out.length - 1) / 2) * (fs + 8));
        });
      };
      if (!view) {
        say(L, ['No frame yet', `No frame recorded for ${hName(sel)} at this step`]);
      } else if (fr.err) {
        say(L, ['Frame not exported', `step ${view.t}`]);
      } else if (fr.ok) {
        const sc = half / 640;
        ctx.drawImage(fr.img, L.x, L.y, half, half);
        // tile titles first, so detection and truth labels are placed around them
        const small = half < 300;
        chip(ctx, small ? `${hName(sel)} · full frame` : [[`${hName(sel)} · full frame`, 700], [' · YOLOX-s', 400]], L.x + 8, L.y + 8, C.title, '#ffffff', L, fs);
        const mb = ep.manifest && ep.manifest.beta, eb = ep.beta;
        if (mb != null && Math.abs(Number(mb) - eb) > 1e-6) {
          chip(ctx, small ? `Render β ${Number(mb)} · run β ${eb}` : Number(mb) === 0 ? `Rendered without fog · run β ${eb}` : `Rendered at β ${mb} · run β ${eb}`,
            L.x + 8, L.y + L.h - fs - 16, '#ffffff', C.ink, L, fs - 1);
        }
        if (state.opts.eval && view.truth_box) drawTruth(ctx, view.truth_box, view, L.x, L.y, sc, L, fs);
        if (view.zoom) {         // the loupe: region the detector re-ran on, leaders to the 3x tile
          const zs = 640 / (view.zoom.factor || 3);
          const zx = Math.round(L.x + view.zoom.origin[0] * sc), zy = Math.round(L.y + view.zoom.origin[1] * sc), zw = Math.round(zs * sc);
          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(zx + zw + 0.5, zy + 0.5); ctx.lineTo(R.x, R.y + 0.5);
          ctx.moveTo(zx + zw + 0.5, zy + zw - 0.5); ctx.lineTo(R.x, R.y + R.h - 0.5);
          ctx.stroke();
          ctx.lineWidth = 3; ctx.strokeStyle = C.casing;
          ctx.strokeRect(zx + 0.5, zy + 0.5, zw - 1, zw - 1);
          ctx.lineWidth = 1; ctx.strokeStyle = '#ffffff';
          ctx.strokeRect(zx + 0.5, zy + 0.5, zw - 1, zw - 1);
          ctx.restore();
        }
        if (view.detections) drawBoxes(ctx, view.detections, view, L.x, L.y, sc, L, fs);
      } else {
        say(L, ['Loading frame']);
      }
      if (view && view.zoom) {
        if (zm.ok) {
          const sc = half / 640;
          ctx.drawImage(zm.img, R.x, R.y, half, half);
          chip(ctx, half < 300 ? `${view.zoom.factor || 3}× look-closer` : [[`${view.zoom.factor || 3}× look-closer`, 700], [' · detector re-run', 400]], R.x + 8, R.y + 8, C.title, '#ffffff', R, fs);
          if (state.opts.eval && view.zoom.truth_box) drawTruth(ctx, view.zoom.truth_box, view, R.x, R.y, sc, R, fs);
          drawBoxes(ctx, view.zoom.detections || [], view, R.x, R.y, sc, R, fs);
        } else say(R, [zm.err ? 'Zoom tile not exported' : 'Loading zoom tile']);
      } else if (view && !view.bare) {
        say(R, ['No 3× tile for this frame', "Triggered by an uncertain report or a teammate's sighting."]);
      } else if (view) {
        say(R, ['Detections not exported']);
      }
    }
    // caption (text only changes with the step or the evaluator view)
    const capKey = `${key}|${t}`;
    if (p.capKey === capKey) return;
    p.capKey = capKey;
    p.camCap.innerHTML = camCaption(ep, sel, view, t);
    p.cam.setAttribute('aria-label', view ? `${hName(sel)} camera frame at step ${view.t}` : `${hName(sel)}: no frame yet`);
  }
  function detList(dets) {
    if (!dets || !dets.length) return `none above ${DET_MIN}`;
    return [...dets].sort((a, b) => b.score - a.score).slice(0, 3).map(d => `${esc(d.cls)} ${f2(d.score)}`).join(', ');
  }
  function camCaption(ep, sel, view, t) {
    const stepS = ep.cfg.step_seconds;
    const who = `<span class="who">${hDot(sel)}${hName(sel)} camera</span>`;
    const row = (html, cls = '') => `<span class="cap-row${cls ? ` ${cls}` : ''}">${html}</span>`;
    if (!view) return row(`${who} · no frame yet`);
    const age = t - view.t;
    const when = `step ${view.t}${age > 0 ? `, ${plural(age, 'step')} ago` : ''}`;
    if (view.bare || !view.detections) return row(`${who} · ${when} · detections not exported`);
    let s = row(`${who} · ${when} · YOLOX-s (cv2.dnn): ${detList(view.detections)}${view.zoom ? ` · 3× tile: ${detList(view.zoom.detections)}` : ''}`);
    if (state.opts.eval) {
      // a detection is only "true" when it matches the predator; matching a decoy is still a false alarm
      const all = [...view.detections, ...(view.zoom ? view.zoom.detections || [] : [])];
      const tp = all.filter(d => d.matches_truth && view.kind === 'real').length;
      const onDecoy = all.filter(d => d.matches_truth && view.kind === 'decoy').length;
      const fp = all.length - tp - onDecoy;
      const what = view.kind === 'real' ? `predator at ${Math.round(view.distance_m)} m` : view.kind === 'decoy' ? `decoy at ${Math.round(view.distance_m)} m, no predator` : 'no animal in view';
      let e = `<b>Evaluator</b> ${what}`;
      if (tp) e += ` · <span class="tp">${plural(tp, 'true detection')}</span>`;
      if (onDecoy) e += ` · <span class="fp">${onDecoy} on decoy</span>`;
      if (fp) e += ` · <span class="fp">${plural(fp, 'false positive')}</span>`;
      if (view.kind === 'real' && !tp) e += ' · predator not detected';
      s += row(e, 'eval');
    }
    if (view.render_agrees === false) {
      s += row(view.sim_detected
        ? '<b>Render disagrees with the run:</b> the calibrated draw detected the predator; this render does not.'
        : '<b>Render disagrees with the run:</b> the calibrated draw missed the predator; this render detects it.', 'note');
    }
    return s;
  }

  // ---------- DOM (changes only when the integer step changes) ----------
  function countsAt(ep, t) {
    let out = 0, caught = 0, active = 0;
    for (const h of frameAt(ep, t).h) { if (h[2] === 'escaped') out++; else if (h[2] === 'caught') caught++; else active++; }
    return { out, caught, active };
  }
  function updateDom(t) {
    for (const p of panes) {
      updateOutcome(p, p.ep, t);
      updateBadges(p, p.ep);
      updateToasts(p, p.ep, t);
    }
    updateVerdict(t);
    updateRoster(t);
    updateLog(t);
    $('#scrub').value = t;
    $('#scrub').style.setProperty('--p', `${(100 * t / Math.max(1, state.T)).toFixed(2)}%`);   // CSSOM, allowed under CSP
    $('#scrub').setAttribute('aria-valuetext', `step ${t}, ${clock(t, state.L.cfg.step_seconds)}`);
    $('#clockT').textContent = `step ${t}`;
    $('#clockM').textContent = clock(t, state.L.cfg.step_seconds);
  }
  function updateOutcome(p, ep, t) {
    const done = t >= ep.tEnd, show = state.reveal === 'always' || done, n = ep.nh, stepS = ep.cfg.step_seconds;
    let big, sub, cls;
    if (!show) {
      const c = countsAt(ep, t);
      big = `${c.out} of ${n} out so far`;
      sub = c.caught ? `${c.caught} intercepted so far` : 'outcome hidden until the end';
      cls = 'pending';
    } else {
      big = ep.nOut === n ? `All ${n} reached an exit` : `${ep.nOut} of ${n} reached an exit`;
      const parts = [];
      if (ep.nCaught) parts.push(`${ep.nCaught} intercepted`);
      if (ep.nLost) parts.push(`${ep.nLost} still lost at time limit (${clock(ep.cfg.t_max, stepS)})`);
      if (ep.nOut === n) parts.push(`last exit at ${clock(ep.tEnd, stepS)}`);
      if (ep.metrics.n_sightings_real === 0) parts.push('0 real sightings');
      else if (ep.metrics.n_sightings_real > 0) parts.push(`${plural(ep.metrics.n_sightings_real, 'real sighting')}`);
      sub = parts.join(' · ');
      cls = 'done';
    }
    const html = `<span class="o-big">${esc(big)}</span><span class="o-sub">${esc(sub)}</span>`;
    if (p.outHtml !== html) { p.outcome.innerHTML = html; p.outHtml = html; p.outcome.className = `outcome ${cls}`; }
  }
  function updateBadges(p, ep) {
    const sel = clamp(state.sel, 0, ep.nh - 1);
    const parts = [];
    if (state.opts.map && ep.ownSeen) parts.push(`<span class="badge">${hDot(sel)}${hName(sel)} private map</span>`);
    if (state.opts.risk) parts.push(`<span class="badge">Risk: ${ep.cfg.private_sightings ? `${hName(sel)} evidence only` : 'shared evidence'}</span>`);
    if (state.opts.eval) parts.push('<span class="badge eval">Evaluator view: ground truth</span>');
    const html = parts.join('');
    if (p.badgeHtml !== html) { p.badges.innerHTML = html; p.badgeHtml = html; }
  }
  function updateVerdict(t) {
    const { L, R } = state;
    const shown = state.reveal === 'always' || (t >= L.tEnd && t >= R.tEnd);
    const same = L.raw.seed === R.raw.seed;
    const sub = same ? `Paired by seed ${esc(L.raw.seed)}: same forest, hiker start and predator start.` : 'Different seeds: unpaired comparison.';
    let head;
    if (!shown) head = 'Outcome hidden until the end';
    else {
      const d = R.nOut - L.nOut;
      head = d === 0 ? 'Same outcome in both runs'
        : `+${plural(Math.abs(d), 'hiker')} out with ${esc(lower(d > 0 ? R.label : L.label))}`;
    }
    const html = `<strong>${head}</strong><span>${sub}</span>`;
    if (state.verdictHtml !== html) { $('#verdict').innerHTML = html; state.verdictHtml = html; }
  }
  function buildRoster() {
    const nh = Math.max(state.L.nh, state.R.nh);
    $('#roster tbody').innerHTML = Array.from({ length: nh }, (_, i) =>
      `<tr data-h="${i}"><td><button type="button" class="follow" data-h="${i}" aria-pressed="false" aria-keyshortcuts="${i + 1}">${hDot(i)}${hName(i)}${ICON_CAM}</button></td><td class="mL"></td><td class="mR"></td></tr>`).join('');
    for (const b of $$('#roster .follow')) b.addEventListener('click', () => selectHiker(Number(b.dataset.h)));
  }
  function chipHTML(st) {
    const icon = st === 'escaped' ? ICON_CHECK : '';
    return `<span class="chip ${st}">${icon}${MODE_LABEL[st] || esc(st)}</span>`;
  }
  function updateRoster(t) {
    for (const tr of $$('#roster tbody tr')) {
      const i = Number(tr.dataset.h);
      const on = i === state.sel;
      tr.classList.toggle('sel', on);
      $('.follow', tr).setAttribute('aria-pressed', String(on));
      for (const [p, cls] of [[panes[0], '.mL'], [panes[1], '.mR']]) {
        const st = i < p.ep.nh ? hikerStatus(p.ep, i, t) : null;
        const html = st ? chipHTML(st) : '';
        const td = $(cls, tr);
        if (td.dataset.html !== html) { td.innerHTML = html; td.dataset.html = html; }
      }
    }
  }
  function selectHiker(i) {
    state.sel = i; state.dirty = true; state.lastT = -1;
    for (const p of panes) { p.baseKey = ''; p.riskKey = ''; p.camKey = ''; }
  }

  function eventText(ev, ep) {
    const who = ev.h >= 0 ? `${hDot(ev.h)}<span>${hName(ev.h)}</span>` : '';
    const n2 = v => `<span class="num">${f2(v)}</span>`;
    const tagS = s => (state.opts.eval && s ? (s.real ? '<span class="tag real">real</span>' : '<span class="tag false">false cue</span>') : '');
    switch (ev.kind) {
      case 'sighting': return { cls: 'minor', html: `${who}<span>report, confidence</span>${n2(ev.s.conf)}${tagS(ev.s)}` };
      case 'relook': {
        const res = ev.after <= 0 ? '<span>· dismissed</span>' : (ev.after >= 0.75 ? '<span>· confirmed</span>' : '');
        return { cls: '', html: `${who}<span>look closer:</span>${n2(ev.before)}<span>→</span>${n2(ev.after)}${res}${tagS(ev.s)}` };
      }
      case 'scan': return { cls: ev.found ? 'major' : 'minor', html: `${who}<span>3× scan after teammate report ·</span><span>${ev.found ? 'detection' : 'no detection'}</span>` };
      case 'escape': return { cls: 'major', html: `${who}<span>escape decision</span><span>(alarm ${f2(ev.alarm)}, threshold ${esc(ep.cfg.escape_threshold)})</span>` };
      case 'home': return { cls: '', html: `${who}<span>heading home: walk time elapsed</span>` };
      case 'mode': return { cls: 'major', html: `${who}<span>${ev.mode === 'escape' ? 'joins group escape' : 'heading home'}</span>` };
      case 'escaped': return { cls: 'major', html: `${who}<span>reached an exit</span>` };
      case 'caught': return { cls: 'major', html: `${who}<span>intercepted</span>` };
      case 'predator': {
        const txt = ev.state === 'patrol'
          ? (ev.from === 'handling' ? 'Predator resumed patrol' : 'Predator lost the trail')
          : ({ pursue: 'Predator picked up a trail', handling: 'Predator paused' }[ev.state] || esc(ev.state));
        return { cls: ev.state === 'pursue' ? '' : 'minor', html: `<span class="pred-mark" aria-hidden="true"></span><span>${txt}</span>` };
      }
      default: return { cls: 'minor', html: esc(ev.kind) };
    }
  }
  function updateLog(t) {
    const rows = [];
    for (const p of panes) {
      for (const ev of p.ep.events) {
        if (ev.t > t) break;
        if (ev.kind === 'sighting' && !state.opts.sightLog) continue;
        rows.push([ev, p]);
      }
    }
    rows.sort((a, b) => b[0].t - a[0].t || (a[1].side < b[1].side ? -1 : 1));
    const fresh = state.playing ? t : -1;
    const html = rows.slice(0, 160).map(([ev, p]) => {
      const { cls, html } = eventText(ev, p.ep);
      return `<li class="${cls}${ev.t === fresh ? ' fresh' : ''}"><button type="button" class="row" data-t="${ev.t}" data-h="${ev.h}">` +
        `<span class="lt">${clock(ev.t, p.ep.cfg.step_seconds)}</span><span class="side-key" title="${esc(p.ep.label)}">${p.side}</span><span class="lx">${html}</span></button></li>`;
    }).join('');
    const out = html || '<li class="empty">No events yet. Press Space to play.</li>';
    if (state.logHtml !== out) { $('#log').innerHTML = out; state.logHtml = out; }
  }
  $('#log').addEventListener('click', e => {
    const b = e.target.closest('.row');
    if (!b) return;
    setPlaying(false);
    const h = Number(b.dataset.h);
    if (h >= 0) selectHiker(h);
    seek(Number(b.dataset.t));
  });
  function updateToasts(p, ep, t) {
    const items = [];
    const railed = document.body.classList.contains('railed');     // a rail beside the map holds more, for longer
    const keep = railed ? 4 * TOAST_STEPS : TOAST_STEPS, maxN = railed ? 6 : 3;
    for (const ev of ep.events) {
      if (ev.t > t) break;
      if (t - ev.t >= keep) continue;
      let text = null;
      if (ev.kind === 'escape' || (ev.kind === 'mode' && ev.mode === 'escape')) text = `${hName(ev.h)} escape decision`;
      else if (ev.kind === 'caught') text = `${hName(ev.h)} intercepted`;
      else if (ev.kind === 'escaped') text = `${hName(ev.h)} reached an exit`;
      else if (ev.kind === 'predator' && ev.state === 'pursue') text = 'Predator picked up a trail';
      else if (ev.kind === 'relook' && ev.after >= 0.75) text = `${hName(ev.h)} look closer: ${f2(ev.before)} → ${f2(ev.after)}`;
      else if (ev.kind === 'scan' && ev.found) text = `${hName(ev.h)} 3× scan: detection`;
      if (text) items.push({ ev, text });
    }
    const prev = p.toastKeys || new Set(), keys = new Set();
    const html = items.slice(-maxN).map(({ ev, text }) => {
      const mark = ev.h >= 0 ? hDot(ev.h) : '<span class="pred-mark"></span>';
      const key = `${ev.t}|${ev.kind}|${ev.h}`;
      keys.add(key);
      // only an event that just arrived during playback slides in; everything else is drawn in place
      const cls = [t - ev.t >= TOAST_STEPS ? 'old' : '', state.playing && !prev.has(key) ? 'in' : ''].filter(Boolean).join(' ');
      return `<li class="${cls}" data-k="${esc(key)}">${mark}<span class="tx">${esc(text)}</span><span class="tt">${clock(ev.t, ep.cfg.step_seconds)}</span></li>`;
    }).join('');
    p.toastKeys = keys;
    const cmp = html.replace(/ in"/g, '"').replace(/class="in"/g, 'class=""');
    if (p.toastCmp !== cmp) { p.toasts.innerHTML = html; p.toastCmp = cmp; }
  }
  function buildTicks() {
    const el = $('#ticks'), T = Math.max(1, state.T);
    el.textContent = '';
    for (const p of panes) {
      const s = p.side.toLowerCase();
      const add = (cls, t) => {
        const i = document.createElement('i');
        i.className = cls;
        i.style.left = `${(100 * t / T).toFixed(2)}%`;      // CSSOM, allowed under CSP
        el.append(i);
      };
      for (const ev of p.ep.events) if (ev.kind === 'escaped' || ev.kind === 'caught') add(`${s} ${ev.kind}`, ev.t);
      if (p.ep.tEnd < T) add('end', p.ep.tEnd);
    }
  }
  const MAJOR = ev => ev.kind === 'escape' || ev.kind === 'home' || ev.kind === 'mode' || ev.kind === 'caught' ||
    ev.kind === 'escaped' || (ev.kind === 'relook' && ev.after >= 0.75) || (ev.kind === 'scan' && ev.found) ||
    (ev.kind === 'predator' && ev.state === 'pursue');
  function buildQuiet() {
    const T = state.T, busy = new Uint8Array(T + 1);
    for (const p of panes) {
      for (const ev of p.ep.events) {
        if (!MAJOR(ev)) continue;
        for (let t = Math.max(0, ev.t - 20); t <= Math.min(T, ev.t + 10); t++) busy[t] = 1;
      }
      for (let t = Math.max(0, p.ep.tEnd - 20); t <= Math.min(T, p.ep.tEnd + 10); t++) busy[t] = 1;
    }
    for (let t = 0; t <= Math.min(T, 30); t++) busy[t] = 1;
    state.quiet = busy.map(b => 1 - b);
  }

  // ---------- legend + keys ----------
  function swatch(draw, w = 30, h = 30) {
    const dpr = window.devicePixelRatio || 1;
    const cv = document.createElement('canvas');
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.width = `${w}px`; cv.style.height = `${h}px`;
    cv.setAttribute('aria-hidden', 'true');
    const c = cv.getContext('2d'); c.scale(dpr, dpr);
    draw(c, w / 2, h / 2);
    return cv;
  }
  function fillLegend(ul, items) {
    ul.textContent = '';
    for (const [text, draw] of items) {
      const li = document.createElement('li');
      li.append(swatch(draw), document.createTextNode(text));
      ul.append(li);
    }
  }
  function buildLegend() {
    const terrain = (pal, k) => `rgb(${TERRAIN[pal][k].join(',')})`;
    fillLegend($('#legend'), [
      ['Hiker, walking', (c, x, y) => drawHiker(c, x, y, 9, 0, 'enjoy', false, 0)],
      ['Escaping to an exit', (c, x, y) => drawHiker(c, x, y, 7.5, 1, 'escape', false, 0)],
      ['Heading home', (c, x, y) => drawHiker(c, x, y, 7.5, 2, 'home', false, 0)],
      ['Reached an exit', (c, x, y) => drawHiker(c, x, y, 9, 3, 'escaped', false, 0)],
      ['Intercepted (leaves the map)', (c, x, y) => drawHiker(c, x, y, 9, 0, 'caught', false, 0)],
      ['Selected hiker (camera shown)', (c, x, y) => drawHiker(c, x, y, 5, 0, 'enjoy', true, 0)],
      ['Predator', (c, x, y) => { c.beginPath(); c.moveTo(x, y - 9); c.lineTo(x + 9, y); c.lineTo(x, y + 9); c.lineTo(x - 9, y); c.closePath(); c.fillStyle = C.predator; c.fill(); c.strokeStyle = C.ink; c.lineWidth = 1.5; c.stroke(); }],
      ['Predator sensing radius', (c, x, y) => { c.setLineDash([4, 3]); c.strokeStyle = C.predator; c.lineWidth = 1.5; c.beginPath(); c.arc(x, y, 11, 0, 2 * Math.PI); c.stroke(); }],
      ['Report (fades with age)', (c, x, y) => drawSighting(c, x, y, 9, {}, 0.9, 99, false)],
      ['Real sighting (evaluator)', (c, x, y) => drawSighting(c, x, y, 9, { real: true }, 1, 99, true)],
      ['False cue (evaluator)', (c, x, y) => drawSighting(c, x, y, 9, { real: false }, 1, 99, true)],
      ['Exit', (c, x, y) => { c.fillStyle = C.ink; c.fillRect(x - 9, y - 9, 18, 18); c.fillStyle = C.paper; c.fillRect(x - 3, y - 4.5, 6, 9); }],
      ['Explored: ground, trees', (c, x, y) => { c.fillStyle = terrain('ground', 1); c.fillRect(x - 12, y - 9, 12, 18); c.fillStyle = terrain('trees', 1); c.fillRect(x, y - 9, 12, 18); c.strokeStyle = C.rule; c.strokeRect(x - 12.5, y - 9.5, 25, 19); }],
      ['Unexplored', (c, x, y) => { c.fillStyle = terrain('ground', 0); c.fillRect(x - 12, y - 9, 12, 18); c.fillStyle = terrain('trees', 0); c.fillRect(x, y - 9, 12, 18); }],
      ['Risk (decaying evidence)', (c, x, y) => { for (let r = 12; r > 0; r -= 2) { c.fillStyle = `rgba(230,${Math.round(110 + 5 * r)},0,${(0.75 * (1 - r / 13)).toFixed(2)})`; c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill(); } }],
    ]);
    buildMiniKey();
    const box = (col, dash) => (c, x, y) => { c.fillStyle = C.well; c.fillRect(x - 15, y - 15, 30, 30); c.strokeStyle = col; c.lineWidth = 2; if (dash) c.setLineDash([4, 3]); c.strokeRect(x - 9, y - 8, 18, 16); };
    fillLegend($('#legendCam'), [
      ['Detection (hiker view)', box(C.report)],
      ['True detection: predator (evaluator)', box(C.predator)],
      ['False positive or decoy (evaluator)', box('#ffffff', true)],
      ['Ground-truth location (evaluator)', (c, x, y) => { c.fillStyle = C.well; c.fillRect(x - 15, y - 15, 30, 30); c.strokeStyle = C.truth; c.lineWidth = 2; c.beginPath(); for (const [cx, cy, dx, dy] of [[x - 8, y - 8, 1, 1], [x + 8, y - 8, -1, 1], [x - 8, y + 8, 1, -1], [x + 8, y + 8, -1, -1]]) { c.moveTo(cx + dx * 5, cy); c.lineTo(cx, cy); c.lineTo(cx, cy + dy * 5); } c.stroke(); }],
      ['Look-closer region (re-run at 3×)', (c, x, y) => { c.fillStyle = C.well; c.fillRect(x - 15, y - 15, 30, 30); c.strokeStyle = '#ffffff'; c.lineWidth = 1.5; c.strokeRect(x - 11, y - 3, 9, 9); c.lineWidth = 1; c.beginPath(); c.moveTo(x - 2, y - 3); c.lineTo(x + 14, y - 13); c.moveTo(x - 2, y + 6); c.lineTo(x + 14, y + 14); c.stroke(); }],
    ]);
  }
  // the compact key beside the board: the map marks a viewer meets first, evaluator marks only in that view
  function buildMiniKey() {
    const ul = $('#miniKey');
    if (!ul) return;
    const diamond = (c, x, y) => { c.beginPath(); c.moveTo(x, y - 7); c.lineTo(x + 7, y); c.lineTo(x, y + 7); c.lineTo(x - 7, y); c.closePath(); c.fillStyle = C.predator; c.fill(); c.strokeStyle = C.ink; c.lineWidth = 1.4; c.stroke(); };
    const items = [
      ['Hiker, walking', (c, x, y) => drawHiker(c, x, y, 7, 0, 'enjoy', false, 0)],
      ['Escaping to an exit', (c, x, y) => drawHiker(c, x, y, 5.5, 1, 'escape', false, 0)],
      ['Reached an exit', (c, x, y) => drawHiker(c, x, y, 7, 3, 'escaped', false, 0)],
      ['Intercepted', (c, x, y) => drawHiker(c, x, y, 7, 0, 'caught', false, 0)],
      ['Predator', diamond],
      ['Predator sensing radius', (c, x, y) => { c.setLineDash([3, 2.5]); c.strokeStyle = C.predator; c.lineWidth = 1.5; c.beginPath(); c.arc(x, y, 8, 0, 2 * Math.PI); c.stroke(); }],
      ...(state.opts.eval
        ? [['Real sighting (evaluator)', (c, x, y) => drawSighting(c, x, y, 7, { real: true }, 1, 99, true)],
          ['False cue (evaluator)', (c, x, y) => drawSighting(c, x, y, 7, { real: false }, 1, 99, true)]]
        : [['Report (fades with age)', (c, x, y) => drawSighting(c, x, y, 7, {}, 0.9, 99, false)]]),
      ['Exit', (c, x, y) => { c.fillStyle = C.ink; c.fillRect(x - 7, y - 7, 14, 14); c.fillStyle = C.paper; c.fillRect(x - 2.5, y - 3.5, 5, 7); }],
    ];
    ul.textContent = '';
    for (const [text, draw] of items) {
      const li = document.createElement('li');
      li.append(swatch(draw, 22, 22), document.createTextNode(text));
      ul.append(li);
    }
  }
  const KEYS = [
    ['Space', 'Play / pause'], ['← →', 'Step (Shift: 10 steps)'], ['Home End', 'Start / end'], ['↑ ↓', 'Speed up / down'],
    ['1–4', 'Select hiker (camera; own map when maps are private)'], ['E', 'Evaluator view: ground truth for sightings and boxes'],
    ['R S T M', 'Risk, sense ring, routes, living map'], ['O', 'Outcome: always shown / revealed at end'],
    ['F', 'Skip idle stretches at 5×'], ['C', 'Capture mode (hide controls)'], ['?', 'Keyboard help'],
  ];
  function buildKeys() {
    const html = KEYS.map(([k, d]) => `<dt>${k.split(' ').map(x => `<kbd>${esc(x)}</kbd>`).join(' ')}</dt><dd>${esc(d)}</dd>`).join('');
    $('#keysInline').innerHTML = html;
    $('#keysDialog').innerHTML = html;
  }

  // ---------- findings (data/results.json, optional) ----------
  async function loadFindings() {
    let r;
    try { r = await getJSON('data/results.json'); } catch (_) { return; }
    const items = Array.isArray(r && r.headline) ? r.headline.filter(x => x && x.label != null && x.value != null) : [];
    if (!items.length && !r.figure) return;
    $('#headline').innerHTML = items.map(x =>
      `<div><dt>${esc(x.label)}</dt><dd><span class="v">${esc(x.value)}</span>${x.note ? `<span class="n">${esc(x.note)}</span>` : ''}</dd></div>`).join('');
    if (typeof r.figure === 'string' && !/^[a-z]+:|^\/\//i.test(r.figure)) {
      const img = new Image();
      img.id = 'findingsImg'; img.decoding = 'async';
      $('#findingsFig').prepend(img);
      img.src = `data/${r.figure.replace(/^\.?\//, '')}`;
      img.alt = r.figure_alt || 'Experiment figure';
      $('#findingsCap').textContent = r.figure_caption || '';
      img.addEventListener('error', () => { $('#findingsFig').hidden = true; });
      $('#findingsFig').hidden = false;
    }
    $('#findings').hidden = false;
  }

  // ---------- agent decision traces (data/agent/*.json, optional) ----------
  const DEC_LABEL = { keep_going: 'keep going', look_again: 'look again', regroup: 'regroup', escape: 'escape' };
  async function listAgentFiles() {
    try {
      const j = await getJSON('data/agent/index.json');
      const list = Array.isArray(j) ? j : (j.files || []);
      if (list.length) return list.filter(f => typeof f === 'string');
    } catch (_) { /* fall through to a directory listing */ }
    try {
      const res = await fetch('data/agent/', { cache: 'no-cache' });
      if (!res.ok || !/html/.test(res.headers.get('content-type') || '')) return [];
      const html = await res.text();
      return [...html.matchAll(/href="([^"?#]+\.json)"/g)].map(m => decodeURIComponent(m[1]))
        .filter(f => !/\//.test(f) && f !== 'index.json');
    } catch (_) { return []; }
  }
  async function loadAgent() {
    // Preferred: data/agent/index.json {model, source, traces: [{point, title, note, llm, rule, frame, prev_frame}]}
    let idx = null;
    try { idx = await getJSON('data/agent/index.json'); } catch (_) { /* optional */ }
    if (idx && Array.isArray(idx.traces) && idx.traces.length) {
      const points = (await Promise.all(idx.traces.map(async tr => {
        const get = async f => { if (typeof f !== 'string' || !f) return null; try { return await getJSON(`data/agent/${encodeURIComponent(f)}`); } catch (_) { return null; } };
        const [llm, rule] = await Promise.all([get(tr.llm), get(tr.rule)]);
        return llm || rule ? { tr, llm, rule } : null;
      }))).filter(Boolean);
      if (points.length) { renderPoints(points, idx); return; }
    }
    const files = (await listAgentFiles()).slice(0, 200);
    const decisions = [];
    await Promise.all(files.map(async f => {
      try {
        const j = await getJSON(`data/agent/${encodeURIComponent(f)}`);
        const meta = (j && j.meta) || {};
        const list = Array.isArray(j) ? j : Array.isArray(j.decisions) ? j.decisions : [j];
        for (const d of list) if (d && Array.isArray(d.trace) && d.decision) decisions.push({ d, file: f, meta });
      } catch (_) { /* skip unreadable files */ }
    }));
    decisions.sort((a, b) => String(a.d.point_id).localeCompare(String(b.d.point_id)) || String(a.d.policy).localeCompare(String(b.d.policy)));
    renderAgent(decisions);
  }
  function agentImg(path) {
    if (typeof path !== 'string' || !path) return null;
    if (/^[a-z]+:|^\/\//i.test(path)) return null;
    const m = path.match(/viewer\/(data\/.+)$/);
    if (m) return m[1];
    if (path.startsWith('/')) return null;               // absolute path on the machine that ran it
    return `data/agent/${path.replace(/^\.?\//, '')}`;
  }
  const isStub = x => /stub/i.test([x.file, x.meta.mode, x.meta.model, x.d.mode, x.d.model].filter(Boolean).join(' '));
  function renderAgent(decs) {
    const body = $('#agentBody');
    if (!decs.length) {
      body.innerHTML = `<div class="agent-empty">
        <div>
          <p>No agent traces loaded. Source: decision traces written by <code>lost_together/agent/loop.py</code>, copied to <code>viewer/data/agent/</code>. Live traces follow the GPU run.</p>
          <p class="muted">Each trace records one hiker's observation, OpenCV 5 tool calls, decision and ranger action. Only real traces are shown.</p>
        </div>
        <ol class="loop" aria-label="Trace contents">
          <li><b>Observe</b><span>Hiker frame with numbered OpenCV detection candidates, plus a shared-map summary.</span></li>
          <li><b>Model turn</b><span>qwen3-vl on an AWS GPU reads the observation and may request tools.</span></li>
          <li><b>OpenCV 5 tools</b><span><code>look_closer</code>: re-detect on a zoomed, defogged crop. <code>compare_frames</code>: motion. <code>query_map</code>: living map. Served by the MCP server.</span></li>
          <li><b>Decision</b><span>Keep going, look again, regroup or escape; validated against a strict format.</span></li>
          <li><b>Fallback</b><span>On timeout or invalid output, the rule-based planner decides; the reason is logged.</span></li>
          <li><b>Ranger</b><span>Escape orders are evacuation broadcasts; a person approves or overrides each one, and the choice is logged.</span></li>
        </ol>
      </div>`;
      return;
    }
    const anyStub = decs.some(isStub);
    body.innerHTML = `${anyStub ? '<p class="stub-banner">Pipeline demo with a stub model, not a result.</p>' : ''}
      <div class="agent-grid">
        <ol class="dec-list" id="decList">${decs.map((x, i) => {
          const d = x.d, act = d.action || d.decision;
          return `<li><button type="button" class="dec-btn" data-i="${i}" aria-current="${i === 0}">
            <span class="pid">${esc(d.point_id)}</span>
            <span class="pol">${d.policy === 'rule' ? 'rule-based planner' : `${esc(d.policy)} agent`}${d.source === 'fallback' ? ' · rule fallback' : ''}</span>
            <span class="act"><span class="act-chip ${act === 'escape' ? 'escape' : ''}">${esc(DEC_LABEL[act] || act)}</span></span></button></li>`;
        }).join('')}</ol>
        <div id="decDetail"></div>
      </div>`;
    const show = i => {
      $$('#decList .dec-btn').forEach(b => b.setAttribute('aria-current', String(Number(b.dataset.i) === i)));
      $('#decDetail').innerHTML = decisionHTML(decs[i]);
    };
    $('#decList').addEventListener('click', e => { const b = e.target.closest('.dec-btn'); if (b) show(Number(b.dataset.i)); });
    show(0);
  }
  // ----- decision points: the same frame decided by the LLM agent and by the rule-based planner -----
  const actOf = d => (d ? d.action || d.decision : null);
  const actChip = a => `<span class="act-chip ${a === 'escape' ? 'escape' : ''}">${esc(DEC_LABEL[a] || a || 'none')}</span>`;
  function renderPoints(points, idx) {
    const body = $('#agentBody');
    const model = idx.model || (points[0].llm && points[0].llm.model) || null;
    const stub = /stub/i.test(`${model || ''} ${idx.source || ''}`);
    body.innerHTML = `${stub ? '<p class="stub-banner">Pipeline demo with a stub model, not a result.</p>' : ''}
      <p class="agent-lede">${['One decision point', 'Two decision points', 'Three decision points', 'Four decision points'][points.length - 1] || `${points.length} decision points`} from the live agent run${model ? ` (<b>${esc(model)}</b>, AWS GPU)` : ''}. Tools reference OpenCV 5 detection candidates by number; the rule-based planner decided the same frames. Tool calls, model turns and approvals are copied from the trace${idx.source ? ` (<code>${esc(idx.source)}</code>)` : ''}.</p>
      <div class="agent-grid">
        <ol class="dec-list" id="decList">${points.map((x, i) => `<li><button type="button" class="dec-btn" data-i="${i}" aria-current="${i === 0}">
            <span class="pid">${esc(x.tr.title || x.tr.point)}</span>
            <span class="pol">${esc(x.tr.point)}</span>
            <span class="vs"><span>Agent ${actChip(actOf(x.llm))}</span><span>Rule ${actChip(actOf(x.rule))}</span></span></button></li>`).join('')}</ol>
        <div id="decDetail" class="dec-detail" aria-live="polite"></div>
      </div>`;
    const show = i => {
      $$('#decList .dec-btn').forEach(b => b.setAttribute('aria-current', String(Number(b.dataset.i) === i)));
      $('#decDetail').innerHTML = pointHTML(points[i], model);
      drawPointFrame($('#decDetail canvas'), points[i]);
    };
    $('#decList').addEventListener('click', e => { const b = e.target.closest('.dec-btn'); if (b) show(Number(b.dataset.i)); });
    show(0);
  }
  function pointHTML(x, model) {
    const tr = x.tr;
    const side = (d, name) => {
      if (!d) return `<section class="dec-col"><h3>${name}</h3><p class="dec-why">No trace for this policy.</p></section>`;
      const meta = [
        d.source === 'fallback' ? '<b>rule fallback</b>' : '',
        d.llm_calls ? `${d.llm_calls} model calls` : '',
        `${d.tool_calls ?? 0} tool calls`,
        d.latency_s != null ? `${Number(d.latency_s).toFixed(2)} s` : '',
        d.tokens_in ? `${d.tokens_in.toLocaleString('en')} tokens in` : '',
      ].filter(Boolean).map(s => `<span>${s}</span>`).join('');
      return `<section class="dec-col"><h3>${name} ${actChip(actOf(d))}</h3>
        ${d.why ? `<p class="dec-why">“${esc(d.why)}”</p>` : ''}
        ${d.fallback_reason ? `<p class="dec-why"><b>Rule fallback:</b> ${esc(d.fallback_reason)}</p>` : ''}
        <p class="dec-meta">${meta}</p>
        <ol class="steps">${d.trace.map(stepHTML).join('')}</ol></section>`;
    };
    return `<div class="dec-top">
        <figure class="dec-frame"><canvas role="img" aria-label="Hiker camera frame with numbered detection candidates"></canvas>
          <figcaption>Hiker frame with the numbered candidates the agent's tools reference. Dashed box: region re-run by <code>look_closer</code>.</figcaption></figure>
        <div class="dec-story"><h3>${esc(tr.title || tr.point)}</h3>${tr.note ? `<p>${esc(tr.note)}</p>` : ''}<p class="pt">Point ${esc(tr.point)}</p></div>
      </div>
      <div class="dec-cols">${side(x.llm, model ? `LLM agent <small>${esc(model)}</small>` : 'LLM agent')}${side(x.rule, 'Rule-based planner')}</div>`;
  }
  function drawPointFrame(cv, x) {
    if (!cv || typeof x.tr.frame !== 'string' || /^[a-z]+:|^\/|\.\./i.test(x.tr.frame)) { if (cv) cv.closest('figure').hidden = true; return; }
    const d = x.llm || x.rule;
    const obs = d && d.trace.find(s => s.kind === 'observe');
    const cands = (obs && (obs.candidates || (obs.top ? [obs.top] : []))) || [];
    const look = d && d.trace.find(s => s.kind === 'tool' && s.name === 'look_closer' && s.result && s.result.crop_box);
    const img = new Image();
    img.onload = () => {
      const S = Math.round(cv.clientWidth || 420), dpr = window.devicePixelRatio || 1, sc = S / 640;
      cv.width = cv.height = Math.round(S * dpr);
      const ctx = cv.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.drawImage(img, 0, 0, S, S);
      const B = { x: 0, y: 0, w: S, h: S }, fs = 13;
      chip(ctx, [['Hiker camera', 700], [' · YOLOX and blob-search candidates', 400]], 8, 8, C.title, '#ffffff', B, fs);
      if (look) {
        const [bx, by, bw, bh] = look.result.crop_box;
        ctx.save(); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffffff';
        ctx.strokeRect(bx * sc - 3, by * sc - 3, bw * sc + 6, bh * sc + 6); ctx.restore();
      }
      for (const c of [...cands].sort((a, b) => a.score - b.score)) {
        const x = Math.round(c.box[0] * sc), y = Math.round(c.box[1] * sc);
        const w = Math.max(4, Math.round(c.box[2] * sc)), h = Math.max(4, Math.round(c.box[3] * sc));
        ctx.lineWidth = 4; ctx.strokeStyle = C.casing; ctx.strokeRect(x, y, w, h);
        ctx.lineWidth = 2; ctx.strokeStyle = C.report; ctx.strokeRect(x, y, w, h);
        chip(ctx, [[`#${c.id} ${c.cls}`, 700], [` ${f2(c.score)}`, 400]], x - 1, y - 1 - (fs + 8), C.report, C.ink, B, fs);
      }
    };
    img.onerror = () => { cv.closest('figure').hidden = true; };
    img.src = `data/agent/${x.tr.frame.replace(/^\.?\//, '')}`;
  }

  function decisionHTML(x) {
    const d = x.d, act = d.action || d.decision;
    const model = d.model || x.meta.model;
    const changed = d.action && d.action !== d.decision;
    const meta = [
      model ? `Model: <b>${esc(model)}</b>` : 'Model: not recorded',
      `source <b>${esc(d.source)}</b>`,
      `${d.tool_calls ?? 0} tool calls`, `${d.llm_calls ?? 0} model calls`,
      d.latency_s != null ? `<b>${Number(d.latency_s).toFixed(2)} s</b> total` : '',
      d.tokens_in ? `${d.tokens_in} tokens in, ${d.tokens_out} out` : '',
      d.cost_usd ? `$${Number(d.cost_usd).toFixed(4)}` : '',
    ].filter(Boolean).map(s => `<span>${s}</span>`).join('');
    return `<div class="dec-head"><span class="big">${esc(DEC_LABEL[d.decision] || d.decision)}${changed ? ` → ${esc(DEC_LABEL[act] || act)}` : ''}</span>
      <span>${d.policy === 'rule' ? 'Rule-based planner' : `${esc(d.policy)} agent`} · point ${esc(d.point_id)}</span></div>
      ${d.why ? `<p class="dec-why">${esc(d.why)}</p>` : ''}
      ${d.fallback_reason ? `<p class="dec-why"><b>Rule fallback:</b> ${esc(d.fallback_reason)}</p>` : ''}
      <p class="dec-meta">${meta}</p>
      <ol class="steps">${d.trace.map(stepHTML).join('')}</ol>`;
  }
  function stepHTML(s) {
    const T = s.t_s != null ? `<small>${Number(s.t_s).toFixed(2)} s</small>` : '';
    const img = agentImg(s.image);
    const pic = img ? `<img src="${esc(img)}" alt="" loading="lazy">` : '<span></span>';
    const row = (k, d, cls = '') => `<li class="${cls}"><span class="k">${k}${T}</span><span class="d">${d}</span>${pic}</li>`;
    switch (s.kind) {
      case 'observe': {
        const c = s.top;
        const top = c ? `top #${esc(c.id)}: ${esc(c.cls)} ${f2(c.score)}` : 'no candidates';
        const vis = s.frame && s.frame.visibility != null ? `<span class="sub">frame visibility ${f2(s.frame.visibility)}, haze ${f2(s.frame.haze)}</span>` : '';
        return row('Observe', `${esc(s.n_candidates ?? 0)} candidates from ${esc(s.detector || 'the detector')}, ${top}${vis}`);
      }
      case 'llm': {
        const asked = Array.isArray(s.tool_calls) && s.tool_calls.length ? `requested ${s.tool_calls.map(esc).join(', ')}` : (s.ok ? 'answered' : 'failed');
        const sub = `<span class="sub">${s.latency_s != null ? `${Number(s.latency_s).toFixed(2)} s` : ''}${s.cached ? ', cached' : ''}${s.tokens_in ? `, ${s.tokens_in} tokens in` : ''}${s.error ? `, ${esc(s.error)}` : ''}</span>`;
        return row('Model turn', `${asked}${sub}`, s.ok === false ? 'fallback' : '');
      }
      case 'tool': {
        const r = s.result || {};
        let d;
        if (s.ok === false) d = `failed: ${esc(s.error || 'error')}`;
        else if (s.name === 'look_closer' && r.after) d = `#${esc(r.candidate_id)}: ${esc(r.before?.cls)} ${f2(r.before?.score ?? 0)} → ${esc(r.after.cls)} ${f2(r.after.score)}<span class="sub">zoom ${esc(r.zoom)}×${r.contrast_gain ? `, contrast gain ${esc(r.contrast_gain)}` : ''}</span>`;
        else if (s.name === 'compare_frames') d = `#${esc(r.candidate_id ?? '')} ${r.moving ? 'moving' : 'static'}${r.motion_px != null ? `, ${Number(r.motion_px).toFixed(1)} px` : ''}`;
        else if (s.name === 'query_map') d = `risk near hiker ${f2(r.risk_near_hiker ?? 0)}${r.nearest_exit ? `, nearest exit ${Math.round(r.nearest_exit.distance_m)} m` : ''}<span class="sub">${(r.recent_sightings || []).length} recent group sightings</span>`;
        else d = esc(JSON.stringify(r).slice(0, 160));
        return row(`Tool <small>${esc(s.name)}${s.by ? `, called by ${esc(s.by)}` : ''}</small>`, d, s.ok === false ? 'fallback' : '');
      }
      case 'validate': return row('Validate', s.ok ? 'format valid' : `rejected: ${esc(s.error || '')}`, s.ok ? '' : 'fallback');
      case 'fallback': return row('Fallback', `rule-based planner: ${esc(s.reason || '')}`, 'fallback');
      case 'ranger': {
        const by = s.by || (s.approval || {}).by;
        const auto = by === 'auto';
        const what = auto ? `${esc(s.outcome || 'approved')} automatically (batch run, no human review)` : `${esc(s.outcome || '')}${by ? ` by ${esc(by)}` : ''}`;
        return row('Ranger', `${what}${s.autonomy ? `<span class="sub">autonomy: ${esc(s.autonomy)}${s.note && !auto ? ` · ${esc(s.note)}` : ''}</span>` : ''}`, 'ranger');
      }
      default: return row(esc(s.kind || 'step'), esc(JSON.stringify(s).slice(0, 160)));
    }
  }

  // ---------- playback ----------
  let lastTs = null;
  function loop(ts) {
    if (state.playing) {
      if (lastTs !== null) {
        const fast = state.opts.ff && state.quiet[Math.floor(state.pt)] === 1;
        state.pt += Math.min(0.25, (ts - lastTs) / 1000) * state.speed * (fast ? 5 : 1);
        if (fast !== state.fast) { state.fast = fast; $('#ffInd').hidden = !fast; }
        if (state.pt >= state.T) { state.pt = state.T; setPlaying(false); }
      }
      lastTs = ts; state.dirty = true;
    } else lastTs = null;
    if (state.dirty && state.L && state.R) {
      const t = Math.floor(state.pt);
      if (t !== state.lastT) { updateDom(t); state.lastT = t; }
      for (const p of panes) {
        drawPane(p, state.playing ? state.pt : t);
        drawCamera(p, p.ep, Math.min(t, p.ep.tEnd));
      }
      state.dirty = false;
    }
    requestAnimationFrame(loop);
  }
  const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15L19.5 12z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 4.5h4v15H6zM14 4.5h4v15h-4z"/></svg>';
  function setPlaying(on) {
    if (on && state.pt >= state.T) state.pt = 0;
    state.playing = on;
    if (!on) { state.fast = false; $('#ffInd').hidden = true; }
    const b = $('#btnPlay');
    b.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
    b.setAttribute('aria-label', on ? 'Pause' : 'Play');
    state.dirty = true;
  }
  function seek(t) {
    state.pt = clamp(Math.round(t), 0, state.T);
    state.dirty = true;
  }
  const SPEEDS = [3, 6, 10, 20, 40];
  function setSpeed(v) {
    state.speed = v;
    const sel = $('#speed');
    if (![...sel.options].some(o => Number(o.value) === v)) sel.add(new Option(`${v} steps/s`, v));
    sel.value = String(v);
  }
  const TOGGLES = { optMap: 'map', optRisk: 'risk', optRing: 'ring', optTrail: 'trail', optEval: 'eval', optFF: 'ff' };
  function syncToggles() {
    for (const [id, k] of Object.entries(TOGGLES)) $(`#${id}`).setAttribute('aria-pressed', String(state.opts[k]));
    $('#optSightLog').checked = state.opts.sightLog;
  }
  function toggle(k) {
    state.opts[k] = !state.opts[k];
    syncToggles();
    if (k === 'eval') buildMiniKey();
    for (const p of panes) { p.baseKey = ''; p.riskKey = ''; p.camKey = ''; p.capKey = ''; }
    state.lastT = -1; state.dirty = true;
  }

  // ---------- wiring ----------
  $('#btnPlay').addEventListener('click', () => setPlaying(!state.playing));
  $('#btnBack').addEventListener('click', () => { setPlaying(false); seek(Math.ceil(state.pt) - 1); });
  $('#btnFwd').addEventListener('click', () => { setPlaying(false); seek(Math.floor(state.pt) + 1); });
  $('#scrub').addEventListener('input', e => { setPlaying(false); seek(Number(e.target.value)); });
  $('#speed').addEventListener('change', e => setSpeed(Number(e.target.value)));
  $('#pickL').addEventListener('change', selectEpisodes);
  $('#pickR').addEventListener('change', selectEpisodes);
  for (const [id, k] of Object.entries(TOGGLES)) $(`#${id}`).addEventListener('click', () => toggle(k));
  $('#optSightLog').addEventListener('change', () => toggle('sightLog'));
  $('#btnLayers').addEventListener('click', () => {
    const open = document.body.classList.toggle('layers-open');
    $('#btnLayers').setAttribute('aria-expanded', String(open));
  });
  const help = $('#help');
  const openHelp = () => { if (!help.open) help.showModal(); };
  $('#btnHelp').addEventListener('click', openHelp);
  help.addEventListener('click', e => { if (e.target === help) help.close(); });

  document.addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (help.open) { if (e.key === '?') { help.close(); e.preventDefault(); } return; }
    const tag = e.target.tagName;
    if (tag === 'SELECT' && ['ArrowUp', 'ArrowDown', ' ', 'Enter'].includes(e.key)) return;
    if (tag === 'INPUT' && e.target.type === 'range' && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    if ((tag === 'BUTTON' || tag === 'A') && e.key === 'Enter') return;
    if (!state.L) return;
    const k = e.key;
    let handled = true;
    if (k === ' ' || k === 'k') setPlaying(!state.playing);
    else if (k === 'ArrowRight') { setPlaying(false); seek(Math.floor(state.pt) + (e.shiftKey ? 10 : 1)); }
    else if (k === 'ArrowLeft') { setPlaying(false); seek(Math.ceil(state.pt) - (e.shiftKey ? 10 : 1)); }
    else if (k === 'Home') seek(0);
    else if (k === 'End') { setPlaying(false); seek(state.T); }
    else if (k === 'ArrowUp') setSpeed(SPEEDS.find(s => s > state.speed) ?? SPEEDS[SPEEDS.length - 1]);
    else if (k === 'ArrowDown') setSpeed([...SPEEDS].reverse().find(s => s < state.speed) ?? SPEEDS[0]);
    else if (/^[1-8]$/.test(k)) { const i = Number(k) - 1; if (i < Math.max(state.L.nh, state.R.nh)) selectHiker(i); }
    else if (k === 'e' || k === 'E') toggle('eval');
    else if (k === 'r' || k === 'R') toggle('risk');
    else if (k === 's' || k === 'S') toggle('ring');
    else if (k === 't' || k === 'T') toggle('trail');
    else if (k === 'm' || k === 'M') toggle('map');
    else if (k === 'o' || k === 'O') { state.reveal = state.reveal === 'always' ? 'end' : 'always'; state.lastT = -1; state.dirty = true; }
    else if (k === 'f' || k === 'F') toggle('ff');
    else if (k === 'c' || k === 'C') { document.body.classList.toggle('capture'); queueLayout(); }
    else if (k === '?') openHelp();
    else handled = false;
    if (handled) e.preventDefault();
  });

  init().catch(err => fatal(`<strong>Viewer error</strong>${esc(err && err.message ? err.message : err)}`));
})();
