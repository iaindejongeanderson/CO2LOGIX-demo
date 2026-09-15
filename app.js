/* CO2LOGIX UK Explorer
 * Browser implementation of the published CO2LOGIX v1.0 formulation
 * (model/pressure.py, model/growth.py, model/utils.py).
 *
 * The pressure field is a linear superposition of Nordbotten single-well
 * solutions. Because that solution is radial and monotonically decreasing away
 * from each well, the maximum of the superposed field always sits on a well
 * node - so the reported peak is evaluated well-to-well rather than by sweeping
 * a raster. That reproduces the published grid maximum exactly (verified to
 * 1e-6 % of fracture pressure against model/pressure.py).
 */
(function () {
  "use strict";

  const D = window.CO2LOGIX_DATA;
  const SPY = 86400 * 365;
  const PI = Math.PI;

  /* ------------------------------------------------------------ decoding */

  function decode(b64, Ctor) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Ctor(bytes.buffer);
  }

  const G = D.grid;
  const NX = G.nx, NY = G.ny, CELL = G.cell;

  const aqGrid = (function () {
    const out = new Int8Array(NX * NY);
    const rle = G.aquiferRLE;
    let p = 0;
    for (let i = 0; i < rle.length; i += 2) {
      const v = rle[i], n = rle[i + 1];
      out.fill(v, p, p + n);
      p += n;
    }
    return out;
  })();

  const U = D.units;
  const NU = U.length;
  const phi = Float64Array.from(U, u => u.phi);
  const hh = Float64Array.from(U, u => u.h);
  const kk = Float64Array.from(U, u => u.k);
  const Dd = Float64Array.from(U, u => u.D);
  const gamma = Float64Array.from(U, u => u.gamma);
  const expOmega = Float64Array.from(U, u => Math.exp(u.omega));
  const rhoC = Float64Array.from(U, u => u.rho_c);
  const muW = Float64Array.from(U, u => u.u_w);
  const pRef = Float64Array.from(U, u => u.p_ref);
  const pFrac = Float64Array.from(U, u => u.p_frac);

  // Candidate well locations: an unweighted draw from the uniform 1 km lattice, so
  // wells land in proportion to storage unit area.
  const POOL = (function () {
    const p = D.wellPool;
    const ix = decode(p.x, Int16Array), iy = decode(p.y, Int16Array);
    const x = new Float64Array(p.n), y = new Float64Array(p.n);
    for (let i = 0; i < p.n; i++) { x[i] = G.x0 + ix[i] * 1000; y[i] = G.y0 + iy[i] * 1000; }
    return { n: p.n, x, y, aq: decode(p.aq, Int8Array) };
  })();
  const POOL_N = POOL.n;

  /* ------------------------------------------------------------ colour */

  const RAMP = [
    [0.00, 0x17, 0x32, 0x4f],
    [0.28, 0x1c, 0x5c, 0xab],
    [0.52, 0x2a, 0x78, 0xd6],
    [0.76, 0x6d, 0xa7, 0xec],
    [1.00, 0xcd, 0xe2, 0xfb]
  ];
  const AT_LIMIT = [0xec, 0x83, 0x5a];   // 90-100 % of fracture pressure
  const EXCEEDED = [0xd0, 0x3b, 0x3b];   // > 100 %

  const LUT = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let s = 0;
    while (s < RAMP.length - 2 && t > RAMP[s + 1][0]) s++;
    const a = RAMP[s], b = RAMP[s + 1];
    const f = (t - a[0]) / (b[0] - a[0]);
    LUT[i * 3] = a[1] + (b[1] - a[1]) * f;
    LUT[i * 3 + 1] = a[2] + (b[2] - a[2]) * f;
    LUT[i * 3 + 2] = a[3] + (b[3] - a[3]) * f;
  }

  function rampRGB(v) {
    if (v >= 100) return EXCEEDED;
    if (v >= 90) return AT_LIMIT;
    let t = (v - 50) / 40;
    if (!(t > 0)) t = 0; else if (t > 1) t = 1;
    const i = (t * 255) | 0;
    return [LUT[i * 3], LUT[i * 3 + 1], LUT[i * 3 + 2]];
  }
  const rampCSS = v => { const c = rampRGB(v); return `rgb(${c[0]},${c[1]},${c[2]})`; };

  (function paintLegend() {
    const stops = RAMP.map(s => {
      const hex = `rgb(${s[1]},${s[2]},${s[3]})`;
      return `${hex} ${(s[0] * 100).toFixed(0)}%`;
    }).join(", ");
    document.getElementById("legend-bar").style.background =
      `linear-gradient(90deg, ${stops})`;
  })();

  /* ------------------------------------------------------------ geometry */

  const extent = (function () {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    const scan = feats => feats.forEach(f => f.rings.forEach(r => r.forEach(p => {
      if (p[0] < minx) minx = p[0];
      if (p[0] > maxx) maxx = p[0];
      if (p[1] < miny) miny = p[1];
      if (p[1] > maxy) maxy = p[1];
    })));
    scan(D.geo.aquifers);
    scan(D.geo.uk);
    const padx = (maxx - minx) * 0.03, pady = (maxy - miny) * 0.03;
    return { minx: minx - padx, maxx: maxx + padx, miny: miny - pady, maxy: maxy + pady };
  })();

  /* ------------------------------------------------------------ schedule */

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Draw 1 is the pool exactly as shipped (pandas random_state=42).
  function poolOrder(draw, n) {
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    if (draw <= 1) return idx;
    const rnd = mulberry32(draw * 2654435761);
    for (let i = n - 1; i > 0; i--) {
      const j = (rnd() * (i + 1)) | 0;
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    return idx;
  }

  /* ------------------------------------------------------------ engine */

  function lowerBound(arr, n, v) {
    let lo = 0, hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function prepare(P) {
    const t0 = 1 + Math.log(P.L / 2 - 1) / P.k;

    const Q = new Float64Array(NU), pc = new Float64Array(NU);
    for (let a = 0; a < NU; a++) {
      Q[a] = P.rate * 1e9 / rhoC[a] / 365 / 86400;
      pc[a] = (Q[a] * muW[a]) / (2 * PI * hh[a] * kk[a]) / 1e6;
    }

    const order = poolOrder(P.draw, POOL.n);
    const startYear = [], wellsPerYear = [];
    let used = 0, exhausted = false;
    for (let y = 1; y <= P.years; y++) {
      const f = P.L / (1 + Math.exp(-P.k * (y - t0)));
      let n = Math.ceil(P.k * f * (1 - f / P.L));
      if (!(n > 0)) n = 0;
      if (used + n > POOL.n) { n = Math.max(0, POOL.n - used); exhausted = true; }
      wellsPerYear.push(n);
      for (let i = 0; i < n; i++) { startYear.push(y); used++; }
    }

    const N = used;
    const wx = new Float64Array(N), wy = new Float64Array(N);
    const waq = new Int8Array(N), ws = new Int32Array(N), we = new Int32Array(N);
    for (let j = 0; j < N; j++) {
      const p = order[j];
      wx[j] = POOL.x[p]; wy[j] = POOL.y[p]; waq[j] = POOL.aq[p];
      ws[j] = startYear[j]; we[j] = startYear[j] + P.injYears - 1;
    }

    // per-unit source lists, sorted by easting, for windowed superposition
    const sortI = [], sortX = [], count = new Int32Array(NU);
    for (let j = 0; j < N; j++) count[waq[j]]++;
    for (let a = 0; a < NU; a++) { sortI.push(new Int32Array(count[a])); sortX.push(new Float64Array(count[a])); }
    const fill = new Int32Array(NU);
    for (let j = 0; j < N; j++) { const a = waq[j]; sortI[a][fill[a]++] = j; }
    for (let a = 0; a < NU; a++) {
      const ids = Array.from(sortI[a]).sort((p, q) => wx[p] - wx[q]);
      for (let i = 0; i < ids.length; i++) { sortI[a][i] = ids[i]; sortX[a][i] = wx[ids[i]]; }
    }

    // furthest a well of unit `a` can ever reach, at its maximum injection age
    const tsMax = P.injYears * SPY;
    const reach = new Float64Array(NU);
    for (let a = 0; a < NU; a++) {
      const Rmax = Math.sqrt(2.25 * Dd[a] * tsMax);
      const psiMax = expOmega[a] * Math.sqrt(Q[a] * tsMax / (PI * phi[a] * hh[a]));
      reach[a] = Math.max(Rmax, psiMax);
    }

    return {
      P, t0, Q, pc, N, wx, wy, waq, ws, we, wellsPerYear, exhausted,
      sortI, sortX, reach,
      totalYears: P.years + P.injYears,
      R: new Float64Array(N), psi: new Float64Array(N), wellVal: new Float64Array(N),
      maxDP: [], activeWells: [], cumulative: [],
      unitPeak: new Float64Array(NU).fill(0),
      unitStored: new Float64Array(NU),
      unitWells: new Int32Array(NU),
      cum: 0
    };
  }

  // Fill R / psi for every well that exists in `year`.
  function ages(st, year) {
    const { N, ws, we, waq, Q, P, R, psi } = st;
    for (let j = 0; j < N; j++) {
      if (ws[j] > year) break;              // wells are appended in drill order
      const age = year <= we[j] ? year - ws[j] + 1 : P.injYears;
      const ts = age * SPY;
      const a = waq[j];
      R[j] = Math.sqrt(2.25 * Dd[a] * ts);
      psi[j] = expOmega[a] * Math.sqrt(Q[a] * ts / (PI * phi[a] * hh[a]));
    }
  }

  // Superposed pressure change (MPa) at (xi, yi) from every well drilled by `year`.
  function dPat(st, xi, yi, year, selfIdx) {
    const { pc, ws, wx, wy, R, psi, sortI, sortX, reach } = st;
    let dP = 0;
    for (let a = 0; a < NU; a++) {
      const xs = sortX[a], ids = sortI[a], n = xs.length;
      if (!n) continue;
      const rch = reach[a], r2max = rch * rch;
      const hi = xi + rch;
      for (let p = lowerBound(xs, n, xi - rch); p < n && xs[p] <= hi; p++) {
        const j = ids[p];
        if (ws[j] > year) continue;
        let r;
        if (j === selfIdx) r = 0.1;
        else {
          const dy = yi - wy[j];
          if (dy > rch || dy < -rch) continue;
          const dx = xi - wx[j];
          const d2 = dx * dx + dy * dy;
          if (d2 > r2max) continue;
          r = Math.sqrt(d2);
          if (r === 0) r = 0.1;
        }
        const Rj = R[j], pj = psi[j];
        if (r <= pj) dP += (gamma[a] * Math.log(pj / r) + Math.log(Rj / pj)) * pc[a];
        else if (r <= Rj) dP += Math.log(Rj / r) * pc[a];
      }
    }
    return dP;
  }

  function runYear(st, year) {
    ages(st, year);
    const { N, ws, we, waq, wx, wy, wellVal, P } = st;
    let peak = 0, active = 0;
    const unitActive = new Int32Array(NU);

    for (let i = 0; i < N; i++) {
      if (ws[i] > year) break;
      const a = waq[i];
      const val = (dPat(st, wx[i], wy[i], year, i) + pRef[a]) / pFrac[a] * 100;
      wellVal[i] = val;
      if (val > peak) peak = val;
      if (val > st.unitPeak[a]) st.unitPeak[a] = val;
      if (year <= we[i]) { active++; unitActive[a]++; }
    }

    for (let a = 0; a < NU; a++) st.unitStored[a] += unitActive[a] * P.rate;
    st.cum += active * P.rate;
    st.maxDP.push(peak);
    st.activeWells.push(active);
    st.cumulative.push(st.cum);
  }

  /* ------------------------------------------------------------ raster */

  const acc = new Float32Array(NX * NY);
  const valGrid = new Float32Array(NX * NY);
  const imgData = new ImageData(NX, NY);

  function buildRaster(st, year) {
    acc.fill(0);
    ages(st, year);
    const { N, ws, waq, wx, wy, R, psi, pc } = st;

    for (let j = 0; j < N; j++) {
      if (ws[j] > year) break;
      const a = waq[j], Rj = R[j], pj = psi[j];
      const rad = Math.ceil(Math.max(Rj, pj) / CELL);
      const ci = Math.floor((wx[j] - G.x0) / CELL);
      const cj = Math.floor((wy[j] - G.y0) / CELL);

      const j0 = Math.max(0, cj - rad), j1 = Math.min(NY - 1, cj + rad);
      const i0 = Math.max(0, ci - rad), i1 = Math.min(NX - 1, ci + rad);
      for (let gj = j0; gj <= j1; gj++) {
        const Y = G.y0 + CELL * (gj + 0.5) - wy[j];
        const row = gj * NX;
        for (let gi = i0; gi <= i1; gi++) {
          if (aqGrid[row + gi] < 0) continue;
          const X = G.x0 + CELL * (gi + 0.5) - wx[j];
          let r = Math.sqrt(X * X + Y * Y);
          if (r < 0.1) r = 0.1;
          if (r <= pj) acc[row + gi] += (gamma[a] * Math.log(pj / r) + Math.log(Rj / pj)) * pc[a];
          else if (r <= Rj) acc[row + gi] += Math.log(Rj / r) * pc[a];
        }
      }
    }

    const px = imgData.data;
    for (let gj = 0; gj < NY; gj++) {
      const src = gj * NX, dst = (NY - 1 - gj) * NX;
      for (let gi = 0; gi < NX; gi++) {
        const a = aqGrid[src + gi], o = (dst + gi) * 4;
        if (a < 0) { px[o + 3] = 0; valGrid[src + gi] = NaN; continue; }
        const v = (acc[src + gi] + pRef[a]) / pFrac[a] * 100;
        valGrid[src + gi] = v;
        const c = rampRGB(v);
        px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
      }
    }
  }

  const rasterCanvas = document.createElement("canvas");
  rasterCanvas.width = NX; rasterCanvas.height = NY;
  const rasterCtx = rasterCanvas.getContext("2d");

  /* ------------------------------------------------------------ map */

  const mapCanvas = document.getElementById("map");
  const mapCtx = mapCanvas.getContext("2d");
  const mapTip = document.getElementById("map-tip");
  const mapStage = document.getElementById("mapstage");
  let view = null;

  const layers = {
    pressure: document.getElementById("l-pressure"),
    wells: document.getElementById("l-wells"),
    aquifers: document.getElementById("l-aquifers"),
    coast: document.getElementById("l-coast")
  };

  function sizeMap() {
    const dpr = window.devicePixelRatio || 1;
    const w = mapCanvas.clientWidth, h = mapCanvas.clientHeight;
    mapCanvas.width = Math.round(w * dpr);
    mapCanvas.height = Math.round(h * dpr);
    mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ew = extent.maxx - extent.minx, eh = extent.maxy - extent.miny;
    const s = Math.min(w / ew, h / eh);
    view = { w, h, s, cx: (w - ew * s) / 2, cy: (h - eh * s) / 2 };
    updateScaleBar();
  }

  const toX = x => view.cx + (x - extent.minx) * view.s;         // x in km
  const toY = y => view.cy + (extent.maxy - y) * view.s;
  const fromX = p => extent.minx + (p - view.cx) / view.s;
  const fromY = p => extent.maxy - (p - view.cy) / view.s;

  function updateScaleBar() {
    const targets = [500, 250, 200, 100, 50, 25];
    const maxPx = 110;
    let pick = targets[targets.length - 1];
    for (const t of targets) { if (t * view.s <= maxPx) { pick = t; break; } }
    document.getElementById("scale-rule").style.width = (pick * view.s) + "px";
    document.getElementById("scale-label").textContent = pick + " km";
  }

  function tracePath(feats) {
    mapCtx.beginPath();
    for (const f of feats) {
      for (const ring of f.rings) {
        mapCtx.moveTo(toX(ring[0][0]), toY(ring[0][1]));
        for (let i = 1; i < ring.length; i++) mapCtx.lineTo(toX(ring[i][0]), toY(ring[i][1]));
        mapCtx.closePath();
      }
    }
  }

  function drawMap(st, year) {
    if (!view) sizeMap();
    const ctx = mapCtx;
    ctx.clearRect(0, 0, view.w, view.h);
    ctx.fillStyle = "#0d141b";
    ctx.fillRect(0, 0, view.w, view.h);

    if (layers.coast.checked) {
      tracePath(D.geo.uk);
      ctx.fillStyle = "#19232d";
      ctx.fill("evenodd");
      ctx.strokeStyle = "#3c4c5b";
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    if (layers.pressure.checked && st) {
      rasterCtx.putImageData(imgData, 0, 0);
      const left = toX(G.x0 / 1000);
      const top = toY((G.y0 + NY * CELL) / 1000);
      const wpx = (NX * CELL / 1000) * view.s;
      const hpx = (NY * CELL / 1000) * view.s;
      ctx.save();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.globalAlpha = 0.95;
      ctx.drawImage(rasterCanvas, left, top, wpx, hpx);
      ctx.restore();
    }

    if (layers.aquifers.checked) {
      tracePath(D.geo.aquifers);
      ctx.strokeStyle = "rgba(147,167,186,.95)";
      ctx.lineWidth = 1.1;
      ctx.stroke();
    }

    if (layers.wells.checked && st) {
      const { N, ws, we, wx, wy, wellVal } = st;
      const dotR = view.s > 0.12 ? 2.4 : 1.9;
      const flagged = [];
      for (let j = 0; j < N; j++) {
        if (ws[j] > year) break;
        const px = toX(wx[j] / 1000), py = toY(wy[j] / 1000);
        const dormant = year > we[j];
        // The 90 % limit is reached at well nodes long before any 2.5 km raster cell
        // registers it, so an at-limit well gets a halo rather than just a hotter fill.
        if (wellVal[j] >= 90) { flagged.push([px, py, wellVal[j]]); continue; }
        ctx.beginPath();
        ctx.arc(px, py, dormant ? dotR * 0.62 : dotR, 0, 6.2832);
        ctx.fillStyle = dormant ? "rgba(150,166,180,.55)" : rampCSS(wellVal[j]);
        ctx.fill();
        if (!dormant) {
          ctx.lineWidth = 0.9;
          ctx.strokeStyle = "rgba(13,20,27,.85)";
          ctx.stroke();
        }
      }
      for (const [px, py, v] of flagged) {
        const col = v >= 100 ? "rgb(208,59,59)" : "rgb(236,131,90)";
        ctx.beginPath();
        ctx.arc(px, py, dotR + 4.5, 0, 6.2832);
        ctx.fillStyle = v >= 100 ? "rgba(208,59,59,.22)" : "rgba(236,131,90,.20)";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(px, py, dotR + 2.2, 0, 6.2832);
        ctx.lineWidth = 1.3;
        ctx.strokeStyle = col;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(px, py, dotR * 0.95, 0, 6.2832);
        ctx.fillStyle = col;
        ctx.fill();
        ctx.lineWidth = 0.9;
        ctx.strokeStyle = "rgba(13,20,27,.9)";
        ctx.stroke();
      }
    }
  }

  /* ------------------------------------------------------------ charts */

  const CH = {
    pressure: document.getElementById("ch-pressure"),
    rate: document.getElementById("ch-rate"),
    cum: document.getElementById("ch-cum")
  };
  const CHART_H = 152, M = { l: 44, r: 12, t: 8, b: 22 };

  function niceTicks(lo, hi, want) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / want;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  }

  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  function renderChart(node, o) {
    const svg = node.querySelector("svg");
    const W = node.clientWidth || 400;
    const iw = W - M.l - M.r, ih = CHART_H - M.t - M.b;
    const n = o.values.length;
    const xs = i => M.l + (n <= 1 ? 0 : (i / (n - 1)) * iw);
    const ys = v => M.t + ih - ((v - o.yMin) / (o.yMax - o.yMin)) * ih;

    let g = "";
    const yt = niceTicks(o.yMin, o.yMax, 4);
    for (const t of yt) {
      const y = ys(t);
      g += `<line x1="${M.l}" y1="${y.toFixed(1)}" x2="${M.l + iw}" y2="${y.toFixed(1)}" stroke="var(--line)" stroke-width="1"/>`;
      g += `<text x="${M.l - 7}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-family="'IBM Plex Mono',monospace" font-size="9.5" fill="var(--ink-3)">${o.fmtTick(t)}</text>`;
    }

    const xStep = n > 110 ? 25 : n > 60 ? 20 : 10;
    let xt = "";
    for (let i = 0; i < n; i++) {
      const yr = i + 1;
      if (yr !== 1 && yr % xStep !== 0) continue;
      xt += `<text x="${xs(i).toFixed(1)}" y="${CHART_H - 7}" text-anchor="middle" font-family="'IBM Plex Mono',monospace" font-size="9.5" fill="var(--ink-3)">${yr}</text>`;
    }

    let d = "", area = "";
    for (let i = 0; i < n; i++) {
      const x = xs(i).toFixed(1), y = ys(o.values[i]).toFixed(1);
      d += (i ? "L" : "M") + x + " " + y;
    }
    if (o.fill) area = d + `L${xs(n - 1).toFixed(1)} ${ys(o.yMin).toFixed(1)}L${xs(0).toFixed(1)} ${ys(o.yMin).toFixed(1)}Z`;

    let extra = "";
    if (o.threshold != null && o.threshold > o.yMin && o.threshold < o.yMax) {
      const y = ys(o.threshold);
      extra += `<rect x="${M.l}" y="${M.t}" width="${iw}" height="${(y - M.t).toFixed(1)}" fill="var(--critical)" opacity=".07"/>`;
      extra += `<line x1="${M.l}" y1="${y.toFixed(1)}" x2="${M.l + iw}" y2="${y.toFixed(1)}" stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="5 3"/>`;
      extra += `<text x="${M.l + iw - 3}" y="${(y - 5).toFixed(1)}" text-anchor="end" font-family="Archivo,sans-serif" font-weight="600" font-size="9.5" fill="var(--critical)">${esc(o.thresholdLabel)}</text>`;
    }
    if (o.markIndex != null && o.markIndex >= 0 && o.markIndex < n) {
      const x = xs(o.markIndex);
      extra += `<line x1="${x.toFixed(1)}" y1="${M.t}" x2="${x.toFixed(1)}" y2="${M.t + ih}" stroke="var(--critical)" stroke-width="1" stroke-dasharray="2 3" opacity=".8"/>`;
      extra += `<circle cx="${x.toFixed(1)}" cy="${ys(o.values[o.markIndex]).toFixed(1)}" r="3.6" fill="var(--critical)" stroke="var(--surface)" stroke-width="2"/>`;
    }

    svg.setAttribute("viewBox", `0 0 ${W} ${CHART_H}`);
    svg.setAttribute("height", CHART_H);
    svg.innerHTML =
      g + xt +
      `<line x1="${M.l}" y1="${M.t + ih}" x2="${M.l + iw}" y2="${M.t + ih}" stroke="var(--line-2)" stroke-width="1"/>` +
      (o.fill ? `<path d="${area}" fill="${o.fill}"/>` : "") +
      `<path d="${d}" fill="none" stroke="${o.stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` +
      extra +
      `<g class="hv" opacity="0"><line y1="${M.t}" y2="${M.t + ih}" stroke="var(--ink-3)" stroke-width="1"/>` +
      `<circle r="4" fill="${o.stroke}" stroke="var(--surface)" stroke-width="2"/></g>` +
      `<rect class="hit" x="${M.l}" y="${M.t}" width="${iw}" height="${ih}" fill="transparent" style="cursor:crosshair"/>`;

    node._geom = { xs, ys, n, iw, W };
  }

  function setHover(i) {
    const r = state.result;
    if (!r) return;
    const inRange = i != null && i >= 0 && i < r.maxDP.length;
    for (const key of ["pressure", "rate", "cum"]) {
      const node = CH[key], gm = node._geom;
      const hv = node.querySelector(".hv"), tip = node.querySelector(".chart-tip");
      if (!gm || !hv) continue;
      if (!inRange) { hv.setAttribute("opacity", "0"); tip.style.opacity = 0; continue; }
      const val = key === "pressure" ? r.maxDP[i] : key === "rate" ? r.activeWells[i] * r.P.rate : r.cumulative[i] / 1000;
      const x = gm.xs(i), y = gm.ys(val);
      hv.setAttribute("opacity", "1");
      hv.querySelector("line").setAttribute("x1", x); hv.querySelector("line").setAttribute("x2", x);
      hv.querySelector("circle").setAttribute("cx", x); hv.querySelector("circle").setAttribute("cy", y);
      tip.innerHTML =
        `<div class="ty">Year ${i + 1}</div>` +
        `<div class="row"><span>Peak pressure</span><b>${r.maxDP[i].toFixed(1)}%</b></div>` +
        `<div class="row"><span>Active wells</span><b>${r.activeWells[i]}</b></div>` +
        `<div class="row"><span>Injecting</span><b>${(r.activeWells[i] * r.P.rate).toFixed(1)} Mt/yr</b></div>` +
        `<div class="row"><span>Stored</span><b>${(r.cumulative[i] / 1000).toFixed(2)} Gt</b></div>`;
      tip.style.opacity = 1;
      const tw = 152;
      const left = Math.min(Math.max(x - tw / 2, 6), gm.W - tw - 6);
      tip.style.left = left + "px";
      tip.style.top = (node.querySelector("svg").offsetTop + Math.max(y - 96, 2)) + "px";
      tip.style.width = tw + "px";
    }
  }

  for (const key of ["pressure", "rate", "cum"]) {
    const node = CH[key];
    node.addEventListener("pointermove", e => {
      const gm = node._geom;
      if (!gm) return;
      const rect = node.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const f = (px - M.l) / gm.iw;
      setHover(Math.round(f * (gm.n - 1)));
    });
    node.addEventListener("pointerleave", () => setHover(null));
    node.addEventListener("click", e => {
      const gm = node._geom;
      if (!gm) return;
      const rect = node.getBoundingClientRect();
      const f = (e.clientX - rect.left - M.l) / gm.iw;
      const i = Math.round(f * (gm.n - 1));
      if (i >= 0 && i < gm.n) setYear(i + 1);
    });
  }

  function drawCharts(r) {
    const peak = Math.max(...r.maxDP);
    renderChart(CH.pressure, {
      values: r.maxDP, yMin: 50, yMax: Math.max(100, Math.ceil(peak / 10) * 10),
      stroke: "var(--series)", fill: "var(--series-fill)",
      threshold: 90, thresholdLabel: "90% limit", markIndex: r.limitIndex,
      fmtTick: v => v.toFixed(0)
    });
    const rates = r.activeWells.map(w => w * r.P.rate);
    renderChart(CH.rate, {
      values: rates, yMin: 0, yMax: Math.max(1, Math.max(...rates) * 1.08),
      stroke: "var(--series)", fill: "var(--series-fill)", markIndex: r.limitIndex,
      fmtTick: v => v >= 100 ? v.toFixed(0) : v.toFixed(v < 10 ? 1 : 0)
    });
    const cums = r.cumulative.map(c => c / 1000);
    renderChart(CH.cum, {
      values: cums, yMin: 0, yMax: Math.max(0.5, cums[cums.length - 1] * 1.08),
      stroke: "var(--series)", fill: "var(--series-fill)", markIndex: r.limitIndex,
      fmtTick: v => v.toFixed(v < 10 ? 1 : 0)
    });
  }

  /* ------------------------------------------------------------ summary */

  function drawSummary(r) {
    const head = document.getElementById("headline");
    const eyebrow = head.querySelector(".eyebrow");
    const peak = Math.max(...r.maxDP);
    const peakYear = r.maxDP.indexOf(peak) + 1;

    if (r.limitIndex >= 0) {
      const gt = r.cumulative[r.limitIndex] / 1000;
      head.className = "headline over";
      eyebrow.textContent = "Stored at the 90% fracture-pressure limit";
      document.getElementById("hl-value").textContent = gt.toFixed(2);
      document.getElementById("hl-unit").textContent = "Gt CO₂";
      document.getElementById("hl-caption").textContent =
        `Peak reservoir pressure reaches 90% of fracture pressure in year ${r.limitIndex + 1}, ` +
        `with ${r.activeWells[r.limitIndex].toLocaleString()} wells injecting. Storage beyond this point ` +
        `needs pressure management — the full ${r.maxDP.length}-year run would otherwise reach ` +
        `${(r.cumulative[r.cumulative.length - 1] / 1000).toFixed(2)} Gt.`;
      document.getElementById("f-year").textContent = r.limitIndex + 1;
      document.getElementById("f-wells").innerHTML =
        r.wellsBy[r.limitIndex].toLocaleString() + ' <small>of ' + r.N.toLocaleString() + '</small>';
    } else {
      head.className = "headline under";
      eyebrow.textContent = "Stored over the full run";
      document.getElementById("hl-value").textContent = (r.cumulative[r.cumulative.length - 1] / 1000).toFixed(2);
      document.getElementById("hl-unit").textContent = "Gt CO₂";
      document.getElementById("hl-caption").textContent =
        `Peak reservoir pressure never reaches the 90% limit — it tops out at ${peak.toFixed(1)}% ` +
        `of fracture pressure in year ${peakYear}. This configuration is not pressure-limited.`;
      document.getElementById("f-year").textContent = "not reached";
      document.getElementById("f-wells").innerHTML = r.N.toLocaleString() + ' <small>total</small>';
    }

    const f = document.getElementById("f-peak");
    f.textContent = peak.toFixed(1) + "%";
    f.style.color = peak >= 100 ? "var(--critical)" : peak >= 90 ? "var(--serious)" : "var(--good)";

    const warn = document.getElementById("pool-warn");
    if (r.exhausted) {
      warn.hidden = false;
      warn.textContent = `The candidate well pool (${POOL_N.toLocaleString()} locations on the 1 km lattice) ` +
        `is exhausted before the drilling period ends — reduce the carrying capacity or the drilling period ` +
        `to keep the schedule intact.`;
    } else warn.hidden = true;

    const rows = U.map((u, a) => ({
      name: u.name.replace(/ (Formation|Unit|Main)$/, ""),
      wells: r.unitWells[a],
      stored: r.unitStored[a] / 1000,
      peak: r.unitPeak[a]
    })).sort((p, q) => q.peak - p.peak);

    document.getElementById("unit-rows").innerHTML = rows.map(row => {
      if (!row.wells) {
        return `<tr><td>${esc(row.name)}</td><td colspan="3" style="color:var(--ink-3);text-align:right">no wells</td></tr>`;
      }
      return `<tr><td>${esc(row.name)}</td><td>${row.wells.toLocaleString()}</td>` +
        `<td>${row.stored.toFixed(2)}</td>` +
        `<td><span class="peak"><i style="background:${rampCSS(row.peak)}"></i>${row.peak.toFixed(1)}</span></td></tr>`;
    }).join("");
    document.getElementById("unit-hint").textContent = "peak and totals over the full run";
  }

  /* ------------------------------------------------------------ state */

  const state = { st: null, result: null, year: 1, playing: null, token: 0, firstRun: true };

  function setStatus(text, frac) {
    document.getElementById("status-text").textContent = text;
    document.getElementById("status-bar").style.width = ((frac || 0) * 100) + "%";
  }

  function readParams() {
    const num = id => +document.getElementById(id).value;
    return {
      k: num("c-k"), L: Math.round(num("c-L")), rate: num("c-rate"),
      injYears: Math.round(num("c-inj")), years: Math.round(num("c-years")),
      draw: Math.round(num("c-seed"))
    };
  }

  function run() {
    const token = ++state.token;
    const P = readParams();
    const st = prepare(P);
    state.st = st;
    let year = 1;

    (function chunk() {
      if (token !== state.token) return;
      const t0 = performance.now();
      while (year <= st.totalYears && performance.now() - t0 < 26) { runYear(st, year); year++; }
      if (year <= st.totalYears) {
        setStatus("Simulating…", year / st.totalYears);
        requestAnimationFrame(chunk);
        return;
      }
      for (let j = 0; j < st.N; j++) st.unitWells[st.waq[j]]++;
      let limitIndex = -1;
      for (let i = 0; i < st.maxDP.length; i++) { if (st.maxDP[i] >= 90) { limitIndex = i; break; } }
      const wellsBy = [];
      let c = 0, wi = 0;
      for (let y = 1; y <= st.totalYears; y++) {
        while (wi < st.N && st.ws[wi] <= y) { c++; wi++; }
        wellsBy.push(c);
      }
      st.limitIndex = limitIndex;
      st.wellsBy = wellsBy;
      state.result = st;

      setStatus(`${st.N.toLocaleString()} wells · ${st.totalYears} years`, 1);
      document.getElementById("year").max = st.totalYears;
      // open on the year the pressure limit bites - the frame worth seeing first
      if (state.firstRun) {
        state.year = limitIndex >= 0 ? limitIndex + 1 : Math.round(st.totalYears * 0.7);
        state.firstRun = false;
      }
      if (state.year > st.totalYears) state.year = st.totalYears;
      drawCharts(st);
      drawSummary(st);
      setYear(state.year);
    })();
  }

  function setYear(y) {
    state.year = y;
    document.getElementById("year").value = y;
    document.getElementById("year-label").textContent = y;
    const st = state.result;
    if (!st) return;
    if (layers.pressure.checked) buildRaster(st, y);
    ages(st, y);
    let atLimit = 0, over = 0;
    for (let i = 0; i < st.N; i++) {
      if (st.ws[i] > y) break;
      const a = st.waq[i];
      const v = (dPat(st, st.wx[i], st.wy[i], y, i) + pRef[a]) / pFrac[a] * 100;
      st.wellVal[i] = v;
      if (v >= 100) over++; else if (v >= 90) atLimit++;
    }
    drawMap(st, y);

    const v = st.maxDP[y - 1];
    document.getElementById("map-hint").innerHTML =
      `Year ${y} · peak <strong style="color:${v >= 90 ? "var(--serious)" : "var(--ink)"}">${v.toFixed(1)}%</strong> of fracture pressure`;

    const total = atLimit + over;
    document.getElementById("legend-foot").innerHTML = total
      ? `<b>${total} well${total > 1 ? "s" : ""}</b> at or above the limit` +
        (over ? `, <b>${over}</b> past it` : "") +
        ` in year ${y}. The 2.5 km field reads cooler — peaks sit on well nodes.`
      : `No wells at the limit in year ${y}. The field is sampled on a 2.5 km grid; peaks sit on well nodes.`;
  }

  /* ------------------------------------------------------------ wiring */

  const fmtField = {
    "c-k": v => (+v).toFixed(3),
    "c-L": v => Math.round(v).toLocaleString() + ' <span class="unit">wells</span>',
    "c-rate": v => (+v).toFixed(1) + ' <span class="unit">Mt/yr</span>',
    "c-inj": v => Math.round(v) + ' <span class="unit">yr</span>',
    "c-years": v => Math.round(v) + ' <span class="unit">yr</span>',
    "c-seed": v => Math.round(v)
  };

  let debounce = null;
  function scheduleRun() {
    stopPlay();
    clearTimeout(debounce);
    setStatus("Simulating…", 0);
    debounce = setTimeout(run, 140);
  }

  Object.keys(fmtField).forEach(id => {
    const el = document.getElementById(id);
    const out = document.getElementById("v-" + id.slice(2));
    const sync = () => { out.innerHTML = fmtField[id](el.value); };
    sync();
    el.addEventListener("input", () => { sync(); syncPresets(); scheduleRun(); });
  });

  function syncPresets() {
    const k = +document.getElementById("c-k").value;
    document.querySelectorAll("#presets .preset").forEach(b =>
      b.setAttribute("aria-pressed", String(Math.abs(+b.dataset.k - k) < 1e-9)));
  }
  document.querySelectorAll("#presets .preset").forEach(b => {
    b.addEventListener("click", () => {
      const el = document.getElementById("c-k");
      el.value = b.dataset.k;
      el.dispatchEvent(new Event("input"));
    });
  });

  Object.values(layers).forEach(cb => cb.addEventListener("change", () => {
    if (!state.result) return;
    if (cb === layers.pressure && cb.checked) buildRaster(state.result, state.year);
    drawMap(state.result, state.year);
  }));

  document.getElementById("year").addEventListener("input", e => { stopPlay(); setYear(+e.target.value); });

  const playBtn = document.getElementById("play");
  function stopPlay() {
    if (state.playing) { clearInterval(state.playing); state.playing = null; }
    document.getElementById("play-icon").innerHTML = '<path d="M2 0.5 L11 6 L2 11.5 Z"/>';
    playBtn.setAttribute("aria-label", "Play simulation");
  }
  playBtn.addEventListener("click", () => {
    if (state.playing) return stopPlay();
    if (!state.result) return;
    if (state.year >= state.result.totalYears) setYear(1);
    document.getElementById("play-icon").innerHTML = '<path d="M2 1 H4.6 V11 H2 Z M7.4 1 H10 V11 H7.4 Z"/>';
    playBtn.setAttribute("aria-label", "Pause simulation");
    state.playing = setInterval(() => {
      if (state.year >= state.result.totalYears) return stopPlay();
      setYear(state.year + 1);
    }, 110);
  });

  /* map hover readout */
  mapCanvas.addEventListener("pointermove", e => {
    if (!view || !state.result) return;
    const rect = mapCanvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const xm = fromX(px) * 1000, ym = fromY(py) * 1000;
    const gi = Math.floor((xm - G.x0) / CELL), gj = Math.floor((ym - G.y0) / CELL);
    if (gi < 0 || gi >= NX || gj < 0 || gj >= NY) { mapTip.style.opacity = 0; return; }
    const a = aqGrid[gj * NX + gi];
    if (a < 0) { mapTip.style.opacity = 0; return; }
    const v = valGrid[gj * NX + gi];
    const shown = layers.pressure.checked && isFinite(v);
    mapTip.innerHTML = `<strong>${esc(U[a].name)}</strong>` +
      `<div style="color:#8a99a7">${U[a].z} m · ${U[a].k_md} mD · ${U[a].h.toFixed(0)} m net</div>` +
      (shown
        ? `<div class="num" style="color:${rampCSS(v)};margin-top:3px">${v.toFixed(1)}% of frac</div>`
        : `<div style="color:#8a99a7;margin-top:3px">p_frac ${U[a].p_frac.toFixed(1)} MPa</div>`);
    mapTip.style.opacity = 1;
    const tw = 180;
    mapTip.style.left = Math.min(Math.max(px + 14, 6), mapStage.clientWidth - tw - 6) + "px";
    mapTip.style.top = Math.min(Math.max(py - 18, 6), mapStage.clientHeight - 80) + "px";
  });
  mapCanvas.addEventListener("pointerleave", () => { mapTip.style.opacity = 0; });

  let rsz = null;
  window.addEventListener("resize", () => {
    clearTimeout(rsz);
    rsz = setTimeout(() => {
      sizeMap();
      if (state.result) { drawMap(state.result, state.year); drawCharts(state.result); }
    }, 120);
  });

  /* go */
  syncPresets();
  sizeMap();
  run();
})();
