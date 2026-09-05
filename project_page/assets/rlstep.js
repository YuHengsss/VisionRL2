/* Live training-step illustration for Vision-RL².
 * Data: window.RL_CASES (built by tools/build_cases.py from the export dumps).
 * Every number shown is a value produced by the released trainer's building
 * blocks on the sample (frozen Qwen3.5-4B reader, SD-RPN initialization).
 */
(function () {
  'use strict';
  var app = document.getElementById('rlstep-app');
  if (!app) return;
  var CASES = window.RL_CASES || [];
  if (!CASES.length) { app.innerHTML = '<div class="rlstep-loading">No cases packaged yet.</div>'; return; }

  var REGION_COLORS = ['#2563eb', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#ec4899'];
  var SUPP_COLOR = '#0ea5e9', PROBE_COLOR = '#94a3b8', RL_COLOR = '#22c55e';
  var state = { ci: 0, step: 0, hl: -1, playing: false, timer: null, subTimer: null, img: null };

  function L(en, zh) { return (document.body.getAttribute('data-lang') === 'zh') ? zh : en; }
  function f2(x) { return (x === null || x === undefined) ? '–' : (Math.round(x * 100) / 100).toFixed(2); }
  function f3(x) { return (x === null || x === undefined) ? '–' : (Math.round(x * 1000) / 1000).toFixed(3); }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function cleanQ(q) { return String(q).replace(/\n?Answer the question using a single word or phrase\.?/g, '').trim(); }

  var STEPS = [
    { en: 'Input', zh: '输入' },
    { en: 'RoI map & regions', zh: 'RoI 图与区域' },
    { en: 'Reader scores', zh: '阅读器评分' },
    { en: 'Noise margin', zh: '噪声阈值' },
    { en: 'Advantages', zh: '优势与更新' },
    { en: 'Additive group', zh: '加法组' },
    { en: 'After RL', zh: 'RL 之后' }
  ];

  // ------------------------------------------------------------ skeleton
  app.innerHTML =
    '<div class="rlstep-cases" id="rls-cases"></div>' +
    '<div class="rlstep-steps" id="rls-steps"></div>' +
    '<div class="rlstep-body">' +
    '  <div><div class="rlstep-stage"><canvas id="rls-canvas"></canvas></div><div class="rlstep-legend" id="rls-legend"></div></div>' +
    '  <div class="rlstep-panel" id="rls-panel"></div>' +
    '</div>' +
    '<div class="rlstep-controls">' +
    '  <button id="rls-prev">◀</button><button id="rls-play" class="primary"></button><button id="rls-next">▶</button>' +
    '  <span class="spacer"></span><span class="hint" id="rls-hint"></span>' +
    '</div>';
  var canvas = document.getElementById('rls-canvas'), ctx = canvas.getContext('2d');

  function cur() { return CASES[state.ci]; }
  function D() { return cur().data; }

  // ------------------------------------------------------------ drawing
  function cellRect(r, c) {
    var d = D(), W = canvas.width, H = canvas.height, Hg = d.grid[0], Wg = d.grid[1];
    return [c * W / Wg, r * H / Hg, W / Wg, H / Hg];
  }
  function heatColor(t) { // 0..1 -> blue -> cyan -> yellow -> red
    var stops = [[37, 99, 235], [14, 165, 233], [250, 204, 21], [239, 68, 68]];
    var x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.floor(x), f = x - i;
    var a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
    return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
  }
  function drawHeat(P, pmax, alphaMax) {
    var d = D(), Hg = d.grid[0], Wg = d.grid[1];
    for (var r = 0; r < Hg; r++) for (var c = 0; c < Wg; c++) {
      var t = pmax > 0 ? P[r][c] / pmax : 0;
      if (t < 0.04) continue;
      var col = heatColor(t), rc = cellRect(r, c);
      ctx.fillStyle = 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + (alphaMax * t) + ')';
      ctx.fillRect(rc[0], rc[1], rc[2] + 0.5, rc[3] + 0.5);
    }
  }
  function drawMask(mask, fill, stroke, lineDash) {
    var d = D(), Hg = d.grid[0], Wg = d.grid[1];
    if (fill) { ctx.fillStyle = fill; for (var r = 0; r < Hg; r++) for (var c = 0; c < Wg; c++) if (mask[r][c]) { var rc = cellRect(r, c); ctx.fillRect(rc[0], rc[1], rc[2] + 0.5, rc[3] + 0.5); } }
    if (stroke) { // outline: draw edges of cells whose neighbour is outside
      ctx.strokeStyle = stroke; ctx.lineWidth = Math.max(2, canvas.width / 400); ctx.setLineDash(lineDash || []);
      ctx.beginPath();
      for (var r2 = 0; r2 < Hg; r2++) for (var c2 = 0; c2 < Wg; c2++) {
        if (!mask[r2][c2]) continue;
        var q = cellRect(r2, c2), x = q[0], y = q[1], w = q[2], h = q[3];
        if (r2 === 0 || !mask[r2 - 1][c2]) { ctx.moveTo(x, y); ctx.lineTo(x + w, y); }
        if (r2 === Hg - 1 || !mask[r2 + 1][c2]) { ctx.moveTo(x, y + h); ctx.lineTo(x + w, y + h); }
        if (c2 === 0 || !mask[r2][c2 - 1]) { ctx.moveTo(x, y); ctx.lineTo(x, y + h); }
        if (c2 === Wg - 1 || !mask[r2][c2 + 1]) { ctx.moveTo(x + w, y); ctx.lineTo(x + w, y + h); }
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
  }
  function maskBBox(mask) {
    var r0 = 1e9, c0 = 1e9, r1 = -1, c1 = -1;
    for (var r = 0; r < mask.length; r++) for (var c = 0; c < mask[r].length; c++) if (mask[r][c]) { r0 = Math.min(r0, r); c0 = Math.min(c0, c); r1 = Math.max(r1, r); c1 = Math.max(c1, c); }
    return [r0, c0, r1 + 1, c1 + 1];
  }
  function label(mask, text, color) {
    var bb = maskBBox(mask), rc = cellRect(bb[0], bb[1]);
    var fs = Math.max(12, canvas.width / 46);
    ctx.font = '600 ' + fs + 'px Space Grotesk, Inter, sans-serif';
    var w = ctx.measureText(text).width + fs * 0.8, h = fs * 1.5;
    var x = Math.min(rc[0], canvas.width - w - 2), y = Math.max(rc[1] - h - 2, 2);
    ctx.fillStyle = color; ctx.globalAlpha = 0.92; roundRect(x, y, w, h, 5); ctx.fill(); ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle'; ctx.fillText(text, x + fs * 0.4, y + h / 2);
  }
  function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
  function dimExcept(mask) { // darken everything outside mask
    var d = D(), Hg = d.grid[0], Wg = d.grid[1];
    ctx.fillStyle = 'rgba(15,23,42,0.72)';
    for (var r = 0; r < Hg; r++) for (var c = 0; c < Wg; c++) if (!mask[r][c]) { var rc = cellRect(r, c); ctx.fillRect(rc[0], rc[1], rc[2] + 0.5, rc[3] + 0.5); }
  }
  function unionMask(masks) { var d = D(), Hg = d.grid[0], Wg = d.grid[1], m = []; for (var r = 0; r < Hg; r++) { m.push([]); for (var c = 0; c < Wg; c++) { var v = 0; for (var i = 0; i < masks.length; i++) if (masks[i][r][c]) { v = 1; break; } m[r].push(v); } } return m; }

  function draw() {
    var d = D(); if (!state.img) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
    var regs = d.regions || [], sub = d.sub || {}, acts = sub.actions || [];
    var s = state.step;
    if (s === 1) {
      drawHeat(d.P, d.P_max, 0.8);
      regs.forEach(function (rg, i) { drawMask(rg.mask, null, REGION_COLORS[i % 6]); label(rg.mask, 'R' + (i + 1) + '  z=' + f2(rg.z), REGION_COLORS[i % 6]); });
    } else if (s === 2) {
      var a = acts[state.hl >= 0 ? state.hl : (sub.empty_idx || 0)];
      if (a) {
        dimExcept(a.keep_mask);
        regs.forEach(function (rg, i) { var removed = a.discard.indexOf(i) >= 0; drawMask(rg.mask, removed ? 'rgba(239,68,68,0.35)' : null, removed ? '#ef4444' : REGION_COLORS[i % 6], removed ? [6, 4] : []); label(rg.mask, (removed ? '−' : '') + 'R' + (i + 1), removed ? '#ef4444' : REGION_COLORS[i % 6]); });
      }
    } else if (s === 3) {
      var e = acts[sub.empty_idx || 0]; if (e) dimExcept(e.keep_mask);
      regs.forEach(function (rg, i) { drawMask(rg.mask, null, REGION_COLORS[i % 6]); });
      (sub.probes || []).forEach(function (p, j) { drawMask(p.mask, 'rgba(148,163,184,0.55)', '#e2e8f0', [4, 3]); label(p.mask, 'control ' + (j + 1), '#64748b'); });
    } else if (s === 4) {
      drawHeat(d.P, d.P_max, 0.35);
      regs.forEach(function (rg, i) {
        var act = acts.filter(function (x) { return x.discard.length === 1 && x.discard[0] === i; })[0];
        var keep = act && act.decision === 'keep';
        drawMask(rg.mask, keep ? 'rgba(34,197,94,0.30)' : 'rgba(239,68,68,0.30)', keep ? '#16a34a' : '#ef4444');
        label(rg.mask, 'R' + (i + 1) + (keep ? '  ' + L('keep', '保留') : '  ' + L('prune', '剪除')), keep ? '#16a34a' : '#ef4444');
      });
    } else if (s === 5) {
      if (d.ev_union) drawMask(d.ev_union, 'rgba(14,165,233,0.30)', 'rgba(14,165,233,0.7)', [3, 3]);
      regs.forEach(function (rg, i) { drawMask(rg.mask, null, REGION_COLORS[i % 6]); });
      ((d.add || {}).supp || []).forEach(function (sp, j) { var up = sp.decision === 'raise'; drawMask(sp.mask, up ? 'rgba(14,165,233,0.45)' : 'rgba(148,163,184,0.35)', up ? SUPP_COLOR : '#94a3b8', [5, 3]); label(sp.mask, 'S' + (j + 1) + '  Δ=' + f2(sp.delta), up ? SUPP_COLOR : '#64748b'); });
    } else if (s === 6 && d.rl) {
      drawHeat(d.rl.P, d.rl.P_max, 0.8);
      (d.rl.regions || []).forEach(function (rg, i) { drawMask(rg.mask, null, RL_COLOR); label(rg.mask, 'R\'' + (i + 1), '#16a34a'); });
    }
  }

  // ------------------------------------------------------------ panel
  function qa() {
    var d = D();
    return '<div class="rlstep-qa"><div class="k">' + L('Question', '问题') + '</div><div class="v">' + esc(cleanQ(d.question)) + '</div>' +
      '<div class="k" style="margin-top:0.3rem">' + L('Gold answer', '标准答案') + '</div><div class="v">' + esc(d.gold) + '</div></div>';
  }
  function thumbs(list) {
    return '<div class="rlstep-thumbs">' + list.map(function (t) {
      return '<figure' + (t.hl ? ' class="hl"' : '') + '><img src="' + t.src + '" alt="' + esc(t.cap) + '" loading="lazy"><figcaption>' + t.cap + '</figcaption></figure>';
    }).join('') + '</div>';
  }
  function bar(name, v, vmax, color) {
    var pct = Math.min(1, Math.abs(v) / vmax) * 50;
    var left = v >= 0 ? 50 : 50 - pct;
    return '<div class="rlstep-bar"><span>' + name + '</span><div class="track"><span class="zero"></span><span class="fill" style="left:' + left + '%;width:' + pct + '%;background:' + color + '"></span></div><span class="val">' + f2(v) + '</span></div>';
  }
  function panel() {
    var d = D(), s = state.step, sub = d.sub || {}, acts = sub.actions || [], regs = d.regions || [], dir = cur().dir, h = '';
    var kfrac = function (m) { var n = 0, t = 0; m.forEach(function (row) { row.forEach(function (v) { t++; if (v) n++; }); }); return n / t; };
    if (s === 0) {
      h += '<h4>' + L('A training sample', '一个训练样本') + '</h4>' + qa() +
        '<p>' + L('The pool item is an image, a question, and a gold answer, nothing else: no box, no region label. The predictor must decide where the answer lives from the ' + d.grid[0] + '×' + d.grid[1] + ' visual-token grid (' + (d.grid[0] * d.grid[1]) + ' tokens at the 576-token training limit).',
          '训练池里的条目只有图像、问题和标准答案，没有框，也没有区域标注。预测器必须在 ' + d.grid[0] + '×' + d.grid[1] + ' 的视觉 token 网格上（576 token 训练上限下共 ' + (d.grid[0] * d.grid[1]) + ' 个 token）判断答案在哪里。') + '</p>' +
        '<p>' + L('Source: ' + esc(d.dataset) + ', sample ' + d.sample_id + '.', '来源：' + esc(d.dataset) + '，样本 ' + d.sample_id + '。') + '</p>';
    } else if (s === 1) {
      h += '<h4>' + L('SD-RPN map → regions', 'SD-RPN 图 → 区域') + '</h4>' +
        '<p>' + L('The initial predictor emits a dense map P<sub>θ</sub> (max ' + f3(d.P_max) + '). It is smoothed, thresholded at 0.3 × peak, and split into connected components; the top ' + d.K + ' become the regions. Each region\'s confidence z is the mean logit inside it.',
          '初始预测器输出稠密图 P<sub>θ</sub>（最大值 ' + f3(d.P_max) + '）。经平滑、按 0.3 × 峰值阈值化并拆分为连通分量后，前 ' + d.K + ' 个成为区域。每个区域的置信度 z 是其内部的平均 logit。') + '</p>' +
        '<table class="rlstep-table"><tr><th>' + L('Region', '区域') + '</th><th>' + L('cells', '格数') + '</th><th>z<sub>θ</sub></th></tr>' +
        regs.map(function (r, i) { return '<tr><td><span class="rlstep-dot" style="background:' + REGION_COLORS[i % 6] + '"></span>R' + (i + 1) + '</td><td>' + r.area + '</td><td>' + f2(r.z) + '</td></tr>'; }).join('') + '</table>';
    } else if (s === 2) {
      var e = acts[sub.empty_idx || 0];
      h += '<h4>' + L('Score every leave-one-out mask', '为每个留一掩码打分') + '</h4>' +
        '<div class="rlstep-formula">Δ<sub>k</sub> = h<sub>φ</sub>(M) − h<sub>φ</sub>(M ⊖ R<sub>k</sub>)</div>' +
        '<p>' + L('The frozen reader sees the masked image only (background replaced by the image mean) and is teacher-forced on the gold answer. P<sub>φ</sub> is the geometric-mean answer probability, h<sub>φ</sub> its log-odds. Intact mask: P<sub>φ</sub> = ' + f3(e ? e.p : null) + '. Removing a region that carries evidence makes P<sub>φ</sub> collapse; removing a spurious one barely moves it.',
          '冻结阅读器只看掩码图像（背景替换为图像均值），并对标准答案做 teacher forcing。P<sub>φ</sub> 是答案的几何平均概率，h<sub>φ</sub> 是其对数几率。完整掩码：P<sub>φ</sub> = ' + f3(e ? e.p : null) + '。移除承载证据的区域会让 P<sub>φ</sub> 崩塌，移除无关区域几乎不变。') + '</p>' +
        '<table class="rlstep-table"><tr><th>' + L('Action', '动作') + '</th><th>P<sub>φ</sub></th><th>h<sub>φ</sub></th><th>Δ<sub>k</sub></th></tr>' +
        acts.map(function (a, i) { var nm = a.discard.length ? ('− R' + (a.discard[0] + 1)) : L('intact (∅)', '完整 (∅)'); return '<tr' + (i === state.hl ? ' class="hl"' : '') + '><td>' + nm + '</td><td>' + f3(a.p) + '</td><td>' + f2(a.h) + '</td><td>' + (a.discard.length ? f2(a.delta) : '–') + '</td></tr>'; }).join('') + '</table>' +
        thumbs(acts.map(function (a, i) { return { src: dir + '/act_' + i + '.jpg', cap: (a.discard.length ? '− R' + (a.discard[0] + 1) : '∅') + ' · P=' + f3(a.p), hl: i === state.hl }; }));
    } else if (s === 3) {
      var pr = sub.probes || [];
      h += '<h4>' + L('How big is reader noise on this sample?', '这个样本上的阅读器噪声有多大？') + '</h4>' +
        '<div class="rlstep-formula">b = min(κ · max<sub>c</sub> |h<sub>φ</sub>(M ∪ R<sup>c</sup>) − h<sub>φ</sub>(M)|, 1),  κ = ' + f2(sub.kappa) + '</div>' +
        '<p>' + L('Two control regions are grown in low-evidence areas (P<sub>θ</sub> &lt; 0.02), matched in size to the median predicted region, and added to the intact mask. They carry no evidence, so the change they cause is pure reader noise. The margin b is what a region must beat to count as necessary.',
          '在低证据区（P<sub>θ</sub> &lt; 0.02）生长两个对照区域，面积与预测区域的中位数相当，并加到完整掩码上。它们不含证据，因此引起的变化就是纯粹的阅读器噪声。阈值 b 就是一个区域被认定为“必要”所需超过的量。') + '</p>' +
        '<table class="rlstep-table"><tr><th>' + L('Probe', '对照') + '</th><th>' + L('cells', '格数') + '</th><th>P<sub>φ</sub></th><th>|Δh|</th></tr>' +
        pr.map(function (p, j) { return '<tr><td>control ' + (j + 1) + '</td><td>' + p.area + '</td><td>' + f3(p.p) + '</td><td>' + f2(p.abs_dh) + '</td></tr>'; }).join('') +
        '<tr class="hl"><td colspan="3"><strong>b</strong></td><td><strong>' + f2(sub.bar) + '</strong></td></tr></table>' +
        thumbs(pr.map(function (p, j) { return { src: dir + '/probe_' + j + '.jpg', cap: 'control ' + (j + 1) + ' · P=' + f3(p.p) }; }));
    } else if (s === 4) {
      var drops = acts.filter(function (a) { return a.discard.length === 1; });
      var vmax = Math.max.apply(null, drops.map(function (a) { return Math.abs(a.adv || 0); }).concat([0.5]));
      h += '<h4>' + L('Advantages and the removal policy', '优势与移除策略') + '</h4>' +
        '<div class="rlstep-formula">A<sub>k</sub> = (b − Δ<sub>k</sub>) / (s + 1)</div>' +
        '<p>' + L('A region whose contribution exceeds the margin gets a negative advantage for its removal, so the policy raises its confidence and keeps it. A region below the margin gets a positive advantage: its removal is encouraged and its confidence is lowered. The credit reaches the map through the softmax removal policy π<sub>sub</sub>, whose scores are the negated region confidences.',
          '贡献超过阈值的区域，其“移除”动作得到负优势，策略会抬高其置信度并保留它；低于阈值的区域得到正优势，移除被鼓励、置信度被压低。信用通过 softmax 移除策略 π<sub>sub</sub> 传回 RoI 图，其打分是区域置信度的相反数。') + '</p>' +
        '<div class="rlstep-bars">' + drops.map(function (a) { var k = a.discard[0]; return bar('R' + (k + 1), a.adv, vmax, a.decision === 'keep' ? '#16a34a' : '#ef4444'); }).join('') + '</div>' +
        '<table class="rlstep-table"><tr><th>' + L('Region', '区域') + '</th><th>Δ<sub>k</sub></th><th>b</th><th>A<sub>k</sub></th><th>π<sub>sub</sub>(−R<sub>k</sub>)</th><th></th></tr>' +
        drops.map(function (a) { var k = a.discard[0]; return '<tr><td><span class="rlstep-dot" style="background:' + REGION_COLORS[k % 6] + '"></span>R' + (k + 1) + '</td><td>' + f2(a.delta) + '</td><td>' + f2(sub.bar) + '</td><td>' + f2(a.adv) + '</td><td>' + f3(a.pi) + '</td><td><span class="chip ' + a.decision + '">' + (a.decision === 'keep' ? L('keep', '保留') : L('prune', '剪除')) + '</span></td></tr>'; }).join('') + '</table>';
    } else if (s === 5) {
      var add = d.add || {}, sp = add.supp || [];
      h += '<h4>' + L('Recover what the policy never proposed', '找回策略从未提出的证据') + '</h4>' +
        '<div class="rlstep-formula">Δ<sub>j</sub> = h<sub>φ</sub>(M<sub>aug</sub>) − h<sub>φ</sub>(M<sub>aug</sub> ⊖ S<sub>j</sub>)</div>' +
        '<p>' + L('Frozen response-to-image attention maps from ' + (d.ev_layers || 6) + ' MLLM layers (light blue) propose candidate regions outside the prediction. Each candidate is scored by removing it from the augmented mask (intact ∪ candidates, P<sub>φ</sub> = ' + f3(add.p_full) + '). A positive Δ<sub>j</sub> raises the candidate in the map through its mean inclusion log-likelihood ℓ<sup>+</sup>; a negative one suppresses it. No margin is needed: not including a region is already the zero-gradient default.',
          '来自 ' + (d.ev_layers || 6) + ' 个 MLLM 层的冻结“回答到图像”注意力图（淡蓝色）在预测之外提出候选区域。每个候选通过从增广掩码（完整 ∪ 候选，P<sub>φ</sub> = ' + f3(add.p_full) + '）中移除它来评分。Δ<sub>j</sub> 为正则通过其平均包含对数似然 ℓ<sup>+</sup> 在图中抬高该候选，为负则压制。这里不需要阈值：不包含某区域本来就是零梯度的默认状态。') + '</p>' +
        (sp.length ? '<table class="rlstep-table"><tr><th>' + L('Candidate', '候选') + '</th><th>' + L('cells', '格数') + '</th><th>' + L('layers', '层数') + '</th><th>P<sub>φ</sub>(−S<sub>j</sub>)</th><th>Δ<sub>j</sub></th><th>A<sub>j</sub></th><th></th></tr>' +
          sp.map(function (x, j) { return '<tr><td>S' + (j + 1) + '</td><td>' + x.area + '</td><td>' + x.votes + '</td><td>' + f3(x.p_drop) + '</td><td>' + f2(x.delta) + '</td><td>' + f2(x.adv) + '</td><td><span class="chip ' + x.decision + '">' + (x.decision === 'raise' ? L('raise', '抬高') : L('suppress', '压制')) + '</span></td></tr>'; }).join('') + '</table>' +
          thumbs([{ src: dir + '/supp_full.jpg', cap: 'M<sub>aug</sub> · P=' + f3(add.p_full) }].concat(sp.map(function (x, j) { return { src: dir + '/supp_drop_' + j + '.jpg', cap: '− S' + (j + 1) + ' · P=' + f3(x.p_drop) }; })))
          : '<p>' + L('No supplementary candidate on this sample: the evidence maps are already covered by the prediction.', '此样本没有补充候选：注意力图已被预测覆盖。') + '</p>');
    } else if (s === 6) {
      var rl = d.rl || {};
      h += '<h4>' + L('The trained predictor on the same sample', '训练后的预测器在同一样本上') + '</h4>' +
        '<p>' + L('After one epoch of region-level RL (only the three twig blocks are updated), the map keeps ' + (100 * (rl.keep_frac || 0)).toFixed(1) + '% of the cells instead of ' + (100 * (d.keep_frac_pa || 0)).toFixed(1) + '%. Fed the two foregrounds, the reader gives the gold answer P<sub>φ</sub> = ' + f3(rl.p_fg_pa) + ' (SD-RPN) versus ' + f3(rl.p_fg_rl) + ' (Vision-RL²).',
          '经过一个 epoch 的区域级 RL（只更新三个 twig 模块）后，RoI 图保留 ' + (100 * (rl.keep_frac || 0)).toFixed(1) + '% 的格子，而不是 ' + (100 * (d.keep_frac_pa || 0)).toFixed(1) + '%。把两种前景分别喂给阅读器，标准答案的 P<sub>φ</sub> 从 ' + f3(rl.p_fg_pa) + '（SD-RPN）变为 ' + f3(rl.p_fg_rl) + '（Vision-RL²）。') + '</p>' +
        '<div class="rlstep-compare"><figure><img src="' + dir + '/fg_pa.jpg" alt="SD-RPN foreground"><figcaption>SD-RPN · P<sub>φ</sub>=' + f3(rl.p_fg_pa) + '</figcaption></figure><figure><img src="' + dir + '/fg_rl.jpg" alt="Vision-RL2 foreground"><figcaption>Vision-RL² · P<sub>φ</sub>=' + f3(rl.p_fg_rl) + '</figcaption></figure></div>' +
        '<p>' + L('This is the training signal at work: no box was ever labeled; the reader\'s answer likelihood decided which regions to keep, prune, or add.', '这就是训练信号在起作用：从未标注任何框，是阅读器的答案似然决定了哪些区域该保留、剪除或补充。') + '</p>';
    }
    document.getElementById('rls-panel').innerHTML = h;
    document.getElementById('rls-legend').innerHTML = legend();
  }
  function legend() {
    var s = state.step, items = [];
    if (s === 1 || s === 4) items.push(['linear-gradient(90deg,#2563eb,#0ea5e9,#facc15,#ef4444)', L('P<sub>θ</sub> (relative to max)', 'P<sub>θ</sub>（相对最大值）')]);
    if (s >= 1 && s <= 5) items.push(['#2563eb', L('predicted regions R<sub>k</sub>', '预测区域 R<sub>k</sub>')]);
    if (s === 2) items.push(['#ef4444', L('removed region (this action)', '本动作移除的区域')]);
    if (s === 3) items.push(['#94a3b8', L('control region', '对照区域')]);
    if (s === 4) { items.push(['#16a34a', L('keep', '保留')]); items.push(['#ef4444', L('prune', '剪除')]); }
    if (s === 5) { items.push(['rgba(14,165,233,0.35)', L('attention evidence (6 layers)', '注意力证据（6 层）')]); items.push(['#0ea5e9', L('supplementary candidate S<sub>j</sub>', '补充候选 S<sub>j</sub>')]); }
    if (s === 6) items.push(['#22c55e', L('Vision-RL² regions', 'Vision-RL² 区域')]);
    return items.map(function (it) { return '<span><span class="sw" style="background:' + it[0] + '"></span>' + it[1] + '</span>'; }).join('');
  }

  // ------------------------------------------------------------ controls
  function renderTabs() {
    document.getElementById('rls-cases').innerHTML = CASES.map(function (c, i) { return '<button' + (i === state.ci ? ' class="active"' : '') + ' data-i="' + i + '">' + L(c.title_en, c.title_zh) + '</button>'; }).join('');
    document.getElementById('rls-steps').innerHTML = STEPS.map(function (st, i) { return '<button' + (i === state.step ? ' class="active"' : (i < state.step ? ' class="done"' : '')) + ' data-s="' + i + '"><span class="n">' + (i === 0 ? L('start', '开始') : 'step ' + i) + '</span>' + L(st.en, st.zh) + '</button>'; }).join('');
    document.getElementById('rls-play').textContent = state.playing ? L('⏸ Pause', '⏸ 暂停') : L('▶ Play', '▶ 播放');
    document.getElementById('rls-hint').textContent = L('Values from the released trainer on Qwen3.5-4B (SD-RPN init).', '数值来自发布代码在 Qwen3.5-4B（SD-RPN 初始化）上的计算。');
  }
  function setStep(s) {
    state.step = (s + STEPS.length) % STEPS.length;
    clearInterval(state.subTimer); state.subTimer = null; state.hl = -1;
    if (state.step === 2) { // cycle through the actions
      var acts = (D().sub || {}).actions || []; state.hl = 0;
      state.subTimer = setInterval(function () { state.hl = (state.hl + 1) % Math.max(1, acts.length); draw(); panel(); }, 1400);
    }
    renderTabs(); draw(); panel();
  }
  function loadCase(i) {
    state.ci = i; state.img = null; clearInterval(state.subTimer);
    var img = new Image();
    img.onload = function () {
      state.img = img; canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; setStep(state.step);
    };
    img.src = cur().dir + '/src.jpg';
    renderTabs(); panel();
  }
  function play(on) {
    state.playing = on; clearInterval(state.timer); state.timer = null;
    if (on) state.timer = setInterval(function () {
      if (state.step === STEPS.length - 1) { loadCase((state.ci + 1) % CASES.length); state.step = 0; }
      else setStep(state.step + 1);
    }, 6000);
    renderTabs();
  }
  document.getElementById('rls-cases').addEventListener('click', function (e) { var b = e.target.closest('button'); if (b) { state.step = 0; loadCase(+b.getAttribute('data-i')); } });
  document.getElementById('rls-steps').addEventListener('click', function (e) { var b = e.target.closest('button'); if (b) { play(false); setStep(+b.getAttribute('data-s')); } });
  document.getElementById('rls-prev').addEventListener('click', function () { play(false); setStep(state.step - 1); });
  document.getElementById('rls-next').addEventListener('click', function () { play(false); setStep(state.step + 1); });
  document.getElementById('rls-play').addEventListener('click', function () { play(!state.playing); });
  window.RLSTEP = { rerender: function () { renderTabs(); draw(); panel(); } };

  // precompute keep masks for actions (discard indices -> union of kept regions)
  CASES.forEach(function (c) {
    var d = c.data, regs = d.regions || [];
    ((d.sub || {}).actions || []).forEach(function (a) {
      var keep = regs.filter(function (r, i) { return a.discard.indexOf(i) < 0; }).map(function (r) { return r.mask; });
      a.keep_mask = keep.length ? (function (ms) { var Hg = d.grid[0], Wg = d.grid[1], m = []; for (var r = 0; r < Hg; r++) { m.push([]); for (var cc = 0; cc < Wg; cc++) { var v = 0; for (var k = 0; k < ms.length; k++) if (ms[k][r][cc]) { v = 1; break; } m[r].push(v); } } return m; })(keep) : d.regions[0].mask;
    });
  });
  // optional deep link: #live?case=1&step=4 (also used for visual QA screenshots)
  var deep = /live\?case=(\d+)&step=(\d+)/.exec(location.hash || '');
  if (deep) { state.ci = Math.min(CASES.length - 1, +deep[1]); state.step = Math.min(STEPS.length - 1, +deep[2]); }
  loadCase(state.ci);
  // start playing once the section scrolls into view
  if (!deep && 'IntersectionObserver' in window) {
    var started = false;
    new IntersectionObserver(function (es) { es.forEach(function (en) { if (en.isIntersecting && !started) { started = true; play(true); } }); }, { threshold: 0.35 }).observe(app);
  }
})();
