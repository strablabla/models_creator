// Training tab: start/stop a training, follow its curves, list models.
// 2D datasets train a U-Net (PyTorch or TensorFlow); 3D datasets train nnU-Net.
(() => {
  const form = $('#train-form');
  let polling = null, lastState = null;

  // series: [{values, color, label}]
  function drawChart(canvas, series, { fixed01 = false } = {}) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W) return;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const x = canvas.getContext('2d');
    x.scale(dpr, dpr);
    x.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const muted = css.getPropertyValue('--muted'), line = css.getPropertyValue('--line');
    const all = series.flatMap(s => s.values).filter(v => v != null && isFinite(v));
    const L = 44, R = 10, T = 8, B = 22;
    x.font = '11px system-ui, sans-serif';
    if (!all.length) {
      x.fillStyle = muted; x.textAlign = 'center';
      x.fillText('Waiting for the first epoch…', W / 2, H / 2);
      return;
    }
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.1;
    if (fixed01) { lo = Math.max(0, lo - pad - 0.01); hi = Math.min(1, hi + pad + 0.01); }
    else { lo = lo >= 0 ? Math.max(0, lo - pad) : lo - pad; hi = hi + pad || 1; }  // nnU-Net losses go below 0
    if (hi - lo < 1e-6) { hi += 0.01; lo -= 0.01; }
    const n = Math.max(...series.map(s => s.values.length));
    const px = i => L + (n <= 1 ? (W - L - R) / 2 : i * (W - L - R) / (n - 1));
    const py = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    x.strokeStyle = line; x.fillStyle = muted; x.lineWidth = 1; x.textAlign = 'right'; x.textBaseline = 'middle';
    for (let k = 0; k <= 4; k++) {
      const v = lo + (hi - lo) * k / 4, y = py(v);
      x.beginPath(); x.moveTo(L, y); x.lineTo(W - R, y); x.stroke();
      x.fillText(fixed01 ? (v * 100).toFixed(hi - lo < 0.05 ? 1 : 0) + '%' : v.toPrecision(3), L - 6, y);
    }
    x.textAlign = 'center'; x.textBaseline = 'top';
    const stepE = Math.max(1, Math.ceil(n / 8));
    for (let i = 0; i < n; i += stepE) x.fillText(String(i + 1), px(i), H - B + 6);
    for (const s of series) {
      x.strokeStyle = s.color; x.fillStyle = s.color; x.lineWidth = 2;
      x.setLineDash(s.dash ? [5, 4] : []);
      x.beginPath();
      s.values.forEach((v, i) => (i ? x.lineTo(px(i), py(v)) : x.moveTo(px(i), py(v))));
      x.stroke();
      x.setLineDash([]);
      if (s.values.length <= 40) s.values.forEach((v, i) => { x.beginPath(); x.arc(px(i), py(v), 2.5, 0, 7); x.fill(); });
    }
  }

  function renderCharts(s) {
    const h = s.history || [];
    const css = getComputedStyle(document.documentElement);
    const cTrain = css.getPropertyValue('--train'), cVal = css.getPropertyValue('--val');
    const pick = (split, key) => h.map(r => r[split][key]);
    const nn = s.config && s.config.framework === 'nnunet';
    drawChart($('#ch-loss'), [{ values: pick('train', 'loss'), color: cTrain }, { values: pick('val', 'loss'), color: cVal }]);
    if (nn) {
      $('#ch-accuracy-t').textContent = 'Pseudo Dice (validation)';
      $('#ch-recall-t').textContent = 'Pseudo Dice per class';
      drawChart($('#ch-accuracy'), [{ values: pick('val', 'ema_dice'), color: cTrain }, { values: pick('val', 'dice'), color: cVal }], { fixed01: true });
      const classes = classesOf(s.config.dataset);
      const nc = h.length ? h[h.length - 1].val.per_class.length : 0;
      drawChart($('#ch-recall'), Array.from({ length: nc }, (_, k) => ({
        values: h.map(r => r.val.per_class[k]), color: classes[k]?.color || cVal,
      })), { fixed01: true });
      $('#tr-legend').innerHTML = '<span class="sw val"></span> validation (loss, pseudo Dice) <span class="sw train"></span> training loss, Dice moving average (EMA)' +
        classes.slice(0, nc).map(c => ` <span class="sw" style="background:${c.color}"></span> ${escapeHtml(c.name)}`).join('');
    } else {
      $('#ch-accuracy-t').textContent = 'Accuracy';
      $('#ch-recall-t').textContent = 'Recall';
      drawChart($('#ch-accuracy'), [{ values: pick('train', 'accuracy'), color: cTrain }, { values: pick('val', 'accuracy'), color: cVal }], { fixed01: true });
      drawChart($('#ch-recall'), [{ values: pick('train', 'recall'), color: cTrain }, { values: pick('val', 'recall'), color: cVal }], { fixed01: true });
      $('#tr-legend').innerHTML = '<span class="sw train"></span> training <span class="sw val"></span> validation';
    }
  }

  function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
  function classesOf(ds) { return (VOLUME_SETS.find(d => d.name === ds) || {}).classes || []; }

  function metric(k, v, sub) {
    return `<div class="metric"><div class="k">${k}</div><div class="v">${v}${sub ? ` <small>${sub}</small>` : ''}</div></div>`;
  }

  function renderStatus(s) {
    const running = s.state === 'running' || s.state === 'starting';
    const nn = s.config && s.config.framework === 'nnunet';
    $('#tr-start').disabled = running;
    $('#tr-stop').disabled = !running;
    const h = s.history || [];
    const last = h[h.length - 1];
    let line = escapeHtml(s.message || '');
    if (s.config) {
      const c = s.config;
      const what = nn ? `nnU-Net ${c.configuration}` : (s.params ? (s.params / 1e6).toFixed(2) + ' M parameters' : '');
      line = `<b>${escapeHtml(c.model_name)}</b> on "${escapeHtml(c.dataset)}" · ${escapeHtml(s.device || '')} · ${what}` +
        (nn && s.plan ? ` · patch ${s.plan.patch_size.join('×')}, batch ${s.plan.batch_size}` : '') + '<br>' + line;
    }
    if (!s.config) line = 'No training running.';
    $('#tr-status').innerHTML = line;
    const prog = $('#tr-progress');
    if (s.epochs && (!nn || s.batches)) {
      const doneBatches = ((s.epoch || 1) - 1) * (s.batches || 1) + (s.batch || 0);
      const total = s.epochs * (s.batches || 1);
      let txt = nn && s.phase !== 'training' && s.phase !== 'done' && !h.length
        ? 'Preparing (nnU-Net plans and preprocesses the volumes)…'
        : `Epoch ${s.epoch || 0} / ${s.epochs} · ${nn ? 'iteration' : 'batch'} ${s.batch || 0} / ${s.batches || '?'}`;
      if (s.batch_loss != null && running) txt += ` · batch loss ${s.batch_loss.toFixed(4)}`;
      if (running && nn && h.length) {
        const sec = h.slice(-5).reduce((a, r) => a + r.seconds, 0) / Math.min(5, h.length);
        txt += ` · ≈ ${fmtDuration(sec * (s.epochs - h.length))} left`;
      } else if (running && !nn && s.started && doneBatches > 0) {
        const el = Date.now() / 1000 - s.started;
        txt += ` · ≈ ${fmtDuration(el / doneBatches * (total - doneBatches))} left`;
      }
      if (nn && s.phase === 'validation') txt = 'Final validation on the whole validation volumes…';
      setProgress(prog, s.state === 'done' ? total : doneBatches, total, txt);
    } else {
      setProgress(prog, 0, 1, '');
    }
    const b = s.best;
    if (nn) {
      const classes = classesOf(s.config.dataset), v = s.validation;
      $('#tr-metrics').innerHTML = last ? [
        metric('Loss (val)', fmtNum(last.val.loss), `train ${fmtNum(last.train.loss, 3)}`),
        metric('Pseudo Dice (val)', fmtPct(last.val.dice), `EMA ${fmtPct(last.val.ema_dice)}`),
        ...last.val.per_class.slice(0, 6).map((d, k) => metric(`Dice ${escapeHtml(classes[k]?.name || 'class ' + (k + 1))}`, fmtPct(d))),
        b ? metric('Best epoch', b.epoch, `EMA Dice ${fmtPct(b.val.ema_dice)}`) : '',
        v && v.dice != null ? metric('Validation Dice', fmtPct(v.dice), 'whole volumes') : '',
      ].join('') : '';
    } else {
      $('#tr-metrics').innerHTML = last ? [
        metric('Loss (val)', fmtNum(last.val.loss), `train ${fmtNum(last.train.loss, 3)}`),
        metric('Accuracy (val)', fmtPct(last.val.accuracy), `train ${fmtPct(last.train.accuracy)}`),
        metric('Recall (val)', fmtPct(last.val.recall), `train ${fmtPct(last.train.recall)}`),
        metric('IoU (val)', fmtPct(last.val.iou), `train ${fmtPct(last.train.iou)}`),
        b ? metric('Best epoch', b.epoch, `val loss ${fmtNum(b.val.loss, 3)}`) : '',
      ].join('') : '';
    }
    renderCharts(s);
    const log = $('#tr-log');
    log.classList.toggle('hidden', !(nn && s.log && s.log.length));
    if (nn && s.log) {
      const pre = $('pre', log), atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
      pre.textContent = s.log.join('\n');
      if (atEnd) pre.scrollTop = pre.scrollHeight;
    }
    if (lastState && lastState !== s.state && !running) refreshModels();
    lastState = s.state;
  }

  async function poll() {
    try {
      const s = await api('/api/train/status');
      renderStatus(s);
      const running = s.state === 'running' || s.state === 'starting';
      clearTimeout(polling);
      if (currentTab === 'train' || running) polling = setTimeout(poll, running ? 1000 : 4000);
    } catch (e) {
      polling = setTimeout(poll, 3000);
    }
  }

  // ---------- form ----------
  function selected() {
    const v = $('#tr-dataset').value || '';
    return { is3d: v.startsWith('3d:'), name: v.slice(3) };
  }

  function updateForm() {
    const { is3d } = selected();
    $$('#train-form .only-2d').forEach(e => e.classList.toggle('hidden', is3d));
    $$('#train-form .only-3d').forEach(e => e.classList.toggle('hidden', !is3d));
    const ep = $('#tr-epochs');
    if (is3d && ep.dataset.mode !== '3d') { ep.dataset.v2d = ep.value; ep.value = ep.dataset.v3d || 100; }
    if (!is3d && ep.dataset.mode === '3d') { ep.dataset.v3d = ep.value; ep.value = ep.dataset.v2d || 30; }
    ep.dataset.mode = is3d ? '3d' : '2d';
    renderFramework();
  }

  let SETTINGS = null, NNUNET = null;
  function renderFramework() {
    const { is3d } = selected();
    if (is3d) {
      $('#tr-framework').innerHTML = `Framework: <b>nnU-Net</b>${NNUNET ? ' ' + NNUNET : ''} (3D dataset) · PyTorch`;
      $('#tr-nnunet-note').innerHTML = 'nnU-Net chooses the network, patch size and augmentation from the volumes. ' +
        'Its default is 1000 epochs of 250 iterations: on a CPU an epoch of a 3D network takes minutes, so start with few epochs or ' +
        'iterations to check, or train on a machine with an NVIDIA GPU. The best model (EMA pseudo Dice) is saved after each improvement.';
      return;
    }
    if (!SETTINGS) return;
    $('#tr-framework').innerHTML = `Framework: <b>${FRAMEWORK_NAMES[SETTINGS.framework]}</b> · <a href="#" id="tr-fw-link">change in Settings</a>`;
    $('#tr-fw-link').addEventListener('click', e => { e.preventDefault(); showTab('settings'); });
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(form);
    const { is3d, name } = selected();
    const cfg = {
      dataset: name,
      model_name: fd.get('model_name').trim(),
      epochs: +fd.get('epochs'), val_split: +fd.get('val_split'),
      only_annotated: fd.get('only_annotated') === 'on',
    };
    if (is3d) {
      Object.assign(cfg, {
        framework: 'nnunet', configuration: fd.get('configuration'),
        iterations: +fd.get('iterations'), val_iterations: +fd.get('val_iterations'),
        sparse: fd.get('sparse') === 'on', final_validation: fd.get('final_validation') === 'on',
      });
    } else {
      Object.assign(cfg, {
        batch_size: +fd.get('batch_size'), lr: +fd.get('lr'),
        base: +fd.get('base'), depth: +fd.get('depth'), augment: fd.get('augment') === 'on',
      });
    }
    $('#tr-msg').textContent = '';
    if (!cfg.dataset) { $('#tr-msg').textContent = 'Choose a dataset.'; return; }
    const existing = MODELS.some(m => m.name === cfg.model_name);
    if (existing && !confirm(`The model "${cfg.model_name}" already exists. Replace it?`)) return;
    try {
      await api('/api/train', { method: 'POST', json: cfg });
      poll();
    } catch (err) {
      $('#tr-msg').textContent = err.message;
    }
  });
  $('#tr-stop').addEventListener('click', async () => { await api('/api/train/stop', { method: 'POST' }); poll(); });

  function suggestName() {
    const { is3d, name } = selected();
    const d = new Date();
    const stamp = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
    $('#tr-name').value = `${is3d ? 'nnunet' : 'unet'}_${name || 'model'}_${stamp}`.replace(/[\/\\:*?"<>| ]/g, '_');
  }
  $('#tr-dataset').addEventListener('change', () => { updateForm(); suggestName(); });

  function fillDatasets() {
    const sel = $('#tr-dataset'), cur = sel.value;
    sel.innerHTML = '';
    const group = (label, list, prefix, what) => {
      if (!list.length) return;
      const g = document.createElement('optgroup');
      g.label = label;
      for (const d of list) {
        const o = document.createElement('option');
        o.value = prefix + d.name;
        o.textContent = `${d.name} (${d.annotated}/${d.count} ${what} annotated)`;
        g.appendChild(o);
      }
      sel.appendChild(g);
    };
    group('2D images · U-Net', DATASETS, '2d:', 'images');
    group('3D volumes · nnU-Net', VOLUME_SETS, '3d:', 'volumes');
    if (!sel.options.length) sel.innerHTML = '<option value="">No dataset</option>';
    if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
    updateForm();
    if (!$('#tr-name').value) suggestName();
  }
  bus.on('datasets', fillDatasets);
  bus.on('train-dataset', v => { $('#tr-dataset').value = v; updateForm(); suggestName(); });
  bus.on('tab', t => { if (t === 'train') { refreshDatasets(); refreshModels(); poll(); } });
  window.addEventListener('resize', () => { if (currentTab === 'train') poll(); });
  bus.on('theme', () => { if (currentTab === 'train') poll(); });
  bus.on('settings', s => { SETTINGS = s; renderFramework(); });
  api('/api/system').then(i => { NNUNET = (i.libraries || []).find(l => l.name === 'nnU-Net')?.version || null; renderFramework(); }).catch(() => {});

  // ---------- models ----------
  window.MODELS = [];
  function modelDetails(m) {
    const b = m.best;
    if (m.framework === 'nnunet') {
      const c = m.config || {};
      const cls = (m.classes || []).map(x => x.name).join(', ');
      return `<div class="muted small">${m.dataset ? `dataset "${escapeHtml(m.dataset)}" · ` : ''}${m.date ? m.date.replace('T', ' ') + ' · ' : ''}${m.size_mb} MB
          <br>${c.configuration || ''}${c.patch_size ? `, patch ${c.patch_size.join('×')}` : ''}${c.folds > 1 ? `, ${c.folds} folds` : ''}${cls ? ` · ${escapeHtml(cls)}` : ''}</div>
        ${b ? `<div class="small">epoch ${b.epoch} · EMA pseudo Dice ${fmtPct(b.val.ema_dice)}${m.validation && m.validation.dice != null ? ` · validation Dice ${fmtPct(m.validation.dice)}` : ''}</div>` : ''}`;
    }
    return `<div class="muted small">${m.dataset ? `dataset "${escapeHtml(m.dataset)}" · ` : ''}${m.date ? m.date.replace('T', ' ') + ' · ' : ''}${m.size_mb} MB
          ${m.config ? `<br>${m.config.base} base filters, depth ${m.config.depth}` : ''}</div>
        ${b ? `<div class="small">epoch ${b.epoch} · loss ${fmtNum(b.val.loss, 3)} · acc ${fmtPct(b.val.accuracy)} · recall ${fmtPct(b.val.recall)}</div>` : ''}`;
  }

  async function refreshModels() {
    MODELS = await api('/api/models');
    bus.emit('models', MODELS);
    const el = $('#model-list');
    el.innerHTML = '';
    if (!MODELS.length) { el.innerHTML = '<div class="empty">No model yet.</div>'; return; }
    for (const m of MODELS) {
      const c = document.createElement('div');
      c.className = 'card';
      c.innerHTML = `
        <div class="title"></div>
        ${modelDetails(m)}
        <div class="row">
          <button class="primary" data-a="test">Test</button>
          <a class="btn" href="/api/models/${enc(m.name)}/download" title="${m.framework === 'nnunet' ? 'A zip of the nnU-Net model folder' : ''}">Download</a>
          <span class="spacer"></span>
          <button class="danger" data-a="del">Delete</button>
        </div>`;
      $('.title', c).textContent = m.name + ' ';
      const kind = m.framework === 'nnunet' ? `3D · ${(m.config || {}).configuration || ''}` : (m.file || '').split('.').pop();
      $('.title', c).insertAdjacentHTML('beforeend', `<span class="tag">${FRAMEWORK_NAMES[m.framework] || ''} · ${kind}</span>`);
      c.addEventListener('click', async e => {
        const a = e.target.dataset.a;
        if (a === 'test') { showTab('infer'); bus.emit('use-model', m.name); }
        if (a === 'del' && confirm(`Delete the model "${m.name}"?`)) {
          await api(`/api/models/${enc(m.name)}`, { method: 'DELETE' });
          refreshModels();
        }
      });
      el.appendChild(c);
    }
  }
  window.refreshModels = refreshModels;
  refreshModels();
  poll();
})();
