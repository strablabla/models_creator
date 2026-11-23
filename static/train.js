// Training tab: start/stop a training, follow Loss / Accuracy / Recall, list models.
(() => {
  const form = $('#train-form');
  let polling = null, lastState = null;

  function drawChart(canvas, train, val, { fixed01 = false } = {}) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W) return;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const x = canvas.getContext('2d');
    x.scale(dpr, dpr);
    x.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const muted = css.getPropertyValue('--muted'), line = css.getPropertyValue('--line');
    const cTrain = css.getPropertyValue('--train'), cVal = css.getPropertyValue('--val');
    const all = [...train, ...val].filter(v => v != null && isFinite(v));
    const L = 44, R = 10, T = 8, B = 22;
    x.font = '11px system-ui, sans-serif';
    if (!all.length) {
      x.fillStyle = muted; x.textAlign = 'center';
      x.fillText('Waiting for the first epoch…', W / 2, H / 2);
      return;
    }
    let lo = Math.min(...all), hi = Math.max(...all);
    if (fixed01) { lo = Math.max(0, lo - (hi - lo) * 0.1 - 0.01); hi = Math.min(1, hi + (hi - lo) * 0.1 + 0.01); }
    else { lo = Math.max(0, lo - (hi - lo) * 0.1); hi = hi + (hi - lo) * 0.1 || 1; }
    if (hi - lo < 1e-6) { hi += 0.01; lo = Math.max(0, lo - 0.01); }
    const n = Math.max(train.length, val.length);
    const px = i => L + (n <= 1 ? (W - L - R) / 2 : i * (W - L - R) / (n - 1));
    const py = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    // grid
    x.strokeStyle = line; x.fillStyle = muted; x.lineWidth = 1; x.textAlign = 'right'; x.textBaseline = 'middle';
    for (let k = 0; k <= 4; k++) {
      const v = lo + (hi - lo) * k / 4, y = py(v);
      x.beginPath(); x.moveTo(L, y); x.lineTo(W - R, y); x.stroke();
      x.fillText(fixed01 ? (v * 100).toFixed(hi - lo < 0.05 ? 1 : 0) + '%' : v.toPrecision(3), L - 6, y);
    }
    x.textAlign = 'center'; x.textBaseline = 'top';
    const stepE = Math.max(1, Math.ceil(n / 8));
    for (let i = 0; i < n; i += stepE) x.fillText(String(i + 1), px(i), H - B + 6);
    for (const [s, c] of [[train, cTrain], [val, cVal]]) {
      x.strokeStyle = c; x.fillStyle = c; x.lineWidth = 2;
      x.beginPath();
      s.forEach((v, i) => (i ? x.lineTo(px(i), py(v)) : x.moveTo(px(i), py(v))));
      x.stroke();
      if (s.length <= 40) s.forEach((v, i) => { x.beginPath(); x.arc(px(i), py(v), 2.5, 0, 7); x.fill(); });
    }
  }

  function renderCharts(h) {
    const pick = (split, key) => h.map(r => r[split][key]);
    drawChart($('#ch-loss'), pick('train', 'loss'), pick('val', 'loss'));
    drawChart($('#ch-accuracy'), pick('train', 'accuracy'), pick('val', 'accuracy'), { fixed01: true });
    drawChart($('#ch-recall'), pick('train', 'recall'), pick('val', 'recall'), { fixed01: true });
  }

  function metric(k, v, sub) {
    return `<div class="metric"><div class="k">${k}</div><div class="v">${v}${sub ? ` <small>${sub}</small>` : ''}</div></div>`;
  }

  function renderStatus(s) {
    const running = s.state === 'running' || s.state === 'starting';
    $('#tr-start').disabled = running;
    $('#tr-stop').disabled = !running;
    const h = s.history || [];
    const last = h[h.length - 1];
    let line = s.message || '';
    if (s.config) {
      const c = s.config;
      line = `<b>${c.model_name}</b> on "${c.dataset}" · ${s.device || ''} · ${s.params ? (s.params / 1e6).toFixed(2) + ' M parameters' : ''}<br>` + line;
    }
    if (!s.config) line = 'No training running.';
    $('#tr-status').innerHTML = line;
    const prog = $('#tr-progress');
    if (s.epochs) {
      const doneBatches = ((s.epoch || 1) - 1) * (s.batches || 1) + (s.batch || 0);
      const total = s.epochs * (s.batches || 1);
      let txt = `Epoch ${s.epoch || 0} / ${s.epochs} · batch ${s.batch || 0} / ${s.batches || '?'}`;
      if (s.batch_loss != null && running) txt += ` · batch loss ${s.batch_loss.toFixed(4)}`;
      if (running && s.started && doneBatches > 0) {
        const el = Date.now() / 1000 - s.started;
        txt += ` · ≈ ${fmtDuration(el / doneBatches * (total - doneBatches))} left`;
      }
      setProgress(prog, s.state === 'done' ? total : doneBatches, total, txt);
    } else {
      setProgress(prog, 0, 1, '');
    }
    const b = s.best;
    $('#tr-metrics').innerHTML = last ? [
      metric('Loss (val)', fmtNum(last.val.loss), `train ${fmtNum(last.train.loss, 3)}`),
      metric('Accuracy (val)', fmtPct(last.val.accuracy), `train ${fmtPct(last.train.accuracy)}`),
      metric('Recall (val)', fmtPct(last.val.recall), `train ${fmtPct(last.train.recall)}`),
      metric('IoU (val)', fmtPct(last.val.iou), `train ${fmtPct(last.train.iou)}`),
      b ? metric('Best epoch', b.epoch, `val loss ${fmtNum(b.val.loss, 3)}`) : '',
    ].join('') : '';
    renderCharts(h);
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

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(form);
    const cfg = {
      dataset: fd.get('dataset'),
      model_name: fd.get('model_name').trim(),
      epochs: +fd.get('epochs'), batch_size: +fd.get('batch_size'),
      lr: +fd.get('lr'), val_split: +fd.get('val_split'),
      base: +fd.get('base'), depth: +fd.get('depth'),
      augment: fd.get('augment') === 'on', only_annotated: fd.get('only_annotated') === 'on',
    };
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
    const ds = $('#tr-dataset').value || 'model';
    const d = new Date();
    const stamp = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
    $('#tr-name').value = `unet_${ds}_${stamp}`.replace(/[\/\\:*?"<>| ]/g, '_');
  }
  $('#tr-dataset').addEventListener('change', suggestName);

  bus.on('datasets', list => {
    const cur = $('#tr-dataset').value;
    fillSelect($('#tr-dataset'), list.map(d => ({ value: d.name, label: `${d.name} (${d.annotated}/${d.count} annotated)` })), cur, 'No dataset');
    if (!$('#tr-name').value) suggestName();
  });
  bus.on('train-dataset', name => { $('#tr-dataset').value = name; suggestName(); });
  bus.on('tab', t => { if (t === 'train') { refreshDatasets(); refreshModels(); poll(); } });
  window.addEventListener('resize', () => { if (currentTab === 'train') poll(); });
  bus.on('theme', () => { if (currentTab === 'train') poll(); });
  bus.on('settings', s => {
    $('#tr-framework').innerHTML = `Framework: <b>${FRAMEWORK_NAMES[s.framework]}</b> · <a href="#" id="tr-fw-link">change in Settings</a>`;
    $('#tr-fw-link').addEventListener('click', e => { e.preventDefault(); showTab('settings'); });
  });

  // ---------- models ----------
  window.MODELS = [];
  async function refreshModels() {
    MODELS = await api('/api/models');
    bus.emit('models', MODELS);
    const el = $('#model-list');
    el.innerHTML = '';
    if (!MODELS.length) { el.innerHTML = '<div class="empty">No model yet.</div>'; return; }
    for (const m of MODELS) {
      const c = document.createElement('div');
      c.className = 'card';
      const b = m.best;
      c.innerHTML = `
        <div class="title"></div>
        <div class="muted small">${m.dataset ? `dataset "${m.dataset}" · ` : ''}${m.date ? m.date.replace('T', ' ') + ' · ' : ''}${m.size_mb} MB
          ${m.config ? `<br>${m.config.base} base filters, depth ${m.config.depth}` : ''}</div>
        ${b ? `<div class="small">epoch ${b.epoch} · loss ${fmtNum(b.val.loss, 3)} · acc ${fmtPct(b.val.accuracy)} · recall ${fmtPct(b.val.recall)}</div>` : ''}
        <div class="row">
          <button class="primary" data-a="test">Test</button>
          <a class="btn" href="/api/models/${enc(m.name)}/download">Download</a>
          <span class="spacer"></span>
          <button class="danger" data-a="del">Delete</button>
        </div>`;
      $('.title', c).textContent = m.name + ' ';
      $('.title', c).insertAdjacentHTML('beforeend', `<span class="tag">${FRAMEWORK_NAMES[m.framework] || ''} · ${(m.file || '').split('.').pop()}</span>`);
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
