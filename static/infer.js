// Inference tab: pick or drop a model, drop images, view predictions as a mosaic
// or one under the other with a before/after slider, at an adjustable size.
(() => {
  const st = { sid: null, results: [], files: [], mode: 'mosaic', show: 'overlay', color: [0, 224, 255], running: false };
  const resEl = $('#inf-results');

  // ---------- model ----------
  bus.on('models', list => {
    fillSelect($('#inf-model'), list.map(m => ({ value: m.name, label: `${m.name} · ${FRAMEWORK_NAMES[m.framework] || ''}` + (m.best ? ` (recall ${fmtPct(m.best.val.recall)})` : '') })),
      $('#inf-model').value, 'No model — train one or drop a .pt file');
  });
  bus.on('use-model', name => { $('#inf-model').value = name; });
  bus.on('tab', t => { if (t === 'infer') refreshModels(); });

  async function uploadModel(file) {
    if (!/\.(pt|pth|keras|h5)$/i.test(file.name)) { toast('A .pt, .pth, .keras or .h5 file is needed', true); return; }
    const fd = new FormData();
    fd.append('file', file, file.name);
    toast(/\.(keras|h5)$/i.test(file.name) ? 'Loading the model (TensorFlow starts the first time: up to a minute)…' : 'Loading the model…');
    try {
      const r = await api('/api/models', { method: 'POST', body: fd });
      await refreshModels();
      $('#inf-model').value = r.name;
      toast(`Model "${r.name}" ready (${r.kind})`);
    } catch (e) { toast(e.message, true); }
  }
  setupDrop($('#model-drop'), files => { if (files[0]) uploadModel(files[0].file); });
  $('#model-pick').addEventListener('change', e => { if (e.target.files[0]) uploadModel(e.target.files[0]); e.target.value = ''; });

  // ---------- running ----------
  $('#inf-threshold').addEventListener('input', e => { $('#inf-threshold-v').textContent = (+e.target.value).toFixed(2); });

  async function run(files, append) {
    const model = $('#inf-model').value;
    if (!model) { toast('Choose or drop a model', true); return; }
    files = files.filter(f => IMAGE_RE.test(f.path || f.file.name));
    if (!files.length) { toast('No image in what was dropped', true); return; }
    if (st.running) { toast('A prediction is already running', true); return; }
    st.running = true;
    if (!append || !st.sid) {
      st.sid = (await api('/api/inference/sessions', { method: 'POST' })).id;
      st.results = []; st.files = [];
      resEl.innerHTML = '';
    }
    st.files.push(...files);
    const prog = $('#inf-progress');
    const BATCH = 4;
    let done = 0, errors = 0;
    try {
      for (let i = 0; i < files.length; i += BATCH) {
        const fd = new FormData();
        fd.append('model', model);
        fd.append('threshold', $('#inf-threshold').value);
        for (const f of files.slice(i, i + BATCH)) fd.append('files', f.file, f.file.name);
        try {
          const r = await api(`/api/inference/sessions/${st.sid}/predict`, { method: 'POST', body: fd });
          for (const it of r) { st.results.push(it); resEl.appendChild(makeCard(it)); }
        } catch (e) { errors++; toast(e.message, true); }
        done = Math.min(files.length, i + BATCH);
        setProgress(prog, done, files.length, `Predicting… ${done} / ${files.length}`);
      }
      setProgress(prog, 1, 1, `${st.results.length} image(s) with "${model}" at threshold ${(+$('#inf-threshold').value).toFixed(2)}` + (errors ? ` — ${errors} batch(es) failed` : ''));
    } finally {
      st.running = false;
      updateCount();
    }
  }

  setupDrop($('#img-drop'), files => run(files, true));
  $('#img-pick').addEventListener('change', e => { run(filesFromInput(e.target), true); e.target.value = ''; });
  $('#inf-model').addEventListener('change', () => { if (st.files.length && confirm('Run the images again with this model?')) run([...st.files], false); });
  $('#inf-threshold').addEventListener('change', () => { if (st.files.length) run([...st.files], false); });

  // ---------- display ----------
  function colorize(maskImg, canvas, opaque) {
    canvas.width = maskImg.naturalWidth; canvas.height = maskImg.naturalHeight;
    const x = canvas.getContext('2d');
    x.drawImage(maskImg, 0, 0);
    const id = x.getImageData(0, 0, canvas.width, canvas.height), d = id.data;
    const [r, g, b] = st.color;
    for (let i = 0; i < d.length; i += 4) {
      const on = d[i] > 127;
      if (opaque) { d[i] = d[i + 1] = d[i + 2] = on ? 255 : 0; d[i + 3] = 255; }
      else { d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = on ? 255 : 0; }
    }
    x.putImageData(id, 0, 0);
  }

  function makeCard(it) {
    const c = document.createElement('div');
    c.className = 'res';
    c.innerHTML = `<div class="frame"><img draggable="false"><canvas></canvas>
        <div class="split hidden"></div><span class="lbl l hidden" style="left:6px">image</span><span class="lbl r hidden" style="right:6px">prediction</span></div>
      <div class="cap"><span class="nm"></span><span class="muted">${it.width}×${it.height} · ${(it.coverage * 100).toFixed(1)}%</span></div>`;
    $('.nm', c).textContent = it.name;
    $('.nm', c).title = it.name;
    const base = `/api/inference/sessions/${st.sid}`;
    $('img', c).src = `${base}/imgs/${enc(it.name)}`;
    const m = new Image();
    m.onload = () => { c._mask = m; paint(c); };
    m.src = `${base}/masks/${enc(it.name)}`;
    c._split = 0.5;
    const frame = $('.frame', c);
    let drag = false;
    const setSplit = e => {
      const r = frame.getBoundingClientRect();
      c._split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      placeSplit(c);
    };
    frame.addEventListener('pointerdown', e => {
      if (st.mode !== 'column' || st.show === 'image') return;
      drag = true; frame.setPointerCapture(e.pointerId); setSplit(e);
    });
    frame.addEventListener('pointermove', e => { if (drag) setSplit(e); });
    frame.addEventListener('pointerup', () => (drag = false));
    frame.addEventListener('dblclick', () => {
      if (st.mode === 'mosaic') { setMode('column'); requestAnimationFrame(() => c.scrollIntoView({ block: 'center' })); }
    });
    return c;
  }

  function paint(c) {
    if (!c._mask) return;
    const cv = $('canvas', c);
    colorize(c._mask, cv, st.show === 'mask');
    cv.style.opacity = st.show === 'mask' ? 1 : $('#inf-opacity').value / 100;
    cv.style.display = st.show === 'image' ? 'none' : 'block';
    placeSplit(c);
  }

  function placeSplit(c) {
    const split = st.mode === 'column' && st.show !== 'image';
    const cv = $('canvas', c);
    $('.split', c).classList.toggle('hidden', !split);
    $('.lbl.l', c).classList.toggle('hidden', !split);
    $('.lbl.r', c).classList.toggle('hidden', !split);
    if (split) {
      const p = c._split * 100;
      $('.split', c).style.left = `calc(${p}% - 1px)`;
      cv.style.clipPath = `inset(0 0 0 ${p}%)`;
    } else {
      cv.style.clipPath = 'none';
    }
  }

  function repaintAll() { $$('.res', resEl).forEach(paint); }
  function updateOpacity() {
    if (st.show === 'mask') return;
    $$('.res canvas', resEl).forEach(cv => (cv.style.opacity = $('#inf-opacity').value / 100));
  }

  function setMode(mode) {
    st.mode = mode;
    $$('#inf-mode button').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    resEl.className = mode;
    // each mode keeps its own size
    const sz = $('#inf-size');
    if (mode === 'column') { sz.min = 300; sz.max = 2400; sz.value = st.colSize ||= 800; }
    else { sz.min = 100; sz.max = 1000; sz.value = st.mosSize ||= 260; }
    applySize();
    $$('.res', resEl).forEach(placeSplit);
  }
  function applySize() {
    const v = +$('#inf-size').value;
    if (st.mode === 'column') st.colSize = v; else st.mosSize = v;
    resEl.style.setProperty('--size', v + 'px');
    $('#inf-size-v').textContent = v + ' px';
  }
  $$('#inf-mode button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $$('#inf-show button').forEach(b => b.addEventListener('click', () => {
    st.show = b.dataset.show;
    $$('#inf-show button').forEach(x => x.classList.toggle('active', x === b));
    repaintAll();
  }));
  $('#inf-size').addEventListener('input', applySize);
  $('#inf-opacity').addEventListener('input', updateOpacity);
  $('#inf-color').addEventListener('input', e => { st.color = hexToRgb(e.target.value); repaintAll(); });
  resEl.addEventListener('wheel', e => {
    if (!e.ctrlKey) return; // Ctrl + wheel: image size
    e.preventDefault();
    const sz = $('#inf-size');
    sz.value = Math.round(+sz.value * Math.exp(-e.deltaY * 0.001));
    applySize();
  }, { passive: false });

  function updateCount() {
    $('#inf-count').textContent = st.results.length ? `${st.results.length} result(s)` : '';
  }

  $('#inf-zip').addEventListener('click', () => {
    if (!st.sid || !st.results.length) return toast('No results', true);
    location.href = `/api/inference/sessions/${st.sid}/zip`;
  });
  $('#inf-to-ds').addEventListener('click', async () => {
    if (!st.sid || !st.results.length) return toast('No results', true);
    const name = prompt('Add the images and predicted masks to the dataset (existing or new):',
      DATASETS[0]?.name || 'predictions');
    if (!name) return;
    try {
      const r = await api(`/api/inference/sessions/${st.sid}/to_dataset`, { method: 'POST', json: { dataset: name.trim() } });
      toast(`${r.added} image(s) added to "${name}", to correct in Annotation`);
      refreshDatasets();
    } catch (e) { toast(e.message, true); }
  });
  $('#inf-clear').addEventListener('click', () => {
    st.sid = null; st.results = []; st.files = [];
    resEl.innerHTML = '';
    $('#inf-progress').classList.add('hidden');
    updateCount();
  });

  setMode('mosaic');
})();
