// Inference tab with an nnU-Net model: volumes are sent to a session, predicted
// one after the other by the server, and shown slice by slice with their labels.
(() => {
  const st = { sid: null, items: [], classes: [], sel: null, poll: null, cache: {}, running: false };
  const view = new VolumeView($('#inf3d-stage'), {
    emptyText: 'Drop volumes above: multi-page TIFF files, or folders of 2D slices.',
    onSlice: (k, n) => {
      if (in3d()) v3.setPlane(view.axis, k);
      $('#inf3d-slice').max = n - 1;
      $('#inf3d-slice').value = k;
      $('#inf3d-slice-v').textContent = `${k + 1} / ${n}`;
      view.drawStrip($('#inf3d-strip'));
    },
  });
  view.opacity = 0.5;

  // ---------- 3D view (view3d.js: surfaces of the labels) ----------
  let v3 = null, token3d = 0;
  const in3d = () => v3 && v3.visible;
  async function show3d() {
    if (!v3) {
      if (!window.View3D) { toast('The 3D view is still loading, try again', true); return false; }
      v3 = new View3D($('#inf3d-stage'));
    }
    v3.open();
    v3.setOpacity(view.opacity);
    const it = st.items.find(i => i.name === st.sel);
    if (!it || it.status !== 'done') { v3.clear(); v3.render(); v3.showMessage(it ? 'Not predicted yet.' : 'No volume.'); return true; }
    const key = st.sid + '/' + it.name;
    const t = ++token3d;
    v3.showMessage('Building the surfaces…');
    try {
      const data = await api(`/api/inference3d/sessions/${st.sid}/mesh/${enc(it.name)}`);
      if (t !== token3d || !in3d()) return true;
      v3.setPlane(view.axis, view.slice());
      v3.setData(data, st.classes, v3.lastItem === key);
      v3.lastItem = key;
    } catch (e) { if (t === token3d) v3.showMessage('Could not build the surfaces: ' + e.message); }
    return true;
  }

  async function run(files) {
    const model = $('#inf-model').value;
    const entries = classify3d(files, true).filter(e => e.kind === 'img');
    if (!entries.length) { toast('No volume in what was dropped', true); return; }
    if (st.running) { toast('Volumes are being sent, wait a moment', true); return; }
    st.running = true;
    const prog = $('#inf-progress');
    try {
      if (!st.sid) st.sid = (await api('/api/inference3d/sessions', { method: 'POST' })).id;
      const errors = await uploadBatches(`/api/inference3d/sessions/${st.sid}/files`, entries, prog, 'Sending');
      const r = await api(`/api/inference3d/sessions/${st.sid}/run`, {
        method: 'POST', json: { model, mirroring: $('#inf3d-mirror').checked, again: false },
      });
      const errs = [...errors, ...r.errors];
      setProgress(prog, 1, 1, `${r.queued.length} volume(s) queued for "${model}"` + (errs.length ? ` — ${errs.length} error(s): ${errs[0]}` : ''));
      if (errs.length) toast(errs[0], true);
      poll();
    } catch (e) {
      toast(e.message, true);
    } finally {
      st.running = false;
    }
  }

  async function again() {
    if (!st.sid || !st.items.length) return toast('No volume', true);
    const model = $('#inf-model').value;
    try {
      const r = await api(`/api/inference3d/sessions/${st.sid}/run`, {
        method: 'POST', json: { model, mirroring: $('#inf3d-mirror').checked, again: true },
      });
      st.cache = {};
      setProgress($('#inf-progress'), 1, 1, `${r.queued.length} volume(s) queued for "${model}"`);
      poll();
    } catch (e) { toast(e.message, true); }
  }

  async function poll() {
    clearTimeout(st.poll);
    if (!st.sid) return;
    try {
      const s = await api(`/api/inference3d/sessions/${st.sid}`);
      const before = Object.fromEntries(st.items.map(i => [i.name, i.status]));
      st.items = s.items;
      st.classes = s.classes;
      view.setClasses(st.classes);
      renderList();
      const sel = st.items.find(i => i.name === st.sel);
      if (sel && sel.status === 'done' && before[sel.name] !== 'done') show(sel.name, true);
      if (!st.sel) {
        const first = st.items.find(i => i.status === 'done');
        if (first) show(first.name);
      }
      const busy = st.items.some(i => i.status === 'queued' || i.status === 'running');
      const done = st.items.filter(i => i.status === 'done').length;
      $('#inf3d-count').textContent = st.items.length ? `${done} / ${st.items.length} predicted` : '';
      if (busy) st.poll = setTimeout(poll, 1500);
    } catch (e) {
      st.poll = setTimeout(poll, 3000);
    }
  }

  function statusText(it) {
    if (it.status === 'running') return `predicting… ${fmtDuration(Date.now() / 1000 - (it.started || Date.now() / 1000))}`;
    if (it.status === 'queued') return 'waiting';
    if (it.status === 'error') return 'error: ' + (it.error || '');
    if (it.status === 'done') {
      const cov = (it.coverage || []).map((c, k) => c > 0 ? `${st.classes[k]?.name || 'class ' + (k + 1)} ${(c * 100).toFixed(1)}%` : null).filter(Boolean);
      return `${it.seconds} s · ` + (cov.length ? cov.join(', ') : 'nothing found');
    }
    return '';
  }

  function renderList() {
    const el = $('#inf3d-list');
    el.innerHTML = '';
    for (const it of st.items) {
      const d = document.createElement('div');
      d.className = 'vitem' + (it.name === st.sel ? ' sel' : '');
      d.innerHTML = `<img loading="lazy"><div><div class="t"></div><div class="s muted"></div><div class="s"></div></div>`;
      $('img', d).src = `/api/inference3d/sessions/${st.sid}/thumb/${enc(it.name)}?size=96`;
      $('.t', d).textContent = it.name;
      $$('.s', d)[0].textContent = it.shape.join(' × ');
      $$('.s', d)[1].textContent = statusText(it);
      $$('.s', d)[1].classList.toggle('err', it.status === 'error');
      d.title = $$('.s', d)[1].textContent;
      d.addEventListener('click', () => show(it.name));
      el.appendChild(d);
    }
  }

  async function show(name, reloadLabels = false) {
    const it = st.items.find(i => i.name === name);
    if (!it) return;
    st.sel = name;
    $$('#inf3d-list .vitem').forEach(e => e.classList.toggle('sel', $('.t', e).textContent === name));
    $('#inf3d-name').textContent = `${name} · ${it.shape.join(' × ')}`;
    const base = `/api/inference3d/sessions/${st.sid}`;
    try {
      let c = st.cache[name];
      if (!c) {
        view.empty('Loading…');
        c = st.cache[name] = { img: await fetchVolume(`${base}/image/${enc(name)}`) };
      }
      if (it.status === 'done' && (!c.lab || reloadLabels)) c.lab = await fetchVolume(`${base}/labels/${enc(name)}`);
      if (st.sel !== name) return;
      const lab = c.lab ? c.lab.data : new Uint8Array(c.img.data.length);
      if (view.img === c.img.data) view.setLabels(lab);
      else view.setVolume(c.img.data, lab, c.img.shape, it.spacing);
      if (in3d()) show3d();
      renderLegend(it);
    } catch (e) {
      view.empty('Could not load ' + name + ': ' + e.message);
    }
  }

  function renderLegend(it) {
    $('#inf3d-legend').innerHTML = '';
    st.classes.forEach((c, k) => {
      const s = document.createElement('span');
      s.innerHTML = '<i></i><span></span>';
      $('i', s).style.background = c.color;
      const cov = it && it.coverage ? it.coverage[k] : null;
      $('span', s).textContent = c.name + (cov != null ? ` ${(cov * 100).toFixed(1)}%` : '');
      $('#inf3d-legend').appendChild(s);
    });
  }

  async function setAxis(a) {
    if (a === '3d') { if (!(await show3d())) return; }
    else { if (v3) v3.close(); view.setAxis(a); }
    $$('#inf3d-axis button').forEach(b => b.classList.toggle('active', b.dataset.axis === a));
  }
  $$('#inf3d-axis button').forEach(b => b.addEventListener('click', () => setAxis(b.dataset.axis)));
  const display = () => {
    view.opacity = $('#inf3d-opacity').value / 100;
    if (v3) v3.setOpacity(view.opacity);
    view.showLabels = $('#inf3d-visible').checked;
    view.outline = $('#inf3d-outline').checked;
    view.render();
  };
  ['#inf3d-opacity', '#inf3d-visible', '#inf3d-outline'].forEach(s => $(s).addEventListener('input', display));
  $('#inf3d-slice').addEventListener('input', e => view.setSlice(+e.target.value));

  document.addEventListener('keydown', e => {
    if (currentTab !== 'infer' || $('#inf3d').classList.contains('hidden')) return;
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
    const k = e.key.toLowerCase();
    if (k === 'arrowup') { e.preventDefault(); view.setSlice(view.slice() - (e.shiftKey ? 10 : 1)); }
    else if (k === 'arrowdown') { e.preventDefault(); view.setSlice(view.slice() + (e.shiftKey ? 10 : 1)); }
    else if (k === 'v') setAxis({ z: 'y', y: 'x', x: 'z' }[view.axis]);
    else if (k === 't') setAxis(in3d() ? view.axis : '3d');
    else if (k === 'h') { $('#inf3d-visible').checked = !$('#inf3d-visible').checked; display(); }
    else if (k === 'o') { $('#inf3d-outline').checked = !$('#inf3d-outline').checked; display(); }
    else if (k === '0') view.fit();
    else if (k === 'arrowleft' || k === 'arrowright') {
      const i = st.items.findIndex(x => x.name === st.sel) + (k === 'arrowleft' ? -1 : 1);
      if (st.items[i]) { e.preventDefault(); show(st.items[i].name); }
    }
  });

  $('#inf3d-again').addEventListener('click', again);
  $('#inf3d-zip').addEventListener('click', () => {
    if (!st.sid || !st.items.some(i => i.status === 'done')) return toast('No results', true);
    location.href = `/api/inference3d/sessions/${st.sid}/zip`;
  });
  $('#inf3d-to-ds').addEventListener('click', async () => {
    if (!st.sid || !st.items.some(i => i.status === 'done')) return toast('No results', true);
    const name = prompt('Add the volumes and predicted labels to the 3D dataset (existing or new):',
      VOLUME_SETS[0]?.name || 'predictions3d');
    if (!name) return;
    try {
      const r = await api(`/api/inference3d/sessions/${st.sid}/to_dataset`, { method: 'POST', json: { dataset: name.trim() } });
      toast(`${r.added} volume(s) added to "${name}", to correct in Annotation 3D`);
      refreshDatasets();
    } catch (e) { toast(e.message, true); }
  });
  $('#inf3d-clear').addEventListener('click', () => {
    clearTimeout(st.poll);
    Object.assign(st, { sid: null, items: [], sel: null, cache: {} });
    renderList();
    if (v3) { v3.clear(); v3.showMessage('No volume.'); v3.render(); }
    view.empty('Drop volumes above: multi-page TIFF files, or folders of 2D slices.');
    $('#inf3d-count').textContent = '';
    $('#inf3d-name').textContent = '';
    $('#inf3d-legend').innerHTML = '';
    $('#inf-progress').classList.add('hidden');
  });

  bus.on('infer3d-files', run);
  bus.on('infer-mode', is3d => { if (is3d) requestAnimationFrame(() => view.fit()); });
  $('#vol-pick').addEventListener('change', e => { run(filesFromInput(e.target)); e.target.value = ''; });
  $('#vol-pick-dir').addEventListener('change', e => { run(filesFromInput(e.target)); e.target.value = ''; });
})();
