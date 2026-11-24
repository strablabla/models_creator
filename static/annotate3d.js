// Annotation 3D tab: the volumes of a 3D dataset, slice by slice in XY, XZ or
// YZ, with their labels (one color per class): brush, eraser, fill, copy of the
// previous slice, undo-redo. Labels are saved as a whole volume.
(() => {
  // edits counts the changes of the volume shown: a save that ends after
  // another change leaves it modified
  const st = { ds: null, items: [], classes: [], idx: -1, dirty: false, edits: 0, loadToken: 0 };
  const view = new VolumeView($('#a3-stage'), {
    editable: true,
    emptyText: 'Choose a 3D dataset in the Data tab.',
    onChange: () => { st.edits++; if (!st.dirty) { st.dirty = true; $('#a3-dirty').textContent = '● modified'; } },
    onSlice: (k, n) => {
      if (in3d()) v3.setPlane(view.axis, k);
      $('#a3-slice').max = n - 1;
      $('#a3-slice').value = k;
      $('#a3-slice-v').textContent = `${k + 1} / ${n}`;
      view.drawStrip($('#a3-strip'));
    },
    onStroke: () => { clearTimeout(st.stripTimer); st.stripTimer = setTimeout(() => view.drawStrip($('#a3-strip')), 250); },
  });

  // ---------- 3D view (view3d.js: surfaces of the labels) ----------
  let v3 = null, token3d = 0;
  const in3d = () => v3 && v3.visible;
  async function show3d() {
    if (!v3) {
      if (!window.View3D) { toast('The 3D view is still loading, try again', true); return false; }
      v3 = new View3D($('#a3-stage'));
    }
    v3.open();
    v3.setOpacity(view.opacity);
    const it = st.items[st.idx];
    if (!it) { v3.showMessage('No volume.'); return true; }
    if (st.dirty && !(await save())) return true;
    const key = st.ds + '/' + it.name;
    const t = ++token3d;
    v3.showMessage('Building the surfaces…');
    try {
      const data = await api(`/api/volumes/${enc(st.ds)}/mesh/${enc(it.name)}`);
      if (t !== token3d || !in3d()) return true;
      v3.setPlane(view.axis, view.slice());
      v3.setData(data, st.classes, v3.lastItem === key);
      v3.lastItem = key;
    } catch (e) { if (t === token3d) v3.showMessage('Could not build the surfaces: ' + e.message); }
    return true;
  }

  // ---------- classes ----------
  function renderClasses() {
    const el = $('#a3-classes');
    el.innerHTML = '';
    st.classes.forEach((c, i) => {
      const k = i + 1;
      const row = document.createElement('div');
      row.className = 'cls' + (view.active === k ? ' sel' : '');
      row.title = 'Click: paint with this class · double-click the name: rename';
      row.innerHTML = `<span class="k">${k <= 9 ? k : ''}</span><input type="color"><span class="n"></span>`;
      $('input', row).value = c.color;
      $('.n', row).textContent = c.name;
      row.addEventListener('click', e => { if (e.target.tagName !== 'INPUT') setActive(k); });
      $('input', row).addEventListener('input', e => { c.color = e.target.value; view.setClasses(st.classes); if (v3) v3.setClasses(st.classes); });
      $('input', row).addEventListener('change', saveClasses);
      $('.n', row).addEventListener('dblclick', () => {
        const n = prompt('Class name:', c.name);
        if (!n || n.trim() === c.name) return;
        c.name = n.trim();
        saveClasses();
      });
      el.appendChild(row);
    });
  }

  function setActive(k) {
    if (k < 1 || k > st.classes.length) return;
    view.active = k;
    renderClasses();
    if (view.tool === 'eraser') setTool('brush');
  }

  async function saveClasses() {
    if (!st.ds) return;
    try {
      st.classes = await api(`/api/volumes/${enc(st.ds)}/classes`, { method: 'PUT', json: { classes: st.classes } });
    } catch (e) { toast(e.message, true); }
    view.setClasses(st.classes);
    renderClasses();
  }

  $('#a3-add-class').addEventListener('click', () => {
    if (!st.ds) return;
    const k = st.classes.length + 1;
    const n = prompt('Name of the new class:', `Class ${k}`);
    if (!n) return;
    const palette = ['#ff2d55', '#00e0ff', '#ffd23f', '#3ccf91', '#b36bff', '#ff8a00', '#4c8dff', '#ff5cf0', '#a8e10c', '#00b3a4', '#ff6b6b', '#c9a227'];
    st.classes.push({ name: n.trim(), color: palette[(k - 1) % palette.length] });
    view.active = k;
    saveClasses();
  });
  $('#a3-protect').addEventListener('change', e => { view.protect = e.target.checked; });

  // ---------- toolbar ----------
  function setTool(t) {
    view.setTool(t);
    $$('#a3-tools button').forEach(b => b.classList.toggle('active', b.dataset.tool === t));
  }
  $$('#a3-tools button').forEach(b => b.addEventListener('click', () => setTool(b.dataset.tool)));

  async function setAxis(a) {
    if (a === '3d') { if (!(await show3d())) return; }
    else { if (v3) v3.close(); view.setAxis(a); }
    $$('#a3-axis button').forEach(b => b.classList.toggle('active', b.dataset.axis === a));
  }
  $$('#a3-axis button').forEach(b => b.addEventListener('click', () => setAxis(b.dataset.axis)));

  function setSize(v) {
    view.size = Math.max(1, Math.min(120, Math.round(v)));
    $('#a3-size').value = view.size;
    $('#a3-size-v').textContent = view.size;
  }
  $('#a3-size').addEventListener('input', e => setSize(+e.target.value));
  const display = () => {
    view.opacity = $('#a3-opacity').value / 100;
    if (v3) v3.setOpacity(view.opacity);
    view.showLabels = $('#a3-visible').checked;
    view.outline = $('#a3-outline').checked;
    view.render();
  };
  ['#a3-opacity', '#a3-visible', '#a3-outline'].forEach(s => $(s).addEventListener('input', display));
  $('#a3-slice').addEventListener('input', e => view.setSlice(+e.target.value));
  $('#a3-undo').addEventListener('click', () => view.undo());
  $('#a3-redo').addEventListener('click', () => view.redo());
  $('#a3-copy').addEventListener('click', e => copySlice(e.shiftKey ? 1 : -1));
  $('#a3-clear').addEventListener('click', () => view.clearSlice());
  $('#a3-fit').addEventListener('click', () => view.fit());
  $('#a3-save').addEventListener('click', () => save());
  $('#a3-prev').addEventListener('click', () => go(st.idx - 1));
  $('#a3-next').addEventListener('click', () => go(st.idx + 1));

  function copySlice(offset) {
    if (!view.copySlice(offset)) toast(offset < 0 ? 'No previous slice' : 'No next slice');
  }

  $('#a3-delete-item').addEventListener('click', async () => {
    const it = st.items[st.idx];
    if (!it || !confirm(`Remove the volume "${it.name}" and its labels from the dataset?`)) return;
    await api(`/api/volumes/${enc(st.ds)}/items/${enc(it.name)}`, { method: 'DELETE' });
    st.dirty = false;
    const i = st.idx;
    st.items.splice(i, 1);
    st.idx = -1;
    renderList();
    if (st.items.length) go(Math.min(i, st.items.length - 1)); else showEmpty('This dataset is empty.');
  });

  // ---------- keyboard ----------
  document.addEventListener('keydown', e => {
    if (currentTab !== 'annotate3d') return;
    const tag = e.target.tagName;
    if ((tag === 'INPUT' && !['range', 'checkbox', 'color'].includes(e.target.type)) || tag === 'SELECT' || tag === 'TEXTAREA') return;
    const k = e.key.toLowerCase();
    if (e.ctrlKey || e.metaKey) {
      if (k === 'z' && !e.shiftKey) { e.preventDefault(); view.undo(); }
      else if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); view.redo(); }
      else if (k === 's') { e.preventDefault(); save(); }
      return;
    }
    if (k === ' ') { e.preventDefault(); if (!view.spaceDown) view.setSpace(true); return; }
    const tools = { b: 'brush', e: 'eraser', f: 'fill' };
    if (tools[k]) setTool(tools[k]);
    else if (/^[1-9]$/.test(k)) setActive(+k);
    else if (k === '[') setSize(view.size / 1.25);
    else if (k === ']') setSize(view.size * 1.25 + 1);
    else if (k === 'h') { $('#a3-visible').checked = !$('#a3-visible').checked; display(); }
    else if (k === 'o') { $('#a3-outline').checked = !$('#a3-outline').checked; display(); }
    else if (k === '0') view.fit();
    else if (k === 'v') setAxis({ z: 'y', y: 'x', x: 'z' }[view.axis]);
    else if (k === 't') setAxis(in3d() ? view.axis : '3d');
    else if (k === 'c') copySlice(e.shiftKey ? 1 : -1);
    else if (k === 'arrowup' || k === 'w') { e.preventDefault(); view.setSlice(view.slice() - (e.shiftKey ? 10 : 1)); }
    else if (k === 'arrowdown' || k === 's') { e.preventDefault(); view.setSlice(view.slice() + (e.shiftKey ? 10 : 1)); }
    else if (k === 'pageup') { e.preventDefault(); view.setSlice(view.slice() - 10); }
    else if (k === 'pagedown') { e.preventDefault(); view.setSlice(view.slice() + 10); }
    else if (k === 'arrowleft' || k === 'a') { e.preventDefault(); go(st.idx - 1); }
    else if (k === 'arrowright' || k === 'd') { e.preventDefault(); go(st.idx + 1); }
  });
  document.addEventListener('keyup', e => { if (e.key === ' ') view.setSpace(false); });

  // ---------- loading / saving ----------
  async function save() {
    const it = st.items[st.idx];
    if (!it || !st.dirty || !view.lab) return true;
    const lab = view.lab, name = it.name, ds = st.ds, edits = st.edits;
    $('#a3-dirty').textContent = 'saving…';
    try {
      await putVolume(`/api/volumes/${enc(ds)}/labels/${enc(name)}`, lab.slice());
      if (st.ds === ds && st.items[st.idx] === it) {
        st.dirty = st.edits !== edits;
        $('#a3-dirty').textContent = st.dirty ? '● modified' : 'saved';
      }
      let n = 0;
      const [Z, Y, X] = it.shape;
      for (let z = 0; z < Z; z++) {
        for (let i = z * Y * X, e = i + Y * X; i < e; i++) if (lab[i]) { n++; break; }
      }
      it.labeled_slices = n;
      it.annotated = n > 0;
      const th = $(`#a3-list .thumb[data-name="${CSS.escape(name)}"]`);
      if (th) { $('.dot', th).classList.toggle('on', it.annotated); $('.sub', th).textContent = `${n}/${Z}`; }
      updateCount();
      return true;
    } catch (e) {
      $('#a3-dirty').textContent = '● modified';
      toast('Could not save: ' + e.message, true);
      return false;
    }
  }

  async function go(i) {
    if (i < 0 || i >= st.items.length || i === st.idx) return;
    if (st.dirty && !(await save())) return;
    st.idx = i;
    const it = st.items[i];
    const token = ++st.loadToken;
    $$('#a3-list .thumb.sel').forEach(t => t.classList.remove('sel'));
    const th = $(`#a3-list .thumb[data-name="${CSS.escape(it.name)}"]`);
    if (th) { th.classList.add('sel'); th.scrollIntoView({ block: 'nearest' }); }
    $('#a3-name').textContent = `${it.name} · ${it.shape.join(' × ')}`;
    $('#a3-pos').textContent = `${i + 1} / ${st.items.length}`;
    $('#a3-dirty').textContent = '';
    view.empty('Loading…');
    const base = `/api/volumes/${enc(st.ds)}`;
    try {
      const [img, lab] = await Promise.all([fetchVolume(`${base}/image/${enc(it.name)}`), fetchVolume(`${base}/labels/${enc(it.name)}`)]);
      if (token !== st.loadToken) return;
      st.dirty = false;
      view.setClasses(st.classes);
      view.setVolume(img.data, lab.data, img.shape, it.spacing);
      if (in3d()) show3d();
    } catch (e) {
      if (token === st.loadToken) view.empty('Could not load ' + it.name + ': ' + e.message);
    }
  }

  function updateCount() {
    const a = st.items.filter(i => i.annotated).length;
    $('#a3-count').textContent = `${a} / ${st.items.length} annotated`;
  }

  function renderList() {
    const el = $('#a3-list');
    el.innerHTML = '';
    const onlyEmpty = $('#a3-only-empty').checked;
    st.items.forEach((it, i) => {
      if (onlyEmpty && it.annotated && i !== st.idx) return;
      const d = document.createElement('div');
      d.className = 'thumb' + (i === st.idx ? ' sel' : '');
      d.dataset.name = it.name;
      d.innerHTML = `<img loading="lazy"><span class="dot${it.annotated ? ' on' : ''}"></span><span class="sub"></span><span class="nm"></span>`;
      $('img', d).src = `/api/volumes/${enc(st.ds)}/thumb/${enc(it.name)}?size=128`;
      $('.nm', d).textContent = it.name;
      $('.sub', d).textContent = `${it.labeled_slices}/${it.shape[0]}`;
      d.title = `${it.name}: ${it.shape.join(' × ')} voxels, spacing ${it.spacing.map(s => +s.toFixed(3)).join(' × ')}\n${it.labeled_slices} of ${it.shape[0]} Z slices hold labels`;
      d.addEventListener('click', () => go(i));
      el.appendChild(d);
    });
    updateCount();
  }
  $('#a3-only-empty').addEventListener('change', renderList);

  function showEmpty(msg) {
    view.empty(msg);
    $('#a3-name').textContent = '';
    $('#a3-pos').textContent = '';
    $('#a3-slice-v').textContent = '';
  }

  async function openDataset(name) {
    if (st.dirty) await save();
    st.ds = name;
    st.idx = -1;
    $('#a3-dataset').value = name;
    if (!name) { st.items = []; st.classes = []; renderList(); renderClasses(); showEmpty('Import 3D volumes in the Data tab (3D volumes · nnU-Net).'); return; }
    const r = await api(`/api/volumes/${enc(name)}/items`);
    st.items = r.items;
    st.classes = r.classes;
    if (view.active > st.classes.length) view.active = 1;
    renderList();
    renderClasses();
    view.setClasses(st.classes);
    if (st.items.length) go(0); else showEmpty('This dataset is empty.');
  }

  $('#a3-dataset').addEventListener('change', e => openDataset(e.target.value));
  bus.on('volumes', list => fillSelect($('#a3-dataset'), list.map(d => d.name), st.ds, 'No 3D dataset'));
  bus.on('open-volumes', name => openDataset(name));
  bus.on('tab', async t => {
    if (t !== 'annotate3d') return;
    await refreshDatasets();
    if (!st.ds || !VOLUME_SETS.some(d => d.name === st.ds)) openDataset(VOLUME_SETS[0]?.name || '');
    requestAnimationFrame(() => view.fit());
  });
  bus.on('tab-leave', t => { if (t === 'annotate3d' && st.dirty) save(); });
  bus.on('quit', () => { if (st.dirty) window.pendingSave = save(); });
  window.addEventListener('beforeunload', e => { if (st.dirty) { e.preventDefault(); e.returnValue = ''; } });

  setupDrop($('#a3-drop'), async files => {
    if (!st.ds) { toast('Choose a 3D dataset first', true); return; }
    if (st.dirty) await save();
    const r = await importVolumes(files, st.ds);
    if (!r) return;
    const cur = st.items[st.idx]?.name;
    const res = await api(`/api/volumes/${enc(st.ds)}/items`);
    st.items = res.items; st.classes = res.classes;
    st.idx = -1;
    renderList(); renderClasses();
    go(Math.max(0, st.items.findIndex(i => i.name === cur)));
  });

  setTool('brush');
  setSize(12);
})();
