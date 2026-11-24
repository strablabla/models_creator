// Data tab: import a folder (2D: imgs/ + masks/, or images only; 3D: TIFF
// volumes or folders of slices), list the 2D and 3D datasets.
let dataMode = '2d';

function classify(files) {
  const imgs = [], masks = [];
  for (const f of files) {
    if (!IMAGE_RE.test(f.path)) continue;
    const parts = f.path.split('/').slice(0, -1).map(p => p.toLowerCase());
    if (parts.some(p => MASK_DIRS.includes(p))) masks.push(f);
    else if (parts.some(p => IMG_DIRS.includes(p))) imgs.push(f);
    else imgs.push(f); // a folder of images only: black masks will be created
  }
  return { imgs, masks };
}

function guessName(files) {
  const first = files.find(f => f.path.includes('/'));
  let name = first ? first.path.split('/')[0] : 'dataset_' + new Date().toISOString().slice(0, 10);
  if (IMG_DIRS.includes(name.toLowerCase()) || MASK_DIRS.includes(name.toLowerCase())) name = 'dataset_' + new Date().toISOString().slice(0, 10);
  return name.replace(/[\/\\:*?"<>|]/g, '_');
}

async function uploadToDataset(name, imgs, masks, progressEl) {
  const all = [...imgs.map(f => [f, 'img']), ...masks.map(f => [f, 'mask'])];
  const BATCH = 12, PARALLEL = 3;
  const batches = [];
  for (let i = 0; i < all.length; i += BATCH) batches.push(all.slice(i, i + BATCH));
  let done = 0;
  const errors = [];
  setProgress(progressEl, 0, all.length, `Importing… 0 / ${all.length}`);
  let next = 0;
  async function worker() {
    while (next < batches.length) {
      const b = batches[next++];
      const fd = new FormData();
      for (const [f, kind] of b) {
        fd.append('files', f.file, f.file.name);
        fd.append('kinds', kind);
      }
      try {
        const r = await api(`/api/datasets/${enc(name)}/files`, { method: 'POST', body: fd });
        errors.push(...r.errors);
      } catch (e) {
        errors.push(e.message);
      }
      done += b.length;
      setProgress(progressEl, done, all.length, `Importing… ${done} / ${all.length}`);
    }
  }
  await Promise.all(Array.from({ length: PARALLEL }, worker));
  const fin = await api(`/api/datasets/${enc(name)}/finalize`, { method: 'POST' });
  return { ...fin, errors };
}

// 3D: send the files with their paths, then let the server build the volumes.
async function uploadVolumes(name, entries, progressEl) {
  const errors = await uploadBatches(`/api/volumes/${enc(name)}/files`, entries, progressEl);
  setProgress(progressEl, 1, 1, 'Building the volumes…');
  const fin = await api(`/api/volumes/${enc(name)}/finalize`, { method: 'POST' });
  return { ...fin, errors: [...errors, ...fin.errors] };
}

async function importFolder(files) {
  if (dataMode === '2d' && looksLike3d(files) &&
      confirm('This folder holds sub-folders of slices in imgs/: they look like 3D volumes.\nImport them as 3D volumes (nnU-Net)?')) {
    setDataMode('3d');
  }
  if (dataMode === '3d') return importVolumes(files);
  const { imgs, masks } = classify(files);
  if (!imgs.length) { toast('No image found in what was dropped', true); return; }
  let name = prompt(`${imgs.length} image(s), ${masks.length} mask(s).\nDataset name:`, guessName(files));
  if (!name) return;
  name = name.trim().replace(/[\/\\:*?"<>|]/g, '_');
  const prog = $('#import-progress');
  try {
    const r = await uploadToDataset(name, imgs, masks, prog);
    let msg = `"${name}": ${r.images} images, ${r.paired} masks matched`;
    if (r.black_masks) msg += `, ${r.black_masks} black masks created`;
    if (r.orphan_masks) msg += `, ${r.orphan_masks} masks without an image ignored`;
    if (r.errors.length) msg += ` — ${r.errors.length} error(s): ${r.errors[0]}`;
    toast(msg, r.errors.length > 0);
    setProgress(prog, 1, 1, msg);
    await refreshDatasets();
  } catch (e) {
    toast('Import failed: ' + e.message, true);
  }
}

async function importVolumes(files, into = null) {
  const entries = classify3d(files);
  if (!entries.some(e => e.kind === 'img')) { toast('No volume found in what was dropped', true); return null; }
  let name = into;
  if (!name) {
    const ni = entries.filter(e => e.kind === 'img').length, nm = entries.length - ni;
    name = prompt(`${ni} image file(s), ${nm} label file(s).\nName of the 3D dataset:`, guessName(files));
    if (!name) return null;
    name = name.trim().replace(/[\/\\:*?"<>|]/g, '_');
    if (VOLUME_SETS.some(d => d.name === name) && !confirm(`"${name}" exists: add these volumes to it?`)) return null;
  }
  const prog = $('#import-progress');
  try {
    const r = await uploadVolumes(name, entries, prog);
    let msg = `"${name}": ${r.volumes} volume(s), ${r.paired} with labels, ${r.classes} class(es)`;
    if (r.orphan_masks) msg += `, ${r.orphan_masks} label volume(s) without an image ignored`;
    if (r.errors.length) msg += ` — ${r.errors.length} error(s): ${r.errors[0]}`;
    toast(msg, r.errors.length > 0);
    setProgress(prog, 1, 1, msg);
    await refreshDatasets();
    return r;
  } catch (e) {
    toast('Import failed: ' + e.message, true);
    return null;
  }
}

function setDataMode(m) {
  dataMode = m;
  $$('#data-mode button').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
  $$('#data-drop .mode-2d').forEach(e => e.classList.toggle('hidden', m !== '2d'));
  $$('#data-drop .mode-3d').forEach(e => e.classList.toggle('hidden', m !== '3d'));
  try { localStorage.setItem('dataMode', m); } catch (e) {}
}
$$('#data-mode button').forEach(b => b.addEventListener('click', () => setDataMode(b.dataset.mode)));
try { setDataMode(localStorage.getItem('dataMode') === '3d' ? '3d' : '2d'); } catch (e) { setDataMode('2d'); }

setupDrop($('#data-drop'), importFolder);
$('#data-pick').addEventListener('change', e => { importFolder(filesFromInput(e.target)); e.target.value = ''; });
$('#data-pick-files').addEventListener('change', e => { importVolumes(filesFromInput(e.target)); e.target.value = ''; });

function datasetCard(d, is3d) {
  const c = document.createElement('div');
  c.className = 'card';
  const what = is3d
    ? `${d.count} volume${d.count === 1 ? '' : 's'} · ${d.annotated} annotated · ${d.classes.length} class${d.classes.length === 1 ? '' : 'es'}`
    : `${d.count} images · ${d.annotated} annotated`;
  c.innerHTML = `
    <div class="title"></div>
    <div class="muted small">${what}</div>
    <div class="preview"></div>
    <div class="row">
      <button class="primary" data-a="open">Annotate</button>
      <button data-a="train">Train</button>
      <button data-a="rename">Rename</button>
      <span class="spacer"></span>
      <button class="danger" data-a="del">Delete</button>
    </div>`;
  $('.title', c).textContent = d.name + ' ';
  $('.title', c).insertAdjacentHTML('beforeend', `<span class="tag">${is3d ? '3D · nnU-Net' : '2D · 512×512'}</span>`);
  const base = is3d ? `/api/volumes/${enc(d.name)}` : `/api/datasets/${enc(d.name)}`;
  api(`${base}/items`).then(r => {
    const items = is3d ? r.items : r;
    const pv = $('.preview', c);
    for (const it of items.slice(0, 8)) {
      const im = new Image();
      im.loading = 'lazy';
      im.src = is3d ? `${base}/thumb/${enc(it.name)}?size=128` : `${base}/image/${enc(it.name)}?size=128`;
      if (is3d) im.title = `${it.name}: ${it.shape.join(' × ')}`;
      pv.appendChild(im);
    }
  });
  c.addEventListener('click', async e => {
    const a = e.target.dataset.a;
    if (a === 'open') {
      showTab(is3d ? 'annotate3d' : 'annotate');
      bus.emit(is3d ? 'open-volumes' : 'open-dataset', d.name);
    }
    if (a === 'train') { showTab('train'); bus.emit('train-dataset', (is3d ? '3d:' : '2d:') + d.name); }
    if (a === 'rename') {
      const n = prompt('New name:', d.name);
      if (!n || n === d.name) return;
      try { await api(`${base}/rename`, { method: 'POST', json: { new_name: n.trim() } }); refreshDatasets(); }
      catch (err) { toast(err.message, true); }
    }
    if (a === 'del') {
      const what = is3d ? `${d.count} volumes and their labels` : `${d.count} images and their masks`;
      if (!confirm(`Permanently delete "${d.name}" (${what})?`)) return;
      await api(base, { method: 'DELETE' });
      refreshDatasets();
    }
  });
  return c;
}

function renderDatasets() {
  const el = $('#dataset-list');
  el.innerHTML = '';
  if (!DATASETS.length && !VOLUME_SETS.length) { el.innerHTML = '<div class="empty">No dataset yet.</div>'; return; }
  for (const d of DATASETS) el.appendChild(datasetCard(d, false));
  for (const d of VOLUME_SETS) el.appendChild(datasetCard(d, true));
}
bus.on('datasets', renderDatasets);
bus.on('tab', t => { if (t === 'data') refreshDatasets(); });
refreshDatasets();
