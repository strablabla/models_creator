// Data tab: import a folder (imgs/ + masks/, or images only), list datasets.
const IMG_DIRS = ['imgs', 'img', 'images', 'image'];
const MASK_DIRS = ['masks', 'mask', 'labels', 'label', 'annotations', 'gt'];

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

async function importFolder(files) {
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

setupDrop($('#data-drop'), importFolder);
$('#data-pick').addEventListener('change', e => { importFolder(filesFromInput(e.target)); e.target.value = ''; });

function renderDatasets(list) {
  const el = $('#dataset-list');
  el.innerHTML = '';
  if (!list.length) { el.innerHTML = '<div class="empty">No dataset yet.</div>'; return; }
  for (const d of list) {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = `
      <div class="title"></div>
      <div class="muted small">${d.count} images · ${d.annotated} annotated</div>
      <div class="preview"></div>
      <div class="row">
        <button class="primary" data-a="open">Annotate</button>
        <button data-a="train">Train</button>
        <button data-a="rename">Rename</button>
        <span class="spacer"></span>
        <button class="danger" data-a="del">Delete</button>
      </div>`;
    $('.title', c).textContent = d.name;
    api(`/api/datasets/${enc(d.name)}/items`).then(items => {
      const pv = $('.preview', c);
      for (const it of items.slice(0, 8)) {
        const im = new Image();
        im.loading = 'lazy';
        im.src = `/api/datasets/${enc(d.name)}/image/${enc(it.name)}?size=128`;
        pv.appendChild(im);
      }
    });
    c.addEventListener('click', async e => {
      const a = e.target.dataset.a;
      if (a === 'open') { showTab('annotate'); bus.emit('open-dataset', d.name); }
      if (a === 'train') { showTab('train'); bus.emit('train-dataset', d.name); }
      if (a === 'rename') {
        const n = prompt('New name:', d.name);
        if (!n || n === d.name) return;
        try { await api(`/api/datasets/${enc(d.name)}/rename`, { method: 'POST', json: { new_name: n.trim() } }); refreshDatasets(); }
        catch (err) { toast(err.message, true); }
      }
      if (a === 'del') {
        if (!confirm(`Permanently delete "${d.name}" (${d.count} images and their masks)?`)) return;
        await api(`/api/datasets/${enc(d.name)}`, { method: 'DELETE' });
        refreshDatasets();
      }
    });
    el.appendChild(c);
  }
}
bus.on('datasets', renderDatasets);
bus.on('tab', t => { if (t === 'data') refreshDatasets(); });
refreshDatasets();
