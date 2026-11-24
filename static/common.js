// Shared helpers: API calls, toasts, tabs, reading dropped folders.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const enc = encodeURIComponent;
const IMAGE_RE = /\.(png|jpe?g|tiff?|bmp|webp|gif)$/i;
const IMG_DIRS = ['imgs', 'img', 'images', 'image'];
const MASK_DIRS = ['masks', 'mask', 'labels', 'label', 'annotations', 'gt'];

async function api(url, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) {
    o.body = JSON.stringify(o.json);
    o.headers = { 'Content-Type': 'application/json', ...(o.headers || {}) };
    delete o.json;
  }
  const r = await fetch(url, o);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (e) {}
    throw new Error(msg);
  }
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r;
}

let toastTimer;
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'show' + (err ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = ''), err ? 6000 : 2500);
}

// Tiny event bus between tabs
const bus = {
  h: {},
  on(e, f) { (this.h[e] ||= []).push(f); },
  emit(e, d) { (this.h[e] || []).forEach(f => f(d)); },
};

// ---------- tabs ----------
let currentTab = 'data';
function showTab(name) {
  if (currentTab === name) return;
  bus.emit('tab-leave', currentTab);
  currentTab = name;
  $$('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach(s => s.classList.toggle('active', s.id === 'tab-' + name));
  bus.emit('tab', name);
}
$$('#tabs button').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));

// ---------- drop zones ----------
// Calls onFiles([{file, path}]) with every file of the dropped files/folders.
function setupDrop(el, onFiles) {
  el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('over'); });
  el.addEventListener('dragleave', e => { if (!el.contains(e.relatedTarget)) el.classList.remove('over'); });
  el.addEventListener('drop', async e => {
    e.preventDefault();
    el.classList.remove('over');
    const entries = [...e.dataTransfer.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
    let files;
    if (entries.length) {
      files = [];
      for (const en of entries) await walkEntry(en, '', files);
    } else {
      files = [...e.dataTransfer.files].map(f => ({ file: f, path: f.name }));
    }
    onFiles(files);
  });
}

async function walkEntry(entry, prefix, out) {
  const path = prefix + entry.name;
  if (entry.isFile) {
    const file = await new Promise((res, rej) => entry.file(res, rej));
    out.push({ file, path });
  } else if (entry.isDirectory) {
    const reader = entry.createReader();
    // readEntries returns at most ~100 entries per call
    for (;;) {
      const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
      if (!batch.length) break;
      for (const c of batch) await walkEntry(c, path + '/', out);
    }
  }
}

function filesFromInput(input) {
  return [...input.files].map(f => ({ file: f, path: f.webkitRelativePath || f.name }));
}

function setProgress(el, done, total, text) {
  el.classList.remove('hidden');
  $('.bar', el).style.width = (total ? (100 * done / total) : 0) + '%';
  $('span', el).textContent = text ?? `${done} / ${total}`;
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function fmtPct(v) { return v == null ? '–' : (100 * v).toFixed(1) + '%'; }
function fmtNum(v, d = 4) { return v == null ? '–' : Number(v).toFixed(d); }
function fmtDuration(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), sec = s % 60;
  return h ? `${h} h ${String(m).padStart(2, '0')}` : m ? `${m} min ${String(sec).padStart(2, '0')}` : `${sec} s`;
}

// ---------- datasets list, shared by tabs ----------
let DATASETS = [], VOLUME_SETS = [];
async function refreshDatasets() {
  [DATASETS, VOLUME_SETS] = await Promise.all([api('/api/datasets'), api('/api/volumes')]);
  bus.emit('volumes', VOLUME_SETS);
  bus.emit('datasets', DATASETS);
  return DATASETS;
}

// ---------- 3D volumes ----------
// A volume travels as raw uint8 bytes (Z, Y, X), gzip-compressed, its shape in X-Shape.
async function fetchVolume(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail; } catch (e) {}
    throw new Error(msg);
  }
  const shape = r.headers.get('X-Shape').split(',').map(Number);
  return { data: new Uint8Array(await r.arrayBuffer()), shape };
}

async function putVolume(url, bytes) {
  let body = bytes, headers = { 'Content-Type': 'application/octet-stream' };
  if (window.CompressionStream) {
    body = await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer();
    headers['Content-Encoding'] = 'gzip';
  }
  return api(url, { method: 'PUT', body, headers });
}

// Dropped files of 3D volumes -> [{file, kind: 'img'|'mask', rel}], rel being
// the path under imgs/ or masks/, or else under the dropped folder (keepRoot:
// the whole path, so that a dropped folder of slices keeps its name).
function classify3d(files, keepRoot = false) {
  const out = [];
  for (const f of files) {
    if (!IMAGE_RE.test(f.path)) continue;
    const parts = f.path.split('/');
    const i = parts.slice(0, -1).findIndex(p => IMG_DIRS.includes(p.toLowerCase()) || MASK_DIRS.includes(p.toLowerCase()));
    if (i >= 0) {
      out.push({ file: f.file, kind: MASK_DIRS.includes(parts[i].toLowerCase()) ? 'mask' : 'img', rel: parts.slice(i + 1).join('/') });
    } else {
      out.push({ file: f.file, kind: 'img', rel: parts.length > 1 && !keepRoot ? parts.slice(1).join('/') : f.path });
    }
  }
  return out;
}

// Uploads [{file, kind, rel}] in batches to url (FormData: files, kinds, paths).
async function uploadBatches(url, entries, progressEl, label = 'Importing') {
  const BATCH = 16, PARALLEL = 3, errors = [];
  const batches = [];
  for (let i = 0; i < entries.length; i += BATCH) batches.push(entries.slice(i, i + BATCH));
  let done = 0, next = 0;
  setProgress(progressEl, 0, entries.length, `${label}… 0 / ${entries.length}`);
  async function worker() {
    while (next < batches.length) {
      const b = batches[next++];
      const fd = new FormData();
      for (const e of b) { fd.append('files', e.file, e.file.name); fd.append('kinds', e.kind || 'img'); fd.append('paths', e.rel); }
      try { const r = await api(url, { method: 'POST', body: fd }); errors.push(...(r.errors || [])); }
      catch (err) { errors.push(err.message); }
      done += b.length;
      setProgress(progressEl, done, entries.length, `${label}… ${done} / ${entries.length}`);
    }
  }
  await Promise.all(Array.from({ length: PARALLEL }, worker));
  return errors;
}

function looksLike3d(files) {
  // slices in sub-folders of imgs/ (imgs/vol1/z001.png): volumes, not 2D images
  return files.some(f => {
    const parts = f.path.split('/').map(p => p.toLowerCase());
    const i = parts.findIndex(p => IMG_DIRS.includes(p));
    return i >= 0 && parts.length - i > 2;
  });
}

function classColor(classes, k) {
  return hexToRgb((classes[k - 1] && classes[k - 1].color) || '#ffffff');
}

function fillSelect(sel, values, current, empty = '—') {
  sel.innerHTML = '';
  if (!values.length) {
    const o = document.createElement('option');
    o.value = ''; o.textContent = empty;
    sel.appendChild(o);
    return;
  }
  for (const v of values) {
    const o = document.createElement('option');
    o.value = v.value ?? v; o.textContent = v.label ?? v;
    sel.appendChild(o);
  }
  if (current != null && values.some(v => (v.value ?? v) === current)) sel.value = current;
}

const FRAMEWORK_NAMES = { pytorch: 'PyTorch', tensorflow: 'TensorFlow', nnunet: 'nnU-Net' };
function refreshHeader() {
  api('/api/info').then(i => {
    $('#device').textContent = FRAMEWORK_NAMES[i.framework] + (i.device ? ' · ' + i.device : '');
  }).catch(() => {});
}
bus.on('settings', refreshHeader);
refreshHeader();

// ---------- Quit: stop the server, then close the window ----------
$('#quit').addEventListener('click', async () => {
  let training = false;
  try { const s = await api('/api/train/status'); training = s.state === 'running' || s.state === 'starting'; } catch (e) {}
  const msg = training
    ? 'A training is running. Quit anyway?\nIt stops now; its best model so far is already saved.'
    : 'Quit Models creator?\nThe server stops; start it again with the Desktop icon.';
  if (!confirm(msg)) return;
  bus.emit('quit'); // lets the Annotation tab save a modified mask first
  if (window.pendingSave) await window.pendingSave;
  try { await api('/api/shutdown', { method: 'POST' }); } catch (e) {}
  // Browsers let a page close its own tab only in some cases (a tab opened
  // directly by the shortcut, in Edge or Chrome); otherwise say it is stopped.
  window.close();
  setTimeout(() => {
    const d = document.createElement('div');
    d.id = 'stopped';
    d.innerHTML = '<div class="big">Models creator is stopped</div>' +
      '<div class="muted">You can close this tab. Start the application again with the Desktop icon.</div>';
    document.body.appendChild(d);
  }, 300);
});
