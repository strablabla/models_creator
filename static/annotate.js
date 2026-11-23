// Annotation tab: image + mask overlay, brush / eraser / polygon / rectangle / fill,
// undo-redo, zoom and pan. The mask is a Uint8Array of 512x512 values 0/1.
(() => {
  const S = 512;
  const stage = $('#stage'), box = $('#canvases');
  const imgC = $('#img-canvas'), maskC = $('#mask-canvas'), uiC = $('#ui-canvas');
  const imgX = imgC.getContext('2d'), maskX = maskC.getContext('2d'), uiX = uiC.getContext('2d');
  const cursor = $('#cursor');

  const st = {
    ds: null, items: [], idx: -1, loadToken: 0,
    mask: new Uint8Array(S * S), undo: [], redo: [], dirty: false,
    tool: 'brush', size: 20, color: [255, 45, 85],
    view: { scale: 1, tx: 0, ty: 0 },
    drawing: false, last: null, poly: [], rect: null, panning: null, spaceDown: false, mouse: null,
  };

  // ---------- rendering ----------
  let renderQueued = false;
  function render() {
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
      renderQueued = false;
      const id = maskX.createImageData(S, S), d = id.data, m = st.mask;
      const [r, g, b] = st.color;
      const outline = $('#mask-outline').checked;
      for (let i = 0; i < S * S; i++) {
        if (!m[i]) continue;
        if (outline) {
          const x = i % S, y = (i / S) | 0;
          const inner = x > 0 && x < S - 1 && y > 0 && y < S - 1 && m[i - 1] && m[i + 1] && m[i - S] && m[i + S];
          if (inner) continue;
        }
        const k = i * 4;
        d[k] = r; d[k + 1] = g; d[k + 2] = b; d[k + 3] = 255;
      }
      maskX.putImageData(id, 0, 0);
      drawUI();
    });
  }

  function drawUI() {
    uiX.clearRect(0, 0, S, S);
    const lw = 1.5 / st.view.scale;
    uiX.lineWidth = lw;
    if (st.poly.length) {
      uiX.strokeStyle = '#fff';
      uiX.fillStyle = 'rgba(255,255,255,.15)';
      uiX.beginPath();
      st.poly.forEach((p, i) => (i ? uiX.lineTo(p[0], p[1]) : uiX.moveTo(p[0], p[1])));
      if (st.mouse) uiX.lineTo(st.mouse[0], st.mouse[1]);
      uiX.closePath();
      uiX.fill();
      uiX.stroke();
      for (const p of st.poly) {
        uiX.fillStyle = '#fff';
        uiX.fillRect(p[0] - 2 * lw, p[1] - 2 * lw, 4 * lw, 4 * lw);
      }
    }
    if (st.rect) {
      const [x0, y0, x1, y1] = st.rect;
      uiX.strokeStyle = '#fff';
      uiX.setLineDash([4 * lw, 3 * lw]);
      uiX.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      uiX.setLineDash([]);
    }
  }

  function applyView() {
    const v = st.view;
    box.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`;
    imgC.style.imageRendering = v.scale >= 2 ? 'pixelated' : 'auto';
    updateCursor();
    drawUI();
  }

  function fit() {
    const r = stage.getBoundingClientRect();
    const s = Math.min(r.width / S, r.height / S) * 0.94;
    st.view = { scale: s, tx: (r.width - S * s) / 2, ty: (r.height - S * s) / 2 };
    applyView();
  }

  function toImage(e) {
    const r = stage.getBoundingClientRect();
    return [(e.clientX - r.left - st.view.tx) / st.view.scale, (e.clientY - r.top - st.view.ty) / st.view.scale];
  }

  function updateCursor() {
    const show = st.mouse && ['brush', 'eraser'].includes(effectiveTool()) && st.idx >= 0;
    cursor.style.display = show ? 'block' : 'none';
    if (!show) return;
    const d = st.size * st.view.scale;
    cursor.style.width = cursor.style.height = d + 'px';
    cursor.style.left = st.mouse[0] * st.view.scale + st.view.tx + 'px';
    cursor.style.top = st.mouse[1] * st.view.scale + st.view.ty + 'px';
    cursor.style.borderColor = effectiveTool() === 'eraser' ? '#ffd23f' : '#fff';
  }

  function effectiveTool() { return st.spaceDown ? 'pan' : st.tool; }

  function setStageClass() {
    const t = effectiveTool();
    stage.classList.toggle('pan', t === 'pan');
    stage.classList.toggle('poly', t === 'polygon' || t === 'rect');
    stage.classList.toggle('fillc', t === 'fill');
  }

  // ---------- editing primitives ----------
  function pushUndo() {
    st.undo.push(st.mask.slice());
    if (st.undo.length > 60) st.undo.shift();
    st.redo = [];
  }
  function changed() { st.dirty = true; $('#an-dirty').textContent = '● modified'; render(); }

  function stamp(cx, cy, v) {
    const r = st.size / 2, m = st.mask;
    const r2 = r * r;
    const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(S - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      const dy = y + 0.5 - cy;
      const h = r2 - dy * dy;
      if (h < 0) continue;
      const w = Math.sqrt(h);
      const x0 = Math.max(0, Math.ceil(cx - w - 0.5)), x1 = Math.min(S - 1, Math.floor(cx + w - 0.5));
      m.fill(v, y * S + x0, y * S + x1 + 1);
    }
    if (st.size <= 1.5) {
      const x = Math.floor(cx), y = Math.floor(cy);
      if (x >= 0 && y >= 0 && x < S && y < S) m[y * S + x] = v;
    }
  }

  function line(a, b, v) {
    const dist = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const step = Math.max(0.5, st.size / 4);
    const n = Math.max(1, Math.ceil(dist / step));
    for (let i = 1; i <= n; i++) stamp(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, v);
  }

  function fillPath(drawPath, v) {
    const c = document.createElement('canvas');
    c.width = c.height = S;
    const x = c.getContext('2d');
    x.fillStyle = '#fff';
    x.beginPath();
    drawPath(x);
    x.fill();
    const d = x.getImageData(0, 0, S, S).data, m = st.mask;
    for (let i = 0; i < S * S; i++) if (d[i * 4 + 3] > 127) m[i] = v;
  }

  function closePolygon(erase) {
    if (st.poly.length >= 3) {
      pushUndo();
      const pts = st.poly;
      fillPath(x => pts.forEach((p, i) => (i ? x.lineTo(p[0], p[1]) : x.moveTo(p[0], p[1]))), erase ? 0 : 1);
      changed();
    }
    st.poly = [];
    drawUI();
  }

  function floodFill(px, py, v) {
    const x0 = Math.floor(px), y0 = Math.floor(py);
    if (x0 < 0 || y0 < 0 || x0 >= S || y0 >= S) return;
    const m = st.mask, from = m[y0 * S + x0];
    if (from === v) return;
    pushUndo();
    const stack = [y0 * S + x0];
    m[y0 * S + x0] = v;
    while (stack.length) {
      const i = stack.pop(), x = i % S;
      if (x > 0 && m[i - 1] === from) { m[i - 1] = v; stack.push(i - 1); }
      if (x < S - 1 && m[i + 1] === from) { m[i + 1] = v; stack.push(i + 1); }
      if (i >= S && m[i - S] === from) { m[i - S] = v; stack.push(i - S); }
      if (i < S * (S - 1) && m[i + S] === from) { m[i + S] = v; stack.push(i + S); }
    }
    changed();
  }

  // ---------- pointer ----------
  stage.addEventListener('contextmenu', e => e.preventDefault());
  stage.addEventListener('pointerdown', e => {
    if (st.idx < 0) return;
    stage.setPointerCapture(e.pointerId);
    const p = toImage(e);
    const tool = effectiveTool();
    if (tool === 'pan' || e.button === 1 || e.button === 2) {
      st.panning = { x: e.clientX, y: e.clientY, tx: st.view.tx, ty: st.view.ty };
      stage.classList.add('panning');
      return;
    }
    if (e.button !== 0) return;
    if (tool === 'brush' || tool === 'eraser') {
      pushUndo();
      st.drawing = true;
      st.drawValue = tool === 'eraser' || e.shiftKey ? 0 : 1;
      st.last = p;
      stamp(p[0], p[1], st.drawValue);
      changed();
    } else if (tool === 'polygon') {
      const last = st.poly[st.poly.length - 1];
      if (!last || Math.hypot(last[0] - p[0], last[1] - p[1]) > 2 / st.view.scale) st.poly.push(p);
      drawUI();
    } else if (tool === 'rect') {
      st.rect = [p[0], p[1], p[0], p[1]];
      st.rectErase = e.shiftKey;
    } else if (tool === 'fill') {
      floodFill(p[0], p[1], e.shiftKey ? 0 : 1);
    }
  });

  stage.addEventListener('pointermove', e => {
    const p = toImage(e);
    st.mouse = p;
    if (st.panning) {
      st.view.tx = st.panning.tx + e.clientX - st.panning.x;
      st.view.ty = st.panning.ty + e.clientY - st.panning.y;
      applyView();
      return;
    }
    if (st.drawing) {
      line(st.last, p, st.drawValue);
      st.last = p;
      changed();
    } else if (st.rect) {
      st.rect[2] = p[0]; st.rect[3] = p[1];
      drawUI();
    } else if (st.poly.length) {
      drawUI();
    }
    updateCursor();
  });

  function endPointer(e) {
    if (st.panning) { st.panning = null; stage.classList.remove('panning'); }
    st.drawing = false;
    if (st.rect) {
      const [x0, y0, x1, y1] = st.rect;
      st.rect = null;
      if (Math.abs(x1 - x0) >= 1 && Math.abs(y1 - y0) >= 1) {
        pushUndo();
        fillPath(x => x.rect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)), st.rectErase || e.shiftKey ? 0 : 1);
        changed();
      }
      drawUI();
    }
  }
  stage.addEventListener('pointerup', endPointer);
  stage.addEventListener('pointercancel', endPointer);
  stage.addEventListener('pointerleave', () => { st.mouse = null; updateCursor(); });
  stage.addEventListener('dblclick', e => { if (st.tool === 'polygon') closePolygon(e.shiftKey); });

  stage.addEventListener('wheel', e => {
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const f = Math.exp(-e.deltaY * 0.0015);
    const ns = Math.min(30, Math.max(0.1, st.view.scale * f));
    st.view.tx = mx - (mx - st.view.tx) * ns / st.view.scale;
    st.view.ty = my - (my - st.view.ty) * ns / st.view.scale;
    st.view.scale = ns;
    applyView();
  }, { passive: false });

  new ResizeObserver(() => { if (currentTab === 'annotate') fit(); }).observe(stage);

  // ---------- toolbar ----------
  function setTool(t) {
    if (st.tool === 'polygon' && t !== 'polygon') { st.poly = []; drawUI(); }
    st.tool = t;
    $$('#tools button').forEach(b => b.classList.toggle('active', b.dataset.tool === t));
    setStageClass();
    updateCursor();
  }
  $$('#tools button').forEach(b => b.addEventListener('click', () => setTool(b.dataset.tool)));

  function setSize(v) {
    st.size = Math.max(1, Math.min(120, Math.round(v)));
    $('#brush-size').value = st.size;
    $('#brush-size-v').textContent = st.size;
    updateCursor();
  }
  $('#brush-size').addEventListener('input', e => setSize(+e.target.value));
  const opacity = () => { maskC.style.opacity = $('#mask-visible').checked ? $('#mask-opacity').value / 100 : 0; };
  $('#mask-opacity').addEventListener('input', opacity);
  $('#mask-visible').addEventListener('change', opacity);
  $('#mask-outline').addEventListener('change', render);
  $('#mask-color').addEventListener('input', e => { st.color = hexToRgb(e.target.value); render(); });
  opacity();

  function undo() {
    if (!st.undo.length) return;
    st.redo.push(st.mask);
    st.mask = st.undo.pop();
    changed();
  }
  function redo() {
    if (!st.redo.length) return;
    st.undo.push(st.mask);
    st.mask = st.redo.pop();
    changed();
  }
  $('#undo').addEventListener('click', undo);
  $('#redo').addEventListener('click', redo);
  $('#invert').addEventListener('click', () => { if (st.idx < 0) return; pushUndo(); for (let i = 0; i < S * S; i++) st.mask[i] ^= 1; changed(); });
  $('#clear').addEventListener('click', () => { if (st.idx < 0) return; pushUndo(); st.mask.fill(0); changed(); });
  $('#fit').addEventListener('click', fit);
  $('#save').addEventListener('click', () => save());
  $('#prev').addEventListener('click', () => go(st.idx - 1));
  $('#next').addEventListener('click', () => go(st.idx + 1));
  $('#delete-item').addEventListener('click', async () => {
    const it = st.items[st.idx];
    if (!it || !confirm(`Remove "${it.name}" and its mask from the dataset?`)) return;
    await api(`/api/datasets/${enc(st.ds)}/items/${enc(it.name)}`, { method: 'DELETE' });
    st.dirty = false;
    const i = st.idx;
    st.items.splice(i, 1);
    st.idx = -1;
    renderList();
    if (st.items.length) go(Math.min(i, st.items.length - 1)); else showEmpty('This dataset is empty.');
  });

  // ---------- keyboard ----------
  document.addEventListener('keydown', e => {
    if (currentTab !== 'annotate') return;
    const tag = e.target.tagName;
    if ((tag === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox') || tag === 'SELECT' || tag === 'TEXTAREA') return;
    const k = e.key.toLowerCase();
    if (e.ctrlKey || e.metaKey) {
      if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
      else if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); redo(); }
      else if (k === 's') { e.preventDefault(); save(); }
      return;
    }
    if (k === ' ') { e.preventDefault(); if (!st.spaceDown) { st.spaceDown = true; setStageClass(); updateCursor(); } return; }
    const tools = { b: 'brush', e: 'eraser', p: 'polygon', r: 'rect', f: 'fill' };
    if (tools[k]) setTool(tools[k]);
    else if (k === '[') setSize(st.size / 1.25);
    else if (k === ']') setSize(st.size * 1.25 + 1);
    else if (k === 'h') { $('#mask-visible').checked = !$('#mask-visible').checked; opacity(); }
    else if (k === 'o') { $('#mask-outline').checked = !$('#mask-outline').checked; render(); }
    else if (k === '0') fit();
    else if (k === 'arrowleft' || k === 'a') { e.preventDefault(); go(st.idx - 1); }
    else if (k === 'arrowright' || k === 'd') { e.preventDefault(); go(st.idx + 1); }
    else if (k === 'enter' && st.tool === 'polygon') closePolygon(e.shiftKey);
    else if (k === 'escape') { st.poly = []; st.rect = null; drawUI(); }
    else if (k === 'backspace' && st.poly.length) { st.poly.pop(); drawUI(); }
  });
  document.addEventListener('keyup', e => {
    if (e.key === ' ') { st.spaceDown = false; setStageClass(); updateCursor(); }
  });

  // ---------- loading / saving ----------
  function loadImg(src) {
    return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
  }

  async function save() {
    const it = st.items[st.idx];
    if (!it || !st.dirty) return true;
    const c = document.createElement('canvas');
    c.width = c.height = S;
    const x = c.getContext('2d'), id = x.createImageData(S, S), d = id.data;
    let any = false;
    for (let i = 0; i < S * S; i++) {
      const v = st.mask[i] ? 255 : 0;
      if (v) any = true;
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
    }
    x.putImageData(id, 0, 0);
    const blob = await new Promise(r => c.toBlob(r, 'image/png'));
    const fd = new FormData();
    fd.append('file', blob, it.name + '.png');
    try {
      await api(`/api/datasets/${enc(st.ds)}/mask/${enc(it.name)}`, { method: 'PUT', body: fd });
      st.dirty = false;
      $('#an-dirty').textContent = 'saved';
      it.annotated = any;
      const th = $(`.thumb[data-i="${st.idx}"] .dot`);
      if (th) th.classList.toggle('on', any);
      updateCount();
      return true;
    } catch (e) {
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
    $$('.thumb.sel').forEach(t => t.classList.remove('sel'));
    const th = $(`.thumb[data-i="${i}"]`);
    if (th) { th.classList.add('sel'); th.scrollIntoView({ block: 'nearest' }); }
    $('#an-name').textContent = it.name;
    $('#an-pos').textContent = `${i + 1} / ${st.items.length}`;
    $('#an-dirty').textContent = '';
    const base = `/api/datasets/${enc(st.ds)}`;
    try {
      const [img, mask] = await Promise.all([
        loadImg(`${base}/image/${enc(it.name)}?t=${Date.now()}`),
        loadImg(`${base}/mask/${enc(it.name)}?t=${Date.now()}`),
      ]);
      if (token !== st.loadToken) return;
      imgX.clearRect(0, 0, S, S);
      imgX.drawImage(img, 0, 0, S, S);
      const c = document.createElement('canvas');
      c.width = c.height = S;
      const x = c.getContext('2d');
      x.drawImage(mask, 0, 0, S, S);
      const d = x.getImageData(0, 0, S, S).data;
      const m = new Uint8Array(S * S);
      for (let k = 0; k < S * S; k++) m[k] = d[k * 4] > 127 ? 1 : 0;
      st.mask = m;
      st.undo = []; st.redo = []; st.poly = []; st.dirty = false;
      $('#stage-empty').classList.add('hidden');
      box.classList.remove('hidden');
      render();
    } catch (e) {
      toast('Could not load ' + it.name, true);
    }
  }

  function updateCount() {
    const a = st.items.filter(i => i.annotated).length;
    $('#an-count').textContent = `${a} / ${st.items.length} annotated`;
  }

  function renderList() {
    const el = $('#an-list');
    el.innerHTML = '';
    const onlyEmpty = $('#an-only-empty').checked;
    st.items.forEach((it, i) => {
      if (onlyEmpty && it.annotated && i !== st.idx) return;
      const d = document.createElement('div');
      d.className = 'thumb' + (i === st.idx ? ' sel' : '');
      d.dataset.i = i;
      d.innerHTML = `<img loading="lazy"><span class="dot${it.annotated ? ' on' : ''}"></span><span class="nm"></span>`;
      $('img', d).src = `/api/datasets/${enc(st.ds)}/image/${enc(it.name)}?size=128`;
      $('.nm', d).textContent = it.name;
      d.title = it.name;
      d.addEventListener('click', () => go(i));
      el.appendChild(d);
    });
    updateCount();
  }
  $('#an-only-empty').addEventListener('change', renderList);

  function showEmpty(msg) {
    $('#stage-empty').textContent = msg;
    $('#stage-empty').classList.remove('hidden');
    box.classList.add('hidden');
    $('#an-name').textContent = '';
    $('#an-pos').textContent = '';
  }

  async function openDataset(name) {
    if (st.dirty) await save();
    st.ds = name;
    st.idx = -1;
    $('#an-dataset').value = name;
    if (!name) { st.items = []; renderList(); showEmpty('Import a dataset in the Data tab.'); return; }
    st.items = await api(`/api/datasets/${enc(name)}/items`);
    renderList();
    fit();
    if (st.items.length) go(0); else showEmpty('This dataset is empty.');
  }

  $('#an-dataset').addEventListener('change', e => openDataset(e.target.value));
  bus.on('datasets', list => {
    fillSelect($('#an-dataset'), list.map(d => d.name), st.ds, 'No dataset');
  });
  bus.on('open-dataset', name => openDataset(name));
  bus.on('tab', async t => {
    if (t !== 'annotate') return;
    await refreshDatasets();
    if (!st.ds || !DATASETS.some(d => d.name === st.ds)) openDataset(DATASETS[0]?.name || '');
    requestAnimationFrame(fit);
  });
  bus.on('tab-leave', t => { if (t === 'annotate' && st.dirty) save(); });
  bus.on('quit', () => { if (st.dirty) window.pendingSave = save(); });
  window.addEventListener('beforeunload', e => { if (st.dirty) { e.preventDefault(); e.returnValue = ''; } });

  // add images to the open dataset
  setupDrop($('#an-drop'), async files => {
    if (!st.ds) { toast('Choose a dataset first', true); return; }
    const { imgs, masks } = classify(files);
    if (!imgs.length) return;
    const prog = $('#import-progress');
    const r = await uploadToDataset(st.ds, imgs, masks, prog);
    toast(`${r.images} images in "${st.ds}"` + (r.errors.length ? ` — ${r.errors.length} error(s)` : ''), r.errors.length > 0);
    const cur = st.items[st.idx]?.name;
    st.items = await api(`/api/datasets/${enc(st.ds)}/items`);
    st.idx = -1;
    renderList();
    const j = Math.max(0, st.items.findIndex(i => i.name === cur));
    go(j);
  });

  setTool('brush');
  setSize(20);
})();
