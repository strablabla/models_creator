// VolumeView: one slice of a (Z, Y, X) uint8 volume with its labels drawn over
// it, in the XY, XZ or YZ plane, with the voxel spacing as aspect ratio. Zoom
// (Ctrl + wheel), pan (Space, right or middle button), slices (wheel). When
// editable: brush, eraser and fill on the slice, with undo / redo.
// Labels: Uint8Array, 0 = background, k = class k (colors from setClasses).
class VolumeView {
  constructor(stage, opts = {}) {
    this.stage = stage;
    this.editable = !!opts.editable;
    this.onChange = opts.onChange || (() => {});   // labels edited
    this.onSlice = opts.onSlice || (() => {});     // (index, count, axis)
    this.onStroke = opts.onStroke || (() => {});   // end of an edit
    stage.innerHTML = '<div class="vbox"><canvas></canvas></div><div class="vcursor"></div>' +
      '<div class="vinfo hidden"></div><div class="vempty"></div>';
    this.box = $('.vbox', stage);
    this.canvas = $('canvas', stage);
    this.ctx = this.canvas.getContext('2d');
    this.cursor = $('.vcursor', stage);
    this.info = $('.vinfo', stage);
    this.emptyEl = $('.vempty', stage);
    this.img = null; this.lab = null; this.shape = [0, 0, 0]; this.spacing = [1, 1, 1];
    this.axis = 'z'; this.pos = { z: 0, y: 0, x: 0 };
    this.classes = []; this.lut = this._lut([]);
    this.opacity = 0.5; this.showLabels = true; this.outline = false;
    this.tool = 'brush'; this.size = 12; this.active = 1; this.protect = false;
    this.view = { scale: 1, tx: 0, ty: 0 };
    this.undoStack = []; this.redoStack = [];
    this.spaceDown = false; this.mouse = null; this.drawing = false; this.panning = null; this.wheelAcc = 0;
    if (!this.editable) stage.classList.add('view');
    this._events();
    new ResizeObserver(() => { if (this.img && stage.clientWidth) this.fit(); }).observe(stage);
    this.empty(opts.emptyText || '');
  }

  // ---------- data ----------
  setVolume(img, lab, shape, spacing) {
    this.img = img; this.lab = lab; this.shape = shape;
    this.spacing = (spacing || [1, 1, 1]).map(s => (s > 0 ? s : 1));
    const [Z, Y, X] = shape;
    this.pos = { z: Z >> 1, y: Y >> 1, x: X >> 1 };
    this.undoStack = []; this.redoStack = [];
    this.emptyEl.classList.add('hidden');
    this.box.classList.remove('hidden');
    this.info.classList.remove('hidden');
    this.fit();
    this.render();
    this._slice();
  }

  empty(text) {
    this.img = this.lab = null;
    this.emptyEl.textContent = text;
    this.emptyEl.classList.toggle('hidden', !text);
    this.box.classList.add('hidden');
    this.info.classList.add('hidden');
    this.cursor.style.display = 'none';
  }

  setLabels(lab) { this.lab = lab; this.render(); }

  setClasses(classes) { this.classes = classes; this.lut = this._lut(classes); this.render(); }

  _lut(classes) {
    const lut = [];
    for (let k = 0; k < 256; k++) {
      const c = classes[k - 1];
      lut.push(c ? hexToRgb(c.color) : (k ? [255, 255, 255] : [0, 0, 0]));
    }
    return lut;
  }

  // ---------- geometry ----------
  // The slice shown: W x H pixels, voxel of (u, v) = base + u * du + v * dv.
  plane() {
    const [Z, Y, X] = this.shape, sp = this.spacing, k = this.pos[this.axis];
    const clamp = a => Math.min(20, Math.max(0.05, a));
    if (this.axis === 'z') return { W: X, H: Y, base: k * Y * X, du: 1, dv: X, asp: clamp(sp[1] / sp[2]), n: Z, k };
    if (this.axis === 'y') return { W: X, H: Z, base: k * X, du: 1, dv: Y * X, asp: clamp(sp[0] / sp[2]), n: Y, k };
    return { W: Y, H: Z, base: k, du: X, dv: Y * X, asp: clamp(sp[0] / sp[1]), n: X, k };
  }

  voxel(u, v) {  // (x, y, z) of a pixel of the slice
    const k = this.pos[this.axis];
    if (this.axis === 'z') return [u, v, k];
    if (this.axis === 'y') return [u, k, v];
    return [k, u, v];
  }

  fit() {
    if (!this.img) return;
    const p = this.plane(), r = this.stage.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const s = Math.min(r.width / p.W, r.height / (p.H * p.asp)) * 0.94;
    this.view = { scale: s, tx: (r.width - p.W * s) / 2, ty: (r.height - p.H * p.asp * s) / 2 };
    this.applyView();
  }

  applyView() {
    const v = this.view;
    this.box.style.transform = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`;
    this.canvas.style.imageRendering = v.scale >= 2 ? 'pixelated' : 'auto';
    this._cursor();
  }

  toPlane(e) {
    const r = this.stage.getBoundingClientRect(), p = this.plane();
    return [(e.clientX - r.left - this.view.tx) / this.view.scale,
            (e.clientY - r.top - this.view.ty) / (this.view.scale * p.asp)];
  }

  // ---------- slices ----------
  count() { return this.img ? this.plane().n : 0; }
  slice() { return this.pos[this.axis]; }

  setSlice(k) {
    if (!this.img) return;
    const n = this.plane().n;
    k = Math.max(0, Math.min(n - 1, Math.round(k)));
    if (k === this.pos[this.axis]) return;
    this.pos[this.axis] = k;
    this.render();
    this._slice();
  }

  setAxis(a) {
    if (a === this.axis) return;
    this.axis = a;
    if (!this.img) return;
    this.fit();
    this.render();
    this._slice();
  }

  _slice() { this.onSlice(this.pos[this.axis], this.plane().n, this.axis); }

  // which slices of the current axis hold labels: Uint8Array of 0/1
  labeledSlices() {
    if (!this.lab) return new Uint8Array(0);
    const [Z, Y, X] = this.shape, lab = this.lab, out = new Uint8Array(this.plane().n);
    if (this.axis === 'z') {
      for (let z = 0; z < Z; z++) {
        const a = z * Y * X, b = a + Y * X;
        for (let i = a; i < b; i++) if (lab[i]) { out[z] = 1; break; }
      }
    } else {
      for (let i = 0; i < lab.length; i++) {
        if (!lab[i]) continue;
        const x = i % X, y = ((i / X) | 0) % Y;
        out[this.axis === 'y' ? y : x] = 1;
      }
    }
    return out;
  }

  drawStrip(canvas) {
    const n = this.count();
    if (!n) { canvas.width = 1; canvas.getContext('2d').clearRect(0, 0, 1, 1); return; }
    const has = this.labeledSlices();
    canvas.width = n; canvas.height = 1;
    const x = canvas.getContext('2d'), css = getComputedStyle(document.documentElement);
    x.clearRect(0, 0, n, 1);
    x.fillStyle = css.getPropertyValue('--ok');
    for (let i = 0; i < n; i++) if (has[i]) x.fillRect(i, 0, 1, 1);
    x.fillStyle = css.getPropertyValue('--text');
    x.fillRect(this.slice(), 0, Math.max(1, n / 300), 1);
  }

  // ---------- rendering ----------
  render() {
    if (!this.img || this._queued) return;
    this._queued = true;
    requestAnimationFrame(() => { this._queued = false; this._draw(); });
  }

  _draw() {
    if (!this.img) return;
    const p = this.plane(), { W, H, base, du, dv } = p;
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    this.box.style.width = W + 'px';
    this.box.style.height = H * p.asp + 'px';
    const id = this.ctx.createImageData(W, H), d = id.data;
    const img = this.img, lab = this.lab, lut = this.lut;
    const a = this.showLabels && lab ? this.opacity : 0, outline = this.outline;
    for (let v = 0, o = 0; v < H; v++) {
      const row = base + v * dv;
      for (let u = 0; u < W; u++, o += 4) {
        const i = row + u * du, g = img[i], l = a ? lab[i] : 0;
        let show = l !== 0;
        if (show && outline) {
          show = u === 0 || v === 0 || u === W - 1 || v === H - 1 ||
            lab[i - du] !== l || lab[i + du] !== l || lab[i - dv] !== l || lab[i + dv] !== l;
        }
        if (show) {
          const c = lut[l], al = outline ? Math.max(a, 0.9) : a;
          d[o] = g + (c[0] - g) * al; d[o + 1] = g + (c[1] - g) * al; d[o + 2] = g + (c[2] - g) * al;
        } else {
          d[o] = d[o + 1] = d[o + 2] = g;
        }
        d[o + 3] = 255;
      }
    }
    this.ctx.putImageData(id, 0, 0);
    this._info();
  }

  _info() {
    if (!this.img) return;
    const p = this.plane(), names = { z: 'Z', y: 'Y', x: 'X' };
    let t = `${{ z: 'XY', y: 'XZ', x: 'YZ' }[this.axis]} · ${names[this.axis]} ${p.k + 1} / ${p.n}`;
    if (this.mouse) {
      const u = Math.floor(this.mouse[0]), v = Math.floor(this.mouse[1]);
      if (u >= 0 && v >= 0 && u < p.W && v < p.H) {
        const [x, y, z] = this.voxel(u, v), l = this.lab ? this.lab[p.base + u * p.du + v * p.dv] : 0;
        t += ` · x ${x} y ${y} z ${z}` + (l ? ` · ${this.classes[l - 1]?.name || 'class ' + l}` : '');
      }
    }
    this.info.textContent = t;
  }

  _cursor() {
    const t = this.effectiveTool();
    const show = this.editable && this.img && this.mouse && (t === 'brush' || t === 'eraser');
    this.cursor.style.display = show ? 'block' : 'none';
    if (!show) return;
    const p = this.plane(), d = this.size * this.view.scale;
    this.cursor.style.width = this.cursor.style.height = d + 'px';
    this.cursor.style.left = this.mouse[0] * this.view.scale + this.view.tx + 'px';
    this.cursor.style.top = this.mouse[1] * this.view.scale * p.asp + this.view.ty + 'px';
    this.cursor.style.borderColor = t === 'eraser' ? '#ffd23f' : '#fff';
  }

  effectiveTool() { return this.spaceDown ? 'pan' : (this.editable ? this.tool : 'pan'); }

  setTool(t) { this.tool = t; this._stageClass(); this._cursor(); }

  _stageClass() {
    const t = this.effectiveTool();
    this.stage.classList.toggle('pan', t === 'pan');
    this.stage.classList.toggle('fillc', t === 'fill');
  }

  setSpace(down) { this.spaceDown = down; this._stageClass(); this._cursor(); }

  // ---------- editing ----------
  _snapshot() {
    const p = this.plane(), out = new Uint8Array(p.W * p.H);
    for (let v = 0, j = 0; v < p.H; v++) for (let u = 0; u < p.W; u++, j++) out[j] = this.lab[p.base + u * p.du + v * p.dv];
    return { axis: this.axis, k: p.k, data: out };
  }

  // writes a snapshot back (wherever the view is), returns what it replaced
  _restore(s) {
    const axis = this.axis, k = this.pos[s.axis];
    this.axis = s.axis; this.pos[s.axis] = s.k;
    const now = this._snapshot(), p = this.plane();
    for (let v = 0, j = 0; v < p.H; v++) for (let u = 0; u < p.W; u++, j++) this.lab[p.base + u * p.du + v * p.dv] = s.data[j];
    this.axis = axis; this.pos[s.axis] = k;
    return now;
  }

  pushUndo() {
    this.undoStack.push(this._snapshot());
    if (this.undoStack.length > 80) this.undoStack.shift();
    this.redoStack = [];
  }

  undo() {
    if (!this.undoStack.length) return;
    const s = this.undoStack.pop();
    this.redoStack.push(this._restore(s));
    this._edited(true);
    if (s.axis === this.axis && s.k !== this.slice()) this.setSlice(s.k);
  }

  redo() {
    if (!this.redoStack.length) return;
    const s = this.redoStack.pop();
    this.undoStack.push(this._restore(s));
    this._edited(true);
    if (s.axis === this.axis && s.k !== this.slice()) this.setSlice(s.k);
  }

  _edited(end) { this.render(); this.onChange(); if (end) this.onStroke(); }

  _put(i, val) {
    const lab = this.lab, cur = lab[i];
    if (val === 0) { if (!this.protect || cur === this.active) lab[i] = 0; }
    else if (!this.protect || cur === 0 || cur === val) lab[i] = val;
  }

  stamp(cu, cv, val) {
    const p = this.plane(), r = this.size / 2, r2 = r * r;
    const rv = r / p.asp;
    const v0 = Math.max(0, Math.floor(cv - rv)), v1 = Math.min(p.H - 1, Math.ceil(cv + rv));
    for (let v = v0; v <= v1; v++) {
      const dy = (v + 0.5 - cv) * p.asp, h = r2 - dy * dy;
      if (h < 0) continue;
      const w = Math.sqrt(h);
      const u0 = Math.max(0, Math.ceil(cu - w - 0.5)), u1 = Math.min(p.W - 1, Math.floor(cu + w - 0.5));
      for (let u = u0; u <= u1; u++) this._put(p.base + u * p.du + v * p.dv, val);
    }
    if (this.size <= 1.5) {
      const u = Math.floor(cu), v = Math.floor(cv);
      if (u >= 0 && v >= 0 && u < p.W && v < p.H) this._put(p.base + u * p.du + v * p.dv, val);
    }
  }

  line(a, b, val) {
    const p = this.plane();
    const dist = Math.hypot(b[0] - a[0], (b[1] - a[1]) * p.asp);
    const n = Math.max(1, Math.ceil(dist / Math.max(0.5, this.size / 4)));
    for (let i = 1; i <= n; i++) this.stamp(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, val);
  }

  fill(pu, pv, val) {
    const p = this.plane(), u0 = Math.floor(pu), v0 = Math.floor(pv);
    if (u0 < 0 || v0 < 0 || u0 >= p.W || v0 >= p.H) return;
    const { W, H, base, du, dv } = p, lab = this.lab;
    const at = j => base + (j % W) * du + ((j / W) | 0) * dv;
    const from = lab[at(v0 * W + u0)];
    if (from === val) return;
    this.pushUndo();
    const seen = new Uint8Array(W * H), stack = [v0 * W + u0];
    seen[v0 * W + u0] = 1;
    while (stack.length) {
      const j = stack.pop(), u = j % W;
      this._put(at(j), val);
      const nb = [u > 0 ? j - 1 : -1, u < W - 1 ? j + 1 : -1, j >= W ? j - W : -1, j < W * (H - 1) ? j + W : -1];
      for (const q of nb) if (q >= 0 && !seen[q] && lab[at(q)] === from) { seen[q] = 1; stack.push(q); }
    }
    this._edited(true);
  }

  copySlice(offset) {
    const p = this.plane(), src = p.k + offset;
    if (!this.lab || src < 0 || src >= p.n) return false;
    this.pushUndo();
    const shift = { z: this.shape[1] * this.shape[2], y: this.shape[2], x: 1 }[this.axis] * offset;
    for (let v = 0; v < p.H; v++) for (let u = 0; u < p.W; u++) {
      const i = p.base + u * p.du + v * p.dv;
      this.lab[i] = this.lab[i + shift];
    }
    this._edited(true);
    return true;
  }

  clearSlice() {
    if (!this.lab) return;
    this.pushUndo();
    const p = this.plane();
    for (let v = 0; v < p.H; v++) for (let u = 0; u < p.W; u++) this.lab[p.base + u * p.du + v * p.dv] = 0;
    this._edited(true);
  }

  // ---------- pointer ----------
  _events() {
    const st = this.stage;
    st.addEventListener('contextmenu', e => e.preventDefault());
    st.addEventListener('pointerdown', e => {
      if (!this.img) return;
      st.setPointerCapture(e.pointerId);
      const p = this.toPlane(e), tool = this.effectiveTool();
      if (tool === 'pan' || e.button === 1 || e.button === 2) {
        this.panning = { x: e.clientX, y: e.clientY, tx: this.view.tx, ty: this.view.ty };
        st.classList.add('panning');
        return;
      }
      if (e.button !== 0 || !this.lab) return;
      if (tool === 'brush' || tool === 'eraser') {
        this.pushUndo();
        this.drawing = true;
        this.drawValue = tool === 'eraser' || e.shiftKey ? 0 : this.active;
        this.last = p;
        this.stamp(p[0], p[1], this.drawValue);
        this._edited(false);
      } else if (tool === 'fill') {
        this.fill(p[0], p[1], e.shiftKey ? 0 : this.active);
      }
    });
    st.addEventListener('pointermove', e => {
      if (!this.img) return;
      const p = this.toPlane(e);
      this.mouse = p;
      if (this.panning) {
        this.view.tx = this.panning.tx + e.clientX - this.panning.x;
        this.view.ty = this.panning.ty + e.clientY - this.panning.y;
        this.applyView();
        return;
      }
      if (this.drawing) {
        this.line(this.last, p, this.drawValue);
        this.last = p;
        this._edited(false);
      } else {
        this._info();
      }
      this._cursor();
    });
    const end = () => {
      if (this.panning) { this.panning = null; st.classList.remove('panning'); }
      if (this.drawing) { this.drawing = false; this.onStroke(); }
    };
    st.addEventListener('pointerup', end);
    st.addEventListener('pointercancel', end);
    st.addEventListener('pointerleave', () => { this.mouse = null; this._cursor(); this._info(); });
    st.addEventListener('wheel', e => {
      if (!this.img) return;
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const r = st.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
        const ns = Math.min(60, Math.max(0.05, this.view.scale * Math.exp(-e.deltaY * 0.0015)));
        this.view.tx = mx - (mx - this.view.tx) * ns / this.view.scale;
        this.view.ty = my - (my - this.view.ty) * ns / this.view.scale;
        this.view.scale = ns;
        this.applyView();
        return;
      }
      // a wheel notch = one slice; a touchpad moves by smaller steps
      const step = e.shiftKey ? 10 : 1;
      if (Math.abs(e.deltaY) >= 40) { this.setSlice(this.slice() + Math.sign(e.deltaY) * step); return; }
      this.wheelAcc += e.deltaY;
      if (Math.abs(this.wheelAcc) >= 60) { this.setSlice(this.slice() + Math.sign(this.wheelAcc) * step); this.wheelAcc = 0; }
    }, { passive: false });
  }
}
