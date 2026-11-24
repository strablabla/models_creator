// View3D: the labels of a volume as 3D surfaces (one color per class), with
// Three.js (static/vendor/three, loaded through the import map of index.html).
// Drag: rotate · right drag or Shift: pan · wheel: zoom · double-click: fit.
// The current slice of the 2D view is drawn as a translucent plane.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/three/OrbitControls.js';

function b64(s, Type) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}

class View3D {
  constructor(stage) {
    this.stage = stage;
    this.el = document.createElement('div');
    this.el.className = 'v3d hidden';
    this.el.innerHTML = '<div class="v3d-msg"></div><div class="v3d-help">drag: rotate · right drag: pan · wheel: zoom · double-click: fit</div>';
    stage.appendChild(this.el);
    this.msg = this.el.querySelector('.v3d-msg');
    // the 2D view under it must not paint, pan or change slice
    for (const ev of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'contextmenu', 'dblclick']) {
      this.el.addEventListener(ev, e => { e.stopPropagation(); if (ev === 'contextmenu') e.preventDefault(); });
    }
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.el.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1e6);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(0.4, 0.6, 1);
    this.camera.add(key);  // the light follows the camera
    this.scene.add(this.camera);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.addEventListener('change', () => this.render());
    this.renderer.domElement.addEventListener('dblclick', () => this.fit());
    // image axes: x to the right, y down (as in the XY slices), z towards the viewer
    this.group = new THREE.Group();
    this.group.scale.set(1, -1, -1);
    this.scene.add(this.group);
    this.meshes = [];
    this.opacity = 0.6;
    new ResizeObserver(() => this.resize()).observe(stage);
  }

  get visible() { return !this.el.classList.contains('hidden'); }

  showMessage(t) { this.msg.textContent = t || ''; this.msg.classList.toggle('hidden', !t); }

  open() { this.el.classList.remove('hidden'); this.resize(); }
  close() { this.el.classList.add('hidden'); }

  resize() {
    const r = this.stage.getBoundingClientRect();
    if (!r.width || !r.height || !this.visible) return;
    this.renderer.setSize(r.width, r.height);
    this.camera.aspect = r.width / r.height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  clear() {
    for (const o of [...this.group.children]) {
      this.group.remove(o);
      o.geometry?.dispose();
      o.material?.dispose();
    }
    this.meshes = [];
    this.plane = null;
  }

  // data: answer of /mesh/<item>; classes: [{name, color}]
  setData(data, classes, keepView = false) {
    this.clear();
    const [Z, Y, X] = data.shape, sp = data.spacing;
    this.size = [X * sp[2], Y * sp[1], Z * sp[0]];
    this.spacing = sp;
    this.shape = data.shape;
    for (const m of data.meshes) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(b64(m.vertices, Float32Array), 3));
      g.setIndex(new THREE.BufferAttribute(b64(m.faces, Uint32Array), 1));
      g.computeVertexNormals();
      const color = classes[m.label - 1]?.color || '#ffffff';
      const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(g, mat);
      mesh.userData.label = m.label;
      mesh.renderOrder = m.label;  // inner classes (higher numbers, often) drawn last
      this.group.add(mesh);
      this.meshes.push(mesh);
    }
    // the volume's box
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(...this.size)),
      new THREE.LineBasicMaterial({ color: 0x8b91a1, transparent: true, opacity: 0.6 }));
    box.position.set(this.size[0] / 2, this.size[1] / 2, this.size[2] / 2);
    this.group.add(box);
    this.setOpacity(this.opacity);
    if (!keepView) this.fit();
    if (this.planeArgs) this.setPlane(...this.planeArgs);
    this.showMessage(data.meshes.length ? '' : 'No label in this volume.');
    this.render();
  }

  setClasses(classes) {
    for (const m of this.meshes) m.material.color.set(classes[m.userData.label - 1]?.color || '#ffffff');
    this.render();
  }

  setOpacity(a) {
    this.opacity = a;
    const solid = a >= 0.99;
    for (const m of this.meshes) {
      m.material.transparent = !solid;
      m.material.opacity = Math.max(0.08, a);
      m.material.depthWrite = solid;
      m.material.needsUpdate = true;
    }
    this.render();
  }

  // a translucent plane at slice k of axis 'z' | 'y' | 'x' (null: none)
  setPlane(axis, k) {
    this.planeArgs = axis ? [axis, k] : null;
    if (this.plane) { this.group.remove(this.plane); this.plane.geometry.dispose(); this.plane.material.dispose(); this.plane = null; }
    if (!axis || !this.size) return this.render();
    const [W, H, D] = this.size, sp = this.spacing;
    const dims = { z: [W, H], y: [W, D], x: [D, H] }[axis];
    const p = new THREE.Mesh(new THREE.PlaneGeometry(...dims),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
    if (axis === 'z') p.position.set(W / 2, H / 2, (k + 0.5) * sp[0]);
    if (axis === 'y') { p.rotation.x = Math.PI / 2; p.position.set(W / 2, (k + 0.5) * sp[1], D / 2); }
    if (axis === 'x') { p.rotation.y = Math.PI / 2; p.position.set((k + 0.5) * sp[2], H / 2, D / 2); }
    p.renderOrder = 1000;
    this.plane = p;
    this.group.add(p);
    this.render();
  }

  fit() {
    if (!this.size) return;
    const [W, H, D] = this.size;
    const center = new THREE.Vector3(W / 2, -H / 2, -D / 2);  // group scale (1, -1, -1)
    const radius = Math.hypot(W, H, D) / 2;
    const dist = radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.05;
    // seen from above the first slice, a little tilted: XY as in the slices
    const dir = new THREE.Vector3(0.35, -0.45, 1).normalize();
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.camera.up.set(0, 1, 0);
    this.camera.near = dist / 100;
    this.camera.far = dist * 10;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
    this.render();
  }

  render() {
    if (!this.visible) return;
    if (this._q) return;
    this._q = true;
    requestAnimationFrame(() => { this._q = false; this.renderer.render(this.scene, this.camera); });
  }
}

window.View3D = View3D;
window.dispatchEvent(new Event('view3d-ready'));
