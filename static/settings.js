// Settings tab: theme, framework for new trainings, device and CPU threads.
// Settings live on the server (data/settings.json); the theme is also kept in
// localStorage so the page opens in the right colors without a flash.
let SETTINGS = null;

(() => {
  const media = matchMedia('(prefers-color-scheme: light)');

  function applyTheme(theme) {
    const t = theme === 'system' ? (media.matches ? 'light' : 'dark') : theme;
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem('theme', theme); } catch (e) {}
    bus.emit('theme', t); // charts redraw with the new colors
  }
  media.addEventListener('change', () => { if (SETTINGS?.theme === 'system') applyTheme('system'); });

  function seg(id, value) {
    $$(`#${id} button`).forEach(b => b.classList.toggle('active', b.dataset.v === value));
  }

  async function save(changes) {
    try {
      SETTINGS = await api('/api/settings', { method: 'PUT', json: changes });
      render();
      bus.emit('settings', SETTINGS);
      toast('Saved');
    } catch (e) { toast(e.message, true); }
  }

  let frameworks = null;
  function renderFrameworks() {
    const el = $('#set-frameworks');
    if (!frameworks) { el.innerHTML = '<div class="muted small">Loading…</div>'; return; }
    el.innerHTML = '';
    for (const f of frameworks) {
      const c = document.createElement('label');
      const sel = SETTINGS.framework === f.id;
      c.className = 'fw' + (sel ? ' sel' : '') + (f.installed ? '' : ' disabled');
      let d;
      if (!f.installed) d = 'Not installed.';
      else if (f.id === 'pytorch') d = `${f.build} · runs on ${f.cuda_available ? 'GPU' : 'CPU'}`;
      else d = f.loaded ? `${f.build} · Keras ${f.keras} · runs on ${f.device_used}`
                        : (navigator.userAgent.includes('Windows') ? 'Runs on CPU (no GPU support on native Windows)' : 'Loaded on first use');
      c.innerHTML = `<div class="t"><input type="radio" name="fw" ${sel ? 'checked' : ''} ${f.installed ? '' : 'disabled'}>
        ${f.name} <span class="tag">${f.version || '—'}</span></div>
        <div class="d">${d}</div>
        <div class="d">Saves models as <b>${f.id === 'pytorch' ? '.pt' : '.keras'}</b></div>`;
      if (f.installed) c.addEventListener('click', e => { e.preventDefault(); if (!sel) save({ framework: f.id }); });
      el.appendChild(c);
    }
  }

  function render() {
    if (!SETTINGS) return;
    seg('set-theme', SETTINGS.theme);
    seg('set-device', SETTINGS.device);
    if (document.activeElement !== $('#set-threads')) $('#set-threads').value = SETTINGS.threads;
    renderFrameworks();
  }

  $$('#set-theme button').forEach(b => b.addEventListener('click', () => { applyTheme(b.dataset.v); save({ theme: b.dataset.v }); }));
  $$('#set-device button').forEach(b => b.addEventListener('click', () => save({ device: b.dataset.v })));
  $('#set-threads').addEventListener('change', e => {
    const n = Math.max(0, Math.min(256, Math.round(+e.target.value || 0)));
    save({ threads: n });
  });

  async function loadFrameworks() {
    try {
      const r = await api('/api/system');
      frameworks = r.frameworks.list;
      const cores = r.system.cpu_cores;
      $('#set-threads').max = cores;
      $('#set-threads-help').textContent = `0 = framework default. This computer has ${cores} logical cores.`;
      render();
    } catch (e) { toast(e.message, true); }
  }

  $('#set-restart-note').textContent =
    'TensorFlow reads the device and thread settings when it is first loaded: after changing them, ' +
    'restart the application for TensorFlow to follow. PyTorch follows them at once.';

  bus.on('tab', t => { if (t === 'settings') { render(); loadFrameworks(); } });
  bus.on('settings', s => { if (frameworks) renderFrameworks(); });

  api('/api/settings').then(s => {
    SETTINGS = s;
    applyTheme(s.theme);
    render();
    bus.emit('settings', s);
  }).catch(() => {});
})();
