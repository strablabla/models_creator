// Info tab: deep-learning framework and versions, GPUs (name, VRAM, load...), system.
(() => {
  let timer = null;

  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const gb = mb => mb == null ? '–' : mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB';
  const unit = (v, u, d = 0) => v == null ? '–' : `${Number(v).toFixed(d)} ${u}`;
  const rows = pairs => pairs.filter(Boolean).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

  function meter(label, used, total, text, hotAbove = 0.9) {
    const r = total ? Math.min(1, used / total) : 0;
    return `<div class="meter"><div class="lab"><span>${label}</span><b>${text}</b></div>
      <div class="track"><div class="fill${r > hotAbove ? ' hot' : ''}" style="width:${(r * 100).toFixed(1)}%"></div></div></div>`;
  }

  function fwRows(f) {
    if (!f.installed) return `<div class="note">${esc(f.why)}</div>`;
    if (f.id === 'tensorflow' && !f.loaded) {
      return `<table class="kv">${rows([['Version', esc(f.version)],
        ['Details', '<span class="na">loaded when TensorFlow is selected in Settings or used by a model</span>']])}</table>`;
    }
    const devs = (f.devices || []).map(d =>
      `<tr><td>cuda:${d.index}</td><td>${esc(d.name)} · compute ${d.capability} · ${gb(d.memory_total_mb)}</td></tr>`).join('');
    return `${f.why ? `<div class="note">${esc(f.why)}</div>` : ''}
      <table class="kv">${rows([
        ['Version', esc(f.version)],
        ['Build', esc(f.build)],
        ['Device used', esc(f.device_used)],
        f.id === 'pytorch' ? ['cuDNN', f.cudnn ? esc(f.cudnn) : '<span class="na">–</span>'] : ['Keras', esc(f.keras)],
        f.arch_list?.length ? ['GPU architectures', esc(f.arch_list.join(', '))] : null,
        f.gpus ? ['GPUs seen', f.gpus.length ? esc(f.gpus.join(', ')) : '<span class="na">none</span>'] : null,
        f.threads ? ['CPU threads', esc(f.threads)] : null,
      ])}${devs}</table>`;
  }

  function renderFramework(fws, driverCuda) {
    const sel = fws.list.find(f => f.id === fws.selected) || fws.list[0];
    const other = fws.list.filter(f => f !== sel);
    $('#info-framework').innerHTML = `
      <h2>Framework</h2>
      <div class="big">${esc(sel.name)} <small>${esc(sel.version || '')}</small></div>
      <div style="margin:6px 0 10px">
        <span class="pill ok">selected in Settings</span>
        ${sel.build ? `<span class="pill">${esc(sel.build)}</span>` : ''}
        ${sel.installed ? `<span class="pill ${sel.cuda_available ? 'ok' : 'off'}">${sel.cuda_available ? 'GPU in use' : 'Running on CPU'}</span>` : ''}
      </div>
      ${driverCuda && !sel.cuda_available ? `<div class="muted small">The NVIDIA driver supports CUDA up to ${esc(driverCuda)}.</div>` : ''}
      ${fwRows(sel)}
      ${other.map(f => `<h3 style="margin-top:14px">${esc(f.name)}</h3>${fwRows(f)}`).join('')}`;
  }

  function renderSystem(s) {
    $('#info-system').innerHTML = `
      <h2>System</h2>
      <table class="kv">${rows([
        ['Operating system', esc(s.os)],
        ['Processor', esc(s.cpu) + ` <span class="muted">(${s.cpu_cores} logical cores)</span>`],
        ['Python', esc(s.python) + `<br><span class="muted small">${esc(s.python_path)}</span>`],
      ])}</table>
      ${meter('RAM', s.ram_total_mb - s.ram_available_mb, s.ram_total_mb,
        `${gb(s.ram_total_mb - s.ram_available_mb)} used / ${gb(s.ram_total_mb)}`)}`;
  }

  function gpuCard(g, driverCuda) {
    if (g.vendor !== 'NVIDIA') {
      return `<div class="panel"><h2>${esc(g.name)}</h2><table class="kv">${rows([
        ['Driver', esc(g.driver)],
        ['Memory', g.memory_total_mb ? gb(g.memory_total_mb) + ' <span class="muted">(as reported by Windows, at most 4 GB)</span>' : '–'],
        ['Usable by PyTorch', '<span class="na">no (not an NVIDIA GPU)</span>'],
      ])}</table></div>`;
    }
    const used = g.memory_used_mb, total = g.memory_total_mb;
    return `<div class="panel" data-gpu="${g.index}">
      <h2>${esc(g.name)} <span class="pill">GPU ${g.index}</span></h2>
      ${meter('VRAM', used, total, `${gb(used)} used / ${gb(total)}`)}
      ${meter('GPU load', g.utilization_pct, 100, unit(g.utilization_pct, '%'), 2)}
      <table class="kv">${rows([
        ['Total VRAM', gb(total)],
        ['Driver', esc(g.driver) + (driverCuda ? ` <span class="muted">(CUDA up to ${esc(driverCuda)})</span>` : '')],
        ['Temperature', unit(g.temperature_c, '°C')],
        ['Power', `${unit(g.power_w, 'W', 1)} / ${unit(g.power_limit_w, 'W', 0)}`],
        ['Clock', `${unit(g.clock_mhz, 'MHz')} / ${unit(g.clock_max_mhz, 'MHz')} max`],
        ['Fan', unit(g.fan_pct, '%')],
        ['Performance state', esc(g.pstate)],
        ['PCI bus', esc(g.pci_bus)],
      ])}</table></div>`;
  }

  let last = null;
  function renderGpus(gpus) {
    const el = $('#info-gpus');
    el.innerHTML = gpus.length ? gpus.map(g => gpuCard(g, last?.driver_cuda)).join('')
      : '<div class="empty">No GPU found (nvidia-smi is not available).</div>';
  }

  function renderLibs(libs) {
    $('#info-libs').innerHTML = libs.map(l =>
      `<tr><td>${esc(l.name)}</td><td>${l.version ? esc(l.version) : '<span class="na">not installed</span>'}</td></tr>`).join('');
  }

  async function load() {
    try {
      last = await api('/api/system');
      renderFramework(last.frameworks, last.driver_cuda);
      renderSystem(last.system);
      renderGpus(last.gpus);
      renderLibs(last.libraries);
    } catch (e) {
      toast('Could not read system information: ' + e.message, true);
    }
  }

  // GPU load, VRAM and RAM change: refresh them while the tab is open
  async function live() {
    clearTimeout(timer);
    if (currentTab !== 'info') return;
    try {
      const l = await api('/api/system/live');
      if (last) {
        const others = last.gpus.filter(g => g.vendor !== 'NVIDIA');
        last.gpus = [...l.gpus, ...others];
        last.system = l.system;
        renderSystem(l.system);
        renderGpus(last.gpus);
      }
    } catch (e) { /* server busy or restarting: try again */ }
    timer = setTimeout(live, 2000);
  }

  bus.on('settings', () => { last = null; }); // framework may have changed: reload on next visit
  bus.on('tab', async t => {
    if (t !== 'info') return;
    if (!last) await load();
    timer = setTimeout(live, 2000);
  });
})();
