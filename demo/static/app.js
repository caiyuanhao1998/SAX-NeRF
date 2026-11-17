import {CTMeshViewer} from './mesh.js?v=2';

const $ = id => document.getElementById(id);
const state = {scene: null, meta: null, axis: 1, index: 0, epoch: 0, renderSequence: 0, sliceSequence: 0,
  ready: false, hosted: false, view: {azimuth: 35, elevation: 15}, errors: [], render: null, slice: null};

// Hugging Face forwards a visitor's ZeroGPU quota headers through the same
// MessageChannel handshake used by Gradio. Keep them in memory for one request.
const hfOrigin = 'https://huggingface.co';
let supportsZeroGPUHeaders = false;
window.addEventListener('message', event => {
  if (event.origin === hfOrigin && event.source === window.parent && event.data === 'supports-zerogpu-headers') {
    supportsZeroGPUHeaders = true;
  }
});
if (window.parent !== window) window.parent.postMessage('supports-zerogpu-headers', hfOrigin);

async function zeroGPUHeaders() {
  if (!supportsZeroGPUHeaders) return {};
  return new Promise(resolve => {
    const channel = new MessageChannel();
    const finish = headers => {clearTimeout(timeout); channel.port1.close(); resolve(headers);};
    const timeout = setTimeout(() => finish({}), 1500);
    channel.port1.onmessage = event => {
      const value = event.data;
      finish(value && typeof value === 'object' && !Array.isArray(value)
        ? Object.fromEntries(Object.entries(value).filter(([key, item]) => typeof key === 'string' && typeof item === 'string')) : {});
    };
    window.parent.postMessage('zerogpu-headers', hfOrigin, [channel.port2]);
  });
}

async function refreshUsage() {
  if (!state.hosted) return;
  try {
    const data = await api('/api/usage');
    $('usage-month').textContent = data.available ? Number(data.last_30_days).toLocaleString('en-US') : '—';
    $('usage-today').textContent = data.available ? Number(data.today).toLocaleString('en-US') : '—';
  } catch {} // Statistics must not interrupt reconstruction viewing.
}

function status(message, busy = false) {
  $('status').textContent = message;
  $('status').classList.toggle('busy', busy);
}

function error(message) {
  $('error').hidden = false;
  $('error').querySelector('span').textContent = message;
  state.errors.push(message);
  status('Unable to complete this request');
}

async function api(url, options = {}) {
  if (state.hosted && url === '/api/render') {
    options = {...options, headers: {...(options.headers || {}), ...await zeroGPUHeaders()}};
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    let text = 'Unable to load the selected scene. Check the local terminal and retry.';
    try {text = (await response.json()).detail || text;} catch {}
    throw new Error(text);
  }
  return response.json();
}

function link(id, href, filename) {
  const element = $(id);
  element.href = href;
  if (filename) element.download = filename;
  element.classList.remove('disabled');
  element.setAttribute('aria-disabled', 'false');
}

function cameraChanged(view) {
  state.view = view;
  $('view-angle').textContent = view.azimuth.toFixed(1) + '° / ' + view.elevation.toFixed(1) + '°';
  if (state.ready && state.render && (Math.abs(state.render.azimuth - view.azimuth) > .3 || Math.abs(state.render.elevation - view.elevation) > .3)) {
    $('render-info').textContent = 'View changed · Render to update the projection';
  }
}

let viewer;
try {
  viewer = new CTMeshViewer($('mesh-canvas'), cameraChanged, index => setSlice(index));
} catch (exception) {
  error('The 3D mesh needs WebGL. Enable hardware acceleration in your browser and reload.');
  throw exception;
}

let sliceTimer;
function fitSlice() {
  if (!state.slice) return;
  const box = $('slice-viewport').getBoundingClientRect();
  const width = Math.min(box.width, box.height * state.slice.aspect_ratio);
  $('slice-image').style.width = width + 'px';
  $('slice-image').style.height = width / state.slice.aspect_ratio + 'px';
}
new ResizeObserver(fitSlice).observe($('slice-viewport'));

function setSlice(index, immediate = false) {
  if (!state.meta) return;
  state.index = Math.max(0, Math.min(state.meta.shape[state.axis] - 1, Math.round(index)));
  $('slice-position').value = state.index;
  $('slice-caption').textContent = (state.index + 1) + ' / ' + state.meta.shape[state.axis];
  viewer.updatePlane(state.axis, state.index);
  clearTimeout(sliceTimer);
  if (immediate) return updateSlice();
  sliceTimer = setTimeout(updateSlice, 70);
}

async function updateSlice() {
  if (!state.meta) return;
  const epoch = state.epoch, sequence = ++state.sliceSequence;
  const scene = state.scene, axis = state.axis, index = state.index;
  try {
    const result = await api('/api/slice/' + scene + '?axis=' + axis + '&index=' + index);
    if (epoch !== state.epoch || sequence !== state.sliceSequence) return;
    state.slice = result;
    fitSlice();
    await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        if (epoch === state.epoch && sequence === state.sliceSequence) {
          $('slice-image').src = result.image_url;
          $('slice-image').hidden = false;
          $('slice-loading').hidden = true;
        }
        resolve();
      };
      image.onerror = reject;
      image.src = result.image_url;
    });
    if (epoch === state.epoch) link('slice-download', result.image_url, scene + '-slice-' + axis + '-' + index + '.png');
  } catch (exception) {
    if (epoch === state.epoch) error(exception.message || 'Unable to load the CT slice');
  }
}

async function renderProjection(event) {
  if (!state.ready) return;
  const epoch = state.epoch, sequence = ++state.renderSequence;
  $('render-button').disabled = true;
  $('render-button').textContent = 'Rendering…';
  $('projection-loading').hidden = false;
  $('projection-loading').querySelector('span:last-child').textContent = 'Rendering SAX-NeRF X-ray…';
  const view = event?.type === 'change' && state.render?.scene === state.scene
    ? {azimuth: state.render.azimuth, elevation: state.render.elevation} : viewer.getView();
  const payload = {scene: state.scene, ...view, resolution: Number($('resolution').value),
    exposure: Number($('exposure').value), invert: $('invert').checked, count_use: event?.type === 'click'};
  try {
    const result = await api('/api/render', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    if (epoch !== state.epoch || sequence !== state.renderSequence) return;
    await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        if (epoch === state.epoch && sequence === state.renderSequence) {
          $('projection-image').src = result.image_url;
          $('projection-image').hidden = false;
          $('projection-loading').hidden = true;
        }
        resolve();
      };
      image.onerror = reject;
      image.src = result.image_url;
    });
    if (epoch !== state.epoch || sequence !== state.renderSequence) return;
    state.render = result;
    $('render-info').textContent = result.resolution + ' × ' + result.resolution + ' pixels · '
      + (result.compute_ms / 1000).toFixed(2) + ' s' + (result.cached ? ' · cached' : '');
    cameraChanged(viewer.getView());
    link('projection-download', result.download_url, state.scene + '-projection.npy');
    status('Ready');
    if (payload.count_use) refreshUsage();
  } catch (exception) {
    if (epoch === state.epoch) error(exception.message || 'Unable to render the projection');
  } finally {
    if (epoch === state.epoch && sequence === state.renderSequence) {
      $('render-button').disabled = false;
      $('render-button').textContent = 'Render X-ray projection';
    }
  }
}

async function loadScene(scene) {
  const epoch = ++state.epoch;
  state.scene = scene;
  state.ready = false;
  state.meta = null;
  state.render = null;
  state.slice = null;
  clearTimeout(sliceTimer);
  $('error').hidden = true;
  for (const tile of document.querySelectorAll('.scene-tile')) {
    const selected = tile.dataset.scene === scene;
    tile.classList.toggle('selected', selected);
    tile.setAttribute('aria-pressed', String(selected));
    if (selected) tile.scrollIntoView({block: 'nearest', inline: 'nearest'});
  }
  $('scene-name').textContent = scene[0].toUpperCase() + scene.slice(1);
  $('scene-info').textContent = 'SAX-NeRF reconstruction';
  $('mesh-info').textContent = 'Preparing reconstructed CT…';
  status('Loading ' + $('scene-name').textContent + '…', true);
  for (const id of ['mesh-loading', 'projection-loading', 'slice-loading']) $(id).hidden = false;
  $('mesh-loading').querySelector('span:last-child').textContent = 'Preparing ' + $('scene-name').textContent + ' mesh…';
  $('projection-loading').querySelector('span:last-child').textContent = 'Preparing projection…';
  for (const id of ['projection-image', 'slice-image']) $(id).hidden = true;
  for (const id of ['mesh-download', 'slice-download', 'projection-download']) {
    $(id).removeAttribute('href');
    $(id).classList.add('disabled');
    $(id).setAttribute('aria-disabled', 'true');
  }
  $('render-button').disabled = true;
  $('slice-position').disabled = true;
  try {
    const meta = await api('/api/scene/' + scene, {method: 'POST'});
    if (epoch !== state.epoch) return;
    state.meta = meta;
    $('scene-info').textContent = meta.shape.join(' × ') + ' voxels';
    $('mesh-info').textContent = meta.mesh_vertices.toLocaleString() + ' vertices';
    link('mesh-download', '/api/download/mesh/' + scene, scene + '-sax-nerf.glb');
    const loaded = await viewer.load(meta);
    if (epoch !== state.epoch || !loaded) return;
    viewer.setPlaneVisible($('show-plane').checked);
    $('mesh-loading').hidden = true;
    state.ready = true;
    $('slice-position').max = meta.shape[state.axis] - 1;
    $('slice-position').disabled = false;
    $('render-button').disabled = false;
    await setSlice(Math.floor(meta.shape[state.axis] / 2), true);
    if (epoch === state.epoch) await renderProjection();
  } catch (exception) {
    if (epoch === state.epoch) error(exception.message || 'Unable to prepare this scene');
  }
}

$('retry').addEventListener('click', () => loadScene(state.scene));
$('mesh-download').addEventListener('click', () => setTimeout(refreshUsage, 3000));
$('render-button').addEventListener('click', renderProjection);
$('reset-view').addEventListener('click', () => viewer.reset());
$('mesh-opacity').addEventListener('input', event => viewer.setOpacity(event.target.value));
$('show-plane').addEventListener('change', event => viewer.setPlaneVisible(event.target.checked));
$('slice-position').addEventListener('input', event => setSlice(Number(event.target.value)));
$('slice-position').addEventListener('change', () => setSlice(state.index, true));
for (const button of document.querySelectorAll('[data-axis]')) {
  button.addEventListener('click', () => {
    state.axis = Number(button.dataset.axis);
    for (const sibling of document.querySelectorAll('[data-axis]')) {
      const selected = sibling === button;
      sibling.classList.toggle('selected', selected);
      sibling.setAttribute('aria-pressed', String(selected));
    }
    if (state.meta) {
      $('slice-position').max = state.meta.shape[state.axis] - 1;
      setSlice(Math.floor(state.meta.shape[state.axis] / 2), true);
    }
  });
}
$('slice-viewport').addEventListener('wheel', event => {
  if (!state.ready) return;
  event.preventDefault();
  setSlice(state.index + Math.sign(event.deltaY) * (event.shiftKey ? 5 : 1));
}, {passive: false});
for (const id of ['exposure', 'invert']) $(id).addEventListener('change', renderProjection);
$('resolution').addEventListener('change', renderProjection);

window.__saxDemo = {state, viewer, loadScene, setSlice, renderProjection};
window.addEventListener('error', event => state.errors.push(event.message));
window.addEventListener('unhandledrejection', event => state.errors.push(String(event.reason)));

try {
  const catalog = await api('/api/scenes');
  state.hosted = Boolean(catalog.hosted);
  if (state.hosted) {
    $('local-badge').hidden = true;
    $('usage-counts').hidden = false;
    refreshUsage();
    setInterval(refreshUsage, 30000);
  }
  for (const item of catalog.scenes) {
    const tile = document.createElement('button');
    tile.className = 'scene-tile';
    tile.dataset.scene = item.id;
    tile.title = item.name;
    tile.setAttribute('aria-label', 'Load ' + item.name);
    tile.innerHTML = '<img alt="" loading="lazy" src="' + item.thumbnail + '"><span>' + item.name + '</span>';
    tile.addEventListener('click', () => loadScene(item.id));
    $('scene-gallery').append(tile);
  }
  await loadScene(catalog.default_scene);
} catch (exception) {
  error(exception.message);
}
