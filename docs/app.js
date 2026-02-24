const playBtn = document.getElementById('play-btn');
const progress = document.getElementById('progress');
const progressFill = document.getElementById('progress-fill');

const libraryBtn = document.getElementById('library-btn');
const libraryMenu = document.getElementById('library-menu');
const librarySearch = document.getElementById('library-search');
const libraryList = document.getElementById('library-list');

const srInput = document.getElementById('sr-input');
const toastEl = document.getElementById('toast');

const waveformBox = document.getElementById('waveform-box');
const canvas = document.getElementById('waveform-canvas');
const ctx = canvas.getContext('2d');

const specWrap = document.getElementById('spec-wrap');
const specApplyBtn = document.getElementById('spec-apply');
const specSummary = document.getElementById('spec-summary');
const specBox = document.getElementById('spec-box');
const specImg = document.getElementById('spec-img');

const winSamp = document.getElementById('win-samp');
const winSec  = document.getElementById('win-sec');
const hopSamp = document.getElementById('hop-samp');
const hopSec  = document.getElementById('hop-sec');
const hopOv   = document.getElementById('hop-ov');
const nfftSamp = document.getElementById('nfft-samp');
const nfftSame = document.getElementById('nfft-same');

let audioCtx = null;

let originalBuffer = null;
let nativeSr = null;

let playBuffer = null;

let sourceNode = null;
let isPlaying = false;
let startTime = 0;
let offset = 0;
let rafId = null;

let toastTimer = null;

// Keep the original bytes so we can POST to the backend for spectrograms.
let currentAudioBlob = null;   // Blob
let currentAudioName = null;   // string

// Only update spectrogram when "Enter" is pressed.
// These are the last successfully-applied params (the ones that match the currently shown spectrogram).
let appliedSpec = { win: 512, hop: 256, nfft: 512, overlapPct: 50, sr: 16000 };

function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('is-visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('is-visible'), 1600);
}

function ensureAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
}

function setPlayIconPlaying(playing) {
  playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  playBtn.innerHTML = playing
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7V5zm6 0h4v14h-4V5z"></path></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l12-7L8 5z"></path></svg>';
}

function duration() {
  return playBuffer ? playBuffer.duration : 0;
}

function currentTimeSec() {
  if (!playBuffer) return 0;
  if (!isPlaying) return Math.min(offset, duration());
  return Math.min(offset + (audioCtx.currentTime - startTime), duration());
}

function updateProgress() {
  const dur = duration();
  const t = currentTimeSec();
  const pct = dur > 0 ? (t / dur) * 100 : 0;
  progressFill.style.width = pct + '%';
  progress.setAttribute('aria-valuenow', String(Math.round(pct)));
}

function tick() {
  updateProgress();
  if (isPlaying) rafId = requestAnimationFrame(tick);
}

function stopSource() {
  if (sourceNode) {
    try { sourceNode.stop(); } catch {}
    try { sourceNode.disconnect(); } catch {}
    sourceNode = null;
  }
}

function playFromOffset() {
  if (!playBuffer) return;
  ensureAudioCtx();

  stopSource();
  sourceNode = audioCtx.createBufferSource();
  sourceNode.buffer = playBuffer;
  sourceNode.connect(audioCtx.destination);

  startTime = audioCtx.currentTime;
  isPlaying = true;
  setPlayIconPlaying(true);

  sourceNode.onended = () => {
    if (isPlaying && currentTimeSec() >= duration() - 1e-3) {
      isPlaying = false;
      offset = 0;
      setPlayIconPlaying(false);
      updateProgress();
    }
  };

  sourceNode.start(0, Math.min(offset, duration()));
  cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(tick);
}

function pausePlayback() {
  if (!isPlaying) return;
  ensureAudioCtx();

  offset = currentTimeSec();
  isPlaying = false;
  setPlayIconPlaying(false);
  stopSource();
  cancelAnimationFrame(rafId);
  updateProgress();
}

playBtn.addEventListener('click', () => {
  if (!playBuffer) return;
  if (isPlaying) pausePlayback();
  else playFromOffset();
});

function seekFromClientX(clientX) {
  const dur = duration();
  if (dur <= 0) return;

  const rect = progress.getBoundingClientRect();
  const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
  const ratio = rect.width > 0 ? x / rect.width : 0;
  offset = ratio * dur;

  if (isPlaying) playFromOffset();
  else updateProgress();
}

progress.addEventListener('click', (e) => seekFromClientX(e.clientX));

progress.addEventListener('keydown', (e) => {
  const dur = duration();
  if (dur <= 0) return;

  const step = Math.max(dur * 0.02, 0.25);
  if (e.key === 'ArrowLeft') {
    offset = Math.max(0, currentTimeSec() - step);
    if (isPlaying) playFromOffset(); else updateProgress();
    e.preventDefault();
  }
  if (e.key === 'ArrowRight') {
    offset = Math.min(dur, currentTimeSec() + step);
    if (isPlaying) playFromOffset(); else updateProgress();
    e.preventDefault();
  }
});

/* -------- Library dropdown -------- */
function filterLibrary(q) {
  const query = q.trim().toLowerCase();
  [...libraryList.querySelectorAll('.library-item')].forEach(btn => {
    const txt = btn.textContent.trim().toLowerCase();
    btn.style.display = txt.includes(query) ? '' : 'none';
  });
}

function openLibrary() {
  libraryMenu.classList.add('is-open');
  libraryMenu.setAttribute('aria-hidden', 'false');
  librarySearch.value = '';
  filterLibrary('');
  librarySearch.focus();
}

function closeLibrary() {
  libraryMenu.classList.remove('is-open');
  libraryMenu.setAttribute('aria-hidden', 'true');
}

libraryBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  libraryMenu.classList.contains('is-open') ? closeLibrary() : openLibrary();
});

document.addEventListener('click', () => closeLibrary());
libraryMenu.addEventListener('click', (e) => e.stopPropagation());
librarySearch.addEventListener('input', (e) => filterLibrary(e.target.value));

function setActiveItem(activeBtn) {
  [...libraryList.querySelectorAll('.library-item')].forEach(b => b.classList.remove('is-active'));
  if (activeBtn) activeBtn.classList.add('is-active');
}

/* -------- Resampling + waveform -------- */
async function resampleBuffer(buffer, targetSr) {
  if (buffer.sampleRate === targetSr) return buffer;

  const channels = buffer.numberOfChannels;
  const length = Math.ceil(buffer.length * (targetSr / buffer.sampleRate));
  const offline = new OfflineAudioContext(channels, length, targetSr);

  const src = offline.createBufferSource();
  src.buffer = buffer;
  src.connect(offline.destination);
  src.start(0);

  return await offline.startRendering();
}

function resizeCanvasToBox() {
  const w = waveformBox.clientWidth;
  const h = 160;
  const dpr = window.devicePixelRatio || 1;

  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawWaveformFromBuffer(buffer) {
  if (!buffer) return;

  waveformBox.classList.remove('is-hidden');
  resizeCanvasToBox();

  const W = waveformBox.clientWidth;
  const H = 160;
  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
  ctx.lineWidth = 1;

  const mid = H / 2;

  const pad = 18;
  const x0 = pad;
  const x1 = W - pad;

  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;

  const N = ch0.length;
  const cols = Math.max(200, Math.floor(W));

  for (let i = 0; i < cols; i++) {
    const start = Math.floor((i / cols) * N);
    const end = Math.floor(((i + 1) / cols) * N);

    let peak = 0;
    for (let j = start; j < end; j++) {
      const v = ch1 ? 0.5 * (ch0[j] + ch1[j]) : ch0[j];
      const a = Math.abs(v);
      if (a > peak) peak = a;
    }

    const y = peak * (H * 0.42);
    const x = x0 + (i / (cols - 1)) * (x1 - x0);

    ctx.beginPath();
    ctx.moveTo(x, mid - y);
    ctx.lineTo(x, mid + y);
    ctx.stroke();
  }
}

function setSpecUIVisible(isVisible) {
  if (isVisible) specWrap.classList.remove('is-hidden');
  else specWrap.classList.add('is-hidden');
}

function syncSpecSecondsFromSamples() {
  if (!playBuffer) return;
  const sr = playBuffer.sampleRate;

  const w = parseInt(winSamp.value, 10);
  const h = parseInt(hopSamp.value, 10);

  if (Number.isFinite(w) && w > 0) winSec.value = (w / sr).toFixed(4);
  if (Number.isFinite(h) && h > 0) hopSec.value = (h / sr).toFixed(4);

  if (Number.isFinite(w) && w > 0 && Number.isFinite(h) && h > 0) {
    const ov = Math.max(0, Math.min(99.9, (1 - (h / w)) * 100));
    hopOv.value = ov.toFixed(1).replace(/\.0$/, '');
  }
}

function syncSpecSamplesFromSeconds() {
  if (!playBuffer) return;
  const sr = playBuffer.sampleRate;

  const wsec = parseFloat(winSec.value);
  const hsec = parseFloat(hopSec.value);

  if (Number.isFinite(wsec) && wsec > 0) winSamp.value = String(Math.max(1, Math.round(wsec * sr)));
  if (Number.isFinite(hsec) && hsec > 0) hopSamp.value = String(Math.max(1, Math.round(hsec * sr)));

  const w = parseInt(winSamp.value, 10);
  const h = parseInt(hopSamp.value, 10);
  if (Number.isFinite(w) && w > 0 && Number.isFinite(h) && h > 0) {
    const ov = Math.max(0, Math.min(99.9, (1 - (h / w)) * 100));
    hopOv.value = ov.toFixed(1).replace(/\.0$/, '');
  }
}

function applyOverlapToHop() {
  const w = parseInt(winSamp.value, 10);
  const ov = parseFloat(hopOv.value);
  if (!Number.isFinite(w) || w <= 0) return;
  if (!Number.isFinite(ov)) return;

  const hop = Math.max(1, Math.round(w * (1 - ov / 100)));
  hopSamp.value = String(hop);
  syncSpecSecondsFromSamples();
}

function normalizeSpecInputs() {
  // Keep things consistent and valid, but don't be aggressive.
  const w = Math.max(1, Math.floor(parseFloat(winSamp.value || '1')));
  winSamp.value = String(w);

  const hop = Math.max(1, Math.floor(parseFloat(hopSamp.value || '1')));
  hopSamp.value = String(hop);

  const nfft = Math.max(1, Math.floor(parseFloat(nfftSamp.value || '1')));
  nfftSamp.value = String(nfft);

  if (nfftSame.checked) {
    nfftSamp.value = String(w);
    nfftSamp.disabled = true;
  } else {
    nfftSamp.disabled = false;
  }

  syncSpecSecondsFromSamples();
}

function fmtInt(x){ return String(Math.round(x)); }

function updateAppliedSummary(p) {
  const ov = p.win > 0 ? Math.max(0, Math.min(99.9, (1 - (p.hop / p.win)) * 100)) : 0;
  specSummary.textContent = `Window: ${p.win} | Hop size: ${p.hop} (${ov.toFixed(1).replace(/\.0$/, '')}%) | NFFT: ${p.nfft}`;
}

async function requestSpectrogram(params) {
  if (!currentAudioBlob) throw new Error('No audio blob available');
  if (!playBuffer) throw new Error('No audio loaded');

  const form = new FormData();
  form.append('audio', currentAudioBlob, currentAudioName || 'audio.wav');
  form.append('sr', String(playBuffer.sampleRate));
  form.append('win', String(params.win));
  form.append('hop', String(params.hop));
  form.append('nfft', String(params.nfft));

  // Option A backend endpoint (you'll implement later)
  const res = await fetch('/api/spectrogram', { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);

  const blob = await res.blob();
  if (!blob.type.startsWith('image/')) throw new Error('Expected an image response');

  return blob;
}

async function applySpectrogram() {
  if (!playBuffer) return;

  normalizeSpecInputs();

  // Decide hop from overlap if the overlap field was the most recently edited.
  // Draft behavior: if overlap field is focused when you press Enter, use it.
  const activeId = document.activeElement && document.activeElement.id;

  if (activeId === 'hop-ov') applyOverlapToHop();
  if (activeId === 'win-sec') syncSpecSamplesFromSeconds();
  if (activeId === 'hop-sec') syncSpecSamplesFromSeconds();

  normalizeSpecInputs();

  const w = parseInt(winSamp.value, 10);
  const h = parseInt(hopSamp.value, 10);
  let n = parseInt(nfftSamp.value, 10);

  if (nfftSame.checked) n = w;

  // Minimal validation (backend should validate too)
  if (h <= 0 || w <= 0 || n <= 0) { toast('Invalid STFT parameters'); return; }
  if (h > w) { toast('Hop size must be <= window'); return; }
  if (n < w) { toast('NFFT must be >= window'); return; }

  // Only update the displayed spectrogram after we successfully get an image
  try {
    specApplyBtn.disabled = true;
    specApplyBtn.textContent = '...';

    const imgBlob = await requestSpectrogram({ win: w, hop: h, nfft: n });

    // Swap image (keep old URL to revoke)
    const oldUrl = specImg.dataset.url;
    const url = URL.createObjectURL(imgBlob);
    specImg.src = url;
    specImg.dataset.url = url;
    if (oldUrl) URL.revokeObjectURL(oldUrl);

    specBox.classList.remove('is-hidden');
    appliedSpec = { win: w, hop: h, nfft: n, overlapPct: parseFloat(hopOv.value), sr: playBuffer.sampleRate };
    updateAppliedSummary(appliedSpec);
  } catch (e) {
    console.error(e);
    toast('Spectrogram backend not available');
    // keep current spectrogram + summary (stock behavior)
  } finally {
    specApplyBtn.disabled = false;
    specApplyBtn.textContent = 'Enter';
  }
}

specApplyBtn.addEventListener('click', applySpectrogram);

[winSamp, winSec, hopSamp, hopSec, hopOv, nfftSamp].forEach(el => {
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      applySpectrogram();
    }
  });
});

winSamp.addEventListener('input', () => { if (nfftSame.checked) nfftSamp.value = winSamp.value; syncSpecSecondsFromSamples(); });
hopSamp.addEventListener('input', () => syncSpecSecondsFromSamples());
winSec.addEventListener('input', () => syncSpecSamplesFromSeconds());
hopSec.addEventListener('input', () => syncSpecSamplesFromSeconds());
hopOv.addEventListener('input', () => applyOverlapToHop());

nfftSame.addEventListener('change', () => normalizeSpecInputs());

/* -------- Sample rate change -------- */
async function applySampleRate(requested) {
  if (!originalBuffer) return;

  requested = Math.floor(requested);
  if (!Number.isFinite(requested) || requested < 1) {
    requested = 1;
    srInput.value = '1';
  } else {
    srInput.value = String(requested);
  }

  if (requested > nativeSr) {
    requested = nativeSr;
    srInput.value = String(nativeSr);
    toast("Sample rate too high, selecting file's sample rate");
  }

  const durOld = duration();
  const tOld = currentTimeSec();
  const ratio = durOld > 0 ? tOld / durOld : 0;

  if (isPlaying) pausePlayback();
  else stopSource();

  try {
    playBuffer = await resampleBuffer(originalBuffer, requested);
  } catch (e) {
    // Some browsers fail at very low SRs in OfflineAudioContext; clamp to 8000 as a safe fallback.
    const safe = Math.max(8000, Math.min(requested, nativeSr));
    if (safe !== requested) {
      toast('Sample rate too low, selecting 8000 Hz');
      srInput.value = '8000';
    }
    playBuffer = await resampleBuffer(originalBuffer, safe);
  }

  offset = ratio * duration();
  updateProgress();
  drawWaveformFromBuffer(playBuffer);

  // Keep UI in sync
  setSpecUIVisible(true);
  syncSpecSecondsFromSamples();
}

/* -------- Loading examples -------- */
async function loadExample(btn) {
  ensureAudioCtx();

  const src = btn.getAttribute('data-src');
  setActiveItem(btn);

  const res = await fetch(src);
  const arr = await res.arrayBuffer();

  // Store audio bytes for backend spectrogram calls
  currentAudioBlob = new Blob([arr], { type: 'audio/wav' });
  currentAudioName = (src.split('/').pop() || 'audio.wav');

  originalBuffer = await audioCtx.decodeAudioData(arr.slice(0));
  nativeSr = originalBuffer.sampleRate;

  srInput.max = String(nativeSr);

  let requested = parseInt(srInput.value, 10);
  if (!Number.isFinite(requested) || requested < 1) requested = 16000;
  if (requested > nativeSr) {
    requested = nativeSr;
    srInput.value = String(nativeSr);
    toast("Sample rate too high, selecting file's sample rate");
  }

  try {
    playBuffer = await resampleBuffer(originalBuffer, requested);
  } catch (e) {
    const safe = Math.max(8000, Math.min(requested, nativeSr));
    toast('Sample rate too low, selecting 8000 Hz');
    srInput.value = '8000';
    playBuffer = await resampleBuffer(originalBuffer, safe);
  }

  offset = 0;
  isPlaying = false;
  setPlayIconPlaying(false);
  stopSource();
  updateProgress();

  drawWaveformFromBuffer(playBuffer);

  // show spectrogram controls after audio load
  setSpecUIVisible(true);
  normalizeSpecInputs();
  updateAppliedSummary(appliedSpec);

  closeLibrary();
}

libraryList.addEventListener('click', async (e) => {
  const btn = e.target.closest('.library-item');
  if (!btn) return;

  try {
    await loadExample(btn);
  } catch (err) {
    console.error(err);
    waveformBox.classList.add('is-hidden');
    specWrap.classList.add('is-hidden');
    specBox.classList.add('is-hidden');

    originalBuffer = null;
    playBuffer = null;
    nativeSr = null;
    currentAudioBlob = null;
    currentAudioName = null;

    offset = 0;
    isPlaying = false;
    setPlayIconPlaying(false);
    stopSource();
    updateProgress();
  }
});

srInput.addEventListener('change', async () => {
  if (!originalBuffer) return;
  const requested = parseInt(srInput.value, 10);
  await applySampleRate(requested);
});

window.addEventListener('resize', () => {
  if (waveformBox.classList.contains('is-hidden')) return;
  drawWaveformFromBuffer(playBuffer);
});
