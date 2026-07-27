/**
 * MicroG — renderer entry point. Wires the WebSocket protocol (see
 * runtime/server.py's module docstring for the message shapes) to every
 * view module, and owns app-level state: which conversation is open, what
 * the panel tabs are doing, and the retry-once-silently error policy from
 * UI-SPEC.md.
 */

import { Wake } from './wake.js';
import { NeuronField } from './neurons.js';
import { ProbabilitiesView } from './probabilities.js';
import { ChatView } from './chat.js';
import { Settings } from './settings.js';
import { History } from './history.js';
import { SpringLoop, SPRING } from './spring.js';

const WS_URL = 'ws://localhost:8899';
const PINNED_CHART_LEN = 80;

// ---------------------------------------------------------------- elements --
const $ = (s) => document.querySelector(s);
const wakeOverlay = $('#wake-overlay');
const wakeLabel = $('#wake-label');
const brandParams = $('#brand-params');
const brandSpeed = $('#brand-speed');
const tabs = document.querySelectorAll('.tab');
const neuronsView = $('#neurons-view');
const probsView = $('#probabilities-view');
const inputField = $('#input-field');
const sendBtn = $('#send-btn');
const pinnedChart = $('#pinned-chart');
const pinnedLabel = $('#pinned-label');
const pinnedCanvas = $('#pinned-canvas');
const pinnedUnpin = $('#pinned-unpin');

// ------------------------------------------------------------------- state --
let ws = null;
let modelReady = false;
let generating = false;
let currentConvId = null;
let messages = []; // [{role, text}]
let reconnectedOnce = false;
let pinnedFlat = null;
let pinnedHistory = [];
// Arrival times of the last few tokens. Rate is measured across the whole
// window rather than from the single previous token: WebSocket messages do
// not arrive evenly spaced, and two landing in the same JS tick put ~0.6ms in
// the denominator, which is where the nonsense "1667 tok/s" readings came
// from. Averaging over a span turns those bursts back into the real rate.
const SPEED_WINDOW = 12;
let tokenTimes = [];
let lastAssistantText = '';

// ------------------------------------------------------------------- Wake --
const wake = new Wake($('#wake-canvas'), () => {
  wakeOverlay.style.transition = 'opacity 500ms ease-out';
  wakeOverlay.style.opacity = '0';
  setTimeout(() => { wakeOverlay.style.display = 'none'; }, 520);
});

// --------------------------------------------------------------- neurons --
const neurons = new NeuronField($('#neurons-canvas'), $('#neuron-tooltip'), {
  onPin: (layer, neuronIdx, flat) => {
    pinnedFlat = flat;
    pinnedHistory = [];
    pinnedLabel.textContent = `layer ${layer} · neuron ${neuronIdx}`;
    pinnedChart.classList.add('visible');
    pinnedChart.style.transition = 'opacity 240ms ease-out, transform 240ms cubic-bezier(.2,.8,.2,1)';
    pinnedChart.style.opacity = '1';
    pinnedChart.style.transform = 'translateY(0)';
  },
});
pinnedUnpin.addEventListener('click', () => {
  pinnedFlat = null;
  neurons.unpin();
  pinnedChart.style.opacity = '0';
  pinnedChart.style.transform = 'translateY(8px)';
  setTimeout(() => pinnedChart.classList.remove('visible'), 240);
});

function drawPinnedChart() {
  const ctx = pinnedCanvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = pinnedCanvas.getBoundingClientRect();
  if (pinnedCanvas.width !== rect.width * dpr) {
    pinnedCanvas.width = rect.width * dpr;
    pinnedCanvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  const n = PINNED_CHART_LEN;
  const step = w / (n - 1);

  // Honesty: only known values are drawn; a token where this neuron wasn't
  // in the top-64 leaves a real gap, not an interpolated guess.
  ctx.beginPath();
  let penDown = false;
  pinnedHistory.forEach((v, i) => {
    const x = i * step;
    if (v === null) { penDown = false; return; }
    const y = h - v * h * 0.9 - h * 0.05;
    if (!penDown) { ctx.moveTo(x, y); penDown = true; }
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  ctx.lineWidth = 1.3;
  ctx.stroke();
}

// -------------------------------------------------------------- probabilities --
const probs = new ProbabilitiesView($('#prob-bars'), $('#confidence-fill'), $('#confidence-state'));
window.__debug = { probs, neurons };

// ------------------------------------------------------------------- chat --
const chat = new ChatView($('#messages'));

// --------------------------------------------------------------- settings --
const settings = new Settings($('#settings-popover'));
$('#settings-trigger').addEventListener('click', (e) => {
  settings.toggle(e.currentTarget.getBoundingClientRect());
});

// ----------------------------------------------------------------- history --
const history = new History({
  sidebarEl: $('#sidebar'),
  scrimEl: $('#sidebar-scrim'),
  listEl: $('#sidebar-list'),
  onOpenConversation: (id) => loadConversation(id),
});
$('#history-trigger').addEventListener('click', () => history.toggle());
$('#sidebar-new').addEventListener('click', startNewConversation);

async function loadConversation(id) {
  const conv = await window.microg.history.load(id);
  if (!conv) return;
  currentConvId = id;
  messages = conv.messages || [];
  chat.clear();
  for (const m of messages) {
    if (m.role === 'user') chat.addUserMessage(m.text);
    else { chat.beginAssistantMessage(); chat.appendToken(m.text); chat.endAssistantMessage(); }
  }
  probs.reset();
}

function startNewConversation() {
  currentConvId = null;
  messages = [];
  chat.clear();
  probs.reset();
  history.setActive(null);
  inputField.focus();
}

async function persistConversation() {
  if (!messages.length) return;
  const title = messages[0]?.text?.slice(0, 60) || 'Untitled';
  const id = await window.microg.history.save({
    id: currentConvId, title, updated: Date.now(), messages,
  });
  currentConvId = id;
  history.setActive(id);
  if (history.open) history.refresh();
}

// -------------------------------------------------------------------- tabs --
const tabLoop = new SpringLoop(renderTabIndicator);
const tabX = tabLoop.spring('tab-x', 0, SPRING.default);
const tabW = tabLoop.spring('tab-w', 0, SPRING.default);
let activeTabIndex = 0;

function setTab(index) {
  activeTabIndex = index;
  tabs.forEach((t, i) => t.setAttribute('aria-selected', String(i === index)));
  neuronsView.classList.toggle('active', index === 0);
  probsView.classList.toggle('active', index === 1);
  const el = tabs[index];
  tabX.set(el.offsetLeft);
  tabW.set(el.querySelector('.indicator').offsetWidth);
  tabLoop.start();
}

function renderTabIndicator() {
  const ind = tabs[activeTabIndex].querySelector('.indicator');
  ind.style.transform = `translateX(0)`;
  ind.style.width = `${tabW.value}px`;
  tabs.forEach((t, i) => {
    if (i !== activeTabIndex) return;
    t.style.transform = '';
  });
  // Position all indicators absolutely at the spring-driven x, only the
  // active one visible — simplest correct way to animate between two fixed
  // tab positions without measuring layout every frame.
  tabs.forEach((t) => { t.querySelector('.indicator').style.opacity = '0'; });
  const activeInd = tabs[activeTabIndex].querySelector('.indicator');
  activeInd.style.opacity = '1';
  activeInd.style.left = `${tabX.value - tabs[activeTabIndex].offsetLeft + 8}px`;
}

tabs.forEach((t, i) => t.addEventListener('pointerdown', () => setTab(i)));
requestAnimationFrame(() => setTab(0));

// -------------------------------------------------------------------- input --
function autoGrow() {
  inputField.style.height = 'auto';
  inputField.style.height = `${Math.min(160, inputField.scrollHeight)}px`;
}
inputField.addEventListener('input', () => {
  autoGrow();
  sendBtn.classList.toggle('disabled', !inputField.value.trim() || generating);
});
inputField.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
sendBtn.addEventListener('click', sendMessage);

function sendMessage() {
  const text = inputField.value.trim();
  if (!text || generating || !modelReady) return;
  inputField.value = '';
  autoGrow();
  sendBtn.classList.add('disabled');

  messages.push({ role: 'user', text });
  chat.addUserMessage(text);
  chat.beginAssistantMessage();
  probs.reset(); // last turn's candidates shouldn't linger into this one
  generating = true;
  tokenTimes = [];

  ws.send(JSON.stringify({
    type: 'chat', text,
    temperature: settings.values.temperature,
    top_k: settings.values.top_k,
    rag: settings.values.rag,
  }));
}

// -------------------------------------------------------------- keyboard --
document.addEventListener('keydown', (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key.toLowerCase() === 'n') { e.preventDefault(); startNewConversation(); }
  else if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); history.toggle(); }
  else if (mod && e.key === ',') { e.preventDefault(); settings.toggle($('#settings-trigger').getBoundingClientRect()); }
  else if (e.key === 'Escape') {
    if (generating) { ws?.send(JSON.stringify({ type: 'stop' })); }
    else if (settings.isOpen) settings.close();
    else if (history.open) history.close();
  } else if (e.key === 'Tab' && e.target !== inputField
             && !e.target.closest('#settings-popover, #sidebar')) {
    // Excluding the input field matters: an IME/dead-key sequence composing
    // an accented character (this app's whole reason for existing is Polish
    // text) can synthesize a stray Tab keydown mid-composition — it must
    // never yank focus away from what's being typed.
    e.preventDefault();
    setTab(1 - activeTabIndex);
  }
});

window.microg?.onShortcut?.('new', startNewConversation);
window.microg?.onShortcut?.('history', () => history.toggle());
window.microg?.onShortcut?.('settings', () =>
  settings.toggle($('#settings-trigger').getBoundingClientRect()));

// --------------------------------------------------------------- WS wiring --
function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => { reconnectedOnce = false; };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case 'wake':
        wake.onWakeEvent(msg.layer);
        wakeLabel.textContent = msg.layer === -1
          ? 'loading embeddings'
          : `loading layer ${msg.layer + 1} / ${msg.of}`;
        break;

      case 'ready':
        modelReady = true;
        wake.finish();
        brandParams.textContent = `${(msg.params / 1e6).toFixed(1)}M`;
        inputField.placeholder = 'Message MicroG…';
        break;

      case 'context':
        chat.setContextSource(msg.source);
        break;

      case 'step': {
        neurons.applyStep(msg.neurons);
        probs.update(msg.probs);
        if (msg.token) {
          chat.appendToken(msg.token);
          lastAssistantText += msg.token;
        }

        tokenTimes.push(performance.now());
        if (tokenTimes.length > SPEED_WINDOW) tokenTimes.shift();
        if (tokenTimes.length >= 2) {
          const span = tokenTimes[tokenTimes.length - 1] - tokenTimes[0];
          // A whole window inside one tick would still divide by ~0, so fall
          // back to showing nothing rather than a fabricated number.
          if (span > 0) {
            const hz = ((tokenTimes.length - 1) * 1000) / span;
            brandSpeed.textContent = `${hz.toFixed(1)} tok/s`;
          }
        }

        if (pinnedFlat !== null) {
          pinnedHistory.push(neurons.lastKnown(pinnedFlat));
          if (pinnedHistory.length > PINNED_CHART_LEN) pinnedHistory.shift();
          drawPinnedChart();
        }

        if (msg.done) {
          chat.endAssistantMessage();
          neurons.markGenerationIdle();
          generating = false;
          messages.push({ role: 'assistant', text: lastAssistantText });
          lastAssistantText = '';
          persistConversation();
        }
        break;
      }

      case 'error':
        generating = false;
        chat.showError(msg.message, () => { connect(); });
        break;
    }
  };

  ws.onclose = () => {
    modelReady = false;
    if (!reconnectedOnce) {
      // UI-SPEC.md: "Apka próbuje wstać sama raz w tle, zanim powie" —
      // one silent retry before surfacing anything to the user.
      reconnectedOnce = true;
      setTimeout(connect, 800);
    } else {
      chat.showError('Lost connection to the model.', () => { reconnectedOnce = false; connect(); });
    }
  };
}

connect();
