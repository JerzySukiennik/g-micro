/**
 * G-Micro — renderer entry point. Wires the WebSocket protocol (see
 * runtime/server.py's module docstring for the message shapes) to every
 * view module, and owns app-level state: which conversation is open, what
 * the panel tabs are doing, and the retry-once-silently error policy from
 * UI-SPEC.md.
 */

import { Wake } from './wake.js';
import { ChatView } from './chat.js';
import { History } from './history.js';
import { Onboarding } from './onboarding.js';
import { SpringLoop, SPRING } from './spring.js';

const WS_URL = 'ws://localhost:8899';

// Sampling is fixed rather than exposed. Sliders for temperature and top_k
// ask the reader to understand sampling before they can hold a conversation,
// and every value they could pick is worse than a measured one: identity
// answers came back correct 12/12 at 0.8, 0.4 and greedy alike, so the choice
// only ever changed how varied the wording was.
//
// RAG is off. Once its retrieval was fixed it does find the right article,
// but grounded answers come out as bare extracted spans — "Co to jest
// fotosynteza?" answered "koenzym" — because the grounding data it learned
// from (PoQuAD) has span-shaped answers. Fluent and sometimes wrong reads
// better in a chat than terse and sometimes wrong.
const GENERATION = { temperature: 0.7, top_k: 40, rag: false };

/**
 * Suggestions offered on an empty conversation. Four are drawn at random from
 * this pool each time, so the app does not look identical on every open and a
 * wider slice of what the model handles well gets seen.
 *
 * Every entry is something measured as working: identity (100% on the
 * benchmark), list-shaped instructions (90%), and short factual prompts. Asking
 * it for a precise date or a calculation would showcase exactly what it fails
 * at.
 */
const SUGGESTIONS = [
  'Kim jesteś?', 'Wymień trzy owoce.', 'Czym jest Warszawa?',
  'Opowiedz coś o psach.', 'Wymień trzy kolory.', 'Podaj dwa polskie miasta.',
  'Kto cię stworzył?', 'Wymień cztery pory roku.', 'Jak zrobić naleśniki?',
  'Opowiedz o Krakowie.', 'Wymień trzy zwierzęta.', 'Co to jest rower?',
  'Wymień trzy dni tygodnia.', 'Opowiedz krótko o morzu.',
];

function fillSuggestions() {
  const host = document.querySelector('#suggestions');
  if (!host) return;
  const pool = [...SUGGESTIONS];
  const pick = [];
  while (pick.length < 4 && pool.length) {
    pick.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
  }
  host.innerHTML = pick
    .map((s) => `<button class="suggestion" data-text="${s}">${s}</button>`)
    .join('');
}

/** Greets by name once onboarding has one. The model is never told — it was
 *  not trained to use a name and would spend context on it for nothing. */
function applyName(name) {
  const h = document.querySelector('#welcome h1');
  if (h) h.textContent = name ? `Cześć, ${name}.` : 'Cześć, jestem G-Micro.';
}

// ---------------------------------------------------------------- elements --
const $ = (s) => document.querySelector(s);
const wakeOverlay = $('#wake-overlay');
const wakeLabel = $('#wake-label');
const inputField = $('#input-field');
const sendBtn = $('#send-btn');

// ------------------------------------------------------------------- state --
let ws = null;
let modelReady = false;
let generating = false;
let currentConvId = null;
let messages = []; // [{role, text}]
let reconnectAttempt = 0;
// Distinguishes "never started" from "lost mid-session": only the first
// case should keep the wake overlay's wording.
let modelEverReady = false;
const RECONNECT_LIMIT = 20;   // ~30s of backoff before admitting defeat
// Arrival times of the last few tokens. Rate is measured across the whole
// window rather than from the single previous token: WebSocket messages do
// not arrive evenly spaced, and two landing in the same JS tick put ~0.6ms in
// the denominator, which is where the nonsense "1667 tok/s" readings came
// from. Averaging over a span turns those bursts back into the real rate.
let lastAssistantText = '';

// ------------------------------------------------------------------- Wake --
const wake = new Wake($('#wake-canvas'), () => {
  wakeOverlay.style.transition = 'opacity 500ms ease-out';
  wakeOverlay.style.opacity = '0';
  setTimeout(() => { wakeOverlay.style.display = 'none'; }, 520);
});

// ------------------------------------------------------------------- chat --
const chat = new ChatView($('#messages'));

/**
 * The send button becomes a stop button while the model is writing.
 *
 * Stopping was only ever bound to Escape, which nobody discovers — the
 * onboarding mentions it once and then it is gone. Reusing the button the user
 * is already looking at needs no explanation at all.
 */
function setComposerMode(mode) {
  const stopping = mode === 'stop';
  sendBtn.classList.toggle('stopping', stopping);
  sendBtn.setAttribute('aria-label', stopping ? 'Zatrzymaj' : 'Wyślij');
  sendBtn.innerHTML = stopping
    ? '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2.5" fill="currentColor"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none"><path d="M12 19V5M12 5l-6 6M12 5l6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
}

// ----------------------------------------------------------------- history --
const history = new History({
  sidebarEl: $('#sidebar'),
  scrimEl: $('#sidebar-scrim'),
  listEl: $('#sidebar-list'),
  onOpenConversation: (id) => loadConversation(id),
});
$('#history-trigger').addEventListener('click', () => history.toggle());
$('#sidebar-new').addEventListener('click', startNewConversation);
$('#new-trigger').addEventListener('click', startNewConversation);

// Delegated, because chat.clear() re-inserts the welcome markup and any
// listener bound directly to those buttons would die with the old nodes.
$('#messages').addEventListener('click', (e) => {
  const chip = e.target.closest('.suggestion');
  if (!chip || !modelReady || generating) return;
  inputField.value = chip.dataset.text;
  autoGrow();
  sendMessage();
});

async function loadConversation(id) {
  const conv = await window.gmicro.history.load(id);
  if (!conv) return;
  currentConvId = id;
  messages = conv.messages || [];
  chat.clear();
  for (const m of messages) {
    if (m.role === 'user') chat.addUserMessage(m.text);
    else { chat.beginAssistantMessage(); chat.appendToken(m.text); chat.endAssistantMessage(); }
  }
}

function startNewConversation() {
  currentConvId = null;
  messages = [];
  chat.clear();
  fillSuggestions();
  applyName(Onboarding.name());
  history.setActive(null);
  inputField.focus();
}

async function persistConversation() {
  if (!messages.length) return;
  const title = messages[0]?.text?.slice(0, 60) || 'Untitled';
  const id = await window.gmicro.history.save({
    id: currentConvId, title, updated: Date.now(), messages,
  });
  currentConvId = id;
  history.setActive(id);
  if (history.open) history.refresh();
}

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
sendBtn.addEventListener('click', () => {
  if (generating) { ws?.send(JSON.stringify({ type: 'stop' })); return; }
  sendMessage();
});

function sendMessage() {
  const text = inputField.value.trim();
  if (!text || generating || !modelReady) return;
  inputField.value = '';
  autoGrow();
  sendBtn.classList.add('disabled');

  messages.push({ role: 'user', text });
  chat.addUserMessage(text);
  chat.beginAssistantMessage();
  generating = true;
  setComposerMode('stop');
  // The last completed exchange, so follow-ups like "a co z tym?" work. The
  // server decides how much of it to actually use — measured at one exchange,
  // beyond which quality drops on every message (runtime/server.py).
  const history = [];
  for (let i = messages.length - 3; i >= 0 && history.length < 4; i -= 2) {
    const u = messages[i], a = messages[i + 1];
    if (u?.role === 'user' && a?.role === 'assistant' && a.text) {
      history.unshift({ user: u.text, assistant: a.text });
    }
  }
  ws.send(JSON.stringify({ type: 'chat', text, history, ...GENERATION }));
}

// -------------------------------------------------------------- keyboard --
document.addEventListener('keydown', (e) => {
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key.toLowerCase() === 'n') { e.preventDefault(); startNewConversation(); }
  else if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); history.toggle(); }
  else if (e.key === 'Escape') {
    if (generating) { ws?.send(JSON.stringify({ type: 'stop' })); }
    else if (history.open) history.close();
  }
});

const onboarding = new Onboarding({onFinish: (name) => { applyName(name); inputField.focus(); }});
if (!Onboarding.seen()) onboarding.open();
applyName(Onboarding.name());
fillSuggestions();

window.gmicro?.onShortcut?.('onboarding', () => onboarding.open());
window.gmicro?.onShortcut?.('rag', (_e, enabled) => { GENERATION.rag = Boolean(enabled); });
window.gmicro?.onShortcut?.('new', startNewConversation);
window.gmicro?.onShortcut?.('history', () => history.toggle());

// --------------------------------------------------------------- WS wiring --
function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => { reconnectAttempt = 0; };

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
        modelEverReady = true;
        wake.finish();
        inputField.placeholder = 'Napisz wiadomość…';
        break;

      case 'context':
        chat.setContextSource(msg.source);
        break;

      case 'step': {
        if (msg.token) {
          chat.appendToken(msg.token);
          lastAssistantText += msg.token;
        }

        if (msg.done) {
          setComposerMode('send');
          chat.endAssistantMessage();
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
    // UI-SPEC.md asks the app to get itself back up before saying anything.
    // The original policy was a single retry after 800ms, which is a race the
    // app loses on a cold start: Electron spawns the Python backend and then
    // loads this page, but the backend has to import torch before it can
    // listen — measured at 3.8s. Both attempts land before the socket exists,
    // the wake overlay never receives 'ready', and the window sits on "Budzę
    // model…" forever. Confirmed 2026-07-27, and it had been marginal all
    // along rather than newly broken.
    //
    // Backing off up to RECONNECT_LIMIT covers a slow start without hiding a
    // genuinely dead backend: after ~30s it still surfaces the error.
    reconnectAttempt += 1;
    if (reconnectAttempt <= RECONNECT_LIMIT) {
      if (!modelEverReady) {
        wakeLabel.textContent = reconnectAttempt < 4
          ? 'Budzę model…' : 'Budzę model… (jeszcze chwila)';
      }
      setTimeout(connect, Math.min(300 * reconnectAttempt, 2000));
    } else {
      chat.showError('Nie mogę połączyć się z modelem.',
                     () => { reconnectAttempt = 0; connect(); });
    }
  };
}

connect();
