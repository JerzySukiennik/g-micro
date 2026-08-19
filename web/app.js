/**
 * Web entry point. Same conversation, same composer, same model picker as the
 * desktop app — the one thing that changes is where the answer comes from:
 * a Realtime Database node instead of a WebSocket on localhost.
 *
 * Everything visual is imported from the shared renderer modules (kept in step
 * by web/sync.sh), so this file is only the wiring that differs.
 */

import { ChatView } from './chat.js';
import { History } from './history.js';
import { Attachment, ModelPicker } from './composer.js';
import { store } from './store.js';
import { MacBridge } from './bridge.js';

const $ = (s) => document.querySelector(s);

// Same pool as the desktop app: every one of these is something the model was
// measured to handle well.
const SUGGESTIONS = [
  'Kim jesteś?', 'Wymień trzy owoce.', 'Czym jest Warszawa?',
  'Opowiedz coś o psach.', 'Wymień trzy kolory.', 'Podaj dwa polskie miasta.',
  'Kto cię stworzył?', 'Wymień cztery pory roku.', 'Jak zrobić naleśniki?',
  'Opowiedz o Krakowie.', 'Wymień trzy zwierzęta.', 'Co to jest rower?',
];

const HINTS = {
  'g-micro': 'G-Micro jest bardzo małym modelem i często się myli.',
  'g-images': 'G-Images zna skończoną listę przeróbek. Jedna zajmuje ~35 s.',
};

const FALLBACK_MODELS = [
  {id: 'g-micro', name: 'G-Micro', desc: 'rozmowa', available: true},
  {id: 'g-images', name: 'G-Images', desc: 'edycja zdjęć', available: true, needs_image: true},
];

// ----------------------------------------------------------------- client --
/**
 * This device's identity. Made here, on first visit, and never shown to anyone.
 *
 * It is what replaces a login. Everything this browser sends lives under this
 * id and nothing else can reach it: the database rules let only the Mac list
 * what ids exist, so there is nothing to enumerate, and the id itself is 128
 * bits from the platform's cryptographic generator. Two people on the open
 * address get two separate spaces without either of them typing a password.
 *
 * Clearing site data means a new id — a fresh space, with the old one
 * abandoned. That is the same trade as clearing cookies anywhere else.
 */
const CLIENT_KEY = 'gmicro.client';

function resolveClient() {
  const saved = (localStorage.getItem(CLIENT_KEY) || '').trim();
  if (saved.length >= 20) return saved;
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  const id = [...bytes].map((b) => b.toString(36).padStart(2, '0')).join('').slice(0, 26);
  localStorage.setItem(CLIENT_KEY, id);
  return id;
}

const client = resolveClient();

// ------------------------------------------------------------------ state --
let messages = [];
let currentConvId = null;
let generating = false;
let active = null;                 // the running job's handle, for cancelling
let macOnline = false;

const chat = new ChatView($('#messages'));
const inputField = $('#input-field');
const sendBtn = $('#send-btn');

const bridge = new MacBridge(client, {onPresence: setPresence});

const picker = new ModelPicker({onChange: (id) => {
  $('#composer-hint').textContent = HINTS[id] || '';
  updatePlaceholder();
}});
picker.setModels(FALLBACK_MODELS);

const attachment = new Attachment({onChange: (url) => {
  if (url && picker.current !== 'g-images') picker.select('g-images');
  updatePlaceholder();
}});

const history = new History({
  sidebarEl: $('#sidebar'),
  scrimEl: $('#sidebar-scrim'),
  listEl: $('#sidebar-list'),
  onOpenConversation: (id) => loadConversation(id),
});

// The composer floats over the transcript and grows when a photo is attached,
// so the room to leave under the last message is measured, not guessed.
const composerEl = $('#input-composer');
new ResizeObserver(() => {
  $('#conversation-pane').style.setProperty('--composer-h', `${composerEl.offsetHeight}px`);
}).observe(composerEl);

// --------------------------------------------------------------- presence --
function setPresence(online, models) {
  macOnline = online;
  // The Mac publishes what it can actually run, so a missing G-Images
  // checkpoint greys the entry out here exactly as it does on the desktop.
  if (models?.length) picker.setModels(models);
  const el = $('#presence');
  el.textContent = online ? 'Mac online' : 'Mac śpi';
  el.classList.toggle('offline', !online);
  updatePlaceholder();
  sendBtn.classList.toggle('disabled', !inputField.value.trim() || generating || !online);
}

function updatePlaceholder() {
  inputField.placeholder = !macOnline
    ? 'MacBook jest wyłączony albo śpi'
    : picker.current !== 'g-images' ? 'Napisz wiadomość…'
    : attachment.dataUrl ? 'Co mam zmienić? Np. „zrób to czarno-białe”'
    : 'Dodaj zdjęcie spinaczem obok';
}

// ------------------------------------------------------------------- chat --
function setComposerMode(mode) {
  const stopping = mode === 'stop';
  sendBtn.classList.toggle('stopping', stopping);
  sendBtn.setAttribute('aria-label', stopping ? 'Zatrzymaj' : 'Wyślij');
  sendBtn.innerHTML = stopping
    ? '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2.5" fill="currentColor"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none"><path d="M12 19V5M12 5l-6 6M12 5l6 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
}

function sendMessage() {
  const text = inputField.value.trim();
  if (!text || generating || !macOnline) return;
  const model = picker.current;
  const image = model === 'g-images' ? attachment.dataUrl : null;

  inputField.value = '';
  autoGrow();
  sendBtn.classList.add('disabled');

  messages.push({role: 'user', text, image});
  chat.addUserMessage(text, image);
  chat.beginAssistantMessage();
  generating = true;
  setComposerMode('stop');

  const past = [];
  for (let i = messages.length - 3; i >= 0 && past.length < 1; i -= 2) {
    const u = messages[i], a = messages[i + 1];
    if (u?.role === 'user' && a?.role === 'assistant' && a.text) {
      past.unshift({user: u.text, assistant: a.text});
    }
  }

  // The whole reply is re-rendered on every update rather than appended to.
  // Each write from the Mac carries the full text, which is what makes a lost
  // or out-of-order message harmless — but it means the view has to be
  // replaced, not extended.
  let shown = '';
  let gotImage = false;
  active = bridge.run({model, text, image, history: past}, (out) => {
    if (typeof out.text === 'string' && out.text !== shown) {
      chat.appendToken(out.text.slice(shown.length));
      shown = out.text;
    }
    if (typeof out.progress === 'number' && !gotImage) chat.setProgress(out.progress);
    if (out.image && !gotImage) {
      gotImage = true;
      chat.addImage(out.image, out.label || '');
      attachment.adopt(out.image, `wynik: ${out.label || 'edycja'}`);
    }
    if (out.done) finishReply(shown, out.image, out.label);
  });
}

function finishReply(text, image, label) {
  if (!generating) return;
  generating = false;
  active = null;
  setComposerMode('send');
  chat.endAssistantMessage();
  messages.push({role: 'assistant', text, image: image || null, imageLabel: label || ''});
  persist();
}

async function persist() {
  if (!messages.length) return;
  currentConvId = await store.save({
    id: currentConvId,
    title: messages[0]?.text?.slice(0, 60) || 'Bez tytułu',
    updated: Date.now(),
    messages,
  });
  history.setActive(currentConvId);
  if (history.open) history.refresh();
}

async function loadConversation(id) {
  const conv = await store.load(id);
  if (!conv) return;
  currentConvId = id;
  messages = conv.messages || [];
  chat.clear();
  for (const m of messages) {
    if (m.role === 'user') chat.addUserMessage(m.text, m.image);
    else {
      chat.beginAssistantMessage();
      chat.appendToken(m.text);
      if (m.image) chat.addImage(m.image, m.imageLabel || '');
      chat.endAssistantMessage();
    }
  }
}

function startNewConversation() {
  currentConvId = null;
  messages = [];
  chat.clear();
  fillSuggestions();
  history.setActive(null);
}

function fillSuggestions() {
  const host = $('#suggestions');
  if (!host) return;
  const pool = [...SUGGESTIONS];
  const pick = [];
  while (pick.length < 4 && pool.length) {
    pick.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
  }
  host.innerHTML = pick.map((s) => `<button class="suggestion" data-text="${s}">${s}</button>`).join('');
}

// ------------------------------------------------------------------ input --
function autoGrow() {
  inputField.style.height = 'auto';
  inputField.style.height = `${Math.min(160, inputField.scrollHeight)}px`;
}

inputField.addEventListener('input', () => {
  autoGrow();
  sendBtn.classList.toggle('disabled', !inputField.value.trim() || generating || !macOnline);
});
inputField.addEventListener('keydown', (e) => {
  // Enter sends on a keyboard, but on a phone it should insert a newline —
  // there is no shift key in reach and the send button is right there.
  if (e.key === 'Enter' && !e.shiftKey && !matchMedia('(pointer: coarse)').matches) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener('click', () => {
  if (generating) { active?.cancel(); return; }
  sendMessage();
});

$('#messages').addEventListener('click', (e) => {
  const chip = e.target.closest('.suggestion');
  if (!chip || generating || !macOnline) return;
  inputField.value = chip.dataset.text;
  autoGrow();
  sendMessage();
});

$('#history-trigger').addEventListener('click', () => history.toggle());
$('#sidebar-new').addEventListener('click', startNewConversation);
$('#new-trigger').addEventListener('click', startNewConversation);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (generating) active?.cancel();
    else if (history.open) history.close();
  }
});

// ------------------------------------------------------------------- boot --
fillSuggestions();
setPresence(false);
