/**
 * Conversation storage for the browser.
 *
 * The desktop app keeps conversations as JSON files in userData and reaches
 * them through a preload bridge on `window.gmicro`. Rather than fork the
 * sidebar, this fills in the same shape backed by localStorage — history.js
 * then runs here untouched, and a fix to it lands in both places at once.
 *
 * Conversations hold base64 photos, so this trims the oldest ones once the
 * store passes ~4 MB. localStorage throws when it fills up, and a quota error
 * on save would surface as a conversation that simply refuses to persist with
 * no explanation.
 */

const KEY = 'gmicro.conversations';
const BUDGET = 4_000_000;

function readAll() {
  try { return JSON.parse(localStorage.getItem(KEY) || '[]'); }
  catch { return []; }
}

function writeAll(items) {
  let list = items;
  for (;;) {
    const payload = JSON.stringify(list);
    if (payload.length <= BUDGET || list.length <= 1) {
      try {
        localStorage.setItem(KEY, payload);
        return;
      } catch {
        if (list.length <= 1) return;      // one conversation too big to store
        list = list.slice(0, -1);
        continue;
      }
    }
    list = list.slice(0, -1);              // drop the oldest, keep going
  }
}

export const store = {
  async list() {
    return readAll()
      .map(({id, title, updated}) => ({id, title, updated}))
      .sort((a, b) => b.updated - a.updated);
  },

  async load(id) {
    return readAll().find((c) => c.id === id) || null;
  },

  async save(conv) {
    const items = readAll();
    const id = conv.id || `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
    const next = {...conv, id};
    const at = items.findIndex((c) => c.id === id);
    if (at >= 0) items[at] = next; else items.unshift(next);
    items.sort((a, b) => b.updated - a.updated);
    writeAll(items);
    return id;
  },

  async rename(id, title) {
    const items = readAll();
    const conv = items.find((c) => c.id === id);
    if (conv) { conv.title = title; writeAll(items); }
  },

  async delete(id) {
    writeAll(readAll().filter((c) => c.id !== id));
  },
};

// history.js reaches for this by name; giving it the same surface the desktop
// preload exposes is what lets that file be shared verbatim.
window.gmicro = {...(window.gmicro || {}), history: store};
