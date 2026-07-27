/**
 * Conversation rendering — no bubbles, a transcript. UI-SPEC.md "Rozmowa":
 * user text dim and indented, model text full white, tokens arrive one at a
 * time and flash brighter than resting white before settling — ties the
 * text to the pulse in the neuron panel next to it.
 */

export class ChatView {
  constructor(container) {
    this.container = container;
    this.currentAssistantEl = null;
    this.currentTextNode = null;
    // The welcome block ships in index.html rather than being built here, so
    // the copy and the suggestions live with the rest of the markup. Keeping
    // a copy of it means an emptied conversation can show it again instead of
    // leaving a blank page. Suggestion clicks are handled by delegation in
    // app.js precisely because this markup gets re-inserted.
    this.welcomeHTML = container.innerHTML;
  }

  clear() {
    this.container.innerHTML = this.welcomeHTML;
    this.currentAssistantEl = null;
  }

  addUserMessage(text) {
    document.getElementById('welcome')?.remove();
    const el = document.createElement('div');
    el.className = 'msg-user';
    el.textContent = text;
    this.container.appendChild(el);
    this._scrollToBottom();
  }

  beginAssistantMessage() {
    const el = document.createElement('div');
    el.className = 'msg-assistant';
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    el.appendChild(cursor);
    this.container.appendChild(el);
    this.currentAssistantEl = el;
    this._cursor = cursor;
    this._scrollToBottom();
  }

  /** Names the source RAG injected, above the answer it produced.
   *
   * Without this the retrieved text is invisible: on 2026-07-27 a bad hit
   * (an Argentine film matched to "Jak masz na imię?") made a perfectly
   * healthy model look broken, and nothing on screen hinted that an outside
   * document was in play at all. */
  setContextSource(source) {
    if (!this.currentAssistantEl || !source) return;
    const tag = document.createElement('div');
    tag.className = 'msg-context-source';
    tag.textContent = `kontekst: ${source}`;
    this.currentAssistantEl.insertBefore(tag, this.currentAssistantEl.firstChild);
    this._scrollToBottom();
  }

  /** Appends one token's text with a brief brighten-then-settle flash. */
  appendToken(text) {
    if (!this.currentAssistantEl || !text) return;
    const span = document.createElement('span');
    span.className = 'tok';
    span.textContent = text;
    span.style.color = 'var(--ink-full)';
    this.currentAssistantEl.insertBefore(span, this._cursor);
    // Two-frame settle rather than a CSS transition: guarantees the flash
    // paints before easing back, even under load.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      span.style.transition = 'color 420ms ease-out';
      span.style.color = 'var(--ink)';
    }));
    this._scrollToBottom();
  }

  endAssistantMessage() {
    this._cursor?.remove();
    this.currentAssistantEl = null;
  }

  showError(message, onRetry) {
    const el = document.createElement('div');
    el.className = 'msg-error';
    el.innerHTML = `<span></span><button>Retry</button>`;
    el.querySelector('span').textContent = message;
    el.querySelector('button').addEventListener('click', () => { el.remove(); onRetry(); });
    this.container.appendChild(el);
    this._scrollToBottom();
  }

  _scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }
}
