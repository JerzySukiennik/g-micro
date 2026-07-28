/**
 * Conversation rendering — no bubbles, a transcript. UI-SPEC.md "Rozmowa":
 * user text dim and indented, model text full white, tokens arrive one at a
 * time and flash brighter than resting white before settling — ties the
 * text to the pulse in the neuron panel next to it.
 */

import {renderMarkdown} from './format.js';

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

  addUserMessage(text, image) {
    document.getElementById('welcome')?.remove();
    const el = document.createElement('div');
    el.className = 'msg-user';
    if (image) {
      const img = document.createElement('img');
      img.className = 'msg-photo';
      img.alt = 'załączone zdjęcie';
      // Scroll again once it has decoded: the image has no height at append
      // time, so the first scroll lands short by exactly the picture.
      img.addEventListener('load', () => this._scrollToBottom());
      img.src = image;
      el.appendChild(img);
    }
    const line = document.createElement('div');
    line.textContent = text;
    el.appendChild(line);
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

  /**
   * G-Images progress, counted from real forward passes through the network.
   *
   * A spinner would be dishonest here for a different reason than usual: the
   * wait is around thirty-five seconds, long enough that an indeterminate
   * animation reads as a hang. The bar is the only thing separating "working"
   * from "broken" during a wait that long.
   */
  setProgress(p) {
    if (!this.currentAssistantEl) return;
    let bar = this.currentAssistantEl.querySelector('.msg-progress');
    if (!bar) {
      bar = document.createElement('div');
      bar.className = 'msg-progress';
      bar.innerHTML = '<div class="msg-progress-fill"></div>';
      this.currentAssistantEl.insertBefore(bar, this._cursor);
    }
    bar.querySelector('.msg-progress-fill').style.width = `${Math.round(p * 100)}%`;
    this._scrollToBottom();
  }

  /** The edited photo, captioned with the edit type that was actually applied
   *  — the mapping from a sentence to a type is a guess made on this side, so
   *  it has to be visible and checkable rather than silent. */
  addImage(dataUrl, label) {
    if (!this.currentAssistantEl) return;
    this.currentAssistantEl.querySelector('.msg-progress')?.remove();
    const fig = document.createElement('figure');
    fig.className = 'msg-figure';
    const img = document.createElement('img');
    img.alt = label || 'wynik';
    img.addEventListener('load', () => this._scrollToBottom());
    img.src = dataUrl;
    const cap = document.createElement('figcaption');
    cap.textContent = label || '';
    fig.append(img, cap);
    this.currentAssistantEl.insertBefore(fig, this._cursor);
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

  /**
   * Finish a reply: swap the streamed spans for formatted output and attach a
   * copy button.
   *
   * Formatting deliberately happens here rather than per token. The flash as
   * each token lands is the app's one piece of genuine product behaviour, and
   * re-parsing a half-finished sentence on every token would both fight that
   * animation and render markup from syntax the model had not finished
   * writing. Streaming stays raw; the finished text gets laid out once.
   */
  endAssistantMessage() {
    this._cursor?.remove();
    const el = this.currentAssistantEl;
    this.currentAssistantEl = null;
    if (!el) return;

    const source = el.querySelector('.msg-context-source');
    // Figures are moved, not rebuilt: this method replaces the element's whole
    // contents, and a G-Images result lives in the DOM rather than in the token
    // stream, so anything not carried across here would simply vanish.
    const figures = [...el.querySelectorAll('.msg-figure')];
    el.querySelector('.msg-progress')?.remove();
    const raw = [...el.querySelectorAll('.tok')].map((s) => s.textContent).join('');
    if (!raw.trim() && !figures.length) return;

    el.innerHTML = '';
    if (source) el.appendChild(source);

    if (raw.trim()) {
      const body = document.createElement('div');
      body.className = 'msg-body';
      body.innerHTML = renderMarkdown(raw);
      el.appendChild(body);
    }
    figures.forEach((f) => el.appendChild(f));
    if (raw.trim()) el.appendChild(this._copyButton(raw));
  }

  /** Copy control, revealed on hover. Confirms in place rather than with a
   *  toast — the feedback belongs where the click happened. */
  _copyButton(text) {
    const btn = document.createElement('button');
    btn.className = 'msg-copy';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Kopiuj odpowiedź');
    const idle = '<svg viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="11" height="11" rx="2.5" stroke="currentColor" stroke-width="1.7"/><path d="M5 15V5.5A1.5 1.5 0 0 1 6.5 4H15" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>';
    const done = '<svg viewBox="0 0 24 24" fill="none"><path d="M5 12.5 L10 17.5 L19 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    btn.innerHTML = idle;
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(text);
        btn.innerHTML = done;
        btn.classList.add('copied');
        setTimeout(() => { btn.innerHTML = idle; btn.classList.remove('copied'); }, 1400);
      } catch {
        // Clipboard can be refused; saying nothing would look like the click
        // was ignored.
        btn.setAttribute('aria-label', 'Nie udało się skopiować');
      }
    });
    return btn;
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
