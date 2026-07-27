/**
 * The Probabilities panel — top-10 next-token candidates, springing to new
 * ranks as they change, plus the confidence meter (UI-SPEC.md "Zakładka 2").
 *
 * Rows are keyed by token id, not decoded text: several distinct BPE tokens
 * (whitespace variants, control tokens) can decode to identical or empty
 * strings, and keying by text would let two different candidates fight over
 * one DOM row and spring — which is exactly what happened the first time
 * this ran against a live model. The backend sends the id alongside the
 * text for this reason.
 */

import { SpringLoop, SPRING } from './spring.js';

const ROW_H = 26;
const ROW_GAP = 6;

export class ProbabilitiesView {
  constructor(container, confidenceFillEl, confidenceStateEl) {
    this.container = container;
    this.confidenceFill = confidenceFillEl;
    this.confidenceState = confidenceStateEl;
    this.rows = new Map(); // token id -> { el, y, w }
    this.loop = new SpringLoop(() => this._render());
  }

  update({ top, entropy_norm, label }) {
    const width = this.container.clientWidth;
    const maxProb = top.length ? top[0][1] : 1;

    const seen = new Set();
    top.forEach(([token, prob, id], rank) => {
      seen.add(id);
      let row = this.rows.get(id);
      if (!row) {
        const el = document.createElement('div');
        el.className = 'prob-row';
        el.innerHTML = `<span class="rank mono"></span><span class="token"></span>
                        <span class="track"><span class="fill"></span></span>
                        <span class="pct mono"></span>`;
        this.container.appendChild(el);
        row = { el, y: this.loop.spring(`p${id}-y`, rank * (ROW_H + ROW_GAP), SPRING.default),
               w: this.loop.spring(`p${id}-w`, 0, SPRING.default) };
        this.rows.set(id, row);
      }
      row.fading = false;
      row.y.set(rank * (ROW_H + ROW_GAP));
      row.w.set(Math.max(0, (prob / maxProb) * (width - 178)));
      row.el.querySelector('.rank').textContent = rank + 1;
      row.el.querySelector('.token').textContent = token || '·';
      row.el.querySelector('.pct').textContent = `${(prob * 100).toFixed(1)}%`;
    });

    // Candidates that fell out of the top-10 spring off the bottom and get
    // culled once settled, rather than disappearing instantly.
    for (const [id, row] of this.rows) {
      if (!seen.has(id)) {
        row.y.set(10 * (ROW_H + ROW_GAP));
        row.w.set(0);
        row.fading = true;
      }
    }

    this.loop.start();

    this.confidenceFill.style.width = `${entropy_norm * 100}%`;
    this.confidenceState.textContent = label;
  }

  _render() {
    for (const [id, row] of this.rows) {
      row.el.style.transform = `translateY(${row.y.value}px)`;
      row.el.querySelector('.fill').style.width = `${Math.max(0, row.w.value)}px`;
      if (row.fading && row.w.settled && row.w.value <= 0) {
        row.el.remove();
        this.rows.delete(id);
        this.loop.remove(`p${id}-y`);
        this.loop.remove(`p${id}-w`);
      }
    }
  }

  reset() {
    for (const [id, row] of this.rows) {
      row.el.remove();
      this.loop.remove(`p${id}-y`);
      this.loop.remove(`p${id}-w`);
    }
    this.rows.clear();
    this.confidenceFill.style.width = '0%';
    this.confidenceState.textContent = '—';
  }
}
