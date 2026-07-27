/**
 * Settings popover — temperature/top-k (UI-SPEC.md: "schowane pod jednym
 * przyciskiem") and the RAG toggle (UI-SPEC.md risk #1: off by default,
 * because a 110M model will confabulate and hiding sources by default means
 * you can't tell the difference — this switch is the escape hatch).
 */

const STORAGE_KEY = 'microg.settings';

const defaults = { temperature: 0.8, top_k: 50, rag: false };

export class Settings {
  constructor(popoverEl) {
    this.popover = popoverEl;
    this.values = { ...defaults, ...this._load() };
    this.tempRange = popoverEl.querySelector('#temp-range');
    this.tempVal = popoverEl.querySelector('#temp-val');
    this.topkRange = popoverEl.querySelector('#topk-range');
    this.topkVal = popoverEl.querySelector('#topk-val');
    this.ragToggle = popoverEl.querySelector('#rag-toggle');

    this.tempRange.value = this.values.temperature;
    this.tempVal.textContent = this.values.temperature.toFixed(2);
    this.topkRange.value = this.values.top_k;
    this.topkVal.textContent = this.values.top_k;
    this.ragToggle.classList.toggle('on', this.values.rag);

    this.tempRange.addEventListener('input', () => {
      this.values.temperature = parseFloat(this.tempRange.value);
      this.tempVal.textContent = this.values.temperature.toFixed(2);
      this._save();
    });
    this.topkRange.addEventListener('input', () => {
      this.values.top_k = parseInt(this.topkRange.value, 10);
      this.topkVal.textContent = this.values.top_k;
      this._save();
    });
    this.ragToggle.addEventListener('click', () => {
      this.values.rag = !this.values.rag;
      this.ragToggle.classList.toggle('on', this.values.rag);
      this._save();
    });

    document.addEventListener('click', (e) => {
      if (this.isOpen && !this.popover.contains(e.target) && e.target.id !== 'settings-trigger') {
        this.close();
      }
    });
  }

  get isOpen() { return this.popover.classList.contains('open'); }

  toggle(anchorRect) {
    this.isOpen ? this.close() : this.open(anchorRect);
  }

  open(anchorRect) {
    // Anchored to its trigger, not centred — apple-design: popovers scale
    // from the element that opened them so the spatial relationship reads.
    this.popover.style.top = `${anchorRect.bottom + 8}px`;
    this.popover.style.right = `${window.innerWidth - anchorRect.right}px`;
    this.popover.style.transform = 'scale(0.92) translateY(-4px)';
    this.popover.classList.add('open');
    requestAnimationFrame(() => {
      this.popover.style.transition = 'opacity 220ms ease-out, transform 220ms cubic-bezier(.2,.8,.2,1)';
      this.popover.style.opacity = '1';
      this.popover.style.transform = 'scale(1) translateY(0)';
    });
  }

  close() {
    this.popover.style.opacity = '0';
    this.popover.style.transform = 'scale(0.92) translateY(-4px)';
    this.popover.classList.remove('open');
  }

  _load() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
    catch { return {}; }
  }

  _save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.values));
  }
}
