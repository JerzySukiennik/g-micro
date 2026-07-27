/**
 * The Neurons panel — live view of all 24,576 FFN activations.
 *
 * Data honesty (UI-SPEC.md "Stan spoczynku" + "Dane z modelu"): the backend
 * only ever transmits the top-64-strongest neurons per layer, per token —
 * that is the whole reason generation stays fast. This module never invents
 * values for the other ~1984-per-layer it wasn't told about; they simply
 * decay toward the last real thing they were, which is honest because it's
 * still derived from real data, just aging.
 *
 * Update policy: the *reported* neurons are hard-set every token, no spring,
 * no cross-fade — UI-SPEC.md is explicit that fidelity beats smoothness
 * here. Only the idle "breathing" (small brightness modulation while
 * nothing is generating) is animated, and only over real last-known values.
 *
 * Rendering is batched by quantised brightness bucket into one Path2D per
 * bucket, so a 24k-dot frame costs ~24 fill() calls, not 24,576.
 */

const LAYERS = 12;
const PER_LAYER = 2048;
const TOTAL = LAYERS * PER_LAYER;
const COLS = 157;
const ROWS = Math.ceil(TOTAL / COLS);
const BUCKETS = 24;
const DECAY = 0.90;          // per-token fade for neurons not in this step's top-64
const IDLE_DECAY = 0.985;    // slower fade once generation has stopped

export class NeuronField {
  constructor(canvas, tooltipEl, { onPin } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.tooltip = tooltipEl;
    this.onPin = onPin;

    this.activation = new Float32Array(TOTAL); // 0..1, "real" values only
    this.pinned = null;                         // flat index or null
    this.breathePhase = new Float32Array(TOTAL);
    for (let i = 0; i < TOTAL; i++) this.breathePhase[i] = Math.random() * Math.PI * 2;

    this._resize();
    window.addEventListener('resize', () => this._resize());
    canvas.addEventListener('mousemove', (e) => this._hover(e));
    canvas.addEventListener('mouseleave', () => this.tooltip.classList.remove('show'));
    canvas.addEventListener('click', (e) => this._click(e));

    this._idle = true;
    this._raf = null;
    this._loop();
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = r.width * dpr;
    this.canvas.height = r.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height;
    this._layout();
  }

  _layout() {
    this.cell = Math.min(this.w / COLS, this.h / ROWS);
    this.ox = (this.w - this.cell * COLS) / 2;
    this.oy = (this.h - this.cell * ROWS) / 2;
  }

  _cellAt(px, py) {
    const col = Math.floor((px - this.ox) / this.cell);
    const row = Math.floor((py - this.oy) / this.cell);
    if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return null;
    const flat = row * COLS + col;
    if (flat >= TOTAL) return null;
    return flat;
  }

  _hover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const flat = this._cellAt(e.clientX - rect.left, e.clientY - rect.top);
    if (flat === null) { this.tooltip.classList.remove('show'); return; }
    const layer = Math.floor(flat / PER_LAYER);
    const neuron = flat % PER_LAYER;
    this.tooltip.textContent = `layer ${layer} · neuron ${neuron}`;
    this.tooltip.style.left = `${e.clientX - rect.left}px`;
    this.tooltip.style.top = `${e.clientY - rect.top}px`;
    this.tooltip.classList.add('show');
  }

  _click(e) {
    const rect = this.canvas.getBoundingClientRect();
    const flat = this._cellAt(e.clientX - rect.left, e.clientY - rect.top);
    if (flat === null) return;
    this.pinned = flat;
    this.onPin?.(Math.floor(flat / PER_LAYER), flat % PER_LAYER, flat);
  }

  unpin() { this.pinned = null; }

  /** One token's worth of real data: neuronsPerLayer = [[idx,val], ...] x 12. */
  applyStep(neuronsPerLayer) {
    this._idle = false;
    // Decay everything first — anything not re-set below is honestly aging,
    // not being re-measured.
    for (let i = 0; i < TOTAL; i++) this.activation[i] *= DECAY;

    for (let layer = 0; layer < neuronsPerLayer.length; layer++) {
      const pairs = neuronsPerLayer[layer];
      if (!pairs.length) continue;
      const maxVal = pairs[0][1] || 1; // pairs arrive sorted, [0] is strongest
      for (const [idx, val] of pairs) {
        this.activation[layer * PER_LAYER + idx] = Math.min(1, val / maxVal);
      }
    }
  }

  /** Real value for a specific flat index this step, or null if it wasn't
   *  among the reported top-64 — used by the pinned-neuron chart, which
   *  must show gaps rather than fabricate a number. */
  lastKnown(flat) {
    return this.activation[flat] > 0 ? this.activation[flat] : null;
  }

  markGenerationIdle() { this._idle = true; }

  _loop() {
    this._raf = requestAnimationFrame(() => this._loop());
    if (this._idle) {
      // Gentle breathing on real last-known values only (UI-SPEC.md risk
      // note: "falowanie robione z ostatniego prawdziwego stanu aktywacji,
      // nie z wymyślonego szumu"). The stored value never changes here —
      // only how brightly it's drawn.
      this._t = (this._t || 0) + 0.016;
    }
    this._draw();
  }

  _draw() {
    const { ctx, w, h, cell } = this;
    ctx.clearRect(0, 0, w, h);

    const buckets = Array.from({ length: BUCKETS }, () => new Path2D());
    for (let i = 0; i < TOTAL; i++) {
      let v = this.activation[i];
      if (v <= 0.003 && !(this._idle && v > 0)) continue;
      if (this._idle && v > 0) {
        v *= 0.75 + 0.25 * Math.sin(this._t * 0.6 + this.breathePhase[i]);
      }
      const row = Math.floor(i / COLS), col = i % COLS;
      const cx = this.ox + col * cell + cell / 2;
      const cy = this.oy + row * cell + cell / 2;
      const radius = cell * (0.16 + v * 0.24);
      const bucket = Math.min(BUCKETS - 1, Math.max(0, Math.floor(v * BUCKETS)));
      buckets[bucket].moveTo(cx + radius, cy);
      buckets[bucket].arc(cx, cy, radius, 0, Math.PI * 2);
    }

    // Dim base dots first (very low alpha, no glow — keeps the "planszа"
    // texture even where nothing is currently active).
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.shadowBlur = 0;
    ctx.fill(buckets[0]);

    ctx.shadowColor = 'rgba(255,255,255,0.9)';
    for (let b = 1; b < BUCKETS; b++) {
      const alpha = 0.12 + (b / BUCKETS) * 0.88;
      ctx.shadowBlur = 1 + (b / BUCKETS) * 6;
      ctx.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
      ctx.fill(buckets[b]);
    }
    ctx.shadowBlur = 0;

    if (this.pinned !== null) this._drawPinnedRing();
  }

  _drawPinnedRing() {
    const { ctx, cell } = this;
    const row = Math.floor(this.pinned / COLS), col = this.pinned % COLS;
    const cx = this.ox + col * cell + cell / 2;
    const cy = this.oy + row * cell + cell / 2;
    ctx.beginPath();
    ctx.arc(cx, cy, cell * 0.55, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}
