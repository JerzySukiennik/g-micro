/**
 * Wake — the startup sequence and the app's one signature moment.
 *
 * Per UI-SPEC.md: 24,576 neurons light up layer by layer, in the rhythm of
 * the *real* checkpoint load — not a fabricated progress bar. Each `wake`
 * event from the backend (see runtime/server.py: Backend.load) corresponds
 * to one transformer block that has genuinely just finished loading; this
 * module only decides how to reveal that fact, never invents it.
 */

const LAYERS = 12;
const NEURONS_PER_LAYER = 2048;
const COLS = 157; // 157*157 = 24,649, comfortably covers 24,576

export class Wake {
  constructor(canvas, onDone) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onDone = onDone;
    this.litLayers = 0; // -1 slot (embeddings) counts as "layer 0 lit"
    this.embeddingsLit = false;
    this._resize();
    window.addEventListener('resize', () => this._resize());
    this._raf = null;
    this._draw();
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = r.width * dpr;
    this.canvas.height = r.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height;
  }

  /** Called once per backend "wake" message. layer = -1 means embeddings. */
  onWakeEvent(layer) {
    if (layer === -1) this.embeddingsLit = true;
    else this.litLayers = Math.max(this.litLayers, layer + 1);
    this._draw();
  }

  finish() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this.onDone?.();
  }

  _draw() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);

    const size = Math.min(w, h) * 0.62;
    const ox = (w - size) / 2, oy = (h - size) / 2;
    const cell = size / COLS;

    const litRows = this.embeddingsLit
      ? Math.ceil((this.litLayers / LAYERS) * COLS)
      : 0;

    for (let row = 0; row < COLS; row++) {
      const lit = row < litRows;
      // deterministic pseudo-random brightness per cell, same trick as the
      // logo mark — gives the grid organic variation instead of a flat wash
      for (let col = 0; col < COLS; col++) {
        const seed = Math.sin(row * 12.9898 + col * 78.233) * 43758.5453;
        const t = seed - Math.floor(seed);
        if (!lit && t > 0.15) continue; // sparse dim dots when unlit
        const brightness = lit ? 0.5 + t * 0.5 : 0.03 + t * 0.05;
        const r = cell * 0.28 * (lit ? 0.7 + t * 0.3 : 0.5);
        ctx.beginPath();
        ctx.arc(ox + col * cell + cell / 2, oy + row * cell + cell / 2, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${brightness})`;
        ctx.fill();
      }
    }
  }
}
