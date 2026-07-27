/**
 * A minimal spring physics engine — the whole animation system.
 *
 * Per UI-SPEC.md "Ruch": everything animates on springs, never fixed-duration
 * CSS transitions. Springs are parameterised the way Apple's design language
 * describes them (see the apple-design skill) — damping ratio + response, not
 * raw stiffness/mass — and always retarget from the value currently on
 * screen, never from the old target. That is what makes an interrupted
 * animation redirect smoothly instead of jumping.
 *
 * damping: 1.0 = critically damped, settles with no overshoot (the default
 *          everywhere). ~0.8 = a little bounce — reserved for places a
 *          gesture actually carried momentum (see apple-design).
 * response: seconds — how quickly it moves, not a fixed duration; the
 *          settle time emerges from the physics.
 */

const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;

export class Spring {
  constructor(value = 0, { damping = 1.0, response = 0.35 } = {}) {
    this.value = value;
    this.target = value;
    this.velocity = 0;
    this.damping = damping;
    this.response = response;
  }

  /** Retarget without ever snapping — starts from wherever the spring is now. */
  set(target, { velocity = null } = {}) {
    this.target = target;
    if (velocity !== null) this.velocity = velocity;
    if (REDUCED_MOTION) { this.value = target; this.velocity = 0; }
  }

  /** Advance by dt seconds. Returns the current value. */
  step(dt) {
    if (REDUCED_MOTION || dt <= 0) { this.value = this.target; return this.value; }
    dt = Math.min(dt, 1 / 30); // clamp huge frame gaps (tab switch, GC pause)

    const mass = 1;
    const stiffness = ((2 * Math.PI) / this.response) ** 2 * mass;
    const dampingCoef = (4 * Math.PI * this.damping * mass) / this.response;

    const displacement = this.value - this.target;
    const springForce = -stiffness * displacement;
    const dampingForce = -dampingCoef * this.velocity;
    const accel = (springForce + dampingForce) / mass;

    this.velocity += accel * dt;
    this.value += this.velocity * dt;
    return this.value;
  }

  get settled() {
    return Math.abs(this.value - this.target) < 0.001 && Math.abs(this.velocity) < 0.001;
  }
}

/**
 * Drives a set of named springs on one requestAnimationFrame loop and calls
 * onFrame each tick until every spring settles. Sharing one rAF loop across
 * many springs (bars re-ranking, sidebar slide, glow fades) is cheaper and
 * keeps everything in visual sync, rather than each element running its own
 * timer.
 */
export class SpringLoop {
  constructor(onFrame) {
    this.springs = new Map();
    this.onFrame = onFrame;
    this._running = false;
    this._last = 0;
    this._tick = this._tick.bind(this);
  }

  spring(key, initial, opts) {
    if (!this.springs.has(key)) this.springs.set(key, new Spring(initial, opts));
    return this.springs.get(key);
  }

  /** Drop a spring once its element is gone — without this, a long chat
   *  session accumulates one dead spring per distinct token that ever
   *  passed through the top-10, and every frame keeps stepping all of them. */
  remove(key) { this.springs.delete(key); }

  start() {
    if (this._running) return;
    this._running = true;
    this._last = performance.now();
    requestAnimationFrame(this._tick);
  }

  _tick(now) {
    const dt = (now - this._last) / 1000;
    this._last = now;
    let allSettled = true;
    for (const s of this.springs.values()) {
      s.step(dt);
      if (!s.settled) allSettled = false;
    }
    this.onFrame();
    if (allSettled) { this._running = false; return; }
    requestAnimationFrame(this._tick);
  }
}

export const SPRING = {
  default: { damping: 1.0, response: 0.35 },
  snappy: { damping: 1.0, response: 0.22 },
  bouncy: { damping: 0.8, response: 0.4 },  // only where a gesture carried momentum
};

export { REDUCED_MOTION };
