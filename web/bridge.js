/**
 * Browser side of the link to the Mac.
 *
 * There is no server here in the usual sense. This page writes a job into a
 * Realtime Database, the Mac at home notices it, runs the model, and writes
 * the answer back token by token; both ends only ever make outgoing
 * connections, so nothing at home is exposed to the network.
 *
 * The room id in the URL fragment is the whole of the access control. It never
 * reaches the database as a query — it *is* the path — and the rules refuse
 * anything shorter than twenty characters, so rooms cannot be guessed or
 * enumerated. Treat the link like a password.
 */

import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js';
import {
  getDatabase, ref, push, set, remove, onValue, off,
} from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js';

const CONFIG = {
  apiKey: 'AIzaSyCSoFFWPm9Ts1iLlFkPUuicmIfT_wFrAss',
  authDomain: 'g-micro-web.firebaseapp.com',
  databaseURL: 'https://g-micro-web-default-rtdb.firebaseio.com',
  projectId: 'g-micro-web',
  appId: '1:86314193446:web:76c9d08c2c0c40ad420ce7',
};

// The Mac writes a heartbeat every 20 s. Three missed beats is a clear enough
// signal that the lid is shut, without calling it offline over one hiccup.
const STALE_AFTER = 70_000;

export class MacBridge {
  constructor(room, {onPresence}) {
    this.room = room;
    this.db = getDatabase(initializeApp(CONFIG));
    this.onPresence = onPresence;
    this.online = false;
    this.lastBeat = 0;

    onValue(ref(this.db, `rooms/${room}/mac`), (snap) => {
      const v = snap.val() || {};
      this.lastBeat = v.at || 0;
      this.models = v.models || null;
      this._setOnline(Boolean(v.online) && Date.now() - this.lastBeat < STALE_AFTER);
    });
    // A Mac that goes to sleep stops writing rather than writing "offline", so
    // presence has to expire on a timer as well as on a value change.
    setInterval(() => {
      if (this.online && Date.now() - this.lastBeat > STALE_AFTER) this._setOnline(false);
    }, 10_000);
  }

  _setOnline(value) {
    if (value === this.online) return;
    this.online = value;
    this.onPresence?.(value, this.models);
  }

  /**
   * Post a job and stream its answer.
   *
   * `onUpdate` receives the whole accumulated state each time, not a delta —
   * the Mac writes the full value on purpose, so a dropped or reordered
   * message can never leave half a sentence on screen.
   */
  run(job, onUpdate) {
    const jobRef = push(ref(this.db, `rooms/${this.room}/jobs`));
    const id = jobRef.key;
    const outRef = ref(this.db, `rooms/${this.room}/out/${id}`);

    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      off(outRef);
      remove(outRef);
    };

    onValue(outRef, (snap) => {
      const v = snap.val();
      if (!v) return;
      onUpdate(v);
      if (v.done) finish();
    });

    set(jobRef, {
      model: job.model,
      text: job.text,
      at: Date.now(),
      ...(job.image ? {image: job.image} : {}),
      ...(job.history?.length ? {history: job.history} : {}),
    }).catch((e) => {
      onUpdate({text: `Nie mogę wysłać zadania: ${e.message}`, done: true});
      finish();
    });

    return {
      id,
      // Written *into* the job rather than somewhere else, because the Mac is
      // already streaming that node — a cancel lands mid-generation instead of
      // waiting for the next poll.
      cancel: () => set(ref(this.db, `rooms/${this.room}/jobs/${id}/cancel`), true),
    };
  }
}
