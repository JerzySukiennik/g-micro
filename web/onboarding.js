/**
 * First-run introduction: three screens, skippable, re-openable from the menu.
 *
 * The middle screen is the one that matters. This model invents facts — it will
 * state a wrong year with complete confidence, and it does not know it is
 * wrong. Someone meeting it for the first time has no way to know that, and
 * finding out by being misled is worse than being told up front. A launch
 * screen that only listed features would be lying by omission.
 *
 * The last screen asks for a name. It is stored locally and used only by the
 * welcome screen — the model never sees it. Sending it would spend context
 * window on something the model was never trained to use, and it would use it
 * erratically or not at all.
 */

const KEY = 'gmicro.onboarding';

export const SCREENS = [
  {
    title: 'Cześć, jestem G-Micro.',
    body: 'Mały model językowy, który mówi po polsku. Powstał od zera — nie jest '
        + 'przerobioną wersją niczego innego. Działa w całości na tym komputerze, '
        + 'bez internetu, i nic nie wysyła na żaden serwer.',
    tag: 'czym jestem',
  },
  {
    title: 'Mylę się. Często.',
    body: 'Jestem bardzo mały, więc fakty, daty i liczby regularnie zmyślam — '
        + 'i robię to z pełnym przekonaniem, bo nie wiem, kiedy się mylę. '
        + 'Traktuj to jak rozmowę z kimś, kto dużo czyta i wszystko myli. '
        + 'Nie sprawdzaj u mnie niczego, na czym Ci zależy.',
    tag: 'czego nie oczekiwać',
  },
  {
    title: 'Jak mam się do Ciebie zwracać?',
    body: 'Imię zostaje na tym komputerze i służy tylko do powitania. '
        + 'Możesz pominąć.',
    tag: 'ostatnia rzecz',
    askName: true,
  },
];

export class Onboarding {
  /**
   * `screens` overrides the copy. The web build needs it: there the model runs
   * on a Mac somewhere else and the text travels through a database to reach
   * it, so the first screen's promise that nothing leaves this computer would
   * be a lie. Same wizard, honest wording in each place.
   */
  constructor({onFinish, screens}) {
    this.onFinish = onFinish;
    this.screens = screens || SCREENS;
    this.index = 0;
    this.el = null;
  }

  static seen() {
    try { return Boolean(JSON.parse(localStorage.getItem(KEY) || '{}').done); }
    catch { return false; }
  }

  static name() {
    try { return JSON.parse(localStorage.getItem(KEY) || '{}').name || ''; }
    catch { return ''; }
  }

  static save(patch) {
    let cur = {};
    try { cur = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { /* corrupt, start over */ }
    localStorage.setItem(KEY, JSON.stringify({...cur, ...patch}));
  }

  open() {
    this.index = 0;
    this.el = document.createElement('div');
    this.el.id = 'onboarding';
    document.body.appendChild(this.el);
    requestAnimationFrame(() => this.el.classList.add('visible'));
    this._render();
    document.addEventListener('keydown', this._onKey);
  }

  _onKey = (e) => {
    if (!this.el) return;
    if (e.key === 'Escape') { e.preventDefault(); this._finish(); }
    else if (e.key === 'Enter') { e.preventDefault(); this._next(); }
  };

  _render() {
    const s = this.screens[this.index];
    const last = this.index === this.screens.length - 1;
    this.el.innerHTML = `
      <div class="ob-card">
        <div class="ob-tag">${s.tag}</div>
        <h2>${s.title}</h2>
        <p>${s.body}</p>
        ${s.askName ? '<input id="ob-name" type="text" placeholder="Twoje imię" maxlength="24" autocomplete="off" spellcheck="false">' : ''}
        <div class="ob-foot">
          <div class="ob-dots">${this.screens.map((_, i) =>
            `<span class="${i === this.index ? 'on' : ''}"></span>`).join('')}</div>
          <div class="ob-actions">
            <button class="ob-skip" type="button">${last ? 'Pomiń' : 'Pomiń wprowadzenie'}</button>
            <button class="ob-next" type="button">${last ? 'Zaczynamy' : 'Dalej'}</button>
          </div>
        </div>
      </div>`;
    this.el.querySelector('.ob-next').addEventListener('click', () => this._next());
    this.el.querySelector('.ob-skip').addEventListener('click', () => this._finish());
    this.el.querySelector('#ob-name')?.focus();
  }

  _next() {
    const input = this.el?.querySelector('#ob-name');
    if (input) Onboarding.save({name: input.value.trim().slice(0, 24)});
    if (this.index < this.screens.length - 1) {
      this.index += 1;
      this._render();
    } else {
      this._finish();
    }
  }

  _finish() {
    Onboarding.save({done: true});
    document.removeEventListener('keydown', this._onKey);
    const el = this.el;
    this.el = null;
    el?.classList.remove('visible');
    setTimeout(() => el?.remove(), 260);
    this.onFinish?.(Onboarding.name());
  }
}
