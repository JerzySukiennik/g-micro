/**
 * The two controls under the send button: which model answers, and the photo
 * it answers about.
 *
 * They are one module because they are one decision. G-Micro cannot see a
 * photo and G-Images cannot answer a question, so attaching a picture and
 * choosing a model are the same act expressed twice — attaching switches the
 * model, and every hint in the composer follows from which model is live.
 */

const MAX_SIDE = 512;

/**
 * Prepare a file for sending: centre-crop to a square, scale down, re-encode.
 *
 * The server crops and resizes to 128px anyway, so shipping a 4000px phone
 * photo through a WebSocket and into conversation history would be pure waste.
 * Cropping here rather than only on the server also means the thumbnail the
 * user sees is exactly the region the model will be given — a photo whose
 * subject silently fell outside the square would look like the model ignored
 * it.
 */
export function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('nie mogę odczytać pliku'));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error('to nie wygląda na zdjęcie'));
      img.onload = () => {
        const side = Math.min(img.naturalWidth, img.naturalHeight);
        const out = Math.min(side, MAX_SIDE);
        const c = document.createElement('canvas');
        c.width = c.height = out;
        c.getContext('2d').drawImage(
          img,
          (img.naturalWidth - side) / 2, (img.naturalHeight - side) / 2, side, side,
          0, 0, out, out);
        resolve(c.toDataURL('image/jpeg', 0.9));
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

export class Attachment {
  constructor({onChange}) {
    this.onChange = onChange;
    this.dataUrl = null;
    this.el = document.querySelector('#attachment');
    this.thumb = document.querySelector('#attachment-thumb');
    this.nameEl = document.querySelector('#attachment-name');
    this.input = document.querySelector('#file-input');

    document.querySelector('#attach-btn')
      .addEventListener('click', () => this.input.click());
    this.input.addEventListener('change', () => {
      const file = this.input.files?.[0];
      // Reset first: picking the same file twice in a row fires no change
      // event otherwise, so removing a photo and re-adding it would do nothing.
      this.input.value = '';
      if (file) this.set(file);
    });
    document.querySelector('#attachment-remove')
      .addEventListener('click', () => this.clear());

    // Paste and drop, because both are how someone actually gets a screenshot
    // into a window they already have open.
    //
    // `items` rather than `files`: a picture copied from Preview or Finder on
    // macOS arrives as a clipboard *item* and never appears in `files`, so the
    // obvious version of this silently ignored every real paste. Measured —
    // pasting a JPEG did nothing at all until this was widened.
    document.addEventListener('paste', (e) => {
      const items = [...(e.clipboardData?.items || [])];
      const file = items.find((i) => i.kind === 'file' && i.type.startsWith('image/'))
        ?.getAsFile() || [...(e.clipboardData?.files || [])]
        .find((f) => f.type.startsWith('image/'));
      if (file) { e.preventDefault(); this.set(file); }
    });
    const composer = document.querySelector('#input-composer');
    composer.addEventListener('dragover', (e) => {
      e.preventDefault();
      composer.classList.add('dropping');
    });
    composer.addEventListener('dragleave', () => composer.classList.remove('dropping'));
    composer.addEventListener('drop', (e) => {
      e.preventDefault();
      composer.classList.remove('dropping');
      const file = [...(e.dataTransfer?.files || [])][0];
      if (file?.type.startsWith('image/')) this.set(file);
    });
  }

  async set(file) {
    let url;
    try {
      url = await fileToDataURL(file);
    } catch (err) {
      this.nameEl.textContent = err.message;
      return;
    }
    this.adopt(url, file.name || 'zdjęcie');
  }

  /**
   * Attach an image that is already prepared — used to hand a finished edit
   * back in as the source for the next one.
   *
   * Chaining is what people actually do: the reply to "zrób czarno-białe" is
   * "to teraz rozjaśnij", about the picture on screen. Dropping the photo after
   * one edit made that follow-up answer "dodaj zdjęcie", which reads as the app
   * forgetting what it just produced.
   */
  adopt(dataUrl, name) {
    this.dataUrl = dataUrl;
    this.thumb.src = dataUrl;
    this.nameEl.textContent = name;
    this.el.hidden = false;
    this.onChange?.(dataUrl);
  }

  clear() {
    this.dataUrl = null;
    this.el.hidden = true;
    this.thumb.removeAttribute('src');
    this.onChange?.(null);
  }
}

export class ModelPicker {
  constructor({onChange}) {
    this.onChange = onChange;
    this.models = [];
    this.current = 'g-micro';
    this.btn = document.querySelector('#model-btn');
    this.nameEl = document.querySelector('#model-name');
    this.menu = document.querySelector('#model-menu');

    this.btn.addEventListener('click', (e) => { e.stopPropagation(); this.toggle(); });
    document.addEventListener('click', () => this.close());
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') this.close(); });
  }

  /** The list comes from the backend, which knows whether G-Images actually
   *  has a checkpoint on this machine. Hardcoding it here would offer a model
   *  that cannot run. */
  setModels(models) {
    this.models = models || [];
    if (!this.models.some((m) => m.id === this.current && m.available)) {
      this.select('g-micro', true);
    }
    this._render();
  }

  select(id, silent = false) {
    const model = this.models.find((m) => m.id === id);
    if (model && !model.available) return;
    this.current = id;
    this.nameEl.textContent = model?.name || id;
    this._render();
    if (!silent) this.onChange?.(id);
  }

  get needsImage() {
    return Boolean(this.models.find((m) => m.id === this.current)?.needs_image);
  }

  toggle() {
    this.menu.hidden ? this.open() : this.close();
  }

  open() {
    this.menu.hidden = false;
    this.btn.setAttribute('aria-expanded', 'true');
  }

  close() {
    this.menu.hidden = true;
    this.btn.setAttribute('aria-expanded', 'false');
  }

  _render() {
    this.menu.innerHTML = '';
    for (const m of this.models) {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'model-item';
      item.setAttribute('role', 'option');
      item.disabled = !m.available;
      if (m.id === this.current) item.classList.add('active');
      item.innerHTML = `<span class="model-item-name"></span><span class="model-item-desc"></span>`;
      item.querySelector('.model-item-name').textContent = m.name;
      item.querySelector('.model-item-desc').textContent =
        m.available ? m.desc : 'brak checkpointu';
      item.addEventListener('click', (e) => {
        e.stopPropagation();
        this.select(m.id);
        this.close();
      });
      this.menu.appendChild(item);
    }
  }
}
