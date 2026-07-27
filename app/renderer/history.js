/**
 * History sidebar — UI-SPEC.md: "wysuwana z lewej, domyślnie schowana,
 * przykrywa czat półprzezroczystą warstwą szkła." Persistence goes through
 * main.js via the preload bridge (plain JSON files in userData), not
 * localStorage — this is desktop app data, not a web page's scratch state.
 */

export class History {
  constructor({ sidebarEl, scrimEl, listEl, onOpenConversation }) {
    this.sidebar = sidebarEl;
    this.scrim = scrimEl;
    this.list = listEl;
    this.onOpenConversation = onOpenConversation;
    this.open = false;
    this.activeId = null;

    this.scrim.addEventListener('click', () => this.close());
  }

  toggle() { this.open ? this.close() : this.openSidebar(); }

  async openSidebar() {
    this.open = true;
    this.sidebar.setAttribute('aria-hidden', 'false');
    this.sidebar.style.transition = 'transform 380ms cubic-bezier(.2,.9,.25,1)';
    this.sidebar.style.transform = 'translateX(0)';
    this.scrim.classList.add('show');
    this.scrim.style.transition = 'opacity 380ms ease-out';
    this.scrim.style.opacity = '1';
    await this.refresh();
  }

  close() {
    this.open = false;
    this.sidebar.setAttribute('aria-hidden', 'true');
    this.sidebar.style.transform = 'translateX(-280px)';
    this.scrim.style.opacity = '0';
    this.scrim.classList.remove('show');
  }

  /** Swaps the row's label for an input, committing on Enter or blur. Inline
   *  because a modal for renaming a chat is more ceremony than the act needs. */
  _startRename(row, open, item) {
    const input = document.createElement('input');
    input.className = 'conv-rename';
    input.value = item.title || '';
    input.maxLength = 80;
    row.replaceChild(input, open);
    input.focus();
    input.select();
    let done = false;
    const commit = async (save) => {
      if (done) return;
      done = true;
      if (save && input.value.trim()) {
        await window.gmicro.history.rename(item.id, input.value.trim());
      }
      await this.refresh();
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); commit(true); }
      else if (e.key === 'Escape') { e.preventDefault(); commit(false); }
      e.stopPropagation();
    });
    input.addEventListener('blur', () => commit(true));
  }

  async refresh() {
    const items = await window.gmicro.history.list();
    this.list.innerHTML = '';
    if (!items.length) {
      this.list.innerHTML = '<div class="conv-empty">Nie ma jeszcze żadnych rozmów.<br>Napisz coś, żeby zacząć.</div>';
      return;
    }
    for (const item of items) {
      // A row rather than a bare button now, because it carries its own
      // rename and delete controls. Those appear on hover: conversations are
      // read far more often than they are managed.
      const row = document.createElement('div');
      row.className = 'conv-row' + (item.id === this.activeId ? ' active' : '');

      const open = document.createElement('button');
      open.className = 'conv-item';
      open.innerHTML =
        `${escapeHtml(item.title || 'Bez tytułu')}<span class="date">${formatDate(item.updated)}</span>`;
      open.addEventListener('click', () => {
        this.activeId = item.id;
        this.close();
        this.onOpenConversation(item.id);
      });

      const rename = document.createElement('button');
      rename.className = 'conv-act';
      rename.title = 'Zmień nazwę';
      rename.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none"><path d="M4 20h4L19 9l-4-4L4 16v4z" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/></svg>';
      rename.addEventListener('click', (e) => {
        e.stopPropagation();
        this._startRename(row, open, item);
      });

      const del = document.createElement('button');
      del.className = 'conv-act danger';
      del.title = 'Usuń rozmowę';
      del.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none"><path d="M5 7h14M10 7V5h4v2M8 7l1 12h6l1-12" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      del.addEventListener('click', async (e) => {
        e.stopPropagation();
        // Two-step rather than a dialog box: the second click confirms, and
        // moving the mouse away cancels. Deleting is irreversible, so it
        // should not happen on a single stray click.
        if (!del.classList.contains('confirm')) {
          del.classList.add('confirm');
          del.title = 'Kliknij ponownie, aby usunąć';
          row.addEventListener('mouseleave', () => {
            del.classList.remove('confirm');
            del.title = 'Usuń rozmowę';
          }, {once: true});
          return;
        }
        await window.gmicro.history.delete(item.id);
        if (this.activeId === item.id) this.activeId = null;
        await this.refresh();
      });

      row.append(open, rename, del);
      this.list.appendChild(row);
    }
  }

  setActive(id) { this.activeId = id; }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function formatDate(ts) {
  const d = new Date(ts);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' · ' +
         d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}
