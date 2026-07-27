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

  async refresh() {
    const items = await window.g-micro.history.list();
    this.list.innerHTML = '';
    if (!items.length) {
      this.list.innerHTML = '<div class="conv-empty">No conversations yet.<br>Start typing to begin one.</div>';
      return;
    }
    for (const item of items) {
      const el = document.createElement('button');
      el.className = 'conv-item' + (item.id === this.activeId ? ' active' : '');
      el.innerHTML = `${escapeHtml(item.title || 'Untitled')}<span class="date">${formatDate(item.updated)}</span>`;
      el.addEventListener('click', () => {
        this.activeId = item.id;
        this.close();
        this.onOpenConversation(item.id);
      });
      this.list.appendChild(el);
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
