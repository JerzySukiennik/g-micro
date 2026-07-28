/**
 * Turn a finished reply into readable HTML.
 *
 * Two constraints shape everything here.
 *
 * **It runs after streaming, not during.** Tokens arrive one at a time and the
 * brightness flash as each lands is the app's signature — reformatting on every
 * token would fight it and produce flickering half-parsed markup. So the raw
 * text streams, and this replaces it once the reply is complete.
 *
 * **It is deliberately timid.** This model was never trained on markdown; it
 * writes ordinary Polish prose. A permissive parser would find syntax that was
 * never meant — an asterisk used as a footnote mark, an underscore inside a
 * filename — and silently eat the surrounding text. Every rule below therefore
 * requires unambiguous delimiters, and anything that does not clearly match is
 * left exactly as the model wrote it. Rendering plain text as plain text is a
 * correct outcome; mangling it is not.
 *
 * The one rule that earns its place is the inline list. The model reliably
 * writes "1) jabłka 2) jagody 3) banany" on a single line, which is a list in
 * everything but presentation.
 */

const escapeHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Inline emphasis. Applied to already-escaped text, so the patterns can only
 *  match literal characters the model wrote. */
function inline(text) {
  return text
    // `code` — unambiguous, and the model does occasionally quote tokens
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    // **bold** — both delimiters required, no space just inside them, so an
    // asterisk floating in prose cannot start one
    .replace(/\*\*(?=\S)([^*\n]+?)(?<=\S)\*\*/g, '<strong>$1</strong>')
    // *italic* — same rule, single asterisk. Rare in Polish prose, which is
    // precisely why it is safe to honour when it does appear balanced.
    .replace(/(^|[\s(])\*(?=\S)([^*\n]+?)(?<=\S)\*(?=[\s.,;:!?)]|$)/g, '$1<em>$2</em>');
}

/** "1) a 2) b 3) c" written on one line → the items, or null if this is not
 *  that shape. Requires at least two markers so a sentence that merely opens
 *  with "1)" is not mistaken for a list. */
function inlineEnumeration(line) {
  const markers = [...line.matchAll(/(?:^|\s)(\d{1,2})[.)]\s*/g)];
  if (markers.length < 2) return null;
  // The numbers have to actually count up, otherwise this is prose containing
  // figures rather than an enumeration.
  const nums = markers.map((m) => Number(m[1]));
  if (!nums.every((n, i) => i === 0 || n === nums[i - 1] + 1)) return null;

  const items = [];
  for (let i = 0; i < markers.length; i++) {
    const start = markers[i].index + markers[i][0].length;
    const end = i + 1 < markers.length ? markers[i + 1].index : line.length;
    const item = line.slice(start, end).trim().replace(/[.,;]$/, '');
    if (item) items.push(item);
  }
  return items.length >= 2 ? items : null;
}

/**
 * Render a completed reply.
 *
 * @param {string} text raw model output
 * @returns {string} HTML
 */
export function renderMarkdown(text) {
  const src = String(text ?? '').trim();
  if (!src) return '';

  const blocks = [];
  let list = null;

  const flushList = () => {
    if (list) {
      blocks.push(`<ol>${list.map((i) => `<li>${inline(i)}</li>`).join('')}</ol>`);
      list = null;
    }
  };

  for (const rawLine of src.split('\n')) {
    const line = escapeHtml(rawLine.trim());
    if (!line) { flushList(); continue; }

    // Whole-list-on-one-line is checked FIRST. Order matters: "1) jabłka 2)
    // jagody 3) banany" also matches the single-item pattern below, which
    // swallowed the entire line into one <li> — and that string is the single
    // most common shape this model produces, so the bug would have shipped on
    // the exact case the feature exists for.
    const inlineItems = inlineEnumeration(line);
    if (inlineItems) {
      flushList();
      blocks.push(`<ol>${inlineItems.map((i) => `<li>${inline(i)}</li>`).join('')}</ol>`);
      continue;
    }

    // A line that is itself one numbered or bulleted item
    const own = line.match(/^(?:\d{1,2}[.)]|[-*•])\s+(.*)$/);
    if (own) {
      (list ??= []).push(own[1]);
      continue;
    }

    flushList();
    blocks.push(`<p>${inline(line)}</p>`);
  }
  flushList();
  return blocks.join('');
}
