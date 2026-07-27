"""
Retrieval for the "Use notes (RAG)" setting — two sources, tried in order:

  1. Jurek's own vault (his second-brain notes, ~/Downloads/Claude/
     ClaudeMemory) — plain keyword overlap over the markdown files, not a
     real embedding search. No embedding model is available offline on this
     hardware, and for a few hundred short notes, keyword overlap is good
     enough to find the right file most of the time.
  2. Wikipedia (Polish, falling back to nothing rather than English — this
     is a Polish model project) — for general topics the vault doesn't
     cover. Needs internet; the app is not otherwise online-dependent, so
     every call here is wrapped to fail silently and return None rather
     than surface a network error to the user.

This module only retrieves — server.py wraps whatever it returns in the
tokenizer's <|context|> token before the <|user|> turn. That token was
reserved before pretraining but never used in a single real training
example until the 2026-07-25 SFT pass added synthetic context+question+
answer examples specifically to teach the association (see
data/build_sft.py's CONTEXT_QA_* pipeline) — retrieval built here is only
as good as whether the deployed checkpoint was trained on that data.
"""

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

VAULT = Path.home() / "Downloads" / "Claude" / "ClaudeMemory"
STOPWORDS = {
    "jest", "czy", "jak", "co", "to", "na", "do", "nie", "się", "dla",
    "ale", "tak", "gdzie", "kto", "ile", "przez", "kiedy", "oraz", "the",
    "and", "for", "with", "from",
    # Common conversational function words — confirmed live: "masz" (a
    # generic "you have") matched incidental, unrelated notes for the
    # identity question "jak masz na imię?" purely by chance, at a high
    # enough score to beat MIN_SCORE. Same failure mode as any of these.
    "masz", "mam", "ma", "mają", "może", "można", "chcę", "chcesz",
    "tylko", "także", "żeby", "jestem", "jesteś", "był", "była", "były",
    "będzie", "będą", "sobie", "swoje", "swój", "bardzo", "jakiś",
    # Instruction verbs and interrogatives. These are what the user wants
    # *done*, never what they want it done *about*, and leaving them in the
    # search string is why "Czym jest Warszawa?" retrieved the article
    # "Wikipedia" and "Opowiedz o Krakowie." retrieved an actor.
    "czym", "opowiedz", "powiedz", "wyjaśnij", "wyjasnij", "wytłumacz",
    "wytlumacz", "opisz", "napisz", "wymień", "wymien", "podaj", "wypisz",
    "pokaż", "pokaz", "jakie", "jaki", "jaka", "które", "ktore", "który",
    "ktory", "która", "ktora", "coś", "cos", "krótko", "krotko", "proszę",
    "prosze", "znaczy", "oznacza",
}


def _keywords(query: str):
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
MIN_SCORE = 2  # confirmed live: a score of 1 is often one incidental word
                # (e.g. "masz" in "jak masz na imię?") matching some
                # unrelated note by chance — that produced garbled,
                # single-word replies once the model actually started
                # trying to use the (irrelevant) context it was handed.


def _strip_frontmatter(text: str) -> str:
    """The <|context|> training examples (data/build_sft.py) are all clean
    Wikipedia prose. A raw vault file's YAML frontmatter
    ("---\\ntype: project\\nstatus: active\\n---") is nothing like that
    distribution — confirmed live, handing the model a note WITH its
    frontmatter produced single garbled words ("Skyfall", "Reputacja") where
    clean prose produced a real, if still sometimes wrong, attempt at an
    answer. Stripping it is a formatting fix, not a relevance one."""
    return FRONTMATTER_RE.sub("", text, count=1)


def search_vault(query: str, snippet_chars: int = 600):
    """Best-matching vault note by raw keyword-count overlap. Not
    semantic search — just good enough to find the right file among a
    few hundred short, hand-written notes."""
    if not VAULT.exists():
        return None
    words = _keywords(query)
    if not words:
        return None

    best_score, best_path, best_text = 0, None, None
    for p in VAULT.rglob("*.md"):
        if ".obsidian" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        low = text.lower()
        score = sum(low.count(w) for w in words)
        if score > best_score:
            best_score, best_path, best_text = score, p, text

    if best_path is None or best_score < MIN_SCORE:
        return None
    clean_text = _strip_frontmatter(best_text)
    return {"source": str(best_path.relative_to(VAULT)), "text": clean_text[:snippet_chars]}


# Wikipedia's API rejects requests with no User-Agent (or Python's default
# one) as of their current bot-traffic policy — a bare urlopen() call gets
# a flat 403, confirmed directly against the live API.
_HEADERS = {"User-Agent": "MicroG/1.0 (hobby Polish LLM project; local personal use)"}


# Questions about the assistant itself must never hit retrieval. Confirmed
# live 2026-07-27: with RAG on, "Jak masz na imię?" was answered
# "Argentyńskie", because opensearch matched the *phrasing of the question*
# to the title of a Wikipedia article — "Jak mam na imię", a 2012
# Argentine-Spanish-Brazilian drama — and the model then did exactly what
# three rounds of training taught it to do: answer from the context it was
# given. The model was right; the retrieval was garbage. There is no article
# that can answer "what is your name" better than the identity examples
# already baked into the weights, so the correct hit rate here is zero.
SELF_QUERY_RE = re.compile(
    r"(masz na imię|masz na imie|się nazywasz|sie nazywasz|twoje imię|twoje imie|"
    r"kim jesteś|kim jestes|czym jesteś|czym jestes|przedstaw się|przedstaw sie|"
    r"kto cię (stworzył|wytrenował)|kto cie (stworzyl|wytrenowal)|"
    r"ile masz (lat|parametrów|parametrow)|czy jesteś|czy jestes|"
    r"o sobie|twoja nazwa|masz imię|masz imie)", re.I)

# A bare greeting carries no information need. "Hej!" retrieved "Hej! Kto
# Polak na bagnety!!" — a 1920 propaganda poster — and the reply became a
# monologue about the Polish nation.
GREETING_RE = re.compile(
    r"^\W*(hej|cześć|czesc|siema|witam|witaj|dzień dobry|dzien dobry|hello|hi|"
    r"elo|joł|jol|dobry wieczór|dobry wieczor|no hej)\W*$", re.I)


def should_retrieve(query: str) -> bool:
    """Whether this query should consult an outside source at all.

    Retrieval can only help when the answer lives in a document. For
    identity questions and greetings it can only hurt, and did.
    """
    q = query.strip()
    if not q or GREETING_RE.match(q) or SELF_QUERY_RE.search(q):
        return False
    return bool(_keywords(q))


def _fold(s: str) -> str:
    """Lowercase, accents removed — so 'Kraków' and 'krakowie' can be compared."""
    s = s.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c))


def _is_relevant(query: str, text: str, title: str) -> bool:
    """Reject an article matched on question phrasing rather than subject.

    opensearch does fuzzy *title* matching, so a question shaped like a title
    finds that title whatever it is about. Requiring the article to actually
    mention something the question was about separates "article about the
    thing I asked" from "article whose name resembles my sentence".

    Matching is on folded five-character prefixes, because Polish inflects
    everything: the question says "Tadeusza" and the article says "Tadeusz",
    "Krakowie" against "Kraków". Comparing whole words literally rejected
    every one of those, which is how the first version of this check silently
    switched retrieval off for real questions.
    """
    hay = _fold(text + " " + title)
    for w in _keywords(query):
        stem = _fold(w)[:5]
        if len(stem) >= 4 and stem in hay:
            return True
    return False


def search_wikipedia(query: str, lang: str = "pl", timeout: float = 3.0):
    if not should_retrieve(query):
        return None
    # Full-text search, not opensearch. opensearch matches *title prefixes*,
    # which is the entire mechanism behind the 2026-07-27 failure: it only
    # ever returned a hit when the question happened to look like a title
    # ("Jak masz na imię?" -> "Jak mam na imię", "Hej!" -> "Hej! Kto Polak na
    # bagnety!!"), and returned nothing at all for ordinary questions like
    # "Co to jest fotosynteza?". Precisely backwards. list=search reads
    # article content and copes with Polish inflection, so "Opowiedz o
    # Krakowie." can still reach "Kraków".
    terms = " ".join(_keywords(query))
    if not terms:
        return None
    try:
        search_url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(terms)}&srlimit=1&format=json"
        )
        req = urllib.request.Request(search_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            hits = json.loads(r.read()).get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        summary_url = (
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
            f"{urllib.parse.quote(title)}"
        )
        req = urllib.request.Request(summary_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            extract = json.loads(r.read()).get("extract")
        if not extract:
            return None
        if not _is_relevant(query, extract, title):
            return None
        return {"source": f"wikipedia:{title}", "text": extract}
    except Exception:
        # No internet, DNS failure, Wikipedia hiccup, malformed response —
        # none of these should ever surface as a chat error. RAG failing
        # quietly and falling through to the model's own (confabulated)
        # answer is the correct degrade path, not a crash.
        return None


def retrieve(query: str):
    """Wikipedia only for now. search_vault() is disabled here on purpose,
    not removed: the <|context|> SFT examples (data/build_sft.py) are all
    clean Wikipedia-style prose, and confirmed live 2026-07-25, handing the
    model a vault note's markdown-and-headers style — even with its
    frontmatter stripped — produced garbled single-word replies where the
    same question with Wikipedia prose produced a real (if still sometimes
    wrong) structured attempt at an answer, consistently across repeated
    tries. The model generalised "context = prose paragraph", not "context =
    any text" — it would need vault-styled examples in its own training data
    to fix, which is a separate SFT pass, not built yet. Re-enable by
    changing this to `search_vault(query) or search_wikipedia(query)` once
    that training exists."""
    return search_wikipedia(query)
