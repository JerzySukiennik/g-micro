"""The Mac side of the web app: run jobs posted from a browser, anywhere.

The phone cannot run a 110M-parameter model, and this Mac cannot accept an
inbound connection without opening a port on the home router. So neither end
connects to the other — both connect *out* to a Firebase Realtime Database and
talk through it. The Mac's only network behaviour is an outgoing HTTPS request
it opened itself, which is the whole point: there is nothing here for anyone to
port-scan.

Shape:

    browser  --write-->  /open/<client>/jobs/<id>    {model, text, image}
    mac      --stream->  all of /open over SSE, runs what appears
    mac      --write-->  /open/<client>/out/<id>     {text, progress, image, done}
    browser  --stream->  its own out, and draws it
    mac      --write-->  /status/mac                 heartbeat, readable by all

Nobody logs in, and devices still cannot see each other. Each browser invents a
128-bit client id on first visit and works only under it; the rules let *only*
this Mac list /open, so ids cannot be discovered, and they are too large to
guess. The Mac earns that privilege by signing in anonymously once — see
MacIdentity and database.rules.md for why the asymmetry has to exist.

There is no other gate: while this process runs, anyone who opens the site gets
an answer. That is deliberate (Jurek turns it on while using it) and bounded by
what the bridge can reach — the chat model and the photo editor, with retrieval
off, so no file, note or command is exposed.

Run it standalone, or let the app start it (see app/main.js).
"""

import argparse
import asyncio
import gc
import json
import signal
import sys
import threading
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from runtime.server import Backend, model_list  # noqa: E402
from runtime.images import ImageModel, VERSIONS as IMAGE_MODELS  # noqa: E402

# Two sites now run this same protocol over two different Firebase projects:
# the original G-Micro page, and Narew Labs at ai.gzowo.fun. The wire format,
# the paths and the rules are identical — only the project differs — so the
# bridge takes a profile rather than growing a second copy of itself.
#
# Each project needs its own identity file: the Mac's uid is per-project, and
# writing over G-Micro's would silently unlink a working bridge.
SITES = {
    "g-micro": {
        "db": "https://g-micro-web-default-rtdb.firebaseio.com",
        "site": "https://g-micro-web.web.app",
        "api_key": "AIzaSyCSoFFWPm9Ts1iLlFkPUuicmIfT_wFrAss",
        "identity": Path.home() / ".g-micro" / "identity.json",
    },
    "narew-labs": {
        "db": "https://narew-labs-default-rtdb.europe-west1.firebasedatabase.app",
        "site": "https://ai.gzowo.fun",
        "api_key": "AIzaSyCiNb3wGWfE1xt19CmeEF3M4hh2KQ_QcqM",
        "identity": Path.home() / ".narew-labs" / "identity.json",
    },
}

# Filled in by main() once the profile is known. Module-level names are kept so
# the rest of the file reads the same as before.
DB_URL = SITES["g-micro"]["db"]
SITE_URL = SITES["g-micro"]["site"]
API_KEY = SITES["g-micro"]["api_key"]
CONFIG = SITES["g-micro"]["identity"]


def use_site(name):
    """Point this process at one of the projects above."""
    global DB_URL, SITE_URL, API_KEY, CONFIG
    cfg = SITES[name]
    DB_URL, SITE_URL = cfg["db"], cfg["site"]
    API_KEY, CONFIG = cfg["api_key"], cfg["identity"]

# Every browser invents its own private id and gets its own subtree under here.
# Nobody signs in and nobody shares a space — the isolation comes from the id
# being unguessable and from the rules refusing to let anyone but this Mac list
# what ids exist. See database.rules.json.
ROOT = "open"

# How often the growing answer is pushed. Every token would be ~25 writes a
# second for prose that a person reads at 5 words a second; 150 ms still looks
# like typing and costs a fraction of that.
FLUSH_EVERY = 0.15
HEARTBEAT_EVERY = 20          # seconds; the web app calls the Mac offline at 60

# Refresh well before the hour Firebase gives an id token, so a long photo edit
# never runs out mid-write.
TOKEN_TTL = 50 * 60

# Drop the weights after this long without a job. Reloading the text model off
# an SSD costs about four seconds, paid once by whoever breaks the silence;
# holding ~700 MB through an idle night costs it the whole night.
IDLE_UNLOAD = int(__import__('os').environ.get('IDLE_UNLOAD', 10*60))


class MacIdentity:
    """The Mac's own anonymous Firebase account.

    This is what makes per-device privacy possible without anyone logging in.
    Browsers write into their own subtree and can read only that subtree; the
    Mac is the single party allowed to *list* the tree and see everyone's work,
    and the rules recognise it by uid. Without an authenticated Mac there would
    be no way to grant "read everything" to the bridge alone — anything a
    browser could use to find jobs, every other browser could use too.

    The account is created once and then kept alive by its refresh token, so
    the uid stays stable; the rules name that uid, so losing this file means
    redeploying rules with the new one.
    """

    def __init__(self):
        self.uid = None
        self.refresh_token = None
        self.id_token = None
        self.expires_at = 0
        self._lock = threading.Lock()

    def load_or_create(self):
        if CONFIG.exists():
            saved = json.loads(CONFIG.read_text())
            self.uid = saved.get("uid")
            self.refresh_token = saved.get("refresh_token")
            if self.refresh_token:
                self._refresh()
                return self.uid
        r = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}",
            json={"returnSecureToken": True}, timeout=20)
        r.raise_for_status()
        d = r.json()
        self.uid, self.refresh_token = d["localId"], d["refreshToken"]
        self.id_token = d["idToken"]
        self.expires_at = time.time() + TOKEN_TTL
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps({"uid": self.uid,
                                      "refresh_token": self.refresh_token}, indent=2))
        CONFIG.chmod(0o600)
        return self.uid

    def _refresh(self):
        r = requests.post(f"https://securetoken.googleapis.com/v1/token?key={API_KEY}",
                          data={"grant_type": "refresh_token",
                                "refresh_token": self.refresh_token}, timeout=20)
        r.raise_for_status()
        d = r.json()
        self.id_token = d["id_token"]
        self.uid = d.get("user_id", self.uid)
        self.expires_at = time.time() + TOKEN_TTL

    def token(self):
        """A currently valid id token, refreshed on demand."""
        with self._lock:
            if not self.id_token or time.time() >= self.expires_at:
                self._refresh()
            return self.id_token


class CancelFlag:
    """What `Backend.chat` and `ImageBackend.run` expect as `stop_event` — they
    only ever call `.is_set()`. Here that question is answered by the set the
    SSE thread fills when the browser asks to stop, so a cancel written from a
    phone lands inside a generation loop already in flight."""

    def __init__(self, bridge, job_id):
        self.bridge, self.job_id = bridge, job_id

    def is_set(self):
        with self.bridge.lock:
            return self.job_id in self.bridge.cancelled


class Bridge:
    def __init__(self, identity):
        self.identity = identity
        self.base = f"{DB_URL}/{ROOT}"
        # Built on demand, dropped when nobody has asked anything for a while.
        # A bridge that runs all day to answer a handful of messages should not
        # hold half a gigabyte the whole time: measured idle cost was 704 MB
        # with only the text model resident and 917 MB once a photo had been
        # edited, against ~200 MB for bare Python with torch imported.
        self._backend = None
        self.last_job = time.time()
        self.jobs = []                      # pending (client, id, payload), oldest first
        self.cancelled = set()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = threading.Event()

    # ---------------------------------------------------------------- http --
    def put(self, path, data):
        try:
            requests.put(f"{self.base}/{path}.json?auth={self.identity.token()}",
                         json=data, timeout=20)
        except requests.RequestException as e:
            # A dropped write is not worth killing a 36-second edit over; the
            # next flush carries the whole accumulated state anyway, because
            # every write sends the full value rather than a delta.
            print(f"[bridge] write failed ({path}): {e}", flush=True)

    def delete(self, path):
        try:
            requests.delete(f"{self.base}/{path}.json?auth={self.identity.token()}",
                            timeout=20)
        except requests.RequestException:
            pass

    # ----------------------------------------------------------- listening --
    def listen(self):
        """Server-sent events over every client's jobs at once, in a thread.

        Streaming the whole `open` tree is the point of the Mac having its own
        account: the rules let exactly this uid read across clients, so one
        connection serves everybody while no browser can see past its own id.

        Also the cancel channel: the browser writes `cancel: true` *into* the
        job it wants stopped, and that patch arrives here as an ordinary event
        while the job is mid-generation. That is why a job is not deleted when
        it is picked up — a deleted job has nowhere to receive a cancel.
        """
        headers = {"Accept": "text/event-stream"}
        while not self.stopping.is_set():
            try:
                # The token is minted per connection; the stream is torn down and
                # remade well inside the token's lifetime by the loop below.
                url = f"{self.base}.json?auth={self.identity.token()}"
                with requests.get(url, headers=headers,
                                  stream=True, timeout=(10, None)) as r:
                    r.raise_for_status()
                    event = None
                    # `r.raw.readline()`, not `r.iter_lines()`. iter_lines asks
                    # the socket for a 512-byte chunk and blocks until it has
                    # one, so a 110-byte event just sits there — measured: the
                    # stream connected, curl showed the job arriving instantly,
                    # and the loop yielded nothing for as long as it was
                    # watched. readline stops at the newline, which is what a
                    # line-oriented protocol needs.
                    while not self.stopping.is_set():
                        line = r.raw.readline()
                        if not line:
                            break                    # server closed; reconnect
                        raw = line.decode("utf-8", "replace").rstrip("\n")
                        if not raw:
                            continue
                        if raw.startswith("event: "):
                            event = raw[7:].strip()
                        elif raw.startswith("data: ") and event in ("put", "patch"):
                            self._on_event(json.loads(raw[6:]))
            except requests.RequestException as e:
                if not self.stopping.is_set():
                    print(f"[bridge] stream lost, retrying: {e}", flush=True)
                    time.sleep(3)

    def _on_event(self, msg):
        """Route one stream event.

        Paths are relative to `open`, so they name the client first:
            /                                   whole tree (snapshot on connect)
            /<client>/jobs/<id>                 a new job
            /<client>/jobs/<id>/cancel          stop that job
            /<client>/out/<id>                  our own reply echoing back
        """
        path, data = msg.get("path", "/"), msg.get("data")
        if data is None:
            return                                   # a deletion, ours usually
        parts = [p for p in path.split("/") if p]

        if not parts:
            # The whole tree at once, so anything posted while this Mac was
            # asleep still gets answered when it wakes up.
            for client, sub in (data or {}).items():
                for job_id, payload in sorted((sub.get("jobs") or {}).items(),
                                              key=lambda kv: kv[1].get("at", 0)):
                    self._queue(client, job_id, payload)
        elif len(parts) == 1 and isinstance(data, dict):
            # A client's whole subtree appeared at once — its first ever job.
            for job_id, payload in sorted((data.get("jobs") or {}).items(),
                                          key=lambda kv: kv[1].get("at", 0)):
                self._queue(parts[0], job_id, payload)
        elif len(parts) == 2 and parts[1] == "jobs" and isinstance(data, dict):
            for job_id, payload in sorted(data.items(), key=lambda kv: kv[1].get("at", 0)):
                self._queue(parts[0], job_id, payload)
        elif len(parts) == 3 and parts[1] == "jobs" and isinstance(data, dict):
            self._queue(parts[0], parts[2], data)
        elif len(parts) == 4 and parts[1] == "jobs" and parts[3] == "cancel" and data:
            with self.lock:
                self.cancelled.add(parts[2])
            self.wake.set()

    def _queue(self, client, job_id, payload):
        if not isinstance(payload, dict) or "text" not in payload:
            return
        with self.lock:
            if any(j[1] == job_id for j in self.jobs):
                return
            self.jobs.append((client, job_id, payload))
        self.wake.set()

    def heartbeat(self):
        while not self.stopping.wait(HEARTBEAT_EVERY):
            self.announce(True)

    def announce(self, online):
        """Presence lives outside the per-client tree, at a path every browser
        may read — a device has to know whether the Mac is awake before it has
        anything of its own in the database."""
        try:
            requests.put(f"{DB_URL}/status/mac.json?auth={self.identity.token()}",
                         json=self._presence(online), timeout=20)
        except requests.RequestException as e:
            print(f"[bridge] presence failed: {e}", flush=True)

    # -------------------------------------------------------------- running --
    @property
    def backend(self):
        """The models, loaded the first time something actually needs them."""
        if self._backend is None:
            print("[bridge] wczytuję modele…", flush=True)
            self._backend = Backend()
        return self._backend

    def maybe_unload(self):
        """Give the memory back after a quiet spell.

        Costs a few seconds on the next message — the text model reloads from
        disk in about four. That is a good trade for a process meant to sit in
        the background for days: the alternative is holding the weights through
        every hour nobody is using it.
        """
        if self._backend is None or IDLE_UNLOAD <= 0:
            return
        if time.time() - self.last_job < IDLE_UNLOAD:
            return
        print("[bridge] cisza — zwalniam modele", flush=True)
        self._backend = None
        gc.collect()

    def _presence(self, online):
        """What the web app reads to decide whether to let anyone type.

        The model list rides along: whether G-Images has a checkpoint on disk
        is a fact about this machine, so the browser should be told rather than
        left guessing.
        """
        # Deliberately does NOT touch self.backend: a heartbeat every 20 s
        # that loaded the model would keep it resident forever and undo the
        # idle unloading below. Whether G-Images is available is a question
        # about a file on disk, not about a loaded network.
        return {"online": online, "at": int(time.time() * 1000),
                "models": model_list(ImageModel().available())}

    def run(self):
        # SIGTERM is how this process normally dies — the app sends it on quit.
        # Python's default handler exits without unwinding, so the `finally`
        # below never ran and the last thing left in the database said the Mac
        # was online. The page recovers on its own after a missed-heartbeat
        # timeout, but that is a minute of a phone showing the wrong thing;
        # turning the signal into a normal exit makes the switch feel instant.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

        self.announce(True)
        threading.Thread(target=self.listen, daemon=True).start()
        threading.Thread(target=self.heartbeat, daemon=True).start()

        print(f"[bridge] tożsamość Maca: {self.identity.uid}")
        print(f"[bridge] otwórz: {SITE_URL}")
        print("[bridge] czekam na zadania…", flush=True)

        try:
            while True:
                self.wake.wait(30)
                self.wake.clear()
                while True:
                    with self.lock:
                        if not self.jobs:
                            break
                        # Text first, always. One edit is roughly two minutes of
                        # solid CPU, so a couple of queued pictures used to make
                        # chat look dead: the message sat behind them, the page's
                        # own watchdog gave up, and the Mac was answering the
                        # whole time. Ordering by kind rather than by arrival
                        # costs the pictures nothing they notice and stops a
                        # sentence from waiting on a landscape.
                        idx = next(
                            (i for i, j in enumerate(self.jobs)
                             if j[2].get("model", "g-micro") not in IMAGE_MODELS),
                            0)
                        client, job_id, payload = self.jobs.pop(idx)
                    self._handle(client, job_id, payload)
                    self.last_job = time.time()
                self.maybe_unload()
        except KeyboardInterrupt:
            pass
        finally:
            self.stopping.set()
            self.announce(False)

    def _handle(self, client, job_id, payload):
        with self.lock:
            cancelled = job_id in self.cancelled
        if cancelled:
            self.delete(f"{client}/jobs/{job_id}")
            return
        model = payload.get("model", "g-micro")
        print(f"[bridge] {client[:6]}/{job_id[:8]} {model}: "
              f"{payload.get('text','')[:60]!r}", flush=True)
        try:
            asyncio.run(self._run_job(client, job_id, payload, model))
        except Exception as e:
            self.put(f"{client}/out/{job_id}",
                     {"text": f"Coś poszło nie tak na Macu: {e}", "done": True})
        self.delete(f"{client}/jobs/{job_id}")

    async def _run_job(self, client, job_id, payload, model):
        state = {"text": "", "progress": None, "image": None}
        last_flush = [0.0]

        def flush(force=False):
            now = time.monotonic()
            if not force and now - last_flush[0] < FLUSH_EVERY:
                return
            last_flush[0] = now
            # The full value every time, not a delta: a lost or reordered write
            # then costs nothing, because the next one is authoritative.
            self.put(f"{client}/out/{job_id}",
                     {k: v for k, v in state.items() if v is not None})

        stop = CancelFlag(self, job_id)

        async def send(obj):
            kind = obj.get("type")
            if kind == "step":
                state["text"] += obj.get("token", "")
                flush(force=bool(obj.get("done")))
                if obj.get("done"):
                    self.put(f"{client}/out/{job_id}",
                             {**{k: v for k, v in state.items()
                                 if v is not None}, "done": True})
            elif kind == "image_progress":
                state["progress"] = obj.get("p", 0)
                flush()
            elif kind == "image_result":
                state["image"] = obj.get("image")
                state["label"] = obj.get("label")
                flush(force=True)
            elif kind == "error":
                state["text"] += obj.get("message", "")
                flush(force=True)

        # Every image version routes here under its own wire name; the text
        # model is the only thing that is not one of them. Falling through to
        # chat for an unknown image id used to answer a picture request with a
        # sentence, which reads as the model being broken rather than absent.
        if model in IMAGE_MODELS:
            await self.backend.images.run(send, payload.get("text", ""),
                                          payload.get("image") or "", stop,
                                          version=model)
        else:
            if not self.backend.ready:
                await self.backend.load(lambda _obj: asyncio.sleep(0))
            history = payload.get("history") or []
            await self.backend.chat(send, payload.get("text", ""), 0.7, 40, stop,
                                    rag=False, history=history)

        with self.lock:
            self.cancelled.discard(job_id)


def main():
    p = argparse.ArgumentParser(description="Most między stroną a modelami na Macu.")
    p.add_argument("--site", choices=sorted(SITES), default="g-micro",
                   help="który projekt obsługiwać (domyślnie g-micro)")
    p.add_argument("--print-url", action="store_true",
                   help="wypisz adres strony i zakończ")
    p.add_argument("--print-uid", action="store_true",
                   help="wypisz uid tego Maca (ten, który nazywają reguły bazy)")
    args = p.parse_args()

    use_site(args.site)

    if args.print_url:
        print(SITE_URL)
        return

    identity = MacIdentity()
    uid = identity.load_or_create()
    if args.print_uid:
        print(uid)
        return
    Bridge(identity).run()


if __name__ == "__main__":
    main()
