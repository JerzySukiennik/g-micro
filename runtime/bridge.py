"""The Mac side of the web app: run jobs posted from a browser, anywhere.

The phone cannot run a 110M-parameter model, and this Mac cannot accept an
inbound connection without opening a port on the home router. So neither end
connects to the other — both connect *out* to a Firebase Realtime Database and
talk through it. The Mac's only network behaviour is an outgoing HTTPS request
it opened itself, which is the whole point: there is nothing here for anyone to
port-scan.

Shape:

    browser  --write-->  /rooms/<room>/jobs/<id>     {model, text, image}
    mac      --stream->  same path over SSE, runs it
    mac      --write-->  /rooms/<room>/out/<id>      {text, progress, image, done}
    browser  --stream->  that, and draws it

Access control is the room id: 26 random characters, generated on this machine
on first run and stored in ~/.g-micro/room.json. The database rules refuse any
room name shorter than 20 characters, and a room cannot be discovered by
listing because nothing has read permission on the parent. **Whoever holds the
link can talk to this Mac** — it is a password that happens to live in a URL.
`--new-room` throws the old one away.

Run it standalone, or let the app start it (see app/main.js).
"""

import argparse
import asyncio
import json
import secrets
import sys
import threading
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from runtime.server import Backend  # noqa: E402

DB_URL = "https://g-micro-web-default-rtdb.firebaseio.com"
SITE_URL = "https://g-micro-web.web.app"
CONFIG = Path.home() / ".g-micro" / "room.json"

# How often the growing answer is pushed. Every token would be ~25 writes a
# second for prose that a person reads at 5 words a second; 150 ms still looks
# like typing and costs a fraction of that.
FLUSH_EVERY = 0.15
HEARTBEAT_EVERY = 20          # seconds; the web app calls the Mac offline at 60


def room_id(reset=False):
    if CONFIG.exists() and not reset:
        return json.loads(CONFIG.read_text())["room"]
    room = secrets.token_urlsafe(20)
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"room": room}, indent=2))
    CONFIG.chmod(0o600)
    return room


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
    def __init__(self, room):
        self.room = room
        self.base = f"{DB_URL}/rooms/{room}"
        self.backend = Backend()
        self.jobs = []                      # pending (id, payload), oldest first
        self.cancelled = set()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = threading.Event()

    # ---------------------------------------------------------------- http --
    def put(self, path, data):
        try:
            requests.put(f"{self.base}/{path}.json", json=data, timeout=20)
        except requests.RequestException as e:
            # A dropped write is not worth killing a 36-second edit over; the
            # next flush carries the whole accumulated state anyway, because
            # every write sends the full value rather than a delta.
            print(f"[bridge] write failed ({path}): {e}", flush=True)

    def delete(self, path):
        try:
            requests.delete(f"{self.base}/{path}.json", timeout=20)
        except requests.RequestException:
            pass

    # ----------------------------------------------------------- listening --
    def listen(self):
        """Server-sent events on the job queue, in a thread.

        Also the cancel channel: the browser writes `cancel: true` *into* the
        job it wants stopped, and that patch arrives here as an ordinary event
        while the job is mid-generation. That is why a job is not deleted when
        it is picked up — a deleted job has nowhere to receive a cancel.
        """
        headers = {"Accept": "text/event-stream"}
        while not self.stopping.is_set():
            try:
                with requests.get(f"{self.base}/jobs.json", headers=headers,
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
        path, data = msg.get("path", "/"), msg.get("data")
        if data is None:
            return                                   # a deletion, ours usually
        parts = [p for p in path.split("/") if p]

        if not parts:
            # The whole queue at once — the snapshot sent on connect, so
            # anything posted while this Mac was asleep still gets answered.
            for job_id, payload in sorted(data.items(), key=lambda kv: kv[1].get("at", 0)):
                self._queue(job_id, payload)
        elif len(parts) == 1 and isinstance(data, dict):
            self._queue(parts[0], data)
        elif len(parts) == 2 and parts[1] == "cancel" and data:
            with self.lock:
                self.cancelled.add(parts[0])
            self.wake.set()

    def _queue(self, job_id, payload):
        if not isinstance(payload, dict) or "text" not in payload:
            return
        with self.lock:
            if any(j[0] == job_id for j in self.jobs):
                return
            self.jobs.append((job_id, payload))
        self.wake.set()

    def heartbeat(self):
        while not self.stopping.wait(HEARTBEAT_EVERY):
            self.put("mac", self._presence(True))

    # -------------------------------------------------------------- running --
    def _presence(self, online):
        """What the web app reads to decide whether to let anyone type.

        The model list rides along: whether G-Images has a checkpoint on disk
        is a fact about this machine, so the browser should be told rather than
        left guessing.
        """
        return {"online": online, "at": int(time.time() * 1000),
                "models": self.backend.model_list()}

    def run(self):
        self.put("mac", self._presence(True))
        threading.Thread(target=self.listen, daemon=True).start()
        threading.Thread(target=self.heartbeat, daemon=True).start()

        print(f"[bridge] pokój {self.room}")
        print(f"[bridge] otwórz: {SITE_URL}/#{self.room}")
        print("[bridge] czekam na zadania…", flush=True)

        try:
            while True:
                self.wake.wait(1)
                self.wake.clear()
                while True:
                    with self.lock:
                        if not self.jobs:
                            break
                        job_id, payload = self.jobs.pop(0)
                    self._handle(job_id, payload)
        except KeyboardInterrupt:
            pass
        finally:
            self.stopping.set()
            self.put("mac", self._presence(False))

    def _handle(self, job_id, payload):
        with self.lock:
            cancelled = job_id in self.cancelled
        if cancelled:
            self.delete(f"jobs/{job_id}")
            return
        model = payload.get("model", "g-micro")
        print(f"[bridge] {job_id[:8]} {model}: {payload.get('text','')[:60]!r}", flush=True)
        try:
            asyncio.run(self._run_job(job_id, payload, model))
        except Exception as e:
            self.put(f"out/{job_id}", {"text": f"Coś poszło nie tak na Macu: {e}",
                                       "done": True})
        self.delete(f"jobs/{job_id}")

    async def _run_job(self, job_id, payload, model):
        state = {"text": "", "progress": None, "image": None}
        last_flush = [0.0]

        def flush(force=False):
            now = time.monotonic()
            if not force and now - last_flush[0] < FLUSH_EVERY:
                return
            last_flush[0] = now
            # The full value every time, not a delta: a lost or reordered write
            # then costs nothing, because the next one is authoritative.
            self.put(f"out/{job_id}", {k: v for k, v in state.items() if v is not None})

        stop = CancelFlag(self, job_id)

        async def send(obj):
            kind = obj.get("type")
            if kind == "step":
                state["text"] += obj.get("token", "")
                flush(force=bool(obj.get("done")))
                if obj.get("done"):
                    self.put(f"out/{job_id}", {**{k: v for k, v in state.items()
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

        if model == "g-images":
            await self.backend.images.run(send, payload.get("text", ""),
                                          payload.get("image") or "", stop)
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
    p.add_argument("--new-room", action="store_true",
                   help="wygeneruj nowy adres pokoju (unieważnia stary link)")
    p.add_argument("--print-url", action="store_true",
                   help="wypisz adres i zakończ")
    args = p.parse_args()

    room = room_id(reset=args.new_room)
    if args.print_url:
        print(f"{SITE_URL}/#{room}")
        return
    Bridge(room).run()


if __name__ == "__main__":
    main()
