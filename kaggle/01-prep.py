"""
Kaggle cell 1 of 2 — build the token binaries.

Run this ONCE. It downloads the corpus and packs it into pl_train.bin /
pl_val.bin under /kaggle/working, which you then save as a Dataset and feed to
the training notebook. Doing it here rather than uploading from home is
deliberate: Kaggle's connection is fast, a domestic upload of ~3.8 GB is not.

Settings: GPU off (this stage is CPU only), Internet ON.
Expect roughly an hour.
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-micro.git"
WORK = "/kaggle/working"

# --- code -------------------------------------------------------------------
if os.path.exists(f"{WORK}/g-micro"):
    # A stale checkout from an earlier attempt in this same session would
    # silently run old code even after this script itself was re-fetched —
    # that is exactly what ran a T4 session at half speed on the bf16 fix.
    # Pulling forces the checkout to match what curl just downloaded.
    subprocess.run(["git", "-C", f"{WORK}/g-micro", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-micro"], check=True)
os.chdir(f"{WORK}/g-micro")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "tokenizers"], check=True)

# --- corpus -----------------------------------------------------------------
# A Hugging Face token lifts the anonymous rate limit. Put it in Kaggle's
# "Add-ons -> Secrets" as HF_TOKEN; without it this still works, just slower.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded from Kaggle secrets")
except Exception as e:
    print(f"no HF token ({type(e).__name__}) — downloading anonymously, slower")

# Already packed? Then there is nothing to do, and re-running costs seconds
# instead of an hour. Matters because a Kaggle batch run is all-or-nothing:
# any later cell that raises discards the whole output, and the natural
# response is to run the notebook again.
if os.path.exists(f"{WORK}/pl_train.bin") and os.path.getsize(f"{WORK}/pl_train.bin") > 1e9:
    print(f"pl_train.bin already built "
          f"({os.path.getsize(f'{WORK}/pl_train.bin')/1e9:.2f} GB) — nothing to do")
    sys.exit(0)


def complete(path):
    """A corpus file is finished iff its last line is the document separator.

    Worth checking rather than trusting: a download can die after writing most
    of the file, and re-downloading 6 GB because of an unverified assumption is
    expensive in both directions.
    """
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return False
    with open(path, "rb") as f:
        f.seek(-32, 2)
        return f.read().decode("utf-8", "ignore").strip().endswith("<|doc|>")


def fetch(source, out, extra=()):
    if complete(out):
        print(f"{out} already complete ({os.path.getsize(out)/1e9:.2f} GB) — skipping")
        return
    subprocess.run([sys.executable, "data/fetch_corpus.py", source,
                    "--out", out, *extra], check=True)


fetch("wiki", f"{WORK}/corpus_wiki.txt")
fetch("fineweb", f"{WORK}/corpus_web.txt", ("--max-chars", "6e9"))

# --- tokenizer --------------------------------------------------------------
# Prefer the mixed-corpus tokenizer if the repo has it; fall back to v1.
TOK = "data/tokenizer-v2.json" if os.path.exists("data/tokenizer-v2.json") else "data/tokenizer.json"
print(f"tokenizer: {TOK}")

# --- pack -------------------------------------------------------------------
subprocess.run([sys.executable, "data/pack_data.py",
                f"{WORK}/corpus_wiki.txt", f"{WORK}/corpus_web.txt",
                "--tokenizer", TOK,
                "--out-prefix", f"{WORK}/pl"], check=True)

# The raw text is ~9 GB and Kaggle's output quota is 20 GB. The binaries are
# all the training run needs, so drop the text before saving the notebook.
for f in ("corpus_wiki.txt", "corpus_web.txt"):
    os.remove(f"{WORK}/{f}")

subprocess.run(["cp", TOK, f"{WORK}/tokenizer.json"], check=True)
print("\ndone — save this notebook's output as a Dataset named 'microg-data'")
for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e9:.2f} GB")
