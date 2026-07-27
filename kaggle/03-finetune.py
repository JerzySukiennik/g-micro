"""
Kaggle cell 3 of 3 — instruction tuning.

Run after pretraining finishes. Turns the base model, which continues text,
into one that answers.

Settings: GPU T4 x2, Internet ON.
Inputs: the checkpoint dataset from 02-train.py (must contain run/best.pt).

Much shorter than pretraining — around 20 minutes. The model already knows
Polish; this only teaches it the shape of a conversation.

This file is documentation of intent, not what actually runs: run_finetune.py
pushes Niepotrzebne/kaggle-orchestration/kernel-finetune/finetune_kernel.py
as the kernel's code_file. The two are kept manually in sync — the same
mismatch already cost a full pretraining cycle once (kaggle/02-train.py vs
kernel/train_kernel.py, 2026-07-22).
"""

import glob
import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/g-micro.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/sft"

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

# ------------------------------------------------------------------- base --
# Kaggle's input mount depth isn't fixed (seen both /kaggle/input/<slug>/ and
# /kaggle/input/datasets/<owner>/<slug>/ in practice, in the same session for
# two different attached datasets) — recursive search finds it regardless of
# depth, where a fixed-depth glob silently found nothing and the whole run
# would fail this assert for a reason that looked like a missing dataset
# rather than a path assumption.
base = next(iter(sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True))), None) \
    or next(iter(sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))), None)
assert base, "no pretrained checkpoint in inputs — attach the 02-train output dataset"
print(f"base model: {base}")

# ------------------------------------------------------------------- data --
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

subprocess.run([sys.executable, "data/build_sft.py",
                "--tokenizer", "data/tokenizer-v2.json",
                "--out-prefix", f"{WORK}/pl_sft"], check=True)

# --------------------------------------------------------------- finetune --
subprocess.run([sys.executable, "train/finetune.py",
                "--init", base,
                "--data", f"{WORK}/pl_sft",
                "--out", OUT,
                "--batch-size", "8",
                "--grad-accum", "8",
                "--epochs", "3",
                "--lr", "3e-5",
                "--eval-every", "100",
                "--ckpt-every", "200",
                "--log-every", "20"], check=True)

# The instruction data is regenerable from the script; the weights are not.
for f in glob.glob(f"{WORK}/pl_sft_*.bin"):
    os.remove(f)

print("\ndone — download sft/best.pt, that is the chat model")
