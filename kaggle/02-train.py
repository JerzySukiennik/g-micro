"""
Kaggle cell 2 of 2 — pretraining.

Settings: Accelerator GPU T4 x2, Internet ON, Persistence "Variables and Files".
Inputs: the 'microg-data' Dataset produced by 01-prep.py, plus — on any run
after the first — the previous run's output as 'microg-ckpt'.

A Kaggle session is capped at 12 hours and can die sooner without warning, so
this is built to be interrupted. Everything needed to continue lands in
/kaggle/working/run every CKPT_EVERY steps; add that output as an input to the
next session and it picks up mid-stride.

This file is documentation of intent, not what actually runs: the automated
orchestrator (Niepotrzebne/kaggle-orchestration/orchestrate.py) pushes
Niepotrzebne/kaggle-orchestration/kernel/train_kernel.py as the kernel's
code_file. The two are kept manually in sync — a real resume-breaking bug
once lived only in the difference between them (this copy had the fix,
train_kernel.py didn't), so treat any edit here as incomplete until mirrored
there too.
"""

import os
import shutil
import subprocess
import sys
import glob

REPO = "https://github.com/JerzySukiennik/g-micro.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

# ---------------------------------------------------------------- schedule --
# 16 x 30 x 1024 = 491,520 tokens per step. 4060 steps = 2.0B tokens, matching
# the packed corpus exactly — one epoch, no repeats.
#
# Settled by measurement, not assumption: single-GPU (batch=8, same
# tokens/step) measured 13.2k tok/s; DataParallel across 2 GPUs measured
# 25.2k — almost exactly 2x, textbook data-parallel scaling. DataParallel's
# PCIe broadcast tax was the suspect; it is not the bottleneck. Per-GPU T4
# compute throughput is, and that is a harder problem than a one-line fix —
# accepting ~22h across 2 resumable sessions rather than chasing it further.
BATCH, ACCUM, STEPS, WARMUP = 16, 30, 4060, 200

if os.path.exists(f"{WORK}/g-micro"):
    # A stale checkout from an earlier attempt in this same session would
    # silently run old code even after this script itself was re-fetched —
    # that is exactly what ran a T4 session at half speed on the bf16 fix.
    # Pulling forces the checkout to match what curl just downloaded.
    subprocess.run(["git", "-C", f"{WORK}/g-micro", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/g-micro"], check=True)
os.chdir(f"{WORK}/g-micro")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)

# ------------------------------------------------------------------- data --
# Kaggle's input mount depth isn't fixed (seen both /kaggle/input/<slug>/ and
# /kaggle/input/datasets/<owner>/<slug>/ in practice) — recursive search finds
# it regardless. If nothing turns up, print the tree rather than just
# asserting — "not found" is useless when you cannot see what *is* there.
hits = glob.glob("/kaggle/input/**/pl_train.bin", recursive=True)
if not hits:
    print("pl_train.bin not found. /kaggle/input contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            size = os.path.getsize(os.path.join(root, f)) / 1e9
            print("  " * (depth + 1) + f"{f}  {size:.2f} GB")
    raise SystemExit("attach the microg-data dataset, or wait for it to finish building")

data_dir = os.path.dirname(hits[0])
print(f"data: {data_dir}")

# ------------------------------------------------------- resume if possible --
os.makedirs(OUT, exist_ok=True)
# Kaggle's actual input mount depth has moved before (plain /kaggle/input/<slug>/
# vs. /kaggle/input/datasets/<owner>/<slug>/ seen in practice) and a fixed-depth
# glob silently finds nothing when it moves again — "--resume" then trains from
# scratch with no error. Recursive search doesn't care how deep it's nested.
hits_ckpt = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
if not hits_ckpt:
    print("no ckpt.pt found under /kaggle/input — starting from scratch. tree:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 4:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            print("  " * (depth + 1) + f)
prev = hits_ckpt[0] if hits_ckpt else None
resume = []
if prev:
    shutil.copy(prev, f"{OUT}/ckpt.pt")
    best = prev.replace("ckpt.pt", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, f"{OUT}/best.pt")
    resume = ["--resume"]
    print(f"resuming from {prev}")
else:
    print("starting from scratch")

# ------------------------------------------------------------------ train --
cmd = [sys.executable, "train/train.py",
       "--data", f"{data_dir}/pl",
       "--out", OUT,
       "--batch-size", str(BATCH),
       "--grad-accum", str(ACCUM),
       "--max-steps", str(STEPS),
       "--warmup", str(WARMUP),
       "--eval-every", "100",
       "--ckpt-every", "100",     # ~13 min of work at risk if the session dies
       "--log-every", "10"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('microg-ckpt') to continue "
      "in the next session, or download run/best.pt if training finished.")
