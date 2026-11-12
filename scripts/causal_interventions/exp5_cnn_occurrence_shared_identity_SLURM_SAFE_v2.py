from __future__ import annotations

# ==================================================================================================
# WORK II — EXPERIMENT 5: CNN OCCURRENCE / SHARED-PARAMETER IDENTITY COMPARISON (RQ7)
# WITH INTRINSIC CONDITIONAL REALIZATION FIELDS OVER THE INPUT IMAGE
# MPI3D-TOY + FASHION-MNIST | FROZEN CNN CHECKPOINTS | HPC/SLURM SAFE | NO RETRAINING | STREAMING AGGREGATION
# ==================================================================================================
#
# SCIENTIFIC PURPOSE
# ------------------
# RQ7: For CNNs, is spatial use more informative than shared parameter identity?
#
# A convolutional parameter identity is reused at many spatial execution occurrences.  The established
# CNN conditional realization therefore contains two related carriers:
#
#   occurrence flow   F_a^c          : realized use at one spatial execution occurrence a
#   parameter flow    F_theta^c      : sum of F_a^c over all occurrences sharing parameter identity theta
#
# Experiment 5 compares intervention localization at these two resolutions while holding fixed:
#   - the same frozen CNN checkpoint;
#   - the same declared condition c=(x,0);
#   - the same output coordinate;
#   - alpha=0.25;
#   - the same LAYERWISE NUMBER OF PRIMITIVE SPATIAL EXECUTION OCCURRENCES attenuated.
#
# The shared-identity arm selects parameter identities by F_theta^c and attenuates every spatial reuse
# of each selected identity.  The occurrence arm receives exactly the same primitive-occurrence budget
# and selects the highest-flow individual occurrences from the same positive parameter-support universe.
# Thus the only intended difference is localization resolution.
#
# INTERPRETABILITY FIELDS
# -----------------------
# Field A (intrinsic conditional realization field):
#
#   P_in^c(s) = sum_k F^c_(input node k,s)
#
# This is the input-origin marginal of the complete normalized conditional path measure.  It is NOT a
# gradient saliency map and is not a new path primitive.  It is read directly from the established input
# node flow and satisfies sum_s P_in^c(s)=1 whenever the realization is defined.
#
# Field B (spatial intervention footprint): for the FIRST convolutional layer, whose source coordinates
# map exactly to the input lattice, sum baseline occurrence flow over selected occurrences by their input
# source coordinate.  Two fields are saved:
#
#   B_occ^c(s)    : footprint of the occurrence-selected intervention
#   B_param^c(s)  : footprint after selecting shared identities and unfolding all their spatial reuses
#
# The causal intervention itself acts across ALL convolutional layers.  Restricting Field B to the first
# convolution is only an image-space visualization rule; deeper occurrences are retained in native layer
# audits and are not silently projected onto pixels through an invented receptive-field convention.
#
# REQUIRED EXISTING FILES
# -----------------------
# Only immutable Experiment 1 resources are required:
#   <EXP1_RESULTS>/checkpoints/*.pt
#   <EXP1_RESULTS>/data/
#
# Experiment 2/3/4 tables are optional cohort-consistency audits only.
#
# DEFAULT OUTPUT — NO PRIOR EXPERIMENT IS TOUCHED
# ------------------------------------------------
#   pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results/
#       experiment5_cnn_occurrence_shared_identity/
#           tables/
#           figures/
#           fields/
#           run_parts/
#           audit/
#
# There is deliberately NO destructive "start fresh" mode.  Scientific result files are write-once.
# Existing Experiment 1–4 files are never modified.  Reruns resume from write-once per-condition parts.
#
# HPC RUN
# -------
#   srun python -u exp5_cnn_occurrence_shared_identity_SLURM_SAFE_v2.py
#
# JUPYTER ON AN ALLOCATED SLURM GPU NODE
# --------------------------------------
#   %run exp5_cnn_occurrence_shared_identity_SLURM_SAFE_v2.py
# ==================================================================================================

import os
import math
import json
import csv
import shutil
import time
import random
import copy
import hashlib
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    from torchvision.datasets import FashionMNIST
except Exception as e:
    raise RuntimeError("torchvision is required for Fashion-MNIST") from e

from tqdm.auto import tqdm
from scipy.stats import wilcoxon, spearmanr, ks_2samp

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================================================================================
# 0. CONFIGURATION
# ==================================================================================================

PROFILE = os.environ.get("PATHFLOW_EXP5_PROFILE", "l4_48h").strip().lower()
RESUME = os.environ.get("PATHFLOW_EXP5_RESUME", "1") != "0"
START_FRESH = os.environ.get("PATHFLOW_EXP5_START_FRESH", "0") == "1"
RUN_PREFLIGHT = os.environ.get("PATHFLOW_EXP5_PREFLIGHT", "1") != "0"
PREFLIGHT_ONLY = os.environ.get("PATHFLOW_EXP5_PREFLIGHT_ONLY", "0") == "1"
AGGREGATE_ONLY = os.environ.get("PATHFLOW_EXP5_AGGREGATE_ONLY", "0") == "1"
RUN_FIGURES = os.environ.get("PATHFLOW_EXP5_FIGURES", "1") != "0"
ALLOW_DOWNLOAD = os.environ.get("PATHFLOW_EXP5_ALLOW_DOWNLOAD", "1") != "0"
SELFTEST_ONLY = os.environ.get("PATHFLOW_EXP5_SELFTEST_ONLY", "0") == "1"
REQUIRE_CUDA = os.environ.get("PATHFLOW_EXP5_REQUIRE_CUDA", "1") != "0"
SOFT_WALL_HOURS = float(os.environ.get("PATHFLOW_EXP5_SOFT_WALL_HOURS", "46.0"))
SOFT_START_BUFFER_MIN = float(os.environ.get("PATHFLOW_EXP5_SOFT_START_BUFFER_MIN", "20.0"))
SCRIPT_START_MONOTONIC = time.perf_counter()

SEED = 730031
MODEL_SEED_BASE = 271828
ORDER_SEED_BASE = 141421
NOISE_SEED_BASE = 161803
MATCH_SEED_BASE = 223607
CONDITION_SEED_BASE = 244949

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("PATHFLOW_EXP5_CPU_THREADS", "4")))
torch.set_num_threads(max(1, CPU_THREADS))

def _script_or_submit_dir() -> Path:
    """Persistent default root for both Slurm and Jupyter/%run execution.

    Priority:
      1) directory containing this Python file, when __file__ is available;
      2) SLURM_SUBMIT_DIR, when supplied by Slurm;
      3) current working directory.

    We deliberately never default to /content or any Colab/Google-Drive path.
    """
    try:
        f = globals().get("__file__", None)
        if f:
            return Path(f).expanduser().resolve().parent
    except Exception:
        pass
    submit = os.environ.get("SLURM_SUBMIT_DIR", "").strip()
    if submit:
        return Path(submit).expanduser().resolve()
    return Path.cwd().resolve()


SCRIPT_DIR = _script_or_submit_dir()

# Experiment 5 is designed to run from the SAME working directory as Experiments 1, 2 and 3.
# It reuses only immutable Experiment 1 checkpoints and dataset files.
# All Experiment 5 outputs are isolated under a dedicated subtree.
_EXP1_RESULTS_ENV = os.environ.get("PATHFLOW_EXP1_RESULTS", "").strip()
EXP1_RESULTS = (
    Path(_EXP1_RESULTS_ENV).expanduser().resolve()
    if _EXP1_RESULTS_ENV
    else SCRIPT_DIR / "pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results"
)
SOURCE_CHECKPOINT_DIR = EXP1_RESULTS / "checkpoints"

_EXP5_OUTDIR_ENV = os.environ.get("PATHFLOW_EXP5_OUTDIR", "").strip()
OUTDIR = (
    Path(_EXP5_OUTDIR_ENV).expanduser().resolve()
    if _EXP5_OUTDIR_ENV
    else EXP1_RESULTS / "experiment5_cnn_occurrence_shared_identity"
)

_DATA_ROOT_ENV = os.environ.get("PATHFLOW_EXP5_DATA_ROOT", "").strip()
DATA_ROOT = (
    Path(_DATA_ROOT_ENV).expanduser().resolve()
    if _DATA_ROOT_ENV
    else EXP1_RESULTS / "data"
)

try:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "figures", "run_parts", "audit"]:
        (OUTDIR / sub).mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _write_probe = OUTDIR / "audit" / ".write_probe"
    _write_probe.write_text("ok", encoding="utf-8")
    _write_probe.unlink()
except Exception as e:
    raise RuntimeError(
        "Could not create/write the Experiment 5 output directory. "
        f"Resolved EXP1_RESULTS={EXP1_RESULTS}, OUTDIR={OUTDIR}. "
        "Run this script from the Experiment 1 working directory or set PATHFLOW_EXP1_RESULTS and "
        "PATHFLOW_EXP5_OUTDIR to a writable persistent Slurm/HPC path."
    ) from e

if START_FRESH:
    raise RuntimeError(
        "PATHFLOW_EXP5_START_FRESH=1 is intentionally disabled. Experiment 5 never deletes scientific data. "
        "Use the existing output directory to resume, or set PATHFLOW_EXP5_OUTDIR to a new empty directory."
    )

DATASETS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP5_DATASETS", "mpi3d,fmnist").split(",") if x.strip()]
ACTIVATIONS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP5_ACTIVATIONS", "relu,tanh,silu").split(",") if x.strip()]
OPTIMIZERS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP5_OPTIMIZERS", "sgdm,rmsprop,adam").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("PATHFLOW_EXP5_SEEDS", "0,1,2").split(",") if x.strip()]
ENVIRONMENTS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP5_ENVIRONMENTS", "clean,shifted").split(",") if x.strip()]

IMAGE_RESOLUTION = 16
N_TRAIN = int(os.environ.get("PATHFLOW_EXP5_NTRAIN", "20000"))
N_VAL = int(os.environ.get("PATHFLOW_EXP5_NVAL", "5000"))
N_TEST = int(os.environ.get("PATHFLOW_EXP5_NTEST", "5000"))
BATCH_SIZE = int(os.environ.get("PATHFLOW_EXP5_BATCH", "256"))
MAX_EPOCH = int(os.environ.get("PATHFLOW_EXP5_EPOCHS", "160"))
WEIGHT_DECAY = 0.0
ENDPOINT_INPUTS = int(os.environ.get("PATHFLOW_EXP5_ENDPOINT_INPUTS", "160"))
SHIFT_SNR_DB = float(os.environ.get("PATHFLOW_EXP5_SHIFT_SNR_DB", "15.0"))

LEARNING_RATES = {
    "sgdm": 1.0e-2,
    "rmsprop": 1.0e-3,
    "adam": 1.0e-3,
}

# Shared Experiment 1 targeting conventions; ATTENUATION=0.25 is used only by common fidelity sentinels.
CONDITIONS_PER_MODEL = int(os.environ.get("PATHFLOW_EXP5_CONDITIONS", "40"))
TARGET_FRACTION = float(os.environ.get("PATHFLOW_EXP5_TARGET_FRACTION", "0.10"))
ATTENUATION = float(os.environ.get("PATHFLOW_EXP5_ALPHA", "0.25"))
SAVE_EVERY_CONDITIONS = int(os.environ.get("PATHFLOW_EXP5_SAVE_EVERY", "5"))
MATCH_BINS = int(os.environ.get("PATHFLOW_EXP5_MATCH_BINS", "20"))
PREFLIGHT_TOL = float(os.environ.get("PATHFLOW_EXP5_PREFLIGHT_TOL", "5e-6"))

EPS = 1e-14
PATH_LOG_FLOOR = 1e-300
MANIFEST_EPS = 1e-12

MLP_ARCHS: Dict[str, List[int]] = {
    "M1_taper": [512, 384, 256, 128, 64],
    "M2_sustained": [384, 384, 384, 256, 128],
    "M3_bottleneck": [512, 256, 64, 256, 128],
}
CNN_ARCHS: Dict[str, Dict[str, Any]] = {
    "C1_taper": {"hidden_channels": [4, 8, 8, 12], "kernels": [3, 3, 3, 3, 1]},
    "C2_bottleneck": {"hidden_channels": [4, 8, 3, 8, 10], "kernels": [3, 3, 1, 3, 3, 1]},
    "C3_deepmix": {"hidden_channels": [3, 6, 8, 8, 10, 8], "kernels": [3, 3, 3, 1, 3, 1, 1]},
}
ARCH_IDS = list(MLP_ARCHS) + list(CNN_ARCHS)

if PROFILE == "pilot":
    DATASETS = ["fmnist"]
    ACTIVATIONS = ["relu"]
    OPTIMIZERS = ["adam"]
    SEEDS = [0]
    ARCH_IDS = ["M2_sustained", "C2_bottleneck"]
    N_TRAIN, N_VAL, N_TEST = 1024, 256, 256
    BATCH_SIZE = 128
    MAX_EPOCH = 4
    ENDPOINT_INPUTS = 16
    CONDITIONS_PER_MODEL = 4

if not (0.0 < TARGET_FRACTION <= 0.5):
    raise ValueError("TARGET_FRACTION must lie in (0, 0.5].")
if not (0.0 <= ATTENUATION < 1.0):
    raise ValueError("ATTENUATION must lie in [0,1).")
if CONDITIONS_PER_MODEL <= 0:
    raise ValueError("CONDITIONS_PER_MODEL must be positive.")
if any(e not in {"clean", "shifted"} for e in ENVIRONMENTS):
    raise ValueError(f"Unsupported environments: {ENVIRONMENTS}")
if SOFT_WALL_HOURS <= 0:
    raise ValueError("SOFT_WALL_HOURS must be positive.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision("high")

# ==================================================================================================
# 1. GENERIC HELPERS
# ==================================================================================================

def banner(s: str):
    print("\n" + "=" * 116)
    print(s)
    print("=" * 116)


def human_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ETAReporter:
    def __init__(self, stage: str, total: int, window: int = 8):
        self.stage = stage
        self.total = max(int(total), 1)
        self.window = max(int(window), 3)
        self.start = time.perf_counter()
        self.durations: List[float] = []

    def update(self, duration: float, completed: int):
        self.durations.append(float(duration))
        med = float(np.median(self.durations[-self.window:]))
        remain = max(self.total - completed, 0)
        elapsed = time.perf_counter() - self.start
        tqdm.write(
            f"[ETA:{self.stage}] {completed}/{self.total} | current {human_time(duration)} | "
            f"rolling median {human_time(med)}/item | remaining ~{human_time(med * remain)} | "
            f"elapsed {human_time(elapsed)}"
        )

    def finish(self):
        tqdm.write(f"[ETA:{self.stage}] finished in {human_time(time.perf_counter() - self.start)}")


def stable_hash_dict(d: Dict[str, Any], n: int = 12) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:n]


def atomic_csv(df: pd.DataFrame, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(obj: Any, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_log(x: float) -> float:
    try:
        x = float(x)
        return float(math.log(max(x, EPS))) if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def model_seed(arch: str, dataset: str, seed_index: int) -> int:
    return int(MODEL_SEED_BASE + 1000003 * seed_index + int(stable_hash_dict({"a": arch, "d": dataset}, 8), 16) % 999983)


def family_of(arch: str) -> str:
    return "mlp" if arch in MLP_ARCHS else "cnn"


def output_dim_for(ds: str) -> int:
    return 10 if ds == "fmnist" else 1


def task_for(ds: str) -> str:
    return "classification" if ds == "fmnist" else "regression"


def checkpoint_path(run_id: str) -> Path:
    return SOURCE_CHECKPOINT_DIR / f"{run_id}__best.pt"


def training_part_path(run_id: str, kind: str) -> Path:
    return OUTDIR / "unused_training_parts" / f"{run_id}__{kind}.csv"


def part_path(run_id: str, kind: str) -> Path:
    return OUTDIR / "run_parts" / f"{run_id}__{kind}.csv"


def done_path(run_id: str) -> Path:
    return OUTDIR / "run_parts" / f"{run_id}__DONE.json"


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    if m == 0:
        return np.asarray([], float)
    out = np.ones(m, float)
    order = np.argsort(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    return out


def bootstrap_median_ci(d: np.ndarray, seed: int = 0, B: int = 2000) -> Tuple[float, float]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED + seed)
    vals = np.empty(B, float)
    for b in range(B):
        vals[b] = np.median(rng.choice(d, size=len(d), replace=True))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def hodges_lehmann_onesample(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    vals = [0.5 * (d[i] + d[j]) for i in range(len(d)) for j in range(i, len(d))]
    return float(np.median(vals))


def paired_rank_biserial(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy(float)
    wp = float(ranks[d > 0].sum())
    wm = float(ranks[d < 0].sum())
    den = wp + wm
    return float((wp - wm) / den) if den > 0 else 0.0


def hash_indices(mask: np.ndarray) -> str:
    ids = np.flatnonzero(np.asarray(mask, bool)).astype(np.int64)
    return hashlib.sha1(ids.tobytes()).hexdigest()[:16]


def tensor_close_max(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(torch.max(torch.abs(a.to(torch.float64) - b.to(torch.float64))).item()) if a.numel() else 0.0


def soft_wall_remaining_seconds() -> float:
    return max(0.0, SOFT_WALL_HOURS * 3600.0 - (time.perf_counter() - SCRIPT_START_MONOTONIC))


def can_start_new_unit(stage: str, extra_buffer_min: Optional[float] = None) -> bool:
    """Conservative Slurm wall-time guard.

    It is checked only between atomic units (trained models / intervention models), never in the
    middle of a scientific state update. If the guard fires, all completed work is already on disk
    and the same script can be resumed in a new allocation.
    """
    buffer_min = SOFT_START_BUFFER_MIN if extra_buffer_min is None else float(extra_buffer_min)
    remaining = soft_wall_remaining_seconds()
    if remaining <= buffer_min * 60.0:
        msg = (
            f"[soft-wall] Not starting next {stage}: remaining soft-wall budget "
            f"{human_time(remaining)} <= safety buffer {buffer_min:.1f} min. Resume in a new allocation."
        )
        print(msg)
        atomic_json({
            "stage": stage,
            "remaining_seconds": remaining,
            "soft_wall_hours": SOFT_WALL_HOURS,
            "safety_buffer_minutes": buffer_min,
            "timestamp_unix": time.time(),
        }, OUTDIR / "audit" / "soft_wall_stop.json")
        return False
    return True

# ==================================================================================================
# 2. SELF-CONTAINED TRAINING MANIFEST / DETERMINISTIC TRAINING HELPERS
# ==================================================================================================

def deterministic_batches(n: int, epoch: int, seed_index: int, batch_size: int):
    rng = np.random.default_rng(ORDER_SEED_BASE + 104729 * seed_index + int(epoch))
    ids = rng.permutation(int(n))
    for lo in range(0, len(ids), batch_size):
        yield ids[lo:min(lo + batch_size, len(ids))]


def build_manifest() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for arch in ARCH_IDS:
            for opt in OPTIMIZERS:
                for act in ACTIVATIONS:
                    for seed in SEEDS:
                        rid = f"BENCH__{ds}__{arch}__{opt}__{act}__S{seed:02d}"
                        rows.append({
                            "run_id": rid,
                            "dataset": ds,
                            "task": task_for(ds),
                            "family": family_of(arch),
                            "arch_id": arch,
                            "optimizer": opt,
                            "activation": act,
                            "seed_index": seed,
                        })
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "run_manifest.csv")
    return df


def training_complete(run_id: str) -> bool:
    return (
        checkpoint_path(run_id).exists()
        and training_part_path(run_id, "summary").exists()
        and training_part_path(run_id, "history").exists()
    )

# ==================================================================================================
# 3. DATASETS â€” COPIED IN LOGIC FROM THE AUDITED BENCHMARK
# ==================================================================================================

MPI3D_URL = os.environ.get(
    "MPI3D_URL",
    "https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_toy.npz",
)


def _resize_images_np(x: np.ndarray, size: int) -> np.ndarray:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 3:
        t = t[:, None, :, :]
    elif t.ndim == 4:
        if t.shape[-1] in (1, 3, 4):
            t = t.permute(0, 3, 1, 2)
        elif t.shape[1] not in (1, 3, 4):
            raise ValueError(f"Unsupported 4D image layout: {tuple(t.shape)}")
    else:
        raise ValueError(f"Unsupported image array shape: {tuple(t.shape)}")
    with torch.no_grad():
        y = F.interpolate(t, size=(size, size), mode="area")
    return y.numpy().astype(np.float32)


def _split_indices_stratified(labels: np.ndarray, ntr: int, nv: int, nt: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    tr, va, te = [], [], []
    total = ntr + nv + nt
    for c in classes:
        ids = np.where(labels == c)[0]
        rng.shuffle(ids)
        take = max(1, int(round(total * len(ids) / len(labels))))
        ids = ids[:min(take, len(ids))]
        a = max(1, int(round(ntr * len(ids) / total)))
        b = max(1, int(round(nv * len(ids) / total)))
        tr.extend(ids[:a])
        va.extend(ids[a:a + b])
        te.extend(ids[a + b:])

    def fill(arr, target, pool):
        arr = list(dict.fromkeys(arr))
        need = target - len(arr)
        if need > 0:
            remain = np.setdiff1d(pool, np.asarray(arr, int), assume_unique=False)
            rng.shuffle(remain)
            arr.extend(remain[:need].tolist())
        return np.asarray(arr[:target], int)

    pool = np.arange(len(labels))
    tr = fill(tr, ntr, pool)
    va_pool = np.setdiff1d(pool, tr)
    va = fill(va, nv, va_pool)
    te_pool = np.setdiff1d(pool, np.concatenate([tr, va]))
    te = fill(te, nt, te_pool)
    return tr, va, te


def balanced_indices_from_strata(strata: np.ndarray, n: int, seed: int) -> np.ndarray:
    strata = np.asarray(strata)
    classes = np.asarray(sorted(np.unique(strata).tolist()))
    if len(classes) == 0:
        return np.array([], dtype=int)
    n = min(int(n), len(strata))
    rng = np.random.default_rng(int(seed))
    base = n // len(classes)
    rem = n % len(classes)
    extra = set(rng.permutation(classes)[:rem].tolist())
    chosen: Dict[Any, List[int]] = {}
    leftovers: List[int] = []
    for c in classes:
        ids = np.where(strata == c)[0].copy()
        rng.shuffle(ids)
        take = base + (1 if c in extra else 0)
        chosen[c] = ids[:min(take, len(ids))].tolist()
        leftovers.extend(ids[min(take, len(ids)):].tolist())
    selected = sum((v for v in chosen.values()), [])
    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:n - len(selected)])
    queues = {c: list(chosen[c]) for c in classes}
    out: List[int] = []
    while len(out) < min(n, sum(len(v) for v in queues.values())):
        moved = False
        for c in classes:
            if queues[c] and len(out) < n:
                out.append(queues[c].pop(0))
                moved = True
        if not moved:
            break
    if len(out) < n:
        used = set(out)
        tail = [i for i in selected if i not in used]
        out.extend(tail[:n - len(out)])
    return np.asarray(out[:n], dtype=int)


def _ensure_mpi3d_npz() -> Path:
    env = os.environ.get("MPI3D_NPZ", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend([
        DATA_ROOT / "mpi3d_toy.npz",
        SCRIPT_DIR / "mpi3d_toy.npz",
        SCRIPT_DIR / "data" / "mpi3d_toy.npz",
        Path.cwd() / "mpi3d_toy.npz",
        Path.cwd() / "data" / "mpi3d_toy.npz",
        Path.home() / "data" / "mpi3d_toy.npz",
        Path.home() / "datasets" / "mpi3d_toy.npz",
    ])
    for p in candidates:
        if p.exists():
            return p
    if not ALLOW_DOWNLOAD:
        raise FileNotFoundError(
            "MPI3D-Toy NPZ was not found. Checked:\n  " + "\n  ".join(str(p) for p in candidates) +
            "\nSet MPI3D_NPZ=/path/to/mpi3d_toy.npz or PATHFLOW_EXP5_ALLOW_DOWNLOAD=1."
        )
    p = DATA_ROOT / "mpi3d_toy.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    print(f"Downloading MPI3D-Toy to {p} ...")
    try:
        urllib.request.urlretrieve(MPI3D_URL, tmp)
        # Basic integrity check before promoting the download to the persistent dataset path.
        with np.load(tmp, allow_pickle=False) as z:
            if "images" not in z.files:
                raise RuntimeError(f"Downloaded MPI3D archive has keys {z.files}; expected 'images'.")
        tmp.replace(p)
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise RuntimeError(
            "MPI3D-Toy was not found locally and automatic download failed. "
            "On compute nodes without outbound internet, place mpi3d_toy.npz on persistent storage "
            "and set MPI3D_NPZ=/absolute/path/to/mpi3d_toy.npz."
        ) from e
    return p


def load_mpi3d() -> Dict[str, Any]:
    p = _ensure_mpi3d_npz()
    factor_sizes = np.asarray([6, 6, 2, 3, 3, 40, 40], dtype=np.int64)
    expected_n = int(np.prod(factor_sizes))
    all_ids = np.arange(expected_n, dtype=np.int64)
    xpos = ((all_ids // int(factor_sizes[-1])) % int(factor_sizes[-2])).astype(np.int64)
    tr, va, te = _split_indices_stratified(xpos, N_TRAIN, N_VAL, N_TEST, SEED + 29)
    ids = np.concatenate([tr, va, te])

    with np.load(p, allow_pickle=False) as z:
        if "images" not in z.files:
            raise KeyError(f"MPI3D NPZ has keys {z.files}; expected 'images'.")
        images = z["images"]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Unexpected MPI3D shape {images.shape}; expected [N,64,64,3].")
        if images.shape[0] != expected_n:
            raise ValueError(f"Unexpected MPI3D population size {images.shape[0]}; expected {expected_n}.")
        imgs = images[ids].astype(np.float32) / 255.0

    imgs = _resize_images_np(imgs, IMAGE_RESOLUTION)
    ntr, nv = len(tr), len(va)
    Xtr, Xv, Xt = imgs[:ntr], imgs[ntr:ntr + nv], imgs[ntr + nv:]
    yall = xpos[ids].astype(np.float32)
    ytr, yv, yt = yall[:ntr], yall[ntr:ntr + nv], yall[ntr + nv:]

    mu = float(Xtr.mean())
    sd = float(Xtr.std() + 1e-8)
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xv = ((Xv - mu) / sd).astype(np.float32)
    Xt = ((Xt - mu) / sd).astype(np.float32)

    ym = float(ytr.mean())
    ys = float(ytr.std() + 1e-8)
    ytr_n = ((ytr - ym) / ys).astype(np.float32)[:, None]
    yv_n = ((yv - ym) / ys).astype(np.float32)[:, None]
    yt_n = ((yt - ym) / ys).astype(np.float32)[:, None]

    data = {
        "task": "regression",
        "output_dim": 1,
        "train": (torch.from_numpy(Xtr), torch.from_numpy(ytr_n)),
        "val": (torch.from_numpy(Xv), torch.from_numpy(yv_n)),
        "test": (torch.from_numpy(Xt), torch.from_numpy(yt_n)),
        "train_strata": xpos[tr].copy(),
        "val_strata": xpos[va].copy(),
        "test_strata": xpos[te].copy(),
        "meta": {
            "dataset": "mpi3d",
            "variant": "mpi3d_toy",
            "input_channels": 3,
            "input_mean": mu,
            "input_sd": sd,
            "target_mean": ym,
            "target_sd": ys,
            "target": "horizontal_axis",
            "target_levels": 40,
            "resolution": IMAGE_RESOLUTION,
            "npz": str(p),
        },
    }
    return attach_endpoint_and_causal_population(data, "mpi3d")


def load_fmnist() -> Dict[str, Any]:
    root = Path(os.environ.get("PATHFLOW_EXP5_FMNIST_ROOT", str(DATA_ROOT / "fmnist"))).expanduser().resolve()
    download = ALLOW_DOWNLOAD
    ds_tr = FashionMNIST(root=str(root), train=True, download=download)
    ds_te = FashionMNIST(root=str(root), train=False, download=download)
    Xtr0 = ds_tr.data.numpy().astype(np.float32) / 255.0
    ytr0 = ds_tr.targets.numpy().astype(np.int64)
    Xte0 = ds_te.data.numpy().astype(np.float32) / 255.0
    yte0 = ds_te.targets.numpy().astype(np.int64)
    Xtr0 = _resize_images_np(Xtr0, IMAGE_RESOLUTION)
    Xte0 = _resize_images_np(Xte0, IMAGE_RESOLUTION)
    tr, va, _ = _split_indices_stratified(ytr0, N_TRAIN, N_VAL, max(1, min(1000, len(ytr0) - N_TRAIN - N_VAL)), SEED + 11)
    _, _, te = _split_indices_stratified(
        yte0,
        max(1, min(1000, len(yte0) - N_TEST - 1)),
        max(1, min(500, len(yte0) - N_TEST - 1)),
        N_TEST,
        SEED + 17,
    )
    mu = float(Xtr0[tr].mean())
    sd = float(Xtr0[tr].std() + 1e-8)
    Xtr = ((Xtr0[tr] - mu) / sd).astype(np.float32)
    Xv = ((Xtr0[va] - mu) / sd).astype(np.float32)
    Xt = ((Xte0[te] - mu) / sd).astype(np.float32)
    data = {
        "task": "classification",
        "output_dim": 10,
        "train": (torch.from_numpy(Xtr), torch.from_numpy(ytr0[tr])),
        "val": (torch.from_numpy(Xv), torch.from_numpy(ytr0[va])),
        "test": (torch.from_numpy(Xt), torch.from_numpy(yte0[te])),
        "train_strata": ytr0[tr].copy(),
        "val_strata": ytr0[va].copy(),
        "test_strata": yte0[te].copy(),
        "meta": {
            "dataset": "fmnist",
            "input_mean": mu,
            "input_sd": sd,
            "resolution": IMAGE_RESOLUTION,
        },
    }
    return attach_endpoint_and_causal_population(data, "fmnist")


def attach_endpoint_and_causal_population(data: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    test_strata = np.asarray(data["test_strata"])
    endpoint_ids = balanced_indices_from_strata(
        test_strata,
        ENDPOINT_INPUTS,
        SEED + 1009 + (0 if dataset == "fmnist" else 1),
    )
    endpoint_strata = test_strata[endpoint_ids]
    causal_local_ids = balanced_indices_from_strata(
        endpoint_strata,
        min(CONDITIONS_PER_MODEL, len(endpoint_ids)),
        CONDITION_SEED_BASE + (0 if dataset == "fmnist" else 1),
    )
    data["endpoint_indices"] = endpoint_ids
    data["causal_endpoint_local_indices"] = causal_local_ids
    data["causal_test_indices"] = endpoint_ids[causal_local_ids]
    vals, counts = np.unique(endpoint_strata[causal_local_ids], return_counts=True)
    data["causal_balance"] = {str(v): int(c) for v, c in zip(vals, counts)}
    return data


def load_datasets() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if "mpi3d" in DATASETS:
        out["mpi3d"] = load_mpi3d()
    if "fmnist" in DATASETS:
        out["fmnist"] = load_fmnist()
    rows = []
    for ds, d in out.items():
        endpoint_ids = np.asarray(d["endpoint_indices"], int)
        causal_local = np.asarray(d["causal_endpoint_local_indices"], int)
        strata = np.asarray(d["test_strata"])
        for j, local_id in enumerate(causal_local):
            test_id = int(endpoint_ids[local_id])
            rows.append({
                "dataset": ds,
                "condition_index": int(j),
                "endpoint_local_index": int(local_id),
                "test_index": test_id,
                "stratum": strata[test_id],
            })
    atomic_csv(pd.DataFrame(rows), OUTDIR / "tables" / "causal_condition_population.csv")
    return out


def shifted_environment(X: torch.Tensor, dataset: str, dataset_meta: Dict[str, Any]) -> torch.Tensor:
    x = X.float()
    mu = float(dataset_meta["input_mean"])
    sd = float(dataset_meta["input_sd"])
    raw = x * sd + mu
    signal = float(torch.mean(raw * raw).item())
    noise_var = signal / (10.0 ** (SHIFT_SNR_DB / 10.0))
    g = torch.Generator(device="cpu").manual_seed(
        NOISE_SEED_BASE + int(stable_hash_dict({"d": dataset}, 8), 16) % 1000003
    )
    noise = torch.randn(raw.shape, generator=g, dtype=raw.dtype) * math.sqrt(max(noise_var, EPS))
    shifted_raw = raw + noise
    return (shifted_raw - mu) / max(sd, EPS)


def prepared_endpoint_environment(data: Dict[str, Any], dataset: str, environment: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Reproduce the benchmark ordering exactly: select the 160 endpoint population first, then shift it.

    The deterministic noise generator is shape-dependent. Shifting only the 40 causal conditions would
    not reproduce the benchmark's shifted endpoint values, so we shift the full frozen endpoint population
    and only then take the frozen causal subset.
    """
    Xt, yt = data["test"]
    endpoint_ids = np.asarray(data["endpoint_indices"], int)
    causal_local = np.asarray(data["causal_endpoint_local_indices"], int)
    Xep = Xt[endpoint_ids]
    yep = yt[endpoint_ids]
    if environment == "shifted":
        Xep = shifted_environment(Xep, dataset, data["meta"])
    return Xep[causal_local], yep[causal_local], causal_local

# ==================================================================================================
# 4. MODELS / ACTIVATIONS â€” EXACT BENCHMARK DEFINITIONS
# ==================================================================================================

class ActivationModule(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, x):
        if self.name == "relu":
            return F.relu(x)
        if self.name == "tanh":
            return torch.tanh(x)
        if self.name == "silu":
            return F.silu(x)
        raise ValueError(self.name)


@torch.no_grad()
def finite_secant_torch64(name: str, z: torch.Tensor, z0: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    z = z.to(torch.float64)
    z0 = z0.to(torch.float64)
    dz = z - z0
    if name == "relu":
        f = lambda q: torch.clamp_min(q, 0.0)
    elif name == "tanh":
        f = torch.tanh
    elif name == "silu":
        f = F.silu
    else:
        raise ValueError(name)
    out = torch.empty_like(dz)
    mask = torch.abs(dz) > tol
    if mask.any():
        out[mask] = (f(z[mask]) - f(z0[mask])) / dz[mask]
    if (~mask).any():
        q = .5 * (z[~mask] + z0[~mask])
        if name == "relu":
            der = (q > 0).to(q.dtype)
        elif name == "tanh":
            t = torch.tanh(q)
            der = 1 - t * t
        else:
            s = torch.sigmoid(q)
            der = s + q * s * (1 - s)
        out[~mask] = der
    return out


class DeepMLP(nn.Module):
    def __init__(self, hidden: Sequence[int], activation: str, output_dim: int, seed: int, input_channels: int = 1):
        super().__init__()
        self.activation_name = activation
        self.act = ActivationModule(activation)
        self.input_channels = int(input_channels)
        dims = [self.input_channels * IMAGE_RESOLUTION * IMAGE_RESOLUTION] + list(hidden) + [output_dim]
        torch.manual_seed(seed)
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(dims) - 1)])
        self.reset_common(seed)

    def reset_common(self, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for l in self.layers:
                W = torch.randn(l.weight.shape, generator=g, dtype=l.weight.dtype) / math.sqrt(max(l.in_features, 1))
                l.weight.copy_(W)

    def forward(self, x):
        h = x.reshape(x.shape[0], -1)
        for l in self.layers[:-1]:
            h = self.act(l(h))
        return self.layers[-1](h)

    def forward_with_states(self, x):
        h = x.reshape(x.shape[0], -1)
        zs = []
        hs = [h]
        for l in self.layers[:-1]:
            z = l(h)
            zs.append(z)
            h = self.act(z)
            hs.append(h)
        out = self.layers[-1](h)
        hs.append(out)
        return out, zs, hs

    def parameter_count(self):
        return int(sum(p.numel() for p in self.parameters()))


class DeepCNN(nn.Module):
    def __init__(self, spec: Dict[str, Any], activation: str, output_dim: int, seed: int, input_channels: int = 1):
        super().__init__()
        self.activation_name = activation
        self.act = ActivationModule(activation)
        self.input_channels = int(input_channels)
        hidden = list(spec["hidden_channels"])
        kernels = list(spec["kernels"])
        chans = [self.input_channels] + hidden + [output_dim]
        if len(kernels) != len(chans) - 1:
            raise ValueError("CNN kernels mismatch")
        self.channels = chans
        self.kernels = kernels
        torch.manual_seed(seed)
        self.convs = nn.ModuleList([
            nn.Conv2d(chans[i], chans[i + 1], kernel_size=kernels[i], stride=1, padding=0, bias=False)
            for i in range(len(kernels))
        ])
        self.reset_common(seed)
        if self.output_resolution() < 1:
            raise ValueError("CNN output resolution <1")

    def reset_common(self, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for c in self.convs:
                W = torch.randn(c.weight.shape, generator=g, dtype=c.weight.dtype) / math.sqrt(
                    max(c.in_channels * c.kernel_size[0] * c.kernel_size[1], 1)
                )
                c.weight.copy_(W)

    def spatial_shapes(self):
        h = w = IMAGE_RESOLUTION
        out = [(h, w)]
        for k in self.kernels:
            h = h - k + 1
            w = w - k + 1
            out.append((h, w))
        return out

    def output_resolution(self):
        return int(self.spatial_shapes()[-1][0])

    def forward(self, x):
        h = x
        for i, c in enumerate(self.convs):
            z = c(h)
            h = self.act(z) if i < len(self.convs) - 1 else z
        return h.mean(dim=(-1, -2))

    def forward_with_states(self, x):
        h = x
        zs = []
        hs = [x]
        for i, c in enumerate(self.convs):
            z = c(h)
            if i < len(self.convs) - 1:
                zs.append(z)
                h = self.act(z)
                hs.append(h)
            else:
                h = z
                hs.append(h)
        return h.mean(dim=(-1, -2)), zs, hs

    def parameter_count(self):
        return int(sum(p.numel() for p in self.parameters()))


def build_model(arch_id: str, activation: str, output_dim: int, dataset: str, seed_index: int) -> nn.Module:
    s = model_seed(arch_id, dataset, seed_index)
    input_channels = 1 if dataset == "fmnist" else 3
    if arch_id in MLP_ARCHS:
        return DeepMLP(MLP_ARCHS[arch_id], activation, output_dim, s, input_channels=input_channels).to(DEVICE)
    if arch_id in CNN_ARCHS:
        return DeepCNN(CNN_ARCHS[arch_id], activation, output_dim, s, input_channels=input_channels).to(DEVICE)
    raise KeyError(f"Unknown architecture {arch_id}")

# ==================================================================================================
# 5. INITIAL-SIGNAL CALIBRATION / OPTIMIZERS / TRAINING
# ==================================================================================================

@torch.no_grad()
def calibrate_initial_signal(model: nn.Module, Xcal: torch.Tensor, target_var: float = 1.0):
    model.eval()
    h = Xcal[:min(len(Xcal), 256)].to(DEVICE)
    scales = []
    if isinstance(model, DeepMLP):
        h = h.reshape(h.shape[0], -1)
        for l in model.layers[:-1]:
            z = l(h)
            sd = float(torch.std(z, unbiased=False).item())
            sc = float(np.clip(math.sqrt(target_var) / max(sd, 1e-6), .15, 6.0))
            l.weight.mul_(sc)
            h = model.act(l(h))
            scales.append(sc)
        scales.append(1.0)
    else:
        for i, c in enumerate(model.convs):
            z = c(h)
            if i < len(model.convs) - 1:
                sd = float(torch.std(z, unbiased=False).item())
                sc = float(np.clip(math.sqrt(target_var) / max(sd, 1e-6), .15, 6.0))
                c.weight.mul_(sc)
                h = model.act(c(h))
                scales.append(sc)
            else:
                scales.append(1.0)
    return scales


def make_optimizer(model: nn.Module, name: str):
    lr = LEARNING_RATES[name]
    if name == "sgdm":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=WEIGHT_DECAY)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=.99, weight_decay=WEIGHT_DECAY)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    raise ValueError(name)


@torch.no_grad()
def evaluate_performance(model: nn.Module, X: torch.Tensor, y: torch.Tensor, task: str, chunk: int = 1024) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for lo in range(0, len(X), chunk):
        xb = X[lo:lo + chunk].to(DEVICE)
        yb = y[lo:lo + chunk].to(DEVICE)
        pred = model(xb)
        if task == "classification":
            loss = F.cross_entropy(pred, yb, reduction="sum")
            correct += int((pred.argmax(1) == yb).sum().item())
            total += len(yb)
        else:
            loss = torch.sum((pred - yb) ** 2)
            total += int(yb.numel())
        total_loss += float(loss.item())
    out = {"loss": total_loss / max(total, 1)}
    if task == "classification":
        out["accuracy"] = correct / max(total, 1)
    return out


def loss_batch(pred: torch.Tensor, y: torch.Tensor, task: str):
    return F.cross_entropy(pred, y) if task == "classification" else F.mse_loss(pred, y)


def train_or_resume_one(row: pd.Series, data: Dict[str, Any]):
    """Exact benchmark training logic, stopping only to retain the best-validation checkpoint.

    The trained checkpoint is persisted before any intervention analysis. Resume never retrains a
    completed model. The model is returned only when needed by callers; the normal training stage
    releases it immediately to keep VRAM bounded.
    """
    rid = str(row.run_id)
    if RESUME and training_complete(rid):
        return

    Xtr, ytr = data["train"]
    Xv, yv = data["val"]
    Xt, yt = data["test"]
    cp = checkpoint_path(rid)
    model = build_model(str(row.arch_id), str(row.activation), output_dim_for(str(row.dataset)), str(row.dataset), int(row.seed_index))

    if RESUME and cp.exists():
        chk = torch.load(cp, map_location="cpu")
        model.load_state_dict(chk["state_dict"])
        meta = dict(chk.get("meta", {}))
        meta.update({
            "run_id": rid,
            "dataset": row.dataset,
            "task": row.task,
            "family": row.family,
            "arch_id": row.arch_id,
            "optimizer": row.optimizer,
            "activation": row.activation,
            "seed_index": int(row.seed_index),
            "parameter_count": model.parameter_count(),
        })
        hist_df = safe_read_csv(training_part_path(rid, "history"))
        hist = hist_df.to_dict("records") if not hist_df.empty else []
    else:
        cal_scales = calibrate_initial_signal(model, Xtr)
        opt = make_optimizer(model, str(row.optimizer))
        task = str(row.task)
        init = evaluate_performance(model, Xv, yv, task)
        best_metric = init["loss"]
        best_epoch = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hist = []
        diverged = False

        epoch_bar = tqdm(range(1, MAX_EPOCH + 1), desc=f"Train {rid}", leave=False)
        for epoch in epoch_bar:
            model.train()
            losses = []
            for ids in deterministic_batches(len(Xtr), epoch - 1, int(row.seed_index), BATCH_SIZE):
                xb = Xtr[ids].to(DEVICE)
                yb = ytr[ids].to(DEVICE)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_batch(pred, yb, task)
                if not torch.isfinite(loss):
                    diverged = True
                    break
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            if diverged:
                break
            val = evaluate_performance(model, Xv, yv, task)
            hist.append({
                "run_id": rid,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else np.nan,
                "val_loss": val["loss"],
                "val_accuracy": val.get("accuracy", np.nan),
            })
            epoch_bar.set_postfix(val=f"{val['loss']:.4g}", best=f"{best_metric:.4g}")
            if np.isfinite(val["loss"]) and val["loss"] < best_metric:
                best_metric = val["loss"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not np.isfinite(val["loss"]) or val["loss"] > 1e10:
                diverged = True
                break

        model.load_state_dict(best_state)
        model.to(DEVICE).eval()
        ptest = evaluate_performance(model, Xt, yt, task)
        meta = {
            "run_id": rid,
            "dataset": row.dataset,
            "task": task,
            "family": row.family,
            "arch_id": row.arch_id,
            "optimizer": row.optimizer,
            "activation": row.activation,
            "seed_index": int(row.seed_index),
            "parameter_count": model.parameter_count(),
            "best_epoch": best_epoch,
            "best_val_loss": best_metric,
            "best_test_loss": ptest["loss"],
            "best_test_accuracy": ptest.get("accuracy", np.nan),
            "diverged": diverged,
            "calibration_scales_json": json.dumps([float(x) for x in cal_scales]),
        }
        torch.save({"state_dict": best_state, "meta": meta}, cp)
        atomic_csv(pd.DataFrame([meta]), training_part_path(rid, "summary"))
        atomic_csv(pd.DataFrame(hist), training_part_path(rid, "history"))
        del opt

    if not training_part_path(rid, "summary").exists():
        atomic_csv(pd.DataFrame([meta]), training_part_path(rid, "summary"))
    if not training_part_path(rid, "history").exists():
        atomic_csv(pd.DataFrame(hist), training_part_path(rid, "history"))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def aggregate_training_parts(manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    summaries, histories = [], []
    for rid in manifest.run_id.astype(str):
        s = safe_read_csv(training_part_path(rid, "summary"))
        h = safe_read_csv(training_part_path(rid, "history"))
        if len(s):
            summaries.append(s)
        if len(h):
            histories.append(h)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    atomic_csv(summary, OUTDIR / "tables" / "training_summary.csv")
    atomic_csv(history, OUTDIR / "tables" / "training_history.csv")
    return {"summary": summary, "history": history}

# ==================================================================================================
# 5. BASELINE PATH STATES â€” COPIED FROM THE BENCHMARK LOGIC
# ==================================================================================================

def _normalize_rows(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    s = t.sum(dim=tuple(range(1, t.ndim)))
    valid = torch.isfinite(s) & (s > 0)
    safe = torch.where(valid, s, torch.ones_like(s))
    out = t / safe.reshape((-1,) + ((1,) * (t.ndim - 1)))
    out = torch.where(valid.reshape((-1,) + ((1,) * (t.ndim - 1))), out, torch.zeros_like(out))
    return out, s


@torch.no_grad()
def mlp_response_sq(model: DeepMLP, X: torch.Tensor, X0: torch.Tensor) -> List[torch.Tensor]:
    _, z, _ = model.forward_with_states(X.to(DEVICE))
    _, z0, _ = model.forward_with_states(X0.to(DEVICE))
    return [torch.square(finite_secant_torch64(model.activation_name, a, b)) for a, b in zip(z, z0)]


@torch.no_grad()
def mlp_log_partition_q(model: DeepMLP, source_sq: torch.Tensor, response_sq: List[torch.Tensor], target_idx: torch.Tensor, q: float) -> torch.Tensor:
    src = torch.pow(torch.clamp(source_sq.to(DEVICE, dtype=torch.float64), min=0.0), float(q))
    a, s0 = _normalize_rows(src)
    valid = torch.isfinite(s0) & (s0 > 0)
    logscale = torch.where(valid, torch.log(torch.clamp(s0, min=EPS)), torch.full_like(s0, -torch.inf))
    for i, l in enumerate(model.layers[:-1]):
        w = torch.pow(torch.square(l.weight.to(torch.float64)), float(q))
        a = a @ w.T
        a = a * torch.pow(torch.clamp(response_sq[i].to(DEVICE, dtype=torch.float64), min=0.0), float(q))
        a, sl = _normalize_rows(a)
        good = valid & torch.isfinite(sl) & (sl > 0)
        logscale = torch.where(good, logscale + torch.log(torch.clamp(sl, min=EPS)), torch.full_like(logscale, -torch.inf))
        valid = good
    Wout = torch.pow(torch.square(model.layers[-1].weight.to(torch.float64)), float(q))
    t = target_idx.to(DEVICE).long()
    selected = Wout[t, :]
    z = torch.sum(a * selected, dim=1)
    good = valid & torch.isfinite(z) & (z > 0)
    return torch.where(good, logscale + torch.log(torch.clamp(z, min=EPS)), torch.full_like(logscale, -torch.inf))


@torch.no_grad()
def mlp_factor_state(model: DeepMLP, source_sq: torch.Tensor, response_sq: List[torch.Tensor], target_idx: torch.Tensor) -> Dict[str, Any]:
    B = source_sq.shape[0]
    src = source_sq.to(DEVICE, dtype=torch.float64)
    resp = [r.to(DEVICE, dtype=torch.float64) for r in response_sq]
    logZ1 = mlp_log_partition_q(model, src, resp, target_idx, 1.0)
    logZ2 = mlp_log_partition_q(model, src, resp, target_idx, 2.0)
    valid = torch.isfinite(logZ1)
    alpha = []
    a, _ = _normalize_rows(src)
    alpha.append(a)
    for i, l in enumerate(model.layers[:-1]):
        a = a @ torch.square(l.weight.to(torch.float64)).T
        a = a * resp[i]
        a, _ = _normalize_rows(a)
        alpha.append(a)
    Wout2 = torch.square(model.layers[-1].weight.to(torch.float64))
    t = target_idx.to(DEVICE).long()
    selected = Wout2[t, :]
    beta: List[Optional[torch.Tensor]] = [None] * len(alpha)
    b = selected.clone()
    b, _ = _normalize_rows(b)
    beta[-1] = b
    for i in range(len(model.layers) - 2, -1, -1):
        target = beta[i + 1] * resp[i]
        b = target @ torch.square(model.layers[i].weight.to(torch.float64))
        b, _ = _normalize_rows(b)
        beta[i] = b
    node = []
    for i in range(len(alpha)):
        v = alpha[i] * beta[i]
        v, _ = _normalize_rows(v)
        node.append(v)
    outnodes = torch.zeros((B, model.layers[-1].out_features), dtype=torch.float64, device=DEVICE)
    outnodes[torch.arange(B, device=DEVICE), t] = 1.0
    node.append(outnodes)
    edge = []
    elog = torch.zeros(B, dtype=torch.float64, device=DEVICE)
    for i, l in enumerate(model.layers[:-1]):
        E = (beta[i + 1] * resp[i])[:, :, None] * torch.square(l.weight.to(torch.float64))[None, :, :] * alpha[i][:, None, :]
        den = E.sum((1, 2))
        E = torch.where((den > 0)[:, None, None], E / torch.clamp(den[:, None, None], min=EPS), torch.zeros_like(E))
        edge.append(E)
        logfac = (
            torch.log(torch.clamp(torch.square(l.weight.to(torch.float64)), min=PATH_LOG_FLOOR))[None, :, :]
            + torch.log(torch.clamp(resp[i], min=PATH_LOG_FLOOR))[:, :, None]
        )
        elog += torch.sum(torch.where(E > 0, E * logfac, torch.zeros_like(E)), dim=(1, 2))
    Eout = selected * alpha[-1]
    den = Eout.sum(1)
    Eout = torch.where((den > 0)[:, None], Eout / torch.clamp(den[:, None], min=EPS), torch.zeros_like(Eout))
    EF = torch.zeros((B, model.layers[-1].out_features, model.layers[-1].in_features), dtype=torch.float64, device=DEVICE)
    EF[torch.arange(B, device=DEVICE), t, :] = Eout
    edge.append(EF)
    logwout = torch.log(torch.clamp(Wout2[t, :], min=PATH_LOG_FLOOR))
    elog += torch.sum(torch.where(Eout > 0, Eout * logwout, torch.zeros_like(Eout)), dim=1)
    p0 = node[0]
    elogsrc = torch.sum(torch.where(p0 > 0, p0 * torch.log(torch.clamp(src, min=PATH_LOG_FLOOR)), torch.zeros_like(p0)), dim=1)
    elogq = elogsrc + elog
    H = logZ1 - elogq
    logN = math.log(model.layers[0].in_features) + sum(math.log(l.out_features) for l in model.layers[:-1])
    Sdef = logN - H
    eta1 = torch.exp(torch.clamp(H - logN, min=-745, max=0))
    eta2 = torch.exp(torch.clamp(2 * logZ1 - logZ2 - logN, min=-745, max=0))
    Sdef_norm = Sdef / max(logN, EPS)

    def arr(x):
        y = x.detach().cpu().numpy().astype(float)
        mask = valid.detach().cpu().numpy()
        y[~mask] = np.nan
        return y

    return {
        "valid": valid.detach().cpu().numpy(),
        "logZ1": logZ1.detach().cpu().numpy().astype(float),
        "S_path": arr(H),
        "entropy_deficit": arr(Sdef),
        "entropy_deficit_norm": arr(Sdef_norm),
        "eta1": arr(eta1),
        "eta2": arr(eta2),
        "node_flow": [v.detach().cpu().numpy() for v in node],
        "edge_flow": [e.detach().cpu().numpy() for e in edge],
    }


@torch.no_grad()
def cnn_response_sq(model: DeepCNN, X: torch.Tensor, X0: torch.Tensor) -> List[torch.Tensor]:
    _, z, _ = model.forward_with_states(X.to(DEVICE))
    _, z0, _ = model.forward_with_states(X0.to(DEVICE))
    return [torch.square(finite_secant_torch64(model.activation_name, a, b)) for a, b in zip(z, z0)]


@torch.no_grad()
def cnn_log_partition_q(model: DeepCNN, source_sq: torch.Tensor, response_sq: List[torch.Tensor], target_idx: torch.Tensor, q: float) -> torch.Tensor:
    src = torch.pow(torch.clamp(source_sq.to(DEVICE, dtype=torch.float64), min=0.0), float(q))
    a, s0 = _normalize_rows(src)
    valid = torch.isfinite(s0) & (s0 > 0)
    logscale = torch.where(valid, torch.log(torch.clamp(s0, min=EPS)), torch.full_like(s0, -torch.inf))
    for i, c in enumerate(model.convs):
        w = torch.pow(torch.square(c.weight.to(torch.float64)), float(q))
        a = F.conv2d(a, w, bias=None, stride=1, padding=0)
        if i < len(model.convs) - 1:
            a = a * torch.pow(torch.clamp(response_sq[i].to(DEVICE, dtype=torch.float64), min=0.0), float(q))
        a, sl = _normalize_rows(a)
        good = valid & torch.isfinite(sl) & (sl > 0)
        logscale = torch.where(good, logscale + torch.log(torch.clamp(sl, min=EPS)), torch.full_like(logscale, -torch.inf))
        valid = good
    t = target_idx.to(DEVICE).long()
    tmass = a[torch.arange(a.shape[0], device=DEVICE), t, :, :].sum(dim=(-1, -2))
    good = valid & torch.isfinite(tmass) & (tmass > 0)
    hout, wout = model.spatial_shapes()[-1]
    g2q = (1.0 / (hout * wout)) ** (2.0 * float(q))
    return torch.where(good, logscale + torch.log(torch.clamp(tmass, min=EPS)) + math.log(g2q), torch.full_like(logscale, -torch.inf))


@torch.no_grad()
def cnn_factor_state(model:DeepCNN,source_sq:torch.Tensor,response_sq:List[torch.Tensor],target_idx:torch.Tensor)->Dict[str,Any]:
    src=source_sq.to(DEVICE,dtype=torch.float64);resp=[r.to(DEVICE,dtype=torch.float64) for r in response_sq];B=src.shape[0];logZ1=cnn_log_partition_q(model,src,resp,target_idx,1.0);logZ2=cnn_log_partition_q(model,src,resp,target_idx,2.0);valid=torch.isfinite(logZ1)
    alpha=[];a,_=_normalize_rows(src);alpha.append(a)
    for i,c in enumerate(model.convs):
        a=F.conv2d(a,torch.square(c.weight.to(torch.float64)),bias=None,stride=1,padding=0)
        if i<len(model.convs)-1:a=a*resp[i]
        a,_=_normalize_rows(a);alpha.append(a)
    C,H,W=alpha[-1].shape[1:];t=target_idx.to(DEVICE).long();b=torch.zeros_like(alpha[-1]);b[torch.arange(B,device=DEVICE),t,:,:]=1.0;b,_=_normalize_rows(b);beta=[None]*len(alpha);beta[-1]=b
    for i in range(len(model.convs)-1,-1,-1):
        target=beta[i+1]
        if i<len(model.convs)-1:target=target*resp[i]
        b=F.conv_transpose2d(target,torch.square(model.convs[i].weight.to(torch.float64)),bias=None,stride=1,padding=0);b,_=_normalize_rows(b);beta[i]=b
    node=[]
    for i in range(len(alpha)):
        v=alpha[i]*beta[i];v,_=_normalize_rows(v);node.append(v)
    occurrence=[];parameter=[];layer_metrics=[];elog_edges=torch.zeros(B,dtype=torch.float64,device=DEVICE)
    for i,c in enumerate(model.convs):
        k=int(c.kernel_size[0]);patches=F.unfold(alpha[i],kernel_size=k,stride=1,padding=0);Hout,Wout=alpha[i+1].shape[-2:];L=Hout*Wout;target=beta[i+1]
        if i<len(model.convs)-1:target=target*resp[i]
        target=target.reshape(B,c.out_channels,L);w2=torch.square(c.weight.to(torch.float64)).reshape(c.out_channels,c.in_channels*k*k);E=target[:,:,None,:]*w2[None,:,:,None]*patches[:,None,:,:];den=E.sum((1,2,3));E=torch.where((den>0)[:,None,None,None],E/torch.clamp(den[:,None,None,None],min=EPS),torch.zeros_like(E));E=E.reshape(B,c.out_channels,c.in_channels,k,k,Hout,Wout);P=E.sum((-1,-2));occurrence.append(E);parameter.append(P)
        logfactor=torch.log(torch.clamp(torch.square(c.weight.to(torch.float64)),min=PATH_LOG_FLOOR))[None,:,:,:,:,None,None]
        if i<len(model.convs)-1:logfactor=logfactor+torch.log(torch.clamp(resp[i],min=PATH_LOG_FLOOR))[:,:,None,None,None,:,:]
        elog_edges+=torch.sum(torch.where(E>0,E*logfactor,torch.zeros_like(E)),dim=(1,2,3,4,5,6))
        flat=E.reshape(B,-1);pflat=P.reshape(B,-1);Hocc=-torch.sum(torch.where(flat>0,flat*torch.log(torch.clamp(flat,min=PATH_LOG_FLOOR)),torch.zeros_like(flat)),1);Hpar=-torch.sum(torch.where(pflat>0,pflat*torch.log(torch.clamp(pflat,min=PATH_LOG_FLOOR)),torch.zeros_like(pflat)),1);Hreuse=Hocc-Hpar;ntheta=P[0].numel();m=Hout*Wout;Socc=math.log(E[0].numel())-Hocc;Spar=math.log(ntheta)-Hpar;Sreuse=math.log(m)-Hreuse
        layer_metrics.append({"S_occ":Socc,"S_param":Spar,"S_reuse":Sreuse,"eta1_occ":torch.exp(-Socc),"eta1_param":torch.exp(-Spar),"eta1_reuse":torch.exp(-Sreuse),"reuse_multiplicity":float(m)})
    p0=node[0];elogsrc=torch.sum(torch.where(p0>0,p0*torch.log(torch.clamp(src,min=PATH_LOG_FLOOR)),torch.zeros_like(p0)),dim=(1,2,3));hout,wout=model.spatial_shapes()[-1];elogq=elogsrc+elog_edges+math.log((1.0/(hout*wout))**2);Hpath=logZ1-elogq;logN=math.log(float(hout*wout))+sum(math.log(float(c.in_channels*c.kernel_size[0]*c.kernel_size[1])) for c in model.convs);Sdef=logN-Hpath;eta1=torch.exp(torch.clamp(Hpath-logN,min=-745,max=0));eta2=torch.exp(torch.clamp(2*logZ1-logZ2-logN,min=-745,max=0));Sdef_norm=Sdef/max(logN,EPS)
    def arr(x):
        y=x.detach().cpu().numpy().astype(float);mask=valid.detach().cpu().numpy();y[~mask]=np.nan;return y
    return {"valid":valid.detach().cpu().numpy(),"logZ1":logZ1.detach().cpu().numpy().astype(float),"S_path":arr(Hpath),"entropy_deficit":arr(Sdef),"entropy_deficit_norm":arr(Sdef_norm),"eta1":arr(eta1),"eta2":arr(eta2),"node_flow":[v.detach().cpu().numpy() for v in node],"occurrence_flow":[e.detach().cpu().numpy() for e in occurrence],"parameter_flow":[p.detach().cpu().numpy() for p in parameter],"layer_metrics":[{k:(arr(v) if torch.is_tensor(v) else np.full(B,float(v))) for k,v in d.items()} for d in layer_metrics]}


# ==================================================================================================
# 6. TEMPORARY EXECUTION GATES â€” NEW OPERATOR, PARAMETERS REMAIN FIXED
# ==================================================================================================

def ones_mlp_gates(model: DeepMLP) -> List[np.ndarray]:
    return [np.ones(tuple(l.weight.shape), dtype=np.float64) for l in model.layers]


def ones_cnn_occurrence_gates(model: DeepCNN) -> List[np.ndarray]:
    shapes = model.spatial_shapes()
    out = []
    for i, c in enumerate(model.convs):
        k = int(c.kernel_size[0])
        Hout, Wout = shapes[i + 1]
        out.append(np.ones((c.out_channels, c.in_channels, k, k, Hout, Wout), dtype=np.float64))
    return out


def masks_to_gates(masks: List[np.ndarray], alpha: float) -> List[np.ndarray]:
    return [np.where(np.asarray(m, bool), 1.0 - float(alpha), 1.0).astype(np.float64) for m in masks]


@torch.no_grad()
def mlp_forward_with_gates(model: DeepMLP, X: torch.Tensor, gates: List[np.ndarray]):
    h = X.to(DEVICE).reshape(X.shape[0], -1)
    zs = []
    hs = [h]
    for i, l in enumerate(model.layers):
        g = torch.as_tensor(gates[i], dtype=l.weight.dtype, device=DEVICE)
        z = F.linear(h, l.weight * g, bias=None)
        if i < len(model.layers) - 1:
            zs.append(z)
            h = model.act(z)
            hs.append(h)
        else:
            out = z
            hs.append(out)
    return out, zs, hs


@torch.no_grad()
def cnn_occurrence_linear(h: torch.Tensor, conv: nn.Conv2d, gate: np.ndarray) -> torch.Tensor:
    """Exact occurrence-resolved convolution used only for the intervention.

    gate shape: [Cout,Cin,k,k,Hout,Wout]. The stored Conv2d.weight tensor is never modified.
    """
    k = int(conv.kernel_size[0])
    patches = F.unfold(h, kernel_size=k, stride=1, padding=0)  # [B,P,L]
    B, P, L = patches.shape
    Hout = h.shape[-2] - k + 1
    Wout = h.shape[-1] - k + 1
    w = conv.weight.reshape(conv.out_channels, P)
    g = torch.as_tensor(gate, dtype=h.dtype, device=h.device).reshape(conv.out_channels, P, L)
    coeff = w[:, :, None] * g
    zflat = torch.einsum("bpl,opl->bol", patches, coeff)
    return zflat.reshape(B, conv.out_channels, Hout, Wout)


@torch.no_grad()
def cnn_forward_with_gates(model: DeepCNN, X: torch.Tensor, gates: List[np.ndarray]):
    h = X.to(DEVICE)
    zs = []
    hs = [h]
    for i, c in enumerate(model.convs):
        z = cnn_occurrence_linear(h, c, gates[i])
        if i < len(model.convs) - 1:
            zs.append(z)
            h = model.act(z)
            hs.append(h)
        else:
            h = z
            hs.append(h)
    return h.mean(dim=(-1, -2)), zs, hs


@torch.no_grad()
def gated_response_sq(model: nn.Module, X: torch.Tensor, X0: torch.Tensor, gates: List[np.ndarray]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    if isinstance(model, DeepMLP):
        out, z, hs = mlp_forward_with_gates(model, X, gates)
        out0, z0, hs0 = mlp_forward_with_gates(model, X0, gates)
    else:
        out, z, hs = cnn_forward_with_gates(model, X, gates)
        out0, z0, hs0 = cnn_forward_with_gates(model, X0, gates)
    resp = [torch.square(finite_secant_torch64(model.activation_name, a, b)) for a, b in zip(z, z0)]
    return resp, out, out0, hs, hs0

# ==================================================================================================
# 7. GATED MLP DYNAMIC PROGRAM
# ==================================================================================================

@torch.no_grad()
def mlp_log_partition_q_gated(model: DeepMLP, source_sq: torch.Tensor, response_sq: List[torch.Tensor], target_idx: torch.Tensor, q: float, gates: List[np.ndarray]) -> torch.Tensor:
    src = torch.pow(torch.clamp(source_sq.to(DEVICE, dtype=torch.float64), min=0.0), float(q))
    a, s0 = _normalize_rows(src)
    valid = torch.isfinite(s0) & (s0 > 0)
    logscale = torch.where(valid, torch.log(torch.clamp(s0, min=EPS)), torch.full_like(s0, -torch.inf))
    for i, l in enumerate(model.layers[:-1]):
        g2 = torch.square(torch.as_tensor(gates[i], dtype=torch.float64, device=DEVICE))
        local = torch.square(l.weight.to(torch.float64)) * g2
        wq = torch.pow(torch.clamp(local, min=0.0), float(q))
        a = a @ wq.T
        a = a * torch.pow(torch.clamp(response_sq[i].to(DEVICE, dtype=torch.float64), min=0.0), float(q))
        a, sl = _normalize_rows(a)
        good = valid & torch.isfinite(sl) & (sl > 0)
        logscale = torch.where(good, logscale + torch.log(torch.clamp(sl, min=EPS)), torch.full_like(logscale, -torch.inf))
        valid = good
    gout2 = torch.square(torch.as_tensor(gates[-1], dtype=torch.float64, device=DEVICE))
    local_out = torch.square(model.layers[-1].weight.to(torch.float64)) * gout2
    Wout = torch.pow(torch.clamp(local_out, min=0.0), float(q))
    t = target_idx.to(DEVICE).long()
    selected = Wout[t, :]
    z = torch.sum(a * selected, dim=1)
    good = valid & torch.isfinite(z) & (z > 0)
    return torch.where(good, logscale + torch.log(torch.clamp(z, min=EPS)), torch.full_like(logscale, -torch.inf))


@torch.no_grad()
def mlp_factor_state_gated(model: DeepMLP, source_sq: torch.Tensor, response_sq: List[torch.Tensor], target_idx: torch.Tensor, gates: List[np.ndarray]) -> Dict[str, Any]:
    B = source_sq.shape[0]
    src = source_sq.to(DEVICE, dtype=torch.float64)
    resp = [r.to(DEVICE, dtype=torch.float64) for r in response_sq]
    logZ1 = mlp_log_partition_q_gated(model, src, resp, target_idx, 1.0, gates)
    logZ2 = mlp_log_partition_q_gated(model, src, resp, target_idx, 2.0, gates)
    valid = torch.isfinite(logZ1)

    edge_local = []
    for i, l in enumerate(model.layers):
        g2 = torch.square(torch.as_tensor(gates[i], dtype=torch.float64, device=DEVICE))
        edge_local.append(torch.square(l.weight.to(torch.float64)) * g2)

    alpha = []
    a, _ = _normalize_rows(src)
    alpha.append(a)
    for i, _ in enumerate(model.layers[:-1]):
        a = a @ edge_local[i].T
        a = a * resp[i]
        a, _ = _normalize_rows(a)
        alpha.append(a)

    t = target_idx.to(DEVICE).long()
    selected = edge_local[-1][t, :]
    beta: List[Optional[torch.Tensor]] = [None] * len(alpha)
    b = selected.clone()
    b, _ = _normalize_rows(b)
    beta[-1] = b
    for i in range(len(model.layers) - 2, -1, -1):
        target = beta[i + 1] * resp[i]
        b = target @ edge_local[i]
        b, _ = _normalize_rows(b)
        beta[i] = b

    node = []
    for i in range(len(alpha)):
        v = alpha[i] * beta[i]
        v, _ = _normalize_rows(v)
        node.append(v)
    outnodes = torch.zeros((B, model.layers[-1].out_features), dtype=torch.float64, device=DEVICE)
    outnodes[torch.arange(B, device=DEVICE), t] = 1.0
    node.append(outnodes)

    edge = []
    elog = torch.zeros(B, dtype=torch.float64, device=DEVICE)
    for i, _ in enumerate(model.layers[:-1]):
        E = (beta[i + 1] * resp[i])[:, :, None] * edge_local[i][None, :, :] * alpha[i][:, None, :]
        den = E.sum((1, 2))
        E = torch.where((den > 0)[:, None, None], E / torch.clamp(den[:, None, None], min=EPS), torch.zeros_like(E))
        edge.append(E)
        logfac = torch.log(torch.clamp(edge_local[i], min=PATH_LOG_FLOOR))[None, :, :] + torch.log(torch.clamp(resp[i], min=PATH_LOG_FLOOR))[:, :, None]
        elog += torch.sum(torch.where(E > 0, E * logfac, torch.zeros_like(E)), dim=(1, 2))

    Eout = selected * alpha[-1]
    den = Eout.sum(1)
    Eout = torch.where((den > 0)[:, None], Eout / torch.clamp(den[:, None], min=EPS), torch.zeros_like(Eout))
    EF = torch.zeros((B, model.layers[-1].out_features, model.layers[-1].in_features), dtype=torch.float64, device=DEVICE)
    EF[torch.arange(B, device=DEVICE), t, :] = Eout
    edge.append(EF)
    logwout = torch.log(torch.clamp(selected, min=PATH_LOG_FLOOR))
    elog += torch.sum(torch.where(Eout > 0, Eout * logwout, torch.zeros_like(Eout)), dim=1)

    p0 = node[0]
    elogsrc = torch.sum(torch.where(p0 > 0, p0 * torch.log(torch.clamp(src, min=PATH_LOG_FLOOR)), torch.zeros_like(p0)), dim=1)
    H = logZ1 - (elogsrc + elog)
    logN = math.log(model.layers[0].in_features) + sum(math.log(l.out_features) for l in model.layers[:-1])
    Sdef = logN - H
    eta1 = torch.exp(torch.clamp(H - logN, min=-745, max=0))
    eta2 = torch.exp(torch.clamp(2 * logZ1 - logZ2 - logN, min=-745, max=0))
    Sdef_norm = Sdef / max(logN, EPS)

    def arr(x):
        y = x.detach().cpu().numpy().astype(float)
        mask = valid.detach().cpu().numpy()
        y[~mask] = np.nan
        return y

    return {
        "valid": valid.detach().cpu().numpy(),
        "logZ1": logZ1.detach().cpu().numpy().astype(float),
        "S_path": arr(H),
        "entropy_deficit": arr(Sdef),
        "entropy_deficit_norm": arr(Sdef_norm),
        "eta1": arr(eta1),
        "eta2": arr(eta2),
        "node_flow": [v.detach().cpu().numpy() for v in node],
        "edge_flow": [e.detach().cpu().numpy() for e in edge],
    }

# ==================================================================================================
# 8. GATED CNN OCCURRENCE-RESOLVED DYNAMIC PROGRAM
# ==================================================================================================

@torch.no_grad()
def _cnn_occurrence_coeff(conv: nn.Conv2d, gate: np.ndarray, q: float, dtype=torch.float64) -> torch.Tensor:
    k = int(conv.kernel_size[0])
    w2q = torch.pow(torch.clamp(torch.square(conv.weight.to(dtype)), min=0.0), float(q))
    g = torch.as_tensor(gate, dtype=dtype, device=DEVICE)
    g2q = torch.pow(torch.clamp(torch.square(g), min=0.0), float(q))
    return w2q[:, :, :, :, None, None] * g2q


@torch.no_grad()
def _cnn_occurrence_forward_mass(a: torch.Tensor, coeff: torch.Tensor, k: int) -> torch.Tensor:
    """Apply occurrence-resolved nonnegative transition coefficients.

    a     : [B,Cin,Hin,Win]
    coeff : [Cout,Cin,k,k,Hout,Wout]
    """
    patches = F.unfold(a, kernel_size=k, stride=1, padding=0)  # [B,P,L]
    B, P, L = patches.shape
    O = coeff.shape[0]
    cflat = coeff.reshape(O, P, L)
    out = torch.einsum("bpl,opl->bol", patches, cflat)
    Hout, Wout = coeff.shape[-2:]
    return out.reshape(B, O, Hout, Wout)


@torch.no_grad()
def _cnn_occurrence_backward_mass(target: torch.Tensor, coeff: torch.Tensor, input_hw: Tuple[int, int], k: int) -> torch.Tensor:
    B, O, Hout, Wout = target.shape
    L = Hout * Wout
    P = coeff.shape[1] * k * k
    tflat = target.reshape(B, O, L)
    cflat = coeff.reshape(O, P, L)
    patch_mass = torch.einsum("bol,opl->bpl", tflat, cflat)
    return F.fold(patch_mass, output_size=input_hw, kernel_size=k, stride=1, padding=0)


@torch.no_grad()
def cnn_log_partition_q_gated(model: DeepCNN, source_sq: torch.Tensor, response_sq: List[torch.Tensor], target_idx: torch.Tensor, q: float, gates: List[np.ndarray]) -> torch.Tensor:
    src = torch.pow(torch.clamp(source_sq.to(DEVICE, dtype=torch.float64), min=0.0), float(q))
    a, s0 = _normalize_rows(src)
    valid = torch.isfinite(s0) & (s0 > 0)
    logscale = torch.where(valid, torch.log(torch.clamp(s0, min=EPS)), torch.full_like(s0, -torch.inf))
    for i, c in enumerate(model.convs):
        k = int(c.kernel_size[0])
        coeff = _cnn_occurrence_coeff(c, gates[i], q, dtype=torch.float64)
        a = _cnn_occurrence_forward_mass(a, coeff, k)
        if i < len(model.convs) - 1:
            a = a * torch.pow(torch.clamp(response_sq[i].to(DEVICE, dtype=torch.float64), min=0.0), float(q))
        a, sl = _normalize_rows(a)
        good = valid & torch.isfinite(sl) & (sl > 0)
        logscale = torch.where(good, logscale + torch.log(torch.clamp(sl, min=EPS)), torch.full_like(logscale, -torch.inf))
        valid = good
    t = target_idx.to(DEVICE).long()
    tmass = a[torch.arange(a.shape[0], device=DEVICE), t, :, :].sum(dim=(-1, -2))
    good = valid & torch.isfinite(tmass) & (tmass > 0)
    hout, wout = model.spatial_shapes()[-1]
    gap_q = (1.0 / (hout * wout)) ** (2.0 * float(q))
    return torch.where(good, logscale + torch.log(torch.clamp(tmass, min=EPS)) + math.log(gap_q), torch.full_like(logscale, -torch.inf))


@torch.no_grad()
def cnn_factor_state_gated(model: DeepCNN, source_sq: torch.Tensor, response_sq: List[torch.Tensor], target_idx: torch.Tensor, gates: List[np.ndarray]) -> Dict[str, Any]:
    src = source_sq.to(DEVICE, dtype=torch.float64)
    resp = [r.to(DEVICE, dtype=torch.float64) for r in response_sq]
    B = src.shape[0]
    logZ1 = cnn_log_partition_q_gated(model, src, resp, target_idx, 1.0, gates)
    logZ2 = cnn_log_partition_q_gated(model, src, resp, target_idx, 2.0, gates)
    valid = torch.isfinite(logZ1)

    coeffs = []
    for i, c in enumerate(model.convs):
        coeffs.append(_cnn_occurrence_coeff(c, gates[i], 1.0, dtype=torch.float64))

    alpha = []
    a, _ = _normalize_rows(src)
    alpha.append(a)
    for i, c in enumerate(model.convs):
        k = int(c.kernel_size[0])
        a = _cnn_occurrence_forward_mass(a, coeffs[i], k)
        if i < len(model.convs) - 1:
            a = a * resp[i]
        a, _ = _normalize_rows(a)
        alpha.append(a)

    t = target_idx.to(DEVICE).long()
    b = torch.zeros_like(alpha[-1])
    b[torch.arange(B, device=DEVICE), t, :, :] = 1.0
    b, _ = _normalize_rows(b)
    beta: List[Optional[torch.Tensor]] = [None] * len(alpha)
    beta[-1] = b
    for i in range(len(model.convs) - 1, -1, -1):
        target = beta[i + 1]
        if i < len(model.convs) - 1:
            target = target * resp[i]
        k = int(model.convs[i].kernel_size[0])
        input_hw = tuple(alpha[i].shape[-2:])
        b = _cnn_occurrence_backward_mass(target, coeffs[i], input_hw, k)
        b, _ = _normalize_rows(b)
        beta[i] = b

    node = []
    for i in range(len(alpha)):
        v = alpha[i] * beta[i]
        v, _ = _normalize_rows(v)
        node.append(v)

    occurrence = []
    parameter = []
    elog_edges = torch.zeros(B, dtype=torch.float64, device=DEVICE)
    for i, c in enumerate(model.convs):
        k = int(c.kernel_size[0])
        patches = F.unfold(alpha[i], kernel_size=k, stride=1, padding=0)
        Hout, Wout = alpha[i + 1].shape[-2:]
        L = Hout * Wout
        target = beta[i + 1]
        if i < len(model.convs) - 1:
            target = target * resp[i]
        target_flat = target.reshape(B, c.out_channels, L)
        coeff_flat = coeffs[i].reshape(c.out_channels, c.in_channels * k * k, L)
        E = target_flat[:, :, None, :] * coeff_flat[None, :, :, :] * patches[:, None, :, :]
        den = E.sum((1, 2, 3))
        E = torch.where((den > 0)[:, None, None, None], E / torch.clamp(den[:, None, None, None], min=EPS), torch.zeros_like(E))
        E = E.reshape(B, c.out_channels, c.in_channels, k, k, Hout, Wout)
        P = E.sum((-1, -2))
        occurrence.append(E)
        parameter.append(P)

        logfactor = torch.log(torch.clamp(coeffs[i], min=PATH_LOG_FLOOR))[None, :, :, :, :, :, :]
        if i < len(model.convs) - 1:
            logfactor = logfactor + torch.log(torch.clamp(resp[i], min=PATH_LOG_FLOOR))[:, :, None, None, None, :, :]
        elog_edges += torch.sum(torch.where(E > 0, E * logfactor, torch.zeros_like(E)), dim=(1, 2, 3, 4, 5, 6))

    p0 = node[0]
    elogsrc = torch.sum(
        torch.where(p0 > 0, p0 * torch.log(torch.clamp(src, min=PATH_LOG_FLOOR)), torch.zeros_like(p0)),
        dim=(1, 2, 3),
    )
    hout, wout = model.spatial_shapes()[-1]
    Hpath = logZ1 - (elogsrc + elog_edges + math.log((1.0 / (hout * wout)) ** 2))
    logN = math.log(float(hout * wout)) + sum(math.log(float(c.in_channels * c.kernel_size[0] * c.kernel_size[1])) for c in model.convs)
    Sdef = logN - Hpath
    eta1 = torch.exp(torch.clamp(Hpath - logN, min=-745, max=0))
    eta2 = torch.exp(torch.clamp(2 * logZ1 - logZ2 - logN, min=-745, max=0))
    Sdef_norm = Sdef / max(logN, EPS)

    def arr(x):
        y = x.detach().cpu().numpy().astype(float)
        mask = valid.detach().cpu().numpy()
        y[~mask] = np.nan
        return y

    return {
        "valid": valid.detach().cpu().numpy(),
        "logZ1": logZ1.detach().cpu().numpy().astype(float),
        "S_path": arr(Hpath),
        "entropy_deficit": arr(Sdef),
        "entropy_deficit_norm": arr(Sdef_norm),
        "eta1": arr(eta1),
        "eta2": arr(eta2),
        "node_flow": [v.detach().cpu().numpy() for v in node],
        "occurrence_flow": [e.detach().cpu().numpy() for e in occurrence],
        "parameter_flow": [p.detach().cpu().numpy() for p in parameter],
    }

# ==================================================================================================
# 9. EXACT PATH-MEASURE HELLINGER DISTANCE BY CROSS-PARTITION DYNAMIC PROGRAM
# ==================================================================================================

@torch.no_grad()
def mlp_cross_log_partition(model: DeepMLP, source_sq_a: torch.Tensor, source_sq_b: torch.Tensor,
                            response_sq_a: List[torch.Tensor], response_sq_b: List[torch.Tensor],
                            target_idx: torch.Tensor, gates_b: List[np.ndarray]) -> torch.Tensor:
    src = torch.sqrt(torch.clamp(source_sq_a.to(DEVICE, dtype=torch.float64), min=0.0) *
                     torch.clamp(source_sq_b.to(DEVICE, dtype=torch.float64), min=0.0))
    a, s0 = _normalize_rows(src)
    valid = torch.isfinite(s0) & (s0 > 0)
    logscale = torch.where(valid, torch.log(torch.clamp(s0, min=EPS)), torch.full_like(s0, -torch.inf))
    for i, l in enumerate(model.layers[:-1]):
        g = torch.as_tensor(gates_b[i], dtype=torch.float64, device=DEVICE)
        edge_cross = torch.square(l.weight.to(torch.float64)) * torch.clamp(g, min=0.0)
        a = a @ edge_cross.T
        r_cross = torch.sqrt(torch.clamp(response_sq_a[i].to(DEVICE, dtype=torch.float64), min=0.0) *
                             torch.clamp(response_sq_b[i].to(DEVICE, dtype=torch.float64), min=0.0))
        a = a * r_cross
        a, sl = _normalize_rows(a)
        good = valid & torch.isfinite(sl) & (sl > 0)
        logscale = torch.where(good, logscale + torch.log(torch.clamp(sl, min=EPS)), torch.full_like(logscale, -torch.inf))
        valid = good
    gout = torch.as_tensor(gates_b[-1], dtype=torch.float64, device=DEVICE)
    edge_cross_out = torch.square(model.layers[-1].weight.to(torch.float64)) * torch.clamp(gout, min=0.0)
    t = target_idx.to(DEVICE).long()
    selected = edge_cross_out[t, :]
    z = torch.sum(a * selected, dim=1)
    good = valid & torch.isfinite(z) & (z > 0)
    return torch.where(good, logscale + torch.log(torch.clamp(z, min=EPS)), torch.full_like(logscale, -torch.inf))


@torch.no_grad()
def cnn_cross_log_partition(model: DeepCNN, source_sq_a: torch.Tensor, source_sq_b: torch.Tensor,
                            response_sq_a: List[torch.Tensor], response_sq_b: List[torch.Tensor],
                            target_idx: torch.Tensor, gates_b: List[np.ndarray]) -> torch.Tensor:
    src = torch.sqrt(torch.clamp(source_sq_a.to(DEVICE, dtype=torch.float64), min=0.0) *
                     torch.clamp(source_sq_b.to(DEVICE, dtype=torch.float64), min=0.0))
    a, s0 = _normalize_rows(src)
    valid = torch.isfinite(s0) & (s0 > 0)
    logscale = torch.where(valid, torch.log(torch.clamp(s0, min=EPS)), torch.full_like(s0, -torch.inf))
    for i, c in enumerate(model.convs):
        k = int(c.kernel_size[0])
        g = torch.as_tensor(gates_b[i], dtype=torch.float64, device=DEVICE)
        edge_cross = torch.square(c.weight.to(torch.float64))[:, :, :, :, None, None] * torch.clamp(g, min=0.0)
        a = _cnn_occurrence_forward_mass(a, edge_cross, k)
        if i < len(model.convs) - 1:
            r_cross = torch.sqrt(torch.clamp(response_sq_a[i].to(DEVICE, dtype=torch.float64), min=0.0) *
                                 torch.clamp(response_sq_b[i].to(DEVICE, dtype=torch.float64), min=0.0))
            a = a * r_cross
        a, sl = _normalize_rows(a)
        good = valid & torch.isfinite(sl) & (sl > 0)
        logscale = torch.where(good, logscale + torch.log(torch.clamp(sl, min=EPS)), torch.full_like(logscale, -torch.inf))
        valid = good
    t = target_idx.to(DEVICE).long()
    tmass = a[torch.arange(a.shape[0], device=DEVICE), t, :, :].sum(dim=(-1, -2))
    good = valid & torch.isfinite(tmass) & (tmass > 0)
    hout, wout = model.spatial_shapes()[-1]
    gap_cross = (1.0 / (hout * wout)) ** 2
    return torch.where(good, logscale + torch.log(torch.clamp(tmass, min=EPS)) + math.log(gap_cross), torch.full_like(logscale, -torch.inf))


def hellinger_from_logs(logZ_base: float, logZ_int: float, logZ_cross: float) -> float:
    if not (np.isfinite(logZ_base) and np.isfinite(logZ_int) and np.isfinite(logZ_cross)):
        return np.nan
    log_bc = float(logZ_cross - 0.5 * (logZ_base + logZ_int))
    bc = float(np.exp(np.clip(log_bc, -745, 0.0)))
    bc = min(1.0, max(0.0, bc))
    return float(math.sqrt(max(0.0, 1.0 - bc)))

# ==================================================================================================
# 10. TARGET CONSTRUCTION
# ==================================================================================================

def flow_list_from_state(state: Dict[str, Any], family: str) -> List[np.ndarray]:
    if family == "mlp":
        return [np.asarray(x[0], float) for x in state["edge_flow"]]
    return [np.asarray(x[0], float) for x in state["occurrence_flow"]]


def select_realized_masks(flow_list: List[np.ndarray], fraction: float) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Select high-flow targets within each layer while preserving a non-target realized-support pool.

    The matched control is intentionally drawn from *positive realized support* outside the high-flow
    target, not from inactive / zero-flow occurrences. Therefore each layer must retain at least as many
    positive-support non-target occurrences as realized targets. Layers with fewer than two positive-flow
    occurrences are not intervened upon in the primary matched design. This avoids a trivial realized-vs-
    matched comparison in which the control is composed of computationally inactive occurrences.
    """
    masks = []
    audit = []
    for li, f in enumerate(flow_list):
        f = np.asarray(f, float)
        flat = f.ravel()
        positive = np.flatnonzero(np.isfinite(flat) & (flat > 0))
        mask = np.zeros(flat.shape, dtype=bool)
        raw_k = int(math.ceil(float(fraction) * len(positive))) if len(positive) else 0
        if len(positive) >= 2:
            # Reserve an equally large positive-support pool for the matched control.
            k = min(max(1, raw_k), len(positive) // 2)
            order = positive[np.argsort(-flat[positive], kind="mergesort")[:k]]
            mask[order] = True
        else:
            k = 0
        mass = float(np.sum(flat[mask])) if k else 0.0
        audit.append({
            "layer": li,
            "n_total": int(flat.size),
            "n_positive_flow": int(len(positive)),
            "n_targets": int(k),
            "raw_fraction_target_count": int(raw_k),
            "target_count_capped_for_matched_support": bool(k != raw_k),
            "positive_support_fraction": float(len(positive) / max(flat.size, 1)),
            "selected_flow_mass": mass,
        })
        masks.append(mask.reshape(f.shape))
    return masks, audit


def positive_support_masks(flow_list: List[np.ndarray]) -> List[np.ndarray]:
    return [np.isfinite(np.asarray(f, float)) & (np.asarray(f, float) > 0) for f in flow_list]

def select_top_k_masks(flow_list: List[np.ndarray], counts: List[int]) -> List[np.ndarray]:
    masks = []
    for li, f in enumerate(flow_list):
        flat = np.asarray(f, float).ravel()
        k = int(counts[li])
        mask = np.zeros(flat.shape, dtype=bool)
        eligible = np.flatnonzero(np.isfinite(flat) & (flat > 0))
        if k > 0:
            if len(eligible) < k:
                # Structural flow should normally have ample support. We fail visibly rather than silently changing budget.
                raise RuntimeError(f"Layer {li}: structural positive support {len(eligible)} < required target count {k}.")
            ids = eligible[np.argsort(-flat[eligible], kind="mergesort")[:k]]
            mask[ids] = True
        masks.append(mask.reshape(np.asarray(f).shape))
    return masks


def target_counts(masks: List[np.ndarray]) -> List[int]:
    return [int(np.sum(np.asarray(m, bool))) for m in masks]


def jaccard_masks(a: List[np.ndarray], b: List[np.ndarray]) -> Tuple[float, List[float]]:
    nums, dens, per = 0, 0, []
    for x, y in zip(a, b):
        x = np.asarray(x, bool)
        y = np.asarray(y, bool)
        inter = int(np.sum(x & y))
        union = int(np.sum(x | y))
        per.append(float(inter / union) if union else np.nan)
        nums += inter
        dens += union
    return (float(nums / dens) if dens else np.nan), per


def ordinary_activation_scores_mlp(model: DeepMLP, hs: List[torch.Tensor], hs0: List[torch.Tensor]) -> List[np.ndarray]:
    """Ordinary source-activation magnitude for every MLP execution edge.

    Matching on source activation, rather than on realized flow or weight magnitude, controls the simple
    objection that realization-guided edges matter only because they originate from strongly active units.
    Structural-flow targeting separately controls trained parameter/disposition prominence.
    """
    out = []
    for li, l in enumerate(model.layers):
        hx = hs[li][0].detach().to(torch.float64)
        h0 = hs0[li][0].detach().to(torch.float64)
        score_src = torch.sqrt(hx * hx + h0 * h0 + (hx - h0) * (hx - h0))
        score = score_src[None, :].expand(l.out_features, l.in_features)
        out.append(score.cpu().numpy().astype(float))
    return out


def ordinary_activation_scores_cnn(model: DeepCNN, hs: List[torch.Tensor], hs0: List[torch.Tensor]) -> List[np.ndarray]:
    """Ordinary source-activation magnitude for every spatial convolutional edge occurrence."""
    out = []
    for li, c in enumerate(model.convs):
        k = int(c.kernel_size[0])
        hx = hs[li].detach().to(torch.float64)
        h0 = hs0[li].detach().to(torch.float64)
        sx = F.unfold(hx, kernel_size=k, stride=1, padding=0)[0]   # [P,L]
        s0 = F.unfold(h0, kernel_size=k, stride=1, padding=0)[0]
        src_score = torch.sqrt(sx * sx + s0 * s0 + (sx - s0) * (sx - s0))
        score = src_score[None, :, :].expand(c.out_channels, -1, -1)
        Hout = hx.shape[-2] - k + 1
        Wout = hx.shape[-1] - k + 1
        score = score.reshape(c.out_channels, c.in_channels, k, k, Hout, Wout)
        out.append(score.cpu().numpy().astype(float))
    return out


def select_matched_masks(real_masks: List[np.ndarray], activation_scores: List[np.ndarray],
                         realized_support_masks: List[np.ndarray], seed: int) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Layer-wise matched control from the non-target part of the realized support.

    Scientific role: compare the high-flow realized target against execution occurrences that are
    (i) in the same layer, (ii) part of the same condition's positive realized support, and
    (iii) approximately matched on ordinary source-activation magnitude, but are NOT selected by
    high realized-flow rank. This prevents a trivial comparison against inactive zero-flow carriers
    and automatically preserves output-conditioned topological role at the final layer/channel.

    Realized flow is used only to define the eligible positive-support set and to exclude the high-flow
    target. It is never used to rank matched candidates. Candidate ranking/matching uses only ordinary
    source-activation magnitude.
    """
    masks: List[np.ndarray] = []
    audit: List[Dict[str, Any]] = []
    for li, (rm, sc, support) in enumerate(zip(real_masks, activation_scores, realized_support_masks)):
        rmf = np.asarray(rm, bool).ravel()
        sf = np.asarray(sc, float).ravel()
        sup = np.asarray(support, bool).ravel()
        if not (len(rmf) == len(sf) == len(sup)):
            raise RuntimeError(f"Layer {li}: matched-control arrays have inconsistent sizes.")
        target_ids = np.flatnonzero(rmf)
        eligible_ids = np.flatnonzero(sup & (~rmf) & np.isfinite(sf))
        if len(target_ids) != int(np.sum(rmf)):
            raise RuntimeError(f"Layer {li}: realized target indexing inconsistency.")
        if len(target_ids) and len(eligible_ids) < len(target_ids):
            raise RuntimeError(
                f"Layer {li}: positive realized-support matched pool {len(eligible_ids)} "
                f"< realized target count {len(target_ids)}."
            )

        mask = np.zeros(rmf.shape, dtype=bool)
        if len(target_ids) == 0:
            masks.append(mask.reshape(np.asarray(rm).shape))
            audit.append({
                "layer": li, "n_targets": 0, "n_matched_pool": int(len(eligible_ids)),
                "matched_activation_ks": np.nan,
                "realized_activation_median": np.nan,
                "matched_activation_median": np.nan,
                "matched_to_realized_activation_median_ratio": np.nan,
                "borrowed_fraction": 0.0,
            })
            continue

        if not np.all(np.isfinite(sf[target_ids])):
            raise RuntimeError(f"Layer {li}: realized target contains non-finite ordinary source-activation score.")

        # Build rank bins ONLY on the positive realized support, so target and control share realized support.
        support_ids = np.flatnonzero(sup & np.isfinite(sf))
        sorted_ids = support_ids[np.argsort(sf[support_ids], kind="mergesort")]
        rank_bin = np.full(len(sf), -1, dtype=np.int32)
        bins_sorted = np.minimum(
            MATCH_BINS - 1,
            np.floor(np.arange(len(sorted_ids), dtype=float) * MATCH_BINS / max(len(sorted_ids), 1)).astype(np.int32),
        )
        rank_bin[sorted_ids] = bins_sorted

        rng = np.random.default_rng(int(seed) + 104729 * li)
        selected: List[int] = []
        used = np.zeros(len(sf), dtype=bool)
        borrowed = 0
        target_bin_counts = np.bincount(rank_bin[target_ids], minlength=MATCH_BINS)

        for b in range(MATCH_BINS):
            need = int(target_bin_counts[b])
            if need <= 0:
                continue
            cand = np.flatnonzero((rank_bin == b) & sup & (~rmf) & (~used) & np.isfinite(sf))
            if len(cand) >= need:
                pick = rng.choice(cand, size=need, replace=False)
                selected.extend(np.asarray(pick, int).tolist())
                used[pick] = True
                continue

            if len(cand):
                selected.extend(cand.tolist())
                used[cand] = True
                need -= len(cand)

            distance = 1
            while need > 0 and distance < MATCH_BINS:
                for bb in (b - distance, b + distance):
                    if bb < 0 or bb >= MATCH_BINS or need <= 0:
                        continue
                    pool = np.flatnonzero((rank_bin == bb) & sup & (~rmf) & (~used) & np.isfinite(sf))
                    if len(pool):
                        take = min(need, len(pool))
                        pick = rng.choice(pool, size=take, replace=False)
                        selected.extend(np.asarray(pick, int).tolist())
                        used[pick] = True
                        need -= take
                        borrowed += take
                distance += 1

            if need > 0:
                pool = np.flatnonzero(sup & (~rmf) & (~used) & np.isfinite(sf))
                if len(pool) < need:
                    raise RuntimeError(f"Layer {li}: positive-support matched-control candidate pool exhausted.")
                pick = rng.choice(pool, size=need, replace=False)
                selected.extend(np.asarray(pick, int).tolist())
                used[pick] = True
                borrowed += need

        if len(selected) != len(target_ids):
            raise RuntimeError(f"Layer {li}: matched-control count {len(selected)} != realized count {len(target_ids)}")
        selected_ids = np.asarray(selected, dtype=np.int64)
        if np.any(rmf[selected_ids]):
            raise RuntimeError(f"Layer {li}: matched control overlaps realized target.")
        if not np.all(sup[selected_ids]):
            raise RuntimeError(f"Layer {li}: matched control left positive realized support.")
        mask[selected_ids] = True

        rt = sf[target_ids]
        mt = sf[selected_ids]
        try:
            ks = float(ks_2samp(rt, mt).statistic)
        except Exception:
            ks = np.nan
        med_r = float(np.median(rt))
        med_m = float(np.median(mt))
        audit.append({
            "layer": li,
            "n_targets": int(len(target_ids)),
            "n_matched_pool": int(len(eligible_ids)),
            "matched_activation_ks": ks,
            "realized_activation_median": med_r,
            "matched_activation_median": med_m,
            "matched_to_realized_activation_median_ratio": float(med_m / (med_r + 1e-12)),
            "borrowed_fraction": float(borrowed / max(len(target_ids), 1)),
        })
        masks.append(mask.reshape(np.asarray(rm).shape))
    return masks, audit

# ==================================================================================================
# 11. OUTPUT-MATCHED STRUCTURAL STATES
# ==================================================================================================

@torch.no_grad()
def output_matched_structural_state(model: nn.Module, family: str, target_idx: int) -> Dict[str, Any]:
    t = torch.tensor([int(target_idx)], dtype=torch.long)
    if family == "mlp":
        src = torch.ones((1, model.layers[0].in_features), dtype=torch.float64)
        resp = [torch.ones((1, l.out_features), dtype=torch.float64, device=DEVICE) for l in model.layers[:-1]]
        return mlp_factor_state(model, src, resp, t)
    shapes = model.spatial_shapes()
    src = torch.ones((1, model.channels[0], IMAGE_RESOLUTION, IMAGE_RESOLUTION), dtype=torch.float64)
    resp = [
        torch.ones((1, model.convs[i].out_channels, *shapes[i + 1]), dtype=torch.float64, device=DEVICE)
        for i in range(len(model.convs) - 1)
    ]
    return cnn_factor_state(model, src, resp, t)

# ==================================================================================================
# 12. ONE BASELINE CONDITION + ONE INTERVENTION
# ==================================================================================================

@torch.no_grad()
def baseline_condition(model: nn.Module, family: str, x: torch.Tensor, x0: torch.Tensor, target_idx: int) -> Dict[str, Any]:
    t = torch.tensor([int(target_idx)], dtype=torch.long)
    if family == "mlp":
        resp = mlp_response_sq(model, x, x0)
        source_sq = torch.square((x - x0).reshape(1, -1).to(torch.float64))
        st = mlp_factor_state(model, source_sq, resp, t)
        out, _, hs = model.forward_with_states(x.to(DEVICE))
        out0, _, hs0 = model.forward_with_states(x0.to(DEVICE))
    else:
        resp = cnn_response_sq(model, x, x0)
        source_sq = torch.square((x - x0).to(torch.float64))
        st = cnn_factor_state(model, source_sq, resp, t)
        out, _, hs = model.forward_with_states(x.to(DEVICE))
        out0, _, hs0 = model.forward_with_states(x0.to(DEVICE))
    return {
        "state": st,
        "response_sq": resp,
        "source_sq": source_sq,
        "target_tensor": t,
        "out": out.detach(),
        "out0": out0.detach(),
        "hs": hs,
        "hs0": hs0,
    }


@torch.no_grad()
def intervene_condition(model: nn.Module, family: str, x: torch.Tensor, x0: torch.Tensor, target_idx: int,
                        masks: List[np.ndarray], baseline: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    gates = masks_to_gates(masks, alpha)
    resp_i, out_i, out0_i, hs_i, hs0_i = gated_response_sq(model, x, x0, gates)
    t = baseline["target_tensor"]
    source_sq = baseline["source_sq"]  # same declared input relation; gate acts only inside the network

    if family == "mlp":
        st_i = mlp_factor_state_gated(model, source_sq, resp_i, t, gates)
    else:
        st_i = cnn_factor_state_gated(model, source_sq, resp_i, t, gates)

    valid_i = bool(st_i["valid"][0])
    hdist = np.nan
    if valid_i:
        logZ0 = float(baseline["state"]["logZ1"][0])
        logZi = float(st_i["logZ1"][0])
        if family == "mlp":
            logZx = float(mlp_cross_log_partition(
                model, source_sq, source_sq, baseline["response_sq"], resp_i, t, gates
            )[0].detach().cpu().item())
        else:
            logZx = float(cnn_cross_log_partition(
                model, source_sq, source_sq, baseline["response_sq"], resp_i, t, gates
            )[0].detach().cpu().item())
        hdist = hellinger_from_logs(logZ0, logZi, logZx)

    return {
        "gates": gates,
        "state": st_i,
        "response_sq": resp_i,
        "out": out_i,
        "out0": out0_i,
        "hellinger": hdist,
        "valid": valid_i,
    }

# ==================================================================================================
# 13. MANIFESTATION MEASURES
# ==================================================================================================

def _target_response(out: torch.Tensor, out0: torch.Tensor, target_idx: int) -> float:
    return float((out[0, int(target_idx)] - out0[0, int(target_idx)]).detach().cpu().item())


def _vector_response(out: torch.Tensor, out0: torch.Tensor) -> np.ndarray:
    return (out[0] - out0[0]).detach().cpu().numpy().astype(float)


def classification_margin(logits: torch.Tensor, target_idx: int) -> float:
    x = logits[0].detach().cpu().numpy().astype(float)
    t = int(target_idx)
    other = np.delete(x, t)
    return float(x[t] - np.max(other)) if len(other) else np.nan


def manifestation_record(dataset: str, y: torch.Tensor, target_idx: int,
                           base_out: torch.Tensor, base_out0: torch.Tensor,
                           int_out: torch.Tensor, int_out0: torch.Tensor) -> Dict[str, Any]:
    r0 = _target_response(base_out, base_out0, target_idx)
    r1 = _target_response(int_out, int_out0, target_idx)
    v0 = _vector_response(base_out, base_out0)
    v1 = _vector_response(int_out, int_out0)
    rec = {
        "baseline_target_response": r0,
        "intervened_target_response": r1,
        "manifestation_abs": float(abs(r1 - r0)),
        "manifestation_relative": float(abs(r1 - r0) / (abs(r0) + MANIFEST_EPS)),
        "finite_response_vector_l2": float(np.linalg.norm(v1 - v0)),
        "baseline_response_abs": float(abs(r0)),
    }
    if dataset == "fmnist":
        ytrue = int(y.reshape(-1)[0].item())
        p0 = int(torch.argmax(base_out, dim=1)[0].item())
        p1 = int(torch.argmax(int_out, dim=1)[0].item())
        ce0 = float(F.cross_entropy(base_out, torch.tensor([ytrue], device=base_out.device)).item())
        ce1 = float(F.cross_entropy(int_out, torch.tensor([ytrue], device=int_out.device)).item())
        rec.update({
            "baseline_prediction": p0,
            "intervened_prediction": p1,
            "prediction_changed": bool(p0 != p1),
            "baseline_target_margin": classification_margin(base_out, target_idx),
            "intervened_target_margin": classification_margin(int_out, target_idx),
            "target_margin_change": float(classification_margin(int_out, target_idx) - classification_margin(base_out, target_idx)),
            "baseline_task_loss": ce0,
            "intervened_task_loss": ce1,
            "task_loss_change": ce1 - ce0,
        })
    else:
        yy = float(y.reshape(-1)[0].item())
        pred0 = float(base_out.reshape(-1)[0].item())
        pred1 = float(int_out.reshape(-1)[0].item())
        mse0 = (pred0 - yy) ** 2
        mse1 = (pred1 - yy) ** 2
        rec.update({
            "baseline_prediction": pred0,
            "intervened_prediction": pred1,
            "prediction_changed": np.nan,
            "baseline_target_margin": np.nan,
            "intervened_target_margin": np.nan,
            "target_margin_change": np.nan,
            "baseline_task_loss": mse0,
            "intervened_task_loss": mse1,
            "task_loss_change": mse1 - mse0,
        })
    return rec

# ==================================================================================================
# 14. OPERATOR-LEVEL SELF TEST â€” RUNS BEFORE DATA DOWNLOAD OR LONG TRAINING
# ==================================================================================================

@torch.no_grad()
def run_operator_selftest() -> pd.DataFrame:
    """Fail-fast numerical/theoretical checks for the new intervention machinery.

    This test deliberately needs no dataset and no trained checkpoint. It checks the new operator
    against the already-established ungated implementation in situations where exact equivalence is
    known. It therefore catches tensor-shape bugs, accidental parameter mutation, wrong gate powers,
    and CNN occurrence-DP mistakes before expensive dataset loading or training starts.
    """
    rows: List[Dict[str, Any]] = []

    def add(family: str, check: str, err: float, tol: Optional[float] = None):
        rows.append({
            "family": family,
            "check": check,
            "max_abs_error": float(err),
            "tolerance": float(PREFLIGHT_TOL if tol is None else tol),
        })

    gen = torch.Generator(device="cpu").manual_seed(SEED + 991)
    cases = [
        ("mlp", DeepMLP([8, 6], "tanh", output_dim=2, seed=SEED + 11, input_channels=1),
         torch.randn((1, 1, IMAGE_RESOLUTION, IMAGE_RESOLUTION), generator=gen), 1),
        ("cnn", DeepCNN({"hidden_channels": [2, 2], "kernels": [3, 3, 1]}, "tanh", output_dim=2,
                        seed=SEED + 13, input_channels=1),
         torch.randn((1, 1, IMAGE_RESOLUTION, IMAGE_RESOLUTION), generator=gen), 1),
    ]

    for family, model, x_cpu, target_idx in cases:
        model = model.to(DEVICE).eval()
        x = x_cpu.to(torch.float32)
        x0 = torch.zeros_like(x)
        baseline = baseline_condition(model, family, x, x0, target_idx)
        if not bool(baseline["state"]["valid"][0]):
            raise RuntimeError(f"Operator self-test unexpectedly produced undefined {family} baseline realization.")
        flow_key = "edge_flow" if family == "mlp" else "occurrence_flow"
        weight_before = [p.detach().cpu().clone() for p in model.parameters()]

        # 1) all-one occurrence/edge gates must be exactly the established model and realization.
        gates1 = ones_mlp_gates(model) if family == "mlp" else ones_cnn_occurrence_gates(model)
        resp1, out1, out01, _, _ = gated_response_sq(model, x, x0, gates1)
        if family == "mlp":
            st1 = mlp_factor_state_gated(model, baseline["source_sq"], resp1, baseline["target_tensor"], gates1)
        else:
            st1 = cnn_factor_state_gated(model, baseline["source_sq"], resp1, baseline["target_tensor"], gates1)
        add(family, "ones_gate_forward_x", tensor_close_max(baseline["out"], out1))
        add(family, "ones_gate_forward_x0", tensor_close_max(baseline["out0"], out01))
        add(family, "ones_gate_eta1", abs(float(baseline["state"]["eta1"][0]) - float(st1["eta1"][0])))
        add(family, "ones_gate_eta2", abs(float(baseline["state"]["eta2"][0]) - float(st1["eta2"][0])))
        add(family, "ones_gate_Sdef", abs(float(baseline["state"]["entropy_deficit"][0]) - float(st1["entropy_deficit"][0])))
        add(family, "ones_gate_flow", max_flow_diff(baseline["state"][flow_key], st1[flow_key]))

        # 2) alpha=0 on an actual high-flow target must be the identity and D_H=0.
        real_flow = flow_list_from_state(baseline["state"], family)
        real_masks, _ = select_realized_masks(real_flow, TARGET_FRACTION)
        zero_int = intervene_condition(model, family, x, x0, target_idx, real_masks, baseline, alpha=0.0)
        m0 = manifestation_record("fmnist", torch.tensor([target_idx]), target_idx,
                                  baseline["out"], baseline["out0"], zero_int["out"], zero_int["out0"])
        add(family, "alpha0_manifestation", float(m0["manifestation_abs"]))
        add(family, "alpha0_hellinger", float(zero_int["hellinger"]))

        # 3) the matched-control construction must preserve budget, stay inside positive realized
        # support, and remain disjoint from the high-flow realized target.
        support = positive_support_masks(real_flow)
        if family == "mlp":
            scores = ordinary_activation_scores_mlp(model, baseline["hs"], baseline["hs0"])
        else:
            scores = ordinary_activation_scores_cnn(model, baseline["hs"], baseline["hs0"])
        matched_masks, _ = select_matched_masks(real_masks, scores, support, SEED + 177)
        budget_err = max(abs(a - b) for a, b in zip(target_counts(real_masks), target_counts(matched_masks)))
        support_err = 0.0
        overlap_err = 0.0
        for rm, mm, sm in zip(real_masks, matched_masks, support):
            support_err = max(support_err, float(np.sum(np.asarray(mm, bool) & (~np.asarray(sm, bool)))))
            overlap_err = max(overlap_err, float(np.sum(np.asarray(mm, bool) & np.asarray(rm, bool))))
        add(family, "matched_budget_identity", float(budget_err), tol=0.0)
        add(family, "matched_inside_positive_support", support_err, tol=0.0)
        add(family, "matched_disjoint_realized", overlap_err, tol=0.0)

        # 4) Nontrivial gate equivalence. When a gate is constant across spatial reuses of a CNN
        # parameter, the occurrence-gated system MUST equal an ordinary model whose corresponding
        # weight tensor has been multiplied by that gate. This is a stringent check of the CNN
        # occurrence-resolved forward map and dynamic program.
        clone = copy.deepcopy(model).to(DEVICE).eval()
        if family == "mlp":
            gates_shared = ones_mlp_gates(model)
            for li, g in enumerate(gates_shared):
                flat = g.ravel()
                flat[::max(3, 11 + li)] = 0.83
                gates_shared[li] = flat.reshape(g.shape)
                clone.layers[li].weight.mul_(torch.as_tensor(gates_shared[li], dtype=clone.layers[li].weight.dtype, device=DEVICE))
            respg, outg, out0g, _, _ = gated_response_sq(model, x, x0, gates_shared)
            stg = mlp_factor_state_gated(model, baseline["source_sq"], respg, baseline["target_tensor"], gates_shared)
            bc = baseline_condition(clone, "mlp", x, x0, target_idx)
            add(family, "nontrivial_shared_gate_forward", tensor_close_max(outg, bc["out"]))
            add(family, "nontrivial_shared_gate_forward_x0", tensor_close_max(out0g, bc["out0"]))
            add(family, "nontrivial_shared_gate_eta1", abs(float(stg["eta1"][0]) - float(bc["state"]["eta1"][0])))
            add(family, "nontrivial_shared_gate_eta2", abs(float(stg["eta2"][0]) - float(bc["state"]["eta2"][0])))
            add(family, "nontrivial_shared_gate_flow", max_flow_diff(stg["edge_flow"], bc["state"]["edge_flow"]))
        else:
            gates_shared = ones_cnn_occurrence_gates(model)
            for li, c in enumerate(model.convs):
                pg = np.ones(tuple(c.weight.shape), dtype=np.float64)
                flat = pg.ravel()
                flat[::max(2, 7 + li)] = 0.83
                pg = flat.reshape(pg.shape)
                gates_shared[li] = np.broadcast_to(pg[..., None, None], gates_shared[li].shape).copy()
                clone.convs[li].weight.mul_(torch.as_tensor(pg, dtype=clone.convs[li].weight.dtype, device=DEVICE))
            respg, outg, out0g, _, _ = gated_response_sq(model, x, x0, gates_shared)
            stg = cnn_factor_state_gated(model, baseline["source_sq"], respg, baseline["target_tensor"], gates_shared)
            bc = baseline_condition(clone, "cnn", x, x0, target_idx)
            add(family, "nontrivial_shared_gate_forward", tensor_close_max(outg, bc["out"]))
            add(family, "nontrivial_shared_gate_forward_x0", tensor_close_max(out0g, bc["out0"]))
            add(family, "nontrivial_shared_gate_eta1", abs(float(stg["eta1"][0]) - float(bc["state"]["eta1"][0])))
            add(family, "nontrivial_shared_gate_eta2", abs(float(stg["eta2"][0]) - float(bc["state"]["eta2"][0])))
            add(family, "nontrivial_shared_gate_flow", max_flow_diff(stg["occurrence_flow"], bc["state"]["occurrence_flow"]))

        # 5) Original model parameters must remain bitwise/numerically untouched by all gate tests.
        mutation = max(tensor_close_max(a, b.detach().cpu()) for a, b in zip(weight_before, model.parameters()))
        add(family, "original_parameter_immutability", mutation, tol=0.0)
        del clone, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df["passed"] = pd.to_numeric(df.max_abs_error, errors="coerce") <= pd.to_numeric(df.tolerance, errors="coerce") + 1e-15
    atomic_csv(df, OUTDIR / "tables" / "operator_selftest.csv")
    bad = df[~df.passed.astype(bool)]
    print(df.to_string(index=False))
    if len(bad):
        raise RuntimeError(
            f"Operator self-test failed {len(bad)} check(s). Inspect {OUTDIR / 'tables' / 'operator_selftest.csv'}."
        )
    print(f"[operator self-test] all {len(df)} fail-fast checks passed before dataset/training work.")
    return df


# ==================================================================================================
# 14. CHECKPOINT IMMUTABILITY / BASELINE-FIDELITY PREFLIGHT
# ==================================================================================================

def file_sha1(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def max_flow_diff(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    vals = []
    for x, y in zip(a, b):
        if np.asarray(x).shape != np.asarray(y).shape:
            return float("inf")
        vals.append(float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float)))))
    return max(vals) if vals else 0.0


@torch.no_grad()
def preflight_one_model(row: pd.Series, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rid = str(row.run_id)
    cp = checkpoint_path(rid)
    cp_hash_before = file_sha1(cp)
    model = build_model(str(row.arch_id), str(row.activation), output_dim_for(str(row.dataset)), str(row.dataset), int(row.seed_index))
    chk = torch.load(cp, map_location="cpu")
    model.load_state_dict(chk["state_dict"])
    model.to(DEVICE).eval()
    family = str(row.family)

    Xc, yc, _ = prepared_endpoint_environment(data, str(row.dataset), "clean")
    base = None
    x = y = x0 = None
    target = 0
    for jj in range(len(Xc)):
        xj = Xc[jj:jj + 1]
        yj = yc[jj:jj + 1]
        x0j = torch.zeros_like(xj)
        tj = int(yj.reshape(-1)[0].item()) if str(row.task) == "classification" else 0
        bj = baseline_condition(model, family, xj, x0j, tj)
        if bool(bj["state"]["valid"][0]):
            x, y, x0, target, base = xj, yj, x0j, tj, bj
            break
    if base is None:
        raise RuntimeError(f"Preflight could not find a valid baseline realization among the frozen causal conditions for sentinel {rid}.")

    # All-one gate must reproduce the exact baseline forward map and factor state.
    gates = ones_mlp_gates(model) if family == "mlp" else ones_cnn_occurrence_gates(model)
    resp_g, out_g, out0_g, _, _ = gated_response_sq(model, x, x0, gates)
    if family == "mlp":
        st_g = mlp_factor_state_gated(model, base["source_sq"], resp_g, base["target_tensor"], gates)
        flow_key = "edge_flow"
    else:
        st_g = cnn_factor_state_gated(model, base["source_sq"], resp_g, base["target_tensor"], gates)
        flow_key = "occurrence_flow"

    checks = []
    checks.append({"run_id": rid, "family": family, "check": "forward_identity_x", "max_abs_error": tensor_close_max(base["out"], out_g)})
    checks.append({"run_id": rid, "family": family, "check": "forward_identity_x0", "max_abs_error": tensor_close_max(base["out0"], out0_g)})
    checks.append({"run_id": rid, "family": family, "check": "eta1_identity", "max_abs_error": abs(float(base["state"]["eta1"][0]) - float(st_g["eta1"][0]))})
    checks.append({"run_id": rid, "family": family, "check": "eta2_identity", "max_abs_error": abs(float(base["state"]["eta2"][0]) - float(st_g["eta2"][0]))})
    checks.append({"run_id": rid, "family": family, "check": "Sdef_identity", "max_abs_error": abs(float(base["state"]["entropy_deficit"][0]) - float(st_g["entropy_deficit"][0]))})
    checks.append({"run_id": rid, "family": family, "check": "flow_identity", "max_abs_error": max_flow_diff(base["state"][flow_key], st_g[flow_key])})

    # alpha=0 must yield exactly zero intervention effect and zero Hellinger distance.
    flow_list = flow_list_from_state(base["state"], family)
    real_masks, _ = select_realized_masks(flow_list, TARGET_FRACTION)
    zero_int = intervene_condition(model, family, x, x0, target, real_masks, base, alpha=0.0)
    mrec = manifestation_record(str(row.dataset), y, target, base["out"], base["out0"], zero_int["out"], zero_int["out0"])
    checks.append({"run_id": rid, "family": family, "check": "alpha0_manifestation", "max_abs_error": float(mrec["manifestation_abs"])})
    checks.append({"run_id": rid, "family": family, "check": "alpha0_hellinger", "max_abs_error": float(zero_int["hellinger"])})

    # CNN-specific occurrence identity: one gated occurrence must not rewrite or globally gate the shared kernel.
    if family == "cnn":
        weight_before = [c.weight.detach().cpu().clone() for c in model.convs]
        masks = [np.zeros_like(g, dtype=bool) for g in gates]
        # Pick the first positive occurrence-flow element in the first layer.
        f0 = np.asarray(flow_list[0], float).ravel()
        pos = np.flatnonzero(f0 > 0)
        if len(pos):
            masks[0].ravel()[int(pos[0])] = True
            _ = cnn_forward_with_gates(model, x, masks_to_gates(masks, ATTENUATION))
            weight_error = max(tensor_close_max(a, c.weight.detach().cpu()) for a, c in zip(weight_before, model.convs))
            checks.append({"run_id": rid, "family": family, "check": "cnn_parameter_tensor_immutable", "max_abs_error": weight_error})

    cp_hash_after = file_sha1(cp)
    checks.append({"run_id": rid, "family": family, "check": "checkpoint_hash_immutable", "max_abs_error": 0.0 if cp_hash_before == cp_hash_after else 1.0})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checks


def run_preflight(manifest: pd.DataFrame, data_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Internal fidelity checks on newly trained sentinel checkpoints.

    There is intentionally no comparison to any previous experiment. Fidelity is checked by
    equivalence between the established ungated implementation and the new intervention-aware
    implementation when gates equal one / alpha equals zero, plus checkpoint immutability.
    """
    if not RUN_PREFLIGHT:
        return pd.DataFrame()
    rows = []
    preferred = manifest[
        manifest.arch_id.astype(str).isin(["M2_sustained", "C2_bottleneck"])
        & (manifest.optimizer.astype(str) == "adam")
        & (manifest.activation.astype(str) == "relu")
        & (pd.to_numeric(manifest.seed_index, errors="coerce") == 0)
    ].copy()
    if preferred.empty:
        preferred = pd.concat([
            manifest[manifest.family.astype(str) == "mlp"].head(1),
            manifest[manifest.family.astype(str) == "cnn"].head(1),
        ], ignore_index=True).drop_duplicates("run_id")
    preferred = preferred[preferred.run_id.astype(str).map(lambda r: checkpoint_path(r).exists())]
    if preferred.empty:
        raise RuntimeError("No trained sentinel checkpoint exists for intervention fidelity preflight.")
    for _, row in tqdm(preferred.iterrows(), total=len(preferred), desc="Intervention fidelity preflight"):
        rows.extend(preflight_one_model(row, data_cache[str(row.dataset)]))
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "intervention_preflight.csv")
    bad = df[pd.to_numeric(df.max_abs_error, errors="coerce") > PREFLIGHT_TOL]
    print(df.to_string(index=False))
    if len(bad):
        raise RuntimeError(
            f"Intervention preflight failed {len(bad)} identity check(s) at tolerance {PREFLIGHT_TOL}. "
            f"Inspect {OUTDIR / 'tables' / 'intervention_preflight.csv'}."
        )
    print(f"[preflight] all {len(df)} internal fidelity checks passed at tolerance {PREFLIGHT_TOL:g}.")
    return df





# ==================================================================================================
# 15. EXPERIMENT 5 CONFIGURATION AND NON-DESTRUCTIVE I/O
# ==================================================================================================

import io
import gc
import tempfile

EXP5_VERSION = "EXP5_RQ7_OCCURRENCE_SHARED_IDENTITY_v1"
EXP5_ENGINEERING_PATCH = "v2_streaming_aggregation"
EXP5_CONDITION_COUNT = int(os.environ.get("PATHFLOW_EXP5_MEASURE_CONDITIONS", "10"))
EXP5_ALPHA = float(os.environ.get("PATHFLOW_EXP5_ALPHA", "0.25"))
EXP5_TARGET_FRACTION = float(os.environ.get("PATHFLOW_EXP5_PARAMETER_TARGET_FRACTION", str(TARGET_FRACTION)))
EXP5_BUFFER_MIN = float(os.environ.get("PATHFLOW_EXP5_CONDITION_BUFFER_MIN", "5.0"))
EXP5_RENDER_CONDITIONS_RAW = os.environ.get("PATHFLOW_EXP5_RENDER_CONDITIONS", "0").strip().lower()
EXP5_RENDER_POST_FIELDS = os.environ.get("PATHFLOW_EXP5_RENDER_POST_FIELDS", "0") == "1"
EXP5_FIG_DPI = int(os.environ.get("PATHFLOW_EXP5_FIG_DPI", "220"))
EXP5_FIELD_TOP_FRACTION = float(os.environ.get("PATHFLOW_EXP5_FIELD_TOP_FRACTION", "0.10"))
INTERVENTION_TYPES = ("occurrence", "parameter")

if PROFILE == "pilot":
    EXP5_CONDITION_COUNT = min(2, CONDITIONS_PER_MODEL)
if EXP5_CONDITION_COUNT <= 0 or EXP5_CONDITION_COUNT > CONDITIONS_PER_MODEL:
    raise ValueError(f"EXP5_CONDITION_COUNT must lie in [1,{CONDITIONS_PER_MODEL}].")
if not (0.0 < EXP5_ALPHA < 1.0):
    raise ValueError("EXP5_ALPHA must lie in (0,1).")
if not (0.0 < EXP5_TARGET_FRACTION <= 1.0):
    raise ValueError("EXP5_TARGET_FRACTION must lie in (0,1].")
if not (0.0 < EXP5_FIELD_TOP_FRACTION <= 1.0):
    raise ValueError("EXP5_FIELD_TOP_FRACTION must lie in (0,1].")

# Extra output directories are created without deleting or replacing anything.
for _sub in ["fields"]:
    (OUTDIR / _sub).mkdir(parents=True, exist_ok=True)


def _publish_bytes_once(path: Path, payload: bytes) -> bool:
    """Publish bytes atomically without ever replacing an existing file.

    Returns True if the file was created, False if an identical file already existed.
    Raises if an existing file has different contents.  The hard-link publication step is atomic on the
    same filesystem and cannot overwrite a destination.  A conservative O_EXCL fallback is provided for
    filesystems that do not support hard-link publication.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_bytes()
        if old == payload:
            return False
        raise RuntimeError(
            f"Write-once protection: refusing to overwrite existing scientific file with different contents: {path}"
        )
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(tmp, path)  # fails if destination exists; never overwrites
        except OSError:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(payload)
                    out.flush()
                    os.fsync(out.fileno())
            except Exception:
                try:
                    path.unlink()
                except Exception:
                    pass
                raise
        return True
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def write_once_text(text: str, path: Path) -> bool:
    return _publish_bytes_once(Path(path), text.encode("utf-8"))


def _sha1_file_stream(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _publish_file_once(tmp: Path, path: Path) -> bool:
    """Publish an already-written temporary file atomically without loading it into RAM.

    If an identical destination already exists, keep it and discard the temporary file.  If a
    different destination exists, fail rather than overwrite scientific output.
    """
    tmp = Path(tmp); path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            same = (path.stat().st_size == tmp.stat().st_size) and (_sha1_file_stream(path) == _sha1_file_stream(tmp))
            if same:
                return False
            raise RuntimeError(
                f"Write-once protection: refusing to overwrite existing scientific file with different contents: {path}"
            )
        try:
            os.link(tmp, path)
        except OSError:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(fd, "wb") as out, open(tmp, "rb") as src:
                    shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
                    out.flush(); os.fsync(out.fileno())
            except Exception:
                try:
                    path.unlink()
                except Exception:
                    pass
                raise
        return True
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def atomic_csv(df: pd.DataFrame, path: Path):
    """Memory-bounded, atomic, content-checked, write-once CSV.

    Unlike v1, this never materializes the complete CSV as a Python bytes object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        df.to_csv(tmp, index=False)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        return _publish_file_once(tmp, path)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def atomic_json(obj: Any, path: Path):
    """Experiment-5 redefinition: atomic, content-checked, write-once JSON."""
    payload = (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    _publish_bytes_once(Path(path), payload)


def write_once_npz(path: Path, **arrays) -> bool:
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    return _publish_bytes_once(Path(path), bio.getvalue())


def publish_figure_once(fig, path: Path, **savefig_kwargs) -> bool:
    path = Path(path)
    if path.exists():
        return False
    suffix = path.suffix if path.suffix else ".png"
    tmp = path.parent / f".{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{suffix}"
    fig.savefig(tmp, **savefig_kwargs)
    payload = tmp.read_bytes()
    try:
        return _publish_bytes_once(path, payload)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


def _render_condition_indices(n: int) -> set:
    raw = EXP5_RENDER_CONDITIONS_RAW
    if raw in {"", "none", "off", "-1"}:
        return set()
    if raw == "all":
        return set(range(n))
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if 0 <= i < n:
            out.add(i)
    return out


def exp5_part_stem(run_id: str, environment: str, condition_index: int) -> str:
    return f"{run_id}__EXP5__{environment}__C{int(condition_index):02d}"


def exp5_condition_paths(run_id: str, environment: str, condition_index: int) -> Dict[str, Path]:
    stem = exp5_part_stem(run_id, environment, condition_index)
    return {
        "measurements": OUTDIR / "run_parts" / f"{stem}__measurements.csv",
        "validity": OUTDIR / "run_parts" / f"{stem}__validity.csv",
        "target_audit": OUTDIR / "run_parts" / f"{stem}__target_audit.csv",
        "field_diagnostics": OUTDIR / "run_parts" / f"{stem}__field_diagnostics.csv",
        "fields": OUTDIR / "fields" / f"{stem}__fields.npz",
        "figure": OUTDIR / "figures" / f"{stem}__fieldA_fieldB.png",
        "post_figure": OUTDIR / "figures" / f"{stem}__post_fieldA.png",
        "done": OUTDIR / "run_parts" / f"{stem}__DONE.json",
    }


def exp5_condition_complete(run_id: str, environment: str, condition_index: int) -> bool:
    p = exp5_condition_paths(run_id, environment, condition_index)
    required = [p["measurements"], p["validity"], p["target_audit"], p["field_diagnostics"], p["fields"], p["done"]]
    return all(x.exists() and x.stat().st_size > 0 for x in required)


def exp5_done_path(run_id: str) -> Path:
    return OUTDIR / "run_parts" / f"{run_id}__EXP5__DONE.json"


# ==================================================================================================
# 16. FROZEN CNN POPULATION AND EXACT CONDITION-COHORT RECONSTRUCTION
# ==================================================================================================


def cnn_manifest() -> pd.DataFrame:
    m = build_manifest()
    m = m[m.family.astype(str) == "cnn"].copy().reset_index(drop=True)
    # Respect user filters inherited from the common benchmark configuration.
    m = m[
        m.dataset.astype(str).isin(DATASETS)
        & m.optimizer.astype(str).isin(OPTIMIZERS)
        & m.activation.astype(str).isin(ACTIVATIONS)
        & pd.to_numeric(m.seed_index, errors="coerce").astype(int).isin(SEEDS)
    ].reset_index(drop=True)
    if m.empty:
        raise RuntimeError("Experiment 5 CNN manifest is empty after configuration filters.")
    return m


def verify_exp1_cnn_checkpoints(manifest: pd.DataFrame) -> pd.DataFrame:
    rows, missing = [], []
    for _, r in manifest.iterrows():
        rid = str(r.run_id)
        cp = checkpoint_path(rid)
        exists = cp.exists()
        rows.append({
            "run_id": rid,
            "dataset": str(r.dataset),
            "family": str(r.family),
            "arch_id": str(r.arch_id),
            "optimizer": str(r.optimizer),
            "activation": str(r.activation),
            "seed_index": int(r.seed_index),
            "checkpoint_exists": bool(exists),
            "checkpoint_path": str(cp),
            "checkpoint_bytes": int(cp.stat().st_size) if exists else 0,
            "checkpoint_sha1": file_sha1(cp) if exists else "",
        })
        if not exists:
            missing.append(str(cp))
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "source_checkpoint_inventory.csv")
    if missing:
        raise RuntimeError(
            f"Experiment 5 requires frozen Experiment 1 CNN checkpoints. Missing {len(missing)} checkpoint(s).\n"
            + "\n".join(missing[:20])
        )
    return df


def select_exp5_endpoint_local_indices(data: Dict[str, Any], dataset: str, count: int) -> np.ndarray:
    endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
    causal_local = np.asarray(data["causal_endpoint_local_indices"], dtype=int)
    strata_test = np.asarray(data["test_strata"])
    causal_strata = strata_test[endpoint_ids[causal_local]]
    if dataset == "fmnist" and count == 10:
        selected = []
        for cls in range(10):
            ids = np.flatnonzero(causal_strata == cls)
            if len(ids) == 0:
                raise RuntimeError(f"Frozen Experiment 1 causal population has no FMNIST class {cls}.")
            selected.append(int(causal_local[int(ids[0])]))
        return np.asarray(selected, dtype=int)
    if dataset == "mpi3d" and count == 10:
        selected = []
        for level in range(0, 40, 4):
            ids = np.flatnonzero(causal_strata == level)
            if len(ids) == 0:
                raise RuntimeError(f"Frozen Experiment 1 causal population has no MPI3D level {level}.")
            selected.append(int(causal_local[int(ids[0])]))
        return np.asarray(selected, dtype=int)
    chosen_positions = balanced_indices_from_strata(
        causal_strata,
        min(int(count), len(causal_local)),
        CONDITION_SEED_BASE + 15001 + (0 if dataset == "fmnist" else 1),
    )
    return causal_local[chosen_positions]


def attach_exp5_population(data_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ds, data in data_cache.items():
        local = select_exp5_endpoint_local_indices(data, ds, EXP5_CONDITION_COUNT)
        data["exp5_endpoint_local_indices"] = local
        endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
        strata = np.asarray(data["test_strata"])
        causal_local = np.asarray(data["causal_endpoint_local_indices"], dtype=int)
        pos_map = {int(v): i for i, v in enumerate(causal_local)}
        for ci, ep_local in enumerate(local):
            test_id = int(endpoint_ids[int(ep_local)])
            rows.append({
                "dataset": ds,
                "exp5_condition_index": int(ci),
                "exp1_condition_index": int(pos_map[int(ep_local)]),
                "endpoint_local_index": int(ep_local),
                "test_index": test_id,
                "stratum": strata[test_id],
            })
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "exp5_condition_population.csv")
    return df


def _norm_stratum(v: Any) -> str:
    try:
        x = float(v)
        if np.isfinite(x) and abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
    except Exception:
        pass
    return str(v).strip()


def audit_prior_cohorts_if_present(population: pd.DataFrame) -> Dict[str, Any]:
    audits = []
    candidates = [
        ("experiment2", EXP1_RESULTS / "experiment2_condition_crossover" / "tables" / "crossover_pair_population.csv"),
        ("experiment3", EXP1_RESULTS / "experiment3_dose_response" / "tables" / "dose_condition_population.csv"),
        ("experiment4", EXP1_RESULTS / "experiment4_multiscale_followup" / "tables" / "multiscale_condition_population.csv"),
    ]
    cur = population[["dataset", "test_index", "stratum"]].copy()
    cur["dataset"] = cur["dataset"].astype(str)
    cur["test_index"] = pd.to_numeric(cur["test_index"], errors="raise").astype(int)
    cur["stratum"] = cur["stratum"].map(_norm_stratum)
    cur = cur.sort_values(["dataset", "test_index", "stratum"]).reset_index(drop=True)
    for name, path in candidates:
        rec = {"source": name, "path": str(path), "present": bool(path.exists())}
        if path.exists():
            old = pd.read_csv(path)
            req = {"dataset", "test_index", "stratum"}
            rec["schema_match"] = bool(req.issubset(old.columns))
            if rec["schema_match"]:
                other = old[["dataset", "test_index", "stratum"]].copy()
                other["dataset"] = other["dataset"].astype(str)
                other["test_index"] = pd.to_numeric(other["test_index"], errors="raise").astype(int)
                other["stratum"] = other["stratum"].map(_norm_stratum)
                other = other.sort_values(["dataset", "test_index", "stratum"]).reset_index(drop=True)
                same = len(cur) == len(other) and cur.equals(other)
                rec["row_count_current"] = int(len(cur))
                rec["row_count_prior"] = int(len(other))
                rec["cohort_match"] = bool(same)
                if not same:
                    raise RuntimeError(f"{name} cohort table exists but does not match Experiment 5's reconstructed cohort.")
        audits.append(rec)
    out = {"audits": audits}
    atomic_json(out, OUTDIR / "audit" / "prior_cohort_audit.json")
    return out


def prepared_exp5_environment(data: Dict[str, Any], dataset: str, environment: str) -> Tuple[torch.Tensor, torch.Tensor]:
    Xt, yt = data["test"]
    endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
    local = np.asarray(data["exp5_endpoint_local_indices"], dtype=int)
    Xep = Xt[endpoint_ids]
    yep = yt[endpoint_ids]
    if environment == "shifted":
        # Preserve the benchmark's shape-dependent deterministic noise semantics: shift full 160 endpoint population first.
        Xep = shifted_environment(Xep, dataset, data["meta"])
    return Xep[local], yep[local]


# ==================================================================================================
# 17. EQUAL-BUDGET OCCURRENCE VS SHARED-PARAMETER TARGET CONSTRUCTION
# ==================================================================================================


def _broadcast_parameter_mask_to_occurrences(param_mask: np.ndarray, occurrence_shape: Tuple[int, ...]) -> np.ndarray:
    p = np.asarray(param_mask, bool)
    if len(occurrence_shape) != p.ndim + 2:
        raise RuntimeError(f"Expected occurrence shape = parameter shape + Hout,Wout; got {occurrence_shape} vs {p.shape}")
    return np.broadcast_to(p[..., None, None], occurrence_shape).copy()


def select_occurrence_vs_parameter_masks(state: Dict[str, Any], fraction: float) -> Dict[str, Any]:
    """Construct the two RQ7 intervention supports with an EXACT layerwise primitive-occurrence budget.

    Parameter arm:
      - among parameter identities with positive conditional parameter flow, select the top ceil(fraction*n_positive);
      - every spatial reuse of each selected identity is attenuated.

    Occurrence arm:
      - receives exactly the same number B_l of primitive occurrences in each layer;
      - draws from the same positive-parameter-support universe (all spatial reuses of identities whose
        aggregate parameter flow is positive);
      - chooses the B_l highest occurrence-flow spatial uses.

    This is intentionally a localization-efficiency comparison.  The occurrence arm is allowed to spend
    the same primitive budget more selectively in space; the parameter arm is constrained by sharing fibres.
    """
    occ_masks, param_occ_masks, param_identity_masks, audit = [], [], [], []
    occurrence_layers = state.get("occurrence_flow", [])
    parameter_layers = state.get("parameter_flow", [])
    if len(occurrence_layers) != len(parameter_layers):
        raise RuntimeError("Occurrence/parameter flow layer count mismatch.")

    for li, (Eb, Pb) in enumerate(zip(occurrence_layers, parameter_layers)):
        E = np.asarray(Eb[0], float)
        P = np.asarray(Pb[0], float)
        if E.shape[:-2] != P.shape:
            raise RuntimeError(f"Layer {li}: occurrence/parameter shape mismatch {E.shape} vs {P.shape}.")
        positive_param = np.isfinite(P) & (P > 0)
        pos_ids = np.flatnonzero(positive_param.ravel())
        if len(pos_ids) == 0:
            raise RuntimeError(f"Layer {li}: no positive conditional parameter-flow identities.")

        m = max(1, int(math.ceil(float(fraction) * len(pos_ids))))
        m = min(m, len(pos_ids))
        pflat = P.ravel()
        chosen_param_ids = pos_ids[np.argsort(-pflat[pos_ids], kind="mergesort")[:m]]
        p_mask_flat = np.zeros(P.size, dtype=bool)
        p_mask_flat[chosen_param_ids] = True
        p_mask = p_mask_flat.reshape(P.shape)

        param_occ = _broadcast_parameter_mask_to_occurrences(p_mask, E.shape)
        positive_param_occ_universe = _broadcast_parameter_mask_to_occurrences(positive_param, E.shape)
        primitive_budget = int(param_occ.sum())
        eligible = np.flatnonzero(positive_param_occ_universe.ravel() & np.isfinite(E.ravel()))
        if primitive_budget <= 0 or len(eligible) < primitive_budget:
            raise RuntimeError(
                f"Layer {li}: cannot construct equal primitive budget B={primitive_budget} from eligible={len(eligible)}."
            )
        order = eligible[np.argsort(-E.ravel()[eligible], kind="mergesort")[:primitive_budget]]
        occ_flat = np.zeros(E.size, dtype=bool)
        occ_flat[order] = True
        occ = occ_flat.reshape(E.shape)

        occ_mass = float(np.sum(E[occ]))
        param_mass = float(np.sum(E[param_occ]))
        # Because occurrence selection is the top-B subset of a universe containing the parameter arm,
        # its selected baseline flow mass cannot be lower except for floating-point roundoff.
        if occ_mass + 1e-12 < param_mass:
            raise RuntimeError(f"Layer {li}: top-B occurrence selected mass {occ_mass} < parameter fibre mass {param_mass}.")

        spatial_constancy_error = 0
        for idx in np.ndindex(P.shape):
            vals = param_occ[idx]
            expect = bool(p_mask[idx])
            spatial_constancy_error = max(spatial_constancy_error, int(np.any(vals != expect)))

        audit.append({
            "layer": int(li),
            "parameter_identities_total": int(P.size),
            "parameter_identities_positive_flow": int(len(pos_ids)),
            "parameter_target_identities": int(m),
            "parameter_target_fraction_of_positive": float(m / max(len(pos_ids), 1)),
            "reuse_multiplicity": int(E.shape[-2] * E.shape[-1]),
            "primitive_occurrence_budget": int(primitive_budget),
            "occurrence_selected_count": int(occ.sum()),
            "parameter_affected_occurrence_count": int(param_occ.sum()),
            "equal_budget": bool(int(occ.sum()) == int(param_occ.sum())),
            "parameter_fibre_closed": bool(spatial_constancy_error == 0),
            "eligible_occurrences_same_positive_parameter_support": int(len(eligible)),
            "occurrence_selected_flow_mass": occ_mass,
            "parameter_selected_flow_mass": param_mass,
            "selected_flow_mass_advantage_occ_minus_param": float(occ_mass - param_mass),
            "selected_flow_mass_ratio_occ_over_param": float(occ_mass / (param_mass + EPS)),
            "occurrence_mask_hash": hash_indices(occ),
            "parameter_identity_mask_hash": hash_indices(p_mask),
            "parameter_occurrence_mask_hash": hash_indices(param_occ),
        })
        occ_masks.append(occ)
        param_occ_masks.append(param_occ)
        param_identity_masks.append(p_mask)

    return {
        "occurrence_masks": occ_masks,
        "parameter_occurrence_masks": param_occ_masks,
        "parameter_identity_masks": param_identity_masks,
        "audit": audit,
    }


# ==================================================================================================
# 18. FIELD A AND FIELD B — DIRECTLY FROM EXISTING CONDITIONAL FLOW OBJECTS
# ==================================================================================================


def fieldA_input_origin(state: Dict[str, Any]) -> np.ndarray:
    """Input-origin marginal P_in^c(s) = sum_channel input-node flow.

    The node-flow tensor is already normalized per condition.  At the input layer, complete paths that
    pass through an input node necessarily originate there, so this is exactly the image-space marginal
    of the complete conditional path measure.
    """
    node0 = np.asarray(state["node_flow"][0][0], float)
    if node0.ndim != 3:
        raise RuntimeError(f"Expected CNN input node flow [Cin,H,W], got {node0.shape}.")
    out = np.sum(node0, axis=0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float64)


def first_layer_source_footprint(occurrence_flow: np.ndarray, mask: np.ndarray, input_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Map selected FIRST-LAYER occurrences to exact input source coordinates.

    E / mask shape: [Cout,Cin,k,k,Hout,Wout].  For valid convolution, occurrence (kh,kw,ho,wo)
    reads source pixel (ho+kh, wo+kw).  Returned arrays are:
      flow_field[s]  = selected baseline occurrence-flow mass sourced at input coordinate s
      count_field[s] = number of selected primitive occurrences sourced at s
    """
    E = np.asarray(occurrence_flow, float)
    M = np.asarray(mask, bool)
    if E.shape != M.shape or E.ndim != 6:
        raise RuntimeError(f"First-layer footprint shape mismatch E={E.shape}, M={M.shape}.")
    _, _, k1, k2, Hout, Wout = E.shape
    Hin, Win = map(int, input_hw)
    if (Hout + k1 - 1, Wout + k2 - 1) != (Hin, Win):
        raise RuntimeError(
            f"Valid-convolution spatial mapping mismatch: E gives {(Hout+k1-1,Wout+k2-1)}, input is {(Hin,Win)}."
        )
    flow_field = np.zeros((Hin, Win), dtype=np.float64)
    count_field = np.zeros((Hin, Win), dtype=np.int64)
    for kh in range(k1):
        for kw in range(k2):
            weighted = np.where(M[:, :, kh, kw, :, :], E[:, :, kh, kw, :, :], 0.0).sum(axis=(0, 1))
            counts = M[:, :, kh, kw, :, :].sum(axis=(0, 1)).astype(np.int64)
            flow_field[kh:kh+Hout, kw:kw+Wout] += weighted
            count_field[kh:kh+Hout, kw:kw+Wout] += counts
    return flow_field, count_field


def field_entropy(p: np.ndarray) -> float:
    x = np.asarray(p, float).ravel()
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        return np.nan
    z = float(x.sum())
    if z <= 0:
        return np.nan
    q = x / z
    return float(-np.sum(q * np.log(np.clip(q, PATH_LOG_FLOOR, None))))


def top_fraction_mass(p: np.ndarray, fraction: float) -> float:
    x = np.asarray(p, float).ravel()
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0 or float(x.sum()) <= 0:
        return np.nan
    k = max(1, int(math.ceil(float(fraction) * len(x))))
    ids = np.argsort(-x, kind="mergesort")[:k]
    return float(x[ids].sum() / x.sum())


def field_support_mass(fieldA: np.ndarray, count_field: np.ndarray) -> float:
    A = np.asarray(fieldA, float)
    support = np.asarray(count_field) > 0
    if A.shape != support.shape:
        return np.nan
    return float(np.sum(A[support]))


def field_l1_normalized(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    sx, sy = float(np.sum(x)), float(np.sum(y))
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(np.sum(np.abs(x / sx - y / sy)))


def image_for_storage_and_display(x: torch.Tensor, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    arr = x[0].detach().cpu().numpy().astype(np.float64)
    raw = arr * float(meta["input_sd"]) + float(meta["input_mean"])
    if raw.shape[0] == 1:
        raw_hwc = raw[0]
    else:
        raw_hwc = np.moveaxis(raw, 0, -1)
    display = np.clip(raw_hwc, 0.0, 1.0)
    return raw_hwc, display


def render_fieldA_fieldB_figure(path: Path, image_display: np.ndarray, A: np.ndarray,
                                 Bocc: np.ndarray, Bparam: np.ndarray,
                                 meta_text: str, Docc: float, Dparam: float,
                                 Hocc: float, Hparam: float):
    if not RUN_FIGURES or path.exists():
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))
    ax = axes.ravel()
    if image_display.ndim == 2:
        ax[0].imshow(image_display, cmap="gray", vmin=0, vmax=1)
    else:
        ax[0].imshow(image_display)
    ax[0].set_title("Input condition x")
    imA = ax[1].imshow(A, cmap="magma", vmin=0, vmax=max(float(np.nanmax(A)), EPS))
    ax[1].set_title("Field A: input-origin path mass")
    fig.colorbar(imA, ax=ax[1], fraction=0.046, pad=0.04)
    if image_display.ndim == 2:
        ax[2].imshow(image_display, cmap="gray", vmin=0, vmax=1)
    else:
        ax[2].imshow(image_display)
    ax[2].imshow(A, cmap="magma", alpha=0.58, vmin=0, vmax=max(float(np.nanmax(A)), EPS))
    ax[2].set_title("Field A overlay")
    vmaxB = max(float(np.nanmax(Bocc)), float(np.nanmax(Bparam)), EPS)
    imO = ax[3].imshow(Bocc, cmap="viridis", vmin=0, vmax=vmaxB)
    ax[3].set_title(f"Field B occurrence\nselected flow mass={np.sum(Bocc):.4f}")
    fig.colorbar(imO, ax=ax[3], fraction=0.046, pad=0.04)
    imP = ax[4].imshow(Bparam, cmap="viridis", vmin=0, vmax=vmaxB)
    ax[4].set_title(f"Field B shared identity\nselected flow mass={np.sum(Bparam):.4f}")
    fig.colorbar(imP, ax=ax[4], fraction=0.046, pad=0.04)
    diff = Bocc - Bparam
    vmaxD = max(float(np.nanmax(np.abs(diff))), EPS)
    imD = ax[5].imshow(diff, cmap="coolwarm", vmin=-vmaxD, vmax=vmaxD)
    ax[5].set_title("Field B: occurrence − parameter")
    fig.colorbar(imD, ax=ax[5], fraction=0.046, pad=0.04)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        meta_text + f"\nD_man occ={Docc:.4g}, param={Dparam:.4g} | D_H occ={Hocc:.4g}, param={Hparam:.4g}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    publish_figure_once(fig, path, dpi=EXP5_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def render_post_fieldA_figure(path: Path, image_display: np.ndarray, A0: np.ndarray,
                              Aocc: np.ndarray, Aparam: np.ndarray, meta_text: str):
    if not RUN_FIGURES or not EXP5_RENDER_POST_FIELDS or path.exists():
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))
    fields = [A0, Aocc, Aparam]
    titles = ["Field A baseline", "Field A post occurrence", "Field A post parameter"]
    vmax = max(max(float(np.nanmax(z)), EPS) for z in fields)
    for j, (z, title) in enumerate(zip(fields, titles)):
        im = axes[0, j].imshow(z, cmap="magma", vmin=0, vmax=vmax)
        axes[0, j].set_title(title); fig.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.04)
    diffs = [Aocc-A0, Aparam-A0, (Aocc-A0)-(Aparam-A0)]
    dt = ["Δ Field A occurrence", "Δ Field A parameter", "Δ occurrence − Δ parameter"]
    vm = max(max(float(np.nanmax(np.abs(z))), EPS) for z in diffs)
    for j, (z, title) in enumerate(zip(diffs, dt)):
        im = axes[1, j].imshow(z, cmap="coolwarm", vmin=-vm, vmax=vm)
        axes[1, j].set_title(title); fig.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.04)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(meta_text, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    publish_figure_once(fig, path, dpi=EXP5_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ==================================================================================================
# 19. EXPERIMENT-5-SPECIFIC SELF-TEST: PARAMETER FIBRES, EQUAL BUDGETS, FIELD IDENTITIES
# ==================================================================================================


def _max_list_array_diff(a: List[np.ndarray], b: List[np.ndarray]) -> float:
    vals = []
    for x, y in zip(a, b):
        xx, yy = np.asarray(x, float), np.asarray(y, float)
        vals.append(float(np.nanmax(np.abs(xx - yy))) if xx.size else 0.0)
    return max(vals) if vals else 0.0


@torch.no_grad()
def run_exp5_operator_selftest() -> pd.DataFrame:
    rows = []
    def add(name: str, value: float, tol: float = PREFLIGHT_TOL, note: str = ""):
        rows.append({"check": name, "value": float(value), "tolerance": float(tol), "passed": bool(value <= tol), "note": note})

    model = build_model("C1_taper", "silu", 10, "fmnist", 0).to(DEVICE).eval()
    g = torch.Generator(device="cpu").manual_seed(SEED + 8801)
    x = torch.randn((1, 1, IMAGE_RESOLUTION, IMAGE_RESOLUTION), generator=g, dtype=torch.float32) * 0.35 + 0.1
    x0 = torch.zeros_like(x)
    target = 3
    base = baseline_condition(model, "cnn", x, x0, target)
    if not bool(base["state"]["valid"][0]):
        raise RuntimeError("Experiment 5 dataset-free self-test unexpectedly produced zero complete-path support.")
    sel = select_occurrence_vs_parameter_masks(base["state"], EXP5_TARGET_FRACTION)
    for li, a in enumerate(sel["audit"]):
        add(f"layer{li}_equal_primitive_budget", abs(a["occurrence_selected_count"] - a["parameter_affected_occurrence_count"]), 0.0)
        add(f"layer{li}_parameter_fibre_closed", 0.0 if a["parameter_fibre_closed"] else 1.0, 0.0)
        add(f"layer{li}_occurrence_mass_not_below_parameter", max(0.0, -a["selected_flow_mass_advantage_occ_minus_param"]), 1e-12)

    # alpha=0 must be exact identity for BOTH support constructions.
    for typ, masks in [("occurrence", sel["occurrence_masks"]), ("parameter", sel["parameter_occurrence_masks"])]:
        z = intervene_condition(model, "cnn", x, x0, target, masks, base, alpha=0.0)
        add(f"alpha0_{typ}_output", tensor_close_max(z["out"], base["out"]), PREFLIGHT_TOL)
        add(f"alpha0_{typ}_reference_output", tensor_close_max(z["out0"], base["out0"]), PREFLIGHT_TOL)
        add(f"alpha0_{typ}_occurrence_flow", _max_list_array_diff(z["state"]["occurrence_flow"], base["state"]["occurrence_flow"]), PREFLIGHT_TOL)
        add(f"alpha0_{typ}_fieldA", float(np.max(np.abs(fieldA_input_origin(z["state"]) - fieldA_input_origin(base["state"])))), PREFLIGHT_TOL)

    # Expanded parameter-occurrence gates must be equivalent to scaling the corresponding shared weights.
    p_result = intervene_condition(model, "cnn", x, x0, target, sel["parameter_occurrence_masks"], base, alpha=EXP5_ALPHA)
    clone = copy.deepcopy(model).to(DEVICE).eval()
    with torch.no_grad():
        for conv, pmask in zip(clone.convs, sel["parameter_identity_masks"]):
            scale = torch.as_tensor(np.where(pmask, 1.0-EXP5_ALPHA, 1.0), dtype=conv.weight.dtype, device=conv.weight.device)
            conv.weight.mul_(scale)
    cout, _, _ = clone.forward_with_states(x.to(DEVICE))
    cout0, _, _ = clone.forward_with_states(x0.to(DEVICE))
    add("parameter_gate_equals_shared_weight_scaling_output", tensor_close_max(p_result["out"], cout), PREFLIGHT_TOL)
    add("parameter_gate_equals_shared_weight_scaling_reference", tensor_close_max(p_result["out0"], cout0), PREFLIGHT_TOL)
    cresp = cnn_response_sq(clone, x, x0)
    cstate = cnn_factor_state(clone, base["source_sq"], cresp, base["target_tensor"])
    add("parameter_gate_equals_shared_weight_scaling_occurrence_flow", _max_list_array_diff(p_result["state"]["occurrence_flow"], cstate["occurrence_flow"]), PREFLIGHT_TOL)
    add("parameter_gate_equals_shared_weight_scaling_parameter_flow", _max_list_array_diff(p_result["state"]["parameter_flow"], cstate["parameter_flow"]), PREFLIGHT_TOL)

    A = fieldA_input_origin(base["state"])
    add("fieldA_probability_mass_sums_to_one", abs(float(A.sum()) - 1.0), PREFLIGHT_TOL)
    E0 = np.asarray(base["state"]["occurrence_flow"][0][0], float)
    Bocc, Cocc = first_layer_source_footprint(E0, sel["occurrence_masks"][0], A.shape)
    Bpar, Cpar = first_layer_source_footprint(E0, sel["parameter_occurrence_masks"][0], A.shape)
    add("fieldB_occurrence_sum_equals_selected_first_layer_mass", abs(float(Bocc.sum()) - float(E0[sel["occurrence_masks"][0]].sum())), PREFLIGHT_TOL)
    add("fieldB_parameter_sum_equals_selected_first_layer_mass", abs(float(Bpar.sum()) - float(E0[sel["parameter_occurrence_masks"][0]].sum())), PREFLIGHT_TOL)
    add("fieldB_occurrence_count_equals_budget", abs(int(Cocc.sum()) - int(sel["occurrence_masks"][0].sum())), 0.0)
    add("fieldB_parameter_count_equals_budget", abs(int(Cpar.sum()) - int(sel["parameter_occurrence_masks"][0].sum())), 0.0)

    df = pd.DataFrame(rows)
    bad = df[~df.passed.astype(bool)]
    atomic_csv(df, OUTDIR / "tables" / "exp5_operator_selftest.csv")
    if len(bad):
        raise RuntimeError(f"Experiment 5 operator self-test failed {len(bad)} check(s).")
    print(f"[exp5 selftest] all {len(df)} occurrence/shared-identity and field checks passed.")
    return df


# ==================================================================================================
# 20. REAL-CHECKPOINT EXPERIMENT 5 PREFLIGHT
# ==================================================================================================


@torch.no_grad()
def run_exp5_real_preflight(manifest: pd.DataFrame, data_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    preferred = manifest[
        (manifest.arch_id.astype(str) == "C2_bottleneck")
        & (manifest.optimizer.astype(str) == "adam")
        & (manifest.activation.astype(str) == "relu")
        & (pd.to_numeric(manifest.seed_index, errors="coerce") == 0)
    ].copy()
    if preferred.empty:
        preferred = manifest.groupby("dataset", dropna=False).head(1).copy()
    for _, row in preferred.iterrows():
        rid = str(row.run_id); ds = str(row.dataset); cp = checkpoint_path(rid)
        before = file_sha1(cp)
        model = build_model(str(row.arch_id), str(row.activation), output_dim_for(ds), ds, int(row.seed_index))
        model.load_state_dict(torch.load(cp, map_location="cpu")["state_dict"]); model.to(DEVICE).eval()
        for environment in ENVIRONMENTS:
            X, Y = prepared_exp5_environment(data_cache[ds], ds, environment)
            x, y = X[0:1], Y[0:1]; x0 = torch.zeros_like(x)
            target = int(y.reshape(-1)[0].item()) if str(row.task) == "classification" else 0
            base = baseline_condition(model, "cnn", x, x0, target)
            if not bool(base["state"]["valid"][0]):
                raise RuntimeError(f"Experiment 5 real preflight baseline undefined: {rid} {environment}")
            sel = select_occurrence_vs_parameter_masks(base["state"], EXP5_TARGET_FRACTION)
            max_budget_error = max(abs(a["occurrence_selected_count"] - a["parameter_affected_occurrence_count"]) for a in sel["audit"])
            min_mass_adv = min(a["selected_flow_mass_advantage_occ_minus_param"] for a in sel["audit"])
            occ0 = intervene_condition(model, "cnn", x, x0, target, sel["occurrence_masks"], base, alpha=0.0)
            par0 = intervene_condition(model, "cnn", x, x0, target, sel["parameter_occurrence_masks"], base, alpha=0.0)
            occ = intervene_condition(model, "cnn", x, x0, target, sel["occurrence_masks"], base, alpha=EXP5_ALPHA)
            par = intervene_condition(model, "cnn", x, x0, target, sel["parameter_occurrence_masks"], base, alpha=EXP5_ALPHA)
            A = fieldA_input_origin(base["state"])
            E0 = np.asarray(base["state"]["occurrence_flow"][0][0], float)
            Bo, Co = first_layer_source_footprint(E0, sel["occurrence_masks"][0], A.shape)
            Bp, Cp = first_layer_source_footprint(E0, sel["parameter_occurrence_masks"][0], A.shape)
            checks = {
                "equal_budget_error": float(max_budget_error),
                "occurrence_mass_below_parameter_violation": float(max(0.0, -min_mass_adv)),
                "alpha0_occ_output_error": tensor_close_max(occ0["out"], base["out"]),
                "alpha0_param_output_error": tensor_close_max(par0["out"], base["out"]),
                "fieldA_sum_error": abs(float(A.sum()) - 1.0),
                "fieldB_occ_mass_error": abs(float(Bo.sum()) - float(E0[sel["occurrence_masks"][0]].sum())),
                "fieldB_param_mass_error": abs(float(Bp.sum()) - float(E0[sel["parameter_occurrence_masks"][0]].sum())),
                "fieldB_occ_count_error": float(abs(int(Co.sum()) - int(sel["occurrence_masks"][0].sum()))),
                "fieldB_param_count_error": float(abs(int(Cp.sum()) - int(sel["parameter_occurrence_masks"][0].sum()))),
                "occ_intervened_invalid": 0.0 if bool(occ["valid"]) else 1.0,
                "param_intervened_invalid": 0.0 if bool(par["valid"]) else 1.0,
            }
            for name, val in checks.items():
                tol = 0.0 if ("budget" in name or "count_error" in name or "invalid" in name) else PREFLIGHT_TOL
                rows.append({
                    "run_id": rid, "dataset": ds, "environment": environment,
                    "check": name, "value": float(val), "tolerance": float(tol), "passed": bool(val <= tol)
                })
        after = file_sha1(cp)
        rows.append({"run_id": rid, "dataset": ds, "environment": "all", "check": "checkpoint_sha1_immutable",
                     "value": 0.0 if before == after else 1.0, "tolerance": 0.0, "passed": bool(before == after)})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    df = pd.DataFrame(rows)
    bad = df[~df.passed.astype(bool)]
    atomic_csv(df, OUTDIR / "tables" / "exp5_real_checkpoint_preflight.csv")
    if len(bad):
        raise RuntimeError(f"Experiment 5 real-checkpoint preflight failed {len(bad)} check(s).")
    print(f"[exp5 preflight] all {len(df)} real-checkpoint checks passed.")
    return df


# ==================================================================================================
# 21. ONE CONDITION: TWO EQUAL-BUDGET INTERVENTIONS + FIELD A/B
# ==================================================================================================


def _common_condition_meta(row: pd.Series, environment: str, ci: int, target_idx: int) -> Dict[str, Any]:
    return {
        "run_id": str(row.run_id), "dataset": str(row.dataset), "task": str(row.task), "family": "cnn",
        "arch_id": str(row.arch_id), "optimizer": str(row.optimizer), "activation": str(row.activation),
        "seed_index": int(row.seed_index), "environment": str(environment), "condition_index": int(ci),
        "target_output": int(target_idx), "alpha": float(EXP5_ALPHA),
        "parameter_target_fraction": float(EXP5_TARGET_FRACTION),
    }


def _nan_field(shape=(IMAGE_RESOLUTION, IMAGE_RESOLUTION)):
    return np.full(shape, np.nan, dtype=np.float64)


@torch.no_grad()
def run_exp5_condition(model: DeepCNN, row: pd.Series, data: Dict[str, Any], environment: str,
                       ci: int, x: torch.Tensor, y: torch.Tensor):
    rid = str(row.run_id)
    paths = exp5_condition_paths(rid, environment, ci)
    if exp5_condition_complete(rid, environment, ci):
        return

    x0 = torch.zeros_like(x)
    target_idx = int(y.reshape(-1)[0].item()) if str(row.task) == "classification" else 0
    common = _common_condition_meta(row, environment, ci, target_idx)
    raw_image, display_image = image_for_storage_and_display(x, data["meta"])
    base = baseline_condition(model, "cnn", x, x0, target_idx)
    base_valid = bool(base["state"]["valid"][0])

    validity_rows = [{**common, "state": "baseline", "intervention_type": "none",
                      "baseline_realization_defined": base_valid, "intervention_applicable": False,
                      "intervened_realization_defined": base_valid, "intervention_induced_degeneracy": False,
                      "status": "defined" if base_valid else "baseline_zero_complete_path_support"}]
    measurement_rows = []
    target_rows = []

    if not base_valid:
        for typ in INTERVENTION_TYPES:
            validity_rows.append({**common, "state": typ, "intervention_type": typ,
                                  "baseline_realization_defined": False, "intervention_applicable": False,
                                  "intervened_realization_defined": False, "intervention_induced_degeneracy": False,
                                  "status": "not_applicable_baseline_zero_support"})
            measurement_rows.append({**common, "intervention_type": typ, "path_hellinger": np.nan,
                                     "manifestation_abs": np.nan, "selected_flow_mass_total": np.nan})
        A = _nan_field(); Aocc = _nan_field(); Aparam = _nan_field(); Bocc = _nan_field(); Bparam = _nan_field()
        Cocc = np.zeros_like(A, dtype=np.int64); Cparam = np.zeros_like(A, dtype=np.int64)
        field_diag = pd.DataFrame([{**common, "fieldA_defined": False}])
        # The audit table must be non-empty for write-once completion semantics.
        target_rows.append({**common, "layer": -1, "status": "not_applicable_baseline_zero_support"})
    else:
        sel = select_occurrence_vs_parameter_masks(base["state"], EXP5_TARGET_FRACTION)
        A = fieldA_input_origin(base["state"])
        E0 = np.asarray(base["state"]["occurrence_flow"][0][0], float)
        Bocc, Cocc = first_layer_source_footprint(E0, sel["occurrence_masks"][0], A.shape)
        Bparam, Cparam = first_layer_source_footprint(E0, sel["parameter_occurrence_masks"][0], A.shape)

        for a in sel["audit"]:
            target_rows.append({**common, **a})

        results = {}
        mask_map = {"occurrence": sel["occurrence_masks"], "parameter": sel["parameter_occurrence_masks"]}
        total_occ_mass = float(sum(a["occurrence_selected_flow_mass"] for a in sel["audit"]))
        total_param_mass = float(sum(a["parameter_selected_flow_mass"] for a in sel["audit"]))
        total_budget = int(sum(a["primitive_occurrence_budget"] for a in sel["audit"]))
        for typ in INTERVENTION_TYPES:
            result = intervene_condition(model, "cnn", x, x0, target_idx, mask_map[typ], base, alpha=EXP5_ALPHA)
            results[typ] = result
            valid_i = bool(result["valid"])
            validity_rows.append({**common, "state": typ, "intervention_type": typ,
                                  "baseline_realization_defined": True, "intervention_applicable": True,
                                  "intervened_realization_defined": valid_i,
                                  "intervention_induced_degeneracy": bool(not valid_i),
                                  "status": "defined" if valid_i else "intervened_zero_complete_path_support"})
            mrec = manifestation_record(str(row.dataset), y, target_idx, base["out"], base["out0"], result["out"], result["out0"])
            rec = {**common, "intervention_type": typ,
                   "path_hellinger": float(result["hellinger"]) if np.isfinite(result["hellinger"]) else np.nan,
                   **mrec,
                   "primitive_occurrence_budget_total": total_budget,
                   "selected_flow_mass_total": total_occ_mass if typ == "occurrence" else total_param_mass,
                   "selected_flow_mass_first_layer": float(Bocc.sum()) if typ == "occurrence" else float(Bparam.sum())}
            if valid_i:
                rec.update({
                    "eta1": float(result["state"]["eta1"][0]), "eta2": float(result["state"]["eta2"][0]),
                    "Sdef": float(result["state"]["entropy_deficit"][0]),
                    "Sdef_norm": float(result["state"]["entropy_deficit_norm"][0]),
                    "delta_eta1": float(result["state"]["eta1"][0] - base["state"]["eta1"][0]),
                    "delta_eta2": float(result["state"]["eta2"][0] - base["state"]["eta2"][0]),
                    "delta_Sdef": float(result["state"]["entropy_deficit"][0] - base["state"]["entropy_deficit"][0]),
                    "delta_Sdef_norm": float(result["state"]["entropy_deficit_norm"][0] - base["state"]["entropy_deficit_norm"][0]),
                })
            measurement_rows.append(rec)

        Aocc = fieldA_input_origin(results["occurrence"]["state"]) if bool(results["occurrence"]["valid"]) else _nan_field(A.shape)
        Aparam = fieldA_input_origin(results["parameter"]["state"]) if bool(results["parameter"]["valid"]) else _nan_field(A.shape)
        field_diag = pd.DataFrame([{
            **common, "fieldA_defined": True,
            "fieldA_sum": float(A.sum()), "fieldA_max": float(np.max(A)),
            "fieldA_entropy": field_entropy(A), "fieldA_top_fraction": float(EXP5_FIELD_TOP_FRACTION),
            "fieldA_top_fraction_mass": top_fraction_mass(A, EXP5_FIELD_TOP_FRACTION),
            "fieldB_occurrence_flow_mass": float(Bocc.sum()), "fieldB_parameter_flow_mass": float(Bparam.sum()),
            "fieldB_flow_mass_advantage_occ_minus_param": float(Bocc.sum() - Bparam.sum()),
            "fieldB_occurrence_primitive_count": int(Cocc.sum()), "fieldB_parameter_primitive_count": int(Cparam.sum()),
            "fieldB_occurrence_input_pixel_support": int(np.sum(Cocc > 0)),
            "fieldB_parameter_input_pixel_support": int(np.sum(Cparam > 0)),
            "fieldA_mass_on_occurrence_footprint": field_support_mass(A, Cocc),
            "fieldA_mass_on_parameter_footprint": field_support_mass(A, Cparam),
            "fieldA_footprint_support_mass_occ_minus_param": float(field_support_mass(A, Cocc) - field_support_mass(A, Cparam)),
            "fieldB_normalized_l1_occ_vs_param": field_l1_normalized(Bocc, Bparam),
            "fieldA_post_occurrence_l1": float(np.nansum(np.abs(Aocc - A))),
            "fieldA_post_parameter_l1": float(np.nansum(np.abs(Aparam - A))),
            "fieldA_post_l1_occ_minus_param": float(np.nansum(np.abs(Aocc - A)) - np.nansum(np.abs(Aparam - A))),
        }])

        render_ids = _render_condition_indices(EXP5_CONDITION_COUNT)
        if ci in render_ids:
            md = pd.DataFrame(measurement_rows).set_index("intervention_type")
            meta_text = f"{rid} | {environment} | condition {ci} | target {target_idx}"
            render_fieldA_fieldB_figure(
                paths["figure"], display_image, A, Bocc, Bparam, meta_text,
                float(md.loc["occurrence", "manifestation_abs"]), float(md.loc["parameter", "manifestation_abs"]),
                float(md.loc["occurrence", "path_hellinger"]), float(md.loc["parameter", "path_hellinger"]),
            )
            render_post_fieldA_figure(paths["post_figure"], display_image, A, Aocc, Aparam, meta_text)

    # Persist all numerical evidence atomically and write-once.  NPZ stores raw fields for every condition.
    if not paths["measurements"].exists(): atomic_csv(pd.DataFrame(measurement_rows), paths["measurements"])
    if not paths["validity"].exists(): atomic_csv(pd.DataFrame(validity_rows), paths["validity"])
    if not paths["target_audit"].exists(): atomic_csv(pd.DataFrame(target_rows), paths["target_audit"])
    if not paths["field_diagnostics"].exists(): atomic_csv(field_diag, paths["field_diagnostics"])
    if not paths["fields"].exists():
        write_once_npz(
            paths["fields"],
            image_raw=np.asarray(raw_image), image_display=np.asarray(display_image),
            fieldA_baseline=np.asarray(A, dtype=np.float64),
            fieldA_post_occurrence=np.asarray(Aocc, dtype=np.float64),
            fieldA_post_parameter=np.asarray(Aparam, dtype=np.float64),
            fieldA_delta_occurrence=np.asarray(Aocc-A, dtype=np.float64),
            fieldA_delta_parameter=np.asarray(Aparam-A, dtype=np.float64),
            fieldB_occurrence=np.asarray(Bocc, dtype=np.float64),
            fieldB_parameter=np.asarray(Bparam, dtype=np.float64),
            fieldB_difference=np.asarray(Bocc-Bparam, dtype=np.float64),
            fieldB_occurrence_count=np.asarray(Cocc, dtype=np.int64),
            fieldB_parameter_count=np.asarray(Cparam, dtype=np.int64),
            target_output=np.asarray([target_idx], dtype=np.int64),
            alpha=np.asarray([EXP5_ALPHA], dtype=np.float64),
        )
    # Publish a condition completion marker LAST.  If a Slurm job dies between publications, the next
    # allocation fills only missing files and never erases or overwrites already-published evidence.
    if not paths["done"].exists():
        atomic_json({
            "run_id":rid,"environment":environment,"condition_index":int(ci),"completed":True,
            "artifacts":{k:str(v) for k,v in paths.items() if k not in {"figure","post_figure","done"}},
            "timestamp_unix":time.time(),"version":EXP5_VERSION,
        }, paths["done"])


# ==================================================================================================
# 22. MODEL LOOP, RESUME, CHECKPOINT IMMUTABILITY
# ==================================================================================================


@torch.no_grad()
def run_exp5_one_model(row: pd.Series, data: Dict[str, Any]) -> bool:
    rid = str(row.run_id)
    if RESUME and exp5_done_path(rid).exists():
        return True
    cp = checkpoint_path(rid); hash_before = file_sha1(cp)
    model = build_model(str(row.arch_id), str(row.activation), output_dim_for(str(row.dataset)), str(row.dataset), int(row.seed_index))
    model.load_state_dict(torch.load(cp, map_location="cpu")["state_dict"]); model.to(DEVICE).eval()
    if not isinstance(model, DeepCNN):
        raise RuntimeError(f"Experiment 5 received non-CNN model {rid}.")

    for environment in ENVIRONMENTS:
        X, Y = prepared_exp5_environment(data, str(row.dataset), environment)
        for ci in range(len(X)):
            if exp5_condition_complete(rid, environment, ci):
                continue
            if not can_start_new_unit("EXP5 condition", extra_buffer_min=EXP5_BUFFER_MIN):
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
                return False
            run_exp5_condition(model, row, data, environment, ci, X[ci:ci+1], Y[ci:ci+1])

    hash_after = file_sha1(cp)
    if hash_before != hash_after:
        raise RuntimeError(f"SOURCE CHECKPOINT CHANGED DURING EXPERIMENT 5: {cp}")
    for environment in ENVIRONMENTS:
        for ci in range(EXP5_CONDITION_COUNT):
            if not exp5_condition_complete(rid, environment, ci):
                raise RuntimeError(f"Cannot mark model complete; missing condition part {rid} {environment} c={ci}")
    atomic_json({
        "run_id": rid, "completed": True, "checkpoint_sha1": hash_before,
        "conditions": int(EXP5_CONDITION_COUNT), "environments": list(ENVIRONMENTS),
        "intervention_types": list(INTERVENTION_TYPES), "alpha": float(EXP5_ALPHA),
        "timestamp_unix": time.time(), "version": EXP5_VERSION,
    }, exp5_done_path(rid))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return True


# ==================================================================================================
# 23. AGGREGATION, PAIRED RQ7 EFFECTS, STATISTICS
# ==================================================================================================


def _iter_exp5_part_paths(manifest: pd.DataFrame, kind: str):
    for rid in manifest.run_id.astype(str):
        for environment in ENVIRONMENTS:
            for ci in range(EXP5_CONDITION_COUNT):
                p = exp5_condition_paths(rid, environment, ci)[kind]
                if p.exists() and p.stat().st_size > 0:
                    yield p


def _rss_mb() -> float:
    """Current Linux resident-set size for aggregation diagnostics; NaN elsewhere."""
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float('nan')


def _stream_concat_exp5_parts(manifest: pd.DataFrame, kind: str, destination: Path) -> Dict[str, Any]:
    """Two-pass, schema-union CSV concatenation with bounded memory.

    Pass 1 reads headers only so sparse/degenerate condition files with reduced schemas remain valid.
    Pass 2 streams rows directly to a temporary aggregate CSV.  No list of thousands of pandas
    DataFrames and no giant in-memory CSV byte string is ever constructed.

    If a prior failed v1 aggregation already published a destination CSV, it is reused read-only.
    v1 publication was atomic/write-once, so an existing file is complete rather than partially written.
    """
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        row_count = 0
        columns = []
        with open(destination, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                columns = next(reader)
            except StopIteration:
                columns = []
            for _ in reader:
                row_count += 1
        print(f"[aggregate:{kind}] reusing existing atomic aggregate | rows={row_count} | RSS={_rss_mb():.1f} MiB", flush=True)
        return {"path": str(destination), "row_count": int(row_count), "source_files": -1, "columns": columns, "reused_existing": True}

    paths = list(_iter_exp5_part_paths(manifest, kind))
    if not paths:
        return {"path": str(destination), "row_count": 0, "source_files": 0, "columns": []}

    fieldnames, seen = [], set()
    for p in paths:
        with open(p, 'r', newline='', encoding='utf-8') as f:
            r = csv.reader(f)
            try:
                hdr = next(r)
            except StopIteration:
                continue
        for c in hdr:
            if c not in seen:
                seen.add(c); fieldnames.append(c)
    if not fieldnames:
        return {"path": str(destination), "row_count": 0, "source_files": len(paths), "columns": []}

    destination = Path(destination)
    tmp = destination.parent / f".{destination.name}.streamtmp.{os.getpid()}.{time.time_ns()}"
    row_count = 0
    try:
        with open(tmp, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for i, p in enumerate(paths, 1):
                with open(p, 'r', newline='', encoding='utf-8') as in_f:
                    reader = csv.DictReader(in_f)
                    for row in reader:
                        writer.writerow({k: ('' if v is None else v) for k, v in row.items() if k is not None})
                        row_count += 1
                if i % 500 == 0:
                    out_f.flush()
                    print(f"[aggregate:{kind}] {i}/{len(paths)} files | rows={row_count} | RSS={_rss_mb():.1f} MiB", flush=True)
            out_f.flush(); os.fsync(out_f.fileno())
        _publish_file_once(tmp, destination)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass
    print(f"[aggregate:{kind}] complete | files={len(paths)} | rows={row_count} | RSS={_rss_mb():.1f} MiB", flush=True)
    return {"path": str(destination), "row_count": int(row_count), "source_files": int(len(paths)), "columns": fieldnames}


def _read_small_aggregate(meta: Dict[str, Any]) -> pd.DataFrame:
    p = Path(meta["path"])
    if int(meta.get("row_count", 0)) <= 0 or not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def aggregate_exp5_parts(manifest: pd.DataFrame) -> Dict[str, Any]:
    """Memory-bounded Experiment 5 aggregation.

    measurements, validity and field diagnostics are small enough to load after streaming consolidation.
    target_audit is intentionally left on disk and validated in chunks because it is not required by the
    paired RQ7 statistics.  This removes the v1 aggregation memory spike that caused native SIGSEGV on HPC.
    """
    out: Dict[str, Any] = {}
    metas = {}
    for kind in ["measurements", "validity", "target_audit", "field_diagnostics"]:
        print(f"[aggregate] streaming {kind} | RSS={_rss_mb():.1f} MiB", flush=True)
        meta = _stream_concat_exp5_parts(manifest, kind, OUTDIR / "tables" / f"{kind}.csv")
        metas[kind] = meta
        gc.collect()

    out["measurements"] = _read_small_aggregate(metas["measurements"])
    out["validity"] = _read_small_aggregate(metas["validity"])
    out["field_diagnostics"] = _read_small_aggregate(metas["field_diagnostics"])
    out["target_audit_meta"] = metas["target_audit"]
    out["aggregate_meta"] = metas
    print(f"[aggregate] small tables loaded | RSS={_rss_mb():.1f} MiB", flush=True)

    # Field index points to write-once raw NPZ artifacts without duplicating them.
    rows = []
    for rid in manifest.run_id.astype(str):
        for env in ENVIRONMENTS:
            for ci in range(EXP5_CONDITION_COUNT):
                p = exp5_condition_paths(rid, env, ci)["fields"]
                if p.exists():
                    rows.append({"run_id":rid,"environment":env,"condition_index":ci,"field_npz":str(p),"bytes":p.stat().st_size})
    out["field_index"] = pd.DataFrame(rows)
    if len(out["field_index"]):
        atomic_csv(out["field_index"], OUTDIR / "tables" / "field_index.csv")
    return out




def _bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true","1","yes","y"})

def validate_exp5_tables(tabs: Dict[str, Any], manifest: pd.DataFrame):
    m = tabs.get("measurements", pd.DataFrame()); v = tabs.get("validity", pd.DataFrame())
    ta_meta = tabs.get("target_audit_meta", {})
    if m.empty or v.empty or int(ta_meta.get("row_count", 0)) <= 0:
        raise RuntimeError("Experiment 5 aggregate tables are empty/incomplete.")
    expected_conditions = len(manifest) * len(ENVIRONMENTS) * EXP5_CONDITION_COUNT
    expected_measurements = expected_conditions * len(INTERVENTION_TYPES)
    expected_validity = expected_conditions * (1 + len(INTERVENTION_TYPES))
    if len(m) != expected_measurements:
        raise RuntimeError(f"Expected {expected_measurements} measurement rows, found {len(m)}")
    if len(v) != expected_validity:
        raise RuntimeError(f"Expected {expected_validity} validity rows, found {len(v)}")
    k = ["run_id","environment","condition_index","intervention_type"]
    if m.duplicated(k, keep=False).any():
        raise RuntimeError("Duplicate Experiment 5 measurement keys.")
    kv = ["run_id","environment","condition_index","state"]
    if v.duplicated(kv, keep=False).any():
        raise RuntimeError("Duplicate Experiment 5 validity keys.")
    # All equal-budget audits must pass whenever applicable.  Validate the potentially larger audit
    # table in bounded chunks instead of materializing it in RAM.
    ta_path = Path(ta_meta["path"])
    applicable_rows = 0
    for chunk in pd.read_csv(ta_path, chunksize=4096, low_memory=False):
        if "layer" not in chunk.columns:
            continue
        aa = chunk[pd.to_numeric(chunk["layer"], errors="coerce") >= 0]
        if not len(aa):
            continue
        applicable_rows += len(aa)
        if "equal_budget" not in aa.columns or not _bool_series(aa["equal_budget"]).all():
            raise RuntimeError("Experiment 5 target audit contains unequal primitive budgets.")
        if "parameter_fibre_closed" not in aa.columns or not _bool_series(aa["parameter_fibre_closed"]).all():
            raise RuntimeError("Parameter intervention is not closed over sharing fibres.")
        if "selected_flow_mass_advantage_occ_minus_param" not in aa.columns:
            raise RuntimeError("Experiment 5 target audit lacks selected-flow-mass advantage column.")
        adv = pd.to_numeric(aa["selected_flow_mass_advantage_occ_minus_param"], errors="coerce")
        if (adv < -1e-10).any():
            raise RuntimeError("Occurrence top-B selected mass fell below parameter-arm mass.")
    if applicable_rows <= 0:
        raise RuntimeError("Experiment 5 target audit contains no applicable layer rows.")
    print(f"[aggregate] target audit validated in chunks | applicable_rows={applicable_rows} | RSS={_rss_mb():.1f} MiB", flush=True)


def build_condition_paired_effects(measurements: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_id","dataset","task","family","arch_id","optimizer","activation","seed_index","environment","condition_index","target_output"]
    metrics = ["manifestation_abs","path_hellinger","selected_flow_mass_total","selected_flow_mass_first_layer",
               "task_loss_change","finite_response_vector_l2","delta_eta1","delta_eta2","delta_Sdef","delta_Sdef_norm"]
    occ = measurements[measurements.intervention_type.astype(str)=="occurrence"].copy()
    par = measurements[measurements.intervention_type.astype(str)=="parameter"].copy()
    keep = keys + [c for c in metrics if c in occ.columns]
    oo = occ[keep].copy(); pp = par[keep].copy()
    merged = oo.merge(pp, on=keys, suffixes=("_occ","_param"), how="inner")
    rows = []
    for _, r in merged.iterrows():
        rec = {k:r[k] for k in keys}
        for metric in metrics:
            oc, pc = metric+"_occ", metric+"_param"
            if oc in merged.columns and pc in merged.columns:
                ov = pd.to_numeric(pd.Series([r[oc]]), errors="coerce").iloc[0]
                pv = pd.to_numeric(pd.Series([r[pc]]), errors="coerce").iloc[0]
                rec[f"{metric}_occ"] = ov; rec[f"{metric}_param"] = pv
                rec[f"{metric}_occ_minus_param"] = float(ov-pv) if np.isfinite(ov) and np.isfinite(pv) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(fields):
        fkeys = ["run_id","dataset","arch_id","optimizer","activation","seed_index","environment","condition_index","target_output"]
        cols = [c for c in fields.columns if c not in set(keys) or c in fkeys]
        ff = fields[[c for c in fkeys if c in fields.columns] + [c for c in fields.columns if c not in set(fkeys) and c not in set(keys)]].copy()
        merge_keys = [k for k in fkeys if k in out.columns and k in ff.columns]
        out = out.merge(ff, on=merge_keys, how="left")
    atomic_csv(out, OUTDIR / "tables" / "condition_paired_effects.csv")
    return out


def build_model_level_effects(cpe: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["run_id","dataset","task","family","arch_id","optimizer","activation","seed_index","environment"]
    effect_cols = [
        "manifestation_abs_occ_minus_param", "path_hellinger_occ_minus_param",
        "selected_flow_mass_total_occ_minus_param", "selected_flow_mass_first_layer_occ_minus_param",
        "task_loss_change_occ_minus_param", "finite_response_vector_l2_occ_minus_param",
        "delta_eta1_occ_minus_param", "delta_eta2_occ_minus_param", "delta_Sdef_occ_minus_param", "delta_Sdef_norm_occ_minus_param",
        "fieldB_flow_mass_advantage_occ_minus_param", "fieldA_footprint_support_mass_occ_minus_param",
        "fieldA_post_l1_occ_minus_param", "fieldB_normalized_l1_occ_vs_param",
    ]
    effect_cols = [c for c in effect_cols if c in cpe.columns]
    rows = []
    for keys, sub in cpe.groupby(group_cols, dropna=False):
        rec = {k:v for k,v in zip(group_cols, keys)}
        rec["n_conditions"] = int(len(sub))
        for c in effect_cols:
            x = pd.to_numeric(sub[c], errors="coerce").to_numpy(float); x = x[np.isfinite(x)]
            rec[f"median_{c}"] = float(np.median(x)) if len(x) else np.nan
            rec[f"positive_condition_fraction_{c}"] = float(np.mean(x>0)) if len(x) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    atomic_csv(out, OUTDIR / "tables" / "model_level_effects.csv")
    return out


def _one_sample_test(d: np.ndarray, seed_offset: int = 0) -> Dict[str, Any]:
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n":0,"median":np.nan,"hl":np.nan,"rank_biserial":np.nan,"positive_fraction":np.nan,"p":np.nan,"ci_low":np.nan,"ci_high":np.nan}
    try:
        p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        p = np.nan
    lo, hi = bootstrap_median_ci(d, seed=SEED + seed_offset, B=2000)
    return {"n":int(len(d)),"median":float(np.median(d)),"hl":hodges_lehmann_onesample(d),
            "rank_biserial":paired_rank_biserial(d),"positive_fraction":float(np.mean(d>0)),"p":p,"ci_low":lo,"ci_high":hi}


def rq7_primary_tests(model_effects: pd.DataFrame) -> pd.DataFrame:
    # The two predeclared RQ7 outcomes form the primary correction family: functional manifestation and full-path Hellinger.
    metrics = [
        ("manifestation", "median_manifestation_abs_occ_minus_param"),
        ("path_hellinger", "median_path_hellinger_occ_minus_param"),
    ]
    rows = []
    for ds in sorted(model_effects.dataset.unique()):
        for env in sorted(model_effects.environment.unique()):
            sub = model_effects[(model_effects.dataset==ds)&(model_effects.environment==env)]
            for label, col in metrics:
                if col not in sub.columns: continue
                rows.append({"dataset":ds,"environment":env,"outcome":label,"contrast":"occurrence_minus_parameter",
                             **_one_sample_test(pd.to_numeric(sub[col], errors="coerce").to_numpy(float), seed_offset=len(rows)+1)})
    out = pd.DataFrame(rows)
    if len(out): out["p_holm_primary_family"] = holm_adjust(pd.to_numeric(out.p, errors="coerce").to_numpy(float))
    atomic_csv(out, OUTDIR / "tables" / "RQ7_primary_tests.csv")
    return out


def rq7_stratified_tests(model_effects: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        ("manifestation", "median_manifestation_abs_occ_minus_param"),
        ("path_hellinger", "median_path_hellinger_occ_minus_param"),
    ]
    strata = [("activation",["activation"]),("architecture",["arch_id"]),("optimizer",["optimizer"]),("architecture_activation",["arch_id","activation"])]
    rows = []
    for ds in sorted(model_effects.dataset.unique()):
        for env in sorted(model_effects.environment.unique()):
            de = model_effects[(model_effects.dataset==ds)&(model_effects.environment==env)]
            for outcome, col in outcomes:
                if col not in de.columns: continue
                for sname, cols in strata:
                    for key, sub in de.groupby(cols, dropna=False):
                        if not isinstance(key, tuple): key=(key,)
                        lab = {c:str(v) for c,v in zip(cols,key)}
                        rows.append({"dataset":ds,"environment":env,"outcome":outcome,"stratum":sname,**lab,
                                     "contrast":"occurrence_minus_parameter",
                                     **_one_sample_test(pd.to_numeric(sub[col],errors="coerce").to_numpy(float), seed_offset=len(rows)+101)})
    out = pd.DataFrame(rows)
    if len(out): out["p_holm_stratified_family"] = holm_adjust(pd.to_numeric(out.p, errors="coerce").to_numpy(float))
    atomic_csv(out, OUTDIR / "tables" / "RQ7_stratified_tests.csv")
    return out


def rq7_mechanism_tests(model_effects: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("captured_realized_flow_all_layers", "median_selected_flow_mass_total_occ_minus_param"),
        ("captured_realized_flow_first_layer", "median_selected_flow_mass_first_layer_occ_minus_param"),
        ("fieldB_flow_mass", "median_fieldB_flow_mass_advantage_occ_minus_param"),
        ("fieldA_mass_on_footprint", "median_fieldA_footprint_support_mass_occ_minus_param"),
        ("post_fieldA_l1", "median_fieldA_post_l1_occ_minus_param"),
    ]
    rows = []
    for ds in sorted(model_effects.dataset.unique()):
        for env in sorted(model_effects.environment.unique()):
            sub = model_effects[(model_effects.dataset==ds)&(model_effects.environment==env)]
            for label, col in candidates:
                if col not in sub.columns: continue
                rows.append({"dataset":ds,"environment":env,"diagnostic":label,"contrast":"occurrence_minus_parameter",
                             **_one_sample_test(pd.to_numeric(sub[col],errors="coerce").to_numpy(float), seed_offset=len(rows)+401)})
    out = pd.DataFrame(rows)
    if len(out): out["p_holm_mechanism_family"] = holm_adjust(pd.to_numeric(out.p, errors="coerce").to_numpy(float))
    atomic_csv(out, OUTDIR / "tables" / "RQ7_mechanism_tests.csv")
    return out


def rq7_information_consequence_associations(model_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    xcols = [
        ("captured_flow_advantage", "median_selected_flow_mass_total_occ_minus_param"),
        ("first_layer_flow_advantage", "median_selected_flow_mass_first_layer_occ_minus_param"),
        ("fieldA_footprint_support_advantage", "median_fieldA_footprint_support_mass_occ_minus_param"),
    ]
    ycols = [
        ("manifestation_advantage", "median_manifestation_abs_occ_minus_param"),
        ("hellinger_advantage", "median_path_hellinger_occ_minus_param"),
    ]
    for ds in sorted(model_effects.dataset.unique()):
        for env in sorted(model_effects.environment.unique()):
            sub = model_effects[(model_effects.dataset==ds)&(model_effects.environment==env)]
            for xl, xc in xcols:
                for yl, yc in ycols:
                    if xc not in sub.columns or yc not in sub.columns: continue
                    x = pd.to_numeric(sub[xc], errors="coerce").to_numpy(float)
                    y = pd.to_numeric(sub[yc], errors="coerce").to_numpy(float)
                    good = np.isfinite(x)&np.isfinite(y)
                    if good.sum() >= 8 and np.nanstd(x[good])>0 and np.nanstd(y[good])>0:
                        sr = spearmanr(x[good], y[good])
                        rho, p = float(sr.statistic), float(sr.pvalue)
                    else:
                        rho, p = np.nan, np.nan
                    rows.append({"dataset":ds,"environment":env,"information_diagnostic":xl,"causal_outcome":yl,
                                 "n":int(good.sum()),"spearman_rho":rho,"p":p})
    out = pd.DataFrame(rows)
    if len(out): out["p_holm_association_family"] = holm_adjust(pd.to_numeric(out.p,errors="coerce").to_numpy(float))
    atomic_csv(out, OUTDIR / "tables" / "RQ7_information_consequence_associations.csv")
    return out


def validity_summary(validity: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for keys, sub in validity.groupby(["dataset","environment","state"], dropna=False):
        rows.append({"dataset":keys[0],"environment":keys[1],"state":keys[2],"n":len(sub),
                     "baseline_defined_fraction":float(_bool_series(sub.baseline_realization_defined).mean()),
                     "post_defined_fraction":float(_bool_series(sub.intervened_realization_defined).mean()),
                     "induced_degeneracy_fraction":float(_bool_series(sub.intervention_induced_degeneracy).mean())})
    out=pd.DataFrame(rows); atomic_csv(out,OUTDIR/"tables"/"validity_summary.csv"); return out


# ==================================================================================================
# 24. MANIFEST AND FIGURE INDEX
# ==================================================================================================


def exp5_design_object(manifest: pd.DataFrame, population: pd.DataFrame) -> Dict[str, Any]:
    design = {
        "version": EXP5_VERSION,
        "study": "Work II Experiment 5 CNN occurrence/shared-parameter identity comparison",
        "RQ7": "For CNNs, is spatial use more informative than shared parameter identity?",
        "exp1_results": str(EXP1_RESULTS), "source_checkpoint_dir": str(SOURCE_CHECKPOINT_DIR), "output_dir": str(OUTDIR),
        "datasets": list(DATASETS), "cnn_architectures": sorted(manifest.arch_id.astype(str).unique().tolist()),
        "optimizers": list(OPTIMIZERS), "activations": list(ACTIVATIONS), "seeds": list(SEEDS),
        "trained_cnn_checkpoints_declared": int(len(manifest)), "conditions_per_model": int(EXP5_CONDITION_COUNT),
        "environments": list(ENVIRONMENTS), "condition_definition": "c=(x,0)",
        "attenuation_alpha": float(EXP5_ALPHA), "gate_on_selected": float(1.0-EXP5_ALPHA),
        "parameter_target_fraction_of_positive_identities": float(EXP5_TARGET_FRACTION),
        "equal_budget_rule": "Within each convolutional layer, occurrence and parameter arms attenuate exactly the same number of primitive spatial execution occurrences.",
        "occurrence_selection_rule": "Top-B conditional occurrence-flow uses within the spatial reuse universe of positive-flow parameter identities.",
        "parameter_selection_rule": "Top conditional parameter-flow identities; attenuate every spatial occurrence in each selected sharing fibre.",
        "declared_baseline_conditions": int(len(manifest)*len(ENVIRONMENTS)*EXP5_CONDITION_COUNT),
        "declared_nonzero_interventions": int(len(manifest)*len(ENVIRONMENTS)*EXP5_CONDITION_COUNT*2),
        "primary_functional_outcome": "D_occ - D_param, where D is finite-response manifestation displacement",
        "primary_internal_outcome": "D_H_occ - D_H_param, complete-path Hellinger displacement",
        "mechanism_diagnostic": "equal-budget realized-flow mass captured by occurrence versus parameter selection",
        "field_A": "P_in^c(s)=sum over input channels of established input-node flow; input-origin marginal of normalized complete conditional path measure",
        "field_B": "first-convolution source-coordinate footprint of selected baseline occurrence-flow mass; occurrence and shared-identity arms",
        "field_B_scope_guard": "Causal intervention acts across all CNN layers. Direct raw-image Field B uses first convolution only because its source indices map exactly to the input lattice; no invented deep receptive-field projection.",
        "interpretability_guard": "Field A is not gradient saliency; it is a marginal of the declared conditional path measure. No claim that it is a universal true saliency map.",
        "geometry": "BCANw is intentionally not recomputed in RQ7; Experiment 4 already addressed multiscale geometry.",
        "write_policy": "write-once; no destructive start-fresh mode; Experiment 1-4 files never modified",
        "cohort_rows": int(len(population)), "device": DEVICE, "cpu_threads": int(CPU_THREADS),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    scientific_signature = {
        k: design[k] for k in [
            "version","RQ7","datasets","cnn_architectures","optimizers","activations","seeds",
            "trained_cnn_checkpoints_declared","conditions_per_model","environments","condition_definition",
            "attenuation_alpha","gate_on_selected","parameter_target_fraction_of_positive_identities",
            "equal_budget_rule","occurrence_selection_rule","parameter_selection_rule",
            "primary_functional_outcome","primary_internal_outcome","field_A","field_B","field_B_scope_guard"
        ]
    }
    # Runtime-specific fields such as Slurm job ID, node, CPU count, and device are deliberately excluded
    # so the exact same scientific campaign can resume across multiple allocations.
    design["design_hash"] = stable_hash_dict(scientific_signature, 20)
    return design


def write_or_validate_manifest(obj: Dict[str, Any]):
    path = OUTDIR / "experiment_manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("design_hash") != obj.get("design_hash"):
            raise RuntimeError(
                "Experiment 5 output directory already contains a DIFFERENT design manifest. "
                "No files will be overwritten. Set PATHFLOW_EXP5_OUTDIR to a new empty directory."
            )
        return
    atomic_json(obj, path)


def _aggregate_and_finalize_exp5(manifest: pd.DataFrame, design: Dict[str, Any], start: float):
    """Aggregate completed write-once run parts and produce RQ7 statistics without touching raw evidence."""
    banner("Aggregate and analyze RQ7 — streaming / bounded-memory")
    tabs = aggregate_exp5_parts(manifest)
    validate_exp5_tables(tabs, manifest)
    gc.collect(); print(f"[aggregate] validated | RSS={_rss_mb():.1f} MiB", flush=True)
    cpe = build_condition_paired_effects(tabs["measurements"], tabs["field_diagnostics"])
    gc.collect(); print(f"[aggregate] condition paired effects | rows={len(cpe)} | RSS={_rss_mb():.1f} MiB", flush=True)
    mle = build_model_level_effects(cpe)
    gc.collect(); print(f"[aggregate] model effects | rows={len(mle)} | RSS={_rss_mb():.1f} MiB", flush=True)
    primary = rq7_primary_tests(mle)
    strat = rq7_stratified_tests(mle)
    mechanism = rq7_mechanism_tests(mle)
    assoc = rq7_information_consequence_associations(mle)
    vs = validity_summary(tabs["validity"])

    elapsed = time.perf_counter()-start
    summary = [
        "WORK II EXPERIMENT 5 CNN OCCURRENCE / SHARED-PARAMETER IDENTITY COMPARISON COMPLETE",
        "RQ7: For CNNs, is spatial use more informative than shared parameter identity?",
        f"Device: {DEVICE}", f"Wall time: {human_time(elapsed)}",
        f"Engineering aggregation patch: {EXP5_ENGINEERING_PATCH}",
        f"Frozen CNN checkpoints reused: {len(manifest)}/{len(manifest)}", f"Conditions per model: {EXP5_CONDITION_COUNT}",
        f"Environments: {ENVIRONMENTS}", f"Intervention alpha: {EXP5_ALPHA}",
        f"Equal-budget nonzero interventions: {len(manifest)*len(ENVIRONMENTS)*EXP5_CONDITION_COUNT*2}",
        f"Measurement rows: {len(tabs['measurements'])}", f"Validity rows: {len(tabs['validity'])}",
        f"Target-audit rows: {int(tabs['target_audit_meta']['row_count'])}", f"Field diagnostics: {len(tabs['field_diagnostics'])}",
        f"Raw field NPZ files: {len(tabs['field_index'])}", f"Primary RQ7 test rows: {len(primary)}",
        f"Stratified RQ7 test rows: {len(strat)}", f"Mechanism diagnostic rows: {len(mechanism)}",
        f"Output: {OUTDIR}",
        "Interpretability: Field A is the input-origin marginal of the complete conditional path measure; Field B is the exact first-convolution spatial footprint of each intervention support.",
        "Causal guard: parameter sharing is the learned structural substrate; RQ7 tests whether quotienting spatial realized use loses intervention-relevant localization.",
    ]
    write_once_text("\n".join(summary)+"\n", OUTDIR / "experiment_summary.txt")
    complete_marker = OUTDIR / "EXPERIMENT5_COMPLETE.json"
    atomic_json({"completed":True,"version":EXP5_VERSION,"design_hash":design["design_hash"],"timestamp_unix":time.time(),
                 "wall_seconds":elapsed,"models":len(manifest),"conditions":len(cpe),
                 "engineering_patch":EXP5_ENGINEERING_PATCH}, complete_marker)
    print("\n" + "\n".join(summary))
    return {"manifest":manifest,"tables":tabs,"condition_effects":cpe,"model_effects":mle,
            "primary_tests":primary,"stratified_tests":strat,"mechanism_tests":mechanism,
            "associations":assoc,"validity_summary":vs}


# ==================================================================================================
# 25. ORCHESTRATION
# ==================================================================================================


def _load_or_run_csv(path: Path, runner):
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return runner()


def run_experiment5():
    start = time.perf_counter()
    banner("WORK II — EXPERIMENT 5: CNN OCCURRENCE / SHARED-PARAMETER IDENTITY COMPARISON")
    print(f"Device: {DEVICE} | CPU threads: {CPU_THREADS}")
    print(f"Script/runtime root: {SCRIPT_DIR}")
    print(f"Experiment 1 source: {EXP1_RESULTS}")
    print(f"Experiment 5 output: {OUTDIR}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Slurm job: {os.environ.get('SLURM_JOB_ID','')} | node: {os.environ.get('SLURMD_NODENAME','')}")
    print("Write policy: NON-DESTRUCTIVE / WRITE-ONCE. No Experiment 1–4 result is modified.")

    if AGGREGATE_ONLY:
        print("Aggregation-only mode: no dataset loading, checkpoint loading, model execution, intervention, field generation, or figure rendering will occur.")
        manifest = cnn_manifest()
        missing = [str(r) for r in manifest.run_id.astype(str) if not exp5_done_path(str(r)).exists()]
        if missing:
            raise RuntimeError(f"Aggregation-only mode requires all Experiment 5 model completion markers; missing {len(missing)} model(s).")
        design_path = OUTDIR / "experiment_manifest.json"
        if not design_path.exists():
            raise RuntimeError("Aggregation-only mode requires the existing Experiment 5 design manifest from the completed campaign.")
        design = json.loads(design_path.read_text(encoding="utf-8"))
        complete_marker = OUTDIR / "EXPERIMENT5_COMPLETE.json"
        if complete_marker.exists():
            print(f"Experiment 5 completion marker already exists: {complete_marker}")
            return {"manifest":manifest,"already_complete":True}
        return _aggregate_and_finalize_exp5(manifest, design, start)

    if REQUIRE_CUDA and not SELFTEST_ONLY and DEVICE != "cuda":
        raise RuntimeError("Experiment 5 full profile requires CUDA. Set PATHFLOW_EXP5_REQUIRE_CUDA=0 only for deliberate CPU testing.")

    banner("Dataset-free fidelity tests")
    operator_path = OUTDIR / "tables" / "operator_selftest.csv"
    operator = _load_or_run_csv(operator_path, run_operator_selftest)
    exp5_operator_path = OUTDIR / "tables" / "exp5_operator_selftest.csv"
    exp5_operator = _load_or_run_csv(exp5_operator_path, run_exp5_operator_selftest)
    if SELFTEST_ONLY:
        return {"operator_selftest": operator, "exp5_operator_selftest": exp5_operator}

    manifest = cnn_manifest()
    inventory = verify_exp1_cnn_checkpoints(manifest)
    data_cache = load_datasets()
    population = attach_exp5_population(data_cache)
    cohort_audit = audit_prior_cohorts_if_present(population)
    design = exp5_design_object(manifest, population)
    write_or_validate_manifest(design)
    print(f"Frozen CNN checkpoints: {len(manifest)}")
    print(f"Declared baseline conditions: {design['declared_baseline_conditions']}")
    print(f"Declared nonzero interventions: {design['declared_nonzero_interventions']}")

    preflight = pd.DataFrame()
    if RUN_PREFLIGHT:
        banner("Frozen-checkpoint common intervention fidelity preflight")
        p = OUTDIR / "tables" / "intervention_preflight.csv"
        preflight_common = _load_or_run_csv(p, lambda: run_preflight(manifest, data_cache))
        banner("Experiment 5 real-checkpoint equal-budget/field preflight")
        p5 = OUTDIR / "tables" / "exp5_real_checkpoint_preflight.csv"
        preflight = _load_or_run_csv(p5, lambda: run_exp5_real_preflight(manifest, data_cache))
    if PREFLIGHT_ONLY:
        return {"manifest":manifest,"inventory":inventory,"population":population,"cohort_audit":cohort_audit,
                "operator_selftest":operator,"exp5_operator_selftest":exp5_operator,"exp5_preflight":preflight}

    complete_marker = OUTDIR / "EXPERIMENT5_COMPLETE.json"
    if complete_marker.exists():
        print(f"Experiment 5 completion marker already exists: {complete_marker}")
        print("No scientific files will be rewritten.")
        return {"manifest":manifest,"already_complete":True}

    banner("Full RQ7 causal + interpretability campaign")
    remaining = sum(not exp5_done_path(r).exists() for r in manifest.run_id.astype(str))
    eta = ETAReporter("EXP5 CNN models", remaining); completed_now = 0; wall_stop = False
    for _, row in manifest.iterrows():
        rid = str(row.run_id)
        if RESUME and exp5_done_path(rid).exists():
            continue
        if not can_start_new_unit("EXP5 model", extra_buffer_min=EXP5_BUFFER_MIN):
            wall_stop = True; break
        t0 = time.perf_counter()
        finished = run_exp5_one_model(row, data_cache[str(row.dataset)])
        if not finished:
            wall_stop = True; break
        completed_now += 1; eta.update(time.perf_counter()-t0, completed_now)
    eta.finish()
    n_done = sum(exp5_done_path(r).exists() for r in manifest.run_id.astype(str))
    if n_done < len(manifest):
        pause = {
            "status":"paused_safely","completed_models":int(n_done),"declared_models":int(len(manifest)),
            "software_wall_stop":bool(wall_stop),"remaining_seconds":soft_wall_remaining_seconds(),"timestamp_unix":time.time()
        }
        pause_path = OUTDIR / "audit" / f"PAUSED_{int(time.time())}.json"
        atomic_json(pause, pause_path)
        print(f"Experiment 5 paused safely at {n_done}/{len(manifest)} models. Rerun the same file to resume.")
        return {"manifest":manifest,"incomplete":True,"completed_models":n_done}

    return _aggregate_and_finalize_exp5(manifest, design, start)


if __name__ == "__main__":
    EXP5_RESULTS = run_experiment5()
