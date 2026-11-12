from __future__ import annotations

# ==================================================================================================
# WORK II — EXPERIMENT 2: CONDITION CROSSOVER (RQ3)
# MPI3D-TOY + FASHION-MNIST | DEEP MLPs + CNNs | HPC/SLURM SAFE
# ==================================================================================================
#
# PURPOSE
# -------
# Test whether computational responsibility is condition-dependent in the SAME trained network.
# For each underlying test observation, define two conditions using the already-established benchmark:
#
#       c_A = (x_clean,   0)
#       c_B = (x_shifted, 0)
#
# The network parameters, architecture, semantic test item, target output, zero reference,
# intervention strength, and layer-wise target budget remain fixed. Only target origin and evaluation
# condition vary.
#
# For each layer, because conditional edge/occurrence flows are normalized marginals on the same
# carrier, preferential targets are ranked by the signed excess flow
#
#       delta_a = F_a^{c_A} - F_a^{c_B}.
#
# The c_A target uses the largest positive delta_a; the c_B target uses the most negative delta_a.
# The two target sets have exactly the same cardinality in every layer and are frozen before testing.
# Each target set is then attenuated under BOTH conditions, giving the 2 x 2 design:
#
#                                 evaluation condition
#                             c_A                    c_B
#   target from c_A          D_AA                   D_AB
#   target from c_B          D_BA                   D_BB
#
# PRIMARY RQ3 INTERACTION
# -----------------------
# The primary analysis is the target-origin x evaluation-condition crossover interaction, formed
# from WITHIN-evaluation-condition contrasts so clean/shifted response-scale differences cannot create
# a spurious crossover:
#
#       A advantage = D_AA - D_BA
#       B advantage = D_BB - D_AB
#       crossover   = 0.5 * [(D_AA-D_BA) + (D_BB-D_AB)]
#
# Positive crossover means a target is, on average, more consequential in the condition from which it
# was identified. The older same-target cross-condition contrasts are retained only as secondary output.
#
# DEPENDENCY ON EXPERIMENT 1
# --------------------------
# This file DOES NOT use Experiment 1 intervention-result tables and DOES NOT retrain networks.
# It requires only:
#   1) Experiment 1 best-validation checkpoints: <EXP1_RESULTS>/checkpoints/*.pt
#   2) Experiment 1 dataset cache:             <EXP1_RESULTS>/data/
#      (or MPI3D_NPZ / Fashion-MNIST may be supplied/downloaded again)
#
# The expected 324-model manifest is reconstructed in this file and checked against the checkpoint
# inventory. Experiment 1's experiment_manifest.json is used only as an optional sanity audit if present.
#
# OUTPUT ISOLATION
# ----------------
# By default, if this file sits beside the Experiment 1 result directory, it writes only to:
#
#   pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results/
#       experiment2_condition_crossover/
#           tables/
#           figures/
#           run_parts/
#           audit/
#
# No Experiment 1 table, run-part, checkpoint, or dataset file is overwritten.
#
# DEFAULT SCALE
# -------------
#   324 frozen trained networks
#   10 paired clean/shifted observations per model
#   4 intervention evaluations per pair
#   = 12,960 controlled evaluations
#   alpha = 0.25 (same as Experiment 1)
#   target fraction basis = 10% of positive realized support, capped symmetrically by available
#                           preferential support in both conditions
#
# HPC
# ---
# Designed for one NVIDIA L4 (~22-24 GiB VRAM), 4 CPUs, 64 GiB host RAM, <=48 h allocation.
# Uses a 46 h software wall guard, atomic per-model run-parts, checkpoint hashing, tqdm + rolling ETA,
# and fail-fast operator/checkpoint/crossover preflights before the full campaign.
# Run either:
#       srun python -u exp2_condition_crossover_SLURM_SAFE_v1.py
# or from one HPC Jupyter cell:
#       %run exp2_condition_crossover_SLURM_SAFE_v1.py
# ==================================================================================================

import os
import math
import json
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

PROFILE = os.environ.get("PATHFLOW_CROSSOVER_PROFILE", "l4_48h").strip().lower()
RESUME = os.environ.get("PATHFLOW_CROSSOVER_RESUME", "1") != "0"
START_FRESH = os.environ.get("PATHFLOW_CROSSOVER_START_FRESH", "0") == "1"
RUN_PREFLIGHT = os.environ.get("PATHFLOW_CROSSOVER_PREFLIGHT", "1") != "0"
PREFLIGHT_ONLY = os.environ.get("PATHFLOW_CROSSOVER_PREFLIGHT_ONLY", "0") == "1"
RUN_FIGURES = os.environ.get("PATHFLOW_CROSSOVER_FIGURES", "1") != "0"
ALLOW_DOWNLOAD = os.environ.get("PATHFLOW_CROSSOVER_ALLOW_DOWNLOAD", "1") != "0"
SELFTEST_ONLY = os.environ.get("PATHFLOW_CROSSOVER_SELFTEST_ONLY", "0") == "1"
REQUIRE_CUDA = os.environ.get("PATHFLOW_CROSSOVER_REQUIRE_CUDA", "1") != "0"
SOFT_WALL_HOURS = float(os.environ.get("PATHFLOW_CROSSOVER_SOFT_WALL_HOURS", "46.0"))
SOFT_START_BUFFER_MIN = float(os.environ.get("PATHFLOW_CROSSOVER_SOFT_START_BUFFER_MIN", "20.0"))
SCRIPT_START_MONOTONIC = time.perf_counter()

SEED = 730031
MODEL_SEED_BASE = 271828
ORDER_SEED_BASE = 141421
NOISE_SEED_BASE = 161803
MATCH_SEED_BASE = 223607
CONDITION_SEED_BASE = 244949

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("PATHFLOW_CROSSOVER_CPU_THREADS", "4")))
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

# Experiment 2 is designed to run from the SAME working directory as Experiment 1.
# It reuses only the immutable best-validation checkpoints and dataset files from Experiment 1.
# All Experiment 2 outputs are isolated under a dedicated subtree, so no Experiment 1 table/run-part
# can be overwritten.
_EXP1_RESULTS_ENV = os.environ.get("PATHFLOW_EXP1_RESULTS", "").strip()
EXP1_RESULTS = (
    Path(_EXP1_RESULTS_ENV).expanduser().resolve()
    if _EXP1_RESULTS_ENV
    else SCRIPT_DIR / "pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results"
)
SOURCE_CHECKPOINT_DIR = EXP1_RESULTS / "checkpoints"

_EXP2_OUTDIR_ENV = os.environ.get("PATHFLOW_CROSSOVER_OUTDIR", "").strip()
OUTDIR = (
    Path(_EXP2_OUTDIR_ENV).expanduser().resolve()
    if _EXP2_OUTDIR_ENV
    else EXP1_RESULTS / "experiment2_condition_crossover"
)

_DATA_ROOT_ENV = os.environ.get("PATHFLOW_CROSSOVER_DATA_ROOT", "").strip()
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
        "Could not create/write the Experiment 2 output directory. "
        f"Resolved EXP1_RESULTS={EXP1_RESULTS}, OUTDIR={OUTDIR}. "
        "Run this script from the Experiment 1 working directory or set PATHFLOW_EXP1_RESULTS and "
        "PATHFLOW_CROSSOVER_OUTDIR to writable persistent Slurm/HPC paths."
    ) from e

if START_FRESH:
    # Data are intentionally preserved. Scientific outputs/checkpoints are cleared.
    for sub in ["tables", "figures", "checkpoints", "training_parts", "run_parts", "audit"]:
        for p in (OUTDIR / sub).glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

DATASETS = [x.strip().lower() for x in os.environ.get("PATHFLOW_CROSSOVER_DATASETS", "mpi3d,fmnist").split(",") if x.strip()]
ACTIVATIONS = [x.strip().lower() for x in os.environ.get("PATHFLOW_CROSSOVER_ACTIVATIONS", "relu,tanh,silu").split(",") if x.strip()]
OPTIMIZERS = [x.strip().lower() for x in os.environ.get("PATHFLOW_CROSSOVER_OPTIMIZERS", "sgdm,rmsprop,adam").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("PATHFLOW_CROSSOVER_SEEDS", "0,1,2").split(",") if x.strip()]
ENVIRONMENTS = [x.strip().lower() for x in os.environ.get("PATHFLOW_CROSSOVER_ENVIRONMENTS", "clean,shifted").split(",") if x.strip()]

IMAGE_RESOLUTION = 16
N_TRAIN = int(os.environ.get("PATHFLOW_CROSSOVER_NTRAIN", "20000"))
N_VAL = int(os.environ.get("PATHFLOW_CROSSOVER_NVAL", "5000"))
N_TEST = int(os.environ.get("PATHFLOW_CROSSOVER_NTEST", "5000"))
BATCH_SIZE = int(os.environ.get("PATHFLOW_CROSSOVER_BATCH", "256"))
MAX_EPOCH = int(os.environ.get("PATHFLOW_CROSSOVER_EPOCHS", "160"))
WEIGHT_DECAY = 0.0
ENDPOINT_INPUTS = int(os.environ.get("PATHFLOW_CROSSOVER_ENDPOINT_INPUTS", "160"))
SHIFT_SNR_DB = float(os.environ.get("PATHFLOW_CROSSOVER_SHIFT_SNR_DB", "15.0"))

LEARNING_RATES = {
    "sgdm": 1.0e-2,
    "rmsprop": 1.0e-3,
    "adam": 1.0e-3,
}

# First intervention experiment choices.
CONDITIONS_PER_MODEL = int(os.environ.get("PATHFLOW_CROSSOVER_CONDITIONS", "40"))
TARGET_FRACTION = float(os.environ.get("PATHFLOW_CROSSOVER_TARGET_FRACTION", "0.10"))
ATTENUATION = float(os.environ.get("PATHFLOW_CROSSOVER_ALPHA", "0.25"))
SAVE_EVERY_CONDITIONS = int(os.environ.get("PATHFLOW_CROSSOVER_SAVE_EVERY", "5"))
MATCH_BINS = int(os.environ.get("PATHFLOW_CROSSOVER_MATCH_BINS", "20"))
PREFLIGHT_TOL = float(os.environ.get("PATHFLOW_CROSSOVER_PREFLIGHT_TOL", "5e-6"))

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
            "\nSet MPI3D_NPZ=/path/to/mpi3d_toy.npz or PATHFLOW_CROSSOVER_ALLOW_DOWNLOAD=1."
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
    root = Path(os.environ.get("PATHFLOW_CROSSOVER_FMNIST_ROOT", str(DATA_ROOT / "fmnist"))).expanduser().resolve()
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
# 15. PER-MODEL INTERVENTION LOOP
# ==================================================================================================

def completed_keys(run_id: str) -> set:
    p = part_path(run_id, "validity")
    d = safe_read_csv(p)
    if d.empty:
        return set()
    keys = set()
    for _, r in d.iterrows():
        try:
            keys.add((str(r.environment), int(r.condition_index), str(r.target_type)))
        except Exception:
            pass
    return keys


def load_existing_parts(run_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    m = safe_read_csv(part_path(run_id, "measurements"))
    v = safe_read_csv(part_path(run_id, "validity"))
    a = safe_read_csv(part_path(run_id, "target_audit"))
    return (
        m.to_dict("records") if len(m) else [],
        v.to_dict("records") if len(v) else [],
        a.to_dict("records") if len(a) else [],
    )


def flush_model_parts(run_id: str, measurements: List[Dict[str, Any]], validity: List[Dict[str, Any]], audit: List[Dict[str, Any]]):
    atomic_csv(pd.DataFrame(measurements), part_path(run_id, "measurements"))
    atomic_csv(pd.DataFrame(validity), part_path(run_id, "validity"))
    atomic_csv(pd.DataFrame(audit), part_path(run_id, "target_audit"))


def base_meta(row: pd.Series, environment: str, condition_index: int, target_idx: int) -> Dict[str, Any]:
    return {
        "run_id": str(row.run_id),
        "dataset": str(row.dataset),
        "task": str(row.task),
        "family": str(row.family),
        "arch_id": str(row.arch_id),
        "optimizer": str(row.optimizer),
        "activation": str(row.activation),
        "seed_index": int(row.seed_index),
        "environment": str(environment),
        "condition_index": int(condition_index),
        "target_output": int(target_idx),
        "target_fraction": float(TARGET_FRACTION),
        "attenuation_alpha": float(ATTENUATION),
    }


@torch.no_grad()
def run_one_model(row: pd.Series, data: Dict[str, Any]):
    rid = str(row.run_id)
    if RESUME and done_path(rid).exists():
        return

    cp = checkpoint_path(rid)
    cp_hash_before = file_sha1(cp)
    chk = torch.load(cp, map_location="cpu")
    model = build_model(str(row.arch_id), str(row.activation), output_dim_for(str(row.dataset)), str(row.dataset), int(row.seed_index))
    model.load_state_dict(chk["state_dict"])
    model.to(DEVICE).eval()
    family = str(row.family)

    measurements, validity, audit = load_existing_parts(rid) if RESUME else ([], [], [])
    done = completed_keys(rid) if RESUME else set()

    # Structural states are output-specific and condition-independent. Cache once per output.
    structural_cache: Dict[int, Dict[str, Any]] = {}

    total_conditions = len(ENVIRONMENTS) * min(CONDITIONS_PER_MODEL, len(data["causal_endpoint_local_indices"]))
    cond_bar = tqdm(total=total_conditions, desc=f"{rid[:52]}", leave=False)
    processed_since_flush = 0

    for env in ENVIRONMENTS:
        Xc, yc, causal_local = prepared_endpoint_environment(data, str(row.dataset), env)
        for ci in range(len(Xc)):
            x = Xc[ci:ci + 1]
            y = yc[ci:ci + 1]
            x0 = torch.zeros_like(x)
            target_idx = int(y.reshape(-1)[0].item()) if str(row.task) == "classification" else 0
            meta = base_meta(row, env, ci, target_idx)

            expected_types = ["realized", "structural", "matched"]
            if all((env, ci, t) in done for t in expected_types):
                cond_bar.update(1)
                continue

            baseline = baseline_condition(model, family, x, x0, target_idx)
            baseline_valid = bool(baseline["state"]["valid"][0])

            if not baseline_valid:
                for target_type in expected_types:
                    key = (env, ci, target_type)
                    if key in done:
                        continue
                    validity.append({
                        **meta,
                        "target_type": target_type,
                        "baseline_realization_defined": False,
                        "intervention_applicable": False,
                        "intervened_realization_defined": False,
                        "intervention_induced_degeneracy": False,
                        "n_targets_total": 0,
                        "status": "not_applicable_baseline_zero_support",
                        "reason": "baseline_zero_complete_path_support",
                    })
                    done.add(key)
                processed_since_flush += 1
                if processed_since_flush >= SAVE_EVERY_CONDITIONS:
                    flush_model_parts(rid, measurements, validity, audit)
                    processed_since_flush = 0
                cond_bar.update(1)
                continue

            # ----- Target construction from the unperturbed baseline -----
            real_flow = flow_list_from_state(baseline["state"], family)
            real_masks, real_audit = select_realized_masks(real_flow, TARGET_FRACTION)
            counts = target_counts(real_masks)
            n_targets_total = int(sum(counts))
            if n_targets_total == 0:
                # Baseline measure exists, but no positive marginal was found. This is an integrity pathology.
                raise RuntimeError(f"Defined realization with zero positive marginal targets: {rid}, env={env}, condition={ci}")

            if target_idx not in structural_cache:
                structural_cache[target_idx] = output_matched_structural_state(model, family, target_idx)
            st_struct = structural_cache[target_idx]
            if not bool(st_struct["valid"][0]):
                raise RuntimeError(f"Output-matched structural state unexpectedly undefined: {rid}, target={target_idx}")
            struct_flow = flow_list_from_state(st_struct, family)
            struct_masks = select_top_k_masks(struct_flow, counts)

            if family == "mlp":
                contrib = ordinary_activation_scores_mlp(model, baseline["hs"], baseline["hs0"])
            else:
                contrib = ordinary_activation_scores_cnn(model, baseline["hs"], baseline["hs0"])
            match_seed = MATCH_SEED_BASE + int(stable_hash_dict({"run": rid, "env": env, "condition": ci}, 8), 16) % 1000003
            real_support = positive_support_masks(real_flow)
            matched_masks, matched_audit = select_matched_masks(real_masks, contrib, real_support, match_seed)

            jacc, per_jacc = jaccard_masks(real_masks, struct_masks)
            for li, ra in enumerate(real_audit):
                ma = matched_audit[li]
                audit.append({
                    **meta,
                    "layer": int(li),
                    "n_total": int(ra["n_total"]),
                    "n_positive_flow": int(ra["n_positive_flow"]),
                    "n_targets": int(ra["n_targets"]),
                    "raw_fraction_target_count": int(ra.get("raw_fraction_target_count", ra["n_targets"])),
                    "target_count_capped_for_matched_support": bool(ra.get("target_count_capped_for_matched_support", False)),
                    "positive_support_fraction": float(ra["positive_support_fraction"]),
                    "realized_selected_flow_mass": float(ra["selected_flow_mass"]),
                    "realized_structural_jaccard_layer": float(per_jacc[li]) if np.isfinite(per_jacc[li]) else np.nan,
                    "realized_mask_hash": hash_indices(real_masks[li]),
                    "structural_mask_hash": hash_indices(struct_masks[li]),
                    "matched_mask_hash": hash_indices(matched_masks[li]),
                    "matched_support_pool": int(ma.get("n_matched_pool", 0)),
                    "matched_activation_ks": ma["matched_activation_ks"],
                    "realized_activation_median": ma["realized_activation_median"],
                    "matched_activation_median": ma["matched_activation_median"],
                    "matched_to_realized_activation_median_ratio": ma["matched_to_realized_activation_median_ratio"],
                    "matched_borrowed_fraction": ma["borrowed_fraction"],
                })

            mask_map = {
                "realized": real_masks,
                "structural": struct_masks,
                "matched": matched_masks,
            }

            for target_type in expected_types:
                key = (env, ci, target_type)
                if key in done:
                    continue
                masks = mask_map[target_type]
                t0 = time.perf_counter()
                result = intervene_condition(model, family, x, x0, target_idx, masks, baseline, ATTENUATION)
                elapsed = time.perf_counter() - t0
                int_valid = bool(result["valid"])

                validity.append({
                    **meta,
                    "target_type": target_type,
                    "baseline_realization_defined": True,
                    "intervention_applicable": True,
                    "intervened_realization_defined": int_valid,
                    "intervention_induced_degeneracy": bool(not int_valid),
                    "n_targets_total": n_targets_total,
                    "realized_structural_jaccard": jacc,
                    "status": "defined" if int_valid else "manifestation_only_intervened_zero_support",
                    "reason": "" if int_valid else "intervened_zero_complete_path_support",
                    "intervention_seconds": float(elapsed),
                })

                mrec = manifestation_record(
                    str(row.dataset), y, target_idx,
                    baseline["out"], baseline["out0"],
                    result["out"], result["out0"],
                )
                rec = {
                    **meta,
                    "target_type": target_type,
                    "n_targets_total": n_targets_total,
                    "realized_structural_jaccard": jacc,
                    **mrec,
                    "baseline_eta1": float(baseline["state"]["eta1"][0]),
                    "baseline_eta2": float(baseline["state"]["eta2"][0]),
                    "baseline_Sdef": float(baseline["state"]["entropy_deficit"][0]),
                    "baseline_Sdef_norm": float(baseline["state"]["entropy_deficit_norm"][0]),
                    "path_hellinger": float(result["hellinger"]) if np.isfinite(result["hellinger"]) else np.nan,
                    "intervention_seconds": float(elapsed),
                }
                if int_valid:
                    rec.update({
                        "intervened_eta1": float(result["state"]["eta1"][0]),
                        "intervened_eta2": float(result["state"]["eta2"][0]),
                        "intervened_Sdef": float(result["state"]["entropy_deficit"][0]),
                        "intervened_Sdef_norm": float(result["state"]["entropy_deficit_norm"][0]),
                        "delta_eta1": float(result["state"]["eta1"][0] - baseline["state"]["eta1"][0]),
                        "delta_eta2": float(result["state"]["eta2"][0] - baseline["state"]["eta2"][0]),
                        "delta_Sdef": float(result["state"]["entropy_deficit"][0] - baseline["state"]["entropy_deficit"][0]),
                        "delta_Sdef_norm": float(result["state"]["entropy_deficit_norm"][0] - baseline["state"]["entropy_deficit_norm"][0]),
                    })
                else:
                    rec.update({
                        "intervened_eta1": np.nan,
                        "intervened_eta2": np.nan,
                        "intervened_Sdef": np.nan,
                        "intervened_Sdef_norm": np.nan,
                        "delta_eta1": np.nan,
                        "delta_eta2": np.nan,
                        "delta_Sdef": np.nan,
                        "delta_Sdef_norm": np.nan,
                    })
                measurements.append(rec)
                done.add(key)

            processed_since_flush += 1
            if processed_since_flush >= SAVE_EVERY_CONDITIONS:
                flush_model_parts(rid, measurements, validity, audit)
                processed_since_flush = 0

            cond_bar.set_postfix({"env": env, "c": ci + 1, "targets": n_targets_total})
            cond_bar.update(1)

    cond_bar.close()
    flush_model_parts(rid, measurements, validity, audit)

    cp_hash_after = file_sha1(cp)
    if cp_hash_before != cp_hash_after:
        raise RuntimeError(f"SOURCE CHECKPOINT CHANGED DURING INTERVENTION STUDY: {cp}")

    atomic_json({
        "run_id": rid,
        "completed": True,
        "checkpoint_sha1": cp_hash_before,
        "n_measurement_rows": len(measurements),
        "n_validity_rows": len(validity),
        "n_target_audit_rows": len(audit),
        "timestamp_unix": time.time(),
    }, done_path(rid))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================================================================================================
# 16. AGGREGATION / PRIMARY STATISTICS
# ==================================================================================================

def aggregate_parts(manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    ms, vs, aa = [], [], []
    for rid in manifest.run_id.astype(str):
        for kind, dest in [("measurements", ms), ("validity", vs), ("target_audit", aa)]:
            d = safe_read_csv(part_path(rid, kind))
            if len(d):
                dest.append(d)
    out = {
        "measurements": pd.concat(ms, ignore_index=True) if ms else pd.DataFrame(),
        "validity": pd.concat(vs, ignore_index=True) if vs else pd.DataFrame(),
        "target_audit": pd.concat(aa, ignore_index=True) if aa else pd.DataFrame(),
    }
    atomic_csv(out["measurements"], OUTDIR / "tables" / "intervention_measurements.csv")
    atomic_csv(out["validity"], OUTDIR / "tables" / "intervention_validity.csv")
    atomic_csv(out["target_audit"], OUTDIR / "tables" / "target_audit.csv")
    return out


def validate_intervention_tables(tabs: Dict[str, pd.DataFrame]):
    m = tabs["measurements"]
    v = tabs["validity"]
    if m.empty or v.empty:
        raise RuntimeError("Intervention tables are empty.")
    required_v = {
        "run_id", "environment", "condition_index", "target_type",
        "baseline_realization_defined", "intervention_applicable", "intervened_realization_defined", "status"
    }
    missing = required_v - set(v.columns)
    if missing:
        raise RuntimeError(f"Intervention validity schema missing {sorted(missing)}")
    keys = ["run_id", "environment", "condition_index", "target_type"]
    if v.duplicated(keys).any():
        raise RuntimeError("Duplicate intervention validity keys detected.")
    # Every numerical measurement must correspond to an applicable baseline-defined intervention.
    vk = set(tuple(x) for x in v[v.intervention_applicable.astype(bool)][keys].to_records(index=False))
    mk = set(tuple(x) for x in m[keys].to_records(index=False))
    if not mk.issubset(vk):
        raise RuntimeError("Measurement rows exist without corresponding applicable validity records.")
    # All three target types must have identical budget within each condition whenever baseline is defined.
    q = v[v.intervention_applicable.astype(bool)].copy()
    if len(q):
        piv = q.pivot_table(index=["run_id", "environment", "condition_index"], columns="target_type", values="n_targets_total", aggfunc="first")
        needed = [x for x in ["realized", "structural", "matched"] if x in piv.columns]
        if len(needed) == 3:
            bad = piv[needed].nunique(axis=1) > 1
            if bad.any():
                raise RuntimeError(f"Target-budget mismatch in {int(bad.sum())} condition(s).")
    print(
        f"[intervention tables] measurements={len(m)} | validity={len(v)} | "
        f"intervention-induced degenerate states={int(v.intervention_induced_degeneracy.astype(bool).sum()) if 'intervention_induced_degeneracy' in v else 0}"
    )


def model_level_effects(m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated conditions within each trained model before primary inference."""
    if m.empty:
        return pd.DataFrame()
    metrics = [
        "manifestation_abs", "manifestation_relative", "finite_response_vector_l2",
        "path_hellinger", "delta_eta1", "delta_eta2", "delta_Sdef_norm", "task_loss_change",
    ]
    metrics = [x for x in metrics if x in m.columns]
    group = ["run_id", "dataset", "family", "arch_id", "optimizer", "activation", "seed_index", "environment", "target_type"]
    agg = m.groupby(group, dropna=False)[metrics].median().reset_index()
    atomic_csv(agg, OUTDIR / "tables" / "model_level_intervention_medians.csv")

    rows = []
    idcols = ["run_id", "dataset", "family", "arch_id", "optimizer", "activation", "seed_index", "environment"]
    for metric in metrics:
        piv = agg.pivot_table(index=idcols, columns="target_type", values=metric, aggfunc="first").reset_index()
        if all(x in piv.columns for x in ["realized", "structural", "matched"]):
            for _, r in piv.iterrows():
                rows.append({
                    **{c: r[c] for c in idcols},
                    "metric": metric,
                    "realized": r["realized"],
                    "structural": r["structural"],
                    "matched": r["matched"],
                    "realized_minus_matched": r["realized"] - r["matched"] if np.isfinite(r["realized"]) and np.isfinite(r["matched"]) else np.nan,
                    "realized_minus_structural": r["realized"] - r["structural"] if np.isfinite(r["realized"]) and np.isfinite(r["structural"]) else np.nan,
                })
    eff = pd.DataFrame(rows)
    atomic_csv(eff, OUTDIR / "tables" / "model_level_intervention_effects.csv")
    return eff


def primary_paired_tests(eff: pd.DataFrame) -> pd.DataFrame:
    if eff.empty:
        return pd.DataFrame()
    rows = []
    # Primary scientific endpoint: target-output finite-response displacement.
    # Report dataset/family/environment strata and dataset/environment aggregate over architectures.
    for metric in ["manifestation_abs", "manifestation_relative", "path_hellinger"]:
        d0 = eff[eff.metric == metric]
        for dataset in sorted(d0.dataset.unique()):
            for environment in sorted(d0.environment.unique()):
                for family in ["all", "mlp", "cnn"]:
                    d = d0[(d0.dataset == dataset) & (d0.environment == environment)]
                    if family != "all":
                        d = d[d.family == family]
                    for contrast in ["realized_minus_matched", "realized_minus_structural"]:
                        vals = pd.to_numeric(d[contrast], errors="coerce").to_numpy(float)
                        vals = vals[np.isfinite(vals)]
                        if len(vals) < 3:
                            continue
                        try:
                            stat, p = wilcoxon(vals, zero_method="wilcox", alternative="greater")
                        except Exception:
                            stat, p = np.nan, 1.0
                        lo, hi = bootstrap_median_ci(vals, seed=3301 + len(rows))
                        rows.append({
                            "metric": metric,
                            "dataset": dataset,
                            "environment": environment,
                            "family": family,
                            "contrast": contrast,
                            "alternative": ">0",
                            "n_trained_models": int(len(vals)),
                            "median_difference": float(np.median(vals)),
                            "hodges_lehmann": hodges_lehmann_onesample(vals),
                            "rank_biserial": paired_rank_biserial(vals),
                            "ci_low": lo,
                            "ci_high": hi,
                            "wilcoxon_statistic": stat,
                            "p_raw": p,
                            "positive_fraction": float(np.mean(vals > 0)),
                        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for _, g in out.groupby(["metric", "dataset", "environment"]):
            out.loc[g.index, "p_holm"] = holm_adjust(pd.to_numeric(g.p_raw, errors="coerce").fillna(1.0).to_numpy(float))
    atomic_csv(out, OUTDIR / "tables" / "primary_paired_tests.csv")
    return out


def organization_manifestation_link(m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """RQ5 without pseudo-replication.

    Conditions are repeated observations within a trained network. We therefore first estimate the
    Hellinger-manifestation association separately inside each trained model, then perform population
    inference over those model-level association coefficients. A pooled condition-level correlation is
    not used as inferential evidence.
    """
    if m.empty or "path_hellinger" not in m.columns:
        return {"model_level": pd.DataFrame(), "tests": pd.DataFrame()}

    model_rows = []
    group_cols = [
        "run_id", "dataset", "family", "arch_id", "optimizer", "activation",
        "seed_index", "environment", "target_type",
    ]
    for keys, d in m.groupby(group_cols, dropna=False):
        x = pd.to_numeric(d.path_hellinger, errors="coerce").to_numpy(float)
        y = pd.to_numeric(d.manifestation_abs, errors="coerce").to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        n = int(np.sum(good))
        if n < 6 or np.unique(x[good]).size < 3 or np.unique(y[good]).size < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(x[good], y[good])
        rec = {c: v for c, v in zip(group_cols, keys)}
        rec.update({
            "n_conditions": n,
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p_within_model": float(p) if np.isfinite(p) else np.nan,
            "median_hellinger": float(np.nanmedian(x[good])) if n else np.nan,
            "median_manifestation_abs": float(np.nanmedian(y[good])) if n else np.nan,
        })
        model_rows.append(rec)

    model_df = pd.DataFrame(model_rows)
    atomic_csv(model_df, OUTDIR / "tables" / "organization_manifestation_model_level.csv")

    test_rows = []
    for dataset in sorted(model_df.dataset.unique()) if len(model_df) else []:
        for environment in sorted(model_df.environment.unique()):
            for target_type in ["realized", "structural", "matched"]:
                for family in ["all", "mlp", "cnn"]:
                    d = model_df[
                        (model_df.dataset == dataset)
                        & (model_df.environment == environment)
                        & (model_df.target_type == target_type)
                    ].copy()
                    if family != "all":
                        d = d[d.family == family]
                    vals = pd.to_numeric(d.spearman_rho, errors="coerce").to_numpy(float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) < 3:
                        continue
                    try:
                        stat, pval = wilcoxon(vals, zero_method="wilcox", alternative="greater")
                    except Exception:
                        stat, pval = np.nan, 1.0
                    lo, hi = bootstrap_median_ci(vals, seed=4701 + len(test_rows))
                    test_rows.append({
                        "dataset": dataset,
                        "environment": environment,
                        "target_type": target_type,
                        "family": family,
                        "n_trained_models": int(len(vals)),
                        "median_within_model_spearman_rho": float(np.median(vals)),
                        "hodges_lehmann": hodges_lehmann_onesample(vals),
                        "rank_biserial": paired_rank_biserial(vals),
                        "ci_low": lo,
                        "ci_high": hi,
                        "positive_fraction": float(np.mean(vals > 0)),
                        "wilcoxon_statistic": stat,
                        "p_raw": pval,
                    })
    tests = pd.DataFrame(test_rows)
    if len(tests):
        tests["p_holm"] = np.nan
        for _, g in tests.groupby(["dataset", "environment"]):
            tests.loc[g.index, "p_holm"] = holm_adjust(
                pd.to_numeric(g.p_raw, errors="coerce").fillna(1.0).to_numpy(float)
            )
    atomic_csv(tests, OUTDIR / "tables" / "organization_manifestation_tests.csv")
    return {"model_level": model_df, "tests": tests}

# ==================================================================================================
# 17. LIGHTWEIGHT FIGURES
# ==================================================================================================

def make_figures(tabs: Dict[str, pd.DataFrame], eff: pd.DataFrame):
    if not RUN_FIGURES:
        return
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
    })

    def save(fig, name):
        fig.savefig(OUTDIR / "figures" / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(OUTDIR / "figures" / f"{name}.png", bbox_inches="tight", dpi=600)
        plt.close(fig)

    m = tabs["measurements"]
    if len(m):
        # Condition-level manifestation distributions, shown separately by environment/family.
        for ds in sorted(m.dataset.unique()):
            d = m[m.dataset == ds]
            fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
            for ax, env in zip(axs, [e for e in ["clean", "shifted"] if e in d.environment.unique()]):
                q = d[d.environment == env]
                vals = [pd.to_numeric(q[q.target_type == t].manifestation_abs, errors="coerce").dropna().to_numpy(float)
                        for t in ["realized", "structural", "matched"]]
                ax.boxplot(vals, labels=["realized", "structural", "matched"], showfliers=False)
                ax.set_ylabel(r"$|\widetilde{\Delta f}-\Delta f|$")
                ax.set_title(f"{ds.upper()} | {env}")
            save(fig, f"fig01_{ds}_manifestation_by_target")

        q = m[np.isfinite(pd.to_numeric(m.path_hellinger, errors="coerce"))].copy()
        if len(q):
            fig, ax = plt.subplots(figsize=(4.2, 3.3), constrained_layout=True)
            for t, marker in [("realized", "o"), ("structural", "s"), ("matched", "^")]:
                z = q[q.target_type == t]
                ax.scatter(z.path_hellinger, z.manifestation_abs, s=9, alpha=.18, marker=marker, label=t)
            ax.set_xlabel(r"$D_H(\mu^c,\widetilde{\mu}^c)$")
            ax.set_ylabel(r"$|\widetilde{\Delta f}-\Delta f|$")
            ax.legend(frameon=False)
            save(fig, "fig02_internal_vs_manifestation_displacement")

    if len(eff):
        q = eff[eff.metric == "manifestation_abs"]
        if len(q):
            fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
            for ax, col, title in [
                (axs[0], "realized_minus_matched", "realized - matched"),
                (axs[1], "realized_minus_structural", "realized - structural"),
            ]:
                vals = pd.to_numeric(q[col], errors="coerce").dropna().to_numpy(float)
                ax.axhline(0, lw=.8, color="black")
                ax.scatter(np.arange(len(vals)), np.sort(vals), s=10, alpha=.65)
                ax.set_ylabel("model-level median effect difference")
                ax.set_title(title)
            save(fig, "fig03_model_level_primary_effects")


# ==================================================================================================
# 18. EXPERIMENT 2 CONFIGURATION / SOURCE-AUDIT HELPERS
# ==================================================================================================

PAIR_COUNT = int(os.environ.get("PATHFLOW_CROSSOVER_PAIRS", "10"))
PREFERENCE_TOL = float(os.environ.get("PATHFLOW_CROSSOVER_PREFERENCE_TOL", "1e-15"))
SAVE_EVERY_PAIRS = int(os.environ.get("PATHFLOW_CROSSOVER_SAVE_EVERY", "2"))

if PAIR_COUNT <= 0:
    raise ValueError("PAIR_COUNT must be positive.")
if PAIR_COUNT > CONDITIONS_PER_MODEL:
    raise ValueError(
        f"PAIR_COUNT={PAIR_COUNT} exceeds the reconstructed Experiment 1 causal population "
        f"CONDITIONS_PER_MODEL={CONDITIONS_PER_MODEL}."
    )


def exp2_part_path(run_id: str, kind: str) -> Path:
    return OUTDIR / "run_parts" / f"{run_id}__EXP2__{kind}.csv"


def exp2_done_path(run_id: str) -> Path:
    return OUTDIR / "run_parts" / f"{run_id}__EXP2__DONE.json"


def verify_exp1_source(manifest: pd.DataFrame) -> pd.DataFrame:
    """Require the exact frozen checkpoint population before any dataset-heavy work begins."""
    rows = []
    missing = []
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
        })
        if not exists:
            missing.append(str(cp))
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "source_checkpoint_inventory.csv")
    if missing:
        preview = "\n  ".join(missing[:20])
        raise FileNotFoundError(
            f"Experiment 2 requires all {len(manifest)} frozen Experiment 1 checkpoints; "
            f"{len(missing)} are missing. First missing paths:\n  {preview}\n"
            f"Resolved EXP1_RESULTS={EXP1_RESULTS}. If needed set PATHFLOW_EXP1_RESULTS=/absolute/path/to/"
            "pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results"
        )

    # Optional config audit against the Exp1 manifest. It is never a computational dependency.
    src_manifest = EXP1_RESULTS / "experiment_manifest.json"
    audit = {
        "exp1_results": str(EXP1_RESULTS),
        "source_manifest_present": src_manifest.exists(),
        "expected_checkpoints": int(len(manifest)),
        "found_checkpoints": int(df.checkpoint_exists.sum()),
    }
    if src_manifest.exists():
        try:
            obj = json.loads(src_manifest.read_text(encoding="utf-8"))
            audit.update({
                "exp1_trained_networks_declared": obj.get("trained_networks_declared"),
                "exp1_datasets": obj.get("datasets"),
                "exp1_architectures": obj.get("architectures"),
                "exp1_optimizers": obj.get("optimizers"),
                "exp1_activations": obj.get("activations"),
                "exp1_seeds": obj.get("seeds"),
            })
            checks = [
                list(obj.get("datasets", [])) == list(DATASETS),
                list(obj.get("architectures", [])) == list(ARCH_IDS),
                list(obj.get("optimizers", [])) == list(OPTIMIZERS),
                list(obj.get("activations", [])) == list(ACTIVATIONS),
                list(obj.get("seeds", [])) == list(SEEDS),
                int(obj.get("trained_networks_declared", -1)) == len(manifest),
            ]
            audit["manifest_configuration_match"] = bool(all(checks))
            if not all(checks):
                raise RuntimeError(
                    "Experiment 1 manifest exists but does not match the Experiment 2 declared factorial. "
                    f"Inspect {src_manifest} and {OUTDIR / 'audit' / 'exp1_source_audit.json'}."
                )
        except Exception as e:
            audit["manifest_audit_error"] = repr(e)
            atomic_json(audit, OUTDIR / "audit" / "exp1_source_audit.json")
            raise
    atomic_json(audit, OUTDIR / "audit" / "exp1_source_audit.json")
    return df


# ==================================================================================================
# 19. FROZEN PAIRED CONDITION POPULATION
# ==================================================================================================


def select_pair_endpoint_local_indices(data: Dict[str, Any], dataset: str, pair_count: int) -> np.ndarray:
    """Choose paired clean/shifted instances from the exact 40-condition Exp1 causal population.

    FMNIST default n=10: one item per class.
    MPI3D default n=10: one item at every fourth horizontal-position level (0,4,...,36), because
    Exp1's 40-condition causal population contains exactly one item from every horizontal level.
    """
    endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
    causal_local = np.asarray(data["causal_endpoint_local_indices"], dtype=int)
    strata_test = np.asarray(data["test_strata"])
    causal_strata = strata_test[endpoint_ids[causal_local]]

    if dataset == "fmnist" and pair_count == 10:
        selected = []
        for cls in range(10):
            ids = np.flatnonzero(causal_strata == cls)
            if len(ids) == 0:
                raise RuntimeError(f"Frozen Exp1 causal population has no FMNIST class {cls}.")
            selected.append(int(causal_local[int(ids[0])]))
        return np.asarray(selected, dtype=int)

    if dataset == "mpi3d" and pair_count == 10:
        selected = []
        for level in range(0, 40, 4):
            ids = np.flatnonzero(causal_strata == level)
            if len(ids) == 0:
                raise RuntimeError(f"Frozen Exp1 causal population has no MPI3D horizontal level {level}.")
            selected.append(int(causal_local[int(ids[0])]))
        return np.asarray(selected, dtype=int)

    # Generic deterministic fallback for deliberate pilot/scale changes: balance over strata while
    # remaining strictly inside the frozen Exp1 causal population.
    chosen_positions = balanced_indices_from_strata(
        causal_strata,
        min(int(pair_count), len(causal_local)),
        CONDITION_SEED_BASE + 7001 + (0 if dataset == "fmnist" else 1),
    )
    return causal_local[chosen_positions]


def attach_crossover_pairs(data_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ds, data in data_cache.items():
        pair_local = select_pair_endpoint_local_indices(data, ds, PAIR_COUNT)
        data["crossover_endpoint_local_indices"] = pair_local
        endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
        strata_test = np.asarray(data["test_strata"])
        for pi, ep_local in enumerate(pair_local):
            test_id = int(endpoint_ids[int(ep_local)])
            rows.append({
                "dataset": ds,
                "pair_index": int(pi),
                "endpoint_local_index": int(ep_local),
                "test_index": test_id,
                "stratum": strata_test[test_id],
                "condition_A": "clean",
                "condition_B": "shifted",
            })
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "crossover_pair_population.csv")
    return df


def prepared_crossover_environments(data: Dict[str, Any], dataset: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the SAME underlying paired items under clean and deterministic shifted observation.

    Shift is generated on the full frozen 160-endpoint population first, exactly as in Experiment 1,
    because the noise generator is shape-dependent. Only afterward are the crossover pair indices taken.
    """
    Xt, yt = data["test"]
    endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
    pair_local = np.asarray(data["crossover_endpoint_local_indices"], dtype=int)
    Xep = Xt[endpoint_ids]
    yep = yt[endpoint_ids]
    Xshift = shifted_environment(Xep, dataset, data["meta"])
    return Xep[pair_local], Xshift[pair_local], yep[pair_local]


# ==================================================================================================
# 20. CONDITION-PREFERENTIAL TARGET CONSTRUCTION
# ==================================================================================================


def select_preferential_masks(
    flow_A: List[np.ndarray],
    flow_B: List[np.ndarray],
    fraction: float,
    tol: float = PREFERENCE_TOL,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    """Construct equal-budget, disjoint c_A- and c_B-preferential targets layer by layer.

    Every layer's flow marginal is normalized on the same execution carrier, so the signed excess

        delta = F^{c_A} - F^{c_B}

    is directly comparable within that layer. We use the largest positive delta values for A and the
    most negative values for B. Tiny-flow ratio scores are deliberately avoided.

    The raw budget follows the Experiment 1 rule separately in each condition (10% of positive support),
    then is symmetrically capped by the smaller available budget and by the number of genuinely
    preferential carriers on each side. This ensures exact layer-wise equality without fabricating
    condition preference.
    """
    if len(flow_A) != len(flow_B):
        raise RuntimeError("Condition flows have different layer counts.")
    masks_A, masks_B, audit = [], [], []
    for li, (fa0, fb0) in enumerate(zip(flow_A, flow_B)):
        fa = np.asarray(fa0, float)
        fb = np.asarray(fb0, float)
        if fa.shape != fb.shape:
            raise RuntimeError(f"Layer {li}: condition-flow shapes differ: {fa.shape} vs {fb.shape}")
        a = fa.ravel()
        b = fb.ravel()
        finite = np.isfinite(a) & np.isfinite(b)
        posA = finite & (a > 0)
        posB = finite & (b > 0)
        delta = a - b
        prefA = finite & (delta > float(tol)) & (a > 0)
        prefB = finite & (delta < -float(tol)) & (b > 0)

        rawA = int(math.ceil(float(fraction) * int(np.sum(posA)))) if np.any(posA) else 0
        rawB = int(math.ceil(float(fraction) * int(np.sum(posB)))) if np.any(posB) else 0
        k = min(rawA, rawB, int(np.sum(prefA)), int(np.sum(prefB)))

        ma = np.zeros(a.shape, dtype=bool)
        mb = np.zeros(b.shape, dtype=bool)
        if k > 0:
            idsA = np.flatnonzero(prefA)
            idsB = np.flatnonzero(prefB)
            chooseA = idsA[np.argsort(-delta[idsA], kind="mergesort")[:k]]
            chooseB = idsB[np.argsort(delta[idsB], kind="mergesort")[:k]]
            ma[chooseA] = True
            mb[chooseB] = True

        if np.any(ma & mb):
            raise RuntimeError(f"Layer {li}: preferential target sets overlap despite opposite-sign construction.")
        if int(ma.sum()) != int(mb.sum()):
            raise RuntimeError(f"Layer {li}: crossover target budgets differ.")
        if np.any(ma & ~(delta > tol)) or np.any(mb & ~(delta < -tol)):
            raise RuntimeError(f"Layer {li}: target violates declared preference sign.")

        audit.append({
            "layer": int(li),
            "n_total": int(a.size),
            "n_positive_A": int(np.sum(posA)),
            "n_positive_B": int(np.sum(posB)),
            "n_preferential_A": int(np.sum(prefA)),
            "n_preferential_B": int(np.sum(prefB)),
            "raw_target_count_A": int(rawA),
            "raw_target_count_B": int(rawB),
            "n_targets_each": int(k),
            "budget_capped": bool(k < min(rawA, rawB)) if min(rawA, rawB) > 0 else False,
            "selected_A_origin_flow_mass": float(np.sum(a[ma])) if k else 0.0,
            "selected_A_other_flow_mass": float(np.sum(b[ma])) if k else 0.0,
            "selected_B_origin_flow_mass": float(np.sum(b[mb])) if k else 0.0,
            "selected_B_other_flow_mass": float(np.sum(a[mb])) if k else 0.0,
            "selected_A_mean_delta": float(np.mean(delta[ma])) if k else np.nan,
            "selected_B_mean_delta": float(np.mean(-delta[mb])) if k else np.nan,
            "flow_l1_distance": float(np.sum(np.abs(delta))),
            "flow_overlap_mass": float(np.sum(np.minimum(a[finite], b[finite]))),
            "target_jaccard": 0.0 if k else np.nan,
        })
        masks_A.append(ma.reshape(fa.shape))
        masks_B.append(mb.reshape(fb.shape))
    return masks_A, masks_B, audit


def crossover_target_selftest() -> pd.DataFrame:
    """Dataset-free unit test of the Experiment 2 target logic."""
    A = [np.array([[0.50, 0.20, 0.20, 0.10]], float)]
    B = [np.array([[0.10, 0.20, 0.30, 0.40]], float)]
    ma, mb, aud = select_preferential_masks(A, B, 0.50, tol=1e-15)
    delta = A[0] - B[0]
    rows = [
        {"check": "equal_budget", "error": abs(int(ma[0].sum()) - int(mb[0].sum())), "tolerance": 0.0},
        {"check": "disjoint_targets", "error": int(np.sum(ma[0] & mb[0])), "tolerance": 0.0},
        {"check": "A_sign", "error": int(np.sum(ma[0] & ~(delta > 0))), "tolerance": 0.0},
        {"check": "B_sign", "error": int(np.sum(mb[0] & ~(delta < 0))), "tolerance": 0.0},
        {"check": "nonzero_budget", "error": 0.0 if int(ma[0].sum()) > 0 else 1.0, "tolerance": 0.0},
    ]
    df = pd.DataFrame(rows)
    df["passed"] = df.error <= df.tolerance
    atomic_csv(df, OUTDIR / "tables" / "crossover_target_selftest.csv")
    if not bool(df.passed.all()):
        raise RuntimeError("Condition-preferential target self-test failed.")
    return df


# ==================================================================================================
# 21. EXPERIMENT 2 REAL-DATA PREFLIGHT
# ==================================================================================================


@torch.no_grad()
def preflight_crossover_one_model(row: pd.Series, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rid = str(row.run_id)
    cp = checkpoint_path(rid)
    h0 = file_sha1(cp)
    chk = torch.load(cp, map_location="cpu")
    model = build_model(str(row.arch_id), str(row.activation), output_dim_for(str(row.dataset)), str(row.dataset), int(row.seed_index))
    model.load_state_dict(chk["state_dict"])
    model.to(DEVICE).eval()
    family = str(row.family)

    X_A, X_B, ypair = prepared_crossover_environments(data, str(row.dataset))
    rows = []
    found = False
    for pi in range(len(X_A)):
        xa, xb = X_A[pi:pi+1], X_B[pi:pi+1]
        y = ypair[pi:pi+1]
        x0a, x0b = torch.zeros_like(xa), torch.zeros_like(xb)
        target = int(y.reshape(-1)[0].item()) if str(row.task) == "classification" else 0
        ba = baseline_condition(model, family, xa, x0a, target)
        bb = baseline_condition(model, family, xb, x0b, target)
        if not (bool(ba["state"]["valid"][0]) and bool(bb["state"]["valid"][0])):
            continue
        fA = flow_list_from_state(ba["state"], family)
        fB = flow_list_from_state(bb["state"], family)
        mA, mB, aud = select_preferential_masks(fA, fB, TARGET_FRACTION)
        if sum(target_counts(mA)) <= 0:
            continue
        found = True
        rows.append({"run_id":rid,"family":family,"check":"equal_total_budget","error":abs(sum(target_counts(mA))-sum(target_counts(mB))),"tolerance":0.0})
        rows.append({"run_id":rid,"family":family,"check":"layer_budget_identity","error":max(abs(a-b) for a,b in zip(target_counts(mA),target_counts(mB))),"tolerance":0.0})
        rows.append({"run_id":rid,"family":family,"check":"target_disjointness","error":max(float(np.sum(a & b)) for a,b in zip(mA,mB)),"tolerance":0.0})

        # alpha=0 must be identity for BOTH frozen origin masks under BOTH evaluation conditions.
        for origin, masks in [("A",mA),("B",mB)]:
            for ev, xx, xx0, base in [("A",xa,x0a,ba),("B",xb,x0b,bb)]:
                z = intervene_condition(model, family, xx, xx0, target, masks, base, alpha=0.0)
                mr = manifestation_record(str(row.dataset), y, target, base["out"], base["out0"], z["out"], z["out0"])
                rows.append({"run_id":rid,"family":family,"check":f"alpha0_manifestation_{origin}_{ev}","error":float(mr["manifestation_abs"]),"tolerance":PREFLIGHT_TOL})
                rows.append({"run_id":rid,"family":family,"check":f"alpha0_hellinger_{origin}_{ev}","error":float(z["hellinger"]),"tolerance":PREFLIGHT_TOL})

        # Nonzero intervention must not mutate the underlying checkpoint/model parameters.
        _ = intervene_condition(model, family, xa, x0a, target, mA, ba, alpha=ATTENUATION)
        break

    if not found:
        raise RuntimeError(f"Crossover preflight found no valid preferential target pair for sentinel {rid}.")
    h1 = file_sha1(cp)
    rows.append({"run_id":rid,"family":family,"check":"checkpoint_hash_immutable","error":0.0 if h0==h1 else 1.0,"tolerance":0.0})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def run_crossover_preflight(manifest: pd.DataFrame, data_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    preferred = manifest[
        manifest.arch_id.astype(str).isin(["M2_sustained", "C2_bottleneck"])
        & (manifest.optimizer.astype(str) == "adam")
        & (manifest.activation.astype(str) == "relu")
        & (pd.to_numeric(manifest.seed_index, errors="coerce") == 0)
    ].copy()
    rows = []
    for _, r in tqdm(preferred.iterrows(), total=len(preferred), desc="Crossover real-data preflight"):
        rows.extend(preflight_crossover_one_model(r, data_cache[str(r.dataset)]))
    df = pd.DataFrame(rows)
    df["passed"] = pd.to_numeric(df.error, errors="coerce") <= pd.to_numeric(df.tolerance, errors="coerce") + 1e-15
    atomic_csv(df, OUTDIR / "tables" / "crossover_preflight.csv")
    bad = df[~df.passed.astype(bool)]
    if len(bad):
        print(df.to_string(index=False))
        raise RuntimeError(f"Experiment 2 crossover preflight failed {len(bad)} check(s).")
    print(f"[crossover preflight] all {len(df)} real-data checks passed.")
    return df


# ==================================================================================================
# 22. PER-MODEL CROSSOVER CAMPAIGN
# ==================================================================================================


def exp2_completed_keys(run_id: str) -> set:
    d = safe_read_csv(exp2_part_path(run_id, "validity"))
    if d.empty:
        return set()
    keys = set()
    for _, r in d.iterrows():
        try:
            keys.add((int(r.pair_index), str(r.target_origin), str(r.evaluation_condition)))
        except Exception:
            pass
    return keys


def exp2_load_parts(run_id: str):
    m = safe_read_csv(exp2_part_path(run_id, "measurements"))
    v = safe_read_csv(exp2_part_path(run_id, "validity"))
    a = safe_read_csv(exp2_part_path(run_id, "target_audit"))
    return (
        m.to_dict("records") if len(m) else [],
        v.to_dict("records") if len(v) else [],
        a.to_dict("records") if len(a) else [],
    )


def exp2_flush(run_id: str, measurements, validity, audit):
    atomic_csv(pd.DataFrame(measurements), exp2_part_path(run_id, "measurements"))
    atomic_csv(pd.DataFrame(validity), exp2_part_path(run_id, "validity"))
    atomic_csv(pd.DataFrame(audit), exp2_part_path(run_id, "target_audit"))


@torch.no_grad()
def run_crossover_one_model(row: pd.Series, data: Dict[str, Any]):
    rid = str(row.run_id)
    if RESUME and exp2_done_path(rid).exists():
        return
    cp = checkpoint_path(rid)
    cp_hash_before = file_sha1(cp)
    chk = torch.load(cp, map_location="cpu")
    model = build_model(str(row.arch_id), str(row.activation), output_dim_for(str(row.dataset)), str(row.dataset), int(row.seed_index))
    model.load_state_dict(chk["state_dict"])
    model.to(DEVICE).eval()
    family = str(row.family)

    measurements, validity, audit = exp2_load_parts(rid) if RESUME else ([], [], [])
    done = exp2_completed_keys(rid) if RESUME else set()
    X_A, X_B, ypair = prepared_crossover_environments(data, str(row.dataset))
    endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
    pair_local = np.asarray(data["crossover_endpoint_local_indices"], dtype=int)
    strata_test = np.asarray(data["test_strata"])

    bar = tqdm(total=len(X_A), desc=f"EXP2 {rid[:48]}", leave=False)
    since_flush = 0
    for pi in range(len(X_A)):
        expected = [(pi,o,e) for o in ["clean","shifted"] for e in ["clean","shifted"]]
        if all(k in done for k in expected):
            bar.update(1)
            continue

        xa, xb = X_A[pi:pi+1], X_B[pi:pi+1]
        y = ypair[pi:pi+1]
        x0a, x0b = torch.zeros_like(xa), torch.zeros_like(xb)
        target = int(y.reshape(-1)[0].item()) if str(row.task) == "classification" else 0
        ba = baseline_condition(model, family, xa, x0a, target)
        bb = baseline_condition(model, family, xb, x0b, target)
        base_defined = bool(ba["state"]["valid"][0]) and bool(bb["state"]["valid"][0])
        test_id = int(endpoint_ids[int(pair_local[pi])])
        stratum = strata_test[test_id]
        common = {
            "run_id": rid,
            "dataset": str(row.dataset),
            "task": str(row.task),
            "family": family,
            "arch_id": str(row.arch_id),
            "optimizer": str(row.optimizer),
            "activation": str(row.activation),
            "seed_index": int(row.seed_index),
            "pair_index": int(pi),
            "endpoint_local_index": int(pair_local[pi]),
            "test_index": test_id,
            "stratum": stratum,
            "target_output": int(target),
            "target_fraction": float(TARGET_FRACTION),
            "attenuation_alpha": float(ATTENUATION),
        }

        if not base_defined:
            for _, origin, ev in expected:
                k = (pi,origin,ev)
                if k in done:
                    continue
                validity.append({
                    **common,"target_origin":origin,"evaluation_condition":ev,
                    "baseline_A_defined":bool(ba["state"]["valid"][0]),
                    "baseline_B_defined":bool(bb["state"]["valid"][0]),
                    "intervention_applicable":False,"intervened_realization_defined":False,
                    "n_targets_total":0,"status":"not_applicable_baseline_undefined",
                })
                done.add(k)
            since_flush += 1
            if since_flush >= SAVE_EVERY_PAIRS:
                exp2_flush(rid, measurements, validity, audit); since_flush = 0
            bar.update(1)
            continue

        flowA = flow_list_from_state(ba["state"], family)
        flowB = flow_list_from_state(bb["state"], family)
        masksA, masksB, aud = select_preferential_masks(flowA, flowB, TARGET_FRACTION)
        countsA, countsB = target_counts(masksA), target_counts(masksB)
        if countsA != countsB:
            raise RuntimeError(f"{rid} pair {pi}: target budgets differ after construction.")
        ntargets = int(sum(countsA))

        for li, d in enumerate(aud):
            audit.append({
                **common,
                **d,
                "clean_target_hash": hash_indices(masksA[li]),
                "shifted_target_hash": hash_indices(masksB[li]),
            })

        if ntargets <= 0:
            for _, origin, ev in expected:
                k = (pi,origin,ev)
                if k in done: continue
                validity.append({
                    **common,"target_origin":origin,"evaluation_condition":ev,
                    "baseline_A_defined":True,"baseline_B_defined":True,
                    "intervention_applicable":False,"intervened_realization_defined":False,
                    "n_targets_total":0,"status":"not_applicable_no_preferential_support",
                })
                done.add(k)
            since_flush += 1
            if since_flush >= SAVE_EVERY_PAIRS:
                exp2_flush(rid, measurements, validity, audit); since_flush = 0
            bar.update(1)
            continue

        origin_masks = {"clean":masksA, "shifted":masksB}
        eval_data = {
            "clean": (xa, x0a, ba),
            "shifted": (xb, x0b, bb),
        }
        for origin in ["clean","shifted"]:
            for ev in ["clean","shifted"]:
                key = (pi,origin,ev)
                if key in done:
                    continue
                xx, xx0, base = eval_data[ev]
                t0 = time.perf_counter()
                result = intervene_condition(model, family, xx, xx0, target, origin_masks[origin], base, ATTENUATION)
                elapsed = time.perf_counter() - t0
                int_valid = bool(result["valid"])
                validity.append({
                    **common,"target_origin":origin,"evaluation_condition":ev,
                    "origin_matches_evaluation":bool(origin==ev),
                    "baseline_A_defined":True,"baseline_B_defined":True,
                    "intervention_applicable":True,"intervened_realization_defined":int_valid,
                    "intervention_induced_degeneracy":bool(not int_valid),
                    "n_targets_total":ntargets,
                    "status":"defined" if int_valid else "manifestation_only_intervened_zero_support",
                    "intervention_seconds":float(elapsed),
                })
                mrec = manifestation_record(
                    str(row.dataset), y, target,
                    base["out"], base["out0"], result["out"], result["out0"]
                )
                rec = {
                    **common,"target_origin":origin,"evaluation_condition":ev,
                    "origin_matches_evaluation":bool(origin==ev),
                    "n_targets_total":ntargets,
                    **mrec,
                    "baseline_eta1":float(base["state"]["eta1"][0]),
                    "baseline_eta2":float(base["state"]["eta2"][0]),
                    "baseline_Sdef":float(base["state"]["entropy_deficit"][0]),
                    "baseline_Sdef_norm":float(base["state"]["entropy_deficit_norm"][0]),
                    "path_hellinger":float(result["hellinger"]) if np.isfinite(result["hellinger"]) else np.nan,
                    "intervention_seconds":float(elapsed),
                }
                if int_valid:
                    rec.update({
                        "intervened_eta1":float(result["state"]["eta1"][0]),
                        "intervened_eta2":float(result["state"]["eta2"][0]),
                        "intervened_Sdef":float(result["state"]["entropy_deficit"][0]),
                        "intervened_Sdef_norm":float(result["state"]["entropy_deficit_norm"][0]),
                        "delta_eta1":float(result["state"]["eta1"][0]-base["state"]["eta1"][0]),
                        "delta_eta2":float(result["state"]["eta2"][0]-base["state"]["eta2"][0]),
                        "delta_Sdef":float(result["state"]["entropy_deficit"][0]-base["state"]["entropy_deficit"][0]),
                        "delta_Sdef_norm":float(result["state"]["entropy_deficit_norm"][0]-base["state"]["entropy_deficit_norm"][0]),
                    })
                else:
                    for c in ["intervened_eta1","intervened_eta2","intervened_Sdef","intervened_Sdef_norm","delta_eta1","delta_eta2","delta_Sdef","delta_Sdef_norm"]:
                        rec[c] = np.nan
                measurements.append(rec)
                done.add(key)

        since_flush += 1
        if since_flush >= SAVE_EVERY_PAIRS:
            exp2_flush(rid, measurements, validity, audit); since_flush = 0
        bar.set_postfix(pair=pi+1, targets=ntargets, stratum=str(stratum))
        bar.update(1)

    bar.close()
    exp2_flush(rid, measurements, validity, audit)
    cp_hash_after = file_sha1(cp)
    if cp_hash_before != cp_hash_after:
        raise RuntimeError(f"Experiment 1 source checkpoint changed during crossover: {cp}")
    atomic_json({
        "run_id":rid,"completed":True,"checkpoint_sha1":cp_hash_before,
        "n_measurement_rows":len(measurements),"n_validity_rows":len(validity),
        "n_target_audit_rows":len(audit),"timestamp_unix":time.time(),
    }, exp2_done_path(rid))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ==================================================================================================
# 23. AGGREGATION, RQ3 CONTRASTS, AND INFERENCE
# ==================================================================================================


def aggregate_exp2_parts(manifest: pd.DataFrame) -> Dict[str,pd.DataFrame]:
    mm,vv,aa = [],[],[]
    for rid in manifest.run_id.astype(str):
        for kind,dest in [("measurements",mm),("validity",vv),("target_audit",aa)]:
            d = safe_read_csv(exp2_part_path(rid,kind))
            if len(d): dest.append(d)
    out = {
        "measurements":pd.concat(mm,ignore_index=True) if mm else pd.DataFrame(),
        "validity":pd.concat(vv,ignore_index=True) if vv else pd.DataFrame(),
        "target_audit":pd.concat(aa,ignore_index=True) if aa else pd.DataFrame(),
    }
    atomic_csv(out["measurements"], OUTDIR/"tables"/"crossover_measurements.csv")
    atomic_csv(out["validity"], OUTDIR/"tables"/"crossover_validity.csv")
    atomic_csv(out["target_audit"], OUTDIR/"tables"/"crossover_target_audit.csv")
    return out


def validate_exp2_tables(tabs: Dict[str,pd.DataFrame]):
    m,v = tabs["measurements"],tabs["validity"]
    if m.empty or v.empty: raise RuntimeError("Experiment 2 tables are empty.")
    keys=["run_id","pair_index","target_origin","evaluation_condition"]
    if v.duplicated(keys).any(): raise RuntimeError("Duplicate Experiment 2 validity keys detected.")
    applicable=v[v.intervention_applicable.astype(bool)]
    vk=set(tuple(x) for x in applicable[keys].to_records(index=False))
    mk=set(tuple(x) for x in m[keys].to_records(index=False))
    if not mk.issubset(vk): raise RuntimeError("Experiment 2 measurement without applicable validity record.")
    q=applicable.pivot_table(index=["run_id","pair_index"],columns=["target_origin","evaluation_condition"],values="n_targets_total",aggfunc="first")
    if len(q):
        row_nunique=q.nunique(axis=1)
        if bool((row_nunique>1).any()):
            raise RuntimeError(f"Target-budget mismatch across 2x2 crossover cells in {int((row_nunique>1).sum())} pair(s).")
    print(
        f"[EXP2 tables] measurements={len(m)} | validity={len(v)} | "
        f"applicable={len(applicable)} | degeneracies="
        f"{int(v.get('intervention_induced_degeneracy',pd.Series(dtype=bool)).fillna(False).astype(bool).sum())}"
    )


def pair_level_crossover_contrasts(m: pd.DataFrame) -> pd.DataFrame:
    metrics=[x for x in ["manifestation_abs","manifestation_relative","finite_response_vector_l2","path_hellinger","task_loss_change"] if x in m.columns]
    idcols=["run_id","dataset","family","arch_id","optimizer","activation","seed_index","pair_index","stratum"]
    rows=[]
    for metric in metrics:
        q=m[idcols+["target_origin","evaluation_condition",metric]].copy()
        q[metric]=pd.to_numeric(q[metric],errors="coerce")
        piv=q.pivot_table(index=idcols,columns=["target_origin","evaluation_condition"],values=metric,aggfunc="first")
        required=[("clean","clean"),("clean","shifted"),("shifted","clean"),("shifted","shifted")]
        for idx,r in piv.iterrows():
            if not all(c in piv.columns for c in required):
                continue
            D_AA=float(r[("clean","clean")]); D_AB=float(r[("clean","shifted")])
            D_BA=float(r[("shifted","clean")]); D_BB=float(r[("shifted","shifted")])
            if not all(np.isfinite([D_AA,D_AB,D_BA,D_BB])): continue
            meta={c:v for c,v in zip(idcols, idx if isinstance(idx,tuple) else (idx,))}
            Aadv=D_AA-D_BA
            Badv=D_BB-D_AB
            rows.append({
                **meta,"metric":metric,
                "D_AA":D_AA,"D_AB":D_AB,"D_BA":D_BA,"D_BB":D_BB,
                "clean_origin_advantage":Aadv,
                "shifted_origin_advantage":Badv,
                "crossover_interaction":Aadv+Badv,
                "crossover_mean_advantage":0.5*(Aadv+Badv),
                "matched_origin_mean":0.5*(D_AA+D_BB),
                "cross_origin_mean":0.5*(D_BA+D_AB),
                # secondary same-target cross-condition contrasts, not primary because response scale differs
                "clean_target_same_condition_advantage":D_AA-D_AB,
                "shifted_target_same_condition_advantage":D_BB-D_BA,
            })
    out=pd.DataFrame(rows)
    atomic_csv(out,OUTDIR/"tables"/"pair_level_crossover_contrasts.csv")
    return out


def model_level_crossover(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty: return pd.DataFrame()
    group=["run_id","dataset","family","arch_id","optimizer","activation","seed_index","metric"]
    vals=["clean_origin_advantage","shifted_origin_advantage","crossover_interaction","crossover_mean_advantage","matched_origin_mean","cross_origin_mean","clean_target_same_condition_advantage","shifted_target_same_condition_advantage"]
    out=pair_df.groupby(group,dropna=False)[vals].median().reset_index()
    counts=pair_df.groupby(group,dropna=False).size().rename("n_pairs").reset_index()
    out=out.merge(counts,on=group,how="left")
    atomic_csv(out,OUTDIR/"tables"/"model_level_crossover.csv")
    return out


def rq3_tests(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df.empty: return pd.DataFrame()
    q=model_df[model_df.metric=="manifestation_abs"].copy()
    rows=[]
    group_specs=[]
    # overall, family-specific, activation-specific, and family x activation. Activation is predeclared
    # because Experiment 1 established substantial activation dependence.
    for ds in sorted(q.dataset.unique()):
        dds=q[q.dataset==ds]
        group_specs.append((ds,"overall","all",dds))
        for fam in ["mlp","cnn"]:
            group_specs.append((ds,"family",fam,dds[dds.family==fam]))
        for act in ["relu","tanh","silu"]:
            group_specs.append((ds,"activation",act,dds[dds.activation==act]))
        for fam in ["mlp","cnn"]:
            for act in ["relu","tanh","silu"]:
                group_specs.append((ds,"family_activation",f"{fam}:{act}",dds[(dds.family==fam)&(dds.activation==act)]))

    contrasts=["clean_origin_advantage","shifted_origin_advantage","crossover_mean_advantage"]
    for ds,stype,sname,d in group_specs:
        for contrast in contrasts:
            vals=pd.to_numeric(d[contrast],errors="coerce").to_numpy(float)
            vals=vals[np.isfinite(vals)]
            if len(vals)<3: continue
            try:
                stat,p=wilcoxon(vals,zero_method="wilcox",alternative="greater")
            except Exception:
                stat,p=np.nan,1.0
            lo,hi=bootstrap_median_ci(vals,seed=8801+len(rows))
            rows.append({
                "dataset":ds,"stratum_type":stype,"stratum":sname,"contrast":contrast,
                "alternative":">0","n_trained_models":int(len(vals)),
                "median_difference":float(np.median(vals)),
                "hodges_lehmann":hodges_lehmann_onesample(vals),
                "rank_biserial":paired_rank_biserial(vals),
                "ci_low":lo,"ci_high":hi,"wilcoxon_statistic":stat,"p_raw":p,
                "positive_fraction":float(np.mean(vals>0)),
            })
    out=pd.DataFrame(rows)
    if len(out):
        out["p_holm"]=np.nan
        # Correction within dataset and contrast across the declared reporting strata.
        for _,g in out.groupby(["dataset","contrast"]):
            out.loc[g.index,"p_holm"]=holm_adjust(pd.to_numeric(g.p_raw,errors="coerce").fillna(1.0).to_numpy(float))
    atomic_csv(out,OUTDIR/"tables"/"RQ3_primary_crossover_tests.csv")
    return out


def target_specificity_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty: return pd.DataFrame()
    numeric=["n_targets_each","flow_l1_distance","flow_overlap_mass","selected_A_mean_delta","selected_B_mean_delta","selected_A_origin_flow_mass","selected_A_other_flow_mass","selected_B_origin_flow_mass","selected_B_other_flow_mass"]
    numeric=[c for c in numeric if c in audit.columns]
    group=["dataset","family","activation"]
    out=audit.groupby(group,dropna=False)[numeric].median().reset_index()
    atomic_csv(out,OUTDIR/"tables"/"crossover_target_specificity_summary.csv")
    return out


# ==================================================================================================
# 24. LIGHTWEIGHT EXPERIMENT 2 FIGURES
# ==================================================================================================


def make_exp2_figures(pair_df: pd.DataFrame, model_df: pd.DataFrame):
    if not RUN_FIGURES or pair_df.empty or model_df.empty: return
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix","font.size":9,"pdf.fonttype":42,"savefig.dpi":600})
    def save(fig,name):
        fig.savefig(OUTDIR/"figures"/f"{name}.pdf",bbox_inches="tight")
        fig.savefig(OUTDIR/"figures"/f"{name}.png",bbox_inches="tight",dpi=600)
        plt.close(fig)

    q=model_df[model_df.metric=="manifestation_abs"].copy()
    if len(q):
        for ds in sorted(q.dataset.unique()):
            d=q[q.dataset==ds]
            fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
            for ax,col,title in [
                (axs[0],"clean_origin_advantage","clean evaluation: clean-origin minus shifted-origin"),
                (axs[1],"shifted_origin_advantage","shifted evaluation: shifted-origin minus clean-origin"),
            ]:
                vals=pd.to_numeric(d[col],errors="coerce").dropna().to_numpy(float)
                ax.axhline(0,lw=.8,color="black")
                ax.scatter(np.arange(len(vals)),np.sort(vals),s=10,alpha=.65)
                ax.set_ylabel("model-level median manifestation difference")
                ax.set_title(title)
            save(fig,f"fig01_{ds}_condition_matched_origin_advantage")

        fig,ax=plt.subplots(figsize=(5.0,3.4),constrained_layout=True)
        labels=[]; vals=[]
        for ds in sorted(q.dataset.unique()):
            for act in ["relu","tanh","silu"]:
                z=pd.to_numeric(q[(q.dataset==ds)&(q.activation==act)].crossover_mean_advantage,errors="coerce").dropna().to_numpy(float)
                if len(z): labels.append(f"{ds}\n{act}"); vals.append(z)
        ax.boxplot(vals,labels=labels,showfliers=False)
        ax.axhline(0,lw=.8,color="black")
        ax.set_ylabel("crossover mean advantage")
        save(fig,"fig02_RQ3_crossover_by_activation")


# ==================================================================================================
# 25. EXPERIMENT 2 MANIFEST
# ==================================================================================================


def write_exp2_manifest(manifest: pd.DataFrame, pair_df: pd.DataFrame):
    obj={
        "study":"Work II Experiment 2 condition crossover",
        "research_question":"RQ3: Is computational responsibility condition-dependent?",
        "source_exp1_results":str(EXP1_RESULTS),
        "source_checkpoints":str(SOURCE_CHECKPOINT_DIR),
        "output_dir":str(OUTDIR),
        "data_root":str(DATA_ROOT),
        "device":DEVICE,"cpu_threads":CPU_THREADS,
        "trained_networks_reused":int(len(manifest)),
        "retraining":False,
        "datasets":DATASETS,"architectures":ARCH_IDS,"optimizers":OPTIMIZERS,"activations":ACTIVATIONS,"seeds":SEEDS,
        "condition_A":"c_A=(x_clean,0)","condition_B":"c_B=(x_shifted,0)",
        "same_underlying_item":True,"shift_snr_db":float(SHIFT_SNR_DB),
        "pairs_per_model":int(PAIR_COUNT),
        "declared_intervention_evaluations":int(len(manifest)*PAIR_COUNT*4),
        "attenuation_alpha":float(ATTENUATION),
        "target_fraction_basis":float(TARGET_FRACTION),
        "target_score":"signed layer-normalized realized-flow excess delta = F_clean - F_shifted",
        "budget_rule":"same layer-wise cardinality for clean-origin and shifted-origin targets; raw 10% positive-support budgets symmetrically capped by available preferential support",
        "primary_manifestation":"absolute displacement of the same output-conditioned finite response used in Experiment 1",
        "primary_RQ3_test":"within-evaluation-condition target-origin x evaluation-condition crossover interaction",
        "primary_contrasts":{
            "clean_origin_advantage":"D_AA-D_BA",
            "shifted_origin_advantage":"D_BB-D_AB",
            "crossover_mean_advantage":"0.5*((D_AA-D_BA)+(D_BB-D_AB))",
        },
        "secondary_internal_metrics":["path Hellinger","eta1","eta2","entropy deficit"],
        "pair_population_rows":int(len(pair_df)),
        "geometry_included":False,"dose_response_included":False,"matched_control_included":False,
    }
    atomic_json(obj,OUTDIR/"experiment_manifest.json")


# ==================================================================================================
# 26. ORCHESTRATION
# ==================================================================================================


def run_experiment2():
    start=time.perf_counter()
    banner("WORK II — EXPERIMENT 2: CONDITION CROSSOVER (RQ3)")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | total VRAM: {p.total_memory/1024**3:.1f} GiB")
    print(f"Slurm job ID: {os.environ.get('SLURM_JOB_ID','not-set')}")
    print(f"Slurm node: {os.environ.get('SLURMD_NODENAME',os.environ.get('HOSTNAME','unknown'))}")
    print(f"SLURM_SUBMIT_DIR: {os.environ.get('SLURM_SUBMIT_DIR','not-set')}")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Experiment 1 results: {EXP1_RESULTS}")
    print(f"Source checkpoints: {SOURCE_CHECKPOINT_DIR}")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Experiment 2 output: {OUTDIR}")
    print(f"Pairs/model={PAIR_COUNT} | alpha={ATTENUATION:.3f} | target fraction={TARGET_FRACTION:.3f}")
    print(f"Soft wall={SOFT_WALL_HOURS:.2f} h")

    banner("Fail-fast dataset-free operator and crossover-target self-tests")
    op=run_operator_selftest()
    targtest=crossover_target_selftest()
    if SELFTEST_ONLY:
        print("SELFTEST_ONLY=1: no source checkpoint or dataset was touched.")
        return {"operator_selftest":op,"crossover_target_selftest":targtest,"selftest_only":True}

    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("Experiment 2 full campaign requires CUDA. Set PATHFLOW_CROSSOVER_REQUIRE_CUDA=0 only for an intentional debug run.")

    manifest=build_manifest()
    banner("Verify immutable Experiment 1 checkpoint population")
    inventory=verify_exp1_source(manifest)
    print(f"Checkpoint inventory: {int(inventory.checkpoint_exists.sum())}/{len(inventory)} present")

    banner("Reconstruct datasets and exact frozen Experiment 1 endpoint population")
    data_cache=load_datasets()
    pair_population=attach_crossover_pairs(data_cache)
    for ds,d in data_cache.items():
        X_A,X_B,y=prepared_crossover_environments(d,ds)
        strata=pair_population[pair_population.dataset==ds].stratum.tolist()
        print(f"{ds}: paired n={len(X_A)} | strata={strata}")
    write_exp2_manifest(manifest,pair_population)

    banner("Checkpoint/operator fidelity preflight from Experiment 1 machinery")
    base_preflight=run_preflight(manifest,data_cache) if RUN_PREFLIGHT else pd.DataFrame()
    banner("Experiment 2 crossover-specific real-data preflight")
    cross_preflight=run_crossover_preflight(manifest,data_cache)
    if PREFLIGHT_ONLY:
        print("PREFLIGHT_ONLY=1: all source/operator/crossover checks passed; full RQ3 campaign was not started.")
        return {"manifest":manifest,"inventory":inventory,"data":data_cache,"operator_selftest":op,"base_preflight":base_preflight,"crossover_preflight":cross_preflight,"preflight_only":True}

    banner("Run / resume RQ3 condition-crossover campaign")
    todo=[r for _,r in manifest.iterrows() if not (RESUME and exp2_done_path(str(r.run_id)).exists())]
    eta=ETAReporter("EXP2 crossover models",len(todo))
    completed=0
    for row in tqdm(todo,total=len(todo),desc="EXP2 models"):
        if not can_start_new_unit("Experiment 2 crossover model"):
            break
        t0=time.perf_counter()
        run_crossover_one_model(row,data_cache[str(row.dataset)])
        completed+=1
        eta.update(time.perf_counter()-t0,completed)
    eta.finish()

    n_done=sum(exp2_done_path(str(rid)).exists() for rid in manifest.run_id.astype(str))
    tabs=aggregate_exp2_parts(manifest)
    if n_done!=len(manifest):
        msg=[
            "WORK II EXPERIMENT 2 — CLEAN SOFT-WALL STOP",
            f"Crossover models complete: {n_done}/{len(manifest)}",
            f"Measurement rows safely persisted: {len(tabs['measurements'])}",
            f"Wall time: {human_time(time.perf_counter()-start)}",
            "Resume the same file in another Slurm allocation; completed model/pair parts will be reused.",
            f"Output: {OUTDIR}",
        ]
        (OUTDIR/"experiment_summary.txt").write_text("\n".join(msg)+"\n",encoding="utf-8")
        print("\n"+"\n".join(msg))
        return {"manifest":manifest,"tables":tabs,"incomplete":True}

    banner("Validate and analyze RQ3 crossover")
    validate_exp2_tables(tabs)
    pair_contrasts=pair_level_crossover_contrasts(tabs["measurements"])
    model_contrasts=model_level_crossover(pair_contrasts)
    tests=rq3_tests(model_contrasts)
    target_summary=target_specificity_summary(tabs["target_audit"])
    make_exp2_figures(pair_contrasts,model_contrasts)

    elapsed=time.perf_counter()-start
    applicable=int(tabs["validity"].intervention_applicable.astype(bool).sum()) if len(tabs["validity"]) else 0
    summary=[
        "WORK II EXPERIMENT 2 CONDITION CROSSOVER COMPLETE",
        "RQ3: Is computational responsibility condition-dependent?",
        f"Device: {DEVICE}",
        f"Wall time: {human_time(elapsed)}",
        f"Frozen Experiment 1 checkpoints reused: {len(manifest)}/{len(manifest)}",
        f"Paired conditions/model: {PAIR_COUNT}",
        f"Declared interventions: {len(manifest)*PAIR_COUNT*4}",
        f"Measurement rows: {len(tabs['measurements'])}",
        f"Validity rows: {len(tabs['validity'])}",
        f"Applicable interventions: {applicable}",
        f"Pair-level contrast rows: {len(pair_contrasts)}",
        f"Model-level contrast rows: {len(model_contrasts)}",
        f"RQ3 test rows: {len(tests)}",
        f"Output: {OUTDIR}",
    ]
    (OUTDIR/"experiment_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8")
    print("\n"+"\n".join(summary))
    return {
        "manifest":manifest,"inventory":inventory,"data":data_cache,
        "operator_selftest":op,"base_preflight":base_preflight,"crossover_preflight":cross_preflight,
        "tables":tabs,"pair_contrasts":pair_contrasts,"model_contrasts":model_contrasts,
        "RQ3_tests":tests,"target_summary":target_summary,
    }


if __name__ == "__main__":
    CROSSOVER_RESULTS = run_experiment2()
