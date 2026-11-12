from __future__ import annotations

# ==================================================================================================
# WORK II — EXPERIMENT 5.1
# TRANSFORMATION SYMMETRY OF THE FIELD-A CONDITIONAL REALIZATION OBSERVABLE
# MPI3D-TOY + FASHION-MNIST | FROZEN CNN CHECKPOINTS | HPC/SLURM SAFE | NO RETRAINING | NO INTERVENTION
# ==================================================================================================
#
# SCIENTIFIC QUESTION
# -------------------
# Does the spatial organization of conditional computational realization transform coherently when the
# presentation of the same task-relevant input is translated?
#
# Field A is the already-established input-origin marginal of the complete normalized conditional path
# measure:
#
#       P_in^c(s) = sum_k F^c_(input node k,s),       sum_s P_in^c(s) = 1
#
# Experiment 5.1 does NOT add an intervention.  It extends the Field-A observable introduced in
# Experiment 5.  The trained CNN checkpoint is fixed.  The declared reference remains x0 = 0 in the
# network's normalized input coordinates.  Only the presentation x is translated.
#
# For an integer translation T_g, the primary spatial hypothesis is equivariance:
#
#       P_(T_g x)  ~=  T_g P_x
#
# The direct competitor is spatial invariance:
#
#       P_(T_g x)  ~=  P_x
#
# Primary row-level quantity:
#
#       equivariance_advantage = H(P_(T_g x), P_x) - H(P_(T_g x), T_g P_x)
#
# Positive values mean that the realized field follows the translated input more closely than it remains
# fixed at the original coordinates.
#
# A deliberately incorrect inverse translation is also retained as a direction control:
#
#       wrong_direction_advantage
#           = H(P_(T_g x), T_(-g) P_x) - H(P_(T_g x), T_g P_x)
#
# Positive values mean the correct transformation direction fits better than the opposite direction.
#
# TRANSFORMATIONS
# ---------------
# Fashion-MNIST:
#   horizontal and vertical integer translations by +/-1 and +/-2 pixels.
#
# MPI3D-Toy:
#   vertical integer translations by +/-1 and +/-2 pixels only.
#   Horizontal translation is deliberately excluded because the benchmark regression target is the
#   horizontal-axis factor itself.
#
# Identity translation (0,0) is recomputed as a numerical/deterministic sentinel and is not included in
# the primary inference.
#
# FINITE-IMAGE BOUNDARY RULE
# --------------------------
# Translations are applied in NORMALIZED INPUT COORDINATES with zero fill.  This is deliberate because
# the declared reference is x0=0, so T_g(x0)=x0 exactly.  The same zero-fill translation is applied to
# Field A.  Since finite-frame translation can move some Field-A probability mass outside the 16x16
# lattice, the transported baseline field is renormalized over the mass retained in-frame before Hellinger
# comparison.  The retained mass is always recorded and never hidden.
#
# OUTPUT BEHAVIOUR
# ----------------
# Fashion-MNIST additionally records whether the predicted class is unchanged, target probability drift,
# and target-margin drift.
#
# MPI3D records prediction drift in normalized target units and in original horizontal-axis units.
# No arbitrary binary regression-stability threshold is introduced.
#
# INFERENTIAL UNIT
# ----------------
# Conditions and translations are repeated observations.  The trained CNN is the inferential block.
# Condition/translation effects are first aggregated within model and environment.  Population tests are
# then performed on those model-level summaries.
#
# REQUIRED EXISTING FILES
# -----------------------
# Only immutable Experiment-1 resources are computational dependencies:
#
#   <EXP1_RESULTS>/checkpoints/*.pt
#   <EXP1_RESULTS>/data/
#
# If Experiment 5's condition-population table exists, it is used as a strict audit so Experiment 5.1
# cannot silently drift to a different 10-condition cohort.
#
# DEFAULT OUTPUT — STRICTLY ISOLATED
# ----------------------------------
# The script is intended to sit in the same working directory as the earlier experiment scripts and writes:
#
#   pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results/
#       experiment5_1_fieldA_realization_symmetry/
#           tables/
#           figures/
#           fields/
#           run_parts/
#           audit/
#
# There is NO destructive start-fresh mode.  All scientific files are write-once and content checked.
# Experiment 1, 2, 3, 4 and 5 folders are never modified.
#
# HPC EXAMPLE
# -----------
#   #SBATCH --time=48:00:00
#   #SBATCH --ntasks=1
#   #SBATCH --cpus-per-task=4
#   #SBATCH --mem=64G
#   #SBATCH --gres=gpu:1
#
#   srun python -u exp5_1_fieldA_realization_symmetry_SLURM_SAFE_v1.py
#
# Useful modes:
#
#   PATHFLOW_EXP51_SELFTEST_ONLY=1
#   PATHFLOW_EXP51_PREFLIGHT_ONLY=1
#   PATHFLOW_EXP51_AGGREGATE_ONLY=1
#   PATHFLOW_EXP51_FIGURES=0
#   PATHFLOW_EXP51_ALLOW_DOWNLOAD=1   # only if the immutable source dataset cache is genuinely absent
#
# ==================================================================================================

import os
import math
import json
import time
import random
import hashlib
import shutil
import urllib.request
import warnings
import tempfile
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.datasets import FashionMNIST
except Exception as e:
    raise RuntimeError("torchvision is required for Fashion-MNIST") from e

from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================================================================================
# 0. CONFIGURATION
# ==================================================================================================

EXP51_VERSION = "EXP5_1_FIELDA_TRANSLATION_SYMMETRY_v1"

PROFILE = os.environ.get("PATHFLOW_EXP51_PROFILE", "l4_48h").strip().lower()
RESUME = os.environ.get("PATHFLOW_EXP51_RESUME", "1") != "0"
RUN_PREFLIGHT = os.environ.get("PATHFLOW_EXP51_PREFLIGHT", "1") != "0"
PREFLIGHT_ONLY = os.environ.get("PATHFLOW_EXP51_PREFLIGHT_ONLY", "0") == "1"
AGGREGATE_ONLY = os.environ.get("PATHFLOW_EXP51_AGGREGATE_ONLY", "0") == "1"
RUN_FIGURES = os.environ.get("PATHFLOW_EXP51_FIGURES", "1") != "0"
ALLOW_DOWNLOAD = os.environ.get("PATHFLOW_EXP51_ALLOW_DOWNLOAD", "0") != "0"
SELFTEST_ONLY = os.environ.get("PATHFLOW_EXP51_SELFTEST_ONLY", "0") == "1"
REQUIRE_CUDA = os.environ.get("PATHFLOW_EXP51_REQUIRE_CUDA", "1") != "0"

SOFT_WALL_HOURS = float(os.environ.get("PATHFLOW_EXP51_SOFT_WALL_HOURS", "46.0"))
SOFT_START_BUFFER_MIN = float(os.environ.get("PATHFLOW_EXP51_SOFT_START_BUFFER_MIN", "10.0"))
SCRIPT_START_MONOTONIC = time.perf_counter()

SEED = 730031
MODEL_SEED_BASE = 271828
NOISE_SEED_BASE = 161803
CONDITION_SEED_BASE = 244949

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("PATHFLOW_EXP51_CPU_THREADS", "4")))
torch.set_num_threads(max(1, CPU_THREADS))

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
torch.set_float32_matmul_precision("high")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def _script_or_submit_dir() -> Path:
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

_exp1_env = os.environ.get("PATHFLOW_EXP1_RESULTS", "").strip()
EXP1_RESULTS = (
    Path(_exp1_env).expanduser().resolve()
    if _exp1_env
    else SCRIPT_DIR / "pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results"
)

SOURCE_CHECKPOINT_DIR = EXP1_RESULTS / "checkpoints"

_data_env = os.environ.get("PATHFLOW_EXP51_DATA_ROOT", "").strip()
DATA_ROOT = Path(_data_env).expanduser().resolve() if _data_env else EXP1_RESULTS / "data"

_out_env = os.environ.get("PATHFLOW_EXP51_OUTDIR", "").strip()
OUTDIR = (
    Path(_out_env).expanduser().resolve()
    if _out_env
    else EXP1_RESULTS / "experiment5_1_fieldA_realization_symmetry"
)

# Hard non-destructive guard against accidentally pointing this script at a prior experiment folder.
_FORBIDDEN_OUTPUTS = {
    EXP1_RESULTS.resolve(),
    (EXP1_RESULTS / "experiment2_condition_crossover").resolve(),
    (EXP1_RESULTS / "experiment3_dose_response").resolve(),
    (EXP1_RESULTS / "experiment4_multiscale_followup").resolve(),
    (EXP1_RESULTS / "experiment5_cnn_occurrence_shared_identity").resolve(),
}
if OUTDIR.resolve() in _FORBIDDEN_OUTPUTS:
    raise RuntimeError(
        f"Refusing unsafe PATHFLOW_EXP51_OUTDIR={OUTDIR}. "
        "Experiment 5.1 must have its own dedicated output subtree."
    )

for sub in ["tables", "figures", "fields", "run_parts", "audit"]:
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

# Probe only the new Experiment-5.1 audit directory.  It is immediately removed.
_probe = OUTDIR / "audit" / f".write_probe_{os.getpid()}"
try:
    with open(_probe, "x", encoding="utf-8") as f:
        f.write("ok")
finally:
    try:
        _probe.unlink()
    except Exception:
        pass

DATASETS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP51_DATASETS", "mpi3d,fmnist").split(",") if x.strip()]
ACTIVATIONS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP51_ACTIVATIONS", "relu,tanh,silu").split(",") if x.strip()]
OPTIMIZERS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP51_OPTIMIZERS", "sgdm,rmsprop,adam").split(",") if x.strip()]
SEEDS = [int(x) for x in os.environ.get("PATHFLOW_EXP51_SEEDS", "0,1,2").split(",") if x.strip()]
ARCH_FILTER = [x.strip() for x in os.environ.get("PATHFLOW_EXP51_ARCHS", "C1_taper,C2_bottleneck,C3_deepmix").split(",") if x.strip()]
ENVIRONMENTS = [x.strip().lower() for x in os.environ.get("PATHFLOW_EXP51_ENVIRONMENTS", "clean,shifted").split(",") if x.strip()]

IMAGE_RESOLUTION = 16
N_TRAIN = 20000
N_VAL = 5000
N_TEST = 5000
ENDPOINT_INPUTS = 160
CONDITIONS_PER_MODEL = 40
SHIFT_SNR_DB = 15.0
EXP51_CONDITION_COUNT = int(os.environ.get("PATHFLOW_EXP51_MEASURE_CONDITIONS", "10"))
PREFLIGHT_TOL = float(os.environ.get("PATHFLOW_EXP51_PREFLIGHT_TOL", "5e-6"))
FIG_DPI = int(os.environ.get("PATHFLOW_EXP51_FIG_DPI", "220"))

EPS = 1e-14
PATH_LOG_FLOOR = 1e-300

CNN_ARCHS: Dict[str, Dict[str, Any]] = {
    "C1_taper": {"hidden_channels": [4, 8, 8, 12], "kernels": [3, 3, 3, 3, 1]},
    "C2_bottleneck": {"hidden_channels": [4, 8, 3, 8, 10], "kernels": [3, 3, 1, 3, 3, 1]},
    "C3_deepmix": {"hidden_channels": [3, 6, 8, 8, 10, 8], "kernels": [3, 3, 3, 1, 3, 1, 1]},
}

if PROFILE == "pilot":
    DATASETS = ["fmnist"]
    ACTIVATIONS = ["relu"]
    OPTIMIZERS = ["adam"]
    SEEDS = [0]
    ARCH_FILTER = ["C1_taper"]
    EXP51_CONDITION_COUNT = min(EXP51_CONDITION_COUNT, 2)

if any(ds not in {"mpi3d", "fmnist"} for ds in DATASETS):
    raise ValueError(f"Unsupported dataset(s): {DATASETS}")
if any(a not in {"relu", "tanh", "silu"} for a in ACTIVATIONS):
    raise ValueError(f"Unsupported activation(s): {ACTIVATIONS}")
if any(o not in {"sgdm", "rmsprop", "adam"} for o in OPTIMIZERS):
    raise ValueError(f"Unsupported optimizer(s): {OPTIMIZERS}")
if any(a not in CNN_ARCHS for a in ARCH_FILTER):
    raise ValueError(f"Unsupported CNN architecture filter: {ARCH_FILTER}")
if any(e not in {"clean", "shifted"} for e in ENVIRONMENTS):
    raise ValueError(f"Unsupported environment(s): {ENVIRONMENTS}")
if not (1 <= EXP51_CONDITION_COUNT <= CONDITIONS_PER_MODEL):
    raise ValueError(f"PATHFLOW_EXP51_MEASURE_CONDITIONS must lie in [1,{CONDITIONS_PER_MODEL}].")
if SOFT_WALL_HOURS <= 0:
    raise ValueError("SOFT_WALL_HOURS must be positive.")

MPI3D_URL = os.environ.get(
    "MPI3D_URL",
    "https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_toy.npz",
)

# ==================================================================================================
# 1. WRITE-ONCE, CONTENT-CHECKED I/O
# ==================================================================================================

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
    tmp = Path(tmp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists():
            same = (
                path.stat().st_size == tmp.stat().st_size
                and _sha1_file_stream(path) == _sha1_file_stream(tmp)
            )
            if same:
                return False
            raise RuntimeError(
                f"Write-once protection refused to replace an existing scientific file with different contents: {path}"
            )
        try:
            os.link(tmp, path)
        except OSError:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(fd, "wb") as out, open(tmp, "rb") as src:
                    shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
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

def write_once_bytes(payload: bytes, path: Path) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    return _publish_file_once(tmp, path)

def write_once_text(text: str, path: Path) -> bool:
    return write_once_bytes(text.encode("utf-8"), path)

def write_once_json(obj: Any, path: Path) -> bool:
    payload = (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    return write_once_bytes(payload, path)

def write_once_csv(df: pd.DataFrame, path: Path) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    df.to_csv(tmp, index=False, float_format="%.17g")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    return _publish_file_once(tmp, path)

def write_once_npz(path: Path, **arrays: Any) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}.npz"
    np.savez_compressed(tmp, **arrays)
    return _publish_file_once(tmp, path)

def publish_figure_once(fig, path: Path, **savefig_kwargs: Any) -> bool:
    path = Path(path)
    if path.exists():
        return False
    tmp = path.parent / f".{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{path.suffix}"
    fig.savefig(tmp, **savefig_kwargs)
    return _publish_file_once(tmp, path)

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

def soft_wall_remaining_seconds() -> float:
    return max(0.0, SOFT_WALL_HOURS * 3600.0 - (time.perf_counter() - SCRIPT_START_MONOTONIC))

def can_start_new_unit(stage: str) -> bool:
    remaining = soft_wall_remaining_seconds()
    if remaining <= SOFT_START_BUFFER_MIN * 60.0:
        rec = {
            "stage": stage,
            "remaining_seconds": remaining,
            "soft_wall_hours": SOFT_WALL_HOURS,
            "safety_buffer_minutes": SOFT_START_BUFFER_MIN,
            "status": "paused_safely",
        }
        # Unique audit filename avoids overwriting any prior pause record.
        write_once_json(rec, OUTDIR / "audit" / f"PAUSED_{int(time.time())}.json")
        print(
            f"[soft-wall] Not starting next {stage}. Remaining {human_time(remaining)}. "
            "Rerun the same file in a new allocation to resume."
        )
        return False
    return True



# ==================================================================================================
# SOURCE-FAITHFUL CORE FROM EXPERIMENT 5
# ==================================================================================================

def stable_hash_dict(d: Dict[str, Any], n: int = 12) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:n]

def model_seed(arch: str, dataset: str, seed_index: int) -> int:
    return int(MODEL_SEED_BASE + 1000003 * seed_index + int(stable_hash_dict({"a": arch, "d": dataset}, 8), 16) % 999983)

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
            "\nSet MPI3D_NPZ=/path/to/mpi3d_toy.npz or PATHFLOW_EXP51_ALLOW_DOWNLOAD=1."
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
    root = Path(os.environ.get("PATHFLOW_EXP51_FMNIST_ROOT", str(DATA_ROOT / "fmnist"))).expanduser().resolve()
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

def _normalize_rows(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    s = t.sum(dim=tuple(range(1, t.ndim)))
    valid = torch.isfinite(s) & (s > 0)
    safe = torch.where(valid, s, torch.ones_like(s))
    out = t / safe.reshape((-1,) + ((1,) * (t.ndim - 1)))
    out = torch.where(valid.reshape((-1,) + ((1,) * (t.ndim - 1))), out, torch.zeros_like(out))
    return out, s

def cnn_response_sq(model: DeepCNN, X: torch.Tensor, X0: torch.Tensor) -> List[torch.Tensor]:
    _, z, _ = model.forward_with_states(X.to(DEVICE))
    _, z0, _ = model.forward_with_states(X0.to(DEVICE))
    return [torch.square(finite_secant_torch64(model.activation_name, a, b)) for a, b in zip(z, z0)]

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

def file_sha1(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

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

def image_for_storage_and_display(x: torch.Tensor, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    arr = x[0].detach().cpu().numpy().astype(np.float64)
    raw = arr * float(meta["input_sd"]) + float(meta["input_mean"])
    if raw.shape[0] == 1:
        raw_hwc = raw[0]
    else:
        raw_hwc = np.moveaxis(raw, 0, -1)
    display = np.clip(raw_hwc, 0.0, 1.0)
    return raw_hwc, display

# ==================================================================================================
# 2. EXPERIMENT-5.1 MANIFEST AND FROZEN CONDITION POPULATION
# ==================================================================================================

def output_dim_for(dataset: str) -> int:
    return 10 if dataset == "fmnist" else 1

def task_for(dataset: str) -> str:
    return "classification" if dataset == "fmnist" else "regression"

def checkpoint_path(run_id: str) -> Path:
    return SOURCE_CHECKPOINT_DIR / f"{run_id}__best.pt"

def build_cnn_manifest() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for arch in ARCH_FILTER:
            for opt in OPTIMIZERS:
                for act in ACTIVATIONS:
                    for seed in SEEDS:
                        rows.append({
                            "run_id": f"BENCH__{ds}__{arch}__{opt}__{act}__S{seed:02d}",
                            "dataset": ds,
                            "task": task_for(ds),
                            "family": "cnn",
                            "arch_id": arch,
                            "optimizer": opt,
                            "activation": act,
                            "seed_index": int(seed),
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Experiment 5.1 manifest is empty.")
    return df

def build_model(arch_id: str, activation: str, output_dim: int, dataset: str, seed_index: int) -> DeepCNN:
    if arch_id not in CNN_ARCHS:
        raise KeyError(f"Experiment 5.1 accepts CNN architectures only, got {arch_id}.")
    input_channels = 1 if dataset == "fmnist" else 3
    seed = model_seed(arch_id, dataset, seed_index)
    return DeepCNN(
        CNN_ARCHS[arch_id],
        activation,
        output_dim,
        seed,
        input_channels=input_channels,
    ).to(DEVICE)

def verify_source_checkpoints(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing = []
    for _, r in manifest.iterrows():
        cp = checkpoint_path(str(r.run_id))
        exists = cp.exists()
        rows.append({
            "run_id": str(r.run_id),
            "dataset": str(r.dataset),
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
    write_once_csv(df, OUTDIR / "tables" / "source_checkpoint_inventory.csv")
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(
            f"Experiment 5.1 requires the frozen Experiment-1 CNN checkpoints. "
            f"{len(missing)} checkpoint(s) are missing.\n{preview}"
        )
    return df

def _norm_stratum(v: Any) -> str:
    try:
        x = float(v)
        if np.isfinite(x) and abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
    except Exception:
        pass
    return str(v).strip()

def select_exp51_endpoint_local_indices(data: Dict[str, Any], dataset: str, count: int) -> np.ndarray:
    """
    Reconstruct the exact Experiment-5 selection rule.

    Default count=10:
      FMNIST -> one observation per class.
      MPI3D  -> horizontal levels 0,4,...,36.
    """
    endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
    causal_local = np.asarray(data["causal_endpoint_local_indices"], dtype=int)
    strata_test = np.asarray(data["test_strata"])
    causal_strata = strata_test[endpoint_ids[causal_local]]

    if dataset == "fmnist" and count == 10:
        selected = []
        for cls in range(10):
            ids = np.flatnonzero(causal_strata == cls)
            if len(ids) == 0:
                raise RuntimeError(f"Frozen causal population has no FMNIST class {cls}.")
            selected.append(int(causal_local[int(ids[0])]))
        return np.asarray(selected, dtype=int)

    if dataset == "mpi3d" and count == 10:
        selected = []
        for level in range(0, 40, 4):
            ids = np.flatnonzero(causal_strata == level)
            if len(ids) == 0:
                raise RuntimeError(f"Frozen causal population has no MPI3D horizontal level {level}.")
            selected.append(int(causal_local[int(ids[0])]))
        return np.asarray(selected, dtype=int)

    chosen_positions = balanced_indices_from_strata(
        causal_strata,
        min(int(count), len(causal_local)),
        CONDITION_SEED_BASE + 15001 + (0 if dataset == "fmnist" else 1),
    )
    return causal_local[chosen_positions]

def attach_exp51_population(data_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ds, data in data_cache.items():
        local = select_exp51_endpoint_local_indices(data, ds, EXP51_CONDITION_COUNT)
        data["exp51_endpoint_local_indices"] = local

        endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
        strata = np.asarray(data["test_strata"])
        causal_local = np.asarray(data["causal_endpoint_local_indices"], dtype=int)
        pos_map = {int(v): i for i, v in enumerate(causal_local)}

        for ci, ep_local in enumerate(local):
            test_id = int(endpoint_ids[int(ep_local)])
            rows.append({
                "dataset": ds,
                "exp51_condition_index": int(ci),
                "exp1_condition_index": int(pos_map[int(ep_local)]),
                "endpoint_local_index": int(ep_local),
                "test_index": test_id,
                "stratum": strata[test_id],
            })
    df = pd.DataFrame(rows)
    write_once_csv(df, OUTDIR / "tables" / "experiment5_1_condition_population.csv")
    return df

def audit_prior_condition_cohorts(population: pd.DataFrame) -> Dict[str, Any]:
    """
    Experiment 5 is the strongest audit because 5.1 is explicitly an extension of its Field-A observable.
    Experiments 2-4 are retained as additional optional consistency checks.
    """
    cur = population[["dataset", "test_index", "stratum"]].copy()
    cur["dataset"] = cur["dataset"].astype(str)
    cur["test_index"] = pd.to_numeric(cur["test_index"], errors="raise").astype(int)
    cur["stratum"] = cur["stratum"].map(_norm_stratum)
    cur = cur.sort_values(["dataset", "test_index", "stratum"]).reset_index(drop=True)

    candidates = [
        ("experiment5",
         EXP1_RESULTS / "experiment5_cnn_occurrence_shared_identity" / "tables" / "exp5_condition_population.csv",
         True),
        ("experiment2",
         EXP1_RESULTS / "experiment2_condition_crossover" / "tables" / "crossover_pair_population.csv",
         False),
        ("experiment3",
         EXP1_RESULTS / "experiment3_dose_response" / "tables" / "dose_condition_population.csv",
         False),
        ("experiment4",
         EXP1_RESULTS / "experiment4_multiscale_followup" / "tables" / "multiscale_condition_population.csv",
         False),
    ]

    audits = []
    for name, path, strict_if_present in candidates:
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
                if strict_if_present and not same:
                    raise RuntimeError(
                        "Experiment 5 condition-population table exists but does not match the reconstructed "
                        "Experiment 5.1 cohort. Refusing to continue."
                    )
        audits.append(rec)

    out = {"audits": audits}
    write_once_json(out, OUTDIR / "audit" / "prior_condition_cohort_audit.json")
    return out

def load_datasets_exp51() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if "mpi3d" in DATASETS:
        out["mpi3d"] = load_mpi3d()
    if "fmnist" in DATASETS:
        out["fmnist"] = load_fmnist()
    return out

def prepared_exp51_environment(
    data: Dict[str, Any],
    dataset: str,
    environment: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preserve the benchmark's shifted-environment semantics exactly.

    The full frozen 160-endpoint population is shifted first because the deterministic Gaussian generator
    is shape dependent.  Only then are the fixed Experiment-5/5.1 conditions extracted.
    """
    Xt, yt = data["test"]
    endpoint_ids = np.asarray(data["endpoint_indices"], dtype=int)
    local = np.asarray(data["exp51_endpoint_local_indices"], dtype=int)

    Xep = Xt[endpoint_ids]
    yep = yt[endpoint_ids]
    if environment == "shifted":
        Xep = shifted_environment(Xep, dataset, data["meta"])

    return Xep[local], yep[local]

# ==================================================================================================
# 3. DECLARED TRANSLATION FAMILY
# ==================================================================================================

def declared_transforms(dataset: str) -> List[Dict[str, Any]]:
    identity = [{"transform_id": "identity", "dy": 0, "dx": 0, "magnitude": 0, "kind": "identity"}]
    if dataset == "fmnist":
        primary = [
            {"transform_id": "left_1",  "dy": 0,  "dx": -1, "magnitude": 1, "kind": "primary"},
            {"transform_id": "right_1", "dy": 0,  "dx":  1, "magnitude": 1, "kind": "primary"},
            {"transform_id": "up_1",    "dy": -1, "dx":  0, "magnitude": 1, "kind": "primary"},
            {"transform_id": "down_1",  "dy":  1, "dx":  0, "magnitude": 1, "kind": "primary"},
            {"transform_id": "left_2",  "dy": 0,  "dx": -2, "magnitude": 2, "kind": "primary"},
            {"transform_id": "right_2", "dy": 0,  "dx":  2, "magnitude": 2, "kind": "primary"},
            {"transform_id": "up_2",    "dy": -2, "dx":  0, "magnitude": 2, "kind": "primary"},
            {"transform_id": "down_2",  "dy":  2, "dx":  0, "magnitude": 2, "kind": "primary"},
        ]
    elif dataset == "mpi3d":
        primary = [
            {"transform_id": "up_1",   "dy": -1, "dx": 0, "magnitude": 1, "kind": "primary"},
            {"transform_id": "down_1", "dy":  1, "dx": 0, "magnitude": 1, "kind": "primary"},
            {"transform_id": "up_2",   "dy": -2, "dx": 0, "magnitude": 2, "kind": "primary"},
            {"transform_id": "down_2", "dy":  2, "dx": 0, "magnitude": 2, "kind": "primary"},
        ]
    else:
        raise ValueError(dataset)
    return identity + primary

def translate_zero_reference_tensor(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    Integer translation with zero fill in normalized input coordinates.

    Positive dy moves content down.  Positive dx moves content right.
    Because the declared reference is x0=0, this rule guarantees T_g(x0)=x0 exactly.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(x.shape)}")
    dy, dx = int(dy), int(dx)
    n, c, h, w = x.shape
    if abs(dy) >= h or abs(dx) >= w:
        raise ValueError(f"Translation {(dy, dx)} exceeds finite input support {(h, w)}.")

    out = torch.zeros_like(x)

    sy0 = max(0, -dy)
    sy1 = min(h, h - dy)
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx)

    dy0 = max(0, dy)
    dy1 = dy0 + (sy1 - sy0)
    dx0 = max(0, dx)
    dx1 = dx0 + (sx1 - sx0)

    out[..., dy0:dy1, dx0:dx1] = x[..., sy0:sy1, sx0:sx1]
    return out

def translate_zero_field(field: np.ndarray, dy: int, dx: int) -> Tuple[np.ndarray, float]:
    """
    Apply the same finite-lattice translation to a 2-D probability field.

    Returns:
      normalized_translated_field
      mass_retained_inside_the_finite_lattice_before_renormalization
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"Expected 2-D field, got {f.shape}")
    h, w = f.shape
    dy, dx = int(dy), int(dx)
    if abs(dy) >= h or abs(dx) >= w:
        raise ValueError(f"Translation {(dy, dx)} exceeds field support {(h, w)}.")

    out = np.zeros_like(f)

    sy0 = max(0, -dy)
    sy1 = min(h, h - dy)
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx)

    dy0 = max(0, dy)
    dy1 = dy0 + (sy1 - sy0)
    dx0 = max(0, dx)
    dx1 = dx0 + (sx1 - sx0)

    out[dy0:dy1, dx0:dx1] = f[sy0:sy1, sx0:sx1]
    retained = float(np.sum(out))
    if np.isfinite(retained) and retained > EPS:
        out = out / retained
    else:
        out[:] = np.nan
    return out, retained

def input_contrast_energy_retained(x: torch.Tensor, xt: torch.Tensor) -> float:
    a = float(torch.sum(torch.square(x.to(torch.float64))).item())
    b = float(torch.sum(torch.square(xt.to(torch.float64))).item())
    if not np.isfinite(a) or a <= EPS:
        return np.nan
    return float(b / a)

def _normalize_probability(p: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(p, dtype=np.float64).ravel().copy()
    if np.any(~np.isfinite(x)):
        return None
    x = np.clip(x, 0.0, None)
    z = float(np.sum(x))
    if z <= EPS:
        return None
    return x / z

def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    pp = _normalize_probability(p)
    qq = _normalize_probability(q)
    if pp is None or qq is None or pp.shape != qq.shape:
        return np.nan
    return float(np.linalg.norm(np.sqrt(pp) - np.sqrt(qq)) / math.sqrt(2.0))

def l1_probability_distance(p: np.ndarray, q: np.ndarray) -> float:
    pp = _normalize_probability(p)
    qq = _normalize_probability(q)
    if pp is None or qq is None or pp.shape != qq.shape:
        return np.nan
    return float(np.sum(np.abs(pp - qq)))

def field_center_of_mass(p: np.ndarray) -> Tuple[float, float]:
    pp = _normalize_probability(p)
    if pp is None:
        return np.nan, np.nan
    h, w = np.asarray(p).shape
    a = pp.reshape(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    return float(np.sum(a * yy)), float(np.sum(a * xx))

# ==================================================================================================
# 4. FIELD-A COMPUTATION AND BEHAVIOURAL RECORD
# ==================================================================================================

@torch.no_grad()
def compute_fieldA_state(
    model: DeepCNN,
    x: torch.Tensor,
    target_idx: int,
) -> Dict[str, Any]:
    x = x.to(DEVICE)
    x0 = torch.zeros_like(x)
    response_sq = cnn_response_sq(model, x, x0)
    source_sq = torch.square((x - x0).to(torch.float64))
    state = cnn_factor_state(
        model,
        source_sq,
        response_sq,
        torch.tensor([int(target_idx)], dtype=torch.long),
    )
    out = model(x).detach()
    out0 = model(x0).detach()
    valid = bool(np.asarray(state["valid"]).reshape(-1)[0])

    if valid:
        field = fieldA_input_origin(state)
        field_sum = float(np.sum(field))
        if not np.isfinite(field_sum) or abs(field_sum - 1.0) > PREFLIGHT_TOL:
            raise RuntimeError(f"Valid Field A failed probability normalization, sum={field_sum}.")
    else:
        field = np.full((IMAGE_RESOLUTION, IMAGE_RESOLUTION), np.nan, dtype=np.float64)

    return {
        "valid": valid,
        "field": field,
        "state": state,
        "out": out,
        "out0": out0,
    }

def classification_margin(logits: torch.Tensor, target_idx: int) -> float:
    x = logits[0].detach().cpu().numpy().astype(float)
    t = int(target_idx)
    other = np.delete(x, t)
    return float(x[t] - np.max(other)) if len(other) else np.nan

def behavior_record(
    dataset: str,
    y: torch.Tensor,
    target_idx: int,
    base_out: torch.Tensor,
    transformed_out: torch.Tensor,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    if dataset == "fmnist":
        ytrue = int(y.reshape(-1)[0].item())
        p0 = int(torch.argmax(base_out, dim=1)[0].item())
        p1 = int(torch.argmax(transformed_out, dim=1)[0].item())
        prob0 = float(torch.softmax(base_out, dim=1)[0, target_idx].item())
        prob1 = float(torch.softmax(transformed_out, dim=1)[0, target_idx].item())
        m0 = classification_margin(base_out, target_idx)
        m1 = classification_margin(transformed_out, target_idx)
        return {
            "true_class": ytrue,
            "baseline_prediction": p0,
            "transformed_prediction": p1,
            "prediction_unchanged": bool(p0 == p1),
            "baseline_correct": bool(p0 == ytrue),
            "transformed_correct": bool(p1 == ytrue),
            "baseline_target_probability": prob0,
            "transformed_target_probability": prob1,
            "target_probability_change": float(prob1 - prob0),
            "baseline_target_margin": m0,
            "transformed_target_margin": m1,
            "target_margin_change": float(m1 - m0),
            "prediction_abs_change_normalized": np.nan,
            "prediction_abs_change_raw": np.nan,
        }

    yy = float(y.reshape(-1)[0].item())
    p0 = float(base_out.reshape(-1)[0].item())
    p1 = float(transformed_out.reshape(-1)[0].item())
    ys = float(meta["target_sd"])
    ym = float(meta["target_mean"])
    return {
        "true_class": np.nan,
        "baseline_prediction": p0,
        "transformed_prediction": p1,
        "prediction_unchanged": np.nan,
        "baseline_correct": np.nan,
        "transformed_correct": np.nan,
        "baseline_target_probability": np.nan,
        "transformed_target_probability": np.nan,
        "target_probability_change": np.nan,
        "baseline_target_margin": np.nan,
        "transformed_target_margin": np.nan,
        "target_margin_change": np.nan,
        "prediction_abs_change_normalized": float(abs(p1 - p0)),
        "prediction_abs_change_raw": float(abs((p1 * ys + ym) - (p0 * ys + ym))),
        "baseline_mse_normalized": float((p0 - yy) ** 2),
        "transformed_mse_normalized": float((p1 - yy) ** 2),
        "task_loss_change": float((p1 - yy) ** 2 - (p0 - yy) ** 2),
    }

# ==================================================================================================
# 5. RUN-PART LAYOUT AND RESUME LOGIC
# ==================================================================================================

def unit_stem(run_id: str, environment: str) -> str:
    return f"{run_id}__EXP51__{environment}"

def unit_paths(run_id: str, environment: str) -> Dict[str, Path]:
    stem = unit_stem(run_id, environment)
    return {
        "validity": OUTDIR / "run_parts" / f"{stem}__validity.csv",
        "measurements": OUTDIR / "run_parts" / f"{stem}__measurements.csv",
        "behavior": OUTDIR / "run_parts" / f"{stem}__behavior.csv",
        "fields": OUTDIR / "fields" / f"{stem}__fields.npz",
        "done": OUTDIR / "run_parts" / f"{stem}__DONE.json",
    }

def unit_complete(run_id: str, environment: str) -> bool:
    p = unit_paths(run_id, environment)
    req = [p["validity"], p["measurements"], p["behavior"], p["fields"], p["done"]]
    return all(x.exists() and x.stat().st_size > 0 for x in req)

def load_checkpoint_model(row: pd.Series) -> DeepCNN:
    cp = checkpoint_path(str(row.run_id))
    model = build_model(
        str(row.arch_id),
        str(row.activation),
        output_dim_for(str(row.dataset)),
        str(row.dataset),
        int(row.seed_index),
    )
    chk = torch.load(cp, map_location="cpu")
    if not isinstance(chk, dict) or "state_dict" not in chk:
        raise RuntimeError(f"Checkpoint {cp} does not contain the established state_dict payload.")
    model.load_state_dict(chk["state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model

# ==================================================================================================
# 6. ONE MODEL × ENVIRONMENT ATOMIC SCIENTIFIC UNIT
# ==================================================================================================

def run_one_model_environment(
    row: pd.Series,
    data: Dict[str, Any],
    population: pd.DataFrame,
    environment: str,
) -> None:
    run_id = str(row.run_id)
    if RESUME and unit_complete(run_id, environment):
        return

    dataset = str(row.dataset)
    transforms = declared_transforms(dataset)
    X, y = prepared_exp51_environment(data, dataset, environment)

    pop_ds = population[population.dataset.astype(str) == dataset].sort_values("exp51_condition_index")
    if len(pop_ds) != len(X):
        raise RuntimeError(f"Condition-population length mismatch for {dataset}: {len(pop_ds)} vs {len(X)}.")

    model = load_checkpoint_model(row)

    validity_rows: List[Dict[str, Any]] = []
    measurement_rows: List[Dict[str, Any]] = []
    behavior_rows: List[Dict[str, Any]] = []

    ncond = len(X)
    ntrans = len(transforms)
    baseline_fields = np.full((ncond, IMAGE_RESOLUTION, IMAGE_RESOLUTION), np.nan, dtype=np.float64)
    transformed_fields = np.full((ncond, ntrans, IMAGE_RESOLUTION, IMAGE_RESOLUTION), np.nan, dtype=np.float64)
    expected_fields = np.full_like(transformed_fields, np.nan)

    for ci in range(ncond):
        xb = X[ci:ci+1]
        yb = y[ci:ci+1]
        poprow = pop_ds.iloc[ci]
        target_idx = int(yb.reshape(-1)[0].item()) if dataset == "fmnist" else 0

        baseline = compute_fieldA_state(model, xb, target_idx)
        if baseline["valid"]:
            baseline_fields[ci] = baseline["field"]

        for ti, tr in enumerate(transforms):
            dy, dx = int(tr["dy"]), int(tr["dx"])
            xt = translate_zero_reference_tensor(xb, dy, dx)
            transformed = compute_fieldA_state(model, xt, target_idx)

            if transformed["valid"]:
                transformed_fields[ci, ti] = transformed["field"]

            common = {
                "run_id": run_id,
                "dataset": dataset,
                "arch_id": str(row.arch_id),
                "optimizer": str(row.optimizer),
                "activation": str(row.activation),
                "seed_index": int(row.seed_index),
                "environment": environment,
                "condition_index": int(ci),
                "test_index": int(poprow.test_index),
                "stratum": poprow.stratum,
                "target_idx": int(target_idx),
                "transform_id": str(tr["transform_id"]),
                "transform_kind": str(tr["kind"]),
                "dy": dy,
                "dx": dx,
                "shift_magnitude": int(tr["magnitude"]),
            }

            valid_compare = False
            predicted_field = np.full((IMAGE_RESOLUTION, IMAGE_RESOLUTION), np.nan, dtype=np.float64)
            retained = np.nan
            d_eq = d_inv = d_wrong = np.nan
            l1_eq = l1_inv = l1_wrong = np.nan
            eq_adv = wrong_adv = np.nan
            com_error = np.nan

            if baseline["valid"]:
                predicted_field, retained = translate_zero_field(baseline["field"], dy, dx)
                expected_fields[ci, ti] = predicted_field

            if baseline["valid"] and transformed["valid"] and np.all(np.isfinite(predicted_field)):
                d_eq = hellinger_distance(transformed["field"], predicted_field)
                d_inv = hellinger_distance(transformed["field"], baseline["field"])

                wrong_field, wrong_retained = translate_zero_field(baseline["field"], -dy, -dx)
                d_wrong = hellinger_distance(transformed["field"], wrong_field)

                l1_eq = l1_probability_distance(transformed["field"], predicted_field)
                l1_inv = l1_probability_distance(transformed["field"], baseline["field"])
                l1_wrong = l1_probability_distance(transformed["field"], wrong_field)

                eq_adv = float(d_inv - d_eq)
                wrong_adv = float(d_wrong - d_eq)

                by, bx = field_center_of_mass(baseline["field"])
                ty, tx = field_center_of_mass(transformed["field"])
                py, px = field_center_of_mass(predicted_field)
                if all(np.isfinite(v) for v in [ty, tx, py, px]):
                    com_error = float(math.sqrt((ty - py) ** 2 + (tx - px) ** 2))
                valid_compare = True
            else:
                wrong_retained = np.nan

            validity_rows.append({
                **common,
                "baseline_field_defined": bool(baseline["valid"]),
                "transformed_field_defined": bool(transformed["valid"]),
                "comparison_defined": bool(valid_compare),
                "baseline_logZ1": float(np.asarray(baseline["state"]["logZ1"]).reshape(-1)[0]),
                "transformed_logZ1": float(np.asarray(transformed["state"]["logZ1"]).reshape(-1)[0]),
            })

            beh = behavior_record(
                dataset,
                yb,
                target_idx,
                baseline["out"],
                transformed["out"],
                data["meta"],
            )
            behavior_rows.append({
                **common,
                "input_contrast_energy_retained": input_contrast_energy_retained(xb, xt),
                **beh,
            })

            if valid_compare:
                measurement_rows.append({
                    **common,
                    "field_transport_retained_mass": float(retained),
                    "wrong_direction_transport_retained_mass": float(wrong_retained),
                    "hellinger_equivariant": float(d_eq),
                    "hellinger_invariant": float(d_inv),
                    "hellinger_wrong_direction": float(d_wrong),
                    "equivariance_advantage": float(eq_adv),
                    "wrong_direction_advantage": float(wrong_adv),
                    "l1_equivariant": float(l1_eq),
                    "l1_invariant": float(l1_inv),
                    "l1_wrong_direction": float(l1_wrong),
                    "field_com_equivariance_error_pixels": float(com_error),
                })

    validity_df = pd.DataFrame(validity_rows)
    measurements_df = pd.DataFrame(measurement_rows)
    behavior_df = pd.DataFrame(behavior_rows)

    paths = unit_paths(run_id, environment)
    write_once_csv(validity_df, paths["validity"])
    write_once_csv(measurements_df, paths["measurements"])
    write_once_csv(behavior_df, paths["behavior"])
    write_once_npz(
        paths["fields"],
        baseline_fields=baseline_fields,
        transformed_fields=transformed_fields,
        expected_equivariant_fields=expected_fields,
        transform_id=np.asarray([t["transform_id"] for t in transforms], dtype="U32"),
        dy=np.asarray([t["dy"] for t in transforms], dtype=np.int64),
        dx=np.asarray([t["dx"] for t in transforms], dtype=np.int64),
        transform_kind=np.asarray([t["kind"] for t in transforms], dtype="U16"),
    )

    done = {
        "run_id": run_id,
        "dataset": dataset,
        "environment": environment,
        "conditions": int(ncond),
        "transforms_per_condition": int(ntrans),
        "validity_rows": int(len(validity_df)),
        "measurement_rows": int(len(measurements_df)),
        "behavior_rows": int(len(behavior_df)),
        "validity_sha1": file_sha1(paths["validity"]),
        "measurements_sha1": file_sha1(paths["measurements"]),
        "behavior_sha1": file_sha1(paths["behavior"]),
        "fields_sha1": file_sha1(paths["fields"]),
    }
    write_once_json(done, paths["done"])

    if RUN_FIGURES:
        render_representative_figure(
            row=row,
            data=data,
            environment=environment,
            X=X,
            y=y,
            baseline_fields=baseline_fields,
            transformed_fields=transformed_fields,
            expected_fields=expected_fields,
            transforms=transforms,
            measurements_df=measurements_df,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================================================================================================
# 7. REPRESENTATIVE FIGURE
# ==================================================================================================

def _representative_transform_id(dataset: str) -> str:
    return "right_2" if dataset == "fmnist" else "down_2"

def render_representative_figure(
    row: pd.Series,
    data: Dict[str, Any],
    environment: str,
    X: torch.Tensor,
    y: torch.Tensor,
    baseline_fields: np.ndarray,
    transformed_fields: np.ndarray,
    expected_fields: np.ndarray,
    transforms: List[Dict[str, Any]],
    measurements_df: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    dataset = str(row.dataset)
    ci = 0
    wanted = _representative_transform_id(dataset)
    tids = [t["transform_id"] for t in transforms]
    if wanted not in tids:
        return
    ti = tids.index(wanted)
    tr = transforms[ti]

    A0 = baseline_fields[ci]
    At = transformed_fields[ci, ti]
    Ae = expected_fields[ci, ti]
    if not (np.all(np.isfinite(A0)) and np.all(np.isfinite(At)) and np.all(np.isfinite(Ae))):
        return

    xb = X[ci:ci+1]
    xt = translate_zero_reference_tensor(xb, int(tr["dy"]), int(tr["dx"]))
    _, img0 = image_for_storage_and_display(xb, data["meta"])
    _, imgt = image_for_storage_and_display(xt, data["meta"])

    resid = np.abs(At - Ae)
    vmax = max(float(np.max(A0)), float(np.max(At)), float(np.max(Ae)), EPS)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5))
    ax = axes.ravel()

    if img0.ndim == 2:
        ax[0].imshow(img0, cmap="gray", vmin=0, vmax=1)
        ax[3].imshow(imgt, cmap="gray", vmin=0, vmax=1)
    else:
        ax[0].imshow(img0)
        ax[3].imshow(imgt)
    ax[0].set_title("Original input")
    ax[3].set_title(f"Translated input  {wanted}")

    im1 = ax[1].imshow(A0, cmap="magma", vmin=0, vmax=vmax)
    ax[1].set_title("Original Field A")
    ax[2].imshow(Ae, cmap="magma", vmin=0, vmax=vmax)
    ax[2].set_title("Expected if equivariant")
    ax[4].imshow(At, cmap="magma", vmin=0, vmax=vmax)
    ax[4].set_title("Observed translated Field A")
    im5 = ax[5].imshow(resid, cmap="viridis")
    ax[5].set_title("Absolute residual")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    fig.colorbar(im1, ax=[ax[1], ax[2], ax[4]], fraction=0.025, pad=0.02)
    fig.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

    m = measurements_df[
        (measurements_df.condition_index == ci)
        & (measurements_df.transform_id.astype(str) == wanted)
    ]
    if len(m):
        r = m.iloc[0]
        subtitle = (
            f"H_eq={r.hellinger_equivariant:.4f}   "
            f"H_inv={r.hellinger_invariant:.4f}   "
            f"advantage={r.equivariance_advantage:+.4f}"
        )
    else:
        subtitle = "Field comparison undefined for this condition."

    fig.suptitle(
        f"{row.run_id} | {environment}\n{subtitle}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    path = OUTDIR / "figures" / f"{unit_stem(str(row.run_id), environment)}__representative.png"
    publish_figure_once(fig, path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# ==================================================================================================
# 8. SELF-TEST AND REAL-CHECKPOINT PREFLIGHT
# ==================================================================================================

def run_dataset_free_selftest() -> pd.DataFrame:
    rows = []

    def add(name: str, value: float, tolerance: float, mode: str = "abs_le"):
        if mode == "abs_le":
            passed = bool(np.isfinite(value) and abs(value) <= tolerance)
        elif mode == "ge":
            passed = bool(np.isfinite(value) and value >= tolerance)
        else:
            raise ValueError(mode)
        rows.append({
            "test": name,
            "value": float(value),
            "criterion": mode,
            "tolerance": float(tolerance),
            "passed": passed,
        })

    x0 = torch.zeros((1, 1, IMAGE_RESOLUTION, IMAGE_RESOLUTION), dtype=torch.float32)
    z = translate_zero_reference_tensor(x0, 2, -1)
    add("zero_reference_is_fixed", float(torch.max(torch.abs(z)).item()), 0.0)

    impulse = np.zeros((IMAGE_RESOLUTION, IMAGE_RESOLUTION), dtype=np.float64)
    impulse[7, 7] = 1.0
    moved, retained = translate_zero_field(impulse, 2, -1)
    expected = np.zeros_like(impulse)
    expected[9, 6] = 1.0
    add("field_translation_exact_interior", float(np.max(np.abs(moved - expected))), 0.0)
    add("field_translation_retains_interior_mass", abs(retained - 1.0), 0.0)

    d_correct = hellinger_distance(expected, moved)
    wrong, _ = translate_zero_field(impulse, -2, 1)
    d_wrong = hellinger_distance(expected, wrong)
    add("hellinger_correct_translation_zero", d_correct, 1e-14)
    add("wrong_direction_is_worse_for_impulse", d_wrong - d_correct, 1e-6, mode="ge")

    # Computational Field-A sentinel on an untrained synthetic CNN.
    model = DeepCNN(CNN_ARCHS["C1_taper"], "silu", 10, seed=123, input_channels=1).to(DEVICE).eval()
    x = torch.randn((1, 1, IMAGE_RESOLUTION, IMAGE_RESOLUTION), generator=torch.Generator().manual_seed(987))
    c = compute_fieldA_state(model, x, target_idx=0)
    if c["valid"]:
        add("synthetic_fieldA_probability_sum", abs(float(np.sum(c["field"])) - 1.0), PREFLIGHT_TOL)
    else:
        rows.append({
            "test": "synthetic_fieldA_probability_sum",
            "value": np.nan,
            "criterion": "defined",
            "tolerance": 0.0,
            "passed": False,
        })

    df = pd.DataFrame(rows)
    if not bool(df.passed.all()):
        raise RuntimeError("Experiment 5.1 dataset-free self-test failed.\n" + df.to_string(index=False))
    return df

def run_real_checkpoint_preflight(
    manifest: pd.DataFrame,
    data_cache: Dict[str, Dict[str, Any]],
    population: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    # One real checkpoint from each dataset present in the run.
    for ds in DATASETS:
        subset = manifest[manifest.dataset.astype(str) == ds]
        if subset.empty:
            continue
        row = subset.iloc[0]
        data = data_cache[ds]
        X, y = prepared_exp51_environment(data, ds, "clean")
        x = X[0:1]
        target_idx = int(y[0].reshape(-1)[0].item()) if ds == "fmnist" else 0

        model = load_checkpoint_model(row)
        base = compute_fieldA_state(model, x, target_idx)

        # Identity must be recomputed, not merely copied.
        ident_x = translate_zero_reference_tensor(x, 0, 0)
        ident = compute_fieldA_state(model, ident_x, target_idx)

        identity_h = (
            hellinger_distance(base["field"], ident["field"])
            if base["valid"] and ident["valid"] else np.nan
        )

        # One declared non-identity transform.
        tr = next(t for t in declared_transforms(ds) if t["kind"] == "primary")
        xt = translate_zero_reference_tensor(x, int(tr["dy"]), int(tr["dx"]))
        moved = compute_fieldA_state(model, xt, target_idx)

        rows.append({
            "dataset": ds,
            "run_id": str(row.run_id),
            "baseline_defined": bool(base["valid"]),
            "identity_defined": bool(ident["valid"]),
            "translated_defined": bool(moved["valid"]),
            "baseline_field_sum_error": (
                abs(float(np.sum(base["field"])) - 1.0) if base["valid"] else np.nan
            ),
            "identity_recompute_hellinger": identity_h,
            "zero_reference_translation_error": float(
                torch.max(torch.abs(
                    translate_zero_reference_tensor(torch.zeros_like(x), int(tr["dy"]), int(tr["dx"]))
                )).item()
            ),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        if not bool(r.baseline_defined) or not bool(r.identity_defined):
            raise RuntimeError(f"Real-checkpoint preflight found undefined baseline/identity Field A for {r.dataset}.")
        if float(r.baseline_field_sum_error) > PREFLIGHT_TOL:
            raise RuntimeError(f"Real-checkpoint Field-A sum check failed for {r.dataset}.")
        if float(r.identity_recompute_hellinger) > PREFLIGHT_TOL:
            raise RuntimeError(f"Identity recomputation failed for {r.dataset}.")
        if float(r.zero_reference_translation_error) > PREFLIGHT_TOL:
            raise RuntimeError(f"Zero-reference preservation failed for {r.dataset}.")
    return df

# ==================================================================================================
# 9. AGGREGATION AND MODEL-LEVEL INFERENCE
# ==================================================================================================

def _read_all_run_parts(kind: str, manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in manifest.iterrows():
        for env in ENVIRONMENTS:
            p = unit_paths(str(row.run_id), env)[kind]
            if not p.exists():
                raise RuntimeError(f"Missing completed run part during aggregation: {p}")
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def build_model_level_summary(meas: pd.DataFrame, beh: pd.DataFrame) -> pd.DataFrame:
    primary = meas[meas.transform_kind.astype(str) == "primary"].copy()
    bprimary = beh[beh.transform_kind.astype(str) == "primary"].copy()

    keys = ["run_id", "dataset", "arch_id", "optimizer", "activation", "seed_index", "environment"]
    rows = []
    for key, g in primary.groupby(keys, dropna=False, sort=True):
        rec = dict(zip(keys, key))
        rec.update({
            "n_valid_field_comparisons": int(len(g)),
            "median_hellinger_equivariant": float(np.median(g.hellinger_equivariant)),
            "median_hellinger_invariant": float(np.median(g.hellinger_invariant)),
            "median_equivariance_advantage": float(np.median(g.equivariance_advantage)),
            "positive_equivariance_advantage_fraction": float(np.mean(g.equivariance_advantage > 0)),
            "median_wrong_direction_advantage": float(np.median(g.wrong_direction_advantage)),
            "positive_wrong_direction_advantage_fraction": float(np.mean(g.wrong_direction_advantage > 0)),
            "median_field_transport_retained_mass": float(np.median(g.field_transport_retained_mass)),
            "median_field_com_equivariance_error_pixels": float(np.median(g.field_com_equivariance_error_pixels)),
        })

        bb = bprimary
        for k, v in zip(keys, key):
            bb = bb[bb[k].astype(str) == str(v)]
        if len(bb):
            rec["median_input_contrast_energy_retained"] = float(np.nanmedian(bb.input_contrast_energy_retained))
            if str(rec["dataset"]) == "fmnist":
                rec["prediction_unchanged_fraction"] = float(np.mean(bb.prediction_unchanged.astype(str) == "True"))
                rec["median_abs_target_probability_change"] = float(np.nanmedian(np.abs(bb.target_probability_change)))
                rec["median_abs_target_margin_change"] = float(np.nanmedian(np.abs(bb.target_margin_change)))
                rec["median_prediction_abs_change_raw"] = np.nan
            else:
                rec["prediction_unchanged_fraction"] = np.nan
                rec["median_abs_target_probability_change"] = np.nan
                rec["median_abs_target_margin_change"] = np.nan
                rec["median_prediction_abs_change_raw"] = float(np.nanmedian(bb.prediction_abs_change_raw))
        rows.append(rec)
    return pd.DataFrame(rows)

def build_model_magnitude_summary(meas: pd.DataFrame) -> pd.DataFrame:
    primary = meas[meas.transform_kind.astype(str) == "primary"].copy()
    keys = [
        "run_id", "dataset", "arch_id", "optimizer", "activation", "seed_index",
        "environment", "shift_magnitude"
    ]
    rows = []
    for key, g in primary.groupby(keys, dropna=False, sort=True):
        rec = dict(zip(keys, key))
        rec.update({
            "n_valid_field_comparisons": int(len(g)),
            "median_hellinger_equivariant": float(np.median(g.hellinger_equivariant)),
            "median_hellinger_invariant": float(np.median(g.hellinger_invariant)),
            "median_equivariance_advantage": float(np.median(g.equivariance_advantage)),
            "median_wrong_direction_advantage": float(np.median(g.wrong_direction_advantage)),
            "positive_equivariance_advantage_fraction": float(np.mean(g.equivariance_advantage > 0)),
        })
        rows.append(rec)
    return pd.DataFrame(rows)

def bootstrap_median_ci(x: np.ndarray, seed: int, B: int = 5000) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED + int(seed))
    vals = np.empty(B, dtype=float)
    for b in range(B):
        vals[b] = np.median(rng.choice(x, size=len(x), replace=True))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def hodges_lehmann_onesample(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    vals = [0.5 * (x[i] + x[j]) for i in range(len(x)) for j in range(i, len(x))]
    return float(np.median(vals))

def paired_rank_biserial(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return 0.0
    ranks = pd.Series(np.abs(x)).rank(method="average").to_numpy(float)
    wp = float(ranks[x > 0].sum())
    wm = float(ranks[x < 0].sum())
    den = wp + wm
    return float((wp - wm) / den) if den > 0 else 0.0

def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if len(p) == 0:
        return p
    out = np.ones(len(p), dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return out * np.nan
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    out[~np.isfinite(p)] = np.nan
    return out

def one_sample_signed_test(
    x: np.ndarray,
    metric: str,
    family: str,
    label: str,
    seed: int,
) -> Dict[str, Any]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {
            "test_family": family, "group": label, "metric": metric, "n_models": 0,
            "median": np.nan, "hl": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "positive_fraction": np.nan, "rank_biserial": np.nan, "wilcoxon_stat": np.nan,
            "p_raw": np.nan,
        }

    if np.allclose(a, 0.0):
        stat, p = 0.0, 1.0
    else:
        try:
            stat, p = wilcoxon(a, zero_method="wilcox", alternative="two-sided", mode="auto")
        except Exception:
            stat, p = np.nan, np.nan

    lo, hi = bootstrap_median_ci(a, seed=seed)
    return {
        "test_family": family,
        "group": label,
        "metric": metric,
        "n_models": int(len(a)),
        "median": float(np.median(a)),
        "hl": hodges_lehmann_onesample(a),
        "ci_low": lo,
        "ci_high": hi,
        "positive_fraction": float(np.mean(a > 0)),
        "rank_biserial": paired_rank_biserial(a),
        "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
        "p_raw": float(p) if np.isfinite(p) else np.nan,
    }

def build_population_tests(model_level: pd.DataFrame, model_mag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed_counter = 0

    def add_grouped(df: pd.DataFrame, group_cols: List[str], family: str, metric: str):
        nonlocal seed_counter
        if not group_cols:
            groups = [((), df)]
        else:
            groups = df.groupby(group_cols, dropna=False, sort=True)
        for key, g in groups:
            if not isinstance(key, tuple):
                key = (key,)
            label = "overall" if not group_cols else " | ".join(f"{c}={v}" for c, v in zip(group_cols, key))
            rows.append(one_sample_signed_test(g[metric].to_numpy(float), metric, family, label, seed_counter))
            seed_counter += 1

    # Primary comparison. Positive means correct transformed field is closer than stationary baseline field.
    add_grouped(model_level, [], "primary_overall", "median_equivariance_advantage")
    add_grouped(model_level, ["dataset"], "primary_dataset", "median_equivariance_advantage")
    add_grouped(model_level, ["dataset", "environment"], "primary_dataset_environment", "median_equivariance_advantage")
    add_grouped(model_level, ["dataset", "activation"], "primary_dataset_activation", "median_equivariance_advantage")
    add_grouped(model_level, ["dataset", "arch_id"], "primary_dataset_architecture", "median_equivariance_advantage")
    add_grouped(model_level, ["dataset", "optimizer"], "primary_dataset_optimizer", "median_equivariance_advantage")
    add_grouped(model_mag, ["dataset", "shift_magnitude"], "primary_dataset_magnitude", "median_equivariance_advantage")

    # Direction control. Positive means correct direction beats the deliberately wrong inverse direction.
    add_grouped(model_level, [], "direction_control_overall", "median_wrong_direction_advantage")
    add_grouped(model_level, ["dataset", "environment"], "direction_control_dataset_environment", "median_wrong_direction_advantage")

    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for family, ids in out.groupby("test_family", sort=False).groups.items():
            idx = list(ids)
            out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_raw"].to_numpy(float))
    return out

def build_behavior_realization_association_table(
    meas: pd.DataFrame,
    beh: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive linkage only.  No causal interpretation is assigned.

    FMNIST compares equivariance advantage between prediction-stable and prediction-changed rows.
    MPI3D reports Spearman association between equivariance advantage and absolute raw prediction drift.
    """
    from scipy.stats import spearmanr

    keys = [
        "run_id", "dataset", "arch_id", "optimizer", "activation", "seed_index",
        "environment", "condition_index", "transform_id"
    ]
    m = meas[meas.transform_kind.astype(str) == "primary"].copy()
    b = beh[beh.transform_kind.astype(str) == "primary"].copy()
    joined = m.merge(b, on=keys, how="inner", suffixes=("", "_beh"))

    rows = []
    for ds, g in joined.groupby("dataset", sort=True):
        if ds == "fmnist":
            stable = g[g.prediction_unchanged.astype(str) == "True"].equivariance_advantage.to_numpy(float)
            changed = g[g.prediction_unchanged.astype(str) == "False"].equivariance_advantage.to_numpy(float)
            rows.append({
                "dataset": ds,
                "analysis": "descriptive_prediction_stability_split",
                "n": int(len(g)),
                "stable_n": int(len(stable)),
                "changed_n": int(len(changed)),
                "stable_median_equivariance_advantage": float(np.nanmedian(stable)) if len(stable) else np.nan,
                "changed_median_equivariance_advantage": float(np.nanmedian(changed)) if len(changed) else np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            })
        else:
            x = g.equivariance_advantage.to_numpy(float)
            y = g.prediction_abs_change_raw.to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) >= 3:
                rho, p = spearmanr(x[mask], y[mask])
            else:
                rho, p = np.nan, np.nan
            rows.append({
                "dataset": ds,
                "analysis": "descriptive_equivariance_vs_output_drift",
                "n": int(np.sum(mask)),
                "stable_n": np.nan,
                "changed_n": np.nan,
                "stable_median_equivariance_advantage": np.nan,
                "changed_median_equivariance_advantage": np.nan,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(p) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows)

# ==================================================================================================
# 10. DESIGN MANIFEST
# ==================================================================================================

def design_object(manifest: pd.DataFrame, population: pd.DataFrame) -> Dict[str, Any]:
    transforms = {ds: declared_transforms(ds) for ds in DATASETS}
    obj = {
        "version": EXP51_VERSION,
        "study": "Experiment 5.1, transformation symmetry of Field-A conditional realization",
        "scientific_status": "descriptive extension of Experiment 5 observable field; no intervention and no retraining",
        "primary_question": (
            "Does Field A transform coherently with controlled task-admissible input translations "
            "in the same frozen trained CNN?"
        ),
        "field": "Field A, input-origin marginal of the complete normalized conditional path measure",
        "condition": "c=(x,0) in normalized input coordinates",
        "reference_rule": "x0=0 and the translation operator preserves x0 exactly",
        "transform_rule": (
            "integer zero-fill translation in normalized input coordinates; positive dy=down, positive dx=right"
        ),
        "field_boundary_rule": (
            "the same finite-lattice translation is applied to Field A; retained in-frame mass is recorded; "
            "the transported field is renormalized before probability-distance comparison"
        ),
        "transformations": transforms,
        "primary_metric": (
            "equivariance_advantage = H(P_Tx,P_x) - H(P_Tx,T(P_x)); positive favors equivariance over invariance"
        ),
        "direction_control": (
            "wrong_direction_advantage = H(P_Tx,T_inverse(P_x)) - H(P_Tx,T(P_x)); positive favors correct direction"
        ),
        "identity_control": "identity input is recomputed; H(P_x_recomputed,P_x) should be numerical zero",
        "population": {
            "datasets": list(DATASETS),
            "architectures": list(ARCH_FILTER),
            "optimizers": list(OPTIMIZERS),
            "activations": list(ACTIVATIONS),
            "seeds": list(SEEDS),
            "environments": list(ENVIRONMENTS),
            "frozen_cnn_checkpoints": int(len(manifest)),
            "conditions_per_model": int(EXP51_CONDITION_COUNT),
            "condition_population_rows": int(len(population)),
        },
        "task_specific_scope": {
            "fmnist": "horizontal and vertical +/-1,+/-2 translations; class target unchanged",
            "mpi3d": (
                "vertical +/-1,+/-2 translations only; horizontal translation excluded because horizontal_axis "
                "is the regression target"
            ),
        },
        "validity_semantics": (
            "zero complete-path support is retained in a separate validity table; undefined Field-A comparisons "
            "are not coerced to zero"
        ),
        "inferential_block": "trained CNN; conditions and transformations are repeated observations",
        "output_root": str(OUTDIR),
        "source_checkpoint_root": str(SOURCE_CHECKPOINT_DIR),
        "source_data_root": str(DATA_ROOT),
    }
    obj["design_hash"] = hashlib.sha1(
        json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return obj

def write_or_validate_design(obj: Dict[str, Any]) -> None:
    p = OUTDIR / "audit" / "experiment5_1_design_manifest.json"
    write_once_json(obj, p)

# ==================================================================================================
# 11. AGGREGATION
# ==================================================================================================

def aggregate_and_finalize(manifest: pd.DataFrame, design: Dict[str, Any]) -> Dict[str, Any]:
    validity = _read_all_run_parts("validity", manifest)
    measurements = _read_all_run_parts("measurements", manifest)
    behavior = _read_all_run_parts("behavior", manifest)

    write_once_csv(validity, OUTDIR / "tables" / "field_validity.csv")
    write_once_csv(measurements, OUTDIR / "tables" / "field_symmetry_measurements.csv")
    write_once_csv(behavior, OUTDIR / "tables" / "behavior_measurements.csv")

    model_level = build_model_level_summary(measurements, behavior)
    model_mag = build_model_magnitude_summary(measurements)
    tests = build_population_tests(model_level, model_mag)
    assoc = build_behavior_realization_association_table(measurements, behavior)

    write_once_csv(model_level, OUTDIR / "tables" / "model_level_symmetry_summary.csv")
    write_once_csv(model_mag, OUTDIR / "tables" / "model_level_shift_magnitude_summary.csv")
    write_once_csv(tests, OUTDIR / "tables" / "population_inference.csv")
    write_once_csv(assoc, OUTDIR / "tables" / "behavior_realization_descriptive_linkage.csv")

    identity = measurements[measurements.transform_kind.astype(str) == "identity"].copy()
    identity_audit = {
        "identity_rows_defined": int(len(identity)),
        "max_identity_hellinger": (
            float(np.nanmax(identity.hellinger_equivariant.to_numpy(float))) if len(identity) else np.nan
        ),
        "median_identity_hellinger": (
            float(np.nanmedian(identity.hellinger_equivariant.to_numpy(float))) if len(identity) else np.nan
        ),
    }
    write_once_json(identity_audit, OUTDIR / "audit" / "identity_control_summary.json")

    expected_units = int(len(manifest) * len(ENVIRONMENTS))
    completed_units = sum(
        unit_complete(str(r.run_id), env)
        for _, r in manifest.iterrows()
        for env in ENVIRONMENTS
    )
    if completed_units != expected_units:
        raise RuntimeError(f"Aggregation reached with incomplete units {completed_units}/{expected_units}.")

    completion = {
        "version": EXP51_VERSION,
        "status": "complete",
        "design_hash": design["design_hash"],
        "frozen_cnn_checkpoints": int(len(manifest)),
        "environments": list(ENVIRONMENTS),
        "conditions_per_model": int(EXP51_CONDITION_COUNT),
        "completed_model_environment_units": int(completed_units),
        "validity_rows": int(len(validity)),
        "defined_field_comparison_rows": int(len(measurements)),
        "behavior_rows": int(len(behavior)),
        "model_level_rows": int(len(model_level)),
        "population_test_rows": int(len(tests)),
        "no_retraining": True,
        "no_intervention": True,
        "prior_experiments_modified": False,
    }
    write_once_json(completion, OUTDIR / "EXPERIMENT5_1_COMPLETE.json")

    summary = (
        "WORK II EXPERIMENT 5.1 COMPLETE\n"
        f"Frozen CNN checkpoints: {len(manifest)}\n"
        f"Model-environment units: {completed_units}/{expected_units}\n"
        f"Conditions per model: {EXP51_CONDITION_COUNT}\n"
        f"Defined Field-A transformation comparisons: {len(measurements)}\n"
        f"Output: {OUTDIR}\n"
        "No retraining. No intervention. No prior experiment files modified.\n"
    )
    write_once_text(summary, OUTDIR / "experiment5_1_summary.txt")
    print(summary)
    return completion

# ==================================================================================================
# 12. MAIN
# ==================================================================================================

def run_experiment5_1() -> Dict[str, Any]:
    start = time.perf_counter()

    print("=" * 116)
    print("WORK II — EXPERIMENT 5.1 — FIELD-A REALIZATION SYMMETRY")
    print("=" * 116)
    print(f"Device: {DEVICE}")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Experiment-1 result root: {EXP1_RESULTS}")
    print(f"Experiment-5.1 output root: {OUTDIR}")

    # Dataset-free self-test is always safe and fast.
    selftest_path = OUTDIR / "tables" / "dataset_free_selftest.csv"
    if selftest_path.exists():
        selftest = pd.read_csv(selftest_path)
        if "passed" not in selftest.columns or not bool(selftest.passed.astype(bool).all()):
            raise RuntimeError(f"Existing self-test file is not a passing result: {selftest_path}")
    else:
        selftest = run_dataset_free_selftest()
        write_once_csv(selftest, selftest_path)

    if SELFTEST_ONLY:
        print(selftest.to_string(index=False))
        return {"selftest_only": True}

    if REQUIRE_CUDA and DEVICE != "cuda":
        raise RuntimeError(
            "Experiment 5.1 full run requires CUDA by default. "
            "Set PATHFLOW_EXP51_REQUIRE_CUDA=0 only for a deliberate CPU diagnostic."
        )

    manifest = build_cnn_manifest()
    write_once_csv(manifest, OUTDIR / "tables" / "run_manifest.csv")
    inventory = verify_source_checkpoints(manifest)

    data_cache = load_datasets_exp51()
    population = attach_exp51_population(data_cache)
    cohort_audit = audit_prior_condition_cohorts(population)

    design = design_object(manifest, population)
    write_or_validate_design(design)

    if AGGREGATE_ONLY:
        return aggregate_and_finalize(manifest, design)

    if RUN_PREFLIGHT:
        pf_path = OUTDIR / "tables" / "real_checkpoint_preflight.csv"
        if pf_path.exists():
            preflight = pd.read_csv(pf_path)
        else:
            preflight = run_real_checkpoint_preflight(manifest, data_cache, population)
            write_once_csv(preflight, pf_path)
        if PREFLIGHT_ONLY:
            print(preflight.to_string(index=False))
            return {
                "preflight_only": True,
                "manifest_rows": int(len(manifest)),
                "population_rows": int(len(population)),
            }

    complete_marker = OUTDIR / "EXPERIMENT5_1_COMPLETE.json"
    if complete_marker.exists():
        print(f"Experiment 5.1 is already complete at {complete_marker}. No files will be rewritten.")
        return json.loads(complete_marker.read_text(encoding="utf-8"))

    total_units = int(len(manifest) * len(ENVIRONMENTS))
    done_before = sum(
        unit_complete(str(r.run_id), env)
        for _, r in manifest.iterrows()
        for env in ENVIRONMENTS
    )
    print(f"Completed model-environment units before this run: {done_before}/{total_units}")

    completed_now = 0
    wall_stop = False

    for mi, (_, row) in enumerate(manifest.iterrows(), start=1):
        for environment in ENVIRONMENTS:
            if unit_complete(str(row.run_id), environment):
                continue
            if not can_start_new_unit("Experiment-5.1 model-environment unit"):
                wall_stop = True
                break

            t0 = time.perf_counter()
            run_one_model_environment(
                row=row,
                data=data_cache[str(row.dataset)],
                population=population,
                environment=environment,
            )
            completed_now += 1
            elapsed = time.perf_counter() - t0
            done_total = done_before + completed_now
            print(
                f"[{done_total}/{total_units}] {row.run_id} | {environment} | "
                f"{human_time(elapsed)} | soft-wall remaining {human_time(soft_wall_remaining_seconds())}",
                flush=True,
            )
        if wall_stop:
            break

    completed_total = sum(
        unit_complete(str(r.run_id), env)
        for _, r in manifest.iterrows()
        for env in ENVIRONMENTS
    )

    if completed_total < total_units:
        print(
            f"Experiment 5.1 paused safely at {completed_total}/{total_units} model-environment units. "
            "Rerun the same file to resume."
        )
        return {
            "status": "paused_safely",
            "completed_model_environment_units": int(completed_total),
            "declared_model_environment_units": int(total_units),
            "runtime_seconds_this_allocation": float(time.perf_counter() - start),
        }

    result = aggregate_and_finalize(manifest, design)

    runtime_record = {
        "runtime_seconds_this_allocation": float(time.perf_counter() - start),
        "runtime_human": human_time(time.perf_counter() - start),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "device": DEVICE,
    }
    # Unique runtime audit file, never a mutable scientific result.
    write_once_json(
        runtime_record,
        OUTDIR / "audit" / f"runtime_{os.environ.get('SLURM_JOB_ID', str(int(time.time())))}.json"
    )
    return result

if __name__ == "__main__":
    EXP51_RESULTS = run_experiment5_1()
