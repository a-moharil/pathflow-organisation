from __future__ import annotations

# ==================================================================================================
# WORK II — EXPERIMENT 5 / 5.1 VISUAL EXTENSION V4
# ALL-CONDITION FIELD A + FIELD B + GEOMETRIC-SYMMETRY ATLAS
# FASHION-MNIST + MPI3D-TOY | FROZEN EXISTING RESULTS | HPC/SLURM SAFE | NO RETRAINING | NO INTERVENTION
# ==================================================================================================
#
# PURPOSE
# -------
# Experiment 5 and Experiment 5.1 already computed the numerical fields for TEN balanced conditions per
# trained CNN, but their standard rendered figures showed only condition index 0.
#
# This V4 script does NOT create a new causal experiment and does NOT recompute the intervention campaign.
# It is a non-destructive visualization extension over the already-completed numerical evidence:
#
#   1. For every selected condition, render the Experiment-5 Field A / Field B figure.
#   2. For every selected condition, render the representative Experiment-5.1 geometric Field-A figures.
#
# The condition remains exactly
#
#       c = (x, x0) = (x, 0)
#
# in normalized input coordinates.
#
# IMPORTANT TERMINOLOGY
# ---------------------
# V4 changes WHICH INPUT CLASS / STRATUM x is visualized. It does NOT silently replace x0=0 with an
# exemplar from another class.
#
# Fashion-MNIST:
#   condition 0..9 = one fixed input example from each class 0..9.
#
# MPI3D-Toy:
#   this benchmark is regression, so there are not ten semantic classes. The ten fixed conditions are
#   horizontal_axis strata 0,4,8,...,36.
#
# If a future experiment replaces x0=0 with an actual exemplar x0 from another class/stratum, that is a
# genuinely different conditional contrast and must recompute the finite-response realization itself.
# This script intentionally does not make that change.
#
# SOURCE FIDELITY
# ---------------
# Field A and Field B are READ from Experiment 5's saved NPZ evidence:
#
#   Field A:
#       P_in^c(s) = sum_channel input-node flow
#
#   Field B occurrence:
#       selected first-layer baseline occurrence-flow mass mapped to exact input source coordinates.
#
#   Field B shared identity:
#       first-layer baseline occurrence-flow footprint induced by selecting shared parameter identities
#       and unfolding their entire spatial reuse fibres.
#
# No pixel is intervened upon. Field B is only the image-space visualization of selected computational
# occurrences from the completed Experiment-5 intervention design.
#
# Experiment 5.1 Field-A symmetry arrays are READ from its saved model-environment NPZ:
#
#       original Field A
#       observed transformed Field A
#       expected geometrically transported Field A
#
# The transformed input image is reconstructed with the exact benchmark data/preprocessing/shift ordering
# so the display matches the original Experiment-5.1 renderer.
#
# DEFAULT REPRESENTATIVE SYMMETRY FIGURES
# ---------------------------------------
# Fashion-MNIST:
#       right_2, rot_ccw_20, scale_up_110
#
# MPI3D-Toy:
#       down_2
#
# Set PATHFLOW_EXP51V4_TRANSFORMS=all to render every transform stored in the Experiment-5.1 NPZ.
#
# OUTPUT SIZE AT DEFAULT REPRESENTATIVE MODE
# ------------------------------------------
# Full 162-CNN x 2-environment x 10-condition atlas:
#   3,240 Field-A/Field-B figures
#   6,480 symmetry figures
#   9,720 PNGs total
#
# OUTPUT
# ------
# A completely isolated directory is used:
#
#   pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results/
#       experiment5_1_all_conditions_visual_atlas_v4/
#           figures/
#           run_parts/
#           tables/
#           audit/
#
# No Experiment 1, 2, 3, 4, 5 or 5.1 source file is modified.
#
# HPC EXAMPLE
# -----------
#   #SBATCH --time=48:00:00
#   #SBATCH --ntasks=1
#   #SBATCH --cpus-per-task=4
#   #SBATCH --mem=64G
#   #SBATCH --gres=gpu:1
#
# This renderer itself does not require GPU model execution, so it can safely run CPU-side inside the same
# allocation:
#
#   srun python -u exp5_1_all_conditions_visual_atlas_SLURM_SAFE_v4.py
#
# Useful overrides:
#
#   PATHFLOW_EXP1_RESULTS=/absolute/path/to/pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results
#   PATHFLOW_EXP5_RESULTS=/absolute/path/to/experiment5_cnn_occurrence_shared_identity
#   PATHFLOW_EXP51_RESULTS=/absolute/path/to/experiment5_1_fieldA_realization_symmetry_geometric
#   PATHFLOW_EXP51V4_OUTDIR=/absolute/path/to/new/v4/output
#   PATHFLOW_EXP51V4_TRANSFORMS=representative     # default
#   PATHFLOW_EXP51V4_TRANSFORMS=all
#   PATHFLOW_EXP51V4_INCLUDE_CONDITION0=1         # default, complete atlas
#   PATHFLOW_EXP51V4_DPI=220
#   PATHFLOW_EXP51V4_PREFLIGHT_ONLY=1
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
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

try:
    from torchvision.datasets import FashionMNIST
except Exception as e:
    raise RuntimeError("torchvision is required for Fashion-MNIST") from e

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Headless HPC rendering.
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# ==================================================================================================
# 0. CONFIGURATION
# ==================================================================================================

V4_VERSION = "EXP5_EXP51_ALL_CONDITIONS_VISUAL_ATLAS_v4"

SEED = 730031
NOISE_SEED_BASE = 161803
CONDITION_SEED_BASE = 244949

IMAGE_RESOLUTION = 16
N_TRAIN = 20000
N_VAL = 5000
N_TEST = 5000
ENDPOINT_INPUTS = 160
CONDITIONS_PER_MODEL = 40
CONDITION_COUNT = 10
SHIFT_SNR_DB = 15.0
EPS = 1e-14

CPU_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("PATHFLOW_EXP51V4_CPU_THREADS", "4")))
torch.set_num_threads(max(1, CPU_THREADS))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SOFT_WALL_HOURS = float(os.environ.get("PATHFLOW_EXP51V4_SOFT_WALL_HOURS", "46.0"))
SOFT_START_BUFFER_MIN = float(os.environ.get("PATHFLOW_EXP51V4_SOFT_START_BUFFER_MIN", "10.0"))
START_MONO = time.perf_counter()

FIG_DPI = int(os.environ.get("PATHFLOW_EXP51V4_DPI", "220"))
INCLUDE_CONDITION0 = os.environ.get("PATHFLOW_EXP51V4_INCLUDE_CONDITION0", "1") != "0"
PREFLIGHT_ONLY = os.environ.get("PATHFLOW_EXP51V4_PREFLIGHT_ONLY", "0") == "1"
ALLOW_DOWNLOAD = os.environ.get("PATHFLOW_EXP51V4_ALLOW_DOWNLOAD", "0") == "1"

TRANSFORM_MODE = os.environ.get("PATHFLOW_EXP51V4_TRANSFORMS", "representative").strip().lower()
if TRANSFORM_MODE not in {"representative", "all"}:
    raise ValueError("PATHFLOW_EXP51V4_TRANSFORMS must be 'representative' or 'all'.")

def _script_or_submit_dir() -> Path:
    submit = os.environ.get("SLURM_SUBMIT_DIR", "").strip()
    if submit:
        return Path(submit).expanduser().resolve()
    try:
        f = globals().get("__file__", None)
        if f:
            return Path(f).expanduser().resolve().parent
    except Exception:
        pass
    return Path.cwd().resolve()

SCRIPT_DIR = _script_or_submit_dir()

_exp1 = os.environ.get("PATHFLOW_EXP1_RESULTS", "").strip()
EXP1_RESULTS = (
    Path(_exp1).expanduser().resolve()
    if _exp1
    else SCRIPT_DIR / "pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results"
)

_exp5 = os.environ.get("PATHFLOW_EXP5_RESULTS", "").strip()
EXP5_RESULTS = (
    Path(_exp5).expanduser().resolve()
    if _exp5
    else EXP1_RESULTS / "experiment5_cnn_occurrence_shared_identity"
)

_exp51 = os.environ.get("PATHFLOW_EXP51_RESULTS", "").strip()
EXP51_RESULTS = (
    Path(_exp51).expanduser().resolve()
    if _exp51
    else EXP1_RESULTS / "experiment5_1_fieldA_realization_symmetry_geometric"
)

_data = os.environ.get("PATHFLOW_EXP51V4_DATA_ROOT", "").strip()
DATA_ROOT = Path(_data).expanduser().resolve() if _data else EXP1_RESULTS / "data"

_out = os.environ.get("PATHFLOW_EXP51V4_OUTDIR", "").strip()
OUTDIR = (
    Path(_out).expanduser().resolve()
    if _out
    else EXP1_RESULTS / "experiment5_1_all_conditions_visual_atlas_v4"
)

# Never allow V4 to point at any source experiment directory.
for forbidden in [
    EXP1_RESULTS,
    EXP5_RESULTS,
    EXP51_RESULTS,
    EXP1_RESULTS / "experiment2_condition_crossover",
    EXP1_RESULTS / "experiment3_dose_response",
    EXP1_RESULTS / "experiment4_multiscale_followup",
]:
    if OUTDIR.resolve() == forbidden.resolve():
        raise RuntimeError(f"Unsafe V4 output path {OUTDIR}. Refusing to write into a source experiment directory.")

for sub in ["figures", "run_parts", "tables", "audit"]:
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

# Write probe inside V4 only.
_probe = OUTDIR / "audit" / f".write_probe_{os.getpid()}"
with open(_probe, "x", encoding="utf-8") as f:
    f.write("ok")
_probe.unlink()

FMNIST_CLASS_NAMES = {
    0: "T-shirt_top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle_boot",
}

MPI3D_URL = os.environ.get(
    "MPI3D_URL",
    "https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_toy.npz",
)

# ==================================================================================================
# 1. SAFE I/O
# ==================================================================================================

def _sha1(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def write_once_bytes(payload: bytes, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Existing identical file is accepted, differing file is protected.
        tmp = path.parent / f".{path.name}.compare.{os.getpid()}.{time.time_ns()}"
        tmp.write_bytes(payload)
        try:
            same = path.stat().st_size == tmp.stat().st_size and _sha1(path) == _sha1(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        if same:
            return False
        raise RuntimeError(f"Write-once protection refused to replace different file: {path}")

    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    except OSError:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as out, open(tmp, "rb") as src:
            shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
    finally:
        tmp.unlink(missing_ok=True)
    return True

def write_once_json(obj: Any, path: Path) -> bool:
    payload = (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    return write_once_bytes(payload, path)

def write_once_csv(df: pd.DataFrame, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Aggregate tables can be reconstructed only after completion. If already present, preserve them.
        return False
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    df.to_csv(tmp, index=False, float_format="%.17g")
    try:
        os.link(tmp, path)
    except OSError:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as out, open(tmp, "rb") as src:
            shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    finally:
        tmp.unlink(missing_ok=True)
    return True

def publish_figure_once(fig, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    tmp = path.parent / f".{path.stem}.tmp.{os.getpid()}.{time.time_ns()}{path.suffix}"
    fig.savefig(tmp, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08)
    try:
        os.link(tmp, path)
    except OSError:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as out, open(tmp, "rb") as src:
            shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    finally:
        tmp.unlink(missing_ok=True)
    return True

def soft_remaining() -> float:
    return max(0.0, SOFT_WALL_HOURS * 3600.0 - (time.perf_counter() - START_MONO))

def can_start_unit() -> bool:
    if soft_remaining() <= SOFT_START_BUFFER_MIN * 60:
        write_once_json(
            {
                "status": "paused_safely",
                "remaining_seconds": soft_remaining(),
                "soft_wall_hours": SOFT_WALL_HOURS,
                "buffer_minutes": SOFT_START_BUFFER_MIN,
            },
            OUTDIR / "audit" / f"PAUSED_{int(time.time())}.json",
        )
        return False
    return True

# ==================================================================================================
# 2. EXACT DATASET / COHORT RECONSTRUCTION FROM THE BENCHMARK
# ==================================================================================================

def stable_hash_dict(d: Dict[str, Any], n: int = 12) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:n]

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

def _split_indices_stratified(labels: np.ndarray, ntr: int, nv: int, nt: int, seed: int):
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
        va.extend(ids[a:a+b])
        te.extend(ids[a+b:])

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
    va = fill(va, nv, np.setdiff1d(pool, tr))
    te = fill(te, nt, np.setdiff1d(pool, np.concatenate([tr, va])))
    return tr, va, te

def balanced_indices_from_strata(strata: np.ndarray, n: int, seed: int) -> np.ndarray:
    strata = np.asarray(strata)
    classes = np.asarray(sorted(np.unique(strata).tolist()))
    rng = np.random.default_rng(int(seed))
    n = min(int(n), len(strata))
    base = n // len(classes)
    rem = n % len(classes)
    extra = set(rng.permutation(classes)[:rem].tolist())

    chosen, leftovers = {}, []
    for c in classes:
        ids = np.where(strata == c)[0].copy()
        rng.shuffle(ids)
        take = base + (1 if c in extra else 0)
        chosen[c] = ids[:min(take, len(ids))].tolist()
        leftovers.extend(ids[min(take, len(ids)):].tolist())

    selected = sum((v for v in chosen.values()), [])
    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:n-len(selected)])

    queues = {c: list(chosen[c]) for c in classes}
    out = []
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
        out.extend([i for i in selected if i not in used][:n-len(out)])
    return np.asarray(out[:n], dtype=int)

def attach_endpoint_and_causal_population(data: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    strata = np.asarray(data["test_strata"])
    endpoint_ids = balanced_indices_from_strata(
        strata, ENDPOINT_INPUTS, SEED + 1009 + (0 if dataset == "fmnist" else 1)
    )
    endpoint_strata = strata[endpoint_ids]
    causal_local = balanced_indices_from_strata(
        endpoint_strata,
        min(CONDITIONS_PER_MODEL, len(endpoint_ids)),
        CONDITION_SEED_BASE + (0 if dataset == "fmnist" else 1),
    )
    data["endpoint_indices"] = endpoint_ids
    data["causal_endpoint_local_indices"] = causal_local
    return data

def select_ten_condition_local_indices(data: Dict[str, Any], dataset: str) -> np.ndarray:
    endpoint_ids = np.asarray(data["endpoint_indices"], int)
    causal_local = np.asarray(data["causal_endpoint_local_indices"], int)
    strata = np.asarray(data["test_strata"])
    causal_strata = strata[endpoint_ids[causal_local]]

    if dataset == "fmnist":
        out = []
        for cls in range(10):
            ids = np.flatnonzero(causal_strata == cls)
            if len(ids) == 0:
                raise RuntimeError(f"No frozen FMNIST class {cls} in causal population.")
            out.append(int(causal_local[int(ids[0])]))
        return np.asarray(out, int)

    out = []
    for level in range(0, 40, 4):
        ids = np.flatnonzero(causal_strata == level)
        if len(ids) == 0:
            raise RuntimeError(f"No frozen MPI3D horizontal level {level} in causal population.")
        out.append(int(causal_local[int(ids[0])]))
    return np.asarray(out, int)

def _ensure_mpi3d_npz() -> Path:
    env = os.environ.get("MPI3D_NPZ", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        DATA_ROOT / "mpi3d_toy.npz",
        SCRIPT_DIR / "mpi3d_toy.npz",
        SCRIPT_DIR / "data" / "mpi3d_toy.npz",
        Path.cwd() / "mpi3d_toy.npz",
        Path.cwd() / "data" / "mpi3d_toy.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    if not ALLOW_DOWNLOAD:
        raise FileNotFoundError(
            "MPI3D-Toy not found. Set MPI3D_NPZ=/absolute/path/to/mpi3d_toy.npz "
            "or PATHFLOW_EXP51V4_ALLOW_DOWNLOAD=1."
        )
    p = DATA_ROOT / "mpi3d_toy.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".npz.part")
    urllib.request.urlretrieve(MPI3D_URL, tmp)
    tmp.replace(p)
    return p

def load_mpi3d():
    p = _ensure_mpi3d_npz()
    factor_sizes = np.asarray([6, 6, 2, 3, 3, 40, 40], dtype=np.int64)
    expected_n = int(np.prod(factor_sizes))
    all_ids = np.arange(expected_n, dtype=np.int64)
    xpos = ((all_ids // int(factor_sizes[-1])) % int(factor_sizes[-2])).astype(np.int64)

    tr, va, te = _split_indices_stratified(xpos, N_TRAIN, N_VAL, N_TEST, SEED + 29)
    ids = np.concatenate([tr, va, te])

    with np.load(p, allow_pickle=False) as z:
        images = z["images"]
        if images.shape[0] != expected_n:
            raise RuntimeError(f"Unexpected MPI3D population size {images.shape[0]}.")
        imgs = images[ids].astype(np.float32) / 255.0

    imgs = _resize_images_np(imgs, IMAGE_RESOLUTION)
    ntr, nv = len(tr), len(va)
    Xtr, Xv, Xt = imgs[:ntr], imgs[ntr:ntr+nv], imgs[ntr+nv:]
    yall = xpos[ids].astype(np.float32)
    ytr, yv, yt = yall[:ntr], yall[ntr:ntr+nv], yall[ntr+nv:]

    mu = float(Xtr.mean())
    sd = float(Xtr.std() + 1e-8)

    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xv = ((Xv - mu) / sd).astype(np.float32)
    Xt = ((Xt - mu) / sd).astype(np.float32)

    ym = float(ytr.mean())
    ys = float(ytr.std() + 1e-8)

    data = {
        "test": (torch.from_numpy(Xt), torch.from_numpy(((yt - ym) / ys).astype(np.float32)[:, None])),
        "test_strata": xpos[te].copy(),
        "meta": {
            "dataset": "mpi3d",
            "input_mean": mu,
            "input_sd": sd,
            "target_mean": ym,
            "target_sd": ys,
        },
    }
    return attach_endpoint_and_causal_population(data, "mpi3d")

def load_fmnist():
    root = Path(os.environ.get("PATHFLOW_EXP51V4_FMNIST_ROOT", str(DATA_ROOT / "fmnist"))).expanduser().resolve()
    ds_tr = FashionMNIST(root=str(root), train=True, download=ALLOW_DOWNLOAD)
    ds_te = FashionMNIST(root=str(root), train=False, download=ALLOW_DOWNLOAD)

    Xtr0 = _resize_images_np(ds_tr.data.numpy().astype(np.float32) / 255.0, IMAGE_RESOLUTION)
    Xte0 = _resize_images_np(ds_te.data.numpy().astype(np.float32) / 255.0, IMAGE_RESOLUTION)
    ytr0 = ds_tr.targets.numpy().astype(np.int64)
    yte0 = ds_te.targets.numpy().astype(np.int64)

    tr, va, _ = _split_indices_stratified(
        ytr0, N_TRAIN, N_VAL, max(1, min(1000, len(ytr0)-N_TRAIN-N_VAL)), SEED + 11
    )
    _, _, te = _split_indices_stratified(
        yte0,
        max(1, min(1000, len(yte0)-N_TEST-1)),
        max(1, min(500, len(yte0)-N_TEST-1)),
        N_TEST,
        SEED + 17,
    )

    mu = float(Xtr0[tr].mean())
    sd = float(Xtr0[tr].std() + 1e-8)
    Xt = ((Xte0[te] - mu) / sd).astype(np.float32)

    data = {
        "test": (torch.from_numpy(Xt), torch.from_numpy(yte0[te])),
        "test_strata": yte0[te].copy(),
        "meta": {"dataset": "fmnist", "input_mean": mu, "input_sd": sd},
    }
    return attach_endpoint_and_causal_population(data, "fmnist")

def shifted_environment(X: torch.Tensor, dataset: str, meta: Dict[str, Any]) -> torch.Tensor:
    x = X.float()
    mu = float(meta["input_mean"])
    sd = float(meta["input_sd"])
    raw = x * sd + mu
    signal = float(torch.mean(raw * raw).item())
    noise_var = signal / (10.0 ** (SHIFT_SNR_DB / 10.0))
    g = torch.Generator(device="cpu").manual_seed(
        NOISE_SEED_BASE + int(stable_hash_dict({"d": dataset}, 8), 16) % 1000003
    )
    noise = torch.randn(raw.shape, generator=g, dtype=raw.dtype) * math.sqrt(max(noise_var, EPS))
    return (raw + noise - mu) / max(sd, EPS)

def prepared_ten_environment(data: Dict[str, Any], dataset: str, environment: str):
    Xt, yt = data["test"]
    endpoint_ids = np.asarray(data["endpoint_indices"], int)
    local = select_ten_condition_local_indices(data, dataset)

    Xep = Xt[endpoint_ids]
    yep = yt[endpoint_ids]
    if environment == "shifted":
        # Exact benchmark ordering. Shift the complete 160-endpoint population before subselecting.
        Xep = shifted_environment(Xep, dataset, data["meta"])

    data["v4_condition_local_indices"] = local
    return Xep[local], yep[local]

def image_for_display(x: torch.Tensor, meta: Dict[str, Any]) -> np.ndarray:
    arr = x[0].detach().cpu().numpy().astype(np.float64)
    raw = arr * float(meta["input_sd"]) + float(meta["input_mean"])
    if raw.shape[0] == 1:
        raw = raw[0]
    else:
        raw = np.moveaxis(raw, 0, -1)
    return np.clip(raw, 0.0, 1.0)

# ==================================================================================================
# 3. GEOMETRIC INPUT TRANSFORMS, MATCHING EXPERIMENT 5.1
# ==================================================================================================

def _translation(transform_id: str, dy: int, dx: int):
    return dict(transform_id=transform_id, transform_family="translation", dy=dy, dx=dx, angle_deg=0.0, scale_factor=1.0)

def _rotation(transform_id: str, angle_deg: float):
    return dict(transform_id=transform_id, transform_family="rotation", dy=0, dx=0, angle_deg=float(angle_deg), scale_factor=1.0)

def _scaling(transform_id: str, scale_factor: float):
    return dict(transform_id=transform_id, transform_family="scale", dy=0, dx=0, angle_deg=0.0, scale_factor=float(scale_factor))

def transform_dictionary(dataset: str) -> Dict[str, Dict[str, Any]]:
    if dataset == "fmnist":
        ts = [
            _translation("left_1",0,-1), _translation("right_1",0,1),
            _translation("up_1",-1,0), _translation("down_1",1,0),
            _translation("left_2",0,-2), _translation("right_2",0,2),
            _translation("up_2",-2,0), _translation("down_2",2,0),
            _rotation("rot_cw_10",-10.0), _rotation("rot_ccw_10",10.0),
            _rotation("rot_cw_20",-20.0), _rotation("rot_ccw_20",20.0),
            _scaling("scale_down_090",0.90), _scaling("scale_up_110",1.10),
        ]
    else:
        ts = [
            _translation("up_1",-1,0), _translation("down_1",1,0),
            _translation("up_2",-2,0), _translation("down_2",2,0),
        ]
    return {t["transform_id"]: t for t in ts}

def translate_tensor(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    n,c,h,w = x.shape
    out = torch.zeros_like(x)
    sy0, sy1 = max(0,-dy), min(h,h-dy)
    sx0, sx1 = max(0,-dx), min(w,w-dx)
    dy0, dx0 = max(0,dy), max(0,dx)
    out[..., dy0:dy0+(sy1-sy0), dx0:dx0+(sx1-sx0)] = x[..., sy0:sy1, sx0:sx1]
    return out

def _affine_theta(angle_deg: float, scale_factor: float, dtype, device):
    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)
    inv = 1.0 / float(scale_factor)
    return torch.tensor(
        [[c*inv, -s*inv, 0.0], [s*inv, c*inv, 0.0]],
        dtype=dtype, device=device
    ).unsqueeze(0)

def affine_tensor(x: torch.Tensor, angle_deg=0.0, scale_factor=1.0):
    theta = _affine_theta(angle_deg, scale_factor, x.dtype, x.device)
    if x.shape[0] != 1:
        theta = theta.expand(x.shape[0], -1, -1).contiguous()
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

def apply_input_transform(x: torch.Tensor, tr: Dict[str, Any]) -> torch.Tensor:
    fam = tr["transform_family"]
    if fam == "translation":
        return translate_tensor(x, int(tr["dy"]), int(tr["dx"]))
    if fam == "rotation":
        return affine_tensor(x, angle_deg=float(tr["angle_deg"]), scale_factor=1.0)
    if fam == "scale":
        return affine_tensor(x, angle_deg=0.0, scale_factor=float(tr["scale_factor"]))
    raise ValueError(fam)

def transform_display_label(tr: Dict[str, Any]) -> str:
    fam = tr["transform_family"]
    if fam == "translation":
        return f"{tr['transform_id']} ({int(tr['dy']):+d},{int(tr['dx']):+d}) px"
    if fam == "rotation":
        return f"{float(tr['angle_deg']):+g} degrees"
    return f"scale {float(tr['scale_factor']):.2f}"

# ==================================================================================================
# 4. SOURCE RESULT PATHS AND AUDITS
# ==================================================================================================

def exp5_condition_npz(run_id: str, env: str, ci: int) -> Path:
    return EXP5_RESULTS / "fields" / f"{run_id}__EXP5__{env}__C{ci:02d}__fields.npz"

def exp5_condition_measurements(run_id: str, env: str, ci: int) -> Path:
    return EXP5_RESULTS / "run_parts" / f"{run_id}__EXP5__{env}__C{ci:02d}__measurements.csv"

def exp51_fields(run_id: str, env: str) -> Path:
    return EXP51_RESULTS / "fields" / f"{run_id}__EXP51__{env}__fields.npz"

def exp51_measurements(run_id: str, env: str) -> Path:
    return EXP51_RESULTS / "run_parts" / f"{run_id}__EXP51__{env}__measurements.csv"

def load_and_audit_manifests() -> Tuple[pd.DataFrame, pd.DataFrame]:
    p5 = EXP5_RESULTS / "tables" / "exp5_condition_population.csv"
    p51 = EXP51_RESULTS / "tables" / "experiment5_1_condition_population.csv"
    pm = EXP51_RESULTS / "tables" / "run_manifest.csv"

    for p in [p5,p51,pm]:
        if not p.exists():
            raise FileNotFoundError(f"Required source table not found: {p}")

    c5 = pd.read_csv(p5)
    c51 = pd.read_csv(p51)
    manifest = pd.read_csv(pm)

    cols = ["dataset","test_index","stratum"]
    a = c5[cols].copy()
    b = c51[cols].copy()
    a["dataset"] = a["dataset"].astype(str)
    b["dataset"] = b["dataset"].astype(str)
    a["test_index"] = pd.to_numeric(a["test_index"]).astype(int)
    b["test_index"] = pd.to_numeric(b["test_index"]).astype(int)
    a["stratum"] = pd.to_numeric(a["stratum"], errors="coerce")
    b["stratum"] = pd.to_numeric(b["stratum"], errors="coerce")
    a = a.sort_values(cols).reset_index(drop=True)
    b = b.sort_values(cols).reset_index(drop=True)

    if len(a) != len(b) or not a.equals(b):
        raise RuntimeError("Experiment 5 and Experiment 5.1 condition cohorts do not match. Refusing atlas generation.")

    manifest = manifest[manifest["family"].astype(str) == "cnn"].copy().reset_index(drop=True)
    if len(manifest) != 162:
        raise RuntimeError(f"Expected 162 frozen CNNs in Experiment 5.1 manifest, found {len(manifest)}.")

    write_once_json(
        {
            "experiment5_population_rows": int(len(c5)),
            "experiment5_1_population_rows": int(len(c51)),
            "cohort_match": True,
            "cnn_manifest_rows": int(len(manifest)),
        },
        OUTDIR / "audit" / "source_cohort_audit.json",
    )
    return manifest, c51

def class_label(dataset: str, stratum: int) -> str:
    if dataset == "fmnist":
        return f"class {int(stratum)} {FMNIST_CLASS_NAMES[int(stratum)].replace('_','/')}"
    return f"horizontal_axis stratum {int(stratum)}"

def condition_folder(dataset: str, ci: int, stratum: int) -> str:
    if dataset == "fmnist":
        return f"C{ci:02d}_class{int(stratum):02d}_{FMNIST_CLASS_NAMES[int(stratum)]}"
    return f"C{ci:02d}_xpos{int(stratum):02d}"

def expected_transform_ids(dataset: str, npz_ids: List[str]) -> List[str]:
    ids = [x for x in npz_ids if x != "identity"]
    if TRANSFORM_MODE == "all":
        return ids
    wanted = ["right_2","rot_ccw_20","scale_up_110"] if dataset == "fmnist" else ["down_2"]
    missing = [x for x in wanted if x not in ids]
    if missing:
        raise RuntimeError(f"Experiment 5.1 fields are missing representative transform(s): {missing}")
    return wanted

# ==================================================================================================
# 5. OVERFLOW-SAFE FIELD A / FIELD B FIGURE
# ==================================================================================================

def render_fieldA_fieldB(
    path: Path,
    image_display: np.ndarray,
    A: np.ndarray,
    Bocc: np.ndarray,
    Bparam: np.ndarray,
    title: str,
    Docc: float,
    Dparam: float,
    Hocc: float,
    Hparam: float,
):
    if path.exists():
        return False

    vmaxA = max(float(np.nanmax(A)), EPS)
    vmaxB = max(float(np.nanmax(Bocc)), float(np.nanmax(Bparam)), EPS)
    diff = Bocc - Bparam
    vmaxD = max(float(np.nanmax(np.abs(diff))), EPS)

    # 2 rows x 5 columns.
    # Cols 0..2 are scientific panels.
    # Col 3 is a dedicated shared colorbar column.
    # Col 4 holds the bottom-row signed-difference colorbar only.
    # No colorbar can overlap a data panel.
    fig = plt.figure(figsize=(14.2, 8.1), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1,1,1,0.045,0.045],
        height_ratios=[1,1],
        wspace=0.08, hspace=0.10,
    )

    axes = np.empty((2,3), dtype=object)
    for rr in range(2):
        for cc in range(3):
            axes[rr,cc] = fig.add_subplot(gs[rr,cc])

    caxA = fig.add_subplot(gs[0,3])
    blank = fig.add_subplot(gs[0,4])
    blank.axis("off")
    caxB = fig.add_subplot(gs[1,3])
    caxD = fig.add_subplot(gs[1,4])

    if image_display.ndim == 2:
        axes[0,0].imshow(image_display, cmap="gray", vmin=0, vmax=1)
        axes[0,2].imshow(image_display, cmap="gray", vmin=0, vmax=1)
    else:
        axes[0,0].imshow(image_display)
        axes[0,2].imshow(image_display)

    axes[0,0].set_title("Input condition x")
    imA = axes[0,1].imshow(A, cmap="magma", vmin=0, vmax=vmaxA)
    axes[0,1].set_title("Field A  input-origin path mass")
    axes[0,2].imshow(A, cmap="magma", alpha=0.58, vmin=0, vmax=vmaxA)
    axes[0,2].set_title("Field A overlay")

    imO = axes[1,0].imshow(Bocc, cmap="viridis", vmin=0, vmax=vmaxB)
    axes[1,0].set_title(f"Field B occurrence\nselected flow mass={np.sum(Bocc):.4f}")

    axes[1,1].imshow(Bparam, cmap="viridis", vmin=0, vmax=vmaxB)
    axes[1,1].set_title(f"Field B shared identity\nselected flow mass={np.sum(Bparam):.4f}")

    imD = axes[1,2].imshow(diff, cmap="coolwarm", vmin=-vmaxD, vmax=vmaxD)
    axes[1,2].set_title("Field B  occurrence - parameter")

    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])

    cbA = fig.colorbar(imA, cax=caxA)
    cbA.set_label("Field A probability mass", labelpad=10)
    cbA.ax.tick_params(labelsize=8)

    cbB = fig.colorbar(imO, cax=caxB)
    cbB.set_label("Selected occurrence flow", labelpad=10)
    cbB.ax.tick_params(labelsize=8)

    cbD = fig.colorbar(imD, cax=caxD)
    cbD.set_label("Occurrence - parameter", labelpad=10)
    cbD.ax.tick_params(labelsize=8)

    fig.suptitle(
        title
        + f"\nD_man occ={Docc:.4g}, param={Dparam:.4g}"
        + f" | D_H occ={Hocc:.4g}, param={Hparam:.4g}"
        + " | reference x0=0",
        fontsize=10.5,
    )

    ok = publish_figure_once(fig, path)
    plt.close(fig)
    return ok

# ==================================================================================================
# 6. OVERFLOW-SAFE EXPERIMENT-5.1 SYMMETRY FIGURE
# ==================================================================================================

def render_symmetry(
    path: Path,
    original_display: np.ndarray,
    transformed_display: np.ndarray,
    A0: np.ndarray,
    Aobs: np.ndarray,
    Aexp: np.ndarray,
    title: str,
    H_eq: float,
    H_inv: float,
    advantage: float,
):
    if path.exists():
        return False

    resid = np.abs(Aobs - Aexp)
    vmaxA = max(float(np.nanmax(A0)), float(np.nanmax(Aobs)), float(np.nanmax(Aexp)), EPS)
    vmaxR = max(float(np.nanmax(resid)), EPS)

    fig = plt.figure(figsize=(13.6, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1,1,1,0.055],
        height_ratios=[1,1],
        wspace=0.08, hspace=0.10,
    )

    axes = np.empty((2,3), dtype=object)
    for rr in range(2):
        for cc in range(3):
            axes[rr,cc] = fig.add_subplot(gs[rr,cc])
    caxA = fig.add_subplot(gs[0,3])
    caxR = fig.add_subplot(gs[1,3])

    if original_display.ndim == 2:
        axes[0,0].imshow(original_display, cmap="gray", vmin=0, vmax=1)
        axes[1,0].imshow(transformed_display, cmap="gray", vmin=0, vmax=1)
    else:
        axes[0,0].imshow(original_display)
        axes[1,0].imshow(transformed_display)

    axes[0,0].set_title("Original input")
    axes[1,0].set_title("Transformed input")

    imA = axes[0,1].imshow(A0, cmap="magma", vmin=0, vmax=vmaxA)
    axes[0,1].set_title("Original Field A")

    axes[0,2].imshow(Aexp, cmap="magma", vmin=0, vmax=vmaxA)
    axes[0,2].set_title("Expected if equivariant")

    axes[1,1].imshow(Aobs, cmap="magma", vmin=0, vmax=vmaxA)
    axes[1,1].set_title("Observed transformed Field A")

    imR = axes[1,2].imshow(resid, cmap="viridis", vmin=0, vmax=vmaxR)
    axes[1,2].set_title("Absolute residual")

    # Keep a blank lower-right scientific slot? No. All six slots have content except there are only five conceptual
    # panels. We use the sixth slot for a compact text summary rather than another image.
    axes[0,0].set_title("Original input")
    axes[1,0].set_title("Transformed input")

    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])

    cbA = fig.colorbar(imA, cax=caxA)
    cbA.set_label("Field A probability mass", labelpad=10)
    cbA.ax.tick_params(labelsize=8)

    cbR = fig.colorbar(imR, cax=caxR)
    cbR.set_label("Absolute residual", labelpad=10)
    cbR.ax.tick_params(labelsize=8)

    fig.suptitle(
        title
        + f"\nH_eq={H_eq:.4f}  H_inv={H_inv:.4f}  advantage={advantage:+.4f}"
        + " | reference x0=0",
        fontsize=10.5,
    )

    ok = publish_figure_once(fig, path)
    plt.close(fig)
    return ok

# ==================================================================================================
# 7. UNIT RENDERING
# ==================================================================================================

def unit_done_path(run_id: str, env: str, ci: int) -> Path:
    return OUTDIR / "run_parts" / f"{run_id}__V4__{env}__C{ci:02d}__DONE.json"

def completed_unit_record(run_id: str, env: str, ci: int) -> Optional[Dict[str, Any]]:
    """
    A completion marker is trusted only when every figure it names still exists.

    If a prior allocation wrote the marker but a later manual cleanup removed one or more PNGs, V4 will
    regenerate the missing figure(s) from the immutable source fields instead of silently skipping the unit.
    """
    p = unit_done_path(run_id, env, ci)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        raise RuntimeError(f"Unreadable V4 completion marker: {p}")

    paths = [Path(str(rec.get("fieldA_fieldB_figure", "")))]
    paths += [Path(str(x)) for x in rec.get("symmetry_figures", [])]
    if not paths or any((not q.exists()) or q.stat().st_size <= 0 for q in paths):
        return None
    return rec

def source_measurement_pair(path: Path) -> Tuple[float,float,float,float]:
    df = pd.read_csv(path)
    if "intervention_type" not in df.columns:
        return np.nan,np.nan,np.nan,np.nan
    d = df.set_index("intervention_type")
    try:
        return (
            float(d.loc["occurrence","manifestation_abs"]),
            float(d.loc["parameter","manifestation_abs"]),
            float(d.loc["occurrence","path_hellinger"]),
            float(d.loc["parameter","path_hellinger"]),
        )
    except Exception:
        return np.nan,np.nan,np.nan,np.nan

def render_one_condition(
    row: pd.Series,
    env: str,
    ci: int,
    stratum: int,
    x: torch.Tensor,
    data_meta: Dict[str, Any],
) -> Dict[str, Any]:
    run_id = str(row.run_id)
    dataset = str(row.dataset)

    p5 = exp5_condition_npz(run_id, env, ci)
    pm5 = exp5_condition_measurements(run_id, env, ci)
    p51 = exp51_fields(run_id, env)
    pm51 = exp51_measurements(run_id, env)

    for p in [p5, pm5, p51, pm51]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required completed source result: {p}")

    folder = OUTDIR / "figures" / dataset / run_id / env / condition_folder(dataset,ci,stratum)
    folder.mkdir(parents=True, exist_ok=True)

    with np.load(p5, allow_pickle=False) as z:
        required = ["image_display","fieldA_baseline","fieldB_occurrence","fieldB_parameter"]
        for k in required:
            if k not in z.files:
                raise RuntimeError(f"{p5} missing key {k}.")
        image5 = np.asarray(z["image_display"])
        A5 = np.asarray(z["fieldA_baseline"], dtype=np.float64)
        Bocc = np.asarray(z["fieldB_occurrence"], dtype=np.float64)
        Bparam = np.asarray(z["fieldB_parameter"], dtype=np.float64)

    Docc,Dparam,Hocc,Hparam = source_measurement_pair(pm5)

    reconstructed = image_for_display(x, data_meta)
    display_max_abs = float(np.max(np.abs(np.asarray(image5,float) - np.asarray(reconstructed,float))))
    if display_max_abs > 5e-5:
        raise RuntimeError(
            f"Reconstructed image does not match Experiment-5 saved display for {run_id}/{env}/C{ci:02d}; "
            f"max abs diff={display_max_abs}."
        )

    label = class_label(dataset, stratum)
    common_title = f"{run_id} | {env} | condition {ci} | {label}"

    fab_path = folder / f"{run_id}__{env}__C{ci:02d}__fieldA_fieldB_v4.png"
    made_field = render_fieldA_fieldB(
        fab_path, image5, A5, Bocc, Bparam,
        common_title, Docc,Dparam,Hocc,Hparam
    )

    with np.load(p51, allow_pickle=False) as z:
        for k in ["baseline_fields","transformed_fields","expected_equivariant_fields","transform_id"]:
            if k not in z.files:
                raise RuntimeError(f"{p51} missing key {k}.")
        A51_all = np.asarray(z["baseline_fields"], dtype=np.float64)
        Obs_all = np.asarray(z["transformed_fields"], dtype=np.float64)
        Exp_all = np.asarray(z["expected_equivariant_fields"], dtype=np.float64)
        tids = [str(v) for v in np.asarray(z["transform_id"]).tolist()]

    if ci >= A51_all.shape[0]:
        raise RuntimeError(f"Condition {ci} is outside Experiment-5.1 field array {A51_all.shape}.")

    baseline_diff = float(np.nanmax(np.abs(A5 - A51_all[ci])))
    if baseline_diff > 5e-8:
        raise RuntimeError(
            f"Experiment-5 and Experiment-5.1 baseline Field A mismatch for {run_id}/{env}/C{ci:02d}; "
            f"max abs diff={baseline_diff}."
        )

    m51 = pd.read_csv(pm51)
    m51 = m51[pd.to_numeric(m51["condition_index"]).astype(int) == int(ci)].copy()

    transform_defs = transform_dictionary(dataset)
    selected_ids = expected_transform_ids(dataset, tids)
    sym_paths = []
    made_sym = 0

    for tid in selected_ids:
        ti = tids.index(tid)
        if tid not in transform_defs:
            raise RuntimeError(f"No V4 input-transform definition for stored Experiment-5.1 transform {tid}.")

        mt = m51[m51["transform_id"].astype(str) == tid]
        if len(mt) != 1:
            raise RuntimeError(f"Expected one Experiment-5.1 measurement row for {run_id}/{env}/C{ci:02d}/{tid}, got {len(mt)}.")
        mr = mt.iloc[0]

        xt = apply_input_transform(x, transform_defs[tid])
        transformed_display = image_for_display(xt, data_meta)

        sym_path = folder / f"{run_id}__{env}__C{ci:02d}__{tid}__symmetry_v4.png"
        made_sym += int(render_symmetry(
            sym_path,
            reconstructed,
            transformed_display,
            A51_all[ci],
            Obs_all[ci,ti],
            Exp_all[ci,ti],
            f"{common_title} | {transform_display_label(transform_defs[tid])}",
            float(mr["hellinger_equivariant"]),
            float(mr["hellinger_invariant"]),
            float(mr["equivariance_advantage"]),
        ))
        sym_paths.append(str(sym_path))

    rec = {
        "run_id": run_id,
        "dataset": dataset,
        "architecture": str(row.arch_id),
        "optimizer": str(row.optimizer),
        "activation": str(row.activation),
        "seed_index": int(row.seed_index),
        "environment": env,
        "condition_index": int(ci),
        "stratum": int(stratum),
        "condition_label": label,
        "reference": "x0=0 normalized coordinates",
        "fieldA_fieldB_figure": str(fab_path),
        "symmetry_figure_count": int(len(sym_paths)),
        "symmetry_figures": sym_paths,
        "fieldA_cross_experiment_max_abs_diff": baseline_diff,
        "input_display_reconstruction_max_abs_diff": display_max_abs,
        "fieldB_occurrence_flow_mass": float(np.nansum(Bocc)),
        "fieldB_parameter_flow_mass": float(np.nansum(Bparam)),
    }

    write_once_json(rec, unit_done_path(run_id,env,ci))
    return rec

# ==================================================================================================
# 8. PREFLIGHT
# ==================================================================================================

def run_preflight(manifest: pd.DataFrame, population: pd.DataFrame, data_cache: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset in ["fmnist","mpi3d"]:
        mr = manifest[manifest["dataset"].astype(str)==dataset].iloc[0]
        run_id = str(mr.run_id)
        data = data_cache[dataset]
        X,y = prepared_ten_environment(data,dataset,"clean")

        pop = population[population["dataset"].astype(str)==dataset].sort_values("exp51_condition_index")
        if len(pop) != 10:
            raise RuntimeError(f"Expected 10 source conditions for {dataset}, found {len(pop)}.")

        # Test condition 0 and final condition 9 so this specifically verifies the new V4 condition expansion.
        for ci in [0,9]:
            p5 = exp5_condition_npz(run_id,"clean",ci)
            p51 = exp51_fields(run_id,"clean")
            if not p5.exists() or not p51.exists():
                raise FileNotFoundError(f"Preflight source fields missing for {dataset} condition {ci}.")

            with np.load(p5, allow_pickle=False) as z5, np.load(p51, allow_pickle=False) as z51:
                A5 = np.asarray(z5["fieldA_baseline"],float)
                A51 = np.asarray(z51["baseline_fields"],float)[ci]
                img = np.asarray(z5["image_display"],float)
                recon = image_for_display(X[ci:ci+1],data["meta"])

            rows.append({
                "dataset":dataset,
                "run_id":run_id,
                "condition_index":ci,
                "stratum":int(pop.iloc[ci].stratum),
                "fieldA_max_abs_diff_exp5_vs_exp51":float(np.max(np.abs(A5-A51))),
                "image_display_max_abs_diff":float(np.max(np.abs(img-recon))),
                "fieldA_sum":float(np.sum(A5)),
            })

    df = pd.DataFrame(rows)
    if np.nanmax(df["fieldA_max_abs_diff_exp5_vs_exp51"]) > 5e-8:
        raise RuntimeError("Preflight found Experiment-5 / Experiment-5.1 Field-A mismatch.")
    if np.nanmax(df["image_display_max_abs_diff"]) > 5e-5:
        raise RuntimeError("Preflight found reconstructed input-display mismatch.")
    if np.nanmax(np.abs(df["fieldA_sum"]-1.0)) > 5e-6:
        raise RuntimeError("Preflight found invalid Field-A normalization.")
    return df

# ==================================================================================================
# 9. MAIN
# ==================================================================================================

def main():
    print("="*116)
    print("WORK II — EXPERIMENT 5 / 5.1 ALL-CONDITION VISUAL ATLAS V4")
    print("="*116)
    print(f"Experiment 5 source : {EXP5_RESULTS}")
    print(f"Experiment 5.1 source: {EXP51_RESULTS}")
    print(f"V4 output            : {OUTDIR}")
    print(f"Transform rendering  : {TRANSFORM_MODE}")
    print("Scientific operations: READ EXISTING FIELDS + RENDER ONLY")
    print("Reference condition  : x0=0 is preserved; input class/stratum x varies")

    manifest, population = load_and_audit_manifests()

    data_cache = {
        "fmnist": load_fmnist(),
        "mpi3d": load_mpi3d(),
    }

    # Confirm reconstructed V4 ten-condition cohort against saved source table.
    cohort_rows = []
    for ds,data in data_cache.items():
        local = select_ten_condition_local_indices(data,ds)
        endpoint_ids = np.asarray(data["endpoint_indices"],int)
        strata = np.asarray(data["test_strata"])
        for ci,ep_local in enumerate(local):
            cohort_rows.append({
                "dataset":ds,
                "condition_index":ci,
                "test_index":int(endpoint_ids[int(ep_local)]),
                "stratum":int(strata[int(endpoint_ids[int(ep_local)])]),
            })
    cohort = pd.DataFrame(cohort_rows)

    source = population.rename(columns={"exp51_condition_index":"condition_index"})[
        ["dataset","condition_index","test_index","stratum"]
    ].copy()
    source["condition_index"] = pd.to_numeric(source["condition_index"]).astype(int)
    source["test_index"] = pd.to_numeric(source["test_index"]).astype(int)
    source["stratum"] = pd.to_numeric(source["stratum"]).astype(int)
    source = source.sort_values(["dataset","condition_index"]).reset_index(drop=True)
    cohort = cohort.sort_values(["dataset","condition_index"]).reset_index(drop=True)

    if not cohort.equals(source):
        raise RuntimeError("Reconstructed V4 class/stratum cohort does not match Experiment-5.1 saved cohort.")

    write_once_csv(cohort, OUTDIR/"tables"/"v4_condition_population.csv")

    pf_path = OUTDIR/"tables"/"preflight.csv"
    if pf_path.exists():
        preflight = pd.read_csv(pf_path)
    else:
        preflight = run_preflight(manifest,population,data_cache)
        write_once_csv(preflight,pf_path)

    design = {
        "version":V4_VERSION,
        "purpose":"render every fixed Experiment-5/5.1 condition instead of only condition 0",
        "reference":"c=(x,0), x0 remains zero in normalized coordinates",
        "fmnist_conditions":"classes 0..9, one fixed balanced exemplar each",
        "mpi3d_conditions":"horizontal_axis strata 0,4,...,36",
        "transform_mode":TRANSFORM_MODE,
        "representative_transforms":{
            "fmnist":["right_2","rot_ccw_20","scale_up_110"],
            "mpi3d":["down_2"],
        },
        "source_exp5":str(EXP5_RESULTS),
        "source_exp51":str(EXP51_RESULTS),
        "output":str(OUTDIR),
        "no_retraining":True,
        "no_intervention_recomputation":True,
        "no_source_result_overwrite":True,
    }
    write_once_json(design,OUTDIR/"audit"/"design_manifest.json")

    if PREFLIGHT_ONLY:
        print(preflight.to_string(index=False))
        print("PREFLIGHT ONLY COMPLETE")
        return

    index_rows = []
    total_units = len(manifest)*2*10
    done = 0

    for _,row in manifest.iterrows():
        ds = str(row.dataset)
        data = data_cache[ds]
        pop = population[population["dataset"].astype(str)==ds].sort_values("exp51_condition_index").reset_index(drop=True)

        for env in ["clean","shifted"]:
            X,y = prepared_ten_environment(data,ds,env)
            if len(X) != 10 or len(pop) != 10:
                raise RuntimeError(f"Expected ten conditions for {ds}/{env}.")

            for ci in range(10):
                if ci == 0 and not INCLUDE_CONDITION0:
                    done += 1
                    continue

                existing = completed_unit_record(str(row.run_id), env, ci)
                if existing is not None:
                    index_rows.append({
                        k:(json.dumps(v) if isinstance(v,list) else v)
                        for k,v in existing.items()
                    })
                    done += 1
                    continue

                if not can_start_unit():
                    partial = pd.DataFrame(index_rows)
                    if len(partial):
                        partial.to_csv(OUTDIR/"tables"/f"partial_index_{int(time.time())}.csv",index=False)
                    print(f"Paused safely after {done}/{total_units} units. Rerun the same script to resume.")
                    return

                stratum = int(pop.iloc[ci].stratum)
                rec = render_one_condition(
                    row=row,
                    env=env,
                    ci=ci,
                    stratum=stratum,
                    x=X[ci:ci+1],
                    data_meta=data["meta"],
                )
                index_rows.append({
                    k:(json.dumps(v) if isinstance(v,list) else v)
                    for k,v in rec.items()
                })
                done += 1

                if done % 25 == 0:
                    print(f"[{done}/{total_units}] rendered | remaining soft wall {soft_remaining()/3600:.2f} h", flush=True)

                gc.collect()
                plt.close("all")

    index = pd.DataFrame(index_rows)
    index = index.sort_values(["dataset","run_id","environment","condition_index"]).reset_index(drop=True)
    write_once_csv(index, OUTDIR/"tables"/"figure_index.csv")

    completion = {
        "version":V4_VERSION,
        "status":"complete",
        "model_environment_condition_units":int(len(index)),
        "expected_units":int(total_units if INCLUDE_CONDITION0 else total_units-len(manifest)*2),
        "fieldA_fieldB_figures":int(len(index)),
        "symmetry_figures":int(index["symmetry_figure_count"].sum()),
        "transform_mode":TRANSFORM_MODE,
        "reference":"x0=0 preserved",
        "source_files_modified":False,
    }
    write_once_json(completion,OUTDIR/"EXPERIMENT5_1_V4_VISUAL_ATLAS_COMPLETE.json")

    print("="*116)
    print("V4 VISUAL ATLAS COMPLETE")
    print(json.dumps(completion,indent=2))
    print("="*116)

if __name__ == "__main__":
    main()
