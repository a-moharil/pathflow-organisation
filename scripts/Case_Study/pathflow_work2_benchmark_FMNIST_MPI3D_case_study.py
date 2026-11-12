from __future__ import annotations

# ==================================================================================================
# WORK II — BENCHMARK CASE STUDY
# FMNIST CLASSIFICATION + MPI3D-TOY REGRESSION
# STRUCTURAL DISPOSITION AND CONDITIONAL REALIZATION IN DEEP MLPs / CNNs
# ==================================================================================================
# Self-contained Python file; may also be pasted into one Jupyter cell.
# Designed for one NVIDIA L4 GPU under Slurm with resumable run-parts.
#
# Canonical training factorial:
#   2 datasets x (3 MLP + 3 CNN) x 3 optimizers x 3 activations x 3 seeds = 324 trained networks.
#
# Environments are NOT another training axis. Every trained network is evaluated under:
#   E0 = clean
#   E1 = deterministic Gaussian sensor shift calibrated to 15 dB SNR
# so structural disposition is held fixed while conditional realization changes.
#
# Predeclared full-geometry panel (union design):
#   optimizer == Adam, all activations, seed 0
#   OR activation == ReLU, all optimizers, seed 0
# => 60 unique trained models x {structural, clean, shifted} = 180 population BCANw states.
#
# Audited BCANw conventions:
#   * metric on flow enrichment: lambda_e=(enrichment+eps)^(-1)
#   * undirected shortest-path support, directed quotient flow
#   * constant-NB plateaus represented by geometric midpoint, not first radius
#   * xi_q only if target is genuinely bracketed
#   * xi_q crossing interpolated in log(NB/N) versus log ell_B
#   * d_eff only on midpoint-compressed transition staircase
#   * saturated/non-supported derivative regions are never painted as dimension zero
#   * undefined derived quantities are tracked through dedicated validity/status tables
#   * nodewise d_i^F, alpha_i^F, Delta_i^F retained
#
# Validity semantics:
#   * path_summary.csv contains only states for which a normalized realization measure exists
#   * path_realization_validity.csv contains every structural/clean/shifted state and its support status
#   * population_geometry.csv is finite-only long-form geometry observations
#   * geometry_validity.csv records skipped/unsupported derived geometry without fabricating values
#
# The script writes standalone 600-dpi PNG + vector PDF figures and a multipage report.
# ==================================================================================================

import os, math, json, time, random, hashlib, shutil, urllib.request, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

try:
    import torchvision
    from torchvision.datasets import FashionMNIST
except Exception as e:
    raise RuntimeError("torchvision is required for Fashion-MNIST") from e

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare, ks_2samp

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================================================================================
# 0. CONFIGURATION
# ==================================================================================================

PROFILE = os.environ.get("PATHFLOW_BENCH_PROFILE", "l4_48h").strip().lower()
RESUME = os.environ.get("PATHFLOW_BENCH_RESUME", "1") != "0"
START_FRESH = os.environ.get("PATHFLOW_BENCH_START_FRESH", "0") == "1"
RUN_BCANW = os.environ.get("PATHFLOW_BENCH_RUN_BCANW", "1") != "0"
RUN_INPUT_GEOM = os.environ.get("PATHFLOW_BENCH_RUN_INPUT_GEOM", "1") != "0"
RUN_FIGURES = os.environ.get("PATHFLOW_BENCH_RUN_FIGURES", "1") != "0"

SEED = 730031
MODEL_SEED_BASE = 271828
ORDER_SEED_BASE = 141421
NOISE_SEED_BASE = 161803
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTDIR = Path(os.environ.get(
    "/content/drive/MyDrive/PATHFLOW_BENCH_OUTDIR",
    "/content/drive/MyDrive/pathflow_work2_benchmark_FMNIST_MPI3D_case_study_results"
))
for sub in ["tables", "figures", "checkpoints", "run_parts", "data"]:
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

DATASETS = ["mpi3d"] #"fmnist", 
ACTIVATIONS = ["relu", "tanh", "silu"]
OPTIMIZERS = ["sgdm", "rmsprop", "adam"]
SEEDS = [0, 1, 2]
ENVIRONMENTS = ["clean", "shifted"]

IMAGE_RESOLUTION = 16
N_TRAIN = 20000
N_VAL = 5000
N_TEST = 5000
BATCH_SIZE = 256
MAX_EPOCH = 160
WEIGHT_DECAY = 0.0

LEARNING_RATES = {
    "sgdm": 1.0e-2,
    "rmsprop": 1.0e-3,
    "adam": 1.0e-3,
}

# Conditional/path evaluation
ENDPOINT_INPUTS = 160
CONDITIONAL_BATCH = 16
SHIFT_SNR_DB = 15.0

# Rich-population preflight. This is deliberately cheap relative to the 324-network campaign.
RUN_PREFLIGHT = os.environ.get("PATHFLOW_BENCH_RUN_PREFLIGHT", "1") != "0"
PREFLIGHT_ONLY = os.environ.get("PATHFLOW_BENCH_PREFLIGHT_ONLY", "0") == "1"
# Slurm policy: only genuine numerical/theoretical pathologies abort the allocation.
# ReLU richness shortfalls are scientific warnings unless the user explicitly requests hard gating.
PREFLIGHT_STRICT = os.environ.get("PATHFLOW_BENCH_PREFLIGHT_STRICT", "0") != "0"
PREFLIGHT_ABORT_ON_RELU_WARNING = os.environ.get("PATHFLOW_BENCH_PREFLIGHT_ABORT_ON_RELU_WARNING", "0") == "1"
PREFLIGHT_EPOCHS = int(os.environ.get("PATHFLOW_BENCH_PREFLIGHT_EPOCHS", "5"))
PREFLIGHT_N_TRAIN = int(os.environ.get("PATHFLOW_BENCH_PREFLIGHT_NTRAIN", "2000"))
PREFLIGHT_SEED = int(os.environ.get("PATHFLOW_BENCH_PREFLIGHT_SEED", "0"))
PREFLIGHT_RELU_MIN_VALID_FRACTION = float(os.environ.get("PATHFLOW_BENCH_PREFLIGHT_RELU_MIN_VALID", "0.99"))
PREFLIGHT_RELU_MIN_FINAL_ACTIVE_Q01 = float(os.environ.get("PATHFLOW_BENCH_PREFLIGHT_RELU_MIN_FINAL_ACTIVE_Q01", "0.03"))
PREFLIGHT_RUN_BCANW_SENTINEL = os.environ.get("PATHFLOW_BENCH_PREFLIGHT_BCANW", "1") != "0"
PREFLIGHT_BCANW_SCALES = int(os.environ.get("PATHFLOW_BENCH_PREFLIGHT_BCANW_SCALES", "24"))
PREFLIGHT_BCANW_ARCH_IDS = ["M2_sustained", "C2_bottleneck"]
PREFLIGHT_BCANW_OPTIMIZER = "adam"
PREFLIGHT_BCANW_ACTIVATION = "relu"

# BCANw publication panel
BCANW_SCALES = 64
BCANW_MAX_NODES = 5200
LOCAL_DIMENSION_NODES = 128
LOCAL_RADII_MAX = 40
GEOM_SEEDS = [0]
GEOM_ACTIVATION_PANEL_OPT = "adam"
GEOM_OPTIMIZER_PANEL_ACT = "relu"

# Input-resolved nested geometry sentinel
INPUT_GEOM_ARCH_IDS = ["M2_sustained", "C2_bottleneck"]
INPUT_GEOM_OPTIMIZER = "adam"
INPUT_GEOM_SEEDS = [0]
INPUT_GEOM_INPUTS = 3
INPUT_GEOM_SCALES = 40

# BCANw / response constants
EPS = 1e-14
PATH_LOG_FLOOR = 1e-300
METRIC_POWER = 1.0
LOCAL_SLOPE_WINDOW = 7
LOCAL_SLOPE_R2_MIN = 0.90
RENORM_TARGET_BOX_FRACTIONS = np.array([0.50, 0.25, 0.125, 0.0625])
COARSE_RENYI_Q = [0.5, 1.0, 2.0, 3.0]

SCALING_R2_MIN = 0.97
SCALING_MIN_DIM = 0.25
SCALING_MIN_POINTS = 6
SCALING_MIN_SPAN_DECADES = 0.50
SCALING_MIN_NB_RATIO = 5.0
SCALING_MAX_SLOPE_CV = 0.25
SCALING_MAX_DELTA_BIC = 2.0

LOCAL_COMMON_MIN_POINTS = 7
LOCAL_COMMON_MIN_SPAN_DECADES = 0.50
LOCAL_COMMON_SUPPORT_GROWTH = 4.0
LOCAL_COMMON_MASS_GROWTH = 3.0
LOCAL_COMMON_R2_MIN = 0.95
LOCAL_COMMON_SATURATION = 0.95

# Architectures: deep enough to be structurally nontrivial but small enough for path-flow endpoint analysis.
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

# Allow scalar environment overrides without source edits.
MAX_EPOCH = int(os.environ.get("PATHFLOW_BENCH_EPOCHS", MAX_EPOCH))
N_TRAIN = int(os.environ.get("PATHFLOW_BENCH_NTRAIN", N_TRAIN))
N_VAL = int(os.environ.get("PATHFLOW_BENCH_NVAL", N_VAL))
N_TEST = int(os.environ.get("PATHFLOW_BENCH_NTEST", N_TEST))
ENDPOINT_INPUTS = int(os.environ.get("PATHFLOW_BENCH_ENDPOINT_INPUTS", ENDPOINT_INPUTS))
BCANW_SCALES = int(os.environ.get("PATHFLOW_BENCH_BCANW_SCALES", BCANW_SCALES))
BCANW_MAX_NODES = int(os.environ.get("PATHFLOW_BENCH_BCANW_MAX_NODES", BCANW_MAX_NODES))

if PROFILE == "pilot":
    DATASETS = ["fmnist"]
    ACTV0 = ACTIVATIONS
    OPT0 = OPTIMIZERS
    ACTIVATIONS = ["relu", "tanh"]
    OPTIMIZERS = ["adam"]
    SEEDS = [0]
    N_TRAIN, N_VAL, N_TEST = 1024, 256, 256
    MAX_EPOCH = 4
    BATCH_SIZE = 128
    ENDPOINT_INPUTS = 16
    BCANW_SCALES = 16
    BCANW_MAX_NODES = 3000
    LOCAL_DIMENSION_NODES = 48
    GEOM_SEEDS = [0]
    INPUT_GEOM_INPUTS = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision("high")

if START_FRESH:
    for sub in ["tables", "figures", "checkpoints", "run_parts"]:
        for p in (OUTDIR / sub).glob("*"):
            if p.is_file():
                try: p.unlink()
                except Exception: pass

# ==================================================================================================
# 1. GENERIC HELPERS
# ==================================================================================================

def banner(s: str):
    print("\n" + "=" * 112)
    print(s)
    print("=" * 112)


def human_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0: return "unknown"
    s = int(round(seconds)); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


class ETAReporter:
    def __init__(self, stage: str, total: int, window: int = 8):
        self.stage = stage; self.total = max(int(total), 1); self.window = max(int(window), 3)
        self.start = time.perf_counter(); self.durations: List[float] = []
    def update(self, duration: float, completed: int):
        self.durations.append(float(duration)); med = float(np.median(self.durations[-self.window:]))
        remain = max(self.total - completed, 0)
        tqdm.write(f"[ETA:{self.stage}] {completed}/{self.total} | current {human_time(duration)} | "
                   f"median {human_time(med)}/item | remaining ~{human_time(med*remain)}")
    def finish(self):
        tqdm.write(f"[ETA:{self.stage}] finished in {human_time(time.perf_counter()-self.start)}")


def stable_hash_dict(d: Dict[str, Any], n: int = 12) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:n]


def atomic_csv(df: pd.DataFrame, path: Path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False); tmp.replace(path)


def atomic_json(obj: Any, path: Path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8"); tmp.replace(path)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1: return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()


def r2_score_np(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    den = float(np.sum((y-y.mean())**2))
    return float(1.0 - np.sum((y-pred)**2)/den) if den > EPS else np.nan


def safe_log(x):
    try:
        x = float(x)
        return float(math.log(max(x, EPS))) if np.isfinite(x) else np.nan
    except Exception: return np.nan


def model_seed(arch: str, dataset: str, seed_index: int) -> int:
    return int(MODEL_SEED_BASE + 1000003*seed_index + int(stable_hash_dict({"a":arch,"d":dataset}, 8),16)%999983)


def deterministic_batches(n: int, epoch: int, seed_index: int, batch_size: int):
    rng = np.random.default_rng(ORDER_SEED_BASE + 104729*seed_index + int(epoch))
    ids = rng.permutation(int(n))
    for lo in range(0, len(ids), batch_size):
        yield ids[lo:min(lo+batch_size, len(ids))]


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float); m = len(p); out = np.ones(m, float)
    order = np.argsort(p); running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank)*p[idx]); running = max(running, val); out[idx] = running
    return out


def bootstrap_median_ci(d: np.ndarray, seed: int = 0, B: int = 2000) -> Tuple[float,float]:
    d = np.asarray(d,float); d = d[np.isfinite(d)]
    if len(d) < 2: return np.nan, np.nan
    rng = np.random.default_rng(SEED + seed); vals = np.empty(B,float)
    for b in range(B): vals[b] = np.median(rng.choice(d, size=len(d), replace=True))
    return float(np.quantile(vals,.025)), float(np.quantile(vals,.975))


def hodges_lehmann_onesample(d: np.ndarray) -> float:
    d=np.asarray(d,float); d=d[np.isfinite(d)]
    if len(d)==0:return np.nan
    vals=[0.5*(d[i]+d[j]) for i in range(len(d)) for j in range(i,len(d))]
    return float(np.median(vals))


def paired_rank_biserial(d: np.ndarray) -> float:
    d=np.asarray(d,float); d=d[np.isfinite(d)&(d!=0)]
    if len(d)==0:return 0.0
    ranks=pd.Series(np.abs(d)).rank(method="average").to_numpy(float)
    wp=float(ranks[d>0].sum()); wm=float(ranks[d<0].sum()); den=wp+wm
    return float((wp-wm)/den) if den>0 else 0.0

# ==================================================================================================
# 2. DATASETS
# ==================================================================================================

MPI3D_URL = os.environ.get(
    "MPI3D_URL",
    "https://huggingface.co/datasets/waleedgondal/mpi3d/resolve/main/mpi3d_toy.npz"
)


def _resize_images_np(x: np.ndarray, size: int) -> np.ndarray:
    """Resize grayscale NHW or RGB NHWC arrays to NCHW without changing study logic."""
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 3:
        # FMNIST: [N,H,W] -> [N,1,H,W]
        t = t[:,None,:,:]
    elif t.ndim == 4:
        # MPI3D official storage: [N,H,W,3] -> [N,3,H,W]
        if t.shape[-1] in (1,3,4):
            t = t.permute(0,3,1,2)
        elif t.shape[1] not in (1,3,4):
            raise ValueError(f"Unsupported 4D image layout: {tuple(t.shape)}")
    else:
        raise ValueError(f"Unsupported image array shape: {tuple(t.shape)}")
    with torch.no_grad():
        y = F.interpolate(t, size=(size,size), mode="area")
    return y.numpy().astype(np.float32)


def _split_indices_stratified(labels: np.ndarray, ntr: int, nv: int, nt: int, seed: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    labels=np.asarray(labels); rng=np.random.default_rng(seed); classes=np.unique(labels)
    tr=[]; va=[]; te=[]
    total=ntr+nv+nt
    for c in classes:
        ids=np.where(labels==c)[0]; rng.shuffle(ids)
        take=max(1,int(round(total*len(ids)/len(labels))))
        ids=ids[:min(take,len(ids))]
        a=max(1,int(round(ntr*len(ids)/total))); b=max(1,int(round(nv*len(ids)/total)))
        tr.extend(ids[:a]); va.extend(ids[a:a+b]); te.extend(ids[a+b:])
    def fill(arr, target, pool):
        arr=list(dict.fromkeys(arr)); need=target-len(arr)
        if need>0:
            remain=np.setdiff1d(pool,np.asarray(arr,int),assume_unique=False); rng.shuffle(remain); arr.extend(remain[:need].tolist())
        return np.asarray(arr[:target],int)
    pool=np.arange(len(labels))
    tr=fill(tr,ntr,pool); used=tr
    va_pool=np.setdiff1d(pool,used); va=fill(va,nv,va_pool); used=np.concatenate([tr,va])
    te_pool=np.setdiff1d(pool,used); te=fill(te,nt,te_pool)
    return tr,va,te



def balanced_indices_from_strata(strata: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministically select a near-exactly balanced population and interleave its strata.

    For the canonical endpoint size 160 this is exactly 16/class for Fashion-MNIST
    and 4/horizontal-position level for MPI3D when endpoint_inputs=160. Interleaving prevents conditional mini-batches from
    inheriting class/target blocks from storage order.
    """
    strata=np.asarray(strata)
    classes=np.asarray(sorted(np.unique(strata).tolist()))
    if len(classes)==0:
        return np.array([],dtype=int)
    n=min(int(n),len(strata)); rng=np.random.default_rng(int(seed))
    base=n//len(classes); rem=n%len(classes)
    extra=set(rng.permutation(classes)[:rem].tolist())
    chosen={}
    leftovers=[]
    for c in classes:
        ids=np.where(strata==c)[0].copy(); rng.shuffle(ids)
        take=base+(1 if c in extra else 0)
        chosen[c]=ids[:min(take,len(ids))].tolist()
        leftovers.extend(ids[min(take,len(ids)):].tolist())
    # If a rare stratum could not supply its quota, fill globally without duplication.
    selected=sum((v for v in chosen.values()),[])
    if len(selected)<n:
        rng.shuffle(leftovers); selected.extend(leftovers[:n-len(selected)])
    # Round-robin interleave the balanced selections.
    queues={c:list(chosen[c]) for c in classes}; out=[]
    while len(out)<min(n,sum(len(v) for v in queues.values())):
        moved=False
        for c in classes:
            if queues[c] and len(out)<n:
                out.append(queues[c].pop(0)); moved=True
        if not moved: break
    if len(out)<n:
        used=set(out); tail=[i for i in selected if i not in used]; out.extend(tail[:n-len(out)])
    return np.asarray(out[:n],dtype=int)


def attach_balanced_populations(data: Dict[str,Any], dataset: str) -> Dict[str,Any]:
    """Attach fixed matched endpoint and small-training populations to one dataset."""
    test_strata=np.asarray(data["test_strata"])
    train_strata=np.asarray(data["train_strata"])
    endpoint_ids=balanced_indices_from_strata(test_strata,ENDPOINT_INPUTS,SEED+1009+(0 if dataset=="fmnist" else 1))
    pretrain_ids=balanced_indices_from_strata(train_strata,min(PREFLIGHT_N_TRAIN,len(train_strata)),SEED+2003+(0 if dataset=="fmnist" else 1))
    data["endpoint_indices"]=endpoint_ids
    data["preflight_train_indices"]=pretrain_ids
    # Explicit balance table, useful for manuscript/report auditing.
    vals,counts=np.unique(test_strata[endpoint_ids],return_counts=True)
    data["endpoint_balance"]={str(v):int(c) for v,c in zip(vals,counts)}
    return data

def load_fmnist() -> Dict[str,Any]:
    root=OUTDIR/"data"/"fmnist"
    ds_tr=FashionMNIST(root=str(root),train=True,download=True)
    ds_te=FashionMNIST(root=str(root),train=False,download=True)
    Xtr0=ds_tr.data.numpy().astype(np.float32)/255.0; ytr0=ds_tr.targets.numpy().astype(np.int64)
    Xte0=ds_te.data.numpy().astype(np.float32)/255.0; yte0=ds_te.targets.numpy().astype(np.int64)
    Xtr0=_resize_images_np(Xtr0,IMAGE_RESOLUTION); Xte0=_resize_images_np(Xte0,IMAGE_RESOLUTION)
    tr,va,_=_split_indices_stratified(ytr0,N_TRAIN,N_VAL,max(1,min(1000,len(ytr0)-N_TRAIN-N_VAL)),SEED+11)
    # Preserve the official Fashion-MNIST test pool; take a stratified subset only for compute control.
    _,_,te=_split_indices_stratified(yte0,max(1,min(1000,len(yte0)-N_TEST-1)),max(1,min(500,len(yte0)-N_TEST-1)),N_TEST,SEED+17)
    mu=float(Xtr0[tr].mean()); sd=float(Xtr0[tr].std()+1e-8)
    Xtr=((Xtr0[tr]-mu)/sd).astype(np.float32); Xv=((Xtr0[va]-mu)/sd).astype(np.float32); Xt=((Xte0[te]-mu)/sd).astype(np.float32)
    return {
        "task":"classification","output_dim":10,
        "train":(torch.from_numpy(Xtr),torch.from_numpy(ytr0[tr])),
        "val":(torch.from_numpy(Xv),torch.from_numpy(ytr0[va])),
        "test":(torch.from_numpy(Xt),torch.from_numpy(yte0[te])),
        "train_strata":ytr0[tr].copy(), "val_strata":ytr0[va].copy(), "test_strata":yte0[te].copy(),
        "meta":{"dataset":"fmnist","input_mean":mu,"input_sd":sd,"resolution":IMAGE_RESOLUTION,"split":"official train pool for train/val; official test pool for test"}
    }


def _ensure_mpi3d_npz() -> Path:
    env=os.environ.get("MPI3D_NPZ","").strip()
    if env:
        p=Path(env)
        if not p.exists(): raise FileNotFoundError(f"MPI3D_NPZ does not exist: {p}")
        return p
    p=OUTDIR/"data"/"mpi3d_toy.npz"
    if p.exists(): return p
    print(f"Downloading MPI3D-Toy to {p} ...")
    try:
        urllib.request.urlretrieve(MPI3D_URL, p)
    except Exception as e:
        raise RuntimeError(
            "Could not download MPI3D-Toy. On HPC without outbound internet, download mpi3d_toy.npz once and set "
            "MPI3D_NPZ=/path/to/mpi3d_toy.npz"
        ) from e
    return p


def load_mpi3d() -> Dict[str,Any]:
    """MPI3D-Toy scalar spatial-regression benchmark.

    The original benchmark's scalar-regression semantics are preserved exactly:
        target = horizontal_axis, 0..39

    MPI3D-Toy stores the Cartesian product in C-order with factor sizes
        [6,6,2,3,3,40,40]
    corresponding to
        [object_color, object_shape, object_size, camera_height,
         background_color, horizontal_axis, vertical_axis].

    Thus horizontal_axis = (flat_index // 40) % 40.
    """
    p=_ensure_mpi3d_npz()

    factor_sizes=np.asarray([6,6,2,3,3,40,40],dtype=np.int64)
    expected_n=int(np.prod(factor_sizes))

    # Labels are determined by the declared Cartesian-product ordering and do not
    # require materializing a latent-factor table.
    all_ids=np.arange(expected_n,dtype=np.int64)
    xpos=((all_ids // int(factor_sizes[-1])) % int(factor_sizes[-2])).astype(np.int64)

    # Preserve support of all 40 horizontal-position levels in train/val/test.
    tr,va,te=_split_indices_stratified(xpos,N_TRAIN,N_VAL,N_TEST,SEED+29)
    ids=np.concatenate([tr,va,te])

    # Official NPZ key is "images", shape [1,036,800,64,64,3], uint8 RGB.
    # Only the predeclared train/val/test subset is retained after loading.
    with np.load(p,allow_pickle=False) as z:
        if "images" not in z.files:
            raise KeyError(f"MPI3D NPZ has keys {z.files}; expected an 'images' array.")
        images=z["images"]
        if images.ndim!=4 or images.shape[-1]!=3:
            raise ValueError(f"Unexpected MPI3D image shape: {images.shape}; expected [N,64,64,3].")
        if images.shape[0]!=expected_n:
            raise ValueError(
                f"Unexpected MPI3D-Toy population size {images.shape[0]}; expected {expected_n} "
                "for factor sizes [6,6,2,3,3,40,40]."
            )
        imgs=images[ids].astype(np.float32)/255.0

    imgs=_resize_images_np(imgs,IMAGE_RESOLUTION)

    # Re-index the selected local array exactly as in the original MPI3D arm.
    ntr=len(tr);nv=len(va)
    Xtr=imgs[:ntr];Xv=imgs[ntr:ntr+nv];Xt=imgs[ntr+nv:]
    yall=xpos[ids].astype(np.float32)
    ytr=yall[:ntr];yv=yall[ntr:ntr+nv];yt=yall[ntr+nv:]

    # Preserve the original scalar normalization convention.
    mu=float(Xtr.mean());sd=float(Xtr.std()+1e-8)
    Xtr=((Xtr-mu)/sd).astype(np.float32)
    Xv=((Xv-mu)/sd).astype(np.float32)
    Xt=((Xt-mu)/sd).astype(np.float32)

    ym=float(ytr.mean());ys=float(ytr.std()+1e-8)
    ytr=((ytr-ym)/ys).astype(np.float32)[:,None]
    yv=((yv-ym)/ys).astype(np.float32)[:,None]
    yt=((yt-ym)/ys).astype(np.float32)[:,None]

    return {
        "task":"regression","output_dim":1,
        "train":(torch.from_numpy(Xtr),torch.from_numpy(ytr)),
        "val":(torch.from_numpy(Xv),torch.from_numpy(yv)),
        "test":(torch.from_numpy(Xt),torch.from_numpy(yt)),
        "train_strata":xpos[tr].copy(),
        "val_strata":xpos[va].copy(),
        "test_strata":xpos[te].copy(),
        "meta":{
            "dataset":"mpi3d",
            "variant":"mpi3d_toy",
            "input_channels":3,
            "input_mean":mu,
            "input_sd":sd,
            "target_mean":ym,
            "target_sd":ys,
            "target":"horizontal_axis",
            "target_levels":40,
            "resolution":IMAGE_RESOLUTION,
            "npz":str(p),
            "factor_order":[
                "object_color","object_shape","object_size","camera_height",
                "background_color","horizontal_axis","vertical_axis"
            ],
            "factor_sizes":[6,6,2,3,3,40,40],
        }
    }


def load_datasets() -> Dict[str,Dict[str,Any]]:
    out={}
    if "fmnist" in DATASETS: out["fmnist"]=attach_balanced_populations(load_fmnist(),"fmnist")
    if "mpi3d" in DATASETS: out["mpi3d"]=attach_balanced_populations(load_mpi3d(),"mpi3d")
    rows=[]
    for ds,d in out.items():
        for level,count in d.get("endpoint_balance",{}).items():
            rows.append({"dataset":ds,"stratum":level,"n_endpoint":count,"endpoint_population_size":len(d.get("endpoint_indices",[]))})
    atomic_csv(pd.DataFrame(rows),OUTDIR/"tables"/"balanced_endpoint_population.csv")
    return out


def shifted_environment(X: torch.Tensor, dataset: str, dataset_meta: Dict[str,Any]) -> torch.Tensor:
    """Deterministic additive Gaussian sensor shift in raw image space, then reapply clean-train normalization.

    This implements the declared benchmark intervention literally: the learned parameters, labels/targets,
    reference x0, and normalization statistics remain fixed; only the raw sensor observation changes.
    """
    x=X.float(); mu=float(dataset_meta["input_mean"]); sd=float(dataset_meta["input_sd"])
    raw=x*sd+mu
    signal=float(torch.mean(raw*raw).item()); noise_var=signal/(10.0**(SHIFT_SNR_DB/10.0))
    g=torch.Generator(device="cpu").manual_seed(NOISE_SEED_BASE + int(stable_hash_dict({"d":dataset},8),16)%1000003)
    noise=torch.randn(raw.shape,generator=g,dtype=raw.dtype)*math.sqrt(max(noise_var,EPS))
    shifted_raw=raw+noise
    return (shifted_raw-mu)/max(sd,EPS)

# ==================================================================================================
# 3. MODELS / ACTIVATIONS / OPTIMIZERS
# ==================================================================================================

class ActivationModule(nn.Module):
    def __init__(self,name:str): super().__init__(); self.name=name
    def forward(self,x):
        if self.name=="relu": return F.relu(x)
        if self.name=="tanh": return torch.tanh(x)
        if self.name=="silu": return F.silu(x)
        raise ValueError(self.name)


@torch.no_grad()
def finite_secant_torch64(name:str,z:torch.Tensor,z0:torch.Tensor,tol:float=1e-12)->torch.Tensor:
    z=z.to(torch.float64); z0=z0.to(torch.float64); dz=z-z0
    if name=="relu": f=lambda q:torch.clamp_min(q,0.0)
    elif name=="tanh": f=torch.tanh
    elif name=="silu": f=F.silu
    else: raise ValueError(name)
    out=torch.empty_like(dz); mask=torch.abs(dz)>tol
    if mask.any(): out[mask]=(f(z[mask])-f(z0[mask]))/dz[mask]
    if (~mask).any():
        q=.5*(z[~mask]+z0[~mask])
        if name=="relu": der=(q>0).to(q.dtype)
        elif name=="tanh":
            t=torch.tanh(q); der=1-t*t
        else:
            s=torch.sigmoid(q); der=s+q*s*(1-s)
        out[~mask]=der
    return out


class DeepMLP(nn.Module):
    def __init__(self,hidden:Sequence[int],activation:str,output_dim:int,seed:int,input_channels:int=1):
        super().__init__(); self.activation_name=activation; self.act=ActivationModule(activation); self.input_channels=int(input_channels)
        dims=[self.input_channels*IMAGE_RESOLUTION*IMAGE_RESOLUTION]+list(hidden)+[output_dim]
        torch.manual_seed(seed)
        self.layers=nn.ModuleList([nn.Linear(dims[i],dims[i+1],bias=False) for i in range(len(dims)-1)])
        self.reset_common(seed)
    def reset_common(self,seed:int):
        g=torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for l in self.layers:
                W=torch.randn(l.weight.shape,generator=g,dtype=l.weight.dtype)/math.sqrt(max(l.in_features,1)); l.weight.copy_(W)
    def forward(self,x):
        h=x.reshape(x.shape[0],-1)
        for l in self.layers[:-1]: h=self.act(l(h))
        return self.layers[-1](h)
    def forward_with_states(self,x):
        h=x.reshape(x.shape[0],-1); zs=[]; hs=[h]
        for l in self.layers[:-1]:
            z=l(h); zs.append(z); h=self.act(z); hs.append(h)
        out=self.layers[-1](h); hs.append(out); return out,zs,hs
    def parameter_count(self): return int(sum(p.numel() for p in self.parameters()))


class DeepCNN(nn.Module):
    def __init__(self,spec:Dict[str,Any],activation:str,output_dim:int,seed:int,input_channels:int=1):
        super().__init__(); self.activation_name=activation; self.act=ActivationModule(activation); self.input_channels=int(input_channels)
        hidden=list(spec["hidden_channels"]); kernels=list(spec["kernels"]); chans=[self.input_channels]+hidden+[output_dim]
        if len(kernels)!=len(chans)-1: raise ValueError("CNN kernels mismatch")
        self.channels=chans; self.kernels=kernels
        torch.manual_seed(seed)
        self.convs=nn.ModuleList([nn.Conv2d(chans[i],chans[i+1],kernel_size=kernels[i],stride=1,padding=0,bias=False)
                                  for i in range(len(kernels))])
        self.reset_common(seed)
        if self.output_resolution()<1: raise ValueError("CNN output resolution <1")
    def reset_common(self,seed:int):
        g=torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for c in self.convs:
                W=torch.randn(c.weight.shape,generator=g,dtype=c.weight.dtype)/math.sqrt(max(c.in_channels*c.kernel_size[0]*c.kernel_size[1],1)); c.weight.copy_(W)
    def spatial_shapes(self):
        h=w=IMAGE_RESOLUTION; out=[(h,w)]
        for k in self.kernels: h=h-k+1; w=w-k+1; out.append((h,w))
        return out
    def output_resolution(self): return int(self.spatial_shapes()[-1][0])
    def forward(self,x):
        h=x
        for i,c in enumerate(self.convs):
            z=c(h); h=self.act(z) if i<len(self.convs)-1 else z
        return h.mean(dim=(-1,-2))
    def forward_with_states(self,x):
        h=x; zs=[]; hs=[x]
        for i,c in enumerate(self.convs):
            z=c(h)
            if i<len(self.convs)-1:
                zs.append(z); h=self.act(z); hs.append(h)
            else: h=z; hs.append(h)
        return h.mean(dim=(-1,-2)),zs,hs
    def parameter_count(self): return int(sum(p.numel() for p in self.parameters()))


def build_model(arch_id:str,activation:str,output_dim:int,dataset:str,seed_index:int)->nn.Module:
    s=model_seed(arch_id,dataset,seed_index)
    input_channels=1 if dataset=="fmnist" else 3
    if arch_id in MLP_ARCHS: return DeepMLP(MLP_ARCHS[arch_id],activation,output_dim,s,input_channels=input_channels).to(DEVICE)
    return DeepCNN(CNN_ARCHS[arch_id],activation,output_dim,s,input_channels=input_channels).to(DEVICE)


@torch.no_grad()
def calibrate_initial_signal(model:nn.Module,Xcal:torch.Tensor,target_var:float=1.0):
    model.eval(); h=Xcal[:min(len(Xcal),256)].to(DEVICE); scales=[]
    if isinstance(model,DeepMLP):
        h=h.reshape(h.shape[0],-1)
        for l in model.layers[:-1]:
            z=l(h); sd=float(torch.std(z,unbiased=False).item()); sc=float(np.clip(math.sqrt(target_var)/max(sd,1e-6),.15,6.0)); l.weight.mul_(sc); h=model.act(l(h)); scales.append(sc)
        scales.append(1.0)
    else:
        for i,c in enumerate(model.convs):
            z=c(h)
            if i<len(model.convs)-1:
                sd=float(torch.std(z,unbiased=False).item()); sc=float(np.clip(math.sqrt(target_var)/max(sd,1e-6),.15,6.0)); c.weight.mul_(sc); h=model.act(c(h)); scales.append(sc)
            else: scales.append(1.0)
    return scales


def make_optimizer(model:nn.Module,name:str):
    lr=LEARNING_RATES[name]
    if name=="sgdm": return torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=WEIGHT_DECAY)
    if name=="rmsprop": return torch.optim.RMSprop(model.parameters(),lr=lr,alpha=.99,weight_decay=WEIGHT_DECAY)
    if name=="adam": return torch.optim.Adam(model.parameters(),lr=lr,weight_decay=WEIGHT_DECAY)
    raise ValueError(name)

# ==================================================================================================
# 4. TRAINING / PERFORMANCE
# ==================================================================================================

@torch.no_grad()
def evaluate_performance(model:nn.Module,X:torch.Tensor,y:torch.Tensor,task:str,chunk:int=1024)->Dict[str,float]:
    model.eval(); total_loss=0.0; total=0; correct=0
    for lo in range(0,len(X),chunk):
        xb=X[lo:lo+chunk].to(DEVICE); yb=y[lo:lo+chunk].to(DEVICE); pred=model(xb)
        if task=="classification":
            loss=F.cross_entropy(pred,yb,reduction="sum"); correct+=int((pred.argmax(1)==yb).sum().item()); total+=len(yb)
        else:
            loss=torch.sum((pred-yb)**2); total+=int(yb.numel())
        total_loss+=float(loss.item())
    out={"loss":total_loss/max(total,1)}
    if task=="classification": out["accuracy"]=correct/max(total,1)
    return out


def loss_batch(pred,y,task):
    return F.cross_entropy(pred,y) if task=="classification" else F.mse_loss(pred,y)

# ==================================================================================================
# 5. PATH STATES — MLP
# ==================================================================================================

def _normalize_rows(t:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
    s=t.sum(dim=tuple(range(1,t.ndim))); valid=torch.isfinite(s)&(s>0); safe=torch.where(valid,s,torch.ones_like(s))
    out=t/safe.reshape((-1,)+((1,)*(t.ndim-1))); out=torch.where(valid.reshape((-1,)+((1,)*(t.ndim-1))),out,torch.zeros_like(out)); return out,s


@torch.no_grad()
def mlp_response_sq(model:DeepMLP,X:torch.Tensor,X0:torch.Tensor)->List[torch.Tensor]:
    _,z,_=model.forward_with_states(X.to(DEVICE)); _,z0,_=model.forward_with_states(X0.to(DEVICE))
    return [torch.square(finite_secant_torch64(model.activation_name,a,b)) for a,b in zip(z,z0)]


@torch.no_grad()
def mlp_log_partition_q(model:DeepMLP,source_sq:torch.Tensor,response_sq:List[torch.Tensor],target_idx:torch.Tensor,q:float)->torch.Tensor:
    src=torch.pow(torch.clamp(source_sq.to(DEVICE,dtype=torch.float64),min=0.0),float(q));a,s0=_normalize_rows(src)
    valid=torch.isfinite(s0)&(s0>0);logscale=torch.where(valid,torch.log(torch.clamp(s0,min=EPS)),torch.full_like(s0,-torch.inf))
    for i,l in enumerate(model.layers[:-1]):
        w=torch.pow(torch.square(l.weight.to(torch.float64)),float(q));a=a@w.T
        a=a*torch.pow(torch.clamp(response_sq[i].to(DEVICE,dtype=torch.float64),min=0.0),float(q));a,sl=_normalize_rows(a)
        good=valid&torch.isfinite(sl)&(sl>0);logscale=torch.where(good,logscale+torch.log(torch.clamp(sl,min=EPS)),torch.full_like(logscale,-torch.inf));valid=good
    Wout=torch.pow(torch.square(model.layers[-1].weight.to(torch.float64)),float(q));t=target_idx.to(DEVICE).long();selected=Wout[t,:];z=torch.sum(a*selected,dim=1);good=valid&torch.isfinite(z)&(z>0)
    return torch.where(good,logscale+torch.log(torch.clamp(z,min=EPS)),torch.full_like(logscale,-torch.inf))


@torch.no_grad()
def mlp_factor_state(model:DeepMLP,source_sq:torch.Tensor,response_sq:List[torch.Tensor],target_idx:torch.Tensor)->Dict[str,Any]:
    B=source_sq.shape[0]; src=source_sq.to(DEVICE,dtype=torch.float64); resp=[r.to(DEVICE,dtype=torch.float64) for r in response_sq]
    logZ1=mlp_log_partition_q(model,src,resp,target_idx,1.0);logZ2=mlp_log_partition_q(model,src,resp,target_idx,2.0);valid=torch.isfinite(logZ1)
    alpha=[]; a,_=_normalize_rows(src); alpha.append(a)
    for i,l in enumerate(model.layers[:-1]):
        a=a@torch.square(l.weight.to(torch.float64)).T; a=a*resp[i]; a,_=_normalize_rows(a); alpha.append(a)
    Wout2=torch.square(model.layers[-1].weight.to(torch.float64)); t=target_idx.to(DEVICE).long(); selected=Wout2[t,:]
    beta=[None]*len(alpha); b=selected.clone(); b,_=_normalize_rows(b); beta[-1]=b
    for i in range(len(model.layers)-2,-1,-1):
        target=beta[i+1]*resp[i]; b=target@torch.square(model.layers[i].weight.to(torch.float64)); b,_=_normalize_rows(b); beta[i]=b
    node=[]
    for i in range(len(alpha)):
        v=alpha[i]*beta[i]; v,_=_normalize_rows(v); node.append(v)
    outnodes=torch.zeros((B,model.layers[-1].out_features),dtype=torch.float64,device=DEVICE);outnodes[torch.arange(B,device=DEVICE),t]=1.0;node.append(outnodes)
    edge=[];elog=torch.zeros(B,dtype=torch.float64,device=DEVICE)
    for i,l in enumerate(model.layers[:-1]):
        E=(beta[i+1]*resp[i])[:,:,None]*torch.square(l.weight.to(torch.float64))[None,:,:]*alpha[i][:,None,:]
        den=E.sum((1,2));E=torch.where((den>0)[:,None,None],E/torch.clamp(den[:,None,None],min=EPS),torch.zeros_like(E));edge.append(E)
        logfac=torch.log(torch.clamp(torch.square(l.weight.to(torch.float64)),min=PATH_LOG_FLOOR))[None,:,:]+torch.log(torch.clamp(resp[i],min=PATH_LOG_FLOOR))[:,:,None]
        elog+=torch.sum(torch.where(E>0,E*logfac,torch.zeros_like(E)),dim=(1,2))
    Eout=selected*alpha[-1];den=Eout.sum(1);Eout=torch.where((den>0)[:,None],Eout/torch.clamp(den[:,None],min=EPS),torch.zeros_like(Eout))
    EF=torch.zeros((B,model.layers[-1].out_features,model.layers[-1].in_features),dtype=torch.float64,device=DEVICE);EF[torch.arange(B,device=DEVICE),t,:]=Eout;edge.append(EF)
    logwout=torch.log(torch.clamp(Wout2[t,:],min=PATH_LOG_FLOOR));elog+=torch.sum(torch.where(Eout>0,Eout*logwout,torch.zeros_like(Eout)),dim=1)
    p0=node[0];elogsrc=torch.sum(torch.where(p0>0,p0*torch.log(torch.clamp(src,min=PATH_LOG_FLOOR)),torch.zeros_like(p0)),dim=1);elogq=elogsrc+elog
    H=logZ1-elogq;logN=math.log(model.layers[0].in_features)+sum(math.log(l.out_features) for l in model.layers[:-1]);Sdef=logN-H;eta1=torch.exp(torch.clamp(H-logN,min=-745,max=0));eta2=torch.exp(torch.clamp(2*logZ1-logZ2-logN,min=-745,max=0));Sdef_norm=Sdef/max(logN,EPS)
    def arr(x):
        y=x.detach().cpu().numpy().astype(float);mask=valid.detach().cpu().numpy();y[~mask]=np.nan;return y
    return {"valid":valid.detach().cpu().numpy(),"S_path":arr(H),"entropy_deficit":arr(Sdef),"entropy_deficit_norm":arr(Sdef_norm),"eta1":arr(eta1),"eta2":arr(eta2),"node_flow":[v.detach().cpu().numpy() for v in node],"edge_flow":[e.detach().cpu().numpy() for e in edge]}


@torch.no_grad()
def mlp_population_state(model:DeepMLP,X:torch.Tensor,y:torch.Tensor,task:str,n_inputs:int)->Dict[str,Any]:
    n=min(n_inputs,len(X)); X=X[:n]; y=y[:n]
    edge_sum=None; node_sum=None; rows=[]; validn=0
    for lo in range(0,n,CONDITIONAL_BATCH):
        xb=X[lo:lo+CONDITIONAL_BATCH]; x0=torch.zeros_like(xb); resp=mlp_response_sq(model,xb,x0)
        target=y[lo:lo+len(xb)].long() if task=="classification" else torch.zeros(len(xb),dtype=torch.long)
        st=mlp_factor_state(model,torch.square((xb-x0).reshape(len(xb),-1).to(torch.float64)),resp,target)
        for j,v in enumerate(st["valid"]):
            if v:
                rows.append({"input_index":lo+j,"valid":True,"S_path":st["S_path"][j],"entropy_deficit":st["entropy_deficit"][j],"entropy_deficit_norm":st["entropy_deficit_norm"][j],"eta1":st["eta1"][j],"eta2":st["eta2"][j]}); validn+=1
            else: rows.append({"input_index":lo+j,"valid":False})
        good=np.where(st["valid"])[0]
        if len(good):
            if edge_sum is None:
                edge_sum=[np.sum(e[good],axis=0) for e in st["edge_flow"]]; node_sum=[np.sum(v[good],axis=0) for v in st["node_flow"]]
            else:
                for i,e in enumerate(st["edge_flow"]): edge_sum[i]+=np.sum(e[good],axis=0)
                for i,v in enumerate(st["node_flow"]): node_sum[i]+=np.sum(v[good],axis=0)
    idf=pd.DataFrame(rows)
    validity={
        "n_inputs_evaluated":int(n),
        "n_valid_inputs":int(validn),
        "valid_fraction":float(validn/max(n,1)),
        "realization_defined":bool(validn>0),
        "degenerate":bool(validn==0),
        "status":"degenerate_zero_support" if validn==0 else ("defined_full_support" if validn==n else "defined_partial_support"),
        "reason":"zero_complete_path_support" if validn==0 else "",
    }
    if validn==0:
        # No normalized conditional path measure exists. Do not manufacture numerical observables.
        return {"valid_fraction":0.0,"summary":{},"state":None,"input_df":idf,"validity":validity}
    edge=[e/validn for e in edge_sum]; node=[v/validn for v in node_sum]
    good=idf[idf.valid==True]
    summary={"conditional_valid_fraction":validn/n,"S_path_median":float(good.S_path.median()),"entropy_deficit_median":float(good.entropy_deficit.median()),"entropy_deficit_norm_median":float(good.entropy_deficit_norm.median()),"eta1_median":float(good.eta1.median()),"eta2_median":float(good.eta2.median())}
    return {"valid_fraction":validn/n,"summary":summary,"state":{"edge_flow":edge,"node_flow":node},"input_df":idf,"validity":validity}


@torch.no_grad()
def mlp_structural_population(model:DeepMLP,output_dim:int,task:str)->Dict[str,Any]:
    targets=list(range(output_dim)) if task=="classification" else [0]
    edge_sum=None; node_sum=None; vals=[]
    for t in targets:
        src=torch.ones((1,model.layers[0].in_features),dtype=torch.float64)
        resp=[torch.ones((1,l.out_features),dtype=torch.float64,device=DEVICE) for l in model.layers[:-1]]
        st=mlp_factor_state(model,src,resp,torch.tensor([t]))
        vals.append({k:st[k][0] for k in ["S_path","entropy_deficit","entropy_deficit_norm","eta1","eta2"]})
        if edge_sum is None:
            edge_sum=[e[0].copy() for e in st["edge_flow"]]; node_sum=[v[0].copy() for v in st["node_flow"]]
        else:
            for i,e in enumerate(st["edge_flow"]): edge_sum[i]+=e[0]
            for i,v in enumerate(st["node_flow"]): node_sum[i]+=v[0]
    m=len(targets); edge=[e/m for e in edge_sum]; node=[v/m for v in node_sum]
    return {"summary":{k:float(np.mean([x[k] for x in vals])) for k in ["S_path","entropy_deficit","entropy_deficit_norm","eta1","eta2"]},"state":{"edge_flow":edge,"node_flow":node}}

# ==================================================================================================
# 6. PATH STATES — CNN SHARED-PARAMETER REALIZATION
# ==================================================================================================

@torch.no_grad()
def cnn_response_sq(model:DeepCNN,X:torch.Tensor,X0:torch.Tensor)->List[torch.Tensor]:
    _,z,_=model.forward_with_states(X.to(DEVICE)); _,z0,_=model.forward_with_states(X0.to(DEVICE))
    return [torch.square(finite_secant_torch64(model.activation_name,a,b)) for a,b in zip(z,z0)]


@torch.no_grad()
def cnn_log_partition_q(model:DeepCNN,source_sq:torch.Tensor,response_sq:List[torch.Tensor],target_idx:torch.Tensor,q:float)->torch.Tensor:
    src=torch.pow(torch.clamp(source_sq.to(DEVICE,dtype=torch.float64),min=0.0),float(q));a,s0=_normalize_rows(src);valid=torch.isfinite(s0)&(s0>0);logscale=torch.where(valid,torch.log(torch.clamp(s0,min=EPS)),torch.full_like(s0,-torch.inf))
    for i,c in enumerate(model.convs):
        w=torch.pow(torch.square(c.weight.to(torch.float64)),float(q));a=F.conv2d(a,w,bias=None,stride=1,padding=0)
        if i<len(model.convs)-1:a=a*torch.pow(torch.clamp(response_sq[i].to(DEVICE,dtype=torch.float64),min=0.0),float(q))
        a,sl=_normalize_rows(a);good=valid&torch.isfinite(sl)&(sl>0);logscale=torch.where(good,logscale+torch.log(torch.clamp(sl,min=EPS)),torch.full_like(logscale,-torch.inf));valid=good
    t=target_idx.to(DEVICE).long();tmass=a[torch.arange(a.shape[0],device=DEVICE),t,:,:].sum(dim=(-1,-2));good=valid&torch.isfinite(tmass)&(tmass>0);hout,wout=model.spatial_shapes()[-1];g2q=(1.0/(hout*wout))**(2.0*float(q))
    return torch.where(good,logscale+torch.log(torch.clamp(tmass,min=EPS))+math.log(g2q),torch.full_like(logscale,-torch.inf))


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
    return {"valid":valid.detach().cpu().numpy(),"S_path":arr(Hpath),"entropy_deficit":arr(Sdef),"entropy_deficit_norm":arr(Sdef_norm),"eta1":arr(eta1),"eta2":arr(eta2),"node_flow":[v.detach().cpu().numpy() for v in node],"occurrence_flow":[e.detach().cpu().numpy() for e in occurrence],"parameter_flow":[p.detach().cpu().numpy() for p in parameter],"layer_metrics":[{k:(arr(v) if torch.is_tensor(v) else np.full(B,float(v))) for k,v in d.items()} for d in layer_metrics]}


@torch.no_grad()
def cnn_population_state(model:DeepCNN,X:torch.Tensor,y:torch.Tensor,task:str,n_inputs:int)->Dict[str,Any]:
    n=min(n_inputs,len(X)); X=X[:n]; y=y[:n]; occ_sum=None; par_sum=None; node_sum=None; rows=[]; layer_rows=[]; validn=0
    for lo in range(0,n,CONDITIONAL_BATCH):
        xb=X[lo:lo+CONDITIONAL_BATCH]; x0=torch.zeros_like(xb); resp=cnn_response_sq(model,xb,x0); target=y[lo:lo+len(xb)].long() if task=="classification" else torch.zeros(len(xb),dtype=torch.long)
        st=cnn_factor_state(model,torch.square((xb-x0).to(torch.float64)),resp,target)
        good=np.where(np.asarray(st["valid"],bool))[0]
        for j in range(len(xb)):
            is_valid=bool(st["valid"][j])
            rec={"input_index":lo+j,"valid":is_valid}
            if is_valid:
                rec.update({"S_path":st["S_path"][j],"entropy_deficit":st["entropy_deficit"][j],"entropy_deficit_norm":st["entropy_deficit_norm"][j],"eta1":st["eta1"][j],"eta2":st["eta2"][j]}); validn+=1
                for li,d in enumerate(st["layer_metrics"]): layer_rows.append({"input_index":lo+j,"layer":li,**{k:float(v[j]) for k,v in d.items()}})
            rows.append(rec)
        if len(good):
            if occ_sum is None:
                occ_sum=[np.sum(e[good],0) for e in st["occurrence_flow"]]; par_sum=[np.sum(p[good],0) for p in st["parameter_flow"]]; node_sum=[np.sum(v[good],0) for v in st["node_flow"]]
            else:
                for i,e in enumerate(st["occurrence_flow"]):occ_sum[i]+=np.sum(e[good],0)
                for i,p in enumerate(st["parameter_flow"]):par_sum[i]+=np.sum(p[good],0)
                for i,v in enumerate(st["node_flow"]):node_sum[i]+=np.sum(v[good],0)
    idf=pd.DataFrame(rows)
    # Keep a stable schema even when every conditional contrast is invalid.
    # This is essential because downstream training aggregation is allowed to
    # encounter a zero-valid population without treating it as an exception.
    _cnn_layer_cols = [
        "input_index", "layer",
        "S_occ", "S_param", "S_reuse",
        "eta1_occ", "eta1_param", "eta1_reuse",
        "reuse_multiplicity",
    ]
    ldf=pd.DataFrame(layer_rows, columns=_cnn_layer_cols)
    validity={
        "n_inputs_evaluated":int(n),
        "n_valid_inputs":int(validn),
        "valid_fraction":float(validn/max(n,1)),
        "realization_defined":bool(validn>0),
        "degenerate":bool(validn==0),
        "status":"degenerate_zero_support" if validn==0 else ("defined_full_support" if validn==n else "defined_partial_support"),
        "reason":"zero_complete_path_support" if validn==0 else "",
    }
    if validn==0:
        # The conditional measure itself does not exist, so there is no numerical path-observable row.
        return {"summary":{},"state":None,"input_df":idf,"layer_df":ldf,"validity":validity}
    occ=[e/validn for e in occ_sum]; par=[p/validn for p in par_sum]; node=[v/validn for v in node_sum]
    hidden=ldf[ldf.layer<ldf.layer.max()] if len(ldf) else ldf; goodidf=idf[idf.valid==True]
    summary={"conditional_valid_fraction":validn/max(n,1),"S_path_median":float(goodidf.S_path.median()),"entropy_deficit_median":float(goodidf.entropy_deficit.median()),"entropy_deficit_norm_median":float(goodidf.entropy_deficit_norm.median()),"eta1_median":float(goodidf.eta1.median()),"eta2_median":float(goodidf.eta2.median())}
    for c in ["S_occ","S_param","S_reuse","eta1_occ","eta1_param","eta1_reuse"]:
        if len(hidden):summary[f"hidden_mean_{c}"]=float(hidden.groupby("input_index")[c].mean().median())
    return {"summary":summary,"state":{"occurrence_flow":occ,"parameter_flow":par,"node_flow":node},"input_df":idf,"layer_df":ldf,"validity":validity}


@torch.no_grad()
def cnn_structural_population(model:DeepCNN,output_dim:int,task:str)->Dict[str,Any]:
    targets=list(range(output_dim)) if task=="classification" else [0]; occ_sum=None; par_sum=None; node_sum=None; vals=[]; layers=[]
    shapes=model.spatial_shapes()
    for t in targets:
        src=torch.ones((1,model.channels[0],IMAGE_RESOLUTION,IMAGE_RESOLUTION),dtype=torch.float64)
        resp=[torch.ones((1,model.convs[i].out_channels,*shapes[i+1]),dtype=torch.float64,device=DEVICE) for i in range(len(model.convs)-1)]
        st=cnn_factor_state(model,src,resp,torch.tensor([t])); vals.append({k:st[k][0] for k in ["S_path","entropy_deficit","entropy_deficit_norm","eta1","eta2"]}); layers.append([{k:float(v[0]) for k,v in d.items()} for d in st["layer_metrics"]])
        if occ_sum is None:
            occ_sum=[e[0].copy() for e in st["occurrence_flow"]]; par_sum=[p[0].copy() for p in st["parameter_flow"]]; node_sum=[v[0].copy() for v in st["node_flow"]]
        else:
            for i,e in enumerate(st["occurrence_flow"]):occ_sum[i]+=e[0]
            for i,p in enumerate(st["parameter_flow"]):par_sum[i]+=p[0]
            for i,v in enumerate(st["node_flow"]):node_sum[i]+=v[0]
    m=len(targets); occ=[e/m for e in occ_sum]; par=[p/m for p in par_sum]; node=[v/m for v in node_sum]
    layer_df=pd.DataFrame([{**{"target":ti,"layer":li},**d} for ti,L in enumerate(layers) for li,d in enumerate(L)])
    return {"summary":{k:float(np.mean([x[k] for x in vals])) for k in ["S_path","entropy_deficit","entropy_deficit_norm","eta1","eta2"]},"state":{"occurrence_flow":occ,"parameter_flow":par,"node_flow":node},"layer_df":layer_df}

# ==================================================================================================
# 7. GEOMETRY CARRIERS
# ==================================================================================================

@dataclass
class FlowGraph:
    n_nodes:int
    edge_u:np.ndarray
    edge_v:np.ndarray
    edge_mag:np.ndarray
    edge_layer:np.ndarray
    node_meta:List[Dict[str,Any]]


def mlp_graph_from_state(model:DeepMLP,state:Dict[str,Any],enrichment:bool)->FlowGraph:
    node_sizes=[model.layers[0].in_features]+[l.out_features for l in model.layers]
    offsets=np.cumsum([0]+node_sizes[:-1]).astype(np.int64); meta=[]
    for li,n in enumerate(node_sizes):
        for j in range(n):meta.append({"layer":li,"channel":j,"y":-1,"x":-1})
    U=[];V=[];Fv=[];Lv=[]
    for li,E0 in enumerate(state["edge_flow"]):
        E=np.clip(np.asarray(E0,float),0,None); E=E*(E.size if enrichment else 1.0)
        no,ni=E.shape
        oo,ii=np.indices((no,ni),dtype=np.int64)
        U.append(offsets[li]+ii.ravel()); V.append(offsets[li+1]+oo.ravel()); Fv.append(E.ravel()); Lv.append(np.full(E.size,li,dtype=np.int32))
    return FlowGraph(sum(node_sizes),np.concatenate(U),np.concatenate(V),np.concatenate(Fv),np.concatenate(Lv),meta)


def cnn_graph_from_state(model:DeepCNN,state:Dict[str,Any],enrichment:bool)->FlowGraph:
    shapes_hw=model.spatial_shapes(); layer_shapes=[(model.channels[i],*shapes_hw[i]) for i in range(len(shapes_hw))]
    offsets=[0]
    for c,h,w in layer_shapes[:-1]:offsets.append(offsets[-1]+c*h*w)
    meta=[]
    for li,(c,h,w) in enumerate(layer_shapes):
        for cc in range(c):
            for yy in range(h):
                for xx in range(w):meta.append({"layer":li,"channel":cc,"y":yy,"x":xx})
    U=[];V=[];Fv=[];Lv=[]
    for li,E0 in enumerate(state["occurrence_flow"]):
        E=np.clip(np.asarray(E0,float),0,None); E=E*(E.size if enrichment else 1.0)
        cout,cin,k,_,Hout,Wout=E.shape; _,Hin,Win=layer_shapes[li]
        co,ci,dy,dx,yy,xx=np.indices(E.shape,dtype=np.int64)
        u=(offsets[li] + ci*Hin*Win + (yy+dy)*Win + (xx+dx)).ravel()
        v=(offsets[li+1] + co*Hout*Wout + yy*Wout + xx).ravel()
        U.append(u);V.append(v);Fv.append(E.ravel());Lv.append(np.full(E.size,li,dtype=np.int32))
    return FlowGraph(sum(c*h*w for c,h,w in layer_shapes),np.concatenate(U),np.concatenate(V),np.concatenate(Fv),np.concatenate(Lv),meta)


def global_node_measure(state:Dict[str,Any],family:str)->np.ndarray:
    node=state["node_flow"]; x=np.concatenate([np.asarray(v,float).ravel() for v in node]); x=np.clip(x,0,None); x/=max(float(x.sum()),EPS); return x


def metric_lengths(G:FlowGraph,power:float=METRIC_POWER)->np.ndarray:
    return np.power(np.clip(G.edge_mag,0,None)+EPS,-float(power))


def distance_matrix(G:FlowGraph,lengths:np.ndarray)->np.ndarray:
    rows=np.concatenate([G.edge_u,G.edge_v]); cols=np.concatenate([G.edge_v,G.edge_u]); vals=np.concatenate([lengths,lengths])
    A=csr_matrix((vals,(rows,cols)),shape=(G.n_nodes,G.n_nodes),dtype=float)
    return np.asarray(csgraph_dijkstra(A,directed=False,unweighted=False),float)


def bcanw_scales(D:np.ndarray,n_scales:int)->np.ndarray:
    vals=D[np.triu_indices(D.shape[0],k=1)];vals=vals[np.isfinite(vals)&(vals>0)]
    if len(vals)==0:return np.array([],float)
    lo=max(float(vals.min())*.95,EPS);hi=float(vals.max())*1.005
    return np.geomspace(lo,hi,max(n_scales,4)) if hi>lo else np.array([lo])


def node_strength(G:FlowGraph)->np.ndarray:
    s=np.zeros(G.n_nodes,float); np.add.at(s,G.edge_u,G.edge_mag); np.add.at(s,G.edge_v,G.edge_mag); return s

def greedy_bcanw_coloring(D:np.ndarray,ell:float,order:np.ndarray)->np.ndarray:
    color=np.full(D.shape[0],-1,int)
    for u in order:
        used=set(color[(D[u]>=ell)&(color>=0)].tolist()); c=0
        while c in used:c+=1
        color[u]=c
    return color


def renyi_effective_fraction(p:np.ndarray,q:float)->float:
    p=np.asarray(p,float);p=p[np.isfinite(p)&(p>0)]
    if len(p)==0:return np.nan
    p=p/p.sum(); E=len(p); H=-np.sum(p*np.log(p)) if abs(q-1)<1e-12 else math.log(np.sum(p**q))/(1-q)
    return float(math.exp(H)/E)


def box_flow_observables(colors:np.ndarray,node_mu:np.ndarray,Graw:FlowGraph)->Dict[str,Any]:
    """Box mass, directed quotient flow, retention and coarse Renyi statistics.

    Uses vectorized edge arrays and sparse quotient-pair aggregation. No dense NB x NB matrix is built.
    Because node ids are layer ordered, edge_u -> edge_v retains the computational feed-forward direction.
    """
    nb=int(colors.max())+1
    mass=np.bincount(colors,weights=node_mu,minlength=nb).astype(float); mean=max(float(mass.mean()),EPS); lac=float(np.mean(mass*mass)/(mean*mean))
    a=colors[Graw.edge_u]; b=colors[Graw.edge_v]; f=Graw.edge_mag
    total=float(np.sum(f)); off=(a!=b); inter=float(np.sum(f[off]))
    if np.any(off):
        pair=a[off].astype(np.int64)*np.int64(nb)+b[off].astype(np.int64)
        uniq,inv=np.unique(pair,return_inverse=True)
        vals=np.bincount(inv,weights=f[off]).astype(float)
        vals=vals[np.isfinite(vals)&(vals>0)]
    else:
        vals=np.array([],float)
    if len(vals):
        qnorm=vals/max(float(vals.mean()),EPS); probs=vals/max(float(vals.sum()),EPS); eta={q:renyi_effective_fraction(probs,q) for q in COARSE_RENYI_Q}
    else:qnorm=np.array([]);eta={q:np.nan for q in COARSE_RENYI_Q}
    return {"lacunarity":lac,"quotient_flow_norm":qnorm,"coarse_eta_q":eta,"interbox_flow_fraction":float(inter/max(total,EPS))}

def compress_box_staircase_midpoint(radii,NB,lac=None):
    r=np.asarray(radii,float); n=np.asarray(NB,float); good=np.isfinite(r)&np.isfinite(n)&(r>0); r,n=r[good],n[good]; l=np.asarray(lac,float)[good] if lac is not None else None
    if len(r)==0:return (r,n,np.array([]),pd.DataFrame()) if lac is not None else (r,n,pd.DataFrame())
    rows=[]; rc=[]; nc=[]; lc=[]; start=0
    for k in range(1,len(r)+1):
        end=(k==len(r)) or (n[k] != n[start] if k<len(r) else True)
        if end:
            j=k-1; mid=math.sqrt(r[start]*r[j]); rc.append(mid); nc.append(n[start]);
            if l is not None:lc.append(float(np.nanmedian(l[start:j+1])))
            rows.append({"plateau_index":len(rows),"NB":float(n[start]),"ell_min":float(r[start]),"ell_max":float(r[j]),"ell_mid":float(mid),"n_raw_scales":int(j-start+1),"span_decades":float(np.log10(r[j]/r[start])) if r[j]>r[start] else 0.0})
            start=k
    if l is not None:return np.asarray(rc),np.asarray(nc),np.asarray(lc),pd.DataFrame(rows)
    return np.asarray(rc),np.asarray(nc),pd.DataFrame(rows)


def covering_quantile_scale_loglog(radii,NB,N,q,return_diagnostics:bool=False):
    r=np.asarray(radii,float); y=np.asarray(NB,float)/float(N); good=np.isfinite(r)&np.isfinite(y)&(r>0)&(y>0);r,y=r[good],y[good]
    diag={"monotonicity_corrections":0,"max_upward_cover_fraction_step":0.0}
    if len(r)<2:return (np.nan,diag) if return_diagnostics else np.nan
    order=np.argsort(r);r,y=r[order],y[order]
    dy=np.diff(y); upward=dy>1e-12
    diag["monotonicity_corrections"]=int(np.sum(upward));diag["max_upward_cover_fraction_step"]=float(np.max(dy[upward])) if np.any(upward) else 0.0
    y=np.minimum.accumulate(y)
    if not (y[-1] <= q <= y[0]):return (np.nan,diag) if return_diagnostics else np.nan
    idx=np.where(y<=q)[0]
    if len(idx)==0:return (np.nan,diag) if return_diagnostics else np.nan
    j=int(idx[0])
    if j==0:
        xi=float(r[0]) if np.isclose(y[0],q) else np.nan
        return (xi,diag) if return_diagnostics else xi
    ly0,ly1=math.log(y[j-1]),math.log(y[j]);lr0,lr1=math.log(r[j-1]),math.log(r[j]);lq=math.log(q)
    xi=float(math.sqrt(r[j-1]*r[j])) if abs(ly1-ly0)<EPS else float(math.exp(lr0+(lq-ly0)/(ly1-ly0)*(lr1-lr0)))
    return (xi,diag) if return_diagnostics else xi

def local_effective_dimension_transition(radii_c,NB_c,xi50,window=LOCAL_SLOPE_WINDOW):
    r=np.asarray(radii_c,float);n=np.asarray(NB_c,float);good=np.isfinite(r)&np.isfinite(n)&(r>0)&(n>1);r,n=r[good],n[good]
    if len(r)==0 or not np.isfinite(xi50) or xi50<=0:return np.array([]),np.array([]),np.array([])
    x=r/xi50;lx=np.log(r);ly=np.log(n);d=np.full(len(r),np.nan);rr=np.full(len(r),np.nan);half=max(window//2,1)
    for i in range(len(r)):
        lo=max(0,i-half);hi=min(len(r),i+half+1)
        if hi-lo<3:continue
        b,a=np.polyfit(lx[lo:hi],ly[lo:hi],1);pred=a+b*lx[lo:hi];d[i]=-float(b);rr[i]=r2_score_np(ly[lo:hi],pred)
    return x,d,rr


def strict_scaling_window(radii_c,NB_c,N):
    empty={"dB":np.nan,"r2":np.nan,"span_decades":0.,"n_points":0,"credible":False}
    r=np.asarray(radii_c,float);nb=np.asarray(NB_c,float);good=np.isfinite(r)&np.isfinite(nb)&(r>0)&(nb>1)&(nb<.95*N);ids=np.where(good)[0]
    if len(ids)<SCALING_MIN_POINTS:return empty
    cand=[]
    for a in range(len(ids)-SCALING_MIN_POINTS+1):
        for b in range(a+SCALING_MIN_POINTS,len(ids)+1):
            sel=ids[a:b]
            if np.any(np.diff(sel)!=1):continue
            rs,ns=r[sel],nb[sel];span=float(np.log10(rs[-1]/rs[0]));nbr=float(ns[0]/max(ns[-1],EPS))
            if span<SCALING_MIN_SPAN_DECADES or nbr<SCALING_MIN_NB_RATIO:continue
            x,y=np.log(rs),np.log(ns);slope,inter=np.polyfit(x,y,1);dim=-float(slope);pred=inter+slope*x;rr=r2_score_np(y,pred)
            if not np.isfinite(rr) or rr<SCALING_R2_MIN or dim<SCALING_MIN_DIM:continue
            cand.append((span,rr,dim,len(sel)))
    if not cand:return empty
    cand.sort(reverse=True);span,rr,dim,npnt=cand[0];return {"dB":dim,"r2":rr,"span_decades":span,"n_points":npnt,"credible":True}


def fit_common_local_growth(rgrid,support,mass):
    r=np.asarray(rgrid,float);s=np.asarray(support,float);m=np.asarray(mass,float);good=np.isfinite(r)&np.isfinite(s)&np.isfinite(m)&(r>0)&(s>0)&(m>0);r,s,m=r[good],s[good],m[good]
    keep=[0] if len(r) else []
    for k in range(1,len(r)):
        if not(np.isclose(s[k],s[keep[-1]],rtol=1e-10,atol=1e-14) and np.isclose(m[k],m[keep[-1]],rtol=1e-10,atol=1e-14)):keep.append(k)
    if not keep:return (np.nan,)*7
    r,s,m=r[keep],s[keep],m[keep];valid=(s<LOCAL_COMMON_SATURATION)&(m<LOCAL_COMMON_SATURATION);r,s,m=r[valid],s[valid],m[valid]
    if len(r)<LOCAL_COMMON_MIN_POINTS:return (np.nan,)*7
    best=None
    for i in range(len(r)-LOCAL_COMMON_MIN_POINTS+1):
        for j in range(i+LOCAL_COMMON_MIN_POINTS,len(r)+1):
            rr,ss,mm=r[i:j],s[i:j],m[i:j];span=float(np.log10(rr[-1]/rr[0]))
            if span<LOCAL_COMMON_MIN_SPAN_DECADES or ss[-1]/max(ss[0],EPS)<LOCAL_COMMON_SUPPORT_GROWTH or mm[-1]/max(mm[0],EPS)<LOCAL_COMMON_MASS_GROWTH:continue
            x=np.log(rr);bd,ad=np.polyfit(x,np.log(ss),1);ba,aa=np.polyfit(x,np.log(mm),1)
            if bd<=0 or ba<=0:continue
            r2d=r2_score_np(np.log(ss),ad+bd*x);r2a=r2_score_np(np.log(mm),aa+ba*x)
            if r2d<LOCAL_COMMON_R2_MIN or r2a<LOCAL_COMMON_R2_MIN:continue
            score=(span,min(r2d,r2a),len(rr));rec=(float(bd),float(ba),float(ba-bd),float(r2d),float(r2a),span,int(len(rr)))
            if best is None or score>best[0]:best=(score,rec)
    return best[1] if best else (np.nan,)*7


def local_metric_dimensions(D,node_mu,node_meta,max_nodes=LOCAL_DIMENSION_NODES,max_radii=LOCAL_RADII_MAX)->Tuple[Dict[str,Any],pd.DataFrame]:
    N=D.shape[0];ids=np.arange(N)
    if max_nodes is not None and max_nodes<N:
        rng=np.random.default_rng(SEED+N);ids=np.sort(rng.choice(N,size=max_nodes,replace=False))
    rows=[];deltas=[]
    for i in ids:
        di=np.asarray(D[i],float);vals=np.unique(di[np.isfinite(di)&(di>0)])
        if len(vals)<LOCAL_COMMON_MIN_POINTS:continue
        if len(vals)>max_radii:
            jj=np.unique(np.round(np.linspace(0,len(vals)-1,max_radii)).astype(int));vals=vals[jj]
        support=np.array([np.mean(di<=r) for r in vals],float);mass=np.array([np.sum(node_mu[di<=r]) for r in vals],float)
        bd,ba,dd,r2d,r2a,span,npnt=fit_common_local_growth(vals,support,mass)
        meta=node_meta[i] if i<len(node_meta) else {}
        rows.append({"node_id":int(i),**meta,"d_i_F":bd,"alpha_i_F":ba,"delta_i_F":dd,"r2_d":r2d,"r2_alpha":r2a,"span_decades":span,"n_fit":npnt,"node_measure":float(node_mu[i])})
        if np.isfinite(dd):deltas.append(dd)
    df=pd.DataFrame(rows);summ={"local_delta_median":float(np.nanmedian(deltas)) if len(deltas) else np.nan,"valid_local_delta_fraction":float(np.mean(np.isfinite(df.delta_i_F))) if len(df) else np.nan,"local_nodes_evaluated":int(len(ids))}
    return summ,df


def nearest_resolution_indices(NB,N):
    frac=np.asarray(NB,float)/N;out=[]
    for t in RENORM_TARGET_BOX_FRACTIONS:
        elig=np.where((NB>=3)&(NB<=.90*N))[0]
        if len(elig)==0:continue
        idx=int(elig[np.argmin(np.abs(frac[elig]-t))]);
        if idx not in out:out.append(idx)
    return out


def mean_pairwise_ks(arrays):
    arr=[]
    for a in arrays:
        v=np.asarray(a,float);v=v[np.isfinite(v)&(v>0)]
        if len(v)>=5:arr.append(v)
    if len(arr)<2:return np.nan
    vals=[ks_2samp(arr[i],arr[j]).statistic for i in range(len(arr)) for j in range(i+1,len(arr))]
    return float(np.mean(vals)) if vals else np.nan


def analyze_geometry(model:nn.Module,state:Dict[str,Any],family:str,n_scales:int,compute_local:bool=True):
    Ggeom=mlp_graph_from_state(model,state,True) if family=="mlp" else cnn_graph_from_state(model,state,True)
    if Ggeom.n_nodes>BCANW_MAX_NODES:
        return {"geometry_skipped":True,"geometry_skip_reason":f"n_nodes={Ggeom.n_nodes}>cap={BCANW_MAX_NODES}","n_graph_nodes":Ggeom.n_nodes},pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    Graw=mlp_graph_from_state(model,state,False) if family=="mlp" else cnn_graph_from_state(model,state,False)
    D=distance_matrix(Ggeom,metric_lengths(Ggeom));radii=bcanw_scales(D,n_scales);order=np.lexsort((np.arange(Ggeom.n_nodes),-node_strength(Ggeom)));node_mu=global_node_measure(state,family)
    NB=[];lac=[];inter=[];qnorm=[];coarse={q:[] for q in COARSE_RENYI_Q}
    for ell in radii:
        colors=greedy_bcanw_coloring(D,float(ell),order);NB.append(int(colors.max())+1);b=box_flow_observables(colors,node_mu,Graw);lac.append(b["lacunarity"]);inter.append(b["interbox_flow_fraction"]);qnorm.append(b["quotient_flow_norm"])
        for q in COARSE_RENYI_Q:coarse[q].append(b["coarse_eta_q"][q])
    NB=np.asarray(NB,float);lac=np.asarray(lac,float);inter=np.asarray(inter,float)
    rc,nc,lc,plateaus=compress_box_staircase_midpoint(radii,NB,lac);xi50,xi_diag=covering_quantile_scale_loglog(rc,nc,Ggeom.n_nodes,.5,return_diagnostics=True);xc,deff,deffr2=local_effective_dimension_transition(rc,nc,xi50);sw=strict_scaling_window(rc,nc,Ggeom.n_nodes)
    qidx=nearest_resolution_indices(NB,Ggeom.n_nodes);qks=mean_pairwise_ks([qnorm[i] for i in qidx]);ret=float(np.nanmean(inter[qidx])) if len(qidx) else np.nan
    local_summ={"local_delta_median":np.nan,"valid_local_delta_fraction":np.nan,"local_nodes_evaluated":0};local_df=pd.DataFrame()
    if compute_local:local_summ,local_df=local_metric_dimensions(D,node_mu,Ggeom.node_meta)
    valid_deff=(np.isfinite(deff)&np.isfinite(deffr2)&(deffr2>=LOCAL_SLOPE_R2_MIN)&(deff>0))
    dstar=float(np.nanmax(deff[valid_deff])) if np.any(valid_deff) else np.nan;xdstar=float(xc[np.where(valid_deff)[0][np.nanargmax(deff[valid_deff])]]) if np.any(valid_deff) else np.nan
    frac=NB/Ggeom.n_nodes;mask=np.isfinite(lac)&(frac>=.015)&(frac<=.85);lacstar=float(np.nanmax(lac[mask])) if np.any(mask) else np.nan
    summ={"geometry_skipped":False,"geometry_skip_reason":"","n_graph_nodes":Ggeom.n_nodes,"xi50":xi50,"log_xi50":safe_log(xi50),"d_star":dstar,"x_d_star":xdstar,"lacunarity_star":lacstar,"scaling_dB":sw["dB"],"scaling_r2":sw["r2"],"scaling_credible":sw["credible"],"quotient_mean_pairwise_KS":qks,"renorm_flow_retention_mean":ret,"n_plateaus":len(plateaus),"transition_fraction":len(rc)/max(len(radii),1),"max_plateau_span_decades":float(plateaus.span_decades.max()) if len(plateaus) else np.nan,"xi50_bracketed":bool(np.isfinite(xi50)),"cover_monotonicity_corrections":int(xi_diag["monotonicity_corrections"]),"max_upward_cover_fraction_step":float(xi_diag["max_upward_cover_fraction_step"]),**local_summ}
    # raw scale table + only transition-supported interpolation of d_eff
    xraw=radii/xi50 if np.isfinite(xi50) and xi50>0 else np.full(len(radii),np.nan);deff_interp=np.full(len(radii),np.nan);deffr2_interp=np.full(len(radii),np.nan)
    good=np.isfinite(xc)&np.isfinite(deff)&np.isfinite(deffr2)&(deffr2>=LOCAL_SLOPE_R2_MIN)&(deff>0)
    if np.sum(good)>=2:
        lo,hi=np.nanmin(xc[good]),np.nanmax(xc[good]);inside=np.isfinite(xraw)&(xraw>=lo)&(xraw<=hi);deff_interp[inside]=np.interp(np.log(xraw[inside]),np.log(xc[good]),deff[good]);deffr2_interp[inside]=np.interp(np.log(xraw[inside]),np.log(xc[good]),deffr2[good])
    rows=[]
    for k,ell in enumerate(radii):
        rec={"scale_index":k,"ell_B":float(ell),"ell_over_xi50":float(xraw[k]),"NB":float(NB[k]),"NB_over_N":float(NB[k]/Ggeom.n_nodes),"lacunarity":float(lac[k]),"interbox_flow_fraction":float(inter[k]),"d_eff_interp":float(deff_interp[k]),"d_eff_r2_interp":float(deffr2_interp[k])}
        for q in COARSE_RENYI_Q:rec[f"coarse_eta_q{q:g}"]=float(coarse[q][k])
        rows.append(rec)
    mask_comp=(nc>1)&np.isfinite(rc)&np.isfinite(nc);rcd,ncd,lcd=rc[mask_comp],nc[mask_comp],lc[mask_comp]
    nfit=len(rcd);deff_c=deff if len(deff)==nfit else np.full(nfit,np.nan);deffr2_c=deffr2 if len(deffr2)==nfit else np.full(nfit,np.nan)
    comp=pd.DataFrame({"ell_mid":rcd,"NB":ncd,"NB_over_N":ncd/Ggeom.n_nodes,"ell_over_xi50":rcd/xi50 if np.isfinite(xi50) else np.full(nfit,np.nan),"d_eff":deff_c,"d_eff_r2":deffr2_c,"lacunarity":lcd})
    return summ,pd.DataFrame(rows),comp,local_df.assign(**{}) if len(local_df) else local_df


# ==================================================================================================
# 7B. RICH-POPULATION CONDITIONAL-SUPPORT PREFLIGHT
# ==================================================================================================

@torch.no_grad()
def activation_richness_diagnostics(model:nn.Module,X:torch.Tensor,environment:str,dataset:str,dataset_meta:Dict[str,Any])->Tuple[Dict[str,float],pd.DataFrame]:
    """Measure hidden response richness without constructing BCANw geometry."""
    Xuse=X if environment=="clean" else shifted_environment(X,dataset,dataset_meta)
    _,zs,_=model.forward_with_states(Xuse.to(DEVICE))
    layer_rows=[]
    for li,z in enumerate(zs):
        zz=z.detach().to(torch.float64)
        flat=zz.reshape(zz.shape[0],-1)
        finite=torch.isfinite(flat)
        finite_fraction=float(finite.double().mean().item())
        absmax=float(torch.max(torch.abs(flat[finite])).item()) if finite.any() else float("inf")
        z0=torch.zeros_like(zz)
        sec=finite_secant_torch64(model.activation_name,zz,z0)
        sec2=torch.square(sec).reshape(sec.shape[0],-1)
        response_positive=(sec2>0).double().mean(1).detach().cpu().numpy()
        posvals=sec2[sec2>0].detach().cpu().numpy()
        rec={"layer":li,"n_units":int(flat.shape[1]),"preactivation_finite_fraction":finite_fraction,"preactivation_abs_max":absmax,
             "response_positive_fraction_q01":float(np.quantile(response_positive,.01)),"response_positive_fraction_median":float(np.median(response_positive)),
             "response_sq_positive_q01":float(np.quantile(posvals,.01)) if len(posvals) else 0.0}
        if model.activation_name=="relu":
            active=(flat>0).double().mean(1).detach().cpu().numpy()
            rec.update({"relu_active_q01":float(np.quantile(active,.01)),"relu_active_q05":float(np.quantile(active,.05)),"relu_active_median":float(np.median(active))})
        else:
            rec.update({"relu_active_q01":np.nan,"relu_active_q05":np.nan,"relu_active_median":np.nan})
        layer_rows.append(rec)
    ldf=pd.DataFrame(layer_rows)
    summary={
        "all_preactivations_finite":bool(np.all(pd.to_numeric(ldf.preactivation_finite_fraction,errors="coerce").to_numpy(float)>=1.0-1e-12)) if len(ldf) else True,
        "max_preactivation_abs":float(pd.to_numeric(ldf.preactivation_abs_max,errors="coerce").max()) if len(ldf) else 0.0,
        "min_response_positive_fraction_q01":float(pd.to_numeric(ldf.response_positive_fraction_q01,errors="coerce").min()) if len(ldf) else 1.0,
        "min_response_sq_positive_q01":float(pd.to_numeric(ldf.response_sq_positive_q01,errors="coerce").min()) if len(ldf) else np.nan,
        "final_hidden_relu_active_q01":float(ldf.iloc[-1].relu_active_q01) if len(ldf) and model.activation_name=="relu" else np.nan,
        "min_hidden_relu_active_q01":float(pd.to_numeric(ldf.relu_active_q01,errors="coerce").min()) if len(ldf) and model.activation_name=="relu" else np.nan,
    }
    return summary,ldf


def flow_metric_preflight_diagnostics(state:Optional[Dict[str,Any]],family:str)->Dict[str,float]:
    """Cheap precursor diagnostics for lambda=(enrichment+eps)^-1, with no shortest paths."""
    if state is None:
        return {"flow_state_defined":False,"zero_flow_fraction":1.0,"positive_enrichment_q01":np.nan,"positive_enrichment_median":np.nan,
                "positive_enrichment_q99":np.nan,"positive_metric_log10_span":np.nan,"metric_log10_span_including_floor":np.nan}
    flows=state.get("edge_flow",[]) if family=="mlp" else state.get("occurrence_flow",[])
    enrich=[]; total=0; zero=0
    for f in flows:
        x=np.asarray(f,float).ravel(); total+=x.size; zero+=int(np.sum(x<=0)); enrich.append(x*max(x.size,1))
    if not enrich or total==0:
        return {"flow_state_defined":True,"zero_flow_fraction":np.nan,"positive_enrichment_q01":np.nan,"positive_enrichment_median":np.nan,
                "positive_enrichment_q99":np.nan,"positive_metric_log10_span":np.nan,"metric_log10_span_including_floor":np.nan}
    e=np.concatenate(enrich); pos=e[np.isfinite(e)&(e>0)]
    if len(pos):
        lengths=(pos+EPS)**(-METRIC_POWER); span=float(np.log10(np.max(lengths)/max(np.min(lengths),EPS))) if len(lengths)>1 else 0.0
        all_min=float((np.max(e)+EPS)**(-METRIC_POWER)); all_max=float(EPS**(-METRIC_POWER)); all_span=float(np.log10(all_max/max(all_min,EPS)))
        return {"flow_state_defined":True,"zero_flow_fraction":float(zero/total),"positive_enrichment_q01":float(np.quantile(pos,.01)),
                "positive_enrichment_median":float(np.median(pos)),"positive_enrichment_q99":float(np.quantile(pos,.99)),
                "positive_metric_log10_span":span,"metric_log10_span_including_floor":all_span}
    return {"flow_state_defined":True,"zero_flow_fraction":1.0,"positive_enrichment_q01":np.nan,"positive_enrichment_median":np.nan,
            "positive_enrichment_q99":np.nan,"positive_metric_log10_span":np.nan,"metric_log10_span_including_floor":0.0}


def _preflight_row(ds,arch,opt,act,seed,stage,env,cond,rich,perf=None,diverged=False):
    v=dict(cond.get("validity",{})); rec={"stage":stage,"dataset":ds,"family":family_of(arch),"arch_id":arch,"optimizer":opt,
        "activation":act,"seed_index":int(seed),"environment":env,"training_diverged":bool(diverged),**v,**rich}
    if perf:
        for k,val in perf.items():rec[f"performance_{k}"]=val
    return rec


def _preflight_gate_rows(init_df:pd.DataFrame,stress_df:pd.DataFrame)->pd.DataFrame:
    """Classify preflight outcomes by scientific severity.

    fatal:
      - nonfinite/training divergence;
      - tanh/SiLU losing complete-path support, which should not occur for finite states under x0=0.
    warning:
      - ReLU population support below the preferred richness threshold;
      - ReLU final-hidden activity below the preferred richness threshold.

    Warnings are recorded but do not terminate a Slurm allocation by default.
    """
    rows=[]
    for source,df in [("initialization",init_df),("stress_5epoch",stress_df)]:
        for _,r in df.iterrows():
            fatal=[]; warn=[]; act=str(r.activation); vf=float(r.get("valid_fraction",0.0))
            if bool(r.get("training_diverged",False)): fatal.append("training_diverged")
            if not bool(r.get("all_preactivations_finite",True)): fatal.append("nonfinite_preactivation")
            if act in {"tanh","silu"} and vf < 1.0-1e-12:
                fatal.append("smooth_activation_lost_complete_path_support")
            if act=="relu" and vf < PREFLIGHT_RELU_MIN_VALID_FRACTION:
                warn.append("relu_population_support_below_preferred_richness")
            aq=float(r.get("final_hidden_relu_active_q01",np.nan))
            if act=="relu" and np.isfinite(aq) and aq < PREFLIGHT_RELU_MIN_FINAL_ACTIVE_Q01:
                warn.append("relu_final_hidden_population_below_preferred_richness")
            severity = "fatal" if fatal else ("warning" if warn else "pass")
            reasons = fatal + warn
            rows.append({"stage":source,"dataset":r.dataset,"family":r.family,"arch_id":r.arch_id,"optimizer":r.optimizer,
                         "activation":act,"seed_index":int(r.seed_index),"environment":r.environment,
                         "gate_pass":severity=="pass","gate_severity":severity,"gate_reasons":";".join(reasons),
                         "valid_fraction":vf,"final_hidden_relu_active_q01":aq})
    return pd.DataFrame(rows)


def run_rich_population_preflight(data_cache:Dict[str,Any])->Dict[str,pd.DataFrame]:
    """Cheap preflight that characterizes support before the 324-network campaign.

    Stage A: all architecture x dataset x activation x seed initializations, no optimizer.
    Stage B: one seed, every optimizer, 2k balanced training examples, five epochs.
    Stage C: optional 8-state BCANw sentinel on M2/C2, Adam, ReLU, both datasets/environments.
    """
    if not RUN_PREFLIGHT:
        print("[preflight] disabled by PATHFLOW_BENCH_RUN_PREFLIGHT=0")
        return {}
    init_path=OUTDIR/"tables"/"preflight_initialization_support.csv"
    stress_path=OUTDIR/"tables"/"preflight_stress_support.csv"
    layer_path=OUTDIR/"tables"/"preflight_layer_richness.csv"
    metric_path=OUTDIR/"tables"/"preflight_metric_diagnostics.csv"
    gate_path=OUTDIR/"tables"/"preflight_gates.csv"
    sentinel_path=OUTDIR/"tables"/"preflight_bcanw_sentinel.csv"

    init_rows=[];stress_rows=[];layer_rows=[];metric_rows=[]
    # A. Initialization audit. Optimizer is intentionally labelled 'none'.
    total=len(DATASETS)*len(ARCH_IDS)*len(ACTIVATIONS)*len(SEEDS)
    bar=tqdm(total=total,desc="Preflight A: initialization support")
    for ds in DATASETS:
        d=data_cache[ds];Xtr,_=d["train"];Xt,yt=d["test"];ep=np.asarray(d["endpoint_indices"],int);Xep=Xt[ep];yep=yt[ep]
        for arch in ARCH_IDS:
            for act in ACTIVATIONS:
                for seed in SEEDS:
                    row=pd.Series({"dataset":ds,"task":task_for(ds),"family":family_of(arch),"arch_id":arch,"optimizer":"none","activation":act,"seed_index":seed})
                    model=build_model(arch,act,output_dim_for(ds),ds,seed);calibrate_initial_signal(model,Xtr)
                    for env in ENVIRONMENTS:
                        cond=endpoint_state(model,row,Xt,yt,env,d["meta"],ep)
                        rich,ldf=activation_richness_diagnostics(model,Xep,env,ds,d["meta"])
                        init_rows.append(_preflight_row(ds,arch,"none",act,seed,"initialization",env,cond,rich))
                        for _,lr in ldf.iterrows():layer_rows.append({"stage":"initialization","dataset":ds,"family":family_of(arch),"arch_id":arch,"optimizer":"none","activation":act,"seed_index":seed,"environment":env,**lr.to_dict()})
                        metric_rows.append({"stage":"initialization","dataset":ds,"family":family_of(arch),"arch_id":arch,"optimizer":"none","activation":act,"seed_index":seed,"environment":env,**flow_metric_preflight_diagnostics(cond.get("state"),family_of(arch))})
                    del model
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                    bar.update(1)
    bar.close()

    # B. Tiny optimizer stress map.
    total=len(DATASETS)*len(ARCH_IDS)*len(OPTIMIZERS)*len(ACTIVATIONS)
    bar=tqdm(total=total,desc="Preflight B: 5-epoch optimizer stress")
    for ds in DATASETS:
        d=data_cache[ds];Xtr,ytr=d["train"];Xv,yv=d["val"];Xt,yt=d["test"];trids=np.asarray(d["preflight_train_indices"],int);Xs=Xtr[trids];ys=ytr[trids]
        ep=np.asarray(d["endpoint_indices"],int);Xep=Xt[ep];yep=yt[ep]
        for arch in ARCH_IDS:
            for optname in OPTIMIZERS:
                for act in ACTIVATIONS:
                    seed=PREFLIGHT_SEED; row=pd.Series({"dataset":ds,"task":task_for(ds),"family":family_of(arch),"arch_id":arch,"optimizer":optname,"activation":act,"seed_index":seed})
                    model=build_model(arch,act,output_dim_for(ds),ds,seed);calibrate_initial_signal(model,Xtr);opt=make_optimizer(model,optname);diverged=False
                    for epoch in range(1,PREFLIGHT_EPOCHS+1):
                        model.train()
                        for ids in deterministic_batches(len(Xs),epoch-1,seed,BATCH_SIZE):
                            xb=Xs[ids].to(DEVICE);yb=ys[ids].to(DEVICE);opt.zero_grad(set_to_none=True);pred=model(xb);loss=loss_batch(pred,yb,task_for(ds))
                            if not torch.isfinite(loss):diverged=True;break
                            loss.backward();opt.step()
                        if diverged:break
                    perf=evaluate_performance(model,Xv[:min(1024,len(Xv))],yv[:min(1024,len(yv))],task_for(ds)) if not diverged else {"loss":np.inf}
                    for env in ENVIRONMENTS:
                        cond=endpoint_state(model,row,Xt,yt,env,d["meta"],ep)
                        rich,ldf=activation_richness_diagnostics(model,Xep,env,ds,d["meta"])
                        stress_rows.append(_preflight_row(ds,arch,optname,act,seed,"stress_5epoch",env,cond,rich,perf,diverged))
                        for _,lr in ldf.iterrows():layer_rows.append({"stage":"stress_5epoch","dataset":ds,"family":family_of(arch),"arch_id":arch,"optimizer":optname,"activation":act,"seed_index":seed,"environment":env,**lr.to_dict()})
                        metric_rows.append({"stage":"stress_5epoch","dataset":ds,"family":family_of(arch),"arch_id":arch,"optimizer":optname,"activation":act,"seed_index":seed,"environment":env,**flow_metric_preflight_diagnostics(cond.get("state"),family_of(arch))})
                    del model,opt
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                    bar.update(1)
    bar.close()

    init_df=pd.DataFrame(init_rows);stress_df=pd.DataFrame(stress_rows);layers_df=pd.DataFrame(layer_rows);metric_df=pd.DataFrame(metric_rows)
    gates=_preflight_gate_rows(init_df,stress_df)
    atomic_csv(init_df,init_path);atomic_csv(stress_df,stress_path);atomic_csv(layers_df,layer_path);atomic_csv(metric_df,metric_path);atomic_csv(gates,gate_path)

    # C. Tiny BCANw sentinel. It deliberately uses new 5-epoch models, not publication checkpoints.
    sent_rows=[]
    if PREFLIGHT_RUN_BCANW_SENTINEL:
        for ds in DATASETS:
            d=data_cache[ds];Xtr,ytr=d["train"];Xt,yt=d["test"];trids=np.asarray(d["preflight_train_indices"],int);Xs=Xtr[trids];ys=ytr[trids];ep=np.asarray(d["endpoint_indices"],int)
            for arch in PREFLIGHT_BCANW_ARCH_IDS:
                act=PREFLIGHT_BCANW_ACTIVATION;optname=PREFLIGHT_BCANW_OPTIMIZER;seed=PREFLIGHT_SEED
                row=pd.Series({"dataset":ds,"task":task_for(ds),"family":family_of(arch),"arch_id":arch,"optimizer":optname,"activation":act,"seed_index":seed})
                model=build_model(arch,act,output_dim_for(ds),ds,seed);calibrate_initial_signal(model,Xtr);opt=make_optimizer(model,optname)
                for epoch in range(1,PREFLIGHT_EPOCHS+1):
                    model.train()
                    for ids in deterministic_batches(len(Xs),epoch-1,seed,BATCH_SIZE):
                        xb=Xs[ids].to(DEVICE);yb=ys[ids].to(DEVICE);opt.zero_grad(set_to_none=True);loss=loss_batch(model(xb),yb,task_for(ds));loss.backward();opt.step()
                for env in ENVIRONMENTS:
                    cond=endpoint_state(model,row,Xt,yt,env,d["meta"],ep);state=cond.get("state")
                    base={"dataset":ds,"family":family_of(arch),"arch_id":arch,"optimizer":optname,"activation":act,"seed_index":seed,"environment":env,"realization_defined":state is not None,"valid_fraction":float(cond.get("validity",{}).get("valid_fraction",0.0))}
                    if state is None: sent_rows.append({**base,"geometry_computed":False,"reason":"zero_complete_path_support"});continue
                    gs,_,_,_=analyze_geometry(model,state,family_of(arch),PREFLIGHT_BCANW_SCALES,compute_local=False)
                    sent_rows.append({**base,"geometry_computed":not bool(gs.get("geometry_skipped",False)),"geometry_skip_reason":gs.get("geometry_skip_reason",""),"xi50":gs.get("xi50",np.nan),"xi50_bracketed":gs.get("xi50_bracketed",False),"d_star":gs.get("d_star",np.nan),"transition_fraction":gs.get("transition_fraction",np.nan),"cover_monotonicity_corrections":gs.get("cover_monotonicity_corrections",0)})
                del model,opt
                if torch.cuda.is_available():torch.cuda.empty_cache()
    sentinel_df=pd.DataFrame(sent_rows);atomic_csv(sentinel_df,sentinel_path)

    fatal=gates[gates.gate_severity.astype(str)=="fatal"].copy() if len(gates) else pd.DataFrame()
    warnings_df=gates[gates.gate_severity.astype(str)=="warning"].copy() if len(gates) else pd.DataFrame()
    passes=gates[gates.gate_severity.astype(str)=="pass"].copy() if len(gates) else pd.DataFrame()
    lines=["RICH-POPULATION PREFLIGHT — SLURM POLICY",
           f"Initialization states: {len(init_df)}",f"Stress states: {len(stress_df)}",
           f"Pass states: {len(passes)}",f"Warning states: {len(warnings_df)}",f"Fatal states: {len(fatal)}",
           f"BCANw sentinel states: {len(sentinel_df)}"]
    if len(warnings_df):
        lines.append("Warning condition signatures (campaign continues):\n"+warnings_df[["stage","dataset","arch_id","optimizer","activation","environment","valid_fraction","final_hidden_relu_active_q01","gate_reasons"]].to_string(index=False))
    if len(fatal):
        lines.append("FATAL condition signatures:\n"+fatal[["stage","dataset","arch_id","optimizer","activation","environment","valid_fraction","gate_reasons"]].to_string(index=False))
    (OUTDIR/"preflight_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n"+"\n".join(lines))

    # HPC behavior: never sacrifice a long Slurm allocation for a merely conservative ReLU richness threshold.
    # Abort only on genuine numerical/theoretical pathology. ReLU warnings can be promoted to fatal explicitly.
    if len(fatal):
        raise RuntimeError(f"Rich-population preflight found {len(fatal)} FATAL state(s). Inspect {gate_path} and preflight_summary.txt.")
    if len(warnings_df) and (PREFLIGHT_STRICT or PREFLIGHT_ABORT_ON_RELU_WARNING):
        raise RuntimeError(f"Rich-population preflight found {len(warnings_df)} ReLU richness warning state(s) and strict warning-abort mode is enabled. Inspect {gate_path}.")
    if len(warnings_df):
        print(f"[preflight] {len(warnings_df)} ReLU richness warning state(s) recorded; continuing the 324-network Slurm campaign.")
    else:
        print("[preflight] all state-level gates passed; continuing the 324-network Slurm campaign.")
    return {"initialization":init_df,"stress":stress_df,"layers":layers_df,"metric":metric_df,"gates":gates,"sentinel":sentinel_df}

# ==================================================================================================
# 8. ENDPOINT ANALYSIS / RUN MANIFEST
# ==================================================================================================

def family_of(arch):return "mlp" if arch in MLP_ARCHS else "cnn"


def output_dim_for(ds):return 10 if ds=="fmnist" else 1

def task_for(ds):return "classification" if ds=="fmnist" else "regression"


def build_manifest()->pd.DataFrame:
    rows=[]
    for ds in DATASETS:
        for arch in ARCH_IDS:
            for opt in OPTIMIZERS:
                for act in ACTIVATIONS:
                    for seed in SEEDS:
                        rid=f"BENCH__{ds}__{arch}__{opt}__{act}__S{seed:02d}"
                        rows.append({"run_id":rid,"dataset":ds,"task":task_for(ds),"family":family_of(arch),"arch_id":arch,"optimizer":opt,"activation":act,"seed_index":seed})
    df=pd.DataFrame(rows);atomic_csv(df,OUTDIR/"tables"/"run_manifest.csv");return df


def part_path(rid,kind):return OUTDIR/"run_parts"/f"{rid}__{kind}.csv"
def checkpoint_path(rid):return OUTDIR/"checkpoints"/f"{rid}__best.pt"

def run_complete(rid):
    return checkpoint_path(rid).exists() and all(part_path(rid,k).exists() for k in ["summary","history","path_summary","path_validity","cnn_layers"])


def select_geometry_row(row:pd.Series)->bool:
    if int(row.seed_index) not in GEOM_SEEDS:return False
    return (str(row.optimizer)==GEOM_ACTIVATION_PANEL_OPT) or (str(row.activation)==GEOM_OPTIMIZER_PANEL_ACT)


@torch.no_grad()
def endpoint_state(model,row,X,y,env,dataset_meta,endpoint_indices=None):
    task=str(row.task);family=str(row.family)
    if endpoint_indices is not None:
        ids=np.asarray(endpoint_indices,dtype=int); X=X[ids]; y=y[ids]
    n_use=len(X)
    Xuse=X if env=="clean" else shifted_environment(X,str(row.dataset),dataset_meta)
    if family=="mlp":return mlp_population_state(model,Xuse,y,task,n_use)
    return cnn_population_state(model,Xuse,y,task,n_use)


@torch.no_grad()
def structural_state(model,row):
    if row.family=="mlp":return mlp_structural_population(model,output_dim_for(row.dataset),row.task)
    return cnn_structural_population(model,output_dim_for(row.dataset),row.task)


def _aggregate_cnn_layer_summary(obj:Dict[str,Any], state_name:str, meta:Dict[str,Any])->Optional[pd.DataFrame]:
    """Return one clean layer-level summary, or None when no valid layer rows exist.

    A zero-valid conditional population is a legitimate outcome of the finite-response
    measure. It must not crash training bookkeeping. Only scientific layer metrics are
    aggregated; bookkeeping columns such as input_index/target are deliberately excluded.
    """
    d=obj.get("layer_df", None) if isinstance(obj,dict) else None
    if not isinstance(d,pd.DataFrame) or d.empty or "layer" not in d.columns:
        return None
    metric_cols=[c for c in [
        "S_occ","S_param","S_reuse",
        "eta1_occ","eta1_param","eta1_reuse",
        "reuse_multiplicity",
    ] if c in d.columns]
    if not metric_cols:
        return None
    x=d[["layer"]+metric_cols].copy()
    for c in metric_cols:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["layer"])
    if x.empty:
        return None
    out=x.groupby("layer",as_index=False)[metric_cols].mean()
    out["state"]=str(state_name)
    for k,v in meta.items():
        out[k]=v
    return out


def train_one(row:pd.Series,data:Dict[str,Any]):
    """Train one model, then compute structural/conditional endpoint states.

    Fault-tolerant ordering is deliberate: the best checkpoint and training tables are
    persisted *before* endpoint/path analysis. If endpoint analysis fails, a resumed run
    reloads the saved network and performs analysis only, rather than retraining it.
    """
    rid=str(row.run_id)
    if RESUME and run_complete(rid):
        return

    Xtr,ytr=data["train"]; Xv,yv=data["val"]; Xt,yt=data["test"]
    cp=checkpoint_path(rid)
    model=build_model(row.arch_id,row.activation,output_dim_for(row.dataset),row.dataset,int(row.seed_index))

    # -----------------------------
    # Training or checkpoint resume
    # -----------------------------
    if RESUME and cp.exists():
        chk=torch.load(cp,map_location="cpu")
        model.load_state_dict(chk["state_dict"])
        model.to(DEVICE); model.eval()
        meta=dict(chk.get("meta",{}))
        # Ensure mandatory identifiers are present even for an older compatible checkpoint.
        meta.update({
            "run_id":rid,"dataset":row.dataset,"task":row.task,"family":row.family,
            "arch_id":row.arch_id,"optimizer":row.optimizer,"activation":row.activation,
            "seed_index":int(row.seed_index),"parameter_count":model.parameter_count(),
        })
        hist_df=safe_read_csv(part_path(rid,"history"))
        hist=hist_df.to_dict("records") if not hist_df.empty else []
    else:
        cal_scales=calibrate_initial_signal(model,Xtr)
        opt=make_optimizer(model,row.optimizer); task=row.task
        init=evaluate_performance(model,Xv,yv,task)
        best_metric=init["loss"]; best_epoch=0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        hist=[]; diverged=False
        for epoch in range(1,MAX_EPOCH+1):
            model.train(); losses=[]
            for ids in deterministic_batches(len(Xtr),epoch-1,int(row.seed_index),BATCH_SIZE):
                xb=Xtr[ids].to(DEVICE); yb=ytr[ids].to(DEVICE)
                opt.zero_grad(set_to_none=True); pred=model(xb); loss=loss_batch(pred,yb,task)
                if not torch.isfinite(loss):
                    diverged=True; break
                loss.backward(); opt.step(); losses.append(float(loss.item()))
            if diverged: break
            val=evaluate_performance(model,Xv,yv,task)
            hist.append({"run_id":rid,"epoch":epoch,"train_loss":float(np.mean(losses)),
                         "val_loss":val["loss"],"val_accuracy":val.get("accuracy",np.nan)})
            if np.isfinite(val["loss"]) and val["loss"]<best_metric:
                best_metric=val["loss"]; best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if not np.isfinite(val["loss"]) or val["loss"]>1e10:
                diverged=True; break

        model.load_state_dict(best_state); model.to(DEVICE); model.eval()
        ptest=evaluate_performance(model,Xt,yt,task)
        meta={
            "run_id":rid,"dataset":row.dataset,"task":task,"family":row.family,
            "arch_id":row.arch_id,"optimizer":row.optimizer,"activation":row.activation,
            "seed_index":int(row.seed_index),"parameter_count":model.parameter_count(),
            "best_epoch":best_epoch,"best_val_loss":best_metric,
            "best_test_loss":ptest["loss"],"best_test_accuracy":ptest.get("accuracy",np.nan),
            "diverged":diverged,
            "calibration_scales_json":json.dumps([float(x) for x in cal_scales]),
        }

        # Persist the trained state BEFORE any endpoint/path analysis.
        torch.save({"state_dict":best_state,"meta":meta},cp)
        atomic_csv(pd.DataFrame([meta]),part_path(rid,"summary"))
        atomic_csv(pd.DataFrame(hist),part_path(rid,"history"))
        del opt

    # If resuming from a checkpoint that predates the fault-tolerant ordering, backfill
    # lightweight training tables before endpoint analysis.
    if not part_path(rid,"summary").exists():
        atomic_csv(pd.DataFrame([meta]),part_path(rid,"summary"))
    if not part_path(rid,"history").exists():
        atomic_csv(pd.DataFrame(hist),part_path(rid,"history"))

    # -----------------------------
    # Endpoint/path-state analysis
    # -----------------------------
    structural=structural_state(model,row)
    # Keep primary realization tables task-comparable and free of irrelevant task-specific missing fields.
    path_meta={k:meta[k] for k in ["run_id","dataset","task","family","arch_id","optimizer","activation","seed_index","parameter_count","best_epoch","best_val_loss","best_test_loss"] if k in meta}
    path_rows=[{**path_meta,"state":"structural",**{f"path_{k}_median":v for k,v in structural["summary"].items()}}]
    validity_rows=[{**path_meta,"state":"structural","n_inputs_evaluated":1,"n_valid_inputs":1,"valid_fraction":1.0,
                    "realization_defined":True,"degenerate":False,"status":"defined_structural","reason":""}]
    cnn_layer_rows=[]
    if row.family=="cnn":
        d=_aggregate_cnn_layer_summary(structural,"structural",meta)
        if d is not None: cnn_layer_rows.append(d)

    for env in ENVIRONMENTS:
        cond=endpoint_state(model,row,Xt,yt,env,data["meta"],data.get("endpoint_indices"))
        v=dict(cond.get("validity",{}))
        validity_rows.append({**path_meta,"state":env,**v})
        if cond.get("state") is not None:
            # A numerical row is written only when a normalized conditional measure exists.
            obs_summary={k:val for k,val in cond["summary"].items() if k!="conditional_valid_fraction"}
            prow={**path_meta,"state":env,**{f"path_{k}":val for k,val in obs_summary.items()}}
            metric_vals=[prow.get(m) for m in ["path_eta1_median","path_eta2_median","path_entropy_deficit_norm_median"]]
            if not all(x is not None and np.isfinite(float(x)) for x in metric_vals):
                raise RuntimeError(f"Defined realization produced non-finite primary path observables: run={rid}, state={env}, values={metric_vals}")
            path_rows.append(prow)
        if row.family=="cnn":
            d=_aggregate_cnn_layer_summary(cond,env,meta)
            if d is not None: cnn_layer_rows.append(d)

    atomic_csv(pd.DataFrame(path_rows),part_path(rid,"path_summary"))
    atomic_csv(pd.DataFrame(validity_rows),part_path(rid,"path_validity"))
    atomic_csv(pd.concat(cnn_layer_rows,ignore_index=True) if cnn_layer_rows else pd.DataFrame(),part_path(rid,"cnn_layers"))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ==================================================================================================
# 9. POPULATION / INPUT GEOMETRY BACKFILL
# ==================================================================================================

def geometry_done(rid):return all(part_path(rid,k).exists() for k in ["geometry","geometry_validity","geometry_scales","geometry_compressed"])


def run_geometry_panel(manifest:pd.DataFrame,data_cache:Dict[str,Any]):
    if not RUN_BCANW:return
    sub=manifest[manifest.apply(select_geometry_row,axis=1)].copy();todo=[r for _,r in sub.iterrows() if not(RESUME and geometry_done(str(r.run_id)))]
    eta=ETAReporter("population BCANw",len(todo))
    for k,row in enumerate(tqdm(todo,desc="Benchmark population BCANw"),1):
        t0=time.perf_counter();rid=str(row.run_id);cp=checkpoint_path(rid)
        if not cp.exists():continue
        data=data_cache[row.dataset];Xt,yt=data["test"];model=build_model(row.arch_id,row.activation,output_dim_for(row.dataset),row.dataset,int(row.seed_index));model.load_state_dict(torch.load(cp,map_location="cpu")["state_dict"]);model.to(DEVICE);model.eval();structural=structural_state(model,row)
        gsum=[];gvalid=[];gsc=[];gcomp=[];glocal=[]
        states=[("structural",structural["state"],{"valid_fraction":1.0,"realization_defined":True,"status":"defined_structural"})]
        for env in ENVIRONMENTS:
            cond=endpoint_state(model,row,Xt,yt,env,data["meta"],data.get("endpoint_indices"));states.append((env,cond.get("state"),dict(cond.get("validity",{}))))
        for source,state,rvalid in states:
            valid_fraction=float(rvalid.get("valid_fraction",1.0 if source=="structural" else 0.0))
            realization_defined=bool(rvalid.get("realization_defined",state is not None))
            base={"run_id":rid,"dataset":row.dataset,"task":row.task,"family":row.family,"arch_id":row.arch_id,"optimizer":row.optimizer,"activation":row.activation,"seed_index":int(row.seed_index),"geometry_source":source,"conditional_valid_fraction":valid_fraction}
            vbase={**base,"realization_defined":realization_defined}
            if state is None:
                gvalid.append({**vbase,"geometry_computed":False,"geometry_status":"not_computed_no_realization","geometry_skip_reason":"zero_complete_path_support","n_graph_nodes":0,
                               "xi50_bracketed":False,"deff_supported":False,"local_geometry_supported":False,"quotient_ks_defined":False,"retention_defined":False,"scaling_window_defined":False,"scaling_credible":False,
                               "transition_fraction":0.0,"max_plateau_span_decades":0.0,"cover_monotonicity_corrections":0,"max_upward_cover_fraction_step":0.0})
                continue
            gs,sc,comp,local=analyze_geometry(model,state,row.family,BCANW_SCALES,compute_local=True)
            if bool(gs.get("geometry_skipped",False)):
                gvalid.append({**vbase,"geometry_computed":False,"geometry_status":"skipped_geometry_constraint","geometry_skip_reason":str(gs.get("geometry_skip_reason","")),
                               "n_graph_nodes":int(gs.get("n_graph_nodes",0)),"xi50_bracketed":False,"deff_supported":False,"local_geometry_supported":False,"quotient_ks_defined":False,"retention_defined":False,"scaling_window_defined":False,"scaling_credible":False,
                               "transition_fraction":0.0,"max_plateau_span_decades":0.0,"cover_monotonicity_corrections":0,"max_upward_cover_fraction_step":0.0})
                continue
            clean_gs={k:v for k,v in gs.items() if k not in ["geometry_skipped","geometry_skip_reason","xi50_bracketed","scaling_credible"]}
            gsum.append({**base,**clean_gs})
            xi_ok=bool(gs.get("xi50_bracketed",False));deff_ok=bool(np.isfinite(gs.get("d_star",np.nan)));local_ok=bool(np.isfinite(gs.get("local_delta_median",np.nan)))
            status="computed" if xi_ok and deff_ok else ("computed_xi50_unbracketed" if not xi_ok else "computed_no_valid_deff")
            gvalid.append({**vbase,"geometry_computed":True,"geometry_status":status,"geometry_skip_reason":"","n_graph_nodes":int(gs.get("n_graph_nodes",0)),
                           "xi50_bracketed":xi_ok,"deff_supported":deff_ok,"local_geometry_supported":local_ok,
                           "quotient_ks_defined":bool(np.isfinite(gs.get("quotient_mean_pairwise_KS",np.nan))),"retention_defined":bool(np.isfinite(gs.get("renorm_flow_retention_mean",np.nan))),
                           "scaling_window_defined":bool(np.isfinite(gs.get("scaling_dB",np.nan))),"scaling_credible":bool(gs.get("scaling_credible",False)),
                           "transition_fraction":float(gs.get("transition_fraction",0.0)),"max_plateau_span_decades":float(gs.get("max_plateau_span_decades",0.0)) if np.isfinite(gs.get("max_plateau_span_decades",np.nan)) else 0.0,
                           "cover_monotonicity_corrections":int(gs.get("cover_monotonicity_corrections",0)),"max_upward_cover_fraction_step":float(gs.get("max_upward_cover_fraction_step",0.0))})
            if len(sc):
                for q,v in base.items():sc[q]=v
                gsc.append(sc)
            if len(comp):
                for q,v in base.items():comp[q]=v
                gcomp.append(comp)
            if len(local):
                for q,v in base.items():local[q]=v
                glocal.append(local)
        atomic_csv(pd.DataFrame(gsum),part_path(rid,"geometry"));atomic_csv(pd.DataFrame(gvalid),part_path(rid,"geometry_validity"));atomic_csv(pd.concat(gsc,ignore_index=True) if gsc else pd.DataFrame(),part_path(rid,"geometry_scales"));atomic_csv(pd.concat(gcomp,ignore_index=True) if gcomp else pd.DataFrame(),part_path(rid,"geometry_compressed"));
        if glocal:
            pd.concat(glocal,ignore_index=True).to_csv(part_path(rid,"local_geometry").with_suffix(".csv.gz"),index=False,compression="gzip")
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        eta.update(time.perf_counter()-t0,k)
    eta.finish()


def run_input_geometry(manifest:pd.DataFrame,data_cache:Dict[str,Any])->pd.DataFrame:
    """Compute only defined input-resolved geometries; record every attempted input in a validity table."""
    path=OUTDIR/"tables"/"input_resolved_geometry.csv"
    vpath=OUTDIR/"tables"/"input_geometry_validity.csv"
    if not RUN_INPUT_GEOM:
        return pd.DataFrame()
    existing=safe_read_csv(path) if RESUME else pd.DataFrame()
    existing_v=safe_read_csv(vpath) if RESUME else pd.DataFrame()
    # Migrate defensively from older versions that stored invalid attempts in the primary table.
    if not existing.empty and "valid" in existing.columns:
        old_invalid=existing[~existing["valid"].astype(bool)].copy()
        existing=existing[existing["valid"].astype(bool)].copy()
        if not old_invalid.empty and existing_v.empty:
            migrated=[]
            for _,r in old_invalid.iterrows():
                migrated.append({k:r.get(k) for k in ["run_id","dataset","family","arch_id","optimizer","activation","seed_index","environment","input_index"]} | {
                    "realization_defined":False,"geometry_computed":False,"status":"not_computed_no_realization","reason":"legacy_invalid_conditional_path_state"})
            existing_v=pd.DataFrame(migrated)
    rows=[] if existing.empty else existing.to_dict("records")
    vrows=[] if existing_v.empty else existing_v.to_dict("records")
    done={(str(r["run_id"]),str(r["environment"]),int(r["input_index"])) for r in vrows if "run_id" in r and "environment" in r and "input_index" in r}
    if not done:
        done={(str(r["run_id"]),str(r["environment"]),int(r["input_index"])) for r in rows if "run_id" in r and "environment" in r and "input_index" in r}
    sub=manifest[(manifest.arch_id.isin(INPUT_GEOM_ARCH_IDS))&(manifest.optimizer==INPUT_GEOM_OPTIMIZER)&(manifest.seed_index.isin(INPUT_GEOM_SEEDS))]
    total=sum(1 for _,r in sub.iterrows() for env in ENVIRONMENTS for i in range(INPUT_GEOM_INPUTS) if (str(r.run_id),env,i) not in done)
    eta=ETAReporter("input BCANw",total);c=0
    for _,row in sub.iterrows():
        cp=checkpoint_path(row.run_id)
        if not cp.exists():continue
        model=build_model(row.arch_id,row.activation,output_dim_for(row.dataset),row.dataset,int(row.seed_index));model.load_state_dict(torch.load(cp,map_location="cpu")["state_dict"]);model.to(DEVICE);model.eval();data=data_cache[row.dataset];Xt,yt=data["test"]
        ep=np.asarray(data.get("endpoint_indices",np.arange(len(Xt))),dtype=int); Xt=Xt[ep]; yt=yt[ep]
        for env in ENVIRONMENTS:
            Xenv=Xt if env=="clean" else shifted_environment(Xt,row.dataset,data["meta"])
            for i in range(min(INPUT_GEOM_INPUTS,len(Xenv))):
                key=(str(row.run_id),env,i)
                if key in done:continue
                t0=time.perf_counter();x=Xenv[i:i+1];y=yt[i:i+1]
                if row.family=="mlp":
                    resp=mlp_response_sq(model,x,torch.zeros_like(x));target=y.long() if row.task=="classification" else torch.zeros(1,dtype=torch.long);st=mlp_factor_state(model,torch.square(x.reshape(1,-1).to(torch.float64)),resp,target)
                    state={"edge_flow":[e[0] for e in st["edge_flow"]],"node_flow":[v[0] for v in st["node_flow"]]} if bool(st["valid"][0]) else None
                else:
                    resp=cnn_response_sq(model,x,torch.zeros_like(x));target=y.long() if row.task=="classification" else torch.zeros(1,dtype=torch.long);st=cnn_factor_state(model,torch.square(x.to(torch.float64)),resp,target)
                    state={"occurrence_flow":[e[0] for e in st["occurrence_flow"]],"parameter_flow":[p[0] for p in st["parameter_flow"]],"node_flow":[v[0] for v in st["node_flow"]]} if bool(st["valid"][0]) else None
                base={"run_id":row.run_id,"dataset":row.dataset,"family":row.family,"arch_id":row.arch_id,"optimizer":row.optimizer,"activation":row.activation,"seed_index":int(row.seed_index),"environment":env,"input_index":i}
                if state is None:
                    vrows.append({**base,"realization_defined":False,"geometry_computed":False,"status":"not_computed_no_realization","reason":"zero_complete_path_support","xi50_bracketed":False,"deff_supported":False})
                else:
                    gs,_,_,_=analyze_geometry(model,state,row.family,INPUT_GEOM_SCALES,compute_local=False)
                    if bool(gs.get("geometry_skipped",False)):
                        vrows.append({**base,"realization_defined":True,"geometry_computed":False,"status":"skipped_geometry_constraint","reason":str(gs.get("geometry_skip_reason","")),"xi50_bracketed":False,"deff_supported":False})
                    else:
                        clean_gs={k:v for k,v in gs.items() if k not in ["geometry_skipped","geometry_skip_reason","xi50_bracketed","scaling_credible"]}
                        rows.append({**base,**clean_gs})
                        vrows.append({**base,"realization_defined":True,"geometry_computed":True,"status":"computed" if bool(gs.get("xi50_bracketed",False)) else "computed_xi50_unbracketed","reason":"",
                                      "xi50_bracketed":bool(gs.get("xi50_bracketed",False)),"deff_supported":bool(np.isfinite(gs.get("d_star",np.nan)))})
                done.add(key);atomic_csv(pd.DataFrame(rows),path);atomic_csv(pd.DataFrame(vrows),vpath);c+=1;eta.update(time.perf_counter()-t0,c)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    eta.finish();return pd.DataFrame(rows)


# ==================================================================================================
# 10. AGGREGATION / STATISTICS
# ==================================================================================================

def _geometry_observables_long(g:pd.DataFrame)->pd.DataFrame:
    """Finite-only long-form population geometry table for scientific export.

    Undefined derived quantities are absent observations, not NaN-valued measurements.
    The corresponding state/metric availability is recorded in geometry_validity.csv.
    """
    if g is None or g.empty:return pd.DataFrame()
    meta_cols=[c for c in ["run_id","dataset","task","family","arch_id","optimizer","activation","seed_index","geometry_source","conditional_valid_fraction"] if c in g.columns]
    excluded=set(meta_cols)|{"geometry_skipped","geometry_skip_reason","xi50_bracketed","scaling_credible"}
    value_cols=[c for c in g.columns if c not in excluded]
    rows=[]
    for _,r in g.iterrows():
        base={c:r[c] for c in meta_cols}
        for metric in value_cols:
            v=r[metric]
            if isinstance(v,(bool,np.bool_)):
                rows.append({**base,"metric":metric,"value":float(bool(v))})
            else:
                try:
                    fv=float(v)
                except Exception:
                    continue
                if np.isfinite(fv):rows.append({**base,"metric":metric,"value":fv})
    return pd.DataFrame(rows)


def aggregate_parts(manifest:pd.DataFrame):
    kinds={k:[] for k in ["summary","history","path_summary","path_validity","cnn_layers","geometry","geometry_validity","geometry_scales","geometry_compressed"]};local=[]
    for rid in manifest.run_id:
        for k in kinds:
            d=safe_read_csv(part_path(rid,k))
            if len(d):kinds[k].append(d)
        lp=part_path(rid,"local_geometry").with_suffix(".csv.gz")
        if lp.exists():
            try:local.append(pd.read_csv(lp))
            except Exception:pass
    out={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in kinds.items()};out["local_geometry"]=pd.concat(local,ignore_index=True) if local else pd.DataFrame()
    # Primary observable tables contain defined measurements only.
    atomic_csv(out["summary"],OUTDIR/"tables"/"run_summary.csv")
    atomic_csv(out["history"],OUTDIR/"tables"/"training_history.csv")
    atomic_csv(out["path_summary"],OUTDIR/"tables"/"path_summary.csv")
    atomic_csv(out["path_validity"],OUTDIR/"tables"/"path_realization_validity.csv")
    atomic_csv(out["cnn_layers"],OUTDIR/"tables"/"cnn_layer_decomposition.csv")
    out["geometry_observables"]=_geometry_observables_long(out["geometry"])
    atomic_csv(out["geometry_observables"],OUTDIR/"tables"/"population_geometry.csv")
    atomic_csv(out["geometry_validity"],OUTDIR/"tables"/"geometry_validity.csv")
    atomic_csv(out["geometry_scales"],OUTDIR/"tables"/"population_geometry_scales.csv")
    atomic_csv(out["geometry_compressed"],OUTDIR/"tables"/"population_geometry_compressed.csv")
    if len(out["local_geometry"]):out["local_geometry"].to_csv(OUTDIR/"tables"/"local_geometry_nodes.csv.gz",index=False,compression="gzip")
    return out


PATH_METRICS=["path_eta1_median","path_eta2_median","path_entropy_deficit_norm_median"]

def validate_realization_tables(pathdf:pd.DataFrame,validity:pd.DataFrame,manifest:pd.DataFrame):
    """Validate existence/measurement separation before expensive geometry.

    Hard failures are reserved for programming/data-integrity errors. Legitimate degenerate
    realizations are represented only in the validity table and are therefore not failures.
    """
    req_path=["run_id","state",*PATH_METRICS]
    miss=[c for c in req_path if c not in pathdf.columns]
    if miss:raise RuntimeError(f"Path summary schema mismatch; missing columns: {miss}")
    req_v=["run_id","state","realization_defined","valid_fraction","status","reason"]
    miss=[c for c in req_v if c not in validity.columns]
    if miss:raise RuntimeError(f"Path validity schema mismatch; missing columns: {miss}")
    # Every trained run must have structural + clean + shifted validity records exactly once.
    expected={(str(rid),st) for rid in manifest.run_id.astype(str) for st in ["structural",*ENVIRONMENTS]}
    got=[(str(r.run_id),str(r.state)) for _,r in validity.iterrows()]
    if len(got)!=len(set(got)):raise RuntimeError("Duplicate run/state records in path_realization_validity")
    missing=expected-set(got);extra=set(got)-expected
    if missing or extra:raise RuntimeError(f"Path validity coverage mismatch; missing={len(missing)}, extra={len(extra)}")
    # Observable rows are exactly the defined realizations, with finite primary metrics.
    defined={(str(r.run_id),str(r.state)) for _,r in validity.iterrows() if bool(r.realization_defined)}
    measured={(str(r.run_id),str(r.state)) for _,r in pathdf.iterrows()}
    if defined!=measured:
        raise RuntimeError(f"Defined-realization/path-observation mismatch; defined_without_measure={len(defined-measured)}, measure_without_defined={len(measured-defined)}")
    for m in PATH_METRICS:
        vals=pd.to_numeric(pathdf[m],errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(vals)):raise RuntimeError(f"Primary path table contains non-finite values in {m}; undefined realizations must live only in the validity table")
    structural=validity[validity.state.astype(str)=="structural"]
    if not structural.realization_defined.astype(bool).all():raise RuntimeError("Structural disposition unexpectedly marked undefined")
    print(f"[realization tables] {len(pathdf)} defined observable states; {(~validity.realization_defined.astype(bool)).sum()} degenerate states recorded separately")


def realization_validity_tests(v:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    """Treat conditional support itself as a response variable rather than hidden missingness."""
    if v is None or v.empty:return pd.DataFrame(),pd.DataFrame()
    d=v[v.state.astype(str).isin(ENVIRONMENTS)].copy();d["valid_fraction"]=pd.to_numeric(d.valid_fraction,errors="coerce")
    summary=d.groupby(["dataset","family","arch_id","optimizer","activation","state"],dropna=False).agg(
        n_runs=("run_id","nunique"),median_valid_fraction=("valid_fraction","median"),min_valid_fraction=("valid_fraction","min"),degenerate_rate=("degenerate","mean")
    ).reset_index()
    atomic_csv(summary,OUTDIR/"tables"/"realization_validity_summary.csv")
    rows=[]
    # Environment is an exact within-network contrast.
    piv=d.pivot_table(index=["run_id","dataset","family","arch_id","optimizer","activation","seed_index"],columns="state",values="valid_fraction",aggfunc="first")
    if all(e in piv.columns for e in ENVIRONMENTS):
        comp=piv[ENVIRONMENTS].dropna();delta=comp["shifted"].to_numpy(float)-comp["clean"].to_numpy(float)
        if len(delta):
            try:w,pv=wilcoxon(delta,zero_method="wilcox")
            except Exception:w,pv=np.nan,1.0
            lo,hi=bootstrap_median_ci(delta,seed=1701);rows.append({"factor":"environment","state":"shifted_minus_clean","n_blocks":len(delta),"median_difference":float(np.median(delta)),"hodges_lehmann":hodges_lehmann_onesample(delta),"rank_biserial":paired_rank_biserial(delta),"ci_low":lo,"ci_high":hi,"statistic":w,"p_raw":pv})
    # Activation/optimizer omnibus on support fraction within each environment.
    for state in ENVIRONMENTS:
        ds=d[d.state.astype(str)==state]
        for fac,levels,index_cols in [("activation",ACTIVATIONS,["dataset","arch_id","optimizer","seed_index"]),("optimizer",OPTIMIZERS,["dataset","arch_id","activation","seed_index"])]:
            piv=ds.pivot_table(index=index_cols,columns=fac,values="valid_fraction",aggfunc="first");lev=[x for x in levels if x in piv.columns];comp=piv[lev].dropna()
            if len(lev)>=3 and len(comp)>=3:
                try:st,pv=friedmanchisquare(*[comp[x].to_numpy(float) for x in lev]);W=float(st/(len(comp)*(len(lev)-1)))
                except Exception:st,pv,W=np.nan,np.nan,np.nan
                rows.append({"factor":fac,"state":state,"n_blocks":len(comp),"friedman_chi2":st,"kendall_W":W,"p_raw":pv})
    tests=pd.DataFrame(rows)
    if len(tests):tests["p_holm"]=holm_adjust(pd.to_numeric(tests.p_raw,errors="coerce").fillna(1.0).to_numpy(float))
    atomic_csv(tests,OUTDIR/"tables"/"realization_validity_tests.csv")
    return summary,tests


GEOM_METRICS=["xi50","d_star","lacunarity_star","local_delta_median","quotient_mean_pairwise_KS","renorm_flow_retention_mean"]


def matched_factor_tests(pathdf:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    # activation / optimizer Friedman on conditional and structural path observables.
    omnibus=[];pairs=[];envrows=[]
    for state in ["structural","clean","shifted"]:
        d0=pathdf[pathdf.state==state]
        for fac,levels,index_cols in [("activation",ACTIVATIONS,["dataset","arch_id","optimizer","seed_index"]),("optimizer",OPTIMIZERS,["dataset","arch_id","activation","seed_index"])]:
            for metric in [m for m in PATH_METRICS if m in d0.columns]:
                piv=d0.pivot_table(index=index_cols,columns=fac,values=metric,aggfunc="first");lev=[x for x in levels if x in piv.columns];comp=piv[lev].dropna()
                if len(lev)>=3 and len(comp)>=3:
                    try:st,p=friedmanchisquare(*[comp[x].to_numpy(float) for x in lev]);W=float(st/(len(comp)*(len(lev)-1)))
                    except Exception:st,p,W=np.nan,np.nan,np.nan
                    omnibus.append({"state":state,"factor":fac,"metric":metric,"n_blocks":len(comp),"friedman_chi2":st,"p_raw":p,"kendall_W":W})
                    for i in range(len(lev)):
                        for j in range(i+1,len(lev)):
                            x=comp[lev[i]].to_numpy(float);y=comp[lev[j]].to_numpy(float);delta=x-y
                            try:w,pv=wilcoxon(x,y,zero_method="wilcox")
                            except Exception:w,pv=np.nan,1.0
                            lo,hi=bootstrap_median_ci(delta,seed=100+i*10+j)
                            pairs.append({"state":state,"factor":fac,"metric":metric,"level_a":lev[i],"level_b":lev[j],"n_blocks":len(delta),"median_difference":float(np.median(delta)),"hodges_lehmann":hodges_lehmann_onesample(delta),"rank_biserial":paired_rank_biserial(delta),"ci_low":lo,"ci_high":hi,"p_raw":pv})
    om=pd.DataFrame(omnibus);pa=pd.DataFrame(pairs)
    if len(om):
        om["p_holm"]=np.nan
        for _,g in om.groupby(["state","factor"]):
            om.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    if len(pa):
        pa["p_holm"]=np.nan
        for _,g in pa.groupby(["state","factor","metric"]):pa.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    # environment paired effects: exactly same trained networks.
    for metric in [m for m in PATH_METRICS if m in pathdf.columns]:
        piv=pathdf[pathdf.state.isin(ENVIRONMENTS)].pivot_table(index=["run_id","dataset","family","arch_id","optimizer","activation","seed_index"],columns="state",values=metric,aggfunc="first").dropna()
        if all(e in piv.columns for e in ENVIRONMENTS):
            d=piv["shifted"].to_numpy(float)-piv["clean"].to_numpy(float)
            try:w,p=wilcoxon(d,zero_method="wilcox")
            except Exception:w,p=np.nan,1.0
            lo,hi=bootstrap_median_ci(d,seed=900);envrows.append({"metric":metric,"n_blocks":len(d),"median_shift_minus_clean":float(np.median(d)),"hodges_lehmann":hodges_lehmann_onesample(d),"rank_biserial":paired_rank_biserial(d),"ci_low":lo,"ci_high":hi,"p_raw":p})
    env=pd.DataFrame(envrows)
    if len(env):env["p_holm"]=holm_adjust(env.p_raw.to_numpy(float))
    atomic_csv(om,OUTDIR/"tables"/"matched_omnibus_tests.csv");atomic_csv(pa,OUTDIR/"tables"/"matched_pairwise_tests.csv");atomic_csv(env,OUTDIR/"tables"/"environment_paired_effects.csv")
    return om,pa,env



def architecture_friedman_tests(pathdf:pd.DataFrame)->pd.DataFrame:
    rows=[]
    fam_archs={"mlp":[a for a in ARCH_IDS if a in MLP_ARCHS],"cnn":[a for a in ARCH_IDS if a in CNN_ARCHS]}
    for state in ["structural","clean","shifted"]:
        for fam,levels in fam_archs.items():
            d0=pathdf[(pathdf.state==state)&(pathdf.family==fam)]
            for metric in [m for m in PATH_METRICS if m in d0.columns]:
                piv=d0.pivot_table(index=["dataset","optimizer","activation","seed_index"],columns="arch_id",values=metric,aggfunc="first");lev=[a for a in levels if a in piv.columns];comp=piv[lev].dropna()
                if len(lev)>=3 and len(comp)>=3:
                    try:st,p=friedmanchisquare(*[comp[a].to_numpy(float) for a in lev]);W=float(st/(len(comp)*(len(lev)-1)))
                    except Exception:st,p,W=np.nan,np.nan,np.nan
                    rows.append({"state":state,"family":fam,"metric":metric,"n_blocks":len(comp),"n_architectures":len(lev),"friedman_chi2":st,"p_raw":p,"kendall_W":W})
    df=pd.DataFrame(rows)
    if len(df):
        df["p_holm"]=np.nan
        for _,g in df.groupby(["state","family"]):df.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    atomic_csv(df,OUTDIR/"tables"/"architecture_friedman_tests.csv");return df


def benchmark_context_tests(pathdf:pd.DataFrame)->pd.DataFrame:
    # FMNIST and MPI3D are different tasks; these paired tests are reported only for dimensionless path observables.
    rows=[]
    for state in ["structural","clean","shifted"]:
        d0=pathdf[pathdf.state==state]
        for metric in ["path_eta1_median","path_eta2_median","path_entropy_deficit_norm_median"]:
            if metric not in d0.columns:continue
            piv=d0.pivot_table(index=["arch_id","optimizer","activation","seed_index"],columns="dataset",values=metric,aggfunc="first").dropna()
            if not all(ds in piv.columns for ds in ["fmnist","mpi3d"]):continue
            delta=piv["fmnist"].to_numpy(float)-piv["mpi3d"].to_numpy(float)
            try:w,p=wilcoxon(delta,zero_method="wilcox")
            except Exception:w,p=np.nan,1.0
            lo,hi=bootstrap_median_ci(delta,seed=1500);rows.append({"state":state,"metric":metric,"contrast":"fmnist_minus_mpi3d","n_blocks":len(delta),"median_difference":float(np.median(delta)),"hodges_lehmann":hodges_lehmann_onesample(delta),"rank_biserial":paired_rank_biserial(delta),"ci_low":lo,"ci_high":hi,"p_raw":p})
    df=pd.DataFrame(rows)
    if len(df):
        df["p_holm"]=np.nan
        for _,g in df.groupby("state"):df.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    atomic_csv(df,OUTDIR/"tables"/"benchmark_context_effects.csv");return df


def geometry_factor_tests(g:pd.DataFrame)->pd.DataFrame:
    rows=[]
    if g.empty:return pd.DataFrame()
    for source in ["structural","clean","shifted"]:
        for factor,levels,sub,index_cols in [
            ("activation",ACTIVATIONS,g[(g.geometry_source==source)&(g.optimizer==GEOM_ACTIVATION_PANEL_OPT)],["dataset","arch_id","seed_index"]),
            ("optimizer",OPTIMIZERS,g[(g.geometry_source==source)&(g.activation==GEOM_OPTIMIZER_PANEL_ACT)],["dataset","arch_id","seed_index"])
        ]:
            for metric in [m for m in GEOM_METRICS if m in sub.columns]:
                piv=sub.pivot_table(index=index_cols,columns=factor,values=metric,aggfunc="first");lev=[x for x in levels if x in piv.columns];comp=piv[lev].dropna()
                if len(lev)>=3 and len(comp)>=3:
                    try:st,p=friedmanchisquare(*[comp[x].to_numpy(float) for x in lev]);W=float(st/(len(comp)*(len(lev)-1)))
                    except Exception:st,p,W=np.nan,np.nan,np.nan
                    rows.append({"geometry_source":source,"factor":factor,"metric":metric,"n_blocks":len(comp),"friedman_chi2":st,"p_raw":p,"kendall_W":W})
    df=pd.DataFrame(rows)
    if len(df):
        df["p_holm"]=np.nan
        for _,gg in df.groupby(["geometry_source","factor"]):df.loc[gg.index,"p_holm"]=holm_adjust(gg.p_raw.to_numpy(float))
    atomic_csv(df,OUTDIR/"tables"/"geometry_factor_tests.csv");return df

def geometry_effect_tests(g:pd.DataFrame)->pd.DataFrame:
    rows=[]
    if g.empty:return pd.DataFrame()
    for metric in [m for m in GEOM_METRICS if m in g.columns]:
        piv=g[g.geometry_source.isin(ENVIRONMENTS)].pivot_table(index=["run_id","dataset","family","arch_id","optimizer","activation","seed_index"],columns="geometry_source",values=metric,aggfunc="first").dropna()
        if all(e in piv.columns for e in ENVIRONMENTS):
            d=piv["shifted"].to_numpy(float)-piv["clean"].to_numpy(float)
            if len(d):
                try:w,p=wilcoxon(d,zero_method="wilcox")
                except Exception:w,p=np.nan,1.
                lo,hi=bootstrap_median_ci(d,seed=1200);rows.append({"metric":metric,"n_blocks":len(d),"median_shift_minus_clean":float(np.median(d)),"hodges_lehmann":hodges_lehmann_onesample(d),"rank_biserial":paired_rank_biserial(d),"ci_low":lo,"ci_high":hi,"p_raw":p})
    df=pd.DataFrame(rows)
    if len(df):df["p_holm"]=holm_adjust(df.p_raw.to_numpy(float))
    atomic_csv(df,OUTDIR/"tables"/"geometry_environment_effects.csv");return df

# ==================================================================================================
# 11. PUBLICATION FIGURES
# ==================================================================================================

mpl.rcParams.update({
    "font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
    "font.size":8.8,"axes.labelsize":9.2,"axes.titlesize":10.0,"xtick.labelsize":7.6,"ytick.labelsize":7.6,
    "legend.fontsize":7.1,"axes.linewidth":.8,"xtick.direction":"in","ytick.direction":"in",
    "savefig.dpi":600,"pdf.fonttype":42,"ps.fonttype":42,
})
ACT_COLORS={a:mpl.colormaps["inferno"](v) for a,v in zip(ACTIVATIONS,[.25,.55,.82])}
OPT_MARKERS={"sgdm":"o","rmsprop":"s","adam":"^"}
STATE_STYLES={"structural":"--","clean":"-","shifted":":"}


def savefig(fig,name):
    fig.savefig(OUTDIR/"figures"/(name+".pdf"),bbox_inches="tight");fig.savefig(OUTDIR/"figures"/(name+".png"),bbox_inches="tight",dpi=600);plt.close(fig)


def make_figures(tabs:Dict[str,pd.DataFrame],input_geom:pd.DataFrame,omnibus:pd.DataFrame,pairs:pd.DataFrame,envfx:pd.DataFrame,gfx:pd.DataFrame):
    if not RUN_FIGURES:return
    summary=tabs["summary"];path=tabs["path_summary"];pvalid=tabs.get("path_validity",pd.DataFrame());layers=tabs["cnn_layers"];g=tabs["geometry"];gvalid=tabs.get("geometry_validity",pd.DataFrame());gs=tabs["geometry_scales"];local=tabs["local_geometry"]
    # 1 performance matrix
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),constrained_layout=True)
    for ax,ds in zip(axs,DATASETS):
        d=summary[summary.dataset==ds].copy();metric="best_test_accuracy" if ds=="fmnist" else "best_test_loss";p=d.groupby(["arch_id","activation"])[metric].median().unstack().reindex(index=ARCH_IDS,columns=ACTIVATIONS);im=ax.imshow(p.values,aspect="auto",cmap="inferno");ax.set_yticks(range(len(p.index)),p.index);ax.set_xticks(range(len(ACTIVATIONS)),ACTIVATIONS);ax.set_title(f"{ds.upper()} | {'accuracy' if ds=='fmnist' else 'MSE'}");plt.colorbar(im,ax=ax,fraction=.046,pad=.03)
    savefig(fig,"fig01_benchmark_performance_INFERNO")
    # 2 path concentration structural/realized
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),constrained_layout=True)
    for ax,(metric,title) in zip(axs,[("path_eta1_median",r"effective path fraction $\eta_1$"),("path_entropy_deficit_norm_median",r"normalized entropy deficit $S_{\rm def}/\log|\Gamma|$")]):
        for act in ACTIVATIONS:
            d=path[path.activation==act];x=[];y=[]
            for si,state in enumerate(["structural","clean","shifted"]):x.append(si);y.append(float(d[d.state==state][metric].median()))
            ax.plot(x,y,marker="o",color=ACT_COLORS[act],label=act)
        ax.set_xticks(range(3),["structural","clean","shifted"]);ax.set_ylabel(title)
    axs[0].legend(frameon=False);savefig(fig,"fig02_structural_to_conditional_path_response")
    if len(gs):
        # 3 BCANw master curves, all geometries, split family
        fig,axs=plt.subplots(2,3,figsize=(7.2,5.0),constrained_layout=True)
        vars=[("NB_over_N",r"$N_B/N$"),("d_eff_interp",r"$d_{\rm eff}^F$"),("lacunarity",r"$\Lambda_F$")]
        for ri,fam in enumerate(["mlp","cnn"]):
            d0=gs[gs.family==fam]
            for ci,(var,lab) in enumerate(vars):
                ax=axs[ri,ci]
                for rid,d in d0.groupby("run_id"):
                    for state in ["structural","clean","shifted"]:
                        q=d[d.geometry_source==state].sort_values("ell_over_xi50");x=q.ell_over_xi50.to_numpy(float);y=q[var].to_numpy(float);good=np.isfinite(x)&np.isfinite(y)&(x>0)
                        if np.sum(good)>=3:ax.plot(x[good],y[good],lw=.45,alpha=.12,linestyle=STATE_STYLES[state])
                ax.set_xscale("log");ax.set_xlabel(r"$x=\ell_B/\xi_{50}$");ax.set_ylabel(lab);ax.set_title(f"{fam.upper()} | {lab}")
        savefig(fig,"fig03_BCANw_master_curves_all_models")
        # 4 single all-condition d_eff heatmap, inferno, black labels
        d=gs[gs.geometry_source.isin(["clean","shifted"])].copy();curves=[];labels=[]
        grid=np.geomspace(.12,8,96)
        for keys,q in d.groupby(["dataset","family","arch_id","optimizer","activation","seed_index","geometry_source"]):
            q=q.sort_values("ell_over_xi50");x=q.ell_over_xi50.to_numpy(float);y=q.d_eff_interp.to_numpy(float);good=np.isfinite(x)&np.isfinite(y)&(x>0)
            if np.sum(good)>=3:
                yy=np.full(len(grid),np.nan);inside=(grid>=np.min(x[good]))&(grid<=np.max(x[good]));yy[inside]=np.interp(np.log(grid[inside]),np.log(x[good]),y[good]);curves.append(yy);labels.append(" | ".join(map(str,keys)))
        if curves:
            A=np.vstack(curves);fig,ax=plt.subplots(figsize=(7.2,max(5.0,.11*len(labels))),constrained_layout=True);im=ax.imshow(A,aspect="auto",cmap="inferno",interpolation="nearest",extent=[np.log10(grid[0]),np.log10(grid[-1]),len(labels),0]);ax.set_xlabel(r"$\log_{10}(\ell_B/\xi_{50})$");ax.set_ylabel("trained-model conditional states");ax.set_yticks([]);cb=plt.colorbar(im,ax=ax,fraction=.025,pad=.02);cb.set_label(r"$d_{\rm eff}^F$");savefig(fig,"fig04_all_conditions_deff_heatmap_INFERNO")
        # 5 quotient KS / retention
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
        cg=g[g.geometry_source.isin(["clean","shifted"])]
        for act in ACTIVATIONS:
            d=cg[cg.activation==act];axs[0].scatter(d.quotient_mean_pairwise_KS,d.renorm_flow_retention_mean,s=14,alpha=.5,color=ACT_COLORS[act],label=act);axs[1].scatter(d.xi50,d.lacunarity_star,s=14,alpha=.5,color=ACT_COLORS[act])
        axs[0].set_xlabel("quotient-flow mean pairwise KS");axs[0].set_ylabel("retained inter-box flow");axs[1].set_xlabel(r"$\xi_{50}$");axs[1].set_ylabel(r"$\Lambda_F^*$");axs[0].legend(frameon=False);savefig(fig,"fig05_quotient_retention_geometry")
        # 6 directed Renyi / retention profiles
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),constrained_layout=True)
        d=gs[gs.geometry_source=="clean"]
        grid=np.geomspace(.15,6,90)
        for qv in COARSE_RENYI_Q:
            curves=[]
            for _,z in d.groupby("run_id"):
                x=z.ell_over_xi50.to_numpy(float);y=z[f"coarse_eta_q{qv:g}"].to_numpy(float);good=np.isfinite(x)&np.isfinite(y)&(x>0)
                if np.sum(good)>=3:
                    inside=(grid>=np.min(x[good]))&(grid<=np.max(x[good]));yy=np.full(len(grid),np.nan);yy[inside]=np.interp(np.log(grid[inside]),np.log(x[good]),y[good]);curves.append(yy)
            if curves:axs[0].plot(grid,np.nanmedian(np.vstack(curves),0),label=f"q={qv:g}")
        curves=[]
        for _,z in d.groupby("run_id"):
            x=z.ell_over_xi50.to_numpy(float);y=z.interbox_flow_fraction.to_numpy(float);good=np.isfinite(x)&np.isfinite(y)&(x>0)
            if np.sum(good)>=3:
                inside=(grid>=np.min(x[good]))&(grid<=np.max(x[good]));yy=np.full(len(grid),np.nan);yy[inside]=np.interp(np.log(grid[inside]),np.log(x[good]),y[good]);curves.append(yy)
        if curves:axs[1].plot(grid,np.nanmedian(np.vstack(curves),0))
        for ax in axs:ax.set_xscale("log");ax.set_xlabel(r"$\ell_B/\xi_{50}$")
        axs[0].set_ylabel("coarse Renyi occupation");axs[1].set_ylabel("retained inter-box flow");axs[0].legend(frameon=False);savefig(fig,"fig06_directed_Renyi_retention_profiles")
    # 7 local geometry
    if len(local):
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),constrained_layout=True)
        for ai,act in enumerate(ACTIVATIONS):
            vals=local[(local.activation==act)&(local.geometry_source=="clean")].delta_i_F.to_numpy(float);vals=vals[np.isfinite(vals)];axs[0].boxplot(vals,positions=[ai],widths=.55,showfliers=False)
        axs[0].set_xticks(range(len(ACTIVATIONS)),ACTIVATIONS);axs[0].set_ylabel(r"$\Delta_i^F=\alpha_i^F-d_i^F$")
        q=local[local.geometry_source=="clean"];axs[1].scatter(q.d_i_F,q.alpha_i_F,s=5,alpha=.15);axs[1].set_xlabel(r"$d_i^F$");axs[1].set_ylabel(r"$\alpha_i^F$");savefig(fig,"fig07_local_geometry")
    # 8 CNN occurrence/parameter/reuse decomposition
    if len(layers):
        d=layers[layers.state.isin(["structural","clean","shifted"])];agg=d.groupby(["activation","state"])[["S_occ","S_param","S_reuse"]].median().reset_index();fig,axs=plt.subplots(1,3,figsize=(7.2,3.0),constrained_layout=True)
        for ax,metric in zip(axs,["S_occ","S_param","S_reuse"]):
            for act in ACTIVATIONS:
                q=agg[agg.activation==act].set_index("state").reindex(["structural","clean","shifted"]);ax.plot(range(3),q[metric],marker="o",color=ACT_COLORS[act],label=act)
            ax.set_xticks(range(3),["structural","clean","shifted"],rotation=20);ax.set_title(metric)
        axs[0].legend(frameon=False);savefig(fig,"fig08_CNN_occurrence_parameter_reuse")
    # 9 performance vs geometry
    if len(g):
        m=g[g.geometry_source=="clean"].merge(summary,on="run_id",suffixes=("","_s"));fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
        for ds in DATASETS:
            q=m[m.dataset==ds];perf=q.best_test_accuracy if ds=="fmnist" else -q.best_test_loss;axs[0].scatter(q.xi50,perf,s=15,alpha=.45,label=ds);axs[1].scatter(q.lacunarity_star,perf,s=15,alpha=.45,label=ds)
        axs[0].set_xlabel(r"$\xi_{50}$");axs[1].set_xlabel(r"$\Lambda_F^*$");axs[0].set_ylabel("performance (accuracy or -MSE)");axs[0].legend(frameon=False);savefig(fig,"fig09_geometry_vs_performance")
    # 10 geometry validity diagnostics live in the dedicated status table.
    if len(gvalid):
        fig,axs=plt.subplots(1,3,figsize=(7.2,2.8),constrained_layout=True);cg=gvalid[gvalid.geometry_computed.astype(bool)].copy()
        axs[0].hist(pd.to_numeric(cg.transition_fraction,errors="coerce").dropna(),bins=20);axs[0].set_xlabel("transition fraction")
        axs[1].hist(pd.to_numeric(cg.max_plateau_span_decades,errors="coerce").dropna(),bins=20);axs[1].set_xlabel("max plateau span (decades)")
        rate=cg.groupby("geometry_source").xi50_bracketed.mean().reindex(["structural","clean","shifted"]);axs[2].bar(range(len(rate)),rate.values);axs[2].set_xticks(range(len(rate)),rate.index);axs[2].set_ylabel(r"$\xi_{50}$ bracket rate");savefig(fig,"fig10_geometry_validity_diagnostics")
    # 11 conditional realization existence/support is a response variable, not hidden missingness.
    if len(pvalid):
        q=pvalid[pvalid.state.astype(str).isin(ENVIRONMENTS)].copy();q["valid_fraction"]=pd.to_numeric(q.valid_fraction,errors="coerce")
        fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
        for ai,act in enumerate(ACTIVATIONS):
            vals=q[q.activation==act].valid_fraction.to_numpy(float);vals=vals[np.isfinite(vals)];
            if len(vals):axs[0].boxplot(vals,positions=[ai],widths=.55,showfliers=False)
        axs[0].set_xticks(range(len(ACTIVATIONS)),ACTIVATIONS);axs[0].set_ylabel("conditional valid fraction");axs[0].set_ylim(-.02,1.02)
        deg=q.groupby(["activation","state"]).degenerate.mean().unstack().reindex(index=ACTIVATIONS,columns=ENVIRONMENTS).fillna(0.0)
        im=axs[1].imshow(deg.values,aspect="auto",cmap="inferno",vmin=0,vmax=max(.01,float(deg.values.max()) if deg.size else .01));axs[1].set_yticks(range(len(deg.index)),deg.index);axs[1].set_xticks(range(len(ENVIRONMENTS)),ENVIRONMENTS);axs[1].set_title("degenerate-realization rate");plt.colorbar(im,ax=axs[1],fraction=.046,pad=.03)
        savefig(fig,"fig11_conditional_realization_validity")


def make_report():
    figs=sorted((OUTDIR/"figures").glob("fig*.png"));report=OUTDIR/"benchmark_case_study_scientific_report.pdf"
    with PdfPages(report) as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.suptitle("Work II benchmark case study: FMNIST + MPI3D",fontsize=18,y=.95);fig.text(.07,.85,"324 trained networks | 3 deep MLPs + 3 deep CNNs | SGDM/RMSProp/Adam | ReLU/tanh/SiLU | 3 seeds\nClean and shifted conditional realization are evaluated on the same trained states. Structural disposition is held fixed.\nBCANw uses audited midpoint plateau compression, bracketed log-log xi50 and transition-supported d_eff.",fontsize=11,va="top",wrap=True);pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
        for p in figs:
            img=plt.imread(p);fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_subplot(111);ax.imshow(img);ax.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
    return report

# ==================================================================================================
# 12. MANIFEST / ORCHESTRATION
# ==================================================================================================

def write_experiment_manifest(manifest):
    obj={"study":"Work II benchmark case study","datasets":DATASETS,"architectures":ARCH_IDS,"optimizers":OPTIMIZERS,"activations":ACTIVATIONS,"seeds":SEEDS,"trained_networks":int(len(manifest)),"training":{"epochs":MAX_EPOCH,"n_train":N_TRAIN,"n_val":N_VAL,"n_test":N_TEST,"batch_size":BATCH_SIZE,"learning_rates":LEARNING_RATES},"environments":{"clean":"canonical test distribution","shifted":f"deterministic additive Gaussian sensor shift at {SHIFT_SNR_DB} dB in raw image space before reapplying clean-train normalization","training_axis":False},"conditional_population":{"endpoint_inputs":ENDPOINT_INPUTS,"selection":"fixed stratified balanced population shared by every model and both environments","fmnist_canonical":"16 examples per class when endpoint_inputs=160","mpi3d_canonical":"4 examples per horizontal-position level when endpoint_inputs=160"},"preflight":{"enabled":RUN_PREFLIGHT,"preflight_only":PREFLIGHT_ONLY,"epochs":PREFLIGHT_EPOCHS,"balanced_train_n":PREFLIGHT_N_TRAIN,"relu_min_valid_fraction":PREFLIGHT_RELU_MIN_VALID_FRACTION,"relu_min_final_hidden_active_q01":PREFLIGHT_RELU_MIN_FINAL_ACTIVE_Q01,"smooth_activation_required_valid_fraction":1.0,"bcanw_sentinel":PREFLIGHT_RUN_BCANW_SENTINEL},"geometry":{"scales":BCANW_SCALES,"max_nodes":BCANW_MAX_NODES,"geometry_seeds":GEOM_SEEDS,"selection":"Adam across all activations UNION ReLU across all optimizers","audited_fixes":["geometric-midpoint plateau compression","strict bracketed xi_q","log-log xi_q interpolation","transition-supported d_eff","nodewise local geometry retained"],"validity_semantics":"undefined realizations/geometric derivatives are represented in dedicated status tables, never coerced to zero or numerical path observations"}}
    atomic_json(obj,OUTDIR/"experiment_manifest.json")


def run_experiment():
    start=time.perf_counter();banner("WORK II BENCHMARK CASE STUDY — FMNIST + MPI3D — MLP/CNN CONDITIONAL REALIZATION")
    print(f"Device: {DEVICE} | profile: {PROFILE} | output: {OUTDIR.resolve()}")
    print(f"Datasets={DATASETS} | architectures={ARCH_IDS} | optimizers={OPTIMIZERS} | activations={ACTIVATIONS} | seeds={SEEDS}")
    manifest=build_manifest();write_experiment_manifest(manifest);print(f"Training factorial: {len(manifest)} networks")
    banner("Load benchmark datasets and construct fixed balanced endpoint populations")
    data_cache=load_datasets()
    for ds,d in data_cache.items(): print(f"{ds}: endpoint n={len(d.get('endpoint_indices',[]))} balance={d.get('endpoint_balance',{})}")
    banner("Rich-population support / optimizer / metric preflight")
    preflight=run_rich_population_preflight(data_cache)
    if PREFLIGHT_ONLY:
        lines=["WORK II BENCHMARK PREFLIGHT-ONLY COMPLETE",f"Device: {DEVICE}",f"Balanced endpoint populations: {ENDPOINT_INPUTS}",f"Output: {OUTDIR.resolve()}","Full 324-network training was NOT started."]
        (OUTDIR/"experiment_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8");print("\n"+"\n".join(lines))
        return {"manifest":manifest,"data":data_cache,"preflight":preflight,"preflight_only":True}
    banner("Train / resume 324-network benchmark map")
    todo=[r for _,r in manifest.iterrows() if not(RESUME and run_complete(r.run_id))];eta=ETAReporter("training",len(todo))
    for k,row in enumerate(tqdm(todo,desc="Benchmark networks"),1):
        t0=time.perf_counter();train_one(row,data_cache[row.dataset]);eta.update(time.perf_counter()-t0,k)
    eta.finish()
    banner("Aggregate and validate realization tables before geometry")
    tabs=aggregate_parts(manifest)
    validate_realization_tables(tabs["path_summary"],tabs["path_validity"],manifest)
    validity_summary,validity_tests=realization_validity_tests(tabs["path_validity"])
    banner("Predeclared population BCANw panel")
    run_geometry_panel(manifest,data_cache)
    banner("Nested input-resolved geometry sentinel")
    input_geom=run_input_geometry(manifest,data_cache)
    banner("Aggregate final tables")
    tabs=aggregate_parts(manifest)
    banner("Matched statistical tests")
    omnibus,pairs,envfx=matched_factor_tests(tabs["path_summary"]);archfx=architecture_friedman_tests(tabs["path_summary"]);benchfx=benchmark_context_tests(tabs["path_summary"]);gfx=geometry_effect_tests(tabs["geometry"]);gfac=geometry_factor_tests(tabs["geometry"])
    banner("Publication figures")
    make_figures(tabs,input_geom,omnibus,pairs,envfx,gfx);report=make_report() if RUN_FIGURES else None
    elapsed=time.perf_counter()-start
    summary=["WORK II BENCHMARK CASE STUDY COMPLETE",f"Device: {DEVICE}",f"Wall time: {human_time(elapsed)}",f"Trained networks summarized: {len(tabs['summary'])}/{len(manifest)}",f"Defined path-observable states: {len(tabs['path_summary'])}",f"Degenerate realization states: {int((~tabs['path_validity'].realization_defined.astype(bool)).sum()) if len(tabs['path_validity']) else 0}",f"Population geometry states computed: {len(tabs['geometry'])}",f"Input-resolved geometries computed: {len(input_geom)}",f"Local geometry node rows: {len(tabs['local_geometry'])}",f"Report: {report}",f"Output: {OUTDIR.resolve()}"]
    (OUTDIR/"experiment_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8");print("\n"+"\n".join(summary))
    return {"manifest":manifest,"data":data_cache,"preflight":preflight,"tables":tabs,"input_geometry":input_geom,"realization_validity_summary":validity_summary,"realization_validity_tests":validity_tests,"omnibus":omnibus,"pairwise":pairs,"environment_effects":envfx,"architecture_effects":archfx,"benchmark_context_effects":benchfx,"geometry_effects":gfx,"geometry_factor_effects":gfac,"report":report}


BENCH_RESULTS = run_experiment()
