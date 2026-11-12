from __future__ import annotations

# ==================================================================================================
# WORK II — BENCHMARK CASE STUDY
# FMNIST CLASSIFICATION + dSPRITES REGRESSION
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
#   * saturated/non-supported tails remain NaN, never painted as dimension zero
#   * nodewise d_i^F, alpha_i^F, Delta_i^F retained
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
    "PATHFLOW_BENCH_OUTDIR",
    "pathflow_work2_benchmark_FMNIST_dSprites_case_study_results"
))
for sub in ["tables", "figures", "checkpoints", "run_parts", "data"]:
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

DATASETS = ["fmnist", "dsprites"]
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
ENDPOINT_INPUTS = 128
CONDITIONAL_BATCH = 16
SHIFT_SNR_DB = 15.0

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

DSPRITES_URL = os.environ.get(
    "DSPRITES_URL",
    "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)


def _resize_images_np(x: np.ndarray, size: int) -> np.ndarray:
    t = torch.as_tensor(x[:,None,:,:], dtype=torch.float32)
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
        "meta":{"dataset":"fmnist","input_mean":mu,"input_sd":sd,"resolution":IMAGE_RESOLUTION,"split":"official train pool for train/val; official test pool for test"}
    }


def _ensure_dsprites_npz() -> Path:
    env=os.environ.get("DSPRITES_NPZ","").strip()
    if env:
        p=Path(env)
        if not p.exists(): raise FileNotFoundError(f"DSPRITES_NPZ does not exist: {p}")
        return p
    p=OUTDIR/"data"/"dsprites.npz"
    if p.exists(): return p
    print(f"Downloading dSprites to {p} ...")
    try:
        urllib.request.urlretrieve(DSPRITES_URL, p)
    except Exception as e:
        raise RuntimeError(
            "Could not download dSprites. On HPC without outbound internet, download the official NPZ once and set "
            "DSPRITES_NPZ=/path/to/dsprites_ndarray_...npz"
        ) from e
    return p


def load_dsprites() -> Dict[str,Any]:
    p=_ensure_dsprites_npz()
    z=np.load(p,allow_pickle=True)
    lat=z["latents_classes"]
    xpos=lat[:,4].astype(int)  # x-position class, 0..31
    # Stratify directly on x-position to preserve regression target support.
    tr,va,te=_split_indices_stratified(xpos,N_TRAIN,N_VAL,N_TEST,SEED+29)
    ids=np.concatenate([tr,va,te]); imgs=z["imgs"][ids].astype(np.float32)
    imgs=_resize_images_np(imgs,IMAGE_RESOLUTION)
    # Re-index local selected array.
    ntr=len(tr); nv=len(va)
    Xtr=imgs[:ntr]; Xv=imgs[ntr:ntr+nv]; Xt=imgs[ntr+nv:]
    yall=xpos[ids].astype(np.float32)
    ytr=yall[:ntr]; yv=yall[ntr:ntr+nv]; yt=yall[ntr+nv:]
    mu=float(Xtr.mean()); sd=float(Xtr.std()+1e-8)
    Xtr=((Xtr-mu)/sd).astype(np.float32); Xv=((Xv-mu)/sd).astype(np.float32); Xt=((Xt-mu)/sd).astype(np.float32)
    ym=float(ytr.mean()); ys=float(ytr.std()+1e-8)
    ytr=((ytr-ym)/ys).astype(np.float32)[:,None]
    yv=((yv-ym)/ys).astype(np.float32)[:,None]
    yt=((yt-ym)/ys).astype(np.float32)[:,None]
    return {
        "task":"regression","output_dim":1,
        "train":(torch.from_numpy(Xtr),torch.from_numpy(ytr)),
        "val":(torch.from_numpy(Xv),torch.from_numpy(yv)),
        "test":(torch.from_numpy(Xt),torch.from_numpy(yt)),
        "meta":{"dataset":"dsprites","input_mean":mu,"input_sd":sd,"target_mean":ym,"target_sd":ys,
                "target":"horizontal latent position","resolution":IMAGE_RESOLUTION,"npz":str(p)}
    }


def load_datasets() -> Dict[str,Dict[str,Any]]:
    out={}
    if "fmnist" in DATASETS: out["fmnist"]=load_fmnist()
    if "dsprites" in DATASETS: out["dsprites"]=load_dsprites()
    return out


def shifted_environment(X: torch.Tensor, dataset: str) -> torch.Tensor:
    # Deterministic additive sensor perturbation, calibrated to requested image-space SNR.
    x=X.float(); signal=float(torch.mean(x*x).item()); noise_var=signal/(10.0**(SHIFT_SNR_DB/10.0))
    g=torch.Generator(device="cpu").manual_seed(NOISE_SEED_BASE + int(stable_hash_dict({"d":dataset},8),16)%1000003)
    noise=torch.randn(x.shape,generator=g,dtype=x.dtype)*math.sqrt(max(noise_var,EPS))
    return x+noise

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
    def __init__(self,hidden:Sequence[int],activation:str,output_dim:int,seed:int):
        super().__init__(); self.activation_name=activation; self.act=ActivationModule(activation)
        dims=[IMAGE_RESOLUTION*IMAGE_RESOLUTION]+list(hidden)+[output_dim]
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
    def __init__(self,spec:Dict[str,Any],activation:str,output_dim:int,seed:int):
        super().__init__(); self.activation_name=activation; self.act=ActivationModule(activation)
        hidden=list(spec["hidden_channels"]); kernels=list(spec["kernels"]); chans=[1]+hidden+[output_dim]
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
    if arch_id in MLP_ARCHS: return DeepMLP(MLP_ARCHS[arch_id],activation,output_dim,s).to(DEVICE)
    return DeepCNN(CNN_ARCHS[arch_id],activation,output_dim,s).to(DEVICE)


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
    if validn==0:return {"valid_fraction":0.0,"summary":{},"state":None,"input_df":pd.DataFrame(rows)}
    edge=[e/validn for e in edge_sum]; node=[v/validn for v in node_sum]; idf=pd.DataFrame(rows)
    good=idf[idf.valid==True]
    summary={"conditional_valid_fraction":validn/n,"S_path_median":float(good.S_path.median()),"entropy_deficit_median":float(good.entropy_deficit.median()),"entropy_deficit_norm_median":float(good.entropy_deficit_norm.median()),"eta1_median":float(good.eta1.median()),"eta2_median":float(good.eta2.median())}
    return {"valid_fraction":validn/n,"summary":summary,"state":{"edge_flow":edge,"node_flow":node},"input_df":idf}


@torch.no_grad()
def mlp_structural_population(model:DeepMLP,output_dim:int,task:str)->Dict[str,Any]:
    targets=list(range(output_dim)) if task=="classification" else [0]
    edge_sum=None; node_sum=None; vals=[]
    for t in targets:
        src=torch.ones((1,IMAGE_RESOLUTION*IMAGE_RESOLUTION),dtype=torch.float64)
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
        for j in range(len(xb)):
            rows.append({"input_index":lo+j,"valid":True,"S_path":st["S_path"][j],"entropy_deficit":st["entropy_deficit"][j],"entropy_deficit_norm":st["entropy_deficit_norm"][j],"eta1":st["eta1"][j],"eta2":st["eta2"][j]}); validn+=1
            for li,d in enumerate(st["layer_metrics"]): layer_rows.append({"input_index":lo+j,"layer":li,**{k:float(v[j]) for k,v in d.items()}})
        if occ_sum is None:
            occ_sum=[np.sum(e,0) for e in st["occurrence_flow"]]; par_sum=[np.sum(p,0) for p in st["parameter_flow"]]; node_sum=[np.sum(v,0) for v in st["node_flow"]]
        else:
            for i,e in enumerate(st["occurrence_flow"]):occ_sum[i]+=np.sum(e,0)
            for i,p in enumerate(st["parameter_flow"]):par_sum[i]+=np.sum(p,0)
            for i,v in enumerate(st["node_flow"]):node_sum[i]+=np.sum(v,0)
    occ=[e/validn for e in occ_sum]; par=[p/validn for p in par_sum]; node=[v/validn for v in node_sum]
    idf=pd.DataFrame(rows); ldf=pd.DataFrame(layer_rows); hidden=ldf[ldf.layer<ldf.layer.max()] if len(ldf) else ldf
    goodidf=idf[idf.valid==True];summary={"conditional_valid_fraction":validn/max(n,1),"S_path_median":float(goodidf.S_path.median()),"entropy_deficit_median":float(goodidf.entropy_deficit.median()),"entropy_deficit_norm_median":float(goodidf.entropy_deficit_norm.median()),"eta1_median":float(goodidf.eta1.median()),"eta2_median":float(goodidf.eta2.median())}
    for c in ["S_occ","S_param","S_reuse","eta1_occ","eta1_param","eta1_reuse"]:
        if len(hidden):summary[f"hidden_mean_{c}"]=float(hidden.groupby("input_index")[c].mean().median())
    return {"summary":summary,"state":{"occurrence_flow":occ,"parameter_flow":par,"node_flow":node},"input_df":idf,"layer_df":ldf}


@torch.no_grad()
def cnn_structural_population(model:DeepCNN,output_dim:int,task:str)->Dict[str,Any]:
    targets=list(range(output_dim)) if task=="classification" else [0]; occ_sum=None; par_sum=None; node_sum=None; vals=[]; layers=[]
    shapes=model.spatial_shapes()
    for t in targets:
        src=torch.ones((1,1,IMAGE_RESOLUTION,IMAGE_RESOLUTION),dtype=torch.float64)
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
    edge_mag:Dict[Tuple[int,int],float]
    layer_of_edge:Dict[Tuple[int,int],int]
    node_meta:List[Dict[str,Any]]


def mlp_graph_from_state(model:DeepMLP,state:Dict[str,Any],enrichment:bool)->FlowGraph:
    node_sizes=[model.layers[0].in_features]+[l.out_features for l in model.layers]
    offsets=np.cumsum([0]+node_sizes[:-1]).tolist(); edge_mag={}; layer_of_edge={}; meta=[]
    for li,n in enumerate(node_sizes):
        for j in range(n):meta.append({"layer":li,"channel":j,"y":-1,"x":-1})
    for li,E0 in enumerate(state["edge_flow"]):
        E=np.asarray(E0,float); scale=E.size if enrichment else 1.0; E=E*scale
        for o in range(E.shape[0]):
            for i in range(E.shape[1]):
                f=float(max(E[o,i],0)); u=offsets[li]+i; v=offsets[li+1]+o; e=(u,v) if u<v else (v,u); edge_mag[e]=f; layer_of_edge[e]=li
    return FlowGraph(sum(node_sizes),edge_mag,layer_of_edge,meta)


def cnn_graph_from_state(model:DeepCNN,state:Dict[str,Any],enrichment:bool)->FlowGraph:
    shapes_hw=model.spatial_shapes(); layer_shapes=[(model.channels[i],*shapes_hw[i]) for i in range(len(shapes_hw))]
    offsets=[0]
    for c,h,w in layer_shapes[:-1]:offsets.append(offsets[-1]+c*h*w)
    meta=[]
    for li,(c,h,w) in enumerate(layer_shapes):
        for cc in range(c):
            for yy in range(h):
                for xx in range(w):meta.append({"layer":li,"channel":cc,"y":yy,"x":xx})
    def nid(off,c,y,x,H,W):return int(off+c*H*W+y*W+x)
    edge_mag={}; layer_of_edge={}
    for li,E0 in enumerate(state["occurrence_flow"]):
        E=np.asarray(E0,float); E=E*(E.size if enrichment else 1.0); cin,Hin,Win=layer_shapes[li]; cout,Hout,Wout=layer_shapes[li+1]; k=model.kernels[li]
        for co in range(cout):
            for ci in range(cin):
                for dy in range(k):
                    for dx in range(k):
                        vals=E[co,ci,dy,dx]
                        for yy in range(Hout):
                            sy=yy+dy
                            for xx in range(Wout):
                                sx=xx+dx; u=nid(offsets[li],ci,sy,sx,Hin,Win); v=nid(offsets[li+1],co,yy,xx,Hout,Wout); e=(u,v) if u<v else (v,u); edge_mag[e]=float(max(vals[yy,xx],0)); layer_of_edge[e]=li
    return FlowGraph(sum(c*h*w for c,h,w in layer_shapes),edge_mag,layer_of_edge,meta)


def global_node_measure(state:Dict[str,Any],family:str)->np.ndarray:
    node=state["node_flow"]; x=np.concatenate([np.asarray(v,float).ravel() for v in node]); x=np.clip(x,0,None); x/=max(float(x.sum()),EPS); return x


def metric_lengths(G:FlowGraph,power:float=METRIC_POWER):return {e:float((max(w,0)+EPS)**(-power)) for e,w in G.edge_mag.items()}


def distance_matrix(G:FlowGraph,lengths:Dict[Tuple[int,int],float])->np.ndarray:
    rows=[];cols=[];vals=[]
    for (u,v),L in lengths.items():rows.extend([u,v]);cols.extend([v,u]);vals.extend([L,L])
    A=csr_matrix((vals,(rows,cols)),shape=(G.n_nodes,G.n_nodes),dtype=float)
    return np.asarray(csgraph_dijkstra(A,directed=False,unweighted=False),float)


def bcanw_scales(D:np.ndarray,n_scales:int)->np.ndarray:
    vals=D[np.triu_indices(D.shape[0],k=1)]; vals=vals[np.isfinite(vals)&(vals>0)]
    if len(vals)==0:return np.array([],float)
    lo=max(float(np.min(vals))*.95,EPS); hi=float(np.max(vals))*1.005
    return np.geomspace(lo,hi,max(n_scales,4)) if hi>lo else np.array([lo],float)


def node_strength(G:FlowGraph)->np.ndarray:
    s=np.zeros(G.n_nodes,float)
    for (u,v),w in G.edge_mag.items():s[u]+=w;s[v]+=w
    return s


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
    nb=int(colors.max())+1; mass=np.bincount(colors,weights=node_mu,minlength=nb).astype(float); mean=max(float(mass.mean()),EPS); lac=float(np.mean(mass*mass)/(mean*mean))
    W=np.zeros((nb,nb),float); total=0.; inter=0.
    for (u,v),f in Graw.edge_mag.items():
        total+=f; a,b=int(colors[u]),int(colors[v])
        if a==b:continue
        W[a,b]+=f; inter+=f
    mask=(W>0)&(~np.eye(nb,dtype=bool)); vals=W[mask]; vals=vals[np.isfinite(vals)&(vals>0)]
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


def covering_quantile_scale_loglog(radii,NB,N,q):
    r=np.asarray(radii,float); y=np.asarray(NB,float)/float(N); good=np.isfinite(r)&np.isfinite(y)&(r>0)&(y>0);r,y=r[good],y[good]
    if len(r)<2:return np.nan
    order=np.argsort(r);r,y=r[order],y[order];y=np.minimum.accumulate(y)
    # Strict bracket: target must lie within observed range, not be clamped.
    if not (y[-1] <= q <= y[0]):return np.nan
    idx=np.where(y<=q)[0]
    if len(idx)==0:return np.nan
    j=int(idx[0])
    if j==0:return float(r[0]) if np.isclose(y[0],q) else np.nan
    ly0,ly1=math.log(y[j-1]),math.log(y[j]);lr0,lr1=math.log(r[j-1]),math.log(r[j]);lq=math.log(q)
    if abs(ly1-ly0)<EPS:return float(math.sqrt(r[j-1]*r[j]))
    t=(lq-ly0)/(ly1-ly0); return float(math.exp(lr0+t*(lr1-lr0)))


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
    rc,nc,lc,plateaus=compress_box_staircase_midpoint(radii,NB,lac);xi50=covering_quantile_scale_loglog(rc,nc,Ggeom.n_nodes,.5);xc,deff,deffr2=local_effective_dimension_transition(rc,nc,xi50);sw=strict_scaling_window(rc,nc,Ggeom.n_nodes)
    qidx=nearest_resolution_indices(NB,Ggeom.n_nodes);qks=mean_pairwise_ks([qnorm[i] for i in qidx]);ret=float(np.nanmean(inter[qidx])) if len(qidx) else np.nan
    local_summ={"local_delta_median":np.nan,"valid_local_delta_fraction":np.nan,"local_nodes_evaluated":0};local_df=pd.DataFrame()
    if compute_local:local_summ,local_df=local_metric_dimensions(D,node_mu,Ggeom.node_meta)
    valid_deff=(np.isfinite(deff)&np.isfinite(deffr2)&(deffr2>=LOCAL_SLOPE_R2_MIN)&(deff>0))
    dstar=float(np.nanmax(deff[valid_deff])) if np.any(valid_deff) else np.nan;xdstar=float(xc[np.where(valid_deff)[0][np.nanargmax(deff[valid_deff])]]) if np.any(valid_deff) else np.nan
    frac=NB/Ggeom.n_nodes;mask=np.isfinite(lac)&(frac>=.015)&(frac<=.85);lacstar=float(np.nanmax(lac[mask])) if np.any(mask) else np.nan
    summ={"geometry_skipped":False,"geometry_skip_reason":"","n_graph_nodes":Ggeom.n_nodes,"xi50":xi50,"log_xi50":safe_log(xi50),"d_star":dstar,"x_d_star":xdstar,"lacunarity_star":lacstar,"scaling_dB":sw["dB"],"scaling_r2":sw["r2"],"scaling_credible":sw["credible"],"quotient_mean_pairwise_KS":qks,"renorm_flow_retention_mean":ret,"n_plateaus":len(plateaus),"transition_fraction":len(rc)/max(len(radii),1),"max_plateau_span_decades":float(plateaus.span_decades.max()) if len(plateaus) else np.nan,"xi50_bracketed":bool(np.isfinite(xi50)),**local_summ}
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
    mask_comp=(nc>1)&np.isfinite(rc)&np.isfinite(nc);rcd,ncd,lcd=rc[mask_comp],nc[mask_comp],lc[mask_comp];comp=pd.DataFrame({"ell_mid":rcd,"NB":ncd,"NB_over_N":ncd/Ggeom.n_nodes,"ell_over_xi50":rcd/xi50 if np.isfinite(xi50) else np.full(len(rcd),np.nan),"d_eff":deff,"d_eff_r2":deffr2,"lacunarity":lcd})
    return summ,pd.DataFrame(rows),comp,local_df.assign(**{}) if len(local_df) else local_df

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
    return checkpoint_path(rid).exists() and all(part_path(rid,k).exists() for k in ["summary","history","path_summary","cnn_layers"])


def select_geometry_row(row:pd.Series)->bool:
    if int(row.seed_index) not in GEOM_SEEDS:return False
    return (str(row.optimizer)==GEOM_ACTIVATION_PANEL_OPT) or (str(row.activation)==GEOM_OPTIMIZER_PANEL_ACT)


@torch.no_grad()
def endpoint_state(model,row,X,y,env):
    task=str(row.task);family=str(row.family);Xuse=X if env=="clean" else shifted_environment(X,str(row.dataset))
    if family=="mlp":return mlp_population_state(model,Xuse,y,task,ENDPOINT_INPUTS)
    return cnn_population_state(model,Xuse,y,task,ENDPOINT_INPUTS)


@torch.no_grad()
def structural_state(model,row):
    if row.family=="mlp":return mlp_structural_population(model,output_dim_for(row.dataset),row.task)
    return cnn_structural_population(model,output_dim_for(row.dataset),row.task)


def train_one(row:pd.Series,data:Dict[str,Any]):
    rid=str(row.run_id)
    if RESUME and run_complete(rid):return
    Xtr,ytr=data["train"];Xv,yv=data["val"];Xt,yt=data["test"]
    model=build_model(row.arch_id,row.activation,output_dim_for(row.dataset),row.dataset,int(row.seed_index));cal_scales=calibrate_initial_signal(model,Xtr)
    opt=make_optimizer(model,row.optimizer);task=row.task
    init=evaluate_performance(model,Xv,yv,task);best_metric=init["loss"];best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};hist=[];diverged=False
    for epoch in range(1,MAX_EPOCH+1):
        model.train();losses=[]
        for ids in deterministic_batches(len(Xtr),epoch-1,int(row.seed_index),BATCH_SIZE):
            xb=Xtr[ids].to(DEVICE);yb=ytr[ids].to(DEVICE);opt.zero_grad(set_to_none=True);pred=model(xb);loss=loss_batch(pred,yb,task)
            if not torch.isfinite(loss):diverged=True;break
            loss.backward();opt.step();losses.append(float(loss.item()))
        if diverged:break
        val=evaluate_performance(model,Xv,yv,task);hist.append({"run_id":rid,"epoch":epoch,"train_loss":float(np.mean(losses)),"val_loss":val["loss"],"val_accuracy":val.get("accuracy",np.nan)})
        if np.isfinite(val["loss"]) and val["loss"]<best_metric:
            best_metric=val["loss"];best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if not np.isfinite(val["loss"]) or val["loss"]>1e10:diverged=True;break
    model.load_state_dict(best_state);model.to(DEVICE);model.eval();ptest=evaluate_performance(model,Xt,yt,task)
    meta={"run_id":rid,"dataset":row.dataset,"task":task,"family":row.family,"arch_id":row.arch_id,"optimizer":row.optimizer,"activation":row.activation,"seed_index":int(row.seed_index),"parameter_count":model.parameter_count(),"best_epoch":best_epoch,"best_val_loss":best_metric,"best_test_loss":ptest["loss"],"best_test_accuracy":ptest.get("accuracy",np.nan),"diverged":diverged,"calibration_scales_json":json.dumps([float(x) for x in cal_scales])}
    structural=structural_state(model,row); path_rows=[{**meta,"state":"structural",**{f"path_{k}_median":v for k,v in structural["summary"].items()}}]; cnn_layer_rows=[]
    if row.family=="cnn" and "layer_df" in structural:
        d=structural["layer_df"].groupby("layer").mean(numeric_only=True).reset_index();d["state"]="structural";[d.__setitem__(k,v) for k,v in meta.items()];cnn_layer_rows.append(d)
    for env in ENVIRONMENTS:
        cond=endpoint_state(model,row,Xt,yt,env);path_rows.append({**meta,"state":env,**{f"path_{k}":v for k,v in cond["summary"].items()}})
        if row.family=="cnn" and "layer_df" in cond:
            d=cond["layer_df"].groupby("layer").mean(numeric_only=True).reset_index();d["state"]=env;[d.__setitem__(k,v) for k,v in meta.items()];cnn_layer_rows.append(d)
    torch.save({"state_dict":best_state,"meta":meta},checkpoint_path(rid));atomic_csv(pd.DataFrame([meta]),part_path(rid,"summary"));atomic_csv(pd.DataFrame(hist),part_path(rid,"history"));atomic_csv(pd.DataFrame(path_rows),part_path(rid,"path_summary"));atomic_csv(pd.concat(cnn_layer_rows,ignore_index=True) if cnn_layer_rows else pd.DataFrame(),part_path(rid,"cnn_layers"))
    del model,opt
    if torch.cuda.is_available():torch.cuda.empty_cache()

# ==================================================================================================
# 9. POPULATION / INPUT GEOMETRY BACKFILL
# ==================================================================================================

def geometry_done(rid):return part_path(rid,"geometry").exists() and part_path(rid,"geometry_scales").exists() and part_path(rid,"geometry_compressed").exists()


def run_geometry_panel(manifest:pd.DataFrame,data_cache:Dict[str,Any]):
    if not RUN_BCANW:return
    sub=manifest[manifest.apply(select_geometry_row,axis=1)].copy();todo=[r for _,r in sub.iterrows() if not(RESUME and geometry_done(str(r.run_id)))]
    eta=ETAReporter("population BCANw",len(todo))
    for k,row in enumerate(tqdm(todo,desc="Benchmark population BCANw"),1):
        t0=time.perf_counter();rid=str(row.run_id);cp=checkpoint_path(rid)
        if not cp.exists():continue
        data=data_cache[row.dataset];Xt,yt=data["test"];model=build_model(row.arch_id,row.activation,output_dim_for(row.dataset),row.dataset,int(row.seed_index));model.load_state_dict(torch.load(cp,map_location="cpu")["state_dict"]);model.to(DEVICE);model.eval();structural=structural_state(model,row)
        gsum=[];gsc=[];gcomp=[];glocal=[]
        states=[("structural",structural["state"])]
        for env in ENVIRONMENTS:
            cond=endpoint_state(model,row,Xt,yt,env);states.append((env,cond["state"]))
        for source,state in states:
            gs,sc,comp,local=analyze_geometry(model,state,row.family,BCANW_SCALES,compute_local=True);base={"run_id":rid,"dataset":row.dataset,"task":row.task,"family":row.family,"arch_id":row.arch_id,"optimizer":row.optimizer,"activation":row.activation,"seed_index":int(row.seed_index),"geometry_source":source}
            gsum.append({**base,**gs})
            if len(sc):
                for q,v in base.items():sc[q]=v
                gsc.append(sc)
            if len(comp):
                for q,v in base.items():comp[q]=v
                gcomp.append(comp)
            if len(local):
                for q,v in base.items():local[q]=v
                glocal.append(local)
        atomic_csv(pd.DataFrame(gsum),part_path(rid,"geometry"));atomic_csv(pd.concat(gsc,ignore_index=True) if gsc else pd.DataFrame(),part_path(rid,"geometry_scales"));atomic_csv(pd.concat(gcomp,ignore_index=True) if gcomp else pd.DataFrame(),part_path(rid,"geometry_compressed"));
        if glocal:
            pd.concat(glocal,ignore_index=True).to_csv(part_path(rid,"local_geometry").with_suffix(".csv.gz"),index=False,compression="gzip")
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        eta.update(time.perf_counter()-t0,k)
    eta.finish()


def run_input_geometry(manifest:pd.DataFrame,data_cache:Dict[str,Any])->pd.DataFrame:
    path=OUTDIR/"tables"/"input_resolved_geometry.csv"
    if not RUN_INPUT_GEOM:return pd.DataFrame()
    existing=safe_read_csv(path) if RESUME else pd.DataFrame();rows=[] if existing.empty else existing.to_dict("records");done={(r["run_id"],r["environment"],int(r["input_index"])) for r in rows if "run_id" in r}
    sub=manifest[(manifest.arch_id.isin(INPUT_GEOM_ARCH_IDS))&(manifest.optimizer==INPUT_GEOM_OPTIMIZER)&(manifest.seed_index.isin(INPUT_GEOM_SEEDS))]
    total=sum(1 for _,r in sub.iterrows() for env in ENVIRONMENTS for i in range(INPUT_GEOM_INPUTS) if (r.run_id,env,i) not in done);eta=ETAReporter("input BCANw",total);c=0
    for _,row in sub.iterrows():
        cp=checkpoint_path(row.run_id)
        if not cp.exists():continue
        model=build_model(row.arch_id,row.activation,output_dim_for(row.dataset),row.dataset,int(row.seed_index));model.load_state_dict(torch.load(cp,map_location="cpu")["state_dict"]);model.to(DEVICE);model.eval();data=data_cache[row.dataset];Xt,yt=data["test"]
        for env in ENVIRONMENTS:
            Xenv=Xt if env=="clean" else shifted_environment(Xt,row.dataset)
            for i in range(min(INPUT_GEOM_INPUTS,len(Xenv))):
                key=(row.run_id,env,i)
                if key in done:continue
                t0=time.perf_counter();x=Xenv[i:i+1];y=yt[i:i+1]
                if row.family=="mlp":
                    resp=mlp_response_sq(model,x,torch.zeros_like(x));target=y.long() if row.task=="classification" else torch.zeros(1,dtype=torch.long);st=mlp_factor_state(model,torch.square(x.reshape(1,-1).to(torch.float64)),resp,target);state={"edge_flow":[e[0] for e in st["edge_flow"]],"node_flow":[v[0] for v in st["node_flow"]]}
                else:
                    resp=cnn_response_sq(model,x,torch.zeros_like(x));target=y.long() if row.task=="classification" else torch.zeros(1,dtype=torch.long);st=cnn_factor_state(model,torch.square(x.to(torch.float64)),resp,target);state={"occurrence_flow":[e[0] for e in st["occurrence_flow"]],"parameter_flow":[p[0] for p in st["parameter_flow"]],"node_flow":[v[0] for v in st["node_flow"]]}
                gs,_,_,_=analyze_geometry(model,state,row.family,INPUT_GEOM_SCALES,compute_local=False);rec={"run_id":row.run_id,"dataset":row.dataset,"family":row.family,"arch_id":row.arch_id,"optimizer":row.optimizer,"activation":row.activation,"seed_index":int(row.seed_index),"environment":env,"input_index":i,**gs};rows.append(rec);done.add(key);atomic_csv(pd.DataFrame(rows),path);c+=1;eta.update(time.perf_counter()-t0,c)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    eta.finish();return pd.DataFrame(rows)

# ==================================================================================================
# 10. AGGREGATION / STATISTICS
# ==================================================================================================

def aggregate_parts(manifest:pd.DataFrame):
    kinds={k:[] for k in ["summary","history","path_summary","cnn_layers","geometry","geometry_scales","geometry_compressed"]};local=[]
    for rid in manifest.run_id:
        for k in kinds:
            d=safe_read_csv(part_path(rid,k))
            if len(d):kinds[k].append(d)
        lp=part_path(rid,"local_geometry").with_suffix(".csv.gz")
        if lp.exists():
            try:local.append(pd.read_csv(lp))
            except Exception:pass
    out={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in kinds.items()};out["local_geometry"]=pd.concat(local,ignore_index=True) if local else pd.DataFrame()
    names={"summary":"run_summary.csv","history":"training_history.csv","path_summary":"path_summary.csv","cnn_layers":"cnn_layer_decomposition.csv","geometry":"population_geometry.csv","geometry_scales":"population_geometry_scales.csv","geometry_compressed":"population_geometry_compressed.csv"}
    for k,nm in names.items():atomic_csv(out[k],OUTDIR/"tables"/nm)
    if len(out["local_geometry"]):out["local_geometry"].to_csv(OUTDIR/"tables"/"local_geometry_nodes.csv.gz",index=False,compression="gzip")
    return out


PATH_METRICS=["path_eta1_median","path_eta2_median","path_entropy_deficit_norm_median"]
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
    # FMNIST and dSprites are different tasks; these paired tests are reported only for dimensionless path observables.
    rows=[]
    for state in ["structural","clean","shifted"]:
        d0=pathdf[pathdf.state==state]
        for metric in ["path_eta1_median","path_eta2_median","path_entropy_deficit_norm_median"]:
            if metric not in d0.columns:continue
            piv=d0.pivot_table(index=["arch_id","optimizer","activation","seed_index"],columns="dataset",values=metric,aggfunc="first").dropna()
            if not all(ds in piv.columns for ds in ["fmnist","dsprites"]):continue
            delta=piv["fmnist"].to_numpy(float)-piv["dsprites"].to_numpy(float)
            try:w,p=wilcoxon(delta,zero_method="wilcox")
            except Exception:w,p=np.nan,1.0
            lo,hi=bootstrap_median_ci(delta,seed=1500);rows.append({"state":state,"metric":metric,"contrast":"fmnist_minus_dsprites","n_blocks":len(delta),"median_difference":float(np.median(delta)),"hodges_lehmann":hodges_lehmann_onesample(delta),"rank_biserial":paired_rank_biserial(delta),"ci_low":lo,"ci_high":hi,"p_raw":p})
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
    summary=tabs["summary"];path=tabs["path_summary"];layers=tabs["cnn_layers"];g=tabs["geometry"];gs=tabs["geometry_scales"];local=tabs["local_geometry"]
    # 1 performance matrix
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),constrained_layout=True)
    for ax,ds in zip(axs,DATASETS):
        d=summary[summary.dataset==ds].copy();metric="best_test_accuracy" if ds=="fmnist" else "best_test_loss";p=d.groupby(["arch_id","activation"])[metric].median().unstack().reindex(index=ARCH_IDS,columns=ACTIVATIONS);im=ax.imshow(p.values,aspect="auto",cmap="inferno");ax.set_yticks(range(len(p.index)),p.index);ax.set_xticks(range(len(ACTIVATIONS)),ACTIVATIONS);ax.set_title(f"{ds.upper()} | {'accuracy' if ds=='fmnist' else 'MSE'}");plt.colorbar(im,ax=ax,fraction=.046,pad=.03)
    savefig(fig,"fig01_benchmark_performance_INFERNO")
    # 2 path concentration structural/realized
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1),constrained_layout=True)
    for ax,metric,title in zip(axs,PATH_METRICS,[r"effective path fraction proxy $\eta_1$",r"normalized concentration deficit"]):
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
    # 10 validity diagnostics
    if len(g):
        fig,axs=plt.subplots(1,3,figsize=(7.2,2.8),constrained_layout=True);cg=g[~g.geometry_skipped.astype(bool)]
        axs[0].hist(cg.transition_fraction.dropna(),bins=20);axs[0].set_xlabel("transition fraction")
        axs[1].hist(cg.max_plateau_span_decades.dropna(),bins=20);axs[1].set_xlabel("max plateau span (decades)")
        rate=cg.groupby("geometry_source").xi50_bracketed.mean().reindex(["structural","clean","shifted"]);axs[2].bar(range(len(rate)),rate.values);axs[2].set_xticks(range(len(rate)),rate.index);axs[2].set_ylabel(r"$\xi_{50}$ bracket rate");savefig(fig,"fig10_geometry_validity_diagnostics")


def make_report():
    figs=sorted((OUTDIR/"figures").glob("fig*.png"));report=OUTDIR/"benchmark_case_study_scientific_report.pdf"
    with PdfPages(report) as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.suptitle("Work II benchmark case study: FMNIST + dSprites",fontsize=18,y=.95);fig.text(.07,.85,"324 trained networks | 3 deep MLPs + 3 deep CNNs | SGDM/RMSProp/Adam | ReLU/tanh/SiLU | 3 seeds\nClean and shifted conditional realization are evaluated on the same trained states. Structural disposition is held fixed.\nBCANw uses audited midpoint plateau compression, bracketed log-log xi50 and transition-supported d_eff.",fontsize=11,va="top",wrap=True);pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
        for p in figs:
            img=plt.imread(p);fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_subplot(111);ax.imshow(img);ax.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
    return report

# ==================================================================================================
# 12. MANIFEST / ORCHESTRATION
# ==================================================================================================

def write_experiment_manifest(manifest):
    obj={"study":"Work II benchmark case study","datasets":DATASETS,"architectures":ARCH_IDS,"optimizers":OPTIMIZERS,"activations":ACTIVATIONS,"seeds":SEEDS,"trained_networks":int(len(manifest)),"training":{"epochs":MAX_EPOCH,"n_train":N_TRAIN,"n_val":N_VAL,"n_test":N_TEST,"batch_size":BATCH_SIZE,"learning_rates":LEARNING_RATES},"environments":{"clean":"canonical test distribution","shifted":f"deterministic Gaussian sensor shift at {SHIFT_SNR_DB} dB","training_axis":False},"geometry":{"scales":BCANW_SCALES,"max_nodes":BCANW_MAX_NODES,"geometry_seeds":GEOM_SEEDS,"selection":"Adam across all activations UNION ReLU across all optimizers","audited_fixes":["geometric-midpoint plateau compression","strict bracketed xi_q","log-log xi_q interpolation","transition-supported d_eff","nodewise local geometry retained"]}}
    atomic_json(obj,OUTDIR/"experiment_manifest.json")


def run_experiment():
    start=time.perf_counter();banner("WORK II BENCHMARK CASE STUDY — FMNIST + dSPRITES — MLP/CNN CONDITIONAL REALIZATION")
    print(f"Device: {DEVICE} | profile: {PROFILE} | output: {OUTDIR.resolve()}")
    print(f"Datasets={DATASETS} | architectures={ARCH_IDS} | optimizers={OPTIMIZERS} | activations={ACTIVATIONS} | seeds={SEEDS}")
    manifest=build_manifest();write_experiment_manifest(manifest);print(f"Training factorial: {len(manifest)} networks")
    banner("Load benchmark datasets")
    data_cache=load_datasets()
    banner("Train / resume 324-network benchmark map")
    todo=[r for _,r in manifest.iterrows() if not(RESUME and run_complete(r.run_id))];eta=ETAReporter("training",len(todo))
    for k,row in enumerate(tqdm(todo,desc="Benchmark networks"),1):
        t0=time.perf_counter();train_one(row,data_cache[row.dataset]);eta.update(time.perf_counter()-t0,k)
    eta.finish()
    banner("Predeclared population BCANw panel")
    run_geometry_panel(manifest,data_cache)
    banner("Nested input-resolved geometry sentinel")
    input_geom=run_input_geometry(manifest,data_cache)
    banner("Aggregate tables")
    tabs=aggregate_parts(manifest)
    banner("Matched statistical tests")
    omnibus,pairs,envfx=matched_factor_tests(tabs["path_summary"]);archfx=architecture_friedman_tests(tabs["path_summary"]);benchfx=benchmark_context_tests(tabs["path_summary"]);gfx=geometry_effect_tests(tabs["geometry"]);gfac=geometry_factor_tests(tabs["geometry"])
    banner("Publication figures")
    make_figures(tabs,input_geom,omnibus,pairs,envfx,gfx);report=make_report() if RUN_FIGURES else None
    elapsed=time.perf_counter()-start
    summary=["WORK II BENCHMARK CASE STUDY COMPLETE",f"Device: {DEVICE}",f"Wall time: {human_time(elapsed)}",f"Trained networks summarized: {len(tabs['summary'])}/{len(manifest)}",f"Population geometry states: {len(tabs['geometry'])}",f"Input-resolved geometry rows: {len(input_geom)}",f"Local geometry node rows: {len(tabs['local_geometry'])}",f"Report: {report}",f"Output: {OUTDIR.resolve()}"]
    (OUTDIR/"experiment_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8");print("\n"+"\n".join(summary))
    return {"manifest":manifest,"data":data_cache,"tables":tabs,"input_geometry":input_geom,"omnibus":omnibus,"pairwise":pairs,"environment_effects":envfx,"architecture_effects":archfx,"benchmark_context_effects":benchfx,"geometry_effects":gfx,"geometry_factor_effects":gfac,"report":report}


BENCH_RESULTS = run_experiment()
