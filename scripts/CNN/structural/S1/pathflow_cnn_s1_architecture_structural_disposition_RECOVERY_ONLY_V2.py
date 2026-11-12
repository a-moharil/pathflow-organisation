# ==================================================================================================
# CNN-S1 — ARCHITECTURE-WIDE STRUCTURAL DISPOSITION
# Recovery hardening revision V2: scalar-safe rank persistence + full statistics/figure/report preflight.
# AutoDL architecture discovery + RF-SMAC refinement + fixed longitudinal inference cohort
# ==================================================================================================
# Single-file / single-Jupyter-cell research experiment.
#
# Scientific object
# -----------------
# CNN-S1 studies the input-independent structural disposition of a trained convolutional scaffold:
#
#     O_CNN^S = (G_exec_tilde, Gamma_exec, p^S ; rho)
#
# where G_exec_tilde is the spatially expanded execution DAG, Gamma_exec is the set of complete spatial
# input-output paths, rho maps execution-edge occurrences to shared convolutional parameter identities,
# and
#
#     Q_gamma^S = product_{a in gamma} theta_{rho(a)}^2,
#     p^S(gamma) = Q_gamma^S / sum_{gamma'} Q_{gamma'}^S.
#
# There is NO input condition, NO finite activation response factor, and NO activation-dependent path
# gate inside the observable. ReLU is fixed only as the training nonlinearity that shapes the learned
# parameter state. The structural measure itself is weight-only.
#
# Experimental logic
# ------------------
# Stage S1-A: 600 stratified Sobol CNN architectures, low-fidelity 60-epoch discovery.
# Stage S1-B: 400 RF-SMAC sequential refinements, with 25% random/coverage interleaving.
# Stage S1-anchor: 8 forced reference/control architectures, giving 1008 discovery architectures.
# Stage S1-geometry: 240 architecture/configuration-stratified discovery states receive BCANw at t=0,60.
# Stage S1-C: select 48 architectures by maximin coverage over configuration + structural phenotype.
# Stage S1-D: 48 architectures x 2 seeds = 96 full 360-epoch longitudinal trajectories.
#
# Exact common state grid:
#     0,1,2,5,10,20,40,80,120,160,240,360
# Full BCANw grid on the 48x2 cohort:
#     0,20,80,160,240,360
#
# RF-SMAC implementation note
# ---------------------------
# The script implements the Work-I RF-SMAC mechanism directly with scikit-learn:
# random-forest surrogate + expected improvement + Sobol cold map + 25% random/coverage interleaving.
# This avoids dependence on a specific external SMAC package version while preserving the sequential
# random-forest model-based configuration logic used throughout the programme.
#
# IMPORTANT: held-out validation/test performance is NOT an acquisition target. AutoDL is used as an
# information-seeking experimental-design mechanism to cover architecture/configuration space and seek
# structurally novel response regions. Formal inference is performed only on the frozen 48-architecture
# longitudinal panel, with seeds aggregated within architecture before architecture-level inference.
#
# CNN-specific structural decomposition
# -------------------------------------
# From p^S we compute exact occurrence flow F_a^S and its sharing pushforward F_theta^S. For stride-one
# valid convolutions, each parameter identity in a layer has uniform reuse multiplicity m_l, giving
#
#     H_occ^S = H_param^S + H_reuse^S,
#     S_occ^S = S_param^S + S_reuse^S,
#     eta1_occ^S = eta1_param^S * eta1_reuse^S.
#
# Thus S1 can distinguish architecture-wide concentration arising from selection of shared parameter
# identities from concentration arising within their spatial reuse fibres.
#
# Statistical hierarchy
# ---------------------
# Planned architecture-level inference (48 architecture units after seed aggregation):
#   * paired Wilcoxon signed-rank for t=360 - t=0 changes;
#   * Hodges-Lehmann paired location shift;
#   * paired rank-biserial effect;
#   * bootstrap 95% CI for median architecture change;
#   * architecture-wide sign consistency with exact Clopper-Pearson interval;
#   * Holm correction within predeclared measure/flow, sharing, and geometry families;
#   * Friedman + Kendall W for repeated checkpoint trajectories;
#   * Spearman architecture-rank preservation and seed reproducibility.
#
# Secondary/post-hoc explanatory analyses are explicitly labelled post hoc:
#   * Spearman associations between architecture descriptors and structural responses;
#   * Kruskal-Wallis + epsilon^2 for morphology/kernel-family response differences;
#   * pairwise Mann-Whitney/rank-biserial contrasts only after an omnibus categorical signal;
#   * two-stage finite-scaling analysis: paired credibility first, d_B only where both states are credible.
# These do not retroactively become architecture-population causal claims because the inference cohort is a
# maximin-designed panel rather than a random sample from a hypothetical population of all CNNs.
#
# CUDA / Colab
# ------------
# Publication profile requires CUDA by default. Datasets are staged on GPU, training and exact structural
# dynamic programming use CUDA, and detailed tqdm/ETA/GPU telemetry is printed. BCANw all-pairs shortest
# path geometry remains SciPy/CPU-bound by design, so low GPU utilization during geometry is expected.
#
# Run in Colab:
#     %run /content/pathflow_cnn_s1_architecture_structural_disposition_autodl_rfsmac_single_cell.py
#
# Profiles:
#     PATHFLOW_CNN_S1_PROFILE=publication_colab   (default frozen experiment)
#     PATHFLOW_CNN_S1_PROFILE=pilot               (fast end-to-end smoke test)
#
# Failure recovery:
#     Re-run with the SAME PATHFLOW_CNN_S1_OUTDIR. If all 96 trajectories and 576 BCANw states already
#     exist, the script automatically enters analysis-only recovery and performs ZERO retraining/BCANw.
#     Explicit recovery: PATHFLOW_CNN_S1_ANALYSIS_ONLY=1. Never use START_FRESH for recovery.
#
# ==================================================================================================

from __future__ import annotations

import os, sys, math, json, time, random, hashlib, itertools, warnings, subprocess, traceback, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

# Lightweight dependency fallback for clean Colab runtimes.
def _ensure_pkg(import_name: str, pip_name: Optional[str] = None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])

_ensure_pkg("sklearn", "scikit-learn")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from scipy.stats import (
    spearmanr, wilcoxon, friedmanchisquare, ks_2samp, norm, binomtest,
    kruskal, mannwhitneyu, beta
)
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================================================================================
# 0. PROFILE / EXPERIMENT CONFIGURATION
# ==================================================================================================

S1_PROFILE = os.environ.get("PATHFLOW_CNN_S1_PROFILE", "publication_colab").strip().lower()
S1_RESUME = os.environ.get("PATHFLOW_CNN_S1_RESUME", "1") != "0"
S1_START_FRESH = os.environ.get("PATHFLOW_CNN_S1_START_FRESH", "0") == "1"
S1_REQUIRE_CUDA = os.environ.get("PATHFLOW_CNN_S1_REQUIRE_CUDA", "1") != "0"
S1_GPU_RESIDENT_DATA = os.environ.get("PATHFLOW_CNN_S1_GPU_RESIDENT_DATA", "1") != "0"
S1_RUN_BCANW = os.environ.get("PATHFLOW_CNN_S1_RUN_BCANW", "1") != "0"
S1_RUN_EXACT = os.environ.get("PATHFLOW_CNN_S1_RUN_EXACT", "1") != "0"
S1_RUN_REPORT = os.environ.get("PATHFLOW_CNN_S1_RUN_REPORT", "1") != "0"
S1_CUDA_DEVICE = int(os.environ.get("PATHFLOW_CNN_S1_CUDA_DEVICE", "0"))

# Failure-resilience / recovery controls. These are intentionally ON by default for expensive runs.
S1_ANALYSIS_ONLY = True  # recovery-only build: expensive compute is impossible
S1_AUTO_ANALYSIS_RECOVERY = os.environ.get("PATHFLOW_CNN_S1_AUTO_ANALYSIS_RECOVERY", "1") != "0"
S1_ANALYSIS_SELFTEST = os.environ.get("PATHFLOW_CNN_S1_ANALYSIS_SELFTEST", "1") != "0"
S1_CONFIRM_START_FRESH = os.environ.get("PATHFLOW_CNN_S1_CONFIRM_START_FRESH", "0") == "1"

S1_SEED = 3829
MODEL_SEED_BASE = 271828
DATA_SEED_BASE = 57721
ORDER_SEED_BASE = 141421

if S1_REQUIRE_CUDA and not S1_ANALYSIS_ONLY and not torch.cuda.is_available():
    raise RuntimeError(
        "CNN-S1 publication profile requires CUDA, but torch.cuda.is_available() is False. "
        "In Colab select Runtime > Change runtime type > GPU. Set PATHFLOW_CNN_S1_REQUIRE_CUDA=0 "
        "only for an intentional CPU smoke test."
    )
if torch.cuda.is_available():
    torch.cuda.set_device(S1_CUDA_DEVICE)
    DEVICE = torch.device(f"cuda:{S1_CUDA_DEVICE}")
else:
    DEVICE = torch.device("cpu")

OUTDIR = Path(os.environ.get("PATHFLOW_CNN_S1_OUTDIR", "pathflow_cnn_s1_architecture_structural_disposition_results"))
for sub in ["tables","figures","checkpoints_discovery","checkpoints_longitudinal","run_parts","geometry_parts"]:
    (OUTDIR/sub).mkdir(parents=True,exist_ok=True)

# Fixed S1 scientific environment/training protocol.
INPUT_CHANNELS = 1
RESOLUTION = 16
ACTIVATION = "relu"
OPTIMIZER = "adam"
LEARNING_RATE = 3e-3  # same mature CNN Adam operating point used in the optimizer-conditioned programme
WEIGHT_DECAY = 0.0
BATCH_SIZE = 256
N_TRAIN, N_VAL, N_TEST = 20_000, 8_000, 8_000
DISCOVERY_N_TRAIN, DISCOVERY_N_VAL = 4_000, 1_000
DISCOVERY_EPOCHS = 60
FULL_EPOCHS = 360
STATE_GRID = [0,1,2,5,10,20,40,80,120,160,240,360]
GEOMETRY_GRID = [0,20,80,160,240,360]
LONGITUDINAL_SEEDS = [0,1]
N_LONG_ARCH = 48
N_S2_PANEL = 12
N_S3_SENTINELS = 6

# Architecture discovery: 600 stratified Sobol + 400 RF-SMAC + 8 anchors = 1008.
MORPHOLOGIES = ["constant","expansion","taper","bottleneck","hourglass"]
DEPTH_VALUES = [3,4,5,6,7]
COMPUTE_FACTORS = [0.5,1.0,2.0,4.0]
KERNEL_FAMILIES = ["all3","alternating","spatial_then_pointwise"]
SOBOL_REPS_PER_CELL = 2
SOBOL_TOTAL = len(MORPHOLOGIES)*len(DEPTH_VALUES)*len(COMPUTE_FACTORS)*len(KERNEL_FAMILIES)*SOBOL_REPS_PER_CELL
RFSMAC_STEPS = 400
RFSMAC_POOL = 768
RFSMAC_RF_TREES = 160
RFSMAC_RANDOM_INTERLEAVE = 0.25
RFSMAC_EI_XI = 0.03
RFSMAC_KNN_NOVELTY = 8
ANCHOR_TOTAL = 8
DISCOVERY_TOTAL = SOBOL_TOTAL + RFSMAC_STEPS + ANCHOR_TOTAL
DISCOVERY_GEOMETRY_N = 240

# Architecture constraints.
MIN_CHANNELS = 2
MAX_CHANNELS = 40
MIN_FINAL_RESOLUTION = 4
MAX_EXEC_NODES = 6500
TARGET_MAC_TOL = 0.18
if S1_PROFILE == "pilot":
    MAX_EXEC_NODES = 1100

# Structural/BCANw constants faithful to the corrected Work-I / Work-II conventions.
EPS = 1e-14
PATH_LOG_FLOOR = 1e-300
METRIC_POWER = 1.0
BCANW_SCALES_DISCOVERY = 26
BCANW_SCALES_LONG = 36
BCANW_MAX_NODES = 7000
LOCAL_DIMENSION_NODES = 128
if S1_PROFILE == "pilot":
    BCANW_MAX_NODES = 1100
    LOCAL_DIMENSION_NODES = 16
LOCAL_SLOPE_WINDOW = 7
LOCAL_SLOPE_R2_MIN = 0.90
SCALING_R2_MIN = 0.97
SCALING_MIN_DIM = 0.25
SCALING_MIN_POINTS = 6
SCALING_MIN_SPAN_DECADES = 0.50
SCALING_MIN_NB_RATIO = 5.0
SCALING_MAX_SLOPE_CV = 0.25
SCALING_MAX_DELTA_BIC = 2.0
RENORM_TARGET_BOX_FRACTIONS = np.array([0.50,0.25,0.125,0.0625])
COARSE_RENYI_Q = [0.5,1.0,2.0,3.0]
LOCAL_COMMON_MIN_POINTS = 7
LOCAL_COMMON_MIN_SPAN_DECADES = 0.50
LOCAL_COMMON_SUPPORT_GROWTH = 4.0
LOCAL_COMMON_MASS_GROWTH = 3.0
LOCAL_COMMON_R2_MIN = 0.95
LOCAL_COMMON_SATURATION = 0.95

BOOTSTRAP_B = 4000
POSTHOC_ALPHA = 0.05

# Pilot reduces every stage but executes the identical scientific code path.
if S1_PROFILE == "pilot":
    RESOLUTION = 10
    N_TRAIN,N_VAL,N_TEST = 768,256,256
    DISCOVERY_N_TRAIN,DISCOVERY_N_VAL = 384,128
    DISCOVERY_EPOCHS = 2
    FULL_EPOCHS = 4
    STATE_GRID = [0,1,2,4]
    GEOMETRY_GRID = [0,4]
    LONGITUDINAL_SEEDS = [0]
    MORPHOLOGIES = ["constant","bottleneck"]
    DEPTH_VALUES = [3,4]
    COMPUTE_FACTORS = [0.7,1.2]
    KERNEL_FAMILIES = ["all3","alternating"]
    SOBOL_REPS_PER_CELL = 1
    SOBOL_TOTAL = len(MORPHOLOGIES)*len(DEPTH_VALUES)*len(COMPUTE_FACTORS)*len(KERNEL_FAMILIES)
    RFSMAC_STEPS = 4
    ANCHOR_TOTAL = 2
    DISCOVERY_TOTAL = SOBOL_TOTAL+RFSMAC_STEPS+ANCHOR_TOTAL
    N_LONG_ARCH = 4
    N_S2_PANEL = 3
    N_S3_SENTINELS = 2
    DISCOVERY_GEOMETRY_N = 4
    RFSMAC_POOL = 80
    RFSMAC_RF_TREES = 40
    BCANW_SCALES_DISCOVERY = 10
    BCANW_SCALES_LONG = 10
    LOCAL_DIMENSION_NODES = 24
    BOOTSTRAP_B = 300

# Optional direct scalar overrides.
DISCOVERY_EPOCHS = int(os.environ.get("PATHFLOW_CNN_S1_DISCOVERY_EPOCHS",DISCOVERY_EPOCHS))
FULL_EPOCHS = int(os.environ.get("PATHFLOW_CNN_S1_FULL_EPOCHS",FULL_EPOCHS))
DISCOVERY_N_TRAIN = int(os.environ.get("PATHFLOW_CNN_S1_DISCOVERY_NTRAIN",DISCOVERY_N_TRAIN))
DISCOVERY_N_VAL = int(os.environ.get("PATHFLOW_CNN_S1_DISCOVERY_NVAL",DISCOVERY_N_VAL))

random.seed(S1_SEED); np.random.seed(S1_SEED); torch.manual_seed(S1_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(S1_SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ==================================================================================================
# 1. GENERIC HELPERS / CUDA TELEMETRY / STATISTICAL UTILITIES
# ==================================================================================================

def banner(title:str):
    print("\n"+"="*108); print(title); print("="*108)

def human_time(seconds:float)->str:
    if not np.isfinite(seconds) or seconds<0:return "unknown"
    s=int(round(seconds));h,s=divmod(s,3600);m,s=divmod(s,60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")

class ETAReporter:
    def __init__(self,stage:str,total:int,window:int=12):
        self.stage=stage;self.total=max(int(total),1);self.window=max(int(window),3);self.start=time.perf_counter();self.d=[]
    def update(self,duration:float,completed:int):
        self.d.append(float(duration));self.d=self.d[-self.window:]
        rate=float(np.mean(self.d));rem=max(self.total-completed,0)*rate
        if completed==1 or completed%max(1,self.total//20)==0 or completed==self.total:
            print(f"[ETA] {self.stage}: {completed}/{self.total} | recent {human_time(rate)}/item | remaining ~{human_time(rem)}")
    def finish(self): print(f"[ETA] {self.stage} completed in {human_time(time.perf_counter()-self.start)}")

def gpu_snapshot(prefix="GPU"):
    if not torch.cuda.is_available():
        print(f"[{prefix}] CUDA unavailable"); return {}
    d=torch.cuda.current_device();p=torch.cuda.get_device_properties(d)
    alloc=torch.cuda.memory_allocated(d)/2**30;res=torch.cuda.memory_reserved(d)/2**30;peak=torch.cuda.max_memory_allocated(d)/2**30
    util="?";used="?"
    try:
        raw=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True).strip().splitlines()[d]
        util,used=[x.strip() for x in raw.split(",")]
    except Exception:pass
    print(f"[{prefix}] {p.name} cuda:{d} | util={util}% | nvidia-mem={used} MiB | allocated={alloc:.3f} GiB | reserved={res:.3f} GiB | peak={peak:.3f} GiB")
    return {"allocated_gib":alloc,"reserved_gib":res,"peak_gib":peak,"total_gib":p.total_memory/2**30}

def stable_hash_dict(d:Dict[str,Any],n:int=12)->str:
    s=json.dumps(d,sort_keys=True,separators=(",",":"),default=str).encode();return hashlib.sha256(s).hexdigest()[:n]

def atomic_csv(df:pd.DataFrame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");df.to_csv(tmp,index=False);tmp.replace(path)

def atomic_json(obj:Any,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8");tmp.replace(path)

def safe_read(path:Path)->pd.DataFrame:
    """Read a CSV safely. Missing files mean "not produced yet"; corrupt existing files are fatal.

    Returning an empty frame for a corrupt checkpoint/result file can silently trigger many hours of
    recomputation. For an expensive experiment that is the wrong failure mode, so an existing file
    that cannot be parsed raises immediately and preserves all other artifacts untouched.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Existing result file is unreadable; refusing silent recomputation: {path}") from exc

def r2_score_np(y,pred):
    y=np.asarray(y,float);pred=np.asarray(pred,float);ss=np.sum((y-y.mean())**2);return float(1-np.sum((y-pred)**2)/ss) if ss>0 else np.nan

def safe_log(x): return float(np.log(x)) if np.isfinite(x) and x>0 else np.nan

def holm_adjust(pvals:Sequence[float])->np.ndarray:
    p=np.asarray(pvals,float);n=len(p);out=np.full(n,np.nan);good=np.isfinite(p);idx=np.where(good)[0]
    if not len(idx):return out
    o=idx[np.argsort(p[idx])];adj=[];mx=0.0
    for rank,j in enumerate(o):
        v=min(1.0,(len(o)-rank)*p[j]);mx=max(mx,v);adj.append(mx)
    for j,v in zip(o,adj):out[j]=v
    return out

def hodges_lehmann_onesample(d:np.ndarray)->float:
    x=np.asarray(d,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan
    vals=[]
    for i in range(len(x)):
        vals.extend(((x[i]+x[i:])/2.0).tolist())
    return float(np.median(vals))

def paired_rank_biserial(d:np.ndarray)->float:
    x=np.asarray(d,float);x=x[np.isfinite(x)&(x!=0)]
    if not len(x):return 0.0
    ranks=pd.Series(np.abs(x)).rank(method="average").to_numpy(float);wp=float(ranks[x>0].sum());wn=float(ranks[x<0].sum());return float((wp-wn)/max(wp+wn,EPS))

def mannwhitney_rank_biserial(x,y)->float:
    x=np.asarray(x,float);y=np.asarray(y,float);x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    if not len(x) or not len(y):return np.nan
    U=mannwhitneyu(x,y,alternative="two-sided").statistic
    return float(2*U/(len(x)*len(y))-1)

def bootstrap_median_ci(d:np.ndarray,seed:int=0,B:int=BOOTSTRAP_B)->Tuple[float,float]:
    x=np.asarray(d,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan
    rng=np.random.default_rng(seed);vals=np.empty(B,float)
    for b in range(B):vals[b]=np.median(rng.choice(x,size=len(x),replace=True))
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))


def _scalar_float(x:Any, default:float=np.nan)->float:
    """Return a genuine scalar float or a safe default.

    SciPy may return ndarray-valued statistics when a 2-D input slips through. Publication/report
    code must never place an array into a scalar result table or plotting matrix.
    """
    try:
        arr=np.asarray(x)
        if arr.size==1:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    return float(default)

def _spearman_scalar(x:Sequence[float], y:Sequence[float], min_n:int=5)->Tuple[float,float,int]:
    """Robust scalar Spearman result for paired 1-D vectors.

    Explicitly converts both inputs to 1-D numeric arrays and removes non-finite pairs. This prevents
    pandas duplicate-column selections from silently turning a vector correlation into a correlation matrix.
    """
    xa=np.asarray(x,dtype=float).reshape(-1); ya=np.asarray(y,dtype=float).reshape(-1)
    if xa.size!=ya.size:
        raise ValueError(f"Spearman inputs have unequal lengths: {xa.size} vs {ya.size}")
    ok=np.isfinite(xa)&np.isfinite(ya); xa=xa[ok]; ya=ya[ok]; n=int(ok.sum())
    if n<min_n:
        return np.nan,np.nan,n
    try:
        res=spearmanr(xa,ya)
        return _scalar_float(res.statistic),_scalar_float(res.pvalue),n
    except Exception:
        return np.nan,np.nan,n

def clopper_pearson(k:int,n:int,alpha:float=.05)->Tuple[float,float]:
    if n<=0:return np.nan,np.nan
    lo=0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1));hi=1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return lo,hi

def deterministic_batches(n:int,epoch:int,seed_index:int,batch_size:int):
    rng=np.random.default_rng(ORDER_SEED_BASE+100003*seed_index+epoch);ids=rng.permutation(n)
    for lo in range(0,n,batch_size):yield ids[lo:lo+batch_size]

def model_seed(arch_id:str,seed_index:int)->int:
    return MODEL_SEED_BASE+int(stable_hash_dict({"arch":arch_id,"seed":int(seed_index)},8),16)%1_000_003

if S1_START_FRESH:
    if not S1_CONFIRM_START_FRESH:
        raise RuntimeError(
            "PATHFLOW_CNN_S1_START_FRESH=1 is destructive. To delete existing expensive results, also set "
            "PATHFLOW_CNN_S1_CONFIRM_START_FRESH=1. Otherwise leave START_FRESH=0 and resume safely."
        )
    for sub in ["tables","figures","checkpoints_discovery","checkpoints_longitudinal","run_parts","geometry_parts"]:
        for p in (OUTDIR/sub).glob("*"):
            if p.is_file():
                try:p.unlink()
                except Exception:pass
    for p in [OUTDIR/"COMPUTE_COMPLETE.json", OUTDIR/"analysis_status.json", OUTDIR/"experiment_summary.txt"]:
        if p.exists():
            try:p.unlink()
            except Exception:pass

# ==================================================================================================
# 2. CNN MODEL, COMPUTE DESCRIPTORS, FIXED SPATIAL ENVIRONMENT
# ==================================================================================================

class ActivationModule(nn.Module):
    def __init__(self,name:str):super().__init__();self.name=str(name).lower()
    def forward(self,x):
        if self.name=="relu":return F.relu(x)
        if self.name=="tanh":return torch.tanh(x)
        if self.name=="silu":return F.silu(x)
        if self.name=="identity":return x
        raise ValueError(self.name)

class ConvTraceNet(nn.Module):
    """Bias-free, stride-one VALID CNN with learned scalar feature map and fixed global average."""
    def __init__(self,resolution:int,hidden_channels:Sequence[int],kernel_sizes:Sequence[int],activation:str,seed:int):
        super().__init__();self.resolution=int(resolution);self.hidden_channels=[int(x) for x in hidden_channels];self.kernel_sizes=[int(k) for k in kernel_sizes];self.activation_name=str(activation);self.activation=ActivationModule(activation)
        channels=[INPUT_CHANNELS]+self.hidden_channels+[1];self.channels=channels
        if len(self.kernel_sizes)!=len(channels)-1:raise ValueError("kernel_sizes must have one entry per learned convolution")
        self.convs=nn.ModuleList();torch.manual_seed(int(seed))
        for cin,cout,k in zip(channels[:-1],channels[1:],self.kernel_sizes):self.convs.append(nn.Conv2d(cin,cout,kernel_size=k,stride=1,padding=0,bias=False))
        self.reset_common_directions(seed)
        if self.output_resolution()<MIN_FINAL_RESOLUTION:raise ValueError("output spatial resolution below S1 minimum")
    def reset_common_directions(self,seed:int):
        g=torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for conv in self.convs:
                W=torch.randn(conv.weight.shape,generator=g,dtype=conv.weight.dtype);W/=math.sqrt(max(conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1],1));conv.weight.copy_(W)
    def spatial_shapes(self):
        h=w=self.resolution;out=[(h,w)]
        for k in self.kernel_sizes:h=h-k+1;w=w-k+1;out.append((h,w))
        return out
    def output_resolution(self):return int(self.spatial_shapes()[-1][0])
    def parameter_count(self):return int(sum(p.numel() for p in self.parameters()))
    def forward(self,x):
        h=x
        for l,conv in enumerate(self.convs):
            z=conv(h);h=self.activation(z) if l<len(self.convs)-1 else z
        return h.mean(dim=(-1,-2))

@torch.no_grad()
def calibrate_initial_signal(model:ConvTraceNet,Xcal:torch.Tensor,target_var:float=1.0):
    model.eval();h=Xcal.to(DEVICE);scales=[]
    for l,conv in enumerate(model.convs):
        z=conv(h)
        if l<len(model.convs)-1:
            sd=float(torch.std(z,unbiased=False).item());scale=float(np.clip(math.sqrt(target_var)/max(sd,1e-6),.15,6.0));conv.weight.mul_(scale);z=conv(h);h=model.activation(z);scales.append(scale)
        else:scales.append(1.0)
    return scales

def architecture_descriptors(hidden_channels:Sequence[int],kernel_sizes:Sequence[int],resolution:int=RESOLUTION)->Dict[str,float]:
    ch=[INPUT_CHANNELS]+list(map(int,hidden_channels))+[1];h=w=int(resolution);mac=0;occ=0;nodes=INPUT_CHANNELS*h*w;rf=1
    for cin,cout,k in zip(ch[:-1],ch[1:],kernel_sizes):
        h=h-int(k)+1;w=w-int(k)+1
        if h<1:return {"valid":False}
        layer_occ=cout*cin*k*k*h*w;occ+=layer_occ;mac+=layer_occ;nodes+=cout*h*w;rf+=k-1
    params=sum(cout*cin*k*k for cin,cout,k in zip(ch[:-1],ch[1:],kernel_sizes))
    return {"valid":h>=MIN_FINAL_RESOLUTION and nodes<=MAX_EXEC_NODES,"parameter_count":int(params),"occurrence_count":int(occ),"macs":int(mac),"execution_nodes":int(nodes),"output_resolution":int(h),"receptive_field":int(rf),"depth":len(kernel_sizes),"mean_hidden_width":float(np.mean(hidden_channels)) if hidden_channels else 1.0,"max_hidden_width":int(max(hidden_channels)) if hidden_channels else 1}

# Fixed mature CNN anchors from the conditional programme plus two controls.
ANCHOR_LIBRARY={
    "C1_shallow3":{"hidden_channels":[4,4],"kernel_sizes":[3,3,1]},
    "C2_medium3":{"hidden_channels":[4,8,8],"kernel_sizes":[3,3,3,1]},
    "C3_bottleneck":{"hidden_channels":[6,3,6],"kernel_sizes":[3,3,3,1]},
    "C4_pointwise":{"hidden_channels":[4,8,8],"kernel_sizes":[1,1,1,1]},
    "C5_deep3":{"hidden_channels":[4,8,8,12],"kernel_sizes":[3,3,3,3,1]},
    "C6_large5":{"hidden_channels":[4,8,8],"kernel_sizes":[5,3,3,1]},
    "CTRL_small3":{"hidden_channels":[3,3],"kernel_sizes":[3,3,1]},
    "CTRL_deepmix":{"hidden_channels":[4,5,6,6,6],"kernel_sizes":[3,1,3,1,3,1]},
}
for _k,_v in list(ANCHOR_LIBRARY.items()):
    if not architecture_descriptors(_v["hidden_channels"],_v["kernel_sizes"])["valid"]:
        ANCHOR_LIBRARY.pop(_k)

REF_DESC=architecture_descriptors(ANCHOR_LIBRARY["C2_medium3"]["hidden_channels"],ANCHOR_LIBRARY["C2_medium3"]["kernel_sizes"])
REFERENCE_MACS=float(REF_DESC["macs"])


def generate_fixed_environment():
    """Canonical distributed synthetic spatial regression environment used throughout CNN-S1."""
    n=N_TRAIN+N_VAL+N_TEST;R=RESOLUTION;rng=np.random.default_rng(DATA_SEED_BASE)
    X=np.zeros((n,1,R,R),np.float32);yclean=np.zeros((n,1),np.float32)
    yy,xx=np.meshgrid(np.linspace(-1,1,R),np.linspace(-1,1,R),indexing="ij")
    basis1=(np.cos(np.pi*xx)*np.cos(np.pi*yy)).astype(np.float32)
    basis2=(np.sin(1.5*np.pi*xx)*np.cos(.5*np.pi*yy)).astype(np.float32)
    for i in range(n):
        offset=rng.normal(0,.8);scale=np.exp(rng.normal(-.1,.25));amp1=rng.normal(0,.7);amp2=rng.normal(0,.45);noise=rng.normal(0,1,(R,R)).astype(np.float32)
        X[i,0]=.62*offset+scale*noise+.50*amp1*basis1+.30*amp2*basis2
        yclean[i,0]=.55*offset+.25*(scale-1)+.30*amp1+.18*amp2+.16*offset*amp1+.08*amp1*amp2
    xmu=float(X[:N_TRAIN].mean());xsd=float(X[:N_TRAIN].std()+1e-8);X=((X-xmu)/xsd).astype(np.float32)
    ym=float(yclean[:N_TRAIN].mean());ys=float(yclean[:N_TRAIN].std()+1e-8);yclean=((yclean-ym)/ys).astype(np.float32)
    snr=25.0;y=yclean+rng.normal(0,math.sqrt(1/snr),size=yclean.shape).astype(np.float32);ym2=float(y[:N_TRAIN].mean());ys2=float(y[:N_TRAIN].std()+1e-8);y=((y-ym2)/ys2).astype(np.float32)
    splits={"train":(torch.from_numpy(X[:N_TRAIN]),torch.from_numpy(y[:N_TRAIN])),"val":(torch.from_numpy(X[N_TRAIN:N_TRAIN+N_VAL]),torch.from_numpy(y[N_TRAIN:N_TRAIN+N_VAL])),"test":(torch.from_numpy(X[N_TRAIN+N_VAL:]),torch.from_numpy(y[N_TRAIN+N_VAL:]))}
    if torch.cuda.is_available() and S1_GPU_RESIDENT_DATA:
        splits={k:(a.to(DEVICE,non_blocking=True),b.to(DEVICE,non_blocking=True)) for k,(a,b) in splits.items()}
    return splits,{"environment":"canonical_distributed_regression","resolution":R,"snr":snr,"seed":DATA_SEED_BASE}

@torch.no_grad()
def eval_mse(model:nn.Module,X:torch.Tensor,y:torch.Tensor,chunk:int=2048)->float:
    model.eval();total=0.;count=0
    for lo in range(0,len(X),chunk):
        hi=min(lo+chunk,len(X));xb=X[lo:hi].to(DEVICE,non_blocking=True);yb=y[lo:hi].to(DEVICE,non_blocking=True);p=model(xb);total+=float(torch.sum((p-yb)**2));count+=yb.numel()
    return total/max(count,1)

# ==================================================================================================
# 3. EXACT STRUCTURAL COMPLETE-PATH ENGINE AND SHARING PUSHFORWARD
# ==================================================================================================

def _global_normalize(t:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
    dims=tuple(range(1,t.ndim));s=t.sum(dim=dims);valid=torch.isfinite(s)&(s>0);safe=torch.where(valid,s,torch.ones_like(s));out=t/safe.reshape((-1,)+((1,)*(t.ndim-1)));out=torch.where(valid.reshape((-1,)+((1,)*(t.ndim-1))),out,torch.zeros_like(out));return out,s

@torch.no_grad()
def cnn_log_partition(model:ConvTraceNet,source_sq:torch.Tensor,response_sq:List[torch.Tensor],q:float)->torch.Tensor:
    a=torch.pow(torch.clamp(source_sq.to(torch.float64),min=0),float(q));a,s=_global_normalize(a);valid=torch.isfinite(s)&(s>0);logscale=torch.where(valid,torch.log(torch.clamp(s,min=EPS)),torch.full_like(s,-torch.inf))
    for l,conv in enumerate(model.convs):
        w2q=torch.pow(torch.square(conv.weight.to(torch.float64)),float(q));a=F.conv2d(a,w2q,bias=None,stride=1,padding=0)
        if l<len(model.convs)-1:a=a*torch.pow(torch.clamp(response_sq[l].to(torch.float64),min=0),float(q))
        a,sl=_global_normalize(a);good=valid&torch.isfinite(sl)&(sl>0);logscale=torch.where(good,logscale+torch.log(torch.clamp(sl,min=EPS)),torch.full_like(logscale,-torch.inf));valid=good
    hout,wout=model.spatial_shapes()[-1];g2q=float((1/(hout*wout))**(2*float(q)));return torch.where(valid,logscale+math.log(g2q),torch.full_like(logscale,-torch.inf))

def cnn_topological_log_count(model:ConvTraceNet)->float:
    h,w=model.spatial_shapes()[-1];logn=math.log(float(model.channels[-1]*h*w))
    for conv in model.convs:logn+=math.log(float(conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1]))
    return float(logn)

def _entropy_prob(p:torch.Tensor,dims:Tuple[int,...])->torch.Tensor:
    return -torch.where(p>0,p*torch.log(torch.clamp(p,min=PATH_LOG_FLOOR)),torch.zeros_like(p)).sum(dim=dims)

@torch.no_grad()
def cnn_factor_state(model:ConvTraceNet,source_sq:torch.Tensor,response_sq:List[torch.Tensor],return_flows:bool=True)->Dict[str,Any]:
    source_sq=source_sq.to(DEVICE,dtype=torch.float64);response_sq=[r.to(DEVICE,dtype=torch.float64) for r in response_sq];B=source_sq.shape[0]
    logZ1=cnn_log_partition(model,source_sq,response_sq,1.0);logZ2=cnn_log_partition(model,source_sq,response_sq,2.0);valid=torch.isfinite(logZ1)
    alpha=[];a,_=_global_normalize(source_sq);alpha.append(a)
    for l,conv in enumerate(model.convs):
        a=F.conv2d(a,torch.square(conv.weight.to(torch.float64)),bias=None,stride=1,padding=0)
        if l<len(model.convs)-1:a=a*response_sq[l]
        a,_=_global_normalize(a);alpha.append(a)
    beta=[None]*(len(model.convs)+1);b=torch.ones_like(alpha[-1],dtype=torch.float64);b,_=_global_normalize(b);beta[-1]=b
    for l in range(len(model.convs)-1,-1,-1):
        target=beta[l+1]
        if l<len(model.convs)-1:target=target*response_sq[l]
        b=F.conv_transpose2d(target,torch.square(model.convs[l].weight.to(torch.float64)),bias=None,stride=1,padding=0);b,_=_global_normalize(b);beta[l]=b
    node_flow=[]
    for l in range(len(alpha)):
        v=alpha[l]*beta[l];v,_=_global_normalize(v);node_flow.append(v)
    occurrence=[];param=[];layer_metrics=[];elog_edges=torch.zeros(B,dtype=torch.float64,device=DEVICE)
    max_occ=torch.zeros(B,dtype=torch.float64,device=DEVICE);max_push=torch.zeros(B,dtype=torch.float64,device=DEVICE)
    for l,conv in enumerate(model.convs):
        k=int(conv.kernel_size[0]);cout=conv.out_channels;cin=conv.in_channels;patches=F.unfold(alpha[l],kernel_size=k,stride=1,padding=0);Hout,Wout=alpha[l+1].shape[-2:];L=Hout*Wout;target=beta[l+1]
        if l<len(model.convs)-1:target=target*response_sq[l]
        target=target.reshape(B,cout,L);w2=torch.square(conv.weight.to(torch.float64)).reshape(cout,cin*k*k);E=target[:,:,None,:]*w2[None,:,:,None]*patches[:,None,:,:];den=E.sum(dim=(1,2,3));good=den>0;E=torch.where(good[:,None,None,None],E/torch.clamp(den[:,None,None,None],min=EPS),torch.zeros_like(E));E=E.reshape(B,cout,cin,k,k,Hout,Wout);P=E.sum(dim=(-1,-2));occurrence.append(E);param.append(P)
        occsum=E.sum(dim=(1,2,3,4,5,6));psum=P.sum(dim=(1,2,3,4));max_occ=torch.maximum(max_occ,torch.abs(occsum-1));max_push=torch.maximum(max_push,torch.abs(psum-occsum))
        logw=torch.log(torch.clamp(torch.square(conv.weight.to(torch.float64)),min=PATH_LOG_FLOOR));logfactor=logw[None,:,:,:,:,None,None]
        if l<len(model.convs)-1:logfactor=logfactor+torch.log(torch.clamp(response_sq[l],min=PATH_LOG_FLOOR))[:, :, None, None, None, :, :]
        elog_edges+=torch.sum(torch.where(E>0,E*logfactor,torch.zeros_like(E)),dim=(1,2,3,4,5,6))
        Hocc=_entropy_prob(E,(1,2,3,4,5,6));Hparam=_entropy_prob(P,(1,2,3,4));Pb=P[...,None,None];ratio=torch.where(E>0,E/torch.clamp(Pb,min=PATH_LOG_FLOOR),torch.ones_like(E));Hreuse_direct=-torch.sum(torch.where(E>0,E*torch.log(torch.clamp(ratio,min=PATH_LOG_FLOOR)),torch.zeros_like(E)),dim=(1,2,3,4,5,6));Hreuse=Hocc-Hparam
        ntheta=cout*cin*k*k;m=Hout*Wout;nocc=ntheta*m;Socc=math.log(nocc)-Hocc;Sparam=math.log(ntheta)-Hparam;Sreuse=math.log(m)-Hreuse;eta_occ=torch.exp(torch.clamp(-Socc,min=-745,max=0));eta_param=torch.exp(torch.clamp(-Sparam,min=-745,max=0));eta_reuse=torch.exp(torch.clamp(-Sreuse,min=-745,max=0));occflat=E.reshape(B,-1);pflat=P.reshape(B,-1)
        layer_metrics.append({"H_occ":Hocc,"H_param":Hparam,"H_reuse":Hreuse,"S_occ":Socc,"S_param":Sparam,"S_reuse":Sreuse,"eta1_occ":eta_occ,"eta1_param":eta_param,"eta1_reuse":eta_reuse,"occurrence_cv":occflat.std(1,unbiased=False)/torch.clamp(occflat.mean(1),min=EPS),"parameter_cv":pflat.std(1,unbiased=False)/torch.clamp(pflat.mean(1),min=EPS),"entropy_identity_residual":Hocc-Hparam-Hreuse_direct,"deficit_identity_residual":Socc-Sparam-Sreuse,"eta_factorization_residual":eta_occ-eta_param*eta_reuse,"reuse_multiplicity":torch.full((B,),float(m),dtype=torch.float64,device=DEVICE)})
    p0=node_flow[0];elogsource=torch.sum(torch.where(p0>0,p0*torch.log(torch.clamp(source_sq,min=PATH_LOG_FLOOR)),torch.zeros_like(p0)),dim=(1,2,3));hout,wout=model.spatial_shapes()[-1];logg2=math.log((1/(hout*wout))**2);elogq=elogsource+elog_edges+logg2;H=logZ1-elogq;logN=cnn_topological_log_count(model);Sdef=logN-H;logeta1=H-logN;logeta2=2*logZ1-logZ2-logN
    def arr(x):
        y=x.detach().cpu().numpy().astype(float);y[~valid.detach().cpu().numpy()]=np.nan;return y
    return {"valid":valid.detach().cpu().numpy().astype(bool),"logZ1":arr(logZ1),"logZ2":arr(logZ2),"S_path":arr(H),"entropy_deficit":arr(Sdef),"eta1":arr(torch.exp(torch.clamp(logeta1,min=-745,max=0))),"eta2":arr(torch.exp(torch.clamp(logeta2,min=-745,max=0))),"node_flow":[v.detach().cpu().numpy() for v in node_flow] if return_flows else [],"occurrence_flow":[e.detach().cpu().numpy() for e in occurrence] if return_flows else [],"parameter_flow":[p.detach().cpu().numpy() for p in param] if return_flows else [],"layer_metrics":[{k:arr(v) for k,v in d.items()} for d in layer_metrics],"occurrence_conservation_error":arr(max_occ),"pushforward_conservation_error":arr(max_push)}

@torch.no_grad()
def cnn_structural_state(model:ConvTraceNet,return_flows:bool=True)->Dict[str,Any]:
    source=torch.ones((1,INPUT_CHANNELS,model.resolution,model.resolution),dtype=torch.float64,device=DEVICE);responses=[];shapes=model.spatial_shapes()
    for l,conv in enumerate(model.convs[:-1]):
        h,w=shapes[l+1];responses.append(torch.ones((1,conv.out_channels,h,w),dtype=torch.float64,device=DEVICE))
    st=cnn_factor_state(model,source,responses,return_flows=return_flows)
    if not st["valid"][0]:raise RuntimeError("degenerate structural path partition")
    hidden=list(range(max(len(model.convs)-1,0)))
    summary={"logZ1":float(st["logZ1"][0]),"logZ2":float(st["logZ2"][0]),"S_path":float(st["S_path"][0]),"entropy_deficit":float(st["entropy_deficit"][0]),"eta1":float(st["eta1"][0]),"eta2":float(st["eta2"][0]),"occurrence_conservation_error":float(st["occurrence_conservation_error"][0]),"pushforward_conservation_error":float(st["pushforward_conservation_error"][0])}
    for key in ["S_occ","S_param","S_reuse","eta1_occ","eta1_param","eta1_reuse","occurrence_cv","parameter_cv"]:
        vals=[float(st["layer_metrics"][l][key][0]) for l in hidden];summary[f"hidden_mean_{key}"]=float(np.mean(vals)) if vals else np.nan
    summary["max_entropy_identity_residual"]=float(max(abs(d["entropy_identity_residual"][0]) for d in st["layer_metrics"]))
    summary["max_deficit_identity_residual"]=float(max(abs(d["deficit_identity_residual"][0]) for d in st["layer_metrics"]))
    summary["max_eta_factorization_residual"]=float(max(abs(d["eta_factorization_residual"][0]) for d in st["layer_metrics"]))
    if return_flows:
        summary.update({"node_flow":[v[0] for v in st["node_flow"]],"occurrence_flow":[e[0] for e in st["occurrence_flow"]],"parameter_flow":[p[0] for p in st["parameter_flow"]],"layer_metrics":[{k:float(v[0]) for k,v in d.items()} for d in st["layer_metrics"]]})
    return summary

# ==================================================================================================
# 4. SPATIALLY UNROLLED EXECUTION GRAPH + STRUCTURAL BCANw
# ==================================================================================================

@dataclass
class ExecutionGraph:
    n_nodes:int
    layer_shapes:List[Tuple[int,int,int]]
    offsets:List[int]
    edge_mag:Dict[Tuple[int,int],float]
    layer_of_edge:Dict[Tuple[int,int],int]

def _node_id(offset,c,y,x,H,W):return int(offset+c*H*W+y*W+x)

def execution_graph_from_occurrence(model:ConvTraceNet,occurrence_flow:List[np.ndarray],enrichment:bool)->ExecutionGraph:
    shapes_hw=model.spatial_shapes();layer_shapes=[(model.channels[l],h,w) for l,(h,w) in enumerate(shapes_hw)];offsets=[0]
    for c,h,w in layer_shapes[:-1]:offsets.append(offsets[-1]+c*h*w)
    edge_mag={};layer_of_edge={}
    for l,(conv,E0) in enumerate(zip(model.convs,occurrence_flow)):
        E=np.asarray(E0,float);E=E*(E.size if enrichment else 1.0);cin,Hin,Win=layer_shapes[l];cout,Hout,Wout=layer_shapes[l+1];k=int(conv.kernel_size[0])
        for co in range(cout):
            for ci in range(cin):
                for dy in range(k):
                    for dx in range(k):
                        vals=E[co,ci,dy,dx]
                        for y in range(Hout):
                            sy=y+dy
                            for x in range(Wout):
                                sx=x+dx;u=_node_id(offsets[l],ci,sy,sx,Hin,Win);v=_node_id(offsets[l+1],co,y,x,Hout,Wout);e=(u,v) if u<v else (v,u);edge_mag[e]=float(max(vals[y,x],0));layer_of_edge[e]=l
    return ExecutionGraph(sum(c*h*w for c,h,w in layer_shapes),layer_shapes,offsets,edge_mag,layer_of_edge)

def metric_lengths(P:ExecutionGraph,power:float=METRIC_POWER):return {e:float((max(w,0)+EPS)**(-power)) for e,w in P.edge_mag.items()}

def distance_matrix(P:ExecutionGraph,lengths):
    rows=[];cols=[];vals=[]
    for (u,v),L in lengths.items():rows.extend([u,v]);cols.extend([v,u]);vals.extend([L,L])
    A=csr_matrix((vals,(rows,cols)),shape=(P.n_nodes,P.n_nodes),dtype=float);return np.asarray(csgraph_dijkstra(A,directed=False,unweighted=False),float)

def bcanw_scales(D,n_scales):
    vals=D[np.triu_indices(D.shape[0],k=1)];vals=vals[np.isfinite(vals)&(vals>0)]
    if not len(vals):return np.array([],float)
    lo=max(float(vals.min())*.95,EPS);hi=float(vals.max())*1.005
    return np.geomspace(lo,hi,max(int(n_scales),4)) if hi>lo else np.array([lo])

def node_strength(P):
    s=np.zeros(P.n_nodes,float)
    for (u,v),w in P.edge_mag.items():s[u]+=w;s[v]+=w
    return s

def greedy_bcanw_coloring(D,ell,order):
    color=np.full(D.shape[0],-1,int)
    for u in order:
        used=set(color[(D[u]>=float(ell))&(color>=0)].tolist());c=0
        while c in used:c+=1
        color[u]=c
    return color

def global_node_measure(node_flow):
    x=np.concatenate([np.asarray(v,float).ravel() for v in node_flow]);x=np.clip(x,0,None);x/=max(x.sum(),EPS);return x

def renyi_effective_fraction(probabilities,q):
    p=np.asarray(probabilities,float);p=p[np.isfinite(p)&(p>0)]
    if not len(p):return np.nan
    p/=p.sum();E=len(p);H=-np.sum(p*np.log(p)) if abs(q-1)<1e-12 else math.log(np.sum(p**q))/(1-q);return float(math.exp(H)/E)

def box_flow_observables(colors,node_measure,raw_graph):
    nb=int(colors.max())+1;mass=np.bincount(colors,weights=node_measure,minlength=nb).astype(float);mean=max(mass.mean(),EPS);lac=float(np.mean(mass*mass)/(mean*mean));W=np.zeros((nb,nb),float);total=0.;inter=0.
    for (u,v),f in raw_graph.edge_mag.items():
        total+=f;a,b=int(colors[u]),int(colors[v])
        if a!=b:W[a,b]+=f;inter+=f
    mask=(W>0)&(~np.eye(nb,dtype=bool));vals=W[mask];vals=vals[np.isfinite(vals)&(vals>0)]
    if len(vals):
        vals_norm=vals/max(vals.mean(),EPS);probs=vals/max(vals.sum(),EPS);eta={q:renyi_effective_fraction(probs,q) for q in COARSE_RENYI_Q}
    else:vals_norm=np.array([]);eta={q:np.nan for q in COARSE_RENYI_Q}
    return {"lacunarity":lac,"quotient_flow_norm":vals_norm,"coarse_eta_q":eta,"interbox_flow_fraction":float(inter/max(total,EPS))}

def compress_box_staircase(radii,NB,lac=None):
    r=np.asarray(radii,float);n=np.asarray(NB,float);good=np.isfinite(r)&np.isfinite(n)&(r>0);r,n=r[good],n[good];l=np.asarray(lac,float)[good] if lac is not None else None
    if not len(r):return (r,n,np.array([])) if lac is not None else (r,n)
    keep=[0]
    for k in range(1,len(r)):
        if not np.isclose(n[k],n[keep[-1]],rtol=0,atol=0):keep.append(k)
    keep=np.array(keep,int);return (r[keep],n[keep],l[keep]) if lac is not None else (r[keep],n[keep])

def covering_quantile_scale(radii,NB,N,q):
    r=np.asarray(radii,float);y=np.asarray(NB,float)/float(N);good=np.isfinite(r)&np.isfinite(y)&(r>0);r,y=r[good],y[good]
    if not len(r):return np.nan
    o=np.argsort(r);r,y=r[o],y[o];y=np.minimum.accumulate(y)
    if q>=y[0]:return float(r[0])
    if q<=y[-1]:return float(r[-1])
    idx=np.where(y<=q)[0]
    if not len(idx):return np.nan
    j=int(idx[0]);
    if j==0:return float(r[0])
    y0,y1=y[j-1],y[j];lr0,lr1=np.log(r[j-1]),np.log(r[j]);t=(q-y0)/max(y1-y0,EPS);return float(np.exp(lr0+t*(lr1-lr0)))

def local_effective_dimension(radii_c,NB_c,xi50,window=LOCAL_SLOPE_WINDOW):
    r=np.asarray(radii_c,float);n=np.asarray(NB_c,float);good=np.isfinite(r)&np.isfinite(n)&(r>0)&(n>1);r,n=r[good],n[good]
    if not len(r) or not np.isfinite(xi50) or xi50<=0:return np.array([]),np.array([]),np.array([])
    x=r/xi50;lx=np.log(r);ly=np.log(n);d=np.full(len(r),np.nan);rr=np.full(len(r),np.nan);half=max(window//2,1)
    for i in range(len(r)):
        lo,hi=max(0,i-half),min(len(r),i+half+1)
        if hi-lo<3:continue
        b,a=np.polyfit(lx[lo:hi],ly[lo:hi],1);pred=a+b*lx[lo:hi];d[i]=-b;rr[i]=r2_score_np(ly[lo:hi],pred)
    return x,d,rr

def _window_slope_cv(x,y,subwin=5):
    if len(x)<max(4,subwin):return np.nan
    ds=[]
    for i in range(len(x)-subwin+1):b,_=np.polyfit(x[i:i+subwin],y[i:i+subwin],1);ds.append(-b)
    ds=np.asarray(ds,float);return float(ds.std()/max(abs(ds.mean()),EPS)) if len(ds)>=2 else np.nan

def strict_scaling_window(radii_c,NB_c,N):
    empty={"dB":np.nan,"r2":np.nan,"span_decades":0.,"n_points":0,"nb_ratio":np.nan,"slope_cv":np.nan,"delta_bic":np.nan,"credible":False}
    r=np.asarray(radii_c,float);nb=np.asarray(NB_c,float);ids=np.where(np.isfinite(r)&np.isfinite(nb)&(r>0)&(nb>1)&(nb<.95*N))[0]
    if len(ids)<SCALING_MIN_POINTS:return empty
    cand=[]
    for a in range(len(ids)-SCALING_MIN_POINTS+1):
        for b in range(a+SCALING_MIN_POINTS,len(ids)+1):
            sel=ids[a:b]
            if np.any(np.diff(sel)!=1):continue
            rs,ns=r[sel],nb[sel];span=np.log10(rs[-1]/rs[0]);nbr=ns[0]/max(ns[-1],EPS)
            if span<SCALING_MIN_SPAN_DECADES or nbr<SCALING_MIN_NB_RATIO:continue
            x,y=np.log(rs),np.log(ns);slope,inter=np.polyfit(x,y,1);dim=-slope;pred=inter+slope*x;rr=r2_score_np(y,pred)
            if not np.isfinite(rr) or rr<SCALING_R2_MIN or dim<SCALING_MIN_DIM:continue
            scv=_window_slope_cv(x,y,min(5,len(x)));rss1=max(np.sum((y-pred)**2),EPS);bic1=len(x)*np.log(rss1/len(x))+2*np.log(len(x));q2,q1,q0=np.polyfit(x,y,2);pred2=q0+q1*x+q2*x*x;rss2=max(np.sum((y-pred2)**2),EPS);bic2=len(x)*np.log(rss2/len(x))+3*np.log(len(x));dbic=bic1-bic2;cred=np.isfinite(scv) and scv<=SCALING_MAX_SLOPE_CV and dbic<=SCALING_MAX_DELTA_BIC;cand.append((cred,span,rr,dim,len(sel),nbr,scv,dbic))
    if not cand:return empty
    cand.sort(key=lambda z:(z[0],z[1],z[2]),reverse=True);cred,span,rr,dim,npnt,nbr,scv,dbic=cand[0];return {"dB":float(dim),"r2":float(rr),"span_decades":float(span),"n_points":int(npnt),"nb_ratio":float(nbr),"slope_cv":float(scv),"delta_bic":float(dbic),"credible":bool(cred)}

def crossover_from_deff(xscale,deff,r2loc,NB_c,N):
    x=np.asarray(xscale,float);d=np.asarray(deff,float);rr=np.asarray(r2loc,float);frac=np.asarray(NB_c,float)/float(N);m=min(len(x),len(frac));x,d,rr,frac=x[:m],d[:m],rr[:m],frac[:m];good=np.isfinite(x)&np.isfinite(d)&np.isfinite(rr)&(rr>=LOCAL_SLOPE_R2_MIN)&(d>0)&(frac>=.015)&(frac<=.85)
    if not np.any(good):return {"x_d_star":np.nan,"d_star":np.nan}
    ids=np.where(good)[0];idx=int(ids[np.argmax(d[ids])]);return {"x_d_star":float(x[idx]),"d_star":float(d[idx])}

def lacunarity_crossover(radii,lac,xi50,NB,N):
    r=np.asarray(radii,float);l=np.asarray(lac,float);frac=np.asarray(NB,float)/float(N)
    if not np.isfinite(xi50) or xi50<=0:return {"x_lac_star":np.nan,"lac_star":np.nan}
    x=r/xi50;good=np.isfinite(x)&np.isfinite(l)&(frac>=.015)&(frac<=.85)
    if not np.any(good):return {"x_lac_star":np.nan,"lac_star":np.nan}
    ids=np.where(good)[0];idx=int(ids[np.argmax(l[ids])]);return {"x_lac_star":float(x[idx]),"lac_star":float(l[idx])}

def local_metric_dimensions(D,node_measure,max_radii=36,max_nodes=LOCAL_DIMENSION_NODES):
    N=D.shape[0];delta=np.full(N,np.nan);rng=np.random.default_rng(S1_SEED+N);ids_eval=np.arange(N) if max_nodes>=N else np.sort(rng.choice(N,size=int(max_nodes),replace=False))
    for i in ids_eval:
        di=np.asarray(D[i],float);vals=np.unique(di[np.isfinite(di)&(di>0)])
        if len(vals)<LOCAL_COMMON_MIN_POINTS:continue
        if len(vals)>max_radii:vals=vals[np.unique(np.round(np.linspace(0,len(vals)-1,max_radii)).astype(int))]
        support=np.array([np.mean(di<=r) for r in vals]);mass=np.array([np.sum(node_measure[di<=r]) for r in vals]);good=(support>0)&(mass>0)&(support<LOCAL_COMMON_SATURATION)&(mass<LOCAL_COMMON_SATURATION)
        vals,support,mass=vals[good],support[good],mass[good]
        if len(vals)<LOCAL_COMMON_MIN_POINTS:continue
        best=None
        for a in range(len(vals)-LOCAL_COMMON_MIN_POINTS+1):
            for b in range(a+LOCAL_COMMON_MIN_POINTS,len(vals)+1):
                rr,ss,mm=vals[a:b],support[a:b],mass[a:b];span=np.log10(rr[-1]/rr[0])
                if span<LOCAL_COMMON_MIN_SPAN_DECADES or ss[-1]/max(ss[0],EPS)<LOCAL_COMMON_SUPPORT_GROWTH or mm[-1]/max(mm[0],EPS)<LOCAL_COMMON_MASS_GROWTH:continue
                x=np.log(rr);bd,ad=np.polyfit(x,np.log(ss),1);ba,aa=np.polyfit(x,np.log(mm),1);r2d=r2_score_np(np.log(ss),ad+bd*x);r2a=r2_score_np(np.log(mm),aa+ba*x)
                if bd>0 and ba>0 and r2d>=LOCAL_COMMON_R2_MIN and r2a>=LOCAL_COMMON_R2_MIN:
                    score=(span,min(r2d,r2a),len(rr));rec=ba-bd
                    if best is None or score>best[0]:best=(score,rec)
        if best is not None:delta[i]=best[1]
    return {"delta":delta,"valid_delta_fraction":float(np.mean(np.isfinite(delta[ids_eval]))),"local_nodes_evaluated":len(ids_eval)}

def nearest_resolution_indices(NB,N,target_fractions=RENORM_TARGET_BOX_FRACTIONS):
    frac=np.asarray(NB,float)/float(N);out=[]
    for t in target_fractions:
        elig=np.where((NB>=3)&(NB<=.90*N))[0]
        if not len(elig):continue
        idx=int(elig[np.argmin(np.abs(frac[elig]-t))]);
        if idx not in out:out.append(idx)
    return out

def mean_pairwise_ks(arrays):
    arr=[]
    for a in arrays:
        v=np.asarray(a,float);v=v[np.isfinite(v)&(v>0)]
        if len(v)>=5:arr.append(v)
    if len(arr)<2:return np.nan
    vals=[ks_2samp(arr[i],arr[j]).statistic for i in range(len(arr)) for j in range(i+1,len(arr))];return float(np.mean(vals))

def analyze_execution_geometry(model:ConvTraceNet,state:Dict[str,Any],n_scales:int,compute_local:bool=True)->Tuple[Dict[str,Any],pd.DataFrame]:
    Pgeom=execution_graph_from_occurrence(model,state["occurrence_flow"],True)
    if Pgeom.n_nodes>BCANW_MAX_NODES:return {"geometry_skipped":True,"geometry_skip_reason":f"n_nodes={Pgeom.n_nodes}>cap={BCANW_MAX_NODES}","n_graph_nodes":Pgeom.n_nodes},pd.DataFrame()
    Praw=execution_graph_from_occurrence(model,state["occurrence_flow"],False);D=distance_matrix(Pgeom,metric_lengths(Pgeom));radii=bcanw_scales(D,n_scales);order=np.lexsort((np.arange(Pgeom.n_nodes),-node_strength(Pgeom)));node_mu=global_node_measure(state["node_flow"]);NB=[];lac=[];qnorm=[];inter=[];coarse={q:[] for q in COARSE_RENYI_Q}
    for ell in radii:
        colors=greedy_bcanw_coloring(D,float(ell),order);NB.append(int(colors.max())+1);b=box_flow_observables(colors,node_mu,Praw);lac.append(b["lacunarity"]);qnorm.append(b["quotient_flow_norm"]);inter.append(b["interbox_flow_fraction"])
        for q in COARSE_RENYI_Q:coarse[q].append(b["coarse_eta_q"][q])
    NB=np.asarray(NB,float);lac=np.asarray(lac,float);inter=np.asarray(inter,float);rc,nc,lc=compress_box_staircase(radii,NB,lac);xi50=covering_quantile_scale(rc,nc,Pgeom.n_nodes,.5);x,deff,deffr2=local_effective_dimension(rc,nc,xi50);sw=strict_scaling_window(rc,nc,Pgeom.n_nodes);cross=crossover_from_deff(x,deff,deffr2,nc,Pgeom.n_nodes);lcross=lacunarity_crossover(radii,lac,xi50,NB,Pgeom.n_nodes);qidx=nearest_resolution_indices(NB,Pgeom.n_nodes);qks=mean_pairwise_ks([qnorm[i] for i in qidx]);ret=float(np.nanmean(inter[qidx])) if len(qidx) else np.nan;local_delta=np.nan;valid_local=np.nan;local_nodes=0
    if compute_local:
        ld=local_metric_dimensions(D,node_mu);v=np.isfinite(ld["delta"]);local_delta=float(np.nanmedian(ld["delta"][v])) if np.any(v) else np.nan;valid_local=ld["valid_delta_fraction"];local_nodes=ld["local_nodes_evaluated"]
    summ={"geometry_skipped":False,"geometry_skip_reason":"","n_graph_nodes":Pgeom.n_nodes,"xi50":float(xi50),"log_xi50":safe_log(xi50),"d_star":float(cross["d_star"]),"x_d_star":float(cross["x_d_star"]),"lacunarity_star":float(lcross["lac_star"]),"x_lacunarity_star":float(lcross["x_lac_star"]),"scaling_dB":float(sw["dB"]),"scaling_r2":float(sw["r2"]),"scaling_credible":bool(sw["credible"]),"scaling_span":float(sw["span_decades"]),"local_delta_median":float(local_delta),"valid_local_delta_fraction":float(valid_local),"local_nodes_evaluated":int(local_nodes),"quotient_mean_pairwise_KS":float(qks),"renorm_flow_retention_mean":float(ret)}
    rows=[];xb=radii/max(xi50,EPS) if np.isfinite(xi50) and xi50>0 else np.full(len(radii),np.nan)
    for k,ell in enumerate(radii):
        rec={"scale_index":k,"ell_B":float(ell),"ell_over_xi50":float(xb[k]),"NB":float(NB[k]),"NB_over_N":float(NB[k]/Pgeom.n_nodes),"lacunarity":float(lac[k]),"interbox_flow_fraction":float(inter[k])}
        for q in COARSE_RENYI_Q:
            rec[f"coarse_eta_q{q:g}"]=float(coarse[q][k])
        rows.append(rec)
    return summ,pd.DataFrame(rows)

# ==================================================================================================
# 5. EXACT PRE-FLIGHT CONTROLS
# ==================================================================================================

def _tiny_structural_enumeration(model:ConvTraceNet):
    # Explicitly enumerate structural paths for a one-hidden-layer tiny CNN [1->1->1].
    if model.channels!=[1,1,1]:raise ValueError("tiny enumeration expects one hidden channel")
    W=[c.weight.detach().cpu().numpy().astype(float) for c in model.convs];R=model.resolution;k1,k2=model.kernel_sizes;H1=R-k1+1;H2=H1-k2+1;g=1/(H2*H2);vals=[]
    for y2 in range(H2):
        for x2 in range(H2):
            for dy2 in range(k2):
                for dx2 in range(k2):
                    y1,x1=y2+dy2,x2+dx2
                    for dy1 in range(k1):
                        for dx1 in range(k1):
                            vals.append(float(W[0][0,0,dy1,dx1]**2*W[1][0,0,dy2,dx2]**2*g*g))
    q=np.asarray(vals,float);p=q/max(q.sum(),EPS);H=-np.sum(p[p>0]*np.log(p[p>0]));return q,H

def run_preflight_controls()->pd.DataFrame:
    path=OUTDIR/"tables"/"preflight_controls.csv"
    if S1_RESUME and path.exists():return pd.read_csv(path)
    rows=[]
    # Exact enumeration.
    model=ConvTraceNet(7,[1],[2,2],"relu",S1_SEED).to(DEVICE);st=cnn_structural_state(model,True);q,H=_tiny_structural_enumeration(model);rows.append({"control":"tiny_enumeration","partition_rel_error":abs(math.exp(st["logZ1"])-q.sum())/max(q.sum(),EPS),"entropy_abs_error":abs(st["S_path"]-H),"identity_residual":st["max_deficit_identity_residual"]})
    # Channel permutation and positive ReLU rescaling controls on C2-like network.
    model=ConvTraceNet(12,[3,4],[3,3,1],"relu",S1_SEED+1).to(DEVICE);base=cnn_structural_state(model,False);x=torch.randn(5,1,12,12,device=DEVICE);y0=model(x).detach()
    with torch.no_grad():
        p=torch.tensor([2,0,1],device=DEVICE);model.convs[0].weight.copy_(model.convs[0].weight[p]);model.convs[1].weight.copy_(model.convs[1].weight[:,p])
    perm=cnn_structural_state(model,False);y1=model(x).detach();rows.append({"control":"channel_permutation","function_max_abs_error":float((y1-y0).abs().max()),"entropy_abs_error":abs(perm["S_path"]-base["S_path"]),"deficit_abs_error":abs(perm["entropy_deficit"]-base["entropy_deficit"])})
    with torch.no_grad():
        c=2.3;model.convs[0].weight[0].mul_(c);model.convs[1].weight[:,0].div_(c)
    scale=cnn_structural_state(model,False);y2=model(x).detach();rows.append({"control":"positive_channel_rescaling","function_max_abs_error":float((y2-y1).abs().max()),"entropy_abs_error":abs(scale["S_path"]-perm["S_path"]),"deficit_abs_error":abs(scale["entropy_deficit"]-perm["entropy_deficit"])})
    # Analytic circular translation-sharing control: all occurrences of one shared parameter are equivalent.
    rng=np.random.default_rng(S1_SEED);k=3;R=7;w=rng.normal(size=(k,k));E=np.zeros((k,k,R,R),float)
    for dy in range(k):
        for dx in range(k):E[dy,dx,:,:]=w[dy,dx]**2
    E/=E.sum();P=E.sum(axis=(-1,-2));Hocc=-np.sum(E[E>0]*np.log(E[E>0]));Hpar=-np.sum(P[P>0]*np.log(P[P>0]));Hreuse=Hocc-Hpar;Sreuse=math.log(R*R)-Hreuse;rows.append({"control":"circular_translation_structural_reuse","S_reuse":Sreuse})
    df=pd.DataFrame(rows);atomic_csv(df,path);return df

# ==================================================================================================
# 6. ARCHITECTURE CONFIGURATION SPACE + 600 SOBOL + 400 RF-SMAC + 8 ANCHORS
# ==================================================================================================

def kernel_schedule(depth:int,family:str)->List[int]:
    hidden_layers=int(depth)-1
    if family=="all3":return [3]*hidden_layers+[1]
    if family=="alternating":return [3 if i%2==0 else 1 for i in range(hidden_layers)]+[1]
    if family=="spatial_then_pointwise":
        nsp=max(1,int(math.ceil(hidden_layers/2)));return [3]*nsp+[1]*(hidden_layers-nsp)+[1]
    raise ValueError(family)

def morphology_profile(n:int,morphology:str,rho:float,curve:float,center:float)->np.ndarray:
    if n<=0:return np.array([],float)
    t=np.linspace(0,1,n);rho=float(np.clip(rho,.05,.9));curve=float(np.clip(curve,.4,2.5));center=float(np.clip(center,.2,.8))
    if morphology=="constant":p=np.ones(n)
    elif morphology=="expansion":p=np.exp(rho*curve*(t-.5))
    elif morphology=="taper":p=np.exp(-rho*curve*(t-.5))
    elif morphology=="bottleneck":
        sigma=.10+.16/curve;p=1-rho*np.exp(-.5*((t-center)/sigma)**2)
    elif morphology=="hourglass":p=1-rho*np.power(np.clip(4*t*(1-t),0,1),curve)
    else:raise ValueError(morphology)
    p=np.clip(p,.18,None);p=p/np.exp(np.mean(np.log(p)));return p

def realize_architecture(morphology:str,depth:int,compute_factor:float,kernel_family:str,rho:float,curve:float,center:float)->Optional[Dict[str,Any]]:
    kernels=kernel_schedule(depth,kernel_family);profile=morphology_profile(depth-1,morphology,rho,curve,center);target=REFERENCE_MACS*float(compute_factor);best=None
    for base in range(MIN_CHANNELS,MAX_CHANNELS+1):
        hidden=np.clip(np.rint(base*profile),MIN_CHANNELS,MAX_CHANNELS).astype(int).tolist();desc=architecture_descriptors(hidden,kernels)
        if not desc.get("valid",False):continue
        err=abs(math.log(max(desc["macs"],1)/max(target,1)));rec=(err,hidden,desc)
        if best is None or rec[0]<best[0]:best=rec
    if best is None:return None
    err,hidden,desc=best
    # Keep realized architectures within a broad but controlled band; a nearest feasible architecture is
    # still allowed when rounding prevents exact 18% matching, but the mismatch is recorded.
    arch={"morphology":morphology,"depth":int(depth),"compute_factor_nominal":float(compute_factor),"kernel_family":kernel_family,"rho":float(rho),"curve":float(curve),"center":float(center),"hidden_channels":hidden,"kernel_sizes":kernels,"target_macs":float(target),"compute_log_error":float(err),**desc}
    arch["arch_key"]=stable_hash_dict({"hidden":hidden,"kernels":kernels},16);arch["arch_id"]="S1_"+arch["arch_key"];return arch

def anchor_rows()->List[Dict[str,Any]]:
    out=[]
    keys=list(ANCHOR_LIBRARY.keys())[:ANCHOR_TOTAL]
    for name in keys:
        sp=ANCHOR_LIBRARY[name];desc=architecture_descriptors(sp["hidden_channels"],sp["kernel_sizes"]);depth=len(sp["kernel_sizes"]);ratio=desc["macs"]/REFERENCE_MACS
        # assign nearest descriptive family without pretending the anchor was generated by it
        morph="bottleneck" if "bottleneck" in name.lower() else ("constant" if len(set(sp["hidden_channels"]))==1 else "anchor")
        kfam="all3" if all(k==3 for k in sp["kernel_sizes"][:-1]) else ("alternating" if 1 in sp["kernel_sizes"][:-1] else "anchor")
        rec={"morphology":morph,"depth":depth,"compute_factor_nominal":float(ratio),"kernel_family":kfam,"rho":np.nan,"curve":np.nan,"center":np.nan,"hidden_channels":sp["hidden_channels"],"kernel_sizes":sp["kernel_sizes"],"target_macs":float(desc["macs"]),"compute_log_error":0.,**desc,"arch_key":stable_hash_dict({"hidden":sp["hidden_channels"],"kernels":sp["kernel_sizes"]},16),"arch_id":name,"anchor_name":name,"stage":"anchor"}
        out.append(rec)
    return out

def config_vector(row:Dict[str,Any])->np.ndarray:
    morph=row.get("morphology","constant");midx=MORPHOLOGIES.index(morph) if morph in MORPHOLOGIES else -1;kfam=row.get("kernel_family","all3");kidx=KERNEL_FAMILIES.index(kfam) if kfam in KERNEL_FAMILIES else -1
    return np.array([float(row.get("depth",0))/max(DEPTH_VALUES),math.log(max(float(row.get("macs",1)),1))/20.0,math.log(max(float(row.get("parameter_count",1)),1))/15.0,float(midx)/max(len(MORPHOLOGIES)-1,1),float(kidx)/max(len(KERNEL_FAMILIES)-1,1),float(np.nan_to_num(row.get("rho",.5),nan=.5)),float(np.nan_to_num(row.get("curve",1),nan=1))/2.5,float(np.nan_to_num(row.get("center",.5),nan=.5))],float)

PHENOTYPE_COLS=["delta_entropy_deficit","delta_log_eta1","delta_log_eta2","delta_hidden_mean_occurrence_cv","delta_hidden_mean_parameter_cv","delta_hidden_mean_S_param","delta_hidden_mean_S_reuse"]

def phenotype_vector(row:Dict[str,Any])->np.ndarray:
    return np.array([float(row.get(c,np.nan)) for c in PHENOTYPE_COLS],float)

def candidate_cell(row:Dict[str,Any])->str:
    return f"{row.get('morphology')}|D{row.get('depth')}|B{row.get('compute_factor_nominal')}|{row.get('kernel_family')}"

def generate_sobol_candidates(existing_keys:set)->List[Dict[str,Any]]:
    sampler=qmc.Sobol(d=3,scramble=True,seed=S1_SEED);u=sampler.random_base2(int(math.ceil(math.log2(max(SOBOL_TOTAL,1)))))[:SOBOL_TOTAL];rows=[];j=0
    for morph in MORPHOLOGIES:
        for depth in DEPTH_VALUES:
            for cf in COMPUTE_FACTORS:
                for kfam in KERNEL_FAMILIES:
                    for rep in range(SOBOL_REPS_PER_CELL):
                        found=None
                        for retry in range(80):
                            if j<len(u) and retry==0:uu=u[j];j+=1
                            else:uu=np.random.default_rng(S1_SEED+j+retry).random(3)
                            rho=.12+.76*uu[0];curve=.5+1.8*uu[1];center=.25+.5*uu[2];# slight within-cell target jitter improves realized uniqueness
                            cf_eff=float(cf*np.exp((uu[2]-.5)*.12));cand=realize_architecture(morph,depth,cf_eff,kfam,rho,curve,center)
                            if cand is not None and cand["arch_key"] not in existing_keys:
                                found=cand;break
                        if found is not None:
                            found.update({"stage":"sobol","sobol_rep":rep,"cell":candidate_cell({**found,"compute_factor_nominal":cf})});found["compute_factor_stratum"]=cf;rows.append(found);existing_keys.add(found["arch_key"])
    return rows

def random_candidate_pool(n:int,seed:int,existing_keys:set)->List[Dict[str,Any]]:
    rng=np.random.default_rng(seed);rows=[];attempts=0
    while len(rows)<n and attempts<n*20:
        attempts+=1;morph=random.choice(MORPHOLOGIES);depth=random.choice(DEPTH_VALUES);cf=random.choice(COMPUTE_FACTORS);kfam=random.choice(KERNEL_FAMILIES);rho=rng.uniform(.12,.88);curve=rng.uniform(.5,2.3);center=rng.uniform(.25,.75);cf_eff=float(cf*np.exp(rng.uniform(-.08,.08)));cand=realize_architecture(morph,depth,cf_eff,kfam,rho,curve,center)
        if cand is None or cand["arch_key"] in existing_keys or any(cand["arch_key"]==x["arch_key"] for x in rows):continue
        cand.update({"stage":"bo_pool","compute_factor_stratum":cf,"cell":candidate_cell({**cand,"compute_factor_nominal":cf})});rows.append(cand)
    return rows

def expected_improvement_max(mu,sd,best,xi):
    sd=np.maximum(np.asarray(sd,float),1e-12);imp=np.asarray(mu,float)-float(best)-float(xi);z=imp/sd;return imp*norm.cdf(z)+sd*norm.pdf(z)

def config_novelty(cand:Dict[str,Any],archive:pd.DataFrame)->float:
    if archive.empty:return 1.0
    X=np.vstack([config_vector(r) for r in archive.to_dict("records")]);v=config_vector(cand);return float(np.min(np.linalg.norm(X-v[None,:],axis=1)))

def phenotype_novelty(rec:Dict[str,Any],archive:pd.DataFrame)->float:
    if archive.empty:return 1.0
    valid=archive.dropna(subset=PHENOTYPE_COLS)
    v=phenotype_vector(rec)
    if len(valid)<3 or not np.all(np.isfinite(v)):return 1.0
    X=np.vstack([phenotype_vector(r) for r in valid.to_dict("records")]);sc=RobustScaler().fit(X);Xs=sc.transform(X);vs=sc.transform(v[None,:])[0];d=np.linalg.norm(Xs-vs[None,:],axis=1);k=min(RFSMAC_KNN_NOVELTY,len(d));return float(np.median(np.partition(d,k-1)[:k]))

def coverage_reward(cand:Dict[str,Any],archive:pd.DataFrame)->float:
    if archive.empty:return 1.0
    cell=cand.get("cell",candidate_cell(cand));n=int(np.sum(archive.get("cell",pd.Series(dtype=str)).astype(str)==str(cell)));return float(1/math.sqrt(1+n))

def normalized_component(value:float,reference:Sequence[float])->float:
    x=np.asarray(reference,float);x=x[np.isfinite(x)]
    if not len(x):return float(value)
    med=np.median(x);iqr=np.quantile(x,.75)-np.quantile(x,.25);return float((value-med)/max(iqr,1e-8))

# ==================================================================================================
# 7. DISCOVERY TRAINING, STRUCTURAL PHENOTYPE AND RESUMABLE RF-SMAC ARCHIVE
# ==================================================================================================

def discovery_checkpoint_path(arch_id:str)->Path:return OUTDIR/"checkpoints_discovery"/f"{arch_id}__epoch{DISCOVERY_EPOCHS}.pt"

def discovery_train_candidate(cand:Dict[str,Any],splits:Dict[str,Any])->Dict[str,Any]:
    arch_id=str(cand["arch_id"]);hidden=cand["hidden_channels"] if isinstance(cand["hidden_channels"],list) else json.loads(cand["hidden_channels"]);kernels=cand["kernel_sizes"] if isinstance(cand["kernel_sizes"],list) else json.loads(cand["kernel_sizes"]);seed=model_seed(arch_id,0);model=ConvTraceNet(RESOLUTION,hidden,kernels,ACTIVATION,seed).to(DEVICE);Xtr,ytr=splits["train"];Xv,yv=splits["val"];calibrate_initial_signal(model,Xtr[:min(256,len(Xtr))]);init=cnn_structural_state(model,False);initial_val=eval_mse(model,Xv[:DISCOVERY_N_VAL],yv[:DISCOVERY_N_VAL]);opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY);diverged=False;train_last=np.nan
    for epoch in range(1,DISCOVERY_EPOCHS+1):
        model.train();losses=[]
        for ids in deterministic_batches(DISCOVERY_N_TRAIN,epoch-1,0,BATCH_SIZE):
            ids_t=torch.as_tensor(ids,device=DEVICE);xb=Xtr.index_select(0,ids_t) if Xtr.device.type=="cuda" else Xtr[ids].to(DEVICE);yb=ytr.index_select(0,ids_t) if ytr.device.type=="cuda" else ytr[ids].to(DEVICE);opt.zero_grad(set_to_none=True);pred=model(xb);loss=F.mse_loss(pred,yb)
            if not torch.isfinite(loss):diverged=True;break
            loss.backward();opt.step();losses.append(float(loss.item()))
        if diverged:break
        train_last=float(np.mean(losses))
    final=cnn_structural_state(model,False) if not diverged else {k:np.nan for k in init};final_val=eval_mse(model,Xv[:DISCOVERY_N_VAL],yv[:DISCOVERY_N_VAL]) if not diverged else np.inf
    if not diverged:
        torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"arch":cand,"calibrated_seed":seed},discovery_checkpoint_path(arch_id))
    rec={**cand,"hidden_channels":json.dumps(hidden),"kernel_sizes":json.dumps(kernels),"diverged":diverged,"initial_val_mse":initial_val,"final_val_mse":final_val,"final_train_mse":train_last}
    for key in ["S_path","entropy_deficit","eta1","eta2","hidden_mean_S_occ","hidden_mean_S_param","hidden_mean_S_reuse","hidden_mean_occurrence_cv","hidden_mean_parameter_cv"]:
        rec[f"init_{key}"]=float(init.get(key,np.nan));rec[f"final_{key}"]=float(final.get(key,np.nan))
    rec["delta_entropy_deficit"]=rec["final_entropy_deficit"]-rec["init_entropy_deficit"];rec["delta_log_eta1"]=safe_log(rec["final_eta1"])-safe_log(rec["init_eta1"]);rec["delta_log_eta2"]=safe_log(rec["final_eta2"])-safe_log(rec["init_eta2"])
    for key in ["hidden_mean_occurrence_cv","hidden_mean_parameter_cv","hidden_mean_S_occ","hidden_mean_S_param","hidden_mean_S_reuse"]:rec[f"delta_{key}"]=rec[f"final_{key}"]-rec[f"init_{key}"]
    del model,opt
    return rec

def add_discovery_scores(rec:Dict[str,Any],archive:pd.DataFrame)->Dict[str,Any]:
    pn=phenotype_novelty(rec,archive);cn=config_novelty(rec,archive);cr=coverage_reward(rec,archive);# robustly place components on comparable positive scales
    p_ref=archive.get("phenotype_novelty",pd.Series(dtype=float)).to_numpy(float) if not archive.empty else []
    c_ref=archive.get("config_novelty",pd.Series(dtype=float)).to_numpy(float) if not archive.empty else []
    pn_pos=max(0.,normalized_component(pn,p_ref)+1.5) if len(p_ref) else pn;cn_pos=max(0.,normalized_component(cn,c_ref)+1.5) if len(c_ref) else cn
    rec["phenotype_novelty"]=pn;rec["config_novelty"]=cn;rec["coverage_reward"]=cr;rec["discovery_score"]=.55*pn_pos+.25*cn_pos+.20*cr;return rec

def run_architecture_discovery(splits:Dict[str,Any])->pd.DataFrame:
    path=OUTDIR/"tables"/"architecture_discovery.csv";archive=safe_read(path) if S1_RESUME else pd.DataFrame();existing=set(archive.arch_key.astype(str)) if not archive.empty else set()
    # Pre-reserve anchor realized keys so Sobol/BO cannot accidentally duplicate them.
    anchors=anchor_rows();anchor_keys={r["arch_key"] for r in anchors};gen_existing=set(existing)|anchor_keys
    # Sobol map.
    sobol=generate_sobol_candidates(gen_existing);needed=[c for c in sobol if c["arch_key"] not in existing]
    eta=ETAReporter("S1-A stratified Sobol discovery",len(needed));done=0
    for cand in tqdm(needed,desc="CNN-S1 Sobol architectures"):
        t0=time.perf_counter();rec=discovery_train_candidate(cand,splits);rec=add_discovery_scores(rec,archive);archive=pd.concat([archive,pd.DataFrame([rec])],ignore_index=True);atomic_csv(archive,path);existing.add(rec["arch_key"]);done+=1;eta.update(time.perf_counter()-t0,done)
    eta.finish()
    # Rounding and graph-feasibility constraints can occasionally collapse two Sobol configurations to the
    # same realized CNN. Top up to the frozen Sobol count with configuration-space random fills, still
    # labelled as the space-filling stage, so the scientific accounting remains exactly 600/400/8.
    sobol_now=int(np.sum(archive.stage.astype(str)=="sobol")) if not archive.empty else 0
    if sobol_now < SOBOL_TOTAL:
        fill_needed=SOBOL_TOTAL-sobol_now; eta_fill=ETAReporter("S1-A Sobol uniqueness top-up",fill_needed);fdone=0
        while sobol_now < SOBOL_TOTAL:
            pool=random_candidate_pool(max(32,fill_needed*4),S1_SEED+91_000+sobol_now,existing)
            if not pool: raise RuntimeError("Unable to top up unique Sobol architecture map")
            cand=pool[0];cand["stage"]="sobol";cand["sobol_rep"]=-1;cand["cell"]=candidate_cell(cand)
            tfill=time.perf_counter();rec=discovery_train_candidate(cand,splits);rec=add_discovery_scores(rec,archive);archive=pd.concat([archive,pd.DataFrame([rec])],ignore_index=True);atomic_csv(archive,path);existing.add(rec["arch_key"]);sobol_now+=1;fdone+=1;eta_fill.update(time.perf_counter()-tfill,fdone)
        eta_fill.finish()
    # Anchors are evaluated before BO so the surrogate sees known programme reference points.
    pending=[a for a in anchors if a["arch_key"] not in existing];eta=ETAReporter("S1 anchor evaluations",len(pending));done=0
    for cand in tqdm(pending,desc="CNN-S1 anchors"):
        t0=time.perf_counter();cand["cell"]="anchor|"+cand["arch_id"];rec=discovery_train_candidate(cand,splits);rec=add_discovery_scores(rec,archive);archive=pd.concat([archive,pd.DataFrame([rec])],ignore_index=True);atomic_csv(archive,path);existing.add(rec["arch_key"]);done+=1;eta.update(time.perf_counter()-t0,done)
    eta.finish()
    # RF-SMAC sequential refinement.
    current_bo=int(np.sum(archive.stage.astype(str)=="bo")) if not archive.empty and "stage" in archive else 0;target=RFSMAC_STEPS;eta=ETAReporter("S1-B RF-SMAC refinement",max(target-current_bo,0));completed=0
    while current_bo<target:
        t0=time.perf_counter();pool=random_candidate_pool(RFSMAC_POOL,S1_SEED+50_000+current_bo,existing)
        if not pool:raise RuntimeError("RF-SMAC candidate pool exhausted")
        valid=archive[(~archive.diverged.astype(bool))&np.isfinite(archive.discovery_score.astype(float))].copy()
        # 25% random/coverage interleaving. Otherwise RF expected improvement on accumulated discovery score.
        if len(valid)<20 or random.random()<RFSMAC_RANDOM_INTERLEAVE:
            weights=np.array([coverage_reward(c,archive) for c in pool],float);weights/=weights.sum();cand=pool[int(np.random.default_rng(S1_SEED+current_bo).choice(len(pool),p=weights))];acq=np.nan;mode="random_coverage"
        else:
            X=np.vstack([config_vector(r) for r in valid.to_dict("records")]);y=valid.discovery_score.to_numpy(float);rf=RandomForestRegressor(n_estimators=RFSMAC_RF_TREES,min_samples_leaf=2,max_features="sqrt",random_state=S1_SEED+current_bo,n_jobs=-1);rf.fit(X,y);Xp=np.vstack([config_vector(c) for c in pool]);pred=np.vstack([t.predict(Xp) for t in rf.estimators_]);mu=pred.mean(0);sd=pred.std(0);ei=expected_improvement_max(mu,sd,float(np.max(y)),RFSMAC_EI_XI);cn=np.array([config_novelty(c,archive) for c in pool]);cr=np.array([coverage_reward(c,archive) for c in pool]);score=ei+.12*cn+.08*cr;idx=int(np.nanargmax(score));cand=pool[idx];acq=float(score[idx]);mode="rf_ei"
        cand["stage"]="bo";cand["rfsmac_mode"]=mode;cand["rfsmac_acquisition"]=acq;rec=discovery_train_candidate(cand,splits);rec=add_discovery_scores(rec,archive);archive=pd.concat([archive,pd.DataFrame([rec])],ignore_index=True);atomic_csv(archive,path);existing.add(rec["arch_key"]);current_bo+=1;completed+=1;eta.update(time.perf_counter()-t0,completed)
        if completed%20==0:gpu_snapshot("RF-SMAC")
    eta.finish();archive["discovery_eval_index"]=np.arange(len(archive));atomic_csv(archive,path);return archive

# ==================================================================================================
# 8. DISCOVERY-GEOMETRY SUBSET AND MAXIMIN COHORT SELECTION
# ==================================================================================================

def maximin_select(df:pd.DataFrame,n:int,feature_cols:Optional[List[str]]=None,force_ids:Optional[List[str]]=None)->pd.DataFrame:
    d=df.copy().reset_index(drop=True);force_ids=force_ids or []
    if feature_cols is None:
        cfg=np.vstack([config_vector(r) for r in d.to_dict("records")]);phen=d[PHENOTYPE_COLS].to_numpy(float);phen=np.nan_to_num(RobustScaler().fit_transform(np.nan_to_num(phen,nan=np.nanmedian(phen,axis=0))),nan=0);X=np.hstack([cfg,phen])
    else:
        X=d[feature_cols].to_numpy(float);X=RobustScaler().fit_transform(np.nan_to_num(X,nan=np.nanmedian(X,axis=0)))
    selected=[]
    for aid in force_ids:
        ids=np.where(d.arch_id.astype(str).to_numpy()==str(aid))[0]
        if len(ids) and ids[0] not in selected:selected.append(int(ids[0]))
    if not selected:
        # begin with point farthest from global centre
        center=np.nanmedian(X,axis=0);selected=[int(np.argmax(np.linalg.norm(X-center[None,:],axis=1)))]
    while len(selected)<min(n,len(d)):
        dist=np.min(np.linalg.norm(X[:,None,:]-X[np.array(selected)][None,:,:],axis=2),axis=1);dist[selected]=-np.inf;selected.append(int(np.argmax(dist)))
    return d.iloc[selected[:n]].copy().reset_index(drop=True)

def select_discovery_geometry(archive:pd.DataFrame)->pd.DataFrame:
    path=OUTDIR/"tables"/"discovery_geometry_panel.csv"
    if S1_RESUME and path.exists():return pd.read_csv(path)
    viable=archive[(~archive.diverged.astype(bool))&archive.arch_key.notna()].copy();force=[x for x in ANCHOR_LIBRARY if x in set(viable.arch_id.astype(str))];sel=maximin_select(viable,DISCOVERY_GEOMETRY_N,force_ids=force);atomic_csv(sel,path);return sel

def discovery_geometry(panel:pd.DataFrame,splits:Dict[str,Any])->Tuple[pd.DataFrame,pd.DataFrame]:
    spath=OUTDIR/"tables"/"discovery_geometry_summary.csv";cpath=OUTDIR/"tables"/"discovery_geometry_scales.csv";old=safe_read(spath) if S1_RESUME else pd.DataFrame();oldsc=safe_read(cpath) if S1_RESUME else pd.DataFrame();done={(str(r.arch_id),int(r.epoch)) for _,r in old.iterrows()} if not old.empty else set();rows=old.to_dict("records");scrows=oldsc.to_dict("records");total=sum((str(r.arch_id),e) not in done for _,r in panel.iterrows() for e in [0,DISCOVERY_EPOCHS]);eta=ETAReporter("S1 discovery BCANw",total);completed=0;Xtr=splits["train"] [0]
    for _,r in tqdm(panel.iterrows(),total=len(panel),desc="S1 discovery geometry architectures"):
        hidden=json.loads(r.hidden_channels) if isinstance(r.hidden_channels,str) else r.hidden_channels;kernels=json.loads(r.kernel_sizes) if isinstance(r.kernel_sizes,str) else r.kernel_sizes;seed=model_seed(str(r.arch_id),0)
        for epoch in [0,DISCOVERY_EPOCHS]:
            if (str(r.arch_id),epoch) in done:continue
            t0=time.perf_counter();model=ConvTraceNet(RESOLUTION,hidden,kernels,ACTIVATION,seed).to(DEVICE);calibrate_initial_signal(model,Xtr[:min(256,len(Xtr))])
            if epoch==DISCOVERY_EPOCHS:
                cp=discovery_checkpoint_path(str(r.arch_id));chk=torch.load(cp,map_location="cpu");model.load_state_dict(chk["state_dict"]);model.to(DEVICE)
            st=cnn_structural_state(model,True);gs,gsc=analyze_execution_geometry(model,st,BCANW_SCALES_DISCOVERY,compute_local=True);rec={"arch_id":r.arch_id,"arch_key":r.arch_key,"epoch":epoch,"stage":r.stage,"morphology":r.morphology,"kernel_family":r.kernel_family,"depth":r.depth,"macs":r.macs,"parameter_count":r.parameter_count,**gs};rows.append(rec)
            if not gsc.empty:
                for k,v in rec.items():
                    if k not in gsc.columns:gsc[k]=v
                scrows.extend(gsc.to_dict("records"))
            atomic_csv(pd.DataFrame(rows),spath);atomic_csv(pd.DataFrame(scrows),cpath);done.add((str(r.arch_id),epoch));completed+=1;eta.update(time.perf_counter()-t0,completed);del model
    eta.finish();return pd.DataFrame(rows),pd.DataFrame(scrows)

def select_longitudinal_cohorts(archive:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    p48=OUTDIR/"tables"/"longitudinal_48_architectures.csv";p12=OUTDIR/"tables"/"s2_12_architecture_panel.csv";p6=OUTDIR/"tables"/"s3_6_architecture_sentinels.csv"
    if S1_RESUME and p48.exists() and p12.exists() and p6.exists():return pd.read_csv(p48),pd.read_csv(p12),pd.read_csv(p6)
    viable=archive[(~archive.diverged.astype(bool))].dropna(subset=PHENOTYPE_COLS).copy();force=[x for x in ["C1_shallow3","C2_medium3","C3_bottleneck","C4_pointwise","C5_deep3","C6_large5"] if x in set(viable.arch_id.astype(str))];cohort=maximin_select(viable,N_LONG_ARCH,force_ids=force);panel12=maximin_select(cohort,N_S2_PANEL,force_ids=[x for x in force if x in set(cohort.arch_id)]);sent6=maximin_select(panel12,N_S3_SENTINELS,force_ids=[x for x in force if x in set(panel12.arch_id)]);atomic_csv(cohort,p48);atomic_csv(panel12,p12);atomic_csv(sent6,p6);return cohort,panel12,sent6

# ==================================================================================================
# 9. FULL 48-ARCHITECTURE x 2-SEED LONGITUDINAL COHORT
# ==================================================================================================

def long_run_id(arch_id:str,seed:int)->str:return f"S1LONG__{arch_id}__S{seed:02d}"
def long_part(run_id:str,kind:str)->Path:return OUTDIR/"run_parts"/f"{run_id}__{kind}.csv"
def long_checkpoint(run_id:str,epoch:int)->Path:return OUTDIR/"checkpoints_longitudinal"/f"{run_id}__E{epoch:03d}.pt"

def long_complete(run_id:str)->bool:
    return all(long_part(run_id,k).exists() for k in ["states","layers","history","summary"]) and all(long_checkpoint(run_id,e).exists() for e in STATE_GRID)

def record_structural_checkpoint(model:ConvTraceNet,epoch:int,meta:Dict[str,Any]):
    st=cnn_structural_state(model,True);row={**meta,"epoch":int(epoch)}
    for k,v in st.items():
        if k not in {"node_flow","occurrence_flow","parameter_flow","layer_metrics"}:row[k]=v
    layers=[]
    for l,d in enumerate(st["layer_metrics"]):layers.append({**meta,"epoch":int(epoch),"layer":l,**d})
    return row,pd.DataFrame(layers)

def train_longitudinal_run(r:pd.Series,seed_index:int,splits:Dict[str,Any]):
    aid=str(r.arch_id);rid=long_run_id(aid,seed_index)
    if S1_RESUME and long_complete(rid):return
    hidden=json.loads(r.hidden_channels) if isinstance(r.hidden_channels,str) else r.hidden_channels;kernels=json.loads(r.kernel_sizes) if isinstance(r.kernel_sizes,str) else r.kernel_sizes;seed=model_seed(aid,seed_index);model=ConvTraceNet(RESOLUTION,hidden,kernels,ACTIVATION,seed).to(DEVICE);Xtr,ytr=splits["train"];Xv,yv=splits["val"];Xt,yt=splits["test"];cal_scales=calibrate_initial_signal(model,Xtr[:min(256,len(Xtr))]);meta={"run_id":rid,"arch_id":aid,"arch_key":str(r.arch_key),"seed_index":int(seed_index),"morphology":str(r.morphology),"kernel_family":str(r.kernel_family),"depth":int(r.depth),"macs":float(r.macs),"parameter_count":int(r.parameter_count),"occurrence_count":int(r.occurrence_count),"execution_nodes":int(r.execution_nodes),"receptive_field":int(r.receptive_field),"output_resolution":int(r.output_resolution),"mean_hidden_width":float(r.mean_hidden_width),"max_hidden_width":int(r.max_hidden_width)}
    states=[];layer_frames=[];row,ldf=record_structural_checkpoint(model,0,meta);states.append(row);layer_frames.append(ldf);torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"meta":meta,"calibration_scales":cal_scales},long_checkpoint(rid,0))
    opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY);history=[{**meta,"epoch":0,"train_mse":np.nan,"val_mse":eval_mse(model,Xv,yv)}];best_val=history[-1]["val_mse"];best_epoch=0;diverged=False
    epoch_bar=tqdm(range(1,FULL_EPOCHS+1),desc=f"{aid} s{seed_index}",leave=False)
    for epoch in epoch_bar:
        model.train();losses=[]
        for ids in deterministic_batches(len(Xtr),epoch-1,seed_index,BATCH_SIZE):
            ids_t=torch.as_tensor(ids,device=DEVICE);xb=Xtr.index_select(0,ids_t) if Xtr.device.type=="cuda" else Xtr[ids].to(DEVICE);yb=ytr.index_select(0,ids_t) if ytr.device.type=="cuda" else ytr[ids].to(DEVICE);opt.zero_grad(set_to_none=True);pred=model(xb);loss=F.mse_loss(pred,yb)
            if not torch.isfinite(loss):diverged=True;break
            loss.backward();opt.step();losses.append(float(loss.item()))
        if diverged:break
        train_mse=float(np.mean(losses));val=np.nan
        if epoch in STATE_GRID or epoch%10==0 or epoch==FULL_EPOCHS:
            val=eval_mse(model,Xv,yv)
            if val<best_val:best_val=float(val);best_epoch=epoch
        history.append({**meta,"epoch":epoch,"train_mse":train_mse,"val_mse":val})
        if epoch in STATE_GRID:
            row,ldf=record_structural_checkpoint(model,epoch,meta);states.append(row);layer_frames.append(ldf);torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"meta":meta,"calibration_scales":cal_scales},long_checkpoint(rid,epoch));atomic_csv(pd.DataFrame(states),long_part(rid,"states"));atomic_csv(pd.concat(layer_frames,ignore_index=True),long_part(rid,"layers"));atomic_csv(pd.DataFrame(history),long_part(rid,"history"))
        if epoch%40==0:epoch_bar.set_postfix(train=f"{train_mse:.3g}",val=f"{val:.3g}" if np.isfinite(val) else "-")
    final_test=eval_mse(model,Xt,yt) if not diverged else np.inf;summary={**meta,"diverged":bool(diverged),"best_val_mse":float(best_val),"best_epoch":int(best_epoch),"final_test_mse":float(final_test),"epochs_completed":int(history[-1]["epoch"]),"calibration_scales_json":json.dumps([float(x) for x in cal_scales])};atomic_csv(pd.DataFrame(states),long_part(rid,"states"));atomic_csv(pd.concat(layer_frames,ignore_index=True),long_part(rid,"layers"));atomic_csv(pd.DataFrame(history),long_part(rid,"history"));atomic_csv(pd.DataFrame([summary]),long_part(rid,"summary"));del model,opt

def run_longitudinal_cohort(cohort:pd.DataFrame,splits:Dict[str,Any]):
    jobs=[(r,s) for _,r in cohort.iterrows() for s in LONGITUDINAL_SEEDS if not (S1_RESUME and long_complete(long_run_id(str(r.arch_id),s)))];eta=ETAReporter("S1-C/D full longitudinal trajectories",len(jobs));done=0
    for r,s in tqdm(jobs,desc="CNN-S1 full longitudinal runs"):
        t0=time.perf_counter();train_longitudinal_run(r,s,splits);done+=1;eta.update(time.perf_counter()-t0,done)
        if done%4==0:gpu_snapshot("longitudinal")
    eta.finish()

def aggregate_longitudinal(cohort:pd.DataFrame)->Dict[str,pd.DataFrame]:
    kinds={k:[] for k in ["states","layers","history","summary"]}
    for _,r in cohort.iterrows():
        for s in LONGITUDINAL_SEEDS:
            rid=long_run_id(str(r.arch_id),s)
            for k in kinds:
                d=safe_read(long_part(rid,k))
                if not d.empty:kinds[k].append(d)
    out={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in kinds.items()}
    for k,d in out.items():atomic_csv(d,OUTDIR/"tables"/f"longitudinal_{k}.csv")
    return out

# ==================================================================================================
# 10. FULL LONGITUDINAL BCANw AT EXACT COMMON GEOMETRY CHECKPOINTS
# ==================================================================================================

def run_longitudinal_geometry(cohort:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    spath=OUTDIR/"tables"/"longitudinal_geometry.csv";cpath=OUTDIR/"tables"/"longitudinal_geometry_scales.csv";old=safe_read(spath) if S1_RESUME else pd.DataFrame();oldsc=safe_read(cpath) if S1_RESUME else pd.DataFrame();done={(str(r.run_id),int(r.epoch)) for _,r in old.iterrows()} if not old.empty else set();rows=old.to_dict("records");scrows=oldsc.to_dict("records");jobs=[]
    for _,r in cohort.iterrows():
        for s in LONGITUDINAL_SEEDS:
            rid=long_run_id(str(r.arch_id),s)
            for e in GEOMETRY_GRID:
                if e<=FULL_EPOCHS and (rid,e) not in done:jobs.append((r,s,e))
    eta=ETAReporter("S1 longitudinal BCANw",len(jobs));completed=0
    for r,s,e in tqdm(jobs,desc="CNN-S1 longitudinal BCANw"):
        t0=time.perf_counter();rid=long_run_id(str(r.arch_id),s);hidden=json.loads(r.hidden_channels) if isinstance(r.hidden_channels,str) else r.hidden_channels;kernels=json.loads(r.kernel_sizes) if isinstance(r.kernel_sizes,str) else r.kernel_sizes;model=ConvTraceNet(RESOLUTION,hidden,kernels,ACTIVATION,model_seed(str(r.arch_id),s)).to(DEVICE);chk=torch.load(long_checkpoint(rid,e),map_location="cpu");model.load_state_dict(chk["state_dict"]);model.to(DEVICE);st=cnn_structural_state(model,True);gs,gsc=analyze_execution_geometry(model,st,BCANW_SCALES_LONG,compute_local=True);rec={"run_id":rid,"arch_id":str(r.arch_id),"arch_key":str(r.arch_key),"seed_index":s,"epoch":e,"morphology":str(r.morphology),"kernel_family":str(r.kernel_family),"depth":int(r.depth),"macs":float(r.macs),"parameter_count":int(r.parameter_count),"occurrence_count":int(r.occurrence_count),"execution_nodes":int(r.execution_nodes),**gs};rows.append(rec)
        if not gsc.empty:
            for k,v in rec.items():
                if k not in gsc.columns:gsc[k]=v
            scrows.extend(gsc.to_dict("records"))
        atomic_csv(pd.DataFrame(rows),spath);atomic_csv(pd.DataFrame(scrows),cpath);done.add((rid,e));completed+=1;eta.update(time.perf_counter()-t0,completed);del model
        if torch.cuda.is_available() and completed%8==0:gpu_snapshot("BCANw (CPU-heavy)")
    eta.finish();return pd.DataFrame(rows),pd.DataFrame(scrows)

# ==================================================================================================
# 11. PLANNED ARCHITECTURE-LEVEL INFERENCE AND EXPLICIT POST-HOC ANALYSES
# ==================================================================================================

STATE_PRIMARY_FAMILIES={
    "measure_flow":["entropy_deficit","log_eta1","log_eta2","hidden_mean_occurrence_cv","hidden_mean_parameter_cv"],
    "sharing":["hidden_mean_S_occ","hidden_mean_S_param","hidden_mean_S_reuse"],
}
GEOM_PRIMARY=["log_xi50"]
GEOM_SECONDARY=["d_star","lacunarity_star","local_delta_median","quotient_mean_pairwise_KS","renorm_flow_retention_mean"]

# Raw tables store eta1/eta2 and xi50. Log quantities are analysis coordinates and must be derived
# consistently before *every* analysis path, including seed-level analyses. The original failure occurred
# because seed_reproducibility() received the raw state table but requested the derived column log_eta1.
def add_state_analysis_columns(states:pd.DataFrame)->pd.DataFrame:
    if states.empty:
        return states.copy()
    s=states.copy()
    if "log_eta1" not in s.columns:
        if "eta1" not in s.columns:
            raise KeyError("State table has neither 'eta1' nor derived 'log_eta1'.")
        s["log_eta1"]=np.log(np.clip(pd.to_numeric(s["eta1"],errors="coerce"),1e-300,None))
    if "log_eta2" not in s.columns:
        if "eta2" not in s.columns:
            raise KeyError("State table has neither 'eta2' nor derived 'log_eta2'.")
        s["log_eta2"]=np.log(np.clip(pd.to_numeric(s["eta2"],errors="coerce"),1e-300,None))
    return s

def add_geometry_analysis_columns(geometry:pd.DataFrame)->pd.DataFrame:
    if geometry.empty:
        return geometry.copy()
    g=geometry.copy()
    if "log_xi50" not in g.columns:
        if "xi50" not in g.columns:
            raise KeyError("Geometry table has neither 'xi50' nor derived 'log_xi50'.")
        g["log_xi50"]=np.log(np.clip(pd.to_numeric(g["xi50"],errors="coerce"),1e-300,None))
    return g

def require_columns(df:pd.DataFrame,cols:Sequence[str],context:str):
    missing=[c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{context}: missing required columns {missing}. Available columns: {list(df.columns)}")

def prepare_architecture_level(states:pd.DataFrame,geometry:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    s=add_state_analysis_columns(states);idcols=["arch_id","epoch"]
    require_columns(s,idcols+sum(STATE_PRIMARY_FAMILIES.values(),[]),"prepare_architecture_level(states)")
    num=[c for c in s.columns if c not in ["run_id","arch_key","morphology","kernel_family"]+idcols and pd.api.types.is_numeric_dtype(s[c])]
    sa=s.groupby(idcols,as_index=False)[num].median(numeric_only=True)
    if not geometry.empty:
        g=add_geometry_analysis_columns(geometry)
        require_columns(g,idcols+GEOM_PRIMARY+GEOM_SECONDARY,"prepare_architecture_level(geometry)")
        gn=[c for c in g.columns if c not in ["run_id","arch_key","morphology","kernel_family","geometry_skip_reason"]+idcols and pd.api.types.is_numeric_dtype(g[c])]
        ga=g.groupby(idcols,as_index=False)[gn].median(numeric_only=True)
    else:ga=pd.DataFrame()
    atomic_csv(sa,OUTDIR/"tables"/"architecture_level_states.csv");atomic_csv(ga,OUTDIR/"tables"/"architecture_level_geometry.csv");return sa,ga

def planned_change_table(arch_states:pd.DataFrame,arch_geom:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for fam,metrics in STATE_PRIMARY_FAMILIES.items():
        for metric in metrics:
            if metric not in arch_states.columns: raise KeyError(f"planned_change_table: missing state metric {metric}")
            piv=arch_states.pivot(index="arch_id",columns="epoch",values=metric)
            if 0 not in piv.columns or FULL_EPOCHS not in piv.columns:continue
            d=(piv[FULL_EPOCHS]-piv[0]).dropna().to_numpy(float);n=len(d)
            if not n:continue
            try:stat,p=wilcoxon(d,zero_method="wilcox")
            except Exception:stat,p=np.nan,1.
            lo,hi=bootstrap_median_ci(d,S1_SEED+len(rows));k=int(np.sum(d>0));slo,shi=clopper_pearson(k,n);rows.append({"family":fam,"metric":metric,"n_architectures":n,"median_change":float(np.median(d)),"hodges_lehmann":hodges_lehmann_onesample(d),"rank_biserial":paired_rank_biserial(d),"wilcoxon_stat":stat,"p_raw":p,"bootstrap_median_lo":lo,"bootstrap_median_hi":hi,"positive_fraction":k/n,"positive_fraction_lo":slo,"positive_fraction_hi":shi})
    if not arch_geom.empty:
        for metric in GEOM_PRIMARY:
            if metric not in arch_geom.columns: raise KeyError(f"planned_change_table: missing geometry metric {metric}")
            piv=arch_geom.pivot(index="arch_id",columns="epoch",values=metric)
            if 0 in piv.columns and FULL_EPOCHS in piv.columns:
                d=(piv[FULL_EPOCHS]-piv[0]).dropna().to_numpy(float);n=len(d)
                if n:
                    try:stat,p=wilcoxon(d,zero_method="wilcox")
                    except Exception:stat,p=np.nan,1.
                    lo,hi=bootstrap_median_ci(d,S1_SEED+len(rows));k=int(np.sum(d>0));slo,shi=clopper_pearson(k,n);rows.append({"family":"geometry_primary","metric":metric,"n_architectures":n,"median_change":float(np.median(d)),"hodges_lehmann":hodges_lehmann_onesample(d),"rank_biserial":paired_rank_biserial(d),"wilcoxon_stat":stat,"p_raw":p,"bootstrap_median_lo":lo,"bootstrap_median_hi":hi,"positive_fraction":k/n,"positive_fraction_lo":slo,"positive_fraction_hi":shi})
    df=pd.DataFrame(rows)
    if df.empty:
        df=pd.DataFrame(columns=["family","metric","n_architectures","median_change","hodges_lehmann","rank_biserial","wilcoxon_stat","p_raw","bootstrap_median_lo","bootstrap_median_hi","positive_fraction","positive_fraction_lo","positive_fraction_hi","p_holm"])
    else:
        df["p_holm"]=np.nan
        for fam,g in df.groupby("family"):df.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    atomic_csv(df,OUTDIR/"tables"/"planned_architecture_change_inference.csv");return df

def temporal_friedman(arch_states:pd.DataFrame,arch_geom:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for source,df,metrics in [("state",arch_states,sum(STATE_PRIMARY_FAMILIES.values(),[])),("geometry",arch_geom,GEOM_PRIMARY+GEOM_SECONDARY)]:
        if df.empty:continue
        for metric in metrics:
            if metric not in df.columns: raise KeyError(f"temporal_friedman({source}): missing metric {metric}")
            piv=df.pivot(index="arch_id",columns="epoch",values=metric)
            cols=[c for c in (STATE_GRID if source=="state" else GEOMETRY_GRID) if c in piv.columns]
            comp=piv[cols].dropna() if cols else pd.DataFrame()
            if len(cols)>=3 and len(comp)>=5:
                try:st,p=friedmanchisquare(*[comp[c].to_numpy(float) for c in cols]);W=float(st/(len(comp)*(len(cols)-1)))
                except Exception:st,p,W=np.nan,np.nan,np.nan
                rows.append({"source":source,"metric":metric,"n_architectures":len(comp),"n_checkpoints":len(cols),"friedman_chi2":st,"p_value":p,"kendall_W":W})
    df=pd.DataFrame(rows);atomic_csv(df,OUTDIR/"tables"/"temporal_friedman.csv");return df

def seed_reproducibility(states:pd.DataFrame,geometry:pd.DataFrame)->pd.DataFrame:
    """Seed-to-seed preservation of architecture ordering.

    IMPORTANT: this function operates on seed-level raw tables, therefore it must derive log_eta1,
    log_eta2 and log_xi50 itself. This is the exact defect that caused job 1216 to fail after all BCANw work.
    """
    rows=[];s=add_state_analysis_columns(states);g=add_geometry_analysis_columns(geometry)
    for source,df,metrics in [("state",s,sum(STATE_PRIMARY_FAMILIES.values(),[])),("geometry",g,GEOM_PRIMARY+GEOM_SECONDARY)]:
        if df.empty or len(LONGITUDINAL_SEEDS)<2:continue
        require_columns(df,["arch_id","seed_index","epoch"]+metrics,f"seed_reproducibility({source})")
        for metric in metrics:
            for e in sorted(pd.to_numeric(df.epoch,errors="coerce").dropna().unique()):
                sub=df[df.epoch==e];piv=sub.pivot_table(index="arch_id",columns="seed_index",values=metric,aggfunc="first")
                if all(sid in piv.columns for sid in LONGITUDINAL_SEEDS[:2]):
                    comp=piv[LONGITUDINAL_SEEDS[:2]].dropna();rho,p=spearmanr(comp.iloc[:,0],comp.iloc[:,1]) if len(comp)>=5 else (np.nan,np.nan);rows.append({"source":source,"metric":metric,"epoch":e,"n_architectures":len(comp),"seed_spearman":rho,"p_value":p})
    out=pd.DataFrame(rows);atomic_csv(out,OUTDIR/"tables"/"seed_reproducibility.csv");return out

def rank_preservation(arch_states:pd.DataFrame,arch_geom:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    """Architecture-rank preservation with an explicitly scalar diagonal.

    The earlier recovery build selected piv[[e,e]] on diagonal cells. Pandas then returned duplicate
    columns, comp[e] became a 2-D DataFrame, and scipy.stats.spearmanr returned a correlation matrix.
    That ndarray was later written into a scalar plotting matrix and caused the post-compute failure.
    Diagonal entries are now set analytically to 1 whenever at least five finite architectures exist,
    and every off-diagonal correlation is forced through the 1-D scalar helper above.
    """
    init_rows=[];matrix_rows=[]
    for source,df,metrics,grid in [("state",arch_states,sum(STATE_PRIMARY_FAMILIES.values(),[]),STATE_GRID),("geometry",arch_geom,GEOM_PRIMARY+GEOM_SECONDARY,GEOMETRY_GRID)]:
        if df.empty:continue
        require_columns(df,["arch_id","epoch"]+metrics,f"rank_preservation({source})")
        for metric in metrics:
            piv=df.pivot(index="arch_id",columns="epoch",values=metric);avail=[g for g in grid if g in piv.columns]
            if 0 in piv.columns and avail:
                end=max(avail)
                if end==0:
                    vals=pd.to_numeric(piv[0],errors="coerce").to_numpy(float);n=int(np.isfinite(vals).sum());rho=1.0 if n>=5 else np.nan;p=0.0 if n>=5 else np.nan
                else:
                    rho,p,n=_spearman_scalar(pd.to_numeric(piv[0],errors="coerce"),pd.to_numeric(piv[end],errors="coerce"),5)
                init_rows.append({"source":source,"metric":metric,"end_epoch":end,"n_architectures":n,"init_final_spearman":rho,"p_value":p})
            for e1 in avail:
                for e2 in avail:
                    if e1==e2:
                        vals=pd.to_numeric(piv[e1],errors="coerce").to_numpy(float);n=int(np.isfinite(vals).sum());rho=1.0 if n>=5 else np.nan
                    else:
                        rho,_,n=_spearman_scalar(pd.to_numeric(piv[e1],errors="coerce"),pd.to_numeric(piv[e2],errors="coerce"),5)
                    matrix_rows.append({"source":source,"metric":metric,"epoch_i":e1,"epoch_j":e2,"spearman":float(rho) if np.isfinite(rho) else np.nan,"n_architectures":n})
    a=pd.DataFrame(init_rows);b=pd.DataFrame(matrix_rows)
    # Contract: the persisted Spearman column must be numeric scalars, never object/array values.
    if not b.empty:
        b["spearman"]=pd.to_numeric(b["spearman"],errors="coerce")
        if b["spearman"].dtype==object:
            raise TypeError("rank_preservation produced non-scalar/object Spearman values")
    atomic_csv(a,OUTDIR/"tables"/"init_final_rank_preservation.csv");atomic_csv(b,OUTDIR/"tables"/"rank_persistence_matrices.csv");return a,b

def scaling_two_stage(geometry:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    if geometry.empty:return pd.DataFrame(),pd.DataFrame()
    require_columns(geometry,["arch_id","epoch","scaling_credible","scaling_dB"],"scaling_two_stage")
    g0=geometry.copy();g0["scaling_credible"]=pd.to_numeric(g0["scaling_credible"],errors="coerce")
    cred=g0.groupby(["arch_id","epoch"],as_index=False).scaling_credible.mean().rename(columns={"scaling_credible":"credible_seed_fraction"});p=cred.pivot(index="arch_id",columns="epoch",values="credible_seed_fraction");rows=[]
    if 0 in p.columns and FULL_EPOCHS in p.columns:
        d=(p[FULL_EPOCHS]-p[0]).dropna().to_numpy(float)
        try:st,pv=wilcoxon(d,zero_method="wilcox")
        except Exception:st,pv=np.nan,1.
        if len(d): rows.append({"analysis":"credible_seed_fraction_change","n_architectures":len(d),"median_change":float(np.median(d)),"rank_biserial":paired_rank_biserial(d),"p_value":pv})
    cred_df=pd.DataFrame(rows)
    credible_mask=g0.scaling_credible.fillna(0).astype(float)>0.5
    g=g0[credible_mask].groupby(["arch_id","epoch"],as_index=False).scaling_dB.median();pv=g.pivot(index="arch_id",columns="epoch",values="scaling_dB") if not g.empty else pd.DataFrame();rows=[]
    if not pv.empty and 0 in pv.columns and FULL_EPOCHS in pv.columns:
        d=(pv[FULL_EPOCHS]-pv[0]).dropna().to_numpy(float)
        if len(d):
            try:st,pval=wilcoxon(d,zero_method="wilcox")
            except Exception:st,pval=np.nan,1.
            rows.append({"analysis":"dB_change_given_endpoint_credibility","n_architectures":len(d),"median_change":float(np.median(d)),"hodges_lehmann":hodges_lehmann_onesample(d),"rank_biserial":paired_rank_biserial(d),"p_value":pval})
    db_df=pd.DataFrame(rows);atomic_csv(cred_df,OUTDIR/"tables"/"scaling_credibility_two_stage.csv");atomic_csv(db_df,OUTDIR/"tables"/"scaling_dB_conditional.csv");return cred_df,db_df

def architecture_change_frame(arch_states:pd.DataFrame,arch_geom:pd.DataFrame,cohort:pd.DataFrame)->pd.DataFrame:
    metrics=sum(STATE_PRIMARY_FAMILIES.values(),[]);base=cohort.copy();require_columns(base,["arch_id","macs","parameter_count","occurrence_count"],"architecture_change_frame(cohort)");base["log_macs"]=np.log(np.clip(base.macs.astype(float),1e-300,None));base["log_parameter_count"]=np.log(np.clip(base.parameter_count.astype(float),1e-300,None));base["log_occurrence_count"]=np.log(np.clip(base.occurrence_count.astype(float),1e-300,None))
    for m in metrics:
        if m not in arch_states.columns: raise KeyError(f"architecture_change_frame: missing state metric {m}")
        piv=arch_states.pivot(index="arch_id",columns="epoch",values=m)
        if 0 in piv.columns and FULL_EPOCHS in piv.columns:base=base.merge((piv[FULL_EPOCHS]-piv[0]).rename(f"delta_{m}").reset_index(),on="arch_id",how="left")
    if not arch_geom.empty:
        for m in GEOM_PRIMARY+GEOM_SECONDARY:
            if m not in arch_geom.columns: raise KeyError(f"architecture_change_frame: missing geometry metric {m}")
            piv=arch_geom.pivot(index="arch_id",columns="epoch",values=m)
            if 0 in piv.columns and FULL_EPOCHS in piv.columns:base=base.merge((piv[FULL_EPOCHS]-piv[0]).rename(f"delta_{m}").reset_index(),on="arch_id",how="left")
    atomic_csv(base,OUTDIR/"tables"/"architecture_change_frame.csv");return base

def posthoc_architecture_associations(change:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    continuous=["depth","log_macs","log_parameter_count","log_occurrence_count","receptive_field","execution_nodes","output_resolution","mean_hidden_width","max_hidden_width"]
    responses=[c for c in change.columns if c.startswith("delta_")];rows=[]
    for y in responses:
        for x in continuous:
            if x not in change.columns: continue
            sub=change[[x,y]].dropna();rho,p=spearmanr(sub[x],sub[y]) if len(sub)>=8 else (np.nan,np.nan);rows.append({"analysis":"posthoc_spearman","descriptor":x,"response":y,"n_architectures":len(sub),"rho_s":rho,"p_raw":p})
    corr=pd.DataFrame(rows)
    if not corr.empty:
        corr["p_holm"]=np.nan
        for y,g in corr.groupby("response"):corr.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    atomic_csv(corr,OUTDIR/"tables"/"posthoc_architecture_spearman.csv")
    if "stage" in change.columns: generated=change[change.stage.astype(str).isin(["sobol","bo"])].copy()
    else: generated=change.copy()
    om=[];pairs=[]
    for y in responses:
        for factor in ["morphology","kernel_family"]:
            if factor not in generated.columns: continue
            grouped=[(str(k),v[y].dropna().to_numpy(float)) for k,v in generated.groupby(factor) if len(v[y].dropna())>=3];labels=[x[0] for x in grouped];groups=[x[1] for x in grouped]
            if len(groups)>=2:
                n=sum(len(z) for z in groups);k=len(groups)
                try:H,p=kruskal(*groups);eps2=max(0.,(H-k+1)/max(n-k,1))
                except Exception:H,p,eps2=np.nan,np.nan,np.nan
                om.append({"analysis":"posthoc_kruskal","factor":factor,"response":y,"n_architectures":n,"groups":k,"H":H,"epsilon_squared":eps2,"p_raw":p})
                if np.isfinite(p) and p<POSTHOC_ALPHA:
                    for i in range(len(groups)):
                        for j in range(i+1,len(groups)):
                            try:U,pij=mannwhitneyu(groups[i],groups[j],alternative="two-sided")
                            except Exception:U,pij=np.nan,np.nan
                            pairs.append({"factor":factor,"response":y,"level_a":labels[i],"level_b":labels[j],"n_a":len(groups[i]),"n_b":len(groups[j]),"median_a_minus_b":float(np.median(groups[i])-np.median(groups[j])),"rank_biserial":mannwhitney_rank_biserial(groups[i],groups[j]),"p_raw":pij})
    om=pd.DataFrame(om);pairs=pd.DataFrame(pairs)
    if not om.empty:
        om["p_holm"]=np.nan
        for fac,g in om.groupby("factor"):om.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    if not pairs.empty:
        pairs["p_holm"]=np.nan
        for (fac,y),g in pairs.groupby(["factor","response"]):pairs.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    atomic_csv(om,OUTDIR/"tables"/"posthoc_architecture_categorical_omnibus.csv");atomic_csv(pairs,OUTDIR/"tables"/"posthoc_architecture_categorical_pairs.csv");return corr,om,pairs

# A production-style statistical contract self-test is run BEFORE expensive discovery/training. It deliberately
# includes two seeds, unlike the old pilot profile, so seed-level analysis bugs are caught before GPU/BCANw work.
def run_analysis_contract_selftest():
    global OUTDIR
    if not S1_ANALYSIS_SELFTEST: return
    rng=np.random.default_rng(1907);archs=[f"MOCK_{i:02d}" for i in range(12)];epochs=STATE_GRID;gepochs=GEOMETRY_GRID
    srows=[];grows=[]
    for ai,a in enumerate(archs):
        for seed in [0,1]:
            for e in epochs:
                frac=e/max(FULL_EPOCHS,1);base=0.1*ai+0.01*seed
                srows.append({"run_id":f"{a}_{seed}","arch_id":a,"arch_key":a,"seed_index":seed,"epoch":e,"morphology":MORPHOLOGIES[ai%len(MORPHOLOGIES)],"kernel_family":KERNEL_FAMILIES[ai%len(KERNEL_FAMILIES)],"entropy_deficit":1+base+frac,"eta1":math.exp(-(1+base+frac)),"eta2":math.exp(-2*(1+base+frac)),"hidden_mean_occurrence_cv":1+base+.2*frac,"hidden_mean_parameter_cv":1+base+.1*frac,"hidden_mean_S_occ":.5+base+.1*frac,"hidden_mean_S_param":.3+base+.04*frac,"hidden_mean_S_reuse":.2+.06*frac})
            for e in gepochs:
                frac=e/max(FULL_EPOCHS,1);base=0.01*ai+0.001*seed
                grows.append({"run_id":f"{a}_{seed}","arch_id":a,"arch_key":a,"seed_index":seed,"epoch":e,"morphology":MORPHOLOGIES[ai%len(MORPHOLOGIES)],"kernel_family":KERNEL_FAMILIES[ai%len(KERNEL_FAMILIES)],"xi50":1.0-.05*frac+base,"d_star":2-.1*frac+base,"lacunarity_star":3+.2*frac+base,"local_delta_median":-.2+.02*frac+base,"quotient_mean_pairwise_KS":.2+.01*frac+base,"renorm_flow_retention_mean":.5-.02*frac+base,"scaling_credible":1.0,"scaling_dB":1.2+.02*frac+base})
    cohort=pd.DataFrame({"arch_id":archs,"stage":["sobol"]*12,"morphology":[MORPHOLOGIES[i%len(MORPHOLOGIES)] for i in range(12)],"kernel_family":[KERNEL_FAMILIES[i%len(KERNEL_FAMILIES)] for i in range(12)],"depth":[3+i%5 for i in range(12)],"macs":[1e6*(1+i/10) for i in range(12)],"parameter_count":[1000+50*i for i in range(12)],"occurrence_count":[10000+500*i for i in range(12)],"receptive_field":[3+2*(i%4) for i in range(12)],"execution_nodes":[500+20*i for i in range(12)],"output_resolution":[4+i%3 for i in range(12)],"mean_hidden_width":[8+i for i in range(12)],"max_hidden_width":[12+i for i in range(12)]})
    states=pd.DataFrame(srows);geometry=pd.DataFrame(grows)
    old_out=OUTDIR
    with tempfile.TemporaryDirectory(prefix="cnn_s1_analysis_selftest_") as td:
        OUTDIR=Path(td);(OUTDIR/"tables").mkdir(parents=True,exist_ok=True)
        try:
            (OUTDIR/"figures").mkdir(parents=True,exist_ok=True)
            sa,ga=prepare_architecture_level(states,geometry)
            planned=planned_change_table(sa,ga);temporal=temporal_friedman(sa,ga);seedrep=seed_reproducibility(states,geometry);initfinal,rankmat=rank_preservation(sa,ga);scalecred,scaledb=scaling_two_stage(geometry)
            ch=architecture_change_frame(sa,ga,cohort);postcorr,postom,postpairs=posthoc_architecture_associations(ch)
            # Contract checks that specifically catch ndarray-valued diagonal correlations.
            if not rankmat.empty:
                if rankmat["spearman"].dtype==object: raise TypeError("selftest: rankmat.spearman is object-valued")
                diag=rankmat[rankmat.epoch_i==rankmat.epoch_j].spearman.dropna().to_numpy(float)
                if len(diag) and not np.allclose(diag,1.0,atol=1e-12): raise AssertionError("selftest: rank-matrix diagonal is not scalar 1")
            archive=cohort.copy();archive["stage"]=["bo" if i%4==0 else ("anchor" if i%7==0 else "sobol") for i in range(len(archive))];archive["discovery_score"]=np.linspace(0.2,1.1,len(archive));archive["rfsmac_acquisition"]=np.linspace(0.1,0.9,len(archive))
            gsrows=[]
            for a in archs:
                for seed in [0,1]:
                    rid=f"{a}_{seed}"
                    for x in np.geomspace(.1,5,8):
                        gsrows.append({"run_id":rid,"arch_id":a,"epoch":FULL_EPOCHS,"ell_over_xi50":x,"NB_over_N":max(.02,min(1.0,x**-0.7)),"lacunarity":1+0.3*np.exp(-((np.log(x))**2)),"interbox_flow_fraction":1/(1+x),"coarse_eta_q2":1/(1+.4*x)})
            gsc=pd.DataFrame(gsrows)
            stems=make_figures(archive,sa,ga,states,geometry,gsc,planned,postcorr,rankmat,seedrep)
            report=make_report(stems,archive,cohort,planned,temporal,pd.DataFrame({"control":["mock"],"value":[1.0]}),postcorr,postom,scalecred,scaledb)
            if not Path(report).exists(): raise RuntimeError("selftest: report artifact was not created")
        finally:
            OUTDIR=old_out
    print("[preflight] Production-style statistics + plotting/report contract: PASS (12 mock architectures x 2 seeds).")

# ==================================================================================================
# 12. PUBLICATION-STYLE FIGURES AND AUTOMATIC SCIENTIFIC REPORT
# ==================================================================================================

mpl.rcParams.update({
    "font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix",
    "font.size":8.6,"axes.labelsize":8.8,"axes.titlesize":9.5,"xtick.labelsize":7.6,"ytick.labelsize":7.6,
    "legend.fontsize":7.2,"axes.linewidth":0.8,"xtick.direction":"in","ytick.direction":"in",
    "xtick.top":True,"ytick.right":True,"pdf.fonttype":42,"ps.fonttype":42,"figure.dpi":130,"savefig.dpi":360,
})
STRUCT_COLOR="#243b6b"; STRUCT_LIGHT="#b8c4dc"; BO_COLOR="#8c564b"; SOBOL_COLOR="#4c78a8"; ANCHOR_COLOR="#303030"

def panel_label(ax,label):ax.text(-.12,1.05,label,transform=ax.transAxes,fontweight="bold",fontsize=10,va="top")
def subtle_grid(ax,axis="both"):ax.grid(True,axis=axis,alpha=.12,lw=.55);ax.set_axisbelow(True)
def footer(fig,text):fig.text(.015,.012,text,fontsize=6.8,ha="left",va="bottom",wrap=True)

def save_figure(fig,stem):
    fig.tight_layout(rect=[.02,.04,.99,.97]);fig.savefig(OUTDIR/"figures"/f"{stem}.pdf",bbox_inches="tight");fig.savefig(OUTDIR/"figures"/f"{stem}.png",dpi=420,bbox_inches="tight");plt.close(fig)

def make_figures(archive:pd.DataFrame,arch_states:pd.DataFrame,arch_geom:pd.DataFrame,states:pd.DataFrame,geometry:pd.DataFrame,gscales:pd.DataFrame,planned:pd.DataFrame,postcorr:pd.DataFrame,rankmat:pd.DataFrame,seedrep:pd.DataFrame):
    stems=[]
    # 1 Architecture discovery / RF-SMAC map.
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.3));stages=[("sobol",SOBOL_COLOR),("bo",BO_COLOR),("anchor",ANCHOR_COLOR)]
    ax=axs[0,0]
    for st,c in stages:
        g=archive[archive.stage.astype(str)==st];ax.scatter(g.depth,np.log10(g.macs.astype(float)),s=9,alpha=.45,label=st,color=c)
    ax.set_xlabel("convolutional depth");ax.set_ylabel(r"$\log_{10}$ MACs");ax.set_title("Architecture-space coverage");ax.legend(frameon=False);subtle_grid(ax);panel_label(ax,"a")
    ax=axs[0,1];ax.plot(np.arange(len(archive)),archive.discovery_score.astype(float),lw=.8,color=STRUCT_COLOR);n0=np.sum(archive.stage.astype(str).isin(["sobol","anchor"]));ax.axvline(n0,color="0.5",ls="--",lw=.8);ax.set_xlabel("discovery evaluation");ax.set_ylabel("information score");ax.set_title("Structural information acquisition");subtle_grid(ax);panel_label(ax,"b")
    ax=axs[1,0];tab=archive[archive.stage.astype(str).isin(["sobol","bo"])].groupby(["morphology","kernel_family"]).size().unstack(fill_value=0).reindex(index=MORPHOLOGIES,columns=KERNEL_FAMILIES,fill_value=0);im=ax.imshow(tab.values,aspect="auto");ax.set_yticks(range(len(tab.index)),tab.index);ax.set_xticks(range(len(tab.columns)),[x.replace("_","\n") for x in tab.columns],rotation=20,ha="right");ax.set_title("Morphology × kernel coverage");plt.colorbar(im,ax=ax,fraction=.046,pad=.03,label="evaluations");panel_label(ax,"c")
    ax=axs[1,1];bo=archive[archive.stage.astype(str)=="bo"];ax.scatter(np.arange(len(bo)),bo.rfsmac_acquisition.astype(float),s=10,alpha=.55,color=BO_COLOR);ax.set_xlabel("RF-SMAC refinement step");ax.set_ylabel("accepted acquisition");ax.set_title("Sequential RF-SMAC refinement");subtle_grid(ax);panel_label(ax,"d");save_figure(fig,"S1_F01_architecture_discovery");stems.append("S1_F01_architecture_discovery")

    # helper longitudinal architecture medians/IQR.
    def qtraj(metric,df=arch_states,grid=STATE_GRID):
        d=df[df.epoch.isin(grid)].groupby("epoch")[metric].agg(median="median",q1=lambda x:x.quantile(.25),q3=lambda x:x.quantile(.75)).reindex([e for e in grid if e in set(df.epoch)]);return d
    # 2 Path-flow / heterogeneity trajectories.
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2));specs=[("entropy_deficit",r"$S_{\rm deficit}^{S}$","Structural path concentration"),("eta1",r"$\eta_1^{S}$","Effective path occupation"),("hidden_mean_occurrence_cv",r"$CV_{\rm occ}$","Occurrence-flow heterogeneity"),("hidden_mean_parameter_cv",r"$CV_{\theta}$","Parameter-flow heterogeneity")]
    for ax,(m,yl,title),lab in zip(axs.ravel(),specs,list("abcd")):
        d=qtraj(m);ax.plot(d.index,d["median"],marker="o",ms=3,color=STRUCT_COLOR);ax.fill_between(d.index,d.q1,d.q3,color=STRUCT_LIGHT,alpha=.35,lw=0);ax.set_xlabel("epoch");ax.set_ylabel(yl);ax.set_title(title);subtle_grid(ax);panel_label(ax,lab)
    save_figure(fig,"S1_F02_structural_trajectories");stems.append("S1_F02_structural_trajectories")

    # 3 Sharing decomposition.
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.65));specs=[("hidden_mean_S_occ",r"$S_{\rm occ}^{S}$","Occurrence specialization"),("hidden_mean_S_param",r"$S_{\rm param}^{S}$","Parameter identity"),("hidden_mean_S_reuse",r"$S_{\rm reuse}^{S}$","Spatial reuse")]
    for ax,(m,yl,title),lab in zip(axs,specs,list("abc")):
        d=qtraj(m);ax.plot(d.index,d["median"],marker="o",ms=3,color=STRUCT_COLOR);ax.fill_between(d.index,d.q1,d.q3,color=STRUCT_LIGHT,alpha=.35,lw=0);ax.set_xlabel("epoch");ax.set_ylabel(yl);ax.set_title(title);subtle_grid(ax);panel_label(ax,lab)
    save_figure(fig,"S1_F03_sharing_decomposition");stems.append("S1_F03_sharing_decomposition")

    # 4 Planned architecture-level effect sizes.
    fig,ax=plt.subplots(figsize=(7.0,4.1));d=planned.copy();d["label"]=d.metric.str.replace("hidden_mean_","",regex=False).str.replace("_"," ",regex=False);d=d.sort_values("rank_biserial");y=np.arange(len(d));ax.hlines(y,0,d.rank_biserial,color="0.55",lw=.8);ax.plot(d.rank_biserial,y,"o",color=STRUCT_COLOR,ms=4);ax.axvline(0,color="0.35",lw=.8);ax.set_yticks(y,d.label);ax.set_xlabel("paired rank-biserial effect, final − initialization");ax.set_title("Architecture-wide training-direction effects");ax.set_xlim(-1.05,1.05);subtle_grid(ax,axis="x");save_figure(fig,"S1_F04_planned_effects");stems.append("S1_F04_planned_effects")

    # 5 Initialization vs trained architecture ordering.
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2));pairs=[("entropy_deficit",arch_states,r"$S_{\rm deficit}^{S}$"),("hidden_mean_S_param",arch_states,r"$S_{\rm param}^{S}$"),("hidden_mean_S_reuse",arch_states,r"$S_{\rm reuse}^{S}$"),("log_xi50",arch_geom,r"$\log\xi_{50}$")]
    for ax,(m,df,label),lab in zip(axs.ravel(),pairs,list("abcd")):
        if df.empty:continue
        piv=df.pivot(index="arch_id",columns="epoch",values=m);end=FULL_EPOCHS if FULL_EPOCHS in piv.columns else max(piv.columns);comp=piv[[0,end]].dropna();rho=spearmanr(comp[0],comp[end]).statistic if len(comp)>=5 else np.nan;ax.scatter(comp[0],comp[end],s=14,alpha=.6,color=STRUCT_COLOR);lo=min(comp.min());hi=max(comp.max());ax.plot([lo,hi],[lo,hi],color="0.45",lw=.8,ls="--");ax.set_xlabel("initialization");ax.set_ylabel("trained");ax.set_title(f"{label}; $\\rho_s$={rho:.2f}");subtle_grid(ax);panel_label(ax,lab)
    save_figure(fig,"S1_F05_initial_trained_preservation");stems.append("S1_F05_initial_trained_preservation")

    # 6 Rank persistence matrices.
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));forplots=[("entropy_deficit","state",STATE_GRID,r"$S_{\rm deficit}^{S}$"),("log_xi50","geometry",GEOMETRY_GRID,r"$\log\xi_{50}$")];im=None
    for ax,(m,src,grid,title),lab in zip(axs,forplots,list("ab")):
        if rankmat.empty or not {"metric","source","epoch_i","epoch_j","spearman"}.issubset(rankmat.columns):
            ax.axis("off");ax.set_title(title+" (unavailable)");panel_label(ax,lab);continue
        sub=rankmat[(rankmat.metric.astype(str)==m)&(rankmat.source.astype(str)==src)].copy()
        if sub.empty:
            ax.axis("off");ax.set_title(title+" (unavailable)");panel_label(ax,lab);continue
        sub["spearman"]=pd.to_numeric(sub["spearman"],errors="coerce")
        avail=[e for e in grid if e in set(pd.to_numeric(sub.epoch_i,errors="coerce").dropna())]
        if not avail:
            ax.axis("off");ax.set_title(title+" (unavailable)");panel_label(ax,lab);continue
        M=np.full((len(avail),len(avail)),np.nan,dtype=float)
        for i,e1 in enumerate(avail):
            for j,e2 in enumerate(avail):
                v=pd.to_numeric(sub[(pd.to_numeric(sub.epoch_i,errors="coerce")==e1)&(pd.to_numeric(sub.epoch_j,errors="coerce")==e2)].spearman,errors="coerce").dropna()
                if len(v):M[i,j]=float(v.iloc[0])
        im=ax.imshow(M,vmin=-1,vmax=1,aspect="auto");ax.set_xticks(range(len(avail)),avail,rotation=45,ha="right");ax.set_yticks(range(len(avail)),avail);ax.set_xlabel("epoch");ax.set_ylabel("epoch");ax.set_title(title);panel_label(ax,lab)
    if im is not None:fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.025,pad=.03,label=r"architecture-rank $\rho_s$")
    save_figure(fig,"S1_F06_rank_persistence");stems.append("S1_F06_rank_persistence")

    # 7 Post-hoc architecture-descriptor association matrix (top responses).
    if not postcorr.empty:
        responses=["delta_entropy_deficit","delta_hidden_mean_S_param","delta_hidden_mean_S_reuse","delta_log_xi50"]
        desc=["depth","log_macs","log_parameter_count","log_occurrence_count","receptive_field","mean_hidden_width"]
        M=np.full((len(responses),len(desc)),np.nan)
        for i,y in enumerate(responses):
            for j,x in enumerate(desc):
                v=postcorr[(postcorr.response==y)&(postcorr.descriptor==x)].rho_s
                if len(v):M[i,j]=v.iloc[0]
        fig,ax=plt.subplots(figsize=(7.0,3.3));im=ax.imshow(M,vmin=-1,vmax=1,aspect="auto");ax.set_xticks(range(len(desc)),[d.replace("_","\n") for d in desc]);ax.set_yticks(range(len(responses)),[r.replace("delta_","Δ ").replace("hidden_mean_","").replace("_"," ") for r in responses]);ax.set_title("Post-hoc architecture–response monotonic associations");fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label=r"Spearman $\rho_s$");save_figure(fig,"S1_F07_posthoc_architecture_associations");stems.append("S1_F07_posthoc_architecture_associations")

    # 8 BCANw master curves at final checkpoint.
    if not gscales.empty:
        final=gscales[gscales.epoch==FULL_EPOCHS].copy();fig,axs=plt.subplots(2,2,figsize=(7.2,5.25));specs=[("NB_over_N",r"$N_B/N$","Normalized covering"),("lacunarity",r"$\Lambda_F$","Flow lacunarity"),("interbox_flow_fraction","inter-box flow","Directed flow retention"),("coarse_eta_q2",r"coarse $\eta_2$","Coarse Rényi occupation")]
        for ax,(ycol,yl,title),lab in zip(axs.ravel(),specs,list("abcd")):
            curves=[]
            for rid,g in final.groupby("run_id"):
                x=g.ell_over_xi50.to_numpy(float);y=g[ycol].to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)&(x>0)
                if ok.sum()>=4:
                    o=np.argsort(x[ok]);curves.append((x[ok][o],y[ok][o]))
            if curves:
                xmin=max(x.min() for x,y in curves);xmax=min(x.max() for x,y in curves)
                if xmax>xmin:
                    grid=np.geomspace(xmin,xmax,120);C=[]
                    for x,y in curves:
                        yy=np.interp(np.log(grid),np.log(x),y);C.append(yy);ax.plot(grid,yy,color=STRUCT_LIGHT,alpha=.18,lw=.5)
                    C=np.vstack(C);med=np.nanmedian(C,0);q1=np.nanquantile(C,.25,0);q3=np.nanquantile(C,.75,0);ax.fill_between(grid,q1,q3,color=STRUCT_COLOR,alpha=.13,lw=0);ax.plot(grid,med,color=STRUCT_COLOR,lw=1.8,label=f"median; n={len(C)}")
            ax.set_xscale("log");ax.set_xlabel(r"$\ell_B/\xi_{50}$");ax.set_ylabel(yl);ax.set_title(title);ax.legend(frameon=False);subtle_grid(ax);panel_label(ax,lab)
        save_figure(fig,"S1_F08_structural_BCANw_master_curves");stems.append("S1_F08_structural_BCANw_master_curves")

    # 9 Geometry trajectories.
    if not arch_geom.empty:
        fig,axs=plt.subplots(2,2,figsize=(7.2,5.2));specs=[("log_xi50",r"$\log\xi_{50}$","Characteristic metric scale"),("d_star",r"$d^\ast$","Crossover response"),("lacunarity_star",r"peak $\Lambda_F$","Peak flow lacunarity"),("local_delta_median",r"median $\Delta_i^F$","Local mass–support mismatch")]
        for ax,(m,yl,title),lab in zip(axs.ravel(),specs,list("abcd")):
            d=qtraj(m,arch_geom,GEOMETRY_GRID);ax.plot(d.index,d["median"],marker="o",ms=3,color=STRUCT_COLOR);ax.fill_between(d.index,d.q1,d.q3,color=STRUCT_LIGHT,alpha=.35,lw=0);ax.set_xlabel("epoch");ax.set_ylabel(yl);ax.set_title(title);subtle_grid(ax);panel_label(ax,lab)
        save_figure(fig,"S1_F09_geometry_trajectories");stems.append("S1_F09_geometry_trajectories")

    # 10 Seed reproducibility.
    if not seedrep.empty:
        fig,ax=plt.subplots(figsize=(7.0,3.7))
        for m,label in [("entropy_deficit",r"$S_{deficit}$"),("hidden_mean_S_param",r"$S_{param}$"),("hidden_mean_S_reuse",r"$S_{reuse}$")]:
            g=seedrep[(seedrep.metric==m)&(seedrep.source=="state")].sort_values("epoch");ax.plot(g.epoch,g.seed_spearman,marker="o",ms=3,label=label)
        ax.set_ylim(-.1,1.02);ax.set_xlabel("epoch");ax.set_ylabel(r"seed-to-seed architecture-rank $\rho_s$");ax.set_title("Architecture phenotype reproducibility across independent seeds");ax.legend(frameon=False,ncol=3);subtle_grid(ax);save_figure(fig,"S1_F10_seed_reproducibility");stems.append("S1_F10_seed_reproducibility")
    return stems

def make_report(stems:List[str],archive:pd.DataFrame,cohort:pd.DataFrame,planned:pd.DataFrame,temporal:pd.DataFrame,preflight:pd.DataFrame,postcorr:pd.DataFrame,postom:pd.DataFrame,scaling_cred:pd.DataFrame,scaling_db:pd.DataFrame)->Path:
    path=OUTDIR/"CNN_S1_architecture_structural_disposition_scientific_report.pdf"
    with PdfPages(path) as pdf:
        def textpage(title,paras,table=None,cols=None):
            fig=plt.figure(figsize=(11.7,8.3));fig.suptitle(title,fontsize=17,y=.95);y=.88
            for p in paras:
                fig.text(.06,y,p,fontsize=10.2,va="top",wrap=True);y-=.095+.012*(len(p)//140)
            if table is not None and not table.empty and y>.17:
                show=table[(cols or list(table.columns))].head(14).copy();ax=fig.add_axes([.06,.07,.88,max(.12,y-.10)]);ax.axis("off");tb=ax.table(cellText=show.round(5).astype(str).values,colLabels=show.columns,cellLoc="center",loc="center");tb.auto_set_font_size(False);tb.set_fontsize(7);tb.scale(1,1.2)
            footer(fig,"CNN-S1 | Architecture-Wide Structural Disposition | planned inference uses 48 architecture units after within-architecture seed aggregation.");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
        textpage("CNN-S1 — Architecture-Wide Structural Disposition",[
            "CNN-S1 is the architecture-wide structural arm of the six-phase CNN programme. It observes the input-independent complete-path disposition p^S on the spatially expanded execution DAG, with rho preserving shared parameter identity. ReLU, Adam, the synthetic distributed regression environment, data split, training exposure and spatial resolution are fixed; only architecture is systematically varied.",
            f"Discovery contains {len(archive)} realized architectures from a stratified Sobol map, sequential RF-SMAC refinements and forced programme anchors. The discovery map is adaptive and is not treated as an IID inferential sample. Formal architecture-wide inference is restricted to the frozen {len(cohort)}-architecture longitudinal panel, with independent seeds aggregated inside architecture before statistical testing.",
            "The principal replication hypotheses are the Work-I structural directions: increasing entropy deficit, decreasing effective path occupation, increasing marginal-flow heterogeneity and metric-scale contraction. CNN sharing adds a new exact decomposition, S_occ = S_param + S_reuse, which distinguishes structural selection among shared parameter identities from structural selectivity among their spatial uses."
        ],preflight)
        textpage("Why these statistics?",[
            "The primary experimental unit is architecture, not training seed, layer, edge occurrence or BCANw box. For each selected architecture the two seed trajectories are first summarized by their median. Final-minus-initial architecture changes are therefore paired repeated observations on 48 architecture units. Wilcoxon signed-rank is used because entropy deficits, effective fractions, CVs and geometry responses are not assumed Gaussian. Hodges-Lehmann shifts, paired rank-biserial effects and bootstrap intervals accompany p-values so effect magnitude and direction remain visible.",
            "Architecture-wide sign consistency is reported separately because a recurrent law means more than a nonzero median. Friedman tests with Kendall W describe repeated common-checkpoint trajectories without requiring sphericity. Spearman coefficients quantify whether architecture ordering is already imposed at initialization and whether it persists through training. Seed-to-seed Spearman coefficients quantify whether architecture differences exceed stochastic training variation.",
            "Post-hoc architecture-descriptor analyses are explanatory rather than confirmatory. Continuous descriptors use Spearman monotonic association. Generated morphology/kernel classes use Kruskal-Wallis with epsilon-squared and Holm-corrected pairwise rank tests after an omnibus signal. These analyses do not imply causal effects over a hypothetical random population of CNN architectures because the 48-network panel is deliberately maximin-designed. Strict finite-window d_B is analysed in two stages: scaling credibility first, exponent only where endpoint windows are credible."
        ],planned,["family","metric","n_architectures","median_change","rank_biserial","positive_fraction","p_holm"])
        for stem in stems:
            f=OUTDIR/"figures"/f"{stem}.png"
            if f.exists():
                img=plt.imread(f)
                # The standalone figures remain high-resolution/vector. The integrated diagnostic report
                # uses a bounded raster preview so report assembly does not spend minutes re-embedding
                # multi-megapixel PNGs.
                step=max(1,int(max(img.shape[0]/1400,img.shape[1]/1900)))
                img=img[::step,::step]
                fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([.04,.05,.92,.90]);ax.imshow(img);ax.axis("off");pdf.savefig(fig,bbox_inches="tight",dpi=140);plt.close(fig)
        textpage("Statistical summary and post-hoc boundary",[
            "Planned inference is organized into measure/flow, sharing and primary-geometry families with Holm correction inside family. Secondary geometry remains descriptive unless explicitly tested. The adaptive RF-SMAC discovery map is used for experimental coverage and cohort construction, not for population inference.",
            "Post-hoc architecture correlations and categorical comparisons are retained because they can explain which architecture coordinates accompany the discovered structural responses and can generate explicit hypotheses for CNN-S2 and CNN-C1. They are labelled post hoc in every output table and are not used to redefine the planned S1 laws after observing the data."
        ],temporal,["source","metric","n_architectures","n_checkpoints","kendall_W","p_value"])
        if not scaling_cred.empty or not scaling_db.empty:textpage("Strict finite-window scaling: two-stage analysis",["A finite-window exponent is not assigned when the strict scaling criteria fail. CNN-S1 therefore first asks whether credible scaling windows become more or less prevalent across training, using architecture-level seed-credibility fractions. Only architectures with credible endpoint windows contribute to the paired d_B change analysis. This preserves the programme-wide distinction between existence of a scaling regime and the value of its local exponent."],pd.concat([scaling_cred,scaling_db],ignore_index=True,sort=False))
    return path

# ==================================================================================================
# 13. EXPERIMENT MANIFEST + ORCHESTRATION
# ==================================================================================================

def write_manifest():
    obj={"phase":"CNN-S1","name":"Architecture-Wide Structural Disposition","profile":S1_PROFILE,"object":"(G_exec_tilde, Gamma_exec, p^S ; rho)","device":str(DEVICE),"fixed":{"activation":ACTIVATION,"optimizer":OPTIMIZER,"lr":LEARNING_RATE,"environment":"canonical_distributed_regression","resolution":RESOLUTION,"batch_size":BATCH_SIZE,"full_epochs":FULL_EPOCHS,"train":N_TRAIN,"val":N_VAL,"test":N_TEST},"discovery":{"sobol":SOBOL_TOTAL,"rf_smac":RFSMAC_STEPS,"anchors":ANCHOR_TOTAL,"target_total":DISCOVERY_TOTAL,"discovery_epochs":DISCOVERY_EPOCHS,"discovery_train":DISCOVERY_N_TRAIN,"discovery_val":DISCOVERY_N_VAL,"random_interleave":RFSMAC_RANDOM_INTERLEAVE,"acquisition":"RF expected improvement + configuration novelty + under-covered-cell reward; observed target is structural-information score; held-out performance excluded"},"longitudinal":{"architectures":N_LONG_ARCH,"seeds":LONGITUDINAL_SEEDS,"runs":N_LONG_ARCH*len(LONGITUDINAL_SEEDS),"state_grid":STATE_GRID,"geometry_grid":GEOMETRY_GRID},"statistics":{"planned":"architecture-level seed aggregation; Wilcoxon + Hodges-Lehmann + rank-biserial + bootstrap CI + sign consistency; Friedman/Kendall W; rank preservation","posthoc":"Spearman architecture descriptors; Kruskal-Wallis/epsilon2 categorical factors; Holm; two-stage scaling"},"resilience":{"resume":S1_RESUME,"analysis_only":S1_ANALYSIS_ONLY,"auto_analysis_recovery":S1_AUTO_ANALYSIS_RECOVERY,"analysis_selftest":S1_ANALYSIS_SELFTEST,"corrupt_existing_csv_policy":"fail rather than silently recompute"}}
    atomic_json(obj,OUTDIR/"experiment_manifest.json")

def _unique_pairs(df:pd.DataFrame,cols:Sequence[str])->int:
    return int(len(df[list(cols)].drop_duplicates())) if (not df.empty and all(c in df.columns for c in cols)) else 0

def load_existing_analysis_artifacts(require_complete:bool=True):
    archive=safe_read(OUTDIR/"tables"/"architecture_discovery.csv")
    cohort=safe_read(OUTDIR/"tables"/"longitudinal_48_architectures.csv")
    panel12=safe_read(OUTDIR/"tables"/"s2_12_architecture_panel.csv")
    sent6=safe_read(OUTDIR/"tables"/"s3_6_architecture_sentinels.csv")
    if cohort.empty:
        if require_complete: raise RuntimeError("Cannot enter analysis recovery: longitudinal_48_architectures.csv is missing/empty.")
        return None
    tabs={k:safe_read(OUTDIR/"tables"/f"longitudinal_{k}.csv") for k in ["states","layers","history","summary"]}
    if tabs["states"].empty or tabs["summary"].empty:
        # Re-aggregation only reads saved per-run CSVs; it does not retrain.
        tabs=aggregate_longitudinal(cohort)
    geom=safe_read(OUTDIR/"tables"/"longitudinal_geometry.csv") if S1_RUN_BCANW else pd.DataFrame()
    gsc=safe_read(OUTDIR/"tables"/"longitudinal_geometry_scales.csv") if S1_RUN_BCANW else pd.DataFrame()
    dgeom=safe_read(OUTDIR/"tables"/"discovery_geometry_summary.csv")
    dgsc=safe_read(OUTDIR/"tables"/"discovery_geometry_scales.csv")
    preflight=safe_read(OUTDIR/"tables"/"preflight_controls.csv")
    if require_complete:
        expected_runs=N_LONG_ARCH*len(LONGITUDINAL_SEEDS);expected_states=expected_runs*len(STATE_GRID);expected_geom=expected_runs*len(GEOMETRY_GRID)
        problems=[]
        if len(cohort)!=N_LONG_ARCH: problems.append(f"cohort architectures {len(cohort)} != {N_LONG_ARCH}")
        if tabs["summary"].empty or tabs["summary"].run_id.astype(str).nunique()!=expected_runs: problems.append(f"run summaries != {expected_runs}")
        if _unique_pairs(tabs["states"],["run_id","epoch"])!=expected_states: problems.append(f"unique state checkpoints {_unique_pairs(tabs['states'],['run_id','epoch'])} != {expected_states}")
        if S1_RUN_BCANW and _unique_pairs(geom,["run_id","epoch"])!=expected_geom: problems.append(f"unique BCANw checkpoints {_unique_pairs(geom,['run_id','epoch'])} != {expected_geom}")
        if archive.empty or len(archive)<DISCOVERY_TOTAL: problems.append(f"discovery archive {len(archive)} < {DISCOVERY_TOTAL}")
        if problems: raise RuntimeError("Saved expensive artifacts are incomplete: " + "; ".join(problems))
        # Validate actual raw schema before statistics.
        add_state_analysis_columns(tabs["states"]);add_geometry_analysis_columns(geom)
    return archive,cohort,panel12,sent6,tabs,geom,gsc,dgeom,dgsc,preflight

def expensive_artifacts_complete()->bool:
    try:
        load_existing_analysis_artifacts(require_complete=True);return True
    except Exception as exc:
        print(f"[resume-check] full expensive artifact set not yet complete: {exc}");return False

def mark_compute_complete(archive,cohort,tabs,geom):
    atomic_json({"status":"complete","timestamp":time.time(),"discovery_architectures":len(archive),"cohort_architectures":len(cohort),"full_runs":int(tabs["summary"].run_id.nunique()) if not tabs["summary"].empty else 0,"state_checkpoints":_unique_pairs(tabs["states"],["run_id","epoch"]),"geometry_checkpoints":_unique_pairs(geom,["run_id","epoch"]) if not geom.empty else 0},OUTDIR/"COMPUTE_COMPLETE.json")

def run_analysis_pipeline(archive,cohort,panel12,sent6,tabs,geom,gsc,dgeom,dgsc,preflight,start_time:Optional[float]=None):
    start=start_time or time.perf_counter();status={"status":"running","stage":"prepare","timestamp":time.time()};atomic_json(status,OUTDIR/"analysis_status.json")
    try:
        banner("Planned architecture-level inference")
        arch_states,arch_geom=prepare_architecture_level(tabs["states"],geom);status["stage"]="planned";atomic_json(status,OUTDIR/"analysis_status.json")
        planned=planned_change_table(arch_states,arch_geom);temporal=temporal_friedman(arch_states,arch_geom);seedrep=seed_reproducibility(tabs["states"],geom);initfinal,rankmat=rank_preservation(arch_states,arch_geom);scalecred,scaledb=scaling_two_stage(geom)
        print(planned.sort_values(["family","p_holm"]).to_string(index=False) if not planned.empty else "No planned table")
        banner("Explicitly post-hoc architecture-response analyses")
        status["stage"]="posthoc";atomic_json(status,OUTDIR/"analysis_status.json")
        change=architecture_change_frame(arch_states,arch_geom,cohort);postcorr,postom,postpairs=posthoc_architecture_associations(change)
        stems=[];report=None
        if S1_RUN_REPORT:
            banner("Publication-style figures and integrated PDF");status["stage"]="report";atomic_json(status,OUTDIR/"analysis_status.json")
            stems=make_figures(archive,arch_states,arch_geom,tabs["states"],geom,gsc,planned,postcorr,rankmat,seedrep);report=make_report(stems,archive,cohort,planned,temporal,preflight,postcorr,postom,scalecred,scaledb)
        elapsed=time.perf_counter()-start;summary=["CNN-S1 COMPLETE",f"Profile: {S1_PROFILE}",f"Wall time for this invocation: {human_time(elapsed)}",f"Discovery architectures: {len(archive)}",f"Longitudinal architectures: {len(cohort)}",f"Full trajectories: {len(tabs['summary'])}",f"Geometry states: {len(geom)}",f"Output: {OUTDIR.resolve()}",f"Report: {report}"];(OUTDIR/"experiment_summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8");print("\n"+"\n".join(summary));status.update({"status":"complete","stage":"done","timestamp":time.time(),"report":str(report)});atomic_json(status,OUTDIR/"analysis_status.json");gpu_snapshot("final")
        return {"archive":archive,"discovery_geometry":dgeom,"discovery_geometry_scales":dgsc,"cohort48":cohort,"panel12":panel12,"sentinels6":sent6,"states":tabs["states"],"layers":tabs["layers"],"history":tabs["history"],"run_summary":tabs["summary"],"geometry":geom,"geometry_scales":gsc,"planned":planned,"temporal":temporal,"seed_reproducibility":seedrep,"rank_preservation":initfinal,"rank_matrices":rankmat,"scaling_credibility":scalecred,"scaling_dB":scaledb,"posthoc_spearman":postcorr,"posthoc_categorical":postom,"posthoc_pairs":postpairs,"report":report}
    except Exception as exc:
        tb=traceback.format_exc();status.update({"status":"failed","timestamp":time.time(),"error":repr(exc),"traceback":tb,"recovery":"Re-run this script with the SAME OUTDIR. It will auto-detect complete expensive artifacts, or set PATHFLOW_CNN_S1_ANALYSIS_ONLY=1 explicitly. Do not use START_FRESH."});atomic_json(status,OUTDIR/"analysis_status.json")
        print("\n[ANALYSIS FAILURE] All expensive run/checkpoint/BCANw artifacts remain on disk and are reusable.")
        print("Recovery: re-run with the same PATHFLOW_CNN_S1_OUTDIR; auto-recovery will skip compute. Alternatively set PATHFLOW_CNN_S1_ANALYSIS_ONLY=1.")
        raise

def run_cnn_s1_analysis_only():
    banner("CNN-S1 ANALYSIS-ONLY / FAILURE-RECOVERY MODE")
    artifacts=load_existing_analysis_artifacts(require_complete=True)
    archive,cohort,panel12,sent6,tabs,geom,gsc,dgeom,dgsc,preflight=artifacts
    print(f"[recovery] Reusing {tabs['summary'].run_id.nunique()} completed trajectories and {_unique_pairs(geom,['run_id','epoch']) if not geom.empty else 0} BCANw states. No training or BCANw will be executed.")
    return run_analysis_pipeline(archive,cohort,panel12,sent6,tabs,geom,gsc,dgeom,dgsc,preflight)

def run_cnn_s1():
    banner("CNN-S1 — ANALYSIS-ONLY RECOVERY V2")
    print(f"Recovery output directory: {OUTDIR.resolve()}")
    print("GUARD: this recovery-only build cannot run discovery, training, or BCANw.")
    run_analysis_contract_selftest()
    return run_cnn_s1_analysis_only()

# Single-cell convention: executing/pasting this file starts the complete experiment.
CNN_S1_RESULTS = run_cnn_s1()
