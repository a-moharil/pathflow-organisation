# ==================================================================================================
# CNN-S2 — ENVIRONMENT-WIDE STRUCTURAL DISPOSITION
# RF-SMAC environment discovery + frozen S1 architecture handoff + balanced A x E longitudinal matrix
# ==================================================================================================
# Single-file / single-Jupyter-cell research experiment.
#
# Scientific object
# -----------------
# CNN-S2 retains EXACTLY the input-independent structural disposition object established in CNN-S1:
#
#     O_CNN^S = (G_exec_tilde, Gamma_exec, p^S ; rho)
#
#     Q_gamma^S = product_{a in gamma} theta_{rho(a)}^2,
#     p^S(gamma) = Q_gamma^S / sum_{gamma'} Q_gamma'^S.
#
# The statistical environment never enters this observable as an input condition. Instead it changes the
# learned parameter state through training:
#
#     E -> theta_{A,E,s}(t) -> p^S_{A,E,s}(t) -> F^S -> d_F^S -> BCANw.
#
# Hence this is STRUCTURAL DISPOSITION, not conditional realization.
#
# Experimental logic
# ------------------
# Architecture handoff: 12 architectures frozen by CNN-S1; 4 S1-fixed/maximin sentinels for discovery.
# Environment discovery: 36 initial environments (34 Sobol + 2 canonical anchors) + 24 RF-SMAC = 60.
# Low-fidelity discovery: 60 environments x 4 sentinels, 80 epochs, structural states at 0,5,20,80.
# Discovery BCANw: sentinel geometry at t=0 and 80 (initial geometry is cached because it is environment-free).
# Environment inference panel: 24 selected environments (8 initial/coverage + 16 RF-SMAC) with anchors forced.
# Full matrix: 24 environments x 12 architectures x 2 seeds = 576 trajectories, 360 epochs.
# Exact common state grid:
#     0,1,2,5,10,20,40,80,120,160,240,360
# Full-panel BCANw:
#     0,80,160,360
# Dense-sentinel BCANw supplement:
#     0,20,40,80,120,160,240,360
#
# Environment family
# ------------------
#     E = (teacher_family, informative_spatial_fraction, spatial_correlation,
#          SNR, interaction_strength, localization)
#
# The task is scalar synthetic spatial REGRESSION throughout S2. Classification/regression task variation
# belongs to CNN-C2, not S2. Activation is fixed to ReLU and optimizer to Adam.
#
# RF-SMAC design
# --------------
# RF-SMAC is used as information-seeking experimental design, not performance HPO. The acquisition follows
# the Work-I / MLP-Phase-II logic:
#
#     A_env = .30 U + .25 N_env + .25 N_struct + .20 I_AxE
#
# with 25% random/coverage interleaving. Validation/test performance is NEVER an acquisition target.
# The adaptive 60-environment discovery map is not treated as an IID inferential sample. Formal inference is
# performed only on the frozen 24 x 12 x 2 matrix.
#
# Statistical hierarchy
# ---------------------
# Planned analyses:
#   * freeze the CNN-S1 law registry BEFORE S2 full-matrix inference;
#   * sign-law replication after seed aggregation within architecture x environment cells;
#   * exact architecture-environment balanced variance decomposition with seed/residual shown separately;
#   * temporal variance decomposition on the common state grid;
#   * cross-environment architecture-rank universality (276 environment pairs);
#   * parameter-identity vs spatial-reuse decomposition as a central CNN-specific outcome;
#   * functional arm uses validation-selected checkpoints, never test-selected checkpoints;
#   * finite scaling is two-stage: credibility first, d_B only conditional on credible windows.
#
# Explicitly post-hoc / explanatory analyses:
#   * environment-level Spearman associations after averaging over architectures and seeds;
#   * teacher-family Kruskal-Wallis + epsilon^2, followed by Holm-corrected pairwise Mann-Whitney only
#     after an omnibus signal;
#   * structural-versus-functional environment-level associations;
#   * these are descriptive for the adaptively selected environment panel, not population-causal effects.
#
# Required S1 handoff
# -------------------
# Publication mode requires the actual CNN-S1 12-architecture panel. The script looks for, in order:
#   1) CNN_S1_RESULTS['panel12'] in the current notebook kernel;
#   2) PATHFLOW_CNN_S2_S1_PANEL=/path/to/s2_12_architecture_panel.csv;
#   3) standard CNN-S1 output locations.
# It also loads CNN-S1 planned_architecture_change_inference.csv to freeze the S1 law registry. Set
# PATHFLOW_CNN_S2_S1_STATS if it is elsewhere. Publication mode does not silently invent an S1 panel/law set.
#
# Run in Colab/Jupyter:
#     %run /content/pathflow_cnn_s2_environment_structural_disposition_rfsmac_single_cell.py
#
# Profiles:
#     PATHFLOW_CNN_S2_PROFILE=publication_colab   (default frozen experiment)
#     PATHFLOW_CNN_S2_PROFILE=pilot               (small development profile; still needs S1 handoff unless
#                                                   PATHFLOW_CNN_S2_ALLOW_FALLBACK_PANEL=1)
# ==================================================================================================

from __future__ import annotations

import os, sys, math, json, time, random, hashlib, itertools, warnings, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence


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
    spearmanr, wilcoxon, friedmanchisquare, ks_2samp, norm, kruskal,
    mannwhitneyu, beta, binomtest
)
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================================================================================
# 0. PROFILE / FROZEN EXPERIMENT CONFIGURATION
# ==================================================================================================

S2_PROFILE = os.environ.get("PATHFLOW_CNN_S2_PROFILE", "publication_colab").strip().lower()
S2_RESUME = os.environ.get("PATHFLOW_CNN_S2_RESUME", "1") != "0"
S2_START_FRESH = os.environ.get("PATHFLOW_CNN_S2_START_FRESH", "0") == "1"
S2_REQUIRE_CUDA = os.environ.get("PATHFLOW_CNN_S2_REQUIRE_CUDA", "1") != "0"
S2_GPU_RESIDENT_DATA = os.environ.get("PATHFLOW_CNN_S2_GPU_RESIDENT_DATA", "1") != "0"
S2_RUN_BCANW = os.environ.get("PATHFLOW_CNN_S2_RUN_BCANW", "1") != "0"
S2_RUN_EXACT = os.environ.get("PATHFLOW_CNN_S2_RUN_EXACT", "1") != "0"
S2_RUN_REPORT = os.environ.get("PATHFLOW_CNN_S2_RUN_REPORT", "1") != "0"
S2_ALLOW_FALLBACK_PANEL = os.environ.get("PATHFLOW_CNN_S2_ALLOW_FALLBACK_PANEL", "0") == "1"
S2_CUDA_DEVICE = int(os.environ.get("PATHFLOW_CNN_S2_CUDA_DEVICE", "0"))

S2_SEED = 3829
MODEL_SEED_BASE = 271828
DATA_SEED_BASE = 57721
ORDER_SEED_BASE = 141421
BOOTSTRAP_B = 4000
POSTHOC_ALPHA = 0.05

if S2_REQUIRE_CUDA and not torch.cuda.is_available():
    raise RuntimeError(
        "CNN-S2 publication profile requires CUDA, but torch.cuda.is_available() is False. "
        "In Colab select Runtime > Change runtime type > GPU. Set PATHFLOW_CNN_S2_REQUIRE_CUDA=0 "
        "only for an intentional CPU smoke test."
    )
if torch.cuda.is_available():
    torch.cuda.set_device(S2_CUDA_DEVICE)
    DEVICE = torch.device(f"cuda:{S2_CUDA_DEVICE}")
else:
    DEVICE = torch.device("cpu")

OUTDIR = Path(os.environ.get("PATHFLOW_CNN_S2_OUTDIR", "pathflow_cnn_s2_environment_structural_disposition_results"))
for sub in ["tables","figures","environment_cache","checkpoints_discovery","checkpoints_full","run_parts","geometry_parts"]:
    (OUTDIR/sub).mkdir(parents=True, exist_ok=True)

# Fixed scientific core inherited from CNN-S1 / CNN-C3 conventions.
INPUT_CHANNELS = 1
RESOLUTION = 16
ACTIVATION = "relu"
OPTIMIZER = "adam"
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 256
N_TRAIN, N_VAL, N_TEST = 20_000, 8_000, 8_000
DISCOVERY_N_TRAIN, DISCOVERY_N_VAL, DISCOVERY_N_TEST = 6_000, 2_000, 2_000
DISCOVERY_EPOCHS = 80
FULL_EPOCHS = 360
DISCOVERY_STATE_GRID = [0,5,20,80]
STATE_GRID = [0,1,2,5,10,20,40,80,120,160,240,360]
GEOMETRY_GRID = [0,80,160,360]
DENSE_SENTINEL_GEOMETRY_GRID = [0,20,40,80,120,160,240,360]
FULL_SEEDS = [0,1]
N_ARCH = 12
N_SENTINELS = 4
N_SELECTED_ENV = 24

# Environment discovery: 34 Sobol + 2 canonical anchors + 24 RF-SMAC = 60.
TEACHER_FAMILIES = ["linear_spatial","shallow_nonlinear","deep_compositional"]
SOBOL_ENVIRONMENTS = 34
CANONICAL_ENVIRONMENTS = 2
INITIAL_ENVIRONMENTS = SOBOL_ENVIRONMENTS + CANONICAL_ENVIRONMENTS
RFSMAC_ENV_STEPS = 24
DISCOVERY_ENV_TOTAL = INITIAL_ENVIRONMENTS + RFSMAC_ENV_STEPS
RFSMAC_POOL = 1024
RFSMAC_TREES = 240
RFSMAC_RANDOM_INTERLEAVE = 0.25
RFSMAC_MIN_TRAIN = 12
SELECT_INITIAL = 8
SELECT_BO = 16

# Environment domains.
QINFO_RANGE = (0.10, 1.00)
SPATIAL_CORR_RANGE = (0.00, 0.90)
SNR_RANGE = (2.0, 50.0)
INTERACTION_RANGE = (0.00, 1.00)
LOCALIZATION_RANGE = (0.00, 1.00)

# Structural / BCANw constants inherited from the corrected programme conventions.
EPS = 1e-14
PATH_LOG_FLOOR = 1e-300
METRIC_POWER = 1.0
BCANW_SCALES_DISCOVERY = 26
BCANW_SCALES_LONG = 36
BCANW_MAX_NODES = 7000
LOCAL_DIMENSION_NODES = 128
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
MIN_FINAL_RESOLUTION = 4
MAX_EXEC_NODES = 6500

if S2_PROFILE == "pilot":
    DISCOVERY_N_TRAIN, DISCOVERY_N_VAL, DISCOVERY_N_TEST = 768, 192, 192
    DISCOVERY_EPOCHS = 3
    DISCOVERY_STATE_GRID = [0,1,3]
    SOBOL_ENVIRONMENTS = 4
    INITIAL_ENVIRONMENTS = 6
    RFSMAC_ENV_STEPS = 2
    DISCOVERY_ENV_TOTAL = 8
    RFSMAC_POOL = 64
    N_SELECTED_ENV = 4
    SELECT_INITIAL, SELECT_BO = 2,2
    N_ARCH = 4
    N_SENTINELS = 2
    N_TRAIN, N_VAL, N_TEST = 1024, 256, 256
    FULL_EPOCHS = 5
    STATE_GRID = [0,1,2,5]
    GEOMETRY_GRID = [0,5]
    DENSE_SENTINEL_GEOMETRY_GRID = [0,1,2,5]
    FULL_SEEDS = [0,1]
    BCANW_MAX_NODES = 1100
    LOCAL_DIMENSION_NODES = 16
    MAX_EXEC_NODES = 1100
    BOOTSTRAP_B = 500

# ==================================================================================================
# 1. GENERAL UTILITIES / EXACT RANK-BASED STATISTICS
# ==================================================================================================

def banner(title:str):
    print("\n" + "="*100 + f"\n{title}\n" + "="*100, flush=True)

def human_time(seconds:float)->str:
    if not np.isfinite(seconds): return "?"
    seconds=max(0,int(seconds));h,rem=divmod(seconds,3600);m,s=divmod(rem,60)
    return f"{h:d}h {m:02d}m" if h else (f"{m:d}m {s:02d}s" if m else f"{s:d}s")

class ETAReporter:
    def __init__(self,label,total):self.label=label;self.total=max(int(total),1);self.times=[];self.t0=time.perf_counter()
    def update(self,dt,done):
        self.times.append(float(dt));self.times=self.times[-20:];rate=np.median(self.times);remain=max(self.total-int(done),0)*rate
        print(f"[{self.label}] {done}/{self.total} | last {human_time(dt)} | ETA {human_time(remain)}",flush=True)
    def finish(self):print(f"[{self.label}] complete in {human_time(time.perf_counter()-self.t0)}",flush=True)

def gpu_snapshot(prefix="GPU"):
    if not torch.cuda.is_available():return
    try:
        props=torch.cuda.get_device_properties(DEVICE);alloc=torch.cuda.memory_allocated(DEVICE)/2**30;reserved=torch.cuda.memory_reserved(DEVICE)/2**30
        print(f"[{prefix}] {props.name} | allocated {alloc:.2f} GiB | reserved {reserved:.2f} GiB",flush=True)
    except Exception:pass

def stable_hash_dict(d:Dict[str,Any],n:int=12)->str:
    s=json.dumps(d,sort_keys=True,separators=(",",":"),default=str).encode();return hashlib.sha256(s).hexdigest()[:n]

def atomic_csv(df:pd.DataFrame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");df.to_csv(tmp,index=False);tmp.replace(path)

def atomic_json(obj:Any,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8");tmp.replace(path)

def safe_read(path:Path)->pd.DataFrame:
    try:return pd.read_csv(path)
    except Exception:return pd.DataFrame()

def safe_log(x):return float(np.log(x)) if np.isfinite(x) and x>0 else np.nan

def r2_score_np(y,pred):
    y=np.asarray(y,float);pred=np.asarray(pred,float);ss=np.sum((y-y.mean())**2);return float(1-np.sum((y-pred)**2)/ss) if ss>0 else np.nan

def holm_adjust(pvals:Sequence[float])->np.ndarray:
    p=np.asarray(pvals,float);out=np.full(len(p),np.nan);idx=np.where(np.isfinite(p))[0]
    if not len(idx):return out
    order=idx[np.argsort(p[idx])];mx=0.0
    for rank,j in enumerate(order):
        mx=max(mx,min(1.0,(len(order)-rank)*p[j]));out[j]=mx
    return out

def hodges_lehmann_onesample(d:np.ndarray)->float:
    x=np.asarray(d,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan
    vals=[]
    for i in range(len(x)):vals.extend(((x[i]+x[i:])/2.0).tolist())
    return float(np.median(vals))

def paired_rank_biserial(d:np.ndarray)->float:
    x=np.asarray(d,float);x=x[np.isfinite(x)&(x!=0)]
    if not len(x):return 0.0
    ranks=pd.Series(np.abs(x)).rank(method="average").to_numpy(float);wp=float(ranks[x>0].sum());wn=float(ranks[x<0].sum())
    return float((wp-wn)/max(wp+wn,EPS))

def mannwhitney_rank_biserial(x,y)->float:
    x=np.asarray(x,float);y=np.asarray(y,float);x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    if not len(x) or not len(y):return np.nan
    U=mannwhitneyu(x,y,alternative="two-sided").statistic;return float(2*U/(len(x)*len(y))-1)

def bootstrap_median_ci(d:np.ndarray,seed:int=0,B:int=BOOTSTRAP_B)->Tuple[float,float]:
    x=np.asarray(d,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan,np.nan
    rng=np.random.default_rng(seed);vals=np.empty(B,float)
    for b in range(B):vals[b]=np.median(rng.choice(x,size=len(x),replace=True))
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def clopper_pearson(k:int,n:int,alpha:float=.05)->Tuple[float,float]:
    if n<=0:return np.nan,np.nan
    lo=0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1));hi=1.0 if k==n else float(beta.ppf(1-alpha/2,k+1,n-k))
    return lo,hi

def deterministic_batches(n:int,epoch:int,seed_index:int,batch_size:int):
    rng=np.random.default_rng(ORDER_SEED_BASE+100003*int(seed_index)+int(epoch));ids=rng.permutation(n)
    for lo in range(0,n,batch_size):yield ids[lo:lo+batch_size]

def model_seed(arch_id:str,seed_index:int)->int:
    return MODEL_SEED_BASE+int(stable_hash_dict({"arch":str(arch_id),"seed":int(seed_index)},8),16)%1_000_003

if S2_START_FRESH:
    for sub in ["tables","figures","environment_cache","checkpoints_discovery","checkpoints_full","run_parts","geometry_parts"]:
        for p in (OUTDIR/sub).glob("*"):
            if p.is_file():
                try:p.unlink()
                except Exception:pass

# ==================================================================================================
# 2. CNN MODEL / EXECUTION DESCRIPTORS
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
        if self.output_resolution()<MIN_FINAL_RESOLUTION:raise ValueError("output spatial resolution below S2 minimum")
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



# ==================================================================================================
# 3. EXACT STRUCTURAL COMPLETE-PATH ENGINE AND SHARING PUSHFORWARD
# ==================================================================================================

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
    N=D.shape[0];delta=np.full(N,np.nan);rng=np.random.default_rng(S2_SEED+N);ids_eval=np.arange(N) if max_nodes>=N else np.sort(rng.choice(N,size=int(max_nodes),replace=False))
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


# ==================================================================================================
# 5. EXACT PREFLIGHT CONTROLS
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
    if S2_RESUME and path.exists():return pd.read_csv(path)
    rows=[]
    # Exact enumeration.
    model=ConvTraceNet(7,[1],[2,2],"relu",S2_SEED).to(DEVICE);st=cnn_structural_state(model,True);q,H=_tiny_structural_enumeration(model);rows.append({"control":"tiny_enumeration","partition_rel_error":abs(math.exp(st["logZ1"])-q.sum())/max(q.sum(),EPS),"entropy_abs_error":abs(st["S_path"]-H),"identity_residual":st["max_deficit_identity_residual"]})
    # Channel permutation and positive ReLU rescaling controls on C2-like network.
    model=ConvTraceNet(12,[3,4],[3,3,1],"relu",S2_SEED+1).to(DEVICE);base=cnn_structural_state(model,False);x=torch.randn(5,1,12,12,device=DEVICE);y0=model(x).detach()
    with torch.no_grad():
        p=torch.tensor([2,0,1],device=DEVICE);model.convs[0].weight.copy_(model.convs[0].weight[p]);model.convs[1].weight.copy_(model.convs[1].weight[:,p])
    perm=cnn_structural_state(model,False);y1=model(x).detach();rows.append({"control":"channel_permutation","function_max_abs_error":float((y1-y0).abs().max()),"entropy_abs_error":abs(perm["S_path"]-base["S_path"]),"deficit_abs_error":abs(perm["entropy_deficit"]-base["entropy_deficit"])})
    with torch.no_grad():
        c=2.3;model.convs[0].weight[0].mul_(c);model.convs[1].weight[:,0].div_(c)
    scale=cnn_structural_state(model,False);y2=model(x).detach();rows.append({"control":"positive_channel_rescaling","function_max_abs_error":float((y2-y1).abs().max()),"entropy_abs_error":abs(scale["S_path"]-perm["S_path"]),"deficit_abs_error":abs(scale["entropy_deficit"]-perm["entropy_deficit"])})
    # Analytic circular translation-sharing control: all occurrences of one shared parameter are equivalent.
    rng=np.random.default_rng(S2_SEED);k=3;R=7;w=rng.normal(size=(k,k));E=np.zeros((k,k,R,R),float)
    for dy in range(k):
        for dx in range(k):E[dy,dx,:,:]=w[dy,dx]**2
    E/=E.sum();P=E.sum(axis=(-1,-2));Hocc=-np.sum(E[E>0]*np.log(E[E>0]));Hpar=-np.sum(P[P>0]*np.log(P[P>0]));Hreuse=Hocc-Hpar;Sreuse=math.log(R*R)-Hreuse;rows.append({"control":"circular_translation_structural_reuse","S_reuse":Sreuse})
    df=pd.DataFrame(rows);atomic_csv(df,path);return df


# ==================================================================================================
# 6. CNN-S1 ARCHITECTURE HANDOFF AND FROZEN S1 LAW REGISTRY
# ==================================================================================================

REQUIRED_ARCH_COLS = ["arch_id","hidden_channels","kernel_sizes"]
ARCH_DESCRIPTOR_COLS = ["depth","macs","parameter_count","occurrence_count","execution_nodes","receptive_field","output_resolution","mean_hidden_width","max_hidden_width"]

FALLBACK_ARCHITECTURES = [
    ("C1_shallow3", [4,4], [3,3,1]),
    ("C2_medium3", [4,8,8], [3,3,3,1]),
    ("C3_bottleneck", [6,3,6], [3,3,3,1]),
    ("C4_pointwise", [4,8,8], [1,1,1,1]),
    ("C5_deep3", [4,8,8,12], [3,3,3,3,1]),
    ("C6_large5", [4,8,8], [5,3,3,1]),
    ("S2F07", [3,5,7], [3,1,3,1]),
    ("S2F08", [8,4,8], [3,3,1,1]),
    ("S2F09", [5,5,5,5], [1,3,1,3,1]),
    ("S2F10", [3,6,9,6], [3,1,3,1,1]),
    ("S2F11", [8,6,4], [1,3,3,1]),
    ("S2F12", [5,10,10], [3,3,1,1]),
]

def _complete_arch_descriptors(df:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for _,r in df.iterrows():
        hidden=json.loads(r.hidden_channels) if isinstance(r.hidden_channels,str) else list(r.hidden_channels)
        kernels=json.loads(r.kernel_sizes) if isinstance(r.kernel_sizes,str) else list(r.kernel_sizes)
        d=architecture_descriptors(hidden,kernels);rec=r.to_dict();rec["hidden_channels"]=json.dumps(list(map(int,hidden)));rec["kernel_sizes"]=json.dumps(list(map(int,kernels)))
        for k,v in d.items():
            if k!="valid" and (k not in rec or pd.isna(rec.get(k,np.nan))):rec[k]=v
        rec.setdefault("arch_key",stable_hash_dict({"hidden":hidden,"kernels":kernels},16));rows.append(rec)
    return pd.DataFrame(rows)

def _fallback_panel()->pd.DataFrame:
    rows=[]
    for aid,h,k in FALLBACK_ARCHITECTURES[:max(N_ARCH,1)]:rows.append({"arch_id":aid,"hidden_channels":json.dumps(h),"kernel_sizes":json.dumps(k),"stage":"fallback"})
    return _complete_arch_descriptors(pd.DataFrame(rows))

def _runtime_base_dirs()->List[Path]:
    """Small, deterministic set of directories in which an S1 handoff may legitimately live.

    This intentionally avoids broad recursive filesystem searches on HPC filesystems.  It covers:
      * the current working directory;
      * the directory containing this script;
      * the sibling structural/S1 directory when this script lives in structural/S2;
      * standard Colab /content and /mnt/data locations.
    """
    dirs=[]
    try:
        dirs.append(Path.cwd().resolve())
    except Exception:
        dirs.append(Path.cwd())
    try:
        script_dir=Path(__file__).resolve().parent
        dirs.append(script_dir)
        # Typical project layout: .../structural/S1 and .../structural/S2
        if script_dir.name.lower()=="s2":
            dirs.append(script_dir.parent/"S1")
        else:
            dirs.append(script_dir.parent/"S1")
    except Exception:
        script_dir=Path.cwd()
    dirs.extend([Path('/content'),Path('/mnt/data')])
    out=[];seen=set()
    for d in dirs:
        key=str(d)
        if key not in seen:
            out.append(d);seen.add(key)
    return out


def _expand_handoff_candidate(raw:Path,names:Sequence[str])->List[Path]:
    """Accept either a CSV file path or a directory containing one of the expected CSV names."""
    p=Path(raw).expanduser()
    # Existing directories are expanded.  A path with no .csv suffix is also treated as a possible directory,
    # which makes PATHFLOW_CNN_S2_S1_PANEL=/path/to/tables work naturally.
    if p.exists() and p.is_dir():
        return [p/name for name in names]
    if p.suffix.lower()!='.csv' and not p.exists():
        return [p/name for name in names] + [p]
    return [p]


def _dedupe_paths(paths:Sequence[Path])->List[Path]:
    out=[];seen=set()
    for p in paths:
        try:key=str(p.expanduser().resolve(strict=False))
        except Exception:key=str(p)
        if key not in seen:
            out.append(p);seen.add(key)
    return out


S1_PANEL_FILENAMES=(
    's2_12_architecture_panel.csv',
    'frozen_s1_12_architecture_panel.csv',
)
S1_STATS_FILENAMES=(
    'planned_architecture_change_inference.csv',
    'frozen_s1_law_registry.csv',
)


def _candidate_s1_panel_paths()->List[Path]:
    out=[]
    env=os.environ.get('PATHFLOW_CNN_S2_S1_PANEL','').strip()
    if env:
        out.extend(_expand_handoff_candidate(Path(env),S1_PANEL_FILENAMES))
    for base in _runtime_base_dirs():
        # User may place the handoff CSV directly in the working/script directory.
        out.extend(base/name for name in S1_PANEL_FILENAMES)
        # Native CNN-S1 output layout.
        out.extend(base/'pathflow_cnn_s1_architecture_structural_disposition_results'/'tables'/name for name in S1_PANEL_FILENAMES)
        # If `base` is the sibling S1 directory, this is the most common HPC layout.
        out.extend(base/'pathflow_cnn_s1_architecture_structural_disposition_results'/'tables'/name for name in S1_PANEL_FILENAMES)
        out.extend(base/'tables'/name for name in S1_PANEL_FILENAMES)
    return _dedupe_paths(out)


def _read_csv_file_candidate(p:Path,required:Sequence[str],kind:str)->Optional[pd.DataFrame]:
    """Read only regular files and give useful diagnostics instead of calling pandas on a directory."""
    try:
        p=Path(p).expanduser()
        if not p.exists():
            return None
        if not p.is_file():
            print(f'[{kind}] skipping non-file candidate: {p}',flush=True)
            return None
        d=pd.read_csv(p)
    except Exception as exc:
        print(f'[{kind}] could not read {p}: {type(exc).__name__}: {exc}',flush=True)
        return None
    missing=[c for c in required if c not in d.columns]
    if missing:
        print(f'[{kind}] CSV exists but is not the required handoff table: {p} | missing columns={missing}',flush=True)
        return None
    d.attrs['source_path']=str(p.resolve())
    return d


def load_s1_architecture_panel()->pd.DataFrame:
    if 'CNN_S1_RESULTS' in globals():
        try:
            d=globals()['CNN_S1_RESULTS'].get('panel12',pd.DataFrame()).copy()
            if not d.empty:
                if not all(c in d.columns for c in REQUIRED_ARCH_COLS):
                    raise ValueError(f"CNN_S1_RESULTS['panel12'] is missing required columns {REQUIRED_ARCH_COLS}")
                d=_complete_arch_descriptors(d)
                print("[S1 handoff] using CNN_S1_RESULTS['panel12'] from current kernel",flush=True)
                return d.iloc[:N_ARCH].copy()
        except Exception as exc:
            print(f'[S1 handoff] notebook object was present but invalid: {exc}',flush=True)
    searched=_candidate_s1_panel_paths()
    for p in searched:
        d=_read_csv_file_candidate(p,REQUIRED_ARCH_COLS,'S1 handoff')
        if d is not None:
            if len(d)<N_ARCH:
                print(f'[S1 handoff] candidate has only {len(d)} architectures; need at least {N_ARCH}: {p}',flush=True)
                continue
            d=_complete_arch_descriptors(d)
            print(f'[S1 handoff] loaded architecture panel: {Path(d.attrs.get("source_path",p))}',flush=True)
            return d.iloc[:N_ARCH].copy()
    if S2_ALLOW_FALLBACK_PANEL:
        print('[S1 handoff] WARNING: using explicit development fallback panel; do not use this for publication inference.',flush=True)
        return _fallback_panel().iloc[:N_ARCH].copy()
    msg='\n  '.join(str(p) for p in searched)
    raise FileNotFoundError(
        'CNN-S2 could not locate a valid frozen CNN-S1 12-architecture panel.\n'
        'The file may be placed directly in the current working directory as s2_12_architecture_panel.csv, '
        'or specify either the CSV itself or its containing directory with PATHFLOW_CNN_S2_S1_PANEL.\n'
        f'Searched candidates:\n  {msg}'
    )


def _candidate_s1_stats_paths()->List[Path]:
    out=[]
    env=os.environ.get('PATHFLOW_CNN_S2_S1_STATS','').strip()
    if env:
        out.extend(_expand_handoff_candidate(Path(env),S1_STATS_FILENAMES))
    for base in _runtime_base_dirs():
        out.extend(base/name for name in S1_STATS_FILENAMES)
        out.extend(base/'pathflow_cnn_s1_architecture_structural_disposition_results'/'tables'/name for name in S1_STATS_FILENAMES)
        out.extend(base/'tables'/name for name in S1_STATS_FILENAMES)
    return _dedupe_paths(out)


def load_s1_law_registry()->pd.DataFrame:
    searched=_candidate_s1_stats_paths()
    for p in searched:
        # planned_architecture_change_inference requires median_change. A pre-frozen registry may instead
        # already contain s1_direction, so support both schemas faithfully.
        try:
            pp=Path(p).expanduser()
            if not pp.exists():
                continue
            if not pp.is_file():
                print(f'[S1 law registry] skipping non-file candidate: {pp}',flush=True)
                continue
            d=pd.read_csv(pp)
        except Exception as exc:
            print(f'[S1 law registry] could not read {p}: {type(exc).__name__}: {exc}',flush=True)
            continue
        if {'metric','median_change'}.issubset(d.columns):
            if 'p_holm' in d.columns:
                ph=pd.to_numeric(d.p_holm,errors='coerce')
                d=d[(ph<=0.05)|ph.isna()].copy()
            if 'positive_fraction' in d.columns:
                pf=pd.to_numeric(d.positive_fraction,errors='coerce')
                consistency=np.maximum(pf,1-pf)
                d=d[consistency>=0.80].copy()
            d['s1_direction']=np.where(pd.to_numeric(d.median_change,errors='coerce')>0,1,-1)
        elif {'metric','s1_direction'}.issubset(d.columns):
            d=d.copy()
            d['s1_direction']=pd.to_numeric(d.s1_direction,errors='raise').astype(int)
        else:
            print(f'[S1 law registry] CSV exists but has incompatible schema: {pp} | columns={list(d.columns)}',flush=True)
            continue
        if d.empty:
            print(f'[S1 law registry] valid source produced zero frozen laws after S1 criteria: {pp}',flush=True)
            continue
        d['registry_source']=str(pp.resolve())
        d=d[[c for c in ['family','metric','median_change','p_holm','positive_fraction','s1_direction','registry_source'] if c in d.columns]].copy()
        atomic_csv(d,OUTDIR/'tables'/'frozen_s1_law_registry.csv')
        print(f'[S1 law registry] {len(d)} laws frozen from {pp}',flush=True)
        return d
    if S2_ALLOW_FALLBACK_PANEL:
        d=pd.DataFrame([
            {'family':'measure_flow','metric':'entropy_deficit','s1_direction':1},
            {'family':'measure_flow','metric':'log_eta1','s1_direction':-1},
            {'family':'measure_flow','metric':'log_eta2','s1_direction':-1},
            {'family':'measure_flow','metric':'hidden_mean_occurrence_cv','s1_direction':1},
            {'family':'measure_flow','metric':'hidden_mean_parameter_cv','s1_direction':1},
            {'family':'geometry_primary','metric':'log_xi50','s1_direction':-1},
        ])
        d['registry_source']='development_fallback';atomic_csv(d,OUTDIR/'tables'/'frozen_s1_law_registry.csv');return d
    msg='\n  '.join(str(p) for p in searched)
    raise FileNotFoundError(
        'CNN-S2 could not locate the CNN-S1 planned inference / frozen law registry CSV.\n'
        'Place planned_architecture_change_inference.csv directly in the current working directory, '
        'or specify either the CSV itself or its containing directory with PATHFLOW_CNN_S2_S1_STATS.\n'
        f'Searched candidates:\n  {msg}'
    )


def validate_s1_handoff_contract(panel:pd.DataFrame,registry:pd.DataFrame)->None:
    """Cheap pre-compute contract: fail before any environment generation/training if the handoff is wrong."""
    if len(panel)!=N_ARCH:
        raise RuntimeError(f'S1 handoff contract failed: expected {N_ARCH} architectures, got {len(panel)}')
    if panel.arch_id.astype(str).duplicated().any():
        raise RuntimeError('S1 handoff contract failed: duplicated arch_id values in the 12-architecture panel')
    if registry.empty or not {'metric','s1_direction'}.issubset(registry.columns):
        raise RuntimeError('S1 handoff contract failed: frozen S1 law registry is empty or malformed')
    bad=~registry.s1_direction.astype(int).isin([-1,1])
    if bad.any():
        raise RuntimeError('S1 handoff contract failed: s1_direction must contain only -1 or +1')
    print(f'[S1 handoff preflight] PASS | panel={len(panel)} architectures | frozen laws={len(registry)}',flush=True)

def select_discovery_sentinels(panel:pd.DataFrame)->pd.DataFrame:
    p=OUTDIR/"tables"/"s2_discovery_sentinels.csv"
    if S2_RESUME and p.exists():return pd.read_csv(p)
    if len(panel)<=N_SENTINELS:sent=panel.copy()
    else:
        numeric=[c for c in ARCH_DESCRIPTOR_COLS if c in panel.columns]
        # Add S1 discovery phenotype if those columns survived into the S1 panel.
        numeric += [c for c in panel.columns if c.startswith("delta_") and pd.api.types.is_numeric_dtype(panel[c])]
        X=panel[numeric].replace([np.inf,-np.inf],np.nan).fillna(panel[numeric].median(numeric_only=True)).to_numpy(float)
        X=RobustScaler().fit_transform(X);chosen=[int(np.argmax(np.linalg.norm(X-X.mean(0),axis=1)))]
        while len(chosen)<N_SENTINELS:
            dist=np.min(np.stack([np.linalg.norm(X-X[j],axis=1) for j in chosen]),axis=0);dist[chosen]=-np.inf;chosen.append(int(np.argmax(dist)))
        sent=panel.iloc[chosen].copy()
    atomic_csv(sent,p);return sent

# ==================================================================================================
# 7. CONTROLLED SPATIAL ENVIRONMENT FAMILY
# ==================================================================================================

@dataclass(frozen=True)
class EnvironmentSpec:
    env_id:str
    stage:str
    teacher_family:str
    informative_fraction:float
    spatial_correlation:float
    snr:float
    interaction_strength:float
    localization:float
    source_index:int=-1

    def as_dict(self):return dict(self.__dict__)

def canonical_environment_specs()->List[EnvironmentSpec]:
    # Defaults reproduce the programme's distributed/localized contrast in a continuous family. If exact C3
    # environment coordinates are available, override these by supplying PATHFLOW_CNN_S2_CANONICAL_ENV_JSON.
    override=os.environ.get("PATHFLOW_CNN_S2_CANONICAL_ENV_JSON","").strip()
    if override and Path(override).exists():
        raw=json.loads(Path(override).read_text());rows=[]
        for i,d in enumerate(raw):rows.append(EnvironmentSpec(source_index=i,stage="anchor",**d))
        return rows[:2]
    return [
        EnvironmentSpec("E_CAN_DISTRIBUTED","anchor","deep_compositional",0.85,0.20,25.0,0.35,0.05,0),
        EnvironmentSpec("E_CAN_LOCALIZED","anchor","deep_compositional",0.20,0.20,25.0,0.35,0.95,1),
    ]

def environment_key_dict(d:Dict[str,Any])->Dict[str,Any]:
    return {"teacher_family":str(d["teacher_family"]),"informative_fraction":round(float(d["informative_fraction"]),6),"spatial_correlation":round(float(d["spatial_correlation"]),6),"snr":round(float(d["snr"]),6),"interaction_strength":round(float(d["interaction_strength"]),6),"localization":round(float(d["localization"]),6)}

def make_env_id(d:Dict[str,Any],prefix="E")->str:return f"{prefix}_{stable_hash_dict(environment_key_dict(d),10)}"

def environment_vector(d:Dict[str,Any])->np.ndarray:
    one=[1.0 if str(d["teacher_family"])==f else 0.0 for f in TEACHER_FAMILIES]
    q=(float(d["informative_fraction"])-QINFO_RANGE[0])/(QINFO_RANGE[1]-QINFO_RANGE[0])
    r=(float(d["spatial_correlation"])-SPATIAL_CORR_RANGE[0])/(SPATIAL_CORR_RANGE[1]-SPATIAL_CORR_RANGE[0])
    ls=(math.log(float(d["snr"]))-math.log(SNR_RANGE[0]))/(math.log(SNR_RANGE[1])-math.log(SNR_RANGE[0]))
    k=float(d["interaction_strength"]);loc=float(d["localization"])
    return np.asarray(one+[q,r,ls,k,loc],float)

def generate_initial_environments()->List[Dict[str,Any]]:
    anchors=[x.as_dict() for x in canonical_environment_specs()]
    n=max(SOBOL_ENVIRONMENTS,1);power=int(math.ceil(math.log2(n)));u=qmc.Sobol(d=6,scramble=True,seed=S2_SEED).random_base2(power)[:n];rows=[]
    for i,z in enumerate(u):
        family=TEACHER_FAMILIES[min(int(z[0]*len(TEACHER_FAMILIES)),len(TEACHER_FAMILIES)-1)]
        d={"stage":"sobol","teacher_family":family,"informative_fraction":QINFO_RANGE[0]+z[1]*(QINFO_RANGE[1]-QINFO_RANGE[0]),"spatial_correlation":SPATIAL_CORR_RANGE[0]+z[2]*(SPATIAL_CORR_RANGE[1]-SPATIAL_CORR_RANGE[0]),"snr":math.exp(math.log(SNR_RANGE[0])+z[3]*(math.log(SNR_RANGE[1])-math.log(SNR_RANGE[0]))),"interaction_strength":z[4],"localization":z[5],"source_index":i}
        d["env_id"]=make_env_id(d,"E_SOB");rows.append(d)
    return anchors+rows

def random_environment_pool(n:int,seed:int,existing_keys:set)->List[Dict[str,Any]]:
    rng=np.random.default_rng(seed);rows=[];tries=0
    while len(rows)<n and tries<n*30:
        tries+=1;d={"stage":"bo_pool","teacher_family":str(rng.choice(TEACHER_FAMILIES)),"informative_fraction":float(rng.uniform(*QINFO_RANGE)),"spatial_correlation":float(rng.uniform(*SPATIAL_CORR_RANGE)),"snr":float(np.exp(rng.uniform(math.log(SNR_RANGE[0]),math.log(SNR_RANGE[1])))),"interaction_strength":float(rng.uniform(*INTERACTION_RANGE)),"localization":float(rng.uniform(*LOCALIZATION_RANGE)),"source_index":tries}
        key=stable_hash_dict(environment_key_dict(d),16)
        if key in existing_keys:continue
        d["env_id"]=make_env_id(d,"E_BO");rows.append(d);existing_keys.add(key)
    return rows

def _spatial_support_mask(cfg:Dict[str,Any],rng:np.random.Generator)->np.ndarray:
    R=RESOLUTION;n=max(1,min(R*R,int(round(float(cfg["informative_fraction"])*R*R))));loc=float(cfg["localization"])
    yy,xx=np.meshgrid(np.arange(R),np.arange(R),indexing="ij");cy=rng.uniform(0,R-1);cx=rng.uniform(0,R-1);sigma=max(1.0,R*(0.10+0.35*(1-loc)))
    compact=np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*sigma*sigma));scattered=rng.random((R,R));score=(1-loc)*scattered+loc*compact
    ids=np.argpartition(score.ravel(),-n)[-n:];mask=np.zeros(R*R,np.float32);mask[ids]=1.;return mask.reshape(R,R)

def _correlated_images(n:int,rho:float,rng:np.random.Generator)->np.ndarray:
    R=RESOLUTION;eps=rng.normal(0,1,(n,1,R,R)).astype(np.float32);t=torch.from_numpy(eps);low=F.avg_pool2d(t,kernel_size=5,stride=1,padding=2).numpy();low=(low-low.mean())/(low.std()+1e-8)
    X=math.sqrt(max(1-rho,0))*eps+math.sqrt(max(rho,0))*low.astype(np.float32);return X.astype(np.float32)

def _teacher_clean_target(X:np.ndarray,cfg:Dict[str,Any],mask:np.ndarray,rng:np.random.Generator)->np.ndarray:
    n=len(X);Xm=X*mask[None,None,:,:];fam=str(cfg["teacher_family"])
    xt=torch.from_numpy(Xm);torch.manual_seed(int(rng.integers(1,2**31-1)))
    with torch.no_grad():
        if fam=="linear_spatial":
            w=torch.from_numpy((rng.normal(0,1,(1,1,RESOLUTION,RESOLUTION))*mask[None,None]).astype(np.float32));base=(xt*w).sum((1,2,3))/math.sqrt(max(mask.sum(),1))
        elif fam=="shallow_nonlinear":
            W=torch.randn(6,1,3,3)/3.0;h=F.relu(F.conv2d(xt,W,padding=1));read=torch.randn(6)/math.sqrt(6);base=(h.mean((2,3))*read).sum(1)
        elif fam=="deep_compositional":
            W1=torch.randn(6,1,3,3)/3.0;W2=torch.randn(8,6,3,3)/math.sqrt(54);W3=torch.randn(6,8,3,3)/math.sqrt(72);h=F.relu(F.conv2d(xt,W1,padding=1));h=F.relu(F.conv2d(h,W2,padding=1));h=F.relu(F.conv2d(h,W3,padding=1));read=torch.randn(6)/math.sqrt(6);base=(h.mean((2,3))*read).sum(1)
        else:raise ValueError(fam)
    ids=np.flatnonzero(mask.ravel()>0);m=min(16,len(ids)//2);flat=X[:,0].reshape(n,-1)
    if m>=1:
        perm=rng.permutation(ids)[:2*m];inter=np.mean(flat[:,perm[:m]]*flat[:,perm[m:2*m]],axis=1)
    else:inter=np.zeros(n,float)
    b=base.numpy().astype(np.float64);b=(b-b.mean())/(b.std()+1e-8);inter=(inter-inter.mean())/(inter.std()+1e-8)
    k=float(cfg["interaction_strength"]);return ((1-k)*b+k*inter).astype(np.float32)[:,None]

def environment_dataset(cfg:Dict[str,Any],full:bool)->Tuple[Dict[str,Tuple[torch.Tensor,torch.Tensor]],Dict[str,Any]]:
    nt,nv,nte=(N_TRAIN,N_VAL,N_TEST) if full else (DISCOVERY_N_TRAIN,DISCOVERY_N_VAL,DISCOVERY_N_TEST);n=nt+nv+nte
    seed=DATA_SEED_BASE+int(stable_hash_dict(environment_key_dict(cfg),8),16)%10_000_019;rng=np.random.default_rng(seed);mask=_spatial_support_mask(cfg,rng);X=_correlated_images(n,float(cfg["spatial_correlation"]),rng);yclean=_teacher_clean_target(X,cfg,mask,rng)
    xmu=float(X[:nt].mean());xsd=float(X[:nt].std()+1e-8);X=((X-xmu)/xsd).astype(np.float32);ym=float(yclean[:nt].mean());ys=float(yclean[:nt].std()+1e-8);yclean=((yclean-ym)/ys).astype(np.float32)
    y=yclean+rng.normal(0,math.sqrt(1/float(cfg["snr"])),size=yclean.shape).astype(np.float32);ym2=float(y[:nt].mean());ys2=float(y[:nt].std()+1e-8);y=((y-ym2)/ys2).astype(np.float32)
    splits={"train":(torch.from_numpy(X[:nt]),torch.from_numpy(y[:nt])),"val":(torch.from_numpy(X[nt:nt+nv]),torch.from_numpy(y[nt:nt+nv])),"test":(torch.from_numpy(X[nt+nv:]),torch.from_numpy(y[nt+nv:]))}
    if torch.cuda.is_available() and S2_GPU_RESIDENT_DATA:splits={k:(a.to(DEVICE,non_blocking=True),b.to(DEVICE,non_blocking=True)) for k,(a,b) in splits.items()}
    meta={**environment_key_dict(cfg),"environment_seed":seed,"n_train":nt,"n_val":nv,"n_test":nte,"support_pixels":int(mask.sum())}
    return splits,meta

# ==================================================================================================
# 8. LOW-FIDELITY ENVIRONMENT DISCOVERY + RF-SMAC
# ==================================================================================================

DISC_RESPONSE_METRICS=["delta_entropy_deficit","delta_log_eta1","delta_log_eta2","delta_hidden_mean_occurrence_cv","delta_hidden_mean_parameter_cv","delta_hidden_mean_S_param","delta_hidden_mean_S_reuse"]

def _arch_from_row(r:pd.Series,seed_index:int)->ConvTraceNet:
    hidden=json.loads(r.hidden_channels) if isinstance(r.hidden_channels,str) else list(r.hidden_channels);kernels=json.loads(r.kernel_sizes) if isinstance(r.kernel_sizes,str) else list(r.kernel_sizes)
    return ConvTraceNet(RESOLUTION,hidden,kernels,ACTIVATION,model_seed(str(r.arch_id),seed_index)).to(DEVICE)

def discovery_part(env_id:str,arch_id:str)->Path:return OUTDIR/"run_parts"/f"DISC__{env_id}__{arch_id}.csv"
def discovery_checkpoint(env_id:str,arch_id:str)->Path:return OUTDIR/"checkpoints_discovery"/f"DISC__{env_id}__{arch_id}__E{DISCOVERY_EPOCHS:03d}.pt"

def _record_state_row(model,epoch,base):
    st=cnn_structural_state(model,False);row={**base,"epoch":int(epoch)}
    for k,v in st.items():
        if k not in {"node_flow","occurrence_flow","parameter_flow","layer_metrics"}:row[k]=v
    row["log_eta1"]=safe_log(row.get("eta1",np.nan));row["log_eta2"]=safe_log(row.get("eta2",np.nan));return row

def train_discovery_cell(env:Dict[str,Any],arch:pd.Series,splits:Dict[str,Any])->pd.DataFrame:
    eid=str(env["env_id"]);aid=str(arch.arch_id);path=discovery_part(eid,aid)
    if S2_RESUME and path.exists():return pd.read_csv(path)
    model=_arch_from_row(arch,0);Xtr,ytr=splits["train"];Xv,yv=splits["val"];Xt,yt=splits["test"];g=torch.Generator(device="cpu").manual_seed(model_seed(aid,0)+777);Xcal=torch.randn((256,1,RESOLUTION,RESOLUTION),generator=g).to(DEVICE);calibrate_initial_signal(model,Xcal);base={"env_id":eid,"arch_id":aid,"seed_index":0,**environment_key_dict(env)};rows=[_record_state_row(model,0,base)];opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY);diverged=False
    for epoch in range(1,DISCOVERY_EPOCHS+1):
        model.train()
        for ids in deterministic_batches(len(Xtr),epoch-1,0,BATCH_SIZE):
            ids_t=torch.as_tensor(ids,device=DEVICE);xb=Xtr.index_select(0,ids_t) if Xtr.device.type=="cuda" else Xtr[ids].to(DEVICE);yb=ytr.index_select(0,ids_t) if ytr.device.type=="cuda" else ytr[ids].to(DEVICE);opt.zero_grad(set_to_none=True);pred=model(xb);loss=F.mse_loss(pred,yb)
            if not torch.isfinite(loss):diverged=True;break
            loss.backward();opt.step()
        if diverged:break
        if epoch in DISCOVERY_STATE_GRID:rows.append(_record_state_row(model,epoch,base))
    val=eval_mse(model,Xv,yv) if not diverged else np.inf;test=eval_mse(model,Xt,yt) if not diverged else np.inf
    df=pd.DataFrame(rows);df["diverged"]=diverged;df["final_val_mse"]=val;df["final_test_mse"]=test;atomic_csv(df,path)
    if not diverged:torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"env":env,"arch_id":aid},discovery_checkpoint(eid,aid))
    del model,opt;return df

def summarize_discovery_environment(env:Dict[str,Any],cell_frames:List[pd.DataFrame],archive:pd.DataFrame)->Dict[str,Any]:
    rows=[]
    for d in cell_frames:
        if d.empty:continue
        aid=str(d.arch_id.iloc[0]);p=d.pivot(index="arch_id",columns="epoch")
        if 0 not in d.epoch.values or DISCOVERY_EPOCHS not in d.epoch.values:continue
        a=d[d.epoch==0].iloc[0];b=d[d.epoch==DISCOVERY_EPOCHS].iloc[0];rec={"arch_id":aid}
        for m in ["entropy_deficit","log_eta1","log_eta2","hidden_mean_occurrence_cv","hidden_mean_parameter_cv","hidden_mean_S_param","hidden_mean_S_reuse"]:rec[f"delta_{m}"]=float(b[m]-a[m])
        rows.append(rec)
    rr=pd.DataFrame(rows);out={**env,"n_sentinels_completed":len(rr)}
    for m in DISC_RESPONSE_METRICS:out[f"median_{m}"]=float(rr[m].median()) if m in rr else np.nan
    # AxE disagreement is robust dispersion across sentinel response vectors.
    if len(rr)>=2:
        X=rr[DISC_RESPONSE_METRICS].to_numpy(float);X=RobustScaler().fit_transform(X);dist=[]
        for i in range(len(X)):
            for j in range(i+1,len(X)):dist.append(np.linalg.norm(X[i]-X[j]))
        out["ae_disagreement"]=float(np.median(dist)) if dist else 0.0
    else:out["ae_disagreement"]=np.nan
    # Structural novelty is distance from prior environment-level structural response vectors.
    cols=[f"median_{m}" for m in DISC_RESPONSE_METRICS];v=np.asarray([out[c] for c in cols],float)
    prev=archive.dropna(subset=cols) if not archive.empty and all(c in archive.columns for c in cols) else pd.DataFrame()
    if len(prev)>=2 and np.all(np.isfinite(v)):
        X=prev[cols].to_numpy(float);sc=RobustScaler().fit(X);Xs=sc.transform(X);vs=sc.transform(v[None,:])[0];dd=np.linalg.norm(Xs-vs,axis=1);out["structural_novelty"]=float(np.median(np.partition(dd,min(3,len(dd))-1)[:min(3,len(dd))]))
    else:out["structural_novelty"]=1.0
    return out

def environment_novelty(cand:Dict[str,Any],archive:pd.DataFrame)->float:
    if archive.empty:return 1.0
    X=np.vstack([environment_vector(r) for r in archive.to_dict("records")]);v=environment_vector(cand);d=np.linalg.norm(X-v[None,:],axis=1);return float(np.min(d))

def _rf_predict_mean_sd(rf:RandomForestRegressor,X:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
    preds=np.stack([t.predict(X) for t in rf.estimators_],axis=0);return preds.mean(0),preds.std(0)

def propose_rfsmac_environment(archive:pd.DataFrame,step:int)->Dict[str,Any]:
    existing={stable_hash_dict(environment_key_dict(r),16) for r in archive.to_dict("records")};pool=random_environment_pool(RFSMAC_POOL,S2_SEED+1000+step,existing);rng=np.random.default_rng(S2_SEED+2000+step)
    if not pool:raise RuntimeError("RF-SMAC candidate pool exhausted")
    if rng.random()<RFSMAC_RANDOM_INTERLEAVE or len(archive)<RFSMAC_MIN_TRAIN:
        # Coverage/random interleaving: choose one of the high-novelty candidates stochastically.
        nov=np.asarray([environment_novelty(c,archive) for c in pool]);top=np.argsort(nov)[-max(1,len(pool)//5):];cand=pool[int(rng.choice(top))];cand["rfsmac_acquisition"]=np.nan;cand["rfsmac_mode"]="random_coverage";return cand
    X=np.vstack([environment_vector(r) for r in archive.to_dict("records")]);Xp=np.vstack([environment_vector(r) for r in pool])
    yN=archive.structural_novelty.astype(float).to_numpy();yI=archive.ae_disagreement.astype(float).to_numpy();rfN=RandomForestRegressor(n_estimators=RFSMAC_TREES,min_samples_leaf=2,max_features=.8,random_state=S2_SEED+step,n_jobs=-1).fit(X,yN);rfI=RandomForestRegressor(n_estimators=RFSMAC_TREES,min_samples_leaf=2,max_features=.8,random_state=S2_SEED+300+step,n_jobs=-1).fit(X,yI);muN,sdN=_rf_predict_mean_sd(rfN,Xp);muI,sdI=_rf_predict_mean_sd(rfI,Xp);unc=(sdN+sdI)/2;nov=np.asarray([environment_novelty(c,archive) for c in pool])
    def robust01(v):
        v=np.asarray(v,float);lo=np.nanquantile(v,.05);hi=np.nanquantile(v,.95);return np.clip((v-lo)/max(hi-lo,1e-12),0,1)
    acq=.30*robust01(unc)+.25*robust01(nov)+.25*robust01(muN)+.20*robust01(muI);j=int(np.nanargmax(acq));cand=pool[j];cand["rfsmac_acquisition"]=float(acq[j]);cand["rfsmac_mode"]="model_based";cand["pred_structural_novelty"]=float(muN[j]);cand["pred_ae_disagreement"]=float(muI[j]);cand["pred_uncertainty"]=float(unc[j]);return cand

def evaluate_environment(env:Dict[str,Any],sentinels:pd.DataFrame,archive:pd.DataFrame)->Dict[str,Any]:
    splits,_=environment_dataset(env,full=False);cells=[]
    for _,a in tqdm(sentinels.iterrows(),total=len(sentinels),desc=f"{env['env_id']} sentinels",leave=False):cells.append(train_discovery_cell(env,a,splits))
    rec=summarize_discovery_environment(env,cells,archive);rec["environment_novelty_at_evaluation"]=environment_novelty(env,archive);del splits
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return rec

def run_environment_discovery(sentinels:pd.DataFrame)->pd.DataFrame:
    path=OUTDIR/"tables"/"environment_discovery_archive.csv";archive=safe_read(path) if S2_RESUME else pd.DataFrame();done=set(archive.env_id.astype(str)) if not archive.empty else set();initial=generate_initial_environments();eta=ETAReporter("S2 environment discovery",DISCOVERY_ENV_TOTAL);completed=len(done)
    for env in initial:
        if env["env_id"] in done:continue
        t0=time.perf_counter();rec=evaluate_environment(env,sentinels,archive);archive=pd.concat([archive,pd.DataFrame([rec])],ignore_index=True);atomic_csv(archive,path);done.add(env["env_id"]);completed+=1;eta.update(time.perf_counter()-t0,completed)
    for step in range(RFSMAC_ENV_STEPS):
        # Resume-safe: count existing BO records.
        nbo=int(np.sum(archive.stage.astype(str)=="bo")) if not archive.empty else 0
        if nbo>step:continue
        cand=propose_rfsmac_environment(archive,step);cand["stage"]="bo";t0=time.perf_counter();rec=evaluate_environment(cand,sentinels,archive);archive=pd.concat([archive,pd.DataFrame([rec])],ignore_index=True);atomic_csv(archive,path);completed+=1;eta.update(time.perf_counter()-t0,completed);gpu_snapshot("RF-SMAC")
    eta.finish();return archive

# ==================================================================================================
# 9. DISCOVERY BCANw AND 24-ENVIRONMENT PANEL SELECTION
# ==================================================================================================

def discovery_geometry(archive:pd.DataFrame,sentinels:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    sp=OUTDIR/"tables"/"discovery_geometry.csv";scp=OUTDIR/"tables"/"discovery_geometry_scales.csv";old=safe_read(sp) if S2_RESUME else pd.DataFrame();oldsc=safe_read(scp) if S2_RESUME else pd.DataFrame();done={(str(r.env_id),str(r.arch_id),int(r.epoch)) for _,r in old.iterrows()} if not old.empty else set();rows=old.to_dict("records");scrows=oldsc.to_dict("records")
    # Structural initialization is environment independent for fixed architecture/seed. Cache once per sentinel.
    init_cache={};jobs=[]
    for _,e in archive.iterrows():
        for _,a in sentinels.iterrows():
            for ep in [0,DISCOVERY_EPOCHS]:
                if (str(e.env_id),str(a.arch_id),ep) not in done:jobs.append((e,a,ep))
    eta=ETAReporter("S2 discovery BCANw",len(jobs));n=0
    for e,a,ep in tqdm(jobs,desc="S2 discovery BCANw"):
        t0=time.perf_counter();aid=str(a.arch_id);eid=str(e.env_id)
        if ep==0 and aid in init_cache:gs,gsc=init_cache[aid]
        else:
            model=_arch_from_row(a,0)
            if ep==0:
                # reproduce the same calibration using this environment's training calibration slice; structural init
                # calibration is held fixed by arch+seed by using a deterministic standard Gaussian calibration tensor.
                g=torch.Generator(device="cpu").manual_seed(model_seed(aid,0)+777);Xcal=torch.randn((256,1,RESOLUTION,RESOLUTION),generator=g).to(DEVICE);calibrate_initial_signal(model,Xcal)
            else:
                chk=torch.load(discovery_checkpoint(eid,aid),map_location="cpu");model.load_state_dict(chk["state_dict"])
            st=cnn_structural_state(model,True);gs,gsc=analyze_execution_geometry(model,st,BCANW_SCALES_DISCOVERY,compute_local=True)
            if ep==0:init_cache[aid]=(gs,gsc.copy())
            del model
        rec={"env_id":eid,"arch_id":aid,"seed_index":0,"epoch":ep,**environment_key_dict(e.to_dict()),**gs};rows.append(rec)
        if not gsc.empty:
            gg=gsc.copy()
            for k,v in rec.items():
                if k not in gg.columns:gg[k]=v
            scrows.extend(gg.to_dict("records"))
        atomic_csv(pd.DataFrame(rows),sp);atomic_csv(pd.DataFrame(scrows),scp);done.add((eid,aid,ep));n+=1;eta.update(time.perf_counter()-t0,n)
    eta.finish();return pd.DataFrame(rows),pd.DataFrame(scrows)

def _maximin_env(df:pd.DataFrame,n:int,force_ids:Sequence[str]=())->pd.DataFrame:
    if len(df)<=n:return df.copy()
    X=np.vstack([environment_vector(r) for r in df.to_dict("records")]);chosen=[]
    for fid in force_ids:
        q=np.where(df.env_id.astype(str).to_numpy()==str(fid))[0]
        if len(q):chosen.append(int(q[0]))
    if not chosen:chosen=[int(np.argmax(np.linalg.norm(X-X.mean(0),axis=1)))]
    while len(chosen)<n:
        d=np.min(np.stack([np.linalg.norm(X-X[j],axis=1) for j in chosen]),axis=0);d[chosen]=-np.inf;chosen.append(int(np.argmax(d)))
    return df.iloc[chosen].copy()

def select_environment_panel(archive:pd.DataFrame)->pd.DataFrame:
    path=OUTDIR/"tables"/"selected_24_environments.csv"
    if S2_RESUME and path.exists():return pd.read_csv(path)
    anchors=archive[archive.stage.astype(str)=="anchor"];forced=anchors.env_id.astype(str).tolist();initial=archive[archive.stage.astype(str).isin(["anchor","sobol"])].copy();bo=archive[archive.stage.astype(str)=="bo"].copy();p0=_maximin_env(initial,min(SELECT_INITIAL,len(initial)),force_ids=forced)
    # BO selection balances environment-space distance to already selected points with actual structural novelty and AxE disagreement.
    selected=p0.copy();remaining=bo.copy()
    while len(selected)<min(N_SELECTED_ENV,len(archive)) and len(remaining):
        vals=[]
        for idx,r in remaining.iterrows():
            nov=environment_novelty(r.to_dict(),selected);sn=float(r.get("structural_novelty",0));ae=float(r.get("ae_disagreement",0));vals.append((idx,nov,sn,ae))
        arr=np.asarray([[x[1],x[2],x[3]] for x in vals],float)
        if len(arr)>1:arr=RobustScaler().fit_transform(arr)
        score=.40*arr[:,0]+.35*arr[:,1]+.25*arr[:,2];idx=vals[int(np.argmax(score))][0];selected=pd.concat([selected,remaining.loc[[idx]]]);remaining=remaining.drop(idx)
    if len(selected)<min(N_SELECTED_ENV,len(archive)):
        rest=archive[~archive.env_id.astype(str).isin(selected.env_id.astype(str))];extra=_maximin_env(pd.concat([selected,rest]),min(N_SELECTED_ENV,len(archive)),force_ids=selected.env_id.astype(str).tolist());selected=extra
    selected=selected.drop_duplicates("env_id").iloc[:N_SELECTED_ENV].copy();selected["panel_order"]=np.arange(len(selected));atomic_csv(selected,path);return selected

# ==================================================================================================
# 10. FULL 24 x 12 x 2 LONGITUDINAL MATRIX
# ==================================================================================================

def full_run_id(env_id:str,arch_id:str,seed:int)->str:return f"S2FULL__{env_id}__{arch_id}__S{seed:02d}"
def full_part(run_id:str,kind:str)->Path:return OUTDIR/"run_parts"/f"{run_id}__{kind}.csv"
def full_checkpoint(run_id:str,epoch:int)->Path:return OUTDIR/"checkpoints_full"/f"{run_id}__E{epoch:03d}.pt"

def full_complete(run_id:str)->bool:
    sp=full_part(run_id,"summary")
    if not sp.exists():return False
    sm=safe_read(sp)
    if not sm.empty and bool(sm.iloc[0].get("diverged",False)):return True
    return all(full_part(run_id,k).exists() for k in ["states","history","summary"]) and all(full_checkpoint(run_id,e).exists() for e in STATE_GRID)

def train_full_run(env:Dict[str,Any],arch:pd.Series,seed_index:int,splits:Dict[str,Any]):
    eid=str(env["env_id"]);aid=str(arch.arch_id);rid=full_run_id(eid,aid,seed_index)
    if S2_RESUME and full_complete(rid):return
    model=_arch_from_row(arch,seed_index);Xtr,ytr=splits["train"];Xv,yv=splits["val"];Xt,yt=splits["test"]
    # Calibration must not vary with environment. Use deterministic standard-normal calibration keyed only by arch+seed.
    g=torch.Generator(device="cpu").manual_seed(model_seed(aid,seed_index)+777);Xcal=torch.randn((256,1,RESOLUTION,RESOLUTION),generator=g).to(DEVICE);cal_scales=calibrate_initial_signal(model,Xcal)
    base={"run_id":rid,"env_id":eid,"arch_id":aid,"seed_index":int(seed_index),**environment_key_dict(env)}
    for c in ARCH_DESCRIPTOR_COLS:
        if c in arch.index:base[c]=arch[c]
    states=[_record_state_row(model,0,base)];history=[];torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"base":base,"calibration_scales":cal_scales},full_checkpoint(rid,0));opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY);diverged=False
    # Evaluate function only on common state grid; checkpoint selection is validation-based.
    val0=eval_mse(model,Xv,yv);test0=eval_mse(model,Xt,yt);history.append({**base,"epoch":0,"train_mse":np.nan,"val_mse":val0,"test_mse":test0})
    pbar=tqdm(range(1,FULL_EPOCHS+1),desc=f"{eid} | {aid} | s{seed_index}",leave=False)
    for epoch in pbar:
        model.train();losses=[]
        for ids in deterministic_batches(len(Xtr),epoch-1,seed_index,BATCH_SIZE):
            ids_t=torch.as_tensor(ids,device=DEVICE);xb=Xtr.index_select(0,ids_t) if Xtr.device.type=="cuda" else Xtr[ids].to(DEVICE);yb=ytr.index_select(0,ids_t) if ytr.device.type=="cuda" else ytr[ids].to(DEVICE);opt.zero_grad(set_to_none=True);pred=model(xb);loss=F.mse_loss(pred,yb)
            if not torch.isfinite(loss):diverged=True;break
            loss.backward();opt.step();losses.append(float(loss.item()))
        if diverged:break
        if epoch in STATE_GRID:
            val=eval_mse(model,Xv,yv);test=eval_mse(model,Xt,yt);history.append({**base,"epoch":epoch,"train_mse":float(np.mean(losses)),"val_mse":val,"test_mse":test});states.append(_record_state_row(model,epoch,base));torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"base":base,"calibration_scales":cal_scales},full_checkpoint(rid,epoch));atomic_csv(pd.DataFrame(states),full_part(rid,"states"));atomic_csv(pd.DataFrame(history),full_part(rid,"history"))
    h=pd.DataFrame(history);best=h.loc[h.val_mse.astype(float).idxmin()] if len(h) else None;summary={**base,"diverged":bool(diverged),"epochs_completed":int(h.epoch.max()) if len(h) else 0,"best_val_epoch":int(best.epoch) if best is not None else -1,"best_val_mse":float(best.val_mse) if best is not None else np.inf,"test_at_best_val":float(best.test_mse) if best is not None else np.inf,"final_test_mse":float(h[h.epoch==h.epoch.max()].test_mse.iloc[0]) if len(h) else np.inf,"post_best_val_test_deterioration":float(h[h.epoch==h.epoch.max()].test_mse.iloc[0]-best.test_mse) if best is not None else np.nan}
    atomic_csv(pd.DataFrame(states),full_part(rid,"states"));atomic_csv(h,full_part(rid,"history"));atomic_csv(pd.DataFrame([summary]),full_part(rid,"summary"));del model,opt

def run_full_matrix(selected:pd.DataFrame,panel:pd.DataFrame):
    jobs_total=sum(not (S2_RESUME and full_complete(full_run_id(str(e.env_id),str(a.arch_id),s))) for _,e in selected.iterrows() for _,a in panel.iterrows() for s in FULL_SEEDS);eta=ETAReporter("S2 full 24x12x2 matrix",jobs_total);done=0
    for _,e in tqdm(selected.iterrows(),total=len(selected),desc="CNN-S2 selected environments"):
        pending=[(a,s) for _,a in panel.iterrows() for s in FULL_SEEDS if not (S2_RESUME and full_complete(full_run_id(str(e.env_id),str(a.arch_id),s)))]
        if not pending:continue
        splits,_=environment_dataset(e.to_dict(),full=True)
        for a,s in pending:
            t0=time.perf_counter();train_full_run(e.to_dict(),a,s,splits);done+=1;eta.update(time.perf_counter()-t0,done)
        del splits
        if torch.cuda.is_available():torch.cuda.empty_cache();gpu_snapshot("S2 full matrix")
    eta.finish()

def aggregate_full(selected:pd.DataFrame,panel:pd.DataFrame)->Dict[str,pd.DataFrame]:
    kinds={k:[] for k in ["states","history","summary"]}
    for _,e in selected.iterrows():
        for _,a in panel.iterrows():
            for s in FULL_SEEDS:
                rid=full_run_id(str(e.env_id),str(a.arch_id),s)
                for k in kinds:
                    d=safe_read(full_part(rid,k))
                    if not d.empty:kinds[k].append(d)
    out={k:(pd.concat(v,ignore_index=True) if v else pd.DataFrame()) for k,v in kinds.items()}
    for k,d in out.items():atomic_csv(d,OUTDIR/"tables"/f"full_{k}.csv")
    return out

# ==================================================================================================
# 11. FULL BCANw MATRIX + DENSE SENTINEL SUPPLEMENT
# ==================================================================================================

def run_full_geometry(selected:pd.DataFrame,panel:pd.DataFrame,sentinels:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    sp=OUTDIR/"tables"/"full_geometry.csv";scp=OUTDIR/"tables"/"full_geometry_scales.csv";old=safe_read(sp) if S2_RESUME else pd.DataFrame();oldsc=safe_read(scp) if S2_RESUME else pd.DataFrame();done={(str(r.run_id),int(r.epoch)) for _,r in old.iterrows()} if not old.empty else set();rows=old.to_dict("records");scrows=oldsc.to_dict("records");sent_ids=set(sentinels.arch_id.astype(str));jobs=[]
    for _,e in selected.iterrows():
        for _,a in panel.iterrows():
            grid=DENSE_SENTINEL_GEOMETRY_GRID if str(a.arch_id) in sent_ids else GEOMETRY_GRID
            for s in FULL_SEEDS:
                rid=full_run_id(str(e.env_id),str(a.arch_id),s)
                for ep in grid:
                    if ep<=FULL_EPOCHS and (rid,ep) not in done:jobs.append((e,a,s,ep))
    eta=ETAReporter("S2 full BCANw",len(jobs));n=0
    for e,a,s,ep in tqdm(jobs,desc="CNN-S2 full BCANw"):
        t0=time.perf_counter();rid=full_run_id(str(e.env_id),str(a.arch_id),s);model=_arch_from_row(a,s);chk=torch.load(full_checkpoint(rid,ep),map_location="cpu");model.load_state_dict(chk["state_dict"]);model.to(DEVICE);st=cnn_structural_state(model,True);gs,gsc=analyze_execution_geometry(model,st,BCANW_SCALES_LONG,compute_local=True);rec={"run_id":rid,"env_id":str(e.env_id),"arch_id":str(a.arch_id),"seed_index":int(s),"epoch":int(ep),**environment_key_dict(e.to_dict()),**gs};rows.append(rec)
        if not gsc.empty:
            gg=gsc.copy()
            for k,v in rec.items():
                if k not in gg.columns:gg[k]=v
            scrows.extend(gg.to_dict("records"))
        atomic_csv(pd.DataFrame(rows),sp);atomic_csv(pd.DataFrame(scrows),scp);done.add((rid,ep));n+=1;eta.update(time.perf_counter()-t0,n);del model
        if torch.cuda.is_available() and n%8==0:gpu_snapshot("BCANw (CPU-heavy)")
    eta.finish();return pd.DataFrame(rows),pd.DataFrame(scrows)

# ==================================================================================================
# 12. PLANNED S2 INFERENCE
# ==================================================================================================

STATE_METRICS=["entropy_deficit","log_eta1","log_eta2","hidden_mean_occurrence_cv","hidden_mean_parameter_cv","hidden_mean_S_occ","hidden_mean_S_param","hidden_mean_S_reuse"]
GEOM_METRICS=["log_xi50","d_star","lacunarity_star","local_delta_median","quotient_mean_pairwise_KS","renorm_flow_retention_mean"]

def _add_derived_columns(states:pd.DataFrame,geometry:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    states=states.copy();geometry=geometry.copy()
    if "eta1" in states:states["log_eta1"]=states.eta1.map(safe_log)
    if "eta2" in states:states["log_eta2"]=states.eta2.map(safe_log)
    if not geometry.empty and "xi50" in geometry:geometry["log_xi50"]=geometry.xi50.map(safe_log)
    return states,geometry

def seed_aggregate(df:pd.DataFrame,metrics:Sequence[str])->pd.DataFrame:
    keep=[c for c in ["env_id","arch_id","epoch","teacher_family","informative_fraction","spatial_correlation","snr","interaction_strength","localization"] if c in df.columns];ms=[m for m in metrics if m in df.columns]
    if df.empty or not ms:return pd.DataFrame(columns=keep+ms)
    med=df.groupby(keep,as_index=False)[ms].median(numeric_only=True);cnt=df.groupby(keep,as_index=False).seed_index.nunique().rename(columns={"seed_index":"n_seeds_observed"});out=med.merge(cnt,on=keep,how="left")
    out.loc[out.n_seeds_observed<len(FULL_SEEDS),ms]=np.nan
    return out

def delta_cell_frame(seedagg:pd.DataFrame,metrics:Sequence[str],end_epoch:int)->pd.DataFrame:
    keys=["env_id","arch_id"];meta=seedagg[[c for c in seedagg.columns if c in keys+["teacher_family","informative_fraction","spatial_correlation","snr","interaction_strength","localization"]]].drop_duplicates(keys)
    base=meta.copy()
    for m in metrics:
        if m not in seedagg:continue
        piv=seedagg.pivot_table(index=keys,columns="epoch",values=m,aggfunc="first")
        if 0 in piv.columns and end_epoch in piv.columns:base=base.merge((piv[end_epoch]-piv[0]).rename(f"delta_{m}").reset_index(),on=keys,how="left")
    return base

def s1_sign_law_replication(cell_changes:pd.DataFrame,registry:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    env_rows=[];overall=[]
    for _,law in registry.iterrows():
        m=str(law.metric);col=f"delta_{m}";direction=int(law.s1_direction)
        if col not in cell_changes:continue
        for eid,g in cell_changes.groupby("env_id"):
            x=g[col].to_numpy(float);x=x[np.isfinite(x)];ok=(direction*x)>0;k=int(np.sum(ok));lo,hi=clopper_pearson(k,len(x));env_rows.append({"metric":m,"s1_direction":direction,"env_id":eid,"n_architectures":len(x),"consistent":k,"consistency_fraction":k/max(len(x),1),"consistency_lo":lo,"consistency_hi":hi})
        x=cell_changes[col].to_numpy(float);x=x[np.isfinite(x)];k=int(np.sum(direction*x>0));lo,hi=clopper_pearson(k,len(x));overall.append({"metric":m,"s1_direction":direction,"n_architecture_environment_cells":len(x),"consistent":k,"overall_consistency":k/max(len(x),1),"overall_consistency_lo":lo,"overall_consistency_hi":hi})
    a=pd.DataFrame(env_rows);b=pd.DataFrame(overall)
    if not a.empty:
        summ=a.groupby("metric").consistency_fraction.agg(["median","min","max"]).reset_index();summ=summ.rename(columns={"median":"median_environment_consistency","min":"min_environment_consistency","max":"max_environment_consistency"});thr=a.assign(over80=a.consistency_fraction>=.8,over90=a.consistency_fraction>=.9).groupby("metric")[["over80","over90"]].mean().reset_index();b=b.merge(summ,on="metric",how="left").merge(thr,on="metric",how="left")
    atomic_csv(a,OUTDIR/"tables"/"planned_s1_law_consistency_by_environment.csv");atomic_csv(b,OUTDIR/"tables"/"planned_s1_law_consistency_summary.csv");return a,b

def balanced_variance_components(df:pd.DataFrame,metric:str)->Dict[str,float]:
    d=df[["arch_id","env_id","seed_index",metric]].dropna().copy();A=sorted(d.arch_id.unique());E=sorted(d.env_id.unique());S=sorted(d.seed_index.unique())
    if len(A)<2 or len(E)<2 or len(S)<2 or len(d)!=len(A)*len(E)*len(S):return {"valid_balanced":False}
    y=d.set_index(["arch_id","env_id","seed_index"])[metric];grand=float(y.mean());meanA=d.groupby("arch_id")[metric].mean();meanE=d.groupby("env_id")[metric].mean();meanAE=d.groupby(["arch_id","env_id"])[metric].mean();a,e,s=len(A),len(E),len(S)
    SSA=e*s*float(np.sum((meanA-grand)**2));SSEn=a*s*float(np.sum((meanE-grand)**2));SSAE=s*sum((float(meanAE.loc[(aa,ee)])-float(meanA.loc[aa])-float(meanE.loc[ee])+grand)**2 for aa in A for ee in E);SSe=sum((float(r[metric])-float(meanAE.loc[(r.arch_id,r.env_id)]))**2 for _,r in d.iterrows());dfA=a-1;dfE=e-1;dfAE=(a-1)*(e-1);dferr=a*e*(s-1);MSA=SSA/dfA;MSE=SSEn/dfE;MSAE=SSAE/dfAE;MSerr=SSe/dferr
    sig_err=max(MSerr,0);sig_ae=max((MSAE-MSerr)/s,0);sig_a=max((MSA-MSAE)/(e*s),0);sig_e=max((MSE-MSAE)/(a*s),0);structured=sig_a+sig_e+sig_ae;total=structured+sig_err
    return {"valid_balanced":True,"n_arch":a,"n_env":e,"n_seed":s,"sigma2_arch":sig_a,"sigma2_env":sig_e,"sigma2_AxE":sig_ae,"sigma2_residual":sig_err,"U_arch":sig_a/structured if structured>0 else np.nan,"U_env":sig_e/structured if structured>0 else np.nan,"U_AxE":sig_ae/structured if structured>0 else np.nan,"residual_total_fraction":sig_err/total if total>0 else np.nan}

def variance_decomposition(states:pd.DataFrame,geometry:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    final_rows=[];time_rows=[]
    for source,df,metrics in [("state",states,STATE_METRICS),("geometry",geometry,GEOM_METRICS)]:
        if df.empty:continue
        for m in metrics:
            if m not in df:continue
            for ep in sorted(df.epoch.unique()):
                rec={"source":source,"metric":m,"epoch":int(ep),**balanced_variance_components(df[df.epoch==ep],m)};time_rows.append(rec)
            end=max(df.epoch.unique());final_rows.append({"source":source,"metric":m,"epoch":int(end),**balanced_variance_components(df[df.epoch==end],m)})
    a=pd.DataFrame(final_rows);b=pd.DataFrame(time_rows);atomic_csv(a,OUTDIR/"tables"/"planned_final_variance_decomposition.csv");atomic_csv(b,OUTDIR/"tables"/"planned_temporal_variance_decomposition.csv");return a,b

def architecture_rank_universality(seedagg:pd.DataFrame,metrics:Sequence[str],epoch:int)->Tuple[pd.DataFrame,pd.DataFrame]:
    rows=[];summ=[];d=seedagg[seedagg.epoch==epoch]
    for m in metrics:
        if m not in d:continue
        piv=d.pivot(index="arch_id",columns="env_id",values=m);envs=list(piv.columns)
        vals=[]
        for i in range(len(envs)):
            for j in range(i+1,len(envs)):
                comp=piv[[envs[i],envs[j]]].dropna();rho=spearmanr(comp.iloc[:,0],comp.iloc[:,1]).statistic if len(comp)>=5 else np.nan;rows.append({"metric":m,"env_a":envs[i],"env_b":envs[j],"n_architectures":len(comp),"spearman":rho});vals.append(rho)
        x=np.asarray(vals,float);x=x[np.isfinite(x)];summ.append({"metric":m,"n_environment_pairs":len(x),"median_spearman":float(np.median(x)) if len(x) else np.nan,"p16":float(np.quantile(x,.16)) if len(x) else np.nan,"p84":float(np.quantile(x,.84)) if len(x) else np.nan,"fraction_rho_gt_0p8":float(np.mean(x>.8)) if len(x) else np.nan})
    a=pd.DataFrame(rows);b=pd.DataFrame(summ);atomic_csv(a,OUTDIR/"tables"/"planned_cross_environment_architecture_rank_pairs.csv");atomic_csv(b,OUTDIR/"tables"/"planned_cross_environment_architecture_rank_summary.csv");return a,b

def scaling_two_stage(geometry:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    if geometry.empty:return pd.DataFrame(),pd.DataFrame()
    rows_c=[];rows_d=[]
    for eid,g in geometry.groupby("env_id"):
        p=g.pivot_table(index=["arch_id","seed_index"],columns="epoch",values="scaling_credible",aggfunc="first")
        if 0 in p.columns and FULL_EPOCHS in p.columns:
            both=p[[0,FULL_EPOCHS]].dropna();b0=both[0].astype(int);b1=both[FULL_EPOCHS].astype(int);n01=int(np.sum((b0==0)&(b1==1)));n10=int(np.sum((b0==1)&(b1==0)));n=n01+n10
            # Exact McNemar via binomial sign test on discordant pairs.
            if n:pv=float(binomtest(n01,n,.5,alternative="two-sided").pvalue)
            else:pv=1.0
            rows_c.append({"env_id":eid,"n_blocks":len(both),"credible_rate_init":float(b0.mean()),"credible_rate_final":float(b1.mean()),"n_0_to_1":n01,"n_1_to_0":n10,"mcnemar_exact_p":pv})
        q=g[g.scaling_credible.astype(bool)].pivot_table(index=["arch_id","seed_index"],columns="epoch",values="scaling_dB",aggfunc="first")
        if 0 in q.columns and FULL_EPOCHS in q.columns:
            d=(q[FULL_EPOCHS]-q[0]).dropna().to_numpy(float)
            if len(d):
                try:st,pv=wilcoxon(d,zero_method="wilcox")
                except Exception:st,pv=np.nan,1.
                rows_d.append({"env_id":eid,"n_both_credible":len(d),"median_dB_change":float(np.median(d)),"hodges_lehmann":hodges_lehmann_onesample(d),"rank_biserial":paired_rank_biserial(d),"p_raw":pv})
    a=pd.DataFrame(rows_c);b=pd.DataFrame(rows_d)
    if not a.empty:a["p_holm"]=holm_adjust(a.mcnemar_exact_p.to_numpy(float))
    if not b.empty:b["p_holm"]=holm_adjust(b.p_raw.to_numpy(float))
    atomic_csv(a,OUTDIR/"tables"/"planned_scaling_credibility_by_environment.csv");atomic_csv(b,OUTDIR/"tables"/"planned_dB_change_given_credibility.csv");return a,b

# ==================================================================================================
# 13. EXPLICIT POST-HOC / EXPLANATORY ANALYSES
# ==================================================================================================

def posthoc_environment_associations(cell_changes:pd.DataFrame,summary:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    responses=[c for c in cell_changes.columns if c.startswith("delta_")];envmean=cell_changes.groupby(["env_id","teacher_family","informative_fraction","spatial_correlation","snr","interaction_strength","localization"],as_index=False)[responses].median(numeric_only=True)
    continuous=["informative_fraction","spatial_correlation","snr","interaction_strength","localization"];rows=[]
    for y in responses:
        for x in continuous:
            sub=envmean[[x,y]].dropna();rho,p=spearmanr(sub[x],sub[y]) if len(sub)>=8 else (np.nan,np.nan);rows.append({"analysis":"posthoc_environment_spearman","response":y,"coordinate":x,"n_environments":len(sub),"rho_s":rho,"p_raw":p})
    corr=pd.DataFrame(rows)
    if not corr.empty:
        corr["p_holm"]=np.nan
        for y,g in corr.groupby("response"):corr.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    omnibus=[];pairs=[]
    for y in responses:
        groups=[];labels=[]
        for k,g in envmean.groupby("teacher_family"):
            x=g[y].dropna().to_numpy(float)
            if len(x)>=3:groups.append(x);labels.append(str(k))
        if len(groups)>=2:
            try:H,p=kruskal(*groups);n=sum(len(x) for x in groups);kk=len(groups);eps2=max(0.,(H-kk+1)/max(n-kk,1))
            except Exception:H,p,eps2=np.nan,np.nan,np.nan
            omnibus.append({"analysis":"posthoc_teacher_kruskal","response":y,"n_environments":n,"groups":kk,"H":H,"epsilon_squared":eps2,"p_raw":p})
            if np.isfinite(p) and p<POSTHOC_ALPHA:
                for i in range(len(groups)):
                    for j in range(i+1,len(groups)):
                        try:U,pij=mannwhitneyu(groups[i],groups[j],alternative="two-sided")
                        except Exception:U,pij=np.nan,np.nan
                        pairs.append({"response":y,"family_a":labels[i],"family_b":labels[j],"n_a":len(groups[i]),"n_b":len(groups[j]),"median_a_minus_b":float(np.median(groups[i])-np.median(groups[j])),"rank_biserial":mannwhitney_rank_biserial(groups[i],groups[j]),"p_raw":pij})
    om=pd.DataFrame(omnibus);pa=pd.DataFrame(pairs)
    if not om.empty:om["p_holm"]=holm_adjust(om.p_raw.to_numpy(float))
    if not pa.empty:
        pa["p_holm"]=np.nan
        for y,g in pa.groupby("response"):pa.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    # Exploratory structural-functional bridge, aggregated at the environment level.
    bridge=[]
    if not summary.empty:
        fs=summary.groupby("env_id",as_index=False)[["test_at_best_val","post_best_val_test_deterioration"]].median(numeric_only=True);em=envmean.merge(fs,on="env_id",how="left")
        for y in responses:
            for f in ["test_at_best_val","post_best_val_test_deterioration"]:
                sub=em[[y,f]].dropna();rho,p=spearmanr(sub[y],sub[f]) if len(sub)>=8 else (np.nan,np.nan);bridge.append({"analysis":"posthoc_structural_functional","structural_response":y,"functional_response":f,"n_environments":len(sub),"rho_s":rho,"p_raw":p})
    br=pd.DataFrame(bridge)
    if not br.empty:
        br["p_holm"]=np.nan
        for f,g in br.groupby("functional_response"):br.loc[g.index,"p_holm"]=holm_adjust(g.p_raw.to_numpy(float))
    atomic_csv(envmean,OUTDIR/"tables"/"posthoc_environment_level_response_means.csv");atomic_csv(corr,OUTDIR/"tables"/"posthoc_environment_spearman.csv");atomic_csv(om,OUTDIR/"tables"/"posthoc_teacher_omnibus.csv");atomic_csv(pa,OUTDIR/"tables"/"posthoc_teacher_pairs.csv");atomic_csv(br,OUTDIR/"tables"/"posthoc_structural_functional_bridge.csv");return corr,om,pa,br

# ==================================================================================================
# 14. PUBLICATION-STYLE FIGURES + AUTOMATIC REPORT
# ==================================================================================================

mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],"mathtext.fontset":"stix","font.size":8.6,"axes.labelsize":8.8,"axes.titlesize":9.5,"xtick.labelsize":7.4,"ytick.labelsize":7.4,"legend.fontsize":7.1,"axes.linewidth":0.8,"xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,"pdf.fonttype":42,"ps.fonttype":42,"figure.dpi":130,"savefig.dpi":360})
COL_ARCH="#34495e";COL_ENV="#b05a3c";COL_AXE="#728e6b";COL_RES="#b4b4b4";COL_STRUCT="#263f75";COL_REUSE="#76528b";COL_PARAM="#3d7f6f";COL_BO="#9a5b3f";COL_INIT="#4c78a8"

def panel_label(ax,label):ax.text(-.12,1.05,label,transform=ax.transAxes,fontweight="bold",fontsize=10,va="top")
def subtle_grid(ax,axis="both"):ax.grid(True,axis=axis,alpha=.12,lw=.55);ax.set_axisbelow(True)
def save_figure(fig,stem):
    fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(OUTDIR/"figures"/f"{stem}.pdf",bbox_inches="tight");fig.savefig(OUTDIR/"figures"/f"{stem}.png",bbox_inches="tight",dpi=300);plt.close(fig)

def make_figures(archive,selected,states,geometry,var_final,var_time,law_env,rank_summary,postcorr,summary):
    stems=[]
    # F01 environment discovery and selection.
    fig,axs=plt.subplots(2,2,figsize=(8.0,6.0));
    for stage,color,mark in [("anchor","0.2","s"),("sobol",COL_INIT,"o"),("bo",COL_BO,"^")]:
        d=archive[archive.stage.astype(str)==stage];axs[0,0].scatter(d.informative_fraction,d.snr,s=18,alpha=.7,label=stage,color=color,marker=mark)
    sel=set(selected.env_id.astype(str));d=archive[archive.env_id.astype(str).isin(sel)];axs[0,0].scatter(d.informative_fraction,d.snr,s=48,facecolors="none",edgecolors="black",linewidths=.8);axs[0,0].set_yscale("log");axs[0,0].set_xlabel("informative spatial fraction");axs[0,0].set_ylabel("SNR");axs[0,0].set_title("Explored and selected environments");axs[0,0].legend(frameon=False);subtle_grid(axs[0,0]);panel_label(axs[0,0],"a")
    axs[0,1].scatter(archive.localization,archive.spatial_correlation,c=archive.structural_novelty,s=22,alpha=.75);axs[0,1].set_xlabel("localization");axs[0,1].set_ylabel("spatial correlation");axs[0,1].set_title("Structural novelty over environment space");subtle_grid(axs[0,1]);panel_label(axs[0,1],"b")
    bo=archive[archive.stage.astype(str)=="bo"];axs[1,0].plot(np.arange(len(bo)),bo.structural_novelty,marker="o",ms=3,lw=.8,color=COL_BO,label="structural novelty");axs[1,0].plot(np.arange(len(bo)),bo.ae_disagreement,marker="s",ms=3,lw=.8,color=COL_ARCH,label=r"$A\times E$ disagreement");axs[1,0].set_xlabel("RF-SMAC accepted environment");axs[1,0].set_title("Information-seeking refinement");axs[1,0].legend(frameon=False);subtle_grid(axs[1,0]);panel_label(axs[1,0],"c")
    tab=selected.groupby("teacher_family").size().reindex(TEACHER_FAMILIES,fill_value=0);axs[1,1].bar(np.arange(len(tab)),tab.values,color="0.55");axs[1,1].set_xticks(np.arange(len(tab)),[x.replace("_","\n") for x in tab.index]);axs[1,1].set_ylabel("selected environments");axs[1,1].set_title("Selected teacher-family coverage");panel_label(axs[1,1],"d");save_figure(fig,"S2_F01_environment_discovery");stems.append("S2_F01_environment_discovery")
    # F02 law replication + final variance decomposition.
    fig,axs=plt.subplots(1,2,figsize=(8.0,3.5));
    if not law_env.empty:
        s=law_env.groupby("metric").consistency_fraction.median().sort_values();axs[0].barh(np.arange(len(s)),s.values,color=COL_STRUCT);axs[0].axvline(.8,color="0.45",ls="--",lw=.8);axs[0].axvline(.9,color="0.45",ls=":",lw=.8);axs[0].set_yticks(np.arange(len(s)),s.index);axs[0].set_xlim(0,1.02);axs[0].set_xlabel("median environment sign consistency");axs[0].set_title("Replication of frozen S1 laws");subtle_grid(axs[0],"x");panel_label(axs[0],"a")
    vf=var_final[var_final.valid_balanced.astype(bool)] if not var_final.empty and "valid_balanced" in var_final else var_final
    if not vf.empty:
        d=vf.set_index("metric")[["U_arch","U_env","U_AxE"]].dropna();y=np.arange(len(d));left=np.zeros(len(d));
        for c,col in [("U_arch",COL_ARCH),("U_env",COL_ENV),("U_AxE",COL_AXE)]:axs[1].barh(y,d[c].values,left=left,label=c.replace("U_",""),color=col);left+=d[c].values
        axs[1].set_yticks(y,d.index);axs[1].set_xlim(0,1);axs[1].set_xlabel("structured variance fraction");axs[1].set_title("Architecture / environment / interaction");axs[1].legend(frameon=False,ncol=3,loc="lower right");panel_label(axs[1],"b")
    save_figure(fig,"S2_F02_laws_and_variance");stems.append("S2_F02_laws_and_variance")
    # F03 parameter/reuse trajectories.
    fig,axs=plt.subplots(1,3,figsize=(8.2,2.8));
    for ax,m,title,col in zip(axs,["hidden_mean_S_occ","hidden_mean_S_param","hidden_mean_S_reuse"],[r"$S_{occ}$",r"$S_{param}$",r"$S_{reuse}$"],[COL_STRUCT,COL_PARAM,COL_REUSE]):
        if m in states:
            d=states.groupby("epoch")[m].agg(["median",lambda x:np.quantile(x,.25),lambda x:np.quantile(x,.75)]);d.columns=["med","q1","q3"];ax.plot(d.index,d.med,marker="o",ms=3,color=col);ax.fill_between(d.index,d.q1,d.q3,color=col,alpha=.18,lw=0);ax.set_title(title);ax.set_xlabel("epoch");subtle_grid(ax);panel_label(ax,chr(97+list(axs).index(ax)))
    axs[0].set_ylabel("structural specialization (nats)");save_figure(fig,"S2_F03_sharing_trajectories");stems.append("S2_F03_sharing_trajectories")
    # F04 temporal variance shares for selected metrics.
    metrics=[m for m in ["entropy_deficit","hidden_mean_S_param","hidden_mean_S_reuse","log_xi50"] if m in set(var_time.metric.astype(str))] if not var_time.empty else []
    if metrics:
        fig,axs=plt.subplots(1,len(metrics),figsize=(2.7*len(metrics),2.8),squeeze=False);axs=axs[0]
        for ax,m in zip(axs,metrics):
            d=var_time[var_time.metric==m].sort_values("epoch");ax.plot(d.epoch,d.U_arch,label="architecture",color=COL_ARCH);ax.plot(d.epoch,d.U_env,label="environment",color=COL_ENV);ax.plot(d.epoch,d.U_AxE,label=r"$A\times E$",color=COL_AXE);ax.set_ylim(0,1);ax.set_title(m);ax.set_xlabel("epoch");subtle_grid(ax);panel_label(ax,chr(97+list(axs).index(ax)))
        axs[0].set_ylabel("structured variance fraction");axs[-1].legend(frameon=False,fontsize=6.5);save_figure(fig,"S2_F04_temporal_variance");stems.append("S2_F04_temporal_variance")
    # F05 rank universality and environment associations.
    fig,axs=plt.subplots(1,2,figsize=(8.0,3.4));
    if not rank_summary.empty:
        d=rank_summary.sort_values("median_spearman");axs[0].errorbar(d.median_spearman,np.arange(len(d)),xerr=[d.median_spearman-d.p16,d.p84-d.median_spearman],fmt="o",ms=4,color=COL_STRUCT,capsize=2);axs[0].set_yticks(np.arange(len(d)),d.metric);axs[0].axvline(0,color="0.6",lw=.7);axs[0].set_xlim(-1,1);axs[0].set_xlabel("cross-environment architecture rank $\\rho_s$");axs[0].set_title("Architecture-rank universality");subtle_grid(axs[0],"x");panel_label(axs[0],"a")
    if not postcorr.empty:
        piv=postcorr.pivot(index="response",columns="coordinate",values="rho_s");im=axs[1].imshow(piv.values,vmin=-1,vmax=1,aspect="auto");axs[1].set_yticks(np.arange(len(piv.index)),piv.index);axs[1].set_xticks(np.arange(len(piv.columns)),piv.columns,rotation=35,ha="right");axs[1].set_title("Post-hoc environment associations");plt.colorbar(im,ax=axs[1],fraction=.046,pad=.03,label=r"$\\rho_s$");panel_label(axs[1],"b")
    save_figure(fig,"S2_F05_rank_and_environment_associations");stems.append("S2_F05_rank_and_environment_associations")
    return stems

def make_report(stems,archive,selected,panel,sentinels,law_summary,var_final,rank_summary,registry,postcorr,scaling_cred,scaling_db):
    path=OUTDIR/"CNN_S2_environment_structural_disposition_scientific_report.pdf"
    if not S2_RUN_REPORT:return path
    with PdfPages(path) as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.92,"CNN-S2 — Environment-Wide Structural Disposition",fontsize=18,fontweight="bold");fig.text(.08,.875,"RF-SMAC environment discovery, frozen S1 architecture panel, and balanced environment-universality inference",fontsize=10.5);txt=(f"Structural object: O_CNN^S=(G_exec, Gamma_exec, p^S; rho). The environment affects the learned parameter state but never enters the observable as an input condition.\n\n"
            f"Architecture handoff: {len(panel)} frozen S1 architectures; discovery sentinels: {len(sentinels)}. Environment discovery: {len(archive)} explored; inference panel: {len(selected)} selected. Full intended matrix: {len(selected)} x {len(panel)} x {len(FULL_SEEDS)} = {len(selected)*len(panel)*len(FULL_SEEDS)} trajectories.\n\n"
            "Planned inference freezes S1 sign laws before S2, decomposes balanced variance into architecture, environment, AxE and residual components, measures cross-environment architecture-rank preservation, and keeps scaling credibility separate from d_B. Post-hoc environment-coordinate and teacher-family analyses are explicitly descriptive because the selected panel is adaptively designed.")
        fig.text(.08,.80,txt,fontsize=10,va="top",wrap=True);pdf.savefig(fig);plt.close(fig)
        for stem in stems:
            p=OUTDIR/"figures"/f"{stem}.png"
            if p.exists():
                im=plt.imread(p);fig=plt.figure(figsize=(8.27,11.69));ax=fig.add_axes([.06,.08,.88,.84]);ax.imshow(im);ax.axis("off");fig.text(.06,.95,stem.replace("_"," "),fontsize=12,fontweight="bold");pdf.savefig(fig);plt.close(fig)
        tables=[("Frozen S1 law registry",registry),("S2 law replication summary",law_summary),("Final variance decomposition",var_final),("Cross-environment architecture rank summary",rank_summary),("Scaling credibility",scaling_cred),("dB conditional on credibility",scaling_db),("Post-hoc environment associations",postcorr)]
        for title,d in tables:
            if d is None or d.empty:continue
            view=d.head(24).copy();fig=plt.figure(figsize=(11.69,8.27));ax=fig.add_axes([.03,.05,.94,.86]);ax.axis("off");ax.set_title(title,loc="left",fontsize=12,fontweight="bold");tbl=ax.table(cellText=np.round(view.select_dtypes(include=[np.number]).to_numpy(),4) if len(view.select_dtypes(include=[np.number]).columns)==len(view.columns) else view.astype(str).to_numpy(),colLabels=view.columns,loc="center",cellLoc="center");tbl.auto_set_font_size(False);tbl.set_fontsize(5.8);tbl.scale(1,1.15);pdf.savefig(fig);plt.close(fig)
    return path

# ==================================================================================================
# 15. MANIFEST AND ONE-CELL ENTRY POINT
# ==================================================================================================

def write_manifest(panel,sentinels,selected,registry):
    manifest={"study":"CNN-S2 Environment-Wide Structural Disposition","profile":S2_PROFILE,"structural_object":"O_CNN^S=(G_exec_tilde,Gamma_exec,p^S;rho)","architecture_panel_n":len(panel),"sentinel_n":len(sentinels),"environment_discovery":{"initial":INITIAL_ENVIRONMENTS,"sobol":SOBOL_ENVIRONMENTS,"canonical_anchors":CANONICAL_ENVIRONMENTS,"rfsmac":RFSMAC_ENV_STEPS,"total":DISCOVERY_ENV_TOTAL,"epochs":DISCOVERY_EPOCHS,"state_grid":DISCOVERY_STATE_GRID},"environment_panel_n":len(selected),"full_matrix":{"environments":len(selected),"architectures":len(panel),"seeds":FULL_SEEDS,"intended_runs":len(selected)*len(panel)*len(FULL_SEEDS),"epochs":FULL_EPOCHS,"state_grid":STATE_GRID,"geometry_grid":GEOMETRY_GRID,"dense_sentinel_geometry_grid":DENSE_SENTINEL_GEOMETRY_GRID},"fixed":{"task":"synthetic spatial scalar regression","activation":ACTIVATION,"optimizer":OPTIMIZER,"learning_rate":LEARNING_RATE,"resolution":RESOLUTION,"batch_size":BATCH_SIZE,"weight_decay":WEIGHT_DECAY},"environment_coordinates":["teacher_family","informative_fraction","spatial_correlation","snr","interaction_strength","localization"],"rfsmac_acquisition":{"uncertainty":.30,"environment_novelty":.25,"structural_novelty":.25,"architecture_environment_disagreement":.20,"random_interleave":RFSMAC_RANDOM_INTERLEAVE,"heldout_performance_used":False},"statistics":{"planned":["frozen S1 sign-law replication","balanced A/E/AxE variance components + separate residual","temporal variance decomposition","cross-environment architecture-rank universality","two-stage finite scaling"],"posthoc":["environment-level Spearman after A/seed aggregation","teacher-family Kruskal-Wallis + epsilon2 and conditional pairwise tests","structural-functional environment bridge"]},"frozen_s1_laws":registry.to_dict("records")}
    atomic_json(manifest,OUTDIR/"experiment_manifest.json")

def run_cnn_s2():
    banner("CNN-S2 | ENVIRONMENT-WIDE STRUCTURAL DISPOSITION")
    print(f"profile={S2_PROFILE} | device={DEVICE} | output={OUTDIR.resolve()}");gpu_snapshot("startup")
    panel=load_s1_architecture_panel();registry=load_s1_law_registry();validate_s1_handoff_contract(panel,registry);sentinels=select_discovery_sentinels(panel);atomic_csv(panel,OUTDIR/"tables"/"frozen_s1_12_architecture_panel.csv")
    print("Frozen architecture panel:",panel.arch_id.astype(str).tolist());print("Discovery sentinels:",sentinels.arch_id.astype(str).tolist());print("Frozen S1 laws:",registry[["metric","s1_direction"]].to_dict("records"))
    preflight=run_preflight_controls() if S2_RUN_EXACT else pd.DataFrame();archive=run_environment_discovery(sentinels);selected=select_environment_panel(archive);print(f"Environment discovery={len(archive)} | selected={len(selected)}")
    dgeom,dgsc=discovery_geometry(archive,sentinels) if S2_RUN_BCANW else (pd.DataFrame(),pd.DataFrame())
    run_full_matrix(selected,panel);tabs=aggregate_full(selected,panel);geometry,gsc=run_full_geometry(selected,panel,sentinels) if S2_RUN_BCANW else (pd.DataFrame(),pd.DataFrame());states,geometry=_add_derived_columns(tabs["states"],geometry);atomic_csv(states,OUTDIR/"tables"/"full_states_derived.csv");
    if not geometry.empty:atomic_csv(geometry,OUTDIR/"tables"/"full_geometry_derived.csv")
    seed_states=seed_aggregate(states,STATE_METRICS);seed_geom=seed_aggregate(geometry,GEOM_METRICS) if not geometry.empty else pd.DataFrame();cell_state=delta_cell_frame(seed_states,STATE_METRICS,FULL_EPOCHS);cell_geom=delta_cell_frame(seed_geom,GEOM_METRICS,FULL_EPOCHS) if not seed_geom.empty else pd.DataFrame();cell=cell_state.merge(cell_geom,on=[c for c in ["env_id","arch_id","teacher_family","informative_fraction","spatial_correlation","snr","interaction_strength","localization"] if c in cell_state.columns and c in cell_geom.columns],how="left") if not cell_geom.empty else cell_state;atomic_csv(cell,OUTDIR/"tables"/"architecture_environment_cell_changes.csv")
    law_env,law_summary=s1_sign_law_replication(cell,registry);var_final,var_time=variance_decomposition(states,geometry);rank_pairs,rank_summary=architecture_rank_universality(seed_states,STATE_METRICS,FULL_EPOCHS)
    if not seed_geom.empty:
        rp2,rs2=architecture_rank_universality(seed_geom,GEOM_METRICS,FULL_EPOCHS);rank_pairs=pd.concat([rank_pairs,rp2],ignore_index=True);rank_summary=pd.concat([rank_summary,rs2],ignore_index=True);atomic_csv(rank_pairs,OUTDIR/"tables"/"planned_cross_environment_architecture_rank_pairs.csv");atomic_csv(rank_summary,OUTDIR/"tables"/"planned_cross_environment_architecture_rank_summary.csv")
    scaling_cred,scaling_db=scaling_two_stage(geometry);postcorr,postom,postpairs,postbridge=posthoc_environment_associations(cell,tabs["summary"]);stems=make_figures(archive,selected,states,geometry,var_final,var_time,law_env,rank_summary,postcorr,tabs["summary"]);report=make_report(stems,archive,selected,panel,sentinels,law_summary,var_final,rank_summary,registry,postcorr,scaling_cred,scaling_db);write_manifest(panel,sentinels,selected,registry)
    banner("CNN-S2 COMPLETE");print(f"Report: {report}");print(f"Tables: {OUTDIR/'tables'}");print(f"Figures: {OUTDIR/'figures'}")
    return {"panel12":panel,"sentinels4":sentinels,"s1_registry":registry,"environment_archive":archive,"selected_environments":selected,"discovery_geometry":dgeom,"states":states,"history":tabs["history"],"run_summary":tabs["summary"],"geometry":geometry,"geometry_scales":gsc,"law_by_environment":law_env,"law_summary":law_summary,"variance_final":var_final,"variance_temporal":var_time,"rank_pairs":rank_pairs,"rank_summary":rank_summary,"scaling_credibility":scaling_cred,"scaling_dB":scaling_db,"posthoc_environment_spearman":postcorr,"posthoc_teacher_omnibus":postom,"posthoc_teacher_pairs":postpairs,"posthoc_structural_functional":postbridge,"report":report}

CNN_S2_RESULTS = run_cnn_s2()
