# WORK II — MLP CONDITIONAL REALIZATION PROGRAMME — PHASE C1
# ARCHITECTURE-WIDE CONDITIONAL REALIZATION IN MULTILAYER PERCEPTRONS
# ==================================================================================================
# Single-file / single-Jupyter-cell research experiment.
#
# Paste this entire file into ONE Jupyter notebook cell and run, or execute it as a normal Python file.
#
# Scientific position
# -------------------
# The MLP programme now has two parallel empirical sequences:
#
#     Structural disposition:       S1 architecture | S2 environment | S3 optimizer
#     Conditional realization:      C1 architecture | C2 environment | C3 optimizer
#
# C3 is already completed.  C1 is the missing architecture-wide study of the SAME finite conditional
# MLP observable established in Phase II-B:
#
#     p_Delta(pi | x,x0) = Q_pi^Delta / sum_pi Q_pi^Delta,
#
#     Q_pi^Delta = | Delta x_i0 * prod_l w_l * prod_hidden_l s_l(x,x0) |^2.
#
# For every trained architecture the script retains the weight-only structural path disposition
# p_str(pi) and the conditionally realized path measure p_Delta(pi|x,x0).  The scientific question is
# therefore not merely whether architectures differ, but which S1 architecture-wide laws survive the
# transition
#
#     structural disposition  -->  conditional realization.
#
# Primary architecture experiment
# -------------------------------
# Stage 0: RF-SMAC architecture exploration
#   80 candidate architectures = 32 Sobol cold-start + 48 RF/EI proposals.
#   Discovery uses one central synthetic regression environment, ReLU, Adam, one discovery seed,
#   120 epochs and a validation-MSE trajectory objective.  Path-flow quantities never enter the
#   acquisition objective.  Search observations are discovery data only, not inferential replicates.
#
# Stage 1: Frozen publication panel
#   20 viable architectures are selected by morphology-aware maximin coverage in a declared descriptor
#   space (depth, parameter count, log path count, width CV, bottleneck ratio, expansion index and
#   morphology).  Selection is for design-space coverage, not best validation loss.
#
# Stage 2: Primary C1 ReLU map
#   20 architectures x 8 matched seeds x 1 environment x 1 optimizer x ReLU
#   = 160 full 280-epoch trajectories.
#
# Stage 3: Smooth-activation robustness sentinel
#   6 morphology-spanning architectures x the same 8 seeds x tanh
#   = 48 additional full trajectories.
#
# Total publication trajectories = 208, plus 80 short RF-SMAC discovery evaluations.
#
# What is fixed
# -------------
#   task            : synthetic regression
#   environment     : one canonical central environment (mixed teacher, moderate correlation/SNR/
#                     interaction)
#   optimizer       : Adam, one frozen global operating point (lr=3e-3), no scheduler, no weight decay
#   primary act.    : ReLU
#   train horizon   : 280 epochs, no early stopping
#   condition       : c=(x,x0), with x0=0 after standardization, unchanged from II-B/C3
#   model bias      : disabled in C1 so the structural complete-path disposition is the exact Work-I
#                     bias-free MLP object rather than a descriptive weight-only baseline.
#
# Exact matched trajectory grid
# -----------------------------
#     t = {0,5,10,20,40,80,120,160,200,240,280}
#
# Population conditional state uses a fixed 128-condition test bank.  A 64-condition subset is used
# at every trajectory checkpoint.  BCANw geometry is evaluated longitudinally at
#
#     t_G = {0,40,80,160,240,280}
#
# plus the best-validation and final endpoint states.  Eight fixed test conditions per trained model
# are retained for input-resolved geometry so population geometry and typical realized geometry are
# never conflated.
#
# Observable hierarchy
# --------------------
#   p_str, p_Delta
#     -> S_deficit, eta_1, eta_2
#     -> exact conditional edge/node flow and CV
#     -> KL[p_Delta || p_str], H^2[p_Delta,p_str], structural-realized flow-rank agreement
#     -> input specialization I(X;I_l)
#     -> BCANw: xi_50, d*, lacunarity, local Delta_i^F, quotient-flow KS, retained flow,
#               coarse Renyi occupation, and strict finite-window d_B^F when a credible window exists.
#
# Primary C1 inference
# --------------------
#   * Architecture omnibus: Friedman repeated-block test over the 20 architectures with the 8 matched
#     seeds as blocks; Kendall W_A reports architecture concordance.
#   * Architecture-rank persistence through training on the common epoch grid.
#   * Structural-to-realized architecture-rank survival at matched states.
#   * Architecture-descriptor Spearman associations on seed-averaged architecture responses.
#   * Training-direction survival table: which S1 signs are preserved, attenuated, reversed or absent.
#   * Equal-functional-progress architecture analysis at tau_L={.25,.50,.75,.90,.99}.
#   * Tanh sentinel: paired ReLU/tanh effects and architecture-rank preservation.
#   * Strict d_B^F: scaling-window credibility is reported before conditional exponent comparisons.
#
# Important boundaries
# --------------------
# 1. Squared path amplitudes define a positive route-prominence measure, not additive output energy.
# 2. Inputs are repeated conditions nested within a trained network, never independent replicates.
# 3. RF-SMAC observations are discovery data; only the frozen fresh panel enters architecture inference.
# 4. BCANw is downstream of the conditional measure and flow.  d_B^F is a finite-window descriptor.
# 5. C1 is an architecture phase.  Environment, task and optimizer remain fixed by construction.
#
# Principal outputs
# -----------------
# pathflow_work2_C1_architecture_conditional_realization_results/
#   experiment_manifest.json
#   tables/
#     rf_smac_archive.csv
#     c1_architecture_panel.csv
#     c1_tanh_sentinels.csv
#     c1_run_manifest.csv
#     run_summary.csv
#     endpoint_input_summary.csv
#     training_history.csv
#     trajectory_summary.csv
#     functional_progress_summary.csv
#     late_training_summary.csv
#     trajectory_geometry.csv
#     trajectory_geometry_scales.csv
#     input_resolved_geometry.csv
#     exact_validation.csv
#     c1_architecture_friedman.csv
#     c1_architecture_rank_persistence.csv
#     c1_structural_realized_rank_bridge.csv
#     c1_architecture_descriptor_associations.csv
#     c1_direction_survival.csv
#     c1_functional_progress_friedman.csv
#     c1_tanh_paired_effects.csv
#     c1_tanh_architecture_rank_preservation.csv
#     c1_input_population_geometry_comparison.csv
#   figures/
#     publication-oriented diagnostic atlas for search coverage, endpoint states, trajectories,
#     functional progress, structural-realized bridge, BCANw and tanh sentinel.
#   checkpoints/ and run_parts/
#
from __future__ import annotations

import os
import math
import json
import time
import random
import hashlib
import itertools
import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from scipy.stats import norm, qmc, ks_2samp, spearmanr, friedmanchisquare, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================================================================================
# 0. C1 CONFIGURATION
# ==================================================================================================

C1_PROFILE = os.environ.get("PATHFLOW_W2_C1_PROFILE", "publication_colab").strip().lower()
C1_RESUME = os.environ.get("PATHFLOW_W2_C1_RESUME", "1") != "0"
C1_START_FRESH = os.environ.get("PATHFLOW_W2_C1_START_FRESH", "0") == "1"
C1_RUN_INPUT_GEOM = os.environ.get("PATHFLOW_W2_C1_RUN_INPUT_GEOM", "1") != "0"
C1_RUN_TRAJECTORY_GEOM = os.environ.get("PATHFLOW_W2_C1_RUN_TRAJECTORY_GEOM", "1") != "0"
C1_RUN_EXACT_PREFLIGHT = os.environ.get("PATHFLOW_W2_C1_RUN_EXACT_PREFLIGHT", "1") != "0"
C1_REQUIRE_CUDA = os.environ.get("PATHFLOW_W2_C1_REQUIRE_CUDA", "1") != "0"

# Compatibility aliases used by inherited exact conditional-path / BCANw helpers.
P2B2_PROFILE = C1_PROFILE
P2B2_RESUME = C1_RESUME
P2B2_START_FRESH = C1_START_FRESH
P2B2_RUN_CONTROLLED_ENV = False
P2B2_RUN_INPUT_GEOM = C1_RUN_INPUT_GEOM
P2B2_RUN_TRAJECTORY_GEOM = C1_RUN_TRAJECTORY_GEOM
P2B2_RUN_EXACT_PREFLIGHT = C1_RUN_EXACT_PREFLIGHT
P2B2_ALLOW_ARCH_DISCOVERY = True
P2B_PROFILE = C1_PROFILE
P2B_RESUME = C1_RESUME
P2B_START_FRESH = C1_START_FRESH

P2B_SEED = 72031
MODEL_INIT_BASE_SEED = 31415
DATA_BASE_SEED = 880301
DATA_ORDER_BASE_SEED = 161803
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTDIR = Path(os.environ.get(
    "PATHFLOW_W2_C1_OUTDIR",
    "pathflow_work2_C1_architecture_conditional_realization_results"
))
ARCH_SOURCE_DIR = Path(os.environ.get("PATHFLOW_W2_P2B_EXPANDED_DIR", "pathflow_work2_phase2b_expanded_results"))
for sub in ["tables", "figures", "checkpoints", "run_parts", "architecture_discovery", "calibration"]:
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

# Exact primary C1 training design.
INPUT_DIM = 32
N_TRAIN = 10_000
N_VAL = 2_000
N_TEST = 2_000
BATCH_SIZE = 512
MAX_EPOCH = 280
WEIGHT_DECAY = 0.0
MODEL_BIAS = False
MAIN_SEEDS = list(range(8))
PRIMARY_ACTIVATION = "relu"
SENTINEL_ACTIVATION = "tanh"
ACTIVATIONS = [PRIMARY_ACTIVATION, SENTINEL_ACTIVATION]
ANCHOR_ACTIVATIONS = [PRIMARY_ACTIVATION]  # architecture discovery must not optimize activation.
OPTIMIZERS = ["adam"]
FIXED_OPTIMIZER = "adam"
FIXED_ADAM_LR = float(os.environ.get("PATHFLOW_W2_C1_ADAM_LR", "0.003"))
LEARNING_RATE = FIXED_ADAM_LR

# RF-SMAC architecture exploration.  Search objective is functional competence only.
RFSMAC_TOTAL_CANDIDATES = 80
RFSMAC_SOBOL_COLD = 32
RFSMAC_BO_STEPS = RFSMAC_TOTAL_CANDIDATES - RFSMAC_SOBOL_COLD
RFSMAC_POOL = 1536
RFSMAC_RANDOM_INTERLEAVE = 0.25
RFSMAC_RF_TREES = 320
RFSMAC_EI_XI = 0.01
RFSMAC_DISCOVERY_EPOCHS = 120
RFSMAC_DISCOVERY_N_TRAIN = 4_000
RFSMAC_DISCOVERY_N_VAL = 1_000
RFSMAC_SELECT_N = 20
RFSMAC_VIABLE_FRACTION = 0.75
MIN_HIDDEN_DEPTH = 3
MAX_HIDDEN_DEPTH = 20
MIN_PARAMETER_BUDGET = 15_000
MAX_PARAMETER_BUDGET = 1_200_000
MIN_WIDTH = 16
MAX_WIDTH = 512
MORPHOLOGIES = ["constant", "contracting", "expanding", "bottleneck", "expand_contract", "irregular"]
C1_ARCHITECTURES = 20
C1_TANH_SENTINELS = 6

# Compatibility variables for inherited but unused II-B.2 helpers.
B2_ARCHITECTURES = C1_ARCHITECTURES
CONTROLLED_ENV_ARCHITECTURES = 0
INPUT_GEOM_ARCHITECTURES = C1_ARCHITECTURES
CALIBRATION_ARCHITECTURES = 0
CALIBRATION_ACTIVATIONS = [PRIMARY_ACTIVATION]
CALIBRATION_SEEDS = [0]
CALIBRATION_EPOCHS = 1
CALIBRATION_N_TRAIN = 128
CALIBRATION_N_VAL = 64
OPTIMIZER_LR_GRIDS = {"adam": [FIXED_ADAM_LR]}

# Longitudinal conditional evaluation.
TRAJECTORY_CHECKPOINTS = [0, 5, 10, 20, 40, 80, 120, 160, 200, 240, 280]
TRAJECTORY_GEOMETRY_CHECKPOINTS = [0, 40, 80, 160, 240, 280]
TRAJECTORY_INPUTS = 64
ENDPOINT_INPUTS = 128
CONDITIONAL_EVAL_INPUTS = ENDPOINT_INPUTS
CONDITIONAL_BATCH = 64
FUNCTIONAL_PROGRESS_LEVELS = [0.25, 0.50, 0.75, 0.90, 0.99]
RESPONSE_SUPPORT_TOL = 1e-12
LEAKY_REGIME_MIDPOINT = 0.55

# Input-resolved BCANw: eight fixed test conditions for every completed publication run.
INPUT_GEOM_INPUTS = 8
INPUT_BCANW_SCALES = 56

# Exact finite-response preflight.
EXACT_RANDOM_STATES = 50
EXACT_CONTRASTS_PER_STATE = 20
EXACT_HIDDEN = [12, 12]
REFERENCE_SENSITIVITY_INPUTS = 64
EXACT_PREFLIGHT_TOL = 1e-10

# BCANw constants inherited unchanged from the corrected Work-I/III and Work-II MLP pipeline.
EPS = 1e-14
PATH_LOG_FLOOR = 1e-300
METRIC_POWER = 1.0
LOCAL_SLOPE_WINDOW = 7
LOCAL_SLOPE_R2_MIN = 0.90
SCALING_R2_MIN = 0.97
SCALING_MIN_DIM = 0.25
SCALING_MIN_POINTS = 6
SCALING_MIN_SPAN_DECADES = 0.50
SCALING_MIN_NB_RATIO = 5.0
SCALING_MAX_SLOPE_CV = 0.25
SCALING_MAX_DELTA_BIC = 2.0
RENORM_TARGET_BOX_FRACTIONS = np.array([0.50, 0.25, 0.125, 0.0625])
COARSE_RENYI_Q = [0.5, 1.0, 2.0, 3.0]
LOCAL_COMMON_MIN_POINTS = 7
LOCAL_COMMON_MIN_SPAN_DECADES = 0.50
LOCAL_COMMON_SUPPORT_GROWTH = 4.0
LOCAL_COMMON_MASS_GROWTH = 3.0
LOCAL_COMMON_R2_MIN = 0.95
LOCAL_COMMON_SATURATION = 0.95
BCANW_SCALES = 72

# Compatibility-only controlled-environment values; C1 never executes this arm.
CONTROLLED_ENVIRONMENTS = []
CONTROLLED_FEATURE_CORRELATION = 0.35
CONTROLLED_INTERACTION = 0.50
CONTROLLED_SNR_HIGH = 20.0
CONTROLLED_SNR_LOW = 8.0
MAIN_ENVIRONMENTS = ["C1_CANON"]

# Pilot is an orchestration/smoke test only.  It may run on CPU when REQUIRE_CUDA=0.
if C1_PROFILE == "pilot":
    N_TRAIN, N_VAL, N_TEST = 600, 160, 160
    BATCH_SIZE = 128
    MAX_EPOCH = 4
    MAIN_SEEDS = [0, 1]
    RFSMAC_TOTAL_CANDIDATES = 6
    RFSMAC_SOBOL_COLD = 4
    RFSMAC_BO_STEPS = 2
    RFSMAC_POOL = 96
    RFSMAC_RF_TREES = 48
    RFSMAC_DISCOVERY_EPOCHS = 2
    RFSMAC_DISCOVERY_N_TRAIN = 400
    RFSMAC_DISCOVERY_N_VAL = 120
    RFSMAC_SELECT_N = 3
    C1_ARCHITECTURES = 3
    C1_TANH_SENTINELS = 2
    B2_ARCHITECTURES = C1_ARCHITECTURES
    INPUT_GEOM_ARCHITECTURES = C1_ARCHITECTURES
    MIN_HIDDEN_DEPTH, MAX_HIDDEN_DEPTH = 2, 4
    MIN_PARAMETER_BUDGET, MAX_PARAMETER_BUDGET = 500, 3_000
    MIN_WIDTH, MAX_WIDTH = 4, 24
    MORPHOLOGIES = ["constant", "contracting", "bottleneck"]
    TRAJECTORY_CHECKPOINTS = [0, 1, 2, 4]
    TRAJECTORY_GEOMETRY_CHECKPOINTS = [0, 4]
    TRAJECTORY_INPUTS = 8
    ENDPOINT_INPUTS = 12
    CONDITIONAL_EVAL_INPUTS = ENDPOINT_INPUTS
    CONDITIONAL_BATCH = 6
    FUNCTIONAL_PROGRESS_LEVELS = [0.5, 0.9]
    INPUT_GEOM_INPUTS = 1
    INPUT_BCANW_SCALES = 10
    EXACT_RANDOM_STATES = 2
    EXACT_CONTRASTS_PER_STATE = 2
    EXACT_HIDDEN = [4, 3]
    BCANW_SCALES = 12
    if "PATHFLOW_W2_C1_RUN_INPUT_GEOM" not in os.environ:
        C1_RUN_INPUT_GEOM = False
        P2B2_RUN_INPUT_GEOM = False

random.seed(P2B_SEED)
np.random.seed(P2B_SEED)
torch.manual_seed(P2B_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(P2B_SEED)
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if C1_REQUIRE_CUDA and C1_PROFILE != "pilot" and not torch.cuda.is_available():
    raise RuntimeError(
        "C1 publication profile requires CUDA. In Colab choose Runtime > Change runtime type > GPU. "
        "For a CPU smoke test set PATHFLOW_W2_C1_PROFILE=pilot and PATHFLOW_W2_C1_REQUIRE_CUDA=0."
    )

if C1_START_FRESH:
    for sub in ["tables", "figures", "checkpoints", "run_parts", "architecture_discovery", "calibration"]:
        for p in (OUTDIR / sub).glob("*"):
            if p.is_file():
                try: p.unlink()
                except Exception: pass

# ==================================================================================================
# 1. GENERIC HELPERS
# ==================================================================================================

def banner(title: str):
    print("\n" + "="*104)
    print(title)
    print("="*104)


def human_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    s = int(round(seconds)); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


class ETAReporter:
    def __init__(self, stage: str, total: int, window: int = 12):
        self.stage = stage; self.total = max(int(total), 1); self.window = max(int(window), 3)
        self.start = time.perf_counter(); self.durations: List[float] = []
    def update(self, duration: float, completed: int):
        self.durations.append(float(duration))
        recent = np.asarray(self.durations[-self.window:], float)
        typical = float(np.median(recent)); remain = max(self.total-int(completed), 0)
        tqdm.write(
            f"[ETA:{self.stage}] {completed}/{self.total} | current {human_time(duration)} | "
            f"median {human_time(typical)}/item | elapsed {human_time(time.perf_counter()-self.start)} | "
            f"remaining ~{human_time(typical*remain)}"
        )
    def finish(self):
        tqdm.write(f"[ETA:{self.stage}] finished in {human_time(time.perf_counter()-self.start)}")


def stable_hash_dict(d: Dict[str, Any], n: int = 12) -> str:
    def clean(v):
        if isinstance(v, (np.integer, np.floating)): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        return v
    return hashlib.sha1(json.dumps({str(k):clean(v) for k,v in d.items()}, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()[:n]


def atomic_csv(df: pd.DataFrame, path: Path):
    path = Path(path); tmp = path.with_suffix(path.suffix+".tmp")
    df.to_csv(tmp, index=False); tmp.replace(path)


def atomic_json(obj: Any, path: Path):
    path = Path(path); tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8"); tmp.replace(path)


def r2_score_np(y, pred):
    y = np.asarray(y,float); pred = np.asarray(pred,float)
    den = float(np.sum((y-y.mean())**2))
    return float(1.0-np.sum((y-pred)**2)/den) if den > EPS else np.nan


def safe_log(x):
    try:
        x = float(x)
        return float(math.log(max(x, EPS))) if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def model_seed(seed_index: int) -> int:
    return int(MODEL_INIT_BASE_SEED + 1_000_003*int(seed_index))


def order_seed(seed_index: int) -> int:
    return int(DATA_ORDER_BASE_SEED + 104_729*int(seed_index))


def deterministic_batches(n: int, epoch: int, seed_index: int, batch_size: int):
    rng = np.random.default_rng(order_seed(seed_index)+int(epoch))
    ids = rng.permutation(int(n))
    for lo in range(0, len(ids), int(batch_size)):
        yield ids[lo:min(lo+int(batch_size), len(ids))]


def bhattacharyya_from_logs(logp: np.ndarray, logq: np.ndarray) -> float:
    a = np.asarray(logp,float); b = np.asarray(logq,float)
    m = np.maximum(0.5*(a+b), -745)
    return float(np.sum(np.exp(m)))


# ==================================================================================================
# 2. ACTIVATIONS + MLP
# ==================================================================================================

class ActivationModule(nn.Module):
    def __init__(self, name: str):
        super().__init__(); self.name = str(name).lower()
    def forward(self, x):
        if self.name == "relu": return F.relu(x)
        if self.name == "leaky_relu": return F.leaky_relu(x, negative_slope=0.1)
        if self.name == "tanh": return torch.tanh(x)
        if self.name == "sigmoid": return torch.sigmoid(x)
        if self.name == "silu": return F.silu(x)
        if self.name == "identity": return x
        raise ValueError(f"Unknown activation {self.name}")


def activation_numpy(name: str, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float); name = str(name).lower()
    if name == "relu": return np.maximum(z, 0.0)
    if name == "leaky_relu": return np.where(z >= 0, z, 0.1*z)
    if name == "tanh": return np.tanh(z)
    if name == "sigmoid":
        zz = np.clip(z, -60, 60); return 1.0/(1.0+np.exp(-zz))
    if name == "silu":
        zz = np.clip(z, -60, 60); sig = 1.0/(1.0+np.exp(-zz)); return z*sig
    if name == "identity": return z
    raise ValueError(name)


def activation_derivative_numpy(name: str, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float); name = str(name).lower()
    if name == "relu": return (z > 0).astype(float)
    if name == "leaky_relu": return np.where(z >= 0, 1.0, 0.1)
    if name == "tanh":
        t = np.tanh(z); return 1.0-t*t
    if name == "sigmoid":
        zz = np.clip(z, -60, 60); s = 1.0/(1.0+np.exp(-zz)); return s*(1.0-s)
    if name == "silu":
        zz = np.clip(z, -60, 60); s = 1.0/(1.0+np.exp(-zz)); return s + z*s*(1.0-s)
    if name == "identity": return np.ones_like(z)
    raise ValueError(name)


def finite_secant_numpy(name: str, z: np.ndarray, z0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    z = np.asarray(z,float); z0 = np.asarray(z0,float); dz = z-z0
    out = np.empty_like(dz, dtype=float)
    mask = np.abs(dz) > tol
    if np.any(mask):
        out[mask] = (activation_numpy(name,z[mask])-activation_numpy(name,z0[mask]))/dz[mask]
    if np.any(~mask):
        out[~mask] = activation_derivative_numpy(name, 0.5*(z[~mask]+z0[~mask]))
    return out


class GeneralMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], activation: str,
                 out_dim: int = 1, bias: bool = MODEL_BIAS, seed: int = MODEL_INIT_BASE_SEED):
        super().__init__()
        self.in_dim = int(in_dim); self.hidden = [int(x) for x in hidden]; self.out_dim = int(out_dim)
        self.activation_name = str(activation).lower(); self.widths = [self.in_dim]+self.hidden+[self.out_dim]
        self.activation = ActivationModule(self.activation_name)
        torch.manual_seed(int(seed))
        self.linears = nn.ModuleList([
            nn.Linear(self.widths[i], self.widths[i+1], bias=bool(bias))
            for i in range(len(self.widths)-1)
        ])
        self.reset_common_directions(seed)

    def reset_common_directions(self, seed: int):
        # Same random directions for every activation in a matched block.
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for layer in self.linears:
                W = torch.randn(layer.weight.shape, generator=g, dtype=layer.weight.dtype)
                W /= math.sqrt(max(layer.in_features,1))
                layer.weight.copy_(W)
                if layer.bias is not None:
                    layer.bias.zero_()

    def forward(self, x):
        for k, layer in enumerate(self.linears):
            x = layer(x)
            if k < len(self.linears)-1:
                x = self.activation(x)
        return x

    def forward_with_states(self, x):
        """Return output, hidden preactivations, hidden activations."""
        zlist, hlist = [], [x]
        h = x
        for k, layer in enumerate(self.linears):
            z = layer(h)
            if k < len(self.linears)-1:
                h = self.activation(z); zlist.append(z); hlist.append(h)
            else:
                h = z
        return h, zlist, hlist


@torch.no_grad()
def calibrate_initial_signal(model: GeneralMLP, Xcal: torch.Tensor, target_var: float = 1.0):
    """Activation-neutral layerwise scalar calibration of preactivation variance.

    Weight DIRECTIONS are unchanged. Each hidden affine layer is multiplied by one positive scalar so
    that the empirical preactivation variance on a common calibration batch is approximately target_var.
    The final readout is left at the common random-direction scale because output variance is task-scale
    dependent and targets are standardized.
    """
    model.eval(); h = Xcal.to(DEVICE)
    scales = []
    for k, layer in enumerate(model.linears):
        z = layer(h)
        if k < len(model.linears)-1:
            sd = float(torch.std(z, unbiased=False).item())
            scale = math.sqrt(float(target_var))/max(sd, 1e-6)
            scale = float(np.clip(scale, 0.15, 6.0))
            layer.weight.mul_(scale)
            if layer.bias is not None: layer.bias.mul_(scale)
            z = layer(h); h = model.activation(z); scales.append(scale)
        else:
            scales.append(1.0)
    return scales


def extract_weights(model: GeneralMLP) -> List[np.ndarray]:
    return [layer.weight.detach().cpu().numpy().astype(np.float64).copy() for layer in model.linears]


def extract_biases(model: GeneralMLP) -> List[Optional[np.ndarray]]:
    return [None if layer.bias is None else layer.bias.detach().cpu().numpy().astype(np.float64).copy()
            for layer in model.linears]


# ==================================================================================================
# 3. SYNTHETIC REGRESSION ENVIRONMENTS
# ==================================================================================================

ENVIRONMENTS = {
    "DISC":     {"family":"mixed", "feature_correlation":0.35, "snr":15.0, "interaction":0.50, "informative_dim":16},
    "C1_CANON": {"family":"mixed", "feature_correlation":0.35, "snr":15.0, "interaction":0.50, "informative_dim":16},
}
MAIN_ENVIRONMENTS = ["C1_CANON"]


def correlated_gaussian(rng, n: int, d: int, rho: float):
    eps = rng.standard_normal((n,d), dtype=np.float32)
    if rho <= 0: return eps
    g = rng.standard_normal((n,1), dtype=np.float32)
    return (math.sqrt(1-rho)*eps + math.sqrt(rho)*g).astype(np.float32)


def deterministic_target(X: np.ndarray, cfg: Dict[str,Any]) -> np.ndarray:
    """Activation-neutral diagnostic target families. No student activation is used as a teacher."""
    x = np.asarray(X,float); fam = cfg["family"]; k = float(cfg["interaction"])
    if fam == "smooth":
        y = (0.75*np.sin(1.1*x[:,0]) + 0.55*np.tanh(x[:,1]+0.7*x[:,2]) +
             0.40*x[:,3]*x[:,4] + 0.25*(x[:,5]**2-1.0) +
             0.20*np.sin(x[:,6]*x[:,7]))
    elif fam == "piecewise":
        y = (0.70*np.maximum(x[:,0]+x[:,1],0.0) + 0.55*np.abs(x[:,2]-x[:,3]) +
             0.40*np.maximum(x[:,4]*x[:,5],0.0) - 0.25*x[:,6] +
             0.20*np.sign(x[:,7])*np.sqrt(np.abs(x[:,7])+1e-8))
    elif fam == "mixed":
        y = (0.55*np.sin(x[:,0]*x[:,1]) + 0.45*np.tanh(x[:,2]+np.sin(x[:,3])) +
             0.35*np.maximum(x[:,4]-0.5*x[:,5],0.0) + 0.25*x[:,6]*x[:,7])
    else:
        raise ValueError(fam)
    # Shared interaction term makes the tasks non-additive without privileging one student activation.
    inter = np.mean(x[:,8:16:2]*x[:,9:16:2], axis=1)
    y = (1.0-k)*y + k*inter
    return y.reshape(-1,1).astype(np.float32)


def generate_environment(env_id: str, discovery: bool = False):
    cfg = dict(ENVIRONMENTS[str(env_id)])
    ntr = RFSMAC_DISCOVERY_N_TRAIN if discovery else N_TRAIN
    nva = RFSMAC_DISCOVERY_N_VAL if discovery else N_VAL
    nte = 0 if discovery else N_TEST
    seed = int(DATA_BASE_SEED + int(stable_hash_dict({"env":env_id},8),16)%1_000_000_007)
    rng = np.random.default_rng(seed); n = ntr+nva+nte
    X = correlated_gaussian(rng,n,INPUT_DIM,float(cfg["feature_correlation"]))
    xm = X[:ntr].mean(0,keepdims=True); xs = X[:ntr].std(0,keepdims=True)+1e-8
    X = ((X-xm)/xs).astype(np.float32)
    clean = deterministic_target(X,cfg)
    clean = ((clean-clean[:ntr].mean())/(clean[:ntr].std()+1e-8)).astype(np.float32)
    noise_sd = math.sqrt(1/max(float(cfg["snr"]),EPS))
    y = clean + rng.normal(0,noise_sd,size=clean.shape).astype(np.float32)
    y = ((y-y[:ntr].mean())/(y[:ntr].std()+1e-8)).astype(np.float32)
    out = {
        "train": (torch.from_numpy(X[:ntr]), torch.from_numpy(y[:ntr])),
        "val": (torch.from_numpy(X[ntr:ntr+nva]), torch.from_numpy(y[ntr:ntr+nva])),
    }
    if nte:
        out["test"] = (torch.from_numpy(X[ntr+nva:]), torch.from_numpy(y[ntr+nva:]))
    meta = {"environment_id":env_id,"environment_seed":seed,"noise_sd":noise_sd,**cfg}
    return out, meta


@torch.no_grad()
def eval_mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor, chunk: int = 4096) -> float:
    model.eval(); total=0.0; count=0
    for lo in range(0,len(X),chunk):
        hi=min(lo+chunk,len(X)); xb=X[lo:hi].to(DEVICE); yb=y[lo:hi].to(DEVICE)
        pred=model(xb); total += float(torch.sum((pred-yb)**2).item()); count += int(yb.numel())
    return total/max(count,1)


# ==================================================================================================
# 4. CONFIGURATION SPACE + RF-SMAC EXPLORATION
# ==================================================================================================

def parameter_count(widths: Sequence[int], bias: bool = MODEL_BIAS) -> int:
    total=0
    for a,b in zip(widths[:-1],widths[1:]):
        total += int(a)*int(b)
        if bias: total += int(b)
    return int(total)


def morphology_profile(depth: int, morphology: str, strength: float, curvature: float) -> np.ndarray:
    """Low-dimensional deterministic morphology generator used by C1 RF-SMAC.

    `strength` controls amplitude; `curvature` controls taper/bottleneck sharpness or the deterministic
    irregular phase.  Width vectors are generated from these normalized profiles and a target budget.
    """
    t = np.linspace(0.0, 1.0, int(depth))
    s = float(np.clip(strength, 0.0, 1.6)); c = float(np.clip(curvature, 0.0, 1.0))
    if morphology == "constant":
        p = np.ones_like(t)
    elif morphology == "contracting":
        expo = 0.35 + 1.65*c
        p = 1.45 - (0.35 + 0.85*s) * np.power(t, expo)
    elif morphology == "expanding":
        expo = 0.35 + 1.65*c
        p = 0.55 + (0.35 + 0.85*s) * np.power(t, expo)
    elif morphology == "expand_contract":
        amp = 0.25 + 0.75*min(s, 1.4)
        p = 1.0 + amp*np.sin(np.pi*t)
    elif morphology == "bottleneck":
        sigma = max(0.08, 0.28 - 0.16*c)
        center = 0.35 + 0.30*c
        dip = np.exp(-0.5*((t-center)/sigma)**2)
        p = 1.0 - min(0.82, 0.35 + 0.40*s)*dip
    elif morphology == "irregular":
        # Deterministic non-monotone profile; no hidden random variable is introduced into the BO space.
        phase = 2*np.pi*c
        p = (1.0 + (0.12 + 0.22*s)*np.sin(2*np.pi*t + phase)
               + (0.06 + 0.12*s)*np.sin(5*np.pi*t + 0.5*phase))
    else:
        raise ValueError(morphology)
    p = np.clip(p, 0.12, None)
    return p / np.median(p)


def realize_widths(depth: int, target_budget: int, morphology: str,
                   shape_strength: float, shape_curvature: float) -> Optional[List[int]]:
    prof = morphology_profile(depth,morphology,shape_strength,shape_curvature)
    lo,hi=4.0,512.0; best=None
    for _ in range(40):
        mid=0.5*(lo+hi)
        hidden=np.clip(np.rint(mid*prof),MIN_WIDTH,MAX_WIDTH).astype(int).tolist()
        P=parameter_count([INPUT_DIM]+hidden+[1],bias=MODEL_BIAS)
        best=(hidden,P)
        if P < target_budget: lo=mid
        else: hi=mid
    hidden,P=best
    ratio=P/max(int(target_budget),1)
    if not (0.75 <= ratio <= 1.30): return None
    if np.median(hidden)<MIN_WIDTH or np.max(hidden)>MAX_WIDTH: return None
    return [int(x) for x in hidden]


def unit_to_candidate(u: Sequence[float]) -> Optional[Dict[str,Any]]:
    u=np.asarray(u,float)
    depth=int(np.clip(round(MIN_HIDDEN_DEPTH+u[0]*(MAX_HIDDEN_DEPTH-MIN_HIDDEN_DEPTH)),
                      MIN_HIDDEN_DEPTH,MAX_HIDDEN_DEPTH))
    logb=math.log(MIN_PARAMETER_BUDGET)+u[1]*(math.log(MAX_PARAMETER_BUDGET)-math.log(MIN_PARAMETER_BUDGET))
    budget=int(round(math.exp(logb)))
    morph=MORPHOLOGIES[min(int(math.floor(u[2]*len(MORPHOLOGIES))),len(MORPHOLOGIES)-1)]
    strength=0.20+1.25*float(u[3]); curvature=0.05+0.90*float(u[4])
    hidden=realize_widths(depth,budget,morph,strength,curvature)
    if hidden is None: return None
    widths=[INPUT_DIM]+hidden+[1]; realized=parameter_count(widths,bias=MODEL_BIAS)
    h=np.asarray(hidden,float)
    width_cv=float(np.std(h)/max(np.mean(h),EPS))
    bottleneck_ratio=float(np.min(h)/max(np.max(h),EPS))
    expansion_index=float((h[-1]-h[0])/max(np.median(h),EPS)) if len(h)>1 else 0.0
    log_paths=float(math.log(INPUT_DIM)+np.sum(np.log(np.clip(h,1,None))))
    return {
        "hidden_depth":int(depth),"target_budget":int(budget),"morphology":morph,
        "shape_strength":float(strength),"shape_curvature":float(curvature),
        "hidden_widths":[int(x) for x in hidden],"realized_parameters":int(realized),
        "median_width":float(np.median(h)),"min_width":int(np.min(h)),"max_width":int(np.max(h)),
        "width_cv":width_cv,"bottleneck_ratio":bottleneck_ratio,
        "expansion_index":expansion_index,"log_topological_paths":log_paths,
    }


def candidate_key(c: Dict[str,Any]) -> str:
    return stable_hash_dict({k:c[k] for k in ["hidden_depth","target_budget","morphology","shape_strength","shape_curvature","hidden_widths"]},12)


def candidate_features(c: Dict[str,Any]) -> np.ndarray:
    m=[1.0 if str(c["morphology"])==x else 0.0 for x in MORPHOLOGIES]
    hidden = c.get("hidden_widths", [])
    if isinstance(hidden, str): hidden=json.loads(hidden)
    h=np.asarray(hidden,float)
    width_cv=float(c.get("width_cv", np.std(h)/max(np.mean(h),EPS)))
    br=float(c.get("bottleneck_ratio", np.min(h)/max(np.max(h),EPS)))
    ex=float(c.get("expansion_index", (h[-1]-h[0])/max(np.median(h),EPS) if len(h)>1 else 0.0))
    lp=float(c.get("log_topological_paths", math.log(INPUT_DIM)+np.sum(np.log(np.clip(h,1,None)))))
    return np.asarray([
        float(c["hidden_depth"]), math.log(max(float(c["realized_parameters"]),1.0)), lp,
        float(c["median_width"]), width_cv, br, ex,
        float(c.get("shape_strength",0.0)), float(c.get("shape_curvature",0.0)), *m
    ],float)


def random_candidate_pool(n: int, seed: int, existing: Optional[set]=None) -> List[Dict[str,Any]]:
    existing=set(existing or []); out=[]; seen=set(existing)
    sampler=qmc.Sobol(d=5,scramble=True,seed=int(seed))
    m=int(math.ceil(math.log2(max(int(n)*3,8))))
    U=sampler.random_base2(m=m)
    for u in U:
        c=unit_to_candidate(u)
        if c is None: continue
        k=candidate_key(c)
        if k in seen: continue
        c["config_id"]="C_"+k; out.append(c); seen.add(k)
        if len(out)>=int(n): break
    return out


def expected_improvement_min(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    mu=np.asarray(mu,float); sigma=np.asarray(sigma,float)
    improvement=best-mu-float(xi); z=np.zeros_like(mu)
    good=sigma>1e-12; z[good]=improvement[good]/sigma[good]
    ei=np.zeros_like(mu); ei[good]=improvement[good]*norm.cdf(z[good])+sigma[good]*norm.pdf(z[good])
    ei[~good]=np.maximum(improvement[~good],0.0)
    return ei


def train_discovery_candidate(c: Dict[str,Any], splits, candidate_index: int) -> Dict[str,Any]:
    records=[]
    Xtr,ytr=splits["train"]; Xv,yv=splits["val"]
    for ai,act in enumerate(ANCHOR_ACTIVATIONS):
        seed_idx=1000  # fixed discovery seed across candidate architectures
        model=GeneralMLP(INPUT_DIM,c["hidden_widths"],act,bias=MODEL_BIAS,seed=model_seed(seed_idx)).to(DEVICE)
        cal_n=min(512,len(Xtr)); calibrate_initial_signal(model,Xtr[:cal_n])
        opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
        best=np.inf
        model.train()
        for epoch in range(RFSMAC_DISCOVERY_EPOCHS):
            for ids in deterministic_batches(len(Xtr),epoch,seed_idx,BATCH_SIZE):
                xb=Xtr[ids].to(DEVICE); yb=ytr[ids].to(DEVICE)
                opt.zero_grad(set_to_none=True); pred=model(xb); loss=F.mse_loss(pred,yb)
                if not torch.isfinite(loss):
                    best=np.inf; break
                loss.backward(); opt.step()
            val=eval_mse(model,Xv,yv); best=min(best,val)
            if not np.isfinite(val): break
        records.append({"activation":act,"best_val_mse":float(best)})
        del model,opt
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    vals=[r["best_val_mse"] for r in records]
    return {
        "discovery_objective":float(np.max(vals)) if np.all(np.isfinite(vals)) else np.inf,
        "discovery_median":float(np.median(vals)) if np.any(np.isfinite(vals)) else np.inf,
        "anchor_results":records,
    }


def run_rf_smac_discovery() -> pd.DataFrame:
    path=OUTDIR/"tables"/"rf_smac_archive.csv"
    if P2B_RESUME and path.exists():
        old=pd.read_csv(path)
    else:
        old=pd.DataFrame()
    splits,_=generate_environment("DISC",discovery=True)
    archive=[] if old.empty else old.to_dict("records")
    observed_keys=set(str(r["config_key"]) for r in archive if "config_key" in r)

    # Sobol cold-start.
    need=max(0,RFSMAC_SOBOL_COLD-len(archive))
    if need:
        cold=random_candidate_pool(need, P2B_SEED+11, observed_keys)
        eta=ETAReporter("RF-SMAC cold start",len(cold))
        for k,c in enumerate(cold,1):
            t0=time.perf_counter(); res=train_discovery_candidate(c,splits,len(archive))
            rec={**c,"config_key":candidate_key(c),"search_stage":"sobol","search_index":len(archive),
                 "discovery_objective":res["discovery_objective"],"discovery_median":res["discovery_median"],
                 "anchor_json":json.dumps(res["anchor_results"])}
            archive.append(rec); observed_keys.add(rec["config_key"]); atomic_csv(pd.DataFrame(archive),path)
            eta.update(time.perf_counter()-t0,k)
        eta.finish()

    # RF-SMAC BO refinement. Each step fits an RF surrogate on the accumulated archive.
    target_total=RFSMAC_TOTAL_CANDIDATES
    eta=ETAReporter("RF-SMAC refinement",max(target_total-len(archive),0))
    completed=0
    while len(archive)<target_total:
        valid=[r for r in archive if np.isfinite(float(r["discovery_objective"]))]
        pool=random_candidate_pool(RFSMAC_POOL,P2B_SEED+1000+len(archive),observed_keys)
        if not pool: raise RuntimeError("RF-SMAC candidate pool exhausted")
        if len(valid)<max(6,RFSMAC_SOBOL_COLD//2) or random.random()<RFSMAC_RANDOM_INTERLEAVE:
            chosen=pool[np.random.default_rng(P2B_SEED+len(archive)).integers(0,len(pool))]
            acq=np.nan; pred_mu=np.nan; pred_sd=np.nan; stage="random_interleave"
        else:
            X=np.vstack([candidate_features(r) for r in valid]); y=np.asarray([float(r["discovery_objective"]) for r in valid])
            scaler=StandardScaler().fit(X); Xs=scaler.transform(X)
            rf=RandomForestRegressor(n_estimators=RFSMAC_RF_TREES,min_samples_leaf=2,max_features="sqrt",
                                     random_state=P2B_SEED+len(archive),n_jobs=-1)
            rf.fit(Xs,y)
            Xp=scaler.transform(np.vstack([candidate_features(c) for c in pool]))
            trees=np.vstack([t.predict(Xp) for t in rf.estimators_])
            mu=trees.mean(0); sd=trees.std(0,ddof=1); best=float(np.min(y))
            ei=expected_improvement_min(mu,sd,best,RFSMAC_EI_XI)
            idx=int(np.argmax(ei)); chosen=pool[idx]; acq=float(ei[idx]); pred_mu=float(mu[idx]); pred_sd=float(sd[idx]); stage="rf_ei"
        t0=time.perf_counter(); res=train_discovery_candidate(chosen,splits,len(archive))
        rec={**chosen,"config_key":candidate_key(chosen),"search_stage":stage,"search_index":len(archive),
             "discovery_objective":res["discovery_objective"],"discovery_median":res["discovery_median"],
             "acquisition_ei":acq,"pred_mu":pred_mu,"pred_sd":pred_sd,
             "anchor_json":json.dumps(res["anchor_results"])}
        archive.append(rec); observed_keys.add(rec["config_key"]); atomic_csv(pd.DataFrame(archive),path)
        completed+=1; eta.update(time.perf_counter()-t0,completed)
    eta.finish(); return pd.DataFrame(archive)


def select_diverse_configurations(archive: pd.DataFrame) -> pd.DataFrame:
    outpath=OUTDIR/"tables"/"selected_configurations.csv"
    if P2B_RESUME and outpath.exists():
        df=pd.read_csv(outpath)
        df["hidden_widths"] = df["hidden_widths"].apply(lambda s: json.loads(s) if isinstance(s,str) else s)
        return df
    a=archive.copy(); a=a[np.isfinite(a.discovery_objective.astype(float))].copy()
    if len(a)<RFSMAC_SELECT_N: raise RuntimeError("Too few viable RF-SMAC configurations")
    cutoff=float(a.discovery_objective.quantile(min(max(RFSMAC_VIABLE_FRACTION,0.1),1.0)))
    viable=a[a.discovery_objective<=cutoff].copy()
    if len(viable)<RFSMAC_SELECT_N: viable=a.nsmallest(max(RFSMAC_SELECT_N,len(viable)),"discovery_objective")
    Z=np.vstack([candidate_features(r) for _,r in viable.iterrows()]); Z=StandardScaler().fit_transform(Z)
    # Seed with best viable configuration; then maximin/farthest-point coverage.
    selected=[int(np.argmin(viable.discovery_objective.to_numpy(float)))]
    while len(selected)<min(RFSMAC_SELECT_N,len(viable)):
        d=np.min(np.linalg.norm(Z[:,None,:]-Z[np.asarray(selected)][None,:,:],axis=2),axis=1)
        d[np.asarray(selected)]=-np.inf; selected.append(int(np.argmax(d)))
    sel=viable.iloc[selected].copy().reset_index(drop=True)
    sel["selected_rank"]=np.arange(len(sel)); sel["config_id"]=[f"A{k:02d}" for k in range(len(sel))]
    save=sel.copy(); save["hidden_widths"]=save["hidden_widths"].apply(lambda x: json.dumps(x) if not isinstance(x,str) else x)
    atomic_csv(save,outpath); return sel


# ==================================================================================================
# 5. STRUCTURAL PATH MEASURE (WORK-I BASELINE)
# ==================================================================================================

def path_factor_matrices(weights: List[np.ndarray]) -> List[np.ndarray]:
    return [np.square(np.asarray(W,np.float64)) for W in weights]


def log_partition(A: List[np.ndarray], q: float = 1.0) -> float:
    z=np.ones(A[0].shape[1],dtype=np.float64); z/=max(float(z.sum()),EPS); log_scale=math.log(A[0].shape[1])
    for M in A:
        Mq=np.power(np.clip(M,PATH_LOG_FLOOR,None),q); z=Mq@z; s=float(z.sum())
        if s<=0 or not np.isfinite(s): return -np.inf
        z/=s; log_scale+=math.log(s)
    return float(log_scale)


def normalized_forward_backward(A: List[np.ndarray]):
    alpha=[np.ones(A[0].shape[1],float)]; alpha[0]/=alpha[0].sum()
    for M in A:
        z=M@alpha[-1]; s=float(z.sum())
        if s<=EPS or not np.isfinite(s): raise FloatingPointError("Degenerate forward path partition")
        alpha.append(z/s)
    beta=[None]*(len(A)+1); beta[-1]=np.ones(A[-1].shape[0],float); beta[-1]/=beta[-1].sum()
    for l in range(len(A)-1,-1,-1):
        z=A[l].T@beta[l+1]; s=float(z.sum())
        if s<=EPS or not np.isfinite(s): raise FloatingPointError("Degenerate backward path partition")
        beta[l]=z/s
    return alpha,beta


def path_flow(A: List[np.ndarray]):
    alpha,beta=normalized_forward_backward(A); edge_flow=[]
    for l,M in enumerate(A):
        E=M*beta[l+1][:,None]*alpha[l][None,:]; E/=max(float(E.sum()),EPS); edge_flow.append(E)
    node_flow=[]
    for l in range(len(alpha)):
        v=alpha[l]*beta[l]; v/=max(float(v.sum()),EPS); node_flow.append(v)
    return edge_flow,node_flow


def flow_enrichment(edge_flow,node_flow):
    return [E*E.size for E in edge_flow],[v*len(v) for v in node_flow]


def layer_conservation_errors(edge_flow,node_flow):
    return (float(max(abs(float(E.sum())-1.0) for E in edge_flow)),
            float(max(abs(float(v.sum())-1.0) for v in node_flow)))


def flow_concentration(edge_enr,node_enr):
    def one(layers):
        cv=[]; top=[]
        for x in layers:
            a=np.asarray(x,float).ravel(); a=a[np.isfinite(a)&(a>=0)]
            if not len(a): continue
            cv.append(float(a.std()/max(a.mean(),EPS)))
            k=max(1,int(math.ceil(.10*len(a)))); top.append(float(np.sort(a)[-k:].sum()/max(a.sum(),EPS)))
        return float(np.mean(cv)) if cv else np.nan,float(np.mean(top)) if top else np.nan
    ecv,etop=one(edge_enr); ncv,ntop=one(node_enr); return ecv,etop,ncv,ntop


def structural_state(model: GeneralMLP) -> Dict[str,Any]:
    weights=extract_weights(model); A=path_factor_matrices(weights); widths=model.widths
    logZ1=log_partition(A,1.0); logZ2=log_partition(A,2.0); logN=float(np.sum(np.log(np.asarray(widths,float))))
    ef,nf=path_flow(A); ee,nn=flow_enrichment(ef,nf); eerr,nerr=layer_conservation_errors(ef,nf)
    if max(eerr,nerr)>1e-9: raise RuntimeError("Structural conservation failed")
    # Exact Shannon entropy from local edge expectation of log Q.
    elogq=sum(float(np.sum(Fm*np.log(np.clip(M,PATH_LOG_FLOOR,None)))) for Fm,M in zip(ef,A))
    H=float(logZ1-elogq); logeta2=float(2*logZ1-logZ2-logN); ecv,etop,ncv,ntop=flow_concentration(ee,nn)
    return {
        "logZ1":logZ1,"logZ2":logZ2,"log_N_topological":logN,"S_path":H,
        "entropy_deficit":float(logN-H),"eta1":float(np.exp(np.clip(H-logN,-745,0))),
        "eta2":float(np.exp(np.clip(logeta2,-745,0))),"log_eta1":float(H-logN),"log_eta2":logeta2,
        "edge_flow":ef,"node_flow":nf,"edge_enrichment":ee,"node_enrichment":nn,
        "edge_cv":ecv,"edge_top10_share":etop,"node_cv":ncv,"node_top10_share":ntop,
        "edge_conservation_error":eerr,"node_conservation_error":nerr,
    }


# ==================================================================================================
# 6. BATCHED CONDITIONAL PATH MEASURE
# ==================================================================================================

@torch.no_grad()
def _finite_secant_torch(name: str, z: torch.Tensor, z0: torch.Tensor, tol: float = 1e-8) -> torch.Tensor:
    dz=z-z0
    if name=="relu": f=lambda q:F.relu(q)
    elif name=="leaky_relu": f=lambda q:F.leaky_relu(q,negative_slope=0.1)
    elif name=="tanh": f=torch.tanh
    elif name=="sigmoid": f=torch.sigmoid
    elif name=="silu": f=F.silu
    elif name=="identity": f=lambda q:q
    else: raise ValueError(name)
    out=torch.empty_like(dz); mask=torch.abs(dz)>tol
    if mask.any(): out[mask]=(f(z[mask])-f(z0[mask]))/dz[mask]
    if (~mask).any():
        zz=0.5*(z[~mask]+z0[~mask])
        if name=="relu": der=(zz>0).to(zz.dtype)
        elif name=="leaky_relu": der=torch.where(zz>=0,torch.ones_like(zz),torch.full_like(zz,0.1))
        elif name=="tanh":
            t=torch.tanh(zz); der=1-t*t
        elif name=="sigmoid":
            s=torch.sigmoid(zz); der=s*(1-s)
        elif name=="silu":
            s=torch.sigmoid(zz); der=s+zz*s*(1-s)
        elif name=="identity": der=torch.ones_like(zz)
        out[~mask]=der
    return out


@torch.no_grad()
def conditional_factors_batch(model: GeneralMLP, X: torch.Tensor, X0: torch.Tensor,
                              local_direction: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
    """Return factor matrices M_l[b,out,in] whose path product equals |A_pi|^2.

    Finite case: source factor is (X-X0)^2.
    Local directional case: set X0=X and provide local_direction v; hidden response uses phi'(z(X)) and
    the source factor is v^2. This is the exact directional Jacobian-path limit.
    """
    model.eval(); X=X.to(DEVICE); X0=X0.to(DEVICE)
    _,zlist,_=model.forward_with_states(X); _,z0list,_=model.forward_with_states(X0)
    if local_direction is None:
        source=torch.square(X-X0)
        responses=[torch.square(_finite_secant_torch(model.activation_name,z,z0)) for z,z0 in zip(zlist,z0list)]
    else:
        source=torch.square(local_direction.to(DEVICE))
        responses=[]
        for z in zlist:
            name=model.activation_name
            if name=="relu": der=(z>0).to(z.dtype)
            elif name=="leaky_relu": der=torch.where(z>=0,torch.ones_like(z),torch.full_like(z,0.1))
            elif name=="tanh":
                t=torch.tanh(z); der=1-t*t
            elif name=="sigmoid":
                s=torch.sigmoid(z); der=s*(1-s)
            elif name=="silu":
                s=torch.sigmoid(z); der=s+z*s*(1-s)
            elif name=="identity": der=torch.ones_like(z)
            responses.append(torch.square(der))
    factors=[]
    for l,layer in enumerate(model.linears):
        W2=torch.square(layer.weight).unsqueeze(0)
        M=W2.expand(X.shape[0],-1,-1)
        if l==0:
            M=M*source[:,None,:]
        if l < len(model.linears)-1:
            M=M*responses[l][:,:,None]
        factors.append(M.to(torch.float64))
    return factors


@torch.no_grad()
def _batched_log_partition(factors: List[torch.Tensor], q: float = 1.0) -> torch.Tensor:
    B=factors[0].shape[0]; nin=factors[0].shape[2]
    z=torch.ones((B,nin),dtype=torch.float64,device=DEVICE)/float(nin)
    logscale=torch.full((B,),math.log(nin),dtype=torch.float64,device=DEVICE)
    for M in factors:
        Mq=torch.pow(torch.clamp(M,min=PATH_LOG_FLOOR),float(q))
        z=torch.bmm(Mq,z.unsqueeze(-1)).squeeze(-1); s=z.sum(1)
        bad=(s<=0)|(~torch.isfinite(s))
        if bad.any(): s=torch.where(bad,torch.full_like(s,EPS),s)
        z=z/s[:,None]; logscale=logscale+torch.log(s)
    return logscale


@torch.no_grad()
def _batched_support_exists(factors: List[torch.Tensor]) -> torch.Tensor:
    """Exact positive-support existence for each nonnegative multiplicative path system.

    A normalized conditional measure exists iff at least one complete path has strictly positive
    product.  This Boolean dynamic program avoids inventing epsilon mass on hard-gated ReLU routes.
    """
    B=factors[0].shape[0]; nin=factors[0].shape[2]
    support=torch.ones((B,nin),dtype=torch.bool,device=factors[0].device)
    for M in factors:
        A=(M>0).to(torch.float64)
        support=(torch.bmm(A,support.to(torch.float64).unsqueeze(-1)).squeeze(-1)>0)
    return support.any(dim=1)


@torch.no_grad()
def conditional_batch_state(model: GeneralMLP, X: torch.Tensor, X0: torch.Tensor,
                            structural: Dict[str,Any], local_direction: Optional[torch.Tensor] = None,
                            return_flows: bool = True) -> Dict[str,Any]:
    factors=conditional_factors_batch(model,X,X0,local_direction=local_direction); B=factors[0].shape[0]
    measure_defined=_batched_support_exists(factors)
    logZ1=_batched_log_partition(factors,1.0); logZ2=_batched_log_partition(factors,2.0)
    alpha=[torch.ones((B,factors[0].shape[2]),dtype=torch.float64,device=DEVICE)]
    alpha[0]/=alpha[0].sum(1,keepdim=True)
    for M in factors:
        z=torch.bmm(M,alpha[-1].unsqueeze(-1)).squeeze(-1); s=z.sum(1,keepdim=True).clamp_min(EPS); alpha.append(z/s)
    beta=[None]*(len(factors)+1); beta[-1]=torch.ones((B,factors[-1].shape[1]),dtype=torch.float64,device=DEVICE)
    beta[-1]/=beta[-1].sum(1,keepdim=True)
    for l in range(len(factors)-1,-1,-1):
        z=torch.bmm(factors[l].transpose(1,2),beta[l+1].unsqueeze(-1)).squeeze(-1)
        beta[l]=z/z.sum(1,keepdim=True).clamp_min(EPS)
    edge_flow=[]; node_flow=[]; elogq=torch.zeros(B,dtype=torch.float64,device=DEVICE)
    edge_cv_layers=[]; node_cv_layers=[]
    for l,M in enumerate(factors):
        E=M*beta[l+1][:,:,None]*alpha[l][:,None,:]; E=E/E.sum((1,2),keepdim=True).clamp_min(EPS)
        elogq += torch.sum(E*torch.log(torch.clamp(M,min=PATH_LOG_FLOOR)),dim=(1,2))
        mean=E.mean((1,2)); sd=E.std((1,2),unbiased=False); edge_cv_layers.append(sd/mean.clamp_min(EPS))
        if return_flows: edge_flow.append(E.detach().cpu().numpy())
    for l in range(len(alpha)):
        v=alpha[l]*beta[l]; v=v/v.sum(1,keepdim=True).clamp_min(EPS)
        mean=v.mean(1); sd=v.std(1,unbiased=False); node_cv_layers.append(sd/mean.clamp_min(EPS))
        if return_flows: node_flow.append(v.detach().cpu().numpy())
    H=logZ1-elogq; logN=float(np.sum(np.log(np.asarray(model.widths,float))))
    logeta1=H-logN; logeta2=2*logZ1-logZ2-logN
    # Exact KL p_delta || p_struct from additive log-factor ratio expectation.
    e_log_qstr=torch.zeros(B,dtype=torch.float64,device=DEVICE)
    for l,E_np in enumerate(edge_flow if return_flows else []):
        E=torch.from_numpy(E_np).to(DEVICE)
        W2=torch.square(model.linears[l].weight).to(torch.float64).unsqueeze(0)
        e_log_qstr += torch.sum(E*torch.log(torch.clamp(W2,min=PATH_LOG_FLOOR)),dim=(1,2))
    if return_flows:
        kl_struct=elogq-e_log_qstr-logZ1+float(structural["logZ1"])
    else:
        kl_struct=torch.full_like(logZ1,float("nan"))

    # Exact Bhattacharyya/Hellinger comparison to the weight-only structural path measure.
    cross_factors=[]
    for l,M in enumerate(factors):
        W2=torch.square(model.linears[l].weight).to(torch.float64).unsqueeze(0).expand(B,-1,-1)
        cross_factors.append(torch.sqrt(torch.clamp(M,min=0.0)*torch.clamp(W2,min=0.0)))
    logZcross=_batched_log_partition(cross_factors,1.0)
    bc=torch.exp(torch.clamp(logZcross-0.5*(logZ1+float(structural["logZ1"])),min=-745.0,max=0.0))
    h2_struct=torch.clamp(1.0-bc,min=0.0,max=1.0)

    return {
        "logZ1":logZ1.cpu().numpy(),"logZ2":logZ2.cpu().numpy(),"S_path":H.cpu().numpy(),
        "entropy_deficit":(logN-H).cpu().numpy(),"eta1":torch.exp(torch.clamp(logeta1,-745,0)).cpu().numpy(),
        "eta2":torch.exp(torch.clamp(logeta2,-745,0)).cpu().numpy(),
        "edge_cv":torch.stack(edge_cv_layers,1).mean(1).cpu().numpy(),
        "node_cv":torch.stack(node_cv_layers,1).mean(1).cpu().numpy(),
        "kl_to_structural":kl_struct.cpu().numpy(),
        "hellinger2_to_structural":h2_struct.cpu().numpy(),
        "measure_defined":measure_defined.cpu().numpy(),
        "edge_flow":edge_flow,"node_flow":node_flow,
    }


def aggregate_conditional_evaluation(model: GeneralMLP, Xeval: torch.Tensor,
                                     structural: Dict[str,Any], reference: Optional[torch.Tensor] = None,
                                     keep_per_input_flows: bool = False) -> Dict[str,Any]:
    """Aggregate only mathematically defined finite conditional measures.

    For hard-gated activations a declared contrast can have zero positive path partition.  Such a
    condition has no normalized p_Delta and is therefore recorded as undefined rather than repaired by
    epsilon smoothing.  Population flow is the mean over the defined condition subset and the validity
    fraction is reported explicitly.
    """
    n=min(int(CONDITIONAL_EVAL_INPUTS),len(Xeval)); Xeval=Xeval[:n]
    if reference is None: reference=torch.zeros_like(Xeval)
    elif reference.ndim==1: reference=reference[None,:].repeat(n,1)
    rows=[]; edge_sum=None; node_sum=None; per_input_flows=[] if keep_per_input_flows else None
    node_all=[[] for _ in model.widths]; n_defined=0
    scalar_names=["S_path","entropy_deficit","eta1","eta2","edge_cv","node_cv","kl_to_structural","hellinger2_to_structural"]
    for lo in range(0,n,CONDITIONAL_BATCH):
        hi=min(lo+CONDITIONAL_BATCH,n); xb=Xeval[lo:hi]; x0=reference[lo:hi]
        st=conditional_batch_state(model,xb,x0,structural,return_flows=True)
        valid=np.asarray(st["measure_defined"],bool)
        for k in range(hi-lo):
            rec={"measure_defined":bool(valid[k])}
            for m in scalar_names: rec[m]=float(st[m][k]) if valid[k] and np.isfinite(st[m][k]) else np.nan
            rows.append(rec)
        if np.any(valid):
            nv=int(np.sum(valid)); n_defined+=nv
            if edge_sum is None:
                edge_sum=[np.sum(E[valid],axis=0) for E in st["edge_flow"]]
                node_sum=[np.sum(v[valid],axis=0) for v in st["node_flow"]]
            else:
                for l,E in enumerate(st["edge_flow"]): edge_sum[l]+=np.sum(E[valid],axis=0)
                for l,v in enumerate(st["node_flow"]): node_sum[l]+=np.sum(v[valid],axis=0)
            for l,v in enumerate(st["node_flow"]): node_all[l].append(v[valid])
            if keep_per_input_flows:
                for k in np.where(valid)[0]: per_input_flows.append(([E[k].copy() for E in st["edge_flow"]],[v[k].copy() for v in st["node_flow"]]))
    df=pd.DataFrame(rows)
    if n_defined<=0:
        summary={f"conditional_{c}_median":np.nan for c in scalar_names}
        summary.update({f"conditional_{c}_mean":np.nan for c in scalar_names})
        summary.update({"conditional_structural_edge_spearman":np.nan,"input_specialization_mean":np.nan,"input_specialization_max":np.nan,
                        "conditional_defined_fraction":0.0,"conditional_defined_count":0,"conditional_requested_count":int(n)})
        return {"input_df":df,"population_state":{"defined":False},"specialization":[np.nan for _ in model.widths],"summary":summary,"per_input_flows":per_input_flows}
    edge_avg=[E/n_defined for E in edge_sum]; node_avg=[v/n_defined for v in node_sum]
    ee,nn=flow_enrichment(edge_avg,node_avg); eerr,nerr=layer_conservation_errors(edge_avg,node_avg)
    if max(eerr,nerr)>1e-8: raise RuntimeError(f"Conditional population conservation failed {eerr} {nerr}")
    specialization=[]
    for l,chunks in enumerate(node_all):
        if not chunks: specialization.append(np.nan);continue
        V=np.concatenate(chunks,axis=0); mean=V.mean(0); ratio=np.where(V>0,V/np.clip(mean[None,:],EPS,None),1.0)
        I=float(np.mean(np.sum(np.where(V>0,V*np.log(np.clip(ratio,EPS,None)),0.0),axis=1)))
        specialization.append(I)
    pflat=np.concatenate([E.ravel() for E in edge_avg]); sflat=np.concatenate([E.ravel() for E in structural["edge_flow"]])
    rank=float(spearmanr(pflat,sflat).statistic) if len(pflat)>2 else np.nan
    pop_state={"defined":True,"edge_flow":edge_avg,"node_flow":node_avg,"edge_enrichment":ee,"node_enrichment":nn,
               "edge_conservation_error":eerr,"node_conservation_error":nerr}
    numeric=[c for c in df.columns if c!="measure_defined"]
    summary={f"conditional_{c}_median":float(df[c].median(skipna=True)) for c in numeric}
    summary.update({f"conditional_{c}_mean":float(df[c].mean(skipna=True)) for c in numeric})
    summary["conditional_structural_edge_spearman"]=rank
    summary["input_specialization_mean"]=float(np.nanmean(specialization))
    summary["input_specialization_max"]=float(np.nanmax(specialization))
    summary["conditional_defined_fraction"]=float(n_defined/max(n,1));summary["conditional_defined_count"]=int(n_defined);summary["conditional_requested_count"]=int(n)
    return {"input_df":df,"population_state":pop_state,"specialization":specialization,"summary":summary,"per_input_flows":per_input_flows}



# ==================================================================================================
# 7. BCANw GEOMETRY — FAITHFUL WORK-I/III IMPLEMENTATION
# ==================================================================================================
@dataclass
class LayeredGraph:
    n_nodes: int
    widths: List[int]
    offsets: List[int]
    edge_mag: Dict[Tuple[int, int], float]
    layer_of_edge: Dict[Tuple[int, int], int]

def graph_from_matrices(mats: List[np.ndarray]) -> LayeredGraph:
    widths = [int(mats[0].shape[1])] + [int(M.shape[0]) for M in mats]
    offsets = [0]
    for w in widths[:-1]:
        offsets.append(offsets[-1] + int(w))
    edge_mag: Dict[Tuple[int, int], float] = {}
    layer_of_edge: Dict[Tuple[int, int], int] = {}
    for l, M in enumerate(mats):
        M = np.asarray(M, float)
        for j in range(M.shape[0]):
            v = offsets[l + 1] + j
            for i in range(M.shape[1]):
                u = offsets[l] + i
                e = (u, v) if u < v else (v, u)
                edge_mag[e] = float(max(M[j, i], 0.0))
                layer_of_edge[e] = int(l)
    return LayeredGraph(sum(widths), widths, offsets, edge_mag, layer_of_edge)

def metric_lengths(P: LayeredGraph, power: float = METRIC_POWER) -> Dict[Tuple[int, int], float]:
    return {e: float((max(w, 0.0) + EPS) ** (-power)) for e, w in P.edge_mag.items()}

def distance_matrix(P: LayeredGraph, lengths: Dict[Tuple[int, int], float]) -> np.ndarray:
    rows, cols, vals = [], [], []
    for (u, v), L in lengths.items():
        rows.extend([u, v]); cols.extend([v, u]); vals.extend([L, L])
    A = csr_matrix((vals, (rows, cols)), shape=(P.n_nodes, P.n_nodes), dtype=float)
    return np.asarray(csgraph_dijkstra(A, directed=False, unweighted=False), float)

def bcanw_scales(D: np.ndarray, n_scales: int) -> np.ndarray:
    vals = D[np.triu_indices(D.shape[0], k=1)]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return np.array([], dtype=float)
    lo = max(float(np.min(vals)) * 0.95, EPS)
    hi = float(np.max(vals)) * 1.005
    if not np.isfinite(hi) or hi <= lo:
        return np.array([lo], dtype=float)
    return np.geomspace(lo, hi, max(int(n_scales), 4))

def node_strength(P: LayeredGraph) -> np.ndarray:
    s = np.zeros(P.n_nodes, float)
    for (u, v), w in P.edge_mag.items():
        s[u] += w; s[v] += w
    return s

def greedy_bcanw_coloring(D: np.ndarray, ell: float, order: np.ndarray) -> np.ndarray:
    """Wei-style greedy coloring of the distance conflict graph, with canonical strength order."""
    n = D.shape[0]
    color = np.full(n, -1, int)
    for u in order:
        conflicts = D[u] >= float(ell)
        used = set(color[conflicts & (color >= 0)].tolist())
        c = 0
        while c in used:
            c += 1
        color[u] = c
    return color

def global_node_measure(node_flow: List[np.ndarray]) -> np.ndarray:
    x = np.concatenate([np.asarray(v, float).ravel() for v in node_flow])
    x = np.clip(x, 0.0, None)
    x /= max(float(x.sum()), EPS)
    return x

def renyi_effective_fraction(probabilities: np.ndarray, q: float) -> float:
    p = np.asarray(probabilities, float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan
    p = p / p.sum(); E = len(p)
    if abs(float(q) - 1.0) < 1e-12:
        H = -float(np.sum(p * np.log(p)))
    else:
        H = float(math.log(np.sum(p ** q)) / (1.0 - q))
    return float(math.exp(H) / E)

def box_flow_observables(colors: np.ndarray, node_measure: np.ndarray,
                         raw_flow_graph: LayeredGraph) -> Dict[str, Any]:
    """
    BCANw covering uses an undirected metric, but quotient path flow preserves the original
    feed-forward direction. This matches the corrected Phase-I/II construction.
    """
    nb = int(colors.max()) + 1
    mass = np.bincount(colors, weights=node_measure, minlength=nb).astype(float)
    mean = max(float(np.mean(mass)), EPS)
    lac = float(np.mean(mass * mass) / (mean * mean))

    W = np.zeros((nb, nb), float)
    total_flow = 0.0; inter_flow = 0.0
    for (u, v), f in raw_flow_graph.edge_mag.items():
        total_flow += float(f)
        a, b = int(colors[u]), int(colors[v])
        if a == b:
            continue
        W[a, b] += float(f)
        inter_flow += float(f)
    mask = (W > 0) & (~np.eye(nb, dtype=bool))
    vals = W[mask]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        vals_norm = vals / max(float(np.mean(vals)), EPS)
        probs = vals / max(float(np.sum(vals)), EPS)
        eta = {q: renyi_effective_fraction(probs, q) for q in COARSE_RENYI_Q}
    else:
        vals_norm = np.array([]); probs = np.array([])
        eta = {q: np.nan for q in COARSE_RENYI_Q}
    return {
        "box_mass": mass,
        "lacunarity": lac,
        "quotient_flow": vals,
        "quotient_flow_norm": vals_norm,
        "quotient_prob": probs,
        "coarse_eta_q": eta,
        "n_quotient_edges": int(len(vals)),
        "interbox_flow_fraction": float(inter_flow / max(total_flow, EPS)),
    }

def compress_box_staircase(radii: np.ndarray, NB: np.ndarray,
                           lac: Optional[np.ndarray] = None):
    r = np.asarray(radii, float); n = np.asarray(NB, float)
    good = np.isfinite(r) & np.isfinite(n) & (r > 0)
    r, n = r[good], n[good]
    l = np.asarray(lac, float)[good] if lac is not None else None
    if len(r) == 0:
        return (r, n, np.array([])) if lac is not None else (r, n)
    keep = [0]
    for k in range(1, len(r)):
        if not np.isclose(n[k], n[keep[-1]], rtol=0, atol=0):
            keep.append(k)
    keep = np.array(keep, int)
    if lac is None:
        return r[keep], n[keep]
    return r[keep], n[keep], l[keep]

def covering_quantile_scale(radii: np.ndarray, NB: np.ndarray, N: int, q: float) -> float:
    """Log-radius interpolation of the radius at which N_B/N crosses q."""
    r = np.asarray(radii, float); y = np.asarray(NB, float) / float(N)
    good = np.isfinite(r) & np.isfinite(y) & (r > 0)
    r, y = r[good], y[good]
    if len(r) == 0:
        return np.nan
    order = np.argsort(r); r, y = r[order], y[order]
    # enforce a non-increasing envelope against tiny greedy-cover irregularities
    y = np.minimum.accumulate(y)
    if q >= y[0]:
        return float(r[0])
    if q <= y[-1]:
        return float(r[-1])
    idx = np.where(y <= q)[0]
    if len(idx) == 0:
        return np.nan
    j = int(idx[0])
    if j == 0:
        return float(r[0])
    y0, y1 = y[j - 1], y[j]
    lr0, lr1 = math.log(r[j - 1]), math.log(r[j])
    if abs(y1 - y0) < EPS:
        return float(math.sqrt(r[j - 1] * r[j]))
    t = float((q - y0) / (y1 - y0))
    return float(math.exp(lr0 + t * (lr1 - lr0)))

def local_effective_dimension(radii_c: np.ndarray, NB_c: np.ndarray, xi50: float,
                              window: int = LOCAL_SLOPE_WINDOW):
    r = np.asarray(radii_c, float); n = np.asarray(NB_c, float)
    good = np.isfinite(r) & np.isfinite(n) & (r > 0) & (n > 1)
    r, n = r[good], n[good]
    if len(r) == 0 or not np.isfinite(xi50) or xi50 <= 0:
        return np.array([]), np.array([]), np.array([])
    x = r / xi50
    lx = np.log(r); ly = np.log(n)
    d = np.full(len(r), np.nan); rr = np.full(len(r), np.nan)
    half = max(int(window) // 2, 1)
    for i in range(len(r)):
        lo, hi = max(0, i - half), min(len(r), i + half + 1)
        if hi - lo < 3:
            continue
        b, a = np.polyfit(lx[lo:hi], ly[lo:hi], 1)
        pred = a + b * lx[lo:hi]
        d[i] = -float(b); rr[i] = r2_score_np(ly[lo:hi], pred)
    return x, d, rr

def _window_slope_cv(x: np.ndarray, y: np.ndarray, subwin: int = 5) -> float:
    if len(x) < max(4, subwin):
        return np.nan
    ds = []
    for i in range(0, len(x) - subwin + 1):
        b, _ = np.polyfit(x[i:i + subwin], y[i:i + subwin], 1)
        ds.append(-float(b))
    ds = np.asarray(ds, float)
    return float(np.std(ds) / max(abs(float(np.mean(ds))), EPS)) if len(ds) >= 2 else np.nan

def strict_scaling_window(radii_c: np.ndarray, NB_c: np.ndarray, N: int) -> Dict[str, Any]:
    empty = {
        "dB": np.nan, "r2": np.nan, "span_decades": 0.0, "n_points": 0,
        "nb_ratio": np.nan, "i0": -1, "i1": -1, "slope_cv": np.nan,
        "delta_bic": np.nan, "credible": False,
    }
    r = np.asarray(radii_c, float); nb = np.asarray(NB_c, float)
    good = np.isfinite(r) & np.isfinite(nb) & (r > 0) & (nb > 1) & (nb < 0.95 * N)
    ids = np.where(good)[0]
    if len(ids) < SCALING_MIN_POINTS:
        return empty
    candidates = []
    for a in range(0, len(ids) - SCALING_MIN_POINTS + 1):
        for b in range(a + SCALING_MIN_POINTS, len(ids) + 1):
            sel = ids[a:b]
            if np.any(np.diff(sel) != 1):
                continue
            rs, ns = r[sel], nb[sel]
            span = float(np.log10(rs[-1] / rs[0]))
            nb_ratio = float(ns[0] / max(ns[-1], EPS))
            if span < SCALING_MIN_SPAN_DECADES or nb_ratio < SCALING_MIN_NB_RATIO:
                continue
            x, y = np.log(rs), np.log(ns)
            slope, intercept = np.polyfit(x, y, 1); dim = -float(slope)
            pred = intercept + slope * x; rr = r2_score_np(y, pred)
            if not np.isfinite(rr) or rr < SCALING_R2_MIN or dim < SCALING_MIN_DIM:
                continue
            scv = _window_slope_cv(x, y, subwin=min(5, len(x)))
            rss1 = max(float(np.sum((y - pred) ** 2)), EPS)
            bic1 = len(x) * math.log(rss1 / len(x)) + 2 * math.log(len(x))
            q2, q1, q0 = np.polyfit(x, y, 2); pred2 = q0 + q1 * x + q2 * x * x
            rss2 = max(float(np.sum((y - pred2) ** 2)), EPS)
            bic2 = len(x) * math.log(rss2 / len(x)) + 3 * math.log(len(x))
            dbic = float(bic1 - bic2)
            credible = bool(np.isfinite(scv) and scv <= SCALING_MAX_SLOPE_CV and dbic <= SCALING_MAX_DELTA_BIC)
            candidates.append((credible, span, rr, sel[0], sel[-1], dim, len(sel), nb_ratio, scv, dbic))
    if not candidates:
        return empty
    candidates.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
    cred, span, rr, i0, i1, dim, npnt, nbr, scv, dbic = candidates[0]
    return {
        "dB": float(dim), "r2": float(rr), "span_decades": float(span),
        "n_points": int(npnt), "nb_ratio": float(nbr), "i0": int(i0), "i1": int(i1),
        "slope_cv": float(scv), "delta_bic": float(dbic), "credible": bool(cred),
    }

def crossover_from_deff(xscale: np.ndarray, deff: np.ndarray, r2loc: np.ndarray,
                        NB_c: np.ndarray, N: int) -> Dict[str, float]:
    x = np.asarray(xscale, float); d = np.asarray(deff, float); rr = np.asarray(r2loc, float)
    frac = np.asarray(NB_c, float) / float(N)
    m = min(len(x), len(frac)); x, d, rr, frac = x[:m], d[:m], rr[:m], frac[:m]
    good = np.isfinite(x) & np.isfinite(d) & np.isfinite(rr) & (rr >= LOCAL_SLOPE_R2_MIN) & (d > 0)
    good &= (frac >= 0.015) & (frac <= 0.85)
    if not np.any(good):
        return {"x_d_star": np.nan, "d_star": np.nan, "nb_fraction_at_d_star": np.nan}
    ids = np.where(good)[0]; idx = int(ids[np.argmax(d[ids])])
    return {"x_d_star": float(x[idx]), "d_star": float(d[idx]),
            "nb_fraction_at_d_star": float(frac[idx])}

def lacunarity_crossover(radii: np.ndarray, lac: np.ndarray, xi50: float,
                         NB: np.ndarray, N: int) -> Dict[str, float]:
    r = np.asarray(radii, float); l = np.asarray(lac, float); frac = np.asarray(NB, float) / float(N)
    if not np.isfinite(xi50) or xi50 <= 0:
        return {"x_lac_star": np.nan, "lac_star": np.nan, "relative_lac_star": np.nan}
    x = r / xi50
    base = float(l[0]) if len(l) and np.isfinite(l[0]) and l[0] > 0 else 1.0
    rel = l / base
    good = np.isfinite(x) & np.isfinite(l) & (frac >= 0.015) & (frac <= 0.85)
    if not np.any(good):
        return {"x_lac_star": np.nan, "lac_star": np.nan, "relative_lac_star": np.nan}
    ids = np.where(good)[0]; idx = int(ids[np.argmax(l[ids])])
    return {"x_lac_star": float(x[idx]), "lac_star": float(l[idx]),
            "relative_lac_star": float(rel[idx])}

def _compress_common_local_states(rgrid, support, mass):
    r = np.asarray(rgrid, float); s = np.asarray(support, float); m = np.asarray(mass, float)
    good = np.isfinite(r) & np.isfinite(s) & np.isfinite(m) & (r > 0) & (s > 0) & (m > 0)
    r, s, m = r[good], s[good], m[good]
    if len(r) == 0:
        return r, s, m
    keep = [0]
    for k in range(1, len(r)):
        if not (np.isclose(s[k], s[keep[-1]], rtol=1e-10, atol=1e-14) and
                np.isclose(m[k], m[keep[-1]], rtol=1e-10, atol=1e-14)):
            keep.append(k)
    return r[keep], s[keep], m[keep]

def fit_common_local_growth(rgrid, support, mass,
                            min_points=LOCAL_COMMON_MIN_POINTS,
                            min_span=LOCAL_COMMON_MIN_SPAN_DECADES,
                            support_growth=LOCAL_COMMON_SUPPORT_GROWTH,
                            mass_growth=LOCAL_COMMON_MASS_GROWTH,
                            r2_min=LOCAL_COMMON_R2_MIN):
    r, s, m = _compress_common_local_states(rgrid, support, mass)
    valid = (s < LOCAL_COMMON_SATURATION) & (m < LOCAL_COMMON_SATURATION)
    r, s, m = r[valid], s[valid], m[valid]
    if len(r) < min_points:
        return (np.nan,) * 8
    best = None
    for i in range(0, len(r) - min_points + 1):
        for j in range(i + min_points, len(r) + 1):
            rr, ss, mm = r[i:j], s[i:j], m[i:j]
            span = float(np.log10(rr[-1] / rr[0]))
            if span < min_span:
                continue
            if ss[-1] / max(ss[0], EPS) < support_growth:
                continue
            if mm[-1] / max(mm[0], EPS) < mass_growth:
                continue
            x = np.log(rr)
            bd, ad = np.polyfit(x, np.log(ss), 1)
            ba, aa = np.polyfit(x, np.log(mm), 1)
            if bd <= 0 or ba <= 0:
                continue
            pd = ad + bd * x; pa = aa + ba * x
            r2d = r2_score_np(np.log(ss), pd); r2a = r2_score_np(np.log(mm), pa)
            if r2d < r2_min or r2a < r2_min:
                continue
            score = (span, min(r2d, r2a), len(rr))
            rec = (float(bd), float(ba), float(ba - bd), float(r2d), float(r2a), span, int(len(rr)), float(rr[0]))
            if best is None or score > best[0]:
                best = (score, rec)
    return best[1] if best is not None else (np.nan,) * 8

def local_metric_dimensions(D: np.ndarray, node_measure: np.ndarray,
                            max_radii: int = 48) -> Dict[str, np.ndarray]:
    N = D.shape[0]
    dloc = np.full(N, np.nan); alpha = np.full(N, np.nan); delta = np.full(N, np.nan)
    r2d = np.full(N, np.nan); r2a = np.full(N, np.nan); span = np.full(N, np.nan)
    npts = np.zeros(N, int)
    for i in range(N):
        di = np.asarray(D[i], float)
        vals = np.unique(di[np.isfinite(di) & (di > 0)])
        if len(vals) < LOCAL_COMMON_MIN_POINTS:
            continue
        if len(vals) > max_radii:
            ids = np.unique(np.round(np.linspace(0, len(vals) - 1, max_radii)).astype(int))
            vals = vals[ids]
        support = np.array([np.mean(di <= r) for r in vals], float)
        mass = np.array([np.sum(node_measure[di <= r]) for r in vals], float)
        bd, ba, dd, rd, ra, sp, npnt, _ = fit_common_local_growth(vals, support, mass)
        dloc[i], alpha[i], delta[i], r2d[i], r2a[i], span[i], npts[i] = bd, ba, dd, rd, ra, sp, int(npnt) if np.isfinite(npnt) else 0
    valid = np.isfinite(delta)
    return {
        "dloc": dloc, "alpha": alpha, "delta": delta,
        "r2_dloc": r2d, "r2_alpha": r2a, "span": span, "points": npts,
        "valid_delta_fraction": float(np.mean(valid)),
    }

def nearest_resolution_indices(NB: np.ndarray, N: int,
                               target_fractions=RENORM_TARGET_BOX_FRACTIONS) -> List[int]:
    frac = np.asarray(NB, float) / float(N); out: List[int] = []
    for t in target_fractions:
        eligible = np.where((NB >= 3) & (NB <= 0.90 * N))[0]
        if eligible.size == 0:
            continue
        idx = int(eligible[np.argmin(np.abs(frac[eligible] - t))])
        if idx not in out:
            out.append(idx)
    return out

def mean_pairwise_ks(arrays: List[np.ndarray]) -> float:
    arr = []
    for a in arrays:
        v = np.asarray(a, float); v = v[np.isfinite(v) & (v > 0)]
        if len(v) >= 5:
            arr.append(v)
    if len(arr) < 2:
        return np.nan
    vals = [ks_2samp(arr[i], arr[j]).statistic for i in range(len(arr)) for j in range(i + 1, len(arr))]
    return float(np.mean(vals)) if vals else np.nan

def analyze_geometry(weights: List[np.ndarray], state: Dict[str, Any], n_scales: int,
                     compute_local: bool = False) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute the inherited Phase-I/II geometry at one state; never serializes D."""
    Pgeom = graph_from_matrices(state["edge_enrichment"])
    Praw = graph_from_matrices(state["edge_flow"])
    lengths = metric_lengths(Pgeom, power=METRIC_POWER)
    D = distance_matrix(Pgeom, lengths)
    radii = bcanw_scales(D, n_scales)
    order = np.lexsort((np.arange(Pgeom.n_nodes), -node_strength(Pgeom)))
    node_mu = global_node_measure(state["node_flow"])

    NB, lac, colors_by_scale, qnorm, interfrac = [], [], [], [], []
    coarse_eta = {q: [] for q in COARSE_RENYI_Q}
    for ell in radii:
        colors = greedy_bcanw_coloring(D, float(ell), order)
        colors_by_scale.append(colors)
        NB.append(int(colors.max()) + 1)
        b = box_flow_observables(colors, node_mu, Praw)
        lac.append(b["lacunarity"]); qnorm.append(b["quotient_flow_norm"])
        interfrac.append(b["interbox_flow_fraction"])
        for q in COARSE_RENYI_Q:
            coarse_eta[q].append(b["coarse_eta_q"][q])
    NB = np.asarray(NB, float); lac = np.asarray(lac, float); interfrac = np.asarray(interfrac, float)
    rc, nc, lc = compress_box_staircase(radii, NB, lac)
    xi50 = covering_quantile_scale(rc, nc, Pgeom.n_nodes, 0.50)
    x, deff, deff_r2 = local_effective_dimension(rc, nc, xi50)
    sw = strict_scaling_window(rc, nc, Pgeom.n_nodes)
    cross = crossover_from_deff(x, deff, deff_r2, nc, Pgeom.n_nodes)
    lcross = lacunarity_crossover(radii, lac, xi50, NB, Pgeom.n_nodes)

    qidx = nearest_resolution_indices(NB, Pgeom.n_nodes)
    qarr = [qnorm[i] for i in qidx]
    qks = mean_pairwise_ks(qarr)
    retention = float(np.nanmean(interfrac[qidx])) if len(qidx) else np.nan

    local_delta_median = np.nan; valid_local = np.nan
    if compute_local:
        ld = local_metric_dimensions(D, node_mu)
        valid = np.isfinite(ld["delta"])
        local_delta_median = float(np.nanmedian(ld["delta"][valid])) if np.any(valid) else np.nan
        valid_local = float(np.mean(valid))

    summary = {
        "xi50": float(xi50), "log_xi50": safe_log(xi50),
        "scaling_dB": float(sw["dB"]), "scaling_r2": float(sw["r2"]),
        "scaling_credible": bool(sw["credible"]), "scaling_span": float(sw["span_decades"]),
        "scaling_slope_cv": float(sw["slope_cv"]), "scaling_delta_bic": float(sw["delta_bic"]),
        "d_star": float(cross["d_star"]), "x_d_star": float(cross["x_d_star"]),
        "lacunarity_star": float(lcross["lac_star"]), "x_lacunarity_star": float(lcross["x_lac_star"]),
        "quotient_mean_pairwise_KS": float(qks),
        "renorm_flow_retention_mean": float(retention),
        "local_delta_median": float(local_delta_median),
        "valid_local_delta_fraction": float(valid_local),
        "n_graph_nodes": int(Pgeom.n_nodes), "n_bcanw_scales": int(len(radii)),
    }

    # Scale rows retain exact BCANw profiles. d_eff lives on the compressed staircase;
    # interpolate it in log x only for visualization/aggregation, never for fitting.
    if len(radii):
        xb = radii / max(xi50, EPS) if np.isfinite(xi50) and xi50 > 0 else np.full(len(radii), np.nan)
        di = np.full(len(radii), np.nan); ri = np.full(len(radii), np.nan)
        if len(x) >= 2 and np.all(np.isfinite(x)):
            goodd = np.isfinite(deff) & np.isfinite(x) & (x > 0)
            goodr = np.isfinite(deff_r2) & np.isfinite(x) & (x > 0)
            if np.sum(goodd) >= 2:
                di = np.interp(np.log(np.clip(xb, EPS, None)), np.log(x[goodd]), deff[goodd], left=np.nan, right=np.nan)
            if np.sum(goodr) >= 2:
                ri = np.interp(np.log(np.clip(xb, EPS, None)), np.log(x[goodr]), deff_r2[goodr], left=np.nan, right=np.nan)
        rows = []
        for k, ell in enumerate(radii):
            rec = {
                "scale_index": int(k), "ell_B": float(ell), "ell_over_xi50": float(xb[k]),
                "NB": float(NB[k]), "NB_over_N": float(NB[k] / Pgeom.n_nodes),
                "lacunarity": float(lac[k]), "interbox_flow_fraction": float(interfrac[k]),
                "d_eff_interp": float(di[k]), "d_eff_r2_interp": float(ri[k]),
            }
            for q in COARSE_RENYI_Q:
                rec[f"coarse_eta_q{q:g}"] = float(coarse_eta[q][k])
            rows.append(rec)
        scale_df = pd.DataFrame(rows)
    else:
        scale_df = pd.DataFrame()
    return summary, scale_df


# ==================================================================================================
# 8. EXACT ENUMERATION + FACTOR-MEASURE UTILITIES
# ==================================================================================================

def factor_state_scalar(A: List[np.ndarray]) -> Dict[str,Any]:
    A=[np.asarray(M,float) for M in A]; logZ=log_partition(A,1.0); ef,nf=path_flow(A)
    elogq=sum(float(np.sum(Fm*np.log(np.clip(M,PATH_LOG_FLOOR,None)))) for Fm,M in zip(ef,A))
    return {"logZ":float(logZ),"edge_flow":ef,"node_flow":nf,"entropy":float(logZ-elogq)}


def path_hellinger_from_factors(A: List[np.ndarray], B: List[np.ndarray]) -> float:
    """Exact Hellinger distance between two normalized multiplicative path measures.

    BC = sum_pi sqrt(p_pi q_pi) = Z_sqrt / sqrt(Z_A Z_B), where the path factors of Z_sqrt
    are sqrt(A_e B_e). H^2=1-BC. This remains finite when the two measures have different support.
    """
    logZa=log_partition(A,1.0); logZb=log_partition(B,1.0)
    C=[np.sqrt(np.clip(a,0,None)*np.clip(b,0,None)) for a,b in zip(A,B)]
    logZc=log_partition(C,1.0)
    bc=float(np.exp(np.clip(logZc-0.5*(logZa+logZb),-745,0)))
    return float(math.sqrt(max(0.0,1.0-min(max(bc,0.0),1.0))))


@torch.no_grad()
def factors_single(model: GeneralMLP, x: np.ndarray, x0: np.ndarray,
                   local_direction: Optional[np.ndarray] = None) -> List[np.ndarray]:
    xt=torch.as_tensor(x[None,:],dtype=next(model.parameters()).dtype)
    x0t=torch.as_tensor(x0[None,:],dtype=next(model.parameters()).dtype)
    vt=None if local_direction is None else torch.as_tensor(local_direction[None,:],dtype=next(model.parameters()).dtype)
    Fs=conditional_factors_batch(model,xt,x0t,local_direction=vt)
    return [M[0].detach().cpu().numpy().astype(np.float64) for M in Fs]


@torch.no_grad()
def secants_for_single(model: GeneralMLP, x: np.ndarray, x0: np.ndarray):
    dtype=next(model.parameters()).dtype; xt=torch.as_tensor(x[None,:],dtype=dtype,device=DEVICE)
    x0t=torch.as_tensor(x0[None,:],dtype=dtype,device=DEVICE)
    _,z,_=model.forward_with_states(xt); _,z0,_=model.forward_with_states(x0t)
    return [finite_secant_numpy(model.activation_name,a.detach().cpu().numpy()[0],b.detach().cpu().numpy()[0]) for a,b in zip(z,z0)]


def enumerate_path_amplitudes(model: GeneralMLP, x: np.ndarray, x0: np.ndarray):
    weights=extract_weights(model); sec=secants_for_single(model,x,x0); dx=np.asarray(x,float)-np.asarray(x0,float)
    hidden=model.hidden; paths=[]; amps=[]
    for i0 in range(model.in_dim):
        for hs in itertools.product(*[range(w) for w in hidden]):
            inds=(i0,)+tuple(hs)+(0,)
            a=float(dx[i0])
            for l in range(len(weights)):
                a *= float(weights[l][inds[l+1],inds[l]])
                if l < len(sec): a *= float(sec[l][inds[l+1]])
            paths.append(inds); amps.append(a)
    return paths,np.asarray(amps,float)


def enumerate_flow_from_paths(paths: List[Tuple[int,...]], probs: np.ndarray, widths: List[int]):
    ef=[np.zeros((widths[l+1],widths[l]),float) for l in range(len(widths)-1)]
    nf=[np.zeros(widths[l],float) for l in range(len(widths))]
    for path,p in zip(paths,probs):
        for l,i in enumerate(path): nf[l][i]+=p
        for l in range(len(path)-1): ef[l][path[l+1],path[l]]+=p
    return ef,nf


def run_exact_validation() -> pd.DataFrame:
    path=OUTDIR/"tables"/"exact_validation.csv"
    if P2B_RESUME and path.exists(): return pd.read_csv(path)
    rng=np.random.default_rng(P2B_SEED+501); rows=[]
    total=len(ACTIVATIONS)*EXACT_RANDOM_STATES*EXACT_CONTRASTS_PER_STATE
    bar=tqdm(total=total,desc="Arm A exact enumeration")
    for ai,act in enumerate(ACTIVATIONS):
        for state_idx in range(EXACT_RANDOM_STATES):
            model=GeneralMLP(8,EXACT_HIDDEN,act,bias=MODEL_BIAS,seed=P2B_SEED+10000*ai+state_idx).to(DEVICE).double()
            # Random but controlled biases make the exact contrast test genuinely affine/nonlinear.
            with torch.no_grad():
                for layer in model.linears:
                    if layer.bias is not None: layer.bias.normal_(mean=0.0,std=0.12)
            structural=structural_state(model)
            for ci in range(EXACT_CONTRASTS_PER_STATE):
                x=rng.normal(size=8); x0=rng.normal(size=8)*0.5
                paths,amps=enumerate_path_amplitudes(model,x,x0); q=amps*amps; Z=float(q.sum())
                if Z<=EPS: probs=np.zeros_like(q)
                else: probs=q/Z
                dtype=next(model.parameters()).dtype
                with torch.no_grad():
                    fx=float(model(torch.as_tensor(x[None,:],dtype=dtype,device=DEVICE)).item())
                    f0=float(model(torch.as_tensor(x0[None,:],dtype=dtype,device=DEVICE)).item())
                recon=float(amps.sum()); denom=max(abs(fx-f0),1e-12)
                measure_defined=bool(Z>EPS and np.isfinite(Z))
                if measure_defined:
                    A=factors_single(model,x,x0); st=factor_state_scalar(A)
                    efe,nfe=enumerate_flow_from_paths(paths,probs,model.widths)
                    edge_err=max(float(np.max(np.abs(a-b))) for a,b in zip(efe,st["edge_flow"]))
                    node_err=max(float(np.max(np.abs(a-b))) for a,b in zip(nfe,st["node_flow"]))
                    H_enum=-float(np.sum(probs[probs>0]*np.log(probs[probs>0]))) if np.any(probs>0) else 0.0
                    partition_rel_error=abs(math.exp(st["logZ"])-Z)/max(Z,1e-300)
                    entropy_abs_error=abs(st["entropy"]-H_enum)
                else:
                    # A zero conditional partition means the positive normalized route measure is not
                    # defined for this contrast.  Signed finite-response reconstruction is still valid;
                    # measure-level validation is explicitly marked unavailable rather than epsilon-smoothed.
                    partition_rel_error=np.nan; entropy_abs_error=np.nan; edge_err=np.nan; node_err=np.nan
                rows.append({
                    "activation":act,"state_index":state_idx,"contrast_index":ci,"n_paths":len(paths),
                    "measure_defined":measure_defined,"delta_output":fx-f0,"amplitude_sum":recon,
                    "reconstruction_abs_error":abs((fx-f0)-recon),"reconstruction_rel_error":abs((fx-f0)-recon)/denom,
                    "partition_rel_error":partition_rel_error,
                    "entropy_abs_error":entropy_abs_error,"edge_flow_max_abs_error":edge_err,
                    "node_flow_max_abs_error":node_err,
                }); bar.update(1)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    bar.close(); df=pd.DataFrame(rows); atomic_csv(df,path); return df


# ==================================================================================================
# 9. PHASE-II-B ARCHITECTURE PANEL + CONTROLLED ENVIRONMENTS
# ==================================================================================================

# Extend the inherited environment dictionary without changing the original E1/E2 definitions.
ENVIRONMENTS.update({
    "CF_SMOOTH_HIGH": {
        "family": "smooth", "feature_correlation": CONTROLLED_FEATURE_CORRELATION,
        "snr": CONTROLLED_SNR_HIGH, "interaction": CONTROLLED_INTERACTION,
    },
    "CF_SMOOTH_LOW": {
        "family": "smooth", "feature_correlation": CONTROLLED_FEATURE_CORRELATION,
        "snr": CONTROLLED_SNR_LOW, "interaction": CONTROLLED_INTERACTION,
    },
    "CF_PIECE_HIGH": {
        "family": "piecewise", "feature_correlation": CONTROLLED_FEATURE_CORRELATION,
        "snr": CONTROLLED_SNR_HIGH, "interaction": CONTROLLED_INTERACTION,
    },
    "CF_PIECE_LOW": {
        "family": "piecewise", "feature_correlation": CONTROLLED_FEATURE_CORRELATION,
        "snr": CONTROLLED_SNR_LOW, "interaction": CONTROLLED_INTERACTION,
    },
})


def _parse_hidden_widths_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hidden_widths" in out.columns:
        out["hidden_widths"] = out["hidden_widths"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else [int(x) for x in s]
        )
    return out


def load_or_discover_expanded_architecture_panel() -> pd.DataFrame:
    """Load the 20-architecture expanded II-B panel; faithfully rediscover it if unavailable.

    Priority:
      1) PATHFLOW_W2_P2B_EXPANDED_DIR/tables/selected_configurations.csv
      2) a previously materialized Phase-II-B.2 architecture_panel_source.csv
      3) the exact expanded Phase-II-B RF-SMAC fallback implemented above.
    """
    materialized = OUTDIR / "tables" / "architecture_panel_source.csv"
    source = ARCH_SOURCE_DIR / "tables" / "selected_configurations.csv"

    if source.exists():
        df = _parse_hidden_widths_column(pd.read_csv(source))
        save = df.copy()
        save["hidden_widths"] = save["hidden_widths"].apply(json.dumps)
        atomic_csv(save, materialized)
        return df

    if P2B2_RESUME and materialized.exists():
        return _parse_hidden_widths_column(pd.read_csv(materialized))

    if not P2B2_ALLOW_ARCH_DISCOVERY:
        raise FileNotFoundError(
            f"Expanded Phase-II-B architecture panel not found at {source}. "
            "Set PATHFLOW_W2_P2B_EXPANDED_DIR correctly or enable "
            "PATHFLOW_W2_P2B2_ALLOW_ARCH_DISCOVERY=1."
        )

    banner("Expanded Phase-II-B architecture panel not found — running faithful RF-SMAC fallback")
    archive = run_rf_smac_discovery()
    df = select_diverse_configurations(archive)
    save = df.copy()
    save["hidden_widths"] = save["hidden_widths"].apply(json.dumps)
    atomic_csv(save, materialized)
    return df


def _design_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([candidate_features(r) for _, r in df.iterrows()])


def select_b2_architecture_panel(source: pd.DataFrame) -> pd.DataFrame:
    """Deterministic six-architecture coverage panel with morphology coverage where possible.

    We deliberately do not choose the six lowest-loss models. Two geometric extremes seed the panel:
    shallow/wide and deep/narrow. Remaining morphologies are represented by farthest candidates, then
    maximin design-space coverage fills any remaining slots.
    """
    path = OUTDIR / "tables" / "b2_selected_architectures.csv"
    if P2B2_RESUME and path.exists():
        return _parse_hidden_widths_column(pd.read_csv(path))

    src = source.copy().reset_index(drop=True)
    if len(src) < B2_ARCHITECTURES:
        raise RuntimeError(f"Need at least {B2_ARCHITECTURES} source architectures; found {len(src)}")

    Zraw = _design_feature_matrix(src)
    Z = StandardScaler().fit_transform(Zraw)
    depth = src["hidden_depth"].to_numpy(float)
    medw = src["median_width"].to_numpy(float) if "median_width" in src else np.array([
        np.median(h) for h in src["hidden_widths"]
    ], float)
    zd = (depth - depth.mean()) / max(depth.std(), EPS)
    zw = (medw - medw.mean()) / max(medw.std(), EPS)

    shallow_wide = int(np.argmin(zd - zw))
    deep_narrow = int(np.argmax(zd - zw))
    selected = []
    for i in [shallow_wide, deep_narrow]:
        if i not in selected:
            selected.append(i)

    def min_dist_to_selected(i: int) -> float:
        if not selected:
            return np.inf
        return float(np.min(np.linalg.norm(Z[i][None, :] - Z[np.asarray(selected)], axis=1)))

    # Explicitly fill missing morphology classes when the source panel contains them.
    available_morphs = [m for m in MORPHOLOGIES if np.any(src.morphology.astype(str) == m)]
    for morph in available_morphs:
        if len(selected) >= B2_ARCHITECTURES:
            break
        if any(str(src.iloc[i].morphology) == morph for i in selected):
            continue
        candidates_idx = [i for i in range(len(src)) if str(src.iloc[i].morphology) == morph and i not in selected]
        if candidates_idx:
            selected.append(max(candidates_idx, key=min_dist_to_selected))

    while len(selected) < B2_ARCHITECTURES:
        candidates_idx = [i for i in range(len(src)) if i not in selected]
        selected.append(max(candidates_idx, key=min_dist_to_selected))

    panel = src.iloc[selected[:B2_ARCHITECTURES]].copy().reset_index(drop=True)
    panel["b2_panel_rank"] = np.arange(len(panel))
    # Preserve the Phase-II-B configuration ID; add an explicit II-B.2 alias for traceability.
    panel["b2_config_id"] = [f"B2A{k:02d}" for k in range(len(panel))]
    save = panel.copy()
    save["hidden_widths"] = save["hidden_widths"].apply(json.dumps)
    atomic_csv(save, path)
    return panel


def select_depth_spanning_architectures(panel: pd.DataFrame, n: int) -> pd.DataFrame:
    p = panel.sort_values(["hidden_depth", "realized_parameters"]).reset_index(drop=True)
    if len(p) <= n:
        return p.copy()
    ids = np.unique(np.round(np.linspace(0, len(p) - 1, n)).astype(int))
    if len(ids) < n:
        remaining = [i for i in range(len(p)) if i not in ids]
        ids = np.concatenate([ids, remaining[:n-len(ids)]])
    return p.iloc[ids[:n]].copy().reset_index(drop=True)


# ==================================================================================================
# 10. OPTIMIZER FACTORY + GLOBAL LEARNING-RATE CALIBRATION
# ==================================================================================================


def make_optimizer(name: str, parameters, lr: float):
    name = str(name).lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=float(lr), momentum=0.0, weight_decay=WEIGHT_DECAY)
    if name == "sgdm":
        return torch.optim.SGD(parameters, lr=float(lr), momentum=0.9, weight_decay=WEIGHT_DECAY)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=float(lr), alpha=0.99, momentum=0.0,
                                   weight_decay=WEIGHT_DECAY)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=float(lr), betas=(0.9, 0.999),
                                weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unknown optimizer: {name}")


def generate_optimizer_calibration_environment():
    """Discovery-only mixed environment with independent data seed and reduced sample count."""
    cfg = {"family": "mixed", "feature_correlation": 0.35, "snr": 15.0, "interaction": 0.50}
    ntr, nva = CALIBRATION_N_TRAIN, CALIBRATION_N_VAL
    seed = int(DATA_BASE_SEED + 919_271)
    rng = np.random.default_rng(seed)
    X = correlated_gaussian(rng, ntr + nva, INPUT_DIM, cfg["feature_correlation"])
    xm = X[:ntr].mean(0, keepdims=True); xs = X[:ntr].std(0, keepdims=True) + 1e-8
    X = ((X - xm) / xs).astype(np.float32)
    clean = deterministic_target(X, cfg)
    clean = ((clean - clean[:ntr].mean()) / (clean[:ntr].std() + 1e-8)).astype(np.float32)
    noise_sd = math.sqrt(1.0 / max(cfg["snr"], EPS))
    y = clean + rng.normal(0, noise_sd, size=clean.shape).astype(np.float32)
    y = ((y - y[:ntr].mean()) / (y[:ntr].std() + 1e-8)).astype(np.float32)
    return {
        "train": (torch.from_numpy(X[:ntr]), torch.from_numpy(y[:ntr])),
        "val": (torch.from_numpy(X[ntr:]), torch.from_numpy(y[ntr:])),
    }


def _calibration_initial_seed(config_id: str, seed_index: int) -> int:
    return model_seed(seed_index) + int(stable_hash_dict({"config": str(config_id)}, 8), 16) % 1_000_003


def run_one_optimizer_calibration(arch_row: pd.Series, activation: str, optimizer_name: str,
                                  lr: float, seed_index: int, splits: Dict[str, Any]) -> Dict[str, Any]:
    hidden = arch_row.hidden_widths if not isinstance(arch_row.hidden_widths, str) else json.loads(arch_row.hidden_widths)
    cid = str(getattr(arch_row, "config_id", getattr(arch_row, "b2_config_id", "ARCH")))
    initseed = _calibration_initial_seed(cid, int(seed_index))
    model = GeneralMLP(INPUT_DIM, hidden, activation, bias=MODEL_BIAS, seed=initseed).to(DEVICE)
    Xtr, ytr = splits["train"]; Xv, yv = splits["val"]
    scales = calibrate_initial_signal(model, Xtr[:min(512, len(Xtr))])
    opt = make_optimizer(optimizer_name, model.parameters(), lr)

    initial_val = eval_mse(model, Xv, yv)
    vals = [float(initial_val)]
    best_val = float(initial_val)
    diverged = False
    for epoch in range(1, CALIBRATION_EPOCHS + 1):
        model.train()
        for ids in deterministic_batches(len(Xtr), epoch-1, int(seed_index), BATCH_SIZE):
            xb = Xtr[ids].to(DEVICE); yb = ytr[ids].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward(); opt.step()
        if diverged:
            break
        val = eval_mse(model, Xv, yv)
        vals.append(float(val)); best_val = min(best_val, float(val))
        if not np.isfinite(val) or val > 1e8:
            diverged = True
            break

    if diverged or len(vals) < 2:
        norm_auc = 1e6
        final_val = np.inf
    else:
        normalized = np.asarray(vals, float) / max(float(initial_val), EPS)
        # Mean area under validation-loss curve; smaller is better.
        norm_auc = float(np.trapezoid(normalized, dx=1.0) / max(len(normalized)-1, 1))
        final_val = float(vals[-1])

    rec = {
        "config_id": cid,
        "hidden_depth": int(arch_row.hidden_depth),
        "activation": activation,
        "optimizer": optimizer_name,
        "learning_rate": float(lr),
        "seed_index": int(seed_index),
        "initial_val_mse": float(initial_val),
        "best_val_mse": float(best_val),
        "final_val_mse": float(final_val),
        "normalized_val_auc": float(norm_auc),
        "diverged": bool(diverged),
        "calibration_scales_json": json.dumps([float(x) for x in scales]),
    }
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def run_optimizer_calibration(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    runs_path = OUTDIR / "tables" / "optimizer_calibration_runs.csv"
    summary_path = OUTDIR / "tables" / "optimizer_calibration_summary.csv"
    lr_path = OUTDIR / "tables" / "optimizer_learning_rates.csv"

    if P2B2_RESUME and lr_path.exists() and summary_path.exists() and runs_path.exists():
        lrdf = pd.read_csv(lr_path)
        return pd.read_csv(runs_path), pd.read_csv(summary_path), {
            str(r.optimizer): float(r.learning_rate) for r in lrdf.itertuples()
        }

    archs = select_depth_spanning_architectures(panel, min(CALIBRATION_ARCHITECTURES, len(panel)))
    splits = generate_optimizer_calibration_environment()
    existing = pd.read_csv(runs_path) if (P2B2_RESUME and runs_path.exists()) else pd.DataFrame()
    rows = [] if existing.empty else existing.to_dict("records")
    done = set()
    for r in rows:
        done.add((str(r["config_id"]), str(r["activation"]), str(r["optimizer"]),
                  float(r["learning_rate"]), int(r["seed_index"])))

    tasks = []
    for opt_name in OPTIMIZERS:
        for lr in OPTIMIZER_LR_GRIDS[opt_name]:
            for _, arow in archs.iterrows():
                cid = str(arow.config_id)
                for act in CALIBRATION_ACTIVATIONS:
                    for seed_idx in CALIBRATION_SEEDS:
                        key = (cid, act, opt_name, float(lr), int(seed_idx))
                        if key not in done:
                            tasks.append((arow, act, opt_name, float(lr), int(seed_idx)))

    eta = ETAReporter("optimizer LR calibration", len(tasks))
    for k, (arow, act, opt_name, lr, seed_idx) in enumerate(tqdm(tasks, desc="II-B.2 optimizer calibration"), 1):
        t0 = time.perf_counter()
        rec = run_one_optimizer_calibration(arow, act, opt_name, lr, seed_idx, splits)
        rows.append(rec)
        atomic_csv(pd.DataFrame(rows), runs_path)
        eta.update(time.perf_counter() - t0, k)
    eta.finish()

    rdf = pd.DataFrame(rows)
    agg_rows = []
    for (opt_name, lr), g in rdf.groupby(["optimizer", "learning_rate"]):
        diverged_any = bool(g["diverged"].astype(bool).any())
        worst_auc = float(g["normalized_val_auc"].max()) if len(g) else np.inf
        objective = 1e6 + worst_auc if diverged_any else worst_auc
        agg_rows.append({
            "optimizer": str(opt_name), "learning_rate": float(lr),
            "n_anchor_runs": int(len(g)), "diverged_any": diverged_any,
            "worst_normalized_val_auc": worst_auc,
            "median_normalized_val_auc": float(g["normalized_val_auc"].median()),
            "median_best_val_mse": float(g["best_val_mse"].median()),
            "selection_objective": float(objective),
        })
    sdf = pd.DataFrame(agg_rows).sort_values(["optimizer", "learning_rate"]).reset_index(drop=True)
    atomic_csv(sdf, summary_path)

    chosen = []
    for opt_name in OPTIMIZERS:
        sub = sdf[sdf.optimizer == opt_name].sort_values(
            ["selection_objective", "median_best_val_mse", "learning_rate"]
        )
        if sub.empty:
            raise RuntimeError(f"No calibration result for optimizer {opt_name}")
        chosen.append(sub.iloc[0][["optimizer", "learning_rate", "selection_objective",
                                   "median_best_val_mse", "diverged_any"]].to_dict())
    lrdf = pd.DataFrame(chosen)
    atomic_csv(lrdf, lr_path)
    lr_map = {str(r.optimizer): float(r.learning_rate) for r in lrdf.itertuples()}
    return rdf, sdf, lr_map
# ==================================================================================================
# 11. LONGITUDINAL CONDITIONAL-TRACE COMPARISONS
# ==================================================================================================


def clone_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_reference_model(hidden: Sequence[int], activation: str, seed: int) -> GeneralMLP:
    # Parameters are immediately overwritten by a stored state; seed only completes construction.
    return GeneralMLP(INPUT_DIM, hidden, activation, bias=MODEL_BIAS, seed=seed).to(DEVICE)


@torch.no_grad()
def conditional_path_hellinger_models(model_a: GeneralMLP, model_b: GeneralMLP,
                                      X: torch.Tensor, X0: torch.Tensor,
                                      batch_size: int = CONDITIONAL_BATCH) -> np.ndarray:
    """Exact Hellinger distance between two conditional complete-path measures, input by input.

    If A_l and B_l are the two models' nonnegative multiplicative path-factor matrices, then

        BC = Z_{sqrt(A B)} / sqrt(Z_A Z_B),   H^2 = 1-BC.

    No complete-path enumeration is required.
    """
    vals = []
    n = len(X)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        xa = X[lo:hi]; x0a = X0[lo:hi]
        Fa = conditional_factors_batch(model_a, xa, x0a)
        Fb = conditional_factors_batch(model_b, xa, x0a)
        va=_batched_support_exists(Fa); vb=_batched_support_exists(Fb); valid=va & vb
        logZa = _batched_log_partition(Fa, 1.0)
        logZb = _batched_log_partition(Fb, 1.0)
        Fc = [torch.sqrt(torch.clamp(a, min=0.0) * torch.clamp(b, min=0.0)) for a, b in zip(Fa, Fb)]
        logZc = _batched_log_partition(Fc, 1.0)
        bc = torch.exp(torch.clamp(logZc - 0.5 * (logZa + logZb), min=-745.0, max=0.0))
        h = torch.sqrt(torch.clamp(1.0 - bc, min=0.0, max=1.0))
        h=torch.where(valid,h,torch.full_like(h,float("nan")))
        vals.append(h.detach().cpu().numpy())
    return np.concatenate(vals) if vals else np.array([], float)


@torch.no_grad()
def _secant_signature_batch(model: GeneralMLP, X: torch.Tensor, X0: torch.Tensor) -> List[np.ndarray]:
    model.eval()
    Xd = X.to(DEVICE); X0d = X0.to(DEVICE)
    _, z, _ = model.forward_with_states(Xd)
    _, z0, _ = model.forward_with_states(X0d)
    out = []
    for a, b in zip(z, z0):
        s = _finite_secant_torch(model.activation_name, a, b)
        out.append(s.detach().cpu().numpy().astype(np.float64))
    return out


def response_transition_metrics(model_prev: GeneralMLP, model_curr: GeneralMLP,
                                X: torch.Tensor, X0: torch.Tensor,
                                batch_size: int = CONDITIONAL_BATCH) -> Dict[str, float]:
    """Compare activation-response factors between two training checkpoints.

    All activations receive a continuous log-squared-response drift measure. ReLU additionally receives
    true zero/nonzero support turnover. Leaky ReLU receives a coarse response-regime turnover using the
    midpoint between slopes 0.1 and 1.0; continuous drift remains the primary Leaky-ReLU quantity.
    """
    drift_sum = 0.0; drift_n = 0
    support_switch = 0; support_n = 0
    regime_switch = 0; regime_n = 0
    abs_change_sum = 0.0; abs_change_n = 0

    for lo in range(0, len(X), batch_size):
        hi = min(lo + batch_size, len(X))
        sp = _secant_signature_batch(model_prev, X[lo:hi], X0[lo:hi])
        sc = _secant_signature_batch(model_curr, X[lo:hi], X0[lo:hi])
        for a, b in zip(sp, sc):
            la = np.log(np.square(a) + EPS)
            lb = np.log(np.square(b) + EPS)
            d = np.abs(lb - la)
            drift_sum += float(np.sum(d)); drift_n += int(d.size)
            ad = np.abs(b - a)
            abs_change_sum += float(np.sum(ad)); abs_change_n += int(ad.size)
            if model_curr.activation_name == "relu":
                pa = np.abs(a) > RESPONSE_SUPPORT_TOL
                pb = np.abs(b) > RESPONSE_SUPPORT_TOL
                support_switch += int(np.sum(pa != pb)); support_n += int(pa.size)
            elif model_curr.activation_name == "leaky_relu":
                ra = a > LEAKY_REGIME_MIDPOINT
                rb = b > LEAKY_REGIME_MIDPOINT
                regime_switch += int(np.sum(ra != rb)); regime_n += int(ra.size)

    return {
        "response_logsq_drift_mean": float(drift_sum / max(drift_n, 1)),
        "response_abs_secant_change_mean": float(abs_change_sum / max(abs_change_n, 1)),
        "relu_support_turnover_fraction": float(support_switch / support_n) if support_n else np.nan,
        "leaky_regime_turnover_fraction": float(regime_switch / regime_n) if regime_n else np.nan,
    }


def summarize_array(x: Sequence[float], prefix: str) -> Dict[str, float]:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan,
                f"{prefix}_q25": np.nan, f"{prefix}_q75": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_q25": float(np.quantile(a, 0.25)),
        f"{prefix}_q75": float(np.quantile(a, 0.75)),
    }


def conditional_checkpoint_summary(model: GeneralMLP, Xeval: torch.Tensor,
                                   structural: Optional[Dict[str, Any]] = None,
                                   keep_input_rows: bool = False) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Dict[str, Any]]:
    """Exact conditional path statistics on the supplied fixed input panel."""
    if structural is None:
        structural = structural_state(model)
    Xeval = Xeval[:min(len(Xeval), ENDPOINT_INPUTS)]
    cond = aggregate_conditional_evaluation(model, Xeval, structural, reference=None, keep_per_input_flows=False)
    summary = dict(cond["summary"])
    # Explicit hidden-only specialization: do not dilute it with input/scalar-output layers.
    spec = np.asarray(cond["specialization"], float)
    hidden_spec = spec[1:-1] if len(spec) > 2 else np.array([], float)
    summary["input_specialization_hidden_mean"] = float(np.mean(hidden_spec)) if len(hidden_spec) else np.nan
    summary["input_specialization_hidden_max"] = float(np.max(hidden_spec)) if len(hidden_spec) else np.nan
    summary["input_specialization_profile_json"] = json.dumps([float(v) for v in spec])
    inputs = cond["input_df"].copy() if keep_input_rows else None
    return summary, inputs, cond["population_state"]


def structural_scalar_summary(st: Dict[str, Any]) -> Dict[str, float]:
    return {
        "structural_S_path": float(st["S_path"]),
        "structural_entropy_deficit": float(st["entropy_deficit"]),
        "structural_eta1": float(st["eta1"]),
        "structural_eta2": float(st["eta2"]),
        "structural_edge_cv": float(st["edge_cv"]),
        "structural_node_cv": float(st["node_cv"]),
    }


def geometry_from_population_state(pop_state: Dict[str, Any], structural: Dict[str, Any],
                                   meta: Dict[str, Any], checkpoint_label: str,
                                   n_eval_inputs: int, include_structural: bool = True,
                                   n_scales: int = BCANW_SCALES) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []; scales = []
    if pop_state is not None and bool(pop_state.get("defined",True)) and "edge_flow" in pop_state:
        cs, cscale = analyze_geometry([], pop_state, n_scales=n_scales, compute_local=True)
        crow = {**meta, "checkpoint": checkpoint_label, "geometry_source": "conditional_population",
                "evaluation_inputs": int(n_eval_inputs), **cs}
        rows.append(crow)
        if not cscale.empty:
            cscale = cscale.copy()
            for k, v in meta.items(): cscale[k] = v
            cscale["checkpoint"] = checkpoint_label
            cscale["geometry_source"] = "conditional_population"
            cscale["evaluation_inputs"] = int(n_eval_inputs)
            scales.append(cscale)

    if include_structural:
        ss, sscale = analyze_geometry([], structural, n_scales=n_scales, compute_local=True)
        srow = {**meta, "checkpoint": checkpoint_label, "geometry_source": "structural",
                "evaluation_inputs": int(n_eval_inputs), **ss}
        rows.append(srow)
        if not sscale.empty:
            sscale = sscale.copy()
            for k, v in meta.items(): sscale[k] = v
            sscale["checkpoint"] = checkpoint_label
            sscale["geometry_source"] = "structural"
            sscale["evaluation_inputs"] = int(n_eval_inputs)
            scales.append(sscale)
    return pd.DataFrame(rows), pd.concat(scales, ignore_index=True) if scales else pd.DataFrame()
# ==================================================================================================
# 12. RUN MANIFESTS, RESUMABLE PARTS AND 280-EPOCH TRAINING
# ==================================================================================================


def parameter_digest(model: nn.Module) -> str:
    h = hashlib.sha1()
    for name, p in model.state_dict().items():
        h.update(name.encode("utf-8")); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def build_main_manifest(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in panel.iterrows():
        hidden = r.hidden_widths if not isinstance(r.hidden_widths, str) else json.loads(r.hidden_widths)
        cid = str(r.config_id); bid = str(r.b2_config_id)
        for env_id in MAIN_ENVIRONMENTS:
            for seed_idx in MAIN_SEEDS:
                for act in ACTIVATIONS:
                    for opt_name in OPTIMIZERS:
                        rid = f"CORE__{bid}__{env_id}__S{seed_idx:02d}__{act}__{opt_name}"
                        rows.append({
                            "study_arm": "core", "run_id": rid,
                            "config_id": cid, "b2_config_id": bid,
                            "environment_id": env_id, "seed_index": int(seed_idx),
                            "activation": act, "optimizer": opt_name,
                            "hidden_widths": json.dumps(hidden),
                            "hidden_depth": int(r.hidden_depth),
                            "realized_parameters": int(r.realized_parameters),
                            "morphology": str(r.morphology),
                            "shape_strength": float(r.shape_strength),
                            "shape_curvature": float(r.shape_curvature),
                            "longitudinal": True,
                        })
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "main_run_manifest.csv")
    return df


def build_controlled_environment_manifest(panel: pd.DataFrame) -> pd.DataFrame:
    if not P2B2_RUN_CONTROLLED_ENV:
        df = pd.DataFrame()
        atomic_csv(df, OUTDIR / "tables" / "controlled_environment_manifest.csv")
        return df
    archs = select_depth_spanning_architectures(panel, min(CONTROLLED_ENV_ARCHITECTURES, len(panel)))
    rows = []
    for _, r in archs.iterrows():
        hidden = r.hidden_widths if not isinstance(r.hidden_widths, str) else json.loads(r.hidden_widths)
        cid = str(r.config_id); bid = str(r.b2_config_id)
        for env_id in CONTROLLED_ENVIRONMENTS:
            for seed_idx in MAIN_SEEDS:
                for act in ACTIVATIONS:
                    rid = f"CENV__{bid}__{env_id}__S{seed_idx:02d}__{act}__adam"
                    rows.append({
                        "study_arm": "controlled_environment", "run_id": rid,
                        "config_id": cid, "b2_config_id": bid,
                        "environment_id": env_id, "seed_index": int(seed_idx),
                        "activation": act, "optimizer": "adam",
                        "hidden_widths": json.dumps(hidden),
                        "hidden_depth": int(r.hidden_depth),
                        "realized_parameters": int(r.realized_parameters),
                        "morphology": str(r.morphology),
                        "shape_strength": float(r.shape_strength),
                        "shape_curvature": float(r.shape_curvature),
                        "longitudinal": False,
                    })
    df = pd.DataFrame(rows)
    atomic_csv(df, OUTDIR / "tables" / "controlled_environment_manifest.csv")
    return df


def _part_path(run_id: str, kind: str) -> Path:
    return OUTDIR / "run_parts" / f"{run_id}__{kind}.csv"


def _best_checkpoint_path(run_id: str) -> Path:
    return OUTDIR / "checkpoints" / f"{run_id}__best.pt"


def run_complete(run_id: str) -> bool:
    # A run is complete only after all endpoint/geometry artifacts and the best checkpoint exist.
    # Trajectory CSV may be empty for the controlled-environment arm, so it is not required here.
    required = ["summary", "endpoint_inputs", "history", "geometry", "geometry_scales"]
    return all(_part_path(run_id, k).exists() for k in required) and _best_checkpoint_path(run_id).exists()


def _add_meta(df: pd.DataFrame, meta: Dict[str, Any], checkpoint: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    for k, v in meta.items(): out[k] = v
    if checkpoint is not None: out["checkpoint"] = checkpoint
    return out


def _trajectory_row(model: GeneralMLP, Xtraj: torch.Tensor, meta: Dict[str, Any], t: int,
                    prev_model: Optional[GeneralMLP], init_model: Optional[GeneralMLP],
                    X0: torch.Tensor) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    structural = structural_state(model)
    cond_summary, _, pop_state = conditional_checkpoint_summary(model, Xtraj, structural, keep_input_rows=False)
    row = {**meta, "epoch": int(t), **structural_scalar_summary(structural), **cond_summary}

    if prev_model is None:
        row.update({
            "path_hellinger_from_previous_mean": 0.0,
            "path_hellinger_from_previous_median": 0.0,
            "path_hellinger_from_previous_q25": 0.0,
            "path_hellinger_from_previous_q75": 0.0,
            "path_hellinger_from_init_mean": 0.0,
            "path_hellinger_from_init_median": 0.0,
            "path_hellinger_from_init_q25": 0.0,
            "path_hellinger_from_init_q75": 0.0,
            "response_logsq_drift_mean": 0.0,
            "response_abs_secant_change_mean": 0.0,
            "relu_support_turnover_fraction": 0.0 if model.activation_name == "relu" else np.nan,
            "leaky_regime_turnover_fraction": 0.0 if model.activation_name == "leaky_relu" else np.nan,
        })
    else:
        hp = conditional_path_hellinger_models(prev_model, model, Xtraj, X0)
        hi = conditional_path_hellinger_models(init_model, model, Xtraj, X0)
        row.update(summarize_array(hp, "path_hellinger_from_previous"))
        row.update(summarize_array(hi, "path_hellinger_from_init"))
        row.update(response_transition_metrics(prev_model, model, Xtraj, X0))
    return row, structural, pop_state


def _endpoint_evaluation(model: GeneralMLP, Xend: torch.Tensor, meta: Dict[str, Any], label: str,
                         compute_geometry: bool = True) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    structural = structural_state(model)
    csum, inputs, pop = conditional_checkpoint_summary(model, Xend, structural, keep_input_rows=True)
    scalar = {**structural_scalar_summary(structural), **csum}
    idf = _add_meta(inputs, meta, checkpoint=label)
    idf.insert(0, "input_index", np.arange(len(idf)))
    if compute_geometry:
        gdf, sdf = geometry_from_population_state(pop, structural, meta, label, len(Xend), include_structural=True)
    else:
        gdf, sdf = pd.DataFrame(), pd.DataFrame()
    return scalar, idf, gdf, sdf


def train_one_phase2b2_run(row: pd.Series, splits: Dict[str, Any], lr_map: Dict[str, float]):
    run_id = str(row.run_id)
    if P2B2_RESUME and run_complete(run_id):
        return

    hidden = json.loads(row.hidden_widths) if isinstance(row.hidden_widths, str) else list(row.hidden_widths)
    seed_idx = int(row.seed_index); act = str(row.activation); opt_name = str(row.optimizer)
    lr = float(lr_map[opt_name])
    initseed = _calibration_initial_seed(str(row.config_id), seed_idx)
    model = GeneralMLP(INPUT_DIM, hidden, act, bias=MODEL_BIAS, seed=initseed).to(DEVICE)
    Xtr, ytr = splits["train"]; Xv, yv = splits["val"]; Xt, yt = splits["test"]
    calibration_scales = calibrate_initial_signal(model, Xtr[:min(512, len(Xtr))])
    initial_digest = parameter_digest(model)
    init_state = clone_state_dict_cpu(model)

    meta = {
        "study_arm": str(row.study_arm), "run_id": run_id,
        "config_id": str(row.config_id), "b2_config_id": str(row.b2_config_id),
        "environment_id": str(row.environment_id), "seed_index": seed_idx,
        "activation": act, "optimizer": opt_name, "learning_rate": lr,
        "hidden_depth": int(row.hidden_depth), "realized_parameters": int(row.realized_parameters),
        "morphology": str(row.morphology),
    }
    longitudinal = bool(row.longitudinal)
    Xtraj = Xt[:min(TRAJECTORY_INPUTS, len(Xt))]
    Xtraj0 = torch.zeros_like(Xtraj)
    Xend = Xt[:min(ENDPOINT_INPUTS, len(Xt))]

    endpoint_rows = []
    endpoint_input_frames = []
    trajectory_rows = []
    geometry_frames = []
    geometry_scale_frames = []
    history_rows = []

    # Full initialization endpoint.
    init_scalar, init_inputs, init_geom, init_scales = _endpoint_evaluation(
        model, Xend, meta, "init", compute_geometry=not longitudinal
    )
    endpoint_rows.append({"checkpoint": "init", **init_scalar})
    endpoint_input_frames.append(init_inputs)
    if not init_geom.empty: geometry_frames.append(init_geom)
    if not init_scales.empty: geometry_scale_frames.append(init_scales)

    initial_val = eval_mse(model, Xv, yv)
    initial_test = eval_mse(model, Xt, yt)
    history_rows.append({**meta, "epoch": 0, "train_mse": np.nan,
                         "val_mse": float(initial_val), "test_mse_if_evaluated": float(initial_test)})

    # Reference models hold exact prior/init states for path-space comparisons.
    prev_model = None
    init_model = None
    prev_state = clone_state_dict_cpu(model)
    if longitudinal:
        init_model = make_reference_model(hidden, act, initseed + 17)
        init_model.load_state_dict(init_state); init_model.eval()
        tr0, st0, pop0 = _trajectory_row(model, Xtraj, meta, 0, None, init_model, Xtraj0)
        trajectory_rows.append(tr0)
        if P2B2_RUN_TRAJECTORY_GEOM and 0 in TRAJECTORY_GEOMETRY_CHECKPOINTS:
            g, s = geometry_from_population_state(pop0, st0, meta, "t000", len(Xtraj), include_structural=True)
            geometry_frames.append(g); geometry_scale_frames.append(s)
        prev_model = make_reference_model(hidden, act, initseed + 19)
        prev_model.load_state_dict(prev_state); prev_model.eval()

    opt = make_optimizer(opt_name, model.parameters(), lr)
    best_val = float(initial_val); best_epoch = 0; best_state = clone_state_dict_cpu(model)
    diverged = False

    for epoch in range(1, MAX_EPOCH + 1):
        model.train(); losses = []
        for ids in deterministic_batches(len(Xtr), epoch-1, seed_idx, BATCH_SIZE):
            xb = Xtr[ids].to(DEVICE); yb = ytr[ids].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb); loss = F.mse_loss(pred, yb)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward(); opt.step(); losses.append(float(loss.item()))
        if diverged:
            history_rows.append({**meta, "epoch": epoch,
                                 "train_mse": float(np.mean(losses)) if losses else np.nan,
                                 "val_mse": np.inf, "test_mse_if_evaluated": np.nan})
            break

        val = eval_mse(model, Xv, yv)
        history_rows.append({**meta, "epoch": epoch,
                             "train_mse": float(np.mean(losses)) if losses else np.nan,
                             "val_mse": float(val), "test_mse_if_evaluated": np.nan})
        if np.isfinite(val) and val < best_val:
            best_val = float(val); best_epoch = int(epoch); best_state = clone_state_dict_cpu(model)

        if longitudinal and epoch in TRAJECTORY_CHECKPOINTS:
            # prev_model still contains the immediately preceding scheduled state.
            tr, st, pop = _trajectory_row(model, Xtraj, meta, epoch, prev_model, init_model, Xtraj0)
            trajectory_rows.append(tr)
            if P2B2_RUN_TRAJECTORY_GEOM and epoch in TRAJECTORY_GEOMETRY_CHECKPOINTS:
                g, s = geometry_from_population_state(pop, st, meta, f"t{epoch:03d}", len(Xtraj), include_structural=True)
                geometry_frames.append(g); geometry_scale_frames.append(s)
            prev_state = clone_state_dict_cpu(model)
            prev_model.load_state_dict(prev_state); prev_model.eval()

        if not np.isfinite(val) or val > 1e12:
            diverged = True
            break

    final_epoch = int(history_rows[-1]["epoch"])
    final_state = clone_state_dict_cpu(model)
    final_val = eval_mse(model, Xv, yv) if not diverged else np.inf
    final_test = eval_mse(model, Xt, yt) if not diverged else np.inf

    # Full final endpoint; do not confuse it with the 64-input t=280 trajectory geometry.
    if not diverged:
        final_scalar, final_inputs, final_geom, final_scales = _endpoint_evaluation(
            model, Xend, meta, "final", compute_geometry=True
        )
        endpoint_rows.append({"checkpoint": "final", **final_scalar})
        endpoint_input_frames.append(final_inputs)
        if not final_geom.empty: geometry_frames.append(final_geom)
        if not final_scales.empty: geometry_scale_frames.append(final_scales)

    # Full best-validation endpoint and best geometry.
    model.load_state_dict(best_state); model.to(DEVICE); model.eval()
    best_test = eval_mse(model, Xt, yt)
    best_scalar, best_inputs, best_geom, best_scales = _endpoint_evaluation(
        model, Xend, meta, "best", compute_geometry=True
    )
    endpoint_rows.append({"checkpoint": "best", **best_scalar})
    endpoint_input_frames.append(best_inputs)
    if not best_geom.empty: geometry_frames.append(best_geom)
    if not best_scales.empty: geometry_scale_frames.append(best_scales)

    torch.save({
        "state_dict": best_state, "best_epoch": int(best_epoch), "best_val_mse": float(best_val),
        "optimizer": opt_name, "learning_rate": lr, "activation": act,
        "config_id": str(row.config_id), "environment_id": str(row.environment_id),
        "seed_index": seed_idx,
    }, _best_checkpoint_path(run_id))

    endpoint_df = pd.DataFrame(endpoint_rows)
    summary = {
        **meta, "profile": P2B2_PROFILE, "max_epoch_requested": int(MAX_EPOCH),
        "final_epoch_reached": final_epoch, "diverged": bool(diverged),
        "initial_parameter_digest": initial_digest,
        "calibration_scales_json": json.dumps([float(x) for x in calibration_scales]),
        "initial_val_mse": float(initial_val), "initial_test_mse": float(initial_test),
        "best_epoch": int(best_epoch), "best_val_mse": float(best_val), "best_test_mse": float(best_test),
        "final_val_mse": float(final_val), "final_test_mse": float(final_test),
    }
    for _, er in endpoint_df.iterrows():
        label = str(er["checkpoint"])
        for k, v in er.items():
            if k == "checkpoint": continue
            summary[f"{label}_{k}"] = v

    atomic_csv(pd.DataFrame([summary]), _part_path(run_id, "summary"))
    atomic_csv(pd.concat(endpoint_input_frames, ignore_index=True), _part_path(run_id, "endpoint_inputs"))
    atomic_csv(pd.DataFrame(history_rows), _part_path(run_id, "history"))
    atomic_csv(pd.DataFrame(trajectory_rows), _part_path(run_id, "trajectory"))
    atomic_csv(pd.concat(geometry_frames, ignore_index=True) if geometry_frames else pd.DataFrame(),
               _part_path(run_id, "geometry"))
    atomic_csv(pd.concat(geometry_scale_frames, ignore_index=True) if geometry_scale_frames else pd.DataFrame(),
               _part_path(run_id, "geometry_scales"))

    del model, opt
    if prev_model is not None: del prev_model
    if init_model is not None: del init_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_manifest(manifest: pd.DataFrame, env_cache: Dict[str, Any], lr_map: Dict[str, float], stage_name: str):
    if manifest.empty:
        return
    todo = [r for _, r in manifest.iterrows() if not (P2B2_RESUME and run_complete(str(r.run_id)))]
    eta = ETAReporter(stage_name, len(todo))
    for k, r in enumerate(tqdm(todo, desc=stage_name), 1):
        t0 = time.perf_counter()
        train_one_phase2b2_run(r, env_cache[str(r.environment_id)][0], lr_map)
        eta.update(time.perf_counter() - t0, k)
    eta.finish()
# ==================================================================================================
# 13. AGGREGATION, FUNCTIONAL-PROGRESS COORDINATE AND LATE-TRAINING ANALYSIS
# ==================================================================================================


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def aggregate_run_parts(all_manifest: pd.DataFrame):
    kinds = {
        "summary": [], "endpoint_inputs": [], "history": [], "trajectory": [],
        "geometry": [], "geometry_scales": []
    }
    for rid in all_manifest.run_id.astype(str):
        for kind in kinds:
            d = _safe_read_csv(_part_path(rid, kind))
            if not d.empty:
                kinds[kind].append(d)

    summary = pd.concat(kinds["summary"], ignore_index=True) if kinds["summary"] else pd.DataFrame()
    endpoints = pd.concat(kinds["endpoint_inputs"], ignore_index=True) if kinds["endpoint_inputs"] else pd.DataFrame()
    history = pd.concat(kinds["history"], ignore_index=True) if kinds["history"] else pd.DataFrame()
    trajectory = pd.concat(kinds["trajectory"], ignore_index=True) if kinds["trajectory"] else pd.DataFrame()
    geometry = pd.concat(kinds["geometry"], ignore_index=True) if kinds["geometry"] else pd.DataFrame()
    scales = pd.concat(kinds["geometry_scales"], ignore_index=True) if kinds["geometry_scales"] else pd.DataFrame()

    atomic_csv(summary, OUTDIR / "tables" / "run_summary.csv")
    atomic_csv(endpoints, OUTDIR / "tables" / "endpoint_input_summary.csv")
    atomic_csv(history, OUTDIR / "tables" / "training_history.csv")
    atomic_csv(trajectory, OUTDIR / "tables" / "trajectory_summary_raw.csv")
    atomic_csv(geometry, OUTDIR / "tables" / "trajectory_geometry.csv")
    atomic_csv(scales, OUTDIR / "tables" / "trajectory_geometry_scales.csv")
    return summary, endpoints, history, trajectory, geometry, scales


def _history_progress(g: pd.DataFrame) -> pd.DataFrame:
    h = g.sort_values("epoch").copy()
    v = h["val_mse"].to_numpy(float)
    finite = np.isfinite(v)
    if not np.any(finite):
        h["functional_progress"] = np.nan
        return h
    # Running best ignores nonfinite tail values after divergence.
    vf = np.where(finite, v, np.inf)
    running = np.minimum.accumulate(vf)
    L0 = float(vf[0])
    Lbest = float(np.min(vf))
    denom = L0 - Lbest
    if not np.isfinite(denom) or denom <= EPS:
        progress = np.zeros_like(running)
    else:
        progress = np.clip((L0 - running) / denom, 0.0, 1.0)
    progress[~np.isfinite(running)] = np.nan
    h["running_best_val_mse"] = running
    h["functional_progress"] = progress
    return h


def build_functional_progress_tables(history: pd.DataFrame, trajectory: pd.DataFrame):
    traj_path = OUTDIR / "tables" / "trajectory_summary.csv"
    fp_path = OUTDIR / "tables" / "functional_progress_summary.csv"
    late_path = OUTDIR / "tables" / "late_training_summary.csv"
    if history.empty or trajectory.empty:
        atomic_csv(trajectory, traj_path); atomic_csv(pd.DataFrame(), fp_path); atomic_csv(pd.DataFrame(), late_path)
        return trajectory, pd.DataFrame(), pd.DataFrame()

    hparts = []
    for rid, g in history.groupby("run_id"):
        hparts.append(_history_progress(g))
    hp = pd.concat(hparts, ignore_index=True)

    # Attach exact running-best progress to each scheduled trajectory checkpoint.
    tr = trajectory.copy()
    progress_lookup = hp[["run_id", "epoch", "functional_progress", "running_best_val_mse"]].copy()
    tr = tr.merge(progress_lookup, on=["run_id", "epoch"], how="left")
    tr = tr.sort_values(["run_id", "epoch"]).reset_index(drop=True)
    tr["checkpoint_interval_epochs"] = tr.groupby("run_id")["epoch"].diff()
    for stat in ["mean", "median"]:
        hcol = f"path_hellinger_from_previous_{stat}"
        if hcol in tr.columns:
            denom = tr["checkpoint_interval_epochs"].replace(0, np.nan)
            tr[f"path_hellinger_velocity_{stat}"] = tr[hcol] / denom
            tr[f"realized_path_length_cumulative_{stat}"] = tr.groupby("run_id")[hcol].cumsum()
            tr.loc[tr["epoch"] == 0, f"path_hellinger_velocity_{stat}"] = 0.0
    atomic_csv(tr, traj_path)

    id_cols = {"study_arm", "run_id", "config_id", "b2_config_id", "environment_id", "seed_index",
               "activation", "optimizer", "learning_rate", "hidden_depth", "realized_parameters",
               "morphology", "epoch", "functional_progress", "running_best_val_mse"}
    numeric_metrics = [c for c in tr.columns if c not in id_cols and pd.api.types.is_numeric_dtype(tr[c])]

    fp_rows = []
    for rid, g in tr.groupby("run_id"):
        g = g.sort_values("epoch").copy()
        base_meta = {c: g.iloc[0][c] for c in [
            "study_arm", "run_id", "config_id", "b2_config_id", "environment_id", "seed_index",
            "activation", "optimizer", "learning_rate", "hidden_depth", "realized_parameters", "morphology"
        ] if c in g.columns}
        p = g["functional_progress"].to_numpy(float)
        validp = np.isfinite(p)
        if np.sum(validp) < 2:
            continue
        # Monotonic by construction. Collapse repeated progress levels using the latest scheduled state.
        gv = g.loc[validp].copy().sort_values(["functional_progress", "epoch"])
        gv = gv.groupby("functional_progress", as_index=False).tail(1).sort_values("functional_progress")
        xp = gv["functional_progress"].to_numpy(float)
        for target in FUNCTIONAL_PROGRESS_LEVELS:
            rec = {**base_meta, "target_functional_progress": float(target)}
            if target < np.nanmin(xp) - 1e-12 or target > np.nanmax(xp) + 1e-12:
                for m in numeric_metrics: rec[m] = np.nan
            else:
                for m in numeric_metrics:
                    y = gv[m].to_numpy(float)
                    good = np.isfinite(xp) & np.isfinite(y)
                    rec[m] = float(np.interp(target, xp[good], y[good])) if np.sum(good) >= 2 else np.nan
            fp_rows.append(rec)
    fp = pd.DataFrame(fp_rows)
    atomic_csv(fp, fp_path)

    # Post-functional regime: first history epoch reaching 90% progress, then use the first scheduled
    # trajectory checkpoint at or after that epoch and compare it to the final scheduled state.
    late_rows = []
    late_metrics = [
        "structural_entropy_deficit", "structural_eta1", "structural_eta2", "structural_edge_cv", "structural_node_cv",
        "conditional_entropy_deficit_median", "conditional_eta1_median", "conditional_eta2_median",
        "conditional_edge_cv_median", "conditional_node_cv_median", "conditional_kl_to_structural_median",
        "input_specialization_hidden_mean", "path_hellinger_from_init_median"
    ]
    for rid, hg in hp.groupby("run_id"):
        tg = tr[tr.run_id == rid].sort_values("epoch")
        if tg.empty: continue
        hit = hg[np.isfinite(hg.functional_progress) & (hg.functional_progress >= 0.90)]
        if hit.empty: continue
        t90 = int(hit.epoch.min())
        after = tg[tg.epoch >= t90]
        if after.empty: continue
        start = after.iloc[0]; end = tg.iloc[-1]
        rec = {c: tg.iloc[0][c] for c in [
            "study_arm", "run_id", "config_id", "b2_config_id", "environment_id", "seed_index",
            "activation", "optimizer", "learning_rate", "hidden_depth", "realized_parameters", "morphology"
        ] if c in tg.columns}
        rec.update({"t90_epoch": t90, "first_scheduled_epoch_ge_t90": int(start.epoch),
                    "final_scheduled_epoch": int(end.epoch)})
        for m in late_metrics:
            if m in tg.columns:
                rec[f"post90_delta_{m}"] = float(end[m] - start[m]) if np.isfinite(start[m]) and np.isfinite(end[m]) else np.nan
        path_rows = tg[tg.epoch > int(start.epoch)]
        rec["post90_realized_path_length"] = float(np.nansum(path_rows.get(
            "path_hellinger_from_previous_median", pd.Series(dtype=float)
        ).to_numpy(float)))
        late_rows.append(rec)
    late = pd.DataFrame(late_rows)
    atomic_csv(late, late_path)
    return tr, fp, late
# ==================================================================================================
# 14. INPUT-RESOLVED BCANw AT THE BEST CHECKPOINT
# ==================================================================================================


def run_input_resolved_geometry(core_manifest: pd.DataFrame, panel: pd.DataFrame,
                                env_cache: Dict[str, Any]) -> pd.DataFrame:
    path = OUTDIR / "tables" / "input_resolved_geometry.csv"
    if not P2B2_RUN_INPUT_GEOM:
        atomic_csv(pd.DataFrame(), path)
        return pd.DataFrame()

    existing = _safe_read_csv(path) if P2B2_RESUME else pd.DataFrame()
    rows = [] if existing.empty else existing.to_dict("records")
    done = set()
    for r in rows:
        done.add((str(r["run_id"]), int(r["input_index"])))

    archs = select_depth_spanning_architectures(panel, min(INPUT_GEOM_ARCHITECTURES, len(panel)))
    bids = set(archs.b2_config_id.astype(str))
    seeds = MAIN_SEEDS[:min(2, len(MAIN_SEEDS))]
    sub = core_manifest[
        core_manifest.b2_config_id.astype(str).isin(bids) &
        core_manifest.seed_index.astype(int).isin(seeds)
    ].copy()

    total = sum(1 for _, r in sub.iterrows() for i in range(INPUT_GEOM_INPUTS)
                if (str(r.run_id), i) not in done)
    eta = ETAReporter("input-resolved best BCANw", total)
    completed = 0

    for _, r in sub.iterrows():
        cp = _best_checkpoint_path(str(r.run_id))
        if not cp.exists():
            continue
        pending = [i for i in range(INPUT_GEOM_INPUTS) if (str(r.run_id), i) not in done]
        if not pending:
            continue
        hidden = json.loads(r.hidden_widths)
        initseed = _calibration_initial_seed(str(r.config_id), int(r.seed_index))
        model = GeneralMLP(INPUT_DIM, hidden, str(r.activation), bias=MODEL_BIAS, seed=initseed).to(DEVICE)
        chk = torch.load(cp, map_location="cpu")
        model.load_state_dict(chk["state_dict"]); model.to(DEVICE); model.eval()
        structural = structural_state(model)
        Xt = env_cache[str(r.environment_id)][0]["test"][0]
        for i in pending:
            if i >= len(Xt): continue
            t0 = time.perf_counter()
            x = Xt[i:i+1]; x0 = torch.zeros_like(x)
            st = conditional_batch_state(model, x, x0, structural, return_flows=True)
            ef = [E[0] for E in st["edge_flow"]]; nf = [v[0] for v in st["node_flow"]]
            ee, nn = flow_enrichment(ef, nf)
            state = {"edge_flow": ef, "node_flow": nf, "edge_enrichment": ee, "node_enrichment": nn}
            summ, _ = analyze_geometry([], state, n_scales=INPUT_BCANW_SCALES, compute_local=True)
            rows.append({
                "study_arm": "core", "run_id": str(r.run_id), "config_id": str(r.config_id),
                "b2_config_id": str(r.b2_config_id), "environment_id": str(r.environment_id),
                "seed_index": int(r.seed_index), "activation": str(r.activation),
                "optimizer": str(r.optimizer), "input_index": int(i), **summ
            })
            done.add((str(r.run_id), int(i)))
            atomic_csv(pd.DataFrame(rows), path)
            completed += 1; eta.update(time.perf_counter() - t0, completed)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    eta.finish()
    return pd.DataFrame(rows)


# ==================================================================================================
# 15. MATCHED STATISTICS + DIRECT WORK-I STRUCTURAL/REALIZED BRIDGE
# ==================================================================================================


def _holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p); out = np.ones(m, float)
    order = np.argsort(p); running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank) * p[idx]); running = max(running, val); out[idx] = running
    return out


def _attach_best_geometry(summary: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or geometry.empty:
        return summary.copy()
    g = geometry[geometry.checkpoint.astype(str) == "best"].copy()
    cols = ["xi50", "d_star", "x_d_star", "lacunarity_star", "x_lacunarity_star",
            "local_delta_median", "scaling_dB", "quotient_mean_pairwise_KS", "renorm_flow_retention_mean"]
    cp = g[g.geometry_source == "conditional_population"][["run_id"]+cols].rename(
        columns={c: f"best_condgeom_{c}" for c in cols}
    )
    sp = g[g.geometry_source == "structural"][["run_id"]+cols].rename(
        columns={c: f"best_structgeom_{c}" for c in cols}
    )
    return summary.merge(cp, on="run_id", how="left").merge(sp, on="run_id", how="left")


def matched_optimizer_activation_statistics(summary: pd.DataFrame, geometry: pd.DataFrame):
    core = summary[summary.study_arm == "core"].copy() if "study_arm" in summary else summary.copy()
    core = _attach_best_geometry(core, geometry)
    if core.empty:
        return pd.DataFrame(), pd.DataFrame()
    core["block_id"] = (core.b2_config_id.astype(str) + "__" + core.environment_id.astype(str) +
                        "__S" + core.seed_index.astype(str))

    metrics = [
        "best_conditional_entropy_deficit_median", "best_conditional_eta1_median", "best_conditional_eta2_median",
        "best_conditional_edge_cv_median", "best_conditional_node_cv_median",
        "best_conditional_kl_to_structural_median", "best_input_specialization_hidden_mean",
        "best_conditional_structural_edge_spearman", "best_test_mse",
        "best_condgeom_xi50", "best_condgeom_d_star", "best_condgeom_x_d_star",
        "best_condgeom_lacunarity_star", "best_condgeom_local_delta_median", "best_condgeom_scaling_dB",
    ]
    metrics = [m for m in metrics if m in core.columns]
    omni = []; pair = []

    # Optimizer effect within each activation.
    for act in ACTIVATIONS:
        sub = core[core.activation == act]
        for metric in metrics:
            piv = sub.pivot_table(index="block_id", columns="optimizer", values=metric, aggfunc="first")
            levels = [o for o in OPTIMIZERS if o in piv.columns]
            comp = piv[levels].dropna()
            if len(comp) < 3 or len(levels) < 2: continue
            try:
                stat, p = friedmanchisquare(*[comp[o].to_numpy(float) for o in levels])
                W = float(stat / (len(comp) * (len(levels)-1)))
            except Exception:
                stat, p, W = np.nan, np.nan, np.nan
            omni.append({"test_factor": "optimizer_within_activation", "fixed_level": act,
                         "metric": metric, "n_blocks": len(comp), "n_levels": len(levels),
                         "friedman_chi2": stat, "p_value": p, "kendall_W": W})
            raw = []
            for i in range(len(levels)):
                for j in range(i+1, len(levels)):
                    a, b = levels[i], levels[j]; x = comp[a].to_numpy(float); y = comp[b].to_numpy(float)
                    try: st, pv = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                    except Exception: st, pv = np.nan, 1.0
                    raw.append((a, b, st, pv, float(np.median(x-y))))
            adj = _holm_adjust([r[3] for r in raw])
            for k, r in enumerate(raw):
                pair.append({"test_factor": "optimizer_within_activation", "fixed_level": act,
                             "metric": metric, "level_a": r[0], "level_b": r[1],
                             "wilcoxon_W": r[2], "p_raw": r[3], "p_holm": float(adj[k]),
                             "median_paired_difference_a_minus_b": r[4], "n_blocks": len(comp)})

    # Activation effect within each optimizer.
    for opt_name in OPTIMIZERS:
        sub = core[core.optimizer == opt_name]
        for metric in metrics:
            piv = sub.pivot_table(index="block_id", columns="activation", values=metric, aggfunc="first")
            levels = [a for a in ACTIVATIONS if a in piv.columns]
            comp = piv[levels].dropna()
            if len(comp) < 3 or len(levels) < 2: continue
            try:
                stat, p = friedmanchisquare(*[comp[a].to_numpy(float) for a in levels])
                W = float(stat / (len(comp) * (len(levels)-1)))
            except Exception:
                stat, p, W = np.nan, np.nan, np.nan
            omni.append({"test_factor": "activation_within_optimizer", "fixed_level": opt_name,
                         "metric": metric, "n_blocks": len(comp), "n_levels": len(levels),
                         "friedman_chi2": stat, "p_value": p, "kendall_W": W})
            raw = []
            for i in range(len(levels)):
                for j in range(i+1, len(levels)):
                    a, b = levels[i], levels[j]; x = comp[a].to_numpy(float); y = comp[b].to_numpy(float)
                    try: st, pv = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                    except Exception: st, pv = np.nan, 1.0
                    raw.append((a, b, st, pv, float(np.median(x-y))))
            adj = _holm_adjust([r[3] for r in raw])
            for k, r in enumerate(raw):
                pair.append({"test_factor": "activation_within_optimizer", "fixed_level": opt_name,
                             "metric": metric, "level_a": r[0], "level_b": r[1],
                             "wilcoxon_W": r[2], "p_raw": r[3], "p_holm": float(adj[k]),
                             "median_paired_difference_a_minus_b": r[4], "n_blocks": len(comp)})

    odf = pd.DataFrame(omni); pdf = pd.DataFrame(pair)
    atomic_csv(odf, OUTDIR / "tables" / "optimizer_activation_friedman.csv")
    atomic_csv(pdf, OUTDIR / "tables" / "optimizer_activation_pairwise.csv")
    return odf, pdf


def build_work1_bridge(summary: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Compare optimizer ordering of structural disposition and conditional realization at best state."""
    core = summary[summary.study_arm == "core"].copy() if "study_arm" in summary else summary.copy()
    core = _attach_best_geometry(core, geometry)
    if core.empty:
        out = pd.DataFrame(); atomic_csv(out, OUTDIR / "tables" / "work1_bridge_optimizer_ranks.csv"); return out

    metric_pairs = {
        "entropy_deficit": ("best_structural_entropy_deficit", "best_conditional_entropy_deficit_median"),
        "eta1": ("best_structural_eta1", "best_conditional_eta1_median"),
        "eta2": ("best_structural_eta2", "best_conditional_eta2_median"),
        "edge_cv": ("best_structural_edge_cv", "best_conditional_edge_cv_median"),
        "node_cv": ("best_structural_node_cv", "best_conditional_node_cv_median"),
        "xi50": ("best_structgeom_xi50", "best_condgeom_xi50"),
        "d_star": ("best_structgeom_d_star", "best_condgeom_d_star"),
        "lacunarity_star": ("best_structgeom_lacunarity_star", "best_condgeom_lacunarity_star"),
    }
    rows = []
    block_cols = ["b2_config_id", "environment_id", "seed_index", "activation"]
    for keys, g in core.groupby(block_cols):
        if len(set(g.optimizer)) < len(OPTIMIZERS): continue
        for label, (sm, cm) in metric_pairs.items():
            if sm not in g.columns or cm not in g.columns: continue
            piv = g.set_index("optimizer")[[sm, cm]].reindex(OPTIMIZERS).dropna()
            if len(piv) < 3: continue
            rho = float(spearmanr(piv[sm], piv[cm]).statistic)
            rows.append({
                "b2_config_id": keys[0], "environment_id": keys[1], "seed_index": int(keys[2]),
                "activation": keys[3], "metric": label, "optimizer_rank_spearman_structural_vs_realized": rho,
                "structural_optimizer_order": ">".join(piv[sm].sort_values(ascending=False).index.tolist()),
                "realized_optimizer_order": ">".join(piv[cm].sort_values(ascending=False).index.tolist()),
            })
    out = pd.DataFrame(rows)
    atomic_csv(out, OUTDIR / "tables" / "work1_bridge_optimizer_ranks.csv")

    # Pairwise win fractions are easier to interpret than a four-point rank coefficient alone.
    win_rows = []
    for act in ACTIVATIONS:
        gact = core[core.activation == act]
        for label, (sm, cm) in metric_pairs.items():
            if sm not in gact.columns or cm not in gact.columns: continue
            for a, b in itertools.combinations(OPTIMIZERS, 2):
                block = gact.pivot_table(index=["b2_config_id", "environment_id", "seed_index"],
                                         columns="optimizer", values=[sm, cm], aggfunc="first")
                try:
                    sa = block[(sm, a)]; sb = block[(sm, b)]; ca = block[(cm, a)]; cb = block[(cm, b)]
                    mask = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(ca) & np.isfinite(cb)
                    if np.sum(mask) == 0: continue
                    win_rows.append({
                        "activation": act, "metric": label, "optimizer_a": a, "optimizer_b": b,
                        "n_blocks": int(np.sum(mask)),
                        "structural_fraction_a_gt_b": float(np.mean(sa[mask] > sb[mask])),
                        "realized_fraction_a_gt_b": float(np.mean(ca[mask] > cb[mask])),
                    })
                except Exception:
                    continue
    atomic_csv(pd.DataFrame(win_rows), OUTDIR / "tables" / "work1_bridge_pairwise_wins.csv")
    return out
# ==================================================================================================
# 16. CONTROLLED-ENVIRONMENT SUMMARIES
# ==================================================================================================


def controlled_environment_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "study_arm" not in summary:
        out = pd.DataFrame(); atomic_csv(out, OUTDIR / "tables" / "controlled_environment_summary.csv"); return out
    c = summary[summary.study_arm == "controlled_environment"].copy()
    if c.empty:
        atomic_csv(c, OUTDIR / "tables" / "controlled_environment_summary.csv"); return c
    c["task_family"] = c.environment_id.map({
        "CF_SMOOTH_HIGH": "smooth", "CF_SMOOTH_LOW": "smooth",
        "CF_PIECE_HIGH": "piecewise", "CF_PIECE_LOW": "piecewise",
    })
    c["snr_level"] = c.environment_id.map({
        "CF_SMOOTH_HIGH": "high", "CF_PIECE_HIGH": "high",
        "CF_SMOOTH_LOW": "low", "CF_PIECE_LOW": "low",
    })
    metrics = [m for m in [
        "best_test_mse", "best_conditional_entropy_deficit_median", "best_conditional_eta1_median",
        "best_conditional_eta2_median", "best_conditional_edge_cv_median", "best_conditional_node_cv_median",
        "best_conditional_kl_to_structural_median", "best_input_specialization_hidden_mean"
    ] if m in c.columns]
    out = c.groupby(["task_family", "snr_level", "activation"])[metrics].agg(["median", "mean", "std"]).reset_index()
    out.columns = ["__".join([str(x) for x in col if str(x) != ""]) if isinstance(col, tuple) else str(col) for col in out.columns]
    atomic_csv(out, OUTDIR / "tables" / "controlled_environment_summary.csv")
    return out


# ==================================================================================================
# 17. COMPACT DIAGNOSTIC FIGURES
# ==================================================================================================

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9.0, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 130, "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def savefig(fig, stem: str):
    fig.tight_layout()
    fig.savefig(OUTDIR / "figures" / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTDIR / "figures" / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _heatmap_activation_optimizer(ax, df: pd.DataFrame, metric: str, title: str, transform=None):
    sub = df[df.study_arm == "core"] if "study_arm" in df else df
    p = sub.groupby(["activation", "optimizer"])[metric].median().unstack("optimizer")
    p = p.reindex(index=[a for a in ACTIVATIONS if a in p.index],
                  columns=[o for o in OPTIMIZERS if o in p.columns])
    arr = p.to_numpy(float)
    if transform is not None: arr = transform(arr)
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(p.columns)), p.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(p.index)), p.index)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.03)


def make_diagnostic_figures(cal_summary: pd.DataFrame, lr_map: Dict[str, float], summary: pd.DataFrame,
                            trajectory: pd.DataFrame, fp: pd.DataFrame, geometry: pd.DataFrame,
                            bridge: pd.DataFrame):
    if not cal_summary.empty:
        fig, ax = plt.subplots(figsize=(6.8, 4.1))
        for o in OPTIMIZERS:
            g = cal_summary[cal_summary.optimizer == o].sort_values("learning_rate")
            if len(g):
                ax.plot(g.learning_rate, g.selection_objective, marker="o", label=f"{o} (chosen={lr_map[o]:g})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("learning rate"); ax.set_ylabel("worst normalized validation-loss AUC")
        ax.set_title("Optimizer-specific global learning-rate calibration")
        ax.legend(frameon=False)
        savefig(fig, "figure1_optimizer_calibration")

    if not summary.empty:
        fig, axs = plt.subplots(2, 3, figsize=(11.4, 6.4))
        specs = [
            ("best_conditional_entropy_deficit_median", "entropy deficit", None),
            ("best_conditional_eta1_median", r"log10 $\eta_1$", lambda x: np.log10(np.clip(x, 1e-300, None))),
            ("best_conditional_eta2_median", r"log10 $\eta_2$", lambda x: np.log10(np.clip(x, 1e-300, None))),
            ("best_conditional_edge_cv_median", "edge-flow CV", None),
            ("best_conditional_kl_to_structural_median", "KL(realized || weight)", None),
            ("best_input_specialization_hidden_mean", "hidden input specialization", None),
        ]
        for ax, (m, title, trf) in zip(axs.flat, specs):
            if m in summary.columns: _heatmap_activation_optimizer(ax, summary, m, title, trf)
            else: ax.axis("off")
        fig.suptitle("Best-checkpoint activation × optimizer realized-path organization", y=1.01)
        savefig(fig, "figure2_activation_optimizer_endpoint_maps")

    core_tr = trajectory[trajectory.study_arm == "core"] if (not trajectory.empty and "study_arm" in trajectory) else trajectory
    if not core_tr.empty:
        acts = [a for a in ACTIVATIONS if a in set(core_tr.activation)]
        ncol = min(3, max(1, len(acts))); nrow = int(math.ceil(len(acts) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.0*ncol, 3.1*nrow), squeeze=False)
        for ax, act in zip(axs.flat, acts):
            g = core_tr[core_tr.activation == act]
            for o in OPTIMIZERS:
                q = g[g.optimizer == o].groupby("epoch")["conditional_entropy_deficit_median"].median().sort_index()
                if len(q): ax.plot(q.index, q.values, marker="o", ms=3, label=o)
            ax.set_title(act); ax.set_xlabel("epoch"); ax.set_ylabel("conditional entropy deficit")
        for ax in axs.flat[len(acts):]: ax.axis("off")
        axs.flat[0].legend(frameon=False, ncol=2)
        fig.suptitle("Optimizer-conditioned trajectory of realized path concentration", y=1.01)
        savefig(fig, "figure3_entropy_deficit_training_trajectories")

    if fp is not None and not fp.empty:
        acts = [a for a in ACTIVATIONS if a in set(fp.activation)]
        ncol = min(3, max(1, len(acts))); nrow = int(math.ceil(len(acts) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.0*ncol, 3.1*nrow), squeeze=False)
        for ax, act in zip(axs.flat, acts):
            g = fp[fp.activation == act]
            for o in OPTIMIZERS:
                q = g[g.optimizer == o].groupby("target_functional_progress")["conditional_entropy_deficit_median"].median().sort_index()
                if len(q): ax.plot(q.index, q.values, marker="o", ms=3, label=o)
            ax.set_title(act); ax.set_xlabel("normalized functional progress"); ax.set_ylabel("conditional entropy deficit")
        for ax in axs.flat[len(acts):]: ax.axis("off")
        axs.flat[0].legend(frameon=False, ncol=2)
        fig.suptitle("Realized organization at matched functional progress", y=1.01)
        savefig(fig, "figure4_functional_progress_trajectories")

    if not geometry.empty:
        g = geometry[(geometry.checkpoint.astype(str) == "best") &
                     (geometry.geometry_source == "conditional_population")].copy()
        if not g.empty:
            fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.6))
            for ax, m, title in zip(axs, ["xi50", "d_star", "lacunarity_star"],
                                    [r"$\xi_{50}$", r"$d^*$", r"peak $\Lambda_F$"]):
                _heatmap_activation_optimizer(ax, g, m, title)
            fig.suptitle("Best-checkpoint conditional BCANw geometry", y=1.02)
            savefig(fig, "figure5_activation_optimizer_geometry")

    if bridge is not None and not bridge.empty:
        b = bridge[bridge.metric == "entropy_deficit"].copy()
        if not b.empty:
            med = b.groupby("activation")["optimizer_rank_spearman_structural_vs_realized"].median().reindex(
                [a for a in ACTIVATIONS if a in set(b.activation)]
            )
            fig, ax = plt.subplots(figsize=(6.8, 3.8))
            ax.bar(np.arange(len(med)), med.values)
            ax.set_xticks(np.arange(len(med)), med.index, rotation=25)
            ax.set_ylim(-1.05, 1.05); ax.axhline(0, lw=.8)
            ax.set_ylabel("median optimizer-rank Spearman correlation")
            ax.set_title("Work-I structural optimizer ordering versus realized ordering")
            savefig(fig, "figure6_work1_structural_realized_optimizer_bridge")


# ==================================================================================================
# 18. EXPERIMENT MANIFEST + MAIN ORCHESTRATION
# ==================================================================================================


def write_experiment_manifest(panel: pd.DataFrame, core_manifest: pd.DataFrame,
                              controlled_manifest: pd.DataFrame, lr_map: Dict[str, float]):
    obj = {
        "work": "II", "phase": "II-B.2", "profile": P2B2_PROFILE, "device": DEVICE,
        "scientific_question": "optimizer-conditioned realized computation in path-decomposable MLPs",
        "observable": "finite conditional path measure p_Delta(pi|x,x0); object unchanged from Phase II-B",
        "architecture_source_directory": str(ARCH_SOURCE_DIR),
        "architecture_panel": {
            "source_count": 20 if P2B2_PROFILE == "research" else int(max(len(panel), B2_ARCHITECTURES)),
            "selected_for_core": int(len(panel)),
            "selection": "deterministic design-space coverage with shallow/wide, deep/narrow and morphology coverage",
        },
        "optimizers": {
            "families": OPTIMIZERS, "learning_rates": lr_map,
            "sgdm_momentum": 0.9, "rmsprop_alpha": 0.99, "adam_betas": [0.9, 0.999],
            "weight_decay": WEIGHT_DECAY, "scheduler": None,
            "calibration": {
                "global_per_optimizer": True, "activation_specific_tuning": False,
                "architectures": CALIBRATION_ARCHITECTURES, "anchor_activations": CALIBRATION_ACTIVATIONS,
                "seeds": CALIBRATION_SEEDS, "epochs": CALIBRATION_EPOCHS,
                "objective": "worst normalized validation-loss AUC across discovery anchor conditions",
            },
        },
        "core_factorial": {
            "architectures": int(len(panel)), "environments": MAIN_ENVIRONMENTS,
            "seeds": MAIN_SEEDS, "activations": ACTIVATIONS, "optimizers": OPTIMIZERS,
            "full_runs": int(len(core_manifest)),
            "matched_blocks": int(len(panel) * len(MAIN_ENVIRONMENTS) * len(MAIN_SEEDS)),
        },
        "training": {
            "epochs": MAX_EPOCH, "batch_size": BATCH_SIZE, "no_early_stopping": True,
            "trajectory_checkpoints": TRAJECTORY_CHECKPOINTS,
            "geometry_checkpoints": TRAJECTORY_GEOMETRY_CHECKPOINTS,
            "same_minibatch_order_across_optimizers": True,
            "same_activation_calibrated_initial_state_across_optimizers": True,
        },
        "conditional_analysis": {
            "trajectory_inputs": TRAJECTORY_INPUTS, "endpoint_inputs": ENDPOINT_INPUTS,
            "reference": "x0=0 after input standardization",
            "primary_path_metrics": ["S_deficit", "eta1", "eta2", "edge_CV", "node_CV", "KL_to_weight", "input_specialization"],
            "new_longitudinal_metrics": ["exact path-space Hellinger movement", "response-factor drift", "ReLU support turnover", "Leaky regime turnover"],
            "functional_progress_levels": FUNCTIONAL_PROGRESS_LEVELS,
        },
        "BCANw": {
            "faithfulness": "unchanged corrected Work-I/III and Phase-II-B implementation",
            "metric_power": METRIC_POWER, "xi50_normalization": True,
            "primary": ["xi50", "d_star", "x_d_star", "lacunarity_star", "local_delta_median", "quotient KS", "flow retention"],
            "secondary": ["strict finite-window dB"],
        },
        "controlled_environment_extension": {
            "enabled": bool(P2B2_RUN_CONTROLLED_ENV), "runs": int(len(controlled_manifest)),
            "task_family": ["smooth", "piecewise"], "snr": ["high", "low"],
            "feature_correlation_fixed": CONTROLLED_FEATURE_CORRELATION,
            "interaction_fixed": CONTROLLED_INTERACTION,
            "optimizer": "adam",
        },
        "input_resolved_geometry": {
            "enabled": bool(P2B2_RUN_INPUT_GEOM), "architectures": INPUT_GEOM_ARCHITECTURES,
            "environments": MAIN_ENVIRONMENTS, "seeds": MAIN_SEEDS[:min(2, len(MAIN_SEEDS))],
            "activations": ACTIVATIONS, "optimizers": OPTIMIZERS, "inputs_per_model": INPUT_GEOM_INPUTS,
        },
        "boundaries": [
            "route prominence is not additive output energy",
            "weight-only baseline is descriptive outside symmetry regimes that make it intrinsic",
            "dB is a secondary finite-window descriptor",
            "inputs are nested observations, not independent experimental units",
            "calibration runs are excluded from inference",
        ],
    }
    atomic_json(obj, OUTDIR / "experiment_manifest.json")


def run_phase2b2_experiment():
    total_start = time.perf_counter()
    banner("WORK II — PHASE II-B.2: OPTIMIZER-CONDITIONED REALIZED COMPUTATION")
    print(f"Profile: {P2B2_PROFILE} | Device: {DEVICE} | Training horizon: {MAX_EPOCH} epochs")
    print("Object: p_Delta(pi|x,x0), unchanged from II-B; optimizer now conditions the learned state.")

    banner("Architecture source and six-configuration coverage panel")
    source_panel = load_or_discover_expanded_architecture_panel()
    b2_panel = select_b2_architecture_panel(source_panel)
    print(b2_panel[["b2_config_id", "config_id", "hidden_depth", "realized_parameters", "morphology", "hidden_widths"]].to_string(index=False))

    if P2B2_RUN_EXACT_PREFLIGHT:
        banner("Inherited finite-response exactness preflight")
        exact = run_exact_validation()
        if not exact.empty:
            print(exact.groupby("activation")[["reconstruction_rel_error", "entropy_abs_error", "edge_flow_max_abs_error"]].median())
    else:
        exact = pd.DataFrame()

    banner("Optimizer-specific global learning-rate calibration")
    cal_runs, cal_summary, lr_map = run_optimizer_calibration(b2_panel)
    print("Selected optimizer learning rates:")
    for o in OPTIMIZERS: print(f"  {o:8s}: {lr_map[o]:.6g}")

    core_manifest = build_main_manifest(b2_panel)
    controlled_manifest = build_controlled_environment_manifest(b2_panel)
    all_manifest = pd.concat([core_manifest, controlled_manifest], ignore_index=True) if not controlled_manifest.empty else core_manifest.copy()
    write_experiment_manifest(b2_panel, core_manifest, controlled_manifest, lr_map)

    print(f"Core crossed matrix: {len(core_manifest)} runs")
    print(f"Controlled environment extension: {len(controlled_manifest)} runs")
    print(f"Total inferential training conditions: {len(all_manifest)}")

    env_ids = sorted(set(all_manifest.environment_id.astype(str)))
    env_cache = {eid: generate_environment(eid, discovery=False) for eid in env_ids}

    banner("Core 6 architecture × 2 environment × 4 seed × 6 activation × 4 optimizer experiment")
    run_manifest(core_manifest, env_cache, lr_map, f"II-B.2 core {MAX_EPOCH}-epoch runs")

    if not controlled_manifest.empty:
        banner("Controlled 2 × 2 task-family/SNR environment extension")
        run_manifest(controlled_manifest, env_cache, lr_map, f"II-B.2 controlled-environment {MAX_EPOCH}-epoch runs")

    banner("Aggregate resumable run parts")
    summary, endpoints, history, trajectory_raw, geometry, scales = aggregate_run_parts(all_manifest)

    banner("Equal-functional-progress and post-functional analyses")
    trajectory, fp, late = build_functional_progress_tables(history, trajectory_raw)

    banner("Input-resolved best-checkpoint BCANw")
    input_geom = run_input_resolved_geometry(core_manifest, b2_panel, env_cache)

    banner("Matched optimizer/activation statistics")
    omnibus, pairwise = matched_optimizer_activation_statistics(summary, geometry)
    if not omnibus.empty:
        print(omnibus.head(24).to_string(index=False))

    banner("Direct Work-I structural-to-realized optimizer bridge")
    bridge = build_work1_bridge(summary, geometry)
    cenv_summary = controlled_environment_summary(summary)

    banner("Diagnostic figures")
    make_diagnostic_figures(cal_summary, lr_map, summary, trajectory, fp, geometry, bridge)

    elapsed = time.perf_counter() - total_start
    report = [
        "WORK II PHASE II-B.2 COMPLETE",
        f"Profile: {P2B2_PROFILE}", f"Device: {DEVICE}", f"Wall time: {human_time(elapsed)}",
        f"Architecture panel: {len(b2_panel)}",
        f"Core 280-epoch runs summarized: {len(summary[summary.study_arm=='core']) if (not summary.empty and 'study_arm' in summary) else 0} / {len(core_manifest)}",
        f"Controlled-environment runs: {len(summary[summary.study_arm=='controlled_environment']) if (not summary.empty and 'study_arm' in summary) else 0} / {len(controlled_manifest)}",
        f"Trajectory rows: {len(trajectory)}", f"Endpoint input rows: {len(endpoints)}",
        f"Geometry rows: {len(geometry)}", f"Input-resolved geometry rows: {len(input_geom)}",
        f"Functional-progress rows: {len(fp)}", f"Late-training rows: {len(late)}",
        f"Outputs: {OUTDIR.resolve()}",
    ]
    (OUTDIR / "experiment_summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report))
    return {
        "architecture_source": source_panel, "architecture_panel": b2_panel,
        "calibration_runs": cal_runs, "calibration_summary": cal_summary, "learning_rates": lr_map,
        "core_manifest": core_manifest, "controlled_manifest": controlled_manifest,
        "summary": summary, "endpoint_inputs": endpoints, "history": history,
        "trajectory": trajectory, "geometry": geometry, "geometry_scales": scales,
        "functional_progress": fp, "late_training": late, "input_geometry": input_geom,
        "omnibus": omnibus, "pairwise": pairwise, "work1_bridge": bridge,
        "controlled_environment_summary": cenv_summary, "exact": exact,
    }


# Run immediately when pasted into one Jupyter cell, matching the Work-I/II single-cell convention.
# P2B2_RESULTS intentionally disabled in the C1 specialization.


# ==================================================================================================
# 19. C1-SPECIFIC ARCHITECTURE PANEL, MANIFEST AND INFERENCE
# ==================================================================================================


def _refresh_arch_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    out=_parse_hidden_widths_column(df)
    rows=[]
    for _,r in out.iterrows():
        d=r.to_dict(); h=np.asarray(d["hidden_widths"],float)
        d["hidden_depth"]=int(len(h))
        d["median_width"]=float(np.median(h)); d["min_width"]=int(np.min(h)); d["max_width"]=int(np.max(h))
        d["width_cv"]=float(np.std(h)/max(np.mean(h),EPS))
        d["bottleneck_ratio"]=float(np.min(h)/max(np.max(h),EPS))
        d["expansion_index"]=float((h[-1]-h[0])/max(np.median(h),EPS)) if len(h)>1 else 0.0
        d["log_topological_paths"]=float(math.log(INPUT_DIM)+np.sum(np.log(np.clip(h,1,None))))
        d["realized_parameters"]=int(parameter_count([INPUT_DIM]+[int(x) for x in h]+[1],bias=MODEL_BIAS))
        rows.append(d)
    return pd.DataFrame(rows)


def select_c1_architecture_panel(archive: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_architecture_panel.csv"
    if C1_RESUME and path.exists():
        return _refresh_arch_descriptors(pd.read_csv(path))
    a=_refresh_arch_descriptors(archive.copy())
    a=a[np.isfinite(pd.to_numeric(a.discovery_objective,errors="coerce"))].copy()
    if len(a)<C1_ARCHITECTURES:
        raise RuntimeError(f"C1 requires {C1_ARCHITECTURES} viable RF-SMAC architectures; found {len(a)}")
    cutoff=float(a.discovery_objective.quantile(RFSMAC_VIABLE_FRACTION))
    viable=a[a.discovery_objective<=cutoff].copy()
    if len(viable)<C1_ARCHITECTURES:
        viable=a.nsmallest(C1_ARCHITECTURES,"discovery_objective").copy()
    Z=StandardScaler().fit_transform(np.vstack([candidate_features(r) for _,r in viable.iterrows()]))
    # Performance only defines the viability gate above.  Frozen-panel selection itself is entirely
    # design-space based: initialize with the maximally separated pair, then morphology-aware maximin.
    D=np.linalg.norm(Z[:,None,:]-Z[None,:,:],axis=2); np.fill_diagonal(D,-np.inf)
    i0,i1=np.unravel_index(int(np.argmax(D)),D.shape); selected=[int(i0)]
    if int(i1)!=int(i0): selected.append(int(i1))
    def md(i):
        return float(np.min(np.linalg.norm(Z[i][None,:]-Z[np.asarray(selected)],axis=1))) if selected else np.inf
    # Force morphology coverage where the viable archive contains that family.
    for morph in MORPHOLOGIES:
        if len(selected)>=C1_ARCHITECTURES: break
        if any(str(viable.iloc[i].morphology)==morph for i in selected): continue
        ids=[i for i in range(len(viable)) if str(viable.iloc[i].morphology)==morph and i not in selected]
        if ids: selected.append(max(ids,key=md))
    while len(selected)<C1_ARCHITECTURES:
        ids=[i for i in range(len(viable)) if i not in selected]
        selected.append(max(ids,key=md))
    panel=viable.iloc[selected[:C1_ARCHITECTURES]].copy().reset_index(drop=True)
    panel["c1_panel_rank"]=np.arange(len(panel))
    panel["c1_config_id"]=[f"C1A{k:02d}" for k in range(len(panel))]
    panel["b2_config_id"]=panel["c1_config_id"]  # compatibility alias in inherited storage helpers.
    save=panel.copy(); save["hidden_widths"]=save.hidden_widths.apply(json.dumps); atomic_csv(save,path)
    return panel


def select_c1_tanh_sentinels(panel: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_tanh_sentinels.csv"
    if C1_RESUME and path.exists(): return _refresh_arch_descriptors(pd.read_csv(path))
    p=_refresh_arch_descriptors(panel).reset_index(drop=True)
    Z=StandardScaler().fit_transform(np.vstack([candidate_features(r) for _,r in p.iterrows()]))
    selected=[]
    def md(i): return float(np.min(np.linalg.norm(Z[i][None,:]-Z[np.asarray(selected)],axis=1))) if selected else np.inf
    # One representative per morphology where possible; choose the most isolated member of each family.
    for morph in MORPHOLOGIES:
        ids=[i for i in range(len(p)) if str(p.iloc[i].morphology)==morph]
        if ids:
            if not selected: selected.append(ids[len(ids)//2])
            else: selected.append(max([i for i in ids if i not in selected] or ids,key=md))
        if len(selected)>=C1_TANH_SENTINELS: break
    while len(selected)<min(C1_TANH_SENTINELS,len(p)):
        ids=[i for i in range(len(p)) if i not in selected]; selected.append(max(ids,key=md))
    out=p.iloc[selected[:C1_TANH_SENTINELS]].copy().reset_index(drop=True)
    out["sentinel_rank"]=np.arange(len(out))
    save=out.copy(); save["hidden_widths"]=save.hidden_widths.apply(json.dumps); atomic_csv(save,path)
    return out


def build_c1_manifest(panel: pd.DataFrame, sentinels: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_run_manifest.csv"
    if C1_RESUME and path.exists(): return pd.read_csv(path)
    rows=[]
    sentinel_ids=set(sentinels.c1_config_id.astype(str))
    for _,r in panel.iterrows():
        hidden=r.hidden_widths if not isinstance(r.hidden_widths,str) else json.loads(r.hidden_widths)
        base={
            "config_id":str(r.config_id),"c1_config_id":str(r.c1_config_id),"b2_config_id":str(r.c1_config_id),
            "environment_id":"C1_CANON","optimizer":FIXED_OPTIMIZER,"learning_rate":FIXED_ADAM_LR,
            "hidden_widths":json.dumps([int(x) for x in hidden]),"hidden_depth":int(r.hidden_depth),
            "realized_parameters":int(r.realized_parameters),"morphology":str(r.morphology),
            "median_width":float(r.median_width),"min_width":int(r.min_width),"max_width":int(r.max_width),
            "width_cv":float(r.width_cv),"bottleneck_ratio":float(r.bottleneck_ratio),
            "expansion_index":float(r.expansion_index),"log_topological_paths":float(r.log_topological_paths),
            "longitudinal":True,
        }
        for seed in MAIN_SEEDS:
            rid=f"C1__{r.c1_config_id}__RELU__S{seed:02d}"
            rows.append({**base,"study_arm":"c1_primary_relu","run_id":rid,"seed_index":int(seed),"activation":"relu"})
            if str(r.c1_config_id) in sentinel_ids:
                rid=f"C1__{r.c1_config_id}__TANH__S{seed:02d}"
                rows.append({**base,"study_arm":"c1_tanh_sentinel","run_id":rid,"seed_index":int(seed),"activation":"tanh"})
    out=pd.DataFrame(rows).sort_values(["study_arm","c1_config_id","seed_index"]).reset_index(drop=True)
    atomic_csv(out,path); return out


def run_c1_manifest(manifest: pd.DataFrame, env_cache: Dict[str,Any]):
    lr_map={"adam":FIXED_ADAM_LR}
    todo=[r for _,r in manifest.iterrows() if not (C1_RESUME and run_complete(str(r.run_id)))]
    eta=ETAReporter("C1 publication trajectories",len(todo))
    for k,r in enumerate(tqdm(todo,desc="C1 architecture-wide conditional realization",dynamic_ncols=True),1):
        t0=time.perf_counter(); train_one_phase2b2_run(r,env_cache[str(r.environment_id)][0],lr_map)
        eta.update(time.perf_counter()-t0,k)
    eta.finish()


def _rank_biserial_signed(d: np.ndarray) -> float:
    d=np.asarray(d,float); d=d[np.isfinite(d)&(d!=0)]
    if not len(d): return np.nan
    from scipy.stats import rankdata
    rr=rankdata(np.abs(d),method="average"); wp=float(rr[d>0].sum()); wm=float(rr[d<0].sum())
    return float((wp-wm)/max(wp+wm,EPS))


def _hodges_lehmann_paired(d: np.ndarray) -> float:
    d=np.asarray(d,float); d=d[np.isfinite(d)]
    if not len(d): return np.nan
    vals=[]
    for i in range(len(d)): vals.extend(((d[i]+d[i:])/2.0).tolist())
    return float(np.median(vals))


def _friedman_architecture(df: pd.DataFrame, metric: str, arch_col: str="c1_config_id") -> Dict[str,Any]:
    piv=df.pivot_table(index="seed_index",columns=arch_col,values=metric,aggfunc="first")
    levels=sorted(piv.columns.astype(str)); comp=piv[levels].dropna()
    if len(comp)<2 or len(levels)<3: return {"metric":metric,"n_seed_blocks":len(comp),"n_architectures":len(levels),"friedman_chi2":np.nan,"p_raw":np.nan,"kendall_W":np.nan}
    stat,p=friedmanchisquare(*[comp[a].to_numpy(float) for a in levels]); W=float(stat/(len(comp)*(len(levels)-1)))
    return {"metric":metric,"n_seed_blocks":int(len(comp)),"n_architectures":int(len(levels)),"friedman_chi2":float(stat),"p_raw":float(p),"kendall_W":W}


def _c1_endpoint_table(summary: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    s=_attach_best_geometry(summary,geometry)
    return s


def c1_architecture_statistics(summary: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_architecture_friedman.csv"
    core=_c1_endpoint_table(summary[summary.study_arm=="c1_primary_relu"].copy(),geometry)
    metrics=[
        "best_test_mse","best_conditional_entropy_deficit_median","best_conditional_eta1_median",
        "best_conditional_eta2_median","best_conditional_edge_cv_median","best_conditional_node_cv_median",
        "best_conditional_kl_to_structural_median","best_conditional_hellinger2_to_structural_median",
        "best_input_specialization_hidden_mean","best_conditional_structural_edge_spearman",
        "best_structural_entropy_deficit","best_structural_eta1","best_structural_eta2",
        "best_structural_edge_cv","best_structural_node_cv",
        "best_condgeom_xi50","best_condgeom_d_star","best_condgeom_lacunarity_star",
        "best_condgeom_local_delta_median","best_condgeom_quotient_mean_pairwise_KS","best_condgeom_renorm_flow_retention_mean",
    ]
    metrics=[m for m in metrics if m in core.columns]
    rows=[_friedman_architecture(core,m) for m in metrics]
    out=pd.DataFrame(rows)
    if len(out): out["p_holm"]=_holm_adjust(out.p_raw.fillna(1.0))
    atomic_csv(out,path); return out


def c1_rank_persistence(trajectory: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_architecture_rank_persistence.csv"
    tr=trajectory[trajectory.study_arm=="c1_primary_relu"].copy()
    metrics=["conditional_entropy_deficit_median","conditional_edge_cv_median","conditional_node_cv_median",
             "conditional_kl_to_structural_median","input_specialization_hidden_mean","structural_entropy_deficit"]
    rows=[]
    for metric in [m for m in metrics if m in tr.columns]:
        for t0,t1 in itertools.combinations_with_replacement(sorted(tr.epoch.unique()),2):
            rhos=[]
            for seed in MAIN_SEEDS:
                a=tr[(tr.seed_index==seed)&(tr.epoch==t0)].set_index("c1_config_id")[metric]
                b=tr[(tr.seed_index==seed)&(tr.epoch==t1)].set_index("c1_config_id")[metric]
                idx=a.index.intersection(b.index); av=a.loc[idx];bv=b.loc[idx]; good=np.isfinite(av)&np.isfinite(bv)
                if int(np.sum(good))>=4: rhos.append(float(spearmanr(av[good],bv[good]).statistic))
            if rhos:
                rows.append({"metric":metric,"epoch_a":int(t0),"epoch_b":int(t1),"n_seed_blocks":len(rhos),
                             "rho_median":float(np.median(rhos)),"rho_mean":float(np.mean(rhos)),
                             "rho_q25":float(np.quantile(rhos,.25)),"rho_q75":float(np.quantile(rhos,.75))})
    out=pd.DataFrame(rows); atomic_csv(out,path); return out


def c1_structural_realized_bridge(summary: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_structural_realized_rank_bridge.csv"
    core=_c1_endpoint_table(summary[summary.study_arm=="c1_primary_relu"].copy(),geometry)
    pairs={
        "entropy_deficit":("best_structural_entropy_deficit","best_conditional_entropy_deficit_median"),
        "eta1":("best_structural_eta1","best_conditional_eta1_median"),
        "eta2":("best_structural_eta2","best_conditional_eta2_median"),
        "edge_cv":("best_structural_edge_cv","best_conditional_edge_cv_median"),
        "node_cv":("best_structural_node_cv","best_conditional_node_cv_median"),
        "xi50":("best_structgeom_xi50","best_condgeom_xi50"),
        "d_star":("best_structgeom_d_star","best_condgeom_d_star"),
        "lacunarity_star":("best_structgeom_lacunarity_star","best_condgeom_lacunarity_star"),
        "retained_flow":("best_structgeom_renorm_flow_retention_mean","best_condgeom_renorm_flow_retention_mean"),
    }
    rows=[]
    for seed,g in core.groupby("seed_index"):
        for label,(sm,cm) in pairs.items():
            if sm not in g or cm not in g: continue
            q=g.set_index("c1_config_id")[[sm,cm]].dropna()
            if len(q)>=4:
                rows.append({"seed_index":int(seed),"metric":label,"n_architectures":int(len(q)),
                             "architecture_rank_spearman_structural_vs_realized":float(spearmanr(q[sm],q[cm]).statistic)})
    out=pd.DataFrame(rows); atomic_csv(out,path); return out


def c1_descriptor_associations(summary: pd.DataFrame, geometry: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_architecture_descriptor_associations.csv"
    core=_c1_endpoint_table(summary[summary.study_arm=="c1_primary_relu"].copy(),geometry)
    metrics=["best_test_mse","best_conditional_entropy_deficit_median","best_conditional_edge_cv_median",
             "best_conditional_node_cv_median","best_conditional_kl_to_structural_median",
             "best_conditional_hellinger2_to_structural_median","best_input_specialization_hidden_mean",
             "best_condgeom_xi50","best_condgeom_d_star","best_condgeom_lacunarity_star","best_condgeom_renorm_flow_retention_mean"]
    metrics=[m for m in metrics if m in core]
    desc=["hidden_depth","realized_parameters","log_topological_paths","median_width","width_cv","bottleneck_ratio","expansion_index"]
    av=core.groupby("c1_config_id")[metrics].median(numeric_only=True).reset_index()
    pp=_refresh_arch_descriptors(panel)[["c1_config_id"]+desc].drop_duplicates("c1_config_id")
    z=av.merge(pp,on="c1_config_id",how="inner")
    rows=[]
    for m in metrics:
        for d in desc:
            q=z[[m,d]].dropna()
            if len(q)>=6:
                rho,p=spearmanr(q[d],q[m]); rows.append({"metric":m,"descriptor":d,"n_architectures":len(q),"spearman_rho":float(rho),"p_raw":float(p)})
    out=pd.DataFrame(rows)
    if len(out):
        out["p_holm_within_metric"]=np.nan
        for m,g in out.groupby("metric"): out.loc[g.index,"p_holm_within_metric"]=_holm_adjust(g.p_raw)
    atomic_csv(out,path); return out


def c1_direction_survival(trajectory: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_direction_survival.csv"
    tr=trajectory[trajectory.study_arm=="c1_primary_relu"].copy()
    metrics=["structural_entropy_deficit","structural_eta1","structural_eta2","structural_edge_cv","structural_node_cv",
             "conditional_entropy_deficit_median","conditional_eta1_median","conditional_eta2_median",
             "conditional_edge_cv_median","conditional_node_cv_median","conditional_kl_to_structural_median",
             "input_specialization_hidden_mean"]
    rows=[]
    for m in [x for x in metrics if x in tr]:
        deltas=[]
        for (arch,seed),g in tr.groupby(["c1_config_id","seed_index"]):
            g=g.sort_values("epoch"); a=g.iloc[0][m]; b=g.iloc[-1][m]
            if np.isfinite(a) and np.isfinite(b): deltas.append((arch,seed,float(b-a)))
        if not deltas: continue
        dd=pd.DataFrame(deltas,columns=["arch","seed","delta"])
        archmed=dd.groupby("arch").delta.median()
        rows.append({"metric":m,"n_architectures":len(archmed),"n_arch_seed_cells":len(dd),
                     "architecture_fraction_positive":float(np.mean(archmed>0)),
                     "architecture_fraction_negative":float(np.mean(archmed<0)),
                     "architecture_median_delta":float(np.median(archmed)),
                     "cell_fraction_positive":float(np.mean(dd.delta>0)),"cell_median_delta":float(np.median(dd.delta))})
    out=pd.DataFrame(rows); atomic_csv(out,path); return out


def c1_functional_progress_stats(fp: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_functional_progress_friedman.csv"
    f=fp[fp.study_arm=="c1_primary_relu"].copy()
    metrics=["conditional_entropy_deficit_median","conditional_edge_cv_median","conditional_node_cv_median",
             "conditional_kl_to_structural_median","input_specialization_hidden_mean","structural_entropy_deficit"]
    rows=[]
    for tau in sorted(f.target_functional_progress.dropna().unique()):
        d=f[np.isclose(f.target_functional_progress,tau)]
        for m in [x for x in metrics if x in d]:
            r=_friedman_architecture(d,m); r["target_functional_progress"]=float(tau); rows.append(r)
    out=pd.DataFrame(rows)
    if len(out):
        out["p_holm_within_progress"]=np.nan
        for tau,g in out.groupby("target_functional_progress"): out.loc[g.index,"p_holm_within_progress"]=_holm_adjust(g.p_raw.fillna(1))
    atomic_csv(out,path); return out


def c1_tanh_statistics(summary: pd.DataFrame, geometry: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    effects_path=OUTDIR/"tables"/"c1_tanh_paired_effects.csv"
    ranks_path=OUTDIR/"tables"/"c1_tanh_architecture_rank_preservation.csv"
    s=_c1_endpoint_table(summary,geometry)
    sent=s[s.study_arm.isin(["c1_primary_relu","c1_tanh_sentinel"])].copy()
    # Only architectures that possess tanh sentinel runs.
    tids=set(sent[sent.activation=="tanh"].c1_config_id.astype(str)); sent=sent[sent.c1_config_id.astype(str).isin(tids)]
    metrics=["best_test_mse","best_conditional_entropy_deficit_median","best_conditional_edge_cv_median",
             "best_conditional_node_cv_median","best_conditional_kl_to_structural_median",
             "best_conditional_hellinger2_to_structural_median","best_input_specialization_hidden_mean",
             "best_condgeom_xi50","best_condgeom_d_star","best_condgeom_lacunarity_star","best_condgeom_renorm_flow_retention_mean"]
    effects=[]
    for m in [x for x in metrics if x in sent]:
        piv=sent.pivot_table(index=["c1_config_id","seed_index"],columns="activation",values=m,aggfunc="first")
        if "relu" not in piv or "tanh" not in piv: continue
        comp=piv[["relu","tanh"]].dropna(); d=(comp.relu-comp.tanh).to_numpy(float)
        try: W,p=wilcoxon(d,zero_method="wilcox",alternative="two-sided")
        except Exception: W,p=np.nan,1.0
        effects.append({"metric":m,"n_pairs":len(d),"median_relu_minus_tanh":float(np.median(d)) if len(d) else np.nan,
                        "hodges_lehmann_shift":_hodges_lehmann_paired(d),"rank_biserial":_rank_biserial_signed(d),
                        "wilcoxon_W":W,"p_raw":p})
    eff=pd.DataFrame(effects)
    if len(eff): eff["p_holm"]=_holm_adjust(eff.p_raw.fillna(1))
    atomic_csv(eff,effects_path)
    rankrows=[]
    for seed,g in sent.groupby("seed_index"):
        for m in [x for x in metrics if x in g]:
            piv=g.pivot_table(index="c1_config_id",columns="activation",values=m,aggfunc="first").dropna()
            if {"relu","tanh"}.issubset(piv.columns) and len(piv)>=4:
                rankrows.append({"seed_index":int(seed),"metric":m,"n_architectures":len(piv),
                                 "architecture_rank_spearman_relu_vs_tanh":float(spearmanr(piv.relu,piv.tanh).statistic)})
    ranks=pd.DataFrame(rankrows); atomic_csv(ranks,ranks_path); return eff,ranks


def run_c1_input_resolved_geometry(manifest: pd.DataFrame, env_cache: Dict[str,Any]) -> pd.DataFrame:
    path=OUTDIR/"tables"/"input_resolved_geometry.csv"
    if not C1_RUN_INPUT_GEOM:
        atomic_csv(pd.DataFrame(),path); return pd.DataFrame()
    existing=_safe_read_csv(path) if C1_RESUME else pd.DataFrame(); rows=[] if existing.empty else existing.to_dict("records")
    done={(str(r["run_id"]),int(r["input_index"])) for r in rows if "run_id" in r and "input_index" in r}
    tasks=[]
    for _,r in manifest.iterrows():
        cp=_best_checkpoint_path(str(r.run_id))
        if not cp.exists(): continue
        for i in range(INPUT_GEOM_INPUTS):
            if (str(r.run_id),i) not in done: tasks.append((r,i))
    eta=ETAReporter("C1 input-resolved BCANw",len(tasks)); completed=0
    # Cache model per run while processing its eight input geometries.
    for run_id,g in pd.DataFrame([dict(r) for r,i in tasks]).groupby("run_id") if tasks else []:
        r=g.iloc[0]; cp=_best_checkpoint_path(str(run_id)); payload=torch.load(cp,map_location="cpu")
        hidden=json.loads(r.hidden_widths) if isinstance(r.hidden_widths,str) else list(r.hidden_widths)
        initseed=_calibration_initial_seed(str(r.config_id),int(r.seed_index))
        model=GeneralMLP(INPUT_DIM,hidden,str(r.activation),bias=MODEL_BIAS,seed=initseed).to(DEVICE)
        model.load_state_dict(payload["state_dict"]); model.eval(); structural=structural_state(model)
        Xt=env_cache[str(r.environment_id)][0]["test"][0]
        indices=list(range(min(INPUT_GEOM_INPUTS,len(Xt))))
        for i in indices:
            if (str(run_id),i) in done: continue
            t0=time.perf_counter(); x=Xt[i:i+1]; x0=torch.zeros_like(x)
            st=conditional_batch_state(model,x,x0,structural,return_flows=True)
            defined=bool(np.asarray(st["measure_defined"],bool)[0])
            if defined:
                pop={"defined":True,"edge_flow":[e[0] for e in st["edge_flow"]],"node_flow":[v[0] for v in st["node_flow"]]}
                ee,nn=flow_enrichment(pop["edge_flow"],pop["node_flow"]); pop["edge_enrichment"]=ee;pop["node_enrichment"]=nn
                pop["edge_conservation_error"],pop["node_conservation_error"]=layer_conservation_errors(pop["edge_flow"],pop["node_flow"])
                gs,_=analyze_geometry([],pop,n_scales=INPUT_BCANW_SCALES,compute_local=True)
            else:
                gs={k:np.nan for k in ["xi50","d_star","x_d_star","lacunarity_star","x_lacunarity_star","local_delta_median","scaling_dB","quotient_mean_pairwise_KS","renorm_flow_retention_mean"]}
            rows.append({"run_id":str(run_id),"c1_config_id":str(r.c1_config_id),"seed_index":int(r.seed_index),
                         "activation":str(r.activation),"study_arm":str(r.study_arm),"input_index":int(i),"measure_defined":defined,**gs})
            atomic_csv(pd.DataFrame(rows),path); completed+=1;eta.update(time.perf_counter()-t0,completed)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    eta.finish(); return pd.DataFrame(rows)


def c1_input_population_comparison(input_geom: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_input_population_geometry_comparison.csv"
    if input_geom.empty or geometry.empty:
        out=pd.DataFrame();atomic_csv(out,path);return out
    ig=input_geom.groupby("run_id").agg(input_median_xi50=("xi50","median"),input_median_d_star=("d_star","median"),
        input_median_lacunarity_star=("lacunarity_star","median"),input_median_retention=("renorm_flow_retention_mean","median")).reset_index()
    pg=geometry[(geometry.geometry_source=="conditional_population")&(geometry.checkpoint.astype(str)=="best")][
        ["run_id","xi50","d_star","lacunarity_star","renorm_flow_retention_mean"]].copy()
    z=pg.merge(ig,on="run_id",how="inner"); rows=[]
    pairs=[("xi50","input_median_xi50"),("d_star","input_median_d_star"),("lacunarity_star","input_median_lacunarity_star"),("renorm_flow_retention_mean","input_median_retention")]
    for a,b in pairs:
        q=z[[a,b]].dropna();d=(q[a]-q[b]).to_numpy(float)
        if not len(d):continue
        try:W,p=wilcoxon(d,zero_method="wilcox")
        except Exception:W,p=np.nan,1.0
        rows.append({"metric":a,"n_models":len(d),"median_population_minus_median_input":float(np.median(d)),
                     "hodges_lehmann_shift":_hodges_lehmann_paired(d),"rank_biserial":_rank_biserial_signed(d),"wilcoxon_W":W,"p_raw":p})
    out=pd.DataFrame(rows)
    if len(out):out["p_holm"]=_holm_adjust(out.p_raw.fillna(1))
    atomic_csv(out,path);return out




def c1_geometry_rank_persistence(geometry: pd.DataFrame) -> pd.DataFrame:
    path=OUTDIR/"tables"/"c1_geometry_rank_persistence.csv"
    g=geometry[(geometry.study_arm=="c1_primary_relu")&(geometry.geometry_source=="conditional_population")].copy()
    if g.empty:
        out=pd.DataFrame();atomic_csv(out,path);return out
    g["epoch_numeric"]=g.checkpoint.astype(str).str.extract(r"(\d+)")[0].astype(float)
    # best is intentionally excluded because its epoch differs by architecture; common-state ranks use tXXX only.
    g=g[g.checkpoint.astype(str).str.startswith("t")]
    metrics=["xi50","d_star","lacunarity_star","renorm_flow_retention_mean","local_delta_median"]
    rows=[]
    for m in [x for x in metrics if x in g]:
        epochs=sorted(g.epoch_numeric.dropna().unique())
        for t0,t1 in itertools.combinations_with_replacement(epochs,2):
            rhos=[]
            for seed in MAIN_SEEDS:
                a=g[(g.seed_index==seed)&np.isclose(g.epoch_numeric,t0)].set_index("c1_config_id")[m]
                b=g[(g.seed_index==seed)&np.isclose(g.epoch_numeric,t1)].set_index("c1_config_id")[m]
                idx=a.index.intersection(b.index);av=a.loc[idx];bv=b.loc[idx];good=np.isfinite(av)&np.isfinite(bv)
                if int(np.sum(good))>=4:rhos.append(float(spearmanr(av[good],bv[good]).statistic))
            if rhos:rows.append({"metric":m,"epoch_a":int(t0),"epoch_b":int(t1),"n_seed_blocks":len(rhos),"rho_median":float(np.median(rhos)),"rho_mean":float(np.mean(rhos)),"rho_q25":float(np.quantile(rhos,.25)),"rho_q75":float(np.quantile(rhos,.75))})
    out=pd.DataFrame(rows);atomic_csv(out,path);return out


def _cochran_q_binary(X: np.ndarray) -> Tuple[float,float]:
    X=np.asarray(X,int); n,k=X.shape
    if n<2 or k<2:return np.nan,np.nan
    C=X.sum(axis=0);R=X.sum(axis=1);T=C.sum();den=k*T-np.sum(R*R)
    if den<=0:return np.nan,np.nan
    Q=float((k-1)*(k*np.sum(C*C)-T*T)/den)
    from scipy.stats import chi2
    return Q,float(chi2.sf(Q,k-1))


def c1_scaling_window_analysis(geometry: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    summary_path=OUTDIR/"tables"/"c1_scaling_window_summary.csv"
    db_path=OUTDIR/"tables"/"c1_scaling_dB_complete_block_friedman.csv"
    g=geometry[(geometry.study_arm=="c1_primary_relu")&(geometry.geometry_source=="conditional_population")&(geometry.checkpoint.astype(str)=="best")].copy()
    if g.empty:
        a=pd.DataFrame();b=pd.DataFrame();atomic_csv(a,summary_path);atomic_csv(b,db_path);return a,b
    g["scaling_credible_bool"]=g.scaling_credible.astype(bool)
    rates=g.groupby("c1_config_id").agg(n_seed_states=("scaling_credible_bool","size"),credible_rate=("scaling_credible_bool","mean"),median_dB_when_defined=("scaling_dB","median")).reset_index()
    piv=g.pivot_table(index="seed_index",columns="c1_config_id",values="scaling_credible_bool",aggfunc="first")
    levels=sorted(piv.columns.astype(str));comp=piv[levels].dropna()
    Q,pq=_cochran_q_binary(comp.astype(int).to_numpy()) if len(comp) else (np.nan,np.nan)
    rates["cochran_Q_all_architectures"]=Q;rates["cochran_p_all_architectures"]=pq;rates["cochran_complete_seed_blocks"]=len(comp)
    atomic_csv(rates,summary_path)
    # d_B omnibus only if entire architecture panel has credible windows inside at least 3 seed blocks.
    valid_seeds=[]
    for seed,sg in g.groupby("seed_index"):
        q=sg.set_index("c1_config_id").reindex(levels)
        if len(q)==len(levels) and q.scaling_credible_bool.fillna(False).all() and np.isfinite(q.scaling_dB).all():valid_seeds.append(int(seed))
    if len(valid_seeds)>=3:
        d=g[g.seed_index.isin(valid_seeds)].copy();r=_friedman_architecture(d,"scaling_dB");r["interpretation"]="conditional on all 20 architectures having a credible finite scaling window in the matched seed block";db=pd.DataFrame([r])
    else:
        db=pd.DataFrame([{"metric":"scaling_dB","n_seed_blocks":len(valid_seeds),"n_architectures":len(levels),"friedman_chi2":np.nan,"p_raw":np.nan,"kendall_W":np.nan,"interpretation":"omnibus not performed: fewer than 3 complete seed blocks had credible windows for every architecture"}])
    atomic_csv(db,db_path);return rates,db


def c1_posthoc_ridge_models(summary: pd.DataFrame, geometry: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Parsimonious post-hoc architecture model with leave-one-architecture-out prediction.

    This is explanatory only.  The inferential architecture result remains the matched Friedman/Kendall
    analysis.  Four predeclared architecture descriptors are used to limit overfitting at N_arch=20.
    """
    path=OUTDIR/"tables"/"c1_posthoc_ridge_models.csv"
    core=_c1_endpoint_table(summary[summary.study_arm=="c1_primary_relu"].copy(),geometry)
    metrics=["best_conditional_entropy_deficit_median","best_conditional_kl_to_structural_median","best_input_specialization_hidden_mean","best_condgeom_xi50","best_condgeom_lacunarity_star"]
    metrics=[m for m in metrics if m in core]
    desc=["hidden_depth","realized_parameters","log_topological_paths","bottleneck_ratio"]
    av=core.groupby("c1_config_id")[metrics].median(numeric_only=True).reset_index();pp=_refresh_arch_descriptors(panel)[["c1_config_id"]+desc].drop_duplicates("c1_config_id");z=av.merge(pp,on="c1_config_id")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut
    rows=[];alphas=np.logspace(-3,3,25)
    for m in metrics:
        q=z[[m]+desc].dropna();X=q[desc].to_numpy(float);y=q[m].to_numpy(float)
        if len(y)<8:continue
        Xs=StandardScaler().fit_transform(X);ys=(y-y.mean())/max(y.std(),EPS)
        loo=LeaveOneOut();best=None
        for a in alphas:
            pred=np.full(len(y),np.nan)
            for tr,te in loo.split(Xs):pred[te]=Ridge(alpha=float(a)).fit(Xs[tr],ys[tr]).predict(Xs[te])
            mse=float(np.mean((pred-ys)**2))
            if best is None or mse<best[0]:best=(mse,float(a),pred)
        mse,a,pred=best;ss=float(np.sum((ys-ys.mean())**2));r2=float(1-np.sum((ys-pred)**2)/ss) if ss>EPS else np.nan
        model=Ridge(alpha=a).fit(Xs,ys)
        rec={"metric":m,"n_architectures":len(y),"selected_alpha":a,"loocv_standardized_mse":mse,"loocv_R2":r2}
        for d,c in zip(desc,model.coef_):rec[f"coef_std_{d}"]=float(c)
        rows.append(rec)
    out=pd.DataFrame(rows);atomic_csv(out,path);return out

# ==================================================================================================
# 20. C1 DIAGNOSTIC FIGURES
# ==================================================================================================


def c1_savefig(fig,stem):
    fig.tight_layout();fig.savefig(OUTDIR/"figures"/f"{stem}.png",dpi=250,bbox_inches="tight");fig.savefig(OUTDIR/"figures"/f"{stem}.pdf",bbox_inches="tight");plt.close(fig)


def make_c1_figures(archive,panel,summary,trajectory,fp,geometry,bridge,rank_persist,tanh_eff):
    mpl.rcParams.update({"font.family":"serif","font.size":9,"axes.titlesize":10,"axes.labelsize":9,"pdf.fonttype":42})
    # 1 search coverage
    if not archive.empty:
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        axs[0].scatter(archive.hidden_depth,archive.realized_parameters,c=archive.discovery_objective,s=28,alpha=.65)
        if not panel.empty: axs[0].scatter(panel.hidden_depth,panel.realized_parameters,facecolors="none",edgecolors="k",s=70,label="frozen C1 panel")
        axs[0].set_yscale("log");axs[0].set_xlabel("hidden depth");axs[0].set_ylabel("parameters");axs[0].legend(frameon=False)
        axs[1].scatter(archive.search_index,archive.discovery_objective,s=22,alpha=.7);axs[1].set_xlabel("RF-SMAC evaluation");axs[1].set_ylabel("discovery objective")
        axs[1].set_title("RF-SMAC architecture exploration");c1_savefig(fig,"figure01_rf_smac_architecture_exploration")
    core=_attach_best_geometry(summary[summary.study_arm=="c1_primary_relu"].copy(),geometry) if not summary.empty else pd.DataFrame()
    if not core.empty:
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for ax,m,title in zip(axs.flat,["best_conditional_entropy_deficit_median","best_conditional_edge_cv_median","best_conditional_kl_to_structural_median","best_input_specialization_hidden_mean"],
                              ["Conditional path concentration","Conditional edge heterogeneity","Departure from structural disposition","Input specialization"]):
            med=core.groupby("c1_config_id")[m].median().sort_values();ax.bar(np.arange(len(med)),med.values);ax.set_xticks(np.arange(len(med)),med.index,rotation=70,fontsize=7);ax.set_title(title)
        c1_savefig(fig,"figure02_architecture_endpoint_atlas")
    if not trajectory.empty:
        tr=trajectory[trajectory.study_arm=="c1_primary_relu"]
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for ax,m,title in zip(axs.flat,["conditional_entropy_deficit_median","conditional_edge_cv_median","conditional_kl_to_structural_median","input_specialization_hidden_mean"],
                              ["Entropy deficit","Edge-flow CV","KL to structural","Input specialization"]):
            for arch,g in tr.groupby("c1_config_id"):
                q=g.groupby("epoch")[m].median().sort_index();ax.plot(q.index,q.values,lw=.8,alpha=.45)
            ax.set_xscale("symlog",linthresh=1);ax.set_xlabel("epoch");ax.set_title(title)
        c1_savefig(fig,"figure03_common_state_architecture_trajectories")
    if fp is not None and not fp.empty:
        f=fp[fp.study_arm=="c1_primary_relu"]
        fig,ax=plt.subplots(figsize=(7,4))
        for arch,g in f.groupby("c1_config_id"):
            q=g.groupby("target_functional_progress")["conditional_entropy_deficit_median"].median().sort_index();ax.plot(q.index,q.values,lw=.9,alpha=.55)
        ax.set_xlabel("functional progress");ax.set_ylabel("conditional entropy deficit");ax.set_title("Architecture states at matched functional progress");c1_savefig(fig,"figure04_functional_progress_architecture_trajectories")
    if bridge is not None and not bridge.empty:
        fig,ax=plt.subplots(figsize=(7,4));med=bridge.groupby("metric").architecture_rank_spearman_structural_vs_realized.median().sort_values();ax.barh(np.arange(len(med)),med.values);ax.set_yticks(np.arange(len(med)),med.index);ax.set_xlim(-1,1);ax.axvline(0,lw=.7,color="k");ax.set_xlabel("median structural-realized architecture-rank Spearman")
        c1_savefig(fig,"figure05_structural_realized_architecture_bridge")
    if rank_persist is not None and not rank_persist.empty:
        m="conditional_entropy_deficit_median";d=rank_persist[rank_persist.metric==m];epochs=sorted(set(d.epoch_a)|set(d.epoch_b));M=np.full((len(epochs),len(epochs)),np.nan);idx={e:i for i,e in enumerate(epochs)}
        for r in d.itertuples():M[idx[r.epoch_a],idx[r.epoch_b]]=r.rho_median;M[idx[r.epoch_b],idx[r.epoch_a]]=r.rho_median
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(M,vmin=-1,vmax=1,cmap="coolwarm");ax.set_xticks(range(len(epochs)),epochs,rotation=45);ax.set_yticks(range(len(epochs)),epochs);ax.set_title("Architecture-rank persistence: conditional entropy deficit");fig.colorbar(im,ax=ax,label="median Spearman rho");c1_savefig(fig,"figure06_architecture_rank_persistence")
    if not geometry.empty:
        g=geometry[(geometry.geometry_source=="conditional_population")&(geometry.checkpoint.astype(str)=="best")]
        if not g.empty:
            fig,axs=plt.subplots(1,3,figsize=(11,3.5))
            for ax,m,t in zip(axs,["xi50","d_star","lacunarity_star"],[r"$\xi_{50}$",r"$d^*$",r"peak $\Lambda_F$"]):
                q=g[g.study_arm=="c1_primary_relu"].groupby("c1_config_id")[m].median().sort_values();ax.bar(np.arange(len(q)),q.values);ax.set_xticks(np.arange(len(q)),q.index,rotation=70,fontsize=6);ax.set_title(t)
            c1_savefig(fig,"figure07_architecture_conditional_BCANw")
    if tanh_eff is not None and not tanh_eff.empty:
        fig,ax=plt.subplots(figsize=(7,4));d=tanh_eff.sort_values("rank_biserial");ax.barh(np.arange(len(d)),d.rank_biserial);ax.set_yticks(np.arange(len(d)),[x.replace("best_","") for x in d.metric]);ax.axvline(0,lw=.7,color="k");ax.set_xlim(-1,1);ax.set_xlabel("paired ReLU - tanh rank-biserial effect");ax.set_title("C1 smooth-activation sentinel")
        c1_savefig(fig,"figure08_tanh_sentinel_effects")


# ==================================================================================================
# 21. C1 MANIFEST + MAIN ORCHESTRATION
# ==================================================================================================


def write_c1_experiment_manifest(archive,panel,sentinels,manifest):
    obj={
        "work":"II","programme":"MLP conditional realization","phase":"C1","profile":C1_PROFILE,"device":DEVICE,
        "scientific_question":"architecture-wide conditional realization in MLPs",
        "primary_observable":"finite conditional complete-path measure p_Delta(pi|x,x0)",
        "structural_companion":"weight-only complete-path disposition p_str(pi)",
        "RF_SMAC":{"evaluations":RFSMAC_TOTAL_CANDIDATES,"sobol":RFSMAC_SOBOL_COLD,"rf_ei":RFSMAC_BO_STEPS,
                   "discovery_activation":PRIMARY_ACTIVATION,"discovery_epochs":RFSMAC_DISCOVERY_EPOCHS,
                   "objective":"validation MSE trajectory only; no path-flow quantity enters acquisition"},
        "architecture_space":{"depth":[MIN_HIDDEN_DEPTH,MAX_HIDDEN_DEPTH],"width":[MIN_WIDTH,MAX_WIDTH],
                              "parameter_budget":[MIN_PARAMETER_BUDGET,MAX_PARAMETER_BUDGET],"morphologies":MORPHOLOGIES,
                              "frozen_panel":int(len(panel)),"tanh_sentinels":int(len(sentinels))},
        "primary_factorial":{"architectures":int(len(panel)),"seeds":MAIN_SEEDS,"activation":"relu","optimizer":"adam","learning_rate":FIXED_ADAM_LR,
                             "environment":"C1_CANON","task":"regression","runs":int(np.sum(manifest.study_arm=="c1_primary_relu"))},
        "tanh_sentinel":{"architectures":int(len(sentinels)),"seeds":MAIN_SEEDS,"runs":int(np.sum(manifest.study_arm=="c1_tanh_sentinel"))},
        "training":{"epochs":MAX_EPOCH,"batch_size":BATCH_SIZE,"weight_decay":WEIGHT_DECAY,"scheduler":None,"early_stopping":False,
                    "trajectory_checkpoints":TRAJECTORY_CHECKPOINTS,"geometry_checkpoints":TRAJECTORY_GEOMETRY_CHECKPOINTS,"bias":MODEL_BIAS},
        "conditions":{"reference":"x0=0 after input standardization","endpoint_conditions":ENDPOINT_INPUTS,"trajectory_conditions":TRAJECTORY_INPUTS,
                      "input_resolved_geometries_per_model":INPUT_GEOM_INPUTS},
        "inference":{"architecture_primary":"Friedman repeated blocks + Kendall W_A","blocks":"8 matched seeds",
                     "secondary":["rank persistence","structural-realized architecture-rank bridge","descriptor Spearman associations","direction survival","functional-progress Friedman","paired tanh sentinel"]},
        "boundaries":["RF-SMAC discovery rows excluded from architecture inference","inputs nested within runs","BCANw downstream of measure/flow","strict dB interpreted only within credible finite windows"]
    }
    atomic_json(obj,OUTDIR/"experiment_manifest.json")


def run_c1_experiment():
    total_start=time.perf_counter();banner("WORK II — MLP CONDITIONAL REALIZATION — C1 ARCHITECTURE-WIDE STUDY")
    print(f"Profile: {C1_PROFILE} | Device: {DEVICE} | horizon: {MAX_EPOCH} epochs | bias={MODEL_BIAS}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    print("Object pair: p_str(pi)  <->  p_Delta(pi|x,x0).  Architecture is the principal varied factor.")

    banner("C1 RF-SMAC architecture exploration")
    archive=run_rf_smac_discovery(); panel=select_c1_architecture_panel(archive); sentinels=select_c1_tanh_sentinels(panel)
    print(panel[["c1_config_id","hidden_depth","realized_parameters","morphology","width_cv","bottleneck_ratio","expansion_index","hidden_widths"]].to_string(index=False))

    if C1_RUN_EXACT_PREFLIGHT:
        banner("Exact finite-response preflight")
        exact=run_exact_validation()
        check_cols=[c for c in ["reconstruction_abs_error","partition_rel_error","entropy_abs_error","edge_flow_max_abs_error","node_flow_max_abs_error"] if c in exact]
        worst=float(np.nanmax(exact[check_cols].to_numpy(float))) if (not exact.empty and check_cols) else 0.0
        print(f"Worst exactness residual: {worst:.3e}")
        if worst>EXACT_PREFLIGHT_TOL:
            raise RuntimeError(f"Exact path preflight failed: residual {worst:.3e} > {EXACT_PREFLIGHT_TOL:.1e}")
    else: exact=pd.DataFrame()

    manifest=build_c1_manifest(panel,sentinels);write_c1_experiment_manifest(archive,panel,sentinels,manifest)
    print(f"Primary ReLU runs: {int(np.sum(manifest.study_arm=='c1_primary_relu'))}")
    print(f"Tanh sentinel runs: {int(np.sum(manifest.study_arm=='c1_tanh_sentinel'))}")
    print(f"Total full trajectories: {len(manifest)}")
    env_cache={"C1_CANON":generate_environment("C1_CANON",discovery=False)}

    banner(f"C1 full {MAX_EPOCH}-epoch publication trajectories")
    run_c1_manifest(manifest,env_cache)

    banner("Aggregate C1 run parts")
    summary,endpoints,history,tr_raw,geometry,scales=aggregate_run_parts(manifest)
    # Restore explicit C1 identifier in inherited part tables if old helper fields only contain b2_config_id.
    for df in [summary,endpoints,history,tr_raw,geometry,scales]:
        if not df.empty and "c1_config_id" not in df and "b2_config_id" in df: df["c1_config_id"]=df["b2_config_id"].astype(str)
    # Re-save any enriched aggregate tables.
    atomic_csv(summary,OUTDIR/"tables"/"run_summary.csv");atomic_csv(endpoints,OUTDIR/"tables"/"endpoint_input_summary.csv")
    atomic_csv(history,OUTDIR/"tables"/"training_history.csv");atomic_csv(tr_raw,OUTDIR/"tables"/"trajectory_summary_raw.csv")
    atomic_csv(geometry,OUTDIR/"tables"/"trajectory_geometry.csv");atomic_csv(scales,OUTDIR/"tables"/"trajectory_geometry_scales.csv")

    banner("Equal-functional-progress and late-training C1 tables")
    trajectory,fp,late=build_functional_progress_tables(history,tr_raw)
    for df,path in [(trajectory,OUTDIR/"tables"/"trajectory_summary.csv"),(fp,OUTDIR/"tables"/"functional_progress_summary.csv"),(late,OUTDIR/"tables"/"late_training_summary.csv")]:
        if not df.empty and "c1_config_id" not in df and "b2_config_id" in df:df["c1_config_id"]=df["b2_config_id"].astype(str);atomic_csv(df,path)

    banner("C1 input-resolved conditional geometry")
    input_geom=run_c1_input_resolved_geometry(manifest,env_cache)

    banner("C1 architecture-wide inferential tables")
    arch_stats=c1_architecture_statistics(summary,geometry)
    rank_persist=c1_rank_persistence(trajectory)
    geom_rank_persist=c1_geometry_rank_persistence(geometry)
    bridge=c1_structural_realized_bridge(summary,geometry)
    descriptor_assoc=c1_descriptor_associations(summary,geometry,panel)
    ridge_models=c1_posthoc_ridge_models(summary,geometry,panel)
    direction=c1_direction_survival(trajectory)
    fp_stats=c1_functional_progress_stats(fp)
    tanh_eff,tanh_rank=c1_tanh_statistics(summary,geometry)
    scaling_rates,scaling_db=c1_scaling_window_analysis(geometry)
    pop_input=c1_input_population_comparison(input_geom,geometry)
    print(arch_stats[[c for c in ["metric","n_seed_blocks","n_architectures","kendall_W","p_holm"] if c in arch_stats]].to_string(index=False) if not arch_stats.empty else "No architecture statistics")

    banner("C1 diagnostic figure atlas")
    make_c1_figures(archive,panel,summary,trajectory,fp,geometry,bridge,rank_persist,tanh_eff)

    elapsed=time.perf_counter()-total_start
    report=["WORK II MLP CONDITIONAL REALIZATION — C1 COMPLETE",f"Profile: {C1_PROFILE}",f"Device: {DEVICE}",f"Wall time: {human_time(elapsed)}",
            f"RF-SMAC evaluations: {len(archive)} / {RFSMAC_TOTAL_CANDIDATES}",f"Frozen architectures: {len(panel)}",f"Tanh sentinels: {len(sentinels)}",
            f"Full trajectories summarized: {len(summary)} / {len(manifest)}",f"Trajectory rows: {len(trajectory)}",f"Endpoint input rows: {len(endpoints)}",
            f"Geometry rows: {len(geometry)}",f"Input-resolved geometry rows: {len(input_geom)}",f"Functional-progress rows: {len(fp)}",f"Outputs: {OUTDIR.resolve()}"]
    (OUTDIR/"experiment_summary.txt").write_text("\n".join(report)+"\n",encoding="utf-8");print("\n"+"\n".join(report))
    return {"rf_smac_archive":archive,"architecture_panel":panel,"tanh_sentinels":sentinels,"manifest":manifest,
            "summary":summary,"endpoint_inputs":endpoints,"history":history,"trajectory":trajectory,"functional_progress":fp,"late_training":late,
            "geometry":geometry,"geometry_scales":scales,"input_geometry":input_geom,"architecture_friedman":arch_stats,
            "rank_persistence":rank_persist,"geometry_rank_persistence":geom_rank_persist,"structural_realized_bridge":bridge,"descriptor_associations":descriptor_assoc,"posthoc_ridge_models":ridge_models,"direction_survival":direction,
            "functional_progress_friedman":fp_stats,"tanh_effects":tanh_eff,"tanh_rank_preservation":tanh_rank,"scaling_window_summary":scaling_rates,"scaling_dB_complete_blocks":scaling_db,"population_input_comparison":pop_input,"exact":exact}


# Run immediately when pasted into one Jupyter cell.
C1_RESULTS = run_c1_experiment()
