# PATH-FLOW PHASE III — OPTIMIZER DYNAMICS + KINEMATICS OF PATH-MEASURE REDISTRIBUTION
# ======================================================================================
# Single-file / single-Jupyter-cell experiment.
#
# Scientific objective
# --------------------
# Phase I established architecture-wide path-flow states; Phase II tested whether their
# dominant directions survive task-environment variation.  Phase III asks what dynamical
# law generates those directions and whether it is optimizer-dependent.
#
# The central finite-step object is the complete-path log-growth field
#
#     G_pi = log[Q_pi(t1)/Q_pi(t0)],       Q_pi = prod_{e in pi} w_e^2,
#
# under which the normalized complete-path measure obeys the exact exponential tilt
#
#     p_pi(t1) = p_pi(t0) exp(G_pi) / E_{p_t0}[exp(G)].
#
# This script therefore measures both STATE variables inherited from Phases I/II and
# DYNAMIC variables between states.  No path enumeration is performed: all path moments,
# conserved edge/node flows, KL displacements, and selection-alignment statistics are
# evaluated by layerwise dynamic programming.
#
# Phase III-A — Adam mechanistic map
#   6 architecture representatives x 6 Phase-II environments x 5 seeds = 180 Adam runs.
#
# Phase III-B — optimizer universality
#   a nested 5 x 5 panel x 4 optimizers x 5 seeds = 500 runs.  The 125 Adam runs in this
#   nested panel are REUSED from Phase III-A, so the union is 555 unique main runs.
#
# Optimizers: Adam, SGD, SGD+momentum, RMSProp.  Weight decay is fixed at zero.  AdamW is
# intentionally excluded from the primary comparison because decoupled weight decay adds a
# distinct dynamical pressure.
#
# Fair-comparison principles
# --------------------------
# * For a given architecture/seed, every optimizer starts from EXACTLY the same He/Kaiming
#   initialization (the same Phase-II seeding rule).
# * For a given environment/seed, every optimizer sees EXACTLY the same deterministic batch
#   permutations (the same Phase-II data-order rule).
# * Adam is locked to lr=1e-3 exactly as in Phases I/II. Optional learning-rate calibration for
#   non-Adam comparators uses TRAINING LOSS ONLY; no validation/test metric and no path-flow
#   descriptor is used to choose an optimizer learning rate.
# * Comparisons are retained in epoch time, optimizer-step time, functional-progress time,
#   and cumulative structural time tau_J = sum Jeffreys path-measure displacement.
#
# Primary dynamic observables
# ---------------------------
#   Var_p(G)                         path-selection intensity
#   Cov_p(G, log p), Corr_p(G,log p) selection alignment
#   D_KL(p1||p0), D_KL(p0||p1)      directed measure displacement
#   J = D_KL(p1||p0)+D_KL(p0||p1)  symmetric redistribution distance
#   Delta S_deficit                  net concentration/equalization direction
#   tau_J                            cumulative structural exposure
#
# For each finite interval the code additionally evaluates the exact identity
#
#   Delta S_deficit = integral_0^1 Cov_{p_a}(G, log p_a) da,
#   p_a proportional to p_0 exp(a G),
#
# by Gauss-Legendre quadrature.  This turns the continuous-time covariance law into a
# finite-interval numerical consistency test without enumerating paths.
#
# Geometry inherited from Phases I/II
# ------------------------------------
#   eta_1, eta_2, S_deficit, edge/node flow CV,
#   xi_50, strict finite-window d_B^F, d*, x_d*, flow lacunarity,
#   final strict common-window local Delta_i^F, directed quotient-flow KS/retention.
#
# Research handoff
# ----------------
# Research mode REQUIRES the actual Phase-II outputs:
#   pathflow_phase2_environment_universality_results/tables/architecture_panel.csv
#   pathflow_phase2_environment_universality_results/tables/selected_environments.csv
# and uses final_summary.csv when present to make the 6x6 panel selection phenotype-aware.
# No fabricated research architectures/environments are substituted.  Pilot mode has a tiny
# explicitly-labelled fallback solely for orchestration/smoke testing.
#
# Paste this entire file into ONE Jupyter cell and run.

from __future__ import annotations

import os
import math
import json
import time
import random
import hashlib
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

from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from scipy.stats import ks_2samp, spearmanr, binomtest, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================================================================================
# P3.1 CONFIGURATION
# ======================================================================================

P3_PROFILE = os.environ.get("PATHFLOW_PHASE3_PROFILE", "research").strip().lower()
P3_SEED = 53017
MODEL_INIT_SEED = 31415
PHASE2_DATA_BASE_SEED = 880301
PHASE2_DATA_ORDER_BASE_SEED = 161803
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Phase-II handoff.  Override PATHFLOW_PHASE2_DIR if the result folder lives elsewhere.
P3_PHASE2_DIR = Path(os.environ.get(
    "PATHFLOW_PHASE2_DIR", "pathflow_phase2_environment_universality_results"
))
P3_ARCH_CSV = P3_PHASE2_DIR / "tables" / "architecture_panel.csv"
P3_ENV_CSV = P3_PHASE2_DIR / "tables" / "selected_environments.csv"
P3_PHASE2_FINAL_CSV = P3_PHASE2_DIR / "tables" / "final_summary.csv"
P3_REQUIRE_HANDOFF = P3_PROFILE == "research"

P3_OUTDIR = Path(os.environ.get(
    "PATHFLOW_PHASE3_OUTDIR", "pathflow_phase3_optimizer_dynamics_results"
))
for sub in [
    "tables", "figures", "run_state", "run_dynamics", "run_geometry",
    "run_micro", "checkpoints", "calibration"
]:
    (P3_OUTDIR / sub).mkdir(parents=True, exist_ok=True)

P3_RESUME = True
P3_START_FRESH = False

# A single optimizer/environment cell must not destroy the entire factorial sweep.
# Genuine numerical optimizer divergence is recorded as a run-level outcome and the sweep
# continues; unexpected programming errors still abort immediately.
P3_CONTINUE_ON_RUN_DIVERGENCE = True
P3_RETRY_RECORDED_DIVERGENCES = False

# Same controlled synthetic regression family as Phase II.
P3_INPUT_DIM = 32
P3_N_TRAIN = 60_000
P3_N_VAL = 10_000
P3_N_TEST = 10_000
P3_BATCH_SIZE = 1024
P3_MAX_EPOCH = 400
P3_WEIGHT_DECAY = 0.0

# Two nested arms.  The union contains 555 unique research runs with five seeds.
P3_MECH_ARCHITECTURES = 6
P3_MECH_ENVIRONMENTS = 6
P3_UNIV_ARCHITECTURES = 5
P3_UNIV_ENVIRONMENTS = 5
P3_SEEDS = [0, 1, 2, 3, 4]
P3_OPTIMIZERS = ["adam", "sgd", "sgd_momentum", "rmsprop"]

# Dense path-state checkpoints are cheap; BCANw geometry is deliberately sparser.
P3_ADAM_STATE_CHECKPOINTS = [
    0, 1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 50, 60, 80,
    120, 160, 200, 250, 300, 350, 399
]
P3_OTHER_STATE_CHECKPOINTS = [0, 1, 2, 5, 8, 12, 20, 40, 80, 120, 200, 300, 399]
P3_GEOMETRY_CHECKPOINTS = [0, 5, 20, 80, 200, 399]
# Extra BCANw checkpoints only for the microscopic sentinel arm.
P3_SENTINEL_GEOMETRY_CHECKPOINTS = [0, 1, 2, 5, 8, 12, 16, 20, 30, 40, 60, 80, 120, 160, 200, 300, 399]
P3_BCANW_SCALES = 72
P3_SENTINEL_BCANW_SCALES = 72

# First-step / short-step validation arm: one maximin architecture, two environments,
# all four optimizers and all seeds.  These are NOT extra runs; diagnostics are collected
# inside the corresponding main trajectories.
P3_MICRO_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
P3_MICRO_ENVIRONMENTS = 2

# Optional training-only LR calibration for NON-ADAM comparators. Adam is locked to the
# exact Phase-I/II operating point (lr=1e-3) so the mechanistic arm is a true continuation.
# Comparator calibration is a frozen pre-experiment calibration, not HPO over validation/test/path-flow outcomes.
P3_CALIBRATE_LR = True
P3_CALIBRATE_OPTIMIZERS = ["sgd", "sgd_momentum", "rmsprop"]
P3_CALIBRATION_EPOCHS = 12
P3_CALIBRATION_ARCHITECTURES = 2
P3_CALIBRATION_ENVIRONMENTS = 2
P3_CALIBRATION_SEEDS = [0]
P3_LR_GRIDS = {
    "adam": [3e-4, 1e-3, 3e-3],
    "rmsprop": [3e-4, 1e-3, 3e-3],
    "sgd": [1e-3, 3e-3, 1e-2, 3e-2],
    "sgd_momentum": [1e-3, 3e-3, 1e-2, 3e-2],
}
P3_FIXED_LR = {
    "adam": 1e-3,
    "rmsprop": 1e-3,
    "sgd": 1e-2,
    "sgd_momentum": 1e-2,
}

# Path-flow / geometry constants kept consistent with Phases I/II.
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

# Strict common-window local geometry, exactly as in the later Phase-I/II pipeline.
LOCAL_COMMON_MIN_POINTS = 7
LOCAL_COMMON_MIN_SPAN_DECADES = 0.50
LOCAL_COMMON_SUPPORT_GROWTH = 4.0
LOCAL_COMMON_MASS_GROWTH = 3.0
LOCAL_COMMON_R2_MIN = 0.95
LOCAL_COMMON_SATURATION = 0.95
P3_COMPUTE_FINAL_LOCAL = True
P3_COMPUTE_FINAL_QUOTIENT = True

# Finite-tilt entropy identity: 5-point Gauss-Legendre is already extremely accurate for
# the smooth interpolation p_a proportional to p_0 exp(aG).
P3_TILT_QUADRATURE_POINTS = 5

# Pilot profile is only an orchestration test.
if P3_PROFILE == "pilot":
    P3_N_TRAIN, P3_N_VAL, P3_N_TEST = 2_000, 500, 500
    P3_MAX_EPOCH = 4
    P3_MECH_ARCHITECTURES = 2
    P3_MECH_ENVIRONMENTS = 2
    P3_UNIV_ARCHITECTURES = 2
    P3_UNIV_ENVIRONMENTS = 2
    P3_SEEDS = [0]
    P3_OPTIMIZERS = ["adam", "sgd"]
    P3_ADAM_STATE_CHECKPOINTS = [0, 1, 2, 4]
    P3_OTHER_STATE_CHECKPOINTS = [0, 1, 2, 4]
    P3_GEOMETRY_CHECKPOINTS = [0, 2, 4]
    P3_SENTINEL_GEOMETRY_CHECKPOINTS = [0, 1, 2, 4]
    P3_BCANW_SCALES = 18
    P3_SENTINEL_BCANW_SCALES = 18
    P3_MICRO_STEPS = [0, 1, 2, 4]
    P3_MICRO_ENVIRONMENTS = 1
    P3_CALIBRATE_LR = False
    P3_COMPUTE_FINAL_LOCAL = False

random.seed(P3_SEED)
np.random.seed(P3_SEED)
torch.manual_seed(P3_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(P3_SEED)
torch.set_float32_matmul_precision("high")

if P3_START_FRESH:
    for sub in ["run_state", "run_dynamics", "run_geometry", "run_micro", "checkpoints", "calibration"]:
        for p in (P3_OUTDIR / sub).glob("*"):
            if p.is_file():
                try: p.unlink()
                except Exception: pass

# ======================================================================================
# P3.2 TRACEABILITY / GENERIC HELPERS
# ======================================================================================


class RunDivergenceError(RuntimeError):
    """Expected run-level numerical failure (optimizer divergence or zero path support).

    These failures are scientifically meaningful operating-point outcomes and are recorded
    without terminating the full Phase-III factorial. Unexpected exceptions are NOT caught.
    """


def _rescale_path_layers(A: List[np.ndarray], context: str = "path measure"):
    """Return per-layer positively rescaled matrices and their log scales.

    Multiplying every edge in one layer by the same positive scalar multiplies every complete
    path gain by the same scalar and therefore leaves the normalized path measure, edge/node
    marginals, BCANw flow enrichment, and all selection covariances unchanged.  We exploit
    this exact invariance to prevent false `Degenerate forward path partition` failures when
    an optimizer makes an entire layer numerically small.
    """
    out, log_scales = [], []
    for l, M in enumerate(A):
        M = np.asarray(M, np.float64)
        if M.ndim != 2:
            raise RunDivergenceError(f"{context}: layer {l} is not a matrix (shape={M.shape}).")
        if not np.all(np.isfinite(M)):
            bad = int(np.size(M) - np.count_nonzero(np.isfinite(M)))
            raise RunDivergenceError(f"{context}: layer {l} contains {bad} non-finite path factors.")
        if np.any(M < 0):
            raise RunDivergenceError(f"{context}: layer {l} contains negative path factors.")
        scale = float(np.max(M)) if M.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            raise RunDivergenceError(
                f"{context}: layer {l} has zero/invalid total path support (max factor={scale!r})."
            )
        out.append(M / scale)
        log_scales.append(float(math.log(scale)))
    return out, log_scales


def banner(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def human_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    s = int(round(seconds)); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


class ETAReporter:
    def __init__(self, stage: str, total: int, window: int = 12):
        self.stage, self.total, self.window = stage, max(int(total), 1), max(int(window), 3)
        self.start = time.perf_counter(); self.durations = []
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
    clean = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
    return hashlib.sha1(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:n]


def atomic_json(obj: Any, path: Path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8"); tmp.replace(path)


def atomic_csv(df: pd.DataFrame, path: Path):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False); tmp.replace(path)


def r2_score_np(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    den = float(np.sum((y-y.mean())**2))
    return float(1.0 - np.sum((y-pred)**2)/den) if den > EPS else np.nan


def safe_log(x):
    try:
        x = float(x)
        return float(math.log(max(x, EPS))) if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def parse_widths(x) -> List[int]:
    if isinstance(x, str):
        return list(map(int, json.loads(x)))
    return list(map(int, x))


def model_seed(seed_idx: int) -> int:
    # Exact Phase-II rule.
    return int(MODEL_INIT_SEED + 1_000_003 * int(seed_idx))


def order_seed(seed_idx: int) -> int:
    # Exact Phase-II rule.
    return int(PHASE2_DATA_ORDER_BASE_SEED + 104_729 * int(seed_idx))


def epoch_batches(n: int, epoch: int, seed_idx: int = 0):
    # Exact Phase-II rule: optimizer-independent permutation for paired comparisons.
    rng = np.random.default_rng(order_seed(seed_idx) + int(epoch))
    ids = rng.permutation(int(n))
    for lo in range(0, len(ids), P3_BATCH_SIZE):
        yield ids[lo:min(lo+P3_BATCH_SIZE, len(ids))]


def farthest_points(Z: np.ndarray, n: int, seeds: Optional[Sequence[int]] = None) -> List[int]:
    Z = np.asarray(Z, float); n = min(int(n), len(Z))
    selected = list(dict.fromkeys(int(i) for i in (seeds or []) if 0 <= int(i) < len(Z)))[:n]
    if n <= 0: return []
    if not selected:
        center = np.nanmean(Z, axis=0)
        selected = [int(np.nanargmax(np.linalg.norm(Z-center[None,:], axis=1)))]
    while len(selected) < n:
        rem = [i for i in range(len(Z)) if i not in selected]
        if not rem: break
        d = [min(np.linalg.norm(Z[i]-Z[j]) for j in selected) for i in rem]
        selected.append(int(rem[int(np.argmax(d))]))
    return selected[:n]


def standardize_frame(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(df), 1), float)
    X = df[columns].replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    return StandardScaler().fit_transform(X)

# Journal-like restrained plotting style.
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9.0,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.dpi": 130,
    "savefig.dpi": 400, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ======================================================================================
# P3.3 MODEL + PHASE-II ENVIRONMENT GENERATOR
# ======================================================================================


class BiasFreeReLUMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int = 1, seed: int = MODEL_INIT_SEED):
        super().__init__(); torch.manual_seed(int(seed))
        self.widths = [int(in_dim)] + [int(x) for x in hidden] + [int(out_dim)]
        self.linears = nn.ModuleList([
            nn.Linear(self.widths[i], self.widths[i+1], bias=False)
            for i in range(len(self.widths)-1)
        ])
        for layer in self.linears:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears)-1:
                x = torch.relu(x)
        return x


def extract_weights(model: BiasFreeReLUMLP) -> List[np.ndarray]:
    weights = [x.weight.detach().cpu().numpy().astype(np.float64).copy() for x in model.linears]
    for l, W in enumerate(weights):
        if not np.all(np.isfinite(W)):
            bad = int(W.size - np.count_nonzero(np.isfinite(W)))
            finite = W[np.isfinite(W)]
            maxabs = float(np.max(np.abs(finite))) if finite.size else np.nan
            raise RunDivergenceError(
                f"model weights became non-finite in layer {l}: bad={bad}/{W.size}, "
                f"max|finite w|={maxabs:g}"
            )
    return weights


def environment_cfg_from_row(er) -> Dict[str, Any]:
    return {
        "teacher_family": str(er.teacher_family),
        "informative_dim": int(er.informative_dim),
        "feature_correlation": float(er.feature_correlation),
        "snr": float(er.snr),
        "interaction_strength": float(er.interaction_strength),
    }


def environment_seed(cfg: Dict[str, Any]) -> int:
    return int(PHASE2_DATA_BASE_SEED + int(stable_hash_dict(cfg, 8), 16) % 1_000_000_007)


def correlated_gaussian(rng, n, d, rho):
    eps = rng.standard_normal((n, d), dtype=np.float32)
    if rho <= 0: return eps
    g = rng.standard_normal((n, 1), dtype=np.float32)
    return (math.sqrt(1-rho)*eps + math.sqrt(rho)*g).astype(np.float32)


def teacher_target(X: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed+17)
    dinf = int(cfg["informative_dim"]); family = str(cfg["teacher_family"])
    kappa = float(cfg["interaction_strength"]); Xin = X[:, :dinf].astype(np.float32)
    if family == "linear":
        w = rng.standard_normal((dinf, 1)).astype(np.float32)/math.sqrt(max(dinf, 1))
        base = Xin @ w
    else:
        hidden = [64] if family == "shallow_nonlinear" else [128, 64, 32]
        teacher = BiasFreeReLUMLP(dinf, hidden, 1, seed=seed+101).to(DEVICE)
        teacher.eval(); base = np.empty((len(Xin), 1), np.float32)
        with torch.no_grad():
            for lo in range(0, len(Xin), 8192):
                hi = min(lo+8192, len(Xin))
                base[lo:hi] = teacher(torch.from_numpy(Xin[lo:hi]).to(DEVICE)).cpu().numpy()
        del teacher
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if dinf >= 2 and kappa > 0:
        pairs = min(16, dinf//2); ids = np.arange(2*pairs).reshape(pairs, 2)
        inter = np.mean(Xin[:, ids[:,0]]*Xin[:, ids[:,1]], axis=1, keepdims=True).astype(np.float32)
        base = ((1-kappa)*(base/(np.std(base)+1e-8)) +
                kappa*(inter/(np.std(inter)+1e-8)))
    return np.asarray(base, np.float32)


def generate_environment(er):
    cfg = environment_cfg_from_row(er); seed = environment_seed(cfg)
    rng = np.random.default_rng(seed); n = P3_N_TRAIN + P3_N_VAL + P3_N_TEST
    X = correlated_gaussian(rng, n, P3_INPUT_DIM, float(cfg["feature_correlation"]))
    xm = X[:P3_N_TRAIN].mean(0, keepdims=True)
    xs = X[:P3_N_TRAIN].std(0, keepdims=True) + 1e-8
    X = ((X-xm)/xs).astype(np.float32)
    clean = teacher_target(X, cfg, seed)
    clean = ((clean-clean[:P3_N_TRAIN].mean())/(clean[:P3_N_TRAIN].std()+1e-8)).astype(np.float32)
    noise_sd = math.sqrt(1/max(float(cfg["snr"]), EPS))
    y = clean + rng.normal(0, noise_sd, size=clean.shape).astype(np.float32)
    y = ((y-y[:P3_N_TRAIN].mean())/(y[:P3_N_TRAIN].std()+1e-8)).astype(np.float32)
    splits = {
        "train": (torch.from_numpy(X[:P3_N_TRAIN]), torch.from_numpy(y[:P3_N_TRAIN])),
        "val": (torch.from_numpy(X[P3_N_TRAIN:P3_N_TRAIN+P3_N_VAL]),
                torch.from_numpy(y[P3_N_TRAIN:P3_N_TRAIN+P3_N_VAL])),
        "test": (torch.from_numpy(X[P3_N_TRAIN+P3_N_VAL:]),
                 torch.from_numpy(y[P3_N_TRAIN+P3_N_VAL:])),
    }
    return splits, {"environment_seed": seed, "noise_sd": noise_sd, **cfg}


@torch.no_grad()
def eval_mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor, chunk: int = 8192) -> float:
    model.eval(); total = 0.0; count = 0
    for lo in range(0, len(X), chunk):
        hi = min(lo+chunk, len(X)); xb = X[lo:hi].to(DEVICE, non_blocking=True)
        yb = y[lo:hi].to(DEVICE, non_blocking=True); pred = model(xb)
        total += float(torch.sum((pred-yb)**2).item()); count += int(yb.numel())
    return total/max(count, 1)

# ======================================================================================
# P3.4 PHASE-II HANDOFF + MAXIMIN PANEL SELECTION
# ======================================================================================


def pilot_handoff():
    arch = pd.DataFrame([
        {"arch_id":"PILOT_A0","hidden_widths":json.dumps([24,20,16]),"depth":3,
         "morphology":"taper","actual_params":32*24+24*20+20*16+16,"width_mean":20.,"width_cv":.163,
         "shape":.7,"curvature":1.0},
        {"arch_id":"PILOT_A1","hidden_widths":json.dumps([20,12,20,12]),"depth":4,
         "morphology":"hourglass","actual_params":32*20+20*12+12*20+20*12+12,"width_mean":16.,"width_cv":.25,
         "shape":.6,"curvature":1.2},
    ])
    env = pd.DataFrame([
        {"environment_id":"PILOT_E0","teacher_family":"linear","informative_dim":8,
         "feature_correlation":.1,"snr":10.,"interaction_strength":.1},
        {"environment_id":"PILOT_E1","teacher_family":"shallow_nonlinear","informative_dim":16,
         "feature_correlation":.5,"snr":5.,"interaction_strength":.5},
    ])
    return arch, env, pd.DataFrame()


def load_phase2_handoff():
    if P3_ARCH_CSV.exists() and P3_ENV_CSV.exists():
        arch = pd.read_csv(P3_ARCH_CSV); env = pd.read_csv(P3_ENV_CSV)
        final = pd.read_csv(P3_PHASE2_FINAL_CSV) if P3_PHASE2_FINAL_CSV.exists() else pd.DataFrame()
    elif P3_REQUIRE_HANDOFF:
        raise FileNotFoundError(
            "Phase III research mode requires the actual Phase-II handoff tables.\n"
            f"Expected: {P3_ARCH_CSV}\nExpected: {P3_ENV_CSV}\n"
            "Set PATHFLOW_PHASE2_DIR to the Phase-II result directory, or use "
            "PATHFLOW_PHASE3_PROFILE=pilot for a smoke test."
        )
    else:
        print("[pilot] Phase-II handoff unavailable; using explicitly-labelled tiny smoke-test panel.")
        arch, env, final = pilot_handoff()
    if "hidden_widths" not in arch:
        raise ValueError("architecture_panel.csv must contain hidden_widths.")
    arch = arch.drop_duplicates("arch_id").reset_index(drop=True)
    env = env.drop_duplicates("environment_id").reset_index(drop=True)
    return arch, env, final


def augment_handoff_with_phase2(arch, env, final):
    a = arch.copy(); e = env.copy()
    if len(final):
        acols = [c for c in [
            "log_eta1_final","log_eta2_final","entropy_deficit_delta","edge_cv_delta","node_cv_delta",
            "xi50_fold","scaling_dB_final","d_star_final","local_delta_median",
            "renorm_mean_pairwise_KS","renorm_flow_retention_mean"
        ] if c in final]
        if acols:
            ag = final.groupby("arch_id")[acols].median(numeric_only=True).add_prefix("p2_").reset_index()
            a = a.merge(ag, on="arch_id", how="left")
        ecols = [c for c in [
            "entropy_deficit_delta","eta1_fold","eta2_fold","xi50_fold","d_star_delta",
            "local_delta_median","post_test_deterioration"
        ] if c in final]
        if ecols:
            eg = final.groupby("environment_id")[ecols].median(numeric_only=True).add_prefix("p2_").reset_index()
            e = e.merge(eg, on="environment_id", how="left")
    return a, e


def select_architectures(arch: pd.DataFrame):
    d = arch.copy()
    d["log_actual_params"] = np.log(np.maximum(pd.to_numeric(d.get("actual_params", 1), errors="coerce"), 1))
    base = [c for c in ["depth","log_actual_params","width_mean","width_cv","shape","curvature"] if c in d]
    p2 = [c for c in d.columns if c.startswith("p2_")][:10]
    morph = pd.get_dummies(d.get("morphology", pd.Series(["unknown"]*len(d))).astype(str), prefix="m", dtype=float)
    temp = pd.concat([d[base+p2].reset_index(drop=True), morph.reset_index(drop=True)], axis=1)
    Z = standardize_frame(temp, list(temp.columns))
    seeds = []
    if "morphology" in d:
        center = np.mean(Z, axis=0)
        for m in sorted(d.morphology.astype(str).unique()):
            ids = np.where(d.morphology.astype(str).to_numpy()==m)[0]
            if len(ids): seeds.append(int(ids[np.argmax(np.linalg.norm(Z[ids]-center[None,:], axis=1))]))
    idx = farthest_points(Z, min(P3_MECH_ARCHITECTURES, len(d)), seeds)
    sel = d.iloc[idx].copy().reset_index(drop=True); Zs = Z[idx]
    useeds = []
    if "morphology" in sel:
        center_s = np.mean(Zs, axis=0)
        for m in sorted(sel.morphology.astype(str).unique()):
            ids = np.where(sel.morphology.astype(str).to_numpy()==m)[0]
            if len(ids): useeds.append(int(ids[np.argmax(np.linalg.norm(Zs[ids]-center_s[None,:], axis=1))]))
    uidx_local = farthest_points(Zs, min(P3_UNIV_ARCHITECTURES, len(sel)), useeds)
    univ_ids = set(sel.iloc[uidx_local].arch_id.astype(str))
    sel["phase3_optimizer_subset"] = sel.arch_id.astype(str).isin(univ_ids)
    sel["phase3_mechanistic_subset"] = True
    atomic_csv(sel, P3_OUTDIR/"tables"/"phase3_architecture_panel.csv")
    return sel


def env_encoding_frame(env: pd.DataFrame) -> pd.DataFrame:
    d = env.copy(); d["log_snr"] = np.log(np.maximum(d.snr.astype(float), EPS))
    fam = pd.get_dummies(d.teacher_family.astype(str), prefix="teacher", dtype=float)
    base = d[["informative_dim","feature_correlation","log_snr","interaction_strength"]].astype(float)
    extra = d[[c for c in d.columns if c.startswith("p2_")][:8]].copy() if any(c.startswith("p2_") for c in d.columns) else pd.DataFrame(index=d.index)
    return pd.concat([base, fam, extra], axis=1)


def select_environments(env: pd.DataFrame):
    d = env.copy(); X = env_encoding_frame(d); Z = standardize_frame(X, list(X.columns))
    center = np.mean(Z, axis=0); seeds = []
    for f in sorted(d.teacher_family.astype(str).unique()):
        ids = np.where(d.teacher_family.astype(str).to_numpy()==f)[0]
        if len(ids): seeds.append(int(ids[np.argmax(np.linalg.norm(Z[ids]-center[None,:], axis=1))]))
    idx = farthest_points(Z, min(P3_MECH_ENVIRONMENTS, len(d)), seeds)
    sel = d.iloc[idx].copy().reset_index(drop=True); Zs = Z[idx]
    useeds = []
    center_s = np.mean(Zs, axis=0)
    for f in sorted(sel.teacher_family.astype(str).unique()):
        ids = np.where(sel.teacher_family.astype(str).to_numpy()==f)[0]
        if len(ids): useeds.append(int(ids[np.argmax(np.linalg.norm(Zs[ids]-center_s[None,:], axis=1))]))
    uidx_local = farthest_points(Zs, min(P3_UNIV_ENVIRONMENTS, len(sel)), useeds)
    univ_ids = set(sel.iloc[uidx_local].environment_id.astype(str))
    sel["phase3_optimizer_subset"] = sel.environment_id.astype(str).isin(univ_ids)
    sel["phase3_mechanistic_subset"] = True
    atomic_csv(sel, P3_OUTDIR/"tables"/"phase3_environment_panel.csv")
    return sel


def build_run_manifest(arch_panel, env_panel):
    a_univ = set(arch_panel.loc[arch_panel.phase3_optimizer_subset, "arch_id"].astype(str))
    e_univ = set(env_panel.loc[env_panel.phase3_optimizer_subset, "environment_id"].astype(str))
    # One sentinel architecture and two maximin environments for true short-step diagnostics.
    micro_aid = str(arch_panel.iloc[0].arch_id)
    eZ = standardize_frame(env_encoding_frame(env_panel), list(env_encoding_frame(env_panel).columns))
    micro_eidx = farthest_points(eZ, min(P3_MICRO_ENVIRONMENTS, len(env_panel)))
    micro_eids = set(env_panel.iloc[micro_eidx].environment_id.astype(str))
    rows = []
    for _, ar in arch_panel.iterrows():
        for _, er in env_panel.iterrows():
            for s in P3_SEEDS:
                # Adam mechanistic map uses the entire 6x6 panel.
                rows.append({
                    "run_id": f"{er.environment_id}__{ar.arch_id}__adam__S{int(s):02d}",
                    "arch_id": str(ar.arch_id), "environment_id": str(er.environment_id),
                    "optimizer": "adam", "seed_index": int(s),
                    "mechanistic_arm": True,
                    "optimizer_universality_arm": bool(str(ar.arch_id) in a_univ and str(er.environment_id) in e_univ),
                    "micro_arm": bool(str(ar.arch_id)==micro_aid and str(er.environment_id) in micro_eids),
                })
                # Non-Adam optimizers only on nested 5x5 panel.
                if str(ar.arch_id) in a_univ and str(er.environment_id) in e_univ:
                    for opt in P3_OPTIMIZERS:
                        if opt == "adam": continue
                        rows.append({
                            "run_id": f"{er.environment_id}__{ar.arch_id}__{opt}__S{int(s):02d}",
                            "arch_id": str(ar.arch_id), "environment_id": str(er.environment_id),
                            "optimizer": opt, "seed_index": int(s),
                            "mechanistic_arm": False, "optimizer_universality_arm": True,
                            "micro_arm": bool(str(ar.arch_id)==micro_aid and str(er.environment_id) in micro_eids),
                        })
    out = pd.DataFrame(rows).drop_duplicates("run_id").reset_index(drop=True)
    atomic_csv(out, P3_OUTDIR/"tables"/"run_manifest.csv")
    return out

# ======================================================================================
# P3.5 GAUGE-INVARIANT COMPLETE-PATH MEASURE + CONSERVED FLOW
# ======================================================================================


def path_factor_matrices(weights: List[np.ndarray]) -> List[np.ndarray]:
    A = []
    with np.errstate(over="ignore", invalid="ignore"):
        for l, W in enumerate(weights):
            W = np.asarray(W, np.float64)
            if not np.all(np.isfinite(W)):
                raise RunDivergenceError(f"path factors: non-finite weights in layer {l}.")
            M = np.square(W)
            if not np.all(np.isfinite(M)):
                raise RunDivergenceError(
                    f"path factors overflowed in layer {l}; max|w|={float(np.max(np.abs(W))):g}."
                )
            A.append(M)
    return A


def log_partition(A: List[np.ndarray], q: float = 1.0) -> float:
    """Numerically stable exact layerwise log Z_q = log sum_pi Q_pi^q.

    Each layer is divided by its own positive maximum before propagation and the removed
    factor is accumulated analytically in log space.  This is algebraically exact and avoids
    underflow when trained layers become very small.
    """
    As, layer_log_scales = _rescale_path_layers(A, context=f"log Z_{q:g}")
    z = np.ones(As[0].shape[1], dtype=np.float64)
    z /= float(z.sum())
    log_scale = math.log(As[0].shape[1])
    for M, lscale in zip(As, layer_log_scales):
        # Z_q picks up scale^q from this whole layer.  Preserve exact zeros; no artificial
        # PATH_LOG_FLOOR is needed after layer rescaling.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            Mq = np.power(M, q)
        z = Mq @ z
        ss = float(z.sum())
        if ss <= 0.0 or not np.isfinite(ss):
            raise RunDivergenceError(f"log Z_{q:g}: degenerate propagated mass after layer rescaling.")
        z /= ss
        log_scale += float(q) * lscale + math.log(ss)
    return float(log_scale)


def path_partition_stats(A: List[np.ndarray], widths: List[int]) -> Dict[str, float]:
    logZ1 = log_partition(A, 1.0); logZ2 = log_partition(A, 2.0)
    logNtopo = float(np.sum(np.log(np.asarray(widths, float))))
    dq = 1e-4
    dlogZ = (log_partition(A, 1+dq)-log_partition(A, 1-dq))/(2*dq)
    S = float(logZ1-dlogZ); log_eta1 = S-logNtopo; log_eta2 = 2*logZ1-logZ2-logNtopo
    return {
        "logZ1": float(logZ1), "logZ2": float(logZ2), "log_N_topological": logNtopo,
        "S_path": S, "entropy_deficit": float(max(logNtopo-S, 0.0)),
        "log_eta1": float(log_eta1), "log_eta2": float(log_eta2),
        "eta1": float(np.exp(np.clip(log_eta1, -745, 0))),
        "eta2": float(np.exp(np.clip(log_eta2, -745, 0))),
    }


def normalized_forward_backward(A: List[np.ndarray]):
    """Stable forward/backward messages for the normalized complete-path measure.

    The old implementation rejected any layer whose propagated unnormalized mass was below
    EPS=1e-14.  That criterion is mathematically invalid: an arbitrarily small positive layer
    factor merely rescales every complete path equally and must cancel after normalization.
    We therefore normalize each layer by a positive scalar first and only reject *true* zero
    support or non-finite factors.
    """
    As, _ = _rescale_path_layers(A, context="forward/backward path flow")
    alpha = [np.ones(As[0].shape[1], float)]
    alpha[0] /= float(alpha[0].sum())
    for l, M in enumerate(As):
        z = M @ alpha[-1]
        ss = float(z.sum())
        if ss <= 0.0 or not np.isfinite(ss):
            raise RunDivergenceError(f"forward path partition has zero/non-finite support at layer {l}.")
        alpha.append(z / ss)
    beta = [None] * (len(As) + 1)
    beta[-1] = np.ones(As[-1].shape[0], float)
    beta[-1] /= float(beta[-1].sum())
    for l in range(len(As)-1, -1, -1):
        z = As[l].T @ beta[l+1]
        ss = float(z.sum())
        if ss <= 0.0 or not np.isfinite(ss):
            raise RunDivergenceError(f"backward path partition has zero/non-finite support at layer {l}.")
        beta[l] = z / ss
    return alpha, beta, As


def path_flow(A: List[np.ndarray]):
    alpha, beta, As = normalized_forward_backward(A)
    edge_flow = []
    for l, M in enumerate(As):
        E = M * beta[l+1][:, None] * alpha[l][None, :]
        ss = float(E.sum())
        if ss <= 0.0 or not np.isfinite(ss):
            raise RunDivergenceError(f"edge-flow normalization failed at layer {l}.")
        E /= ss
        edge_flow.append(E)
    node_flow = []
    for l in range(len(alpha)):
        v = alpha[l] * beta[l]
        ss = float(v.sum())
        if ss <= 0.0 or not np.isfinite(ss):
            raise RunDivergenceError(f"node-flow normalization failed at layer {l}.")
        v /= ss
        node_flow.append(v)
    return edge_flow, node_flow


def flow_enrichment(edge_flow, node_flow):
    return [E*E.size for E in edge_flow], [v*len(v) for v in node_flow]


def layer_conservation_errors(edge_flow, node_flow):
    ee = max(abs(float(E.sum())-1.0) for E in edge_flow)
    nn = max(abs(float(v.sum())-1.0) for v in node_flow)
    return float(ee), float(nn)


def flow_concentration(edge_enr, node_enr):
    def one(layers):
        cv, top = [], []
        for x in layers:
            a = np.asarray(x,float).ravel(); a = a[np.isfinite(a)&(a>=0)]
            if not len(a): continue
            cv.append(float(a.std()/max(a.mean(), EPS)))
            k = max(1, int(math.ceil(.10*len(a))))
            top.append(float(np.sort(a)[-k:].sum()/max(a.sum(), EPS)))
        return float(np.mean(cv)) if cv else np.nan, float(np.mean(top)) if top else np.nan
    ecv, etop = one(edge_enr); ncv, ntop = one(node_enr)
    return ecv, etop, ncv, ntop


def path_state(weights: List[np.ndarray], widths: List[int]) -> Dict[str, Any]:
    A = path_factor_matrices(weights); pstats = path_partition_stats(A, widths)
    edge_flow, node_flow = path_flow(A); edge_enr, node_enr = flow_enrichment(edge_flow, node_flow)
    eerr, nerr = layer_conservation_errors(edge_flow, node_flow)
    if eerr > 1e-9 or nerr > 1e-9:
        raise RuntimeError(f"Path-flow conservation failed: edge={eerr:g}, node={nerr:g}")
    ecv, etop, ncv, ntop = flow_concentration(edge_enr, node_enr)
    return {
        **pstats,
        "edge_cv": ecv, "edge_top10_share": etop,
        "node_cv": ncv, "node_top10_share": ntop,
        "edge_conservation_error": eerr, "node_conservation_error": nerr,
        "edge_flow": edge_flow, "node_flow": node_flow,
        "edge_enrichment": edge_enr, "node_enrichment": node_enr,
    }

# ======================================================================================
# P3.6 EXACT FINITE-INTERVAL PATH-GROWTH DYNAMICS WITHOUT PATH ENUMERATION
# ======================================================================================


def additive_path_moments(A: List[np.ndarray], g_layers: List[np.ndarray], h_layers: List[np.ndarray]):
    """Moments under p(A) for additive path observables G=sum g_e and H=sum h_e.

    Returns E[G], Var(G), E[H], Var(H), Cov(G,H).  The recurrence is exact up to
    floating-point arithmetic; all moment accumulators are rescaled by the same layerwise
    normalization used for the path partition.
    """
    As, _ = _rescale_path_layers(A, context="additive path moments")
    z = np.ones(As[0].shape[1], float); z /= float(z.sum())
    mg = np.zeros_like(z); mg2 = np.zeros_like(z)
    mh = np.zeros_like(z); mh2 = np.zeros_like(z); mgh = np.zeros_like(z)
    for M, g, h in zip(As, g_layers, h_layers):
        Mg, Mh = M*g, M*h
        z1 = M @ z
        mg1 = M @ mg + Mg @ z
        mh1 = M @ mh + Mh @ z
        mg21 = M @ mg2 + 2*Mg @ mg + (M*g*g) @ z
        mh21 = M @ mh2 + 2*Mh @ mh + (M*h*h) @ z
        mgh1 = M @ mgh + Mg @ mh + Mh @ mg + (M*g*h) @ z
        s = float(z1.sum())
        if s <= 0.0 or not np.isfinite(s):
            raise RunDivergenceError("additive path moments encountered zero/non-finite propagated support.")
        z, mg, mh, mg2, mh2, mgh = [q/s for q in (z1,mg1,mh1,mg21,mh21,mgh1)]
    norm = max(float(z.sum()), EPS)
    EG = float(mg.sum()/norm); EH = float(mh.sum()/norm)
    EG2 = float(mg2.sum()/norm); EH2 = float(mh2.sum()/norm); EGH = float(mgh.sum()/norm)
    vG = max(EG2-EG*EG, 0.0); vH = max(EH2-EH*EH, 0.0); cov = EGH-EG*EH
    corr = cov/math.sqrt(vG*vH) if vG>0 and vH>0 else np.nan
    return {"mean_G":EG,"var_G":vG,"mean_H":EH,"var_H":vH,"cov_GH":cov,"corr_GH":corr}


def weighted_edge_sum(edge_flow: List[np.ndarray], edge_values: List[np.ndarray]) -> float:
    return float(sum(np.sum(F*V) for F,V in zip(edge_flow, edge_values)))


def parameter_displacement(old_weights, new_weights):
    s2 = 0.0; old2 = 0.0; maxabs = 0.0
    for a,b in zip(old_weights,new_weights):
        d = np.asarray(b,float)-np.asarray(a,float)
        s2 += float(np.sum(d*d)); old2 += float(np.sum(np.asarray(a,float)**2)); maxabs=max(maxabs,float(np.max(np.abs(d))))
    l2 = math.sqrt(max(s2,0.0)); rel = l2/max(math.sqrt(max(old2,0.0)),EPS)
    return l2, rel, maxabs


def finite_tilt_integrated_alignment(A0, A1, g_layers, h0_layers, nquad=P3_TILT_QUADRATURE_POINTS):
    """Integral_0^1 Cov_{p_a}(G, log p_a) da, p_a ∝ p0 exp(aG).

    Since log p_a differs from H0 + aG only by a path-independent constant,
    Cov(G,log p_a)=Cov(G,H0)+a Var(G).  Gauss-Legendre quadrature therefore supplies a
    high-accuracy finite-interval entropy-production identity check.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(nquad))
    aa = .5*(nodes+1.0); ww = .5*weights; vals=[]
    logA0 = [np.log(np.clip(M, PATH_LOG_FLOOR, None)) for M in A0]
    logA1 = [np.log(np.clip(M, PATH_LOG_FLOOR, None)) for M in A1]
    for a in aa:
        Aa = []
        for l0, l1 in zip(logA0, logA1):
            lm = (1.0-float(a))*l0 + float(a)*l1
            # A positive scalar per layer leaves p_a and all covariances invariant.
            lm = lm - float(np.max(lm))
            Aa.append(np.exp(lm))
        mom = additive_path_moments(Aa,g_layers,h0_layers)
        vals.append(float(mom["cov_GH"] + float(a)*mom["var_G"]))
    return float(np.sum(ww*np.asarray(vals,float)))


def flatten_layer_arrays(xs):
    return np.concatenate([np.asarray(x,float).ravel() for x in xs]) if xs else np.array([])


def interval_dynamics(old_weights, new_weights, widths, old_state, new_state,
                      old_perf: Dict[str,float], new_perf: Dict[str,float],
                      interval_steps: int, interval_epochs: float) -> Dict[str, float]:
    A0 = path_factor_matrices(old_weights); A1 = path_factor_matrices(new_weights)
    support_zero_old = np.mean([np.mean(M<=0) for M in A0]); support_zero_new = np.mean([np.mean(M<=0) for M in A1])
    support_change = np.mean([np.mean((M0<=0)!=(M1<=0)) for M0,M1 in zip(A0,A1)])
    g = [np.log(np.clip(M1,PATH_LOG_FLOOR,None))-np.log(np.clip(M0,PATH_LOG_FLOOR,None)) for M0,M1 in zip(A0,A1)]
    h0 = [np.log(np.clip(M0,PATH_LOG_FLOOR,None)) for M0 in A0]
    mom = additive_path_moments(A0, g, h0)
    # Cov(G,log p0)=Cov(G,log Q0)=Cov(G,H0).
    cov_align = float(mom["cov_GH"]); corr_align = float(mom["corr_GH"])
    logZ_delta = float(new_state["logZ1"]-old_state["logZ1"])
    EG0 = weighted_edge_sum(old_state["edge_flow"], g)
    EG1 = weighted_edge_sum(new_state["edge_flow"], g)
    # Exact whenever p0 and p1 share support (true almost surely for dense floating weights).
    kl_10 = float(EG1-logZ_delta); kl_01 = float(logZ_delta-EG0)
    if support_change > 0:
        # Preserve finite-floor diagnostics but make the exact-support caveat explicit.
        exact_support = False
    else:
        exact_support = True
    kl_10 = max(kl_10,0.0) if kl_10 > -1e-9 else kl_10
    kl_01 = max(kl_01,0.0) if kl_01 > -1e-9 else kl_01
    J = float(kl_10+kl_01)
    dSdef = float(new_state["entropy_deficit"]-old_state["entropy_deficit"])
    integrated_cov = finite_tilt_integrated_alignment(A0,A1,g,h0)
    l2, rel_l2, maxabs = parameter_displacement(old_weights,new_weights)
    train_delta = float(new_perf.get("train_mse",np.nan)-old_perf.get("train_mse",np.nan))
    improvement = -train_delta
    ef0 = flatten_layer_arrays(old_state["edge_flow"]); ef1 = flatten_layer_arrays(new_state["edge_flow"])
    nf0 = flatten_layer_arrays(old_state["node_flow"]); nf1 = flatten_layer_arrays(new_state["node_flow"])
    edge_l1 = float(np.mean(np.abs(ef1-ef0))) if len(ef0)==len(ef1) else np.nan
    node_l1 = float(np.mean(np.abs(nf1-nf0))) if len(nf0)==len(nf1) else np.nan
    # Exact exponential-tilt flow reconstruction.  Construct in log space and remove an
    # arbitrary layerwise constant before exponentiation; that constant cancels from p.
    Atilt = []
    for M0, gg in zip(A0, g):
        lm = np.log(np.clip(M0, PATH_LOG_FLOOR, None)) + gg
        lm = lm - float(np.max(lm))
        Atilt.append(np.exp(lm))
    tilt_edge, tilt_node = path_flow(Atilt)
    tilt_edge_err = max(float(np.max(np.abs(a-b))) for a,b in zip(tilt_edge,new_state["edge_flow"]))
    tilt_node_err = max(float(np.max(np.abs(a-b))) for a,b in zip(tilt_node,new_state["node_flow"]))
    # Exact local metric monotonicity check under lambda=(Fcal+eps)^(-p).
    oldE = flatten_layer_arrays(old_state["edge_enrichment"]); newE = flatten_layer_arrays(new_state["edge_enrichment"])
    oldL = np.power(oldE+EPS,-METRIC_POWER); newL=np.power(newE+EPS,-METRIC_POWER)
    dE, dL = newE-oldE, newL-oldL
    active = np.abs(dE)>1e-15
    anti = float(np.mean(dE[active]*dL[active] <= 1e-14)) if np.any(active) else np.nan
    return {
        "interval_steps":int(interval_steps), "interval_epochs":float(interval_epochs),
        "mean_G_p0":float(mom["mean_G"]), "var_G_p0":float(mom["var_G"]),
        "sd_G_p0":float(math.sqrt(max(mom["var_G"],0.0))),
        "mean_logQ_p0":float(mom["mean_H"]), "var_logQ_p0":float(mom["var_H"]),
        "cov_G_logp_p0":cov_align, "corr_G_logp_p0":corr_align,
        "logZ_delta":logZ_delta, "E_G_p0_flowcheck":EG0, "E_G_p1_flowcheck":EG1,
        "kl_p1_p0":kl_10, "kl_p0_p1":kl_01, "jeffreys_J":J,
        "delta_entropy_deficit":dSdef,
        "integrated_selection_covariance":integrated_cov,
        "entropy_identity_residual":float(dSdef-integrated_cov),
        "local_covariance_residual":float(dSdef-cov_align),
        "parameter_l2":l2, "relative_parameter_l2":rel_l2, "parameter_maxabs":maxabs,
        "J_per_parameter_l2_sq":float(J/max(l2*l2,EPS)),
        "delta_Sdef_per_parameter_l2":float(dSdef/max(l2,EPS)),
        "train_mse_delta":train_delta,
        "J_per_train_loss_improvement":float(J/improvement) if np.isfinite(improvement) and improvement>EPS else np.nan,
        "edge_flow_mean_abs_delta":edge_l1, "node_flow_mean_abs_delta":node_l1,
        "tilt_edge_flow_max_abs_residual":float(tilt_edge_err),
        "tilt_node_flow_max_abs_residual":float(tilt_node_err),
        "metric_antialignment_fraction":anti,
        "zero_edge_fraction_old":float(support_zero_old), "zero_edge_fraction_new":float(support_zero_new),
        "zero_support_change_fraction":float(support_change), "exact_common_support":bool(exact_support),
    }

# ======================================================================================
# P3.8 FLOW-INDUCED BCANw GEOMETRY — SAME PRIMARY CONSTRUCTION AS PHASES I/II
# ======================================================================================

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


# ======================================================================================
# P3.9 OPTIMIZER CONSTRUCTION + TRAINING-ONLY LEARNING-RATE CALIBRATION
# ======================================================================================


def make_optimizer(model: nn.Module, name: str, lr: float):
    name = str(name).lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=P3_WEIGHT_DECAY)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=float(lr), weight_decay=P3_WEIGHT_DECAY, momentum=0.0)
    if name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=float(lr), weight_decay=P3_WEIGHT_DECAY,
                               momentum=0.9, nesterov=False)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=float(lr), weight_decay=P3_WEIGHT_DECAY,
                                   alpha=0.99, momentum=0.0)
    raise ValueError(f"Unknown optimizer: {name}")


def _calibration_subset(arch_panel: pd.DataFrame, env_panel: pd.DataFrame):
    # Maximin again, so the calibration is not tied to the first rows in a table.
    ad = arch_panel.copy()
    acols = [c for c in ["depth", "actual_params", "width_mean", "width_cv", "shape", "curvature"] if c in ad]
    if "actual_params" in acols:
        ad["actual_params"] = np.log(np.maximum(pd.to_numeric(ad["actual_params"], errors="coerce"), 1))
    AZ = standardize_frame(ad, acols) if acols else np.arange(len(ad))[:, None]
    ai = farthest_points(AZ, min(P3_CALIBRATION_ARCHITECTURES, len(ad)))
    EZ = standardize_frame(env_encoding_frame(env_panel), list(env_encoding_frame(env_panel).columns))
    ei = farthest_points(EZ, min(P3_CALIBRATION_ENVIRONMENTS, len(env_panel)))
    return ad.iloc[ai].copy(), env_panel.iloc[ei].copy()


def calibrate_learning_rates(arch_panel: pd.DataFrame, env_panel: pd.DataFrame) -> Dict[str, float]:
    path = P3_OUTDIR / "calibration" / "selected_learning_rates.json"
    if (not P3_CALIBRATE_LR) or P3_PROFILE == "pilot":
        chosen = {o: float(P3_FIXED_LR[o]) for o in P3_OPTIMIZERS}
        chosen["adam"] = 1e-3
        atomic_json({"method": "fixed_preregistered", "learning_rates": chosen,
                     "adam_anchor": "Phase-I/II exact lr=1e-3"}, path)
        return chosen
    if P3_RESUME and path.exists():
        obj = json.loads(path.read_text())
        if "learning_rates" in obj:
            chosen = {k: float(v) for k, v in obj["learning_rates"].items() if k in P3_OPTIMIZERS}
            chosen["adam"] = 1e-3
            return chosen

    banner("PHASE III — TRAINING-ONLY LEARNING-RATE CALIBRATION")
    a_sub, e_sub = _calibration_subset(arch_panel, env_panel)
    rows = []
    # Cache generated environments because all optimizers must see exactly the same tensors.
    env_cache = {str(er.environment_id): generate_environment(er)[0] for _, er in e_sub.iterrows()}
    total = sum(len(P3_LR_GRIDS[o]) for o in P3_CALIBRATE_OPTIMIZERS) * len(a_sub) * len(e_sub) * len(P3_CALIBRATION_SEEDS)
    bar = tqdm(total=total, desc="LR calibration", leave=True)
    for opt_name in P3_CALIBRATE_OPTIMIZERS:
        for lr in P3_LR_GRIDS[opt_name]:
            for _, ar in a_sub.iterrows():
                widths = parse_widths(ar.hidden_widths)
                for _, er in e_sub.iterrows():
                    splits = env_cache[str(er.environment_id)]
                    Xtr, ytr = splits["train"]
                    for sidx in P3_CALIBRATION_SEEDS:
                        model = BiasFreeReLUMLP(P3_INPUT_DIM, widths, 1, seed=model_seed(int(sidx))).to(DEVICE)
                        opt = make_optimizer(model, opt_name, lr)
                        lossfn = nn.MSELoss()
                        init = eval_mse(model, Xtr, ytr)
                        diverged = False
                        for ep in range(1, P3_CALIBRATION_EPOCHS + 1):
                            model.train()
                            for ids in epoch_batches(len(Xtr), ep, int(sidx)):
                                it = torch.as_tensor(ids, dtype=torch.long)
                                xb = Xtr.index_select(0, it).to(DEVICE, non_blocking=True)
                                yb = ytr.index_select(0, it).to(DEVICE, non_blocking=True)
                                opt.zero_grad(set_to_none=True)
                                loss = lossfn(model(xb), yb)
                                if not torch.isfinite(loss):
                                    diverged = True; break
                                loss.backward(); opt.step()
                            if diverged: break
                        fin = eval_mse(model, Xtr, ytr) if not diverged else np.inf
                        ratio = float(fin / max(init, EPS)) if np.isfinite(fin) else np.inf
                        score = float(math.log(max(ratio, EPS))) if np.isfinite(ratio) else 50.0
                        # A trial that worsens training loss badly is not a fair operating point.
                        if (not np.isfinite(fin)) or ratio > 2.0:
                            score = 50.0 + min(ratio if np.isfinite(ratio) else 50.0, 50.0)
                        rows.append({
                            "optimizer": opt_name, "lr": float(lr), "arch_id": str(ar.arch_id),
                            "environment_id": str(er.environment_id), "seed_index": int(sidx),
                            "train_mse_init": float(init), "train_mse_final": float(fin),
                            "train_ratio": float(ratio), "log_train_ratio_score": float(score),
                            "diverged": bool(diverged),
                        })
                        del model, opt
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        bar.update(1)
    bar.close()
    cdf = pd.DataFrame(rows); atomic_csv(cdf, P3_OUTDIR / "calibration" / "calibration_results.csv")
    chosen = {"adam": 1e-3}
    summaries = [{"optimizer": "adam", "lr": 1e-3, "median_log_train_ratio": np.nan,
                  "selected": True, "selection_note": "locked Phase-I/II anchor"}]
    for opt_name in P3_CALIBRATE_OPTIMIZERS:
        g = cdf[cdf.optimizer == opt_name]
        med = g.groupby("lr")["log_train_ratio_score"].median().sort_values()
        best = float(med.index[0]) if len(med) else float(P3_FIXED_LR[opt_name])
        chosen[opt_name] = best
        for lr, val in med.items():
            summaries.append({"optimizer": opt_name, "lr": float(lr), "median_log_train_ratio": float(val),
                              "selected": bool(float(lr) == best)})
    atomic_csv(pd.DataFrame(summaries), P3_OUTDIR / "calibration" / "calibration_summary.csv")
    atomic_json({
        "method": "training_loss_only_maximin_panel",
        "epochs": int(P3_CALIBRATION_EPOCHS),
        "architectures": int(len(a_sub)), "environments": int(len(e_sub)),
        "seeds": list(map(int, P3_CALIBRATION_SEEDS)),
        "learning_rates": chosen,
        "selection_rule": "minimum median log(final_train_MSE / initial_train_MSE); divergence penalized",
        "validation_or_test_used": False, "pathflow_used": False,
        "adam_anchor": "locked to lr=1e-3 exactly as Phases I/II",
    }, path)
    print("Selected learning rates:", chosen)
    return chosen


# ======================================================================================
# P3.10 INTERNAL MATHEMATICAL VALIDATION
# ======================================================================================


def enumerate_tiny_paths(weights: List[np.ndarray]):
    """Only for a tiny unit test; never used by the research experiment."""
    widths = [weights[0].shape[1]] + [W.shape[0] for W in weights]
    import itertools
    paths = list(itertools.product(*[range(w) for w in widths]))
    q = []
    for p in paths:
        val = 1.0
        for l, W in enumerate(weights):
            val *= float(W[p[l + 1], p[l]] ** 2)
        q.append(val)
    q = np.asarray(q, float); p = q / q.sum()
    return paths, q, p


def internal_math_self_test() -> Dict[str, Any]:
    rng = np.random.default_rng(190731)
    W0 = rng.normal(size=(3, 2)); W1 = rng.normal(size=(2, 3)); W2 = rng.normal(size=(1, 2))
    old = [W0, W1, W2]
    new = [W0 * (1 + 0.015 * rng.normal(size=W0.shape)),
           W1 * (1 + 0.015 * rng.normal(size=W1.shape)),
           W2 * (1 + 0.015 * rng.normal(size=W2.shape))]
    widths = [2, 3, 2, 1]
    s0, s1 = path_state(old, widths), path_state(new, widths)
    d = interval_dynamics(old, new, widths, s0, s1,
                          {"train_mse": np.nan}, {"train_mse": np.nan}, 1, 0.0)
    paths, q0, p0 = enumerate_tiny_paths(old); _, q1, p1 = enumerate_tiny_paths(new)
    G = np.log(np.clip(q1, PATH_LOG_FLOOR, None)) - np.log(np.clip(q0, PATH_LOG_FLOOR, None))
    tilt = p0 * np.exp(G - np.max(G)); tilt /= tilt.sum()
    tilt_err = float(np.max(np.abs(tilt - p1)))

    # Positive hidden-unit gauge rescaling: incoming row x c, outgoing column x 1/c.
    gauged = [x.copy() for x in old]
    c = 3.7; unit = 1
    gauged[0][unit, :] *= c
    gauged[1][:, unit] /= c
    sg = path_state(gauged, widths)
    flow_err = max(
        max(float(np.max(np.abs(a - b))) for a, b in zip(s0["edge_flow"], sg["edge_flow"])),
        max(float(np.max(np.abs(a - b))) for a, b in zip(s0["node_flow"], sg["node_flow"])),
    )
    gauge_eta_err = max(abs(s0["eta1"] - sg["eta1"]), abs(s0["eta2"] - sg["eta2"]))
    eerr, nerr = layer_conservation_errors(s0["edge_flow"], s0["node_flow"])
    result = {
        "path_tilt_max_abs_error": tilt_err,
        "gauge_flow_max_abs_error": float(flow_err),
        "gauge_eta_max_abs_error": float(gauge_eta_err),
        "edge_conservation_error": float(eerr), "node_conservation_error": float(nerr),
        "finite_entropy_identity_residual": float(d["entropy_identity_residual"]),
        "kl_p1_p0": float(d["kl_p1_p0"]), "kl_p0_p1": float(d["kl_p0_p1"]),
    }
    result["passed"] = bool(
        tilt_err < 1e-10 and flow_err < 1e-10 and gauge_eta_err < 1e-10 and
        eerr < 1e-10 and nerr < 1e-10 and abs(d["entropy_identity_residual"]) < 5e-8 and
        d["kl_p1_p0"] >= -1e-9 and d["kl_p0_p1"] >= -1e-9
    )
    atomic_json(result, P3_OUTDIR / "tables" / "internal_validation.json")
    if not result["passed"]:
        raise RuntimeError(f"Phase-III mathematical self-test failed: {result}")
    return result


# ======================================================================================
# P3.11 LONGITUDINAL TRAINING + STATE / VELOCITY COLLECTION
# ======================================================================================


def state_checkpoints(optimizer_name: str) -> List[int]:
    base = P3_ADAM_STATE_CHECKPOINTS if optimizer_name == "adam" else P3_OTHER_STATE_CHECKPOINTS
    return sorted(set([e for e in base if 0 <= e <= P3_MAX_EPOCH] + [0, P3_MAX_EPOCH]))


def geometry_checkpoints(run_row) -> List[int]:
    base = P3_SENTINEL_GEOMETRY_CHECKPOINTS if bool(run_row.micro_arm) else P3_GEOMETRY_CHECKPOINTS
    return sorted(set([e for e in base if 0 <= e <= P3_MAX_EPOCH] + [0, P3_MAX_EPOCH]))


def state_scalar_dict(s: Dict[str, Any]) -> Dict[str, float]:
    keys = ["logZ1", "logZ2", "log_N_topological", "S_path", "entropy_deficit", "log_eta1", "log_eta2",
            "eta1", "eta2", "edge_cv", "edge_top10_share", "node_cv", "node_top10_share",
            "edge_conservation_error", "node_conservation_error"]
    return {k: float(s.get(k, np.nan)) for k in keys}


def run_metadata(run_row, ar, er, lr: float) -> Dict[str, Any]:
    return {
        "run_id": str(run_row.run_id), "arch_id": str(ar.arch_id),
        "environment_id": str(er.environment_id), "optimizer": str(run_row.optimizer),
        "learning_rate": float(lr), "seed_index": int(run_row.seed_index),
        "mechanistic_arm": bool(run_row.mechanistic_arm),
        "optimizer_universality_arm": bool(run_row.optimizer_universality_arm),
        "micro_arm": bool(run_row.micro_arm),
        "teacher_family": str(er.teacher_family), "informative_dim": int(er.informative_dim),
        "feature_correlation": float(er.feature_correlation), "snr": float(er.snr),
        "interaction_strength": float(er.interaction_strength),
        "morphology": str(getattr(ar, "morphology", "unknown")),
        "depth": int(getattr(ar, "depth", len(parse_widths(ar.hidden_widths)))),
        "actual_params": int(getattr(ar, "actual_params", -1)),
    }


def evaluate_splits(model, splits) -> Dict[str, float]:
    return {
        "train_mse": eval_mse(model, *splits["train"]),
        "val_mse": eval_mse(model, *splits["val"]),
        "test_mse": eval_mse(model, *splits["test"]),
    }


def _write_run_partials(run_id: str, state_rows, dyn_rows, geom_rows, micro_rows):
    atomic_csv(pd.DataFrame(state_rows), P3_OUTDIR / "run_state" / f"{run_id}__partial.csv")
    atomic_csv(pd.DataFrame(dyn_rows), P3_OUTDIR / "run_dynamics" / f"{run_id}__partial.csv")
    if geom_rows:
        atomic_csv(pd.concat(geom_rows, ignore_index=True), P3_OUTDIR / "run_geometry" / f"{run_id}__partial.csv")
    if micro_rows:
        atomic_csv(pd.DataFrame(micro_rows), P3_OUTDIR / "run_micro" / f"{run_id}__partial.csv")


def run_one(run_row, ar, er, splits, learning_rates: Dict[str, float]):
    run_id = str(run_row.run_id); opt_name = str(run_row.optimizer); sidx = int(run_row.seed_index)
    state_path = P3_OUTDIR / "run_state" / f"{run_id}.csv"
    dyn_path = P3_OUTDIR / "run_dynamics" / f"{run_id}.csv"
    geom_path = P3_OUTDIR / "run_geometry" / f"{run_id}.csv"
    micro_path = P3_OUTDIR / "run_micro" / f"{run_id}.csv"
    done_path = P3_OUTDIR / "checkpoints" / f"{run_id}__DONE.json"
    failed_path = P3_OUTDIR / "checkpoints" / f"{run_id}__DIVERGED.json"
    if P3_RESUME and done_path.exists() and state_path.exists() and dyn_path.exists():
        return (
            pd.read_csv(state_path), pd.read_csv(dyn_path),
            pd.read_csv(geom_path) if geom_path.exists() else pd.DataFrame(),
            pd.read_csv(micro_path) if micro_path.exists() else pd.DataFrame(),
        )
    if (P3_RESUME and failed_path.exists() and not P3_RETRY_RECORDED_DIVERGENCES):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    widths_hidden = parse_widths(ar.hidden_widths); widths = [P3_INPUT_DIM] + widths_hidden + [1]
    lr = float(learning_rates[opt_name]); meta = run_metadata(run_row, ar, er, lr)
    model = BiasFreeReLUMLP(P3_INPUT_DIM, widths_hidden, 1, seed=model_seed(sidx)).to(DEVICE)
    opt = make_optimizer(model, opt_name, lr); lossfn = nn.MSELoss()
    Xtr, ytr = splits["train"]
    gcheck = geometry_checkpoints(run_row)
    scheck = sorted(set(state_checkpoints(opt_name) + gcheck))
    steps_per_epoch = int(math.ceil(len(Xtr) / P3_BATCH_SIZE))

    state_rows: List[Dict[str, Any]] = []; dyn_rows: List[Dict[str, Any]] = []
    geom_rows: List[pd.DataFrame] = []; micro_rows: List[Dict[str, Any]] = []
    global_step = 0; tauJ = 0.0

    weights0 = extract_weights(model); ps0 = path_state(weights0, widths); perf0 = evaluate_splits(model, splits)
    grow0 = {}
    if 0 in gcheck:
        gs0, gdf0 = analyze_geometry(weights0, ps0, P3_SENTINEL_BCANW_SCALES if run_row.micro_arm else P3_BCANW_SCALES,
                                     compute_local=False)
        grow0 = gs0
        if len(gdf0):
            for k, v in meta.items(): gdf0[k] = v
            gdf0["epoch"] = 0; gdf0["global_step"] = 0
            geom_rows.append(gdf0)
    state_rows.append({**meta, "epoch": 0, "global_step": 0, "tau_J": 0.0, **perf0,
                       **state_scalar_dict(ps0), **grow0})
    prev_weights, prev_state, prev_perf, prev_epoch, prev_step = weights0, ps0, perf0, 0, 0
    micro_prev_weights, micro_prev_state, micro_prev_step = weights0, ps0, 0

    for ep in range(1, P3_MAX_EPOCH + 1):
        model.train()
        for ids in epoch_batches(len(Xtr), ep, sidx):
            it = torch.as_tensor(ids, dtype=torch.long)
            xb = Xtr.index_select(0, it).to(DEVICE, non_blocking=True)
            yb = ytr.index_select(0, it).to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RunDivergenceError(
                    f"non-finite training loss at run={run_id}, epoch={ep}, step={global_step+1}, "
                    f"optimizer={opt_name}, lr={lr:g}"
                )
            loss.backward()
            opt.step()
            global_step += 1

            if bool(run_row.micro_arm) and global_step in P3_MICRO_STEPS:
                mw = extract_weights(model); ms = path_state(mw, widths)
                md = interval_dynamics(
                    micro_prev_weights, mw, widths, micro_prev_state, ms,
                    {"train_mse": np.nan}, {"train_mse": np.nan},
                    global_step - micro_prev_step, (global_step - micro_prev_step) / max(steps_per_epoch, 1),
                )
                micro_rows.append({**meta, "start_step": int(micro_prev_step), "end_step": int(global_step),
                                   "epoch_at_end": float(ep - 1 + min(1.0, global_step / max(steps_per_epoch, 1))), **md})
                micro_prev_weights, micro_prev_state, micro_prev_step = mw, ms, global_step

        if ep in scheck:
            w = extract_weights(model); ps = path_state(w, widths); perf = evaluate_splits(model, splits)
            dd = interval_dynamics(prev_weights, w, widths, prev_state, ps, prev_perf, perf,
                                   global_step - prev_step, float(ep - prev_epoch))
            tauJ += max(float(dd["jeffreys_J"]), 0.0) if np.isfinite(dd["jeffreys_J"]) else 0.0
            dyn_rows.append({**meta, "start_epoch": int(prev_epoch), "end_epoch": int(ep),
                             "start_step": int(prev_step), "end_step": int(global_step),
                             "tau_J_end": float(tauJ), **dd})
            grow = {}
            if ep in gcheck:
                # Final local dimensions are intentionally computed only on the final state.
                glocal = bool(P3_COMPUTE_FINAL_LOCAL and ep == P3_MAX_EPOCH)
                gs, gdf = analyze_geometry(w, ps,
                                           P3_SENTINEL_BCANW_SCALES if run_row.micro_arm else P3_BCANW_SCALES,
                                           compute_local=glocal)
                grow = gs
                if len(gdf):
                    for k, v in meta.items(): gdf[k] = v
                    gdf["epoch"] = int(ep); gdf["global_step"] = int(global_step)
                    geom_rows.append(gdf)
            state_rows.append({**meta, "epoch": int(ep), "global_step": int(global_step),
                               "tau_J": float(tauJ), **perf, **state_scalar_dict(ps), **grow})
            prev_weights, prev_state, prev_perf, prev_epoch, prev_step = w, ps, perf, ep, global_step
            _write_run_partials(run_id, state_rows, dyn_rows, geom_rows, micro_rows)

    sdf = pd.DataFrame(state_rows).sort_values("epoch").reset_index(drop=True)
    ddf = pd.DataFrame(dyn_rows).sort_values("end_epoch").reset_index(drop=True)
    gdf = pd.concat(geom_rows, ignore_index=True) if geom_rows else pd.DataFrame()
    mdf = pd.DataFrame(micro_rows)
    atomic_csv(sdf, state_path); atomic_csv(ddf, dyn_path)
    if len(gdf): atomic_csv(gdf, geom_path)
    if len(mdf): atomic_csv(mdf, micro_path)
    atomic_json({"run_id": run_id, "completed": True, "epochs": int(P3_MAX_EPOCH),
                 "global_steps": int(global_step), "optimizer": opt_name, "learning_rate": lr}, done_path)
    # Remove partials only after final files are atomically present.
    for p in [P3_OUTDIR/"run_state"/f"{run_id}__partial.csv",
              P3_OUTDIR/"run_dynamics"/f"{run_id}__partial.csv",
              P3_OUTDIR/"run_geometry"/f"{run_id}__partial.csv",
              P3_OUTDIR/"run_micro"/f"{run_id}__partial.csv"]:
        if p.exists():
            try: p.unlink()
            except Exception: pass
    del model, opt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return sdf, ddf, gdf, mdf


# ======================================================================================
# P3.12 POST-HOC TIME COORDINATES + RUN-LEVEL SUMMARIES
# ======================================================================================


def add_time_coordinates(states: pd.DataFrame) -> pd.DataFrame:
    if len(states) == 0:
        return states
    out = []
    for rid, g in states.groupby("run_id", sort=False):
        g = g.sort_values("epoch").copy()
        L0 = float(g.iloc[0].train_mse); Lmin = float(np.nanmin(g.train_mse.to_numpy(float)))
        den = max(L0 - Lmin, EPS)
        g["tau_L"] = np.clip((L0 - g.train_mse.astype(float)) / den, 0.0, 1.0)
        jf = float(g.tau_J.iloc[-1]) if "tau_J" in g and np.isfinite(g.tau_J.iloc[-1]) else np.nan
        g["tau_J_fraction"] = g.tau_J.astype(float) / max(jf, EPS) if np.isfinite(jf) and jf > EPS else 0.0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_final_summary(states: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in states.groupby("run_id", sort=False):
        g = g.sort_values("epoch").copy(); init = g.iloc[0]; fin = g.iloc[-1]
        # Selection checkpoint is validation, not test.
        valid_val = g[np.isfinite(g.val_mse.astype(float))]
        best = valid_val.loc[valid_val.val_mse.astype(float).idxmin()] if len(valid_val) else fin
        d = dynamics[dynamics.run_id.astype(str) == str(rid)].sort_values("end_epoch") if len(dynamics) else pd.DataFrame()
        post = d[d.start_epoch.astype(float) >= float(best.epoch)] if len(d) else pd.DataFrame()
        def fold(col):
            a, b = float(init.get(col, np.nan)), float(fin.get(col, np.nan))
            return float(b / a) if np.isfinite(a) and np.isfinite(b) and abs(a) > EPS else np.nan
        def delta(col):
            a, b = float(init.get(col, np.nan)), float(fin.get(col, np.nan))
            return float(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan
        rec = {k: fin.get(k) for k in [
            "run_id", "arch_id", "environment_id", "optimizer", "learning_rate", "seed_index",
            "mechanistic_arm", "optimizer_universality_arm", "micro_arm", "teacher_family",
            "informative_dim", "feature_correlation", "snr", "interaction_strength",
            "morphology", "depth", "actual_params"
        ] if k in fin.index}
        rec.update({
            "best_val_epoch": int(best.epoch), "best_val_mse": float(best.val_mse),
            "best_test_mse_at_best_val": float(best.test_mse),
            "final_train_mse": float(fin.train_mse), "final_val_mse": float(fin.val_mse),
            "final_test_mse": float(fin.test_mse),
            "post_best_val_test_deterioration": float(fin.test_mse - best.test_mse),
            "eta1_fold": fold("eta1"), "eta2_fold": fold("eta2"),
            "entropy_deficit_delta": delta("entropy_deficit"),
            "edge_cv_fold": fold("edge_cv"), "node_cv_fold": fold("node_cv"),
            "xi50_fold": fold("xi50"), "scaling_dB_delta": delta("scaling_dB"),
            "d_star_delta": delta("d_star"), "x_d_star_delta": delta("x_d_star"),
            "entropy_deficit_final": float(fin.get("entropy_deficit", np.nan)),
            "log_eta1_final": float(fin.get("log_eta1", np.nan)), "log_eta2_final": float(fin.get("log_eta2", np.nan)),
            "edge_cv_final": float(fin.get("edge_cv", np.nan)), "node_cv_final": float(fin.get("node_cv", np.nan)),
            "xi50_final": float(fin.get("xi50", np.nan)), "log_xi50_final": float(fin.get("log_xi50", np.nan)),
            "scaling_dB_final": float(fin.get("scaling_dB", np.nan)),
            "d_star_final": float(fin.get("d_star", np.nan)), "x_d_star_final": float(fin.get("x_d_star", np.nan)),
            "lacunarity_star_final": float(fin.get("lacunarity_star", np.nan)),
            "local_delta_median": float(fin.get("local_delta_median", np.nan)),
            "valid_local_delta_fraction": float(fin.get("valid_local_delta_fraction", np.nan)),
            "quotient_mean_pairwise_KS": float(fin.get("quotient_mean_pairwise_KS", np.nan)),
            "renorm_flow_retention_mean": float(fin.get("renorm_flow_retention_mean", np.nan)),
            "tau_J_final": float(fin.get("tau_J", np.nan)),
        })
        if len(d):
            cov = d.cov_G_logp_p0.astype(float).to_numpy(); cor = d.corr_G_logp_p0.astype(float).to_numpy()
            ds = d.delta_entropy_deficit.astype(float).to_numpy(); J = d.jeffreys_J.astype(float).to_numpy()
            rec.update({
                "dynamic_intervals": int(len(d)),
                "cov_positive_interval_fraction": float(np.mean(cov[np.isfinite(cov)] > 0)) if np.any(np.isfinite(cov)) else np.nan,
                "corr_positive_interval_fraction": float(np.mean(cor[np.isfinite(cor)] > 0)) if np.any(np.isfinite(cor)) else np.nan,
                "delta_Sdef_positive_interval_fraction": float(np.mean(ds[np.isfinite(ds)] > 0)) if np.any(np.isfinite(ds)) else np.nan,
                "median_cov_G_logp": float(np.nanmedian(cov)), "median_corr_G_logp": float(np.nanmedian(cor)),
                "median_var_G": float(np.nanmedian(d.var_G_p0.astype(float))),
                "total_Jeffreys": float(np.nansum(J)),
                "median_J_per_parameter_l2_sq": float(np.nanmedian(d.J_per_parameter_l2_sq.astype(float))),
                "max_abs_entropy_identity_residual": float(np.nanmax(np.abs(d.entropy_identity_residual.astype(float)))),
            })
        else:
            rec.update({"dynamic_intervals": 0})
        if len(post):
            pc = post.cov_G_logp_p0.astype(float).to_numpy()
            rec.update({
                "post_best_interval_count": int(len(post)),
                "post_best_cov_positive_fraction": float(np.mean(pc[np.isfinite(pc)] > 0)) if np.any(np.isfinite(pc)) else np.nan,
                "post_best_delta_Sdef": float(np.nansum(post.delta_entropy_deficit.astype(float))),
                "post_best_Jeffreys": float(np.nansum(post.jeffreys_J.astype(float))),
            })
        else:
            rec.update({"post_best_interval_count": 0, "post_best_cov_positive_fraction": np.nan,
                        "post_best_delta_Sdef": np.nan, "post_best_Jeffreys": np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


def matched_progress_table(states: pd.DataFrame, coordinate: str, grid=None) -> pd.DataFrame:
    if grid is None:
        grid = np.linspace(0, 1, 11)
    metrics = ["entropy_deficit", "log_eta1", "log_eta2", "edge_cv", "node_cv", "log_xi50", "d_star"]
    rows = []
    for rid, g in states.groupby("run_id", sort=False):
        g = g.sort_values(coordinate).copy()
        x = g[coordinate].astype(float).to_numpy()
        goodx = np.isfinite(x)
        if np.sum(goodx) < 2:
            continue
        # Collapse duplicate coordinates by median before interpolation.
        gg = g.loc[goodx].groupby(coordinate, as_index=False).median(numeric_only=True).sort_values(coordinate)
        xx = gg[coordinate].astype(float).to_numpy()
        if len(xx) < 2 or xx[-1] <= xx[0] + EPS:
            continue
        meta = g.iloc[-1]
        for z in grid:
            rec = {"run_id": rid, "optimizer": str(meta.optimizer), "arch_id": str(meta.arch_id),
                   "environment_id": str(meta.environment_id), "seed_index": int(meta.seed_index),
                   "coordinate": coordinate, "coordinate_value": float(z)}
            for m in metrics:
                if m not in gg:
                    rec[m] = np.nan; continue
                yy = gg[m].astype(float).to_numpy(); good = np.isfinite(xx) & np.isfinite(yy)
                rec[m] = float(np.interp(z, xx[good], yy[good])) if np.sum(good) >= 2 and z >= np.min(xx[good]) and z <= np.max(xx[good]) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


# ======================================================================================
# P3.13 HYPOTHESIS TESTS + MATCHED OPTIMIZER CONTRASTS
# ======================================================================================


def optimizer_sign_summary(finals: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for opt in P3_OPTIMIZERS:
        f = finals[(finals.optimizer.astype(str) == opt) & (finals.optimizer_universality_arm.astype(bool))]
        d = dynamics[(dynamics.optimizer.astype(str) == opt) & (dynamics.optimizer_universality_arm.astype(bool))] if len(dynamics) else pd.DataFrame()
        run_cov = f.median_cov_G_logp.astype(float).to_numpy() if "median_cov_G_logp" in f else np.array([])
        run_cov = run_cov[np.isfinite(run_cov)]
        k = int(np.sum(run_cov > 0)); n = int(len(run_cov))
        pbin = float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else np.nan
        rec = {
            "optimizer": opt, "n_runs": int(len(f)),
            "run_fraction_median_cov_positive": float(k / n) if n else np.nan,
            "run_sign_binomial_p_greater_half": pbin,
            "run_fraction_net_entropy_deficit_up": float(np.mean(f.entropy_deficit_delta.astype(float) > 0)) if len(f) else np.nan,
            "run_fraction_eta1_down": float(np.mean(f.eta1_fold.astype(float) < 1)) if len(f) else np.nan,
            "run_fraction_edge_cv_up": float(np.mean(f.edge_cv_fold.astype(float) > 1)) if len(f) else np.nan,
            "run_fraction_node_cv_up": float(np.mean(f.node_cv_fold.astype(float) > 1)) if len(f) else np.nan,
            "run_fraction_xi50_down": float(np.mean(f.xi50_fold.astype(float) < 1)) if len(f) and "xi50_fold" in f else np.nan,
        }
        if len(d):
            cov = d.cov_G_logp_p0.astype(float).to_numpy(); ds = d.delta_entropy_deficit.astype(float).to_numpy()
            rec["n_intervals"] = int(len(d))
            rec["interval_fraction_cov_positive"] = float(np.mean(cov[np.isfinite(cov)] > 0)) if np.any(np.isfinite(cov)) else np.nan
            rec["interval_fraction_delta_Sdef_positive"] = float(np.mean(ds[np.isfinite(ds)] > 0)) if np.any(np.isfinite(ds)) else np.nan
            rec["median_interval_Jeffreys"] = float(np.nanmedian(d.jeffreys_J.astype(float)))
            rec["median_interval_var_G"] = float(np.nanmedian(d.var_G_p0.astype(float)))
            rec["median_interval_corr_alignment"] = float(np.nanmedian(d.corr_G_logp_p0.astype(float)))
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_optimizer_contrasts(finals: pd.DataFrame) -> pd.DataFrame:
    d = finals[finals.optimizer_universality_arm.astype(bool)].copy()
    metrics = [
        "entropy_deficit_delta", "eta1_fold", "edge_cv_fold", "node_cv_fold", "xi50_fold",
        "d_star_delta", "tau_J_final", "median_cov_G_logp", "median_corr_G_logp",
        "median_var_G", "median_J_per_parameter_l2_sq", "post_best_delta_Sdef",
    ]
    keys = ["arch_id", "environment_id", "seed_index"]
    rows = []
    adam = d[d.optimizer == "adam"].set_index(keys)
    for opt in [o for o in P3_OPTIMIZERS if o != "adam"]:
        other = d[d.optimizer == opt].set_index(keys)
        common = adam.index.intersection(other.index)
        for m in metrics:
            if m not in adam.columns or m not in other.columns or len(common) == 0:
                continue
            a = adam.loc[common, m].astype(float).to_numpy(); b = other.loc[common, m].astype(float).to_numpy()
            good = np.isfinite(a) & np.isfinite(b); a, b = a[good], b[good]
            if len(a) == 0:
                continue
            diff = b - a
            try:
                p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue) if np.any(np.abs(diff) > 0) else 1.0
            except Exception:
                p = np.nan
            rows.append({
                "optimizer": opt, "reference": "adam", "metric": m, "n_pairs": int(len(diff)),
                "median_optimizer_minus_adam": float(np.median(diff)),
                "median_absolute_difference": float(np.median(np.abs(diff))),
                "wilcoxon_p": p,
            })
    return pd.DataFrame(rows)


def hypothesis_table(finals: pd.DataFrame, sign: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    adam = finals[finals.optimizer == "adam"]
    rows.append({
        "hypothesis": "H3.2", "statement": "Adam has positive path-selection alignment for a substantial majority of runs.",
        "primary_statistic": "fraction of Adam runs with median Cov_p(G,log p)>0",
        "value": float(np.mean(adam.median_cov_G_logp.astype(float) > 0)) if len(adam) else np.nan,
        "status_rule": ">0.5 is directional support; confidence assessed by run-level sign test",
    })
    rows.append({
        "hypothesis": "H3.3", "statement": "Positive alignment persists across the selected Phase-II environment panel under Adam.",
        "primary_statistic": "fraction of environments whose across-architecture/seed median alignment is positive",
        "value": float(np.mean(adam.groupby("environment_id").median(numeric_only=True).median_cov_G_logp > 0)) if len(adam) else np.nan,
        "status_rule": "descriptive environment-level robustness; no independence assumed among runs within environment",
    })
    if len(sign):
        rows.append({
            "hypothesis": "H3.4", "statement": "The sign of path-selection alignment is preserved across optimizer families.",
            "primary_statistic": "minimum optimizer-specific run fraction with median alignment >0",
            "value": float(np.nanmin(sign.run_fraction_median_cov_positive.astype(float))),
            "status_rule": "universality supported only if all optimizer families remain predominantly positive",
        })
    rows.append({
        "hypothesis": "H3.5", "statement": "Finite edge/node flow changes are induced by differential path amplification.",
        "primary_statistic": "maximum exact exponential-tilt edge/node flow reconstruction residual",
        "value": float(max(np.nanmax(np.abs(dynamics.tilt_edge_flow_max_abs_residual.astype(float))),
                           np.nanmax(np.abs(dynamics.tilt_node_flow_max_abs_residual.astype(float))))) if len(dynamics) else np.nan,
        "status_rule": "numerical reconstruction residual should be near floating-point precision on common support",
    })
    rows.append({
        "hypothesis": "H3.6", "statement": "Increasing flow enrichment shortens the primary p=1 metric edge length.",
        "primary_statistic": "median metric antialignment fraction",
        "value": float(np.nanmedian(dynamics.metric_antialignment_fraction.astype(float))) if len(dynamics) else np.nan,
        "status_rule": "exact local monotonicity predicts value approximately 1 for changed edges",
    })
    post = adam[adam.post_best_interval_count.astype(float) > 0] if len(adam) else adam
    rows.append({
        "hypothesis": "H3.7", "statement": "Positive alignment and concentration continue after the best validation checkpoint.",
        "primary_statistic": "fraction of eligible Adam runs with post-best Delta S_deficit >0",
        "value": float(np.mean(post.post_best_delta_Sdef.astype(float) > 0)) if len(post) else np.nan,
        "status_rule": "best checkpoint selected on validation, never test",
    })
    return pd.DataFrame(rows)


# ======================================================================================
# P3.14 PUBLICATION-STYLE SUMMARY FIGURES
# ======================================================================================


def panel_label(ax, letter: str):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def savefig(fig, stem: str):
    fig.savefig(P3_OUTDIR / "figures" / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(P3_OUTDIR / "figures" / f"{stem}.pdf", bbox_inches="tight")


def _median_band(ax, df, xcol, ycol, groupcol="optimizer", logx=False):
    for name, g in df.groupby(groupcol):
        z = g[[xcol, ycol]].dropna().copy()
        if len(z) == 0: continue
        # exact x checkpoints; aggregate run distributions at each checkpoint
        agg = z.groupby(xcol)[ycol].agg(
            med="median", lo=lambda x: np.percentile(x, 16), hi=lambda x: np.percentile(x, 84)
        ).reset_index().sort_values(xcol)
        ax.plot(agg[xcol], agg.med, label=str(name))
        ax.fill_between(agg[xcol].to_numpy(float), agg.lo.to_numpy(float), agg.hi.to_numpy(float), alpha=0.15)
    if logx: ax.set_xscale("log")


def make_figures(states: pd.DataFrame, dynamics: pd.DataFrame, finals: pd.DataFrame,
                 sign: pd.DataFrame, matched_L: pd.DataFrame, matched_J: pd.DataFrame):
    # Figure 1 — direct kinematics.
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.4))
    _median_band(axs[0,0], dynamics[dynamics.optimizer_universality_arm.astype(bool)], "end_epoch", "corr_G_logp_p0")
    axs[0,0].axhline(0, lw=.8, ls="--"); axs[0,0].set(xlabel="epoch", ylabel=r"$\rho_p(G,\log p)$", title="Selection alignment")
    _median_band(axs[0,1], dynamics[dynamics.optimizer_universality_arm.astype(bool)], "end_epoch", "jeffreys_J")
    axs[0,1].set(xlabel="epoch", ylabel=r"$J(p_{t+1},p_t)$", title="Path-measure redistribution")
    d = dynamics[np.isfinite(dynamics.integrated_selection_covariance.astype(float)) & np.isfinite(dynamics.delta_entropy_deficit.astype(float))]
    if len(d):
        for opt, g in d.groupby("optimizer"):
            axs[1,0].scatter(g.integrated_selection_covariance, g.delta_entropy_deficit, s=9, alpha=.35, label=str(opt))
        lo = min(float(d.integrated_selection_covariance.min()), float(d.delta_entropy_deficit.min()))
        hi = max(float(d.integrated_selection_covariance.max()), float(d.delta_entropy_deficit.max()))
        axs[1,0].plot([lo,hi],[lo,hi], ls="--", lw=.9)
    axs[1,0].set(xlabel=r"$\int_0^1 \mathrm{Cov}_{p_a}(G,\log p_a) da$", ylabel=r"$\Delta S_{deficit}$", title="Exact finite-tilt identity")
    _median_band(axs[1,1], dynamics[dynamics.optimizer_universality_arm.astype(bool)], "end_epoch", "var_G_p0")
    axs[1,1].set(xlabel="epoch", ylabel=r"$\mathrm{Var}_p(G)$", title="Selection intensity")
    for i, ax in enumerate(axs.ravel()): panel_label(ax, chr(97+i))
    handles, labels = axs[0,0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="upper center", ncol=max(1,len(labels)), frameon=False)
    fig.tight_layout(rect=[0,0,1,.95]); savefig(fig, "figure1_path_selection_kinematics"); plt.close(fig)

    # Figure 2 — epoch-time states.
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.4))
    u = states[states.optimizer_universality_arm.astype(bool)]
    specs = [("entropy_deficit", r"$S_{deficit}$", "Path concentration"),
             ("edge_cv", "edge-flow CV", "Marginal heterogeneity"),
             ("log_xi50", r"$\log \xi_{50}$", "Metric scale"),
             ("d_star", r"$d^*$", "Crossover response")]
    for ax, (m,y,t) in zip(axs.ravel(), specs):
        if m in u: _median_band(ax, u, "epoch", m)
        ax.set(xlabel="epoch", ylabel=y, title=t)
    for i, ax in enumerate(axs.ravel()): panel_label(ax, chr(97+i))
    handles, labels = axs[0,0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="upper center", ncol=max(1,len(labels)), frameon=False)
    fig.tight_layout(rect=[0,0,1,.95]); savefig(fig, "figure2_optimizer_state_trajectories"); plt.close(fig)

    # Figure 3 — matched functional progress and structural time.
    fig, axs = plt.subplots(2, 2, figsize=(10.2, 7.4))
    for opt, g in matched_L.groupby("optimizer"):
        agg = g.groupby("coordinate_value")["entropy_deficit"].median().reset_index()
        axs[0,0].plot(agg.coordinate_value, agg.entropy_deficit, label=str(opt))
        agg2 = g.groupby("coordinate_value")["edge_cv"].median().reset_index()
        axs[0,1].plot(agg2.coordinate_value, agg2.edge_cv, label=str(opt))
    for opt, g in matched_J.groupby("optimizer"):
        agg = g.groupby("coordinate_value")["entropy_deficit"].median().reset_index()
        axs[1,0].plot(agg.coordinate_value, agg.entropy_deficit, label=str(opt))
        agg2 = g.groupby("coordinate_value")["edge_cv"].median().reset_index()
        axs[1,1].plot(agg2.coordinate_value, agg2.edge_cv, label=str(opt))
    axs[0,0].set(xlabel=r"functional progress $\tau_L$", ylabel=r"$S_{deficit}$", title="Concentration at matched loss progress")
    axs[0,1].set(xlabel=r"functional progress $\tau_L$", ylabel="edge-flow CV", title="Heterogeneity at matched loss progress")
    axs[1,0].set(xlabel=r"normalized structural time $\tau_J/\tau_J(T)$", ylabel=r"$S_{deficit}$", title="Concentration along structural time")
    axs[1,1].set(xlabel=r"normalized structural time $\tau_J/\tau_J(T)$", ylabel="edge-flow CV", title="Heterogeneity along structural time")
    for i, ax in enumerate(axs.ravel()): panel_label(ax, chr(97+i))
    handles, labels = axs[0,0].get_legend_handles_labels()
    if handles: fig.legend(handles, labels, loc="upper center", ncol=max(1,len(labels)), frameon=False)
    fig.tight_layout(rect=[0,0,1,.95]); savefig(fig, "figure3_matched_progress_and_structural_time"); plt.close(fig)

    # Figure 4 — universality signs.
    if len(sign):
        metrics = ["run_fraction_median_cov_positive", "run_fraction_net_entropy_deficit_up",
                   "run_fraction_edge_cv_up", "run_fraction_xi50_down"]
        labels = ["median alignment > 0", r"$\Delta S_{def}>0$", "edge CV up", r"$\xi_{50}$ down"]
        x = np.arange(len(sign)); width = .18
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        for j, (m, lab) in enumerate(zip(metrics, labels)):
            if m in sign:
                ax.bar(x + (j - 1.5) * width, sign[m].astype(float), width=width, label=lab)
        ax.axhline(.5, ls="--", lw=.8); ax.set_ylim(0,1.05)
        ax.set_xticks(x, sign.optimizer.astype(str)); ax.set_ylabel("fraction of runs")
        ax.set_title("Optimizer-specific directional laws"); ax.legend(frameon=False, ncol=2)
        panel_label(ax, "a"); fig.tight_layout(); savefig(fig, "figure4_optimizer_directional_universality"); plt.close(fig)


# ======================================================================================
# P3.15 AGGREGATION + MAIN EXPERIMENT
# ======================================================================================


def aggregate_run_outputs(manifest: pd.DataFrame):
    states, dyns, geoms, micros = [], [], [], []
    for rid in manifest.run_id.astype(str):
        paths = {
            "s": P3_OUTDIR/"run_state"/f"{rid}.csv",
            "d": P3_OUTDIR/"run_dynamics"/f"{rid}.csv",
            "g": P3_OUTDIR/"run_geometry"/f"{rid}.csv",
            "m": P3_OUTDIR/"run_micro"/f"{rid}.csv",
        }
        if paths["s"].exists(): states.append(pd.read_csv(paths["s"]))
        if paths["d"].exists(): dyns.append(pd.read_csv(paths["d"]))
        if paths["g"].exists(): geoms.append(pd.read_csv(paths["g"]))
        if paths["m"].exists(): micros.append(pd.read_csv(paths["m"]))
    return (
        pd.concat(states, ignore_index=True) if states else pd.DataFrame(),
        pd.concat(dyns, ignore_index=True) if dyns else pd.DataFrame(),
        pd.concat(geoms, ignore_index=True) if geoms else pd.DataFrame(),
        pd.concat(micros, ignore_index=True) if micros else pd.DataFrame(),
    )


def cleanup_if_requested():
    if not P3_START_FRESH:
        return
    for sub in ["run_state", "run_dynamics", "run_geometry", "run_micro", "checkpoints"]:
        for p in (P3_OUTDIR/sub).glob("*"):
            if p.is_file():
                try: p.unlink()
                except Exception: pass


def aggregate_run_divergences(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "run_id", "arch_id", "environment_id", "optimizer", "seed_index",
        "learning_rate", "reason", "failure_type", "timestamp",
    ]
    for rid in manifest.run_id.astype(str):
        p = P3_OUTDIR / "checkpoints" / f"{rid}__DIVERGED.json"
        if p.exists():
            try:
                rows.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                rows.append({"run_id": rid, "reason": "unreadable divergence marker"})
    return pd.DataFrame(rows, columns=columns)


def run_phase3_experiment():
    total_start = time.perf_counter()
    cleanup_if_requested()
    banner("PATH-FLOW RESEARCH PROGRAM — PHASE III OPTIMIZER DYNAMICS")
    validation = internal_math_self_test()
    print("Mathematical self-test: PASS", validation)

    arch, env, p2final = load_phase2_handoff()
    arch, env = augment_handoff_with_phase2(arch, env, p2final)
    arch_panel = select_architectures(arch); env_panel = select_environments(env)
    manifest = build_run_manifest(arch_panel, env_panel)
    learning_rates = calibrate_learning_rates(arch_panel, env_panel)

    experiment_manifest = {
        "phase": "III",
        "scientific_question": "optimizer-induced dynamics of conserved path-measure redistribution",
        "profile": P3_PROFILE, "device": DEVICE,
        "phase2_handoff": {"directory": str(P3_PHASE2_DIR), "architectures_available": int(len(arch)),
                           "environments_available": int(len(env))},
        "design": {
            "mechanistic_adam": {"architectures": int(len(arch_panel)), "environments": int(len(env_panel)),
                                 "seeds": list(map(int,P3_SEEDS))},
            "optimizer_universality": {"architectures": int(arch_panel.phase3_optimizer_subset.sum()),
                                        "environments": int(env_panel.phase3_optimizer_subset.sum()),
                                        "optimizers": list(P3_OPTIMIZERS), "seeds": list(map(int,P3_SEEDS))},
            "unique_main_runs": int(len(manifest)),
        },
        "training": {"max_epoch": int(P3_MAX_EPOCH), "batch_size": int(P3_BATCH_SIZE),
                     "weight_decay": float(P3_WEIGHT_DECAY), "learning_rates": learning_rates,
                     "activation": "ReLU", "bias": False, "initialization": "He/Kaiming normal"},
        "dynamic_observables": ["Var_p(G)", "Cov_p(G,log p)", "Corr_p(G,log p)",
                                "KL(p1||p0)", "KL(p0||p1)", "Jeffreys J", "Delta S_deficit", "tau_J"],
        "finite_interval_identity": "Delta S_deficit = integral_0^1 Cov_{p_a}(G,log p_a) da",
        "geometry": {"metric_power": METRIC_POWER, "bcanw_scales": int(P3_BCANW_SCALES),
                     "strict_scaling": {"min_points": SCALING_MIN_POINTS, "min_span_decades": SCALING_MIN_SPAN_DECADES,
                                        "min_nb_ratio": SCALING_MIN_NB_RATIO, "r2_min": SCALING_R2_MIN,
                                        "slope_cv_max": SCALING_MAX_SLOPE_CV, "delta_bic_max": SCALING_MAX_DELTA_BIC}},
        "hypotheses": {
            "H3.2": "positive Adam path-selection alignment",
            "H3.3": "alignment survives environment variation",
            "H3.4": "optimizer universality vs specificity",
            "H3.5": "flow redistribution follows path selection",
            "H3.6": "flow enrichment shortens primary metric edges",
            "H3.7": "post-best-validation structural motion continues",
        },
    }
    atomic_json(experiment_manifest, P3_OUTDIR / "experiment_manifest.json")

    # Generate each selected environment once, then run all matching architecture/optimizer/seed cells.
    amap = {str(r.arch_id): r for _, r in arch_panel.iterrows()}
    emap = {str(r.environment_id): r for _, r in env_panel.iterrows()}
    total = len(manifest); eta = ETAReporter("Phase-III main runs", total); bar = tqdm(total=total, desc="Phase III", leave=True)
    completed = 0
    for eid in env_panel.environment_id.astype(str):
        er = emap[eid]; splits, env_summary = generate_environment(er)
        block = manifest[manifest.environment_id.astype(str) == eid]
        for _, rr in block.iterrows():
            st = time.perf_counter(); ar = amap[str(rr.arch_id)]
            try:
                run_one(rr, ar, er, splits, learning_rates)
            except RunDivergenceError as exc:
                fail = {
                    "run_id": str(rr.run_id), "arch_id": str(rr.arch_id),
                    "environment_id": str(rr.environment_id), "optimizer": str(rr.optimizer),
                    "seed_index": int(rr.seed_index),
                    "learning_rate": float(learning_rates[str(rr.optimizer)]),
                    "reason": str(exc), "failure_type": "RunDivergenceError",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                atomic_json(fail, P3_OUTDIR / "checkpoints" / f"{rr.run_id}__DIVERGED.json")
                tqdm.write(f"[DIVERGED] {rr.run_id}: {exc}")
                if not P3_CONTINUE_ON_RUN_DIVERGENCE:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            completed += 1; bar.update(1); eta.update(time.perf_counter()-st, completed)
        # release environment tensors before generating the next one
        del splits
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    bar.close(); eta.finish()

    states, dynamics, geometry_profiles, micro = aggregate_run_outputs(manifest)
    divergences = aggregate_run_divergences(manifest)
    states = add_time_coordinates(states)
    finals = build_final_summary(states, dynamics)
    sign = optimizer_sign_summary(finals, dynamics)
    contrasts = paired_optimizer_contrasts(finals)
    matched_L = matched_progress_table(states[states.optimizer_universality_arm.astype(bool)], "tau_L")
    matched_J = matched_progress_table(states[states.optimizer_universality_arm.astype(bool)], "tau_J_fraction")
    hypotheses = hypothesis_table(finals, sign, dynamics)

    atomic_csv(states, P3_OUTDIR / "tables" / "state_summary.csv")
    atomic_csv(dynamics, P3_OUTDIR / "tables" / "dynamic_intervals.csv")
    atomic_csv(finals, P3_OUTDIR / "tables" / "final_summary.csv")
    atomic_csv(sign, P3_OUTDIR / "tables" / "optimizer_directional_summary.csv")
    atomic_csv(contrasts, P3_OUTDIR / "tables" / "optimizer_paired_contrasts.csv")
    atomic_csv(matched_L, P3_OUTDIR / "tables" / "functional_progress_matched.csv")
    atomic_csv(matched_J, P3_OUTDIR / "tables" / "structural_time_matched.csv")
    atomic_csv(hypotheses, P3_OUTDIR / "tables" / "hypothesis_tests.csv")
    if len(geometry_profiles): atomic_csv(geometry_profiles, P3_OUTDIR / "tables" / "geometry_scale_profiles.csv")
    if len(micro): atomic_csv(micro, P3_OUTDIR / "tables" / "microstep_dynamics.csv")
    atomic_csv(divergences, P3_OUTDIR / "tables" / "run_divergences.csv")

    make_figures(states, dynamics, finals, sign, matched_L, matched_J)

    elapsed = time.perf_counter() - total_start
    completion = {
        "completed": True, "elapsed_seconds": float(elapsed), "elapsed": human_time(elapsed),
        "unique_runs": int(len(manifest)),
        "completed_run_files": int(states.run_id.astype(str).nunique()) if len(states) and "run_id" in states else 0,
        "diverged_runs": int(len(divergences)),
        "state_rows": int(len(states)), "dynamic_intervals": int(len(dynamics)),
        "geometry_profile_rows": int(len(geometry_profiles)), "micro_intervals": int(len(micro)),
        "output_directory": str(P3_OUTDIR),
    }
    atomic_json(completion, P3_OUTDIR / "completion.json")
    banner("PHASE III COMPLETE")
    print(json.dumps(completion, indent=2))
    print("Core tables:")
    print("  ", P3_OUTDIR / "tables" / "final_summary.csv")
    print("  ", P3_OUTDIR / "tables" / "dynamic_intervals.csv")
    print("  ", P3_OUTDIR / "tables" / "hypothesis_tests.csv")
    return {
        "manifest": manifest, "states": states, "dynamics": dynamics, "finals": finals,
        "sign_summary": sign, "paired_contrasts": contrasts, "hypotheses": hypotheses,
        "matched_functional": matched_L, "matched_structural": matched_J,
        "divergences": divergences,
    }


# ======================================================================================
# P3.16 EXECUTE — THE ENTIRE FILE IS DESIGNED TO BE PASTED INTO ONE JUPYTER CELL
# ======================================================================================

PHASE3_RESULTS = None
if __name__ == "__main__":
    PHASE3_RESULTS = run_phase3_experiment()

