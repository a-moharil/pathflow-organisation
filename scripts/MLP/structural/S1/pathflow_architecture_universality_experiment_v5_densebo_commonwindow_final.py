# PATH-FLOW ARCHITECTURE UNIVERSALITY EXPERIMENT V2 — COMPARABLE TEMPORAL GEOMETRY + COMPLETE V5 PLOT SUITE
# ===========================================================================
# Single-file / single-Jupyter-cell experiment.
#gg
# Scientific question
# -------------------
# How much of the gauge-invariant path-flow geometry of a bias-free ReLU MLP is
# imposed by architecture + He initialization, and how much is deformed by learning?
#
# The experiment has two coupled maps:
#   MAP A: architecture -> initialization structural phenotype S0(A)
#          Sobol cold start + SMAC Bayesian optimization of structural novelty.
#   MAP B1: a FIXED longitudinal cohort is trained to 160 epochs and analyzed at the
#           same geometry checkpoints. This cohort is never pruned, so variance across
#           epochs is not confounded by survivor bias.
#   MAP B2: the remaining selected architectures use conservative multi-fidelity racing.
#
# Environment is intentionally fixed:
#   * synthetic teacher-student TABULAR regression
#   * 60,000 train + 10,000 validation + 10,000 test samples by default
#   * 32 input attributes by default, so N_train / d = 1875
#   * fixed teacher, fixed noise, fixed data ordering, fixed optimizer
#
# Search axes are architecture only:
#   * hidden depth
#   * target parameter budget
#   * morphology: constant / taper / expansion / bottleneck / hourglass
#   * morphology strength and curvature
#
# The path-flow object is the same foundational object as V5:
#   Q_pi = prod_e w_e^2
#   p_pi = Q_pi / sum_pi Q_pi
#   F_e  = sum_{pi contains e} p_pi
#   F_i  = sum_{pi contains i} p_pi
#   Fcal_e = |E_l| F_e
#   lambda_e = (Fcal_e + eps)^(-p), p=1 primary
#
# IMPORTANT:
#   - This is NOT ordinary HPO. SMAC minimizes exp(-novelty), where novelty is the
#     structural distance of S0(A) from the cold-start phenotype cloud.
#   - EI supplies model-based exploration/exploitation; 15% of proposals are random.
#   - Training promotion is a deliberately conservative scientific intensifier based on
#     structural novelty, deformation, structural velocity, and functional viability.
#   - Dataset/optimizer/activation are deliberately held fixed in this experiment.
#
# Traceability
# ------------
# Every long stage has tqdm progress bars. After each completed expensive item the script
# prints moving-average time/item and projected ETA. Every architecture/rung prints a
# detailed structured status line. Intermediate CSV/JSON outputs are flushed to disk so
# interrupted Colab runs retain completed work.
#
# Plot-consistency principle
# --------------------------
# Every trained architecture is rendered with the SAME V5-style axes, epoch-color mapping,
# scale coordinate x=ell/xi50, and panel definitions. The fixed longitudinal cohort also
# yields scale-by-epoch ensemble means and variance bands / heatmaps.
#
# COMPLETE PLOT SUITE + EXHAUSTIVE PDF REPORT
# -------------------------------------------
# The script now reproduces, per architecture wherever the required diagnostics exist:
#   V5 Figure 1  — path-space formation
#   V5 Figure 2  — microscopic flow + same-tail model comparison
#   V5 Figure 3  — intrinsic covering / d_eff / lacunarity / crossover
#   V5 Figure 4  — local dimensions + directed renormalization
#   V5 Figure S1 — metric-power + greedy-cover robustness
#   Explore 01    — temporal structural dashboard
#   Explore 02    — master-curve residual + scale-time fields
#   Explore 03    — finite scaling window vs global crossover
#   Explore 04    — microscopic layer hierarchy
#   Explore 05    — node-level local geometry by depth
#   Explore 06    — directed quotient-flow KS + Renyi spectrum
#   Explore 07    — coupled temporal phase-space trajectories
#   Explore 08    — robustness
#   Explore 09    — temporal Spearman correlation matrix
# Existing cross-architecture ribbons, scale-time mean/variance fields, preservation,
# PCA trajectories, and final architecture comparison figures are retained.
#
# Paste this entire file into ONE Jupyter cell and run.

from __future__ import annotations

import os
import sys
import math
import json
import time
import random
import hashlib
import warnings
import textwrap
from contextlib import nullcontext
import subprocess
import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ======================================================================================
# 0. OPTIONAL DEPENDENCY BOOTSTRAP
# ======================================================================================

AUTO_INSTALL_SMAC = True
SMAC_AVAILABLE = False


def _try_import_smac():
    global SMAC_AVAILABLE
    try:
        import smac  # noqa: F401
        import ConfigSpace  # noqa: F401
        SMAC_AVAILABLE = True
        return True
    except Exception:
        SMAC_AVAILABLE = False
        return False


if not _try_import_smac() and AUTO_INSTALL_SMAC:
    print("[setup] SMAC / ConfigSpace not found. Attempting a quiet pip install...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "smac", "ConfigSpace"
        ])
        _try_import_smac()
    except Exception as exc:
        print(f"[setup] SMAC installation was unavailable ({exc}).")
        print("[setup] Falling back to an internal RF-LCB Bayesian explorer. The experiment remains runnable,")
        print("        but install `smac ConfigSpace` in Colab to use the intended SMAC engine.")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn

from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from scipy.stats import qmc, ks_2samp, norm, wasserstein_distance, spearmanr
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import minimize
from scipy.integrate import quad

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ======================================================================================
# 1. CONFIGURATION
# ======================================================================================

RUN_PROFILE = "research"   # "research" or "pilot"
SEED = 31
MODEL_INIT_SEED = 31415     # SAME initialization seed for every architecture
TEACHER_SEED = 271828
DATA_ORDER_SEED = 161803
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed task environment: deliberately N >> d.
INPUT_DIM = 32
N_TRAIN = 60_000
N_VAL = 10_000
N_TEST = 10_000
TEACHER_HIDDEN = [512, 256, 256, 128, 128, 112, 95, 64, 53, 32, 27, 16, 16, 8]
TEACHER_NOISE_STD = 0.05     # fraction of clean target standard deviation before y standardization

# Fixed training exposure.
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 0.0
BUDGETS = [260, 360, 400]
EVAL_EVERY = 2

# Longitudinal geometry design.
# These exact checkpoints and plotting conventions are shared by every fixed-cohort architecture.
GEOMETRY_CHECKPOINTS = [0, 1, 2, 5, 8, 12, 16, 20, 30, 40, 50, 60, 80, 120, 160, 200, 250, 300, 350, 399]
LONGITUDINAL_COHORT = 30
LONGITUDINAL_BCANW_SCALES = 72
START_FRESH = True   # clear stale model checkpoints/curve caches from a previous run

# Common scale coordinate for cross-architecture comparisons.
COMMON_X_MIN = 0.08
COMMON_X_MAX = 20.0
COMMON_X_POINTS = 180
ENSEMBLE_PERCENTILES = (16.0, 50.0, 84.0)

# Architecture configuration space.
# V5 deliberately keeps the architecture family deep (>8 hidden layers) but removes the
# extreme depth/low-capacity combinations that caused many layers to collapse onto MIN_WIDTH.
DEPTH_MIN, DEPTH_MAX = 9, 32
PARAM_MIN, PARAM_MAX = 20_000, 180_000
MIN_WIDTH, MAX_WIDTH = 8, 512
MORPHOLOGIES = ["constant", "taper", "expansion", "bottleneck", "hourglass"]
SHAPE_MIN, SHAPE_MAX = 0.45, 0.95
CURVE_MIN, CURVE_MAX = 0.60, 2.00
MAX_GRAPH_NODES = 6_800

# Realized-architecture validity constraints.  These are evaluated AFTER width synthesis,
# so we constrain the actual graph rather than only the requested hyperparameters.
ARCH_MIN_MEDIAN_WIDTH = 18
ARCH_MIN_Q10_WIDTH = 12
ARCH_MAX_FLOOR_FRACTION = 0.20
ARCH_MAX_CEILING_FRACTION = 0.20
ARCH_PARAM_RATIO_MIN = 0.80
ARCH_PARAM_RATIO_MAX = 1.25
ARCH_MAX_RESAMPLE_ATTEMPTS = 250

# Dense stratified initialization map.  Every morphology x depth stratum x parameter stratum
# receives the same number of Sobol points.  5 x 6 x 4 x 2 = 240 initial architectures.
MAP_DEPTH_BINS = 6
MAP_PARAM_BINS = 4

# 5 morphologies × 6 depth bins × 4 parameter bins × 5 samples/cell
# = 600 space-filling cold-start experiments
MAP_SOBOL_PER_CELL = 5

MAP_SOBOL = (
    len(MORPHOLOGIES)
    * MAP_DEPTH_BINS
    * MAP_PARAM_BINS
    * MAP_SOBOL_PER_CELL
)

# Adaptive exploration after the dense initial map
MAP_BO = 400
# BO exploration/exploitation controls.  EI remains the exploitation engine, while a larger
# random-design probability, nonzero xi, a hard cell-density guard, and periodic strict
# under-covered-cell selection prevent the surrogate from collapsing onto one local ridge.
SMAC_RANDOM_PROBABILITY = 0.25
SMAC_EI_XI = 0.15
SMAC_RF_TREES = 96
SMAC_TRIAL_BUDGET_MULTIPLIER = 12
BO_FORCE_UNDERCOVERED_EVERY = 4
BO_CELL_HARD_SLACK = 3
BO_MAX_REJECTIONS_PER_ACCEPTED = 48
BO_STRUCTURAL_WEIGHT = 0.60
BO_CONFIG_DISTANCE_WEIGHT = 0.25
BO_COVERAGE_WEIGHT = 0.15
NOVELTY_K = 3

# MAP B: larger structurally diverse longitudinal cohort.
TRAIN_START = 48
PROMOTION_FRACTIONS = [0.50, 0.50, 0.50]
RANDOM_RESCUE_FRACTION = 0.10
FINAL_EXTRA_SEEDS = 0

# Geometry throughput / strictness.
EPS = 1e-14
METRIC_POWER = 1.0
MAP_BCANW_SCALES = 52
TRAIN_BCANW_SCALES = 72
LOCAL_SLOPE_WINDOW = 7
LOCAL_SLOPE_R2_MIN = 0.90
SCALING_R2_MIN = 0.97
SCALING_MIN_DIM = 0.25
SCALING_MIN_POINTS = 6
SCALING_MIN_SPAN_DECADES = 0.50
SCALING_MIN_NB_RATIO = 5.0
SCALING_MAX_SLOPE_CV = 0.25
SCALING_MAX_DELTA_BIC = 2.0

# Full diagnostics are computed only for final promoted architectures.
LOCAL_DEPTH_GRID = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
RENORM_TARGET_BOX_FRACTIONS = np.array([0.50, 0.25, 0.125, 0.0625])
RENYI_Q = [0.5, 1.0, 2.0, 3.0]

# Strict common-window local-dimension estimator.  Support and flow-mass exponents are now
# fitted on EXACTLY the same radial interval before Delta_i^F = alpha_i^F - d_i^F is formed.
LOCAL_COMMON_MIN_POINTS = 7
LOCAL_COMMON_MIN_SPAN_DECADES = 0.50
LOCAL_COMMON_SUPPORT_GROWTH = 4.0
LOCAL_COMMON_MASS_GROWTH = 3.0
LOCAL_COMMON_R2_MIN = 0.95
LOCAL_COMMON_SATURATION = 0.95
LOCAL_LAYER_MIN_VALID_FRACTION = 0.20

# Comprehensive V5/exploratory diagnostics requested for architecture comparison.
MIN_TAIL_POINTS = 50
TAIL_XMIN_QUANTILES = np.linspace(0.50, 0.90, 21)
# Full metric-family sensitivity requested for the exhaustive report.
# Negative p reverses the usual path-flow distance semantics, so those rows are explicitly
# treated as sign-reversed diagnostic sensitivity rather than primary geometry.
METRIC_POWER_ROBUSTNESS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
COVER_ORDER_REPEATS = 8
COVER_ORDER_NOISE = 0.02
RUN_COMPREHENSIVE_FINAL_DIAGNOSTICS = True
REQUIRE_EXHAUSTIVE_PER_ARCH = True   # every selected architecture gets full 0->max epoch longitudinal analysis
FINAL_REPORT_FILENAME = "pathflow_architecture_universality_EXHAUSTIVE_RESULTS.pdf"
COMPREHENSIVE_PNG_DPI = 300
SAVE_INDIVIDUAL_PNG = True
SAVE_INDIVIDUAL_FIGURE_PDF = False   # final exhaustive PDF already contains every panel
WRITE_PER_ARCHITECTURE_BOOKS = False # turn on only if separate multipage books are also needed
SHOW_COMPREHENSIVE_FIGURES_IN_NOTEBOOK = False  # files are always saved; set True to display hundreds of panels

# Visual consistency for the two diagnostics the project now treats as first-class.
ENTROPY_COLOR = "#B279A2"
ENTROPY_DARK = "#7A5195"
LOCAL_SUPPORT_COLOR = "#4C78A8"
LOCAL_ALPHA_COLOR = "#E45756"
LOCAL_DELTA_COLOR = "#54A24B"
MIN_SPEARMAN_POINTS = 4

OUTDIR = Path("pathflow_architecture_universality_v5_dense_results")
OUTDIR.mkdir(exist_ok=True)
(OUTDIR / "curves").mkdir(exist_ok=True)
(OUTDIR / "checkpoints").mkdir(exist_ok=True)
(OUTDIR / "smac").mkdir(exist_ok=True)
(OUTDIR / "longitudinal_curves").mkdir(exist_ok=True)
(OUTDIR / "architecture_figures").mkdir(exist_ok=True)
(OUTDIR / "microscopic").mkdir(exist_ok=True)
(OUTDIR / "comprehensive_figures").mkdir(exist_ok=True)
(OUTDIR / "comprehensive_tables").mkdir(exist_ok=True)

if START_FRESH:
    for sub,pattern in [("checkpoints","*.pt"),("longitudinal_curves","*.npz"),
                        ("microscopic","*.npz")]:
        for old in (OUTDIR/sub).glob(pattern):
            try: old.unlink()
            except Exception: pass

if RUN_PROFILE == "pilot":
    N_TRAIN, N_VAL, N_TEST = 6_000, 1_000, 1_000
    DEPTH_MIN, DEPTH_MAX = 9, 10
    PARAM_MIN, PARAM_MAX = 15_000, 30_000
    MAX_WIDTH = 96
    MAP_DEPTH_BINS, MAP_PARAM_BINS, MAP_SOBOL_PER_CELL = 1, 1, 1
    MAP_SOBOL, MAP_BO = len(MORPHOLOGIES), 3
    TRAIN_START = 4
    BUDGETS = [2, 4, 8]
    PROMOTION_FRACTIONS = [0.50, 0.50]
    MAP_BCANW_SCALES = 20
    TRAIN_BCANW_SCALES = 24
    LONGITUDINAL_BCANW_SCALES = 24
    LONGITUDINAL_COHORT = 2
    GEOMETRY_CHECKPOINTS = [0, 1, 2, 4, 8]
    EVAL_EVERY = 1

# Reproducibility.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_float32_matmul_precision("high")

# ======================================================================================
# 2. TRACEABILITY + ETA UTILITIES
# ======================================================================================


def human_time(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class ETAReporter:
    def __init__(self, stage: str, total: int, window: int = 12):
        self.stage = stage
        self.total = max(int(total), 1)
        self.window = max(int(window), 3)
        self.start = time.perf_counter()
        self.durations: List[float] = []

    def update(self, duration: float, completed: int):
        duration = float(duration)
        self.durations.append(duration)
        recent = np.asarray(self.durations[-min(self.window, len(self.durations)):], float)
        # Median is deliberately used for ETA because BCANw occasionally has a single slow scale.
        typical = float(np.median(recent))
        mean_recent = float(np.mean(recent))
        remaining = max(self.total - completed, 0)
        eta = typical * remaining
        elapsed = time.perf_counter() - self.start
        tqdm.write(
            f"[ETA:{self.stage}] {completed}/{self.total} | current {human_time(duration)} | "
            f"recent median {human_time(typical)}/item (mean {human_time(mean_recent)}) | "
            f"elapsed {human_time(elapsed)} | remaining ~{human_time(eta)}"
        )

    def finish(self):
        elapsed = time.perf_counter() - self.start
        print(f"[stage complete] {self.stage}: {human_time(elapsed)}")


def banner(title: str):
    line = "=" * 94
    print(f"\n{line}\n{title}\n{line}")


def config_key_dict(d: Dict[str, Any]) -> str:
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def arch_id_from_widths(widths: List[int]) -> str:
    h = hashlib.sha1(json.dumps(list(map(int, widths))).encode()).hexdigest()[:10]
    return f"A_{h}"


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

# ======================================================================================
# 3. JOURNAL FIGURE STYLE
# ======================================================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 130,
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def savefig(fig, name: str):
    fig.savefig(OUTDIR / f"{name}.png", dpi=500)
    fig.savefig(OUTDIR / f"{name}.pdf")


def _pdf_save(book, fig):
    """Save a fixed-size page to PdfPages regardless of the global tight-bbox setting."""
    if book is None:
        return
    with mpl.rc_context({"savefig.bbox": None}):
        book.savefig(fig)

# ======================================================================================
# 4. FIXED SYNTHETIC TASK ENVIRONMENT (> 50K TABULAR POINTS)
# ======================================================================================


class BiasFreeReLUMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int = 1, seed: int = MODEL_INIT_SEED):
        super().__init__()
        torch.manual_seed(int(seed))
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


def make_environment():
    banner("STAGE 0 — GENERATING FIXED TEACHER-STUDENT TABULAR ENVIRONMENT")
    t0 = time.perf_counter()
    n_total = N_TRAIN + N_VAL + N_TEST
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_total, INPUT_DIM), dtype=np.float32)

    # Standardize using train split only (normally almost identity here, but methodologically explicit).
    x_mean = X[:N_TRAIN].mean(axis=0, keepdims=True)
    x_std = X[:N_TRAIN].std(axis=0, keepdims=True) + 1e-8
    X = ((X - x_mean) / x_std).astype(np.float32)

    teacher = BiasFreeReLUMLP(INPUT_DIM, TEACHER_HIDDEN, 1, seed=TEACHER_SEED).to(DEVICE)
    teacher.eval()
    y_clean = np.empty((n_total, 1), np.float32)
    chunk = 8192
    with torch.no_grad():
        for lo in tqdm(range(0, n_total, chunk), desc="Teacher generating targets", unit="chunk"):
            hi = min(lo + chunk, n_total)
            xb = torch.from_numpy(X[lo:hi]).to(DEVICE)
            y_clean[lo:hi] = teacher(xb).cpu().numpy()

    clean_sd = float(y_clean[:N_TRAIN].std())
    noise = rng.normal(0.0, TEACHER_NOISE_STD * max(clean_sd, 1e-8), size=y_clean.shape).astype(np.float32)
    y = y_clean + noise

    y_mean = float(y[:N_TRAIN].mean())
    y_std = float(y[:N_TRAIN].std()) + 1e-8
    y = ((y - y_mean) / y_std).astype(np.float32)

    splits = {
        "train": (torch.from_numpy(X[:N_TRAIN]), torch.from_numpy(y[:N_TRAIN])),
        "val": (torch.from_numpy(X[N_TRAIN:N_TRAIN+N_VAL]), torch.from_numpy(y[N_TRAIN:N_TRAIN+N_VAL])),
        "test": (torch.from_numpy(X[N_TRAIN+N_VAL:]), torch.from_numpy(y[N_TRAIN+N_VAL:])),
    }

    # Empirical signal-to-noise proxy before standardization.
    noise_var = float(np.var(noise[:N_TRAIN]))
    signal_var = float(np.var(y_clean[:N_TRAIN]))
    snr = signal_var / max(noise_var, 1e-15)
    elapsed = time.perf_counter() - t0

    summary = {
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_total": n_total,
        "input_dim": INPUT_DIM,
        "train_samples_per_attribute": N_TRAIN / INPUT_DIM,
        "teacher_hidden": TEACHER_HIDDEN,
        "teacher_noise_std_fraction": TEACHER_NOISE_STD,
        "empirical_signal_to_noise_ratio": snr,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "elapsed_seconds": elapsed,
    }
    (OUTDIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if N_TRAIN < 50_000 and RUN_PROFILE != "pilot":
        raise RuntimeError("Research profile requires >=50,000 training samples.")
    print(f"[data adequacy] N_train / d = {N_TRAIN/INPUT_DIM:.1f}; sample size is far larger than dimensionality.")
    print(f"[stage complete] environment generation: {human_time(elapsed)}")
    return splits, summary

# ======================================================================================
# 5. ARCHITECTURE MORPHOLOGY PARAMETERIZATION
# ======================================================================================


def profile_for_morphology(depth: int, morphology: str, rho: float, curvature: float) -> np.ndarray:
    z = np.linspace(0.0, 1.0, depth)
    rho = float(np.clip(rho, SHAPE_MIN, SHAPE_MAX))
    curvature = float(np.clip(curvature, CURVE_MIN, CURVE_MAX))
    if morphology == "constant":
        p = np.ones(depth)
    elif morphology == "taper":
        p = np.power(rho, np.power(z, curvature))
    elif morphology == "expansion":
        p = np.power(rho, np.power(1.0-z, curvature))
    elif morphology == "bottleneck":
        p = np.ones(depth)
        center = (depth - 1) / 2.0
        width = max(depth / 6.0, 0.75)
        dip = np.exp(-0.5 * ((np.arange(depth)-center)/width)**2)
        p = 1.0 - (1.0-rho) * dip
    elif morphology == "hourglass":
        p = rho + (1.0-rho) * np.power(np.abs(2.0*z - 1.0), curvature)
    else:
        raise ValueError(morphology)
    return np.clip(p, 1e-3, None)


def param_count_from_widths(widths: List[int], in_dim: int = INPUT_DIM, out_dim: int = 1) -> int:
    allw = [in_dim] + list(map(int, widths)) + [out_dim]
    return int(sum(allw[i]*allw[i+1] for i in range(len(allw)-1)))


def architecture_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    depth = int(config["depth"])
    target = int(config["target_params"])
    morphology = str(config["morphology"])
    rho = float(config.get("shape", 1.0 if morphology == "constant" else 0.70))
    curvature = float(config.get("curvature", 1.0))
    if morphology == "constant":
        rho, curvature = 1.0, 1.0

    p = profile_for_morphology(depth, morphology, rho, curvature)

    def make(scale):
        w = np.rint(scale * p).astype(int)
        return np.clip(w, MIN_WIDTH, MAX_WIDTH).tolist()

    # Binary search for multiplicative width scale matching target parameter count.
    lo, hi = 1.0, float(MAX_WIDTH*2)
    best = None
    best_err = float("inf")
    for _ in range(60):
        mid = 0.5*(lo+hi)
        widths = make(mid)
        pc = param_count_from_widths(widths)
        err = abs(math.log(max(pc,1)/max(target,1)))
        if err < best_err:
            best_err = err
            best = (widths, pc)
        if pc < target:
            lo = mid
        else:
            hi = mid

    widths, actual = best
    allw = [INPUT_DIM] + widths + [1]
    nodes = int(sum(allw))
    edges = actual
    if nodes > MAX_GRAPH_NODES:
        # Rescale widths down just enough for graph tractability, then report realized budget.
        scale = MAX_GRAPH_NODES / nodes
        widths = [max(MIN_WIDTH, int(round(x*scale))) for x in widths]
        actual = param_count_from_widths(widths)
        allw = [INPUT_DIM] + widths + [1]
        nodes = int(sum(allw))
        edges = actual

    w_arr = np.asarray(widths, float)
    return {
        "hidden_widths": widths,
        "depth": depth,
        "target_params": target,
        "actual_params": actual,
        "param_budget_ratio": actual / target,
        "morphology": morphology,
        "shape": rho,
        "curvature": curvature,
        "n_nodes": nodes,
        "n_edges": edges,
        "width_mean": float(np.mean(widths)),
        "width_median": float(np.median(widths)),
        "width_q10": float(np.quantile(w_arr, .10)),
        "width_min": int(np.min(widths)),
        "width_max": int(np.max(widths)),
        "width_cv": float(np.std(widths)/max(np.mean(widths), EPS)),
        "floor_fraction": float(np.mean(w_arr <= MIN_WIDTH)),
        "ceiling_fraction": float(np.mean(w_arr >= MAX_WIDTH)),
    }


def architecture_feasibility(arch: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate the REALIZED architecture; no geometry/training is spent on pathological clips."""
    reasons = []
    if float(arch.get("width_median", 0)) < ARCH_MIN_MEDIAN_WIDTH:
        reasons.append(f"median_width<{ARCH_MIN_MEDIAN_WIDTH}")
    if float(arch.get("width_q10", 0)) < ARCH_MIN_Q10_WIDTH:
        reasons.append(f"q10_width<{ARCH_MIN_Q10_WIDTH}")
    if float(arch.get("floor_fraction", 1)) > ARCH_MAX_FLOOR_FRACTION:
        reasons.append(f"floor_fraction>{ARCH_MAX_FLOOR_FRACTION}")
    if float(arch.get("ceiling_fraction", 1)) > ARCH_MAX_CEILING_FRACTION:
        reasons.append(f"ceiling_fraction>{ARCH_MAX_CEILING_FRACTION}")
    ratio = float(arch.get("param_budget_ratio", np.nan))
    if (not np.isfinite(ratio)) or ratio < ARCH_PARAM_RATIO_MIN or ratio > ARCH_PARAM_RATIO_MAX:
        reasons.append("parameter_budget_distorted")
    if int(arch.get("n_nodes", MAX_GRAPH_NODES + 1)) > MAX_GRAPH_NODES:
        reasons.append("graph_node_limit")
    return len(reasons) == 0, reasons


def depth_bin_index(depth: int) -> int:
    u = (int(depth) - DEPTH_MIN) / max((DEPTH_MAX - DEPTH_MIN + 1), 1)
    return int(np.clip(math.floor(u * MAP_DEPTH_BINS), 0, MAP_DEPTH_BINS - 1))


def param_bin_index(target_params: int) -> int:
    u = (math.log(max(int(target_params), PARAM_MIN)) - math.log(PARAM_MIN)) / max(
        math.log(PARAM_MAX) - math.log(PARAM_MIN), EPS
    )
    return int(np.clip(math.floor(u * MAP_PARAM_BINS), 0, MAP_PARAM_BINS - 1))


def coverage_cell_from_config(config: Dict[str, Any]) -> Tuple[str, int, int]:
    return (str(config["morphology"]), depth_bin_index(int(config["depth"])),
            param_bin_index(int(config["target_params"])))


def all_coverage_cells():
    return [(m, d, p) for m in MORPHOLOGIES for d in range(MAP_DEPTH_BINS) for p in range(MAP_PARAM_BINS)]


def coverage_counts_from_rows(rows):
    counts = {cell: 0 for cell in all_coverage_cells()}
    for r in rows:
        try:
            cfg = json.loads(r["config_json"]) if isinstance(r.get("config_json"), str) else r
            cell = coverage_cell_from_config(cfg)
            counts[cell] = counts.get(cell, 0) + 1
        except Exception:
            pass
    return counts


def manual_config_from_unit(u: np.ndarray) -> Dict[str, Any]:
    depth = int(DEPTH_MIN + math.floor(float(u[0])*(DEPTH_MAX-DEPTH_MIN+1)))
    depth = min(depth, DEPTH_MAX)
    logp = math.log(PARAM_MIN) + float(u[1])*(math.log(PARAM_MAX)-math.log(PARAM_MIN))
    target = int(round(math.exp(logp)))
    mi = min(int(math.floor(float(u[2])*len(MORPHOLOGIES))), len(MORPHOLOGIES)-1)
    morphology = MORPHOLOGIES[mi]
    shape = SHAPE_MIN + float(u[3])*(SHAPE_MAX-SHAPE_MIN)
    curvature = CURVE_MIN + float(u[4])*(CURVE_MAX-CURVE_MIN)
    d = {"depth": depth, "target_params": target, "morphology": morphology}
    if morphology != "constant":
        d["shape"] = shape
        d["curvature"] = curvature
    return d

# ======================================================================================
# 6. GAUGE-INVARIANT COMPLETE-PATH FLOW
# ======================================================================================


def extract_weights(model: BiasFreeReLUMLP) -> List[np.ndarray]:
    return [layer.weight.detach().cpu().numpy().astype(np.float64).copy() for layer in model.linears]


def path_factor_matrices(weights: List[np.ndarray]) -> List[np.ndarray]:
    return [np.square(np.asarray(W, float)) for W in weights]


def log_partition(A: List[np.ndarray], q: float) -> float:
    z = np.ones(A[0].shape[1], dtype=np.float64)
    z /= max(z.sum(), EPS)
    log_scale = math.log(A[0].shape[1])
    for M in A:
        Mq = np.power(np.clip(M, EPS, None), q)
        z = Mq @ z
        s = float(z.sum())
        if s <= 0 or not np.isfinite(s):
            return -np.inf
        z /= s
        log_scale += math.log(s)
    return float(log_scale)


def path_partition_stats(A: List[np.ndarray], widths: List[int]) -> Dict[str, float]:
    logZ1 = log_partition(A, 1.0)
    logZ2 = log_partition(A, 2.0)
    logNtopo = float(np.sum(np.log(np.asarray(widths, float))))
    dq = 1e-4
    dlogZ = (log_partition(A, 1+dq)-log_partition(A, 1-dq))/(2*dq)
    S = logZ1 - dlogZ
    log_eta1 = S - logNtopo
    log_eta2 = 2*logZ1 - logZ2 - logNtopo
    return {
        "log_N_topological": logNtopo,
        "S_path": float(S),
        "entropy_deficit": float(max(logNtopo-S, 0.0)),
        "log_eta1": float(log_eta1),
        "log_eta2": float(log_eta2),
        "eta1": float(np.exp(np.clip(log_eta1, -745, 0))),
        "eta2": float(np.exp(np.clip(log_eta2, -745, 0))),
    }


def normalized_forward_backward(A: List[np.ndarray]):
    alpha = [np.ones(A[0].shape[1], float)]
    alpha[0] /= alpha[0].sum()
    for M in A:
        z = M @ alpha[-1]
        s = float(z.sum())
        if s <= EPS or not np.isfinite(s):
            raise FloatingPointError("Degenerate forward partition")
        alpha.append(z/s)
    beta = [None]*(len(A)+1)
    beta[-1] = np.ones(A[-1].shape[0], float)
    beta[-1] /= beta[-1].sum()
    for l in range(len(A)-1, -1, -1):
        z = A[l].T @ beta[l+1]
        s = float(z.sum())
        if s <= EPS or not np.isfinite(s):
            raise FloatingPointError("Degenerate backward partition")
        beta[l] = z/s
    return alpha, beta


def path_flow(A: List[np.ndarray]):
    alpha, beta = normalized_forward_backward(A)
    edge_flow = []
    for l, M in enumerate(A):
        E = M * beta[l+1][:,None] * alpha[l][None,:]
        E /= max(float(E.sum()), EPS)
        edge_flow.append(E)
    node_flow = []
    for l in range(len(alpha)):
        v = alpha[l]*beta[l]
        v /= max(float(v.sum()), EPS)
        node_flow.append(v)
    return edge_flow, node_flow


def flow_enrichment(edge_flow, node_flow):
    return [E*E.size for E in edge_flow], [v*len(v) for v in node_flow]


def flow_concentration(edge_enr, node_enr):
    def summarize(layers):
        vals = []
        for x in layers:
            a = np.asarray(x,float).ravel()
            m = max(float(a.mean()), EPS)
            topn = max(1, int(math.ceil(.10*len(a))))
            vals.append((float(a.std()/m), float(np.sort(a)[-topn:].sum()/max(a.sum(),EPS))))
        return float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))
    ecv, etop = summarize(edge_enr)
    ncv, ntop = summarize(node_enr)
    return ecv, etop, ncv, ntop


# ======================================================================================
# 6B. TAIL MODELS + GLOBAL TRANSITION / RENORMALIZATION SUMMARY HELPERS
# ======================================================================================

def ccdf_positive(x):
    x=np.asarray(x,float).ravel()
    x=x[np.isfinite(x)&(x>0)]
    if len(x)==0: return np.array([]),np.array([])
    x=np.sort(x)
    return x,(len(x)-np.arange(len(x)))/len(x)


def rank_curve(x):
    x=np.asarray(x,float).ravel()
    x=x[np.isfinite(x)&(x>0)]
    if len(x)==0: return np.array([]),np.array([])
    x=np.sort(x)[::-1]
    return np.arange(1,len(x)+1),x


def continuous_powerlaw_tail_fit(values, min_tail=MIN_TAIL_POINTS):
    """Continuous-tail power-law diagnostic; xmin chosen by KS over upper quantiles."""
    x=np.asarray(values,float).ravel()
    x=x[np.isfinite(x)&(x>0)]
    if len(x)<max(min_tail,10):
        return {"alpha":np.nan,"xmin":np.nan,"ks":np.nan,"n_tail":0,"tail_fraction":np.nan}
    best=None
    for xmin in np.unique(np.quantile(x,TAIL_XMIN_QUANTILES)):
        tail=x[x>=xmin]; n=len(tail)
        if n<min_tail: continue
        den=np.sum(np.log(tail/xmin))
        if den<=EPS: continue
        alpha=1.0+n/den
        st=np.sort(tail); emp=np.arange(1,n+1)/n
        model=1.0-np.power(st/xmin,1.0-alpha)
        ks=float(np.max(np.abs(emp-model)))
        cand={"alpha":float(alpha),"xmin":float(xmin),"ks":ks,
              "n_tail":int(n),"tail_fraction":float(n/len(x))}
        if best is None or ks<best["ks"]: best=cand
    return best if best is not None else {
        "alpha":np.nan,"xmin":np.nan,"ks":np.nan,"n_tail":0,"tail_fraction":np.nan
    }


def _loglik_powerlaw(tail,xmin):
    n=len(tail); den=np.sum(np.log(tail/xmin))
    if n<2 or den<=EPS: return np.nan,{"alpha":np.nan}
    alpha=1.0+n/den
    ll=n*math.log(max(alpha-1.0,EPS))-n*math.log(xmin)-alpha*np.sum(np.log(tail/xmin))
    return float(ll),{"alpha":float(alpha)}


def _loglik_exponential(tail,xmin):
    y=tail-xmin; m=float(np.mean(y))
    if m<=EPS: return np.nan,{"lambda":np.nan}
    lam=1.0/m; n=len(tail)
    return float(n*math.log(lam)-lam*np.sum(y)),{"lambda":float(lam)}


def _loglik_lognormal_truncated(tail,xmin):
    z=np.log(tail); zmin=math.log(xmin)
    mu0=float(np.mean(z)); sig0=max(float(np.std(z)),0.2)
    def nll(theta):
        mu,logsig=theta; sig=math.exp(logsig)
        a=(zmin-mu)/sig; sf=max(float(norm.sf(a)),1e-300)
        ll=(-np.log(tail)-math.log(sig)-0.5*math.log(2*math.pi)
            -0.5*((z-mu)/sig)**2-math.log(sf))
        return -float(np.sum(ll))
    res=minimize(nll,[mu0,math.log(sig0)],method="L-BFGS-B",
                 bounds=[(mu0-6*sig0-5,mu0+6*sig0+5),(math.log(1e-3),math.log(20.0))])
    if not res.success or not np.isfinite(res.fun):
        return np.nan,{"mu":np.nan,"sigma":np.nan}
    return -float(res.fun),{"mu":float(res.x[0]),"sigma":float(math.exp(res.x[1]))}


def _tpl_normalizer(alpha,lam_scaled):
    try:
        val,_=quad(lambda y: math.exp(-alpha*math.log(y)-lam_scaled*y),
                   1.0,np.inf,epsabs=1e-9,epsrel=1e-8,limit=150)
        return max(float(val),1e-300)
    except Exception:
        return np.nan


def _loglik_truncated_powerlaw(tail,xmin):
    y=tail/xmin; n=len(y)
    _,pars=_loglik_powerlaw(tail,xmin)
    alpha0=pars.get("alpha",2.5)
    if not np.isfinite(alpha0): alpha0=2.5
    def nll(theta):
        alpha,loglam=theta; lam=math.exp(loglam)
        I=_tpl_normalizer(alpha,lam)
        if not np.isfinite(I) or I<=0: return 1e100
        ll=-alpha*np.sum(np.log(y))-lam*np.sum(y)-n*math.log(xmin)-n*math.log(I)
        return -float(ll)
    res=minimize(nll,[max(alpha0,1.05),math.log(0.02)],method="L-BFGS-B",
                 bounds=[(0.2,10.0),(math.log(1e-6),math.log(100.0))],
                 options={"maxiter":120})
    if not res.success or not np.isfinite(res.fun):
        return np.nan,{"alpha":np.nan,"lambda_scaled":np.nan}
    return -float(res.fun),{"alpha":float(res.x[0]),"lambda_scaled":float(math.exp(res.x[1]))}


def compare_tail_models(values,min_tail=None):
    """Compare PL/lognormal/exponential/truncated-PL on the SAME selected tail."""
    x=np.asarray(values,float).ravel()
    x=x[np.isfinite(x)&(x>0)]
    if min_tail is None:
        min_tail=min(MIN_TAIL_POINTS,max(12,len(x)//4))
    fit=continuous_powerlaw_tail_fit(x,min_tail=min_tail)
    xmin=fit["xmin"]
    if not np.isfinite(xmin):
        return {"xmin":np.nan,"n_tail":0,"tail_fraction":np.nan,
                "powerlaw_alpha":np.nan,"powerlaw_ks":np.nan,"best_model":"insufficient",
                "bic_powerlaw":np.nan,"bic_lognormal":np.nan,"bic_exponential":np.nan,
                "bic_truncated_powerlaw":np.nan,"delta_bic_powerlaw_vs_best_alt":np.nan}
    tail=x[x>=xmin]; n=len(tail)
    models={}
    ll,pa=_loglik_powerlaw(tail,xmin); models["powerlaw"]=(ll,1,pa)
    ll,pa=_loglik_lognormal_truncated(tail,xmin); models["lognormal"]=(ll,2,pa)
    ll,pa=_loglik_exponential(tail,xmin); models["exponential"]=(ll,1,pa)
    ll,pa=_loglik_truncated_powerlaw(tail,xmin); models["truncated_powerlaw"]=(ll,2,pa)
    bics={}
    for name,(ll,k,_) in models.items():
        bics[name]=float(k*math.log(max(n,2))-2*ll) if np.isfinite(ll) else np.nan
    finite={k:v for k,v in bics.items() if np.isfinite(v)}
    best=min(finite,key=finite.get) if finite else "insufficient"
    alts=[bics[k] for k in ["lognormal","exponential","truncated_powerlaw"] if np.isfinite(bics[k])]
    delta=(bics["powerlaw"]-min(alts)) if np.isfinite(bics.get("powerlaw",np.nan)) and alts else np.nan
    return {"xmin":float(xmin),"n_tail":int(n),"tail_fraction":float(n/max(len(x),1)),
            "powerlaw_alpha":float(fit["alpha"]),"powerlaw_ks":float(fit["ks"]),
            "best_model":best,
            "bic_powerlaw":bics.get("powerlaw",np.nan),
            "bic_lognormal":bics.get("lognormal",np.nan),
            "bic_exponential":bics.get("exponential",np.nan),
            "bic_truncated_powerlaw":bics.get("truncated_powerlaw",np.nan),
            "delta_bic_powerlaw_vs_best_alt":float(delta) if np.isfinite(delta) else np.nan}


def global_transition_fit(radii_c,NB_c,N):
    mask=(np.isfinite(radii_c)&np.isfinite(NB_c)&(np.asarray(radii_c)>0)&
          (np.asarray(NB_c)>1)&(np.asarray(NB_c)<0.95*N))
    x=np.log(np.asarray(radii_c,float)[mask]); y=np.log(np.asarray(NB_c,float)[mask])
    if len(x)<4:
        return {"transition_effective_slope":np.nan,"transition_fit_r2":np.nan,
                "transition_delta_bic":np.nan}
    b1,a1=np.polyfit(x,y,1); y1=a1+b1*x
    rss1=max(float(np.sum((y-y1)**2)),EPS)
    bic1=len(x)*math.log(rss1/len(x))+2*math.log(len(x))
    q2,q1,q0=np.polyfit(x,y,2); y2=q0+q1*x+q2*x*x
    rss2=max(float(np.sum((y-y2)**2)),EPS)
    bic2=len(x)*math.log(rss2/len(x))+3*math.log(len(x))
    return {"transition_effective_slope":-float(b1),"transition_fit_r2":r2_score(y,y1),
            "transition_delta_bic":float(bic1-bic2)}


def safe_spearman(x,y):
    """Spearman only when there are enough non-degenerate paired observations."""
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if np.sum(m)<MIN_SPEARMAN_POINTS:
        return np.nan
    xv=x[m]; yv=y[m]
    if np.nanstd(xv)<=EPS or np.nanstd(yv)<=EPS:
        return np.nan
    return float(spearmanr(xv,yv).statistic)


def mean_pairwise_ks(arrays):
    good=[]
    for a in arrays:
        v=np.asarray(a,float); v=v[np.isfinite(v)&(v>0)]
        if len(v)>=5: good.append(v)
    if len(good)<2: return np.nan
    vals=[]
    for i in range(len(good)):
        for j in range(i+1,len(good)):
            vals.append(float(ks_2samp(good[i],good[j]).statistic))
    return float(np.mean(vals)) if vals else np.nan


# ======================================================================================
# 7. PATH-FLOW GRAPH + BCANw GEOMETRY
# ======================================================================================


@dataclass
class LayeredGraph:
    n_nodes: int
    widths: List[int]
    offsets: List[int]
    edge_mag: Dict[Tuple[int,int], float]
    layer_of_edge: Dict[Tuple[int,int], int]


def graph_from_matrices(mats: List[np.ndarray]) -> LayeredGraph:
    widths = [mats[0].shape[1]] + [M.shape[0] for M in mats]
    offsets = [0]
    for w in widths[:-1]:
        offsets.append(offsets[-1]+w)
    edge_mag = {}
    layer_of_edge = {}
    for l, M in enumerate(mats):
        for j in range(M.shape[0]):
            base_v = offsets[l+1] + j
            for i in range(M.shape[1]):
                u = offsets[l] + i
                v = base_v
                e = (u,v) if u < v else (v,u)
                edge_mag[e] = float(max(M[j,i],0.0))
                layer_of_edge[e] = l
    return LayeredGraph(sum(widths), widths, offsets, edge_mag, layer_of_edge)


def metric_lengths(P: LayeredGraph, power: float = METRIC_POWER):
    return {e: float((max(w,0.0)+EPS)**(-power)) for e,w in P.edge_mag.items()}


def distance_matrix(P: LayeredGraph, lengths):
    rows, cols, vals = [], [], []
    for (u,v), L in lengths.items():
        rows.extend([u,v]); cols.extend([v,u]); vals.extend([L,L])
    A = csr_matrix((vals,(rows,cols)),shape=(P.n_nodes,P.n_nodes),dtype=float)
    return np.asarray(csgraph_dijkstra(A,directed=False,unweighted=False),float)


def bcanw_scales(D: np.ndarray, n_scales: int):
    vals = D[np.triu_indices(D.shape[0],k=1)]
    vals = vals[np.isfinite(vals)&(vals>0)]
    if not len(vals):
        return np.array([])
    lo=max(float(vals.min())*.95,EPS)
    hi=float(vals.max())*1.005
    return np.geomspace(lo,hi,n_scales)


def node_strength(P: LayeredGraph):
    s=np.zeros(P.n_nodes,float)
    for (u,v),w in P.edge_mag.items():
        s[u]+=w; s[v]+=w
    return s


def greedy_bcanw_coloring(D: np.ndarray, ell: float, order: np.ndarray):
    n=D.shape[0]
    color=np.full(n,-1,int)
    for u in order:
        conflicts = D[u] >= ell
        used = set(color[conflicts & (color>=0)].tolist())
        c=0
        while c in used:
            c+=1
        color[u]=c
    return color


def global_node_measure(node_flow: List[np.ndarray]):
    x=np.concatenate([np.asarray(v,float) for v in node_flow])
    x=np.clip(x,0,None)
    x/=max(float(x.sum()),EPS)
    return x


def box_mass_lacunarity(colors, node_measure):
    nb=int(colors.max())+1
    mass=np.bincount(colors,weights=node_measure,minlength=nb).astype(float)
    mean=max(float(mass.mean()),EPS)
    lac=float(np.mean(mass*mass)/(mean*mean))
    return mass,lac


def compress_staircase(radii, NB, lac=None):
    radii=np.asarray(radii,float); NB=np.asarray(NB,float)
    out_r=[]; out_n=[]; out_l=[]
    i=0
    while i<len(NB):
        j=i+1
        while j<len(NB) and NB[j]==NB[i]:
            j+=1
        rr=np.sqrt(radii[i]*radii[j-1])
        out_r.append(rr); out_n.append(NB[i])
        if lac is not None:
            out_l.append(float(np.nanmedian(np.asarray(lac)[i:j])))
        i=j
    return np.asarray(out_r),np.asarray(out_n),np.asarray(out_l) if lac is not None else None


def covering_scale(radii, NB, N, target_fraction):
    x=np.asarray(radii,float); y=np.asarray(NB,float)/N
    good=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    x=x[good]; y=y[good]
    if len(x)<2 or target_fraction>y.max() or target_fraction<y.min():
        return np.nan
    # y decreases as x grows; interpolate log x against reversed y.
    order=np.argsort(y)
    ys=y[order]; xs=np.log(x[order])
    return float(np.exp(np.interp(target_fraction,ys,xs)))


def r2_score(y,yhat):
    y=np.asarray(y,float); yhat=np.asarray(yhat,float)
    sst=float(np.sum((y-y.mean())**2)); ssr=float(np.sum((y-yhat)**2))
    return float(1-ssr/sst) if sst>0 else np.nan


def local_effective_dimension(radii,NB,window=LOCAL_SLOPE_WINDOW):
    x=np.log(np.asarray(radii,float)); y=np.log(np.asarray(NB,float))
    d=np.full(len(x),np.nan); rr=np.full(len(x),np.nan)
    half=max(window//2,1)
    for i in range(len(x)):
        lo=max(0,i-half); hi=min(len(x),i+half+1)
        if hi-lo<4: continue
        b,a=np.polyfit(x[lo:hi],y[lo:hi],1)
        pred=a+b*x[lo:hi]
        d[i]=-b; rr[i]=r2_score(y[lo:hi],pred)
    return d,rr


def best_scaling_window(radii,NB,N):
    r=np.asarray(radii,float); n=np.asarray(NB,float)
    good=np.isfinite(r)&np.isfinite(n)&(r>0)&(n>1)&(n<.95*N)
    ids=np.where(good)[0]
    empty={"dB":np.nan,"r2":np.nan,"span":0.0,"nb_ratio":np.nan,"slope_cv":np.nan,
           "delta_bic":np.nan,"credible":False,"i0":-1,"i1":-1}
    if len(ids)<SCALING_MIN_POINTS: return empty
    best=None
    for a in range(len(ids)):
        for b in range(a+SCALING_MIN_POINTS,len(ids)+1):
            sel=ids[a:b]
            if np.any(np.diff(sel)!=1): continue
            span=float(np.log10(r[sel[-1]]/r[sel[0]]))
            ratio=float(n[sel[0]]/n[sel[-1]])
            if span<SCALING_MIN_SPAN_DECADES or ratio<SCALING_MIN_NB_RATIO: continue
            x=np.log(r[sel]); y=np.log(n[sel])
            slope,intercept=np.polyfit(x,y,1)
            dim=-float(slope)
            if dim<=SCALING_MIN_DIM: continue
            pred=intercept+slope*x
            rr=r2_score(y,pred)
            if not np.isfinite(rr) or rr<SCALING_R2_MIN: continue
            ds=[]
            w=min(5,len(x))
            for ii in range(len(x)-w+1):
                bb,aa=np.polyfit(x[ii:ii+w],y[ii:ii+w],1)
                ds.append(-bb)
            ds=np.asarray(ds,float)
            scv=float(np.std(ds)/max(abs(np.mean(ds)),EPS)) if len(ds)>=2 else np.nan
            rss1=max(float(np.sum((y-pred)**2)),EPS)
            bic1=len(x)*math.log(rss1/len(x))+2*math.log(len(x))
            q2,q1,q0=np.polyfit(x,y,2)
            pred2=q0+q1*x+q2*x*x
            rss2=max(float(np.sum((y-pred2)**2)),EPS)
            bic2=len(x)*math.log(rss2/len(x))+3*math.log(len(x))
            dbic=float(bic1-bic2)
            credible=(np.isfinite(scv) and scv<=SCALING_MAX_SLOPE_CV and dbic<=SCALING_MAX_DELTA_BIC)
            cand=(span,ratio,rr,dim,scv,dbic,credible,int(sel[0]),int(sel[-1]))
            if credible and (best is None or (span,ratio,rr)>(best[0],best[1],best[2])):
                best=cand
    if best is None: return empty
    span,ratio,rr,dim,scv,dbic,credible,i0,i1=best
    return {"dB":dim,"r2":rr,"span":span,"nb_ratio":ratio,"slope_cv":scv,
            "delta_bic":dbic,"credible":credible,"i0":i0,"i1":i1}


def crossover_peaks(x, d, r2_local):
    x=np.asarray(x,float); d=np.asarray(d,float); rr=np.asarray(r2_local,float)
    good=np.isfinite(x)&np.isfinite(d)&np.isfinite(rr)&(x>0)&(d>0)&(rr>=LOCAL_SLOPE_R2_MIN)
    x=x[good]; d=d[good]
    if len(x)<5:
        return [{"x":np.nan,"d":np.nan,"prom":np.nan,"width":np.nan} for _ in range(2)]
    # modest 3-point median smoothing for peak topology only; raw d is retained in outputs.
    ds=pd.Series(d).rolling(3,center=True,min_periods=1).median().to_numpy()
    prom=max(0.05,0.08*(np.nanmax(ds)-np.nanmin(ds)))
    peaks,props=find_peaks(ds,prominence=prom,distance=2)
    if not len(peaks):
        peaks=np.array([int(np.nanargmax(ds))])
        props={"prominences":np.array([0.0])}
    widths=peak_widths(ds,peaks,rel_height=.5)[0] if len(peaks) else np.array([])
    candidates=[]
    for j,p in enumerate(peaks):
        candidates.append({"x":float(x[p]),"d":float(ds[p]),
                           "prom":float(props["prominences"][j]),"width":float(widths[j])})
    # Keep two most prominent, then order them from fine -> coarse scale.
    candidates=sorted(candidates,key=lambda z:z["prom"],reverse=True)[:2]
    candidates=sorted(candidates,key=lambda z:z["x"])
    while len(candidates)<2:
        candidates.append({"x":np.nan,"d":np.nan,"prom":np.nan,"width":np.nan})
    return candidates


def analyze_core_geometry(weights: List[np.ndarray], n_scales: int, save_curve: Optional[Path]=None,
                          metric_power: float = METRIC_POWER):
    t0=time.perf_counter()
    A=path_factor_matrices(weights)
    widths=[A[0].shape[1]]+[M.shape[0] for M in A]
    pstats=path_partition_stats(A,widths)
    edge_flow,node_flow=path_flow(A)
    edge_enr,node_enr=flow_enrichment(edge_flow,node_flow)
    ecv,etop,ncv,ntop=flow_concentration(edge_enr,node_enr)

    Pgeom=graph_from_matrices(edge_enr)
    Praw=graph_from_matrices(edge_flow)
    lengths=metric_lengths(Pgeom,power=metric_power)
    D=distance_matrix(Pgeom,lengths)
    radii=bcanw_scales(D,n_scales)
    order=np.argsort(-node_strength(Pgeom))
    node_mu=global_node_measure(node_flow)

    NB=[]; lac=[]; colors_by_scale=[]
    for ell in radii:
        colors=greedy_bcanw_coloring(D,float(ell),order)
        colors_by_scale.append(colors)
        NB.append(int(colors.max())+1)
        _,ll=box_mass_lacunarity(colors,node_mu)
        lac.append(ll)
    NB=np.asarray(NB,float); lac=np.asarray(lac,float)

    rc,nc,lc=compress_staircase(radii,NB,lac)
    xi75=covering_scale(rc,nc,Pgeom.n_nodes,.75)
    xi50=covering_scale(rc,nc,Pgeom.n_nodes,.50)
    xi25=covering_scale(rc,nc,Pgeom.n_nodes,.25)
    xi10=covering_scale(rc,nc,Pgeom.n_nodes,.10)
    x=rc/max(xi50,EPS) if np.isfinite(xi50) else rc/np.nanmedian(rc)
    dloc,r2loc=local_effective_dimension(rc,nc)
    sw=best_scaling_window(rc,nc,Pgeom.n_nodes)
    peaks=crossover_peaks(x,dloc,r2loc)
    transition=global_transition_fit(rc,nc,Pgeom.n_nodes)

    # Dominant accepted d_eff response, kept separately from the two-peak topology.
    good_peak=np.isfinite(x)&np.isfinite(dloc)&np.isfinite(r2loc)&(r2loc>=LOCAL_SLOPE_R2_MIN)&(dloc>0)
    if np.any(good_peak):
        ids=np.where(good_peak)[0]
        ip=int(ids[np.argmax(dloc[ids])])
        d_star=float(dloc[ip]); x_d_star=float(x[ip])
        nb_fraction_at_d_star=float(nc[ip]/Pgeom.n_nodes) if ip<len(nc) else np.nan
    else:
        d_star=x_d_star=nb_fraction_at_d_star=np.nan

    # Pooled microscopic tail diagnostic at every structural checkpoint.
    pooled_edge=np.concatenate([np.asarray(v,float).ravel() for v in edge_enr])
    pooled_tail=continuous_powerlaw_tail_fit(
        pooled_edge,min_tail=min(MIN_TAIL_POINTS,max(12,len(pooled_edge)//100))
    )

    if len(lc):
        valid_lac=np.isfinite(lc)&np.isfinite(x)&(nc>1)
        if valid_lac.any():
            il=np.where(valid_lac)[0][np.argmax(lc[valid_lac])]
            x_lac=float(x[il]); lac_star=float(lc[il])
        else:
            x_lac=lac_star=np.nan
    else:
        x_lac=lac_star=np.nan

    # Directed quotient-flow summary at canonical BCANw resolutions.
    q_arrays=[]; q_retentions=[]; q_eta={q:[] for q in RENYI_Q}
    frac_raw=NB/max(Pgeom.n_nodes,1)
    selected_q=[]
    for target in RENORM_TARGET_BOX_FRACTIONS:
        eligible=np.where((NB>=3)&(NB<=.90*Pgeom.n_nodes))[0]
        if not len(eligible): continue
        idx=int(eligible[np.argmin(np.abs(frac_raw[eligible]-target))])
        if idx in selected_q: continue
        selected_q.append(idx)
        vals_q,p_q,norm_q,ret_q=directed_quotient(edge_flow,colors_by_scale[idx],widths)
        if len(norm_q): q_arrays.append(norm_q)
        q_retentions.append(ret_q)
        for q in RENYI_Q: q_eta[q].append(renyi_effective_fraction(p_q,q))
    quotient_ks=mean_pairwise_ks(q_arrays)
    quotient_ret=float(np.nanmean(q_retentions)) if q_retentions else np.nan

    result={
        **pstats,
        "metric_power":float(metric_power),
        "edge_cv":ecv,"edge_top10_share":etop,"node_cv":ncv,"node_top10_share":ntop,
        "xi75":xi75,"xi50":xi50,"xi25":xi25,"xi10":xi10,
        "xi25_over_xi50":xi25/xi50 if np.isfinite(xi25) and np.isfinite(xi50) else np.nan,
        "xi10_over_xi50":xi10/xi50 if np.isfinite(xi10) and np.isfinite(xi50) else np.nan,
        "scaling_dB":sw["dB"],"scaling_r2":sw["r2"],"scaling_span":sw["span"],
        "scaling_nb_ratio":sw["nb_ratio"],"scaling_slope_cv":sw["slope_cv"],
        "scaling_delta_bic":sw["delta_bic"],"scaling_credible":sw["credible"],
        **transition,
        "x_d_star":x_d_star,"d_star":d_star,"NB_fraction_at_d_star":nb_fraction_at_d_star,
        "pooled_edge_tail_alpha":pooled_tail["alpha"],"pooled_edge_tail_xmin":pooled_tail["xmin"],
        "pooled_edge_tail_ks":pooled_tail["ks"],"pooled_edge_tail_fraction":pooled_tail["tail_fraction"],
        "quotient_flow_mean_pairwise_KS":quotient_ks,
        "renorm_flow_retention_mean_snapshot":quotient_ret,
        **{f"coarse_eta_q{q:g}_mean":(float(np.nanmean(q_eta[q])) if len(q_eta[q]) else np.nan) for q in RENYI_Q},
        "scaling_credible_num":float(bool(sw["credible"])),
        "peak1_x":peaks[0]["x"],"peak1_d":peaks[0]["d"],"peak1_prom":peaks[0]["prom"],"peak1_width":peaks[0]["width"],
        "peak2_x":peaks[1]["x"],"peak2_d":peaks[1]["d"],"peak2_prom":peaks[1]["prom"],"peak2_width":peaks[1]["width"],
        "n_crossover_peaks":int(sum(np.isfinite(z["x"]) for z in peaks)),
        "x_lacunarity_star":x_lac,"lacunarity_star":lac_star,
        "geometry_seconds":time.perf_counter()-t0,
        "n_graph_nodes":Pgeom.n_nodes,
    }

    if save_curve is not None:
        np.savez_compressed(save_curve,
                            radii=radii,NB=NB,lacunarity=lac,
                            radii_compressed=rc,NB_compressed=nc,lacunarity_compressed=lc,
                            ell_over_xi50=x,d_eff=dloc,d_eff_r2=r2loc)

    # Return full objects only to immediate caller; never serialize D for map-wide archive.
    aux={"D":D,"node_mu":node_mu,"edge_flow":edge_flow,"node_flow":node_flow,
         "edge_enrichment":edge_enr,"node_enrichment":node_enr,
         "Praw":Praw,"radii":radii,"NB":NB,"colors_by_scale":colors_by_scale,
         "xi50":xi50,"radii_compressed":rc,"NB_compressed":nc,
         "lacunarity_compressed":lc,"ell_over_xi50":x,
         "d_eff":dloc,"d_eff_r2":r2loc}
    return result,aux

# ======================================================================================
# 8. FULL FINAL-SNAPSHOT DIAGNOSTICS: LOCAL DEPTH + DIRECTED RENORMALIZATION
# ======================================================================================


def _compress_common_local_states(rgrid, support, mass):
    r=np.asarray(rgrid,float); s=np.asarray(support,float); m=np.asarray(mass,float)
    good=np.isfinite(r)&np.isfinite(s)&np.isfinite(m)&(r>0)&(s>0)&(m>0)
    r,s,m=r[good],s[good],m[good]
    if len(r)==0:
        return r,s,m
    keep=[0]
    for k in range(1,len(r)):
        # A new radius contributes only when either geometric support or flow mass actually changes.
        if not (np.isclose(s[k],s[keep[-1]],rtol=1e-10,atol=1e-14) and
                np.isclose(m[k],m[keep[-1]],rtol=1e-10,atol=1e-14)):
            keep.append(k)
    return r[keep],s[keep],m[keep]


def fit_common_local_growth(rgrid, support, mass,
                            min_points=LOCAL_COMMON_MIN_POINTS,
                            min_span=LOCAL_COMMON_MIN_SPAN_DECADES,
                            support_growth=LOCAL_COMMON_SUPPORT_GROWTH,
                            mass_growth=LOCAL_COMMON_MASS_GROWTH,
                            r2_min=LOCAL_COMMON_R2_MIN):
    """
    Fit d_i^F and alpha_i^F on ONE common radial interval.
    The widest/highest-quality contiguous valid window is selected; if no shared scaling
    window exists, both exponents remain NaN.  Delta_i^F is therefore always apples-to-apples.
    """
    r,s,m=_compress_common_local_states(rgrid,support,mass)
    valid=(s<LOCAL_COMMON_SATURATION)&(m<LOCAL_COMMON_SATURATION)
    r,s,m=r[valid],s[valid],m[valid]
    if len(r)<min_points:
        return (np.nan,)*8

    best=None
    for i in range(0,len(r)-min_points+1):
        for j in range(i+min_points,len(r)+1):
            rr=r[i:j]; ss=s[i:j]; mm=m[i:j]
            span=np.log10(rr[-1]/rr[0])
            if span<min_span: continue
            if ss[-1]/max(ss[0],EPS)<support_growth: continue
            if mm[-1]/max(mm[0],EPS)<mass_growth: continue
            x=np.log(rr)
            bd,ad=np.polyfit(x,np.log(ss),1); ba,aa=np.polyfit(x,np.log(mm),1)
            if bd<=0 or ba<=0: continue
            pd=ad+bd*x; pa=aa+ba*x
            r2d=r2_score(np.log(ss),pd); r2a=r2_score(np.log(mm),pa)
            if (not np.isfinite(r2d)) or (not np.isfinite(r2a)) or r2d<r2_min or r2a<r2_min:
                continue
            # Prefer radial span first, then number of distinct states, then the weaker R2.
            score=(span, len(rr), min(r2d,r2a))
            if best is None or score>best[0]:
                best=(score,float(bd),float(ba),float(r2d),float(r2a),float(rr[0]),float(rr[-1]),int(len(rr)))
    if best is None:
        return (np.nan,)*8
    _,d,alpha,r2d,r2a,rmin,rmax,npts=best
    return d,alpha,r2d,r2a,rmin,rmax,npts,float(alpha-d)


def fit_local_growth(rgrid,vals,min_points=7,min_span=.50,min_growth=3.0):
    """Legacy single-field helper retained for unrelated diagnostics; Delta never uses it."""
    r=np.asarray(rgrid,float); v=np.asarray(vals,float)
    m=np.isfinite(r)&np.isfinite(v)&(r>0)&(v>0)&(v<LOCAL_COMMON_SATURATION)
    if m.sum()<min_points: return np.nan,np.nan
    rr=r[m]; vv=v[m]
    if np.log10(rr.max()/rr.min())<min_span or vv.max()/max(vv.min(),EPS)<min_growth:
        return np.nan,np.nan
    x=np.log(rr); y=np.log(vv)
    b,a=np.polyfit(x,y,1); pred=a+b*x
    rr2=r2_score(y,pred)
    if not np.isfinite(rr2) or rr2<LOCAL_COMMON_R2_MIN or b<=0: return np.nan,rr2
    return float(b),float(rr2)


def local_depth_profile(D,node_mu,widths):
    n=D.shape[0]
    dloc=np.full(n,np.nan); alpha=np.full(n,np.nan)
    r2d=np.full(n,np.nan); r2a=np.full(n,np.nan)
    for i in range(n):
        dv=D[i][np.isfinite(D[i])&(D[i]>0)]
        if len(dv)<10: continue
        rlo=max(float(np.quantile(dv,.03)),EPS)
        rhi=float(np.quantile(dv,.80))
        if rhi<=rlo: continue
        rgrid=np.geomspace(rlo,rhi,24)
        support=[]; mass=[]
        for r in rgrid:
            inside=D[i]<=r
            support.append(np.sum(inside)/n)
            mass.append(float(node_mu[inside].sum()))
        d,aa,rd,ra,_,_,_,_=fit_common_local_growth(rgrid,support,mass)
        dloc[i],alpha[i],r2d[i],r2a[i]=d,aa,rd,ra
    delta=alpha-dloc

    offsets=np.cumsum([0]+list(widths))
    layer_rows=[]
    for l in range(len(widths)):
        sl=slice(offsets[l],offsets[l+1])
        vd=np.isfinite(dloc[sl]); va=np.isfinite(alpha[sl]); vp=np.isfinite(delta[sl])
        paired_fraction=float(np.mean(vp))
        layer_rows.append({
            "layer":l,"z":l/max(len(widths)-1,1),"n_nodes":widths[l],
            "delta_median":float(np.nanmedian(delta[sl])) if paired_fraction>=LOCAL_LAYER_MIN_VALID_FRACTION else np.nan,
            "support_median":float(np.nanmedian(dloc[sl])) if np.mean(vd)>=LOCAL_LAYER_MIN_VALID_FRACTION else np.nan,
            "alpha_median":float(np.nanmedian(alpha[sl])) if np.mean(va)>=LOCAL_LAYER_MIN_VALID_FRACTION else np.nan,
            "valid_support_fraction":float(np.mean(vd)),
            "valid_alpha_fraction":float(np.mean(va)),
            "valid_delta_fraction":paired_fraction,
        })
    ldf=pd.DataFrame(layer_rows)
    good=np.isfinite(ldf.delta_median)
    interp=np.full(len(LOCAL_DEPTH_GRID),np.nan)
    if good.sum()>=2:
        interp=np.interp(LOCAL_DEPTH_GRID,ldf.loc[good,"z"],ldf.loc[good,"delta_median"])
    return ldf,interp


def renyi_effective_fraction(p,q):
    p=np.asarray(p,float); p=p[np.isfinite(p)&(p>0)]
    if not len(p): return np.nan
    p=p/p.sum()
    if abs(q-1.0)<1e-12:
        H=-float(np.sum(p*np.log(p)))
    else:
        H=float(np.log(np.sum(p**q))/(1-q))
    return float(np.exp(H)/len(p))


def directed_quotient(edge_flow,colors,widths):
    nb=int(colors.max())+1
    W=np.zeros((nb,nb),float)
    offsets=np.cumsum([0]+list(widths))
    total=0.0; inter=0.0
    for l,E in enumerate(edge_flow):
        for j in range(E.shape[0]):
            v=offsets[l+1]+j
            for i in range(E.shape[1]):
                u=offsets[l]+i
                f=float(E[j,i]); total+=f
                a=int(colors[u]); b=int(colors[v])
                if a!=b:
                    W[a,b]+=f; inter+=f
    vals=W[(W>0)&(~np.eye(nb,dtype=bool))]
    p=vals/max(vals.sum(),EPS) if len(vals) else np.array([])
    norm=vals/max(vals.mean(),EPS) if len(vals) else np.array([])
    return vals,p,norm,float(inter/max(total,EPS))


def full_final_diagnostics(core_aux,widths,arch_id):
    D=core_aux["D"]; node_mu=core_aux["node_mu"]; edge_flow=core_aux["edge_flow"]
    ldf,delta_grid=local_depth_profile(D,node_mu,widths)
    ldf.insert(0,"arch_id",arch_id)

    radii=core_aux["radii"]; NB=core_aux["NB"]; colors_by_scale=core_aux["colors_by_scale"]
    N=D.shape[0]; frac=NB/N
    renorm_rows=[]; arrays=[]
    selected=[]
    for target in RENORM_TARGET_BOX_FRACTIONS:
        eligible=np.where((NB>=3)&(NB<=.90*N))[0]
        if not len(eligible): continue
        idx=int(eligible[np.argmin(np.abs(frac[eligible]-target))])
        if idx in selected: continue
        selected.append(idx)
        vals,p,norm,ret=directed_quotient(edge_flow,colors_by_scale[idx],widths)
        row={"arch_id":arch_id,"target_box_fraction":target,"N_B":NB[idx],"N_B_over_N":frac[idx],
             "ell_over_xi50":radii[idx]/max(core_aux["xi50"],EPS),"n_directed_edges":len(vals),
             "inter_box_flow_retention":ret}
        for q in RENYI_Q:
            row[f"eta_q{q:g}"]=renyi_effective_fraction(p,q)
        renorm_rows.append(row)
        arrays.append(norm)
    ks=[]
    for i in range(len(arrays)):
        for j in range(i+1,len(arrays)):
            if len(arrays[i])>=5 and len(arrays[j])>=5:
                ks.append(ks_2samp(arrays[i],arrays[j]).statistic)
    return ldf,delta_grid,pd.DataFrame(renorm_rows),float(np.mean(ks)) if ks else np.nan

# ======================================================================================
# 9. STRUCTURAL FINGERPRINT SPACE
# ======================================================================================

CORE_FEATURES = [
    "log_eta1","log_eta2","edge_cv",
    "xi25_over_xi50","xi10_over_xi50",
    "scaling_dB","scaling_span","scaling_credible_num","n_crossover_peaks",
    "peak1_x","peak1_d","peak1_prom",
    "peak2_x","peak2_d","peak2_prom",
    "x_lacunarity_star","lacunarity_star",
]


class FingerprintSpace:
    def __init__(self):
        self.imputer=SimpleImputer(strategy="median",keep_empty_features=True)
        self.scaler=StandardScaler()
        self.X0=None

    def fit(self,df):
        X=self.imputer.fit_transform(df[CORE_FEATURES])
        X=self.scaler.fit_transform(X)
        self.X0=X
        return X

    def transform_row(self,row: Dict[str,Any]):
        frame=pd.DataFrame([{k:row.get(k,np.nan) for k in CORE_FEATURES}])
        return self.scaler.transform(self.imputer.transform(frame))[0]

    def transform_df(self,df):
        return self.scaler.transform(self.imputer.transform(df[CORE_FEATURES]))

    def novelty(self,z,k=NOVELTY_K,exclude_index=None):
        X=self.X0
        dist=np.linalg.norm(X-z[None,:],axis=1)
        if exclude_index is not None and 0<=exclude_index<len(dist):
            dist[exclude_index]=np.inf
        good=np.sort(dist[np.isfinite(dist)])
        if not len(good): return 0.0
        return float(np.mean(good[:min(k,len(good))]))

# ======================================================================================
# 10. ARCHITECTURE EVALUATION AT INITIALIZATION
# ======================================================================================


def evaluate_initial_architecture(config: Dict[str,Any], stage: str, index: int):
    arch=architecture_from_config(config)
    widths=arch["hidden_widths"]
    arch_id=arch_id_from_widths(widths)
    curve_path=OUTDIR/"curves"/f"{arch_id}_init.npz"
    t0=time.perf_counter()
    model=BiasFreeReLUMLP(INPUT_DIM,widths,1,seed=MODEL_INIT_SEED)
    weights=extract_weights(model)
    geom,_=analyze_core_geometry(weights,MAP_BCANW_SCALES,save_curve=curve_path)
    elapsed=time.perf_counter()-t0
    row={
        "arch_id":arch_id,"stage":stage,"map_index":index,
        "config_key":config_key_dict(config),"config_json":json.dumps(config,sort_keys=True),
        **{k:v for k,v in arch.items() if k!="hidden_widths"},
        "hidden_widths":json.dumps(widths),
        **geom,
        "total_eval_seconds":elapsed,
    }
    tqdm.write(
        f"[{stage} #{index:03d}] {arch_id} | morph={arch['morphology']:<10s} depth={arch['depth']:2d} "
        f"params={arch['actual_params']:,} nodes={arch['n_nodes']:,} | "
        f"dB={geom['scaling_dB']:.3f} peak1=({geom['peak1_x']:.2f},{geom['peak1_d']:.2f}) "
        f"eta1={geom['eta1']:.2e} | {human_time(elapsed)}"
    )
    del model,weights
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row

# ======================================================================================
# 11. CONFIGSPACE + SMAC / FALLBACK BAYESIAN EXPLORER
# ======================================================================================


def build_configspace():
    if not SMAC_AVAILABLE:
        return None
    from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, InCondition
    cs=ConfigurationSpace(seed=SEED)
    depth=Integer("depth",(DEPTH_MIN,DEPTH_MAX),default=max(DEPTH_MIN,min(12,DEPTH_MAX)))
    target=Integer("target_params",(PARAM_MIN,PARAM_MAX),log=True,default=min(50_000,PARAM_MAX))
    morph=Categorical("morphology",MORPHOLOGIES,default="constant")
    shape=Float("shape",(SHAPE_MIN,SHAPE_MAX),default=0.70)
    curve=Float("curvature",(CURVE_MIN,CURVE_MAX),default=1.0)
    cs.add([depth,target,morph,shape,curve])
    active=[m for m in MORPHOLOGIES if m!="constant"]
    cs.add(InCondition(shape,morph,active))
    cs.add(InCondition(curve,morph,active))
    return cs


def config_obj_to_dict(config):
    if isinstance(config,dict): return {k:(v.item() if hasattr(v,"item") else v) for k,v in config.items()}
    return {k:(v.item() if hasattr(v,"item") else v) for k,v in dict(config).items()}


def _sample_config_in_cell(morphology: str, db: int, pb: int, rng: np.random.Generator):
    # Depth strata are uniform in integer depth; parameter strata are uniform in log-budget.
    d_edges = np.linspace(DEPTH_MIN, DEPTH_MAX + 1, MAP_DEPTH_BINS + 1)
    dlo = int(math.ceil(d_edges[db])); dhi = int(math.floor(d_edges[db + 1] - 1e-9))
    dlo = max(DEPTH_MIN, min(dlo, DEPTH_MAX)); dhi = max(dlo, min(dhi, DEPTH_MAX))
    depth = int(rng.integers(dlo, dhi + 1))

    lp_edges = np.linspace(math.log(PARAM_MIN), math.log(PARAM_MAX), MAP_PARAM_BINS + 1)
    logp = float(rng.uniform(lp_edges[pb], lp_edges[pb + 1]))
    target = int(round(math.exp(logp)))

    cfg = {"depth": depth, "target_params": target, "morphology": morphology}
    if morphology != "constant":
        cfg["shape"] = float(rng.uniform(SHAPE_MIN, SHAPE_MAX))
        cfg["curvature"] = float(rng.uniform(CURVE_MIN, CURVE_MAX))
    return cfg


def sample_feasible_config_in_cell(cell, seed: int, forbidden_signatures=None):
    morphology, db, pb = cell
    forbidden_signatures = forbidden_signatures if forbidden_signatures is not None else set()
    rng = np.random.default_rng(seed)
    last_reason = None
    for _ in range(ARCH_MAX_RESAMPLE_ATTEMPTS):
        cfg = _sample_config_in_cell(morphology, db, pb, rng)
        arch = architecture_from_config(cfg)
        ok, reasons = architecture_feasibility(arch)
        sig = tuple(arch["hidden_widths"])
        if ok and sig not in forbidden_signatures:
            return cfg
        last_reason = reasons
    raise RuntimeError(f"Could not find feasible architecture in cell {cell}; last rejection={last_reason}")


def generate_sobol_configs(n=None, cs=None):
    """
    Dense stratified cold start.  Every morphology-depth-budget cell is represented before BO.
    The argument n is accepted for backward compatibility; research mode uses the exact grid size.
    """
    cells = all_coverage_cells()
    configs = []
    signatures = set()
    base_seed = SEED * 100003
    per_cell = MAP_SOBOL_PER_CELL
    for ci, cell in enumerate(cells):
        # Within each stratum we use a scrambled Sobol sequence for shape/curvature and continuous budget.
        sampler = qmc.Sobol(d=4, scramble=True, seed=base_seed + ci)
        m = int(math.ceil(math.log2(max(per_cell * 4, 2))))
        points = sampler.random_base2(m=m)
        accepted = 0
        for pi, u in enumerate(points):
            if accepted >= per_cell:
                break
            morphology, db, pb = cell
            d_edges = np.linspace(DEPTH_MIN, DEPTH_MAX + 1, MAP_DEPTH_BINS + 1)
            dlo = int(math.ceil(d_edges[db])); dhi = int(math.floor(d_edges[db + 1] - 1e-9))
            dlo = max(DEPTH_MIN, min(dlo, DEPTH_MAX)); dhi = max(dlo, min(dhi, DEPTH_MAX))
            depth = int(dlo + min(int(math.floor(float(u[0]) * (dhi - dlo + 1))), dhi - dlo))
            lp_edges = np.linspace(math.log(PARAM_MIN), math.log(PARAM_MAX), MAP_PARAM_BINS + 1)
            logp = lp_edges[pb] + float(u[1]) * (lp_edges[pb + 1] - lp_edges[pb])
            cfg = {"depth": depth, "target_params": int(round(math.exp(logp))), "morphology": morphology}
            if morphology != "constant":
                cfg["shape"] = SHAPE_MIN + float(u[2]) * (SHAPE_MAX - SHAPE_MIN)
                cfg["curvature"] = CURVE_MIN + float(u[3]) * (CURVE_MAX - CURVE_MIN)
            arch = architecture_from_config(cfg)
            ok, _ = architecture_feasibility(arch)
            sig = tuple(arch["hidden_widths"])
            if ok and sig not in signatures:
                configs.append(cfg); signatures.add(sig); accepted += 1
        while accepted < per_cell:
            cfg = sample_feasible_config_in_cell(cell, base_seed + ci * 1000 + accepted + 77, signatures)
            arch = architecture_from_config(cfg); sig = tuple(arch["hidden_widths"])
            configs.append(cfg); signatures.add(sig); accepted += 1
    return configs


def fallback_encode_config(d):
    morph_idx=MORPHOLOGIES.index(str(d["morphology"]))
    return np.array([
        (int(d["depth"])-DEPTH_MIN)/max(DEPTH_MAX-DEPTH_MIN,1),
        (math.log(int(d["target_params"]))-math.log(PARAM_MIN))/max(math.log(PARAM_MAX)-math.log(PARAM_MIN),EPS),
        morph_idx/max(len(MORPHOLOGIES)-1,1),
        (float(d.get("shape",1.0))-SHAPE_MIN)/(SHAPE_MAX-SHAPE_MIN),
        (float(d.get("curvature",1.0))-CURVE_MIN)/(CURVE_MAX-CURVE_MIN),
    ])


def config_space_distance_to_rows(config, rows):
    if not rows:
        return 1.0
    x = fallback_encode_config(config)
    xs = []
    for r in rows:
        try:
            cfg = json.loads(r["config_json"]) if isinstance(r.get("config_json"), str) else r
            xs.append(fallback_encode_config(cfg))
        except Exception:
            pass
    if not xs:
        return 1.0
    d = np.linalg.norm(np.vstack(xs) - x[None, :], axis=1) / math.sqrt(len(x))
    return float(np.clip(np.min(d), 0.0, 1.0))


def coverage_bonus_for_config(config, rows):
    counts = coverage_counts_from_rows(rows)
    count = counts.get(coverage_cell_from_config(config), 0)
    return float(1.0 / (1.0 + count))


def combined_map_score(structural_novelty, config, prior_rows):
    structural_component = float(1.0 - math.exp(-max(float(structural_novelty), 0.0)))
    config_component = config_space_distance_to_rows(config, prior_rows)
    coverage_component = coverage_bonus_for_config(config, prior_rows)
    score = (BO_STRUCTURAL_WEIGHT * structural_component +
             BO_CONFIG_DISTANCE_WEIGHT * config_component +
             BO_COVERAGE_WEIGHT * coverage_component)
    return float(np.clip(score, 0.0, 1.0)), config_component, coverage_component


def fallback_propose(history_df, n_pool=5000):
    # Diversity-constrained RF Bayesian explorer used only when SMAC is unavailable.
    rows = history_df.to_dict("records")
    if len(history_df) < 12 or np.random.rand() < SMAC_RANDOM_PROBABILITY:
        counts = coverage_counts_from_rows(rows)
        min_count = min(counts.values()) if counts else 0
        cells = [c for c,v in counts.items() if v == min_count]
        return sample_feasible_config_in_cell(random.choice(cells), SEED + len(rows) * 7919)

    X=np.vstack([fallback_encode_config(json.loads(x)) for x in history_df.config_json])
    y=history_df.smac_cost.to_numpy(float)
    rf=RandomForestRegressor(n_estimators=SMAC_RF_TREES,random_state=SEED+len(rows),min_samples_leaf=2,n_jobs=-1)
    rf.fit(X,y)

    counts = coverage_counts_from_rows(rows)
    min_count = min(counts.values()) if counts else 0
    candidate_cfg=[]
    rng=np.random.default_rng(SEED + len(rows)*104729)
    attempts=0
    while len(candidate_cfg)<n_pool and attempts<n_pool*8:
        attempts += 1
        cfg=manual_config_from_unit(rng.random(5))
        arch=architecture_from_config(cfg); ok,_=architecture_feasibility(arch)
        if not ok: continue
        cell=coverage_cell_from_config(cfg)
        if counts.get(cell,0)>min_count+BO_CELL_HARD_SLACK: continue
        candidate_cfg.append(cfg)
    if not candidate_cfg:
        cells=[c for c,v in counts.items() if v==min_count]
        return sample_feasible_config_in_cell(random.choice(cells), SEED+len(rows)*3571)

    XX=np.vstack([fallback_encode_config(c) for c in candidate_cfg])
    preds=np.vstack([t.predict(XX) for t in rf.estimators_])
    mu=preds.mean(0); sd=preds.std(0)
    cfgdist=np.array([config_space_distance_to_rows(c,rows) for c in candidate_cfg])
    cover=np.array([coverage_bonus_for_config(c,rows) for c in candidate_cfg])
    # Lower confidence bound + explicit configuration-space and cell-coverage rewards.
    utility = -mu + 1.35*sd + 0.35*cfgdist + 0.20*cover
    return candidate_cfg[int(np.argmax(utility))]


# ======================================================================================
# 12. FIXED TRAINING ENGINE
# ======================================================================================


def epoch_batches(n:int,epoch:int,batch_size:int=BATCH_SIZE):
    rng=np.random.default_rng(DATA_ORDER_SEED+int(epoch))
    idx=rng.permutation(n)
    for lo in range(0,n,batch_size):
        yield idx[lo:min(lo+batch_size,n)]


@torch.no_grad()
def eval_mse(model,X,y,batch_size=4096):
    model.eval(); total=0.0; n=0
    for lo in range(0,len(X),batch_size):
        hi=min(lo+batch_size,len(X))
        xb=X[lo:hi].to(DEVICE,non_blocking=True)
        yb=y[lo:hi].to(DEVICE,non_blocking=True)
        pred=model(xb)
        total+=nn.functional.mse_loss(pred,yb,reduction="sum").item(); n+=yb.numel()
    return float(total/max(n,1))


def checkpoint_path(arch_id):
    return OUTDIR/"checkpoints"/f"{arch_id}.pt"


def train_to_budget(arch_row: pd.Series, target_epoch: int, splits, epoch_progress=None):
    arch_id=str(arch_row.arch_id)
    widths=json.loads(arch_row.hidden_widths)
    ckpt=checkpoint_path(arch_id)
    model=BiasFreeReLUMLP(INPUT_DIM,widths,1,seed=MODEL_INIT_SEED).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    start_epoch=0
    history=[]
    if ckpt.exists():
        state=torch.load(ckpt,map_location="cpu",weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        # move optimizer tensors to device
        for st in opt.state.values():
            for k,v in st.items():
                if torch.is_tensor(v): st[k]=v.to(DEVICE)
        start_epoch=int(state["epoch"])

    Xtr,ytr=splits["train"]; Xv,yv=splits["val"]
    loss_fn=nn.MSELoss()
    local_bar=tqdm(range(start_epoch+1,target_epoch+1),desc=f"{arch_id} train",leave=False,unit="epoch")
    for epoch in local_bar:
        model.train(); total=0.0; n=0
        et=time.perf_counter()
        for ids in epoch_batches(len(Xtr),epoch):
            ids_t=torch.as_tensor(ids,dtype=torch.long)
            xb=Xtr.index_select(0,ids_t).to(DEVICE,non_blocking=True); yb=ytr.index_select(0,ids_t).to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred=model(xb); loss=loss_fn(pred,yb); loss.backward(); opt.step()
            total+=float(loss.item())*len(ids); n+=len(ids)
        train_epoch=total/max(n,1)
        val=np.nan
        if epoch==target_epoch or epoch%EVAL_EVERY==0:
            val=eval_mse(model,Xv,yv)
        history.append({"arch_id":arch_id,"epoch":epoch,"train_mse_epoch":train_epoch,"val_mse":val,
                        "epoch_seconds":time.perf_counter()-et})
        local_bar.set_postfix(train=f"{train_epoch:.3g}",val=(f"{val:.3g}" if np.isfinite(val) else "-"))
        if epoch_progress is not None: epoch_progress.update(1)

    # Save resumable state after each rung.
    cpu_model={k:v.detach().cpu() for k,v in model.state_dict().items()}
    cpu_opt=opt.state_dict()
    for st in cpu_opt["state"].values():
        for k,v in st.items():
            if torch.is_tensor(v): st[k]=v.detach().cpu()
    torch.save({"model":cpu_model,"optimizer":cpu_opt,"epoch":target_epoch},ckpt)
    return model,history

# ======================================================================================
# 13. MAIN EXPERIMENT
# ======================================================================================





# ======================================================================================
# 13. LONGITUDINAL COHORT + COMPARABLE TEMPORAL / SCALE VISUALIZATION HELPERS
# ======================================================================================

def gini_coefficient(x):
    a=np.asarray(x,float).ravel()
    a=a[np.isfinite(a)&(a>=0)]
    if len(a)==0 or float(a.sum())<=0:
        return np.nan
    a=np.sort(a); n=len(a)
    return float((2.0*np.sum((np.arange(1,n+1))*a)/(n*np.sum(a)))-(n+1)/n)


def farthest_point_subset(df: pd.DataFrame, fps: FingerprintSpace, n: int):
    """Select a reproducibly diverse fixed cohort in the initialization fingerprint space."""
    if len(df)==0 or n<=0:
        return []
    Z=fps.transform_df(df)
    n=min(int(n),len(df))
    center=np.mean(Z,axis=0)
    selected=[]
    # Seed the cohort with morphology representatives when possible.
    for morph in MORPHOLOGIES:
        ids=np.where(df.morphology.to_numpy()==morph)[0]
        if len(ids) and len(selected)<n:
            j=ids[int(np.argmax(np.linalg.norm(Z[ids]-center[None,:],axis=1)))]
            if int(j) not in selected:
                selected.append(int(j))
    if not selected:
        selected=[int(np.argmax(np.linalg.norm(Z-center[None,:],axis=1)))]
    while len(selected)<n:
        remain=[i for i in range(len(Z)) if i not in selected]
        dmin=[min(np.linalg.norm(Z[i]-Z[j]) for j in selected) for i in remain]
        selected.append(int(remain[int(np.argmax(dmin))]))
    return selected[:n]


def checkpoint_scale_rows(arch_row, epoch, aux):
    aid=str(arch_row.arch_id)
    N=int(aux["D"].shape[0])
    x=np.asarray(aux["ell_over_xi50"],float)
    nb=np.asarray(aux["NB_compressed"],float)
    de=np.asarray(aux["d_eff"],float)
    rr=np.asarray(aux["d_eff_r2"],float)
    lac=np.asarray(aux["lacunarity_compressed"],float)
    rows=[]
    m=min(len(x),len(nb),len(de),len(rr),len(lac))
    for i in range(m):
        rows.append({
            "arch_id":aid,"morphology":arch_row.morphology,"depth":int(arch_row.depth),
            "actual_params":int(arch_row.actual_params),"epoch":int(epoch),
            "ell_over_xi50":float(x[i]),"N_B":float(nb[i]),"N_B_over_N":float(nb[i]/N),
            "d_eff":float(de[i]) if np.isfinite(de[i]) else np.nan,
            "d_eff_r2":float(rr[i]) if np.isfinite(rr[i]) else np.nan,
            "lacunarity_flow":float(lac[i]) if np.isfinite(lac[i]) else np.nan,
        })
    return rows


def checkpoint_layer_rows(arch_row, epoch, aux):
    rows=[]
    for typ,layers in [("edge",aux["edge_enrichment"]),("node",aux["node_enrichment"])]:
        denom=max(len(layers)-1,1)
        for l,a in enumerate(layers):
            v=np.asarray(a,float).ravel()
            v=v[np.isfinite(v)&(v>=0)]
            if len(v)==0:
                continue
            topn=max(1,int(math.ceil(.10*len(v))))
            rows.append({
                "arch_id":str(arch_row.arch_id),"morphology":arch_row.morphology,
                "depth":int(arch_row.depth),"actual_params":int(arch_row.actual_params),
                "epoch":int(epoch),"type":typ,"layer":int(l),"z":float(l/denom),
                "n":int(len(v)),"cv":float(np.std(v)/max(np.mean(v),EPS)),
                "gini":gini_coefficient(v),
                "top10_share":float(np.sort(v)[-topn:].sum()/max(v.sum(),EPS)),
                "median":float(np.median(v)),"p90":float(np.quantile(v,.90)),
            })
    return rows


def save_microscopic_snapshot(arch_id, epoch, aux):
    payload={}
    for l,a in enumerate(aux["edge_enrichment"]): payload[f"edge_{l}"]=np.asarray(a,float)
    for l,a in enumerate(aux["node_enrichment"]): payload[f"node_{l}"]=np.asarray(a,float)
    np.savez_compressed(OUTDIR/"microscopic"/f"{arch_id}_epoch{int(epoch)}.npz",**payload)


def train_longitudinal_architecture(arch_row: pd.Series, splits, fps: FingerprintSpace,
                                    geometry_bar=None, geometry_eta=None, geometry_counter=None,
                                    epoch_bar=None):
    """
    Train one fixed-cohort architecture ONCE from epoch 0 to the maximum budget.
    Geometry is analyzed at identical checkpoints for every cohort member.
    """
    aid=str(arch_row.arch_id)
    widths=json.loads(arch_row.hidden_widths)
    max_epoch=int(max(BUDGETS))
    checkpoints=sorted(set([e for e in GEOMETRY_CHECKPOINTS if 0<=e<=max_epoch] + [0,max_epoch]))

    model=BiasFreeReLUMLP(INPUT_DIM,widths,1,seed=MODEL_INIT_SEED).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    loss_fn=nn.MSELoss()
    Xtr,ytr=splits["train"]

    temporal=[]; scale_rows=[]; layer_rows=[]; history=[]
    local_final=None; renorm_final=None; final_diag_row=None
    z_init=None; z_prev=None; epoch_prev=0
    val0=eval_mse(model,*splits["val"]); test0=eval_mse(model,*splits["test"])

    def snapshot(epoch):
        nonlocal z_init,z_prev,epoch_prev,local_final,renorm_final,final_diag_row
        gt=time.perf_counter()
        train_mse=eval_mse(model,*splits["train"])
        val_mse=eval_mse(model,*splits["val"])
        test_mse=eval_mse(model,*splits["test"])
        weights=extract_weights(model)
        curve_path=OUTDIR/"longitudinal_curves"/f"{aid}_epoch{int(epoch)}.npz"
        geom,aux=analyze_core_geometry(weights,LONGITUDINAL_BCANW_SCALES,save_curve=curve_path)
        z=fps.transform_row(geom)
        if z_init is None:
            z_init=z.copy(); z_prev=z.copy(); epoch_prev=int(epoch)
        deformation=float(np.linalg.norm(z-z_init))
        velocity=float(np.linalg.norm(z-z_prev)/max(int(epoch)-epoch_prev,1)) if int(epoch)>epoch_prev else 0.0
        functional_gain=float((val0-val_mse)/max(abs(val0),EPS))

        row={
            "arch_id":aid,"cohort":"longitudinal","epoch":int(epoch),
            "morphology":arch_row.morphology,"depth":int(arch_row.depth),
            "actual_params":int(arch_row.actual_params),
            "init_structural_cluster":int(arch_row.init_structural_cluster),
            "train_mse":train_mse,"val_mse":val_mse,"test_mse":test_mse,
            "val_mse_init":val0,"test_mse_init":test0,
            "functional_gain":functional_gain,
            "structural_deformation":deformation,"structural_velocity":velocity,
            **geom,
        }
        temporal.append(row)
        scale_rows.extend(checkpoint_scale_rows(arch_row,epoch,aux))
        layer_rows.extend(checkpoint_layer_rows(arch_row,epoch,aux))

        if int(epoch)==max_epoch:
            save_microscopic_snapshot(aid,epoch,aux)
            allwidths=[INPUT_DIM]+list(widths)+[1]
            ldf,delta_grid,rdf,ks=full_final_diagnostics(aux,allwidths,aid)
            ldf["epoch"]=int(epoch); ldf["morphology"]=arch_row.morphology
            rdf["epoch"]=int(epoch); rdf["morphology"]=arch_row.morphology
            local_final=ldf; renorm_final=rdf
            final_diag_row=dict(row)
            for zz,val in zip(LOCAL_DEPTH_GRID,delta_grid):
                final_diag_row[f"delta_z{zz:.2f}"]=val
            final_diag_row["renorm_mean_pairwise_KS"]=ks
            final_diag_row["renorm_flow_retention_mean"]=(
                float(rdf.inter_box_flow_retention.mean()) if len(rdf) else np.nan
            )

        z_prev=z.copy(); epoch_prev=int(epoch)
        elapsed=time.perf_counter()-gt
        tqdm.write(
            f"[longitudinal geometry] {aid} epoch={int(epoch):3d} | "
            f"train={train_mse:.4f} val={val_mse:.4f} test={test_mse:.4f} | "
            f"dB={geom['scaling_dB']:.3f} peaks={geom['n_crossover_peaks']} "
            f"eta1={geom['eta1']:.2e} deform={deformation:.3f} | {human_time(elapsed)}"
        )
        if geometry_bar is not None: geometry_bar.update(1)
        if geometry_counter is not None:
            geometry_counter[0]+=1
            if geometry_eta is not None: geometry_eta.update(elapsed,geometry_counter[0])
        del weights,aux

    # Epoch zero is an explicit longitudinal observation, not borrowed from the lower-resolution map.
    snapshot(0)

    local_train=tqdm(range(1,max_epoch+1),desc=f"{aid} longitudinal",leave=False,unit="epoch")
    for epoch in local_train:
        model.train(); total=0.0; n=0; et=time.perf_counter()
        for ids in epoch_batches(len(Xtr),epoch):
            ids_t=torch.as_tensor(ids,dtype=torch.long)
            xb=Xtr.index_select(0,ids_t).to(DEVICE,non_blocking=True)
            yb=ytr.index_select(0,ids_t).to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred=model(xb); loss=loss_fn(pred,yb); loss.backward(); opt.step()
            total+=float(loss.item())*len(ids); n+=len(ids)
        train_epoch=total/max(n,1)
        val=np.nan
        if epoch%EVAL_EVERY==0 or epoch in checkpoints or epoch==max_epoch:
            val=eval_mse(model,*splits["val"])
        history.append({
            "arch_id":aid,"cohort":"longitudinal","morphology":arch_row.morphology,
            "epoch":int(epoch),"train_mse_epoch":train_epoch,"val_mse":val,
            "epoch_seconds":time.perf_counter()-et,
        })
        local_train.set_postfix(train=f"{train_epoch:.3g}",val=(f"{val:.3g}" if np.isfinite(val) else "-"))
        if epoch_bar is not None: epoch_bar.update(1)
        if epoch in checkpoints:
            snapshot(epoch)

    # Save final state for reproducibility / later diagnostics.
    cpu_model={k:v.detach().cpu() for k,v in model.state_dict().items()}
    cpu_opt=opt.state_dict()
    for st in cpu_opt["state"].values():
        for k,v in st.items():
            if torch.is_tensor(v): st[k]=v.detach().cpu()
    torch.save({"model":cpu_model,"optimizer":cpu_opt,"epoch":max_epoch},
               OUTDIR/"checkpoints"/f"{aid}_longitudinal.pt")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return (pd.DataFrame(temporal),pd.DataFrame(scale_rows),pd.DataFrame(layer_rows),
            pd.DataFrame(history),local_final,renorm_final,final_diag_row)


def curve_file_to_scale_rows(arch_row, epoch, curve_path):
    if not Path(curve_path).exists():
        return []
    z=np.load(curve_path)
    x=np.asarray(z["ell_over_xi50"],float)
    nb=np.asarray(z["NB_compressed"],float)
    de=np.asarray(z["d_eff"],float)
    rr=np.asarray(z["d_eff_r2"],float)
    lac=np.asarray(z["lacunarity_compressed"],float)
    N=int(arch_row.n_graph_nodes) if "n_graph_nodes" in arch_row else int(arch_row.depth+2)
    rows=[]
    m=min(len(x),len(nb),len(de),len(rr),len(lac))
    for i in range(m):
        rows.append({
            "arch_id":str(arch_row.arch_id),"morphology":arch_row.morphology,
            "depth":int(arch_row.depth),"actual_params":int(arch_row.actual_params),
            "epoch":int(epoch),"ell_over_xi50":float(x[i]),"N_B":float(nb[i]),
            "N_B_over_N":float(nb[i]/max(N,1)),
            "d_eff":float(de[i]) if np.isfinite(de[i]) else np.nan,
            "d_eff_r2":float(rr[i]) if np.isfinite(rr[i]) else np.nan,
            "lacunarity_flow":float(lac[i]) if np.isfinite(lac[i]) else np.nan,
        })
    return rows


def build_all_trained_comparison_tables(train_df,map_df,rung_df,long_tdf,long_sdf,
                                        funnel_hist,long_hist,cohort_ids):
    """Build one harmonized table so every trained architecture uses identical plotting code."""
    temporal=[]; scales=[]

    # Fixed cohort already has dense longitudinal records.
    if len(long_tdf): temporal.extend(long_tdf.to_dict("records"))
    if len(long_sdf): scales.extend(long_sdf.to_dict("records"))

    # Funnel architectures: initialization + whichever rung endpoints they reached.
    cohort_ids=set(map(str,cohort_ids))
    for _,ar in train_df.iterrows():
        aid=str(ar.arch_id)
        if aid in cohort_ids:
            continue
        init=map_df[map_df.arch_id==aid].iloc[0].to_dict()
        init_row={
            "arch_id":aid,"cohort":"funnel","epoch":0,"morphology":ar.morphology,
            "depth":int(ar.depth),"actual_params":int(ar.actual_params),
            "train_mse":np.nan,"val_mse":np.nan,"test_mse":np.nan,
            "structural_deformation":0.0,"structural_velocity":0.0,
        }
        init_row.update({k:init.get(k,np.nan) for k in CORE_FEATURES})
        # Keep the scalar path/geometry fields used by consistent plots.
        for k in ["eta1","eta2","entropy_deficit","edge_cv","node_cv","xi50",
                  "xi25_over_xi50","xi10_over_xi50","scaling_dB","scaling_span",
                  "peak1_x","peak1_d","peak2_x","peak2_d","x_lacunarity_star","lacunarity_star"]:
            init_row[k]=init.get(k,np.nan)
        temporal.append(init_row)
        scales.extend(curve_file_to_scale_rows(ar,0,OUTDIR/"curves"/f"{aid}_init.npz"))

        if len(rung_df):
            for _,rr in rung_df[rung_df.arch_id==aid].iterrows():
                r=rr.to_dict(); r["cohort"]="funnel"; temporal.append(r)
                scales.extend(curve_file_to_scale_rows(
                    ar,int(rr.epoch),OUTDIR/"curves"/f"{aid}_epoch{int(rr.epoch)}.npz"
                ))

    tdf=pd.DataFrame(temporal)
    sdf=pd.DataFrame(scales)
    hdf=pd.concat([funnel_hist,long_hist],ignore_index=True) if len(funnel_hist) or len(long_hist) else pd.DataFrame()
    return tdf,sdf,hdf


def interpolate_scale_ensemble(long_sdf: pd.DataFrame):
    """Fixed-cohort ensemble statistics on one common intrinsic x-grid."""
    xgrid=np.geomspace(COMMON_X_MIN,COMMON_X_MAX,COMMON_X_POINTS)
    rows=[]
    if len(long_sdf)==0:
        return pd.DataFrame()
    for epoch in sorted(long_sdf.epoch.unique()):
        g=long_sdf[long_sdf.epoch==epoch]
        archs=sorted(g.arch_id.unique())
        fields={"log_nb":[],"d_eff":[],"log_lac":[]}
        for aid in archs:
            a=g[g.arch_id==aid].sort_values("ell_over_xi50")
            x=a.ell_over_xi50.to_numpy(float)
            nb=a.N_B_over_N.to_numpy(float)
            de=a.d_eff.to_numpy(float)
            lac=a.lacunarity_flow.to_numpy(float)
            good=np.isfinite(x)&(x>0)
            x=x[good]; nb=nb[good]; de=de[good]; lac=lac[good]
            if len(x)<2: continue
            def interp(y,logy=False):
                m=np.isfinite(y)
                if logy: m&=(y>0)
                if np.sum(m)<2: return np.full_like(xgrid,np.nan)
                xx=x[m]; yy=y[m]
                o=np.argsort(xx); xx=xx[o]; yy=yy[o]
                vals=np.log10(yy) if logy else yy
                out=np.interp(np.log10(xgrid),np.log10(xx),vals,left=np.nan,right=np.nan)
                return out
            fields["log_nb"].append(interp(nb,True))
            fields["d_eff"].append(interp(de,False))
            fields["log_lac"].append(interp(lac,True))
        arrays={k:np.asarray(v,float) for k,v in fields.items()}
        for ix,xv in enumerate(xgrid):
            row={"epoch":int(epoch),"ell_over_xi50":float(xv)}
            for name,A in arrays.items():
                if A.ndim!=2 or A.shape[0]==0:
                    vals=np.array([])
                else:
                    vals=A[:,ix]; vals=vals[np.isfinite(vals)]
                row[f"{name}_n"]=int(len(vals))
                if len(vals):
                    row[f"{name}_mean"]=float(np.mean(vals))
                    row[f"{name}_std"]=float(np.std(vals))
                    row[f"{name}_p16"]=float(np.percentile(vals,16))
                    row[f"{name}_median"]=float(np.percentile(vals,50))
                    row[f"{name}_p84"]=float(np.percentile(vals,84))
                else:
                    for s in ["mean","std","p16","median","p84"]: row[f"{name}_{s}"]=np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def common_plot_limits(scale_df):
    xlim=(COMMON_X_MIN,COMMON_X_MAX)
    d=scale_df.d_eff.to_numpy(float) if len(scale_df) else np.array([1.0])
    d=d[np.isfinite(d)&(d>=0)]
    dmax=max(3.0,float(np.nanpercentile(d,99.5))*1.08) if len(d) else 4.0
    lac=scale_df.lacunarity_flow.to_numpy(float) if len(scale_df) else np.array([1.0])
    lac=lac[np.isfinite(lac)&(lac>0)]
    lacmax=max(2.0,float(np.nanmax(lac))*1.05) if len(lac) else 10.0
    return {"xlim":xlim,"dylim":(0,dmax),"lacylim":(1,lacmax)}


def plot_one_architecture_page(aid,arch_row,temporal_df,scale_df,history_df,limits,pdf=None):
    t=temporal_df[temporal_df.arch_id==aid].sort_values("epoch")
    s=scale_df[scale_df.arch_id==aid]
    h=history_df[history_df.arch_id==aid].sort_values("epoch") if len(history_df) else pd.DataFrame()
    if len(t)==0 or len(s)==0:
        return
    max_epoch=max(BUDGETS)
    cmap=plt.get_cmap("plasma"); norm=mpl.colors.Normalize(vmin=0,vmax=max_epoch)

    fig,axs=plt.subplots(3,2,figsize=(8.4,9.0))
    fig.suptitle(
        f"{aid} | {arch_row.morphology} | depth={int(arch_row.depth)} | "
        f"params={int(arch_row.actual_params):,}",y=.995
    )

    # a functional trajectory
    ax=axs[0,0]
    if len(h):
        ax.plot(h.epoch,h.train_mse_epoch,label="Train")
        hv=h[np.isfinite(h.val_mse)]
        if len(hv): ax.plot(hv.epoch,hv.val_mse,"o-",ms=3,label="Validation")
    if "test_mse" in t:
        gt=t[np.isfinite(t.test_mse)]
        if len(gt): ax.plot(gt.epoch,gt.test_mse,"s--",ms=3,label="Test @ geometry checkpoints")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_yscale("log"); ax.legend(frameon=False)

    # b path-space
    ax=axs[0,1]
    if "eta1" in t: ax.plot(t.epoch,t.eta1,"o-",label=r"$\eta_1$")
    if "eta2" in t: ax.plot(t.epoch,t.eta2,"s-",label=r"$\eta_2$")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Effective path fraction"); ax.set_yscale("log"); ax.legend(frameon=False)

    # c covering curves
    ax=axs[1,0]
    epoch_curves=[]
    for ep,g in s.groupby("epoch"):
        g=g.sort_values("ell_over_xi50")
        ax.loglog(g.ell_over_xi50,g.N_B_over_N,"o-",ms=2.2,lw=.9,
                  color=cmap(norm(ep)),label=f"{int(ep)}")
        epoch_curves.append(g)
    # architecture-specific temporal master
    xg=np.geomspace(COMMON_X_MIN,COMMON_X_MAX,COMMON_X_POINTS)
    YY=[]
    for g in epoch_curves:
        x=g.ell_over_xi50.to_numpy(float); y=g.N_B_over_N.to_numpy(float)
        m=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
        if np.sum(m)>=2:
            xx=x[m]; yy=np.log10(y[m]); o=np.argsort(xx)
            YY.append(np.interp(np.log10(xg),np.log10(xx[o]),yy[o],left=np.nan,right=np.nan))
    if YY:
        master=np.nanmedian(np.asarray(YY),axis=0)
        good=np.isfinite(master)
        ax.loglog(xg[good],10**master[good],"--",lw=1.4,label="temporal median")
    ax.set_xlim(*limits["xlim"]); ax.set_ylim(0.008,1.05)
    ax.set_xlabel(r"$x=\ell_B/\xi_{50}$"); ax.set_ylabel(r"$N_B/N$")
    ax.legend(frameon=False,ncol=2,title="Epoch")

    # d effective dimension
    ax=axs[1,1]
    for ep,g in s.groupby("epoch"):
        g=g.sort_values("ell_over_xi50")
        good=np.isfinite(g.d_eff)
        ax.semilogx(g.loc[good,"ell_over_xi50"],g.loc[good,"d_eff"],"o-",ms=2.2,lw=.9,
                    color=cmap(norm(ep)),label=f"{int(ep)}")
    ax.set_xlim(*limits["xlim"]); ax.set_ylim(*limits["dylim"])
    ax.set_xlabel(r"$x=\ell_B/\xi_{50}$"); ax.set_ylabel(r"$d_{\rm eff}^{F}(x)$")

    # e lacunarity
    ax=axs[2,0]
    for ep,g in s.groupby("epoch"):
        g=g.sort_values("ell_over_xi50")
        good=np.isfinite(g.lacunarity_flow)&(g.lacunarity_flow>0)
        ax.loglog(g.loc[good,"ell_over_xi50"],g.loc[good,"lacunarity_flow"],"o-",ms=2.2,lw=.9,
                  color=cmap(norm(ep)),label=f"{int(ep)}")
    ax.set_xlim(*limits["xlim"]); ax.set_ylim(*limits["lacylim"])
    ax.set_xlabel(r"$x=\ell_B/\xi_{50}$"); ax.set_ylabel(r"$\Lambda_F(x)$")

    # f temporal geometric descriptors
    ax=axs[2,1]
    if "scaling_dB" in t: ax.plot(t.epoch,t.scaling_dB,"o-",label=r"$d_B^F$")
    if "peak1_d" in t: ax.plot(t.epoch,t.peak1_d,"s-",label=r"peak 1 $d$")
    if "peak2_d" in t: ax.plot(t.epoch,t.peak2_d,"^-",label=r"peak 2 $d$")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dimension / crossover strength")
    ax2=ax.twinx()
    if "xi25_over_xi50" in t: ax2.plot(t.epoch,t.xi25_over_xi50,"--",label=r"$\xi_{25}/\xi_{50}$")
    if "xi10_over_xi50" in t: ax2.plot(t.epoch,t.xi10_over_xi50,":",label=r"$\xi_{10}/\xi_{50}$")
    ax2.set_ylabel("Covering-shape ratio")
    lines=ax.get_lines()+ax2.get_lines()
    ax.legend(lines,[z.get_label() for z in lines],frameon=False,loc="best")

    for letter,ax in zip("abcdef",axs.ravel()):
        ax.text(-.14,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top")
        ax.minorticks_on()

    fig.tight_layout(rect=(0,0,1,.975),w_pad=1.4,h_pad=1.5)
    stem=OUTDIR/"architecture_figures"/f"{aid}_V5_style_temporal"
    fig.savefig(stem.with_suffix(".png"),dpi=500)
    fig.savefig(stem.with_suffix(".pdf"))
    if pdf is not None: _pdf_save(pdf,fig)
    plt.close(fig)


def plot_longitudinal_ensemble(long_tdf,long_sdf,ensemble_df):
    if len(long_tdf)==0 or len(long_sdf)==0 or len(ensemble_df)==0:
        return
    cmap=plt.get_cmap("plasma"); norm=mpl.colors.Normalize(vmin=0,vmax=max(BUDGETS))
    selected=[e for e in [0,5,20,40,80,120,max(BUDGETS)] if e in set(ensemble_df.epoch)]
    limits=common_plot_limits(long_sdf)

    # 1. Ensemble median + 16-84% ribbons at common epochs.
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.4))
    for ep in selected:
        g=ensemble_df[ensemble_df.epoch==ep].sort_values("ell_over_xi50")
        x=g.ell_over_xi50
        col=cmap(norm(ep))
        axs[0].fill_between(x,10**g.log_nb_p16,10**g.log_nb_p84,alpha=.10,color=col)
        axs[0].loglog(x,10**g.log_nb_median,lw=1.2,color=col,label=f"{ep}")
        axs[1].fill_between(x,g.d_eff_p16,g.d_eff_p84,alpha=.10,color=col)
        axs[1].semilogx(x,g.d_eff_median,lw=1.2,color=col)
        axs[2].fill_between(x,10**g.log_lac_p16,10**g.log_lac_p84,alpha=.10,color=col)
        axs[2].loglog(x,10**g.log_lac_median,lw=1.2,color=col)
    axs[0].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$N_B/N$",xlim=limits["xlim"],ylim=(.008,1.05))
    axs[1].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$d_{\rm eff}^{F}$",xlim=limits["xlim"],ylim=limits["dylim"])
    axs[2].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$\Lambda_F$",xlim=limits["xlim"],ylim=limits["lacylim"])
    axs[0].legend(frameon=False,ncol=2,title="Epoch")
    for letter,ax in zip("abc",axs):
        ax.text(-.14,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top"); ax.minorticks_on()
    fig.tight_layout(); savefig(fig,"figure4_longitudinal_scale_ribbons"); plt.close(fig)

    # 2. Mean and variance fields across scale x epoch.
    epochs=sorted(ensemble_df.epoch.unique())
    xvals=sorted(ensemble_df.ell_over_xi50.unique())
    def matrix(col):
        p=ensemble_df.pivot(index="epoch",columns="ell_over_xi50",values=col)
        return p.reindex(index=epochs,columns=xvals).to_numpy(float)
    Zdm=matrix("d_eff_mean"); Zds=matrix("d_eff_std")
    Znm=matrix("log_nb_mean"); Zns=matrix("log_nb_std")
    Zlm=matrix("log_lac_mean"); Zls=matrix("log_lac_std")
    fig,axs=plt.subplots(3,2,figsize=(8.6,8.8))
    ims=[
        axs[0,0].pcolormesh(np.log10(xvals),epochs,Zdm,shading="auto",cmap="viridis"),
        axs[0,1].pcolormesh(np.log10(xvals),epochs,Zds,shading="auto",cmap="magma"),
        axs[1,0].pcolormesh(np.log10(xvals),epochs,Znm,shading="auto",cmap="viridis"),
        axs[1,1].pcolormesh(np.log10(xvals),epochs,Zns,shading="auto",cmap="magma"),
        axs[2,0].pcolormesh(np.log10(xvals),epochs,Zlm,shading="auto",cmap="viridis"),
        axs[2,1].pcolormesh(np.log10(xvals),epochs,Zls,shading="auto",cmap="magma"),
    ]
    labels=[r"Mean $d_{\rm eff}^{F}$",r"SD across architectures of $d_{\rm eff}^{F}$",
            r"Mean $\log_{10}(N_B/N)$",r"SD across architectures of $\log_{10}(N_B/N)$",
            r"Mean $\log_{10}\Lambda_F$",r"SD across architectures of $\log_{10}\Lambda_F$"]
    for ax,im,lab,letter in zip(axs.ravel(),ims,labels,"abcdef"):
        ax.set_xlabel(r"$\log_{10}(\ell/\xi_{50})$"); ax.set_ylabel("Epoch")
        fig.colorbar(im,ax=ax,pad=.02,label=lab)
        ax.text(-.14,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top")
    fig.tight_layout(); savefig(fig,"figure5_longitudinal_scale_time_mean_variance"); plt.close(fig)

    # 3. Temporal descriptor median and architecture variance.
    fig,axs=plt.subplots(2,2,figsize=(8.4,6.1))
    metrics=[
        ("eta1",True,r"$\eta_1$"),
        ("scaling_dB",False,r"$d_B^F$"),
        ("peak1_d",False,"First crossover peak"),
        ("edge_cv",False,"Edge-flow CV"),
    ]
    for ax,(metric,logy,label),letter in zip(axs.ravel(),metrics,"abcd"):
        rows=[]
        for ep,g in long_tdf.groupby("epoch"):
            v=g[metric].to_numpy(float); v=v[np.isfinite(v)]
            if len(v):
                rows.append((ep,np.percentile(v,16),np.median(v),np.percentile(v,84)))
        if rows:
            a=np.asarray(rows,float)
            ax.fill_between(a[:,0],a[:,1],a[:,3],alpha=.18)
            ax.plot(a[:,0],a[:,2],"o-")
        if logy: ax.set_yscale("log")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.text(-.14,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top")
        ax.minorticks_on()
    fig.tight_layout(); savefig(fig,"figure6_longitudinal_descriptor_variance"); plt.close(fig)


def plot_final_architecture_comparison(long_tdf,long_sdf,cohort_df):
    if len(long_tdf)==0 or len(long_sdf)==0:
        return
    final_epoch=int(max(BUDGETS))
    sf=long_sdf[long_sdf.epoch==final_epoch]
    tf=long_tdf[long_tdf.epoch==final_epoch]
    if len(sf)==0: return
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.4))
    morphs={m:i for i,m in enumerate(MORPHOLOGIES)}
    cmap=plt.get_cmap("tab10")
    for aid,g in sf.groupby("arch_id"):
        ar=cohort_df[cohort_df.arch_id==aid].iloc[0]
        col=cmap(morphs.get(ar.morphology,0)%10)
        g=g.sort_values("ell_over_xi50")
        axs[0].loglog(g.ell_over_xi50,g.N_B_over_N,lw=1.0,color=col,alpha=.8)
        good=np.isfinite(g.d_eff)
        axs[1].semilogx(g.loc[good,"ell_over_xi50"],g.loc[good,"d_eff"],lw=1.0,color=col,alpha=.8)
    for m,idx in morphs.items():
        if m in set(cohort_df.morphology): axs[0].plot([],[],color=cmap(idx%10),label=m)
    axs[0].legend(frameon=False,ncol=2)
    if len(tf):
        sc=axs[2].scatter(tf.scaling_dB,tf.peak1_d,c=tf.depth,cmap="viridis",s=36)
        fig.colorbar(sc,ax=axs[2],pad=.02,label="Hidden depth")
        for _,r in tf.iterrows():
            axs[2].annotate(str(r.arch_id)[-4:],(r.scaling_dB,r.peak1_d),
                            xytext=(3,3),textcoords="offset points",fontsize=6)
    axs[0].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$N_B/N$")
    axs[1].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$d_{\rm eff}^{F}(x)$")
    axs[2].set(xlabel=r"$d_B^F$",ylabel="First crossover peak height")
    for letter,ax in zip("abc",axs):
        ax.text(-.14,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top"); ax.minorticks_on()
    fig.tight_layout(); savefig(fig,"figure7_final_longitudinal_architecture_comparison"); plt.close(fig)




def _ccdf_values(x):
    a=np.asarray(x,float).ravel()
    a=a[np.isfinite(a)&(a>0)]
    if len(a)==0: return np.array([]),np.array([])
    a=np.sort(a)
    return a,(len(a)-np.arange(len(a)))/len(a)


def plot_full_budget_diagnostics(full_final_df,local_depth_df,renorm_df,train_df):
    """V5-like final microscopic + local/RG pages for every full-budget architecture."""
    if len(full_final_df)==0:
        return
    with PdfPages(OUTDIR/"all_full_budget_final_diagnostics.pdf") as book:
        for _,fr in tqdm(full_final_df.iterrows(),total=len(full_final_df),
                         desc="Full-budget diagnostic pages",unit="architecture"):
            aid=str(fr.arch_id)
            ar=train_df[train_df.arch_id==aid]
            if len(ar)==0: continue
            ar=ar.iloc[0]
            micro_path=OUTDIR/"microscopic"/f"{aid}_epoch{int(max(BUDGETS))}.npz"

            # Page 1 — microscopic path-flow organization.
            if micro_path.exists():
                z=np.load(micro_path)
                edge_keys=sorted([k for k in z.files if k.startswith("edge_")],
                                 key=lambda s:int(s.split("_")[1]))
                node_keys=sorted([k for k in z.files if k.startswith("node_")],
                                 key=lambda s:int(s.split("_")[1]))
                fig,axs=plt.subplots(2,2,figsize=(8.4,6.2))
                cmap=plt.get_cmap("viridis")
                edge_stats=[]
                for j,k in enumerate(edge_keys):
                    v=np.asarray(z[k],float).ravel(); x,y=_ccdf_values(v)
                    if len(x):
                        axs[0,0].loglog(x,y,lw=.9,color=cmap(j/max(len(edge_keys)-1,1)),label=f"L{j+1}")
                    topn=max(1,int(math.ceil(.1*len(v))))
                    edge_stats.append((j+1,np.std(v)/max(np.mean(v),EPS),gini_coefficient(v),
                                       np.sort(v)[-topn:].sum()/max(v.sum(),EPS)))
                if edge_stats:
                    es=np.asarray(edge_stats,float)
                    axs[0,1].plot(es[:,0],es[:,1],"o-",label="CV")
                    axs[0,1].plot(es[:,0],es[:,2],"s-",label="Gini")
                    axs[0,1].plot(es[:,0],es[:,3],"^-",label="Top-10% share")
                for j,k in enumerate(node_keys):
                    x,y=_ccdf_values(z[k])
                    if len(x): axs[1,0].loglog(x,y,lw=.9,color=cmap(j/max(len(node_keys)-1,1)))
                edge_pool=np.concatenate([np.asarray(z[k],float).ravel() for k in edge_keys]) if edge_keys else np.array([])
                node_pool=np.concatenate([np.asarray(z[k],float).ravel() for k in node_keys]) if node_keys else np.array([])
                if len(edge_pool):
                    v=np.sort(edge_pool[edge_pool>0])[::-1]; axs[1,1].loglog(np.arange(1,len(v)+1)/len(v),v,label="Edges")
                if len(node_pool):
                    v=np.sort(node_pool[node_pool>0])[::-1]; axs[1,1].loglog(np.arange(1,len(v)+1)/len(v),v,label="Nodes")
                axs[0,0].set(xlabel=r"Edge enrichment $\mathcal{F}_e$",ylabel="CCDF")
                axs[0,0].legend(frameon=False,ncol=2)
                axs[0,1].set(xlabel="Edge layer",ylabel="Heterogeneity"); axs[0,1].legend(frameon=False)
                axs[1,0].set(xlabel=r"Node enrichment $\mathcal{F}_i$",ylabel="CCDF")
                axs[1,1].set(xlabel="Normalized rank",ylabel="Flow enrichment"); axs[1,1].legend(frameon=False)
                fig.suptitle(f"{aid} | final microscopic path-flow | {ar.morphology}, depth={int(ar.depth)}",y=.995)
                for letter,ax in zip("abcd",axs.ravel()):
                    ax.text(-.14,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top"); ax.minorticks_on()
                fig.tight_layout(rect=(0,0,1,.975))
                stem=OUTDIR/"architecture_figures"/f"{aid}_final_microscopic"
                fig.savefig(stem.with_suffix(".png"),dpi=500); fig.savefig(stem.with_suffix(".pdf")); _pdf_save(book,fig)
                plt.close(fig)

            # Page 2 — local depth geometry + directed renormalization.
            ld=local_depth_df[local_depth_df.arch_id==aid].sort_values("z") if len(local_depth_df) else pd.DataFrame()
            rg=renorm_df[renorm_df.arch_id==aid].sort_values("ell_over_xi50") if len(renorm_df) else pd.DataFrame()
            fig,axs=plt.subplots(2,2,figsize=(8.4,6.2))
            if len(ld):
                axs[0,0].plot(ld.z,ld.support_median,"o-",label=r"support $d^F$")
                axs[0,0].plot(ld.z,ld.alpha_median,"s-",label=r"flow mass $\alpha^F$")
                axs[0,0].legend(frameon=False)
                axs[0,1].plot(ld.z,ld.delta_median,"o-")
                axs[0,1].axhline(0,ls="--",lw=.8)
                ax2=axs[0,1].twinx()
                ax2.plot(ld.z,ld.valid_delta_fraction,":")
                ax2.set_ylabel("Valid-fit fraction")
            if len(rg):
                for q in RENYI_Q:
                    col=f"eta_q{q:g}"
                    if col in rg: axs[1,0].semilogx(rg.ell_over_xi50,rg[col],"o-",label=rf"$q={q:g}$")
                axs[1,0].legend(frameon=False,ncol=2)
                axs[1,1].semilogx(rg.ell_over_xi50,rg.inter_box_flow_retention,"o-")
            axs[0,0].set(xlabel="Normalized depth z",ylabel="Local dimension")
            axs[0,1].set(xlabel="Normalized depth z",ylabel=r"$\Delta^F=\alpha^F-d^F$")
            axs[1,0].set(xlabel=r"$\ell/\xi_{50}$",ylabel=r"Coarse effective fraction $\eta_q$")
            axs[1,1].set(xlabel=r"$\ell/\xi_{50}$",ylabel="Inter-box flow retention")
            fig.suptitle(f"{aid} | final local + renormalization geometry | {ar.morphology}",y=.995)
            for letter,ax in zip("abcd",axs.ravel()):
                ax.text(-.14,1.05,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top"); ax.minorticks_on()
            fig.tight_layout(rect=(0,0,1,.975))
            stem=OUTDIR/"architecture_figures"/f"{aid}_final_local_renorm"
            fig.savefig(stem.with_suffix(".png"),dpi=500); fig.savefig(stem.with_suffix(".pdf")); _pdf_save(book,fig)
            plt.close(fig)


# ======================================================================================
# 12B. COMPREHENSIVE V5 + EXPLORATORY DIAGNOSTICS FOR EVERY TRAINED ARCHITECTURE
# ======================================================================================

def detailed_local_node_dimensions(D,node_mu,widths,arch_id):
    """Node-level strict COMMON-WINDOW local support / flow-mass dimensions."""
    n=D.shape[0]
    dloc=np.full(n,np.nan); alpha=np.full(n,np.nan)
    r2d=np.full(n,np.nan); r2a=np.full(n,np.nan)
    fit_rmin=np.full(n,np.nan); fit_rmax=np.full(n,np.nan); fit_n=np.full(n,np.nan)
    for i in range(n):
        dv=D[i][np.isfinite(D[i])&(D[i]>0)]
        if len(dv)<10: continue
        rlo=max(float(np.quantile(dv,.03)),EPS)
        rhi=float(np.quantile(dv,.80))
        if rhi<=rlo: continue
        rgrid=np.geomspace(rlo,rhi,24)
        support=[]; mass=[]
        for r in rgrid:
            inside=D[i]<=r
            support.append(np.sum(inside)/n)
            mass.append(float(node_mu[inside].sum()))
        d,aa,rd,ra,rmin,rmax,npts,_=fit_common_local_growth(rgrid,support,mass)
        dloc[i],alpha[i],r2d[i],r2a[i]=d,aa,rd,ra
        fit_rmin[i],fit_rmax[i],fit_n[i]=rmin,rmax,npts
    delta=alpha-dloc
    offsets=np.cumsum([0]+list(widths))
    layer=np.full(n,-1,int); z=np.full(n,np.nan)
    for l in range(len(widths)):
        layer[offsets[l]:offsets[l+1]]=l
        z[offsets[l]:offsets[l+1]]=l/max(len(widths)-1,1)
    return pd.DataFrame({
        "arch_id":arch_id,"node":np.arange(n),"node_layer":layer,"z":z,
        "support_dimension":dloc,"flow_mass_dimension":alpha,
        "delta_alpha_minus_support":delta,
        "R2_support":r2d,"R2_flow_mass":r2a,
        "common_fit_rmin":fit_rmin,"common_fit_rmax":fit_rmax,"common_fit_points":fit_n,
        "common_window_valid":np.isfinite(delta),
    })


def architecture_master_products(scale_df):
    """Temporal master curve, residual matrix, and epoch-wise RMS for one architecture."""
    if len(scale_df)==0:
        return pd.DataFrame(),{}
    epochs=sorted(scale_df.epoch.unique())
    xgrid=np.geomspace(COMMON_X_MIN,COMMON_X_MAX,COMMON_X_POINTS)
    Y=[]; eps=[]
    for ep in epochs:
        g=scale_df[scale_df.epoch==ep].sort_values("ell_over_xi50")
        x=g.ell_over_xi50.to_numpy(float); y=g.N_B_over_N.to_numpy(float)
        m=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
        if np.sum(m)<2: continue
        x=x[m]; y=y[m]; o=np.argsort(x)
        yi=np.interp(np.log10(xgrid),np.log10(x[o]),np.log10(y[o]),left=np.nan,right=np.nan)
        Y.append(yi); eps.append(int(ep))
    if len(Y)<2:
        return pd.DataFrame(),{}
    Y=np.asarray(Y,float)
    master=np.nanmedian(Y,axis=0)
    errors={}
    rows=[]
    for i,ep in enumerate(eps):
        good=np.isfinite(Y[i])&np.isfinite(master)
        errors[ep]=float(np.sqrt(np.mean((Y[i,good]-master[good])**2))) if np.any(good) else np.nan
    for j,x in enumerate(xgrid):
        row={"ell_over_xi50":x,"master_NB_over_N":10**master[j] if np.isfinite(master[j]) else np.nan}
        for i,ep in enumerate(eps):
            row[f"epoch_{ep}_NB_over_N_interp"]=10**Y[i,j] if np.isfinite(Y[i,j]) else np.nan
            row[f"epoch_{ep}_residual_log10"]=Y[i,j]-master[j] if np.isfinite(Y[i,j]) and np.isfinite(master[j]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows),errors


def add_master_error_column(temporal_df,scale_df):
    t=temporal_df.copy()
    if "master_curve_error_xi50" not in t.columns:
        t["master_curve_error_xi50"]=np.nan
    for aid,g in scale_df.groupby("arch_id"):
        _,errors=architecture_master_products(g)
        for ep,val in errors.items():
            mask=(t.arch_id.astype(str)==str(aid))&(t.epoch==ep)
            t.loc[mask,"master_curve_error_xi50"]=val
    return t


def renorm_all_scales_from_aux(aux,widths,arch_id):
    N=aux["D"].shape[0]
    rows=[]; target_detail=[]
    NB=np.asarray(aux["NB"],float); radii=np.asarray(aux["radii"],float)
    xi50=float(aux["xi50"])
    selected=[]
    for i in range(len(radii)):
        if NB[i]<3 or NB[i]>.90*N: continue
        vals,p,norm,ret=directed_quotient(aux["edge_flow"],aux["colors_by_scale"][i],widths)
        if len(p)<3: continue
        row={"arch_id":arch_id,"ell_over_xi50":float(radii[i]/max(xi50,EPS)),
             "N_B":float(NB[i]),"N_B_over_N":float(NB[i]/N),
             "n_directed_quotient_edges":int(len(p)),
             "inter_box_flow_retention":float(ret)}
        for q in RENYI_Q: row[f"eta_q{q:g}"]=renyi_effective_fraction(p,q)
        rows.append(row)

    frac=NB/N
    for target in RENORM_TARGET_BOX_FRACTIONS:
        eligible=np.where((NB>=3)&(NB<=.90*N))[0]
        if not len(eligible): continue
        idx=int(eligible[np.argmin(np.abs(frac[eligible]-target))])
        if idx in selected: continue
        selected.append(idx)
        vals,p,norm,ret=directed_quotient(aux["edge_flow"],aux["colors_by_scale"][idx],widths)
        for j,v in enumerate(norm):
            target_detail.append({
                "arch_id":arch_id,"target_box_fraction":float(target),
                "ell_over_xi50":float(radii[idx]/max(xi50,EPS)),
                "N_B_over_N":float(frac[idx]),
                "normalized_directed_quotient_flow":float(v),
                "directed_quotient_probability":float(p[j]) if j<len(p) else np.nan,
                "inter_box_flow_retention":float(ret),
            })
    return pd.DataFrame(rows),pd.DataFrame(target_detail)


def tail_tables_from_aux(aux,arch_id):
    layer_rows=[]
    for l,E in enumerate(aux["edge_enrichment"]):
        vals=np.asarray(E,float).ravel()
        res=compare_tail_models(vals,min_tail=min(MIN_TAIL_POINTS,max(12,len(vals)//4)))
        res.update({"arch_id":arch_id,"layer":l+1,"n_edges":len(vals)})
        layer_rows.append(res)
    pool=np.concatenate([np.asarray(E,float).ravel() for E in aux["edge_enrichment"]])
    pooled=compare_tail_models(pool,min_tail=min(MIN_TAIL_POINTS,max(12,len(pool)//100)))
    pooled.update({"arch_id":arch_id,"layer":"pooled","n_edges":len(pool)})
    return pd.DataFrame(layer_rows),pd.DataFrame([pooled])


def metric_and_cover_robustness(weights,base_geom,base_aux,widths,arch_id):
    metric_rows=[]; metric_curve_rows=[]
    # p=1 can reuse base; other positive powers are recomputed.
    for pwr in METRIC_POWER_ROBUSTNESS:
        if abs(pwr-METRIC_POWER)<1e-12:
            geom,aux=base_geom,base_aux
            owns=False
        else:
            geom,aux=analyze_core_geometry(weights,TRAIN_BCANW_SCALES,metric_power=pwr)
            owns=True
        metric_rows.append({
            "arch_id":arch_id,"metric_power":float(pwr),"xi50":geom.get("xi50",np.nan),
            "x_d_star":geom.get("x_d_star",np.nan),"d_star":geom.get("d_star",np.nan),
            "x_lacunarity_star":geom.get("x_lacunarity_star",np.nan),
            "scaling_window_credible":geom.get("scaling_credible",False),
            "scaling_window_dB":geom.get("scaling_dB",np.nan),
        })
        x=np.asarray(aux["ell_over_xi50"],float)
        nb=np.asarray(aux["NB_compressed"],float)/max(aux["D"].shape[0],1)
        de=np.asarray(aux["d_eff"],float)
        for i in range(min(len(x),len(nb),len(de))):
            metric_curve_rows.append({
                "arch_id":arch_id,"metric_power":float(pwr),"ell_over_xi50":float(x[i]),
                "N_B_over_N":float(nb[i]),"d_eff":float(de[i]) if np.isfinite(de[i]) else np.nan
            })
        if owns:
            del aux

    # Greedy-cover order uncertainty at the primary p=1 metric.
    D=base_aux["D"]; radii=np.asarray(base_aux["radii"],float)
    Pgeom=graph_from_matrices(base_aux["edge_enrichment"])
    score=node_strength(Pgeom); N=D.shape[0]
    rng=np.random.default_rng(SEED+abs(hash(arch_id))%100000)
    cover_rows=[]; cover_curve_rows=[]
    for r in range(COVER_ORDER_REPEATS):
        noisy=score*np.exp(rng.normal(0,COVER_ORDER_NOISE,size=len(score)))
        order=np.argsort(-noisy)
        NB=[]
        for ell in radii:
            c=greedy_bcanw_coloring(D,float(ell),order)
            NB.append(int(c.max())+1)
        NB=np.asarray(NB,float)
        rc,nc,_=compress_staircase(radii,NB,None)
        xi50=covering_scale(rc,nc,N,.50)
        x=rc/max(xi50,EPS)
        de,rr=local_effective_dimension(rc,nc)
        peaks=crossover_peaks(x,de,rr)
        good=np.isfinite(x)&np.isfinite(de)&np.isfinite(rr)&(rr>=LOCAL_SLOPE_R2_MIN)&(de>0)
        if np.any(good):
            ids=np.where(good)[0]; ip=int(ids[np.argmax(de[ids])])
            xd=float(x[ip]); ds=float(de[ip])
        else:
            xd=ds=np.nan
        cover_rows.append({"arch_id":arch_id,"repeat":r,"xi50":xi50,"x_d_star":xd,"d_star":ds})
        xraw=radii/max(xi50,EPS)
        for j in range(len(radii)):
            cover_curve_rows.append({"arch_id":arch_id,"repeat":r,
                                     "ell_over_xi50":float(xraw[j]),
                                     "N_B_over_N":float(NB[j]/N)})
    return (pd.DataFrame(metric_rows),pd.DataFrame(metric_curve_rows),
            pd.DataFrame(cover_rows),pd.DataFrame(cover_curve_rows))


def load_final_model_for_arch(ar):
    aid=str(ar.arch_id); widths=json.loads(ar.hidden_widths)
    candidates=[
        OUTDIR/"checkpoints"/f"{aid}_longitudinal.pt",
        OUTDIR/"checkpoints"/f"{aid}.pt",
    ]
    ckpt=next((p for p in candidates if p.exists()),None)
    if ckpt is None: return None,None
    model=BiasFreeReLUMLP(INPUT_DIM,widths,1,seed=MODEL_INIT_SEED).to(DEVICE)
    state=torch.load(ckpt,map_location="cpu",weights_only=False)
    model.load_state_dict(state["model"])
    return model,[INPUT_DIM]+list(widths)+[1]


def build_comprehensive_final_tables(full_final_df,train_df):
    """Recompute final p=1 once per full-budget architecture and derive every V5/E4-E8 table."""
    tables={k:[] for k in [
        "tail_layer","tail_pool","local_node","renorm_all","quotient_detail",
        "metric","metric_curves","cover","cover_curves"
    ]}
    if not RUN_COMPREHENSIVE_FINAL_DIAGNOSTICS or len(full_final_df)==0:
        return {k:pd.DataFrame() for k in tables}
    eta=ETAReporter("comprehensive final diagnostics",len(full_final_df))
    for ii,(_,fr) in enumerate(tqdm(full_final_df.iterrows(),total=len(full_final_df),
                                    desc="Comprehensive final diagnostics",unit="architecture"),1):
        st=time.perf_counter(); aid=str(fr.arch_id)
        ar=train_df[train_df.arch_id.astype(str)==aid]
        if len(ar)==0: continue
        ar=ar.iloc[0]
        model,widths=load_final_model_for_arch(ar)
        if model is None: continue
        weights=extract_weights(model)
        geom,aux=analyze_core_geometry(weights,TRAIN_BCANW_SCALES,metric_power=METRIC_POWER)

        tl,tp=tail_tables_from_aux(aux,aid)
        ln=detailed_local_node_dimensions(aux["D"],aux["node_mu"],widths,aid)
        ra,qd=renorm_all_scales_from_aux(aux,widths,aid)
        mr,mc,cr,cc=metric_and_cover_robustness(weights,geom,aux,widths,aid)

        for key,df in [("tail_layer",tl),("tail_pool",tp),("local_node",ln),
                       ("renorm_all",ra),("quotient_detail",qd),("metric",mr),
                       ("metric_curves",mc),("cover",cr),("cover_curves",cc)]:
            if len(df): tables[key].append(df)
        del model,weights,aux
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        eta.update(time.perf_counter()-st,ii)
    eta.finish()

    out={}
    for key,frames in tables.items():
        df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        out[key]=df
        if len(df):
            atomic_csv(df,OUTDIR/"comprehensive_tables"/f"{key}.csv")
    return out


def _maybe_show_or_close(fig):
    if SHOW_COMPREHENSIVE_FIGURES_IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


def _save_arch_fig(fig,folder,name,pdf=None):
    folder.mkdir(parents=True,exist_ok=True)
    if SAVE_INDIVIDUAL_PNG:
        fig.savefig(folder/f"{name}.png",dpi=COMPREHENSIVE_PNG_DPI)
    if SAVE_INDIVIDUAL_FIGURE_PDF:
        fig.savefig(folder/f"{name}.pdf")
    if pdf is not None:
        targets = pdf if isinstance(pdf, (list, tuple)) else [pdf]
        for book in targets:
            if book is not None:
                _pdf_save(book,fig)


def _style_panels(axs):
    arr=np.asarray(axs,dtype=object).ravel()
    for letter,ax in zip("abcdefghijklmnopqrstuvwxyz",arr):
        ax.text(-.15,1.06,letter,transform=ax.transAxes,fontweight="bold",fontsize=11,va="top")
        ax.minorticks_on()
        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)


def _arch_epoch_fields(sdf):
    eps=sorted(sdf.epoch.unique())
    xgrid=np.geomspace(COMMON_X_MIN,COMMON_X_MAX,COMMON_X_POINTS)
    def field(col,logy=False):
        Z=[]
        for ep in eps:
            g=sdf[sdf.epoch==ep].sort_values("ell_over_xi50")
            x=g.ell_over_xi50.to_numpy(float); y=g[col].to_numpy(float)
            m=np.isfinite(x)&np.isfinite(y)&(x>0)
            if logy: m&=(y>0)
            if np.sum(m)<2:
                Z.append(np.full_like(xgrid,np.nan)); continue
            x=x[m]; y=y[m]; o=np.argsort(x)
            yy=np.log10(y[o]) if logy else y[o]
            Z.append(np.interp(np.log10(xgrid),np.log10(x[o]),yy,left=np.nan,right=np.nan))
        return np.asarray(Z,float)
    return eps,xgrid,field("d_eff"),field("lacunarity_flow"),field("N_B_over_N",logy=True)


def _architecture_microscopic_arrays(aid):
    p=OUTDIR/"microscopic"/f"{aid}_epoch{int(max(BUDGETS))}.npz"
    return np.load(p) if p.exists() else None




def _finite_value(row, key, default=np.nan):
    try:
        v=row[key]
        return float(v) if np.isfinite(v) else default
    except Exception:
        return default


def _series_cv(x):
    a=np.asarray(x,float)
    a=a[np.isfinite(a)]
    if len(a)==0: return np.nan
    return float(np.std(a)/max(abs(np.mean(a)),EPS))



def _fixed_temporal_for_ensemble(temporal_df):
    """Use the unchanged longitudinal cohort for across-architecture temporal variability."""
    if len(temporal_df)==0:
        return temporal_df.copy()
    if "cohort" in temporal_df.columns:
        g=temporal_df[temporal_df["cohort"].astype(str)=="longitudinal"].copy()
        if len(g):
            return g
    return temporal_df.copy()


def build_entropy_ensemble_table(temporal_df):
    """
    Across-architecture entropy statistics on the fixed longitudinal cohort.
    Stores raw deficit, deficit normalized by log(N_topological), and change from epoch 0.
    """
    t=_fixed_temporal_for_ensemble(temporal_df)
    if len(t)==0 or "entropy_deficit" not in t.columns:
        return pd.DataFrame()

    work=t.copy()
    # Architecture-specific baseline.
    init=(work.sort_values("epoch").groupby("arch_id",as_index=False)
          .first()[["arch_id","entropy_deficit"]]
          .rename(columns={"entropy_deficit":"entropy_deficit_init"}))
    work=work.merge(init,on="arch_id",how="left")
    work["entropy_deficit_delta"]=work["entropy_deficit"]-work["entropy_deficit_init"]

    if "log_N_topological" in work.columns:
        denom=work["log_N_topological"].to_numpy(float)
        work["entropy_deficit_fraction"]=work["entropy_deficit"].to_numpy(float)/np.maximum(np.abs(denom),EPS)
    else:
        work["entropy_deficit_fraction"]=np.nan

    rows=[]
    metrics=["entropy_deficit","entropy_deficit_delta","entropy_deficit_fraction"]
    for ep,g in work.groupby("epoch"):
        row={"epoch":int(ep),"n_architectures":int(g.arch_id.nunique())}
        for col in metrics:
            v=g[col].to_numpy(float)
            v=v[np.isfinite(v)]
            if len(v):
                row[f"{col}_mean"]=float(np.mean(v))
                row[f"{col}_std"]=float(np.std(v))
                row[f"{col}_p16"]=float(np.percentile(v,16))
                row[f"{col}_median"]=float(np.median(v))
                row[f"{col}_p84"]=float(np.percentile(v,84))
            else:
                for sfx in ["mean","std","p16","median","p84"]:
                    row[f"{col}_{sfx}"]=np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def build_local_dimension_ensemble_density(local_node_df, n_bins=70):
    """
    Median + 16-84% density envelope across architectures for strictly validated
    final local support dimension, flow-mass dimension, and Delta=alpha-d.
    """
    if len(local_node_df)==0:
        return pd.DataFrame()

    df=local_node_df.copy()
    quantities=[
        ("support_dimension","support"),
        ("flow_mass_dimension","flow_mass"),
        ("delta_alpha_minus_support","delta"),
    ]
    rows=[]
    for col,label in quantities:
        allv=df[col].to_numpy(float)
        allv=allv[np.isfinite(allv)]
        if len(allv)<10:
            continue
        lo=float(np.percentile(allv,0.5))
        hi=float(np.percentile(allv,99.5))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
            lo=float(np.nanmin(allv)); hi=float(np.nanmax(allv)+1e-6)
        edges=np.linspace(lo,hi,int(n_bins)+1)
        centers=.5*(edges[:-1]+edges[1:])
        per_arch=[]
        for aid,g in df.groupby("arch_id"):
            v=g[col].to_numpy(float)
            v=v[np.isfinite(v)]
            if len(v)<5:
                continue
            dens,_=np.histogram(v,bins=edges,density=True)
            per_arch.append(dens)
        A=np.asarray(per_arch,float)
        if A.ndim!=2 or A.shape[0]==0:
            continue
        pooled,_=np.histogram(allv,bins=edges,density=True)
        for i,x in enumerate(centers):
            rows.append({
                "quantity":label,
                "x":float(x),
                "n_architectures":int(A.shape[0]),
                "density_mean":float(np.mean(A[:,i])),
                "density_std":float(np.std(A[:,i])),
                "density_p16":float(np.percentile(A[:,i],16)),
                "density_median":float(np.median(A[:,i])),
                "density_p84":float(np.percentile(A[:,i],84)),
                "density_pooled":float(pooled[i]),
            })
    return pd.DataFrame(rows)


def plot_entropy_across_architectures(temporal_df, entropy_ensemble_df, pdf=None):
    """Purple temporal entropy curves + across-architecture variability, matching the reference visual."""
    t=_fixed_temporal_for_ensemble(temporal_df)
    if len(t)==0 or len(entropy_ensemble_df)==0:
        return None
    e=entropy_ensemble_df.sort_values("epoch")

    fig,axs=plt.subplots(1,3,figsize=(11.0,3.45))

    # Raw entropy deficit: show each architecture faintly, then ensemble ribbon/median.
    ax=axs[0]
    for aid,g in t.groupby("arch_id"):
        g=g.sort_values("epoch")
        ax.plot(g.epoch,g.entropy_deficit,color=ENTROPY_COLOR,alpha=.13,lw=.8)
    ax.fill_between(e.epoch,e.entropy_deficit_p16,e.entropy_deficit_p84,
                    color=ENTROPY_COLOR,alpha=.22,label="16-84% architecture band")
    ax.plot(e.epoch,e.entropy_deficit_median,color=ENTROPY_DARK,lw=2.0,
            marker="o",ms=3,label="Median")
    ax.set(xlabel="Epoch",ylabel=r"Entropy deficit $\log N_{\rm topo}-S_{\rm path}$")
    ax.legend(frameon=False)

    # Change from initialization.
    ax=axs[1]
    ax.axhline(0,ls="--",lw=.8,color=".45")
    ax.fill_between(e.epoch,e.entropy_deficit_delta_p16,e.entropy_deficit_delta_p84,
                    color=ENTROPY_COLOR,alpha=.22)
    ax.plot(e.epoch,e.entropy_deficit_delta_median,color=ENTROPY_DARK,lw=2.0,
            marker="o",ms=3)
    ax.set(xlabel="Epoch",ylabel=r"$\Delta$ entropy deficit from initialization")

    # Architecture-normalized deficit: directly comparable across different depths/path counts.
    ax=axs[2]
    good=np.isfinite(e.entropy_deficit_fraction_median)
    if np.any(good):
        ax.fill_between(e.loc[good,"epoch"],e.loc[good,"entropy_deficit_fraction_p16"],
                        e.loc[good,"entropy_deficit_fraction_p84"],
                        color=ENTROPY_COLOR,alpha=.22)
        ax.plot(e.loc[good,"epoch"],e.loc[good,"entropy_deficit_fraction_median"],
                color=ENTROPY_DARK,lw=2.0,marker="o",ms=3)
    ax.set(xlabel="Epoch",ylabel=r"Normalized deficit $(\log N_{\rm topo}-S)/\log N_{\rm topo}$")

    _style_panels(axs)
    fig.suptitle("Across-architecture path entropy organization | fixed longitudinal cohort",y=.995)
    fig.tight_layout(rect=(0,0,1,.965))
    savefig(fig,"figure8_entropy_across_architectures")
    if pdf is not None:
        _pdf_save(pdf,fig)
    plt.close(fig)
    return fig


def plot_local_dimensions_across_architectures(local_node_df, density_df, pdf=None):
    """Reference two-panel local-dimension view, but with 16-84% variability across architectures."""
    if len(local_node_df)==0 or len(density_df)==0:
        return None

    fig,axs=plt.subplots(1,2,figsize=(8.6,3.65))

    ax=axs[0]
    for quantity,label,color in [
        ("support","support $d_i^F$",LOCAL_SUPPORT_COLOR),
        ("flow_mass","flow mass $\\alpha_i^F$",LOCAL_ALPHA_COLOR),
    ]:
        g=density_df[density_df.quantity==quantity].sort_values("x")
        if len(g):
            ax.fill_between(g.x,g.density_p16,g.density_p84,color=color,alpha=.12)
            ax.plot(g.x,g.density_median,color=color,lw=1.7,label=label)
            ax.plot(g.x,g.density_pooled,color=color,lw=.9,ls=":",alpha=.8)
    ax.set(xlabel="Strictly validated local dimension",ylabel="Density")
    ax.legend(frameon=False)

    ax=axs[1]
    g=density_df[density_df.quantity=="delta"].sort_values("x")
    if len(g):
        ax.fill_between(g.x,g.density_p16,g.density_p84,color=LOCAL_DELTA_COLOR,alpha=.18)
        ax.plot(g.x,g.density_median,color=LOCAL_DELTA_COLOR,lw=1.8,label="Median density")
        ax.plot(g.x,g.density_pooled,color=LOCAL_DELTA_COLOR,lw=1.0,ls=":",label="Pooled density")
    ax.axvline(0,ls="--",lw=.9,color=".4")
    ax.set(xlabel=r"$\Delta_i^F=\alpha_i^F-d_i^F$",ylabel="Density")
    ax.legend(frameon=False)

    _style_panels(axs)
    fig.suptitle("Strictly validated local geometry across architectures | final snapshot",y=.995)
    fig.tight_layout(rect=(0,0,1,.965))
    savefig(fig,"figure9_local_dimension_distributions_across_architectures")
    if pdf is not None:
        _pdf_save(pdf,fig)
    plt.close(fig)
    return fig


def _phase_space_panel(ax,t,xc,yc,xlab,ylab):
    """
    Plot a temporal relation without displaying a misleading rho=nan.
    Scientific metrics are unchanged; this only diagnoses data sufficiency.
    """
    if xc not in t.columns or yc not in t.columns:
        ax.text(.5,.5,"metric unavailable",transform=ax.transAxes,ha="center",va="center",color=".45")
        ax.set(xlabel=xlab,ylabel=ylab)
        return None

    good=np.isfinite(t[xc].to_numpy(float)) & np.isfinite(t[yc].to_numpy(float))
    n=int(np.sum(good))
    if n:
        tt=t.loc[good].sort_values("epoch")
        sc=ax.scatter(tt[xc],tt[yc],c=tt.epoch,cmap="viridis",s=36,zorder=3)
        if n>=2:
            ax.plot(tt[xc],tt[yc],color=".72",lw=.8,zorder=1)
    else:
        sc=None

    rho=safe_spearman(t.loc[good,xc],t.loc[good,yc]) if n else np.nan
    if np.isfinite(rho):
        label=rf"$\rho_s={rho:.2f}$  $(n={n})$"
    elif n<MIN_SPEARMAN_POINTS:
        label=rf"Spearman not defined: $n={n}<{MIN_SPEARMAN_POINTS}$"
    else:
        label=rf"Spearman not defined: degenerate variable $(n={n})$"
    ax.text(.04,.95,label,transform=ax.transAxes,va="top",fontsize=8)
    ax.set(xlabel=xlab,ylabel=ylab)
    return sc


def build_architecture_statistical_results(train_df,temporal_df,scale_df,history_df,ext):
    """One exhaustive, human-readable statistical row per selected architecture."""
    rows=[]
    for _,ar in train_df.iterrows():
        aid=str(ar.arch_id)
        t=temporal_df[temporal_df.arch_id.astype(str)==aid].sort_values('epoch').copy()
        h=history_df[history_df.arch_id.astype(str)==aid].sort_values('epoch').copy() if len(history_df) else pd.DataFrame()
        if len(t)==0:
            continue
        init=t.iloc[0]; fin=t.iloc[-1]
        widths=json.loads(ar.hidden_widths)

        gt=t[np.isfinite(t.test_mse)] if 'test_mse' in t else pd.DataFrame()
        gv=t[np.isfinite(t.val_mse)] if 'val_mse' in t else pd.DataFrame()
        hv=h[np.isfinite(h.val_mse)] if len(h) and 'val_mse' in h else pd.DataFrame()
        best_test=gt.loc[gt.test_mse.idxmin()] if len(gt) else None
        best_val=(hv.loc[hv.val_mse.idxmin()] if len(hv) else (gv.loc[gv.val_mse.idxmin()] if len(gv) else None))

        ln=ext.get('local_node',pd.DataFrame())
        ln=ln[ln.arch_id.astype(str)==aid] if len(ln) else pd.DataFrame()
        tp=ext.get('tail_pool',pd.DataFrame())
        tp=tp[tp.arch_id.astype(str)==aid] if len(tp) else pd.DataFrame()
        rg=ext.get('renorm_all',pd.DataFrame())
        rg=rg[rg.arch_id.astype(str)==aid] if len(rg) else pd.DataFrame()
        mr=ext.get('metric',pd.DataFrame())
        mr=mr[mr.arch_id.astype(str)==aid] if len(mr) else pd.DataFrame()
        cr=ext.get('cover',pd.DataFrame())
        cr=cr[cr.arch_id.astype(str)==aid] if len(cr) else pd.DataFrame()

        def delta(col):
            return (_finite_value(fin,col)-_finite_value(init,col)
                    if np.isfinite(_finite_value(fin,col)) and np.isfinite(_finite_value(init,col)) else np.nan)
        def fold(col):
            a=_finite_value(init,col); b=_finite_value(fin,col)
            return b/a if np.isfinite(a) and np.isfinite(b) and abs(a)>EPS else np.nan

        local_support=ln.support_dimension.dropna().to_numpy(float) if len(ln) else np.array([])
        local_alpha=ln.flow_mass_dimension.dropna().to_numpy(float) if len(ln) else np.array([])
        local_delta=ln.delta_alpha_minus_support.dropna().to_numpy(float) if len(ln) else np.array([])
        poolrow=tp.iloc[0] if len(tp) else None

        row={
            'arch_id':aid,'morphology':ar.morphology,'hidden_depth':int(ar.depth),
            'hidden_widths':json.dumps(widths),'actual_params':int(ar.actual_params),
            'target_params':int(ar.target_params),'n_nodes':int(ar.n_nodes),'n_edges':int(ar.n_edges),
            'width_mean':float(ar.width_mean),'width_min':int(ar.width_min),'width_max':int(ar.width_max),
            'width_cv':float(ar.width_cv),'shape':float(ar['shape']),'curvature':float(ar['curvature']),
            'init_structural_cluster':int(ar.init_structural_cluster),
            'final_epoch':int(fin.epoch),
            'final_train_mse':_finite_value(fin,'train_mse'),
            'final_val_mse':_finite_value(fin,'val_mse'),
            'final_test_mse':_finite_value(fin,'test_mse'),
            'best_val_mse':float(best_val.val_mse) if best_val is not None else np.nan,
            'best_val_epoch':int(best_val.epoch) if best_val is not None else np.nan,
            'best_test_mse':float(best_test.test_mse) if best_test is not None else np.nan,
            'best_test_epoch':int(best_test.epoch) if best_test is not None else np.nan,
            'functional_gain_final':_finite_value(fin,'functional_gain'),
            'structural_deformation_final':_finite_value(fin,'structural_deformation'),
            'eta1_init':_finite_value(init,'eta1'),'eta1_final':_finite_value(fin,'eta1'),
            'eta1_fold':fold('eta1'),'eta2_init':_finite_value(init,'eta2'),'eta2_final':_finite_value(fin,'eta2'),
            'eta2_fold':fold('eta2'),'entropy_deficit_init':_finite_value(init,'entropy_deficit'),
            'entropy_deficit_final':_finite_value(fin,'entropy_deficit'),'entropy_deficit_delta':delta('entropy_deficit'),
            'edge_cv_init':_finite_value(init,'edge_cv'),'edge_cv_final':_finite_value(fin,'edge_cv'),'edge_cv_delta':delta('edge_cv'),
            'node_cv_init':_finite_value(init,'node_cv'),'node_cv_final':_finite_value(fin,'node_cv'),'node_cv_delta':delta('node_cv'),
            'xi50_init':_finite_value(init,'xi50'),'xi50_final':_finite_value(fin,'xi50'),'xi50_fold':fold('xi50'),
            'xi25_over_xi50_final':_finite_value(fin,'xi25_over_xi50'),
            'xi10_over_xi50_final':_finite_value(fin,'xi10_over_xi50'),
            'scaling_dB_init':_finite_value(init,'scaling_dB'),'scaling_dB_final':_finite_value(fin,'scaling_dB'),
            'scaling_dB_delta':delta('scaling_dB'),
            'scaling_credible_fraction':float(np.mean(t.scaling_credible.astype(float))) if 'scaling_credible' in t else np.nan,
            'scaling_span_final':_finite_value(fin,'scaling_span'),'scaling_delta_bic_final':_finite_value(fin,'scaling_delta_bic'),
            'd_star_init':_finite_value(init,'d_star'),'d_star_final':_finite_value(fin,'d_star'),'d_star_delta':delta('d_star'),
            'x_d_star_init':_finite_value(init,'x_d_star'),'x_d_star_final':_finite_value(fin,'x_d_star'),'x_d_star_delta':delta('x_d_star'),
            'x_lacunarity_star_final':_finite_value(fin,'x_lacunarity_star'),'lacunarity_star_final':_finite_value(fin,'lacunarity_star'),
            'master_curve_error_mean':float(np.nanmean(t.master_curve_error_xi50)) if 'master_curve_error_xi50' in t else np.nan,
            'master_curve_error_final':_finite_value(fin,'master_curve_error_xi50'),
            'quotient_KS_mean':float(np.nanmean(t.quotient_flow_mean_pairwise_KS)) if 'quotient_flow_mean_pairwise_KS' in t else np.nan,
            'quotient_KS_final':_finite_value(fin,'quotient_flow_mean_pairwise_KS'),
            'rho_eta2_vs_dB':safe_spearman(t.eta2,t.scaling_dB) if 'eta2' in t and 'scaling_dB' in t else np.nan,
            'rho_entropy_vs_dstar':safe_spearman(t.entropy_deficit,t.d_star) if 'entropy_deficit' in t and 'd_star' in t else np.nan,
            'rho_edgeCV_vs_dstar':safe_spearman(t.edge_cv,t.d_star) if 'edge_cv' in t and 'd_star' in t else np.nan,
            'local_support_median':float(np.median(local_support)) if len(local_support) else np.nan,
            'local_support_IQR':float(np.percentile(local_support,75)-np.percentile(local_support,25)) if len(local_support) else np.nan,
            'local_alpha_median':float(np.median(local_alpha)) if len(local_alpha) else np.nan,
            'local_alpha_IQR':float(np.percentile(local_alpha,75)-np.percentile(local_alpha,25)) if len(local_alpha) else np.nan,
            'local_delta_median':float(np.median(local_delta)) if len(local_delta) else np.nan,
            'local_delta_IQR':float(np.percentile(local_delta,75)-np.percentile(local_delta,25)) if len(local_delta) else np.nan,
            'valid_local_delta_fraction':float(ln.delta_alpha_minus_support.notna().mean()) if len(ln) else np.nan,
            'pooled_tail_best_model':str(poolrow.best_model) if poolrow is not None else '',
            'pooled_tail_powerlaw_alpha':float(poolrow.powerlaw_alpha) if poolrow is not None and np.isfinite(poolrow.powerlaw_alpha) else np.nan,
            'pooled_tail_fraction':float(poolrow.tail_fraction) if poolrow is not None and np.isfinite(poolrow.tail_fraction) else np.nan,
            'pooled_tail_deltaBIC_PL_vs_alt':float(poolrow.delta_bic_powerlaw_vs_best_alt) if poolrow is not None and np.isfinite(poolrow.delta_bic_powerlaw_vs_best_alt) else np.nan,
            'renorm_flow_retention_mean':float(rg.inter_box_flow_retention.mean()) if len(rg) else np.nan,
            'renorm_eta_q1_mean':float(rg['eta_q1'].mean()) if len(rg) and 'eta_q1' in rg else np.nan,
            'renorm_eta_q2_mean':float(rg['eta_q2'].mean()) if len(rg) and 'eta_q2' in rg else np.nan,
            'metric_xdstar_CV':_series_cv(mr.x_d_star) if len(mr) else np.nan,
            'metric_dstar_CV':_series_cv(mr.d_star) if len(mr) else np.nan,
            'cover_xdstar_CV':_series_cv(cr.x_d_star) if len(cr) else np.nan,
            'cover_dstar_CV':_series_cv(cr.d_star) if len(cr) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_morphology_group_statistics(stats_df):
    rows=[]
    metrics=['final_test_mse','eta1_fold','eta2_fold','entropy_deficit_delta','scaling_dB_final',
             'd_star_final','x_d_star_final','local_delta_median','quotient_KS_final']
    for morph,g in stats_df.groupby('morphology'):
        row={'morphology':morph,'n_architectures':len(g),
             'depth_min':int(g.hidden_depth.min()),'depth_max':int(g.hidden_depth.max()),
             'params_median':float(g.actual_params.median())}
        for c in metrics:
            v=g[c].to_numpy(float); v=v[np.isfinite(v)]
            row[f'{c}_median']=float(np.median(v)) if len(v) else np.nan
            row[f'{c}_q25']=float(np.percentile(v,25)) if len(v) else np.nan
            row[f'{c}_q75']=float(np.percentile(v,75)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_report(v, digits=4):
    if v is None: return '—'
    if isinstance(v,str): return v if v else '—'
    try:
        if not np.isfinite(v): return '—'
        av=abs(float(v))
        if av!=0 and (av<1e-3 or av>=1e4): return f'{float(v):.{digits}e}'
        return f'{float(v):.{digits}f}'
    except Exception:
        return str(v)


def architecture_statistics_page(ar, stat_row):
    """A4-style architecture description + statistical result table page."""
    fig=plt.figure(figsize=(8.27,11.69))
    ax=fig.add_axes([0.055,0.045,0.89,0.91]); ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_autoscale_on(False)
    aid=str(ar.arch_id)
    widths=json.loads(ar.hidden_widths)
    ax.text(0,0.985,'PATH-FLOW ARCHITECTURE REPORT',fontsize=16,fontweight='bold',va='top')
    ax.text(0,0.955,f'{aid}  |  {ar.morphology}  |  hidden depth = {int(ar.depth)}',fontsize=11,va='top')
    ax.plot([0,1],[0.938,0.938],color='0.25',lw=.8,transform=ax.transAxes)

    width_txt=', '.join(map(str,widths))
    width_txt='\n'.join(textwrap.wrap(width_txt,80))
    config=[
        ('Hidden widths',width_txt),('Target parameters',f'{int(ar.target_params):,}'),('Realized parameters',f'{int(ar.actual_params):,}'),
        ('Graph nodes / edges',f'{int(ar.n_nodes):,} / {int(ar.n_edges):,}'),('Width mean / CV',f'{ar.width_mean:.2f} / {ar.width_cv:.3f}'),
        ('Width min / max',f'{int(ar.width_min)} / {int(ar.width_max)}'),('Morphology shape / curvature',f"{float(ar['shape']):.3f} / {float(ar['curvature']):.3f}"),
        ('Initialization cluster',str(int(ar.init_structural_cluster)))
    ]
    sections=[
        ('ARCHITECTURE CONFIGURATION',config),
        ('FUNCTIONAL TRAJECTORY',[
            ('Final train / val / test MSE',f"{_fmt_report(stat_row.get('final_train_mse'))} / {_fmt_report(stat_row.get('final_val_mse'))} / {_fmt_report(stat_row.get('final_test_mse'))}"),
            ('Best validation MSE (epoch)',f"{_fmt_report(stat_row.get('best_val_mse'))}  ({_fmt_report(stat_row.get('best_val_epoch'),0)})"),
            ('Best test MSE (epoch)',f"{_fmt_report(stat_row.get('best_test_mse'))}  ({_fmt_report(stat_row.get('best_test_epoch'),0)})"),
            ('Final functional gain',_fmt_report(stat_row.get('functional_gain_final'))),
            ('Final structural deformation',_fmt_report(stat_row.get('structural_deformation_final'))),
        ]),
        ('PATH-SPACE / ENTROPY',[
            ('eta1 init -> final (fold)',f"{_fmt_report(stat_row.get('eta1_init'))} -> {_fmt_report(stat_row.get('eta1_final'))}  ({_fmt_report(stat_row.get('eta1_fold'))}x)"),
            ('eta2 init -> final (fold)',f"{_fmt_report(stat_row.get('eta2_init'))} -> {_fmt_report(stat_row.get('eta2_final'))}  ({_fmt_report(stat_row.get('eta2_fold'))}x)"),
            ('Entropy deficit init -> final',f"{_fmt_report(stat_row.get('entropy_deficit_init'))} -> {_fmt_report(stat_row.get('entropy_deficit_final'))}"),
            ('Entropy deficit change',_fmt_report(stat_row.get('entropy_deficit_delta'))),
            ('Edge CV init -> final',f"{_fmt_report(stat_row.get('edge_cv_init'))} -> {_fmt_report(stat_row.get('edge_cv_final'))}"),
            ('Node CV init -> final',f"{_fmt_report(stat_row.get('node_cv_init'))} -> {_fmt_report(stat_row.get('node_cv_final'))}"),
        ]),
        ('INTRINSIC GEOMETRY',[
            ('xi50 init -> final (fold)',f"{_fmt_report(stat_row.get('xi50_init'))} -> {_fmt_report(stat_row.get('xi50_final'))}  ({_fmt_report(stat_row.get('xi50_fold'))}x)"),
            ('xi25/xi50 ; xi10/xi50 final',f"{_fmt_report(stat_row.get('xi25_over_xi50_final'))} ; {_fmt_report(stat_row.get('xi10_over_xi50_final'))}"),
            ('Scaling dB init -> final',f"{_fmt_report(stat_row.get('scaling_dB_init'))} -> {_fmt_report(stat_row.get('scaling_dB_final'))}"),
            ('Credible-window fraction',_fmt_report(stat_row.get('scaling_credible_fraction'))),
            ('d* init -> final',f"{_fmt_report(stat_row.get('d_star_init'))} -> {_fmt_report(stat_row.get('d_star_final'))}"),
            ('x_d* init -> final',f"{_fmt_report(stat_row.get('x_d_star_init'))} -> {_fmt_report(stat_row.get('x_d_star_final'))}"),
            ('x_Lambda* / Lambda* final',f"{_fmt_report(stat_row.get('x_lacunarity_star_final'))} / {_fmt_report(stat_row.get('lacunarity_star_final'))}"),
            ('Master-curve RMS mean / final',f"{_fmt_report(stat_row.get('master_curve_error_mean'))} / {_fmt_report(stat_row.get('master_curve_error_final'))}"),
        ]),
        ('LOCAL / RENORMALIZATION / TAIL / ROBUSTNESS',[
            ('Local support median [IQR]',f"{_fmt_report(stat_row.get('local_support_median'))} [{_fmt_report(stat_row.get('local_support_IQR'))}]"),
            ('Local alpha median [IQR]',f"{_fmt_report(stat_row.get('local_alpha_median'))} [{_fmt_report(stat_row.get('local_alpha_IQR'))}]"),
            ('Local Delta median [IQR]',f"{_fmt_report(stat_row.get('local_delta_median'))} [{_fmt_report(stat_row.get('local_delta_IQR'))}]"),
            ('Valid local Delta fraction',_fmt_report(stat_row.get('valid_local_delta_fraction'))),
            ('Pooled tail best model',_fmt_report(stat_row.get('pooled_tail_best_model'))),
            ('Pooled PL alpha / DeltaBIC',f"{_fmt_report(stat_row.get('pooled_tail_powerlaw_alpha'))} / {_fmt_report(stat_row.get('pooled_tail_deltaBIC_PL_vs_alt'))}"),
            ('Quotient KS mean / final',f"{_fmt_report(stat_row.get('quotient_KS_mean'))} / {_fmt_report(stat_row.get('quotient_KS_final'))}"),
            ('Mean inter-box flow retention',_fmt_report(stat_row.get('renorm_flow_retention_mean'))),
            ('Metric-family CV: x_d* / d*',f"{_fmt_report(stat_row.get('metric_xdstar_CV'))} / {_fmt_report(stat_row.get('metric_dstar_CV'))}"),
            ('Cover-order CV: x_d* / d*',f"{_fmt_report(stat_row.get('cover_xdstar_CV'))} / {_fmt_report(stat_row.get('cover_dstar_CV'))}"),
        ]),
        ('TEMPORAL ASSOCIATIONS (SPEARMAN)',[
            ('rho(eta2, dB)',_fmt_report(stat_row.get('rho_eta2_vs_dB'))),
            ('rho(entropy deficit, d*)',_fmt_report(stat_row.get('rho_entropy_vs_dstar'))),
            ('rho(edge CV, d*)',_fmt_report(stat_row.get('rho_edgeCV_vs_dstar'))),
        ]),
    ]

    y=0.915
    for title,items in sections:
        ax.text(0,y,title,fontsize=9.5,fontweight='bold',va='top'); y-=0.022
        for label,val in items:
            if '\n' in str(val):
                ax.text(0.015,y,label,fontsize=7.5,va='top',color='0.25')
                ax.text(0.34,y,str(val),fontsize=7.5,va='top')
                y-=0.018*(str(val).count('\n')+1)
            else:
                ax.text(0.015,y,label,fontsize=7.5,va='top',color='0.25')
                ax.text(0.52,y,str(val),fontsize=7.5,va='top',ha='left')
                y-=0.018
        y-=0.010
    ax.text(0,0.005,'All scale-dependent plots use the intrinsic coordinate x = ell_B / xi_50. Negative-p metric curves are sign-reversed sensitivity diagnostics.',fontsize=6.8,color='0.35')
    return fig


def global_report_cover_pages(global_book,stats_df):
    fig=plt.figure(figsize=(8.27,11.69)); ax=fig.add_axes([.08,.08,.84,.84]); ax.axis('off')
    ax.text(0,.94,'PATH-FLOW ARCHITECTURE UNIVERSALITY',fontsize=21,fontweight='bold',va='top')
    ax.text(0,.885,'Exhaustive per-architecture temporal geometry and statistical report',fontsize=12,va='top')
    ax.text(0,.82,f'Selected architectures: {len(stats_df)}',fontsize=10)
    ax.text(0,.785,f'Hidden depth range: {int(stats_df.hidden_depth.min())}-{int(stats_df.hidden_depth.max())} (all > 8)',fontsize=10)
    ax.text(0,.75,f'Training horizon: 0-{int(stats_df.final_epoch.max())} epochs',fontsize=10)
    ax.text(0,.715,'Each architecture section contains: configuration/statistical page, path-space entropy, microscopic flow, intrinsic geometry, local dimensions, renormalization, robustness, scale-time fields, phase-space relations, and temporal correlation matrix.',fontsize=9,wrap=True)
    ax.text(0,.60,'Primary interpretation rule',fontsize=11,fontweight='bold')
    ax.text(0,.565,'Use p=1 as the primary path-flow metric. Negative p values are included only because the full sensitivity family was requested; they reverse the usual high-flow = short-distance semantics.',fontsize=9,wrap=True)
    ax.text(0,.08,'Generated automatically by the single-cell experiment script.',fontsize=8,color='0.35')
    _pdf_save(global_book,fig); plt.close(fig)

    # Architecture index pages.
    cols=['arch_id','morphology','hidden_depth','actual_params','final_test_mse','eta1_fold','entropy_deficit_delta','scaling_dB_final','d_star_final']
    per=18
    for start in range(0,len(stats_df),per):
        sub=stats_df.iloc[start:start+per][cols].copy()
        fig,ax=plt.subplots(figsize=(11.69,8.27)); ax.axis('off')
        ax.set_title(f'Architecture index / statistical overview  ({start+1}-{start+len(sub)} of {len(stats_df)})',fontsize=13,pad=16)
        cell=[]
        for _,r in sub.iterrows():
            cell.append([str(r.arch_id),str(r.morphology),str(int(r.hidden_depth)),f"{int(r.actual_params):,}",
                         _fmt_report(r.final_test_mse,3),_fmt_report(r.eta1_fold,3),_fmt_report(r.entropy_deficit_delta,3),
                         _fmt_report(r.scaling_dB_final,3),_fmt_report(r.d_star_final,3)])
        table=ax.table(cellText=cell,colLabels=['ID','Morphology','Depth','Params','Final test','eta1 fold','Delta entropy','dB final','d* final'],
                       loc='center',cellLoc='center',colLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1,1.35)
        _pdf_save(global_book,fig); plt.close(fig)

def plot_comprehensive_architecture_suite(aid,ar,temporal_df,scale_df,history_df,
                                          local_node_df,renorm_all_df,qdetail_df,
                                          tail_layer_df,tail_pool_df,
                                          metric_df,metric_curve_df,cover_df,cover_curve_df,
                                          stat_row=None,global_book=None):
    """
    Reproduce ALL core V5 figures and ALL comprehensive exploratory figures for one architecture.
    The scientific quantities, axis conventions, panel order, and style mirror the supplied scripts.
    """
    t=temporal_df[temporal_df.arch_id.astype(str)==str(aid)].sort_values("epoch").copy()
    s=scale_df[scale_df.arch_id.astype(str)==str(aid)].sort_values(["epoch","ell_over_xi50"]).copy()
    h=history_df[history_df.arch_id.astype(str)==str(aid)].sort_values("epoch") if len(history_df) else pd.DataFrame()
    if len(t)==0 or len(s)==0: return
    folder=OUTDIR/"comprehensive_figures"/str(aid)
    folder.mkdir(parents=True,exist_ok=True)
    master_df,master_errors=architecture_master_products(s)
    t["master_curve_error_xi50"]=[master_errors.get(int(ep),np.nan) for ep in t.epoch]
    cmap=plt.get_cmap("plasma"); norm_ep=mpl.colors.Normalize(vmin=0,vmax=max(BUDGETS))
    micro=_architecture_microscopic_arrays(str(aid))
    tl=tail_layer_df[tail_layer_df.arch_id.astype(str)==str(aid)] if len(tail_layer_df) else pd.DataFrame()
    ln=local_node_df[local_node_df.arch_id.astype(str)==str(aid)] if len(local_node_df) else pd.DataFrame()
    rg=renorm_all_df[renorm_all_df.arch_id.astype(str)==str(aid)] if len(renorm_all_df) else pd.DataFrame()
    qd=qdetail_df[qdetail_df.arch_id.astype(str)==str(aid)] if len(qdetail_df) else pd.DataFrame()
    mr=metric_df[metric_df.arch_id.astype(str)==str(aid)] if len(metric_df) else pd.DataFrame()
    mc=metric_curve_df[metric_curve_df.arch_id.astype(str)==str(aid)] if len(metric_curve_df) else pd.DataFrame()
    cr=cover_df[cover_df.arch_id.astype(str)==str(aid)] if len(cover_df) else pd.DataFrame()
    cc=cover_curve_df[cover_curve_df.arch_id.astype(str)==str(aid)] if len(cover_curve_df) else pd.DataFrame()

    arch_book_ctx=(PdfPages(folder/f"{aid}_ALL_V5_AND_EXPLORATORY_PLOTS.pdf")
                   if WRITE_PER_ARCHITECTURE_BOOKS else nullcontext(None))
    with arch_book_ctx as book:
        pdf_targets=(book,global_book)
        # Architecture description + statistical result page comes first in both books.
        if stat_row is not None:
            sf=architecture_statistics_page(ar,stat_row)
            if SAVE_INDIVIDUAL_PNG:
                sf.savefig(folder/f"{aid}_architecture_statistics.png",dpi=COMPREHENSIVE_PNG_DPI)
            if SAVE_INDIVIDUAL_FIGURE_PDF:
                sf.savefig(folder/f"{aid}_architecture_statistics.pdf")
            for pb in pdf_targets:
                if pb is not None: _pdf_save(pb,sf)
            _maybe_show_or_close(sf)

        # ------------------------------------------------------------------
        # V5 FIGURE 1 — FORMATION OF COMPLETE-PATH MEASURE
        # ------------------------------------------------------------------
        fig,axs=plt.subplots(2,2,figsize=(7.2,5.7))
        ax=axs[0,0]
        if len(h):
            ax.semilogy(h.epoch,h.train_mse_epoch,label="Train")
            hv=h[np.isfinite(h.val_mse)]
            if len(hv): ax.semilogy(hv.epoch,hv.val_mse,label="Validation")
        gt=t[np.isfinite(t.test_mse)] if "test_mse" in t else pd.DataFrame()
        if len(gt): ax.semilogy(gt.epoch,gt.test_mse,"o-",label="Test")
        ax.set(xlabel="Epoch",ylabel="MSE"); ax.legend(frameon=False)
        ax=axs[0,1]; ax.plot(t.epoch,t.eta1,"o-",label=r"$\eta_1$"); ax.plot(t.epoch,t.eta2,"s-",label=r"$\eta_2$")
        ax.set_yscale("log"); ax.set(xlabel="Epoch",ylabel="Effective path fraction"); ax.legend(frameon=False)
        ax=axs[1,0]; ax.plot(t.epoch,t.entropy_deficit,"o-",color=ENTROPY_COLOR); ax.set(xlabel="Epoch",ylabel=r"Entropy deficit $\log N_{\rm topo}-S$")
        ax=axs[1,1]; ax.plot(t.epoch,t.edge_cv,"o-",label="Edge enrichment CV"); ax.plot(t.epoch,t.node_cv,"s-",label="Node enrichment CV")
        ax.set(xlabel="Epoch",ylabel="Flow heterogeneity"); ax.legend(frameon=False)
        _style_panels(axs); fig.suptitle(f"{aid} | path-space formation | {ar.morphology}",y=.995)
        fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"figure1_path_space",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # V5 FIGURE 1A — DEDICATED PATH-ENTROPY ORGANIZATION
        # ------------------------------------------------------------------
        fig,axs=plt.subplots(1,2,figsize=(8.2,3.45))
        axs[0].plot(t.epoch,t.entropy_deficit,"o-",color=ENTROPY_COLOR,lw=1.8,ms=4)
        axs[0].set(xlabel="Epoch",ylabel=r"Entropy deficit $\log N_{\rm topo}-S_{\rm path}$")
        init_entropy=float(t.entropy_deficit.iloc[0]) if len(t) and np.isfinite(t.entropy_deficit.iloc[0]) else np.nan
        d_entropy=t.entropy_deficit-init_entropy if np.isfinite(init_entropy) else np.full(len(t),np.nan)
        axs[1].axhline(0,ls="--",lw=.8,color=".45")
        axs[1].plot(t.epoch,d_entropy,"o-",color=ENTROPY_DARK,lw=1.8,ms=4,label=r"$\Delta$ deficit")
        if "log_N_topological" in t.columns:
            frac=t.entropy_deficit/np.maximum(np.abs(t.log_N_topological),EPS)
            ax2=axs[1].twinx()
            ax2.plot(t.epoch,frac,":",color=ENTROPY_COLOR,lw=1.3,label="normalized deficit")
            ax2.set_ylabel("Normalized entropy deficit")
            lines=axs[1].get_lines()[1:]+ax2.get_lines()
            axs[1].legend(lines,[z.get_label() for z in lines],frameon=False)
        axs[1].set(xlabel="Epoch",ylabel=r"$\Delta$ entropy deficit from initialization")
        _style_panels(axs); fig.suptitle(f"{aid} | path entropy organization",y=.995)
        fig.tight_layout(rect=(0,0,1,.965)); _save_arch_fig(fig,folder,"figure1A_entropy_organization",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # V5 FIGURE 2 — MICROSCOPIC PATH-FLOW TAIL CHARACTERIZATION
        # ------------------------------------------------------------------
        if micro is not None:
            edge_keys=sorted([k for k in micro.files if k.startswith("edge_")],key=lambda z:int(z.split("_")[1]))
            node_keys=sorted([k for k in micro.files if k.startswith("node_")],key=lambda z:int(z.split("_")[1]))
            fig,axs=plt.subplots(2,2,figsize=(7.2,5.7)); cm=plt.get_cmap("viridis")
            for j,k in enumerate(edge_keys):
                x,y=ccdf_positive(micro[k]); axs[0,0].loglog(x,y,lw=1,color=cm(j/max(len(edge_keys)-1,1)),label=f"L{j+1}")
            axs[0,0].axvline(1,ls="--",lw=.8,color=".5"); axs[0,0].set(xlabel=r"Edge flow enrichment $\mathcal{F}_e$",ylabel="CCDF")
            axs[0,0].legend(frameon=False,ncol=2)
            if len(tl):
                axs[0,1].plot(tl.layer,tl.powerlaw_alpha,"o-",label=r"$\gamma_F$")
                ax2=axs[0,1].twinx(); ax2.plot(tl.layer,tl.delta_bic_powerlaw_vs_best_alt,"s--",label=r"$\Delta$BIC")
                ax2.axhline(0,ls=":",lw=.8,color=".4"); ax2.set_ylabel(r"BIC$_{PL}$-BIC$_{best alt}$")
                axs[0,1].set(xlabel="Edge layer",ylabel=r"Tail exponent $\gamma_F$")
                lines=axs[0,1].get_lines()+ax2.get_lines()[:1]
                axs[0,1].legend(lines,[z.get_label() for z in lines],frameon=False)
            for j,k in enumerate(node_keys):
                x,y=ccdf_positive(micro[k]); axs[1,0].loglog(x,y,lw=1,color=cm(j/max(len(node_keys)-1,1)))
            axs[1,0].axvline(1,ls="--",lw=.8,color=".5"); axs[1,0].set(xlabel=r"Node flow enrichment $\mathcal{F}_i$",ylabel="CCDF")
            epool=np.concatenate([np.asarray(micro[k],float).ravel() for k in edge_keys]); npool=np.concatenate([np.asarray(micro[k],float).ravel() for k in node_keys])
            rr,v=rank_curve(epool); axs[1,1].loglog(rr/len(rr),v,label="Edges")
            rr,v=rank_curve(npool); axs[1,1].loglog(rr/len(rr),v,label="Nodes")
            axs[1,1].set(xlabel="Normalized rank",ylabel="Flow enrichment"); axs[1,1].legend(frameon=False)
            _style_panels(axs); fig.suptitle(f"{aid} | microscopic path flow",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"figure2_microscopic_flow",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # V5 FIGURE 3 — INTRINSIC GEOMETRIC SCALING
        # ------------------------------------------------------------------
        fig,axs=plt.subplots(2,2,figsize=(7.2,5.7))
        selected=sorted(t.epoch.unique())
        if len(selected)>7: selected=[selected[i] for i in np.linspace(0,len(selected)-1,7).astype(int)]
        for ep in selected:
            g=s[s.epoch==ep]; col=cmap(norm_ep(ep))
            axs[0,0].loglog(g.ell_over_xi50,g.N_B_over_N,"o-",ms=2.2,lw=.9,color=col,label=f"{int(ep)}")
            good=np.isfinite(g.d_eff); axs[0,1].semilogx(g.loc[good,"ell_over_xi50"],g.loc[good,"d_eff"],"o-",ms=2.2,lw=.9,color=col)
            good=np.isfinite(g.lacunarity_flow)&(g.lacunarity_flow>0); axs[1,0].semilogx(g.loc[good,"ell_over_xi50"],g.loc[good,"lacunarity_flow"],"o-",ms=2.2,lw=.9,color=col)
        if len(master_df): axs[0,0].loglog(master_df.ell_over_xi50,master_df.master_NB_over_N,"--",lw=1.5,color=".1",label="Temporal median")
        axs[0,0].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$N_B/N$"); axs[0,0].legend(frameon=False,ncol=2,title="Epoch")
        axs[0,1].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$d_{\rm eff}^{F}(x)$")
        axs[1,0].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$\Lambda_F(x)$")
        xi_rel=t.xi50/t.xi50.iloc[0] if "xi50" in t and np.isfinite(t.xi50.iloc[0]) else np.full(len(t),np.nan)
        axs[1,1].plot(t.epoch,xi_rel,"o-",label=r"$\xi_{50}/\xi_{50}(0)$")
        axs[1,1].plot(t.epoch,t.peak1_x,"s-",label=r"$x_1$")
        axs[1,1].plot(t.epoch,t.peak2_x,"^-",label=r"$x_2$")
        axs[1,1].plot(t.epoch,t.x_lacunarity_star,"D-",label=r"$x_\Lambda^*$")
        ax2=axs[1,1].twinx(); ax2.plot(t.epoch,t.d_star,"--",label=r"$d^*$"); ax2.set_ylabel(r"Crossover strength $d^*$")
        lines=axs[1,1].get_lines()+ax2.get_lines(); axs[1,1].legend(lines,[z.get_label() for z in lines],frameon=False)
        axs[1,1].set(xlabel="Epoch",ylabel="Dimensionless scale descriptor")
        _style_panels(axs); fig.suptitle(f"{aid} | path-flow geometry",y=.995)
        fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"figure3_pathflow_geometry",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # V5 FIGURE 4 — LOCAL GEOMETRY + DIRECTED RENORMALIZATION
        # ------------------------------------------------------------------
        if len(ln) or len(qd) or len(rg):
            fig,axs=plt.subplots(2,2,figsize=(7.2,5.7))
            if len(ln):
                vd=ln.support_dimension.dropna().to_numpy(); va=ln.flow_mass_dimension.dropna().to_numpy()
                if len(vd): axs[0,0].hist(vd,bins=18,density=True,histtype="step",lw=1.4,label=r"support $d_i^F$")
                if len(va): axs[0,0].hist(va,bins=18,density=True,histtype="step",lw=1.4,label=r"flow mass $\alpha_i^F$")
                axs[0,0].set(xlabel="Strict local dimension",ylabel="Density"); axs[0,0].legend(frameon=False)
                dv=ln.delta_alpha_minus_support.dropna().to_numpy()
                if len(dv): axs[0,1].hist(dv,bins=18,density=True,histtype="step",lw=1.4)
                axs[0,1].axvline(0,ls="--",lw=.8,color=".4"); axs[0,1].set(xlabel=r"$\Delta_i^F=\alpha_i^F-d_i^F$",ylabel="Density")
            if len(qd):
                cm=plt.get_cmap("magma")
                for j,(frac,g) in enumerate(qd.groupby("N_B_over_N")):
                    x,y=ccdf_positive(g.normalized_directed_quotient_flow)
                    axs[1,0].loglog(x,y,"o-",ms=2,lw=.9,color=cm(j/max(qd.N_B_over_N.nunique()-1,1)),label=rf"$N_B/N={frac:.2f}$")
                axs[1,0].set(xlabel=r"$F'_{AB}/\langle F'\rangle$",ylabel="CCDF"); axs[1,0].legend(frameon=False)
            if len(rg):
                for q in RENYI_Q:
                    axs[1,1].semilogx(rg.ell_over_xi50,rg[f"eta_q{q:g}"],"o-",ms=2.5,label=rf"$q={q:g}$")
                axs[1,1].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"Coarse-flow effective fraction $\eta_q$"); axs[1,1].legend(frameon=False,ncol=2)
            _style_panels(axs); fig.suptitle(f"{aid} | local + renormalized flow",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"figure4_local_renormalization",pdf_targets); _maybe_show_or_close(fig)


        # ------------------------------------------------------------------
        # REFERENCE LOCAL-DIMENSION DISTRIBUTIONS — exact two-panel view
        # ------------------------------------------------------------------
        if len(ln):
            fig,axs=plt.subplots(1,2,figsize=(8.2,3.5))
            vd=ln.support_dimension.dropna().to_numpy(); va=ln.flow_mass_dimension.dropna().to_numpy()
            combined=np.concatenate([vd,va]) if len(vd) and len(va) else (vd if len(vd) else va)
            bins_dim=22
            if len(combined)>=4:
                lo,hi=np.percentile(combined,[0.5,99.5])
                bins_dim=np.linspace(lo,hi,23) if hi>lo else 22
            if len(vd): axs[0].hist(vd,bins=bins_dim,density=True,histtype='step',lw=1.5,color=LOCAL_SUPPORT_COLOR,label=r'support $d_i^F$')
            if len(va): axs[0].hist(va,bins=bins_dim,density=True,histtype='step',lw=1.5,color=LOCAL_ALPHA_COLOR,label=r'flow mass $\alpha_i^F$')
            axs[0].set(xlabel='Strictly validated local dimension',ylabel='Density'); axs[0].legend(frameon=False)
            dv=ln.delta_alpha_minus_support.dropna().to_numpy()
            if len(dv):
                lo,hi=np.percentile(dv,[0.5,99.5]) if len(dv)>=4 else (np.min(dv),np.max(dv))
                bins_delta=np.linspace(lo,hi,23) if np.isfinite(lo) and np.isfinite(hi) and hi>lo else 22
                axs[1].hist(dv,bins=bins_delta,density=True,histtype='stepfilled',alpha=.16,color=LOCAL_DELTA_COLOR)
                axs[1].hist(dv,bins=bins_delta,density=True,histtype='step',lw=1.4,color=LOCAL_DELTA_COLOR)
            axs[1].axvline(0,ls='--',lw=.8,color='.4'); axs[1].set(xlabel=r'$\Delta_i^F=\alpha_i^F-d_i^F$',ylabel='Density')
            _style_panels(axs); fig.suptitle(f'{aid} | strictly validated local support and flow-mass distributions',y=.995)
            fig.tight_layout(rect=(0,0,1,.97)); _save_arch_fig(fig,folder,'figure4A_local_dimension_distributions',pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # V5 FIGURE S1 — ROBUSTNESS
        # ------------------------------------------------------------------
        if len(mr) and len(mc) and len(cr):
            fig,axs=plt.subplots(2,2,figsize=(7.2,5.7))
            for pwr,g in mc.groupby("metric_power"):
                axs[0,0].loglog(g.ell_over_xi50,g.N_B_over_N,"o-",ms=2,lw=.9,label=rf"$p={pwr:g}$")
                good=np.isfinite(g.d_eff); axs[0,1].semilogx(g.loc[good,"ell_over_xi50"],g.loc[good,"d_eff"],"o-",ms=2,lw=.9,label=rf"$p={pwr:g}$")
            axs[0,0].set(xlabel=r"$\ell/\xi_{50}$",ylabel=r"$N_B/N$"); axs[0,0].legend(frameon=False)
            axs[0,1].set(xlabel=r"$\ell/\xi_{50}$",ylabel=r"$d_{\rm eff}^{F}$"); axs[0,1].legend(frameon=False)
            if len(cc):
                xg=np.geomspace(COMMON_X_MIN,COMMON_X_MAX,120); YY=[]
                for rep,g in cc.groupby("repeat"):
                    x=g.ell_over_xi50.to_numpy(float); y=g.N_B_over_N.to_numpy(float)
                    m=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
                    if np.sum(m)>=2:
                        o=np.argsort(x[m]); xx=x[m][o]; yy=np.log10(y[m][o])
                        YY.append(np.interp(np.log10(xg),np.log10(xx),yy,left=np.nan,right=np.nan))
                if YY:
                    YY=np.asarray(YY); med=np.nanmedian(YY,axis=0); lo=np.nanpercentile(YY,16,axis=0); hi=np.nanpercentile(YY,84,axis=0)
                    good=np.isfinite(med)
                    axs[1,0].fill_between(xg[good],10**lo[good],10**hi[good],alpha=.18)
                    axs[1,0].loglog(xg[good],10**med[good],lw=1.3,label="16–84% cover-order band")
                    axs[1,0].legend(frameon=False)
            axs[1,0].set(xlabel=r"$\ell/\xi_{50}$",ylabel=r"$N_B/N$")
            axs[1,1].scatter(cr.x_d_star,cr.d_star,c=cr.repeat,cmap="viridis",s=32)
            axs[1,1].set(xlabel=r"$x_d^*$ under order perturbation",ylabel=r"$d^*$")
            _style_panels(axs); fig.suptitle(f"{aid} | metric + cover robustness",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"figureS1_robustness",pdf_targets); _maybe_show_or_close(fig)


        # ------------------------------------------------------------------
        # REFERENCE ROBUSTNESS — full p family + canonical/perturbed cover view
        # ------------------------------------------------------------------
        if len(mr) and len(mc) and len(cr):
            fig,axs=plt.subplots(2,2,figsize=(8.4,6.2))
            p_order=sorted(mc.metric_power.unique())
            cm=plt.get_cmap('tab10')
            for j,pwr in enumerate(p_order):
                g=mc[mc.metric_power==pwr].sort_values('ell_over_xi50')
                axs[0,0].loglog(g.ell_over_xi50,g.N_B_over_N,'o-',ms=2,lw=.9,color=cm(j%10),label=rf'$p={pwr:g}$')
                good=np.isfinite(g.d_eff)
                axs[0,1].semilogx(g.loc[good,'ell_over_xi50'],g.loc[good,'d_eff'],'o-',ms=2,lw=.9,color=cm(j%10),label=rf'$p={pwr:g}$')
            axs[0,0].set(xlabel=r'$\ell_B/\xi_{50}$',ylabel=r'$N_B/N$'); axs[0,0].legend(frameon=False,ncol=2)
            axs[0,1].set(xlabel=r'$\ell_B/\xi_{50}$',ylabel=r'$d_{\rm eff}^{F}$'); axs[0,1].legend(frameon=False,ncol=2)

            # canonical p=1 curve plus perturbed-order median and 16-84% band
            if len(cc):
                xg=np.geomspace(max(COMMON_X_MIN,1e-3),max(COMMON_X_MAX,30),180); YY=[]
                for rep,g in cc.groupby('repeat'):
                    x=g.ell_over_xi50.to_numpy(float); y=g.N_B_over_N.to_numpy(float)
                    m=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
                    if np.sum(m)>=2:
                        xx=x[m]; yy=np.log10(y[m]); o=np.argsort(xx)
                        YY.append(np.interp(np.log10(xg),np.log10(xx[o]),yy[o],left=np.nan,right=np.nan))
                if YY:
                    YY=np.asarray(YY); med=np.nanmedian(YY,axis=0); lo=np.nanpercentile(YY,16,axis=0); hi=np.nanpercentile(YY,84,axis=0)
                    good=np.isfinite(med)
                    axs[1,0].fill_between(xg[good],10**lo[good],10**hi[good],alpha=.16,label='16–84% order band')
                    axs[1,0].loglog(xg[good],10**med[good],lw=1.3,label='Median perturbed cover')
                can=mc[np.isclose(mc.metric_power,1.0)].sort_values('ell_over_xi50')
                if len(can): axs[1,0].loglog(can.ell_over_xi50,can.N_B_over_N,'--',lw=1.3,color='.15',label='Canonical cover')
            axs[1,0].set(xlabel=r'$\ell_B/\xi_{50}$',ylabel=r'$N_B/N$'); axs[1,0].legend(frameon=False)
            axs[1,1].plot(cr.repeat,cr.x_d_star,'o',label=r'$x_d^*$')
            axs[1,1].plot(cr.repeat,cr.d_star,'s',label=r'$d^*$')
            axs[1,1].set(xlabel='Cover-order perturbation',ylabel='Crossover descriptor'); axs[1,1].legend(frameon=False)
            _style_panels(axs); fig.suptitle(f'{aid} | full metric-family and cover-order sensitivity',y=.995)
            fig.tight_layout(rect=(0,0,1,.97)); _save_arch_fig(fig,folder,'figureS1B_reference_metric_cover_robustness',pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 01 — TEMPORAL STRUCTURAL DASHBOARD
        # ------------------------------------------------------------------
        fig,axs=plt.subplots(3,2,figsize=(8.2,8.0))
        if len(h):
            axs[0,0].semilogy(h.epoch,h.train_mse_epoch,"-",label="Train")
            hv=h[np.isfinite(h.val_mse)]
            if len(hv): axs[0,0].semilogy(hv.epoch,hv.val_mse,"-",label="Validation")
        if len(gt): axs[0,0].semilogy(gt.epoch,gt.test_mse,"o-",label="Test")
        axs[0,0].set(xlabel="Epoch",ylabel="MSE"); axs[0,0].legend(frameon=False)
        axs[0,1].plot(t.epoch,t.eta1,"o-",label=r"$\eta_1$"); axs[0,1].plot(t.epoch,t.eta2,"s-",label=r"$\eta_2$")
        axs[0,1].set_yscale("log"); ax2=axs[0,1].twinx(); ax2.plot(t.epoch,t.entropy_deficit,"^-",alpha=.7,label="entropy deficit")
        axs[0,1].set(xlabel="Epoch",ylabel="Effective path fraction"); ax2.set_ylabel("Entropy deficit")
        lines=axs[0,1].get_lines()+ax2.get_lines(); axs[0,1].legend(lines,[z.get_label() for z in lines],frameon=False)
        axs[1,0].plot(t.epoch,t.xi50,"o-",label=r"$\xi_{50}$"); ax2=axs[1,0].twinx()
        ax2.plot(t.epoch,t.xi25_over_xi50,"s--",label=r"$\xi_{25}/\xi_{50}$"); ax2.plot(t.epoch,t.xi10_over_xi50,"^--",label=r"$\xi_{10}/\xi_{50}$")
        axs[1,0].set(xlabel="Epoch",ylabel=r"$\xi_{50}$"); ax2.set_ylabel("Covering-shape ratio")
        lines=axs[1,0].get_lines()+ax2.get_lines(); axs[1,0].legend(lines,[z.get_label() for z in lines],frameon=False)
        axs[1,1].plot(t.epoch,t.scaling_dB,"o-",label=r"$d_B^F$"); axs[1,1].plot(t.epoch,t.d_star,"s-",label=r"$d^*$")
        axs[1,1].set(xlabel="Epoch",ylabel="Dimension-like descriptor"); axs[1,1].legend(frameon=False)
        axs[2,0].plot(t.epoch,t.x_d_star,"o-",label=r"$x_d^*$")
        axs[2,0].plot(t.epoch,t.x_lacunarity_star,"s-",label=r"$x_\Lambda^*$")
        ratio=t.x_d_star/t.x_lacunarity_star
        axs[2,0].plot(t.epoch,ratio,"^-",label=r"$x_d^*/x_\Lambda^*$")
        axs[2,0].set(xlabel="Epoch",ylabel="Intrinsic crossover scale"); axs[2,0].legend(frameon=False)
        axs[2,1].plot(t.epoch,t.master_curve_error_xi50,"o-",label="master-curve RMS")
        if "quotient_flow_mean_pairwise_KS" in t: axs[2,1].plot(t.epoch,t.quotient_flow_mean_pairwise_KS,"s-",label="quotient-flow KS")
        axs[2,1].set(xlabel="Epoch",ylabel="Collapse / distribution error"); axs[2,1].legend(frameon=False)
        _style_panels(axs); fig.suptitle(f"{aid} | temporal structural dashboard",y=.995)
        fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"explore_01_temporal_dashboard",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 02 — MASTER CURVE + SCALE-TIME FIELDS
        # ------------------------------------------------------------------
        if len(master_df):
            eps,xg,Zd,Zlac,Znb=_arch_epoch_fields(s)
            fig,axs=plt.subplots(2,2,figsize=(8.2,6.3))
            for ep in selected:
                col=f"epoch_{int(ep)}_NB_over_N_interp"
                if col in master_df: axs[0,0].loglog(master_df.ell_over_xi50,master_df[col],lw=1,color=cmap(norm_ep(ep)),label=f"{int(ep)}")
            axs[0,0].loglog(master_df.ell_over_xi50,master_df.master_NB_over_N,"--",lw=1.6,color=".1",label="median master")
            axs[0,0].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$N_B/N$"); axs[0,0].legend(frameon=False,ncol=2)
            resid=[]
            for ep in eps:
                col=f"epoch_{int(ep)}_residual_log10"
                resid.append(master_df[col].to_numpy(float) if col in master_df else np.full(len(master_df),np.nan))
            resid=np.asarray(resid)
            lim=np.nanpercentile(np.abs(resid),97) if np.any(np.isfinite(resid)) else 1
            im=axs[0,1].pcolormesh(np.log10(master_df.ell_over_xi50),eps,resid,shading="auto",cmap="RdBu_r",vmin=-lim,vmax=lim)
            fig.colorbar(im,ax=axs[0,1],pad=.02,label=r"$\Delta\log_{10}(N_B/N)$")
            axs[0,1].set(xlabel=r"$\log_{10}x$",ylabel="Epoch")
            im=axs[1,0].pcolormesh(np.log10(xg),eps,Zd,shading="auto",cmap="viridis"); fig.colorbar(im,ax=axs[1,0],pad=.02,label=r"$d_{\rm eff}^{F}$")
            axs[1,0].set(xlabel=r"$\log_{10}x$",ylabel="Epoch")
            im=axs[1,1].pcolormesh(np.log10(xg),eps,Zlac,shading="auto",cmap="magma"); fig.colorbar(im,ax=axs[1,1],pad=.02,label=r"$\Lambda_F$")
            axs[1,1].set(xlabel=r"$\log_{10}x$",ylabel="Epoch")
            _style_panels(axs); fig.suptitle(f"{aid} | master curve and scale-time fields",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"explore_02_mastercurve_fields",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 03 — FINITE SCALING WINDOW VS GLOBAL CROSSOVER
        # ------------------------------------------------------------------
        fig,axs=plt.subplots(2,2,figsize=(8.2,6.2))
        axs[0,0].plot(t.epoch,t.scaling_dB,"o-",label=r"$d_B^F$"); ax2=axs[0,0].twinx(); ax2.plot(t.epoch,t.scaling_span,"s--",label="window span")
        axs[0,0].set(xlabel="Epoch",ylabel=r"$d_B^F$"); ax2.set_ylabel("Window span (decades)")
        lines=axs[0,0].get_lines()+ax2.get_lines(); axs[0,0].legend(lines,[z.get_label() for z in lines],frameon=False)
        if "transition_delta_bic" in t: axs[0,1].plot(t.epoch,t.transition_delta_bic,"o-",label=r"global $\Delta$BIC")
        axs[0,1].plot(t.epoch,t.scaling_delta_bic,"s-",label=r"window $\Delta$BIC"); axs[0,1].axhline(0,ls=":",lw=.8,color=".4"); axs[0,1].axhline(2,ls="--",lw=.8,color=".6")
        axs[0,1].set(xlabel="Epoch",ylabel=r"BIC$_{lin}$-BIC$_{quad}$"); axs[0,1].legend(frameon=False)
        axs[1,0].plot(t.epoch,t.d_star,"o-",label=r"$d^*$"); ax2=axs[1,0].twinx(); ax2.plot(t.epoch,t.NB_fraction_at_d_star,"s--",label=r"$(N_B/N)_{d^*}$")
        axs[1,0].set(xlabel="Epoch",ylabel=r"$d^*$"); ax2.set_ylabel(r"$N_B/N$ at crossover")
        lines=axs[1,0].get_lines()+ax2.get_lines(); axs[1,0].legend(lines,[z.get_label() for z in lines],frameon=False)
        sc=axs[1,1].scatter(t.scaling_dB,t.d_star,c=t.epoch,cmap="viridis",s=30); fig.colorbar(sc,ax=axs[1,1],pad=.02,label="Epoch")
        axs[1,1].set(xlabel=r"Finite-window exponent $d_B^F$",ylabel=r"Crossover peak $d^*$")
        _style_panels(axs); fig.suptitle(f"{aid} | scaling window versus global crossover",y=.995)
        fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"explore_03_scaling_crossover",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 04 — MICROSCOPIC FLOW BY LAYER
        # ------------------------------------------------------------------
        if micro is not None:
            fig,axs=plt.subplots(2,2,figsize=(8.2,6.3)); cm=plt.get_cmap("viridis")
            edge_keys=sorted([k for k in micro.files if k.startswith("edge_")],key=lambda z:int(z.split("_")[1]))
            stats=[]
            for j,k in enumerate(edge_keys):
                v=np.asarray(micro[k],float).ravel(); x,y=ccdf_positive(v)
                axs[0,0].loglog(x,y,lw=.9,color=cm(j/max(len(edge_keys)-1,1)),label=f"L{j+1}")
                topn=max(1,int(math.ceil(.1*len(v))))
                stats.append((j+1,np.std(v)/max(np.mean(v),EPS),gini_coefficient(v),np.sort(v)[-topn:].sum()/max(v.sum(),EPS)))
            axs[0,0].axvline(1,ls="--",lw=.8,color=".5"); axs[0,0].set(xlabel=r"Edge enrichment $\mathcal{F}_e$",ylabel="CCDF"); axs[0,0].legend(frameon=False,ncol=2)
            if stats:
                st=np.asarray(stats); axs[0,1].plot(st[:,0],st[:,1],"o-",label="CV"); axs[0,1].plot(st[:,0],st[:,2],"s-",label="Gini"); axs[0,1].plot(st[:,0],st[:,3],"^-",label="top-10% share")
                axs[0,1].set(xlabel="Edge layer",ylabel="Heterogeneity"); axs[0,1].legend(frameon=False)
            if len(tl):
                marker={"powerlaw":"o","lognormal":"s","exponential":"^","truncated_powerlaw":"D"}
                for model,g in tl.groupby("best_model"): axs[1,0].scatter(g.layer,g.powerlaw_alpha,s=38,marker=marker.get(model,"o"),label=model)
                axs[1,0].set(xlabel="Edge layer",ylabel=r"Diagnostic $\gamma_F$"); axs[1,0].legend(frameon=False)
                axs[1,1].axhline(0,ls=":",lw=.8,color=".4"); axs[1,1].bar(tl.layer,tl.delta_bic_powerlaw_vs_best_alt)
                for _,rr in tl.iterrows():
                    axs[1,1].text(rr.layer,rr.delta_bic_powerlaw_vs_best_alt,str(rr.best_model).replace("truncated_powerlaw","trunc. PL"),
                                  rotation=90,va="bottom",ha="center",fontsize=6)
                axs[1,1].set(xlabel="Edge layer",ylabel=r"BIC$_{PL}$-BIC$_{best alt}$")
            _style_panels(axs); fig.suptitle(f"{aid} | microscopic flow by layer",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"explore_04_microscopic_layers",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 05 — LOCAL DIMENSION GEOMETRY BY DEPTH
        # ------------------------------------------------------------------
        if len(ln):
            fig,axs=plt.subplots(2,2,figsize=(8.2,6.3))
            v=ln.dropna(subset=["support_dimension","flow_mass_dimension"])
            if len(v):
                sc=axs[0,0].scatter(v.support_dimension,v.flow_mass_dimension,c=v.z,cmap="viridis",s=18,alpha=.75)
                lo=min(v.support_dimension.min(),v.flow_mass_dimension.min()); hi=max(v.support_dimension.max(),v.flow_mass_dimension.max())
                axs[0,0].plot([lo,hi],[lo,hi],"--",lw=.8,color=".4"); fig.colorbar(sc,ax=axs[0,0],pad=.02,label="Normalized depth")
            axs[0,0].set(xlabel=r"Support dimension $d_i^F$",ylabel=r"Flow-mass dimension $\alpha_i^F$")
            layers=sorted(ln.node_layer.unique())
            dat=[ln.loc[ln.node_layer==l,"delta_alpha_minus_support"].dropna().values for l in layers]
            axs[0,1].boxplot(dat,positions=layers,widths=.55,showfliers=False); axs[0,1].axhline(0,ls="--",lw=.8,color=".5")
            axs[0,1].set(xlabel="Node layer",ylabel=r"$\Delta_i^F$")
            med=[];q1=[];q3=[];quality=[]
            for l in layers:
                g=ln[ln.node_layer==l]; d=g.delta_alpha_minus_support.dropna()
                med.append(d.median() if len(d) else np.nan); q1.append(d.quantile(.25) if len(d) else np.nan); q3.append(d.quantile(.75) if len(d) else np.nan)
                quality.append([g.support_dimension.notna().mean(),g.flow_mass_dimension.notna().mean(),g.delta_alpha_minus_support.notna().mean()])
            axs[1,0].plot(layers,med,"o-"); axs[1,0].fill_between(layers,q1,q3,alpha=.18); axs[1,0].axhline(0,ls="--",lw=.8,color=".5")
            axs[1,0].set(xlabel="Node layer",ylabel=r"Layerwise $\Delta_i^F$")
            q=np.asarray(quality,float); axs[1,1].plot(layers,q[:,0],"o-",label="support"); axs[1,1].plot(layers,q[:,1],"s-",label="flow mass"); axs[1,1].plot(layers,q[:,2],"^-",label="paired delta")
            axs[1,1].set_ylim(0,1.05); axs[1,1].set(xlabel="Node layer",ylabel="Valid-fit fraction"); axs[1,1].legend(frameon=False)
            _style_panels(axs); fig.suptitle(f"{aid} | local dimension geometry by depth",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"explore_05_local_dimensions",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 06 — DIRECTED RENORMALIZATION / RENYI
        # ------------------------------------------------------------------
        if len(qd) and len(rg):
            fig,axs=plt.subplots(2,2,figsize=(8.2,6.3)); cm=plt.get_cmap("magma")
            groups=[]; labels=[]
            for j,(frac,g) in enumerate(sorted(qd.groupby("N_B_over_N"),key=lambda z:z[0],reverse=True)):
                vals=g.normalized_directed_quotient_flow.to_numpy(float); x,y=ccdf_positive(vals)
                axs[0,0].loglog(x,y,lw=1,color=cm(j/max(qd.N_B_over_N.nunique()-1,1)),label=rf"$N_B/N={frac:.2f}$")
                groups.append(vals[np.isfinite(vals)&(vals>0)]); labels.append(f"{frac:.2f}")
            axs[0,0].set(xlabel=r"$F'_{AB}/\langle F'\rangle$",ylabel="CCDF"); axs[0,0].legend(frameon=False)
            ng=len(groups); KS=np.full((ng,ng),np.nan)
            for i in range(ng):
                for j in range(ng):
                    if len(groups[i]) and len(groups[j]): KS[i,j]=ks_2samp(groups[i],groups[j]).statistic
            im=axs[0,1].imshow(KS,origin="lower",cmap="viridis",vmin=0,vmax=max(.25,np.nanmax(KS)))
            axs[0,1].set_xticks(range(ng)); axs[0,1].set_yticks(range(ng)); axs[0,1].set_xticklabels(labels); axs[0,1].set_yticklabels(labels)
            for ii in range(ng):
                for jj in range(ng):
                    if np.isfinite(KS[ii,jj]): axs[0,1].text(jj,ii,f"{KS[ii,jj]:.2f}",ha="center",va="center",fontsize=7)
            axs[0,1].set(xlabel=r"$N_B/N$",ylabel=r"$N_B/N$"); fig.colorbar(im,ax=axs[0,1],pad=.02,label="KS distance")
            for q in RENYI_Q: axs[1,0].semilogx(rg.ell_over_xi50,rg[f"eta_q{q:g}"],"o-",ms=2.5,label=rf"$q={q:g}$")
            axs[1,0].set(xlabel=r"$x=\ell/\xi_{50}$",ylabel=r"$\eta_q$"); axs[1,0].legend(frameon=False,ncol=2)
            qs=np.asarray(RENYI_Q,float)
            for j,(frac,g) in enumerate(sorted(qd.groupby("N_B_over_N"),key=lambda z:z[0],reverse=True)):
                target=float(frac); row=rg.iloc[(rg.N_B_over_N-target).abs().argmin()]
                vals=[row[f"eta_q{q:g}"] for q in RENYI_Q]
                axs[1,1].plot(qs,vals,"o-",color=cm(j/max(ng-1,1)),label=rf"$N_B/N={frac:.2f}$")
            axs[1,1].set_yscale("log"); axs[1,1].set(xlabel=r"Rényi order $q$",ylabel=r"$\eta_q$"); axs[1,1].legend(frameon=False)
            _style_panels(axs); fig.suptitle(f"{aid} | directed renormalization",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"explore_06_renormalization",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 07 — COUPLED TEMPORAL PHASE SPACE
        # ------------------------------------------------------------------
        pairs=[("eta2","scaling_dB",r"$\eta_2$",r"$d_B^F$"),
               ("edge_cv","d_star","Edge-flow CV",r"$d^*$"),
               ("entropy_deficit","d_star","Entropy deficit",r"$d^*$"),
               ("master_curve_error_xi50","quotient_flow_mean_pairwise_KS","Master-curve error","Quotient-flow KS")]
        fig,axs=plt.subplots(2,2,figsize=(8.2,6.3))
        last_sc=None
        for ax,(xc,yc,xlab,ylab) in zip(axs.ravel(),pairs):
            sc=_phase_space_panel(ax,t,xc,yc,xlab,ylab)
            if sc is not None:
                last_sc=sc
        if last_sc is not None:
            fig.colorbar(last_sc,ax=axs.ravel().tolist(),pad=.015,label="Epoch")
        _style_panels(axs); fig.suptitle(f"{aid} | coupled temporal phase space",y=.995)
        fig.subplots_adjust(wspace=.30,hspace=.30,right=.88,top=.94)
        _save_arch_fig(fig,folder,"explore_07_phase_space",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 08 — ROBUSTNESS
        # ------------------------------------------------------------------
        if len(mr) and len(cr):
            fig,axs=plt.subplots(2,2,figsize=(8.2,6.2))
            axs[0,0].plot(mr.metric_power,mr.xi50,"o-"); axs[0,0].set(xlabel=r"Metric exponent $p$",ylabel=r"$\xi_{50}$")
            axs[0,1].plot(mr.metric_power,mr.x_d_star,"o-",label=r"$x_d^*$"); axs[0,1].plot(mr.metric_power,mr.x_lacunarity_star,"s-",label=r"$x_\Lambda^*$")
            axs[0,1].set(xlabel=r"Metric exponent $p$",ylabel="Intrinsic crossover scale"); axs[0,1].legend(frameon=False)
            axs[1,0].plot(mr.metric_power,mr.d_star,"o-",label=r"$d^*$"); axs[1,0].plot(mr.metric_power,mr.scaling_window_dB,"s-",label=r"$d_B^F$")
            axs[1,0].set(xlabel=r"Metric exponent $p$",ylabel="Dimension-like descriptor"); axs[1,0].legend(frameon=False)
            axs[1,1].scatter(cr.x_d_star,cr.d_star,c=cr.repeat,cmap="viridis",s=38); axs[1,1].set(xlabel=r"$x_d^*$ under cover perturbation",ylabel=r"$d^*$")
            _style_panels(axs); fig.suptitle(f"{aid} | robustness",y=.995)
            fig.tight_layout(rect=(0,0,1,.975)); _save_arch_fig(fig,folder,"explore_08_robustness",pdf_targets); _maybe_show_or_close(fig)

        # ------------------------------------------------------------------
        # EXPLORE 09 — TEMPORAL CORRELATION MATRIX
        # ------------------------------------------------------------------
        corr_candidates=[
            ("eta1",r"$\eta_1$"),("eta2",r"$\eta_2$"),("entropy_deficit","entropy deficit"),
            ("edge_cv","edge CV"),("node_cv","node CV"),("xi50",r"$\xi_{50}$"),
            ("xi25_over_xi50",r"$\xi_{25}/\xi_{50}$"),("xi10_over_xi50",r"$\xi_{10}/\xi_{50}$"),
            ("scaling_dB",r"$d_B^F$"),("d_star",r"$d^*$"),("x_d_star",r"$x_d^*$"),
            ("x_lacunarity_star",r"$x_\Lambda^*$"),("lacunarity_star",r"$\Lambda^*$"),
            ("quotient_flow_mean_pairwise_KS","quotient KS"),("master_curve_error_xi50","master error"),
            ("pooled_edge_tail_alpha",r"$\gamma_F$"),
        ]
        cols=[c for c,_ in corr_candidates if c in t.columns and np.sum(np.isfinite(t[c]))>=4]
        labs=[lab for c,lab in corr_candidates if c in cols]
        if len(cols)>=3:
            corr=t[cols].corr(method="spearman").to_numpy()
            fig,ax=plt.subplots(figsize=(8.3,7.2)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap="RdBu_r")
            ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs))); ax.set_xticklabels(labs,rotation=60,ha="right"); ax.set_yticklabels(labs)
            fig.colorbar(im,ax=ax,pad=.02,label="Spearman correlation"); ax.minorticks_off()
            fig.tight_layout(); _save_arch_fig(fig,folder,"explore_09_temporal_correlation_matrix",pdf_targets); _maybe_show_or_close(fig)


def plot_all_comprehensive_suites(train_df,temporal_df,scale_df,history_df,ext,stats_df):
    if len(temporal_df)==0 or len(scale_df)==0: return
    global_path=OUTDIR/FINAL_REPORT_FILENAME
    with PdfPages(global_path) as global_book:
        global_report_cover_pages(global_book,stats_df)
        entropy_ensemble=ext.get("entropy_ensemble",pd.DataFrame())
        local_density_ensemble=ext.get("local_density_ensemble",pd.DataFrame())
        plot_entropy_across_architectures(temporal_df,entropy_ensemble,pdf=global_book)
        plot_local_dimensions_across_architectures(
            ext.get("local_node",pd.DataFrame()),local_density_ensemble,pdf=global_book
        )
        for _,ar in tqdm(train_df.iterrows(),total=len(train_df),desc="All V5 + exploratory plot suites",unit="architecture"):
            aid=str(ar.arch_id)
            srow=(stats_df[stats_df.arch_id.astype(str)==aid].iloc[0].to_dict()
                  if len(stats_df) and np.any(stats_df.arch_id.astype(str)==aid) else None)
            plot_comprehensive_architecture_suite(
                aid,ar,temporal_df,scale_df,history_df,
                ext.get("local_node",pd.DataFrame()),ext.get("renorm_all",pd.DataFrame()),
                ext.get("quotient_detail",pd.DataFrame()),ext.get("tail_layer",pd.DataFrame()),
                ext.get("tail_pool",pd.DataFrame()),ext.get("metric",pd.DataFrame()),
                ext.get("metric_curves",pd.DataFrame()),ext.get("cover",pd.DataFrame()),
                ext.get("cover_curves",pd.DataFrame()),stat_row=srow,global_book=global_book
            )
    print(f"[exhaustive report] {global_path.resolve()}")



def run_experiment():
    total_start=time.perf_counter()
    splits,data_summary=make_environment()

    manifest={
        "scientific_question":"architecture-imposed vs training-deformed path-flow geometry",
        "run_profile":RUN_PROFILE,"seed":SEED,"model_init_seed":MODEL_INIT_SEED,
        "device":DEVICE,"dataset":data_summary,
        "configspace":{"depth":[DEPTH_MIN,DEPTH_MAX],"target_params":[PARAM_MIN,PARAM_MAX],
                       "morphologies":MORPHOLOGIES,"shape":[SHAPE_MIN,SHAPE_MAX],
                       "curvature":[CURVE_MIN,CURVE_MAX]},
        "map":{"sobol":MAP_SOBOL,"bo":MAP_BO,"smac_available":SMAC_AVAILABLE,
               "depth_bins":MAP_DEPTH_BINS,"param_bins":MAP_PARAM_BINS,"sobol_per_cell":MAP_SOBOL_PER_CELL,
               "random_probability":SMAC_RANDOM_PROBABILITY,"ei_xi":SMAC_EI_XI,
               "force_undercovered_every":BO_FORCE_UNDERCOVERED_EVERY,"cell_hard_slack":BO_CELL_HARD_SLACK},
        "architecture_validity":{"min_median_width":ARCH_MIN_MEDIAN_WIDTH,"min_q10_width":ARCH_MIN_Q10_WIDTH,
               "max_floor_fraction":ARCH_MAX_FLOOR_FRACTION,"max_ceiling_fraction":ARCH_MAX_CEILING_FRACTION,
               "param_ratio":[ARCH_PARAM_RATIO_MIN,ARCH_PARAM_RATIO_MAX]},
        "training":{"start":TRAIN_START,"budgets":BUDGETS,"promotion_fractions":PROMOTION_FRACTIONS,
                    "optimizer":"Adam","lr":LR,"batch_size":BATCH_SIZE,"weight_decay":WEIGHT_DECAY,
                    "longitudinal_cohort":LONGITUDINAL_COHORT,"require_exhaustive_per_arch":REQUIRE_EXHAUSTIVE_PER_ARCH,
                    "geometry_checkpoints":GEOMETRY_CHECKPOINTS,
                    "longitudinal_bcanw_scales":LONGITUDINAL_BCANW_SCALES},
    }
    (OUTDIR/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    # ------------------------------------------------------------------
    # MAP A1: Sobol cold start
    # ------------------------------------------------------------------
    banner("STAGE 1 — MAP A1: SOBOL COLD-START ARCHITECTURE PHENOTYPE MAP")
    cs=build_configspace()
    configs=generate_sobol_configs(MAP_SOBOL,cs)
    MAP_SOBOL_REALIZED=len(configs)
    rows=[]; seen_widths={}
    eta=ETAReporter("dense stratified Sobol initialization map",len(configs))
    bar=tqdm(total=len(configs),desc="Map A1 — Sobol geometries",unit="architecture")
    for i,cfg in enumerate(configs,1):
        t=time.perf_counter()
        arch=architecture_from_config(cfg); sig=tuple(arch["hidden_widths"])
        if sig in seen_widths:
            # Same realized architecture: reuse geometry but retain distinct config metadata.
            base=dict(seen_widths[sig]); base.update({"stage":"sobol_duplicate","map_index":i,
                "config_key":config_key_dict(cfg),"config_json":json.dumps(cfg,sort_keys=True)})
            row=base
            tqdm.write(f"[Sobol #{i:03d}] duplicate realized widths -> reused {row['arch_id']}")
        else:
            row=evaluate_initial_architecture(cfg,"sobol",i); seen_widths[sig]=dict(row)
        rows.append(row); bar.update(1); eta.update(time.perf_counter()-t,i)
        atomic_csv(pd.DataFrame(rows),OUTDIR/"architecture_map_partial.csv")
    bar.close(); eta.finish()
    sobol_df=pd.DataFrame(rows)

    # Fit fixed cold-start phenotype coordinate system.
    fps=FingerprintSpace(); Z0=fps.fit(sobol_df)
    novelty=[]
    for i,z in enumerate(Z0): novelty.append(fps.novelty(z,exclude_index=i))
    sobol_df["structural_novelty"]=novelty
    cold_records=sobol_df.to_dict("records")
    map_scores=[]; cfg_novs=[]; cov_bonuses=[]
    for i,r in enumerate(cold_records):
        cfg=json.loads(r["config_json"]); prior=cold_records[:i]+cold_records[i+1:]
        score,cn,cb=combined_map_score(r["structural_novelty"],cfg,prior)
        map_scores.append(score); cfg_novs.append(cn); cov_bonuses.append(cb)
    sobol_df["config_space_novelty"]=cfg_novs
    sobol_df["coverage_bonus"]=cov_bonuses
    sobol_df["map_discovery_score"]=map_scores
    sobol_df["smac_cost"]=1.0-np.asarray(map_scores)
    rows=sobol_df.to_dict("records")
    atomic_csv(pd.DataFrame(rows),OUTDIR/"architecture_map_partial.csv")

    print("\n[cold-start phenotype cloud]")
    print(f"  architectures: {len(sobol_df)}")
    print(f"  novelty: mean={np.mean(novelty):.3f}, range=[{np.min(novelty):.3f}, {np.max(novelty):.3f}]")
    print(f"  dB: median={np.nanmedian(sobol_df.scaling_dB):.3f}")
    print(f"  xi25/xi50: median={np.nanmedian(sobol_df.xi25_over_xi50):.3f}")
    cold_counts=coverage_counts_from_rows(rows)
    cold_cov=pd.DataFrame([{"morphology":c[0],"depth_bin":c[1],"param_bin":c[2],"count":v}
                           for c,v in cold_counts.items()])
    atomic_csv(cold_cov,OUTDIR/"architecture_search_coverage_coldstart.csv")
    print(f"  configuration-space cells covered: {(cold_cov['count']>0).sum()}/{len(cold_cov)}; "
          f"cell count range={cold_cov['count'].min()}-{cold_cov['count'].max()}")

    # ------------------------------------------------------------------
    # MAP A2: SMAC BO extension of initialization map
    # ------------------------------------------------------------------
    banner("STAGE 2 — MAP A2: BAYESIAN EXPLORATION OF STRUCTURAL NOVELTY")
    print("Acquisition logic:")
    print("  scientific objective = structural novelty + configuration-space novelty + coverage; SMAC minimizes 1-score")
    print(f"  EI exploration xi = {SMAC_EI_XI}; random interleaving = {SMAC_RANDOM_PROBABILITY:.0%}")
    print("  phenotype novelty uses the standardized V5 fingerprint; a hard cell-density guard prevents local collapse")

    smac_engine=None
    if SMAC_AVAILABLE and cs is not None:
        from ConfigSpace import Configuration
        from smac import Scenario, HyperparameterOptimizationFacade
        from smac.runhistory.dataclasses import TrialValue
        scenario=Scenario(cs,deterministic=True,n_trials=MAP_SOBOL+MAP_BO*SMAC_TRIAL_BUDGET_MULTIPLIER,seed=SEED,
                          output_directory=OUTDIR/"smac"/"architecture_novelty")
        initial_configs=[Configuration(cs,values=json.loads(r["config_json"])) for r in rows[:MAP_SOBOL]]
        initial_design=HyperparameterOptimizationFacade.get_initial_design(
            scenario,n_configs=0,max_ratio=1.0,additional_configs=initial_configs
        )
        random_design=HyperparameterOptimizationFacade.get_random_design(
            scenario,probability=SMAC_RANDOM_PROBABILITY
        )
        acquisition=HyperparameterOptimizationFacade.get_acquisition_function(scenario,xi=SMAC_EI_XI)
        model_rf=HyperparameterOptimizationFacade.get_model(
            scenario,n_trees=SMAC_RF_TREES,min_samples_leaf=2,bootstrapping=True
        )
        intensifier=HyperparameterOptimizationFacade.get_intensifier(scenario,max_config_calls=1)
        smac_engine=HyperparameterOptimizationFacade(
            scenario,lambda config,seed=0:1.0,model=model_rf,acquisition_function=acquisition,
            initial_design=initial_design,random_design=random_design,intensifier=intensifier,
            overwrite=True,logging_level=False
        )
        # Replay already-computed cold-start evaluations into SMAC.
        lookup={r["config_key"]:r for r in rows[:MAP_SOBOL]}
        for _ in range(MAP_SOBOL):
            info=smac_engine.ask(); d=config_obj_to_dict(info.config); key=config_key_dict(d)
            if key not in lookup:
                # Defensive fallback: evaluate if SMAC canonicalization changed a config.
                rr=evaluate_initial_architecture(d,"sobol_replay",len(rows)+1)
                z=fps.transform_row(rr); nov=fps.novelty(z); rr["structural_novelty"]=nov; score,cn,cb=combined_map_score(nov,d,rows); rr["config_space_novelty"]=cn; rr["coverage_bonus"]=cb; rr["map_discovery_score"]=score; rr["smac_cost"]=1-score
                rows.append(rr); lookup[key]=rr
            rr=lookup[key]
            smac_engine.tell(info,TrialValue(cost=float(rr["smac_cost"]),time=float(rr.get("total_eval_seconds",0))))
        print("[SMAC] cold-start observations replayed; RF+EI surrogate is now active.")
    else:
        print("[fallback BO] sklearn RandomForest lower-confidence-bound explorer is active.")

    eta=ETAReporter("diversity-constrained Bayesian architecture map extension",MAP_BO)
    bar=tqdm(total=MAP_BO,desc="Map A2 — diversity-constrained BO",unit="accepted architecture")
    accepted_bo=0; rejected_proposals=0
    while accepted_bo < MAP_BO:
        j=accepted_bo+1
        t=time.perf_counter()
        force_undercovered=(j % BO_FORCE_UNDERCOVERED_EVERY == 0)
        accepted_cfg=None; info=None; origin=None

        for reject_try in range(BO_MAX_REJECTIONS_PER_ACCEPTED):
            if smac_engine is not None:
                info=smac_engine.ask(); cfg=config_obj_to_dict(info.config)
                origin=getattr(info.config,"origin",None)
            else:
                cfg=fallback_propose(pd.DataFrame(rows)); info=None; origin="fallback_RF_diversity_LCB"

            arch=architecture_from_config(cfg); ok,reasons=architecture_feasibility(arch)
            sig=tuple(arch["hidden_widths"])
            counts=coverage_counts_from_rows(rows)
            min_count=min(counts.values()) if counts else 0
            cell=coverage_cell_from_config(cfg); cell_count=counts.get(cell,0)
            over_hard=cell_count > min_count + BO_CELL_HARD_SLACK
            over_forced=force_undercovered and cell_count > min_count
            duplicate=sig in seen_widths

            if ok and (not duplicate) and (not over_hard) and (not over_forced):
                accepted_cfg=cfg; break

            rejected_proposals += 1
            reason = "duplicate" if duplicate else (";".join(reasons) if not ok else
                     ("forced_undercovered_cell" if over_forced else "overrepresented_cell"))
            if smac_engine is not None:
                from smac.runhistory.dataclasses import TrialValue
                # Synthetic penalty is deliberate: it teaches the surrogate that clipped/overdense regions
                # are scientifically undesirable without spending BCANw time on them.
                smac_engine.tell(info,TrialValue(cost=1.25,time=0.0))
            if reject_try in (0, 7, 23):
                tqdm.write(f"[BO guard] rejected proposal ({reason}); cell={cell}, count={cell_count}, min={min_count}")

        if accepted_cfg is None:
            counts=coverage_counts_from_rows(rows); min_count=min(counts.values())
            cells=[c for c,v in counts.items() if v==min_count]
            cfg=sample_feasible_config_in_cell(random.choice(cells),SEED+accepted_bo*10007,set(seen_widths.keys()))
            origin="forced_least_covered_fallback"; info=None
        else:
            cfg=accepted_cfg

        rr=evaluate_initial_architecture(cfg,"bo",MAP_SOBOL+j)
        arch=architecture_from_config(cfg); sig=tuple(arch["hidden_widths"]); seen_widths[sig]=dict(rr)
        z=fps.transform_row(rr); nov=fps.novelty(z)
        score,cn,cb=combined_map_score(nov,cfg,rows)
        rr["structural_novelty"]=nov
        rr["config_space_novelty"]=cn
        rr["coverage_bonus"]=cb
        rr["map_discovery_score"]=score
        rr["smac_cost"]=float(1.0-score)
        rr["proposal_origin"]=origin
        rr["coverage_cell"]=str(coverage_cell_from_config(cfg))
        rows.append(rr)
        if smac_engine is not None and info is not None:
            from smac.runhistory.dataclasses import TrialValue
            smac_engine.tell(info,TrialValue(cost=float(rr["smac_cost"]),time=float(rr.get("total_eval_seconds",0))))
        accepted_bo += 1
        bar.update(1); eta.update(time.perf_counter()-t,accepted_bo)
        atomic_csv(pd.DataFrame(rows),OUTDIR/"architecture_map_partial.csv")
    bar.close(); eta.finish()
    print(f"[BO coverage guard] accepted={accepted_bo}; rejected/penalized proposals={rejected_proposals}")

    map_df=pd.DataFrame(rows).drop_duplicates(subset=["arch_id"],keep="first").reset_index(drop=True)
    Z=fps.transform_df(map_df)
    map_df["structural_novelty_fixed_reference"]=[fps.novelty(z) for z in Z]
    final_counts=coverage_counts_from_rows(map_df.to_dict("records"))
    final_cov=pd.DataFrame([{"morphology":c[0],"depth_bin":c[1],"param_bin":c[2],"count":v}
                            for c,v in final_counts.items()])
    atomic_csv(final_cov,OUTDIR/"architecture_search_coverage_final.csv")
    print(f"[final map coverage] cells={(final_cov['count']>0).sum()}/{len(final_cov)} | "
          f"count range={final_cov['count'].min()}-{final_cov['count'].max()} | "
          f"unique realized architectures={len(map_df)}")

    # Unsupervised initial universality-class suggestion: choose K by silhouette.
    best_k=None; best_score=-np.inf; best_labels=None
    for k in range(2,min(8,len(map_df)-1)+1):
        km=KMeans(n_clusters=k,random_state=SEED,n_init=20).fit(Z)
        sc=silhouette_score(Z,km.labels_)
        if sc>best_score: best_score=sc; best_k=k; best_labels=km.labels_
    if best_labels is None:
        best_labels=np.zeros(len(map_df),int); best_k=1; best_score=np.nan
    map_df["init_structural_cluster"]=best_labels
    atomic_csv(map_df,OUTDIR/"architecture_map.csv")
    print(f"[initial phenotype clustering] selected k={best_k} by silhouette={best_score:.3f}")

    # ------------------------------------------------------------------
    # MAP B selection: farthest-point structural coverage
    # ------------------------------------------------------------------
    banner("STAGE 3 — SELECTING STRUCTURALLY DIVERSE ARCHITECTURES FOR TRAINING")
    nselect=min(TRAIN_START,len(map_df))
    center=Z.mean(axis=0)
    first=int(np.argmin(np.linalg.norm(Z-center[None,:],axis=1)))
    selected=[first]
    # Add most structurally extreme distinct point.
    if nselect>1:
        ranked_extreme=list(np.argsort(-np.linalg.norm(Z-center[None,:],axis=1)))
        for cand in ranked_extreme:
            if int(cand) not in selected:
                selected.append(int(cand)); break
    while len(selected)<nselect:
        remain=[i for i in range(len(Z)) if i not in selected]
        mind=[]
        for i in remain:
            mind.append(min(np.linalg.norm(Z[i]-Z[j]) for j in selected))
        selected.append(remain[int(np.argmax(mind))])
    train_df=map_df.iloc[selected].copy().reset_index(drop=True)
    if RUN_PROFILE == "research" and (train_df.depth <= 8).any():
        raise RuntimeError("Research architecture cohort must contain only >8 hidden-layer configurations.")
    train_df["selected_for_training"]=True
    atomic_csv(train_df,OUTDIR/"selected_training_configs.csv")
    print(train_df[["arch_id","morphology","depth","actual_params","init_structural_cluster",
                    "structural_novelty_fixed_reference"]].to_string(index=False))

    # Fixed longitudinal cohort: held constant through all epochs to avoid survivor-bias
    # in temporal variance and scale-time plots.
    if REQUIRE_EXHAUSTIVE_PER_ARCH:
        # Exhaustive reporting mode: every selected architecture is followed at the same
        # temporal checkpoints and receives the complete final diagnostic/plot suite.
        cohort_idx=list(range(len(train_df)))
        longitudinal_df=train_df.copy().reset_index(drop=True)
    else:
        cohort_idx=farthest_point_subset(train_df,fps,min(LONGITUDINAL_COHORT,len(train_df)))
        longitudinal_df=train_df.iloc[cohort_idx].copy().reset_index(drop=True)
    longitudinal_df["longitudinal_cohort"]=True
    cohort_ids=set(map(str,longitudinal_df.arch_id))
    funnel_df=train_df[~train_df.arch_id.astype(str).isin(cohort_ids)].copy().reset_index(drop=True)
    atomic_csv(longitudinal_df,OUTDIR/"longitudinal_cohort.csv")
    atomic_csv(funnel_df,OUTDIR/"funnel_training_configs.csv")

    print("\n[fixed longitudinal cohort]")
    print(longitudinal_df[["arch_id","morphology","depth","actual_params","init_structural_cluster"]].to_string(index=False))
    print(f"Longitudinal cohort = {len(longitudinal_df)} architectures, all trained to epoch {max(BUDGETS)}.")
    print(f"Funnel cohort = {len(funnel_df)} architectures, subject to conservative promotion.")
    print("IMPORTANT: scale-by-epoch means/variance are computed ONLY from the fixed longitudinal cohort.")

    # PCA coordinates for all initial phenotypes; used for trajectories later.
    pca=PCA(n_components=min(3,Z.shape[1],len(Z))).fit(Z)
    Zp=pca.transform(Z)
    for j in range(Zp.shape[1]): map_df[f"pca{j+1}"]=Zp[:,j]
    atomic_csv(map_df,OUTDIR/"architecture_map.csv")




    # ------------------------------------------------------------------
    # MAP B1: fixed longitudinal cohort — full temporal geometry
    # ------------------------------------------------------------------
    banner("STAGE 4A — FIXED LONGITUDINAL COHORT: TEMPORAL ORGANIZATION WITHOUT SURVIVOR BIAS")
    print(f"Geometry checkpoints: {GEOMETRY_CHECKPOINTS}")
    print(f"Every cohort architecture is trained to epoch {max(BUDGETS)}.")
    print("At each checkpoint we compute the SAME path-space, BCANw covering, d_eff(x), and lacunarity(x).")
    print("All cross-epoch variance plots later use this unchanged architecture set.")

    long_temporal_frames=[]; long_scale_frames=[]; long_layer_frames=[]; long_history_frames=[]
    long_local_frames=[]; long_renorm_frames=[]; long_final_rows=[]

    geom_checks_per_arch=len(sorted(set([e for e in GEOMETRY_CHECKPOINTS if 0<=e<=max(BUDGETS)] + [0,max(BUDGETS)])))
    total_geom_checks=max(len(longitudinal_df)*geom_checks_per_arch,1)
    geom_bar=tqdm(total=total_geom_checks,desc="Longitudinal geometry snapshots",unit="snapshot")
    geom_eta=ETAReporter("longitudinal geometry snapshots",total_geom_checks)
    geom_counter=[0]
    long_epoch_bar=tqdm(total=max(len(longitudinal_df)*max(BUDGETS),1),
                        desc="Longitudinal training work",unit="arch-epoch")
    long_arch_eta=ETAReporter("fixed longitudinal architectures",max(len(longitudinal_df),1))

    for i,(_,ar) in enumerate(longitudinal_df.iterrows(),1):
        t=time.perf_counter()
        tdf_i,sdf_i,ldf_i,hdf_i,local_i,renorm_i,final_i=train_longitudinal_architecture(
            ar,splits,fps,geometry_bar=geom_bar,geometry_eta=geom_eta,
            geometry_counter=geom_counter,epoch_bar=long_epoch_bar
        )
        long_temporal_frames.append(tdf_i); long_scale_frames.append(sdf_i)
        long_layer_frames.append(ldf_i); long_history_frames.append(hdf_i)
        if local_i is not None: long_local_frames.append(local_i)
        if renorm_i is not None: long_renorm_frames.append(renorm_i)
        if final_i is not None: long_final_rows.append(final_i)

        # Flush partial longitudinal archive after every architecture.
        atomic_csv(pd.concat(long_temporal_frames,ignore_index=True),OUTDIR/"longitudinal_temporal_summary_partial.csv")
        atomic_csv(pd.concat(long_scale_frames,ignore_index=True),OUTDIR/"longitudinal_scale_profiles_partial.csv")
        atomic_csv(pd.concat(long_layer_frames,ignore_index=True),OUTDIR/"longitudinal_layer_flow_partial.csv")
        atomic_csv(pd.concat(long_history_frames,ignore_index=True),OUTDIR/"longitudinal_training_history_partial.csv")
        long_arch_eta.update(time.perf_counter()-t,i)

    geom_bar.close(); long_epoch_bar.close(); geom_eta.finish(); long_arch_eta.finish()

    long_tdf=pd.concat(long_temporal_frames,ignore_index=True) if long_temporal_frames else pd.DataFrame()
    long_sdf=pd.concat(long_scale_frames,ignore_index=True) if long_scale_frames else pd.DataFrame()
    long_layer_df=pd.concat(long_layer_frames,ignore_index=True) if long_layer_frames else pd.DataFrame()
    long_hist_df=pd.concat(long_history_frames,ignore_index=True) if long_history_frames else pd.DataFrame()
    long_local_df=pd.concat(long_local_frames,ignore_index=True) if long_local_frames else pd.DataFrame()
    long_renorm_df=pd.concat(long_renorm_frames,ignore_index=True) if long_renorm_frames else pd.DataFrame()
    long_final_df=pd.DataFrame(long_final_rows)

    atomic_csv(long_tdf,OUTDIR/"longitudinal_temporal_summary.csv")
    atomic_csv(long_sdf,OUTDIR/"longitudinal_scale_profiles.csv")
    atomic_csv(long_layer_df,OUTDIR/"longitudinal_layer_flow.csv")
    atomic_csv(long_hist_df,OUTDIR/"longitudinal_training_history.csv")
    if len(long_local_df): atomic_csv(long_local_df,OUTDIR/"longitudinal_local_depth_profiles.csv")
    if len(long_renorm_df): atomic_csv(long_renorm_df,OUTDIR/"longitudinal_renormalization_profiles.csv")
    if len(long_final_df): atomic_csv(long_final_df,OUTDIR/"longitudinal_final_fingerprints.csv")

    ensemble_df=interpolate_scale_ensemble(long_sdf)
    if len(ensemble_df): atomic_csv(ensemble_df,OUTDIR/"longitudinal_scale_ensemble.csv")

    # ------------------------------------------------------------------
    # MAP B2: conservative multi-fidelity training for the non-longitudinal architectures
    # ------------------------------------------------------------------
    banner("STAGE 4B — CONSERVATIVE MULTI-FIDELITY STRUCTURAL DEFORMATION MAP")
    print(f"Fixed exposure: Adam(lr={LR:g}, weight_decay={WEIGHT_DECAY:g}), batch={BATCH_SIZE}")
    print(f"Budgets: {BUDGETS}; random rescue fraction at each promotion = {RANDOM_RESCUE_FRACTION:.0%}")
    print("Promotion score = 0.35 novelty + 0.30 deformation + 0.20 structural velocity + 0.15 functional gain")
    print("The fixed longitudinal cohort is excluded from racing because it is already fully observed.")

    counts=[len(funnel_df)]
    for f in PROMOTION_FRACTIONS[:len(BUDGETS)-1]:
        counts.append(max(1,int(round(counts[-1]*f))) if counts[-1]>0 else 0)
    counts=counts[:len(BUDGETS)]
    prev=0; total_epoch_work=0
    for c,b in zip(counts,BUDGETS):
        total_epoch_work += c*(b-prev); prev=b
    global_epoch_bar=tqdm(total=max(total_epoch_work,1),desc="Funnel training work",unit="arch-epoch")

    active=funnel_df.copy()
    rung_rows=[]; history_rows=[]; previous_z={}; previous_epoch={}
    final_aux_cache={}
    rng_rescue=np.random.default_rng(SEED+99)

    for rung,target_epoch in enumerate(BUDGETS):
        if len(active)==0:
            break
        banner(f"TRAINING RUNG {rung+1}/{len(BUDGETS)} — TARGET EPOCH {target_epoch} — {len(active)} ARCHITECTURES")
        rung_timer=ETAReporter(f"funnel training rung {target_epoch}",len(active))
        current_rows=[]
        for ii,(_,arch_row) in enumerate(active.iterrows(),1):
            t=time.perf_counter(); aid=str(arch_row.arch_id)
            if aid not in previous_epoch:
                base_model=BiasFreeReLUMLP(INPUT_DIM,json.loads(arch_row.hidden_widths),1,seed=MODEL_INIT_SEED).to(DEVICE)
                val0=eval_mse(base_model,*splits["val"]); test0=eval_mse(base_model,*splits["test"])
                del base_model
            else:
                prior=[r for r in rung_rows if r["arch_id"]==aid]
                val0=float(prior[0]["val_mse_init"]); test0=float(prior[0]["test_mse_init"])

            model,hist=train_to_budget(arch_row,target_epoch,splits,epoch_progress=global_epoch_bar)
            for h in hist:
                h["cohort"]="funnel"; h["morphology"]=arch_row.morphology
            history_rows.extend(hist)
            train_mse=eval_mse(model,*splits["train"])
            val_mse=eval_mse(model,*splits["val"])
            test_mse=eval_mse(model,*splits["test"])
            weights=extract_weights(model)
            curve_path=OUTDIR/"curves"/f"{aid}_epoch{target_epoch}.npz"
            geom,aux=analyze_core_geometry(weights,TRAIN_BCANW_SCALES,save_curve=curve_path)
            z=fps.transform_row(geom)
            novelty=fps.novelty(z)
            z0=fps.transform_row(arch_row.to_dict())
            deformation=float(np.linalg.norm(z-z0))
            if aid in previous_z:
                velocity=float(np.linalg.norm(z-previous_z[aid])/max(target_epoch-previous_epoch[aid],1))
            else:
                velocity=float(deformation/max(target_epoch,1))
            functional_gain=float((val0-val_mse)/max(abs(val0),EPS))

            rr={
                "arch_id":aid,"cohort":"funnel","rung":rung,"epoch":target_epoch,
                "morphology":arch_row.morphology,"depth":arch_row.depth,"actual_params":arch_row.actual_params,
                "init_structural_cluster":arch_row.init_structural_cluster,
                "val_mse_init":val0,"test_mse_init":test0,
                "train_mse":train_mse,"val_mse":val_mse,"test_mse":test_mse,
                "functional_gain":functional_gain,"structural_novelty":novelty,
                "structural_deformation":deformation,"structural_velocity":velocity,
                **geom,
            }
            current_rows.append(rr); rung_rows.append(rr)
            previous_z[aid]=z; previous_epoch[aid]=target_epoch
            if rung==len(BUDGETS)-1:
                final_aux_cache[aid]=(aux,json.loads(arch_row.hidden_widths))

            elapsed=time.perf_counter()-t
            tqdm.write(
                f"[funnel rung {target_epoch:3d}] {aid} | train={train_mse:.4f} val={val_mse:.4f} test={test_mse:.4f} | "
                f"gain={functional_gain:+.3f} deform={deformation:.3f} vel={velocity:.4f} novelty={novelty:.3f} | "
                f"dB={geom['scaling_dB']:.3f} peaks={geom['n_crossover_peaks']} | {human_time(elapsed)}"
            )
            rung_timer.update(elapsed,ii)
            atomic_csv(pd.DataFrame(rung_rows),OUTDIR/"training_rungs_partial.csv")
            atomic_csv(pd.DataFrame(history_rows),OUTDIR/"training_history_partial.csv")
            del model,weights
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        rung_timer.finish()

        cur=pd.DataFrame(current_rows)
        for col in ["structural_novelty","structural_deformation","structural_velocity","functional_gain"]:
            cur[f"rank_{col}"]=cur[col].rank(pct=True,method="average")
        cur["promotion_score"]=(.35*cur.rank_structural_novelty + .30*cur.rank_structural_deformation +
                                .20*cur.rank_structural_velocity + .15*cur.rank_functional_gain)
        score_lookup=dict(zip(cur.arch_id,cur.promotion_score))
        for r in rung_rows:
            if r["epoch"]==target_epoch and r["arch_id"] in score_lookup:
                r["promotion_score"]=float(score_lookup[r["arch_id"]])

        if rung < len(BUDGETS)-1:
            frac=PROMOTION_FRACTIONS[rung] if rung<len(PROMOTION_FRACTIONS) else .5
            n_next=max(1,int(round(len(cur)*frac)))
            n_rescue=min(max(1,int(round(len(cur)*RANDOM_RESCUE_FRACTION))),max(n_next-1,0)) if n_next>1 else 0
            top_n=n_next-n_rescue
            ranked=cur.sort_values("promotion_score",ascending=False)
            keep=list(ranked.head(top_n).arch_id)
            remaining=[x for x in ranked.arch_id if x not in keep]
            if n_rescue and remaining:
                rescue=list(rng_rescue.choice(remaining,size=min(n_rescue,len(remaining)),replace=False))
                keep+=rescue
            active=funnel_df[funnel_df.arch_id.isin(keep)].copy()
            print("\n[promotion decision — funnel only]")
            print(cur[["arch_id","promotion_score","structural_novelty","structural_deformation",
                       "structural_velocity","functional_gain","val_mse"]].sort_values(
                           "promotion_score",ascending=False).to_string(index=False))
            print(f"Promoted {len(active)}/{len(cur)} -> epoch {BUDGETS[rung+1]}: {list(active.arch_id)}")

    global_epoch_bar.close()
    rung_df=pd.DataFrame(rung_rows)
    hist_df=pd.DataFrame(history_rows)
    atomic_csv(rung_df,OUTDIR/"training_rungs.csv")
    atomic_csv(hist_df,OUTDIR/"training_history.csv")

    # ------------------------------------------------------------------
    # Final diagnostics for funnel full-budget survivors.
    # Fixed-cohort final diagnostics were already computed during longitudinal tracking.
    # ------------------------------------------------------------------
    banner("STAGE 5 — FINAL LOCAL / RENORMALIZATION DIAGNOSTICS")
    final_epoch=int(max(BUDGETS))
    funnel_survivors=list(rung_df.loc[rung_df.epoch==final_epoch,"arch_id"].unique()) if len(rung_df) else []
    local_frames=[]; renorm_frames=[]; final_rows=[]
    eta=ETAReporter("funnel full final diagnostics",max(len(funnel_survivors),1))
    for i,aid in enumerate(tqdm(funnel_survivors,desc="Funnel final diagnostics",unit="architecture"),1):
        t=time.perf_counter()
        row=rung_df[(rung_df.arch_id==aid)&(rung_df.epoch==final_epoch)].iloc[-1].to_dict()
        if aid in final_aux_cache:
            aux,widths=final_aux_cache[aid]
        else:
            ar=funnel_df[funnel_df.arch_id==aid].iloc[0]
            model=BiasFreeReLUMLP(INPUT_DIM,json.loads(ar.hidden_widths),1,seed=MODEL_INIT_SEED).to(DEVICE)
            state=torch.load(checkpoint_path(aid),map_location="cpu",weights_only=False)
            model.load_state_dict(state["model"])
            _,aux=analyze_core_geometry(extract_weights(model),TRAIN_BCANW_SCALES)
            widths=json.loads(ar.hidden_widths)
        save_microscopic_snapshot(aid,final_epoch,aux)
        allwidths=[INPUT_DIM]+list(widths)+[1]
        ldf,delta_grid,rdf,ks=full_final_diagnostics(aux,allwidths,aid)
        ldf["epoch"]=final_epoch; ldf["morphology"]=row["morphology"]
        rdf["epoch"]=final_epoch; rdf["morphology"]=row["morphology"]
        local_frames.append(ldf); renorm_frames.append(rdf)
        for z,val in zip(LOCAL_DEPTH_GRID,delta_grid): row[f"delta_z{z:.2f}"]=val
        row["renorm_mean_pairwise_KS"]=ks
        row["renorm_flow_retention_mean"]=float(rdf.inter_box_flow_retention.mean()) if len(rdf) else np.nan
        final_rows.append(row)
        eta.update(time.perf_counter()-t,i)
    if funnel_survivors: eta.finish()

    funnel_final_df=pd.DataFrame(final_rows)
    funnel_local_df=pd.concat(local_frames,ignore_index=True) if local_frames else pd.DataFrame()
    funnel_renorm_df=pd.concat(renorm_frames,ignore_index=True) if renorm_frames else pd.DataFrame()

    full_final_df=pd.concat([long_final_df,funnel_final_df],ignore_index=True,sort=False)
    local_depth_df=pd.concat([long_local_df,funnel_local_df],ignore_index=True,sort=False)
    renorm_df=pd.concat([long_renorm_df,funnel_renorm_df],ignore_index=True,sort=False)
    atomic_csv(full_final_df,OUTDIR/"full_final_fingerprints.csv")
    if len(local_depth_df): atomic_csv(local_depth_df,OUTDIR/"local_depth_profiles.csv")
    if len(renorm_df): atomic_csv(renorm_df,OUTDIR/"renormalization_profiles.csv")

    # Full V5/exploratory final diagnostics: tail models, node-level local dimensions,
    # all-scale directed Renyi flow, quotient distributions, metric-p robustness,
    # and greedy-cover order uncertainty. These tables power the complete plot suite.
    banner("STAGE 5B — COMPREHENSIVE V5 / EXPLORATORY FINAL DIAGNOSTICS")
    extended_tables=build_comprehensive_final_tables(full_final_df,train_df)

    # First-class cross-architecture entropy and strictly validated local-dimension ensembles.
    entropy_ensemble_df=build_entropy_ensemble_table(long_tdf)
    local_density_ensemble_df=build_local_dimension_ensemble_density(
        extended_tables.get("local_node",pd.DataFrame())
    )
    extended_tables["entropy_ensemble"]=entropy_ensemble_df
    extended_tables["local_density_ensemble"]=local_density_ensemble_df
    if len(entropy_ensemble_df):
        atomic_csv(entropy_ensemble_df,OUTDIR/"longitudinal_entropy_ensemble.csv")
    if len(local_density_ensemble_df):
        atomic_csv(local_density_ensemble_df,OUTDIR/"ensemble_local_dimension_density.csv")

    # ------------------------------------------------------------------
    # Harmonized tables for consistent plots across ALL trained architectures.
    # ------------------------------------------------------------------
    banner("STAGE 6 — BUILDING HARMONIZED TEMPORAL / SCALE TABLES")
    comparison_tdf,comparison_sdf,comparison_hist=build_all_trained_comparison_tables(
        train_df,map_df,rung_df,long_tdf,long_sdf,hist_df,long_hist_df,cohort_ids
    )
    comparison_tdf=add_master_error_column(comparison_tdf,comparison_sdf)
    atomic_csv(comparison_tdf,OUTDIR/"all_trained_temporal_summary.csv")
    atomic_csv(comparison_sdf,OUTDIR/"all_trained_scale_profiles.csv")
    atomic_csv(comparison_hist,OUTDIR/"all_trained_history.csv")

    # Initialization -> trained preservation is evaluated on the FIXED cohort at every
    # geometry checkpoint so the sample itself does not change with epoch.
    preservation=[]
    if len(long_tdf):
        init_by_arch=long_tdf[long_tdf.epoch==0].set_index("arch_id")
        for ep,g in long_tdf.groupby("epoch"):
            if ep==0: continue
            gg=g.set_index("arch_id")
            common=sorted(set(init_by_arch.index)&set(gg.index))
            for feat in CORE_FEATURES:
                if feat not in init_by_arch.columns or feat not in gg.columns: continue
                a=init_by_arch.loc[common,feat].to_numpy(float)
                b=gg.loc[common,feat].to_numpy(float)
                good=np.isfinite(a)&np.isfinite(b)
                rho=(float(pd.Series(a[good]).corr(pd.Series(b[good]),method="spearman"))
                     if good.sum()>=3 else np.nan)
                preservation.append({"epoch":int(ep),"feature":feat,
                                     "spearman_init_vs_trained":rho,"n":int(good.sum()),
                                     "cohort":"fixed_longitudinal"})
    preserve_df=pd.DataFrame(preservation)
    atomic_csv(preserve_df,OUTDIR/"initialization_to_training_preservation.csv")

    # PCA structural trajectories for the longitudinal cohort only: unbiased trajectories.
    traj=[]
    for _,r in long_tdf.iterrows():
        z=fps.transform_row(r.to_dict())
        pc=pca.transform(z.reshape(1,-1))[0]
        traj.append({"arch_id":r.arch_id,"epoch":int(r.epoch),"morphology":r.morphology,
                     **{f"pca{i+1}":pc[i] for i in range(len(pc))}})
    traj_df=pd.DataFrame(traj)
    atomic_csv(traj_df,OUTDIR/"structural_trajectories.csv")

    # ------------------------------------------------------------------
    # Consistent figures: same axes, same epoch colors, same panels for every architecture.
    # ------------------------------------------------------------------
    banner("STAGE 7 — GENERATING CONSISTENT CROSS-ARCHITECTURE FIGURES")
    limits=common_plot_limits(comparison_sdf)
    with PdfPages(OUTDIR/"all_trained_architectures_V5_style_pages.pdf") as pdf:
        for _,ar in tqdm(train_df.iterrows(),total=len(train_df),
                         desc="Per-architecture V5-style pages",unit="architecture"):
            plot_one_architecture_page(
                str(ar.arch_id),ar,comparison_tdf,comparison_sdf,comparison_hist,limits,pdf=pdf
            )

    plot_longitudinal_ensemble(long_tdf,long_sdf,ensemble_df)
    # Dedicated first-class variability plots requested for interpretation.
    plot_entropy_across_architectures(long_tdf,extended_tables.get("entropy_ensemble",pd.DataFrame()),pdf=None)
    plot_local_dimensions_across_architectures(
        extended_tables.get("local_node",pd.DataFrame()),
        extended_tables.get("local_density_ensemble",pd.DataFrame()),pdf=None
    )
    plot_final_architecture_comparison(long_tdf,long_sdf,longitudinal_df)
    plot_full_budget_diagnostics(full_final_df,local_depth_df,renorm_df,train_df)

    # Exact V5 core figures + all nine comprehensive exploratory figure families,
    # repeated with consistent styling for every trained architecture. Final-only
    # diagnostics appear for every architecture that reaches the maximum budget.
    banner("STAGE 7B — STATISTICAL RESULT TABLES + COMPLETE V5 / EXPLORATORY REPORT")
    stats_df=build_architecture_statistical_results(train_df,comparison_tdf,comparison_sdf,comparison_hist,extended_tables)
    group_stats_df=build_morphology_group_statistics(stats_df)
    atomic_csv(stats_df,OUTDIR/"architecture_statistical_results.csv")
    atomic_csv(group_stats_df,OUTDIR/"morphology_group_statistics.csv")
    # Human-readable architecture table alongside the machine-readable CSV.
    (OUTDIR/"architecture_statistical_results.txt").write_text(stats_df.to_string(index=False),encoding="utf-8")
    plot_all_comprehensive_suites(train_df,comparison_tdf,comparison_sdf,comparison_hist,extended_tables,stats_df)

    # Initialization architecture map remains useful, but is no longer the only visual.
    morph_colors={m:plt.get_cmap("tab10")(i) for i,m in enumerate(MORPHOLOGIES)}
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.5))
    for m,g in map_df.groupby("morphology"):
        axs[0].scatter(g.pca1,g.pca2,s=26,alpha=.75,label=m,color=morph_colors[m])
    axs[0].set_xlabel("Initialization phenotype PC1"); axs[0].set_ylabel("PC2")
    axs[0].legend(frameon=False,ncol=2)
    sc=axs[1].scatter(map_df.depth,map_df.scaling_dB,c=np.log10(map_df.actual_params),cmap="viridis",s=28)
    axs[1].set_xlabel("Hidden depth"); axs[1].set_ylabel(r"Initialization $d_B^F$")
    fig.colorbar(sc,ax=axs[1],pad=.02,label=r"$\log_{10}$ parameters")
    fig.tight_layout(); savefig(fig,"figure1_initial_architecture_map"); plt.close(fig)

    # Longitudinal PCA trajectories.
    if len(traj_df):
        fig,axs=plt.subplots(1,2,figsize=(8.4,3.5))
        for aid,g in traj_df.groupby("arch_id"):
            g=g.sort_values("epoch"); axs[0].plot(g.pca1,g.pca2,"o-",ms=3,alpha=.75)
        axs[0].set_xlabel("Structural PC1"); axs[0].set_ylabel("Structural PC2")
        axs[0].set_title("Fixed-cohort trajectories")
        for ep,g in long_tdf.groupby("epoch"):
            axs[1].scatter(g.structural_deformation,g.val_mse,s=24,label=f"{int(ep)}")
        axs[1].set_xlabel(r"$||S_t-S_0||$"); axs[1].set_ylabel("Validation MSE")
        axs[1].legend(frameon=False,ncol=2,title="epoch")
        fig.tight_layout(); savefig(fig,"figure2_longitudinal_structural_deformation"); plt.close(fig)

    if len(preserve_df):
        fig,ax=plt.subplots(figsize=(5.2,3.6))
        mean_pres=preserve_df.groupby("epoch").spearman_init_vs_trained.mean()
        q16=preserve_df.groupby("epoch").spearman_init_vs_trained.quantile(.16)
        q84=preserve_df.groupby("epoch").spearman_init_vs_trained.quantile(.84)
        ax.fill_between(mean_pres.index,q16.reindex(mean_pres.index),q84.reindex(mean_pres.index),alpha=.18)
        ax.plot(mean_pres.index,mean_pres.values,"o-")
        ax.axhline(0,ls="--",lw=.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Descriptor Spearman: init vs trained")
        fig.tight_layout(); savefig(fig,"figure3_fixed_cohort_preservation"); plt.close(fig)

    # ------------------------------------------------------------------
    # Human-readable summary.
    # ------------------------------------------------------------------
    mean_pres=(preserve_df.groupby("epoch").spearman_init_vs_trained.mean()
               if len(preserve_df) else pd.Series(dtype=float))
    summary=f"""PATH-FLOW ARCHITECTURE UNIVERSALITY EXPERIMENT V2
====================================================

Environment
-----------
N_train={N_TRAIN:,}, N_val={N_VAL:,}, N_test={N_TEST:,}, d={INPUT_DIM}
N_train/d={N_TRAIN/INPUT_DIM:.1f}
Teacher={INPUT_DIM}->{TEACHER_HIDDEN}->1; noise fraction={TEACHER_NOISE_STD}

Architecture map
----------------
Unique realized architectures={len(map_df)}
Dense stratified Sobol requested={MAP_SOBOL}; diversity-constrained BO accepted={MAP_BO}
SMAC used={SMAC_AVAILABLE}
Initial structural clusters={best_k}; silhouette={best_score:.4f}

Temporal design
---------------
Selected for training={len(train_df)}
Fixed longitudinal cohort={len(longitudinal_df)}
Funnel architectures={len(funnel_df)}
Exhaustive per-architecture mode={REQUIRE_EXHAUSTIVE_PER_ARCH}
Geometry checkpoints={GEOMETRY_CHECKPOINTS}
Longitudinal BCANw scales={LONGITUDINAL_BCANW_SCALES}
Common plotting x-range=[{COMMON_X_MIN}, {COMMON_X_MAX}]

Temporal variance principle
---------------------------
ALL scale-by-epoch mean/SD/16-84% bands are computed only from the fixed longitudinal
cohort. Therefore temporal changes in variance are not caused by architectures being
pruned from the sample.

Initialization-to-training preservation — fixed cohort
------------------------------------------------------
{mean_pres.to_string() if len(mean_pres) else "insufficient data"}

Primary comparable outputs
--------------------------
* all_trained_architectures_V5_style_pages.pdf
  One page per trained architecture, identical panels/axes/epoch coloring.
* longitudinal_scale_ensemble.csv
  Common-grid mean, SD, median, and 16-84% intervals at every scale and epoch.
* figure4_longitudinal_scale_ribbons
  Median + architecture variation across scale at selected epochs.
* figure5_longitudinal_scale_time_mean_variance
  Mean and across-architecture SD of d_eff(x,t) and log10[N_B/N](x,t).
* figure6_longitudinal_descriptor_variance
  Temporal median + architecture spread of path/geometry descriptors.
* figure8_entropy_across_architectures
  Purple entropy-deficit curves with median and 16-84% variability across the fixed cohort,
  plus change-from-initialization and depth/path-count normalized entropy deficit.
* longitudinal_entropy_ensemble.csv
  Per-epoch entropy mean/SD/median/16-84% across exactly the same architectures.
* figure9_local_dimension_distributions_across_architectures
  Strictly validated support/flow-mass dimension densities and green Delta density,
  with 16-84% variability across architectures.
* ensemble_local_dimension_density.csv
  Common-bin density envelopes used by the aggregate strictly validated local plots.
* architecture_figures/
  Individual V5-style PNG/PDF pages for every trained architecture.
* all_full_budget_final_diagnostics.pdf
  Consistent final microscopic-flow and local/RG pages for every full-budget architecture.
* comprehensive_figures/<ARCH_ID>/<ARCH_ID>_ALL_V5_AND_EXPLORATORY_PLOTS.pdf
  Complete per-architecture book containing V5 Figures 1-4 + S1 and Exploratory E1-E9.
* comprehensive_tables/
  Tail-model, node-level local geometry, all-scale Renyi, quotient-flow, metric-power,
  and cover-order robustness tables used by the comprehensive plots.
* architecture_statistical_results.csv / .txt
  One exhaustive architecture-description + statistical-result row per configuration.
* morphology_group_statistics.csv
  Median and IQR summaries by architecture morphology.
* {FINAL_REPORT_FILENAME}
  Final exhaustive multipage PDF: report cover + architecture index + one statistical page
  and the complete requested plot suite for every selected architecture.

Interpretation reminder
-----------------------
Compare architectures only on the shared intrinsic coordinate x=ell/xi50 and common axes.
Use the fixed cohort for claims about temporal organization and temporal variance.
Entropy variability is always computed from that unchanged cohort.
Local Delta_i^F now uses a strict shared radial scaling window: d_i^F and alpha_i^F are
fit on exactly the same interval before subtraction.  Invalid shared windows remain NaN.

Architecture proposals are filtered on realized median/q10 width, clipping fractions, parameter
budget fidelity, and graph size.  The initialization map is stratified over morphology x depth x
parameter-budget cells; BO uses EI plus random exploration and a cell-density guard.

Strictly validated local dimensions are final-snapshot diagnostics: per architecture and as
an across-architecture density envelope; failed local fits remain NaN rather than being imputed.
Phase-space panels never substitute a different structural metric when validation is sparse:
they explicitly report insufficient paired checkpoints instead of displaying rho=nan.
Use the funnel only for broader structural discovery; its late-epoch population is selected.
"""
    (OUTDIR/"experiment_summary.txt").write_text(summary,encoding="utf-8")
    print("\n"+summary)

    elapsed=time.perf_counter()-total_start
    print(f"[ALL DONE] Total wall time: {human_time(elapsed)}")
    print(f"[outputs] {OUTDIR.resolve()}")
    return {
        "architecture_map":map_df,
        "selected_training":train_df,
        "longitudinal_cohort":longitudinal_df,
        "longitudinal_temporal":long_tdf,
        "longitudinal_scale":long_sdf,
        "longitudinal_ensemble":ensemble_df,
        "comprehensive_tables":extended_tables,
        "architecture_statistics":stats_df,
        "morphology_group_statistics":group_stats_df,
        "funnel_rungs":rung_df,
        "all_trained_temporal":comparison_tdf,
        "all_trained_scale":comparison_sdf,
        "full_final":full_final_df,
        "trajectories":traj_df,
        "preservation":preserve_df,
    }


# ======================================================================================
# 15. EXECUTE
# ======================================================================================

print("\nPATH-FLOW ARCHITECTURE UNIVERSALITY V2 — COMPARABLE TEMPORAL GEOMETRY")
print(f"Device: {DEVICE}")
print(f"Run profile: {RUN_PROFILE}")
print(f"Research dataset: {N_TRAIN:,} train + {N_VAL:,} val + {N_TEST:,} test; "
      f"d={INPUT_DIM}; N_train/d={N_TRAIN/INPUT_DIM:.1f}")
print(f"Map A: {MAP_SOBOL} stratified Sobol + {MAP_BO} diversity-constrained Bayesian accepted architectures")
print(f"Map B: {TRAIN_START} structurally diverse architectures")
print(f"Fixed longitudinal cohort: {TRAIN_START if REQUIRE_EXHAUSTIVE_PER_ARCH else min(LONGITUDINAL_COHORT,TRAIN_START)}")
print(f"Architecture depth search: {DEPTH_MIN}-{DEPTH_MAX} hidden layers (all > 8); dense grid={MAP_DEPTH_BINS} depth bins x {MAP_PARAM_BINS} budget bins x {len(MORPHOLOGIES)} morphologies")
print(f"Exhaustive per-architecture reporting: {REQUIRE_EXHAUSTIVE_PER_ARCH}")
print(f"Common geometry checkpoints: {GEOMETRY_CHECKPOINTS}")
print("ETA note: machine/GPU dependent. Every expensive stage reports current-step time, recent median throughput, elapsed time, and projected remaining time.")
print("Plot note: every selected architecture receives the full reference plot family, architecture statistics page, and inclusion in the final exhaustive PDF.")

RESULTS = run_experiment()
