# ==================================================================================================
# CNN-C3 — SELF-CONTAINED OPTIMIZER-CONDITIONED REALIZATION (TIME/SPACE BOUNDED)
# ==================================================================================================
# Scientific question
# -------------------
# Does optimizer identity remain visible after input conditioning, and does the optimizer ordering of
# structural path-flow disposition p^S survive in the realized conditional measure mu^c?
#
# This production version is deliberately self-contained: it DOES NOT depend on completion of CNN-S3.
# It trains all four optimizer arms from exactly matched architecture/seed initial states and evaluates
# structural and conditional path-flow on the same trained states.
#
# Bounded production design
# -------------------------
#   5 architecture sentinels (shallow, canonical, bottleneck, pointwise, deep when available)
#   x 3 frozen S2 environments (canonical + two deterministic maximin contrasts)
#   x 2 seeds
#   x 4 optimizers: Adam, RMSProp, SGD+momentum, SGD
#   = 120 fresh matched trajectories
#
# Fixed optimizer learning rates inherit the predeclared S3 defaults; no calibration/pilot is run:
#   Adam=3e-3, RMSProp=3e-3, SGDM=3e-2, SGD=1e-2.
#
# Exposure is deliberately bounded at 80 epochs, an existing common programme checkpoint:
#   t = {0, 1, 5, 20, 40, 80}.
#
# Conditional realization uses 64 fixed inputs at intermediate states and 128 at t=0/t=80.
# BCANw is restricted to a balanced endpoint sentinel arm (3 architectures x 2 environments x 2 seeds
# x 4 optimizers) and computes both structural and population-conditional geometry.
#
# Normal HPC use from CNN_S_C/conditional/:
#   srun python -u pathflow_cnn_c3_optimizer_conditional_realization_SELFCONTAINED_10H.py
#
# One invocation performs metadata/disk/time preflight -> train -> condition -> geometry -> aggregate.
# Completed units and exact mid-epoch optimizer/model state are resumable after interruption.
# Under Slurm, logging is intentionally compact. Jupyter may run the same file in one cell.
# ==================================================================================================

from __future__ import annotations

import os
import sys
import gc
import math
import json
import time
import copy
import signal
import random
import shutil
import hashlib
import itertools
import platform
import socket
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence, Iterable

# --------------------------------------------------------------------------------------------------
# Imports: fail loudly rather than installing packages on a compute node.
# --------------------------------------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm.auto import tqdm
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
    from scipy.stats import spearmanr, friedmanchisquare, wilcoxon
except ImportError as exc:
    raise ImportError(
        "CNN-C3 requires numpy, pandas, scipy, tqdm and torch. Install them in the environment before "
        "submitting the job; this script intentionally performs no runtime package installation."
    ) from exc

# ==================================================================================================
# 0. FROZEN SCIENTIFIC CONFIGURATION + HPC ORCHESTRATION
# ==================================================================================================

# Direct-execution default: a plain `python -u this_file.py` launches the complete C3 science campaign.
# Phase-specific modes remain available only as recovery/operator overrides.
MODE = os.environ.get("PATHFLOW_CNN_C3_MODE", "full").strip().lower()
VALID_MODES = {"full", "train", "condition", "geometry", "input_geometry", "aggregate", "validate", "preflight", "all"}
if MODE not in VALID_MODES:
    raise ValueError(f"PATHFLOW_CNN_C3_MODE must be one of {sorted(VALID_MODES)}; got {MODE!r}")

# Prefer the project HPC hierarchy when it actually exists; otherwise remain portable/self-contained.
PROJECT_CNN_ROOT = Path(
    "<PROJECT_ROOT>/compress/sprime/BCANw_reference_fit/"
    "path_flow/autoDl_search/128_G/CNN_S_C"
)
if PROJECT_CNN_ROOT.exists():
    _default_out = PROJECT_CNN_ROOT / "conditional" / "C3" / "pathflow_cnn_c3_optimizer_conditional_realization_SELFCONTAINED_10H_results"
else:
    _default_out = Path("C3/pathflow_cnn_c3_optimizer_conditional_realization_SELFCONTAINED_10H_results")
OUTDIR = Path(os.environ.get("PATHFLOW_CNN_C3_OUTDIR", str(_default_out))).expanduser()

for sub in [
    "tables", "figures", "checkpoints", "run_parts", "resume_parts", "condition_parts",
    "condition_states", "geometry_parts", "input_geometry_parts", "validation", "audit"
]:
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

# Frozen shared conditional-realization scientific core.
INPUT_CHANNELS = 1
RESOLUTION = 16
ACTIVATION = "relu"
OPTIMIZER = "adam"
ADAM_LR = 3e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 256
N_TRAIN, N_VAL, N_TEST = 20_000, 8_000, 8_000
FULL_EPOCHS = 360
STATE_GRID = [0, 1, 2, 5, 10, 20, 40, 80, 120, 160, 240, 360]
SEEDS = [0, 1, 2, 3]
N_ARCH = 12
LONGITUDINAL_INPUTS = 64
DEEP_INPUTS = 256
INPUT_GEOMETRY_INPUTS = 16
PRINCIPAL_STATES = ["t000", "best", "t360"]
SENTINEL_ARCH_IDS = ["C1_shallow3", "C2_medium3", "C3_bottleneck", "C5_deep3"]

# The exact 12-row architecture coordinate system inherited from the CNN structural programme.
FROZEN_ARCHITECTURES: List[Tuple[str, List[int], List[int]]] = [
    ("C1_shallow3", [4, 4], [3, 3, 1]),
    ("C2_medium3", [4, 8, 8], [3, 3, 3, 1]),
    ("C3_bottleneck", [6, 3, 6], [3, 3, 3, 1]),
    ("C4_pointwise", [4, 8, 8], [1, 1, 1, 1]),
    ("C5_deep3", [4, 8, 8, 12], [3, 3, 3, 3, 1]),
    ("C6_large5", [4, 8, 8], [5, 3, 3, 1]),
    ("S2F07", [3, 5, 7], [3, 1, 3, 1]),
    ("S2F08", [8, 4, 8], [3, 3, 1, 1]),
    ("S2F09", [5, 5, 5, 5], [1, 3, 1, 3, 1]),
    ("S2F10", [3, 6, 9, 6], [3, 1, 3, 1, 1]),
    ("S2F11", [8, 6, 4], [1, 3, 3, 1]),
    ("S2F12", [5, 10, 10], [3, 3, 1, 1]),
]

# One canonical architectural-reference environment.  This is the distributed canonical anchor of the
# controlled spatial family; retained only as a shared engine constant.
CANONICAL_ENV: Dict[str, Any] = {
    "env_id": "E_CAN_DISTRIBUTED",
    "teacher_family": "deep_compositional",
    "informative_fraction": 0.85,
    "spatial_correlation": 0.20,
    "snr": 25.0,
    "interaction_strength": 0.35,
    "localization": 0.05,
}

# Exact programme seed conventions.
C3_SEED = 61031
MODEL_SEED_BASE = 271828
DATA_SEED_BASE = 57721
ORDER_SEED_BASE = 141421
CALIBRATION_SEED_BASE = 918273

# Exact path / BCANw constants.
EPS = 1e-14
PATH_LOG_FLOOR = 1e-300
METRIC_POWER = 1.0
BCANW_SCALES_LONG = int(os.environ.get("PATHFLOW_CNN_C3_BCANW_SCALES", "36"))
BCANW_MAX_NODES = int(os.environ.get("PATHFLOW_CNN_C3_BCANW_MAX_NODES", "7000"))
MIN_FINAL_RESOLUTION = 4
MAX_EXEC_NODES = 6500
SCALING_R2_MIN = 0.97
SCALING_MIN_DIM = 0.25
SCALING_MIN_POINTS = 6
SCALING_MIN_SPAN_DECADES = 0.50
SCALING_MIN_NB_RATIO = 5.0
SCALING_MAX_SLOPE_CV = 0.25
SCALING_MAX_DELTA_BIC = 2.0
COARSE_RENYI_Q = [0.5, 1.0, 2.0, 3.0]

# Conditional batching and descriptive law-survival classification thresholds.
CONDITIONAL_BATCH = max(1, int(os.environ.get("PATHFLOW_CNN_C3_CONDITIONAL_BATCH", "16")))
RANK_PRESERVED_MIN = 0.70
RANK_RELATED_MIN = 0.50
EFFECT_ABSENT_RATIO = 0.10
EFFECT_ATTENUATED_RATIO = 0.50
EFFECT_AMPLIFIED_RATIO = 2.00

# Slurm / resume / walltime controls.
RESUME = os.environ.get("PATHFLOW_CNN_C3_RESUME", "1") != "0"
REQUIRE_CUDA = os.environ.get("PATHFLOW_CNN_C3_REQUIRE_CUDA", "1") != "0"
GPU_RESIDENT_DATA = os.environ.get("PATHFLOW_CNN_C3_GPU_RESIDENT_DATA", "1") != "0"
CUDA_DEVICE = int(os.environ.get("PATHFLOW_CNN_C3_CUDA_DEVICE", os.environ.get("SLURM_LOCALID", "0")))
SAVE_RESUME_EVERY_BATCHES = max(1, int(os.environ.get("PATHFLOW_CNN_C3_RESUME_EVERY_BATCHES", "20")))
SOFT_WALLTIME_SECONDS = float(os.environ.get("PATHFLOW_CNN_C3_SOFT_WALLTIME_SECONDS", "0"))
SOFT_WALLTIME_BUFFER_SECONDS = float(os.environ.get("PATHFLOW_CNN_C3_SOFT_WALLTIME_BUFFER_SECONDS", "240"))
MAX_JOBS_THIS_INVOCATION = int(os.environ.get("PATHFLOW_CNN_C3_MAX_JOBS", "0"))
TQDM_ENABLED = os.environ.get("PATHFLOW_CNN_C3_TQDM", "0" if os.environ.get("SLURM_JOB_ID") else "1") != "0"
TRACE_EVERY_SECONDS = float(os.environ.get("PATHFLOW_CNN_C3_TRACE_EVERY_SECONDS", "600"))
PREFLIGHT_BENCHMARK = os.environ.get("PATHFLOW_CNN_C3_PREFLIGHT_BENCHMARK", "1") != "0"
PREFLIGHT_GEOMETRY_BENCHMARK = os.environ.get("PATHFLOW_CNN_C3_PREFLIGHT_GEOMETRY_BENCHMARK", "1") != "0"
ENFORCE_DISK = os.environ.get("PATHFLOW_CNN_C3_ENFORCE_DISK", "1") != "0"
DISK_SAFETY_FACTOR = float(os.environ.get("PATHFLOW_CNN_C3_DISK_SAFETY_FACTOR", "1.40"))
MIN_FREE_AFTER_GB = float(os.environ.get("PATHFLOW_CNN_C3_MIN_FREE_AFTER_GB", "5.0"))

SHARDS = max(1, int(os.environ.get("PATHFLOW_CNN_C3_SHARDS", "1")))
_raw_shard = os.environ.get("PATHFLOW_CNN_C3_SHARD_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
SHARD_INDEX = int(_raw_shard)
if SHARD_INDEX < 0 or SHARD_INDEX >= SHARDS:
    raise ValueError(f"Shard index {SHARD_INDEX} invalid for PATHFLOW_CNN_C3_SHARDS={SHARDS}; expected 0..{SHARDS-1}")

CPUS_PER_TASK = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("PATHFLOW_CNN_C3_CPUS", "1"))))
os.environ.setdefault("OMP_NUM_THREADS", str(CPUS_PER_TASK))
os.environ.setdefault("MKL_NUM_THREADS", str(CPUS_PER_TASK))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPUS_PER_TASK))
try:
    torch.set_num_threads(CPUS_PER_TASK)
except Exception:
    pass

if REQUIRE_CUDA and not torch.cuda.is_available() and MODE in {"full", "train", "condition", "geometry", "input_geometry", "all"}:
    raise RuntimeError(
        "CNN-C3 is configured to require CUDA for this mode, but torch.cuda.is_available() is False. "
        "Set PATHFLOW_CNN_C3_REQUIRE_CUDA=0 only for an intentional CPU smoke/debug run."
    )
if torch.cuda.is_available():
    n_gpu = max(torch.cuda.device_count(), 1)
    CUDA_DEVICE = CUDA_DEVICE % n_gpu
    torch.cuda.set_device(CUDA_DEVICE)
    DEVICE = torch.device(f"cuda:{CUDA_DEVICE}")
else:
    DEVICE = torch.device("cpu")

if os.environ.get("PATHFLOW_CNN_C3_DETERMINISTIC_KERNELS", "0") == "1":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

RUN_START = time.monotonic()
STOP_REQUESTED = False
STOP_SIGNAL: Optional[int] = None


def _signal_handler(signum, frame):
    global STOP_REQUESTED, STOP_SIGNAL
    STOP_REQUESTED = True
    STOP_SIGNAL = int(signum)
    print(f"[signal] received {signum}; checkpointing at the next safe boundary", flush=True)


for _sig_name in ["SIGUSR1", "SIGTERM", "SIGINT"]:
    if hasattr(signal, _sig_name):
        try:
            signal.signal(getattr(signal, _sig_name), _signal_handler)
        except Exception:
            pass


def soft_walltime_due() -> bool:
    if SOFT_WALLTIME_SECONDS <= 0:
        return False
    elapsed = time.monotonic() - RUN_START
    return elapsed >= max(0.0, SOFT_WALLTIME_SECONDS - SOFT_WALLTIME_BUFFER_SECONDS)

# ==================================================================================================
# 1. TRACE / ATOMIC IO / DETERMINISM UTILITIES
# ==================================================================================================


def fmt_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = float(seconds)
    d, rem = divmod(int(round(seconds)), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d: return f"{d}d {h:02d}h {m:02d}m"
    if h: return f"{h}h {m:02d}m {s:02d}s"
    if m: return f"{m}m {s:02d}s"
    return f"{s}s"


def human_bytes(n: float) -> str:
    n = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while abs(n) >= 1024 and i < len(units) - 1:
        n /= 1024.0; i += 1
    return f"{n:.2f} {units[i]}"


def banner(title: str):
    print("\n" + "=" * 112, flush=True)
    print(title, flush=True)
    print("=" * 112, flush=True)


def trace(msg: str):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tqdm.write(f"[{stamp}] {msg}") if TQDM_ENABLED else print(f"[{stamp}] {msg}", flush=True)


def stable_hash_dict(d: Dict[str, Any], n: int = 12) -> str:
    def clean(v):
        if isinstance(v, (np.integer, np.floating)): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        return v
    payload = {str(k): clean(v) for k, v in d.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:n]


def stable_shard(key: str, n_shards: int = SHARDS) -> int:
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:16], 16) % int(n_shards)


def in_this_shard(key: str) -> bool:
    return stable_shard(key, SHARDS) == SHARD_INDEX


def atomic_csv(df: pd.DataFrame, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False); os.replace(tmp, path)


def atomic_json(obj: Any, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8"); os.replace(tmp, path)


def atomic_torch_save(obj: Any, path: Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp); os.replace(tmp, path)


def atomic_npz(path: Path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)


def safe_read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 1: return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()


def parse_int_list(x: Any) -> List[int]:
    if isinstance(x, str): x = json.loads(x)
    return [int(v) for v in list(x)]


def model_seed(arch_id: str, seed_index: int) -> int:
    return MODEL_SEED_BASE + int(stable_hash_dict({"arch": str(arch_id), "seed": int(seed_index)}, 8), 16) % 1_000_003


def deterministic_batches(n: int, epoch: int, seed_index: int, batch_size: int = BATCH_SIZE):
    rng = np.random.default_rng(ORDER_SEED_BASE + 100003 * int(seed_index) + int(epoch))
    ids = rng.permutation(int(n))
    for lo in range(0, int(n), int(batch_size)):
        yield ids[lo:lo + int(batch_size)]


def run_id(arch_id: str, seed_index: int) -> str:
    return f"C2__{arch_id}__S{int(seed_index):02d}"


def checkpoint_path(rid: str, epoch: int) -> Path:
    return OUTDIR / "checkpoints" / f"{rid}__E{int(epoch):03d}.pt"


def best_checkpoint_path(rid: str) -> Path:
    return OUTDIR / "checkpoints" / f"{rid}__BEST.pt"


def resume_path(rid: str) -> Path:
    return OUTDIR / "resume_parts" / f"{rid}__resume.pt"


def run_part(rid: str, kind: str) -> Path:
    return OUTDIR / "run_parts" / f"{rid}__{kind}.csv"


def condition_part(job_id: str, kind: str) -> Path:
    return OUTDIR / "condition_parts" / f"{job_id}__{kind}.csv"


def condition_state_path(job_id: str) -> Path:
    return OUTDIR / "condition_states" / f"{job_id}.npz"


def geometry_part(job_id: str) -> Path:
    return OUTDIR / "geometry_parts" / f"{job_id}.csv"


def geometry_scale_part(job_id: str) -> Path:
    return OUTDIR / "geometry_parts" / f"{job_id}__scales.csv"


def input_geometry_part(job_id: str) -> Path:
    return OUTDIR / "input_geometry_parts" / f"{job_id}.csv"


def input_geometry_scale_part(job_id: str) -> Path:
    return OUTDIR / "input_geometry_parts" / f"{job_id}__scales.csv"


def state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def tensor_state_digest(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        a = state[k].detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def gpu_memory_text() -> str:
    if not torch.cuda.is_available(): return "CPU"
    try:
        alloc = torch.cuda.memory_allocated(DEVICE); reserved = torch.cuda.memory_reserved(DEVICE)
        peak = torch.cuda.max_memory_allocated(DEVICE)
        return f"GPU alloc={human_bytes(alloc)}, reserved={human_bytes(reserved)}, peak={human_bytes(peak)}"
    except Exception:
        return "GPU memory unavailable"


def process_memory_text() -> str:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux ru_maxrss is KiB, macOS bytes.
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        return f"maxRSS={human_bytes(rss_bytes)}"
    except Exception:
        return "maxRSS unavailable"


def runtime_record() -> Dict[str, Any]:
    out = {
        "timestamp_unix": time.time(), "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE), "mode": MODE, "shards": SHARDS, "shard_index": SHARD_INDEX,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus_per_task": CPUS_PER_TASK, "outdir": str(OUTDIR),
    }
    if torch.cuda.is_available():
        try:
            out["gpu_name"] = torch.cuda.get_device_name(DEVICE)
            out["gpu_total_memory_bytes"] = int(torch.cuda.get_device_properties(DEVICE).total_memory)
        except Exception:
            pass
    return out


def append_rate(phase: str, units: float, seconds: float, note: str = ""):
    if units <= 0 or seconds <= 0: return
    p = OUTDIR / "audit" / f"rates_shard_{SHARD_INDEX:04d}.csv"
    old = safe_read_csv(p)
    row = pd.DataFrame([{
        "timestamp": time.time(), "phase": phase, "units": float(units), "seconds": float(seconds),
        "seconds_per_unit": float(seconds / units), "note": note, "hostname": socket.gethostname(),
    }])
    atomic_csv(pd.concat([old, row], ignore_index=True), p)


def empirical_rate(phase: str, fallback: float) -> float:
    vals = []
    for p in sorted((OUTDIR / "audit").glob("rates_shard_*.csv")):
        d = safe_read_csv(p)
        if not d.empty and "phase" in d and "seconds_per_unit" in d:
            x = pd.to_numeric(d.loc[d.phase.astype(str) == str(phase), "seconds_per_unit"], errors="coerce")
            vals.extend(x[np.isfinite(x) & (x > 0)].tolist())
    return float(np.median(vals[-64:])) if vals else float(fallback)


class PhaseETA:
    def __init__(self, phase: str, total_units: int, fallback_seconds_per_unit: float):
        self.phase = str(phase); self.total = int(total_units); self.done = 0; self.t0 = time.monotonic()
        self.prior = empirical_rate(self.phase, fallback_seconds_per_unit)
        self.last_trace = self.t0
        trace(f"[{self.phase}] planned units={self.total}; planning ETA={fmt_seconds(self.prior*self.total)} "
              f"(rate={self.prior:.3g}s/unit; empirical if available, otherwise conservative prior)")

    def update(self, n: int = 1, force: bool = False):
        self.done += int(n); now = time.monotonic()
        if self.done <= 0: return
        observed = (now - self.t0) / self.done
        rate = 0.70 * observed + 0.30 * self.prior
        remain = max(self.total - self.done, 0) * rate
        if force or now - self.last_trace >= TRACE_EVERY_SECONDS or self.done == self.total:
            trace(f"[{self.phase}] {self.done}/{self.total} complete | observed={observed:.3g}s/unit | "
                  f"predicted remaining={fmt_seconds(remain)} | {process_memory_text()} | {gpu_memory_text()}")
            self.last_trace = now


def fixed_test_indices() -> np.ndarray:
    rng = np.random.default_rng(C3_SEED + 4049)
    ids = rng.permutation(N_TEST)[:DEEP_INPUTS]
    return np.asarray(ids, dtype=int)


def epoch_label(epoch: int) -> str:
    return f"t{int(epoch):03d}"

class ActivationModule(nn.Module):
    def __init__(self, name: str):
        super().__init__(); self.name = str(name).lower()
    def forward(self, x):
        if self.name == "relu": return F.relu(x)
        if self.name == "tanh": return torch.tanh(x)
        if self.name == "silu": return F.silu(x)
        if self.name == "identity": return x
        raise ValueError(self.name)


class ConvTraceNet(nn.Module):
    """Bias-free, stride-one VALID CNN with learned scalar feature map and fixed global average."""
    def __init__(self, resolution: int, hidden_channels: Sequence[int], kernel_sizes: Sequence[int], activation: str, seed: int):
        super().__init__()
        self.resolution = int(resolution)
        self.hidden_channels = [int(x) for x in hidden_channels]
        self.kernel_sizes = [int(k) for k in kernel_sizes]
        self.activation_name = str(activation)
        self.activation = ActivationModule(activation)
        self.channels = [INPUT_CHANNELS] + self.hidden_channels + [1]
        if len(self.kernel_sizes) != len(self.channels) - 1:
            raise ValueError("kernel_sizes must have one entry per learned convolution")
        self.convs = nn.ModuleList()
        torch.manual_seed(int(seed))
        for cin, cout, k in zip(self.channels[:-1], self.channels[1:], self.kernel_sizes):
            self.convs.append(nn.Conv2d(cin, cout, kernel_size=k, stride=1, padding=0, bias=False))
        self.reset_common_directions(seed)
        if self.output_resolution() < MIN_FINAL_RESOLUTION:
            raise ValueError("output spatial resolution below S2 minimum")

    def reset_common_directions(self, seed: int):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            for conv in self.convs:
                W = torch.randn(conv.weight.shape, generator=g, dtype=conv.weight.dtype)
                W /= math.sqrt(max(conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1], 1))
                conv.weight.copy_(W)

    def spatial_shapes(self) -> List[Tuple[int, int]]:
        h = w = self.resolution; out = [(h, w)]
        for k in self.kernel_sizes:
            h = h - k + 1; w = w - k + 1; out.append((h, w))
        return out

    def output_resolution(self) -> int:
        return int(self.spatial_shapes()[-1][0])

    def forward(self, x):
        h = x
        for l, conv in enumerate(self.convs):
            z = conv(h)
            h = self.activation(z) if l < len(self.convs) - 1 else z
        return h.mean(dim=(-1, -2))


def architecture_descriptors(hidden_channels: Sequence[int], kernel_sizes: Sequence[int]) -> Dict[str, float]:
    ch = [INPUT_CHANNELS] + list(map(int, hidden_channels)) + [1]
    h = w = RESOLUTION; occ = 0; nodes = INPUT_CHANNELS * h * w; rf = 1
    for cin, cout, k in zip(ch[:-1], ch[1:], kernel_sizes):
        h = h - int(k) + 1; w = w - int(k) + 1
        if h < 1: return {"valid": False}
        layer_occ = cout * cin * int(k) * int(k) * h * w
        occ += layer_occ; nodes += cout * h * w; rf += int(k) - 1
    params = sum(cout * cin * int(k) * int(k) for cin, cout, k in zip(ch[:-1], ch[1:], kernel_sizes))
    return {
        "valid": bool(h >= MIN_FINAL_RESOLUTION and nodes <= MAX_EXEC_NODES),
        "parameter_count": int(params), "occurrence_count": int(occ), "execution_nodes": int(nodes),
        "output_resolution": int(h), "receptive_field": int(rf), "depth": int(len(kernel_sizes)),
        "mean_hidden_width": float(np.mean(hidden_channels)) if hidden_channels else 1.0,
        "max_hidden_width": int(max(hidden_channels)) if hidden_channels else 1,
    }


def _arch_from_row(r: pd.Series, seed_index: int) -> ConvTraceNet:
    hidden = parse_int_list(r["hidden_channels"])
    kernels = parse_int_list(r["kernel_sizes"])
    return ConvTraceNet(RESOLUTION, hidden, kernels, ACTIVATION, model_seed(str(r["arch_id"]), seed_index)).to(DEVICE)


@torch.no_grad()
def calibrate_initial_signal(model: ConvTraceNet, Xcal: torch.Tensor, target_var: float = 1.0) -> List[float]:
    # Environment-independent architecture/seed calibration retained from the structural programme.
    model.eval(); h = Xcal.to(DEVICE); scales = []
    for l, conv in enumerate(model.convs):
        z = conv(h)
        if l < len(model.convs) - 1:
            sd = float(torch.std(z, unbiased=False).item())
            scale = float(np.clip(math.sqrt(target_var) / max(sd, 1e-6), 0.15, 6.0))
            conv.weight.mul_(scale); z = conv(h); h = model.activation(z); scales.append(scale)
        else:
            scales.append(1.0)
    return scales


ENV_KEYS = ["teacher_family", "informative_fraction", "spatial_correlation", "snr", "interaction_strength", "localization"]


def environment_key_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "teacher_family": str(d["teacher_family"]),
        "informative_fraction": round(float(d["informative_fraction"]), 6),
        "spatial_correlation": round(float(d["spatial_correlation"]), 6),
        "snr": round(float(d["snr"]), 6),
        "interaction_strength": round(float(d["interaction_strength"]), 6),
        "localization": round(float(d["localization"]), 6),
    }


def _spatial_support_mask(cfg: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    R = RESOLUTION
    n = max(1, min(R * R, int(round(float(cfg["informative_fraction"]) * R * R))))
    loc = float(cfg["localization"])
    yy, xx = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    cy = rng.uniform(0, R - 1); cx = rng.uniform(0, R - 1)
    sigma = max(1.0, R * (0.10 + 0.35 * (1 - loc)))
    compact = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    scattered = rng.random((R, R)); score = (1 - loc) * scattered + loc * compact
    ids = np.argpartition(score.ravel(), -n)[-n:]
    mask = np.zeros(R * R, np.float32); mask[ids] = 1.0
    return mask.reshape(R, R)


def _correlated_images(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    R = RESOLUTION
    eps = rng.normal(0, 1, (n, 1, R, R)).astype(np.float32)
    t = torch.from_numpy(eps)
    low = F.avg_pool2d(t, kernel_size=5, stride=1, padding=2).numpy()
    low = (low - low.mean()) / (low.std() + 1e-8)
    X = math.sqrt(max(1 - rho, 0)) * eps + math.sqrt(max(rho, 0)) * low.astype(np.float32)
    return X.astype(np.float32)


def _teacher_clean_target(X: np.ndarray, cfg: Dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(X); Xm = X * mask[None, None, :, :]; fam = str(cfg["teacher_family"])
    xt = torch.from_numpy(Xm)
    torch.manual_seed(int(rng.integers(1, 2 ** 31 - 1)))
    with torch.no_grad():
        if fam == "linear_spatial":
            w = torch.from_numpy((rng.normal(0, 1, (1, 1, RESOLUTION, RESOLUTION)) * mask[None, None]).astype(np.float32))
            base = (xt * w).sum((1, 2, 3)) / math.sqrt(max(mask.sum(), 1))
        elif fam == "shallow_nonlinear":
            W = torch.randn(6, 1, 3, 3) / 3.0
            h = F.relu(F.conv2d(xt, W, padding=1)); read = torch.randn(6) / math.sqrt(6)
            base = (h.mean((2, 3)) * read).sum(1)
        elif fam == "deep_compositional":
            W1 = torch.randn(6, 1, 3, 3) / 3.0
            W2 = torch.randn(8, 6, 3, 3) / math.sqrt(54)
            W3 = torch.randn(6, 8, 3, 3) / math.sqrt(72)
            h = F.relu(F.conv2d(xt, W1, padding=1)); h = F.relu(F.conv2d(h, W2, padding=1)); h = F.relu(F.conv2d(h, W3, padding=1))
            read = torch.randn(6) / math.sqrt(6); base = (h.mean((2, 3)) * read).sum(1)
        else:
            raise ValueError(f"Unknown teacher_family={fam!r}")
    ids = np.flatnonzero(mask.ravel() > 0); m = min(16, len(ids) // 2); flat = X[:, 0].reshape(n, -1)
    if m >= 1:
        perm = rng.permutation(ids)[:2 * m]
        inter = np.mean(flat[:, perm[:m]] * flat[:, perm[m:2 * m]], axis=1)
    else:
        inter = np.zeros(n, float)
    b = base.numpy().astype(np.float64); b = (b - b.mean()) / (b.std() + 1e-8)
    inter = (inter - inter.mean()) / (inter.std() + 1e-8)
    k = float(cfg["interaction_strength"])
    return ((1 - k) * b + k * inter).astype(np.float32)[:, None]


def environment_dataset(cfg: Dict[str, Any]) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    nt, nv, nte = N_TRAIN, N_VAL, N_TEST; n = nt + nv + nte
    seed = DATA_SEED_BASE + int(stable_hash_dict(environment_key_dict(cfg), 8), 16) % 10_000_019
    rng = np.random.default_rng(seed)
    mask = _spatial_support_mask(cfg, rng)
    X = _correlated_images(n, float(cfg["spatial_correlation"]), rng)
    yclean = _teacher_clean_target(X, cfg, mask, rng)
    xmu = float(X[:nt].mean()); xsd = float(X[:nt].std() + 1e-8)
    X = ((X - xmu) / xsd).astype(np.float32)
    ym = float(yclean[:nt].mean()); ys = float(yclean[:nt].std() + 1e-8)
    yclean = ((yclean - ym) / ys).astype(np.float32)
    y = yclean + rng.normal(0, math.sqrt(1 / float(cfg["snr"])), size=yclean.shape).astype(np.float32)
    ym2 = float(y[:nt].mean()); ys2 = float(y[:nt].std() + 1e-8)
    y = ((y - ym2) / ys2).astype(np.float32)
    splits = {
        "train": (torch.from_numpy(X[:nt]), torch.from_numpy(y[:nt])),
        "val": (torch.from_numpy(X[nt:nt + nv]), torch.from_numpy(y[nt:nt + nv])),
        "test": (torch.from_numpy(X[nt + nv:]), torch.from_numpy(y[nt + nv:])),
    }
    if torch.cuda.is_available() and GPU_RESIDENT_DATA:
        splits = {k: (a.to(DEVICE, non_blocking=True), b.to(DEVICE, non_blocking=True)) for k, (a, b) in splits.items()}
    meta = {**environment_key_dict(cfg), "environment_seed": int(seed), "n_train": nt, "n_val": nv, "n_test": nte, "support_pixels": int(mask.sum())}
    return splits, meta


@torch.no_grad()
def eval_mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor, chunk: int = 2048) -> float:
    model.eval(); total = 0.0; count = 0
    for lo in range(0, len(X), chunk):
        hi = min(lo + chunk, len(X))
        if X.device.type == "cuda":
            xb = X[lo:hi]; yb = y[lo:hi]
        else:
            xb = X[lo:hi].to(DEVICE, non_blocking=True); yb = y[lo:hi].to(DEVICE, non_blocking=True)
        pred = model(xb); total += float(torch.sum((pred - yb) ** 2).item()); count += int(yb.numel())
    return total / max(count, 1)

# ==================================================================================================
# 3. EXACT STRUCTURAL COMPLETE-PATH ENGINE + PARAMETER-SHARING PUSHFORWARD
# ==================================================================================================

def _global_normalize(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dims = tuple(range(1, t.ndim)); s = t.sum(dim=dims); valid = torch.isfinite(s) & (s > 0)
    safe = torch.where(valid, s, torch.ones_like(s))
    out = t / safe.reshape((-1,) + ((1,) * (t.ndim - 1)))
    out = torch.where(valid.reshape((-1,) + ((1,) * (t.ndim - 1))), out, torch.zeros_like(out))
    return out, s


@torch.no_grad()
def cnn_log_partition(model: ConvTraceNet, source_sq: torch.Tensor, response_sq: List[torch.Tensor], q: float) -> torch.Tensor:
    a = torch.pow(torch.clamp(source_sq.to(torch.float64), min=0), float(q))
    a, s = _global_normalize(a); valid = torch.isfinite(s) & (s > 0)
    logscale = torch.where(valid, torch.log(torch.clamp(s, min=EPS)), torch.full_like(s, -torch.inf))
    for l, conv in enumerate(model.convs):
        w2q = torch.pow(torch.square(conv.weight.to(torch.float64)), float(q))
        a = F.conv2d(a, w2q, bias=None, stride=1, padding=0)
        if l < len(model.convs) - 1:
            a = a * torch.pow(torch.clamp(response_sq[l].to(torch.float64), min=0), float(q))
        a, sl = _global_normalize(a)
        good = valid & torch.isfinite(sl) & (sl > 0)
        logscale = torch.where(good, logscale + torch.log(torch.clamp(sl, min=EPS)), torch.full_like(logscale, -torch.inf))
        valid = good
    hout, wout = model.spatial_shapes()[-1]
    g2q = float((1 / (hout * wout)) ** (2 * float(q)))
    return torch.where(valid, logscale + math.log(g2q), torch.full_like(logscale, -torch.inf))


def cnn_topological_log_count(model: ConvTraceNet) -> float:
    h, w = model.spatial_shapes()[-1]; logn = math.log(float(model.channels[-1] * h * w))
    for conv in model.convs:
        logn += math.log(float(conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]))
    return float(logn)


def _entropy_prob(p: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    return -torch.where(p > 0, p * torch.log(torch.clamp(p, min=PATH_LOG_FLOOR)), torch.zeros_like(p)).sum(dim=dims)


@torch.no_grad()
def cnn_factor_state(model: ConvTraceNet, source_sq: torch.Tensor, response_sq: List[torch.Tensor], return_flows: bool = True) -> Dict[str, Any]:
    source_sq = source_sq.to(DEVICE, dtype=torch.float64)
    response_sq = [r.to(DEVICE, dtype=torch.float64) for r in response_sq]; B = source_sq.shape[0]
    logZ1 = cnn_log_partition(model, source_sq, response_sq, 1.0)
    logZ2 = cnn_log_partition(model, source_sq, response_sq, 2.0)
    valid = torch.isfinite(logZ1)

    alpha = []; a, _ = _global_normalize(source_sq); alpha.append(a)
    for l, conv in enumerate(model.convs):
        a = F.conv2d(a, torch.square(conv.weight.to(torch.float64)), bias=None, stride=1, padding=0)
        if l < len(model.convs) - 1: a = a * response_sq[l]
        a, _ = _global_normalize(a); alpha.append(a)

    beta = [None] * (len(model.convs) + 1)
    b = torch.ones_like(alpha[-1], dtype=torch.float64); b, _ = _global_normalize(b); beta[-1] = b
    for l in range(len(model.convs) - 1, -1, -1):
        target = beta[l + 1]
        if l < len(model.convs) - 1: target = target * response_sq[l]
        b = F.conv_transpose2d(target, torch.square(model.convs[l].weight.to(torch.float64)), bias=None, stride=1, padding=0)
        b, _ = _global_normalize(b); beta[l] = b

    node_flow = []
    for l in range(len(alpha)):
        v = alpha[l] * beta[l]; v, _ = _global_normalize(v); node_flow.append(v)

    occurrence, param, layer_metrics = [], [], []
    elog_edges = torch.zeros(B, dtype=torch.float64, device=DEVICE)
    max_occ = torch.zeros(B, dtype=torch.float64, device=DEVICE)
    max_push = torch.zeros(B, dtype=torch.float64, device=DEVICE)

    for l, conv in enumerate(model.convs):
        k = int(conv.kernel_size[0]); cout = conv.out_channels; cin = conv.in_channels
        patches = F.unfold(alpha[l], kernel_size=k, stride=1, padding=0)
        Hout, Wout = alpha[l + 1].shape[-2:]; L = Hout * Wout
        target = beta[l + 1]
        if l < len(model.convs) - 1: target = target * response_sq[l]
        target = target.reshape(B, cout, L)
        w2 = torch.square(conv.weight.to(torch.float64)).reshape(cout, cin * k * k)
        E = target[:, :, None, :] * w2[None, :, :, None] * patches[:, None, :, :]
        den = E.sum(dim=(1, 2, 3)); good = den > 0
        E = torch.where(good[:, None, None, None], E / torch.clamp(den[:, None, None, None], min=EPS), torch.zeros_like(E))
        E = E.reshape(B, cout, cin, k, k, Hout, Wout); P = E.sum(dim=(-1, -2))
        occurrence.append(E); param.append(P)
        occsum = E.sum(dim=(1, 2, 3, 4, 5, 6)); psum = P.sum(dim=(1, 2, 3, 4))
        max_occ = torch.maximum(max_occ, torch.abs(occsum - 1)); max_push = torch.maximum(max_push, torch.abs(psum - occsum))

        logw = torch.log(torch.clamp(torch.square(conv.weight.to(torch.float64)), min=PATH_LOG_FLOOR))
        logfactor = logw[None, :, :, :, :, None, None]
        if l < len(model.convs) - 1:
            logfactor = logfactor + torch.log(torch.clamp(response_sq[l], min=PATH_LOG_FLOOR))[:, :, None, None, None, :, :]
        elog_edges += torch.sum(torch.where(E > 0, E * logfactor, torch.zeros_like(E)), dim=(1, 2, 3, 4, 5, 6))

        Hocc = _entropy_prob(E, (1, 2, 3, 4, 5, 6)); Hparam = _entropy_prob(P, (1, 2, 3, 4))
        Pb = P[..., None, None]
        ratio = torch.where(E > 0, E / torch.clamp(Pb, min=PATH_LOG_FLOOR), torch.ones_like(E))
        Hreuse_direct = -torch.sum(torch.where(E > 0, E * torch.log(torch.clamp(ratio, min=PATH_LOG_FLOOR)), torch.zeros_like(E)), dim=(1, 2, 3, 4, 5, 6))
        Hreuse = Hocc - Hparam
        ntheta = cout * cin * k * k; m = Hout * Wout; nocc = ntheta * m
        Socc = math.log(nocc) - Hocc; Sparam = math.log(ntheta) - Hparam; Sreuse = math.log(m) - Hreuse
        eta_occ = torch.exp(torch.clamp(-Socc, min=-745, max=0)); eta_param = torch.exp(torch.clamp(-Sparam, min=-745, max=0)); eta_reuse = torch.exp(torch.clamp(-Sreuse, min=-745, max=0))
        occflat = E.reshape(B, -1); pflat = P.reshape(B, -1)
        layer_metrics.append({
            "H_occ": Hocc, "H_param": Hparam, "H_reuse": Hreuse,
            "S_occ": Socc, "S_param": Sparam, "S_reuse": Sreuse,
            "eta1_occ": eta_occ, "eta1_param": eta_param, "eta1_reuse": eta_reuse,
            "occurrence_cv": occflat.std(1, unbiased=False) / torch.clamp(occflat.mean(1), min=EPS),
            "parameter_cv": pflat.std(1, unbiased=False) / torch.clamp(pflat.mean(1), min=EPS),
            "entropy_identity_residual": Hocc - Hparam - Hreuse_direct,
            "deficit_identity_residual": Socc - Sparam - Sreuse,
            "eta_factorization_residual": eta_occ - eta_param * eta_reuse,
            "reuse_multiplicity": torch.full((B,), float(m), dtype=torch.float64, device=DEVICE),
        })

    p0 = node_flow[0]
    elogsource = torch.sum(torch.where(p0 > 0, p0 * torch.log(torch.clamp(source_sq, min=PATH_LOG_FLOOR)), torch.zeros_like(p0)), dim=(1, 2, 3))
    hout, wout = model.spatial_shapes()[-1]; logg2 = math.log((1 / (hout * wout)) ** 2)
    elogq = elogsource + elog_edges + logg2
    H = logZ1 - elogq; logN = cnn_topological_log_count(model); Sdef = logN - H
    logeta1 = H - logN; logeta2 = 2 * logZ1 - logZ2 - logN

    valid_np = valid.detach().cpu().numpy().astype(bool)
    def arr(x):
        y = x.detach().cpu().numpy().astype(float); y[~valid_np] = np.nan; return y

    return {
        "valid": valid_np, "logZ1": arr(logZ1), "logZ2": arr(logZ2), "S_path": arr(H),
        "entropy_deficit": arr(Sdef), "eta1": arr(torch.exp(torch.clamp(logeta1, min=-745, max=0))),
        "eta2": arr(torch.exp(torch.clamp(logeta2, min=-745, max=0))),
        "node_flow": [v.detach().cpu().numpy() for v in node_flow] if return_flows else [],
        "occurrence_flow": [e.detach().cpu().numpy() for e in occurrence] if return_flows else [],
        "parameter_flow": [p.detach().cpu().numpy() for p in param] if return_flows else [],
        "layer_metrics": [{k: arr(v) for k, v in d.items()} for d in layer_metrics],
        "occurrence_conservation_error": arr(max_occ), "pushforward_conservation_error": arr(max_push),
    }


@torch.no_grad()
def cnn_structural_state(model: ConvTraceNet, return_flows: bool = True) -> Dict[str, Any]:
    source = torch.ones((1, INPUT_CHANNELS, model.resolution, model.resolution), dtype=torch.float64, device=DEVICE)
    responses = []; shapes = model.spatial_shapes()
    for l, conv in enumerate(model.convs[:-1]):
        h, w = shapes[l + 1]; responses.append(torch.ones((1, conv.out_channels, h, w), dtype=torch.float64, device=DEVICE))
    st = cnn_factor_state(model, source, responses, return_flows=return_flows)
    if not st["valid"][0]: raise RuntimeError("degenerate structural path partition")
    hidden = list(range(max(len(model.convs) - 1, 0)))
    summary = {
        "logZ1": float(st["logZ1"][0]), "logZ2": float(st["logZ2"][0]), "S_path": float(st["S_path"][0]),
        "entropy_deficit": float(st["entropy_deficit"][0]), "eta1": float(st["eta1"][0]), "eta2": float(st["eta2"][0]),
        "occurrence_conservation_error": float(st["occurrence_conservation_error"][0]),
        "pushforward_conservation_error": float(st["pushforward_conservation_error"][0]),
    }
    for key in ["S_occ", "S_param", "S_reuse", "eta1_occ", "eta1_param", "eta1_reuse", "occurrence_cv", "parameter_cv"]:
        vals = [float(st["layer_metrics"][l][key][0]) for l in hidden]
        summary[f"hidden_mean_{key}"] = float(np.mean(vals)) if vals else np.nan
    summary["max_entropy_identity_residual"] = float(max(abs(d["entropy_identity_residual"][0]) for d in st["layer_metrics"]))
    summary["max_deficit_identity_residual"] = float(max(abs(d["deficit_identity_residual"][0]) for d in st["layer_metrics"]))
    summary["max_eta_factorization_residual"] = float(max(abs(d["eta_factorization_residual"][0]) for d in st["layer_metrics"]))
    if return_flows:
        summary.update({
            "node_flow": [v[0] for v in st["node_flow"]],
            "occurrence_flow": [e[0] for e in st["occurrence_flow"]],
            "parameter_flow": [p[0] for p in st["parameter_flow"]],
            "layer_metrics": [{k: float(v[0]) for k, v in d.items()} for d in st["layer_metrics"]],
        })
    return summary

# ==================================================================================================
# 4. SPATIALLY UNROLLED EXECUTION GRAPH + BCANw
# ==================================================================================================

@dataclass
class ExecutionGraph:
    n_nodes: int
    layer_shapes: List[Tuple[int, int, int]]
    offsets: List[int]
    edge_mag: Dict[Tuple[int, int], float]
    layer_of_edge: Dict[Tuple[int, int], int]


def _node_id(offset: int, c: int, y: int, x: int, H: int, W: int) -> int:
    return int(offset + c * H * W + y * W + x)


def execution_graph_from_occurrence(model: ConvTraceNet, occurrence_flow: List[np.ndarray], enrichment: bool) -> ExecutionGraph:
    shapes_hw = model.spatial_shapes()
    layer_shapes = [(model.channels[l], h, w) for l, (h, w) in enumerate(shapes_hw)]
    offsets = [0]
    for c, h, w in layer_shapes[:-1]: offsets.append(offsets[-1] + c * h * w)
    edge_mag: Dict[Tuple[int, int], float] = {}; layer_of_edge: Dict[Tuple[int, int], int] = {}
    for l, (conv, E0) in enumerate(zip(model.convs, occurrence_flow)):
        E = np.asarray(E0, float); E = E * (E.size if enrichment else 1.0)
        cin, Hin, Win = layer_shapes[l]; cout, Hout, Wout = layer_shapes[l + 1]; k = int(conv.kernel_size[0])
        for co in range(cout):
            for ci in range(cin):
                for dy in range(k):
                    for dx in range(k):
                        vals = E[co, ci, dy, dx]
                        for y in range(Hout):
                            sy = y + dy
                            for x in range(Wout):
                                sx = x + dx
                                u = _node_id(offsets[l], ci, sy, sx, Hin, Win)
                                v = _node_id(offsets[l + 1], co, y, x, Hout, Wout)
                                e = (u, v) if u < v else (v, u)
                                edge_mag[e] = float(max(vals[y, x], 0)); layer_of_edge[e] = l
    return ExecutionGraph(sum(c * h * w for c, h, w in layer_shapes), layer_shapes, offsets, edge_mag, layer_of_edge)


def metric_lengths(P: ExecutionGraph, power: float = METRIC_POWER) -> Dict[Tuple[int, int], float]:
    return {e: float((max(w, 0.0) + EPS) ** (-float(power))) for e, w in P.edge_mag.items()}


def distance_matrix(P: ExecutionGraph, lengths: Dict[Tuple[int, int], float]) -> np.ndarray:
    rows, cols, vals = [], [], []
    for (u, v), L in lengths.items():
        rows.extend([u, v]); cols.extend([v, u]); vals.extend([L, L])
    A = csr_matrix((vals, (rows, cols)), shape=(P.n_nodes, P.n_nodes), dtype=float)
    return np.asarray(csgraph_dijkstra(A, directed=False, unweighted=False), float)


def bcanw_scales(D: np.ndarray, n_scales: int) -> np.ndarray:
    vals = D[np.triu_indices(D.shape[0], k=1)]; vals = vals[np.isfinite(vals) & (vals > 0)]
    if not len(vals): return np.array([], float)
    lo = max(float(vals.min()) * 0.95, EPS); hi = float(vals.max()) * 1.005
    return np.geomspace(lo, hi, max(int(n_scales), 4)) if hi > lo else np.array([lo])


def node_strength(P: ExecutionGraph) -> np.ndarray:
    s = np.zeros(P.n_nodes, float)
    for (u, v), w in P.edge_mag.items(): s[u] += w; s[v] += w
    return s


def greedy_bcanw_coloring(D: np.ndarray, ell: float, order: np.ndarray) -> np.ndarray:
    color = np.full(D.shape[0], -1, int)
    for u in order:
        used = set(color[(D[u] >= float(ell)) & (color >= 0)].tolist()); c = 0
        while c in used: c += 1
        color[u] = c
    return color


def global_node_measure(node_flow: List[np.ndarray]) -> np.ndarray:
    x = np.concatenate([np.asarray(v, float).ravel() for v in node_flow]); x = np.clip(x, 0, None)
    x /= max(float(x.sum()), EPS); return x


def renyi_effective_fraction(probabilities: np.ndarray, q: float) -> float:
    p = np.asarray(probabilities, float); p = p[np.isfinite(p) & (p > 0)]
    if not len(p): return np.nan
    p = p / p.sum(); E = len(p)
    H = -np.sum(p * np.log(p)) if abs(float(q) - 1.0) < 1e-12 else math.log(np.sum(p ** q)) / (1 - q)
    return float(math.exp(H) / E)


def box_mass_lacunarity(colors: np.ndarray, node_measure: np.ndarray) -> Tuple[np.ndarray, float]:
    nb = int(colors.max()) + 1
    mass = np.bincount(colors, weights=node_measure, minlength=nb).astype(float)
    mean = max(float(mass.mean()), EPS); lac = float(np.mean(mass * mass) / (mean * mean))
    return mass, lac


def compress_staircase(radii: Sequence[float], NB: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    radii = np.asarray(radii, float); NB = np.asarray(NB, float); out_r, out_n = [], []; i = 0
    while i < len(NB):
        j = i + 1
        while j < len(NB) and NB[j] == NB[i]: j += 1
        out_r.append(math.sqrt(radii[i] * radii[j - 1])); out_n.append(NB[i]); i = j
    return np.asarray(out_r), np.asarray(out_n)


def covering_scale(radii: Sequence[float], NB: Sequence[int], N: int, target_fraction: float) -> float:
    x = np.asarray(radii, float); y = np.asarray(NB, float) / float(N)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0); x = x[good]; y = y[good]
    if len(x) < 2 or target_fraction > y.max() or target_fraction < y.min(): return np.nan
    order = np.argsort(y); ys = y[order]; xs = np.log(x[order])
    return float(np.exp(np.interp(target_fraction, ys, xs)))


def _linear_fit_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    A = np.column_stack([np.ones(len(x)), x]); coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef; rss = float(np.sum((y - pred) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - rss / sst) if sst > 0 else np.nan
    return float(coef[0]), float(coef[1]), r2, max(rss, 1e-300)


def _bic(rss: float, n: int, k: int) -> float:
    return float(n * math.log(max(rss / max(n, 1), 1e-300)) + k * math.log(max(n, 2)))


def fit_credible_scaling_window(radii: Sequence[float], NB: Sequence[int]) -> Dict[str, Any]:
    r, n = compress_staircase(radii, NB)
    good = np.isfinite(r) & np.isfinite(n) & (r > 0) & (n >= 2); r = r[good]; n = n[good]
    if len(r) < SCALING_MIN_POINTS:
        return {"scaling_credible": False, "d_B": np.nan, "scaling_reason": "too_few_staircase_points"}
    x = np.log(r); y = np.log(n); best = None
    for i in range(0, len(x)):
        for j in range(i + SCALING_MIN_POINTS, len(x) + 1):
            xx = x[i:j]; yy = y[i:j]; nn = n[i:j]
            span = float((xx.max() - xx.min()) / math.log(10))
            nb_ratio = float(nn.max() / max(nn.min(), EPS))
            if span < SCALING_MIN_SPAN_DECADES or nb_ratio < SCALING_MIN_NB_RATIO: continue
            a, slope, r2, rss1 = _linear_fit_stats(xx, yy); d = -slope
            if not np.isfinite(d) or d < SCALING_MIN_DIM or r2 < SCALING_R2_MIN: continue
            local = -np.diff(yy) / np.diff(xx); local = local[np.isfinite(local)]
            slope_cv = float(np.std(local) / max(abs(np.mean(local)), EPS)) if len(local) else np.inf
            if slope_cv > SCALING_MAX_SLOPE_CV: continue
            Aq = np.column_stack([np.ones(len(xx)), xx, xx * xx]); cq, *_ = np.linalg.lstsq(Aq, yy, rcond=None)
            rss2 = max(float(np.sum((yy - Aq @ cq) ** 2)), 1e-300)
            delta_bic = _bic(rss1, len(xx), 2) - _bic(rss2, len(xx), 3)  # >0 favors curvature
            if delta_bic > SCALING_MAX_DELTA_BIC: continue
            rec = {
                "scaling_credible": True, "d_B": float(d), "scaling_r2": float(r2),
                "scaling_n_points": int(len(xx)), "scaling_span_decades": span,
                "scaling_nb_ratio": nb_ratio, "scaling_slope_cv": slope_cv,
                "scaling_delta_bic_quadratic": float(delta_bic),
                "scaling_r_min": float(np.exp(xx.min())), "scaling_r_max": float(np.exp(xx.max())),
                "scaling_reason": "credible",
            }
            score = (len(xx), span, r2, -slope_cv)
            if best is None or score > best[0]: best = (score, rec)
    if best is None:
        return {"scaling_credible": False, "d_B": np.nan, "scaling_reason": "no_window_passed_credibility"}
    return best[1]


def analyze_execution_geometry(model: ConvTraceNet, state: Dict[str, Any], n_scales: int = BCANW_SCALES_LONG) -> Tuple[Dict[str, Any], pd.DataFrame]:
    Pgeom = execution_graph_from_occurrence(model, state["occurrence_flow"], enrichment=True)
    if Pgeom.n_nodes > BCANW_MAX_NODES:
        return {"geometry_skipped": True, "geometry_skip_reason": f"n_nodes={Pgeom.n_nodes}>cap={BCANW_MAX_NODES}", "n_graph_nodes": Pgeom.n_nodes}, pd.DataFrame()
    lengths = metric_lengths(Pgeom, METRIC_POWER); D = distance_matrix(Pgeom, lengths); radii = bcanw_scales(D, n_scales)
    order = np.lexsort((np.arange(Pgeom.n_nodes), -node_strength(Pgeom))); node_mu = global_node_measure(state["node_flow"])
    rows = []; NB = []
    for ell in radii:
        colors = greedy_bcanw_coloring(D, float(ell), order); nb = int(colors.max()) + 1; NB.append(nb)
        mass, lac = box_mass_lacunarity(colors, node_mu)
        rec = {"radius": float(ell), "N_B": nb, "box_fraction": float(nb / Pgeom.n_nodes), "lacunarity": lac}
        for q in COARSE_RENYI_Q: rec[f"renyi_eff_frac_q{q:g}"] = renyi_effective_fraction(mass, q)
        rows.append(rec)
    scales = pd.DataFrame(rows); summary = {"geometry_skipped": False, "n_graph_nodes": int(Pgeom.n_nodes), "n_scales": int(len(radii))}
    summary.update(fit_credible_scaling_window(radii, NB))
    for frac in [0.50, 0.25, 0.125, 0.0625]:
        summary[f"xi_{str(frac).replace('.', 'p')}"] = covering_scale(radii, NB, Pgeom.n_nodes, frac)
    if len(scales):
        summary["lacunarity_median"] = float(np.nanmedian(scales["lacunarity"]))
        for q in COARSE_RENYI_Q:
            summary[f"renyi_eff_frac_q{q:g}_median"] = float(np.nanmedian(scales[f"renyi_eff_frac_q{q:g}"]))
    return summary, scales

# ==================================================================================================

# ==================================================================================================
# 5. C2 CONDITIONAL REALIZATION ENGINE + FIXED MANIFESTS
# ==================================================================================================

from scipy.stats import ks_2samp, rankdata

LOCAL_SLOPE_WINDOW = int(os.environ.get("PATHFLOW_CNN_C3_LOCAL_SLOPE_WINDOW", "7"))
LOCAL_SLOPE_R2_MIN = float(os.environ.get("PATHFLOW_CNN_C3_LOCAL_SLOPE_R2_MIN", "0.90"))
LOCAL_DIMENSION_NODES = int(os.environ.get("PATHFLOW_CNN_C3_LOCAL_DIMENSION_NODES", "512"))
LOCAL_RADII_MAX = int(os.environ.get("PATHFLOW_CNN_C3_LOCAL_RADII_MAX", "40"))
LOCAL_COMMON_MIN_POINTS = int(os.environ.get("PATHFLOW_CNN_C3_LOCAL_COMMON_MIN_POINTS", "7"))
LOCAL_COMMON_MIN_SPAN_DECADES = float(os.environ.get("PATHFLOW_CNN_C3_LOCAL_COMMON_MIN_SPAN_DECADES", "0.50"))
LOCAL_COMMON_SUPPORT_GROWTH = float(os.environ.get("PATHFLOW_CNN_C3_LOCAL_COMMON_SUPPORT_GROWTH", "4.0"))
LOCAL_COMMON_MASS_GROWTH = float(os.environ.get("PATHFLOW_CNN_C3_LOCAL_COMMON_MASS_GROWTH", "3.0"))
LOCAL_COMMON_R2_MIN = float(os.environ.get("PATHFLOW_CNN_C3_LOCAL_COMMON_R2_MIN", "0.95"))
RENORM_TARGET_BOX_FRACTIONS = [0.50, 0.25, 0.125, 0.0625]


def architecture_panel() -> pd.DataFrame:
    rows = []
    for i, (aid, hidden, kernels) in enumerate(FROZEN_ARCHITECTURES):
        d = architecture_descriptors(hidden, kernels)
        rows.append({
            "architecture_index": i, "arch_id": aid,
            "hidden_channels": json.dumps(hidden), "kernel_sizes": json.dumps(kernels),
            **d,
        })
    out = pd.DataFrame(rows)
    if len(out) != 12 or out.arch_id.nunique() != 12:
        raise RuntimeError("C2 frozen architecture panel must contain exactly 12 unique rows")
    bad = out[~out.valid.astype(bool)]
    if len(bad):
        raise RuntimeError(f"Frozen C2 architecture panel contains invalid execution graphs: {bad.arch_id.tolist()}")
    return out


def training_manifest() -> pd.DataFrame:
    panel = architecture_panel(); rows = []
    for _, a in panel.iterrows():
        for seed in SEEDS:
            rid = run_id(str(a.arch_id), int(seed))
            rows.append({
                "run_id": rid, "arch_id": str(a.arch_id), "architecture_index": int(a.architecture_index),
                "seed_index": int(seed), "activation": ACTIVATION, "optimizer": OPTIMIZER,
                "learning_rate": ADAM_LR, "weight_decay": WEIGHT_DECAY,
                "environment_id": CANONICAL_ENV["env_id"],
            })
    return pd.DataFrame(rows)


def condition_manifest() -> pd.DataFrame:
    rows = []
    for _, r in training_manifest().iterrows():
        rid = str(r.run_id)
        for ep in STATE_GRID:
            label = epoch_label(ep)
            n_inputs = DEEP_INPUTS if ep in (0, FULL_EPOCHS) else LONGITUDINAL_INPUTS
            rows.append({**r.to_dict(), "checkpoint": label, "epoch_nominal": int(ep), "n_inputs": int(n_inputs),
                         "job_id": f"{rid}__{label}"})
        rows.append({**r.to_dict(), "checkpoint": "best", "epoch_nominal": -1, "n_inputs": DEEP_INPUTS,
                     "job_id": f"{rid}__best"})
    return pd.DataFrame(rows)


def geometry_manifest() -> pd.DataFrame:
    rows = []
    for _, r in training_manifest().iterrows():
        for label in PRINCIPAL_STATES:
            rows.append({**r.to_dict(), "checkpoint": label, "job_id": f"{r.run_id}__{label}__geometry"})
    return pd.DataFrame(rows)


def input_geometry_manifest() -> pd.DataFrame:
    rows = []
    train = training_manifest()
    for _, r in train[train.arch_id.isin(SENTINEL_ARCH_IDS)].iterrows():
        for label in PRINCIPAL_STATES:
            for j in range(INPUT_GEOMETRY_INPUTS):
                rows.append({**r.to_dict(), "checkpoint": label, "input_panel_position": int(j),
                             "job_id": f"{r.run_id}__{label}__X{j:03d}__inputgeom"})
    return pd.DataFrame(rows)


def _arch_row(arch_id: str) -> pd.Series:
    p = architecture_panel(); q = p[p.arch_id.astype(str) == str(arch_id)]
    if len(q) != 1: raise KeyError(f"unknown arch_id={arch_id!r}")
    return q.iloc[0]


def build_model(arch_id: str, seed_index: int) -> ConvTraceNet:
    return _arch_from_row(_arch_row(arch_id), int(seed_index))


def calibration_batch(arch_id: str, seed_index: int, n: int = 512) -> torch.Tensor:
    # Environment-independent calibration is deliberately keyed only by architecture and training seed.
    seed = CALIBRATION_SEED_BASE + int(stable_hash_dict({"arch": arch_id, "seed": int(seed_index)}, 8), 16) % 10_000_019
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((int(n), INPUT_CHANNELS, RESOLUTION, RESOLUTION), generator=g, dtype=torch.float32)


def checkpoint_for_label(rid: str, label: str) -> Path:
    if str(label) == "best": return best_checkpoint_path(rid)
    if not str(label).startswith("t"): raise ValueError(label)
    return checkpoint_path(rid, int(str(label)[1:]))


def load_model_checkpoint(arch_id: str, seed_index: int, label: str) -> Tuple[ConvTraceNet, Dict[str, Any]]:
    rid = run_id(arch_id, seed_index); path = checkpoint_for_label(rid, label)
    if not path.exists():
        raise FileNotFoundError(f"required C2 checkpoint missing: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(arch_id, seed_index)
    state = ck.get("model_state", ck.get("state_dict"))
    if state is None: raise KeyError(f"checkpoint {path} has no model_state/state_dict")
    model.load_state_dict(state, strict=True); model.to(DEVICE); model.eval()
    return model, ck


def finite_secant_torch64(name: str, z: torch.Tensor, z0: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    z = z.to(torch.float64); z0 = z0.to(torch.float64); dz = z - z0
    if name == "relu": f = lambda q: torch.clamp_min(q, 0.0)
    elif name == "tanh": f = torch.tanh
    elif name == "silu": f = F.silu
    else: raise ValueError(name)
    out = torch.empty_like(dz); mask = torch.abs(dz) > tol
    if mask.any(): out[mask] = (f(z[mask]) - f(z0[mask])) / dz[mask]
    if (~mask).any():
        q = .5 * (z[~mask] + z0[~mask])
        if name == "relu": der = (q > 0).to(q.dtype)
        elif name == "tanh":
            t = torch.tanh(q); der = 1 - t * t
        else:
            s = torch.sigmoid(q); der = s + q * s * (1 - s)
        out[~mask] = der
    return out


@torch.no_grad()
def forward_with_states(model: ConvTraceNet, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    h = x; zs = []; hs = [x]
    for i, conv in enumerate(model.convs):
        z = conv(h); zs.append(z)
        h = model.activation(z) if i < len(model.convs) - 1 else z
        hs.append(h)
    return h.mean(dim=(-1, -2)), zs, hs


@torch.no_grad()
def conditional_response_sq(model: ConvTraceNet, X: torch.Tensor, X0: torch.Tensor) -> List[torch.Tensor]:
    model.eval(); _, z, _ = forward_with_states(model, X.to(DEVICE)); _, z0, _ = forward_with_states(model, X0.to(DEVICE))
    return [torch.square(finite_secant_torch64(model.activation_name, a, b)) for a, b in zip(z[:-1], z0[:-1])]


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3 or np.nanstd(a[m]) <= 0 or np.nanstd(b[m]) <= 0: return np.nan
    return float(spearmanr(a[m], b[m]).statistic)


def _entropy_np(p: np.ndarray) -> float:
    p = np.asarray(p, float).ravel(); p = p[np.isfinite(p) & (p > 0)]
    return -float(np.sum(p * np.log(p))) if len(p) else np.nan


def _layer_decomposition_from_flow(E: np.ndarray) -> Dict[str, float]:
    E = np.asarray(E, float); E = E / max(float(E.sum()), EPS)
    P = E.sum(axis=(-1, -2)); P = P / max(float(P.sum()), EPS)
    Hocc = _entropy_np(E); Hpar = _entropy_np(P); Hreuse = Hocc - Hpar
    ntheta = int(np.prod(P.shape)); m = int(E.shape[-1] * E.shape[-2]); nocc = ntheta * m
    Socc = math.log(max(nocc, 1)) - Hocc; Sparam = math.log(max(ntheta, 1)) - Hpar; Sreuse = math.log(max(m, 1)) - Hreuse
    return {
        "H_occ": Hocc, "H_param": Hpar, "H_reuse": Hreuse,
        "S_occ": Socc, "S_param": Sparam, "S_reuse": Sreuse,
        "eta1_occ": math.exp(-Socc), "eta1_param": math.exp(-Sparam), "eta1_reuse": math.exp(-Sreuse),
        "occurrence_cv": float(np.std(E) / max(np.mean(E), EPS)),
        "parameter_cv": float(np.std(P) / max(np.mean(P), EPS)),
        "deficit_identity_residual": Socc - Sparam - Sreuse,
    }


def _mutual_information_from_running(sum_prob: np.ndarray, mean_entropy: float, n: int) -> float:
    if n <= 0: return np.nan
    avg = np.asarray(sum_prob, float) / float(n); avg /= max(float(avg.sum()), EPS)
    return float(_entropy_np(avg) - mean_entropy / float(n))


def conditional_distance_metrics(model: ConvTraceNet, st: Dict[str, Any], source_sq: torch.Tensor,
                                 response_sq: List[torch.Tensor], structural: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact KL(mu^c||p^S), Bhattacharyya coefficient and Hellinger distance without path enumeration."""
    B = len(st["valid"]); elog_ratio = np.zeros(B, float)
    src = source_sq.detach().cpu().numpy().astype(float)
    for b in range(B):
        if not st["valid"][b]:
            elog_ratio[b] = np.nan; continue
        p0 = np.asarray(st["node_flow"][0][b], float)
        ls = np.log(np.clip(src[b], PATH_LOG_FLOOR, None))
        elog_ratio[b] = float(np.sum(np.where(p0 > 0, p0 * ls, 0.0)))
        for l, rr in enumerate(response_sq):
            pr = np.asarray(st["node_flow"][l + 1][b], float)
            rv = rr[b].detach().cpu().numpy().astype(float)
            elog_ratio[b] += float(np.sum(np.where(pr > 0, pr * np.log(np.clip(rv, PATH_LOG_FLOOR, None)), 0.0)))
    kl = elog_ratio - np.asarray(st["logZ1"], float) + float(structural["logZ1"])

    # Z_cross = sum sqrt(Qc Qs): source/response factors enter with square root while squared weights remain unchanged.
    cross_source = torch.sqrt(torch.clamp(source_sq.to(DEVICE, dtype=torch.float64), min=0.0))
    cross_resp = [torch.sqrt(torch.clamp(r.to(DEVICE, dtype=torch.float64), min=0.0)) for r in response_sq]
    logzc = cnn_log_partition(model, cross_source, cross_resp, 1.0).detach().cpu().numpy().astype(float)
    log_bc = logzc - 0.5 * (np.asarray(st["logZ1"], float) + float(structural["logZ1"]))
    bc = np.exp(np.clip(log_bc, -745, 0)); bc[~np.asarray(st["valid"], bool)] = np.nan
    h2 = np.clip(1.0 - bc, 0.0, 1.0); hell = np.sqrt(h2)
    return kl, bc, hell


def conditional_batch_analysis(model: ConvTraceNet, X: torch.Tensor, input_indices: np.ndarray,
                               structural: Dict[str, Any], checkpoint: str, rid: str,
                               arch_id: str, seed_index: int) -> Dict[str, Any]:
    """Evaluate individual conditional measures first, then population summaries.

    This function intentionally never substitutes Y(E[F_x]) for E[Y(F_x)].  Input-level observables are
    written before population flow is formed; both objects are returned separately.
    """
    rows: List[Dict[str, Any]] = []; layer_rows: List[Dict[str, Any]] = []
    pop_node = pop_occ = pop_par = None; valid_n = 0
    spec_sum_node = spec_sum_channel = spec_sum_spatial = None
    spec_entropy_node = spec_entropy_channel = spec_entropy_spatial = None
    n_total = len(input_indices)

    bar = tqdm(range(0, n_total, CONDITIONAL_BATCH), desc=f"condition {rid} {checkpoint}", leave=False,
               disable=not TQDM_ENABLED, dynamic_ncols=True)
    for lo in bar:
        hi = min(lo + CONDITIONAL_BATCH, n_total); ids = np.asarray(input_indices[lo:hi], int)
        if X.device.type == "cuda": xb = X[torch.as_tensor(ids, device=X.device, dtype=torch.long)]
        else: xb = X[torch.as_tensor(ids, dtype=torch.long)].to(DEVICE, non_blocking=True)
        x0 = torch.zeros_like(xb); source_sq = torch.square((xb - x0).to(torch.float64))
        response_sq = conditional_response_sq(model, xb, x0)
        st = cnn_factor_state(model, source_sq, response_sq, return_flows=True)
        kl, bc, hell = conditional_distance_metrics(model, st, source_sq, response_sq, structural)

        for bi, data_index in enumerate(ids):
            valid = bool(st["valid"][bi])
            rec: Dict[str, Any] = {
                "run_id": rid, "arch_id": arch_id, "seed_index": int(seed_index), "checkpoint": checkpoint,
                "input_index": int(data_index), "valid": valid,
                "S_path": float(st["S_path"][bi]), "entropy_deficit": float(st["entropy_deficit"][bi]),
                "eta1": float(st["eta1"][bi]), "eta2": float(st["eta2"][bi]),
                "kl_to_structural": float(kl[bi]), "bhattacharyya_to_structural": float(bc[bi]),
                "hellinger_to_structural": float(hell[bi]),
                "occurrence_conservation_error": float(st["occurrence_conservation_error"][bi]),
                "pushforward_conservation_error": float(st["pushforward_conservation_error"][bi]),
            }
            if valid:
                hidden_ids = list(range(max(len(model.convs) - 1, 0)))
                for key in ["S_occ", "S_param", "S_reuse", "eta1_occ", "eta1_param", "eta1_reuse", "occurrence_cv", "parameter_cv"]:
                    vals = [float(st["layer_metrics"][l][key][bi]) for l in hidden_ids]
                    rec[f"hidden_mean_{key}"] = float(np.mean(vals)) if vals else np.nan
                rocc = []; rpar = []
                for l in range(len(model.convs)):
                    rocc.append(_safe_spearman(st["occurrence_flow"][l][bi], structural["occurrence_flow"][l]))
                    rpar.append(_safe_spearman(st["parameter_flow"][l][bi], structural["parameter_flow"][l]))
                    lm = {k: float(v[bi]) for k, v in st["layer_metrics"][l].items()}
                    layer_rows.append({"run_id": rid, "arch_id": arch_id, "seed_index": int(seed_index),
                                       "checkpoint": checkpoint, "input_index": int(data_index), "layer": int(l),
                                       "rho_occurrence_to_structural": rocc[-1], "rho_parameter_to_structural": rpar[-1], **lm})
                rec["rho_occurrence_to_structural"] = float(np.nanmean(rocc))
                rec["rho_parameter_to_structural"] = float(np.nanmean(rpar))
                rec["rho_parameter_minus_occurrence"] = rec["rho_parameter_to_structural"] - rec["rho_occurrence_to_structural"]

                if pop_node is None:
                    pop_node = [np.zeros_like(v[bi], dtype=np.float64) for v in st["node_flow"]]
                    pop_occ = [np.zeros_like(v[bi], dtype=np.float64) for v in st["occurrence_flow"]]
                    pop_par = [np.zeros_like(v[bi], dtype=np.float64) for v in st["parameter_flow"]]
                    spec_sum_node = [np.zeros_like(v[bi], dtype=np.float64) for v in st["node_flow"]]
                    spec_sum_channel = [np.zeros((v.shape[1],), dtype=np.float64) for v in st["node_flow"]]
                    spec_sum_spatial = [np.zeros(v.shape[-2:], dtype=np.float64) for v in st["node_flow"]]
                    spec_entropy_node = [0.0 for _ in st["node_flow"]]
                    spec_entropy_channel = [0.0 for _ in st["node_flow"]]
                    spec_entropy_spatial = [0.0 for _ in st["node_flow"]]
                for l, v in enumerate(st["node_flow"]):
                    p = np.asarray(v[bi], float); pop_node[l] += p; spec_sum_node[l] += p
                    pc = p.sum(axis=(-1, -2)); ps = p.sum(axis=0)
                    spec_sum_channel[l] += pc; spec_sum_spatial[l] += ps
                    spec_entropy_node[l] += _entropy_np(p); spec_entropy_channel[l] += _entropy_np(pc); spec_entropy_spatial[l] += _entropy_np(ps)
                for l, v in enumerate(st["occurrence_flow"]): pop_occ[l] += np.asarray(v[bi], float)
                for l, v in enumerate(st["parameter_flow"]): pop_par[l] += np.asarray(v[bi], float)
                valid_n += 1
            rows.append(rec)

    idf = pd.DataFrame(rows); lidf = pd.DataFrame(layer_rows)
    summary: Dict[str, Any] = {
        "run_id": rid, "arch_id": arch_id, "seed_index": int(seed_index), "checkpoint": checkpoint,
        "n_inputs": int(n_total), "n_valid": int(valid_n), "conditional_valid_fraction": float(valid_n / max(n_total, 1)),
    }
    for k, v in structural.items():
        if np.isscalar(v) and isinstance(v, (int, float, np.integer, np.floating)):
            summary[f"structural_{k}"] = float(v)
    if len(idf):
        metrics = ["S_path", "entropy_deficit", "eta1", "eta2", "kl_to_structural", "hellinger_to_structural",
                   "rho_occurrence_to_structural", "rho_parameter_to_structural", "rho_parameter_minus_occurrence",
                   "hidden_mean_S_occ", "hidden_mean_S_param", "hidden_mean_S_reuse", "hidden_mean_eta1_occ",
                   "hidden_mean_eta1_param", "hidden_mean_eta1_reuse", "hidden_mean_occurrence_cv", "hidden_mean_parameter_cv"]
        q = idf[idf.valid.astype(bool)] if "valid" in idf else idf
        for c in [x for x in metrics if x in q.columns]:
            vals = pd.to_numeric(q[c], errors="coerce").dropna()
            if len(vals):
                summary[f"conditional_{c}_median"] = float(vals.median())
                summary[f"conditional_{c}_mean"] = float(vals.mean())
                summary[f"conditional_{c}_q25"] = float(vals.quantile(.25))
                summary[f"conditional_{c}_q75"] = float(vals.quantile(.75))

    population_state = None; population_layers = pd.DataFrame(); specialization = pd.DataFrame()
    if valid_n > 0:
        pop_node = [v / valid_n for v in pop_node]; pop_occ = [v / valid_n for v in pop_occ]; pop_par = [v / valid_n for v in pop_par]
        population_state = {"node_flow": pop_node, "occurrence_flow": pop_occ, "parameter_flow": pop_par}
        prows = []
        for l, E in enumerate(pop_occ): prows.append({"layer": l, **_layer_decomposition_from_flow(E)})
        population_layers = pd.DataFrame(prows)
        hidden_pop = population_layers[population_layers.layer < max(len(model.convs) - 1, 0)]
        for c in ["S_occ", "S_param", "S_reuse", "eta1_occ", "eta1_param", "eta1_reuse", "occurrence_cv", "parameter_cv"]:
            summary[f"population_hidden_mean_{c}"] = float(hidden_pop[c].mean()) if len(hidden_pop) else np.nan
            medcol = f"conditional_hidden_mean_{c}_median"
            if medcol in summary:
                summary[f"noncommutativity_{c}"] = float(summary[f"population_hidden_mean_{c}"] - summary[medcol])
        cocc = np.concatenate([e.ravel() for e in pop_occ]); socc = np.concatenate([e.ravel() for e in structural["occurrence_flow"]])
        cpar = np.concatenate([p.ravel() for p in pop_par]); spar = np.concatenate([p.ravel() for p in structural["parameter_flow"]])
        summary["population_occurrence_spearman_to_structural"] = _safe_spearman(cocc, socc)
        summary["population_parameter_spearman_to_structural"] = _safe_spearman(cpar, spar)

        srows = []
        for l in range(len(pop_node)):
            inode = _mutual_information_from_running(spec_sum_node[l], spec_entropy_node[l], valid_n)
            ich = _mutual_information_from_running(spec_sum_channel[l], spec_entropy_channel[l], valid_n)
            isp = _mutual_information_from_running(spec_sum_spatial[l], spec_entropy_spatial[l], valid_n)
            srows.append({"layer": l, "I_input_node": inode, "I_input_channel": ich, "I_input_spatial": isp})
        specialization = pd.DataFrame(srows)
        hidden_spec = specialization[(specialization.layer > 0) & (specialization.layer < len(model.convs))]
        for c in ["I_input_node", "I_input_channel", "I_input_spatial"]:
            summary[f"hidden_mean_{c}"] = float(hidden_spec[c].mean()) if len(hidden_spec) else np.nan

    return {"input_df": idf, "layer_input_df": lidf, "summary": summary,
            "population_state": population_state, "population_layers": population_layers,
            "specialization": specialization}

# ==================================================================================================
# 6. TRAINING — EXACT MID-EPOCH RESUME, COMMON CHECKPOINT GRID, BEST VALIDATION STATE
# ==================================================================================================


def _optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=ADAM_LR, weight_decay=WEIGHT_DECAY)


def _save_science_checkpoint(path: Path, model: nn.Module, rid: str, arch_id: str, seed_index: int,
                             epoch: int, val_mse: float, calibration_scales: Sequence[float],
                             initial_digest: str, best: bool = False):
    atomic_torch_save({
        "schema": "cnn_c1_checkpoint_v1", "run_id": rid, "arch_id": arch_id, "seed_index": int(seed_index),
        "epoch": int(epoch), "val_mse": float(val_mse), "best": bool(best),
        "activation": ACTIVATION, "optimizer": OPTIMIZER, "learning_rate": ADAM_LR,
        "environment": CANONICAL_ENV, "calibration_scales": list(map(float, calibration_scales)),
        "initial_digest": initial_digest, "model_state": state_dict_cpu(model),
    }, path)


def train_one(row: pd.Series, splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    rid = str(row.run_id); arch_id = str(row.arch_id); seed_index = int(row.seed_index)
    summary_path = run_part(rid, "summary"); history_path = run_part(rid, "history")
    required = [checkpoint_path(rid, e) for e in STATE_GRID] + [best_checkpoint_path(rid)]
    if RESUME and summary_path.exists() and all(p.exists() for p in required):
        trace(f"[train] skip complete {rid}"); return safe_read_csv(summary_path).iloc[0].to_dict()

    model = build_model(arch_id, seed_index); Xtr, ytr = splits["train"]; Xv, yv = splits["val"]; Xt, yt = splits["test"]
    opt = _optimizer_for(model); rp = resume_path(rid); history_rows: List[Dict[str, Any]] = []
    start_epoch = 1; start_batch = 0; best_val = float("inf"); best_epoch = 0
    calibration_scales: List[float]; initial_digest: str

    if RESUME and rp.exists():
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"], strict=True); model.to(DEVICE)
        opt.load_state_dict(ck["optimizer_state"])
        for state in opt.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v): state[k] = v.to(DEVICE)
        start_epoch = int(ck["epoch"]); start_batch = int(ck["next_batch"])
        best_val = float(ck["best_val"]); best_epoch = int(ck["best_epoch"])
        calibration_scales = list(map(float, ck["calibration_scales"])); initial_digest = str(ck["initial_digest"])
        history_rows = list(ck.get("history_rows", []))
        trace(f"[train] resuming {rid}: epoch={start_epoch}, next_batch={start_batch}, best={best_val:.6g}@{best_epoch}")
    else:
        cX = calibration_batch(arch_id, seed_index)
        calibration_scales = calibrate_initial_signal(model, cX); initial_digest = tensor_state_digest(state_dict_cpu(model))
        val0 = eval_mse(model, Xv, yv); test0 = eval_mse(model, Xt, yt)
        best_val = val0; best_epoch = 0
        _save_science_checkpoint(checkpoint_path(rid, 0), model, rid, arch_id, seed_index, 0, val0, calibration_scales, initial_digest)
        _save_science_checkpoint(best_checkpoint_path(rid), model, rid, arch_id, seed_index, 0, val0, calibration_scales, initial_digest, best=True)
        history_rows.append({"run_id": rid, "arch_id": arch_id, "seed_index": seed_index, "epoch": 0,
                             "train_mse": np.nan, "val_mse": val0, "test_mse_if_state": test0})
        atomic_csv(pd.DataFrame(history_rows), history_path)
        trace(f"[train] {rid} calibrated | scales={np.round(calibration_scales,4).tolist()} | digest={initial_digest} | val0={val0:.6g}")

    n_batches = int(math.ceil(len(Xtr) / BATCH_SIZE)); run_start = time.perf_counter()
    epoch_bar = tqdm(range(start_epoch, FULL_EPOCHS + 1), desc=f"epochs {rid}", leave=False,
                     disable=not TQDM_ENABLED, dynamic_ncols=True)
    for epoch in epoch_bar:
        model.train(); total_loss = 0.0; total_n = 0
        batches = list(deterministic_batches(len(Xtr), epoch, seed_index, BATCH_SIZE))
        b0 = start_batch if epoch == start_epoch else 0
        batch_bar = tqdm(range(b0, len(batches)), desc=f"{rid} e{epoch:03d} batches", leave=False,
                         disable=not TQDM_ENABLED, dynamic_ncols=True)
        for bj in batch_bar:
            ids = batches[bj]; idx = torch.as_tensor(ids, dtype=torch.long, device=Xtr.device)
            xb = Xtr[idx] if Xtr.device.type == "cuda" else Xtr[idx].to(DEVICE, non_blocking=True)
            yb = ytr[idx] if ytr.device.type == "cuda" else ytr[idx].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True); pred = model(xb); loss = F.mse_loss(pred, yb); loss.backward(); opt.step()
            bs = int(len(ids)); total_loss += float(loss.item()) * bs; total_n += bs
            if (bj + 1) % SAVE_RESUME_EVERY_BATCHES == 0 or STOP_REQUESTED or soft_walltime_due():
                atomic_torch_save({
                    "schema": "cnn_c1_resume_v1", "run_id": rid, "epoch": int(epoch), "next_batch": int(bj + 1),
                    "best_val": best_val, "best_epoch": best_epoch, "calibration_scales": calibration_scales,
                    "initial_digest": initial_digest, "history_rows": history_rows,
                    "model_state": state_dict_cpu(model), "optimizer_state": opt.state_dict(),
                }, rp)
            if STOP_REQUESTED or soft_walltime_due():
                trace(f"[train] safe stop requested for {rid} at epoch {epoch} batch {bj+1}/{len(batches)}")
                raise KeyboardInterrupt("safe stop after resume checkpoint")

        train_mse = total_loss / max(total_n, 1); val = eval_mse(model, Xv, yv)
        if val < best_val:
            best_val = float(val); best_epoch = int(epoch)
            _save_science_checkpoint(best_checkpoint_path(rid), model, rid, arch_id, seed_index, epoch, val, calibration_scales, initial_digest, best=True)
        test_if = np.nan
        if epoch in STATE_GRID:
            test_if = eval_mse(model, Xt, yt)
            _save_science_checkpoint(checkpoint_path(rid, epoch), model, rid, arch_id, seed_index, epoch, val, calibration_scales, initial_digest)
        history_rows.append({"run_id": rid, "arch_id": arch_id, "seed_index": seed_index, "epoch": int(epoch),
                             "train_mse": float(train_mse), "val_mse": float(val), "test_mse_if_state": float(test_if)})
        atomic_csv(pd.DataFrame(history_rows), history_path)
        start_batch = 0
        if rp.exists():
            try: rp.unlink()
            except Exception: pass
        elapsed = time.perf_counter() - run_start; done_epochs = epoch - start_epoch + 1
        eta = elapsed / max(done_epochs, 1) * max(FULL_EPOCHS - epoch, 0)
        epoch_bar.set_postfix(train=f"{train_mse:.3g}", val=f"{val:.3g}", best=f"{best_val:.3g}@{best_epoch}", eta=fmt_seconds(eta))
        if done_epochs == 1 or (time.perf_counter() - run_start) > TRACE_EVERY_SECONDS:
            trace(f"[train] {rid} epoch={epoch}/{FULL_EPOCHS} train={train_mse:.6g} val={val:.6g} "
                  f"best={best_val:.6g}@{best_epoch} run-ETA~{fmt_seconds(eta)} {gpu_memory_text()}")
        if STOP_REQUESTED or soft_walltime_due():
            atomic_torch_save({
                "schema": "cnn_c1_resume_v1", "run_id": rid, "epoch": int(epoch + 1), "next_batch": 0,
                "best_val": best_val, "best_epoch": best_epoch, "calibration_scales": calibration_scales,
                "initial_digest": initial_digest, "history_rows": history_rows,
                "model_state": state_dict_cpu(model), "optimizer_state": opt.state_dict(),
            }, rp)
            trace(f"[train] safe stop after completed epoch {epoch}; resume will begin epoch {epoch+1}")
            raise KeyboardInterrupt("safe stop after epoch checkpoint")

    training_seconds = float(time.perf_counter() - run_start)
    append_rate("train_batch", float(max((FULL_EPOCHS-start_epoch+1)*n_batches, 1)), training_seconds, rid)
    final_test = eval_mse(model, Xt, yt)
    best_model, best_ck = load_model_checkpoint(arch_id, seed_index, "best"); best_test = eval_mse(best_model, Xt, yt)
    out = {
        "run_id": rid, "arch_id": arch_id, "seed_index": seed_index, "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val), "best_test_mse": float(best_test), "final_test_mse": float(final_test),
        "initial_digest": initial_digest, "calibration_scales_json": json.dumps(calibration_scales),
        "training_seconds": training_seconds,
    }
    atomic_csv(pd.DataFrame([out]), summary_path)
    if rp.exists():
        try: rp.unlink()
        except Exception: pass
    del model, best_model, opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def run_training_phase():
    banner("CNN-C3 PHASE: TRAIN 12A x 4 seeds = 48 fresh trajectories")
    manifest = training_manifest(); jobs = manifest[manifest.run_id.map(in_this_shard)].reset_index(drop=True)
    if MAX_JOBS_THIS_INVOCATION > 0: jobs = jobs.iloc[:MAX_JOBS_THIS_INVOCATION]
    trace(f"[train] shard {SHARD_INDEX}/{SHARDS}: {len(jobs)} of {len(manifest)} trajectories")
    splits, envmeta = environment_dataset(CANONICAL_ENV)
    eta = PhaseETA("train_run", len(jobs), fallback_seconds_per_unit=1800.0); phase_start = time.perf_counter()
    for i, r in jobs.iterrows():
        t0 = time.perf_counter(); train_one(r, splits); dt = time.perf_counter() - t0
        eta.update(1, force=True); append_rate("train_run", 1.0, dt, str(r.run_id))
    append_rate("train_phase", max(len(jobs), 1), time.perf_counter() - phase_start, f"shard={SHARD_INDEX}/{SHARDS}")
    del splits; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ==================================================================================================
# 7. CONDITIONAL CAMPAIGN — 64 LONGITUDINAL / 256 DEEP INPUTS, NO PREMATURE INPUT AVERAGING
# ==================================================================================================


def _save_population_npz(path: Path, state: Dict[str, Any], meta: Dict[str, Any]):
    arrays: Dict[str, Any] = {"meta_json": np.asarray(json.dumps(meta, default=str))}
    for l, a in enumerate(state["node_flow"]): arrays[f"node_{l}"] = np.asarray(a, dtype=np.float64)
    for l, a in enumerate(state["occurrence_flow"]): arrays[f"occ_{l}"] = np.asarray(a, dtype=np.float64)
    for l, a in enumerate(state["parameter_flow"]): arrays[f"par_{l}"] = np.asarray(a, dtype=np.float64)
    arrays["n_node_layers"] = np.asarray(len(state["node_flow"]), dtype=np.int64)
    arrays["n_occ_layers"] = np.asarray(len(state["occurrence_flow"]), dtype=np.int64)
    atomic_npz(path, **arrays)


def _save_degenerate_population_npz(path: Path, meta: Dict[str, Any]):
    """Write an explicit marker for a scientifically valid all-degenerate conditional job.

    A zero conditional partition for every sampled contrast means that no normalized population
    conditional measure exists for this job.  This is data, not a campaign failure.
    """
    marker = dict(meta)
    marker.update({
        "degenerate_all": True,
        "population_state_defined": False,
        "degeneracy_reason": "all_sampled_conditional_partitions_zero_or_nonfinite",
    })
    atomic_npz(
        path,
        meta_json=np.asarray(json.dumps(marker, default=str)),
        n_node_layers=np.asarray(0, dtype=np.int64),
        n_occ_layers=np.asarray(0, dtype=np.int64),
        degenerate_all=np.asarray(1, dtype=np.int8),
    )


def _load_population_npz(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not Path(path).exists(): raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=False)
    raw = z["meta_json"].item() if np.asarray(z["meta_json"]).ndim == 0 else str(z["meta_json"])
    meta = json.loads(str(raw))
    degenerate = bool(int(np.asarray(z["degenerate_all"]).item())) if "degenerate_all" in z.files else bool(meta.get("degenerate_all", False))
    nn = int(z["n_node_layers"]); no = int(z["n_occ_layers"])
    if degenerate or nn == 0 or no == 0:
        meta["degenerate_all"] = True
        meta["population_state_defined"] = False
        return None, meta
    state = {
        "node_flow": [np.asarray(z[f"node_{l}"], float) for l in range(nn)],
        "occurrence_flow": [np.asarray(z[f"occ_{l}"], float) for l in range(no)],
        "parameter_flow": [np.asarray(z[f"par_{l}"], float) for l in range(no)],
    }
    meta.setdefault("degenerate_all", False)
    meta.setdefault("population_state_defined", True)
    return state, meta


def condition_one(row: pd.Series, splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]], panel_ids: np.ndarray) -> Dict[str, Any]:
    job_id = str(row.job_id); rid = str(row.run_id); arch_id = str(row.arch_id); seed_index = int(row.seed_index)
    label = str(row.checkpoint); n_inputs = int(row.n_inputs)
    summary_path = condition_part(job_id, "summary"); input_path = condition_part(job_id, "inputs")
    layers_path = condition_part(job_id, "layers"); poplayers_path = condition_part(job_id, "population_layers")
    spec_path = condition_part(job_id, "specialization"); state_path = condition_state_path(job_id)
    required = [summary_path, input_path, layers_path, poplayers_path, spec_path, state_path]
    if RESUME and all(p.exists() for p in required):
        trace(f"[condition] skip complete {job_id}"); return safe_read_csv(summary_path).iloc[0].to_dict()

    model, ck = load_model_checkpoint(arch_id, seed_index, label); actual_epoch = int(ck.get("epoch", row.epoch_nominal))
    structural = cnn_structural_state(model, return_flows=True)
    Xt, _ = splits["test"]; ids = np.asarray(panel_ids[:n_inputs], int)
    result = conditional_batch_analysis(model, Xt, ids, structural, label, rid, arch_id, seed_index)
    summ = dict(result["summary"]); summ.update({
        "job_id": job_id, "epoch_actual": actual_epoch, "epoch_nominal": int(row.epoch_nominal),
        "environment_id": CANONICAL_ENV["env_id"], "n_requested_inputs": n_inputs,
    })
    atomic_csv(pd.DataFrame([summ]), summary_path)
    atomic_csv(result["input_df"], input_path); atomic_csv(result["layer_input_df"], layers_path)
    atomic_csv(result["population_layers"], poplayers_path); atomic_csv(result["specialization"], spec_path)
    if result["population_state"] is None:
        raise RuntimeError(f"all conditional contrasts were degenerate for {job_id}")
    _save_population_npz(state_path, result["population_state"], {
        "job_id": job_id, "run_id": rid, "arch_id": arch_id, "seed_index": seed_index,
        "checkpoint": label, "epoch_actual": actual_epoch, "n_inputs": n_inputs,
    })
    del model, structural, result; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summ


def run_condition_phase():
    banner("CNN-C3 PHASE: CONDITION structural p^S -> input-conditioned mu^c")
    manifest = condition_manifest(); jobs = manifest[manifest.job_id.map(in_this_shard)].reset_index(drop=True)
    if MAX_JOBS_THIS_INVOCATION > 0: jobs = jobs.iloc[:MAX_JOBS_THIS_INVOCATION]
    trace(f"[condition] shard {SHARD_INDEX}/{SHARDS}: {len(jobs)} of {len(manifest)} checkpoint-condition jobs")
    splits, _ = environment_dataset(CANONICAL_ENV); ids = fixed_test_indices()
    eta = PhaseETA("condition_job", len(jobs), fallback_seconds_per_unit=120.0); tphase = time.perf_counter()
    for _, r in jobs.iterrows():
        t0 = time.perf_counter(); condition_one(r, splits, ids); dt = time.perf_counter() - t0
        append_rate("condition_job", 1.0, dt, str(r.job_id)); eta.update(1, force=True)
        if STOP_REQUESTED or soft_walltime_due():
            trace("[condition] stopping cleanly at job boundary"); break
    append_rate("condition_phase", max(len(jobs), 1), time.perf_counter() - tphase, f"shard={SHARD_INDEX}/{SHARDS}")
    del splits; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ==================================================================================================
# 8. FAITHFUL BCANw EXTENSION: xi50, d*, FLOW LACUNARITY, LOCAL DELTA_i^F, QUOTIENT KS, RETENTION
# ==================================================================================================


def r2_score_np(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float); pred = np.asarray(pred, float); den = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / den) if den > EPS else np.nan


def compress_box_staircase(radii: np.ndarray, NB: np.ndarray, lac: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    radii = np.asarray(radii, float); NB = np.asarray(NB, float); lac = np.asarray(lac, float)
    rr = []; nn = []; ll = []
    for n in np.unique(NB)[::-1]:
        ids = np.where(NB == n)[0]
        if len(ids):
            j = int(ids[len(ids) // 2]); rr.append(radii[j]); nn.append(NB[j]); ll.append(lac[j])
    order = np.argsort(rr)
    return np.asarray(rr)[order], np.asarray(nn)[order], np.asarray(ll)[order]


def covering_quantile_scale(r: np.ndarray, n: np.ndarray, N: int, target: float = 0.5) -> float:
    return covering_scale(r, n, N, target)


def local_effective_dimension(r: np.ndarray, n: np.ndarray, xi50: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, float); n = np.asarray(n, float); k = max(3, int(LOCAL_SLOPE_WINDOW)); half = k // 2
    xc = np.full(len(r), np.nan); deff = np.full(len(r), np.nan); r2 = np.full(len(r), np.nan)
    if len(r) < k or not np.isfinite(xi50) or xi50 <= 0: return xc, deff, r2
    for c in range(half, len(r) - half):
        ids = np.arange(c - half, c + half + 1); rr = r[ids]; nn = n[ids]
        good = (rr > 0) & (nn > 0)
        if good.sum() < 3: continue
        x = np.log(rr[good]); y = np.log(nn[good]); b, a = np.polyfit(x, y, 1); pred = a + b * x
        xc[c] = r[c] / xi50; deff[c] = -float(b); r2[c] = r2_score_np(y, pred)
    return xc, deff, r2


def crossover_from_deff(x: np.ndarray, deff: np.ndarray, r2: np.ndarray, NB: np.ndarray, N: int) -> Dict[str, float]:
    x = np.asarray(x, float); deff = np.asarray(deff, float); r2 = np.asarray(r2, float); NB = np.asarray(NB, float)
    m = np.isfinite(x) & np.isfinite(deff) & np.isfinite(r2) & (r2 >= LOCAL_SLOPE_R2_MIN) & (deff > 0)
    if not np.any(m): return {"d_star": np.nan, "x_d_star": np.nan, "transition_fraction": 0.0}
    ids = np.where(m)[0]; j = int(ids[np.argmax(deff[ids])])
    return {"d_star": float(deff[j]), "x_d_star": float(x[j]), "transition_fraction": float(np.mean(m))}


def lacunarity_crossover(radii: np.ndarray, lac: np.ndarray, xi50: float, NB: np.ndarray, N: int) -> Dict[str, float]:
    radii = np.asarray(radii, float); lac = np.asarray(lac, float); NB = np.asarray(NB, float)
    m = np.isfinite(lac) & np.isfinite(radii) & (NB >= 3) & (NB <= 0.90 * N)
    if not np.any(m): return {"lacunarity_star": np.nan, "x_lacunarity_star": np.nan}
    ids = np.where(m)[0]; j = int(ids[np.argmax(lac[ids])])
    return {"lacunarity_star": float(lac[j]), "x_lacunarity_star": float(radii[j] / xi50) if np.isfinite(xi50) and xi50 > 0 else np.nan}


def nearest_resolution_indices(NB: np.ndarray, N: int) -> List[int]:
    NB = np.asarray(NB, float); frac = NB / float(N); out: List[int] = []
    for t in RENORM_TARGET_BOX_FRACTIONS:
        eligible = np.where((NB >= 3) & (NB <= .90 * N))[0]
        if not len(eligible): continue
        idx = int(eligible[np.argmin(np.abs(frac[eligible] - t))])
        if idx not in out: out.append(idx)
    return out


def mean_pairwise_ks(arrays: List[np.ndarray]) -> float:
    arr = []
    for a in arrays:
        v = np.asarray(a, float); v = v[np.isfinite(v) & (v > 0)]
        if len(v) >= 5: arr.append(v)
    if len(arr) < 2: return np.nan
    vals = [ks_2samp(arr[i], arr[j]).statistic for i in range(len(arr)) for j in range(i + 1, len(arr))]
    return float(np.mean(vals)) if vals else np.nan


def quotient_flow_distribution(Praw: ExecutionGraph, colors: np.ndarray) -> np.ndarray:
    agg: Dict[Tuple[int, int], float] = {}
    for (u, v), w in Praw.edge_mag.items():
        a, b = int(colors[u]), int(colors[v])
        if a == b: continue
        key = (min(a, b), max(a, b)); agg[key] = agg.get(key, 0.0) + float(w)
    vals = np.asarray(list(agg.values()), float); vals = vals[np.isfinite(vals) & (vals > 0)]
    return vals / vals.sum() if len(vals) and vals.sum() > 0 else np.asarray([], float)


def interbox_flow_fraction(Praw: ExecutionGraph, colors: np.ndarray) -> float:
    total = float(sum(Praw.edge_mag.values()))
    if total <= 0: return np.nan
    inter = sum(float(w) for (u, v), w in Praw.edge_mag.items() if int(colors[u]) != int(colors[v]))
    return float(inter / total)


def fit_common_local_growth(r: np.ndarray, support: np.ndarray, mass: np.ndarray) -> Tuple[float, float, float, float, float, float, int]:
    r = np.asarray(r, float); support = np.asarray(support, float); mass = np.asarray(mass, float); best = None
    n = len(r)
    for i in range(n):
        for j in range(i + LOCAL_COMMON_MIN_POINTS, n + 1):
            rr, ss, mm = r[i:j], support[i:j], mass[i:j]
            if np.any(rr <= 0) or np.any(ss <= 0) or np.any(mm <= 0): continue
            span = float(np.log10(rr[-1] / rr[0]))
            if span < LOCAL_COMMON_MIN_SPAN_DECADES: continue
            if ss[-1] / max(ss[0], EPS) < LOCAL_COMMON_SUPPORT_GROWTH: continue
            if mm[-1] / max(mm[0], EPS) < LOCAL_COMMON_MASS_GROWTH: continue
            x = np.log(rr); bd, ad = np.polyfit(x, np.log(ss), 1); ba, aa = np.polyfit(x, np.log(mm), 1)
            if bd <= 0 or ba <= 0: continue
            r2d = r2_score_np(np.log(ss), ad + bd * x); r2a = r2_score_np(np.log(mm), aa + ba * x)
            if r2d < LOCAL_COMMON_R2_MIN or r2a < LOCAL_COMMON_R2_MIN: continue
            score = (span, min(r2d, r2a), len(rr)); rec = (float(bd), float(ba), float(ba - bd), float(r2d), float(r2a), span, int(len(rr)))
            if best is None or score > best[0]: best = (score, rec)
    return best[1] if best else (np.nan,) * 7


def local_metric_dimensions(D: np.ndarray, node_mu: np.ndarray, max_nodes: int = LOCAL_DIMENSION_NODES,
                            max_radii: int = LOCAL_RADII_MAX) -> Tuple[Dict[str, Any], pd.DataFrame]:
    N = D.shape[0]; ids = np.arange(N)
    if max_nodes is not None and int(max_nodes) < N:
        rng = np.random.default_rng(C3_SEED + N); ids = np.sort(rng.choice(N, size=int(max_nodes), replace=False))
    rows = []; deltas = []
    for i in ids:
        di = np.asarray(D[i], float); vals = np.unique(di[np.isfinite(di) & (di > 0)])
        if len(vals) < LOCAL_COMMON_MIN_POINTS: continue
        if len(vals) > max_radii:
            jj = np.unique(np.round(np.linspace(0, len(vals) - 1, max_radii)).astype(int)); vals = vals[jj]
        support = np.array([np.mean(di <= rr) for rr in vals], float)
        mass = np.array([np.sum(node_mu[di <= rr]) for rr in vals], float)
        bd, ba, dd, r2d, r2a, span, npnt = fit_common_local_growth(vals, support, mass)
        rows.append({"node_id": int(i), "d_i_F": bd, "alpha_i_F": ba, "delta_i_F": dd,
                     "r2_d": r2d, "r2_alpha": r2a, "span_decades": span, "n_fit": npnt,
                     "node_measure": float(node_mu[i])})
        if np.isfinite(dd): deltas.append(dd)
    df = pd.DataFrame(rows)
    return {"local_delta_median": float(np.nanmedian(deltas)) if len(deltas) else np.nan,
            "valid_local_delta_fraction": float(np.mean(np.isfinite(df.delta_i_F))) if len(df) else np.nan,
            "local_nodes_evaluated": int(len(ids))}, df


def analyze_execution_geometry_rich(model: ConvTraceNet, state: Dict[str, Any], n_scales: int = BCANW_SCALES_LONG,
                                    compute_local: bool = True) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    Pgeom = execution_graph_from_occurrence(model, state["occurrence_flow"], enrichment=True)
    if Pgeom.n_nodes > BCANW_MAX_NODES:
        return {"geometry_skipped": True, "geometry_skip_reason": f"n_nodes={Pgeom.n_nodes}>cap={BCANW_MAX_NODES}",
                "n_graph_nodes": Pgeom.n_nodes}, pd.DataFrame(), pd.DataFrame()
    Praw = execution_graph_from_occurrence(model, state["occurrence_flow"], enrichment=False)
    lengths = metric_lengths(Pgeom, METRIC_POWER); D = distance_matrix(Pgeom, lengths); radii = bcanw_scales(D, n_scales)
    order = np.lexsort((np.arange(Pgeom.n_nodes), -node_strength(Pgeom))); node_mu = global_node_measure(state["node_flow"])
    rows = []; NB = []; lac = []; inter = []; qnorm: List[np.ndarray] = []; coarse: Dict[float, List[float]] = {q: [] for q in COARSE_RENYI_Q}
    scale_bar = tqdm(radii, desc=f"BCANw N={Pgeom.n_nodes}", leave=False, disable=not TQDM_ENABLED, dynamic_ncols=True)
    for k, ell in enumerate(scale_bar):
        colors = greedy_bcanw_coloring(D, float(ell), order); nb = int(colors.max()) + 1; mass, lval = box_mass_lacunarity(colors, node_mu)
        NB.append(nb); lac.append(lval); inter.append(interbox_flow_fraction(Praw, colors)); qnorm.append(quotient_flow_distribution(Praw, colors))
        rec = {"scale_index": int(k), "ell_B": float(ell), "NB": float(nb), "NB_over_N": float(nb / Pgeom.n_nodes),
               "lacunarity": float(lval), "interbox_flow_fraction": float(inter[-1])}
        for q in COARSE_RENYI_Q:
            v = renyi_effective_fraction(mass, q); coarse[q].append(v); rec[f"coarse_eta_q{q:g}"] = float(v)
        rows.append(rec)
    NB = np.asarray(NB, float); lac = np.asarray(lac, float); inter = np.asarray(inter, float)
    rc, nc, lc = compress_box_staircase(radii, NB, lac); xi50 = covering_quantile_scale(rc, nc, Pgeom.n_nodes, .5)
    xc, deff, deffr2 = local_effective_dimension(rc, nc, xi50); cross = crossover_from_deff(xc, deff, deffr2, nc, Pgeom.n_nodes)
    lcross = lacunarity_crossover(radii, lac, xi50, NB, Pgeom.n_nodes); sw = fit_credible_scaling_window(radii, NB)
    qidx = nearest_resolution_indices(NB, Pgeom.n_nodes); qks = mean_pairwise_ks([qnorm[i] for i in qidx])
    retention = float(np.nanmean(inter[qidx])) if len(qidx) else np.nan
    local_summ, local_df = ({"local_delta_median": np.nan, "valid_local_delta_fraction": np.nan, "local_nodes_evaluated": 0}, pd.DataFrame())
    if compute_local: local_summ, local_df = local_metric_dimensions(D, node_mu)
    summ = {
        "geometry_skipped": False, "geometry_skip_reason": "", "n_graph_nodes": int(Pgeom.n_nodes),
        "xi50": float(xi50), "log_xi50": float(math.log(xi50)) if np.isfinite(xi50) and xi50 > 0 else np.nan,
        **cross, **lcross,
        "scaling_dB": float(sw.get("d_B", np.nan)), "scaling_r2": float(sw.get("scaling_r2", np.nan)),
        "scaling_credible": bool(sw.get("scaling_credible", False)),
        "scaling_span_decades": float(sw.get("scaling_span_decades", np.nan)),
        "quotient_mean_pairwise_KS": float(qks), "renorm_flow_retention_mean": float(retention), **local_summ,
    }
    sdf = pd.DataFrame(rows)
    if len(sdf): sdf["ell_over_xi50"] = sdf.ell_B / xi50 if np.isfinite(xi50) and xi50 > 0 else np.nan
    del D; gc.collect()
    return summ, sdf, local_df

# ==================================================================================================
# 9. PRINCIPAL + INPUT-RESOLVED GEOMETRY CAMPAIGNS
# ==================================================================================================


def geometry_one(row: pd.Series) -> Dict[str, Any]:
    job_id = str(row.job_id); outp = geometry_part(job_id); scalep = geometry_scale_part(job_id)
    if RESUME and outp.exists() and scalep.exists(): return safe_read_csv(outp).iloc[0].to_dict()
    rid = str(row.run_id); arch_id = str(row.arch_id); seed_index = int(row.seed_index); label = str(row.checkpoint)
    model, ck = load_model_checkpoint(arch_id, seed_index, label); structural = cnn_structural_state(model, True)
    cjob = f"{rid}__{label}"; pop, pmeta = _load_population_npz(condition_state_path(cjob))
    rows = []; scales = []; locals_frames = []
    for source, state in [("structural", structural), ("conditional_population", pop)]:
        t0 = time.perf_counter(); summ, sdf, ldf = analyze_execution_geometry_rich(model, state, BCANW_SCALES_LONG, compute_local=True)
        base = {"job_id": job_id, "run_id": rid, "arch_id": arch_id, "seed_index": seed_index,
                "checkpoint": label, "epoch_actual": int(ck.get("epoch", -1)), "geometry_source": source}
        rows.append({**base, **summ})
        if len(sdf):
            z = sdf.copy()
            for k, v in base.items(): z[k] = v
            scales.append(z)
        if len(ldf):
            lpath = OUTDIR / "geometry_parts" / f"{job_id}__{source}__local.csv"
            z = ldf.copy()
            for k, v in base.items(): z[k] = v
            atomic_csv(z, lpath)
        append_rate("geometry_eval", 1.0, time.perf_counter() - t0, f"{job_id}:{source}")
    atomic_csv(pd.DataFrame(rows), outp); atomic_csv(pd.concat(scales, ignore_index=True) if scales else pd.DataFrame(), scalep)
    del model, structural, pop; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows[0]


def run_geometry_phase():
    banner("CNN-C3 PHASE: PRINCIPAL structural + population-conditional BCANw")
    m = geometry_manifest(); jobs = m[m.job_id.map(in_this_shard)].reset_index(drop=True)
    if MAX_JOBS_THIS_INVOCATION > 0: jobs = jobs.iloc[:MAX_JOBS_THIS_INVOCATION]
    eta = PhaseETA("geometry_job", len(jobs), fallback_seconds_per_unit=600.0)
    for _, r in jobs.iterrows():
        t0 = time.perf_counter(); geometry_one(r); append_rate("geometry_job", 1, time.perf_counter() - t0, str(r.job_id)); eta.update(1, force=True)
        if STOP_REQUESTED or soft_walltime_due(): break


def input_geometry_one(row: pd.Series, splits: Dict[str, Tuple[torch.Tensor, torch.Tensor]], panel_ids: np.ndarray) -> Dict[str, Any]:
    job_id = str(row.job_id); outp = input_geometry_part(job_id); scalep = input_geometry_scale_part(job_id)
    if RESUME and outp.exists() and scalep.exists(): return safe_read_csv(outp).iloc[0].to_dict()
    rid = str(row.run_id); arch_id = str(row.arch_id); seed = int(row.seed_index); label = str(row.checkpoint); pos = int(row.input_panel_position)
    model, ck = load_model_checkpoint(arch_id, seed, label); Xt, _ = splits["test"]; idx = int(panel_ids[pos])
    if Xt.device.type == "cuda": xb = Xt[torch.as_tensor([idx], device=Xt.device, dtype=torch.long)]
    else: xb = Xt[torch.as_tensor([idx], dtype=torch.long)].to(DEVICE)
    x0 = torch.zeros_like(xb); resp = conditional_response_sq(model, xb, x0); st = cnn_factor_state(model, torch.square((xb-x0).to(torch.float64)), resp, True)
    if not bool(st["valid"][0]): raise RuntimeError(f"degenerate conditional state for {job_id}")
    cstate = {"node_flow": [v[0] for v in st["node_flow"]], "occurrence_flow": [v[0] for v in st["occurrence_flow"]],
              "parameter_flow": [v[0] for v in st["parameter_flow"]]}
    t0 = time.perf_counter(); summ, sdf, ldf = analyze_execution_geometry_rich(model, cstate, BCANW_SCALES_LONG, compute_local=True)
    base = {"job_id": job_id, "run_id": rid, "arch_id": arch_id, "seed_index": seed, "checkpoint": label,
            "epoch_actual": int(ck.get("epoch", -1)), "input_panel_position": pos, "input_index": idx}
    atomic_csv(pd.DataFrame([{**base, **summ}]), outp)
    if len(sdf):
        z = sdf.copy()
        for k, v in base.items(): z[k] = v
        atomic_csv(z, scalep)
    else: atomic_csv(pd.DataFrame(), scalep)
    if len(ldf):
        z = ldf.copy()
        for k, v in base.items(): z[k] = v
        atomic_csv(z, OUTDIR / "input_geometry_parts" / f"{job_id}__local.csv")
    append_rate("input_geometry_eval", 1.0, time.perf_counter()-t0, job_id)
    del model, st, cstate; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {**base, **summ}


def run_input_geometry_phase():
    banner("CNN-C3 PHASE: INPUT-RESOLVED BCANw SENTINEL ARM (4A x 4s x 16x x 3 states = 768)")
    m = input_geometry_manifest(); jobs = m[m.job_id.map(in_this_shard)].reset_index(drop=True)
    if MAX_JOBS_THIS_INVOCATION > 0: jobs = jobs.iloc[:MAX_JOBS_THIS_INVOCATION]
    splits, _ = environment_dataset(CANONICAL_ENV); ids = fixed_test_indices()
    eta = PhaseETA("input_geometry_job", len(jobs), fallback_seconds_per_unit=300.0)
    for _, r in jobs.iterrows():
        t0 = time.perf_counter(); input_geometry_one(r, splits, ids); append_rate("input_geometry_job", 1, time.perf_counter()-t0, str(r.job_id)); eta.update(1, force=True)
        if STOP_REQUESTED or soft_walltime_due(): break
    del splits; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ==================================================================================================
# 10. EXACTNESS + SYMMETRY VALIDATION ARMS (IMPLEMENTATION CONTROLS, NOT EXPERIMENTAL FACTORS)
# ==================================================================================================


def _enumerate_tiny_paths(model: ConvTraceNet, x: np.ndarray, x0: np.ndarray) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    if model.channels != [1, 1, 1] or model.kernel_sizes != [2, 2]:
        raise ValueError("tiny enumeration expects channels [1,1,1], kernels [2,2]")
    W = [c.weight.detach().cpu().numpy().astype(float) for c in model.convs]
    xt = torch.as_tensor(x[None, None], dtype=torch.float64, device=DEVICE)
    x0t = torch.as_tensor(x0[None, None], dtype=torch.float64, device=DEVICE)
    _, z, _ = forward_with_states(model, xt); _, z0, _ = forward_with_states(model, x0t)
    sec = finite_secant_torch64(model.activation_name, z[0], z0[0]).detach().cpu().numpy()[0, 0]
    dx = x - x0; H1 = dx.shape[0] - 1; H2 = H1 - 1; g = 1.0 / (H2 * H2)
    amps = []; keys = []
    for y2 in range(H2):
        for x2 in range(H2):
            for dy2 in range(2):
                for dx2 in range(2):
                    y1, x1 = y2 + dy2, x2 + dx2
                    for dy1 in range(2):
                        for dx1 in range(2):
                            y0, xx0 = y1 + dy1, x1 + dx1
                            a = float(dx[y0, xx0] * W[0][0, 0, dy1, dx1] * sec[y1, x1] *
                                      W[1][0, 0, dy2, dx2] * g)
                            amps.append(a); keys.append((y0, xx0, y1, x1, y2, x2, dy1, dx1, dy2, dx2))
    return keys, np.asarray(amps, float)


def run_exact_validation() -> pd.DataFrame:
    outp = OUTDIR / "validation" / "exact_tiny_cnn.csv"
    if RESUME and outp.exists(): return safe_read_csv(outp)
    rng = np.random.default_rng(C3_SEED + 101); rows = []
    total = 8; bar = tqdm(range(total), desc="validate exact tiny CNN", disable=not TQDM_ENABLED, dynamic_ncols=True)
    for ci in bar:
        model = ConvTraceNet(6, [1], [2, 2], "relu", C3_SEED + ci).to(DEVICE).double()
        x = rng.normal(size=(6, 6)); x0 = np.zeros_like(x)
        keys, amps = _enumerate_tiny_paths(model, x, x0); q = amps * amps; Z = float(q.sum()); p = q / max(Z, EPS)
        xb = torch.as_tensor(x[None, None], dtype=torch.float64, device=DEVICE); x0b = torch.zeros_like(xb)
        with torch.no_grad(): fx = float(model(xb).item()); f0 = float(model(x0b).item())
        structural = cnn_structural_state(model, True); resp = conditional_response_sq(model, xb, x0b)
        st = cnn_factor_state(model, torch.square(xb - x0b), resp, True)
        occ1 = np.zeros_like(st["occurrence_flow"][0][0]); occ2 = np.zeros_like(st["occurrence_flow"][1][0])
        for key, pp in zip(keys, p):
            y0, xx0, y1, x1, y2, x2, dy1, dx1, dy2, dx2 = key
            occ1[0, 0, dy1, dx1, y1, x1] += pp; occ2[0, 0, dy2, dx2, y2, x2] += pp
        par1 = occ1.sum(axis=(-1, -2)); par2 = occ2.sum(axis=(-1, -2)); H = _entropy_np(p)
        rows.append({
            "contrast_index": ci, "n_paths": len(amps), "delta_output": fx - f0, "amplitude_sum": float(amps.sum()),
            "reconstruction_abs_error": abs((fx - f0) - float(amps.sum())),
            "partition_rel_error": abs(math.exp(st["logZ1"][0]) - Z) / max(Z, 1e-300),
            "entropy_abs_error": abs(float(st["S_path"][0]) - H),
            "occurrence_flow_max_abs_error": max(float(np.max(np.abs(occ1 - st["occurrence_flow"][0][0]))),
                                                  float(np.max(np.abs(occ2 - st["occurrence_flow"][1][0])))),
            "parameter_flow_max_abs_error": max(float(np.max(np.abs(par1 - st["parameter_flow"][0][0]))),
                                                 float(np.max(np.abs(par2 - st["parameter_flow"][1][0])))),
            "max_pushforward_conservation_error": float(st["pushforward_conservation_error"][0]),
            "max_entropy_identity_residual": float(max(abs(d["entropy_identity_residual"][0]) for d in st["layer_metrics"])),
            "max_deficit_identity_residual": float(max(abs(d["deficit_identity_residual"][0]) for d in st["layer_metrics"])),
        })
        del model
    out = pd.DataFrame(rows); atomic_csv(out, outp); return out


def _conditional_scalar_signature(model: ConvTraceNet, x: torch.Tensor) -> Dict[str, float]:
    x0 = torch.zeros_like(x); structural = cnn_structural_state(model, True); resp = conditional_response_sq(model, x, x0)
    st = cnn_factor_state(model, torch.square((x-x0).to(torch.float64)), resp, True)
    return {"output": float(model(x).detach().cpu().item()), "logZ": float(st["logZ1"][0]), "S_path": float(st["S_path"][0]),
            "param_flow": np.concatenate([p[0].ravel() for p in st["parameter_flow"]]),
            "occ_flow": np.concatenate([e[0].ravel() for e in st["occurrence_flow"]])}


def run_symmetry_validation() -> pd.DataFrame:
    outp = OUTDIR / "validation" / "symmetry_controls.csv"
    if RESUME and outp.exists(): return safe_read_csv(outp)
    rng = np.random.default_rng(C3_SEED + 202); rows = []
    model = ConvTraceNet(10, [3], [3, 1], "relu", C3_SEED + 301).to(DEVICE).double()
    x = torch.as_tensor(rng.normal(size=(1, 1, 10, 10)), dtype=torch.float64, device=DEVICE)
    base = _conditional_scalar_signature(model, x)

    # Hidden-channel permutation: conv0 output channels and conv1 matching input channels are relabelled together.
    mp = copy.deepcopy(model); perm = torch.as_tensor([2, 0, 1], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        mp.convs[0].weight.copy_(mp.convs[0].weight[perm])
        mp.convs[1].weight.copy_(mp.convs[1].weight[:, perm])
    sig = _conditional_scalar_signature(mp, x)
    rows.append({"control": "channel_permutation", "output_abs_error": abs(sig["output"]-base["output"]),
                 "logZ_abs_error": abs(sig["logZ"]-base["logZ"]), "S_path_abs_error": abs(sig["S_path"]-base["S_path"])})

    # Positive ReLU gauge: outgoing channel scaled by a and next-layer corresponding input divided by a.
    ms = copy.deepcopy(model); a = 2.75
    with torch.no_grad():
        ms.convs[0].weight[1].mul_(a); ms.convs[1].weight[:, 1].div_(a)
    sig2 = _conditional_scalar_signature(ms, x)
    rows.append({"control": "positive_channel_rescaling", "output_abs_error": abs(sig2["output"]-base["output"]),
                 "logZ_abs_error": abs(sig2["logZ"]-base["logZ"]), "S_path_abs_error": abs(sig2["S_path"]-base["S_path"])})

    # Circular translation control: shared parameter-identity flow is translation invariant under exact roll.
    R = 11; k = 3; xx = rng.normal(size=(R, R)); w = rng.normal(size=(k, k)); shift = (2, 3)
    def circ_param_flow(img: np.ndarray) -> np.ndarray:
        vals = np.zeros((k, k), float)
        for dy in range(k):
            for dx in range(k):
                rolled = np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)
                vals[dy, dx] = np.sum(np.square(rolled * w[dy, dx]))
        vals /= max(vals.sum(), EPS); return vals
    f0 = circ_param_flow(xx); f1 = circ_param_flow(np.roll(np.roll(xx, shift[0], axis=0), shift[1], axis=1))
    rows.append({"control": "circular_translation_parameter_identity", "output_abs_error": np.nan,
                 "logZ_abs_error": np.nan, "S_path_abs_error": np.nan,
                 "parameter_flow_max_abs_error": float(np.max(np.abs(f0-f1)))})
    out = pd.DataFrame(rows); atomic_csv(out, outp); return out


def run_validation_phase():
    banner("CNN-C3 PHASE: EXACTNESS + SYMMETRY VALIDATION")
    t0 = time.perf_counter(); exact = run_exact_validation(); sym = run_symmetry_validation()
    trace(f"[validate] exact max reconstruction error={exact.reconstruction_abs_error.max():.3e}; "
          f"partition={exact.partition_rel_error.max():.3e}; occurrence={exact.occurrence_flow_max_abs_error.max():.3e}")
    trace(f"[validate] controls written to {OUTDIR/'validation'} in {fmt_seconds(time.perf_counter()-t0)}")

# ==================================================================================================
# 11. DISK + RUNTIME PLANNING — ALWAYS PRINTED BEFORE HEAVY MODES
# ==================================================================================================


def own_output_bytes() -> int:
    total = 0
    if not OUTDIR.exists(): return 0
    for root, dirs, files in os.walk(OUTDIR):
        for f in files:
            try: total += (Path(root) / f).stat().st_size
            except OSError: pass
    return int(total)


def disk_plan() -> Dict[str, Any]:
    panel = architecture_panel(); pmap = dict(zip(panel.arch_id.astype(str), panel.parameter_count.astype(int)))
    omap = dict(zip(panel.arch_id.astype(str), panel.occurrence_count.astype(int)))
    nmap = dict(zip(panel.arch_id.astype(str), panel.execution_nodes.astype(int)))
    checkpoint_bytes = 0; resume_peak = 0; state_bytes = 0
    for aid in pmap:
        p = pmap[aid]
        checkpoint_bytes += len(SEEDS) * (len(STATE_GRID) + 1) * (16_384 + int(p * 4 * 1.35))
        resume_peak = max(resume_peak, 65_536 + int(p * 24))  # model + Adam moments + serialization overhead
        # 13 conditional states per model. Population node + occurrence + parameter arrays are float64.
        par_elems = p; node_elems = nmap[aid]; occ_elems = omap[aid]
        per_state = 16_384 + 8 * (node_elems + occ_elems + par_elems)
        state_bytes += len(SEEDS) * (len(STATE_GRID) + 1) * per_state
    input_rows = len(training_manifest()) * (2 * DEEP_INPUTS + 10 * LONGITUDINAL_INPUTS + DEEP_INPUTS)  # t0,t360,10 middle,best
    input_csv_bytes = int(input_rows * 1700)
    layer_csv_bytes = int(input_rows * 4.5 * 850)
    train_csv_bytes = int(len(training_manifest()) * (FULL_EPOCHS + 1) * 220)
    geometry_scale_rows = len(geometry_manifest()) * 2 * BCANW_SCALES_LONG + len(input_geometry_manifest()) * BCANW_SCALES_LONG
    geometry_csv_bytes = int(geometry_scale_rows * 700 + (len(geometry_manifest()) * 2 + len(input_geometry_manifest())) * 2500)
    local_csv_bytes = int((len(geometry_manifest()) * 2 + len(input_geometry_manifest())) * min(LOCAL_DIMENSION_NODES, 512) * 280)
    tables_figures_audit = 64 * 1024 * 1024
    raw_total = checkpoint_bytes + state_bytes + input_csv_bytes + layer_csv_bytes + train_csv_bytes + geometry_csv_bytes + local_csv_bytes + tables_figures_audit + resume_peak * SHARDS
    final_est = int(raw_total * DISK_SAFETY_FACTOR); current = own_output_bytes(); remaining = max(final_est - current, 0)
    du = shutil.disk_usage(OUTDIR)
    return {
        "estimated_final_bytes_with_safety": final_est, "estimated_raw_bytes": int(raw_total),
        "current_output_bytes": current, "estimated_remaining_bytes": remaining,
        "filesystem_total_bytes": int(du.total), "filesystem_used_bytes": int(du.used), "filesystem_free_bytes": int(du.free),
        "required_free_with_headroom_bytes": int(remaining + MIN_FREE_AFTER_GB * (1024 ** 3)),
        "fits_with_headroom": bool(du.free >= remaining + MIN_FREE_AFTER_GB * (1024 ** 3)),
        "breakdown": {
            "checkpoints": int(checkpoint_bytes), "conditional_population_states": int(state_bytes),
            "conditional_input_csv": int(input_csv_bytes), "conditional_layer_csv": int(layer_csv_bytes),
            "training_csv": int(train_csv_bytes), "geometry_csv": int(geometry_csv_bytes),
            "local_geometry_csv": int(local_csv_bytes), "tables_figures_audit_allowance": int(tables_figures_audit),
            "max_resume_file_per_active_run": int(resume_peak),
        },
        "counts": {"training_runs": len(training_manifest()), "condition_jobs": len(condition_manifest()),
                   "principal_geometry_jobs": len(geometry_manifest()), "input_geometry_jobs": len(input_geometry_manifest()),
                   "conditional_input_rows": int(input_rows)},
    }


def benchmark_runtime() -> Dict[str, Any]:
    """Small hardware calibration. Disk planning is printed before this is called."""
    outp = OUTDIR / "audit" / "preflight_benchmark.json"
    if RESUME and outp.exists():
        try: return json.loads(outp.read_text())
        except Exception: pass
    result: Dict[str, Any] = {"device": str(DEVICE), "timestamp": time.time()}
    model = build_model("C2_medium3", 0); opt = _optimizer_for(model)
    bs = min(BATCH_SIZE, 256); g = torch.Generator(device="cpu").manual_seed(C3_SEED + 909)
    x = torch.randn((bs, 1, RESOLUTION, RESOLUTION), generator=g).to(DEVICE); y = torch.randn((bs, 1), generator=g).to(DEVICE)
    for _ in range(2):
        opt.zero_grad(set_to_none=True); F.mse_loss(model(x), y).backward(); opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter(); reps = 8
    for _ in range(reps):
        opt.zero_grad(set_to_none=True); F.mse_loss(model(x), y).backward(); opt.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    result["seconds_per_train_batch"] = (time.perf_counter()-t0)/reps
    model.eval();
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter(); reps = 20
    with torch.no_grad():
        for _ in range(reps): model(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    result["seconds_per_eval_batch"] = (time.perf_counter()-t0)/reps
    xc = x[:CONDITIONAL_BATCH]; x0 = torch.zeros_like(xc); structural = cnn_structural_state(model, True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter(); reps = 3
    for _ in range(reps):
        resp = conditional_response_sq(model, xc, x0); st = cnn_factor_state(model, torch.square((xc-x0).to(torch.float64)), resp, True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    result["seconds_per_condition_batch"] = (time.perf_counter()-t0)/reps
    if PREFLIGHT_GEOMETRY_BENCHMARK:
        t0 = time.perf_counter(); gs, _, _ = analyze_execution_geometry_rich(model, structural, min(8, BCANW_SCALES_LONG), compute_local=False)
        dt = time.perf_counter()-t0; result["seconds_per_geometry_8scale_no_local"] = dt
        result["seconds_per_geometry_estimated_full"] = dt * (BCANW_SCALES_LONG/max(min(8, BCANW_SCALES_LONG),1)) * 1.35
    atomic_json(result, outp); del model, opt, x, y; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def runtime_plan(bench: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    bench = bench or {}
    train_batch = empirical_rate("train_batch", float(bench.get("seconds_per_train_batch", 0.012)))
    eval_batch = float(bench.get("seconds_per_eval_batch", train_batch * .25))
    cond_batch = empirical_rate("condition_batch", float(bench.get("seconds_per_condition_batch", 0.45)))
    geom_eval = empirical_rate("geometry_eval", float(bench.get("seconds_per_geometry_estimated_full", 300.0)))
    inp_geom = empirical_rate("input_geometry_eval", geom_eval)
    nb_train = int(math.ceil(N_TRAIN/BATCH_SIZE)); nb_val = int(math.ceil(N_VAL/2048)); nb_test = int(math.ceil(N_TEST/2048))
    train_seconds = len(training_manifest()) * (FULL_EPOCHS * nb_train * train_batch + FULL_EPOCHS * nb_val * eval_batch + len(STATE_GRID)*nb_test*eval_batch)
    cond_batches_per_run = 10*math.ceil(LONGITUDINAL_INPUTS/CONDITIONAL_BATCH) + 3*math.ceil(DEEP_INPUTS/CONDITIONAL_BATCH)
    cond_seconds = len(training_manifest()) * cond_batches_per_run * cond_batch
    geometry_seconds = len(geometry_manifest()) * 2 * geom_eval
    input_geometry_seconds = len(input_geometry_manifest()) * inp_geom
    return {
        "train_seconds_compute": float(train_seconds), "condition_seconds_compute": float(cond_seconds),
        "geometry_seconds_compute": float(geometry_seconds), "input_geometry_seconds_compute": float(input_geometry_seconds),
        "total_compute_seconds": float(train_seconds+cond_seconds+geometry_seconds+input_geometry_seconds),
        "idealized_current_shards_seconds": float((train_seconds+cond_seconds+geometry_seconds+input_geometry_seconds)/SHARDS),
        "rates": {"train_batch_s": train_batch, "eval_batch_s": eval_batch, "condition_batch_s": cond_batch,
                  "geometry_eval_s": geom_eval, "input_geometry_eval_s": inp_geom},
        "warning": "Wall-time projection is a planning estimate. BCANw varies strongly with execution-graph size and cluster load; empirical rates replace priors as jobs complete.",
    }


def print_startup_report(bench: Optional[Dict[str, Any]] = None, enforce: bool = True) -> Dict[str, Any]:
    banner("CNN-C3 STARTUP AUDIT — DESIGN / DISK / RUNTIME / SHARD")
    dp = disk_plan(); rp = runtime_plan(bench)
    print(f"mode={MODE} | host={socket.gethostname()} | device={DEVICE} | cpus={CPUS_PER_TASK} | shard={SHARD_INDEX}/{SHARDS}")
    print(f"OUTDIR={OUTDIR}")
    print("DESIGN: 12 architectures x 1 canonical environment x 4 seeds = 48 trained CNNs")
    print(f"state grid={STATE_GRID}; condition jobs={dp['counts']['condition_jobs']}; principal geometry jobs={dp['counts']['principal_geometry_jobs']}; input geometry jobs={dp['counts']['input_geometry_jobs']}")
    print("\nDISK ESTIMATE (conservative; includes safety factor):")
    for k, v in dp["breakdown"].items(): print(f"  {k:40s} {human_bytes(v)}")
    print(f"  {'estimated final (with safety)':40s} {human_bytes(dp['estimated_final_bytes_with_safety'])}")
    print(f"  {'current C2 output':40s} {human_bytes(dp['current_output_bytes'])}")
    print(f"  {'estimated remaining':40s} {human_bytes(dp['estimated_remaining_bytes'])}")
    print(f"  {'filesystem free now':40s} {human_bytes(dp['filesystem_free_bytes'])}")
    print(f"  {'required free incl. headroom':40s} {human_bytes(dp['required_free_with_headroom_bytes'])}")
    print(f"  FITS WITH {MIN_FREE_AFTER_GB:g} GiB HEADROOM: {'YES' if dp['fits_with_headroom'] else 'NO'}")
    print("\nRUNTIME PLANNING ESTIMATE:")
    print(f"  training compute:                 {fmt_seconds(rp['train_seconds_compute'])}")
    print(f"  conditioning compute:             {fmt_seconds(rp['condition_seconds_compute'])}")
    print(f"  principal BCANw compute:          {fmt_seconds(rp['geometry_seconds_compute'])}")
    print(f"  input-resolved BCANw compute:     {fmt_seconds(rp['input_geometry_seconds_compute'])}")
    print(f"  total serial compute:             {fmt_seconds(rp['total_compute_seconds'])}")
    print(f"  idealized / {SHARDS} shards:            {fmt_seconds(rp['idealized_current_shards_seconds'])}")
    print(f"  {rp['warning']}")
    print(f"\nMEMORY: {process_memory_text()} | {gpu_memory_text()}")
    atomic_json({"disk": dp, "runtime": rp, "runtime_environment": runtime_record(), "mode": MODE,
                 "shard_index": SHARD_INDEX, "shards": SHARDS}, OUTDIR / "audit" / f"startup_shard_{SHARD_INDEX:04d}.json")
    if enforce and ENFORCE_DISK and not dp["fits_with_headroom"]:
        raise RuntimeError("Disk guard stopped CNN-C3 before heavy work. Increase free space, change OUTDIR, or explicitly set PATHFLOW_CNN_C3_ENFORCE_DISK=0 after inspecting this estimate.")
    return {"disk": dp, "runtime": rp}


def run_preflight_phase():
    # First report uses conservative priors so disk visibility precedes any benchmark allocation.
    print_startup_report(bench=None, enforce=True)
    banner("CNN-C3 PREFLIGHT HARDWARE CALIBRATION")
    bench = benchmark_runtime() if PREFLIGHT_BENCHMARK else {}
    if bench:
        print(json.dumps({k: v for k, v in bench.items() if isinstance(v, (int, float, str))}, indent=2))
    plan = print_startup_report(bench=bench, enforce=True)
    panel = architecture_panel(); atomic_csv(panel, OUTDIR / "tables" / "frozen_12_architecture_panel.csv")
    atomic_csv(training_manifest(), OUTDIR / "tables" / "training_manifest.csv")
    atomic_csv(condition_manifest(), OUTDIR / "tables" / "condition_manifest.csv")
    atomic_csv(geometry_manifest(), OUTDIR / "tables" / "geometry_manifest.csv")
    atomic_csv(input_geometry_manifest(), OUTDIR / "tables" / "input_geometry_manifest.csv")
    atomic_json({"canonical_environment": CANONICAL_ENV, "environment_hash": stable_hash_dict(environment_key_dict(CANONICAL_ENV), 16),
                 "fixed_test_indices": fixed_test_indices().tolist(), "state_grid": STATE_GRID,
                 "principal_states": PRINCIPAL_STATES, "sentinel_architectures": SENTINEL_ARCH_IDS},
                OUTDIR / "audit" / "frozen_design.json")
    trace("[preflight] design manifests and planning audit written. No science trajectory was trained in preflight mode.")


# ==================================================================================================
# 12. AGGREGATION + C2 LAW-SURVIVAL / PAIRED DISPOSITION-REALIZATION INFERENCE
# ==================================================================================================


def _concat_parts(pattern: str) -> pd.DataFrame:
    frames = []
    for p in sorted(OUTDIR.glob(pattern)):
        d = safe_read_csv(p)
        if len(d): frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _friedman_architecture_table(summary: pd.DataFrame) -> pd.DataFrame:
    d = summary[summary.checkpoint.astype(str) == "t360"].copy() if len(summary) else pd.DataFrame()
    if d.empty: return pd.DataFrame()
    metrics = [c for c in d.columns if c.startswith("conditional_") and c.endswith("_median")]
    rows = []
    for metric in metrics:
        piv = d.pivot_table(index="seed_index", columns="arch_id", values=metric, aggfunc="first")
        piv = piv.reindex(index=SEEDS, columns=[x[0] for x in FROZEN_ARCHITECTURES])
        complete = piv.dropna(axis=0, how="any")
        if complete.shape[0] >= 2 and complete.shape[1] >= 3:
            try:
                stat, p = friedmanchisquare(*[complete[c].to_numpy(float) for c in complete.columns])
                W = float(stat / (complete.shape[0] * (complete.shape[1] - 1)))
            except Exception:
                stat, p, W = np.nan, np.nan, np.nan
        else: stat, p, W = np.nan, np.nan, np.nan
        rows.append({"metric": metric, "n_seed_blocks": int(complete.shape[0]), "n_architectures": int(complete.shape[1]),
                     "friedman_chi2": float(stat), "p_value": float(p), "kendall_W": float(W)})
    return pd.DataFrame(rows)


def _classify_survival(rho: float, var_ratio: float) -> str:
    if not np.isfinite(rho) or not np.isfinite(var_ratio): return "insufficient"
    if rho <= -RANK_RELATED_MIN: return "inverted"
    if var_ratio <= EFFECT_ABSENT_RATIO or abs(rho) < 0.20: return "absent"
    if rho >= RANK_PRESERVED_MIN and EFFECT_ATTENUATED_RATIO <= var_ratio <= EFFECT_AMPLIFIED_RATIO: return "preserved"
    if rho >= RANK_RELATED_MIN and var_ratio < EFFECT_ATTENUATED_RATIO: return "attenuated"
    if rho >= RANK_RELATED_MIN and var_ratio > EFFECT_AMPLIFIED_RATIO: return "amplified"
    return "reorganized"


def _law_survival_tables(summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = summary[summary.checkpoint.astype(str) == "t360"].copy() if len(summary) else pd.DataFrame()
    if d.empty: return pd.DataFrame(), pd.DataFrame()
    pairs = [
        ("S_path", "structural_S_path", "conditional_S_path_median"),
        ("entropy_deficit", "structural_entropy_deficit", "conditional_entropy_deficit_median"),
        ("eta1", "structural_eta1", "conditional_eta1_median"),
        ("eta2", "structural_eta2", "conditional_eta2_median"),
        ("S_occ", "structural_hidden_mean_S_occ", "conditional_hidden_mean_S_occ_median"),
        ("S_param", "structural_hidden_mean_S_param", "conditional_hidden_mean_S_param_median"),
        ("S_reuse", "structural_hidden_mean_S_reuse", "conditional_hidden_mean_S_reuse_median"),
        ("eta1_occ", "structural_hidden_mean_eta1_occ", "conditional_hidden_mean_eta1_occ_median"),
        ("eta1_param", "structural_hidden_mean_eta1_param", "conditional_hidden_mean_eta1_param_median"),
        ("eta1_reuse", "structural_hidden_mean_eta1_reuse", "conditional_hidden_mean_eta1_reuse_median"),
    ]
    pairs = [p for p in pairs if p[1] in d.columns and p[2] in d.columns]
    seed_rows = []; summary_rows = []
    for name, sc, cc in pairs:
        for seed in SEEDS:
            g = d[d.seed_index == seed].set_index("arch_id").reindex([x[0] for x in FROZEN_ARCHITECTURES])
            rho = _safe_spearman(pd.to_numeric(g[sc], errors="coerce"), pd.to_numeric(g[cc], errors="coerce"))
            seed_rows.append({"observable": name, "seed_index": seed, "architecture_rank_spearman": rho})
        gm = d.groupby("arch_id", as_index=False)[[sc, cc]].mean(numeric_only=True).set_index("arch_id").reindex([x[0] for x in FROZEN_ARCHITECTURES])
        xs = pd.to_numeric(gm[sc], errors="coerce").to_numpy(float); yc = pd.to_numeric(gm[cc], errors="coerce").to_numpy(float)
        rho = _safe_spearman(xs, yc); var_s = float(np.nanvar(xs, ddof=1)); var_c = float(np.nanvar(yc, ddof=1)); vr = var_c / var_s if var_s > EPS else np.nan
        deform = pd.to_numeric(d[cc], errors="coerce") - pd.to_numeric(d[sc], errors="coerce")
        seedvals = [r["architecture_rank_spearman"] for r in seed_rows if r["observable"] == name]
        summary_rows.append({
            "observable": name, "rank_survival_rho_architecture_mean": rho,
            "rank_survival_seed_median": float(np.nanmedian(seedvals)), "rank_survival_seed_min": float(np.nanmin(seedvals)),
            "structural_between_arch_variance": var_s, "conditional_between_arch_variance": var_c,
            "effect_size_variance_ratio": vr, "paired_deformation_mean": float(np.nanmean(deform)),
            "paired_deformation_median": float(np.nanmedian(deform)), "classification": _classify_survival(rho, vr),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(seed_rows)


def _paired_deformation_table(summary: pd.DataFrame) -> pd.DataFrame:
    d = summary[summary.checkpoint.astype(str) == "t360"].copy() if len(summary) else pd.DataFrame()
    if d.empty: return pd.DataFrame()
    pairs = {
        "S_path": ("structural_S_path", "conditional_S_path_median"),
        "entropy_deficit": ("structural_entropy_deficit", "conditional_entropy_deficit_median"),
        "eta1": ("structural_eta1", "conditional_eta1_median"),
        "eta2": ("structural_eta2", "conditional_eta2_median"),
        "S_param": ("structural_hidden_mean_S_param", "conditional_hidden_mean_S_param_median"),
        "S_reuse": ("structural_hidden_mean_S_reuse", "conditional_hidden_mean_S_reuse_median"),
    }
    rows = []
    for obs, (sc, cc) in pairs.items():
        if sc not in d or cc not in d: continue
        for _, r in d.iterrows():
            a = float(r[sc]); b = float(r[cc]);
            rows.append({"run_id": r.run_id, "arch_id": r.arch_id, "seed_index": int(r.seed_index), "observable": obs,
                         "structural": a, "conditional_median": b, "delta_c": b-a,
                         "log_ratio_c": math.log(max(b, EPS)/max(a, EPS)) if a > 0 and b > 0 else np.nan})
    return pd.DataFrame(rows)


def _retention_test(summary: pd.DataFrame) -> pd.DataFrame:
    d = summary[summary.checkpoint.astype(str) == "t360"].copy() if len(summary) else pd.DataFrame()
    a = "conditional_rho_parameter_to_structural_median"; b = "conditional_rho_occurrence_to_structural_median"
    if d.empty or a not in d or b not in d: return pd.DataFrame()
    diff = pd.to_numeric(d[a], errors="coerce") - pd.to_numeric(d[b], errors="coerce"); diff = diff[np.isfinite(diff)]
    if len(diff):
        try: stat, p = wilcoxon(diff, alternative="greater", zero_method="wilcox")
        except Exception: stat, p = np.nan, np.nan
    else: stat, p = np.nan, np.nan
    nz = diff[diff != 0]
    if len(nz):
        ranks = rankdata(np.abs(nz)); rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    else: rb = np.nan
    return pd.DataFrame([{"hypothesis": "rho_parameter > rho_occurrence", "n_trained_models": len(diff),
                          "median_difference": float(np.nanmedian(diff)) if len(diff) else np.nan,
                          "wilcoxon_statistic": float(stat), "p_value_one_sided": float(p), "rank_biserial": rb}])


def _variance_decomposition(input_df: pd.DataFrame) -> pd.DataFrame:
    d = input_df[(input_df.checkpoint.astype(str) == "t360") & input_df.valid.astype(bool)].copy() if len(input_df) else pd.DataFrame()
    if d.empty: return pd.DataFrame()
    metrics = [c for c in ["S_path", "entropy_deficit", "eta1", "eta2", "hellinger_to_structural",
                           "hidden_mean_S_param", "hidden_mean_S_reuse", "rho_parameter_to_structural", "rho_occurrence_to_structural"] if c in d]
    rows = []
    for c in metrics:
        x = pd.to_numeric(d[c], errors="coerce"); dd = d.assign(_y=x).dropna(subset=["_y"])
        model_means = dd.groupby(["arch_id", "seed_index"], as_index=False)._y.mean()
        arch_means = model_means.groupby("arch_id")._y.mean()
        between_arch = float(np.var(arch_means, ddof=1)) if len(arch_means) > 1 else np.nan
        model_resid = model_means.merge(arch_means.rename("_arch"), left_on="arch_id", right_index=True)
        between_instance = float(np.mean(model_resid.groupby("arch_id")._y.var(ddof=1))) if model_resid.seed_index.nunique() > 1 else np.nan
        within_input = float(dd.groupby(["arch_id", "seed_index"])._y.var(ddof=1).mean())
        comp = np.array([between_arch, between_instance, within_input], float); den = float(np.nansum(comp))
        rows.append({"metric": c, "between_architecture_variance": between_arch,
                     "between_trained_instance_variance": between_instance, "within_model_input_variance": within_input,
                     "architecture_share_of_component_sum": between_arch/den if den > 0 else np.nan,
                     "trained_instance_share_of_component_sum": between_instance/den if den > 0 else np.nan,
                     "input_share_of_component_sum": within_input/den if den > 0 else np.nan})
    return pd.DataFrame(rows)


def _geometry_noncommutativity(geometry: pd.DataFrame, input_geometry: pd.DataFrame) -> pd.DataFrame:
    if geometry.empty or input_geometry.empty: return pd.DataFrame()
    pop = geometry[geometry.geometry_source.astype(str) == "conditional_population"].copy()
    metrics = [c for c in ["xi50", "d_star", "lacunarity_star", "local_delta_median", "quotient_mean_pairwise_KS", "renorm_flow_retention_mean"] if c in pop and c in input_geometry]
    med = input_geometry.groupby(["run_id", "checkpoint"], as_index=False)[metrics].median(numeric_only=True)
    z = pop.merge(med, on=["run_id", "checkpoint"], suffixes=("_population", "_median_input"))
    rows = []
    for _, r in z.iterrows():
        base = {"run_id": r.run_id, "arch_id": r.arch_id, "seed_index": int(r.seed_index), "checkpoint": r.checkpoint}
        for c in metrics:
            rows.append({**base, "metric": c, "population_geometry": float(r[f"{c}_population"]),
                         "median_input_geometry": float(r[f"{c}_median_input"]),
                         "population_minus_median_input": float(r[f"{c}_population"] - r[f"{c}_median_input"])})
    return pd.DataFrame(rows)


def run_aggregate_phase():
    banner("CNN-C3 PHASE: AGGREGATE + DISPOSITION-TO-REALIZATION LAW SURVIVAL")
    allow_partial = os.environ.get("PATHFLOW_CNN_C3_ALLOW_PARTIAL_AGGREGATE", "0") == "1"
    expected = {
        "training summaries": (len(training_manifest()), len(list((OUTDIR/"run_parts").glob("*__summary.csv")))),
        "conditional summaries": (len(condition_manifest()), len(list((OUTDIR/"condition_parts").glob("*__summary.csv")))),
        "principal geometry": (len(geometry_manifest()), len([p for p in (OUTDIR/"geometry_parts").glob("*.csv") if not p.name.endswith("__scales.csv") and "__local" not in p.name])),
        "input geometry": (len(input_geometry_manifest()), len([p for p in (OUTDIR/"input_geometry_parts").glob("*.csv") if not p.name.endswith("__scales.csv") and "__local" not in p.name])),
    }
    missing = {k: (a,b) for k,(a,b) in expected.items() if b < a}
    if missing and not allow_partial:
        raise RuntimeError("Aggregate refused incomplete campaign. Missing counts: " + json.dumps(missing) + ". Set PATHFLOW_CNN_C3_ALLOW_PARTIAL_AGGREGATE=1 only for an intentional interim audit.")
    if missing: trace(f"[aggregate] PARTIAL aggregation explicitly allowed: {missing}")

    train_summary = _concat_parts("run_parts/*__summary.csv"); train_history = _concat_parts("run_parts/*__history.csv")
    cond_summary = _concat_parts("condition_parts/*__summary.csv"); cond_inputs = _concat_parts("condition_parts/*__inputs.csv")
    cond_layers = _concat_parts("condition_parts/*__layers.csv"); pop_layers = _concat_parts("condition_parts/*__population_layers.csv")
    specialization = _concat_parts("condition_parts/*__specialization.csv")
    geom = pd.concat([safe_read_csv(p) for p in sorted((OUTDIR/"geometry_parts").glob("*.csv")) if not p.name.endswith("__scales.csv") and "__local" not in p.name], ignore_index=True) if any(True for _ in (OUTDIR/"geometry_parts").glob("*.csv")) else pd.DataFrame()
    igeom = pd.concat([safe_read_csv(p) for p in sorted((OUTDIR/"input_geometry_parts").glob("*.csv")) if not p.name.endswith("__scales.csv") and "__local" not in p.name], ignore_index=True) if any(True for _ in (OUTDIR/"input_geometry_parts").glob("*.csv")) else pd.DataFrame()
    gscales = _concat_parts("geometry_parts/*__scales.csv"); igscales = _concat_parts("input_geometry_parts/*__scales.csv")

    tables = {
        "run_summary.csv": train_summary, "training_history.csv": train_history,
        "conditional_checkpoint_summary.csv": cond_summary, "conditional_input_observations.csv": cond_inputs,
        "conditional_layer_input_observations.csv": cond_layers, "conditional_population_layers.csv": pop_layers,
        "input_specialization.csv": specialization, "principal_geometry.csv": geom,
        "principal_geometry_scales.csv": gscales, "input_resolved_geometry.csv": igeom,
        "input_resolved_geometry_scales.csv": igscales,
    }
    for name, df in tables.items(): atomic_csv(df, OUTDIR/"tables"/name)

    law, seed_rank = _law_survival_tables(cond_summary); deformation = _paired_deformation_table(cond_summary)
    friedman = _friedman_architecture_table(cond_summary); retention = _retention_test(cond_summary)
    variance = _variance_decomposition(cond_inputs); noncomm = _geometry_noncommutativity(geom, igeom)
    derived = {
        "architecture_law_survival.csv": law, "architecture_rank_survival_by_seed.csv": seed_rank,
        "paired_structural_conditional_deformation.csv": deformation, "friedman_architecture_tests.csv": friedman,
        "parameter_vs_occurrence_retention_test.csv": retention, "conditional_variance_decomposition.csv": variance,
        "geometry_population_vs_typical_input.csv": noncomm,
    }
    for name, df in derived.items(): atomic_csv(df, OUTDIR/"tables"/name)

    # Compact machine-readable campaign audit.
    audit = {"expected_counts": expected, "partial": bool(missing), "n_train_summary": len(train_summary),
             "n_conditional_summary": len(cond_summary), "n_conditional_inputs": len(cond_inputs),
             "n_principal_geometry_rows": len(geom), "n_input_geometry_rows": len(igeom),
             "law_survival_class_counts": law.classification.value_counts(dropna=False).to_dict() if len(law) else {}}
    atomic_json(audit, OUTDIR/"audit"/"aggregate_audit.json")
    trace(f"[aggregate] wrote publication tables to {OUTDIR/'tables'}")
    if len(law):
        trace("[aggregate] architecture-law survival classifications: " + json.dumps(audit["law_survival_class_counts"]))



# ==================================================================================================
# 13. CNN-C3 SELF-CONTAINED COMPACT OVERRIDES — OPTIMIZER AS THE ONLY EXPERIMENTAL FACTOR
# ==================================================================================================

CONDITIONAL_ROOT = Path(
    "<PROJECT_ROOT>/compress/sprime/BCANw_reference_fit/"
    "path_flow/autoDl_search/128_G/CNN_S_C/conditional"
)
STRUCTURAL_ROOT = CONDITIONAL_ROOT.parent / "structural"
S2_DIR = STRUCTURAL_ROOT / "S2" / "pathflow_cnn_s2_environment_structural_disposition_results"

OUTDIR = Path(os.environ.get(
    "PATHFLOW_CNN_C3_OUTDIR",
    str(CONDITIONAL_ROOT / "C3" / "pathflow_cnn_c3_optimizer_conditional_realization_SELFCONTAINED_10H_results"),
)).expanduser()
for sub in ["tables", "figures", "checkpoints", "run_parts", "resume_parts", "condition_parts",
            "condition_states", "geometry_parts", "audit"]:
    (OUTDIR / sub).mkdir(parents=True, exist_ok=True)

# ------------------------------- frozen compact production design ---------------------------------
FULL_EPOCHS = int(os.environ.get("PATHFLOW_CNN_C3_EPOCHS", "80"))
if FULL_EPOCHS != 80:
    raise ValueError("This bounded C3 production design is frozen at 80 epochs. Do not change PATHFLOW_CNN_C3_EPOCHS.")
STATE_GRID = [0, 1, 5, 20, 40, 80]
SEEDS = [0, 1]
OPTIMIZERS = ["adam", "rmsprop", "sgdm", "sgd"]
OPTIMIZER_LABEL = {"adam":"Adam", "rmsprop":"RMSProp", "sgdm":"SGD+momentum", "sgd":"SGD"}
OPTIMIZER_LR = {"adam":3e-3, "rmsprop":3e-3, "sgdm":3e-2, "sgd":1e-2}
N_COMPACT_ARCH = 5
N_COMPACT_ENV = 3
ARCH_PREFERRED = ["C1_shallow3", "C2_medium3", "C3_bottleneck", "C4_pointwise", "C5_deep3"]
GEOMETRY_ARCH_PREFERRED = ["C2_medium3", "C3_bottleneck", "C4_pointwise"]
N_GEOMETRY_ENV = 2
LONGITUDINAL_INPUTS = int(os.environ.get("PATHFLOW_CNN_C3_LONGITUDINAL_INPUTS", "64"))
DEEP_INPUTS = int(os.environ.get("PATHFLOW_CNN_C3_DEEP_INPUTS", "128"))
CONDITIONAL_BATCH = max(1, int(os.environ.get("PATHFLOW_CNN_C3_CONDITIONAL_BATCH", "16")))
BCANW_SCALES_LONG = int(os.environ.get("PATHFLOW_CNN_C3_BCANW_SCALES", "24"))
C3_SEED = 9143
C3_VERBOSE_TRAIN = max(1, int(os.environ.get("PATHFLOW_CNN_C3_LOG_EVERY_TRAIN", "5")))
C3_VERBOSE_CONDITION = max(1, int(os.environ.get("PATHFLOW_CNN_C3_LOG_EVERY_CONDITION", "50")))
C3_VERBOSE_GEOMETRY = max(1, int(os.environ.get("PATHFLOW_CNN_C3_LOG_EVERY_GEOMETRY", "10")))
SAVE_RESUME_EVERY_BATCHES = max(1, int(os.environ.get("PATHFLOW_CNN_C3_RESUME_EVERY_BATCHES", "40")))

# Hard resource contract.  The scientific design below should sit comfortably below these limits; the
# guards exist so a malformed configuration cannot silently turn this compact experiment back into S3.
C3_OUTPUT_CAP_GB = float(os.environ.get("PATHFLOW_CNN_C3_OUTPUT_CAP_GB", "9.5"))
C3_TARGET_WALL_HOURS = float(os.environ.get("PATHFLOW_CNN_C3_TARGET_WALL_HOURS", "9.75"))
C3_WALL_BUFFER_SECONDS = float(os.environ.get("PATHFLOW_CNN_C3_WALL_BUFFER_SECONDS", "600"))
SOFT_WALLTIME_SECONDS = C3_TARGET_WALL_HOURS * 3600.0
SOFT_WALLTIME_BUFFER_SECONDS = C3_WALL_BUFFER_SECONDS
MIN_FREE_AFTER_GB = float(os.environ.get("PATHFLOW_CNN_C3_MIN_FREE_AFTER_GB", "1.0"))
TQDM_ENABLED = os.environ.get("PATHFLOW_CNN_C3_TQDM", "0" if os.environ.get("SLURM_JOB_ID") else "1") != "0"

ARCH_PANEL_S2 = S2_DIR / "tables" / "frozen_s1_12_architecture_panel.csv"
ARCH_PANEL_S2_FALLBACK = S2_DIR / "tables" / "s2_12_architecture_panel.csv"
ENV_PANEL_S2 = S2_DIR / "tables" / "selected_24_environments.csv"


def c3_run_id(env_id: str, arch_id: str, seed_index: int, optimizer: str) -> str:
    return f"C3SELF__{env_id}__{arch_id}__S{int(seed_index):02d}__{optimizer}"


def c3_checkpoint_path(rid: str, epoch: int) -> Path:
    return OUTDIR / "checkpoints" / f"{rid}__E{int(epoch):03d}.pt"


def c3_resume_path(rid: str) -> Path:
    return OUTDIR / "resume_parts" / f"{rid}__resume.pt"


def _full_architecture_panel() -> pd.DataFrame:
    p = ARCH_PANEL_S2 if ARCH_PANEL_S2.exists() else ARCH_PANEL_S2_FALLBACK
    if p.exists():
        d = pd.read_csv(p).drop_duplicates("arch_id").reset_index(drop=True)
    else:
        rows=[]
        for i,(aid,hidden,kernels) in enumerate(FROZEN_ARCHITECTURES):
            rows.append({"architecture_index":i,"arch_id":aid,"hidden_channels":json.dumps(hidden),
                         "kernel_sizes":json.dumps(kernels), **architecture_descriptors(hidden,kernels)})
        d=pd.DataFrame(rows)
    req={"arch_id","hidden_channels","kernel_sizes"}
    if not req.issubset(d.columns): raise ValueError(f"Architecture panel missing {sorted(req-set(d.columns))}")
    if "architecture_index" not in d: d.insert(0,"architecture_index",np.arange(len(d),dtype=int))
    for i,r in d.iterrows():
        desc=architecture_descriptors(parse_int_list(r.hidden_channels),parse_int_list(r.kernel_sizes))
        for k,v in desc.items():
            if k not in d.columns: d[k]=np.nan
            if pd.isna(d.at[i,k]): d.at[i,k]=v
    return d


def architecture_panel() -> pd.DataFrame:
    d=_full_architecture_panel()
    ids=[x for x in ARCH_PREFERRED if x in set(d.arch_id.astype(str))]
    if len(ids)<N_COMPACT_ARCH:
        # Deterministic maximin completion from structural descriptors.
        cols=[c for c in ["parameter_count","occurrence_count","execution_nodes","output_resolution",
                          "receptive_field","depth","mean_hidden_width","max_hidden_width"] if c in d]
        X=d[cols].astype(float).to_numpy(); med=np.nanmedian(X,axis=0); mad=np.nanmedian(np.abs(X-med),axis=0)
        X=(X-med)/np.where(mad>0,mad,1.0)
        chosen=[int(d.index[d.arch_id.astype(str)==a][0]) for a in ids]
        if not chosen: chosen=[int(np.argmax(np.linalg.norm(X-X.mean(0),axis=1)))]
        while len(chosen)<min(N_COMPACT_ARCH,len(d)):
            dist=np.min(np.stack([np.linalg.norm(X-X[j],axis=1) for j in chosen]),axis=0); dist[chosen]=-np.inf
            chosen.append(int(np.argmax(dist)))
        ids=d.iloc[chosen].arch_id.astype(str).tolist()
    ids=ids[:N_COMPACT_ARCH]
    return d.set_index("arch_id").loc[ids].reset_index()


def _full_environment_panel() -> pd.DataFrame:
    if not ENV_PANEL_S2.exists():
        raise FileNotFoundError(f"C3 self-contained training needs only the frozen S2 environment metadata, missing: {ENV_PANEL_S2}")
    d=pd.read_csv(ENV_PANEL_S2).drop_duplicates("env_id").reset_index(drop=True)
    req={"env_id",*ENV_KEYS}
    if not req.issubset(d.columns): raise ValueError(f"Environment panel missing {sorted(req-set(d.columns))}")
    for c in ["informative_fraction","spatial_correlation","snr","interaction_strength","localization"]:
        d[c]=pd.to_numeric(d[c],errors="raise")
    return d


def _environment_embedding(d: pd.DataFrame) -> np.ndarray:
    num=d[["informative_fraction","spatial_correlation","snr","interaction_strength","localization"]].astype(float).to_numpy()
    med=np.nanmedian(num,axis=0); mad=np.nanmedian(np.abs(num-med),axis=0); num=(num-med)/np.where(mad>0,mad,1.0)
    fams=sorted(d.teacher_family.astype(str).unique()); one=np.column_stack([(d.teacher_family.astype(str).to_numpy()==f).astype(float) for f in fams])
    return np.column_stack([num,one])


def environment_panel() -> pd.DataFrame:
    d=_full_environment_panel(); X=_environment_embedding(d)
    can=np.where(d.env_id.astype(str).to_numpy()=="E_CAN_DISTRIBUTED")[0]
    if len(can): chosen=[int(can[0])]
    else:
        # nearest row to the embedded canonical numeric coordinates when the literal id is absent
        target=np.array([CANONICAL_ENV[k] for k in ["informative_fraction","spatial_correlation","snr","interaction_strength","localization"]],float)
        raw=d[["informative_fraction","spatial_correlation","snr","interaction_strength","localization"]].astype(float).to_numpy()
        chosen=[int(np.argmin(np.linalg.norm((raw-target)/np.maximum(np.nanstd(raw,axis=0),1e-8),axis=1)))]
    while len(chosen)<min(N_COMPACT_ENV,len(d)):
        dist=np.min(np.stack([np.linalg.norm(X-X[j],axis=1) for j in chosen]),axis=0); dist[chosen]=-np.inf
        chosen.append(int(np.argmax(dist)))
    return d.iloc[chosen].copy().reset_index(drop=True)


def environment_row(env_id: str) -> Dict[str,Any]:
    z=environment_panel(); z=z[z.env_id.astype(str)==str(env_id)]
    if len(z)!=1: raise KeyError(env_id)
    r=z.iloc[0]
    return {k:(str(r[k]) if k in {"env_id","teacher_family"} else float(r[k])) for k in ["env_id",*ENV_KEYS]}


def training_manifest() -> pd.DataFrame:
    rows=[]
    for _,e in environment_panel().iterrows():
        for _,a in architecture_panel().iterrows():
            for seed in SEEDS:
                for opt in OPTIMIZERS:
                    rows.append({
                        "run_id":c3_run_id(str(e.env_id),str(a.arch_id),seed,opt),
                        "env_id":str(e.env_id),"arch_id":str(a.arch_id),"architecture_index":int(a.architecture_index),
                        "seed_index":int(seed),"optimizer":opt,"optimizer_label":OPTIMIZER_LABEL[opt],
                        "learning_rate":float(OPTIMIZER_LR[opt]),"teacher_family":str(e.teacher_family),
                        "informative_fraction":float(e.informative_fraction),"spatial_correlation":float(e.spatial_correlation),
                        "snr":float(e.snr),"interaction_strength":float(e.interaction_strength),"localization":float(e.localization),
                    })
    return pd.DataFrame(rows)


def condition_manifest() -> pd.DataFrame:
    rows=[]
    for _,r in training_manifest().iterrows():
        for ep in STATE_GRID:
            n=DEEP_INPUTS if ep in (0,FULL_EPOCHS) else LONGITUDINAL_INPUTS
            label=epoch_label(ep)
            rows.append({**r.to_dict(),"checkpoint":label,"epoch_nominal":int(ep),"n_inputs":int(n),
                         "job_id":f"{r.run_id}__{label}"})
    return pd.DataFrame(rows)


def _geometry_arch_ids() -> List[str]:
    ids=architecture_panel().arch_id.astype(str).tolist()
    out=[a for a in GEOMETRY_ARCH_PREFERRED if a in ids]
    for a in ids:
        if a not in out and len(out)<3: out.append(a)
    return out[:3]


def _geometry_env_ids() -> List[str]:
    e=environment_panel(); ids=e.env_id.astype(str).tolist()
    # environment_panel is ordered canonical -> farthest -> maximin third
    return ids[:min(N_GEOMETRY_ENV,len(ids))]


def geometry_manifest() -> pd.DataFrame:
    aa=set(_geometry_arch_ids()); ee=set(_geometry_env_ids()); rows=[]
    for _,r in training_manifest().iterrows():
        if str(r.arch_id) not in aa or str(r.env_id) not in ee: continue
        label=epoch_label(FULL_EPOCHS)
        rows.append({**r.to_dict(),"checkpoint":label,"job_id":f"{r.run_id}__{label}__geometry"})
    return pd.DataFrame(rows)


def input_geometry_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=["job_id"])


def build_model(arch_id: str, seed_index: int) -> ConvTraceNet:
    q=architecture_panel(); z=q[q.arch_id.astype(str)==str(arch_id)]
    if len(z)!=1: raise KeyError(arch_id)
    return _arch_from_row(z.iloc[0],int(seed_index))


def make_optimizer_c3(name: str, parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    lr=float(OPTIMIZER_LR[str(name)])
    if name=="adam": return torch.optim.Adam(parameters,lr=lr,betas=(0.9,0.999),weight_decay=WEIGHT_DECAY)
    if name=="rmsprop": return torch.optim.RMSprop(parameters,lr=lr,alpha=0.99,eps=1e-8,momentum=0.0,centered=False,weight_decay=WEIGHT_DECAY)
    if name=="sgdm": return torch.optim.SGD(parameters,lr=lr,momentum=0.9,nesterov=False,weight_decay=WEIGHT_DECAY)
    if name=="sgd": return torch.optim.SGD(parameters,lr=lr,momentum=0.0,nesterov=False,weight_decay=WEIGHT_DECAY)
    raise ValueError(name)


_INITIAL_STATE_CACHE: Dict[Tuple[str,int],Tuple[Dict[str,torch.Tensor],List[float],str]] = {}

def exact_common_initial_state(arch_id: str, seed_index: int) -> Tuple[Dict[str,torch.Tensor],List[float],str]:
    key=(str(arch_id),int(seed_index))
    if key in _INITIAL_STATE_CACHE:
        state,scales,digest=_INITIAL_STATE_CACHE[key]
        return {k:v.clone() for k,v in state.items()},list(scales),digest
    model=build_model(arch_id,seed_index)
    scales=calibrate_initial_signal(model,calibration_batch(arch_id,seed_index))
    state=state_dict_cpu(model); digest=tensor_state_digest(state)
    _INITIAL_STATE_CACHE[key]=({k:v.clone() for k,v in state.items()},list(scales),digest)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {k:v.clone() for k,v in state.items()},list(scales),digest


def _optimizer_state_to_device(opt: torch.optim.Optimizer):
    for state in opt.state.values():
        for k,v in list(state.items()):
            if torch.is_tensor(v): state[k]=v.to(DEVICE)


def _save_c3_checkpoint(path: Path, model: nn.Module, row: pd.Series, epoch: int,
                        val_mse: float, test_mse: float, calibration_scales: Sequence[float],
                        initial_digest: str, diverged: bool=False, divergence_epoch: Optional[int]=None):
    atomic_torch_save({
        "schema":"cnn_c3_self_checkpoint_v1","run_id":str(row.run_id),"env_id":str(row.env_id),
        "arch_id":str(row.arch_id),"seed_index":int(row.seed_index),"optimizer":str(row.optimizer),
        "optimizer_label":str(row.optimizer_label),"learning_rate":float(row.learning_rate),"epoch":int(epoch),
        "val_mse":float(val_mse) if np.isfinite(val_mse) else np.nan,
        "test_mse":float(test_mse) if np.isfinite(test_mse) else np.nan,
        "diverged":bool(diverged),"divergence_epoch":None if divergence_epoch is None else int(divergence_epoch),
        "calibration_scales":list(map(float,calibration_scales)),"initial_digest":initial_digest,
        "model_state":state_dict_cpu(model),
    },path)


def train_one_c3(row: pd.Series, splits: Dict[str,Tuple[torch.Tensor,torch.Tensor]]) -> Dict[str,Any]:
    rid=str(row.run_id); summary_path=run_part(rid,"summary"); history_path=run_part(rid,"history")
    required=[c3_checkpoint_path(rid,e) for e in STATE_GRID]
    if RESUME and summary_path.exists() and all(p.exists() for p in required):
        d=safe_read_csv(summary_path); return d.iloc[0].to_dict() if len(d) else {}

    arch_id=str(row.arch_id); seed=int(row.seed_index); opt_name=str(row.optimizer)
    Xtr,ytr=splits["train"]; Xv,yv=splits["val"]; Xt,yt=splits["test"]
    model=build_model(arch_id,seed); init_state,scales,digest=exact_common_initial_state(arch_id,seed)
    model.load_state_dict(init_state,strict=True); model.to(DEVICE)
    opt=make_optimizer_c3(opt_name,model.parameters()); rp=c3_resume_path(rid)
    history=[]; start_epoch=1; start_batch=0; diverged=False; divergence_epoch=None

    if RESUME and rp.exists():
        ck=torch.load(rp,map_location="cpu",weights_only=False)
        model.load_state_dict(ck["model_state"],strict=True); model.to(DEVICE)
        opt.load_state_dict(ck["optimizer_state"]); _optimizer_state_to_device(opt)
        start_epoch=int(ck["epoch"]); start_batch=int(ck["next_batch"]); history=list(ck.get("history",[]))
        scales=list(map(float,ck["calibration_scales"])); digest=str(ck["initial_digest"])
        trace(f"[train] resume {rid} e{start_epoch} batch{start_batch}")
    else:
        val0=eval_mse(model,Xv,yv); test0=eval_mse(model,Xt,yt)
        _save_c3_checkpoint(c3_checkpoint_path(rid,0),model,row,0,val0,test0,scales,digest)
        history.append({"run_id":rid,"env_id":str(row.env_id),"arch_id":arch_id,"seed_index":seed,
                        "optimizer":opt_name,"epoch":0,"train_mse":np.nan,"val_mse":val0,"test_mse":test0})
        atomic_csv(pd.DataFrame(history),history_path)

    t_run=time.perf_counter(); n_batches=int(math.ceil(len(Xtr)/BATCH_SIZE))
    for epoch in range(start_epoch,FULL_EPOCHS+1):
        model.train(); total=0.0; nobs=0; batches=list(deterministic_batches(len(Xtr),epoch,seed,BATCH_SIZE))
        b0=start_batch if epoch==start_epoch else 0
        for bj in range(b0,len(batches)):
            ids=batches[bj]
            if Xtr.device.type=="cuda":
                idx=torch.as_tensor(ids,device=Xtr.device,dtype=torch.long); xb=Xtr.index_select(0,idx); yb=ytr.index_select(0,idx)
            else:
                xb=Xtr[ids].to(DEVICE,non_blocking=True); yb=ytr[ids].to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True); pred=model(xb); loss=F.mse_loss(pred,yb)
            if not torch.isfinite(loss):
                diverged=True; divergence_epoch=epoch; break
            loss.backward(); opt.step(); bs=len(ids); total+=float(loss.item())*bs; nobs+=bs
            if (bj+1)%SAVE_RESUME_EVERY_BATCHES==0 or STOP_REQUESTED or soft_walltime_due():
                atomic_torch_save({"schema":"cnn_c3_self_resume_v1","run_id":rid,"epoch":epoch,"next_batch":bj+1,
                                   "history":history,"calibration_scales":scales,"initial_digest":digest,
                                   "model_state":state_dict_cpu(model),"optimizer_state":opt.state_dict()},rp)
            if STOP_REQUESTED or soft_walltime_due():
                raise KeyboardInterrupt("C3 time/signal guard after exact resume checkpoint")
        if diverged: break
        train_mse=total/max(nobs,1); start_batch=0
        if epoch in STATE_GRID:
            val=eval_mse(model,Xv,yv); test=eval_mse(model,Xt,yt)
            _save_c3_checkpoint(c3_checkpoint_path(rid,epoch),model,row,epoch,val,test,scales,digest)
            history.append({"run_id":rid,"env_id":str(row.env_id),"arch_id":arch_id,"seed_index":seed,
                            "optimizer":opt_name,"epoch":epoch,"train_mse":train_mse,"val_mse":val,"test_mse":test})
            atomic_csv(pd.DataFrame(history),history_path)
        elif epoch%10==0:
            # Minimal longitudinal training trace without validation overhead.
            history.append({"run_id":rid,"env_id":str(row.env_id),"arch_id":arch_id,"seed_index":seed,
                            "optimizer":opt_name,"epoch":epoch,"train_mse":train_mse,"val_mse":np.nan,"test_mse":np.nan})
            atomic_csv(pd.DataFrame(history),history_path)
        if rp.exists():
            try: rp.unlink()
            except Exception: pass

    seconds=time.perf_counter()-t_run
    if diverged:
        # Materialize remaining common states using the last finite model only as an explicit divergence marker.
        # These checkpoints are never interpreted as scientific states; loader checks the divergence flag first.
        for ep in STATE_GRID:
            p=c3_checkpoint_path(rid,ep)
            if not p.exists(): _save_c3_checkpoint(p,model,row,ep,np.nan,np.nan,scales,digest,True,divergence_epoch)
    final_cp=torch.load(c3_checkpoint_path(rid,FULL_EPOCHS),map_location="cpu",weights_only=False)
    out={"run_id":rid,"env_id":str(row.env_id),"arch_id":arch_id,"seed_index":seed,"optimizer":opt_name,
         "optimizer_label":OPTIMIZER_LABEL[opt_name],"learning_rate":float(row.learning_rate),
         "diverged":bool(diverged or final_cp.get("diverged",False)),"divergence_epoch":divergence_epoch,
         "final_val_mse":float(final_cp.get("val_mse",np.nan)),"final_test_mse":float(final_cp.get("test_mse",np.nan)),
         "initial_digest":digest,"training_seconds":float(seconds)}
    atomic_csv(pd.DataFrame([out]),summary_path)
    if rp.exists():
        try: rp.unlink()
        except Exception: pass
    del model,opt; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def run_training_phase_c3():
    banner("CNN-C3 PHASE 1/4 — SELF-CONTAINED MATCHED OPTIMIZER TRAINING")
    m=training_manifest(); jobs=m[m.run_id.map(in_this_shard)].reset_index(drop=True)
    if MAX_JOBS_THIS_INVOCATION>0: jobs=jobs.iloc[:MAX_JOBS_THIS_INVOCATION]
    done=0; diverged=0; t0=time.perf_counter(); env_cache={}
    for _,r in jobs.iterrows():
        eid=str(r.env_id)
        if eid not in env_cache: env_cache[eid]=environment_dataset(environment_row(eid))[0]
        out=train_one_c3(r,env_cache[eid]); done+=1; diverged+=int(bool(out.get("diverged",False)))
        if done==1 or done%C3_VERBOSE_TRAIN==0 or done==len(jobs):
            elapsed=time.perf_counter()-t0; eta=elapsed/max(done,1)*(len(jobs)-done)
            trace(f"[train] {done}/{len(jobs)} | {100*done/max(len(jobs),1):.1f}% | ETA {fmt_seconds(eta)} | "
                  f"diverged={diverged} | last={r.optimizer}/{r.env_id}/{r.arch_id}/S{int(r.seed_index):02d} "
                  f"val={out.get('final_val_mse',np.nan):.4g}")
        if own_output_bytes()>C3_OUTPUT_CAP_GB*(1024**3): raise RuntimeError("C3 output hard cap exceeded")
        if STOP_REQUESTED or soft_walltime_due():
            trace("[train] wall/signal guard reached at run boundary; exact resume state preserved")
            break
    env_cache.clear(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def load_model_checkpoint(arch_id: str, seed_index: int, label: str, env_id: Optional[str]=None,
                          optimizer: Optional[str]=None) -> Tuple[Optional[ConvTraceNet],Dict[str,Any]]:
    if env_id is None or optimizer is None: raise ValueError("C3 loading requires env_id and optimizer")
    if not str(label).startswith("t"): raise ValueError(label)
    ep=int(str(label)[1:]); rid=c3_run_id(env_id,arch_id,seed_index,optimizer); p=c3_checkpoint_path(rid,ep)
    if not p.exists(): raise FileNotFoundError(p)
    ck=torch.load(p,map_location="cpu",weights_only=False)
    if bool(ck.get("diverged",False)): return None,ck
    state=ck.get("model_state",ck.get("state_dict"))
    if state is None: raise KeyError(f"No model state in {p}")
    model=build_model(arch_id,seed_index); model.load_state_dict(state,strict=True); model.to(DEVICE); model.eval()
    return model,ck


def fixed_test_indices_for_env(env_id: str) -> np.ndarray:
    rng=np.random.default_rng(C3_SEED + int(stable_hash_dict({"env":str(env_id)},8),16)%10_000_019 + 4049)
    return np.asarray(rng.permutation(N_TEST)[:DEEP_INPUTS],dtype=int)


def _save_population_marker(path: Path, meta: Dict[str,Any], reason: str):
    mm=dict(meta); mm.update({"degenerate_all":True,"population_state_defined":False,"degeneracy_reason":str(reason)})
    atomic_npz(path,meta_json=np.asarray(json.dumps(mm,default=str)),n_node_layers=np.asarray(0,dtype=np.int64),
               n_occ_layers=np.asarray(0,dtype=np.int64),degenerate_all=np.asarray(1,dtype=np.int8))


def condition_one_c3(row: pd.Series, cache: Dict[str,Any]) -> Dict[str,Any]:
    job_id=str(row.job_id); sp=condition_part(job_id,"summary"); ip=condition_part(job_id,"inputs")
    lp=condition_part(job_id,"layers"); pp=condition_part(job_id,"population_layers"); xp=condition_part(job_id,"specialization")
    npz=condition_state_path(job_id); required=[sp,ip,lp,pp,xp,npz]
    if RESUME and all(p.exists() for p in required):
        d=safe_read_csv(sp); return d.iloc[0].to_dict() if len(d) else {}
    eid=str(row.env_id); aid=str(row.arch_id); seed=int(row.seed_index); opt=str(row.optimizer); label=str(row.checkpoint)
    if eid not in cache: cache[eid]=environment_dataset(environment_row(eid))[0]
    model,ck=load_model_checkpoint(aid,seed,label,env_id=eid,optimizer=opt)
    base={"job_id":job_id,"run_id":str(row.run_id),"env_id":eid,"arch_id":aid,"seed_index":seed,
          "optimizer":opt,"optimizer_label":OPTIMIZER_LABEL[opt],"learning_rate":float(row.learning_rate),
          "checkpoint":label,"epoch_actual":int(ck.get("epoch",row.epoch_nominal)),"epoch_nominal":int(row.epoch_nominal),
          "n_requested_inputs":int(row.n_inputs),"teacher_family":str(row.teacher_family),
          "informative_fraction":float(row.informative_fraction),"spatial_correlation":float(row.spatial_correlation),
          "snr":float(row.snr),"interaction_strength":float(row.interaction_strength),"localization":float(row.localization)}
    if model is None:
        summ={**base,"training_diverged":True,"degenerate_all":True,"population_state_defined":False,
              "conditional_valid_fraction":0.0,"n_valid":0,"degeneracy_reason":"optimizer_training_diverged"}
        atomic_csv(pd.DataFrame([summ]),sp); atomic_csv(pd.DataFrame(),ip); atomic_csv(pd.DataFrame(),lp)
        atomic_csv(pd.DataFrame(),pp); atomic_csv(pd.DataFrame(),xp); _save_population_marker(npz,base,"optimizer_training_diverged")
        return summ
    structural=cnn_structural_state(model,True); Xt,_=cache[eid]["test"]; ids=fixed_test_indices_for_env(eid)[:int(row.n_inputs)]
    result=conditional_batch_analysis(model,Xt,ids,structural,label,str(row.run_id),aid,seed)
    summ={**result["summary"],**base,"training_diverged":False}
    assign={"env_id":eid,"optimizer":opt,"optimizer_label":OPTIMIZER_LABEL[opt],"learning_rate":float(row.learning_rate)}
    atomic_csv(result["input_df"].assign(**assign),ip); atomic_csv(result["layer_input_df"].assign(**assign),lp)
    atomic_csv(result["population_layers"].assign(**assign),pp); atomic_csv(result["specialization"].assign(**assign),xp)
    if result["population_state"] is None:
        summ.update({"degenerate_all":True,"population_state_defined":False,
                     "degeneracy_reason":"all_sampled_conditional_partitions_zero_or_nonfinite"})
        _save_population_marker(npz,base,summ["degeneracy_reason"])
    else:
        summ.update({"degenerate_all":False,"population_state_defined":True,"degeneracy_reason":""})
        _save_population_npz(npz,result["population_state"],base)
    atomic_csv(pd.DataFrame([summ]),sp)
    del model,structural,result; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return summ


def run_condition_phase_c3():
    banner("CNN-C3 PHASE 2/4 — STRUCTURAL -> CONDITIONAL REALIZATION")
    m=condition_manifest(); jobs=m[m.job_id.map(in_this_shard)].reset_index(drop=True)
    if MAX_JOBS_THIS_INVOCATION>0: jobs=jobs.iloc[:MAX_JOBS_THIS_INVOCATION]
    cache={}; done=0; deg=0; t0=time.perf_counter()
    for _,r in jobs.iterrows():
        out=condition_one_c3(r,cache); done+=1; deg+=int(bool(out.get("degenerate_all",False)))
        if done==1 or done%C3_VERBOSE_CONDITION==0 or done==len(jobs):
            elapsed=time.perf_counter()-t0; eta=elapsed/max(done,1)*(len(jobs)-done)
            trace(f"[condition] {done}/{len(jobs)} | {100*done/max(len(jobs),1):.1f}% | ETA {fmt_seconds(eta)} | "
                  f"undefined/diverged={deg} | last={r.optimizer}/{r.checkpoint}")
        if own_output_bytes()>C3_OUTPUT_CAP_GB*(1024**3): raise RuntimeError("C3 output hard cap exceeded")
        if STOP_REQUESTED or soft_walltime_due(): trace("[condition] wall/signal guard reached; atomic jobs preserved"); break
    cache.clear(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def geometry_one_c3(row: pd.Series) -> Dict[str,Any]:
    job_id=str(row.job_id); outp=geometry_part(job_id); scalep=geometry_scale_part(job_id)
    if RESUME and outp.exists() and scalep.exists():
        d=safe_read_csv(outp); return d.iloc[0].to_dict() if len(d) else {}
    eid=str(row.env_id); aid=str(row.arch_id); seed=int(row.seed_index); opt=str(row.optimizer); label=str(row.checkpoint)
    model,ck=load_model_checkpoint(aid,seed,label,env_id=eid,optimizer=opt)
    base={"job_id":job_id,"run_id":str(row.run_id),"env_id":eid,"arch_id":aid,"seed_index":seed,
          "optimizer":opt,"optimizer_label":OPTIMIZER_LABEL[opt],"checkpoint":label,
          "epoch_actual":int(ck.get("epoch",FULL_EPOCHS))}
    if model is None:
        rows=[{**base,"geometry_source":"structural","geometry_skipped":True,"geometry_skip_reason":"optimizer_training_diverged"},
              {**base,"geometry_source":"conditional_population","geometry_skipped":True,"geometry_skip_reason":"optimizer_training_diverged"}]
        atomic_csv(pd.DataFrame(rows),outp); atomic_csv(pd.DataFrame(),scalep); return rows[0]
    structural=cnn_structural_state(model,True); pop,pmeta=_load_population_npz(condition_state_path(f"{row.run_id}__{label}"))
    rows=[]; scales=[]
    for source,state in [("structural",structural),("conditional_population",pop)]:
        if state is None:
            rows.append({**base,"geometry_source":source,"geometry_skipped":True,
                         "geometry_skip_reason":pmeta.get("degeneracy_reason","conditional_population_undefined")})
            continue
        summ,sdf,_=analyze_execution_geometry_rich(model,state,BCANW_SCALES_LONG,compute_local=False)
        rows.append({**base,"geometry_source":source,**summ})
        if len(sdf):
            z=sdf.copy()
            for k,v in {**base,"geometry_source":source}.items(): z[k]=v
            scales.append(z)
    atomic_csv(pd.DataFrame(rows),outp); atomic_csv(pd.concat(scales,ignore_index=True) if scales else pd.DataFrame(),scalep)
    del model,structural,pop; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows[0]


def run_geometry_phase_c3():
    banner("CNN-C3 PHASE 3/4 — ENDPOINT STRUCTURAL + CONDITIONAL BCANw SENTINEL ARM")
    m=geometry_manifest(); jobs=m[m.job_id.map(in_this_shard)].reset_index(drop=True); done=0; skipped=0; t0=time.perf_counter()
    for _,r in jobs.iterrows():
        out=geometry_one_c3(r); done+=1; skipped+=int(bool(out.get("geometry_skipped",False)))
        if done==1 or done%C3_VERBOSE_GEOMETRY==0 or done==len(jobs):
            elapsed=time.perf_counter()-t0; eta=elapsed/max(done,1)*(len(jobs)-done)
            trace(f"[geometry] {done}/{len(jobs)} | {100*done/max(len(jobs),1):.1f}% | ETA {fmt_seconds(eta)} | skipped={skipped}")
        if own_output_bytes()>C3_OUTPUT_CAP_GB*(1024**3): raise RuntimeError("C3 output hard cap exceeded")
        if STOP_REQUESTED or soft_walltime_due(): trace("[geometry] wall/signal guard reached; atomic jobs preserved"); break


def _rank_biserial_paired(diff: np.ndarray) -> float:
    x=np.asarray(diff,float); x=x[np.isfinite(x)&(x!=0)]
    if not len(x): return np.nan
    ranks=rankdata(np.abs(x)); den=float(ranks.sum())
    return float((ranks[x>0].sum()-ranks[x<0].sum())/den) if den>0 else np.nan


def _matched_optimizer_tests(df: pd.DataFrame, metrics: List[str], block_cols: List[str], scope: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    omni=[]; pairs=[]
    for metric in metrics:
        if metric not in df: continue
        p=df.pivot_table(index=block_cols,columns="optimizer",values=metric,aggfunc="first")
        if not set(OPTIMIZERS).issubset(p.columns): continue
        p=p.dropna(subset=OPTIMIZERS)
        if len(p)<2: continue
        arr=[p[o].to_numpy(float) for o in OPTIMIZERS]
        try: stat,pv=friedmanchisquare(*arr)
        except Exception: stat,pv=np.nan,np.nan
        W=float(stat/(len(p)*(len(OPTIMIZERS)-1))) if np.isfinite(stat) and len(p) else np.nan
        omni.append({"scope":scope,"metric":metric,"n_complete_blocks":int(len(p)),"friedman_chi2":float(stat),"p_value":float(pv),"kendall_W":W})
        for a,b in itertools.combinations(OPTIMIZERS,2):
            diff=p[a].to_numpy(float)-p[b].to_numpy(float); diff=diff[np.isfinite(diff)]
            try: ws,wp=wilcoxon(diff,alternative="two-sided",zero_method="wilcox") if np.any(diff!=0) else (0.0,1.0)
            except Exception: ws,wp=np.nan,np.nan
            pairs.append({"scope":scope,"metric":metric,"optimizer_a":a,"optimizer_b":b,"n_complete_blocks":int(len(diff)),
                          "median_a_minus_b":float(np.nanmedian(diff)) if len(diff) else np.nan,
                          "wilcoxon_statistic":float(ws),"p_value":float(wp),"rank_biserial":_rank_biserial_paired(diff)})
    return pd.DataFrame(omni),pd.DataFrame(pairs)


def optimizer_endpoint_inference(cond: pd.DataFrame) -> Dict[str,pd.DataFrame]:
    endpoint=epoch_label(FULL_EPOCHS); d=cond[cond.checkpoint.astype(str)==endpoint].copy() if len(cond) else pd.DataFrame()
    if d.empty: return {k:pd.DataFrame() for k in ["phenotype","structural_omni","structural_pairs","conditional_omni","conditional_pairs","order","order_summary"]}
    cmetrics=[c for c in d.columns if c.startswith("conditional_") and c.endswith("_median")]
    smetrics=[c for c in d.columns if c.startswith("structural_") and pd.api.types.is_numeric_dtype(d[c])]
    phenotype=d.groupby(["optimizer","optimizer_label"],as_index=False)[cmetrics+ [x for x in smetrics if x not in cmetrics]].median(numeric_only=True)
    so,sp=_matched_optimizer_tests(d,smetrics,["env_id","arch_id","seed_index"],"structural_endpoint")
    co,cp=_matched_optimizer_tests(d,cmetrics,["env_id","arch_id","seed_index"],"conditional_endpoint")
    rows=[]
    for cm in cmetrics:
        stem=cm[len("conditional_"):-len("_median")]; sm="structural_"+stem
        if sm not in d: continue
        for key,g in d.groupby(["env_id","arch_id","seed_index"]):
            z=g.set_index("optimizer").reindex(OPTIMIZERS); xs=pd.to_numeric(z[sm],errors="coerce").to_numpy(float); yc=pd.to_numeric(z[cm],errors="coerce").to_numpy(float)
            if np.all(np.isfinite(xs)) and np.all(np.isfinite(yc)):
                rho=_safe_spearman(xs,yc); so_order=">".join([OPTIMIZERS[i] for i in np.argsort(-xs)]); co_order=">".join([OPTIMIZERS[i] for i in np.argsort(-yc)])
                rows.append({"metric":stem,"env_id":key[0],"arch_id":key[1],"seed_index":int(key[2]),
                             "optimizer_rank_spearman_structural_vs_conditional":rho,"structural_order_desc":so_order,
                             "conditional_order_desc":co_order,"exact_order_match":bool(so_order==co_order)})
    order=pd.DataFrame(rows)
    osum=order.groupby("metric",as_index=False).agg(n_blocks=("optimizer_rank_spearman_structural_vs_conditional","count"),
        median_rank_rho=("optimizer_rank_spearman_structural_vs_conditional","median"),
        mean_rank_rho=("optimizer_rank_spearman_structural_vs_conditional","mean"),
        exact_order_match_fraction=("exact_order_match","mean")) if len(order) else pd.DataFrame()
    return {"phenotype":phenotype,"structural_omni":so,"structural_pairs":sp,"conditional_omni":co,"conditional_pairs":cp,
            "order":order,"order_summary":osum}


def run_aggregate_phase_c3():
    banner("CNN-C3 PHASE 4/4 — MATCHED OPTIMIZER INFERENCE")
    train=_concat_parts("run_parts/*__summary.csv"); hist=_concat_parts("run_parts/*__history.csv")
    cond=_concat_parts("condition_parts/*__summary.csv"); inputs=_concat_parts("condition_parts/*__inputs.csv")
    layers=_concat_parts("condition_parts/*__layers.csv"); poplayers=_concat_parts("condition_parts/*__population_layers.csv")
    spec=_concat_parts("condition_parts/*__specialization.csv")
    frames=[safe_read_csv(p) for p in sorted((OUTDIR/"geometry_parts").glob("*.csv")) if not p.name.endswith("__scales.csv")]
    geom=pd.concat([d for d in frames if len(d)],ignore_index=True) if any(len(d) for d in frames) else pd.DataFrame()
    gscales=_concat_parts("geometry_parts/*__scales.csv")
    inf=optimizer_endpoint_inference(cond)
    endpoint=epoch_label(FULL_EPOCHS)
    gd=geom[(geom.checkpoint.astype(str)==endpoint) & (~geom.geometry_skipped.astype(bool))].copy() if len(geom) and "geometry_skipped" in geom else geom.copy()
    gmetrics=[c for c in ["xi50","log_xi50","d_star","x_d_star","lacunarity_star","scaling_dB","quotient_mean_pairwise_KS","renorm_flow_retention_mean"] if c in gd]
    gomni=[]; gpairs=[]
    for source,g in gd.groupby("geometry_source") if len(gd) else []:
        o,p=_matched_optimizer_tests(g,gmetrics,["env_id","arch_id","seed_index"],f"geometry_{source}")
        if len(o): gomni.append(o)
        if len(p): gpairs.append(p)
    gomni=pd.concat(gomni,ignore_index=True) if gomni else pd.DataFrame(); gpairs=pd.concat(gpairs,ignore_index=True) if gpairs else pd.DataFrame()
    tables={
        "training_summary.csv":train,"training_history.csv":hist,"conditional_checkpoint_summary.csv":cond,
        "conditional_input_observations.csv":inputs,"conditional_layer_input_observations.csv":layers,
        "conditional_population_layers.csv":poplayers,"input_specialization.csv":spec,
        "endpoint_geometry.csv":geom,"endpoint_geometry_scales.csv":gscales,
        "optimizer_endpoint_phenotypes.csv":inf["phenotype"],"optimizer_structural_omnibus.csv":inf["structural_omni"],
        "optimizer_structural_pairwise.csv":inf["structural_pairs"],"optimizer_conditional_omnibus.csv":inf["conditional_omni"],
        "optimizer_conditional_pairwise.csv":inf["conditional_pairs"],"optimizer_order_survival_by_block.csv":inf["order"],
        "optimizer_order_survival_summary.csv":inf["order_summary"],"optimizer_geometry_omnibus.csv":gomni,
        "optimizer_geometry_pairwise.csv":gpairs,
    }
    for name,df in tables.items(): atomic_csv(df,OUTDIR/"tables"/name)
    audit={"training_expected":len(training_manifest()),"training_found":int(train.run_id.nunique()) if len(train) else 0,
           "condition_expected":len(condition_manifest()),"condition_found":int(cond.job_id.nunique()) if len(cond) else 0,
           "geometry_expected":len(geometry_manifest()),"geometry_found":int(geom.job_id.nunique()) if len(geom) else 0,
           "diverged_runs":int(pd.to_numeric(train.get("diverged",False),errors="coerce").fillna(0).astype(bool).sum()) if len(train) else 0,
           "undefined_conditional_jobs":int(pd.to_numeric(cond.get("degenerate_all",False),errors="coerce").fillna(0).astype(bool).sum()) if len(cond) else 0,
           "elapsed_seconds":time.monotonic()-RUN_START,"output_bytes":own_output_bytes()}
    atomic_json(audit,OUTDIR/"audit"/"aggregate_audit.json")
    trace(f"[aggregate] train {audit['training_found']}/{audit['training_expected']} | condition {audit['condition_found']}/{audit['condition_expected']} | "
          f"geometry {audit['geometry_found']}/{audit['geometry_expected']} | diverged={audit['diverged_runs']} | undefined={audit['undefined_conditional_jobs']}")


def c3_disk_plan() -> Dict[str,Any]:
    tm=training_manifest(); cm=condition_manifest(); gm=geometry_manifest(); ap=architecture_panel()
    pmap=dict(zip(ap.arch_id.astype(str),pd.to_numeric(ap.parameter_count,errors="coerce").fillna(3000).astype(int)))
    omap=dict(zip(ap.arch_id.astype(str),pd.to_numeric(ap.occurrence_count,errors="coerce").fillna(100000).astype(int)))
    nmap=dict(zip(ap.arch_id.astype(str),pd.to_numeric(ap.execution_nodes,errors="coerce").fillna(5000).astype(int)))
    checkpoints=sum((len(STATE_GRID)*(16384+int(pmap.get(str(r.arch_id),3000)*4*1.3))) for _,r in tm.iterrows())
    resume_peak=max([65536+int(p*20) for p in pmap.values()] or [65536])
    states=sum(16384+8*(pmap.get(str(r.arch_id),3000)+omap.get(str(r.arch_id),100000)+nmap.get(str(r.arch_id),5000)) for _,r in cm.iterrows())
    input_rows=int(cm.n_inputs.sum()); input_csv=int(input_rows*1600); layer_csv=int(input_rows*4.5*800)
    geom=int(len(gm)*2*(BCANW_SCALES_LONG*650+3000)); tables=256*1024**2
    raw=checkpoints+resume_peak+states+input_csv+layer_csv+geom+tables; est=int(raw*1.35); current=own_output_bytes(); du=shutil.disk_usage(OUTDIR)
    return {"estimated_final_bytes":est,"estimated_remaining_bytes":max(est-current,0),"free_bytes":int(du.free),
            "hard_cap_bytes":int(C3_OUTPUT_CAP_GB*1024**3),"fits_cap":est<=C3_OUTPUT_CAP_GB*1024**3,
            "fits_fs":du.free>=max(est-current,0)+MIN_FREE_AFTER_GB*1024**3,
            "breakdown":{"checkpoints":checkpoints,"population_states":states,"input_csv":input_csv,"layer_csv":layer_csv,
                         "geometry":geom,"tables_audit_allowance":tables,"resume_peak":resume_peak}}


def c3_runtime_plan() -> Dict[str,Any]:
    # No pilot/benchmark. Priors are deliberately conservative for small 16x16 CNNs on a single modern GPU.
    train_batches=len(training_manifest())*FULL_EPOCHS*int(math.ceil(N_TRAIN/BATCH_SIZE))
    cond_inputs=int(condition_manifest().n_inputs.sum()); geom_evals=len(geometry_manifest())*2
    train_s=train_batches*0.025; cond_s=cond_inputs*0.05; geom_s=geom_evals*30.0; overhead=1800.0
    return {"train_batches":train_batches,"conditional_realizations":cond_inputs,"geometry_evaluations":geom_evals,
            "planning_seconds":train_s+cond_s+geom_s+overhead,"training_seconds_prior":train_s,
            "conditioning_seconds_prior":cond_s,"geometry_seconds_prior":geom_s,"overhead_seconds_prior":overhead}


def c3_preflight_and_manifest():
    banner("CNN-C3 PREFLIGHT — SELF-CONTAINED 10H / <10GB PRODUCTION DESIGN")
    ap=architecture_panel(); ep=environment_panel(); tm=training_manifest(); cm=condition_manifest(); gm=geometry_manifest()
    dp=c3_disk_plan(); rp=c3_runtime_plan()
    print(f"architectures={ap.arch_id.astype(str).tolist()}")
    print(f"environments={ep.env_id.astype(str).tolist()}")
    print(f"seeds={SEEDS} | optimizers={OPTIMIZERS} | fixed_lr={OPTIMIZER_LR}")
    print(f"training trajectories={len(tm)} | epochs={FULL_EPOCHS} | checkpoints={STATE_GRID}")
    print(f"conditional jobs={len(cm)} | conditional realizations={int(cm.n_inputs.sum()):,} | geometry jobs={len(gm)}")
    print(f"planning runtime (NO benchmark): {fmt_seconds(rp['planning_seconds'])} | hard target={C3_TARGET_WALL_HOURS:.2f}h")
    print("disk estimate:")
    for k,v in dp["breakdown"].items(): print(f"  {k:24s} {human_bytes(v)}")
    print(f"  {'estimated final':24s} {human_bytes(dp['estimated_final_bytes'])}")
    print(f"  {'hard output cap':24s} {C3_OUTPUT_CAP_GB:.2f} GiB")
    print(f"  {'filesystem free':24s} {human_bytes(dp['free_bytes'])}")
    print(f"  {'fits cap/filesystem':24s} {'YES' if dp['fits_cap'] and dp['fits_fs'] else 'NO'}")
    if not dp["fits_cap"]: raise RuntimeError("C3 design exceeds its <10GB hard output contract")
    if ENFORCE_DISK and not dp["fits_fs"]: raise RuntimeError("C3 disk preflight failed")
    if rp["planning_seconds"]>C3_TARGET_WALL_HOURS*3600: raise RuntimeError("C3 conservative no-benchmark runtime plan exceeds the frozen walltime target")
    atomic_csv(ap,OUTDIR/"tables"/"compact_architecture_panel.csv"); atomic_csv(ep,OUTDIR/"tables"/"compact_environment_panel.csv")
    atomic_csv(tm,OUTDIR/"tables"/"training_manifest.csv"); atomic_csv(cm,OUTDIR/"tables"/"condition_manifest.csv"); atomic_csv(gm,OUTDIR/"tables"/"geometry_manifest.csv")
    atomic_json({"design":"5A x 3E x 2s x 4O self-contained","epochs":FULL_EPOCHS,"state_grid":STATE_GRID,
                 "optimizer_lr":OPTIMIZER_LR,"architecture_ids":ap.arch_id.astype(str).tolist(),
                 "environment_ids":ep.env_id.astype(str).tolist(),"geometry_arch_ids":_geometry_arch_ids(),
                 "geometry_env_ids":_geometry_env_ids(),"runtime_plan":rp,"disk_plan":dp,
                 "walltime_target_hours":C3_TARGET_WALL_HOURS,"output_cap_gb":C3_OUTPUT_CAP_GB},OUTDIR/"audit"/"preflight.json")
    trace("[preflight] bounded design passed; continuing directly into full C3 execution")


def _phase_complete(kind: str) -> bool:
    if kind=="train": return len(list((OUTDIR/"run_parts").glob("*__summary.csv")))>=len(training_manifest())
    if kind=="condition": return len(list((OUTDIR/"condition_parts").glob("*__summary.csv")))>=len(condition_manifest())
    if kind=="geometry": return len([p for p in (OUTDIR/"geometry_parts").glob("*.csv") if not p.name.endswith("__scales.csv")])>=len(geometry_manifest())
    return False


def main():
    c3_preflight_and_manifest()
    if MODE=="preflight": return
    if MODE in {"full","all","train"}: run_training_phase_c3()
    if MODE=="train": return
    if not _phase_complete("train"):
        raise RuntimeError("Training phase is incomplete (likely wall/signal guard). Rerun the same command; exact resume files are preserved.")
    if MODE in {"full","all","condition"}: run_condition_phase_c3()
    if MODE=="condition": return
    if not _phase_complete("condition"):
        raise RuntimeError("Condition phase is incomplete. Rerun the same command; completed atomic jobs will be skipped.")
    if MODE in {"full","all","geometry"}: run_geometry_phase_c3()
    if MODE=="geometry": return
    if not _phase_complete("geometry"):
        raise RuntimeError("Geometry phase is incomplete. Rerun the same command; completed atomic jobs will be skipped.")
    if MODE in {"full","all","aggregate"}: run_aggregate_phase_c3()
    trace(f"CNN-C3 SELF-CONTAINED finished | elapsed={fmt_seconds(time.monotonic()-RUN_START)} | output={human_bytes(own_output_bytes())} | results={OUTDIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        trace(f"CNN-C3 interrupted safely: {exc}")
        if STOP_REQUESTED: raise
    except Exception:
        trace("CNN-C3 FAILED — completed atomic parts remain resumable; traceback follows")
        raise
