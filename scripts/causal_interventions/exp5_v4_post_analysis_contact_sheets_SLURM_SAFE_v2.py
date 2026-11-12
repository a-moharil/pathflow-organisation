from __future__ import annotations

# ==================================================================================================
# WORK II — EXPERIMENT 5 + 5.1 V4 POST-ANALYSIS — V2 SCHEMA FIX
# FULL CLASS / STRATUM NUMERICAL ANALYSIS + SYSTEMATIC VISUAL AUDIT CONTACT SHEETS
# HPC / SLURM SAFE | CPU ONLY | READ-ONLY SOURCE RESULTS | SEPARATE WRITE-ONCE OUTPUT
# ==================================================================================================
#
# PURPOSE
# -------
# The V4 visual atlas can contain ~2 GB of PNGs.  Uploading all source images is neither necessary nor
# scientifically desirable.  This script performs the complete numerical post-analysis on the HPC and
# reduces the visual review to a small, systematic contact-sheet package.
#
# Scientific evidence is NOT sampled:
#
#   Experiment 5 numerical evidence
#       field_diagnostics.csv
#       measurements.csv
#
#   Experiment 5.1 numerical evidence
#       field_symmetry_measurements.csv
#       behavior_measurements.csv
#
# are analyzed across the complete trained-CNN population, both environments, all 10 fixed conditions and
# all stored transforms.
#
# Only the VISUAL INSPECTION is reduced.  Representative cases are selected deterministically from the full
# numerical population using 10th / 50th / 90th percentile ranks, then clean and shifted renderings for the
# SAME frozen network are shown side by side.
#
# CONDITION SEMANTICS
# -------------------
# The condition remains
#
#       c = (x, x0) = (x, 0)
#
# throughout.
#
# Fashion-MNIST condition strata are classes 0..9.
# MPI3D is a regression benchmark, so its 10 conditions are horizontal_axis strata 0,4,...,36.
#
# No reference-class experiment is introduced here.
#
# OUTPUT
# ------
# Source root:
#
#   pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results/
#
# Read-only source subdirectories:
#
#   experiment5_cnn_occurrence_shared_identity/
#   experiment5_1_fieldA_realization_symmetry_geometric/
#   experiment5_1_all_conditions_visual_atlas_v4/
#
# New output only:
#
#   experiment5_1_v4_post_analysis/
#       tables/
#       contact_sheets/
#       audit/
#       upload_package/
#
# The three source experiment directories are NEVER modified.
#
# DEFAULT VISUAL PACKAGE
# ----------------------
# Fashion-MNIST, one set per class:
#   10 Field-A/Field-B sheets
#   10 translation sheets   (right_2)
#   10 rotation sheets      (rot_ccw_20)
#   10 scale sheets         (scale_up_110)
#
# MPI3D, one set per horizontal-position stratum:
#   10 Field-A/Field-B sheets
#   10 translation sheets   (down_2)
#
# Total core sheets = 60, plus a small number of global diagnostic sheets.
#
# Every contact sheet contains 3 selected frozen CNNs (low / median / high numerical score)
# and pairs clean versus shifted for the same network.
#
# HPC EXAMPLE
# -----------
# This is a CPU post-analysis.  A GPU is not required.
#
#   #SBATCH --time=24:00:00
#   #SBATCH --ntasks=1
#   #SBATCH --cpus-per-task=4
#   #SBATCH --mem=64G
#
#   srun python -u exp5_v4_post_analysis_contact_sheets_SLURM_SAFE_v1.py
#
# If the script is not placed beside the common result root:
#
#   export PATHFLOW_RESULTS_ROOT=/absolute/path/to/pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results
#
# Useful controls:
#
#   PATHFLOW_V4PA_PREFLIGHT_ONLY=1
#   PATHFLOW_V4PA_SELFTEST_ONLY=1
#   PATHFLOW_V4PA_CONTACT_DPI=120
#   PATHFLOW_V4PA_JPEG_QUALITY=88
#   PATHFLOW_V4PA_SOFT_WALL_HOURS=22
#
# ==================================================================================================

import os
import re
import io
import gc
import math
import json
import time
import zipfile
import hashlib
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, spearmanr

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================================================================================================
# 0. CONFIG
# ==================================================================================================

VERSION = "EXP5_EXP51_V4_POST_ANALYSIS_v2_SCHEMA_FIX"

CPU_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("PATHFLOW_V4PA_CPU_THREADS", "4")))
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, CPU_THREADS)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, CPU_THREADS)))

PREFLIGHT_ONLY = os.environ.get("PATHFLOW_V4PA_PREFLIGHT_ONLY", "0") == "1"
SELFTEST_ONLY = os.environ.get("PATHFLOW_V4PA_SELFTEST_ONLY", "0") == "1"

CONTACT_DPI = int(os.environ.get("PATHFLOW_V4PA_CONTACT_DPI", "120"))
JPEG_QUALITY = int(os.environ.get("PATHFLOW_V4PA_JPEG_QUALITY", "88"))

SOFT_WALL_HOURS = float(os.environ.get("PATHFLOW_V4PA_SOFT_WALL_HOURS", "22.0"))
SOFT_BUFFER_MIN = float(os.environ.get("PATHFLOW_V4PA_SOFT_BUFFER_MIN", "10.0"))
START_MONO = time.perf_counter()

SEED = 730031
rng_global = np.random.default_rng(SEED)

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

_root = os.environ.get("PATHFLOW_RESULTS_ROOT", "").strip()
RESULTS_ROOT = (
    Path(_root).expanduser().resolve()
    if _root
    else SCRIPT_DIR / "pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results"
)

EXP5 = RESULTS_ROOT / "experiment5_cnn_occurrence_shared_identity"
EXP51 = RESULTS_ROOT / "experiment5_1_fieldA_realization_symmetry_geometric"
V4 = RESULTS_ROOT / "experiment5_1_all_conditions_visual_atlas_v4"

_out = os.environ.get("PATHFLOW_V4PA_OUTDIR", "").strip()
OUT = (
    Path(_out).expanduser().resolve()
    if _out
    else RESULTS_ROOT / "experiment5_1_v4_post_analysis"
)

FORBIDDEN = {RESULTS_ROOT.resolve(), EXP5.resolve(), EXP51.resolve(), V4.resolve()}
if OUT.resolve() in FORBIDDEN:
    raise RuntimeError(f"Unsafe output directory {OUT}. Post-analysis must have its own isolated subtree.")

for sub in ["tables", "contact_sheets", "audit", "upload_package"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

_probe = OUT / "audit" / f".write_probe_{os.getpid()}"
with open(_probe, "x", encoding="utf-8") as f:
    f.write("ok")
_probe.unlink()

# ==================================================================================================
# 1. SAFE I/O
# ==================================================================================================

def file_sha1(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def publish_bytes_once(payload: bytes, path: Path) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        tmp = path.parent / f".{path.name}.compare.{os.getpid()}.{time.time_ns()}"
        tmp.write_bytes(payload)
        try:
            same = path.stat().st_size == tmp.stat().st_size and file_sha1(path) == file_sha1(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        if same:
            return False
        raise RuntimeError(f"Write-once protection refused to replace a different existing file: {path}")

    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

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
            path.unlink(missing_ok=True)
            raise
    finally:
        tmp.unlink(missing_ok=True)
    return True

def write_once_json(obj: Any, path: Path) -> bool:
    payload = (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    return publish_bytes_once(payload, path)

def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, float_format="%.17g").encode("utf-8")

def write_once_csv(df: pd.DataFrame, path: Path) -> bool:
    return publish_bytes_once(dataframe_csv_bytes(df), path)

def write_text_once(text: str, path: Path) -> bool:
    return publish_bytes_once(text.encode("utf-8"), path)

def soft_remaining() -> float:
    return max(0.0, SOFT_WALL_HOURS * 3600.0 - (time.perf_counter() - START_MONO))

def can_start_visual_unit() -> bool:
    if soft_remaining() <= SOFT_BUFFER_MIN * 60.0:
        write_once_json(
            {
                "status": "paused_safely",
                "remaining_seconds": soft_remaining(),
                "soft_wall_hours": SOFT_WALL_HOURS,
                "buffer_minutes": SOFT_BUFFER_MIN,
            },
            OUT / "audit" / f"PAUSED_{int(time.time())}.json",
        )
        return False
    return True

# ==================================================================================================
# 2. BASIC STATISTICS
# ==================================================================================================

def finite(a: Sequence[float]) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    return x[np.isfinite(x)]

def bootstrap_median_ci(x: Sequence[float], seed: int, B: int = 3000) -> Tuple[float, float]:
    a = finite(x)
    if len(a) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED + int(seed))
    vals = np.empty(B, dtype=float)
    for b in range(B):
        vals[b] = np.median(rng.choice(a, size=len(a), replace=True))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def hodges_lehmann_onesample(x: Sequence[float]) -> float:
    a = finite(x)
    if len(a) == 0:
        return np.nan
    vals = [0.5 * (a[i] + a[j]) for i in range(len(a)) for j in range(i, len(a))]
    return float(np.median(vals))

def rank_biserial_onesample(x: Sequence[float]) -> float:
    a = finite(x)
    a = a[a != 0]
    if len(a) == 0:
        return 0.0
    ranks = pd.Series(np.abs(a)).rank(method="average").to_numpy(float)
    wp = float(ranks[a > 0].sum())
    wm = float(ranks[a < 0].sum())
    den = wp + wm
    return float((wp - wm) / den) if den > 0 else 0.0

def wilcoxon_onesample_record(
    vals: Sequence[float],
    alternative: str,
    metric: str,
    group_label: str,
    seed: int,
) -> Dict[str, Any]:
    a = finite(vals)
    if len(a) == 0:
        return {
            "metric": metric, "group": group_label, "alternative": alternative, "n_models": 0,
            "median": np.nan, "hodges_lehmann": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "positive_fraction": np.nan, "rank_biserial": np.nan, "wilcoxon_stat": np.nan, "p_raw": np.nan,
        }

    if np.allclose(a, 0):
        stat, p = 0.0, 1.0
    else:
        try:
            stat, p = wilcoxon(a, zero_method="wilcox", alternative=alternative, mode="auto")
        except Exception:
            stat, p = np.nan, np.nan

    lo, hi = bootstrap_median_ci(a, seed=seed)
    return {
        "metric": metric,
        "group": group_label,
        "alternative": alternative,
        "n_models": int(len(a)),
        "median": float(np.median(a)),
        "hodges_lehmann": hodges_lehmann_onesample(a),
        "ci_low": lo,
        "ci_high": hi,
        "positive_fraction": float(np.mean(a > 0)),
        "rank_biserial": rank_biserial_onesample(a),
        "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
        "p_raw": float(p) if np.isfinite(p) else np.nan,
    }

def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ids = np.flatnonzero(np.isfinite(p))
    if not len(ids):
        return out
    order = ids[np.argsort(p[ids])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        q = min(1.0, (m-rank) * p[idx])
        running = max(running, q)
        out[idx] = running
    return out

def add_holm_by(df: pd.DataFrame, family_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["p_holm"] = np.nan
    if out.empty:
        return out
    for _, g in out.groupby(family_cols, dropna=False, sort=False):
        out.loc[g.index, "p_holm"] = holm_adjust(pd.to_numeric(g["p_raw"], errors="coerce"))
    return out

# ==================================================================================================
# 3. SOURCE LOADING AND COMPLETENESS AUDIT
# ==================================================================================================

def require(path: Path, what: str):
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required {what} is missing: {path}")

def class_label(dataset: str, stratum: int) -> str:
    if dataset == "fmnist":
        return f"class {int(stratum)} {FMNIST_CLASS_NAMES[int(stratum)].replace('_','/')}"
    return f"horizontal_axis {int(stratum)}"

def load_sources() -> Dict[str, pd.DataFrame]:
    paths = {
        "exp5_fields": EXP5 / "tables" / "field_diagnostics.csv",
        "exp5_measurements": EXP5 / "tables" / "measurements.csv",
        "exp51_symmetry": EXP51 / "tables" / "field_symmetry_measurements.csv",
        "exp51_behavior": EXP51 / "tables" / "behavior_measurements.csv",
        "population": EXP51 / "tables" / "experiment5_1_condition_population.csv",
        "manifest": EXP51 / "tables" / "run_manifest.csv",
        "v4_index": V4 / "tables" / "figure_index.csv",
        "v4_complete": V4 / "EXPERIMENT5_1_V4_VISUAL_ATLAS_COMPLETE.json",
    }
    for k,p in paths.items():
        require(p, k)

    tabs = {
        "field": pd.read_csv(paths["exp5_fields"], low_memory=False),
        "meas5": pd.read_csv(paths["exp5_measurements"], low_memory=False),
        "sym": pd.read_csv(paths["exp51_symmetry"], low_memory=False),
        "beh": pd.read_csv(paths["exp51_behavior"], low_memory=False),
        "pop": pd.read_csv(paths["population"], low_memory=False),
        "manifest": pd.read_csv(paths["manifest"], low_memory=False),
        "v4": pd.read_csv(paths["v4_index"], low_memory=False),
    }
    completion = json.loads(paths["v4_complete"].read_text(encoding="utf-8"))

    # Persist the exact source schemas. This makes any future table-layout drift visible before analysis.
    write_once_json(
        {name: [str(c) for c in df.columns] for name, df in tabs.items()},
        OUT / "audit" / "source_table_schemas.json",
    )

    manifest = tabs["manifest"]
    manifest = manifest[manifest["family"].astype(str) == "cnn"].copy()
    if len(manifest) != 162:
        raise RuntimeError(f"Expected 162 CNNs, found {len(manifest)} in {paths['manifest']}.")

    # Expected complete campaign sizes.
    checks = [
        ("Experiment 5 field diagnostics", len(tabs["field"]), 3240),
        ("Experiment 5 measurements", len(tabs["meas5"]), 6480),
        ("Experiment 5.1 symmetry measurements", len(tabs["sym"]), 32400),
        ("Experiment 5.1 behavior measurements", len(tabs["beh"]), 32400),
        ("Experiment 5.1 condition population", len(tabs["pop"]), 20),
        ("V4 figure index", len(tabs["v4"]), 3240),
    ]

    audit_rows = []
    for name, observed, expected in checks:
        audit_rows.append({
            "check": name,
            "observed": int(observed),
            "expected": int(expected),
            "passed": bool(observed == expected),
        })
        if observed != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {observed}.")

    if str(completion.get("status","")).lower() != "complete":
        raise RuntimeError("V4 completion marker is not complete.")

    # Validate source keys.
    if tabs["field"].duplicated(["run_id","environment","condition_index"]).any():
        raise RuntimeError("Duplicate Experiment 5 field-diagnostic condition keys.")
    if tabs["meas5"].duplicated(["run_id","environment","condition_index","intervention_type"]).any():
        raise RuntimeError("Duplicate Experiment 5 measurement keys.")
    if tabs["sym"].duplicated(["run_id","environment","condition_index","transform_id"]).any():
        raise RuntimeError("Duplicate Experiment 5.1 symmetry keys.")

    write_once_csv(pd.DataFrame(audit_rows), OUT / "audit" / "source_completeness.csv")
    write_once_json(completion, OUT / "audit" / "source_v4_completion_marker.json")
    return tabs

def attach_condition_labels(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the canonical Experiment-5.1 condition metadata without creating pandas _x/_y suffixes.

    Some source tables, notably field_symmetry_measurements.csv and behavior_measurements.csv, already
    contain `stratum` and/or `test_index`.  V1 merged the population table using the same column names,
    which caused pandas to rename them to stratum_x/stratum_y and test_index_x/test_index_y.  Downstream
    groupby code then correctly failed because the canonical `stratum` column no longer existed.

    V2 treats experiment5_1_condition_population.csv as the canonical cohort map, renames its metadata
    before merging, verifies any metadata already present in the source table, and finally publishes one
    unambiguous set of columns named `stratum`, `test_index`, and `condition_label`.
    """
    p = pop.copy()
    if "exp51_condition_index" in p.columns:
        p = p.rename(columns={"exp51_condition_index":"condition_index"})

    required = {"dataset","condition_index","stratum","test_index"}
    missing = required.difference(p.columns)
    if missing:
        raise RuntimeError(f"Condition population is missing required columns: {sorted(missing)}")

    p["dataset"] = p["dataset"].astype(str)
    p["condition_index"] = pd.to_numeric(p["condition_index"], errors="raise").astype(int)
    p["stratum"] = pd.to_numeric(p["stratum"], errors="raise").astype(int)
    p["test_index"] = pd.to_numeric(p["test_index"], errors="raise").astype(int)
    p["condition_label"] = [
        class_label(str(ds), int(st)) for ds,st in zip(p["dataset"], p["stratum"])
    ]

    if p.duplicated(["dataset","condition_index"]).any():
        dup = p.loc[p.duplicated(["dataset","condition_index"], keep=False),
                    ["dataset","condition_index","stratum","test_index"]]
        raise RuntimeError(
            "Condition population has duplicate dataset/condition_index keys:\n"
            + dup.to_string(index=False)
        )

    canonical = p[
        ["dataset","condition_index","stratum","test_index","condition_label"]
    ].rename(columns={
        "stratum":"_population_stratum",
        "test_index":"_population_test_index",
        "condition_label":"_population_condition_label",
    })

    q = df.copy()
    if "dataset" not in q.columns or "condition_index" not in q.columns:
        raise RuntimeError(
            "Source table must contain dataset and condition_index before attaching condition metadata."
        )

    q["dataset"] = q["dataset"].astype(str)
    q["condition_index"] = pd.to_numeric(q["condition_index"], errors="raise").astype(int)

    q = q.merge(
        canonical,
        on=["dataset","condition_index"],
        how="left",
        validate="many_to_one",
    )

    if q["_population_condition_label"].isna().any():
        bad = q.loc[q["_population_condition_label"].isna(), ["dataset","condition_index"]].drop_duplicates()
        raise RuntimeError(
            "Condition metadata merge produced unmatched rows:\n" + bad.to_string(index=False)
        )

    # Audit any source metadata already present before replacing it with the canonical cohort values.
    if "stratum" in q.columns:
        src = pd.to_numeric(q["stratum"], errors="coerce")
        ref = pd.to_numeric(q["_population_stratum"], errors="coerce")
        mismatch = src.notna() & ref.notna() & (src != ref)
        if mismatch.any():
            bad = q.loc[mismatch, ["dataset","condition_index","stratum","_population_stratum"]].head(20)
            raise RuntimeError(
                "Source-table stratum disagrees with canonical condition population:\n"
                + bad.to_string(index=False)
            )

    if "test_index" in q.columns:
        src = pd.to_numeric(q["test_index"], errors="coerce")
        ref = pd.to_numeric(q["_population_test_index"], errors="coerce")
        mismatch = src.notna() & ref.notna() & (src != ref)
        if mismatch.any():
            bad = q.loc[mismatch, ["dataset","condition_index","test_index","_population_test_index"]].head(20)
            raise RuntimeError(
                "Source-table test_index disagrees with canonical condition population:\n"
                + bad.to_string(index=False)
            )

    q["stratum"] = pd.to_numeric(q["_population_stratum"], errors="raise").astype(int)
    q["test_index"] = pd.to_numeric(q["_population_test_index"], errors="raise").astype(int)
    q["condition_label"] = q["_population_condition_label"].astype(str)

    q = q.drop(
        columns=[
            "_population_stratum",
            "_population_test_index",
            "_population_condition_label",
        ]
    )

    # Hard schema sentinel: suffixes here would mean a future source-schema change escaped the audit.
    forbidden_suffix_cols = [
        c for c in q.columns
        if c in {"stratum_x","stratum_y","test_index_x","test_index_y",
                 "condition_label_x","condition_label_y"}
    ]
    if forbidden_suffix_cols:
        raise RuntimeError(
            f"Unexpected merge-suffix columns after canonical metadata attachment: {forbidden_suffix_cols}"
        )

    return q

# ==================================================================================================
# 4. EXPERIMENT 5 CONDITION-LEVEL TABLE
# ==================================================================================================

def build_exp5_condition_table(tabs: Dict[str,pd.DataFrame]) -> pd.DataFrame:
    field = attach_condition_labels(tabs["field"].copy(), tabs["pop"])

    m = tabs["meas5"].copy()
    idcols = [
        "run_id","dataset","arch_id","optimizer","activation","seed_index","environment","condition_index"
    ]
    # Pivot all numeric measurement columns, keeping occurrence and parameter side by side.
    numeric_candidates = []
    for c in m.columns:
        if c in idcols or c == "intervention_type":
            continue
        x = pd.to_numeric(m[c], errors="coerce")
        if x.notna().any():
            m[c] = x
            numeric_candidates.append(c)

    pieces = []
    for metric in numeric_candidates:
        p = m.pivot_table(
            index=idcols,
            columns="intervention_type",
            values=metric,
            aggfunc="first"
        ).reset_index()
        if "occurrence" in p.columns:
            p = p.rename(columns={"occurrence":f"{metric}_occurrence"})
        if "parameter" in p.columns:
            p = p.rename(columns={"parameter":f"{metric}_parameter"})
        keep = idcols + [c for c in p.columns if c not in idcols]
        pieces.append(p[keep])

    if not pieces:
        raise RuntimeError("No numeric Experiment 5 measurement columns could be pivoted.")

    wide = pieces[0]
    for p in pieces[1:]:
        wide = wide.merge(p, on=idcols, how="outer", validate="one_to_one")

    q = field.merge(wide, on=idcols, how="left", validate="one_to_one")

    def diff_if(a,b,new):
        if a in q.columns and b in q.columns:
            q[new] = pd.to_numeric(q[a], errors="coerce") - pd.to_numeric(q[b], errors="coerce")

    diff_if("manifestation_abs_occurrence","manifestation_abs_parameter","manifestation_adv_occ_minus_param")
    diff_if("path_hellinger_occurrence","path_hellinger_parameter","path_hellinger_adv_occ_minus_param")
    diff_if("selected_flow_mass_total_occurrence","selected_flow_mass_total_parameter","selected_flow_mass_total_adv_occ_minus_param")

    # Mechanism and spatial-density diagnostics from Field B.
    if {
        "fieldB_occurrence_flow_mass","fieldB_parameter_flow_mass",
        "fieldB_occurrence_input_pixel_support","fieldB_parameter_input_pixel_support"
    }.issubset(q.columns):
        q["fieldB_occ_flow_per_supported_pixel"] = (
            pd.to_numeric(q["fieldB_occurrence_flow_mass"],errors="coerce") /
            pd.to_numeric(q["fieldB_occurrence_input_pixel_support"],errors="coerce").replace(0,np.nan)
        )
        q["fieldB_param_flow_per_supported_pixel"] = (
            pd.to_numeric(q["fieldB_parameter_flow_mass"],errors="coerce") /
            pd.to_numeric(q["fieldB_parameter_input_pixel_support"],errors="coerce").replace(0,np.nan)
        )
        q["fieldB_flow_density_ratio_occ_over_param"] = (
            q["fieldB_occ_flow_per_supported_pixel"] /
            q["fieldB_param_flow_per_supported_pixel"].replace(0,np.nan)
        )
        q["fieldB_support_ratio_occ_over_param"] = (
            pd.to_numeric(q["fieldB_occurrence_input_pixel_support"],errors="coerce") /
            pd.to_numeric(q["fieldB_parameter_input_pixel_support"],errors="coerce").replace(0,np.nan)
        )

    if {
        "fieldA_mass_on_occurrence_footprint","fieldA_mass_on_parameter_footprint",
        "fieldB_occurrence_input_pixel_support","fieldB_parameter_input_pixel_support"
    }.issubset(q.columns):
        q["fieldA_mass_per_occ_footprint_pixel"] = (
            pd.to_numeric(q["fieldA_mass_on_occurrence_footprint"],errors="coerce") /
            pd.to_numeric(q["fieldB_occurrence_input_pixel_support"],errors="coerce").replace(0,np.nan)
        )
        q["fieldA_mass_per_param_footprint_pixel"] = (
            pd.to_numeric(q["fieldA_mass_on_parameter_footprint"],errors="coerce") /
            pd.to_numeric(q["fieldB_parameter_input_pixel_support"],errors="coerce").replace(0,np.nan)
        )
        q["fieldA_footprint_density_ratio_occ_over_param"] = (
            q["fieldA_mass_per_occ_footprint_pixel"] /
            q["fieldA_mass_per_param_footprint_pixel"].replace(0,np.nan)
        )

    return q

def summarize_exp5_by_condition(q: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fieldA_max",
        "fieldA_entropy",
        "fieldA_top_fraction_mass",
        "fieldB_occurrence_flow_mass",
        "fieldB_parameter_flow_mass",
        "fieldB_flow_mass_advantage_occ_minus_param",
        "fieldB_occurrence_input_pixel_support",
        "fieldB_parameter_input_pixel_support",
        "fieldB_support_ratio_occ_over_param",
        "fieldB_flow_density_ratio_occ_over_param",
        "fieldA_mass_on_occurrence_footprint",
        "fieldA_mass_on_parameter_footprint",
        "fieldA_footprint_density_ratio_occ_over_param",
        "fieldA_post_occurrence_l1",
        "fieldA_post_parameter_l1",
        "fieldA_post_l1_occ_minus_param",
        "manifestation_abs_occurrence",
        "manifestation_abs_parameter",
        "manifestation_adv_occ_minus_param",
        "path_hellinger_occurrence",
        "path_hellinger_parameter",
        "path_hellinger_adv_occ_minus_param",
    ]
    metrics = [m for m in metrics if m in q.columns]

    group = ["dataset","environment","condition_index","stratum","condition_label"]
    rows = []
    for key,g in q.groupby(group,dropna=False,sort=True):
        rec = dict(zip(group,key))
        rec["n_models"] = int(g["run_id"].nunique())
        for m in metrics:
            a = finite(pd.to_numeric(g[m],errors="coerce"))
            rec[f"{m}__median"] = float(np.median(a)) if len(a) else np.nan
            rec[f"{m}__q10"] = float(np.quantile(a,.10)) if len(a) else np.nan
            rec[f"{m}__q90"] = float(np.quantile(a,.90)) if len(a) else np.nan
            rec[f"{m}__positive_fraction"] = float(np.mean(a>0)) if len(a) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def exp5_condition_inference(q: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("fieldB_flow_mass_advantage_occ_minus_param","greater"),
        ("fieldA_post_l1_occ_minus_param","two-sided"),
        ("manifestation_adv_occ_minus_param","two-sided"),
        ("path_hellinger_adv_occ_minus_param","two-sided"),
    ]
    tests = [(m,a) for m,a in tests if m in q.columns]

    rows = []
    seed = 0
    for (ds,env,ci,st,label),g in q.groupby(
        ["dataset","environment","condition_index","stratum","condition_label"],
        dropna=False,sort=True
    ):
        for metric,alt in tests:
            rec = wilcoxon_onesample_record(
                pd.to_numeric(g[metric],errors="coerce"),
                alt,
                metric,
                f"{ds}|{env}|{label}",
                seed
            )
            seed += 1
            rec.update({
                "dataset":ds,"environment":env,"condition_index":ci,
                "stratum":st,"condition_label":label,
            })
            rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out):
        out = add_holm_by(out, ["metric","dataset","environment"])
    return out

# ==================================================================================================
# 5. EXPERIMENT 5.1 COMPLETE SYMMETRY TABLES
# ==================================================================================================

def build_symmetry_table(tabs: Dict[str,pd.DataFrame]) -> pd.DataFrame:
    sym = attach_condition_labels(tabs["sym"].copy(), tabs["pop"])
    beh = attach_condition_labels(tabs["beh"].copy(), tabs["pop"])

    join_keys = [
        "run_id","dataset","arch_id","optimizer","activation","seed_index",
        "environment","condition_index","transform_id"
    ]
    bcols = [c for c in [
        "prediction_unchanged","baseline_correct","transformed_correct",
        "target_probability_change","target_margin_change",
        "prediction_abs_change_normalized","prediction_abs_change_raw","task_loss_change",
        "input_contrast_energy_ratio","input_contrast_energy_retained",
    ] if c in beh.columns]

    if bcols:
        sym = sym.merge(
            beh[join_keys+bcols],
            on=join_keys,
            how="left",
            validate="one_to_one",
            suffixes=("","_behavior"),
        )
    return sym

def symmetry_transform_summary(sym: pd.DataFrame) -> pd.DataFrame:
    p = sym[sym["transform_kind"].astype(str)=="primary"].copy()
    group = [
        "dataset","environment","condition_index","stratum","condition_label",
        "transform_family","transform_id","transform_magnitude","transform_magnitude_unit"
    ]
    metrics = [
        "hellinger_equivariant","hellinger_invariant","equivariance_advantage",
        "wrong_direction_advantage","field_com_equivariance_error_pixels",
        "prediction_abs_change_raw","prediction_abs_change_normalized",
        "target_probability_change","target_margin_change",
    ]
    metrics = [m for m in metrics if m in p.columns]

    rows = []
    for key,g in p.groupby(group,dropna=False,sort=True):
        rec = dict(zip(group,key))
        rec["n_models"] = int(g["run_id"].nunique())
        for m in metrics:
            a = finite(pd.to_numeric(g[m],errors="coerce"))
            rec[f"{m}__median"] = float(np.median(a)) if len(a) else np.nan
            rec[f"{m}__q10"] = float(np.quantile(a,.10)) if len(a) else np.nan
            rec[f"{m}__q90"] = float(np.quantile(a,.90)) if len(a) else np.nan
            rec[f"{m}__positive_fraction"] = float(np.mean(a>0)) if len(a) else np.nan

        if "prediction_unchanged" in g.columns:
            s = g["prediction_unchanged"]
            if pd.api.types.is_bool_dtype(s):
                vals = s.dropna().astype(bool).to_numpy()
            else:
                vals = s.astype(str).str.strip().str.lower().map(
                    {"true":True,"1":True,"1.0":True,"false":False,"0":False,"0.0":False}
                ).dropna().to_numpy(bool)
            rec["prediction_unchanged_fraction"] = float(np.mean(vals)) if len(vals) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def build_model_condition_family(sym: pd.DataFrame) -> pd.DataFrame:
    p = sym[sym["transform_kind"].astype(str)=="primary"].copy()
    keys = [
        "run_id","dataset","arch_id","optimizer","activation","seed_index",
        "environment","condition_index","stratum","condition_label","transform_family"
    ]
    metrics = [
        "hellinger_equivariant","hellinger_invariant","equivariance_advantage",
        "wrong_direction_advantage","field_com_equivariance_error_pixels",
    ]
    metrics = [m for m in metrics if m in p.columns]
    return p.groupby(keys,dropna=False)[metrics].median().reset_index()

def symmetry_family_inference(model_fam: pd.DataFrame) -> pd.DataFrame:
    rows = []
    seed = 10000
    for key,g in model_fam.groupby(
        ["dataset","environment","condition_index","stratum","condition_label","transform_family"],
        dropna=False,sort=True
    ):
        ds,env,ci,st,label,fam = key
        rec = wilcoxon_onesample_record(
            pd.to_numeric(g["equivariance_advantage"],errors="coerce"),
            "two-sided",
            "equivariance_advantage",
            f"{ds}|{env}|{label}|{fam}",
            seed
        )
        seed += 1
        rec.update({
            "dataset":ds,"environment":env,"condition_index":ci,"stratum":st,
            "condition_label":label,"transform_family":fam,
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    if len(out):
        out = add_holm_by(out, ["dataset","environment","transform_family"])
    return out

def symmetry_environment_paired(model_fam: pd.DataFrame) -> pd.DataFrame:
    idcols = [
        "run_id","dataset","arch_id","optimizer","activation","seed_index",
        "condition_index","stratum","condition_label","transform_family"
    ]
    piv = model_fam.pivot_table(
        index=idcols,
        columns="environment",
        values="equivariance_advantage",
        aggfunc="first",
    ).reset_index()

    if not {"clean","shifted"}.issubset(piv.columns):
        raise RuntimeError("Symmetry environment pairing requires clean and shifted columns.")

    piv["shifted_minus_clean"] = piv["shifted"] - piv["clean"]

    rows = []
    seed = 20000
    for key,g in piv.groupby(
        ["dataset","condition_index","stratum","condition_label","transform_family"],
        dropna=False,sort=True
    ):
        ds,ci,st,label,fam = key
        rec = wilcoxon_onesample_record(
            pd.to_numeric(g["shifted_minus_clean"],errors="coerce"),
            "two-sided",
            "equivariance_advantage_shifted_minus_clean",
            f"{ds}|{label}|{fam}",
            seed
        )
        seed += 1
        rec.update({
            "dataset":ds,"condition_index":ci,"stratum":st,
            "condition_label":label,"transform_family":fam,
        })
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out):
        out = add_holm_by(out, ["dataset","transform_family"])
    return out

# ==================================================================================================
# 6. FIELD A / FIELD B ASSOCIATIONS AT CONDITION LEVEL
# ==================================================================================================

def condition_associations(q: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive model-level associations within each dataset/environment/condition.
    No causal interpretation is assigned.
    """
    pairs = [
        ("fieldB_flow_mass_advantage_occ_minus_param","manifestation_adv_occ_minus_param"),
        ("fieldB_flow_mass_advantage_occ_minus_param","path_hellinger_adv_occ_minus_param"),
        ("fieldB_flow_mass_advantage_occ_minus_param","fieldA_post_l1_occ_minus_param"),
    ]
    pairs = [(a,b) for a,b in pairs if a in q.columns and b in q.columns]

    rows = []
    for key,g in q.groupby(
        ["dataset","environment","condition_index","stratum","condition_label"],
        dropna=False,sort=True
    ):
        ds,env,ci,st,label = key
        for a,b in pairs:
            x = pd.to_numeric(g[a],errors="coerce").to_numpy(float)
            y = pd.to_numeric(g[b],errors="coerce").to_numpy(float)
            mask = np.isfinite(x)&np.isfinite(y)
            if mask.sum() >= 4:
                rho,p = spearmanr(x[mask],y[mask])
            else:
                rho,p = np.nan,np.nan
            rows.append({
                "dataset":ds,"environment":env,"condition_index":ci,"stratum":st,
                "condition_label":label,"x_metric":a,"y_metric":b,
                "n_models":int(mask.sum()),"spearman_rho":rho,"p_raw":p,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = add_holm_by(out, ["dataset","environment","x_metric","y_metric"])
    return out

# ==================================================================================================
# 7. V4 FIGURE INDEX AND PATH RESOLUTION
# ==================================================================================================

def parse_json_list(value: Any) -> List[str]:
    if isinstance(value,list):
        return [str(x) for x in value]
    s = str(value)
    try:
        v = json.loads(s)
        if isinstance(v,list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []

def remap_v4_path(raw: str) -> Path:
    p = Path(str(raw))
    if p.exists():
        return p

    # Robust remap if absolute HPC root moved while the experiment subtree name stayed the same.
    parts = list(p.parts)
    token = "experiment5_1_all_conditions_visual_atlas_v4"
    if token in parts:
        i = parts.index(token)
        candidate = V4.joinpath(*parts[i+1:])
        if candidate.exists():
            return candidate

    # Final fallback by basename. This is slower but only used for selected visual cases.
    hits = list((V4/"figures").rglob(p.name))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous V4 figure basename {p.name}.")
    raise FileNotFoundError(f"V4 source figure not found: {raw}")

def prepare_v4_index(v4: pd.DataFrame) -> pd.DataFrame:
    q = v4.copy()
    q["condition_index"] = pd.to_numeric(q["condition_index"],errors="raise").astype(int)
    q["stratum"] = pd.to_numeric(q["stratum"],errors="raise").astype(int)
    if q.duplicated(["run_id","environment","condition_index"]).any():
        raise RuntimeError("Duplicate V4 figure-index condition rows.")
    return q

def get_v4_condition_row(v4: pd.DataFrame, run_id: str, env: str, ci: int) -> pd.Series:
    q = v4[
        (v4["run_id"].astype(str)==str(run_id))
        & (v4["environment"].astype(str)==str(env))
        & (v4["condition_index"].astype(int)==int(ci))
    ]
    if len(q) != 1:
        raise RuntimeError(f"Expected one V4 figure-index row for {run_id}/{env}/C{ci:02d}, got {len(q)}.")
    return q.iloc[0]

def get_fieldab_path(v4row: pd.Series) -> Path:
    return remap_v4_path(str(v4row["fieldA_fieldB_figure"]))

def get_symmetry_path(v4row: pd.Series, transform_id: str) -> Path:
    paths = parse_json_list(v4row["symmetry_figures"])
    candidates = [x for x in paths if f"__{transform_id}__symmetry_v4.png" in Path(x).name]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one V4 symmetry path for {v4row['run_id']}/{v4row['environment']}/"
            f"C{int(v4row['condition_index']):02d}/{transform_id}; got {len(candidates)}."
        )
    return remap_v4_path(candidates[0])

# ==================================================================================================
# 8. SYSTEMATIC CASE SELECTION
# ==================================================================================================

def nearest_quantile_rows(
    g: pd.DataFrame,
    metric: str,
    quantiles: Sequence[Tuple[str,float]] = (("low",.10),("median",.50),("high",.90)),
) -> List[Tuple[str,pd.Series]]:
    z = g.copy()
    z[metric] = pd.to_numeric(z[metric],errors="coerce")
    z = z[np.isfinite(z[metric])].sort_values(metric).reset_index(drop=True)
    if z.empty:
        return []

    out = []
    used = set()
    vals = z[metric].to_numpy(float)
    for label,q in quantiles:
        target = float(np.quantile(vals,q))
        order = np.argsort(np.abs(vals-target),kind="mergesort")
        chosen = None
        for idx in order:
            rid = str(z.iloc[int(idx)]["run_id"])
            if rid not in used:
                chosen = z.iloc[int(idx)]
                used.add(rid)
                break
        if chosen is None:
            chosen = z.iloc[int(order[0])]
        out.append((label,chosen))
    return out

def build_visual_selection(
    cond5: pd.DataFrame,
    sym: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Field A / Field B atlas.
    score_field = (
        "fieldB_flow_mass_advantage_occ_minus_param"
        if "fieldB_flow_mass_advantage_occ_minus_param" in cond5.columns
        else "fieldA_post_l1_occ_minus_param"
    )

    clean5 = cond5[cond5["environment"].astype(str)=="clean"].copy()
    for key,g in clean5.groupby(
        ["dataset","condition_index","stratum","condition_label"],
        dropna=False,sort=True
    ):
        ds,ci,st,label = key
        for spectrum,r in nearest_quantile_rows(g,score_field):
            rows.append({
                "sheet_type":"fieldAB",
                "dataset":ds,"condition_index":int(ci),"stratum":int(st),"condition_label":label,
                "transform_family":"none","transform_id":"none",
                "spectrum":spectrum,"selection_metric":score_field,
                "selection_value":float(r[score_field]),
                "run_id":str(r["run_id"]),
            })

    # Symmetry atlas uses representative transforms already rendered by V4.
    representative = {
        "fmnist":[("translation","right_2"),("rotation","rot_ccw_20"),("scale","scale_up_110")],
        "mpi3d":[("translation","down_2")],
    }
    clean_sym = sym[
        (sym["environment"].astype(str)=="clean")
        & (sym["transform_kind"].astype(str)=="primary")
    ].copy()

    for ds, specs in representative.items():
        for family,tid in specs:
            d = clean_sym[
                (clean_sym["dataset"].astype(str)==ds)
                & (clean_sym["transform_id"].astype(str)==tid)
            ]
            for key,g in d.groupby(
                ["condition_index","stratum","condition_label"],
                dropna=False,sort=True
            ):
                ci,st,label = key
                for spectrum,r in nearest_quantile_rows(g,"equivariance_advantage"):
                    rows.append({
                        "sheet_type":"symmetry",
                        "dataset":ds,"condition_index":int(ci),"stratum":int(st),"condition_label":label,
                        "transform_family":family,"transform_id":tid,
                        "spectrum":spectrum,"selection_metric":"equivariance_advantage",
                        "selection_value":float(r["equivariance_advantage"]),
                        "run_id":str(r["run_id"]),
                    })

    return pd.DataFrame(rows)

# ==================================================================================================
# 9. CONTACT SHEETS
# ==================================================================================================

def _read_image(path: Path):
    return plt.imread(path)

def safe_name(s: str) -> str:
    x = re.sub(r"[^A-Za-z0-9._-]+","_",str(s)).strip("_")
    return x[:160]

def save_contact_sheet(
    panels: List[Dict[str,Any]],
    title: str,
    path: Path,
) -> bool:
    if path.exists():
        return False
    if len(panels) != 6:
        raise ValueError(f"Contact sheet expects exactly 6 panels, got {len(panels)}.")

    if not can_start_visual_unit():
        raise TimeoutError("soft_wall")

    fig, axs = plt.subplots(3,2,figsize=(16,18),constrained_layout=True)
    for ax,panel in zip(axs.ravel(),panels):
        img = _read_image(panel["path"])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(panel["title"],fontsize=8.2,pad=5)

    fig.suptitle(title,fontsize=13)
    tmp = path.parent / f".{path.stem}.tmp.{os.getpid()}.{time.time_ns()}.jpg"
    fig.savefig(
        tmp,
        format="jpg",
        dpi=CONTACT_DPI,
        bbox_inches="tight",
        pad_inches=0.08,
        pil_kwargs={"quality":JPEG_QUALITY,"optimize":True},
    )
    plt.close(fig)

    payload = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return publish_bytes_once(payload,path)

def metric_text_exp5(row: pd.Series) -> str:
    parts = []
    for col,label in [
        ("fieldB_flow_mass_advantage_occ_minus_param","B flow adv"),
        ("manifestation_adv_occ_minus_param","Dman adv"),
        ("path_hellinger_adv_occ_minus_param","DH adv"),
        ("fieldA_post_l1_occ_minus_param","A-post adv"),
    ]:
        if col in row.index:
            try:
                v = float(row[col])
                if np.isfinite(v):
                    parts.append(f"{label} {v:+.3g}")
            except Exception:
                pass
    return " | ".join(parts)

def metric_text_sym(row: pd.Series) -> str:
    try:
        return (
            f"H_eq {float(row['hellinger_equivariant']):.3f} | "
            f"H_inv {float(row['hellinger_invariant']):.3f} | "
            f"adv {float(row['equivariance_advantage']):+.3f}"
        )
    except Exception:
        return ""

def build_core_contact_sheets(
    selections: pd.DataFrame,
    cond5: pd.DataFrame,
    sym: pd.DataFrame,
    v4: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    grouping = [
        "sheet_type","dataset","condition_index","stratum","condition_label",
        "transform_family","transform_id"
    ]

    for key,g in selections.groupby(grouping,dropna=False,sort=True):
        stype,ds,ci,st,label,fam,tid = key
        g = g.set_index("spectrum")
        if not {"low","median","high"}.issubset(g.index):
            raise RuntimeError(f"Incomplete low/median/high visual selection for {key}.")

        panels = []
        selected_runs = []

        for spectrum in ["low","median","high"]:
            sel = g.loc[spectrum]
            run_id = str(sel["run_id"])
            selected_runs.append(run_id)

            for env in ["clean","shifted"]:
                vrow = get_v4_condition_row(v4,run_id,env,int(ci))

                if stype == "fieldAB":
                    p = get_fieldab_path(vrow)
                    m = cond5[
                        (cond5["run_id"].astype(str)==run_id)
                        & (cond5["environment"].astype(str)==env)
                        & (cond5["condition_index"].astype(int)==int(ci))
                    ]
                    if len(m)!=1:
                        raise RuntimeError(f"Missing Experiment-5 metric row for {run_id}/{env}/C{ci}.")
                    mrow = m.iloc[0]
                    subtitle = metric_text_exp5(mrow)
                else:
                    p = get_symmetry_path(vrow,str(tid))
                    m = sym[
                        (sym["run_id"].astype(str)==run_id)
                        & (sym["environment"].astype(str)==env)
                        & (sym["condition_index"].astype(int)==int(ci))
                        & (sym["transform_id"].astype(str)==str(tid))
                    ]
                    if len(m)!=1:
                        raise RuntimeError(f"Missing symmetry metric row for {run_id}/{env}/C{ci}/{tid}.")
                    mrow = m.iloc[0]
                    subtitle = metric_text_sym(mrow)

                arch = str(mrow.get("arch_id",""))
                act = str(mrow.get("activation",""))
                opt = str(mrow.get("optimizer",""))
                title = f"{spectrum.upper()} | {env} | {arch}/{act}/{opt}\n{subtitle}"
                panels.append({"path":p,"title":title})

        dirname = OUT/"contact_sheets"/str(ds)
        dirname.mkdir(parents=True,exist_ok=True)

        if stype=="fieldAB":
            filename = (
                f"{ds}__C{int(ci):02d}__{safe_name(label)}__FieldA_FieldB__low_med_high_clean_shifted.jpg"
            )
            title = f"{ds.upper()} | {label} | Field A / Field B | low, median, high {g.iloc[0]['selection_metric']}"
        else:
            filename = (
                f"{ds}__C{int(ci):02d}__{safe_name(label)}__{fam}__{tid}"
                "__low_med_high_clean_shifted.jpg"
            )
            title = f"{ds.upper()} | {label} | {fam} | {tid} | low, median, high equivariance advantage"

        outpath = dirname/filename

        try:
            made = save_contact_sheet(panels,title,outpath)
        except TimeoutError:
            raise

        records.append({
            "sheet_type":stype,"dataset":ds,"condition_index":int(ci),"stratum":int(st),
            "condition_label":label,"transform_family":fam,"transform_id":tid,
            "selected_run_low":selected_runs[0],
            "selected_run_median":selected_runs[1],
            "selected_run_high":selected_runs[2],
            "contact_sheet":str(outpath),
            "created_now":bool(made),
        })

        gc.collect()
        plt.close("all")

    return pd.DataFrame(records)

# ==================================================================================================
# 10. GLOBAL DIAGNOSTIC SHEETS
# ==================================================================================================

def six_extreme_rows(df: pd.DataFrame, metric: str, smallest: bool=True) -> pd.DataFrame:
    z = df.copy()
    z[metric] = pd.to_numeric(z[metric],errors="coerce")
    z = z[np.isfinite(z[metric])]
    return z.nsmallest(6,metric) if smallest else z.nlargest(6,metric)

def save_six_panel_global(
    rows: pd.DataFrame,
    v4: pd.DataFrame,
    title: str,
    filename: str,
    kind: str,
    transform_id: Optional[str]=None,
    metric: Optional[str]=None,
) -> Optional[Dict[str,Any]]:
    if len(rows) < 6:
        return None

    panels = []
    for _,r in rows.iloc[:6].iterrows():
        run_id = str(r["run_id"])
        env = str(r["environment"])
        ci = int(r["condition_index"])
        vrow = get_v4_condition_row(v4,run_id,env,ci)
        p = get_fieldab_path(vrow) if kind=="fieldAB" else get_symmetry_path(vrow,str(transform_id or r["transform_id"]))

        value = ""
        if metric and metric in r.index:
            try:
                value = f"{metric}={float(r[metric]):+.3g}"
            except Exception:
                pass

        panels.append({
            "path":p,
            "title":f"{r['condition_label']} | {env}\n{run_id}\n{value}",
        })

    path = OUT/"contact_sheets"/"global"/filename
    path.parent.mkdir(parents=True,exist_ok=True)
    made = save_contact_sheet(panels,title,path)
    return {"contact_sheet":str(path),"created_now":made,"title":title}

def build_global_sheets(cond5: pd.DataFrame, sym: pd.DataFrame, v4: pd.DataFrame) -> pd.DataFrame:
    records = []

    if "manifestation_adv_occ_minus_param" in cond5.columns:
        rev = cond5[
            (pd.to_numeric(cond5["fieldB_flow_mass_advantage_occ_minus_param"],errors="coerce")>0)
            & (pd.to_numeric(cond5["manifestation_adv_occ_minus_param"],errors="coerce")<0)
        ].copy()
        if len(rev)>=6:
            r = six_extreme_rows(rev,"manifestation_adv_occ_minus_param",smallest=True)
            x = save_six_panel_global(
                r,v4,
                "Strong causal reversals despite positive Field-B captured-flow advantage",
                "global__fieldB_positive_flow_but_parameter_functional_wins.jpg",
                "fieldAB",
                metric="manifestation_adv_occ_minus_param",
            )
            if x: records.append(x)

    # Worst symmetry cases by representative family.
    representative = [
        ("fmnist","right_2","translation"),
        ("fmnist","rot_ccw_20","rotation"),
        ("fmnist","scale_up_110","scale"),
        ("mpi3d","down_2","translation"),
    ]
    for ds,tid,fam in representative:
        d = sym[
            (sym["dataset"].astype(str)==ds)
            & (sym["environment"].astype(str)=="clean")
            & (sym["transform_id"].astype(str)==tid)
        ].copy()
        if len(d)>=6:
            r = six_extreme_rows(d,"equivariance_advantage",smallest=True)
            x = save_six_panel_global(
                r,v4,
                f"Worst clean {ds.upper()} {fam} cases | {tid}",
                f"global__{ds}__{tid}__worst_equivariance.jpg",
                "symmetry",
                transform_id=tid,
                metric="equivariance_advantage",
            )
            if x: records.append(x)

    return pd.DataFrame(records)

# ==================================================================================================
# 11. BUILD SMALL UPLOAD PACKAGE
# ==================================================================================================

def build_readme() -> str:
    return f"""Experiment 5 + 5.1 V4 post-analysis package
================================================

Version
-------
{VERSION}

Source result root
------------------
{RESULTS_ROOT}

Read-only sources
-----------------
{EXP5}
{EXP51}
{V4}

Post-analysis output
--------------------
{OUT}

Scientific scope
----------------
The complete numerical evidence is analyzed. Visual inspection is reduced only through deterministic
10th / 50th / 90th percentile case selection.

Condition remains c=(x,0). Fashion-MNIST conditions are classes 0..9. MPI3D conditions are the ten
horizontal_axis strata 0,4,...,36.

Core tables
-----------
exp5_condition_level.csv
exp5_class_stratum_summary.csv
exp5_class_stratum_inference.csv
exp5_condition_associations.csv

exp51_symmetry_condition_transform.csv
exp51_symmetry_class_transform_summary.csv
exp51_model_condition_family.csv
exp51_symmetry_family_inference.csv
exp51_symmetry_environment_paired.csv

visual_case_selection.csv
contact_sheet_index.csv

Visual review
-------------
Each core contact sheet contains three selected frozen CNNs and pairs clean versus shifted for the same
network. Selection is based on numerical ranks, not subjective visual choice.

Fashion-MNIST: Field A/B + translation + rotation + scale for every class.
MPI3D: Field A/B + vertical translation for every selected horizontal-position stratum.

The upload package intentionally excludes the original ~2 GB V4 image corpus.
"""

def make_upload_zip() -> Path:
    package = OUT/"upload_package"/"experiment5_5_1_v4_post_analysis_upload.zip"
    if package.exists():
        return package

    tmp = package.with_suffix(".tmp.zip")
    if tmp.exists():
        tmp.unlink()

    # Fixed timestamps make the zip deterministic enough for write-once use.
    fixed_dt = (2026,1,1,0,0,0)

    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for root_name in ["tables","audit","contact_sheets"]:
            base = OUT/root_name
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                arc = str(p.relative_to(OUT))
                info = zipfile.ZipInfo(arc,date_time=fixed_dt)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info,p.read_bytes())

        readme = OUT/"upload_package"/"README.txt"
        if readme.exists():
            info = zipfile.ZipInfo("README.txt",date_time=fixed_dt)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info,readme.read_bytes())

    payload = tmp.read_bytes()
    tmp.unlink()
    publish_bytes_once(payload,package)
    return package

# ==================================================================================================
# 12. SELFTEST
# ==================================================================================================

def selftest() -> None:
    tmp = OUT/"audit"/"_selftest"
    tmp.mkdir(parents=True,exist_ok=True)

    # Synthetic images only test the contact-sheet renderer and write-once publication.
    panels = []
    rng = np.random.default_rng(123)
    for i in range(6):
        arr = rng.random((160,240,3))
        p = tmp/f"img{i}.png"
        plt.imsave(p,arr)
        panels.append({"path":p,"title":f"synthetic {i}"})

    target = tmp/"contact.jpg"
    save_contact_sheet(panels,"V4 post-analysis synthetic contact sheet",target)
    if not target.exists() or target.stat().st_size<=0:
        raise RuntimeError("Synthetic contact-sheet selftest failed.")

    # Schema regression test: source rows already containing stratum/test_index must retain canonical names
    # after population metadata attachment, with no _x/_y suffixes.
    pop = pd.DataFrame([
        {"dataset":"fmnist","exp51_condition_index":0,"stratum":0,"test_index":123},
        {"dataset":"mpi3d","exp51_condition_index":0,"stratum":0,"test_index":456},
    ])
    source = pd.DataFrame([
        {"run_id":"A","dataset":"fmnist","condition_index":0,"stratum":0,"test_index":123,"value":1.0},
        {"run_id":"B","dataset":"mpi3d","condition_index":0,"stratum":0,"test_index":456,"value":2.0},
    ])
    merged = attach_condition_labels(source,pop)
    required = {"stratum","test_index","condition_label"}
    if not required.issubset(merged.columns):
        raise RuntimeError(f"Schema regression selftest failed; missing {sorted(required.difference(merged.columns))}")
    if any(c.endswith("_x") or c.endswith("_y") for c in merged.columns):
        raise RuntimeError("Schema regression selftest failed; pandas merge suffix survived.")

    print("SELFTEST PASS")
    shutil.rmtree(tmp,ignore_errors=True)

# ==================================================================================================
# 13. MAIN
# ==================================================================================================

def main():
    print("="*116)
    print("WORK II — EXPERIMENT 5 + 5.1 V4 POST-ANALYSIS")
    print("="*116)
    print(f"Results root : {RESULTS_ROOT}")
    print(f"Experiment 5 : {EXP5}")
    print(f"Experiment 5.1: {EXP51}")
    print(f"V4 atlas     : {V4}")
    print(f"Output       : {OUT}")
    print("Source experiments are read-only.")

    if SELFTEST_ONLY:
        selftest()
        return

    tabs = load_sources()
    tabs["v4"] = prepare_v4_index(tabs["v4"])

    # ----------------------------------------------------------------------------------------------
    # Full numerical analysis
    # ----------------------------------------------------------------------------------------------
    print("[analysis] Build Experiment 5 condition-level table",flush=True)
    cond5 = build_exp5_condition_table(tabs)
    summary5 = summarize_exp5_by_condition(cond5)
    inf5 = exp5_condition_inference(cond5)
    assoc5 = condition_associations(cond5)

    print("[analysis] Build Experiment 5.1 class/stratum symmetry tables",flush=True)
    sym = build_symmetry_table(tabs)
    sym_summary = symmetry_transform_summary(sym)
    model_fam = build_model_condition_family(sym)
    sym_inf = symmetry_family_inference(model_fam)
    sym_env = symmetry_environment_paired(model_fam)

    write_once_csv(cond5, OUT/"tables"/"exp5_condition_level.csv")
    write_once_csv(summary5, OUT/"tables"/"exp5_class_stratum_summary.csv")
    write_once_csv(inf5, OUT/"tables"/"exp5_class_stratum_inference.csv")
    write_once_csv(assoc5, OUT/"tables"/"exp5_condition_associations.csv")

    write_once_csv(sym, OUT/"tables"/"exp51_symmetry_condition_transform.csv")
    write_once_csv(sym_summary, OUT/"tables"/"exp51_symmetry_class_transform_summary.csv")
    write_once_csv(model_fam, OUT/"tables"/"exp51_model_condition_family.csv")
    write_once_csv(sym_inf, OUT/"tables"/"exp51_symmetry_family_inference.csv")
    write_once_csv(sym_env, OUT/"tables"/"exp51_symmetry_environment_paired.csv")

    # Numeric visual selection.
    selections = build_visual_selection(cond5,sym)
    write_once_csv(selections,OUT/"tables"/"visual_case_selection.csv")

    preflight = {
        "version":VERSION,
        "source_rows":{
            "exp5_condition_level":int(len(cond5)),
            "exp51_transform_rows":int(len(sym)),
            "v4_figure_index_rows":int(len(tabs["v4"])),
            "visual_selection_rows":int(len(selections)),
        },
        "expected_core_contact_sheets":60,
        "source_experiments_modified":False,
        "output":str(OUT),
    }
    write_once_json(preflight,OUT/"audit"/"post_analysis_preflight.json")

    if PREFLIGHT_ONLY:
        print(json.dumps(preflight,indent=2))
        print("PREFLIGHT ONLY COMPLETE")
        return

    # ----------------------------------------------------------------------------------------------
    # Systematic visual reduction
    # ----------------------------------------------------------------------------------------------
    print("[visual] Build systematic class/stratum contact sheets",flush=True)
    try:
        contact_index = build_core_contact_sheets(selections,cond5,sym,tabs["v4"])
        global_index = build_global_sheets(cond5,sym,tabs["v4"])
    except TimeoutError:
        print("Soft wall reached. Existing contact sheets are preserved. Rerun the same script to resume.")
        return

    write_once_csv(contact_index,OUT/"tables"/"contact_sheet_index.csv")
    write_once_csv(global_index,OUT/"tables"/"global_contact_sheet_index.csv")

    # Audit actual contact sheets.
    sheets = list((OUT/"contact_sheets").rglob("*.jpg"))
    sheet_audit = pd.DataFrame([{
        "path":str(p),
        "bytes":int(p.stat().st_size),
        "sha1":file_sha1(p),
    } for p in sheets])
    write_once_csv(sheet_audit,OUT/"audit"/"contact_sheet_inventory.csv")

    # README + compact upload zip.
    write_text_once(build_readme(),OUT/"upload_package"/"README.txt")
    package = make_upload_zip()

    completion = {
        "version":VERSION,
        "status":"complete",
        "exp5_condition_rows":int(len(cond5)),
        "exp51_symmetry_rows":int(len(sym)),
        "core_contact_sheets":int(len(contact_index)),
        "global_contact_sheets":int(len(global_index)),
        "total_contact_sheets_found":int(len(sheets)),
        "upload_package":str(package),
        "upload_package_bytes":int(package.stat().st_size),
        "source_experiments_modified":False,
    }
    write_once_json(completion,OUT/"POST_ANALYSIS_COMPLETE.json")

    print("="*116)
    print("POST-ANALYSIS COMPLETE")
    print(json.dumps(completion,indent=2))
    print("="*116)

if __name__ == "__main__":
    main()
