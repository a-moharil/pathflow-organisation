# What Organizes in Path-Decomposable Neural Network Architectures: code and result tables

This repository contains the experiment scripts and result tables for the ECML PKDD 2027 Journal Track submission *What Organizes in Path-Decomposable Neural Network Architectures*. The complete results archive, `ECMLPKDD_results_archive.pdf`, reports auxiliary experiments in full.

The paper studies trained multilayer perceptrons (MLPs) and convolutional networks (CNNs) through a probability measure on their complete input-to-output computational paths. The investigation has three levels:

- **Structural disposition (S).** The input-free path measure induced by the trained weights, $\mu^S(\pi) \propto \prod_{e\in\pi} w_e^2$. It is unchanged by positive hidden-unit rescalings and permutations.
- **Conditional realisation (C).** The path measure realised when the network is placed under a declared condition $c=(x,x^\circ)$, here with $x^\circ = 0$ after standardisation. It is obtained from exact finite-response operators, and its support lies inside the structural support.
- **Re-realisation under intervention.** The measure obtained after an execution-only attenuation of the realised part, with the parameters left unchanged.

Each measure is projected onto conserved edge and node flows. A declared ruler $\lambda_e=(\mathcal{F}_e+\varepsilon)^{-p}$ with $p=1$ turns the flow into a geometry, which is analysed by weighted box covering (BCANw).

The first two levels are studied over controlled populations that vary one factor at a time:

| Phase | Factor varied | MLP | CNN |
|---|---|---|---|
| S1 | Architecture | `scripts/MLP/structural/S1` | `scripts/CNN/structural/S1` |
| S2 | Statistical environment | `scripts/MLP/structural/S2` | `scripts/CNN/structural/S2` |
| S3 | Optimiser | `scripts/MLP/structural/S3` | `scripts/CNN/structural/S3` |
| C1 | Architecture | `scripts/MLP/conditional/C1` | `scripts/CNN/conditional/C1` |
| C2 | Environment (and task, for MLPs) | `scripts/MLP/conditional/C2` | `scripts/CNN/conditional/C2` |
| C3 | Optimiser | not included here (<50MB), on Zenodo | `scripts/CNN/conditional/C3` |

Two further parts use real data:

- **Case studies** on MPI3D and Fashion-MNIST, in which structural disposition and conditional realisation are measured in trained MLPs and CNNs under clean and shifted sensing (`scripts/Case_Study`).
- **Causal intervention**, in which the realised part of the computation is attenuated during execution only. The intervention experiments comprise 149,040 controlled interventions (`scripts/causal_interventions`).

Architectures and environments in the controlled phases are chosen by AutoML-style sequential design: Sobol cold starts followed by random-forest model-based search (SMAC or RF-SMAC). The search is used to cover the design space and to seek structurally novel regions. Validation and test performance are never acquisition targets. Statistical inference is carried out only on panels frozen before training, and seeds are aggregated within each architecture or environment before population-level tests.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── ECMLPKDD_results_archive.pdf
├── scripts/
│   ├── MLP/
│   │   ├── structural/   S1/ (with logs/)  S2/  S3/
│   │   └── conditional/  C1/  C2/
│   ├── CNN/
│   │   ├── structural/   S1/  S2/  S3/
│   │   └── conditional/  C1/  C2/  C3/
│   ├── Case_Study/
│   └── causal_interventions/
└── tables/
    ├── MLP/   PhaseS1/  PhaseS2/  PhaseS3/  PhaseC1/  PhaseC2/
    ├── CNN/   PhaseS1/  PhaseS2/  PhaseS3/  PhaseC1/  PhaseC2/  PhaseC3/
    └── interventions_and_casestudy/
               Experiment1/  Experiment2/  Experiment3/  Experiment4/  Experiment5/
```

`tables/<family>/Phase<X>/` holds the tables written by the script in `scripts/<family>/<level>/<X>/` for the runs reported in the paper. `tables/interventions_and_casestudy/Experiment<k>/` holds the tables of intervention experiment *k*. Tables larger than 50 MB are not stored in this repository; see [Large tables](#large-tables).

---

## Script index

Every experiment is one self-contained Python file. It can be run as a script (`python -u file.py`), and most can also be run from a single Jupyter cell. Where a folder contains several files, later revisions keep the same scientific design and add robustness for long cluster runs, such as resumption, recovery and checkpoint validation. The **entry point** names the file to use for a new run. Earlier revisions are kept for provenance.

### MLP, structural disposition

| Phase | Entry point | Other files | Design (default profile) |
|---|---|---|---|
| S1 | `pathflow_architecture_universality_experiment_v5_densebo_commonwindow_final.py` | `logs/MLP_S1_run_log_job6302532.out` (run log) | Bias-free ReLU MLPs with 9-32 hidden layers and five morphologies (constant, taper, expansion, bottleneck, hourglass). Initialisation map over 600 stratified Sobol and 400 Bayesian architectures (SMAC, structural-novelty objective). Fixed longitudinal cohort of 48 architectures, trained on a synthetic teacher-student regression task (60,000 / 10,000 / 10,000 samples, 32 inputs), with geometry at 20 checkpoints from epoch 0 to 399. |
| S2 | `pathflow_phase2_environment_universality_halfscale_singlecell.py` | | 12 architectures inherited from S1, 4 sentinels. Environment discovery over 60 controlled environments (36 Sobol and 24 Bayesian experimental design) at 80 epochs. Balanced matrix of 24 environments x 12 architectures x 2 seeds = 576 runs, to 400 epochs. |
| S3 | `pathflow_phase3_optimizer_dynamics_single_cell_FIXED.py` | | Adam, SGD, SGD with momentum and RMSProp, weight decay 0. Phase III-A: 6 architectures x 6 environments x 5 seeds with Adam (180 runs). Phase III-B: nested 5 x 5 panel x 4 optimisers x 5 seeds (500 runs, of which the 125 Adam runs are reused), giving 555 unique runs. Every optimiser starts from the same initialisation and sees the same batch order. Adam uses lr = 1e-3. When the other learning rates are calibrated, they are chosen on training loss only. |

### MLP, conditional realisation

| Phase | Entry point | Other files | Design (default profile) |
|---|---|---|---|
| C1 | `pathflow_work2_C1_architecture_conditional_realization_280epoch_single_cell.py` | | 80 candidate architectures (32 Sobol and 48 RF/EI proposals), 20 selected by morphology-aware maximin coverage. 20 architectures x 8 seeds with ReLU (160 runs) and a tanh sentinel of 6 architectures x 8 seeds (48 runs), 280 epochs, Adam with lr = 3e-3. |
| C2 | `pathflow_work2_C2_environment_task_conditional_realization_280epoch_single_cell_AUDITED_RESUME_SAFE.py` | `..._SEGFAULT_FIX.py` (earlier revision) | 24 environments matched to MLP S2 x 6 sentinel architectures from C1 x 2 tasks (regression and binary classification on the same inputs and teacher) x 2 seeds with ReLU (576 runs), and a tanh arm of 4 environments x 6 x 2 x 2 (96 runs), 280 epochs. |
| C3 | not included in this release | | |

### CNN, structural disposition

| Phase | Entry point | Other files | Design (default profile) |
|---|---|---|---|
| S1 | `pathflow_cnn_s1_architecture_structural_disposition_autodl_rfsmac_single_cell_FIXED.py` | `..._RECOVERY_ONLY_V2.py` (rebuilds statistics, figures and report from completed outputs); `..._single_cell.py` (earlier revision) | 1,008 discovery architectures (600 Sobol, 400 RF-SMAC, 8 anchors) at 60 epochs, 240 of them with BCANw at epochs 0 and 60. 48 architectures selected by maximin coverage, 48 x 2 seeds = 96 trajectories of 360 epochs. States at 0, 1, 2, 5, 10, 20, 40, 80, 120, 160, 240 and 360; BCANw at 0, 20, 80, 160, 240 and 360. |
| S2 | `pathflow_cnn_s2_environment_structural_disposition_rfsmac_single_cell_FIXED_V3.py` | `..._FIXED.py`, `..._single_cell.py` (earlier revisions) | 12 architectures frozen by CNN S1, 4 sentinels. 60 environments (34 Sobol, 2 canonical anchors, 24 RF-SMAC) at 80 epochs. Matrix of 24 environments x 12 architectures x 2 seeds = 576 runs, 360 epochs, scalar synthetic spatial regression, ReLU, Adam. |
| S3 | `pathflow_cnn_s3_STANDALONE_AUTO_PUBLICATION_single_cell_FIXED.py` | `..._RESUME_POSTTRAINING.py` (resumes geometry, dynamics and aggregation after training, without retraining) | 12 CNNs x 24 environment cells x 2 seeds x 4 optimisers (Adam, RMSProp, SGD with momentum, SGD) = 2,304 runs, 360 epochs, batch 256, weight decay 0. The architecture and environment panels are frozen before training and do not depend on any S1 or S2 outcome. |

### CNN, conditional realisation

| Phase | Entry point | Other files | Design (default profile) |
|---|---|---|---|
| C1 | `pathflow_cnn_c1_architecture_conditional_realization_FULLRUN.py` | `..._hpc_single_cell.py` (same design, run stage by stage through `PATHFLOW_CNN_C1_MODE`) | 12 frozen architectures x 1 canonical spatial-regression environment x 4 seeds = 48 trajectories of 360 epochs, ReLU, Adam with lr = 3e-3. Conditional realisation on 64 fixed test inputs at every checkpoint and on 256 inputs at epoch 0, at the best-validation state and at epoch 360. Structural and population-conditional BCANw at these three states for all 48 models, and 768 input-resolved BCANw jobs (4 architectures x 4 seeds x 16 inputs x 3 states). |
| C2 | `pathflow_cnn_c2_environment_task_conditional_realization_FULLRUN.py` | | Conditional realisation of the trained CNN S2 matrix (12 architectures x 24 environments x 2 seeds, Adam, ReLU), read from the S2 checkpoints without retraining. Principal states at epochs 0, 160 and 360, 64 inputs per state and 256 at the principal states, and 4 environment sentinels for geometry. The script uses the C1 machinery, so its opening comment repeats the C1 description; the C2 design is defined in its section 13. |
| C3 | `pathflow_cnn_c3_optimizer_conditional_realization_SELFCONTAINED_10H.py` | | 5 architecture sentinels x 3 frozen S2 environments x 2 seeds x 4 optimisers (Adam and RMSProp with lr = 3e-3, SGD with momentum with lr = 3e-2, SGD with lr = 1e-2) = 120 matched trajectories of 80 epochs, states at 0, 1, 5, 20, 40 and 80. Conditional realisation on 64 inputs at intermediate states and 128 at epochs 0 and 80. BCANw on a balanced sentinel arm (3 architectures x 2 environments x 2 seeds x 4 optimisers). |

### Case studies on real data (`scripts/Case_Study`)

| Script | Role |
|---|---|
| `pathflow_work2_benchmark_FMNIST_MPI3D_case_study.py` | Fashion-MNIST and MPI3D-Toy benchmark. Source of the MPI3D case study. |
| `pathflow_work2_benchmark_FMNIST_dSprites_MLP_CNN_case_study_HPC_single_cell_AUDITED_V6.py` | Fashion-MNIST and dSprites benchmark. The Fashion-MNIST case study is recomputed on the Fashion-MNIST subset of this run; the dSprites arm is not used in the paper. |
| `pathflow_work2_benchmark_FMNIST_dSprites_MLP_CNN_case_study_HPC_single_cell_V2.py` | Earlier revision of the previous script. |

Both benchmarks use the same computational design, so that differences between the datasets can be read against the same carriers.

| | MPI3D | Fashion-MNIST |
|---|---|---|
| Task | Regression of the normalised horizontal position | Ten-class classification |
| Data | MPI3D-Toy, controlled generative image factors | Natural benchmark images |
| Preprocessing | Area-resized to 16 x 16 and standardised with the training-subset mean and standard deviation; 20,000 / 5,000 / 5,000 train, validation and test examples | Same |
| Architectures | MLPs M1, M2, M3 and CNNs C1, C2, C3 | Same panel |
| Training | SGD with momentum (lr = 1e-2), RMSProp and Adam (lr = 1e-3); ReLU, tanh and SiLU; 3 seeds; at most 160 epochs, batch 256, weight decay 0 | Same |
| Trained networks | 162 | 162 |
| States per network | Structural, clean conditional, shifted conditional | Same |
| Condition population | 160 test inputs, balanced over horizontal-position levels | 160 test images, 16 per class |
| Shift | Deterministic additive Gaussian sensor noise at 15 dB SNR, applied in raw image space before the clean normalisation | Same |
| Population geometries | 90 | 90 |

The three MLPs have hidden widths M1 = [512, 384, 256, 128, 64], M2 = [384, 384, 384, 256, 128] and M3 = [512, 256, 64, 256, 128]. The three CNNs use valid convolutions with global spatial averaging at the output, with hidden channels and kernel sizes C1 = [4, 8, 8, 12] / [3, 3, 3, 3, 1], C2 = [4, 8, 3, 8, 10] / [3, 3, 1, 3, 3, 1] and C3 = [3, 6, 8, 8, 10, 8] / [3, 3, 3, 1, 3, 1, 1]. The two families are not matched in parameter count, so performance differences between them serve as a competence check and not as a comparison of architectures.

### Causal intervention (`scripts/causal_interventions`)

An intervention multiplies a selected set of contributions by $g = 1-\alpha$ during inference. It changes neither the parameters nor the architecture, and the downstream execution of both members of the condition is recomputed. Each experiment answers one question, and the later experiments build on the earlier ones. The paper refers to the experiments by their content; the numbers follow the results archive and the table folders.

| Experiment | Entry point | Question and design | Interventions |
|---|---|---|---|
| 1. Realisation-guided intervention | `causal_int_exp1_SLURM_SAFE_v3.py` (earlier revision: `causal_int_exp1.py`) | Does the realised flow identify load-bearing computation beyond structural prominence and ordinary active support? The script trains 324 networks with the case-study design on Fashion-MNIST and MPI3D-Toy (3 MLPs and 3 CNNs, 3 optimisers, 3 activations, 3 seeds, 160 epochs). Three target sets per condition: the top 10% of positive realised support, an output-matched structural set and an activation-matched set. 40 balanced conditions, clean and shifted, $\alpha = 0.25$. | 324 x 40 x 2 x 3 = 77,760 |
| 2. Condition crossover | `exp2_condition_crossover_SLURM_SAFE_v1.py` | Is responsibility specific to the condition rather than to the network? Ten paired items per network; targets identified under the clean and the shifted member of each pair and evaluated under both. | 324 x 10 x 4 = 12,960 |
| 3. Dose response | `exp3_dose_response_SLURM_SAFE_v1.py` | Is the effect graded rather than tied to one intervention strength? 10 conditions, clean and shifted, the three target sets of Experiment 1, $\alpha \in \{0.10, 0.25, 0.50\}$. The target set is selected once from the unperturbed state and held fixed across dose. | 324 x 10 x 2 x 3 x 3 = 58,320 |
| 4. Multiscale follow-up | `exp4_multiscale_followup_BCANw_SLURM_SAFE_v2.py` | How does the induced geometry change once the realised computation has been changed? BCANw of the population flow before and after intervention at $\alpha = 0.25$, over ten BCANw observables. | |
| 5. Occurrence and shared identity (CNNs) | `exp5_cnn_occurrence_shared_identity_SLURM_SAFE_v2.py` (earlier revision: `_v1`) | Does it matter which spatial use of a shared convolutional weight is attenuated? Occurrence-resolved against shared-parameter targeting under the same primitive-occurrence budget and $\alpha = 0.25$, in the 162 CNNs, 10 conditions, clean and shifted. | |
| 5.1 Equivariance of the realised field (CNNs) | `exp5_1_fieldA_geometric_symmetry_SLURM_SAFE_v3.py` (earlier revision with translations only: `exp5_1_fieldA_realization_symmetry_SLURM_SAFE_v1.py`) | Does the realised input-origin field move with the input? The field is recomputed for transformed inputs and compared with the transported and the untransformed field. Fashion-MNIST: translations of 1 or 2 pixels in four directions, rotations of $\pm10^\circ$ and $\pm20^\circ$, scaling by 0.90 and 1.10. MPI3D: vertical translations of 1 or 2 pixels. No intervention. | 32,400 field comparisons |

Two further scripts work on the completed Experiment 5 and 5.1 results without new computation:

- `exp5_1_all_conditions_visual_atlas_SLURM_SAFE_v4.py` renders the field figures for all ten conditions of every CNN;
- `exp5_v4_post_analysis_contact_sheets_SLURM_SAFE_v2.py` (earlier revision: `_v1`) runs the class- and stratum-level analysis and builds the contact sheets. Its tables are in `tables/interventions_and_casestudy/Experiment5/`.

Experiments 1 to 3 together contain the 149,040 controlled interventions reported in the paper. In all three, every baseline and post-intervention realisation was defined, no intervention produced a degenerate measure, and the intervention operator passed all 34 numerical self-tests.

---

## Order of execution and hand-offs

Some phases read the frozen panels or checkpoints written by an earlier phase. Run them from the same working directory, or point to the earlier output with the variable given.

**MLP**

1. **S1** writes `pathflow_architecture_universality_v5_dense_results/`.
2. **S2** reads the S1 architecture tables from that folder and writes `pathflow_phase2_environment_universality_results/`.
3. **S3** requires the S2 tables `architecture_panel.csv` and `selected_environments.csv` in the research profile. Set `PATHFLOW_PHASE2_DIR` if they are elsewhere.
4. **C1** reads an expanded architecture panel from `PATHFLOW_W2_P2B_EXPANDED_DIR`. If it is absent, the script reruns the RF-SMAC architecture discovery itself.
5. **C2** selects its six sentinel architectures from the C1 panel when it is available (`PATHFLOW_W2_C2_C1_PANEL_CSV` or `PATHFLOW_W2_C2_C1_OUTDIR`), and otherwise uses the frozen sentinel panel embedded in the script.

**CNN**

1. **S1** writes the 12-architecture panel used by S2 (`tables/s2_12_architecture_panel.csv`).
2. **S2** reads that panel from `PATHFLOW_CNN_S2_S1_PANEL`. An embedded fallback panel is used only when `PATHFLOW_CNN_S2_ALLOW_FALLBACK_PANEL=1`.
3. **S3** is standalone.
4. **C1** trains its own networks and needs no earlier output.
5. **C2** reads the CNN S2 checkpoints and panels, and **C3** reads the CNN S2 architecture and environment panels. Both locate them through the constant `CONDITIONAL_ROOT` at the start of their section 13. Set it to the folder that contains `structural/S2/` and `conditional/` before running.

**Causal intervention**

1. **Experiment 1** trains the 324 networks and writes `pathflow_work2_selfcontained_realization_intervention_FMNIST_MPI3D_results/`.
2. **Experiments 2 to 5.1** reuse these checkpoints without retraining. They look for that folder next to the script; set `PATHFLOW_EXP1_RESULTS` if it is elsewhere.
3. The visual atlas and the post-analysis read the completed Experiment 5 and 5.1 folders inside the same results root.

---

## Installation

The scripts need Python 3.9 or later, a CUDA GPU for the training phases and a Linux environment; several of them were written for SLURM clusters and use Unix-only modules. The MLP S1 run log records Python 3.9.25, PyTorch 2.8.0 with CUDA 12.8 and one NVIDIA L4.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from https://pytorch.org if the default wheel does not.

**Data.** The controlled S and C phases generate their synthetic data deterministically, so nothing has to be downloaded for them. The case studies and the intervention experiments use public datasets:

- **Fashion-MNIST** (Xiao, Rasul and Vollgraf, 2017), loaded through `torchvision`;
- **MPI3D-Toy** (Gondal et al., 2019), as `mpi3d_toy.npz`;
- **dSprites** (Matthey et al., 2017), used only by the Fashion-MNIST and dSprites benchmark.

Each script downloads a missing dataset only when its download switch is set, for example `PATHFLOW_CAUSAL_ALLOW_DOWNLOAD=1` for Experiment 1. Alternatively, point to local copies with `MPI3D_NPZ`, `DSPRITES_NPZ` or the script's `..._FMNIST_ROOT` variable. The download addresses are given in each script.

---

## Running an experiment

Each script runs its full design by default. Most offer a reduced profile or a preflight mode for testing the pipeline before a long run, for example:

```bash
# full run
python -u scripts/MLP/structural/S1/pathflow_architecture_universality_experiment_v5_densebo_commonwindow_final.py

# reduced test run (MLP S2)
PATHFLOW_PHASE2_PROFILE=pilot python -u scripts/MLP/structural/S2/pathflow_phase2_environment_universality_halfscale_singlecell.py

# preflight only (intervention Experiment 1)
PATHFLOW_CAUSAL_PREFLIGHT_ONLY=1 python -u scripts/causal_interventions/causal_int_exp1_SLURM_SAFE_v3.py
```

Main controls for the controlled phases, all set through environment variables:

| Script | Profile or mode variable (default) | Other controls |
|---|---|---|
| MLP S1 | `RUN_PROFILE` constant in the file (`research`; `pilot` for testing) | |
| MLP S2 | `PATHFLOW_PHASE2_PROFILE` (`research`; `pilot`) | |
| MLP S3 | `PATHFLOW_PHASE3_PROFILE` (`research`; `pilot`) | `PATHFLOW_PHASE2_DIR`, `PATHFLOW_PHASE3_OUTDIR` |
| MLP C1 | `PATHFLOW_W2_C1_PROFILE` (`publication_colab`; `pilot`) | `PATHFLOW_W2_C1_OUTDIR`, `PATHFLOW_W2_C1_RESUME` |
| MLP C2 | `PATHFLOW_W2_C2_PROFILE` (`publication_colab`; `pilot`) | `PATHFLOW_W2_C2_OUTDIR`, `PATHFLOW_W2_C2_RESUME`, `PATHFLOW_W2_C2_POSTPROCESS_ONLY` |
| CNN S1 | `PATHFLOW_CNN_S1_PROFILE` (`publication_colab`; `pilot`) | `PATHFLOW_CNN_S1_OUTDIR`, `PATHFLOW_CNN_S1_RESUME`, `PATHFLOW_CNN_S1_ANALYSIS_ONLY` |
| CNN S2 | `PATHFLOW_CNN_S2_PROFILE` (`publication_colab`; `pilot`, `smoke`) | `PATHFLOW_CNN_S2_S1_PANEL`, `PATHFLOW_CNN_S2_ALLOW_FALLBACK_PANEL` |
| CNN S3 | `PATHFLOW_CNN_S3_PROFILE` (`publication`; `pilot`, `smoke`) | `PATHFLOW_CNN_S3_MODE`, `PATHFLOW_CNN_S3_SHARDS`, `PATHFLOW_CNN_S3_SHARD_INDEX` |
| CNN C1, C2 | `PATHFLOW_CNN_C1_MODE`, `PATHFLOW_CNN_C2_MODE` (full run; `preflight`, `validate`, `train`, `condition`, `geometry`, `input_geometry`, `aggregate`) | `..._SHARDS`, `..._SHARD_INDEX`, `..._OUTDIR` |
| CNN C3 | full run by default | `PATHFLOW_CNN_C3_EPOCHS` |

The case-study scripts use variables beginning with `PATHFLOW_BENCH_`, and the intervention scripts use `PATHFLOW_CAUSAL_`, `PATHFLOW_CROSSOVER_`, `PATHFLOW_DOSE_`, `PATHFLOW_EXP4_`, `PATHFLOW_EXP5_` and `PATHFLOW_EXP51_`. Each script lists its variables in its opening comment.

**Resuming.** The scripts write their results to disk as they go. All except MLP S1 resume from completed work when rerun with the same output folder; do not set a `START_FRESH` variable when recovering an interrupted run, since it discards completed work. MLP S1 is designed as a single uninterrupted run: it clears its model checkpoints and curve caches at the start of every run (`START_FRESH = True` in the file).

**Sharded runs.** CNN S3, C1 and C2 can be split across a SLURM job array. For CNN S3:

```bash
PATHFLOW_CNN_S3_MODE=calibrate python -u pathflow_cnn_s3_STANDALONE_AUTO_PUBLICATION_single_cell_FIXED.py
PATHFLOW_CNN_S3_MODE=train    PATHFLOW_CNN_S3_SHARDS=64 PATHFLOW_CNN_S3_SHARD_INDEX=$SLURM_ARRAY_TASK_ID python -u pathflow_cnn_s3_STANDALONE_AUTO_PUBLICATION_single_cell_FIXED.py
PATHFLOW_CNN_S3_MODE=geometry PATHFLOW_CNN_S3_SHARDS=64 PATHFLOW_CNN_S3_SHARD_INDEX=$SLURM_ARRAY_TASK_ID python -u pathflow_cnn_s3_STANDALONE_AUTO_PUBLICATION_single_cell_FIXED.py
PATHFLOW_CNN_S3_MODE=dynamics PATHFLOW_CNN_S3_SHARDS=64 PATHFLOW_CNN_S3_SHARD_INDEX=$SLURM_ARRAY_TASK_ID python -u pathflow_cnn_s3_STANDALONE_AUTO_PUBLICATION_single_cell_FIXED.py
PATHFLOW_CNN_S3_MODE=aggregate python -u pathflow_cnn_s3_STANDALONE_AUTO_PUBLICATION_single_cell_FIXED.py
```

Without a mode variable, these scripts run all stages in sequence.

---

## Outputs

Each script writes to its own folder in the working directory. The folder contains:

- `tables/`: CSV and JSON tables, including run manifests, frozen panels, per-state descriptors and statistical tests;
- figures and, for most phases, a PDF report;
- checkpoints used for resumption.

The tables in this repository are the `tables/` folders of the reported runs. Checkpoints and figures are not included.

### Large tables

Ten observation-level tables exceed 50 MB and are therefore not stored in this repository. They will be archived with the Zenodo release in a `large_tables/` folder that mirrors `tables/`:

| File | Size |
|---|---|
| `CNN/PhaseC1/conditional_layer_input_observations.csv` | 89.7 MB |
| `CNN/PhaseC2/conditional_input_observations.csv` | 289.5 MB |
| `CNN/PhaseC2/conditional_layer_input_observations.csv` | 1,137.4 MB |
| `CNN/PhaseC3/conditional_layer_input_observations.csv` | 93.8 MB |
| `CNN/PhaseS3/geometry_scales.csv` | 232.9 MB |
| `interventions_and_casestudy/Experiment4/geometry_scales.csv` | 54.5 MB |
| `interventions_and_casestudy/Experiment4/local_geometry.csv` | 88.0 MB |
| `MLP/PhaseC2/endpoint_input_summary.csv` | 77.1 MB |
| `MLP/PhaseC2/trajectory_geometry_scales.csv` | 295.1 MB |
| `MLP/PhaseS3/geometry_scale_profiles.csv` | 116.6 MB |

Every summary and test table derived from them is in `tables/`.

**Paths in the tables.** Some tables record the location of checkpoints and figures on the computing cluster. The cluster-specific part of these paths has been replaced by `<PROJECT_ROOT>`.


## Notes

- **Reproducibility.** Seeds, data generation, data splits and batch order are fixed in each script. Matched comparisons share initialisation and batch order across the factor being varied.
- **Dependencies installed on the fly.** Some of the controlled-phase scripts call `pip install` if `scikit-learn`, `smac` or `ConfigSpace` is missing, so that they run in a fresh notebook. Installing `requirements.txt` beforehand avoids this.
- **PyTorch 2.6 and later.** From PyTorch 2.6, `torch.load` defaults to `weights_only=True`. Resuming from checkpoints written by an earlier PyTorch version may then fail in scripts that do not set this argument. CNN S2 `FIXED_V3` handles it explicitly.
- **Cluster paths.** Machine-specific paths and account details in the scripts, the run log and the tables have been replaced by placeholders such as `<PROJECT_ROOT>`.

---

## Licence

The code is released under the MIT Licence (see `LICENSE`). The tables and the results archive are released under CC BY 4.0.

## Citation

If you use this code or these tables, please cite the paper. A `CITATION.cff` file is provided, and the archived release will carry a DOI.
