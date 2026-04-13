# Project Context: oPhys_Transcriptomics

## Project Overview

Multi-modal interrogation of visual cortical circuits in mouse primary visual cortex (V1). Combines two-photon calcium imaging with post-hoc spatial transcriptomics (10x Xenium) and hierarchical cell-type classification (Allen Institute taxonomy) to study how genetically defined cell types encode visual information, interact in local circuits, and adapt to changing stimulus contexts and behavioral states.

---

## Animals & Cells

| Mouse ID | n_cells | Excitatory | Inhibitory | GLM cells |
|----------|---------|------------|------------|-----------|
| 778174   | 614     | 558 (L2/3 IT: 245, L4/5 IT: 313) | 56 (Pvalb: 26, Vip: 20, Sst: 6, Lamp5: 4) | 158 |
| 786297   | 1,173   | 1,059 (L2/3 IT: 510, L4/5 IT: 549) | 114 (Pvalb: 49, Sst: 30, Vip: 26, Lamp5: 9) | 1,173 |
| 797371   | 1,037   | 934 (L2/3 IT: 413, L4/5 IT: 498, L5 ET: 23) | 103 (Pvalb: 45, Vip: 29, Sst: 20, Lamp5: 9) | 1,037 |
| **Total** | **2,824** | **2,551 (~90%)** | **273 (~10%)** | **2,368** |

### Cell-Type Hierarchy (AIBS Taxonomy)
```
class (3 per mouse)
 └── subclass (6-7 per mouse)
      └── supertype (22-28 per mouse)
           └── cluster (50-72 per mouse)
```
Each level has `_name`, `_label`, and `_bootstrapping_probability`.

### Subclass Names (used throughout code)
- Excitatory: `007 L2/3 IT CTX Glut`, `006 L4/5 IT CTX Glut`, `022 L5 ET CTX Glut`
- Inhibitory: `052 Pvalb Gaba`, `053 Sst Gaba`, `046 Vip Gaba`, `049 Lamp5 Gaba`

---

## Experimental Design

3 grating sessions per mouse (separate days), each with identical structure:
- **2,186 trials** total per session (including 27 catch / grey-screen trials)
- **41 timepoints** per trial at 10 Hz (−1 s to +3 s relative to stimulus onset)
- **8 directions** × **5 contrasts** (0.05–0.8) × **5 temporal frequencies** (1–15 Hz)
- Spatial frequency fixed at 0.04 cpd; 2-second stimulus presentations

### Block Structure
| Blocks 0, 2 | **Contrast context** | TF fixed at 1 Hz | Contrast × Direction varies |
|---|---|---|---|
| Blocks 1, 3 | **Speed context** | Contrast fixed at 0.8 | TF × Direction varies |

Each block ~800–825 stimulus presentations.

---

## Data Storage

### Zarr Store Locations
Full multimodal stores are at:
```
Data/multimodal_{mouse_id}_ophys_xenium/{mouse_id}_multimodal_data.zarr
```
Partial copies exist at `Data/{mouse_id}_multimodal_data.zarr` (may lack transcriptomics/unique_id).

AnnData files also available: `Data/adata_{mouse_id}.h5ad`

### Zarr Hierarchy

```
{mouse_id}_multimodal_data.zarr/
├── unique_id/                          # (n_cells,) object — cell IDs matching across modalities
│
├── morphology/
│   └── mask_properties/
│       ├── centroid_x_um               # (n_cells,) float64 — 3D spatial coordinates
│       ├── centroid_y_um               # (n_cells,) float64
│       ├── centroid_z_um               # (n_cells,) float64
│       ├── n_voxels                    # (n_cells,) int64   — cell body volume proxy
│       ├── size_pc1_um                 # (n_cells,) float64 — principal axes of soma shape
│       ├── size_pc2_um                 # (n_cells,) float64
│       ├── size_pc3_um                 # (n_cells,) float64
│       ├── size_x_um                   # (n_cells,) float64 — bounding box dimensions
│       ├── size_y_um                   # (n_cells,) float64
│       ├── size_z_um                   # (n_cells,) float64
│       └── angle_deg_xy               # (n_cells,) float64 — soma orientation (−180° to +180°)
│
├── transcriptomics/
│   ├── cell_type/                      # 13 arrays
│   │   ├── class_name                  # (n_cells,) object  — e.g. "01 IT-ET Glut"
│   │   ├── class_label                 # (n_cells,) int/object
│   │   ├── class_bootstrapping_probability  # (n_cells,) float64
│   │   ├── subclass_name               # (n_cells,) object  — e.g. "006 L4/5 IT CTX Glut"
│   │   ├── subclass_label              # (n_cells,) int/object
│   │   ├── subclass_bootstrapping_probability
│   │   ├── supertype_name              # (n_cells,) object  — e.g. "0028 L4/5 IT CTX Glut_6"
│   │   ├── supertype_label
│   │   ├── supertype_bootstrapping_probability
│   │   ├── cluster_name                # (n_cells,) object
│   │   ├── cluster_label
│   │   ├── cluster_bootstrapping_probability
│   │   └── cluster_alias               # (n_cells,) int/object
│   │
│   └── cellxgene/                      # 299 gene expression arrays
│       ├── Pvalb                       # (n_cells,) float64
│       ├── Sst                         # (n_cells,) float64
│       ├── Vip                         # (n_cells,) float64
│       ├── Cck                         # ... (299 total)
│       └── ...
│
└── ophys/
    └── drifting_gratings/
        ├── session_1/
        ├── session_2/
        └── session_3/                  # Each session contains:
            │
            ├── stim_aligned_dff/
            │   ├── GratingStim/
            │   │   ├── dff             # (2186, 41, n_cells) float32 — ΔF/F at 10 Hz
            │   │   ├── running         # (2186, 41, 2) float32 — speed + acceleration
            │   │   ├── time_relative   # (41,) float64 — −1.0 to +3.0 s
            │   │   └── trial_info/     # 13 arrays, each (2186,)
            │   │       ├── orientation           # float64 — 0,45,90,...,315
            │   │       ├── contrast              # float64 — 0.05,0.1,0.2,0.4,0.8
            │   │       ├── temporal_frequency    # float64 — 1,2,4,8,15
            │   │       ├── spatial_frequency     # float64 — fixed 0.04
            │   │       ├── stim_block            # float64 — 0,1,2,3
            │   │       ├── stim_name             # object
            │   │       ├── stim_type             # object
            │   │       ├── gray_screen           # bool
            │   │       ├── start_time            # float64
            │   │       ├── stop_time             # float64
            │   │       ├── duration              # float64
            │   │       ├── stim_index            # float64
            │   │       └── stim_index_block      # float64
            │   │
            │   ├── Catch/
            │   │   ├── dff             # (27, 51, n_cells) float32 — blank trials, −1 to +4 s
            │   │   ├── running         # (27, 51, 2) float32
            │   │   ├── time_relative   # (51,) float64
            │   │   └── trial_info/     # 13 arrays (stimulus fields NaN, gray_screen=True)
            │   │
            │   └── GreyScreen/
            │       ├── dff             # (2, 3624, n_cells) float32 — ~360 s spontaneous epochs
            │       ├── running         # (2, 3624, 2) float32
            │       ├── time_relative   # (3624,) float64
            │       └── trial_info/     # 13 arrays
            │
            ├── tuning_properties/      # 12 precomputed per-cell metrics
            │   ├── OSI                 # (n_cells,) float64 — Orientation Selectivity Index
            │   ├── DSI                 # (n_cells,) float64 — Direction Selectivity Index
            │   ├── pref_ori            # (n_cells,) float64 — Preferred orientation (0–180°)
            │   ├── max_response        # (n_cells,) float64
            │   ├── mean_response       # (n_cells,) float64
            │   ├── bandwidth           # (n_cells,) float64 — von Mises HWHM
            │   ├── C50                 # (n_cells,) float64 — semi-saturation contrast
            │   ├── Rmax_crf            # (n_cells,) float64
            │   ├── n_exponent          # (n_cells,) float64 — Naka-Rushton exponent
            │   ├── pref_TF             # (n_cells,) float64 — preferred temporal frequency
            │   ├── TF_max_response     # (n_cells,) float64
            │   └── TF_lowpass_idx      # (n_cells,) float64
            │
            └── glm/                    # Pre-fitted ridge-regularized encoding model
                ├── alpha/
                │   └── alphas          # (n_glm_cells,) float64 — regularization (10/100/1000)
                ├── coef/               # 167+ coefficient arrays
                │   ├── coef_visual              # (n_glm_cells, 4920) float32
                │   ├── coef_state               # (n_glm_cells, 21) float32
                │   ├── coef_pupil_running       # (n_glm_cells, 21) float32
                │   ├── preferred_direction      # (n_glm_cells,) float64
                │   ├── coef_block_{b}_TF_{t}_contrast_{c}_direction_{d}  # (n_glm_cells, 30) per condition
                │   └── glm-coef_*               # 186 aggregate arrays (marginals, joint, pref-relative)
                ├── score/              # 171+ R² score arrays
                │   ├── score_total              # (n_glm_cells,) float64
                │   ├── score_train              # (n_glm_cells,) float64
                │   ├── score_test               # (n_glm_cells,) float64
                │   ├── score_eval               # (n_glm_cells,) float64
                │   ├── score_visual             # (n_glm_cells,) float64
                │   ├── score_state              # (n_glm_cells,) float64
                │   ├── score_pupil_running      # (n_glm_cells,) float64
                │   └── score_block_{b}_TF_{t}_contrast_{c}_direction_{d}  # per condition
                ├── y/
                │   └── y               # (n_glm_cells, ~37910) float32 — observed continuous ΔF/F
                └── y_hat/
                    ├── y_hat           # (n_glm_cells, ~37910) float32 — full prediction
                    ├── y_hat_visual    # (n_glm_cells, ~37910) float32 — visual-only prediction
                    └── y_hat_state     # (n_glm_cells, ~37910) float32 — state-only prediction
```

**Note on GLM cell counts:** Mouse 778174 has GLM fits for only 158/614 cells; mice 786297 and 797371 have GLM fits for all cells.

---

## Stimulus Parameters

| Parameter | Values |
|-----------|--------|
| Orientations/Directions | 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° |
| Contrasts | 0.05, 0.1, 0.2, 0.4, 0.8 |
| Temporal Frequencies | 1, 2, 4, 8, 15 Hz |
| Spatial Frequency | 0.04 cpd (fixed) |
| Trial Duration | 2 seconds |
| Sampling Rate | 10 Hz (100 ms bins) |
| Time Window | −1 s to +3 s (41 timepoints) |

---

## Analysis Domains

| Domain | Topic | Key Questions |
|--------|-------|---------------|
| **A** | Transcriptomic Identity & Neural Coding | Do cell types have distinct tuning? Do genes predict function? Population coding geometry |
| **B** | Context Adaptation & History | Block-to-block adaptation, context-dependent tuning, multi-day drift |
| **C** | Running State Modulation | Cell-type-specific running modulation, VIP integration hypothesis, sub-second coupling |
| **D** | Spatial Organization | Functional clustering, depth profiles, local neighborhood effects |
| **E** | Functional Connectivity | Noise correlations, directed connectivity (Granger causality), population coupling, spontaneous assemblies |
| **F** | RNN Modeling | Dale's law RNN, data-driven RNN, temporal trajectory RNN |
| **G** | GLM Encoding Analysis | Visual vs. state drive, condition-specific kernels, residual structure |
| **H** | Morphology | Soma shape predicts cell type, morphology–function relationships |

---

## Code Organization

```
code/
├── functions/                          # Shared utility modules
│   ├── __init__.py
│   ├── data_loading.py                 # load_mouse_zarr(), load_zarr_10hz(), zarr_to_df()
│   ├── tuning.py                       # compute_osi/dsi, von_mises_fit, naka_rushton, normalization_model
│   ├── analysis.py                     # get_block_responses, compute_adaptation_index, morans_i, xcorr_pair, pairwise_granger, linear_CKA
│   ├── glm.py                          # add_glm_aggregate_columns, _parse_coef_key
│   └── models.py                       # DalesRNN, PredictiveRNN, TemporalRNN (PyTorch)
├── Preprocessing/                      # Data preparation notebooks
│   ├── Load_all_assets.ipynb
│   ├── Read_coregistration_tables.ipynb
│   ├── Read_morphology_zstack.ipynb
│   ├── Read_ophys_data.ipynb
│   ├── Read_xenium_data.ipynb
│   ├── Save_multimodal_data.ipynb
│   └── analysis_utils.py
├── Domain_A_Transcriptomic_Function.ipynb
├── Domain_B_Context_Adaptation.ipynb
├── Domain_C_Running_State.ipynb
├── Domain_D_Spatial_Organization.ipynb
├── Domain_E_Connectivity.ipynb
├── Domain_F_RNN_Modeling.ipynb
├── Domain_G_GLM_Encoding.ipynb
├── Domain_H_Morphology.ipynb
├── Compute_Tuning_Properties.ipynb     # Writes tuning_properties/ to zarr stores
└── Add_columns_to_glm.ipynb            # Writes aggregate GLM coef/score arrays
```

### Data Loading Pattern
All analysis notebooks use `sys.path.insert(0, '.')` and import from `functions/`:
```python
from functions.data_loading import load_mouse_zarr, load_zarr_10hz
adata_list = [load_mouse_zarr(mid, zarr_dir='multimodal_data', include_genes=True) for mid in MOUSE_IDS]
```
`load_mouse_zarr()` returns a `SimpleNamespace(X, obs, var, n_obs, n_vars)` with:
- `X`: (n_cells, n_trials) matrix of trial-averaged ΔF/F
- `obs`: DataFrame with cell metadata (unique_id, mouse_id, subclass_name, coordinates, gene expression if `include_genes=True`)
- `var`: DataFrame with trial metadata (orientation, contrast, TF, running, block, day)

---

## Key Constants Used in Code

```python
MOUSE_IDS = ['778174', '786297', '797371']
SESSIONS = ['session_1', 'session_2', 'session_3']
ZARR_DIR = 'multimodal_data'  # symlink or path context depends on notebook cwd

ORIENTATIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315])
CONTRASTS = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
TFS = np.array([1, 2, 4, 8, 15])

SUBCLASS_ORDER = [
    '007 L2/3 IT CTX Glut', '006 L4/5 IT CTX Glut', '022 L5 ET CTX Glut',
    '052 Pvalb Gaba', '053 Sst Gaba', '046 Vip Gaba', '049 Lamp5 Gaba'
]
SUBCLASS_SHORT = {
    '007 L2/3 IT CTX Glut': 'L2/3 IT', '006 L4/5 IT CTX Glut': 'L4/5 IT',
    '022 L5 ET CTX Glut': 'L5 ET', '052 Pvalb Gaba': 'Pvalb',
    '053 Sst Gaba': 'Sst', '046 Vip Gaba': 'Vip', '049 Lamp5 Gaba': 'Lamp5'
}
```
