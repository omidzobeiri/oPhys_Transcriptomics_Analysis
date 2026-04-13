# oPhys_Transcriptomics — Workspace Instructions

## Project Context

Neuroscience research project analyzing mouse V1 visual cortex using multi-modal data (two-photon calcium imaging + spatial transcriptomics + morphology). See [CONTEXT.md](/CONTEXT.md) for full data schema, cell-type hierarchy, and zarr structure. See [PROJECT_DOCUMENT.md](/PROJECT_DOCUMENT.md) for scientific questions and analysis plans.

## Architecture

```
code/
├── functions/          # Shared Python modules (data_loading, tuning, analysis, glm, models)
├── Preprocessing/      # Data ingestion notebooks (zarr store creation)
├── Domain_{A-H}_*.ipynb  # Analysis notebooks (one per scientific domain)
├── Compute_Tuning_Properties.ipynb
└── Add_columns_to_glm.ipynb
Data/                   # Zarr stores + h5ad files (gitignored, ~6 GB)
```

- **No package manager**: no requirements.txt/pyproject.toml. Dependencies: numpy, pandas, scipy, zarr, matplotlib, seaborn, scikit-learn, anndata, torch.
- **No tests**: no test framework or test files exist.
- **No CI/CD**: no pipelines configured.

## Data Loading

All notebooks run from `code/` as cwd. Data is loaded via shared functions:

```python
import sys; sys.path.insert(0, '.')
from functions.data_loading import load_mouse_zarr, load_zarr_10hz

# Trial-averaged data → SimpleNamespace(X, obs, var, n_obs, n_vars)
adata_list = [load_mouse_zarr(mid, zarr_dir='multimodal_data', include_genes=True) for mid in MOUSE_IDS]

# 10 Hz trial-resolved data → dict with keys: dff, unique_ids, running, time_rel, trial_info
pk = load_zarr_10hz(mouse_id, session='session_1', zarr_dir='multimodal_data')
```

Zarr stores are at `Data/multimodal_{mouse_id}_ophys_xenium/{mouse_id}_multimodal_data.zarr`. The `zarr_dir='multimodal_data'` parameter resolves via symlinks or relative paths from notebook cwd.

## Key Constants

```python
MOUSE_IDS = ['778174', '786297', '797371']
SESSIONS  = ['session_1', 'session_2', 'session_3']
ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
CONTRASTS    = [0.05, 0.1, 0.2, 0.4, 0.8]
TFS          = [1, 2, 4, 8, 15]
```

Subclass names use AIBS taxonomy codes (e.g., `'007 L2/3 IT CTX Glut'`, `'052 Pvalb Gaba'`). Short labels: `L2/3 IT`, `L4/5 IT`, `L5 ET`, `Pvalb`, `Sst`, `Vip`, `Lamp5`.

## Coding Conventions

- **Style**: snake_case functions/variables, UPPER_SNAKE constants, NumPy-style docstrings
- **No type annotations** in existing code — don't add them unless requested
- **New utility functions** go in `code/functions/` (grouped by domain: data_loading, tuning, analysis, glm, models)
- **Notebooks import from `functions/`** — never define reusable functions inline in notebooks
- **Plotting**: seaborn with `set_context('talk')`, `set_style('whitegrid')`; matplotlib for custom layouts
- **Warnings**: always `warnings.filterwarnings('ignore')` in notebooks

## Gotchas

- Mouse 778174 has GLM fits for only 158/614 cells; other mice have all cells
- `L5 ET` subclass only exists in mouse 797371
- Zarr v2 stores opened with `zarr.open(path, 'r')` — partial stores at `Data/{mouse_id}_multimodal_data.zarr` may lack transcriptomics/unique_id groups
- The `dff` array is `(n_trials, n_timepoints, n_cells)` — note the axis order
- Trial info `gray_screen` is `bool`; other fields like `stim_block` are `float64`
