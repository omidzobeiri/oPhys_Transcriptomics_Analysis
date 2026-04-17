# oPhys_Transcriptomics_Analysis

Integrative analysis of Multi-modal of visual cortical circuits: linking transcriptomic identity, spatial organization, and neural dynamics in mouse V1.

## Overview

This project investigates the relationship between transcriptomic cell-type identity, spatial organization, functional neural dynamics, and behavioral state in the mouse primary visual cortex (V1). It combines large-scale two-photon calcium imaging with post-hoc spatial transcriptomics (10x Xenium) and hierarchical cell-type classification (Allen Institute for Brain Science taxonomy).

## Dataset

- **Species / Region:** Mouse, primary visual cortex (VISp), layers 1–4/5
- **Animals:** 3 mice (IDs: 778174, 786297, 797371)
- **Total Cells:** 2,824 neurons across 3 mice
- **Imaging:** Two-photon calcium imaging (GCaMP), ΔF/F at 10 Hz, 8 planes (40–320 µm depth)
- **Transcriptomics:** 299 genes per cell via 10x Xenium
- **Cell Types:** Excitatory (L2/3 IT, L4/5 IT, L5 ET) and inhibitory (Pvalb, Sst, Vip, Lamp5)
- **Stimuli:** Drifting gratings (8 directions × 5 contrasts × 5 temporal frequencies) across 3 sessions per mouse

## Analysis Domains

| Domain | Notebook | Description |
|--------|----------|-------------|
| A | `Domain_A_Transcriptomic_Function.ipynb` | Transcriptomic identity and neural coding |
| B | `Domain_B_Context_Adaptation.ipynb` | Contextual adaptation and history dependence |
| C | `Domain_C_Running_State.ipynb` | Behavioral state modulation |
| D | `Domain_D_Spatial_Organization.ipynb` | Spatial organization of cell types |
| E | `Domain_E_Connectivity.ipynb` | Functional connectivity |
| F | `Domain_F_RNN_Modeling.ipynb` | RNN modeling |
| G | `Domain_G_GLM_Encoding.ipynb` | GLM encoding models |
| H | `Domain_H_Morphology.ipynb` | Morphological analysis |

Additional notebooks:
- `Compute_Tuning_Properties.ipynb` — Precompute per-cell tuning metrics
- `Add_columns_to_glm.ipynb` — Extend GLM feature columns

## Data

Data files are stored in `Data/` (excluded from this repo due to size). Each mouse has:
- An AnnData file (`adata_{mouse_id}.h5ad`)
- A Zarr multimodal store (`{mouse_id}_multimodal_data.zarr`) containing morphology, ophys, and transcriptomics data

See [PROJECT_DOCUMENT.md](PROJECT_DOCUMENT.md) for full data structure and analysis plan documentation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if available
```
