"""Data loading utilities for zarr multimodal stores."""

import numpy as np
import pandas as pd
import zarr
from types import SimpleNamespace

SESSIONS = ['session_1', 'session_2', 'session_3']


def zarr_to_df(zarr_group):
    """Convert a zarr group to a pandas DataFrame, handling string types."""
    return pd.DataFrame({
        key: zarr_group[key][:].astype(str) if zarr_group[key][:].dtype.kind in {'S', 'O'} else zarr_group[key][:]
        for key in zarr_group.keys()
    })


def _zarr_trial_info_to_df(trial_info_group):
    """Convert a zarr trial_info group to a pandas DataFrame."""
    trial_dict = {}
    for key in trial_info_group.keys():
        arr = trial_info_group[key][:]
        # Decode byte strings when needed
        if hasattr(arr, 'dtype') and arr.dtype.kind in {'S', 'O'}:
            arr = np.array([
                x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else str(x)
                for x in arr
            ])
        trial_dict[key] = arr
    return pd.DataFrame(trial_dict)


def load_mouse_zarr(mouse_id, zarr_dir='multimodal_data', include_genes=True):
    """Load one mouse's data from zarr, returning an adata-like SimpleNamespace.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier (e.g. '778174').
    zarr_dir : str
        Path to directory containing zarr stores.
    include_genes : bool
        If True, load gene expression columns into obs DataFrame.

    Returns
    -------
    SimpleNamespace with X (n_cells, total_trials), obs (DataFrame), var (DataFrame),
    n_obs (int), n_vars (int).
    """
    z = zarr.open(f'{zarr_dir}/{mouse_id}_multimodal_data.zarr', 'r')
    ct = z['transcriptomics/cell_type']
    morph = z['morphology/mask_properties']
    obs_dict = {
        'unique_id': z['unique_id'][:].astype(str),
        'mouse_id': mouse_id,
        'class_name': ct['class_name'][:],
        'class_label': ct['class_label'][:],
        'class_bootstrapping_probability': ct['class_bootstrapping_probability'][:],
        'subclass_name': ct['subclass_name'][:],
        'subclass_label': ct['subclass_label'][:],
        'subclass_bootstrapping_probability': ct['subclass_bootstrapping_probability'][:],
        'supertype_name': ct['supertype_name'][:],
        'supertype_label': ct['supertype_label'][:],
        'supertype_bootstrapping_probability': ct['supertype_bootstrapping_probability'][:],
        'cluster_name': ct['cluster_name'][:],
        'cluster_label': ct['cluster_label'][:],
        'cluster_alias': ct['cluster_alias'][:],
        'cluster_bootstrapping_probability': ct['cluster_bootstrapping_probability'][:],
        'x_loc': morph['centroid_x_um'][:],
        'y_loc': morph['centroid_y_um'][:],
        'z_loc': morph['centroid_z_um'][:],
    }

    if include_genes:
        gene_names = sorted(z['transcriptomics/cellxgene'].keys())
        for g in gene_names:
            obs_dict[g] = z[f'transcriptomics/cellxgene/{g}'][:]

    obs_df = pd.DataFrame(obs_dict)
    n_cells = len(obs_df)

    X_sessions, Layer_sessions, var_sessions = [], [], []
    for si, sess in enumerate(SESSIONS):
        gs = z[f'ophys/drifting_gratings/{sess}/stim_aligned_dff/GratingStim']
        dff = gs['dff'][:]
        time_rel = gs['time_relative'][:]
        running = gs['running'][:]
        gray = gs['trial_info/gray_screen'][:]
        valid = ~gray
        stim_mask = (time_rel >= 0) & (time_rel <= 2.0)
        baseline_mask = (time_rel >= -1.0) & (time_rel < 0)
        dff_avg = dff[valid][:, stim_mask, :].mean(axis=1)
        dff_baseline = dff[valid][:, baseline_mask, :].mean(axis=1)
        
        run_speed = running[valid][:, stim_mask, 0].mean(axis=1)
        X_sessions.append(dff_avg.T)
        Layer_sessions.append(dff_baseline.T)  # Add layer info as a feature
        var_sessions.append(pd.DataFrame({
            'contrast': gs['trial_info/contrast'][:][valid],
            'orientation': gs['trial_info/orientation'][:][valid],
            'temporal_frequency': gs['trial_info/temporal_frequency'][:][valid],
            'spatial_frequency': gs['trial_info/spatial_frequency'][:][valid],
            'stim_block': gs['trial_info/stim_block'][:][valid],
            'stim_name': gs['trial_info/stim_name'][:][valid],
            'start_time': gs['trial_info/start_time'][:][valid],
            'stop_time': gs['trial_info/stop_time'][:][valid],
            'duration': gs['trial_info/duration'][:][valid],
            'avg_running': run_speed,
            'is_running': run_speed > 1.0,
            'day': si + 1,
        }))
    return SimpleNamespace(
        X=np.hstack(X_sessions), Layer=np.hstack(Layer_sessions), obs=obs_df, var=pd.concat(var_sessions, ignore_index=True),
        n_obs=n_cells, n_vars=sum(v.shape[0] for v in var_sessions)
    )
    


def load_zarr_10hz(mouse_id, session='session_1', zarr_dir='multimodal_data'):
    """Load 10 Hz trial-resolved data from zarr.

    Parameters
    ----------
    mouse_id : str
        Mouse identifier.
    session : str
        Session name (e.g. 'session_1').
    zarr_dir : str
        Path to directory containing zarr stores.

    Returns
    -------
    dict with keys: dff, unique_ids, running, time_rel, trial_info.
    """
    z = zarr.open(f'{zarr_dir}/{mouse_id}_multimodal_data.zarr', 'r')
    gs = z[f'ophys/drifting_gratings/{session}/stim_aligned_dff/GratingStim']
    ti_dict = {k: gs[f'trial_info/{k}'][:] for k in gs['trial_info'].keys()}
    return {
        'dff': gs['dff'][:],
        'unique_ids': z['unique_id'][:].astype(str),
        'running': gs['running'][:],
        'time_rel': gs['time_relative'][:],
        'trial_info': pd.DataFrame(ti_dict),
    }