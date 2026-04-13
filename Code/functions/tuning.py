"""Tuning property computation and storage."""

import numpy as np
import pandas as pd
import zarr
from scipy.optimize import curve_fit

ORIENTATIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315])
CONTRASTS = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
TFS = np.array([1, 2, 4, 8, 15])


def compute_osi(responses, orientations_deg):
    """Orientation Selectivity Index: 1 - circular variance."""
    theta = np.deg2rad(2 * orientations_deg)
    R = np.abs(np.sum(responses * np.exp(1j * theta))) / np.sum(np.abs(responses))
    return R


def compute_dsi(responses, orientations_deg):
    """Direction Selectivity Index."""
    theta = np.deg2rad(orientations_deg)
    R = np.abs(np.sum(responses * np.exp(1j * theta))) / np.sum(np.abs(responses))
    return R


def preferred_orientation(responses, orientations_deg):
    """Preferred orientation from vector average (0-180 range)."""
    theta = np.deg2rad(2 * orientations_deg)
    vec = np.sum(responses * np.exp(1j * theta))
    pref = np.rad2deg(np.angle(vec)) / 2
    return pref % 180


def von_mises_fit(theta, A, mu, kappa, b):
    """Von Mises function for orientation tuning."""
    return b + A * np.exp(kappa * np.cos(2 * (theta - mu)))


def naka_rushton(c, Rmax, c50, n, R0):
    """Naka-Rushton contrast response function."""
    return R0 + Rmax * (c**n) / (c**n + c50**n)


def compute_tuning_for_session(dff, trial_info, time_relative, n_cells):
    """Compute all tuning metrics for one session.

    Parameters
    ----------
    dff : array (n_trials, n_timepoints, n_cells)
    trial_info : DataFrame with columns: gray_screen, orientation, contrast,
                 temporal_frequency, stim_block
    time_relative : array (n_timepoints,)
    n_cells : int

    Returns
    -------
    tuning : DataFrame (n_cells rows) with all 12 tuning columns
    """
    gray = trial_info['gray_screen'].astype(bool).values
    valid = ~gray
    stim_mask = (time_relative >= 0) & (time_relative <= 2.0)
    dff_avg = dff[valid][:, stim_mask, :].mean(axis=1)

    var = trial_info.loc[valid].reset_index(drop=True)

    # Orientation tuning (contrast-context blocks 0, 2)
    ori_mask_blocks = var['stim_block'].isin([0.0, 2.0])
    ori_responses = np.zeros((n_cells, len(ORIENTATIONS)))
    for i, ori in enumerate(ORIENTATIONS):
        mask = ori_mask_blocks & (var['orientation'] == ori)
        tidx = np.where(mask.values)[0]
        if len(tidx) > 0:
            ori_responses[:, i] = np.nanmean(dff_avg[tidx], axis=0)

    # Contrast response function (contrast-context blocks 0, 2)
    crf_responses = np.zeros((n_cells, len(CONTRASTS)))
    for i, c in enumerate(CONTRASTS):
        mask = ori_mask_blocks & (var['contrast'] == c)
        tidx = np.where(mask.values)[0]
        if len(tidx) > 0:
            crf_responses[:, i] = np.nanmean(dff_avg[tidx], axis=0)

    # Temporal frequency tuning (speed-context blocks 1, 3)
    tf_mask_blocks = var['stim_block'].isin([1.0, 3.0])
    tf_responses = np.zeros((n_cells, len(TFS)))
    for i, tf in enumerate(TFS):
        mask = tf_mask_blocks & (var['temporal_frequency'] == tf)
        tidx = np.where(mask.values)[0]
        if len(tidx) > 0:
            tf_responses[:, i] = np.nanmean(dff_avg[tidx], axis=0)

    # Per-cell metrics
    osi = np.array([compute_osi(ori_responses[c], ORIENTATIONS) for c in range(n_cells)])
    dsi = np.array([compute_dsi(ori_responses[c], ORIENTATIONS) for c in range(n_cells)])
    pref_ori = np.array([preferred_orientation(ori_responses[c], ORIENTATIONS) for c in range(n_cells)])
    max_resp = np.max(ori_responses, axis=1)
    mean_resp = np.mean(ori_responses, axis=1)

    # Bandwidth (von Mises half-width at half-max)
    bandwidth = np.full(n_cells, np.nan)
    for c in range(n_cells):
        try:
            resp = ori_responses[c]
            if np.ptp(resp) < 0.01:
                continue
            theta_rad = np.deg2rad(ORIENTATIONS)
            popt, _ = curve_fit(
                von_mises_fit, theta_rad, resp,
                p0=[np.ptp(resp), np.deg2rad(ORIENTATIONS[np.argmax(resp)]), 1.0, np.min(resp)],
                maxfev=5000,
            )
            kappa = popt[2]
            if kappa > 0:
                bandwidth[c] = (
                    np.rad2deg(np.arccos(1 - np.log(2) / kappa))
                    if kappa > np.log(2)
                    else 90
                )
        except Exception:
            pass

    # CRF metrics (Naka-Rushton fit)
    c50_arr = np.full(n_cells, np.nan)
    rmax_arr = np.full(n_cells, np.nan)
    n_exp_arr = np.full(n_cells, np.nan)
    for c in range(n_cells):
        try:
            resp = crf_responses[c]
            if np.ptp(resp) < 0.01:
                continue
            popt, _ = curve_fit(
                naka_rushton, CONTRASTS, resp,
                p0=[np.max(resp), 0.3, 2.0, np.min(resp)],
                bounds=([0, 0.01, 0.1, -5], [20, 1.0, 10, 5]),
                maxfev=5000,
            )
            rmax_arr[c], c50_arr[c], n_exp_arr[c] = popt[0], popt[1], popt[2]
        except Exception:
            pass

    # TF metrics
    pref_tf = TFS[np.argmax(tf_responses, axis=1)]
    tf_max_resp = np.max(tf_responses, axis=1)
    tf_lowpass_idx = (
        (tf_responses[:, 0] - tf_responses[:, -1])
        / (np.max(tf_responses, axis=1) + 1e-8)
    )

    return pd.DataFrame({
        'OSI': osi,
        'DSI': dsi,
        'pref_ori': pref_ori,
        'max_response': max_resp,
        'mean_response': mean_resp,
        'bandwidth': bandwidth,
        'C50': c50_arr,
        'Rmax_crf': rmax_arr,
        'n_exponent': n_exp_arr,
        'pref_TF': pref_tf.astype(float),
        'TF_max_response': tf_max_resp,
        'TF_lowpass_idx': tf_lowpass_idx,
    })


def save_tuning_to_zarr(z, session, tuning_df):
    """Write a tuning DataFrame into the zarr store.

    Saves to ophys/drifting_gratings/{session}/tuning_properties.
    """
    sess_group = z.require_group(f'ophys/drifting_gratings/{session}')

    if 'tuning_properties' in sess_group:
        del sess_group['tuning_properties']

    tp = sess_group.create_group('tuning_properties')
    tp.attrs['description'] = 'Per-cell tuning properties computed from this session'
    tp.attrs['n_cells'] = int(len(tuning_df))
    tp.attrs['columns'] = list(tuning_df.columns)

    for col in tuning_df.columns:
        arr = tuning_df[col].to_numpy(dtype=np.float64)
        tp.create_dataset(col, data=arr, shape=arr.shape, dtype=arr.dtype, overwrite=True)
