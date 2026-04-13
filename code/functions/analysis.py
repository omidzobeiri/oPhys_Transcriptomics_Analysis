"""Analysis utilities: adaptation, spatial statistics, connectivity, and similarity."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from statsmodels.tsa.stattools import grangercausalitytests


def get_trial_position_responses(X, var, block, n_bins=5):
    """Bin trials within a block by their position and compute mean response per bin.

    Parameters
    ----------
    X : array (n_cells, n_trials)
    var : DataFrame with trial metadata (must have 'stim_block' and 'start_time').
    block : float
        Block number.
    n_bins : int
        Number of temporal bins.

    Returns
    -------
    array (n_cells, n_bins) or None if block is empty
    """
    block_mask = var['stim_block'] == block
    block_idx = np.where(block_mask.values)[0]
    if len(block_idx) == 0:
        return None
    times = var.iloc[block_idx]['start_time'].values
    sorted_order = np.argsort(times)
    sorted_idx = block_idx[sorted_order]

    bin_size = len(sorted_idx) // n_bins
    bin_responses = []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else len(sorted_idx)
        trial_idx = sorted_idx[start:end]
        bin_responses.append(np.nanmean(X[:, trial_idx], axis=1))
    return np.column_stack(bin_responses)


def exp_decay(x, a, tau, b):
    """Exponential decay function for adaptation fitting."""
    return a * np.exp(-x / tau) + b


def get_block_responses(X, var, block, param_name, param_values):
    """Compute mean response per cell for each unique condition within a block.

    Parameters
    ----------
    X : array (n_cells, n_trials)
    var : DataFrame with trial metadata
    block : float
        Block number (0.0, 1.0, 2.0, or 3.0).
    param_name : str
        Column name in var to group by (e.g. 'contrast', 'temporal_frequency').
    param_values : array
        Unique values of param_name to iterate over.

    Returns
    -------
    dict : {(orientation, param_value): array (n_cells,)}
    """
    orientations = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    block_mask = var['stim_block'] == block
    n_cells = X.shape[0]
    responses = {}
    for ori in orientations:
        for pval in param_values:
            mask = block_mask & (var['orientation'] == ori) & (var[param_name] == pval)
            trial_idx = np.where(mask.values)[0]
            if len(trial_idx) > 0:
                responses[(ori, pval)] = np.nanmean(X[:, trial_idx], axis=1)
            else:
                responses[(ori, pval)] = np.full(n_cells, np.nan)
    return responses


def compute_adaptation_index(resp_early, resp_late, conditions):
    """Adaptation index = (early - late) / (|early| + |late|) averaged over conditions.

    Parameters
    ----------
    resp_early : dict {condition: array (n_cells,)}
    resp_late : dict {condition: array (n_cells,)}
    conditions : list of condition keys

    Returns
    -------
    array (n_cells,) : mean adaptation index across conditions
    """
    ai_all = []
    for cond in conditions:
        r1 = resp_early[cond]
        r2 = resp_late[cond]
        denom = np.abs(r1) + np.abs(r2)
        denom[denom < 1e-8] = np.nan
        ai = (r1 - r2) / denom
        ai_all.append(ai)
    ai_matrix = np.column_stack(ai_all)
    return np.nanmean(ai_matrix, axis=1)


def morans_i(values, coords, distance_threshold):
    """Compute Moran's I for values given coordinates and a distance threshold.

    Parameters
    ----------
    values : array (n,)
    coords : array (n, d)
    distance_threshold : float

    Returns
    -------
    float : Moran's I statistic
    """
    n = len(values)
    mean_v = np.nanmean(values)
    dev = values - mean_v

    dist_mat = squareform(pdist(coords))
    W = (dist_mat < distance_threshold) & (dist_mat > 0)
    W_sum = W.sum()

    if W_sum == 0:
        return np.nan

    numerator = 0
    for i in range(n):
        for j in range(n):
            if W[i, j]:
                numerator += dev[i] * dev[j]

    denominator = np.sum(dev**2)
    I = (n / W_sum) * (numerator / denominator) if denominator > 0 else 0
    return I


def xcorr_pair(x, y, max_lag):
    """Cross-correlation between two signals at integer lags.

    Parameters
    ----------
    x, y : array (n_timepoints,)
    max_lag : int

    Returns
    -------
    array (2*max_lag + 1,)
    """
    n = len(x)
    xn = (x - np.mean(x)) / (np.std(x) + 1e-8)
    yn = (y - np.mean(y)) / (np.std(y) + 1e-8)
    cc = np.zeros(2 * max_lag + 1)
    for li, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            cc[li] = np.mean(xn[:n-lag] * yn[lag:]) if lag < n else 0
        else:
            cc[li] = np.mean(xn[-lag:] * yn[:n+lag]) if -lag < n else 0
    return cc


def xcorr_lagged(run_trials, dff_trials, max_lag):
    """Compute mean cross-correlation at each lag across trials.

    Parameters
    ----------
    run_trials : array (n_trials, n_timepoints)
    dff_trials : array (n_trials, n_timepoints)
    max_lag : int

    Returns
    -------
    array (2*max_lag + 1,) : cross-correlation values
    """
    n_lags = 2 * max_lag + 1
    cc = np.zeros(n_lags)
    for li, lag in enumerate(range(-max_lag, max_lag + 1)):
        rvals = []
        for tr in range(run_trials.shape[0]):
            r = run_trials[tr]
            d = dff_trials[tr]
            if lag >= 0:
                r_seg = r[:len(r)-lag] if lag > 0 else r
                d_seg = d[lag:]
            else:
                r_seg = r[-lag:]
                d_seg = d[:len(d)+lag]
            if len(r_seg) < 3 or np.std(r_seg) < 1e-6 or np.std(d_seg) < 1e-6:
                continue
            rvals.append(np.corrcoef(r_seg, d_seg)[0, 1])
        cc[li] = np.nanmean(rvals) if rvals else np.nan
    return cc


def pairwise_granger(x, y, max_lag=3):
    """Compute Granger causality (y -> x) and (x -> y) using F-test.

    Returns
    -------
    dict with 'y_to_x_p' and 'x_to_y_p' (min p-value across lags).
    """
    results = {}
    try:
        data_xy = np.column_stack([x, y])
        gc_xy = grangercausalitytests(data_xy, maxlag=max_lag, verbose=False)
        results['y_to_x_p'] = min(gc_xy[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1))
    except Exception:
        results['y_to_x_p'] = 1.0
    try:
        data_yx = np.column_stack([y, x])
        gc_yx = grangercausalitytests(data_yx, maxlag=max_lag, verbose=False)
        results['x_to_y_p'] = min(gc_yx[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1))
    except Exception:
        results['x_to_y_p'] = 1.0
    return results


def linear_CKA(X1, X2):
    """Compute linear Centered Kernel Alignment between two matrices.

    Parameters
    ----------
    X1, X2 : array (n_samples, n_features)

    Returns
    -------
    float : CKA similarity score (0 to 1)
    """
    K1, K2 = X1 @ X1.T, X2 @ X2.T
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1c, K2c = H @ K1 @ H, H @ K2 @ H
    hsic = np.trace(K1c @ K2c)
    norm = np.sqrt(np.trace(K1c @ K1c) * np.trace(K2c @ K2c))
    return hsic / norm if norm > 0 else 0
