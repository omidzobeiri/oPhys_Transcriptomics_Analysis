"""Analysis utilities: adaptation, spatial statistics, connectivity, and similarity."""

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import pdist, squareform
from scipy.stats import nct, t
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


def min_detectable_pearson_r(n, alpha=0.05, power=0.8, two_tailed=True):
    """Minimum detectable Pearson r given sample size, alpha, and power.

    Parameters
    ----------
    n : int
        Sample size.
    alpha : float, optional
        Significance level (default 0.05).
    power : float, optional
        Desired statistical power (default 0.8).
    two_tailed : bool, optional
        If True (default), use a two-tailed test. If False, use a one-tailed
        upper-tail test (positive-effect direction).

    Returns
    -------
    float
        Minimum detectable Pearson r (absolute effect size).
    """
    if n < 4:
        raise ValueError('n must be >= 4 for correlation power calculations.')
    if not (0 < alpha < 1):
        raise ValueError('alpha must be between 0 and 1.')
    if not (0 < power < 1):
        raise ValueError('power must be between 0 and 1.')

    df = n - 2
    tcrit = t.ppf(1 - alpha / 2.0, df) if two_tailed else t.ppf(1 - alpha, df)

    def _power_at_r(r):
        r = max(min(r, 1 - 1e-12), 1e-12)
        ncp = r * np.sqrt(df / (1 - r**2))
        if two_tailed:
            return 1.0 - (nct.cdf(tcrit, df, ncp) - nct.cdf(-tcrit, df, ncp))
        return 1.0 - nct.cdf(tcrit, df, ncp)

    if _power_at_r(1e-12) >= power:
        return 0.0

    return brentq(lambda r: _power_at_r(r) - power, 1e-12, 1 - 1e-12)




# =============================================================================
# Metric 1: RUNNING MODULATION INDEX (two-state)
# =============================================================================
#
# For each neuron, compare mean response on running vs stationary trials.
# To avoid stimulus confounds (if running correlates with certain stimuli),
# we compute a **stimulus-matched** running MI:
#   For each stimulus condition, compute mean(running) - mean(stationary),
#   then average across conditions.
#
# MI = (R_run - R_stat) / (R_run + R_stat + epsilon)
#
# We also compute a simpler version and the correlation with running speed.

def compute_running_mi(adata, min_trials_per_condition=2):
    """
    Compute stimulus-matched running modulation index (two-state) per neuron.
    
    For each stimulus condition (contrast x TF x orientation),

    compute the mean response for running and stationary trials separately.

    The MI is averaged across conditions, that have at least min_trials_per_condition in each state.
    
    Returns DataFrame with columns:
        running_mi:      stimulus-matched (R_run - R_stat) / (R_run + R_stat)
        running_mi_raw:  unmatched, all running vs all stationary
        running_corr:    Pearson correlation of response with running speed
        n_run_trials:    total number of running trials
        n_stat_trials:   total number of stationary trials
        n_matched_conds: number of stimulus conditions used for matched MI
    """
    X = adata.X  # cells × presentations
    var = adata.var
    
    n_cells = X.shape[0]
    
    # Boolean running mask
    is_running = var[COL_RUNNING].astype(bool).values # outside the function, set COL_RUNNING to the actual column name in the var that indicates running state
    running_speed = var[COL_RUNNING_SPEED].values.astype(float) # outside the function, set COL_RUNNING_SPEED to the actual column name in the var that indicates running speed
    
    # --- Raw (unmatched) MI ---
    mean_run = X[:, is_running].mean(axis=1) if is_running.sum() > 0 else np.zeros(n_cells)
    mean_stat = X[:, ~is_running].mean(axis=1) if (~is_running).sum() > 0 else np.zeros(n_cells)
    
    # Handle array types (could be matrix or ndarray)
    mean_run = np.asarray(mean_run).flatten()
    mean_stat = np.asarray(mean_stat).flatten()
    
    eps = 1e-6 # small constant to avoid division by zero
    mi_raw = (mean_run - mean_stat) / (np.abs(mean_run) + np.abs(mean_stat) + eps)
    
    # --- Correlation with running speed ---
    run_corr = np.array([
        stats.pearsonr(X[i, :].flatten(), running_speed)[0] 
        if np.std(X[i, :].flatten()) > 0 else 0.0
        for i in range(n_cells)
    ]) # run_corr dims: (n_cells,)
    
    # --- Stimulus-matched MI ---
    # Define stimulus conditions
    # Group by the relevant stimulus parameters (contrast, TF, orientation)
    # outside the function, set COL_CONTRAST, COL_TF, COL_ORI to the actual column names in the var that indicate these stimulus parameters
    conditions = var.groupby([COL_CONTRAST, COL_TF, COL_ORI]).groups
    
    # Initialize arrays to hold matched differences and sums for each condition
    matched_diffs = np.full((n_cells, len(conditions)), np.nan)
    matched_sums = np.full((n_cells, len(conditions)), np.nan)
    
    for ci, (cond, idx) in enumerate(conditions.items()):
        idx_arr = np.array([var.index.get_loc(i) for i in idx])
        run_in_cond = is_running[idx_arr]
        
        n_run = run_in_cond.sum()
        n_stat = (~run_in_cond).sum()
        
        if n_run >= min_trials_per_condition and n_stat >= min_trials_per_condition:
            r_run = np.asarray(X[:, idx_arr[run_in_cond]].mean(axis=1)).flatten()
            r_stat = np.asarray(X[:, idx_arr[~run_in_cond]].mean(axis=1)).flatten()
            matched_diffs[:, ci] = r_run - r_stat # directional 
            matched_sums[:, ci] = np.abs(r_run) + np.abs(r_stat)
    
    # Average across valid conditions
    with np.errstate(invalid='ignore'):
        mi_matched = np.nanmean(matched_diffs, axis=1) / (np.nanmean(matched_sums, axis=1) + eps)
    n_matched = np.sum(~np.isnan(matched_diffs), axis=1)
    
    result = pd.DataFrame({
        'running_mi': mi_matched,
        'running_mi_raw': mi_raw,
        'running_corr': run_corr,
        'n_run_trials': int(is_running.sum()),
        'n_stat_trials': int((~is_running).sum()),
        'n_matched_conds': n_matched,
    }, index=adata.obs_names)
    
    return result

# %%
# =============================================================================
# NESTED PERMUTATION TEST aka "omnibus" test for all individual neuron metrics (e.g., running MI) across the hierarchy of labels (subclass, supertype).
# =============================================================================
#
# Test statistic: sum of squared between-group means (weighted by n)
# Permutation: shuffle labels within the parent group

def nested_permutation_test(values, labels_hierarchy, n_perms=10000, seed=2026):
    """
    Nested permutation test through a hierarchy of labels.
    
    Parameters
    ----------
    values : array, shape (n_cells,)
        The metric to test (e.g., running MI)
    labels_hierarchy : list of array-like
        Each element is a label array at increasing resolution.
        e.g., [subclass_labels, supertype_labels]
    n_perms : int
    seed : int
    
    Returns
    -------
    results : list of dicts with keys:
        'level', 'observed_f', 'p_value', 'null_mean', 'null_std', 'null_distribution'
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values) 
    values = values[valid]
    labels_hierarchy = [np.asarray(l)[valid] for l in labels_hierarchy] # filter labels to valid cells
    
    results = []
    
    # Loop through each level of the hierarchy:
    for level_idx in range(len(labels_hierarchy)):
        current_labels = labels_hierarchy[level_idx]
        
        if level_idx == 0:
            # Top level: permute globally
            parent_labels = np.zeros(len(values), dtype=int)
        else:
            parent_labels = labels_hierarchy[level_idx - 1]
        
        # Observed test statistic: between-group sum of squares within each parent
        def compute_stat(vals, cur_labs, par_labs):
            '''
            Compute the sum of squared between-group means (weighted by n) across the current level, within each parent group.
            Used for both observed and null statistics.
            '''
            ss = 0.0
            for parent in np.unique(par_labs):
                pmask = par_labs == parent
                sub_vals = vals[pmask]
                sub_labs = cur_labs[pmask]
                grand_mean = sub_vals.mean()
                for grp in np.unique(sub_labs):
                    gmask = sub_labs == grp
                    n_g = gmask.sum()
                    if n_g > 0:
                        ss += n_g * (sub_vals[gmask].mean() - grand_mean) ** 2
            return ss
        
        obs_stat = compute_stat(values, current_labels, parent_labels)
        
        # Permutation: shuffle current labels within each parent group
        null_stats = np.zeros(n_perms)
        for pi in range(n_perms):
            shuffled = current_labels.copy()
            for parent in np.unique(parent_labels):
                pmask = parent_labels == parent
                idx = np.where(pmask)[0]
                shuffled[idx] = rng.permutation(shuffled[idx])
            null_stats[pi] = compute_stat(values, shuffled, parent_labels)
        
        p_value = (np.sum(null_stats >= obs_stat) + 1) / (n_perms + 1) # add 1 for observed stat to avoid zero 

        results.append({
            'level': level_idx,
            'observed_stat': obs_stat,
            'p_value': p_value,
            'null_mean': null_stats.mean(),
            'null_std': null_stats.std(),
            'null_distribution': null_stats,
        })
    
    return results



def L2_glm_optimized(X, Y, alphas, n_bootstrap):
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]

    # --- Case 1: Standard OLS (alphas == 0) ---
    # Handle scalar 0 or list [0]
    if np.all(alphas == 0):
        # Use lstsq for stability instead of explicit inverse
        W_best = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ W_best
        Alpha_best = np.zeros(n_targets)

        # Vectorized VAF calculation
        VAF_train = 1 - np.var(Y - Y_hat, axis=0) / np.var(Y, axis=0)
        return Y_hat, W_best, Alpha_best, VAF_train, 0, 0

    # --- Case 2: Ridge Regression (Loop Optimization) ---
    alphas = np.array(alphas)
    n_alphas = len(alphas)
    fold_size = int(n_samples / n_bootstrap)

    # Pre-allocate output tensors
    # Shapes follow the logic: (Targets, Alphas, Bootstraps)
    VAF_train = np.zeros((n_targets, n_alphas, n_bootstrap))
    VAF_test = np.zeros((n_targets, n_alphas, n_bootstrap))
    W_all = np.zeros((n_features, n_targets, n_alphas, n_bootstrap))

    # Precompute Global Covariance Matrices (The "Covariance Subtraction" trick)
    XtX_global = X.T @ X
    XtY_global = X.T @ Y

    # Iterate over Bootstraps (Outer Loop)
    # We flipped the loops: Bootstrap > Alpha. This allows reusing the decomposition.
    for i_bootstrap in tqdm(range(n_bootstrap), desc='Bootstrapping'):
        
        # 1. Define Indices
        test_start = i_bootstrap * fold_size
        test_inds = np.arange(test_start, test_start + fold_size)
        eval_start = (i_bootstrap + 1) * fold_size % n_samples
        eval_inds = (eval_start + np.arange(fold_size)) % n_samples
        
        # Combine indices to remove (Test + Eval) to get Train
        inds_remove = np.concatenate((test_inds, eval_inds))
        X_remove = X[inds_remove]
        Y_remove = Y[inds_remove]

        # 2. Efficiently update XtX and XtY for training set
        # XtX_train = XtX_global - XtX_removed
        XtX_remove = X_remove.T @ X_remove
        XtY_remove = X_remove.T @ Y_remove
        
        XtX_train = XtX_global - XtX_remove
        XtY_train = XtY_global - XtY_remove

        # 3. Eigen Decomposition (The "Solver" Optimization)
        # Decompose once, solve for all alphas instantly
        eigvals, eigvecs = np.linalg.eigh(XtX_train)
        
        # Project XtY onto eigenbasis: Z = V.T @ XtY
        Z = eigvecs.T @ XtY_train  # Shape: (Features, Targets)

        # 4. Solve for all Alphas simultaneously via Broadcasting
        # Formula: W = V @ (1 / (eigvals + alpha)) @ Z
        # diag_inv shape: (Alphas, Features)
        diag_inv = 1.0 / (alphas[:, None] + eigvals[None, :])
        
        # Scale Z by the inverse eigenvalues for each alpha
        # Scaled Z shape: (Alphas, Features, Targets)
        scaled_Z = diag_inv[:, :, None] * Z[None, :, :]
        
        # Project back to original basis to get W
        # einsum: f=features(evecs), k=latent, a=alphas, t=targets
        # W_alphas shape: (Alphas, Features, Targets)
        W_alphas = np.einsum('fk, akt -> aft', eigvecs, scaled_Z)
        
        # Store W (transpose to match expected shape: F, T, A)
        W_all[..., i_bootstrap] = W_alphas.transpose(1, 2, 0)

        # 5. Vectorized Prediction and Evaluation
        # Reconstruct X_train using mask (faster than setdiff1d)
        mask_train = np.ones(n_samples, dtype=bool)
        mask_train[inds_remove] = False
        X_train_fold = X[mask_train]
        Y_train_fold = Y[mask_train]
        X_test_fold = X[test_inds]
        Y_test_fold = Y[test_inds]

        # Predict Train: (N_train, F) @ (A, F, T) -> (N_train, A, T)
        Y_hat_train = np.einsum('nf, aft -> nat', X_train_fold, W_alphas)
        # Predict Test
        Y_hat_test = np.einsum('nf, aft -> nat', X_test_fold, W_alphas)

        # Calculate VAF (Vectorized)
        # Train
        res_train = Y_train_fold[:, None, :] - Y_hat_train
        vaf_train_fold = 1 - np.var(res_train, axis=0) / np.var(Y_train_fold[:, None, :], axis=0)
        VAF_train[..., i_bootstrap] = vaf_train_fold.T # Transpose to (Targets, Alphas)

        # Test
        res_test = Y_test_fold[:, None, :] - Y_hat_test
        vaf_test_fold = 1 - np.var(res_test, axis=0) / np.var(Y_test_fold[:, None, :], axis=0)
        VAF_test[..., i_bootstrap] = vaf_test_fold.T

    # --- Final Selection & Evaluation ---
    # Aggregate statistics
    mu_test = np.mean(VAF_test, axis=2)
    sd_test = np.std(VAF_test, axis=2)
    i_alpha_best = np.argmax(mu_test, axis=1)
    
    Alpha_best = np.zeros(n_targets)
    W_best = np.zeros((n_features, n_targets))
    VAF_eval = np.zeros(n_targets)

    # Reconstruct the indices for the FINAL fold (to match original logic)
    # The original code evaluated on the eval set of the *last* bootstrap iteration.
    last_boot_idx = n_bootstrap - 1
    eval_start = (last_boot_idx + 1) * fold_size % n_samples
    eval_inds = (eval_start + np.arange(fold_size)) % n_samples
    
    # Train set for final eval is everything EXCEPT eval indices
    mask_eval = np.ones(n_samples, dtype=bool)
    mask_eval[eval_inds] = False
    X_tt = X[mask_eval] # "Test + Train" combined
    Y_tt = Y[mask_eval]
    X_eval = X[eval_inds]
    Y_eval = Y[eval_inds]

    # Precompute covariance for final fit
    XtX_tt = X_tt.T @ X_tt

    for i in range(n_targets):
        # 1-Standard-Error Rule
        thresh = mu_test[i, i_alpha_best[i]] - sd_test[i, i_alpha_best[i]]
        candidates = np.where(mu_test[i] >= thresh)[0]
        best_idx_corrected = candidates[-1] # Take the largest alpha in range
        
        Alpha_best[i] = alphas[best_idx_corrected]
        
        # Final Solve (Single solve, so standard solver is fine)
        reg_mat = XtX_tt + Alpha_best[i] * np.eye(n_features)
        rhs = X_tt.T @ Y_tt[:, i]
        W_best[:, i] = np.linalg.solve(reg_mat, rhs)
        
        # Eval Score
        y_hat_val = X_eval @ W_best[:, i]
        VAF_eval[i] = 1 - np.var(Y_eval[:, i] - y_hat_val) / np.var(Y_eval[:, i])

    # Final Full Prediction
    Y_hat = X @ W_best
    VAF_total = 1 - np.var(Y - Y_hat,axis=0) / np.var(Y,axis=0)
    return Y_hat, W_best, Alpha_best, VAF_total, VAF_train, VAF_test, VAF_eval