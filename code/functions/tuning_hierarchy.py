import numpy as np
import pandas as pd
from scipy.stats import kruskal


def _first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _to_numpy(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    valid = np.isfinite(pvals)
    if valid.sum() == 0:
        return out

    idx = np.where(valid)[0]
    pv = pvals[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    back = np.empty_like(q)
    back[order] = q
    out[idx] = back
    return out


def _compute_cell_tuning(X, var_df):
    ori_col = _first_existing(var_df.columns, ["orientation", "ori", "ori_deg", "orientation_deg"])
    con_col = _first_existing(var_df.columns, ["contrast", "stim_contrast"])
    tf_col = _first_existing(var_df.columns, ["temporal_frequency", "tf", "TF"])

    n_cells = X.shape[0]
    out = pd.DataFrame(index=np.arange(n_cells))

    out["mean_response"] = np.nanmean(X, axis=1)
    out["response_std"] = np.nanstd(X, axis=1)

    if ori_col is not None:
        ori_vals = np.sort(pd.unique(var_df[ori_col].dropna()))
        if len(ori_vals) > 0:
            tuning = np.full((n_cells, len(ori_vals)), np.nan)
            for j, ori in enumerate(ori_vals):
                idx = np.where(var_df[ori_col].values == ori)[0]
                if len(idx) > 0:
                    tuning[:, j] = np.nanmean(X[:, idx], axis=1)

            pref_idx = np.nanargmax(np.where(np.isnan(tuning), -np.inf, tuning), axis=1)
            out["pref_orientation"] = ori_vals[pref_idx]

            num = np.nansum(tuning * np.exp(2j * np.deg2rad(ori_vals))[None, :], axis=1)
            den = np.nansum(tuning, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["osi"] = np.abs(num) / den

    # Determine per-cell preferred orientation for conditioning contrast/TF tuning
    have_pref_ori = ori_col is not None and "pref_orientation" in out.columns
    pref_ori_vals = out["pref_orientation"].values if have_pref_ori else None

    if con_col is not None:
        con_vals = np.sort(pd.unique(var_df[con_col].dropna()))
        if len(con_vals) > 0:
            if have_pref_ori and ori_col is not None:
                # Build (n_cells, n_oris, n_contrasts) tensor, then select at pref ori
                ori_vals_con = np.sort(pd.unique(var_df[ori_col].dropna()))
                full_con = np.full((n_cells, len(ori_vals_con), len(con_vals)), np.nan)
                for oi, ori in enumerate(ori_vals_con):
                    for ci, c in enumerate(con_vals):
                        idx = np.where((var_df[ori_col].values == ori) & (var_df[con_col].values == c))[0]
                        if len(idx) > 0:
                            full_con[:, oi, ci] = np.nanmean(X[:, idx], axis=1)
                # Select each cell's preferred orientation slice
                ori_idx = np.array([np.argmin(np.abs(ori_vals_con - po)) for po in pref_ori_vals])
                con_tuning = full_con[np.arange(n_cells), ori_idx, :]
            else:
                con_tuning = np.full((n_cells, len(con_vals)), np.nan)
                for j, c in enumerate(con_vals):
                    idx = np.where(var_df[con_col].values == c)[0]
                    if len(idx) > 0:
                        con_tuning[:, j] = np.nanmean(X[:, idx], axis=1)

            pref_idx = np.nanargmax(np.where(np.isnan(con_tuning), -np.inf, con_tuning), axis=1)
            out["pref_contrast"] = con_vals[pref_idx]

            slope = np.full(n_cells, np.nan)
            x = np.log2(np.maximum(con_vals.astype(float), 1e-6))
            for i in range(n_cells):
                y = con_tuning[i, :]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() >= 2:
                    slope[i] = np.polyfit(x[valid], y[valid], 1)[0]
            out["contrast_slope"] = slope

            mx = np.nanmax(con_tuning, axis=1)
            mn = np.nanmin(con_tuning, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["contrast_selectivity"] = (mx - mn) / (mx + mn)

    if tf_col is not None:
        tf_vals = np.sort(pd.unique(var_df[tf_col].dropna()))
        if len(tf_vals) > 0:
            if have_pref_ori and ori_col is not None:
                # Build (n_cells, n_oris, n_tfs) tensor, then select at pref ori
                ori_vals_tf = np.sort(pd.unique(var_df[ori_col].dropna()))
                full_tf = np.full((n_cells, len(ori_vals_tf), len(tf_vals)), np.nan)
                for oi, ori in enumerate(ori_vals_tf):
                    for ti, tf in enumerate(tf_vals):
                        idx = np.where((var_df[ori_col].values == ori) & (var_df[tf_col].values == tf))[0]
                        if len(idx) > 0:
                            full_tf[:, oi, ti] = np.nanmean(X[:, idx], axis=1)
                ori_idx = np.array([np.argmin(np.abs(ori_vals_tf - po)) for po in pref_ori_vals])
                tf_tuning = full_tf[np.arange(n_cells), ori_idx, :]
            else:
                tf_tuning = np.full((n_cells, len(tf_vals)), np.nan)
                for j, tf in enumerate(tf_vals):
                    idx = np.where(var_df[tf_col].values == tf)[0]
                    if len(idx) > 0:
                        tf_tuning[:, j] = np.nanmean(X[:, idx], axis=1)

            pref_idx = np.nanargmax(np.where(np.isnan(tf_tuning), -np.inf, tf_tuning), axis=1)
            out["pref_tf"] = tf_vals[pref_idx]

            mx = np.nanmax(tf_tuning, axis=1)
            mn = np.nanmin(tf_tuning, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["tf_selectivity"] = (mx - mn) / (mx + mn)

            tf_slope = np.full(n_cells, np.nan)
            x_tf = np.log2(np.maximum(tf_vals.astype(float), 1e-6))
            for i in range(n_cells):
                y = tf_tuning[i, :]
                valid = np.isfinite(x_tf) & np.isfinite(y)
                if valid.sum() >= 2:
                    tf_slope[i] = np.polyfit(x_tf[valid], y[valid], 1)[0]
            out["tf_slope"] = tf_slope

    return out


def compute_tuning_properties_by_session(adata, mouse_id=None):
    obs = adata.obs.copy().reset_index(drop=False)
    var = adata.var.copy().reset_index(drop=False)
    X = _to_numpy(adata.X)

    session_col = _first_existing(var.columns, ["session", "session_name", "ophys_session", "stim_session"])
    if session_col is None:
        session_values = ["all"]
        session_indexer = {"all": np.arange(var.shape[0])}
    else:
        session_values = [v for v in pd.unique(var[session_col]) if pd.notna(v)]
        session_indexer = {s: np.where(var[session_col].values == s)[0] for s in session_values}

    subclass_col = _first_existing(obs.columns, ["subclass", "subclass_name", "subclass_label"])
    supertype_col = _first_existing(obs.columns, ["supertype", "supertype_name", "supertype_label"])
    cluster_col = _first_existing(obs.columns, ["cluster", "cluster_name", "cluster_label"])
    uid_col = _first_existing(obs.columns, ["unique_id", "cell_id", "index"])

    keep_cols = [c for c in [uid_col, subclass_col, supertype_col, cluster_col] if c is not None]
    meta = obs[keep_cols].copy() if len(keep_cols) else pd.DataFrame(index=np.arange(obs.shape[0]))
    rename_map = {}
    if uid_col is not None:
        rename_map[uid_col] = "cell_id"
    if subclass_col is not None:
        rename_map[subclass_col] = "subclass"
    if supertype_col is not None:
        rename_map[supertype_col] = "supertype"
    if cluster_col is not None:
        rename_map[cluster_col] = "cluster"
    meta = meta.rename(columns=rename_map)

    all_parts = []
    for sess in session_values:
        idx = session_indexer[sess]
        if len(idx) == 0:
            continue
        tdf = _compute_cell_tuning(X[:, idx], var.iloc[idx].reset_index(drop=True))
        merged = pd.concat([meta.reset_index(drop=True), tdf.reset_index(drop=True)], axis=1)
        merged["session"] = str(sess)
        merged["mouse_id"] = str(mouse_id) if mouse_id is not None else "unknown"
        all_parts.append(merged)

    if len(all_parts) == 0:
        return pd.DataFrame()
    return pd.concat(all_parts, ignore_index=True)


def compare_hierarchy_levels(tuning_df, metrics=None, min_cells=10):
    if metrics is None:
        metrics = [
            "mean_response",
            "response_std",
            "osi",
            "pref_contrast",
            "contrast_slope",
            "contrast_selectivity",
            "pref_tf",
            "tf_selectivity",
            "tf_slope",
        ]

    rows = []
    group_keys = ["mouse_id", "session"]

    for (mouse_id, session), sdf in tuning_df.groupby(group_keys, dropna=False):
        available_metrics = [m for m in metrics if m in sdf.columns]

        if "subclass" in sdf.columns:
            for metric in available_metrics:
                tmp = sdf[["subclass", metric]].dropna()
                counts = tmp["subclass"].value_counts()
                good = counts[counts >= min_cells].index
                tmp = tmp[tmp["subclass"].isin(good)]
                if tmp["subclass"].nunique() >= 2:
                    groups = [g[metric].values for _, g in tmp.groupby("subclass")]
                    stat, p = kruskal(*groups)
                    rows.append({
                        "mouse_id": mouse_id,
                        "session": session,
                        "level": "subclass",
                        "parent_group": "all",
                        "metric": metric,
                        "n_cells": len(tmp),
                        "n_groups": tmp["subclass"].nunique(),
                        "statistic": stat,
                        "p_value": p,
                    })

        if "subclass" in sdf.columns and "supertype" in sdf.columns:
            for subclass, subdf in sdf.groupby("subclass", dropna=False):
                for metric in available_metrics:
                    tmp = subdf[["supertype", metric]].dropna()
                    counts = tmp["supertype"].value_counts()
                    good = counts[counts >= min_cells].index
                    tmp = tmp[tmp["supertype"].isin(good)]
                    if tmp["supertype"].nunique() >= 2:
                        groups = [g[metric].values for _, g in tmp.groupby("supertype")]
                        stat, p = kruskal(*groups)
                        rows.append({
                            "mouse_id": mouse_id,
                            "session": session,
                            "level": "supertype_within_subclass",
                            "parent_group": subclass,
                            "metric": metric,
                            "n_cells": len(tmp),
                            "n_groups": tmp["supertype"].nunique(),
                            "statistic": stat,
                            "p_value": p,
                        })

        if "subclass" in sdf.columns and "cluster" in sdf.columns:
            for subclass, subdf in sdf.groupby("subclass", dropna=False):
                for metric in available_metrics:
                    tmp = subdf[["cluster", metric]].dropna()
                    counts = tmp["cluster"].value_counts()
                    good = counts[counts >= min_cells].index
                    tmp = tmp[tmp["cluster"].isin(good)]
                    if tmp["cluster"].nunique() >= 2:
                        groups = [g[metric].values for _, g in tmp.groupby("cluster")]
                        stat, p = kruskal(*groups)
                        rows.append({
                            "mouse_id": mouse_id,
                            "session": session,
                            "level": "cluster_within_subclass",
                            "parent_group": subclass,
                            "metric": metric,
                            "n_cells": len(tmp),
                            "n_groups": tmp["cluster"].nunique(),
                            "statistic": stat,
                            "p_value": p,
                        })

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    res["q_value"] = np.nan
    for (mouse_id, session, level), idx in res.groupby(["mouse_id", "session", "level"]).groups.items():
        res.loc[idx, "q_value"] = _bh_fdr(res.loc[idx, "p_value"].values)

    res["significant"] = res["q_value"] < 0.05
    return res