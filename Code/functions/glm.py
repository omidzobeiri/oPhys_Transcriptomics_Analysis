"""GLM coefficient aggregation and zarr I/O."""

import re
import numpy as np

orientations = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
contrasts = np.array([0.05, 0.1, 0.2, 0.4, 0.8], dtype=float)
TFs = np.array([1, 2, 4, 8, 15], dtype=float)
directions = orientations.copy()

COEF_RE = re.compile(
    r'coef_block_(?P<block>[^_]+)_TF_(?P<TF>[^_]+)_contrast_(?P<contrast>[^_]+)_direction_(?P<direction>.+)'
)


def _parse_coef_key(key):
    m = COEF_RE.match(key)
    if not m:
        return None
    return {k: float(v) for k, v in m.groupdict().items()}


def _write_array(grp, name, data):
    if name in grp:
        grp[name][:] = data
    else:
        grp.create_dataset(name, data=data, shape=data.shape, dtype=data.dtype, overwrite=True)


def _pref_remap(raw_aggs, pref_idx, n_cells, n_tp):
    """Re-index direction-keyed aggregates relative to preferred direction."""
    result = {}
    for i_slot, d_slot in enumerate(directions):
        pref_arr = np.zeros((n_cells, n_tp), dtype=np.float32)
        for ci in range(n_cells):
            shifted_dirs = np.roll(directions, shift=-int(pref_idx[ci]))
            actual_dir = shifted_dirs[i_slot]
            if actual_dir in raw_aggs:
                pref_arr[ci] = raw_aggs[actual_dir][ci]
        result[d_slot] = pref_arr
    return result


def add_glm_aggregate_columns(glm_group):
    """Read condition-specific coef arrays, compute aggregates, and write them back.

    Written under glm/coef/:
      Single-variable marginals:
        glm-coef_TF_{t}                          (n_cells, 30)
        glm-coef_contrast_{c}                    (n_cells, 30)
        glm-coef_direction_{d}                   (n_cells, 30)
        glm-coef_direction_pref_{d}              (n_cells, 30)
      Joint marginals:
        glm-coef_TF_{t}_direction_{d}            (n_cells, 30)
        glm-coef_TF_{t}_direction_pref_{d}       (n_cells, 30)
        glm-coef_contrast_{c}_direction_{d}      (n_cells, 30)
        glm-coef_contrast_{c}_direction_pref_{d} (n_cells, 30)
      Preferred direction:
        preferred_direction                      (n_cells,)
    """
    coef_grp = glm_group['coef']
    all_keys = sorted(coef_grp.keys())

    parsed = {}
    for k in all_keys:
        p = _parse_coef_key(k)
        if p is not None and not np.isnan(p['direction']):
            parsed[k] = p

    if len(parsed) == 0:
        return 0

    coef_data = {}
    n_cells = None
    n_tp = 30
    for k, meta in parsed.items():
        arr = coef_grp[k][:]
        if arr.shape[1] != n_tp:
            continue
        coef_data[k] = (meta, arr)
        if n_cells is None:
            n_cells = arr.shape[0]

    n_written = 0

    # TF aggregates
    for tf_val in TFs:
        matching = [arr for meta, arr in coef_data.values()
                    if not np.isnan(meta['TF']) and meta['TF'] == tf_val]
        if matching:
            agg = np.nanmean(np.stack(matching), axis=0)
            _write_array(coef_grp, f'glm-coef_TF_{tf_val:g}', agg)
            n_written += 1

    # Contrast aggregates
    for c_val in contrasts:
        matching = [arr for meta, arr in coef_data.values()
                    if not np.isnan(meta['contrast']) and meta['contrast'] == c_val]
        if matching:
            agg = np.nanmean(np.stack(matching), axis=0)
            _write_array(coef_grp, f'glm-coef_contrast_{c_val:g}', agg)
            n_written += 1

    # Direction aggregates
    dir_aggs = {}
    for d_val in directions:
        matching = [arr for meta, arr in coef_data.values()
                    if not np.isnan(meta['direction']) and meta['direction'] == d_val]
        if matching:
            agg = np.nanmean(np.stack(matching), axis=0)
            dir_aggs[d_val] = agg
            _write_array(coef_grp, f'glm-coef_direction_{d_val:g}', agg)
            n_written += 1

    # Preferred direction (max |mean coef| across timepoints)
    pref_idx = None
    if dir_aggs:
        dir_matrix = np.stack([dir_aggs[d] for d in directions])
        dir_mean = np.nanmean(dir_matrix, axis=2)
        pref_idx = np.argmax(np.abs(dir_mean), axis=0)
        pref_dir = directions[pref_idx]

        _write_array(coef_grp, 'preferred_direction', pref_dir)
        n_written += 1

        for d_slot, pref_arr in _pref_remap(dir_aggs, pref_idx, n_cells, n_tp).items():
            _write_array(coef_grp, f'glm-coef_direction_pref_{d_slot:g}', pref_arr)
            n_written += 1

    # TF x Direction joint aggregates
    for tf_val in TFs:
        td_aggs = {}
        for d_val in directions:
            matching = [arr for meta, arr in coef_data.values()
                        if meta['TF'] == tf_val and meta['direction'] == d_val]
            if matching:
                agg = np.nanmean(np.stack(matching), axis=0)
                td_aggs[d_val] = agg
                _write_array(coef_grp, f'glm-coef_TF_{tf_val:g}_direction_{d_val:g}', agg)
                n_written += 1

        if td_aggs and pref_idx is not None:
            for d_slot, pref_arr in _pref_remap(td_aggs, pref_idx, n_cells, n_tp).items():
                _write_array(coef_grp, f'glm-coef_TF_{tf_val:g}_direction_pref_{d_slot:g}', pref_arr)
                n_written += 1

    # Contrast x Direction joint aggregates
    for c_val in contrasts:
        cd_aggs = {}
        for d_val in directions:
            matching = [arr for meta, arr in coef_data.values()
                        if meta['contrast'] == c_val and meta['direction'] == d_val]
            if matching:
                agg = np.nanmean(np.stack(matching), axis=0)
                cd_aggs[d_val] = agg
                _write_array(coef_grp, f'glm-coef_contrast_{c_val:g}_direction_{d_val:g}', agg)
                n_written += 1

        if cd_aggs and pref_idx is not None:
            for d_slot, pref_arr in _pref_remap(cd_aggs, pref_idx, n_cells, n_tp).items():
                _write_array(coef_grp, f'glm-coef_contrast_{c_val:g}_direction_pref_{d_slot:g}', pref_arr)
                n_written += 1

    return n_written
