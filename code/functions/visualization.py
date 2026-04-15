"""
Visualization utilities for oPhys_Transcriptomics analysis.

Functions for plotting tuning curves, population responses, correlation matrices,
and spatial distributions across cell types and brain regions.
"""

from curses import noecho
# from tkinter import N
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import seaborn as sns
from matplotlib.gridspec import GridSpec


def polar_bar_plot(data, ax = None, radii = None, theta = None, vmin=None, vmax=None, cmap = 'viridis'):
    """
    Plot polar bar plot for tuning data (e.g., orientation tuning).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes to plot on
    data : ndarray
        Tuning data, shape (n_radial, n_angular)
    radii : ndarray, optional
        Radial bin edges
    theta : ndarray, optional
        Angular bin edges in radians
    
    Returns
    -------
    sm : matplotlib.cm.ScalarMappable
        Colormap mapper for colorbar
    """
    if radii is None:
        radii = np.arange(data.shape[0] + 1)
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, data.shape[1] + 1)
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    dr = np.diff(radii)
    dtheta = np.diff(theta)[0]
    
    norm = colors.Normalize(vmin=vmin if vmin is not None else np.nanmin(data),
                            vmax=vmax if vmax is not None else np.nanmax(data))
    cmap = cm.get_cmap(cmap)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.bar(
                theta[j],
                dr[i],
                width=dtheta,
                bottom=radii[i],
                align='edge',
                color=cmap(norm(data[i, j])),
                edgecolor='none'
            )

    ax.set_xticks(theta[:-1])  # 8 direction ticks``
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def plot_tuning_curve(tuning_data, stimulus_param, cell_type=None, ax=None, 
                      error_bars=True, title=None):
    """
    Plot tuning curve for a single stimulus dimension.
    
    Parameters
    ----------
    tuning_data : ndarray
        Mean responses, shape (n_stim,) or (n_stim, n_cells)
    stimulus_param : ndarray or list
        Stimulus values (e.g., orientations, TFs, contrasts)
    cell_type : str, optional
        Cell type label for legend
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; creates new if None
    error_bars : bool
        Whether to plot error bars (if tuning_data is 2D, uses std across cells)
    title : str, optional
        Plot title
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    if tuning_data.ndim == 1:
        ax.plot(stimulus_param, tuning_data, marker='o', linewidth=2, 
                markersize=6, label=cell_type or 'Tuning')
    else:
        mean_resp = tuning_data.mean(axis=1)
        std_resp = tuning_data.std(axis=1)
        ax.errorbar(stimulus_param, mean_resp, yerr=std_resp if error_bars else None,
                   marker='o', linewidth=2, markersize=6, capsize=5, label=cell_type or 'Tuning')
    
    ax.set_xlabel('Stimulus Parameter')
    ax.set_ylabel('Response (ΔF/F)')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_population_heatmap(response_matrix, cell_labels=None, stim_labels=None,
                            cmap='RdBu_r', ax=None, title=None):
    """
    Plot heatmap of population responses across conditions.
    
    Parameters
    ----------
    response_matrix : ndarray
        Population response matrix, shape (n_cells, n_conditions)
    cell_labels : list, optional
        Cell identifiers (e.g., cell IDs, types)
    stim_labels : list, optional
        Stimulus condition labels
    cmap : str
        Colormap name
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(response_matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    
    if stim_labels:
        ax.set_xticks(np.arange(len(stim_labels)))
        ax.set_xticklabels(stim_labels, rotation=45, ha='right')
    ax.set_ylabel('Cell')
    ax.set_xlabel('Stimulus Condition')
    if title:
        ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Response (ΔF/F)')
    
    return ax


def plot_cell_type_comparison(data_dict, stimulus_param, stim_name='Stimulus',
                              figsize=(12, 5)):
    """
    Compare tuning curves across cell types.
    
    Parameters
    ----------
    data_dict : dict
        Keys are cell type names, values are response arrays (n_stim,) or (n_stim, n_cells)
    stimulus_param : ndarray or list
        Stimulus values
    stim_name : str
        Stimulus dimension name (e.g., 'Orientation', 'Temporal Frequency')
    figsize : tuple
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    """
    fig, axes = plt.subplots(1, len(data_dict), figsize=figsize)
    if len(data_dict) == 1:
        axes = [axes]
    
    sns.set_context('talk')
    for ax, (cell_type, response_data) in zip(axes, data_dict.items()):
        plot_tuning_curve(response_data, stimulus_param, cell_type=cell_type,
                         ax=ax, title=cell_type)
        ax.set_xlabel(stim_name)
    
    fig.tight_layout()
    
    return fig, axes


def plot_correlation_matrix(corr_matrix, cell_labels=None, cmap='coolwarm',
                            vmin=-1, vmax=1, ax=None, title=None):
    """
    Plot correlation matrix with dendrogram or hierarchical ordering.
    
    Parameters
    ----------
    corr_matrix : ndarray
        Correlation matrix, shape (n_cells, n_cells)
    cell_labels : list, optional
        Cell identifiers
    cmap : str
    vmin, vmax : float
        Colorbar limits
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    if cell_labels:
        ax.set_xticks(np.arange(len(cell_labels)))
        ax.set_yticks(np.arange(len(cell_labels)))
        ax.set_xticklabels(cell_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(cell_labels, fontsize=8)
    
    ax.set_xlabel('Cell')
    ax.set_ylabel('Cell')
    if title:
        ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    
    return ax


def plot_response_distribution(responses, cell_types=None, ax=None, title=None):
    """
    Plot violin/box plots of response distributions by cell type.
    
    Parameters
    ----------
    responses : ndarray or list of ndarray
        Response values, shape (n_cells,) or list of (n_cells_per_type,)
    cell_types : list, optional
        Cell type labels
    ax : matplotlib.axes.Axes, optional
    title : str, optional
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if isinstance(responses, list):
        data_to_plot = responses
        labels = cell_types or [f'Type {i}' for i in range(len(responses))]
    else:
        data_to_plot = [responses]
        labels = cell_types or ['All']
    
    parts = ax.violinplot(data_to_plot, positions=np.arange(len(data_to_plot)),
                          showmeans=True, showmedians=True)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Response (ΔF/F)')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_trial_responses(responses, time_vector, trial_info=None, ax=None,
                        cmap='viridis', title=None):
    """
    Plot single-trial responses over time.
    
    Parameters
    ----------
    responses : ndarray
        Trial responses, shape (n_trials, n_timepoints) or (n_trials, n_timepoints, n_cells)
    time_vector : ndarray
        Time points, shape (n_timepoints,)
    trial_info : pd.DataFrame, optional
        Trial metadata for coloring by condition
    ax : matplotlib.axes.Axes, optional
    cmap : str
    title : str, optional
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if responses.ndim == 2:
        for trial_idx, trial_resp in enumerate(responses):
            ax.plot(time_vector, trial_resp, alpha=0.5, linewidth=0.8)
    else:
        mean_resp = responses.mean(axis=(0, 2))
        ax.plot(time_vector, mean_resp, linewidth=2, color='black', label='Mean')
        ax.fill_between(time_vector, 
                        mean_resp - responses.std(axis=(0, 2)),
                        mean_resp + responses.std(axis=(0, 2)),
                        alpha=0.3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Response (ΔF/F)')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax