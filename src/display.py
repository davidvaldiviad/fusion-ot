import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from .utils import idx_at_value

def new_axes(n_rows=1, n_cols=1, *, figsize=None):
    """
    Return axes object.

    Args:
        n_rows (int): number of rows.
        n_cols (int): number of columns.
        figsize ([width, height]): size of display. If none is given then it is proportional to n_rows and n_cols.

    Returns:
        ax (matplotlib.Axes): Axes object for further handling.
    """
    if figsize is None:
        figsize=[n_cols * 6, n_rows * 4]

    _, ax = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=figsize)

    return ax

def plot_signal(signal, *, ax=None, times=None, title=None):
    """
    Plot a time signal.

    Args:
        signal (np.ndarray): signal to plot.
        times  (np.ndarray): timestamps.

    """
    if ax is None:
        ax = new_axes()

    if times is not None:
        ax.plot(times, signal, c='black', linewidth=.4)
    else:
        ax.plot(signal, c='black', linewidth=.4)
        ax.set_xticks([])

    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    ax.set_title(title)

def display_spectrogram(spec, 
                        *, 
                        ax=None, 
                        title=None, 
                        f_bins=None, 
                        t_frames=None,
                        low_f=None,
                        high_f=None,
                        low_t=None,
                        high_t=None,
                        log=False,
                        logmin=-80):
    """
    Display spectrogram.

    Args:
        spec (np.ndarray): Spectrogram.
        ax (matplotlib.Axes): Axes object in which to display spectrogram.
        title (string): title for plot.
        f_bins (np.ndarray): frequency bins of spectrogram (see eq. 7).
        t_frames (np.ndarray): time frames of spectrogram (see eq. 8)
        low_f (double): Remove frequencies below low_f (in Hz).
        high_f (double): Remove frequencies above high_f (in Hz).
        low_t (double): Remove frames below low_t (in s).
        high_t (double): Remove frames above high_t (in s).
        log  (bool): Display spectrogram in log/decibel scale.
        logmin (int): Minimum value for scale display if log scale (in dB)
    """

    if ax is None:
        ax = new_axes()

    extent = None

    has_axis = f_bins is not None and t_frames is not None # can only filter if frequency/time axis values given.
    if has_axis:
        if low_f:
            idx = idx_at_value(low_f, f_bins)
            f_bins = f_bins[idx:]
            spec = spec[idx:, :]
        if high_f:
            idx = idx_at_value(high_f, f_bins)
            f_bins = f_bins[:idx]
            spec = spec[:idx, :]
        if low_t:
            idx = idx_at_value(low_t, t_frames)
            t_frames = t_frames[idx:]
            spec = spec[:, idx:]
        if high_t:
            idx = idx_at_value(high_t, t_frames)
            t_frames = t_frames[:idx]
            spec = spec[:, :idx]
        
        extent = [t_frames[0], t_frames[-1], f_bins[0], f_bins[-1]]

    if log:
        spec = 10 * np.log10(spec + 1e-30)
        spec = np.maximum(spec, spec.max() + logmin)

    ax.imshow(spec, origin='lower', aspect='auto', cmap='magma', interpolation='none', extent=extent)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)

def display_support(supp, *, ax=None, title=None, xlabel='Time (s)', ylabel='Frequency (Hz)', add_point=False, point_pos=(1, 1), point_label=''):
    """
        Display time-frequency grid.

        Args:
            supp (np.ndarray (n_pts, n_dim)): support as array. Each item supp[i] corresponds to a point x = (x1, ..., xn).
            ax (matplotlib.Axes): Axes object in which to display spectrogram.
            title (string): title for plot.
            xlabel (string): x axis label.
            ylabel (string): y axis label.
    
    """
    if ax is None:
        ax = new_axes()

    t_frames = np.unique(supp[:, 0])
    f_bins  = np.unique(supp[:, 1])

    ax.scatter(supp[:, 0], supp[:, 1], c='black', marker='.') # display support points

    for f in f_bins:
        for t in t_frames:
            ax.axhline(f, c='black', linewidth=.5,  alpha=.5) # add vertical lines
            ax.axvline(t, c='black', linewidth=.5,  alpha=.5) # add horizontal lines
            ax.set_yticks(f_bins)
            ax.set_xticks(t_frames)
            ax.set_title(title)
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for spine in ax.spines.values(): # remove axis border
        spine.set_visible(False)

    ax.xaxis.set_tick_params(width=0) # remove extra ticks
    ax.yaxis.set_tick_params(width=0) # remove extra ticks

    if add_point:
        ax.text(
            point_pos[0], point_pos[1], point_label,
            transform=ax.transAxes,
            ha="left", va="top",
        )
    