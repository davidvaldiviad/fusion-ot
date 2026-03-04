import numpy as np
from scipy.spatial.distance import cdist
from .utils import *

def sq_cost_matrix(S1, S2, *, norm=True):
    """
    Squared Euclidean cost matrix between supports S1 and S2.

    While in the paper we defined the cost using the sample rate (see eq. 20), in practice we used normalized supports (see utils.time_freq_support) and compute cost between normalized indices.

    Args:
        S1 (np.ndarray, (n_pts1, ndim)) : Support 1.
        S2 (np.ndarray, (n_pts2, ndim)) : Support 2.
        norm (bool)                     : Fit matrix values between 0 and 1.

    Returns:
        C (np.ndarray, (n_pts1, n_pts2) : Cost matrix between supports S1 and S2.
    """
    assert S1.ndim == S2.ndim, "Supports must be of same dimension."
    if S1.ndim == 1:
        S1 = np.expand_dims(S1.copy(), 1)
        S2 = np.expand_dims(S2.copy(), 1)

    C = cdist(S1, S2, metric='sqeuclidean')
    
    if norm:
        C = C / C.max()
    
    return C

def cost_matrix_horizontal(S1, S, norm=True):
    """
        Cost matrix that restricts movement to the time axis, i.e. horizontally (see eq. 21).
        Infinite values are set to np.nan

        Args:
            S1 (np.ndarray, (n_pts2, ndim)) : Support 1.
            S  (np.ndarray, (n_pts2, ndim)) : Target support.
            norm (bool)                     : Fit matrix values between 0 and 1.

        Returns:
            C (np.ndarray, (n_pts1, n_pts) : Structured cost matrix between supports S1 and S.
    
    """
    assert S1.ndim == S.ndim, "Supports must be of same dimension."
    if S.ndim == 1:
        S1 = np.expand_dims(S1.copy(), 1)
        S = np.expand_dims(S.copy(), 1)

    t_frames_1, f_bins_1 = np.unique(S1[:, 0]), np.unique(S1[:, 1])
    t_frames_2           = np.unique(S[:, 0])

    M1, N1 = f_bins_1.size, t_frames_1.size
    N2 = t_frames_2.size

    C = np.empty((M1 * N1, M1 * N2))
    C.fill(np.nan)

    base = ((t_frames_1[:, None] - t_frames_2[None, :]) ** 2)
    
    for f in range(M1):
        C[f::M1, f::M1] = base

    if norm:
        C = C / np.nanmax(C)
    
    return C

def cost_matrix_vertical(S2, S, norm=True):
    """
        Cost matrix that restricts movement to the frequency axis, i.e. vertically (see eq. 22).
        Infinite values are set to np.nan

        Args:
            S2 (np.ndarray, (n_pts2, ndim)) : Support 2.
            S  (np.ndarray, (n_pts2, ndim)) : Target support.
            norm (bool)                     : Fit matrix values between 0 and 1.

        Returns:
            C (np.ndarray, (n_pts2, n_pts) : Structured cost matrix between supports S2 and S.
    
    """
    assert S2.ndim == S.ndim, "Supports must be of same dimension."
    if S.ndim == 1:
        S2 = np.expand_dims(S2.copy(), 1)
        S = np.expand_dims(S.copy(), 1)

    t_frames_2, f_bins_2 = np.unique(S2[:, 0]), np.unique(S2[:, 1])
    f_bins_1             = np.unique(S[:, 0])

    M1 = f_bins_1.size
    M2, N2 = f_bins_2.size, t_frames_2.size

    C = np.empty((M2 * N2, M1 * N2))
    C.fill(np.nan)

    base = ((f_bins_2[:, None] - f_bins_1[None, :]) ** 2)
    
    for t in range(N2):
        C[t * M2: (t + 1) * M2, t * M1 : (t + 1) * M1] = base

    if norm:
        C = C / np.nanmax(C)
    
    return C

def cost_matrix_horizontal_overlap(S1, S, window_size_1, window_size_2, hop_size_1=None, hop_size_2=None, norm=True):
    """
        Cost matrix that restricts movement to the temporal axis considering temporal window overlap (see eq. 23).
        It is optimized to return a vector with finite entries only.
        To handle operations, it also returns a column and row index arrays.
        This leads to vectorized cost c[k] = C[i, j] with C the cost matrix, i = rows[k], j = cols[k].

        Args:
            S1 (np.ndarray, (n_pts1, ndim)) : Support of X1.
            S  (np.ndarray, (n_pts1, ndim)) : Target support.
            window_size_1 (int)             : Size of window used for X1 (in samples).
            window_size_2 (int)             : Size of window used for X2 (in samples).
            hop_size_1 (int)                : Hop size used for X1 (in samples), if None, defaults to window_size_1 / 2.
            hop_size_2 (int)                : Hop size used for X2 (in samples), if None, defaults to window_size_2 / 2.
            norm (bool)                     : Fit matrix values between 0 and 1.

        Returns:
            c (np.ndarray) : Structured cost matrix between supports S1 and S.
    """
    if hop_size_1 is None:
        hop_size_1 = window_size_1 / 2
    if hop_size_2 is None:
        hop_size_2 = window_size_2 / 2

    t_frames_1, f_bins_1 = np.unique(S1[:, 0]), np.unique(S1[:, 1])
    t_frames_2           = np.unique(S[:, 0])

    M1, N1 = f_bins_1.size, t_frames_1.size
    N2 = t_frames_2.size

    c = []
    rows = []
    cols = []

    for t in range(N1):
        # define overlap set (eq. 49)
        # note that it differs slightly since in our code indexes start at 0
        # also window_size in samples removes the sample_rate from the equation

        lower_bound = t * hop_size_1 / hop_size_2 - 1 / 2 / hop_size_2 * (window_size_1 + window_size_2) 
        upper_bound = t * hop_size_1 / hop_size_2 + 1 / 2 / hop_size_2 * (window_size_1 + window_size_2) 
        l = max(0, np.ceil(lower_bound))
        r = min(N2 - 1, np.floor(upper_bound))

        overlap_set = np.arange(int(l), int(r) + 1)
        base = ((np.ones(M1) * t_frames_1[t])[:, None] - t_frames_2[overlap_set][None, :])**2
        c += list(base.flatten())

        current_row_start = M1 * t
        rs = np.arange(current_row_start, current_row_start + M1)
        rows += list(np.repeat(rs, overlap_set.size))

        cs = overlap_set[None, :] * M1 + np.arange(M1)[:, None]
        cols += list(cs.flatten())

    c = np.array(c)
    rows = np.array(rows)
    cols = np.array(cols)

    if norm:
        c = c / np.nanmax(c)
    

    return np.array(c), np.array(rows), np.array(cols)

def cost_matrix_vertical_overlap(S2, S, window_size_1, window_size_2, norm=True):
    """
        Cost matrix that restricts movement to the frequency axis considering frequency window overlap (see eq. 24).
        It is optimized to return a vector with finite entries only.
        To handle operations, it also returns a column and row index arrays.
        This leads to vectorized cost c[k] = C[i, j] with C the cost matrix, i = rows[k], j = cols[k].

        Args:
            S2 (np.ndarray, (n_pts2, ndim)) : Support of X2.
            S  (np.ndarray, (n_pts2, ndim)) : Target support.
            window_size_1 (int)             : Size of window used for X1 (in samples).
            window_size_2 (int)             : Size of window used for X2 (in samples).
            norm (bool)                     : Fit matrix values between 0 and 1.

        Returns:
            C (np.ndarray) : Vectorized overlap cost matrix between supports S2 and S.
    """
    t_frames_2, f_bins_2 = np.unique(S2[:, 0]), np.unique(S2[:, 1])
    f_bins_1             = np.unique(S[:, 1])

    M1 = f_bins_1.size
    M2, N2 = f_bins_2.size, t_frames_2.size

    c_block = []
    row_block = []
    col_block = []

    for f in range(M2):
        # define overlap set (eq. 52)
        # note that it differs slightly since in our code indexes start at 0
        # also window_size in samples removes the sample_rate from the equation

        lower_bound = f * M1 / M2 - 4 * M1 * (1 / window_size_1 + 1 / window_size_2)
        upper_bound = f * M1 / M2 + 4 * M1 * (1 / window_size_1 + 1 / window_size_2)
        l = max(0, np.ceil(lower_bound))
        r = min(M1 - 1, np.floor(upper_bound))

        overlap_set = np.arange(int(l), int(r) + 1)

        base = (f_bins_2[f] - f_bins_1[overlap_set])**2
        c_block += list(base)

        rs = f * np.ones_like(overlap_set)
        row_block += list(rs)

        col_block += list(overlap_set)

    c_block = np.array(c_block)
    row_block = np.array(row_block)
    col_block = np.array(col_block)

    c = np.tile(c_block, N2)
    rows = np.array([row_block + M2 * i for i in range(N2)]).flatten()
    cols = np.array([col_block + M1 * i for i in range(N2)]).flatten()

    if norm:
        c = c / np.nanmax(c)

    return c, rows, cols

def cost_matrix_horizontal_overlap_mel(S1, S, window_size_1, window_size_2, sr, hop_size_1=None, hop_size_2=None, norm=True):
    """
        Similar as `cost_matrix_horizontal_overlap` except that frequency bins of S are on a mel scale, see eq. 46.
        This matrix allows for a small 2D movement.

        Args:
            S1 (np.ndarray, (n_pts2, ndim)) : Support of X1.
            S  (np.ndarray, (n_pts2, ndim)) : Target support.
            window_size_1 (int)             : Size of window used for X1 (in samples).
            window_size_2 (int)             : Size of window used for X2 (in samples).
            hop_size_1 (int)                : Hop size used for X1 (in samples), if None, defaults to window_size_1 / 2.
            hop_size_2 (int)                : Hop size used for X2 (in samples), if None, defaults to window_size_2 / 2.
            sr (int)                        : Sample rate (Hz).
            norm (bool)                     : Fit matrix values between 0 and 1.
    
        Returns:
            C (np.ndarray) : Vectorized overlap cost matrix between supports S1 and S, on mel scale.
    """
    if hop_size_1 is None:
        hop_size_1 = window_size_1 / 2
    if hop_size_2 is None:
        hop_size_2 = window_size_2 / 2

    t_frames_1, f_bins_1 = np.unique(S1[:, 0]), np.unique(S1[:, 1])
    t_frames_2, m_bins   = np.unique(S[:, 0]), np.unique(S[:, 1])

    M1, N1 = f_bins_1.size, t_frames_1.size
    M, N2  = m_bins.size, t_frames_2.size

    c = []
    rows = []
    cols = []

    mr = mel(sr / 2)

    for t in range(N1):
        lower_bound = t * hop_size_1 / hop_size_2 - 1 / 2 / hop_size_2 * (window_size_1 + window_size_2) 
        upper_bound = t * hop_size_1 / hop_size_2 + 1 / 2 / hop_size_2 * (window_size_1 + window_size_2) 
        l = max(0, np.ceil(lower_bound))
        r = min(N2 - 1, np.floor(upper_bound))

        time_overlap_set = np.arange(int(l), int(r) + 1)
        base_time = (t_frames_1[t] - t_frames_2[time_overlap_set])**2
        if len(base_time) > 0 and base_time.max() > 0: # avoid division by zero if there's only one entry and happens to be zero
            base_time /= base_time.max()

        for f in range(f_bins_1.size):
            # Define mel overlap set.
            # To derive the upper and lower bounds we use the same reasoning used in the annex.
            # Previously we replaced freq = f / M * sr
            # For mels we replace freq with 700 * (10^(m'/(M - 1) m_r / 2595) - 1) where m' is the index of the m'-th mel.

            lb_log_arg = f / M1 * sr / 2 / 700 - 2 * sr / 700 * (1 / window_size_1 + 1 / window_size_2) + 1
            ub_log_arg = f / M1 * sr / 2 / 700 + 2 * sr / 700 * (1 / window_size_1 + 1 / window_size_2) + 1

            lower_bound = np.log10(lb_log_arg) * 2595 * (M - 1) / mr
            upper_bound = np.log10(ub_log_arg) * 2595 * (M - 1) / mr

            l = max(0, np.ceil(lower_bound))
            r = min(m_bins.size - 1, np.floor(upper_bound))

            mel_overlap_set = np.arange(int(l), int(r) + 1)

            base_mel = (f_bins_1[f] - m_bins[mel_overlap_set])**2
            if len(base_mel) > 0 and base_mel.max() > 0: # avoid division by zero if there's only one entry and happens to be zero
                base_mel /= base_mel.max()

            c_sum = (base_mel[None, :] + base_time[:, None]).flatten() # euclidean

            c += list(c_sum)

            current_row_start = f_bins_1.size * t
            row_offset = f
            rs = (np.ones(c_sum.size) * (current_row_start + row_offset)).astype(np.int32)
            rows += list(rs)

            cs = mel_overlap_set[None, :] + time_overlap_set[:, None] * m_bins.size
            cols += list(cs.flatten())

    return np.array(c), np.array(rows), np.array(cols)


def cost_matrix_vertical_overlap_mel(S2, S, window_size_1, window_size_2, sr, norm=True):
    """
        Similar as `cost_matrix_vertical_overlap` except that frequency bins of S are on a mel scale.

        Args:
            S2 (np.ndarray, (n_pts2, ndim)) : Support of X2.
            S  (np.ndarray, (n_pts2, ndim)) : Target support.
            window_size_1 (int)             : Size of window used for X1 (in samples).
            window_size_2 (int)             : Size of window used for X2 (in samples).
            hop_size_1 (int)                : Hop size used for X1 (in samples), if None, defaults to window_size_1 / 2.
            hop_size_2 (int)                : Hop size used for X2 (in samples), if None, defaults to window_size_2 / 2.
            sr (int)                        : Sample rate (Hz).
            norm (bool)                     : Fit matrix values between 0 and 1.
        
        Returns:
            C (np.ndarray) : Vectorized overlap cost matrix between supports S2 and S, on mel scale.
            
    """
    t_frames_2, f_bins_2 = np.unique(S2[:, 0]), np.unique(S2[:, 1])
    m_bins               = np.unique(S[:, 1])

    M = m_bins.size
    M2, N2 = f_bins_2.size, t_frames_2.size

    c_block = []
    row_block = []
    col_block = []

    mr = mel(sr / 2)

    for f in range(f_bins_2.size):
        # To derive the upper and lower bounds we use the same reasoning used in the annex.
        # Previously we replaced freq = f / M * sr
        # For mels we replace freq with 700 * (10^(m'/(M - 1) m_r / 2595) - 1) where m' is the index of the m'-th mel.

        lb_log_arg = f / M2 * sr / 2 / 700 - 2 * sr / 700 * (1 / window_size_1 + 1 / window_size_2) + 1
        ub_log_arg = f / M2 * sr / 2 / 700 + 2 * sr / 700 * (1 / window_size_1 + 1 / window_size_2) + 1

        lower_bound = np.log10(lb_log_arg) * 2595 * (M - 1) / mr
        upper_bound = np.log10(ub_log_arg) * 2595 * (M - 1) / mr

        l = max(0, np.ceil(lower_bound))
        r = min(m_bins.size - 1, np.floor(upper_bound))

        overlap_set = np.arange(int(l), int(r) + 1)

        c_freqs = (f_bins_2[f] - m_bins[overlap_set])**2
        c_block += list(c_freqs)

        rs = f * np.ones_like(overlap_set)
        row_block += list(rs)

        col_block += list(overlap_set)

    c_block = np.array(c_block)
    row_block = np.array(row_block)
    col_block = np.array(col_block)

    c = np.tile(c_block, N2)
    rows = np.array([row_block + f_bins_2.size * i for i in range(N2)]).flatten()
    cols = np.array([col_block + m_bins.size * i for i in range(N2)]).flatten()

    if norm:
        c = c / np.nanmax(c)

    return c, rows, cols