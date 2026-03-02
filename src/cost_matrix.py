import numpy as np
from scipy.spatial.distance import cdist

def sq_cost_matrix(S1, S2, *, norm='max'):
    """
    Squared Euclidean cost matrix between supports S1 and S2.

    While in the paper we defined the cost using the sample rate (see eq. 20), in practice we used normalized supports (see utils.time_freq_support) and compute cost between normalized indices.

    Args:
        S1 (np.ndarray, (n_pts1, ndim)) : Support 1.
        S2 (np.ndarray, (n_pts2, ndim)) : Support 2.
        norm (string)                   : Fit matrix values between 0 and 1.

    Returns:
        C (np.ndarray, (n_pts1, n_pts2) : Cost matrix between supports S1 and S2.
    """
    assert S1.ndim == S2.ndim, "Supports must be of same dimension."
    if S1.ndim == 1:
        S1 = np.expand_dims(S1.copy(), 1)
        S2 = np.expand_dims(S2.copy(), 1)

    C = cdist(S1, S2, metric='sqeuclidean')
    if norm == 'max':
        C = C / C.max()
    return C

def cost_matrix_freq_overlap(T_bins1, T_bins2, F, window_size_1, window_size_2, norm=True, hop_size_1=None, hop_size_2=None):
    if norm:
        T_bins1 = T_bins1.copy() / T_bins1.max()
        T_bins2 = T_bins2.copy() / T_bins2.max()

    if hop_size_1 is None:
        hop_size_1 = window_size_1 / 2
    if hop_size_2 is None:
        hop_size_2 = window_size_2 / 2

    c = []
    rows = []
    cols = []

    for t in range(len(T_bins1)):
        l = max(0, np.ceil(t * hop_size_1 / hop_size_2 - (window_size_1 + window_size_2) / (2 * hop_size_2)))
        r = min(T_bins2.size - 1, np.floor(t * hop_size_1 / hop_size_2 + (window_size_1 + window_size_2) / (2 * hop_size_2)))

        taus = np.arange(int(l), int(r) + 1)
        c_taus = ((np.ones(F) * T_bins1[t])[:, None] - T_bins2[taus][None, :])**2
        c += list(c_taus.flatten())

        current_row_start = F * t
        rs = np.arange(current_row_start, current_row_start + F)
        rows += list(np.repeat(rs, taus.size))

        cs = taus[None, :] * F + np.arange(F)[:, None]
        cols += list(cs.flatten())

    return np.array(c), np.array(rows), np.array(cols)

def cost_matrix_time_overlap(F_bins1, F_bins2, T, window_size_1, window_size_2, norm=True):
    if norm:
        F_bins1 = F_bins1 / F_bins1.max()
        F_bins2 = F_bins2 / F_bins2.max()

    c_block = []
    row_block = []
    col_block = []

    M1 = F_bins1.size
    M2 = F_bins2.size

    for f in range(F_bins2.size):
        l = max(0, np.ceil(f * M1 / M2 - 2 * M1 / window_size_2 - 2 * M1 / window_size_1))
        r = min(F_bins1.size - 1, np.floor(f * M1 / M2 + 2 * M1 / window_size_2 + 2 * M1 / window_size_1))

        l = max(0, np.ceil(f * (M1 - 1) / (M2 - 1) - 4 * (M1 - 1) / window_size_1 - 4 * (M1 - 1) / window_size_2))
        r = min(F_bins1.size - 1, np.floor(f * (M1 - 1) / (M2 - 1) + 4 * (M1 - 1) / window_size_1 + 4 * (M1 - 1) / window_size_2))

        freqs = np.arange(int(l), int(r) + 1)

        c_freqs = (F_bins2[f] - F_bins1[freqs])**2
        c_block += list(c_freqs)

        rs = f * np.ones_like(freqs)
        row_block += list(rs)

        col_block += list(freqs)

    c_block = np.array(c_block)
    row_block = np.array(row_block)
    col_block = np.array(col_block)

    c = np.tile(c_block, T)
    rows = np.array([row_block + F_bins2.size * i for i in range(T)]).flatten()
    cols = np.array([col_block + F_bins1.size * i for i in range(T)]).flatten()

    return c, rows, cols