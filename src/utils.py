import numpy as np

def generate_toy_signal(sr=1000):
    """
        Generate toy signal used in Fig. 1.

        Args:
            sr (int): sample rate of the signal.

        Returns:
            t      (np.ndarray): time values of the signal.
            signal (np.ndarray): toy signal.
    """
    duration = .5
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)

    # timestamps
    t1, t2, t3, t4 = .05, .3, .32, .45
    s1, s2, s3, s4 = int(t1 * sr), int(t2 * sr), int(t3 * sr), int(t4 * sr) # as samples

    # sinusoids
    f0, f1 = 100, 120

    signal = np.zeros_like(t)
    signal[s1:s2] += .2 * np.sin(2 * np.pi * f0 * t[s1:s2])
    signal[s1:s2] += .2 * np.sin(2 * np.pi * f1 * t[s1:s2])
    signal[s3:s4] += .2 * np.sin(2 * np.pi * f0 * t[s3:s4])

    return t, signal

def time_freq_support(f_bins, t_frames, *, norm=False):
    """
        Generate time-frequency grids as in eq. 8.

        Args:
            f_bins   (np.ndarray): Frequency bins (in Hz) (see eq. 6).
            t_frames (np.ndarray): Time frames (in s) (see eq. 7).
            norm     (bool): Normalize tf points so that each axis varies between 0 and 1.

        Returns:
            support     (np.ndarray): TF grid.
            idx_support (np.ndarray): 1D-2D index mapping ordered in a column-wise vectorization (see eq. 15).

    """
    support = np.zeros((f_bins.size * t_frames.size, 2))
    idx_support = np.zeros_like(support)

    index = 0
    for t in t_frames:
        for f in f_bins:
            support[index] = [t, f]
            idx_support[index] = [t, f]
            index += 1

    if norm:
        support = (support - support[0]) / support[-1]

    return support, idx_support
