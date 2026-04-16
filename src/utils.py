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
        Generate time-frequency grids as in eq. 9.

        Args:
            f_bins   (np.ndarray): Frequency bins (in Hz) (see eq. 7).
            t_frames (np.ndarray): Time frames (in s) (see eq. 8).
            norm     (bool): Normalize tf points so that each axis varies between 0 and 1.

        Returns:
            support     (np.ndarray): TF grid.

    """
    support = np.zeros((f_bins.size * t_frames.size, 2))

    index = 0
    for t in t_frames:
        for f in f_bins:
            support[index] = [t, f]
            index += 1

    if norm:
        support = normalize(support)

    return support

def idx_at_value(value, array):
    """
        Return index of closest value inside array.

        Args:
            value: value to find.
            array (np.ndarray): array of values

        Returns:
            index of closest value, 0 or array.size - 1 if value outside bounds of array or None.
    """
    if value is None:
        return array.size - 1
    return np.abs(array - value).argmin()

def closest_neighbor(value, array):
    """
        Return closest value inside array.

        Args:
            value: value to find.
            array (np.ndarray): array of values

        Returns:
            closest value, 0 or array.size - 1 if value outside bounds of array or None.
    """   
    return array[idx_at_value(value, array)]


def kullback_leibler(a, b, thr=1e-20):
    """
        Kullback-Leibler divergene.

        Args:
            a (np.ndarray): input vector.
            b (np.ndarray): input vector.
            thr (double):   to avoid log(0) or log(1 / 0)
        
        Returns:
            KL(a, b)
    """
    kl_div = a * np.log((a + thr) / (b + thr)) - a + b
    return kl_div.sum()

def mel(f):
    """
        Mel scale O'Shaughnessy's formula (eq. 54).
        
        Args:
            f (double): Frequency (in Hz).

        Returns:
            Mel associated to f.
    """
    return 2595 * np.log10(1 + f / 700)

def imel(m):
    """
        Inverse-mel scale O'Shaughnessy's formula (eq. 55).
        
        Args:
            m (double): Mels.

        Returns:
            Frequency associated to m.
    """
    return 700 * (10 ** (m / 2595) - 1)

def mel_frequency_bins(n_bins, sr):
    """
        Compute mel frequency scale as explained in Section V-D.

        Args:
            n_bins (int): number of mel bins.
            sr (int)    : sample rate (in Hz).

        Returns:
            Mel scale with n_bins.
    """
    mr = mel(sr / 2)
    mels = np.linspace(0, mr, n_bins)
    m_bins = imel(mels)

    return m_bins

def normalize(array):
    """
        Returns sorted array scaled between 0 and 1.

        Args:
            array (np.ndarray): array to normalize, sorted.
    """

    return (array - array[0]) / array[-1]