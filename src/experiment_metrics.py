import numpy as np

from .utils import *

def generate_sinus(f, t, attenuate=.2):
    """
        Generate a sinusoid at frequency f with temporal support t.
        Onset and offset times are taken as t[0] and t[-1] respectively.

        Args:
            f (double): sinusoid frequency, in Hertz.
            t (np.ndarray): temporal support.
            attenuate (double, [0, 1]): percentage of borders to attenuate, avoids stft issues. E.g. 0.2 attenuates 20% at beginning and end of the signal.
            
        Returns:
            signal (np.ndarray): sinusoid.
    """
    signal = np.sin(2 * np.pi * f * t)
    attenuate_size = int(t.size * attenuate)
    signal[:attenuate_size] *= np.linspace(0, 1, attenuate_size)
    signal[signal.size - attenuate_size:] *= np.linspace(1, 0, attenuate_size)

    return signal

def single_tf_packet(sr, dur, f_min=None, f_max=None, min_dur=None, max_dur=None, min_onset=None, max_offset=None):
    """
        Generate a signal containing a short duration sinusoid.

        Args:
            sr (double): sample rate (in Hz).
            dur (double): duration of the signal (in seconds).
            f_min (double): minimum frequency duration (in Hz).
            f_max (double): maximum frequency duration (in Hz).
            min_dur (double): minimum duration of the sinusoid.
            max_dur (double): maximum duration of the sinusoid.
            min_offset (double): minimum onset time, >= 0.
            max_offset (double): max offset time, <= dur.

        Returns:
            signal (np.ndarray): signal consisting of silence and a short sinusoid packet.
    """
    if f_min is None:
        f_min = 0
    if f_max is None:
        f_max = sr / 2
    if min_dur is None:
        min_dur = dur / 20 # defines an arbitrary minimum duration
    if max_dur is None:
        max_dur = dur / 5  # defines an arbitrary maximum duration
    if min_onset is None:
        min_onset = 0
    if max_offset is None:
        max_offset = dur

    size = int(dur * sr)

    signal = np.zeros(size)
    t_samples = np.arange(size) / sr

    f = np.random.randint(f_min, f_max)
    tf_dur = np.random.uniform(low=min_dur, high=max_dur)
    t_on = np.random.uniform(min_onset, max_offset - tf_dur)
    t_off = t_on + tf_dur

    s_on = int(t_on * sr)
    s_off = int(t_off * sr)

    signal[s_on:s_off] += generate_sinus(f, t_samples[s_on:s_off])

    return signal, f, t_on, t_off

def multi_tf_packet(sr, dur, n_packets, f_min=None, f_max=None, min_dur=None, max_dur=None, min_onset=None, max_offset=None):
    """
        Generate a signal containing a mixture of n_packets short duration sinusoids.

        Args:
            sr (double): sample rate (in Hz).
            dur (double): duration of the signal (in seconds).
            n_packets(int): number of packets.
            f_min (double): minimum frequency duration (in Hz).
            f_max (double): maximum frequency duration (in Hz).
            min_dur (double): minimum duration of the sinusoid.
            max_dur (double): maximum duration of the sinusoid.
            min_offset (double): minimum onset time, >= 0.
            max_offset (double): max offset time, <= dur.

        Returns:
            signal (np.ndarray): signal consisting of silence and a mixture of sinusoid packets.
    """
    if f_min is None:
        f_min = 0
    if f_max is None:
        f_max = sr / 2
    if min_dur is None:
        min_dur = dur / 20 # defines an arbitrary minimum duration
    if max_dur is None:
        max_dur = dur / 5  # defines an arbitrary maximum duration
    if min_onset is None:
        min_onset = 0
    if max_offset is None:
        max_offset = dur

    size = int(dur * sr)

    signal = np.zeros(size)
    t_samples = np.arange(size) / sr

    fs, t_ons, t_offs = np.zeros(n_packets), np.zeros(n_packets), np.zeros(n_packets)

    for i in range(n_packets):
        # compute single packet
        signal_packet, f, t_on, t_off = single_tf_packet(sr, dur, f_min, f_max, min_dur=None, max_dur=None, min_onset=None, max_offset=None)
        signal += signal_packet
        fs[i] = f
        t_ons[i] = t_on
        t_offs[i] = t_off

    return signal, fs, t_ons, t_offs

def error_frequency(X, f, delta_f, f_bins):
    """
        Error in frequency tolerance (see eq. 41).

        Args:
            X (np.ndarray): power spectrogram.
            f (double): center frequency to evaluate (in Hz).
            delta_f (double): tolerance in frequency (in Hz).
            f_bins (np.ndarray): frequency bins of X.
    """
    Pf = closest_neighbor(f, f_bins)
    return X[(f_bins < Pf - delta_f) | (f_bins > Pf + delta_f), :].sum() / X.sum()

def error_time(X, t_on, t_off, delta_t, t_bins):
    """
        Error in time tolerance (see eq. 44).

        Args:
            X (np.ndarray): power spectrogram.
            t_on (double): onset time to evaluate (in seconds).
            t_off (double): offset time to evaluate (in seconds).
            delta_t (double): tolerance in time (in seconds).
            t_bins (np.ndarray): time bins of X.
    """
    Pt_on, Pt_off = closest_neighbor(t_on, t_bins), closest_neighbor(t_off, t_bins)
    return X[:,  (t_bins < Pt_on - delta_t) | (t_bins > Pt_off + delta_t)].sum() / X.sum()

def error_overall(X, freqs, t_ons, t_offs, delta_f, delta_t, f_bins, t_bins):
    """
        Overall time-frequency error (see eq. 47)

        Args:
            X (np.ndarray): power spectrogram.
            freqs (np.ndarray): center frequencys to evaluate (in Hz).
            t_ons (np.ndarray): onset times to evaluate (in seconds).
            t_offs (np.ndarray): offset times to evaluate (in seconds).
            delta_f (double): tolerance in frequency (in Hz).
            delta_t (double): tolerance in time (in seconds).
            f_bins (np.ndarray): frequency bins of X.
            t_bins (np.ndarray): time bins of X.
    """
    keep = np.zeros_like(X, dtype=bool)

    for f, t_on, t_off in zip(freqs, t_ons, t_offs):
        Pf = closest_neighbor(f, f_bins)
        Pt_on, Pt_off = closest_neighbor(t_on, t_bins), closest_neighbor(t_off, t_bins)
        
        f_mask = (f_bins >= Pf - delta_f) & (f_bins <= Pf + delta_f)
        t_mask = (t_bins >= Pt_on - delta_t) & (t_bins <= Pt_off + delta_t)
        keep |= f_mask[:, None] & t_mask[None, :]

    return X[~keep].sum() / X.sum()

def error_harmonic(X, f0_track, f0_times, delta_f, f_bins, t_bins, f_max):
    """
       Harmonic concentration error (see eq. 50)

       Args:
            X (np.ndarray): power spectrogram.
            f0_track (np.ndarray): pitch values provided by database (including 0 for unvoiced frames).
            f0_times (np.ndarray): temporal samplings for the pitch values.
            delta_f (double): tolerance in frequency (in Hz).
            f_bins (np.ndarray): frequency bins of X.
            t_bins (np.ndarray): time bins of X.
            f_max (double): maximum frequency for harmonics.
    """
    voiced_times  = f0_times[f0_track > 0]
    valid_pitches = f0_track[f0_track > 0]

    score = []
    for i in range(len(valid_pitches)):
        f0 = closest_neighbor(valid_pitches[i], f_bins)
        if f0 <= 0:
            continue

        n_harmonics = int(f_max // f0)
        harmonic_freqs = np.zeros(n_harmonics)
        for k in range(n_harmonics):
            harmonic_freqs[k] = closest_neighbor((k + 1) * f0, f_bins)

        keep = np.zeros(X.shape[0], dtype=bool)
        for hf in harmonic_freqs:
            keep |= np.abs(f_bins - hf) <= delta_f

        t_idx = idx_at_value(voiced_times[i], t_bins)

        denom = X[:, t_idx].sum()
        if denom <= 0:
            continue

        score.append(X[~keep, t_idx].sum() / denom)

    return np.mean(score)
