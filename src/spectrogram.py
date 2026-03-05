from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import numpy as np

class Spectrogram:
    """
        Spectrogram class. It is a wrap of the scipy's ShortTimeFFT class, with additional attributes and functions.

        Args:
            signal (np.ndarray)    : signal used for spectrogram.
            sr (int)               : sample rate (in Hz).
            window_size_s (double) : size of the window (in seconds).
            window (np.ndarray)    : window vector (default Hann).
            hop_size_s (double)    : hop size (default window_size_s / 2, 50% overlap).
            nfft (int)             : number of frequency points used to compute stft (defaults: window size in samples).
    """
    def __init__(self, 
                 signal,
                 sr,
                 window_size_s,
                 *,
                 window=None,
                 hop_size_s=None,
                 nfft=None):
        self.signal        = signal
        self.sr            = sr
        self.window_size_s = window_size_s
        self.window_size = int(self.window_size_s * self.sr)

        self.nfft = nfft if nfft is not None else self.window_size

        if window is None:
            self.window = hann(self.window_size)
        else:
            self.window = window

        if hop_size_s is None:
            self.hop_size   = self.window_size // 2
            self.hop_size_s = self.window_size_s / 2
        else:
            self.hop_size = int(hop_size_s * sr)
            self.hop_size_s = hop_size_s

        self.stft_obj = ShortTimeFFT(self.window, hop=self.hop_size, fs=self.sr, mfft=self.nfft)

        self.f_bins = self.stft_obj.f
        self.t_frames = self.stft_obj.t(self.signal.size)
        self.t_frames = self.t_frames[self.t_frames >= 0]

        self.f = len(self.f_bins)
        self.t = len(self.t_frames)
        self.size = self.f * self.t

    def stft(self, p0=0):
        """
            Computes complex STFT of signal.

            Args:
                p0 (int): index of first frame.

            Returns:
                STFT of signal.
        """

        return self.stft_obj.stft(self.signal, p0=p0)
    
    def spectrogram(self, p0=0):
        """
            Computes spectrogram of signal, defined as power of STFT.

            Args:
                p0 (int): index of first frame.

            Returns:
                Spectrogram of signal.
        """

        return np.abs(self.stft(p0=p0))**2
    