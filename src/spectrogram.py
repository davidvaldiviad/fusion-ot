from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
import numpy as np

class Spectrogram:
    def __init__(self, 
                 signal,
                 sr,
                 window_size_s,
                 *,
                 window=None,
                 hop_size_s=None,
                 phase_shift=None,
                 nfft=None,
                 fft_mode='onesided'):
        self.signal        = signal
        self.sr            = sr
        self.signal_dur    = len(self.signal) / self.sr
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

        self.phase_shift = phase_shift

        self.fft_mode = fft_mode
        self.stft_obj = ShortTimeFFT(self.window, hop=self.hop_size, fs=self.sr, phase_shift=self.phase_shift, mfft=self.nfft, fft_mode=self.fft_mode)

        self.mag_stft() # compute f and t bins


    def stft(self, p0=0):
        return self.stft_obj.stft(self.signal, p0=p0)
    
    def mag_stft(self, p0=0):

        self.f_bins = self.stft_obj.f
        self.t_bins = self.stft_obj.t(self.signal.size)
        self.t_bins = self.t_bins[self.t_bins >= 0]

        self.f = len(self.f_bins)
        self.t = len(self.t_bins)
        self.size = self.f * self.t

        return np.abs(self.stft(p0=p0))
    
    def power_spectrogram(self, p0=0):
        return self.mag_stft(p0=p0) ** 2
