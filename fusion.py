#!/usr/bin/python3

import argparse
import librosa
from src.utils import *
from src.spectrogram import *
from src.cost_matrix import *
from src.barycenter import *

parser = argparse.ArgumentParser()

## required or default

# signal
parser.add_argument('filename', 
                    help='Path to the input audio file (any format supported by librosa.load).')
parser.add_argument('--sr',
                    help='Sample rate, in (Hz). Default 16kHz.',
                    type=int)

# spectrograms
parser.add_argument('--window-large-s',
                    help='Large window of high-frequency resolution spectrogram, in seconds. Default 80ms.')
parser.add_argument('--window-short-s',
                    help='Short window of high-temporal resolution spectrogram, in seconds. Default 20ms.',
                    type=float)
parser.add_argument('--hop-size-large-s',
                    help='Hop size of high-frequency resolution spectrogram, in seconds. Default: Half the window size (50% overlap).')
parser.add_argument('--hop-size-short-s',
                    help='Hop size of high-temporal resolution spectrogram, in seconds. Default: Half the window size (50% overlap).')
parser.add_argument('--nfft-large',
                    help='Number of FFT points for high-frequency resolution specgrogram. Default: equal to window size.')
parser.add_argument('--nfft-short',
                    help='Number of FFT points for high-temporal resolution specgrogram. Default: equal to window size.')

# uot
parser.add_argument('--eta',
                    help='Marginal relaxation parameter. Default 1.')
parser.add_argument('--nIter',
                    help='Number of iterations. Default 20.')

# not sure/other

parser.add_argument('--window-large-samples',
                    help='Large window of high-frequency resolution spectrogram, in samples.')
parser.add_argument('--window-short-samples',
                    help='Short window of high-temporal resolution spectrogram, in samples.')
parser.add_argument('--hop-size-large-samples',
                    help='Hop size of high-frequency resolution spectrogram, in samples.')
parser.add_argument('--hop-size-short-samples',
                    help='Hop size of high-temporal resolution spectrogram, in samples.')

args = parser.parse_args()

print(args)


filename = parse(args.filename)
sr = parse(args.sr, 16000)

signal, _ = librosa.load(filename, sr=sr)

window_size_1_s = parse(args.window_large_s, 80e-3)
window_size_2_s = parse(args.window_large_s, 20e-3)

hop_size_1_s = parse(args.hop_size_large_s, window_size_1_s / 2)
hop_size_2_s = parse(args.hop_size_short_s, window_size_2_s / 2)

nfft_1 = parse(args.nfft_large, int(sr * window_size_1_s))
nfft_2 = parse(args.nfft_short, int(sr * window_size_2_s))

X1 = Spectrogram(signal, sr, window_size_1_s, hop_size_s=hop_size_1_s, nfft=nfft_1)
X2 = Spectrogram(signal, sr, window_size_2_s, hop_size_s=hop_size_2_s, nfft=nfft_2)

x1 = X1.power_spectrogram()
x2 = X2.power_spectrogram()

M1, N2 = X1.f, X2.t

c1, rows1, cols1 = cost_matrix_freq_overlap(X1.t_bins, X2.t_bins, X1.f_bins.size, X1.window_size, X2.window_size, norm=True, hop_size_1=X1.hop_size, hop_size_2=X2.hop_size)
c2, rows2, cols2 = cost_matrix_time_overlap(X1.f_bins, X2.f_bins, X2.t, X1.window_size, X2.window_size, norm=True)

c1 /= c1.max()
c2 /= c2.max()

eta   = parse(args.eta, 1)
nIter = parse(args.nIter, 20)

x = uot_sparse_barycenter(x1.T.flatten(),
                          x2.T.flatten(),
                          c1,
                          c2,
                          eta,
                          rows1,
                          cols1,
                          rows2,
                          cols2,
                          X1.f * X2.t,
                          nItermax=nIter,
                          verbose=False)

x = x.reshape(N2, M1).T

import matplotlib.pyplot as plt

plt.imshow(x)
plt.show()