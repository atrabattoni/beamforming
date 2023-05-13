import numpy as np
import scipy.fft
from spectrum import dpss


def multitaper_correlate(x, n_tapers, frequency_band, sampling_rate):
    x = x - np.mean(x, axis=1, keepdims=True)
    weight, freq, X = multitaper_fft(x, n_tapers, sampling_rate)
    if frequency_band is not None:
        mask = np.logical_and(freq >= frequency_band[0], freq < frequency_band[1])
        freq = freq[mask]
        X = X[:, :, mask]
    inv_Px = 1.0 / np.sum(np.real(X * np.conj(X)) * weight[:, None, None], axis=0)
    C = np.sum(
        weight[:, None, None, None] * X[:, :, None, :] * np.conj(X[:, None, :, :]),
        axis=0,
    )
    C = C * (inv_Px[:, None, :] * inv_Px[None, :, :])
    return freq, C


def multitaper_fft(x, n_tapers, sampling_rate):
    n_stations, n_samples = x.shape
    nfft = scipy.fft.next_fast_len(n_samples)
    taper, eigval = dpss(n_samples, n_tapers)
    weight = eigval / (np.arange(len(eigval)) + 1)
    freq = scipy.fft.rfftfreq(n=nfft, d=1.0 / sampling_rate)
    X = scipy.fft.rfft(taper.T[:, None, :] * x[None, :, :], nfft, axis=-1)
    return weight, freq, X
