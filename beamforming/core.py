import numba as nb
import numpy as np
import scipy.fft
from spectrum import dpss


class Beamformer:
    def __init__(
        self,
        coords,
        azimuth_grid,
        speed_grid,
        n_samples,
        sampling_rate,
        frequency_band,
        n_tapers,
    ):
        self.coords = coords
        self.azimuth_grid = azimuth_grid
        self.speed_grid = speed_grid
        self.n_samples = n_samples
        self.frequency_band = frequency_band
        self.sampling_rate = sampling_rate
        self.n_tapers = n_tapers

        # Select frequency band
        nfft = scipy.fft.next_fast_len(n_samples)
        freq = scipy.fft.rfftfreq(n=nfft, d=1.0 / sampling_rate)
        mask = (freq >= frequency_band[0]) & (freq < frequency_band[1])
        freq = freq[mask]

        Sx, Sy = self.get_grid()
        delay = self.get_delay(Sx, Sy)
        self.A = self.get_steering_vector(delay, freq)

    def beamform(self, x):
        # Compute covariance matrix
        Cxy = CMTM(
            x,
            n_tapers=self.n_tapers,
            frequency_band=self.frequency_band,
            sampling_rate=self.sampling_rate,
        )

        # Compute beampower
        Pr = noise_space_projection(Cxy, self.A, n_sources=1)
        P = 1.0 / Pr
        P = P.reshape((len(self.azimuth_grid), len(self.speed_grid)))
        return P

    def get_grid(self):
        slowness_grid = 1 / self.speed_grid
        Sx = -np.sin(self.azimuth_grid)[:, None] * slowness_grid[None, :]
        Sy = -np.cos(self.azimuth_grid)[:, None] * slowness_grid[None, :]
        return Sx, Sy

    def get_delay(self, Sx, Sy):
        x, y = self.coords.T
        return Sx[:, :, None] * x[None, None, :] + Sy[:, :, None] * y[None, None, :]

    def get_steering_vector(self, delay, freq):
        A = np.exp(2j * np.pi * freq[:, None, None, None] * delay[None, :, :, :])
        return A


def CMTM(x, n_tapers, frequency_band=None, sampling_rate=None):
    x = x - np.mean(x, axis=1, keepdims=True)
    weights, freqs, X = multitaper(x, n_tapers, sampling_rate)
    if frequency_band is not None:
        mask = np.logical_and(freqs >= frequency_band[0], freqs < frequency_band[1])
        X = X[:, :, mask]
    inv_Px = 1 / np.sum(np.abs(X.T) ** 2 * weights, axis=-1).T
    Cxy = correlate(X, weights, inv_Px)
    return Cxy


def multitaper(x, n_tapers, sampling_rate):
    n_stations, n_samples = x.shape
    nfft = scipy.fft.next_fast_len(n_samples)
    taper, eigval = dpss(n_samples, n_tapers)
    weight = eigval / (np.arange(len(eigval)) + 1)
    taper = np.tile(taper.T, [n_stations, 1, 1])
    taper = np.swapaxes(taper, 0, 1)
    freq = scipy.fft.rfftfreq(n=nfft, d=1.0 / sampling_rate)
    X = scipy.fft.rfft(np.multiply(taper, x), nfft, axis=-1)
    return weight, freq, X


@nb.njit()
def correlate(X, weight, scale):
    tapers, n_stations, n_samples = X.shape
    Cxy = np.zeros((n_stations, n_stations, n_samples), dtype=nb.complex64)
    for i in range(n_stations):
        for j in range(n_stations):
            for k in range(tapers):
                Cxy[i, j] += weight[k] * X[k, i] * np.conj(X[k, j])
            Cxy[i, j] = Cxy[i, j] * (scale[i] * scale[j])
    return Cxy


def noise_space_projection(Rxx, A, n_sources=1):
    A = A.reshape(A.shape[0], -1, A.shape[-1])
    Nf, Nslow, m = A.shape
    scale = 1.0 / (m * Nf)

    # Total projection onto noise space
    Pm = np.zeros(Nslow, dtype=complex)

    for f in range(Nf):
        Af = A[f]

        # Compute eigenvalues/vectors assuming Rxx is complex Hermitian (conjugate symmetric)
        # Eigenvalues appear in ascending order
        _, v = np.linalg.eigh(Rxx[:, :, f])
        M = n_sources
        # Extract noise space (size n-M)
        # NOTE: in original code, un was labelled "signal space"!
        un = v[:, : m - M]
        # Precompute un.un*
        Un = np.dot(un, un.conj().T)
        # Project steering vector onto subspace
        Pm += np.einsum("sn, nk, sk->s", Af.conj(), Un, Af, optimize=True)

    return np.real(Pm) * scale
