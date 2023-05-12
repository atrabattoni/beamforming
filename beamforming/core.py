import numba as nb
import numpy as np
import scipy
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
        nfft = 2 ** int(np.log2(n_samples) + 1) + 1  # Don't mess with this...
        freqs = scipy.fft.rfftfreq(n=2 * nfft, d=1 / sampling_rate)
        inds = (freqs >= frequency_band[0]) & (freqs < frequency_band[1])
        freqs_select = freqs[inds]

        Sx, Sy = self.get_grid()
        delay = self.get_delay(Sx, Sy)
        self.A = self.get_steering_vector(delay, freqs_select)

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

    def get_steering_vector(self, dt, freqs):
        A = np.exp(2j * np.pi * freqs[:, None, None, None] * dt[None, :, :, :])
        return A


def CMTM(x, n_tapers, frequency_band=None, sampling_rate=None):
    n_stations, n_samples = x.shape
    nfft = 2 ** int(np.log2(n_samples) + 1) + 1  # Next power of 2 (for FFT)
    x = x - np.mean(x, axis=1, keepdims=True)
    tapers, eigen_values = dpss(n_samples, n_tapers)

    # Compute weights from eigenvalues
    weights = eigen_values / (np.arange(len(eigen_values)) + 1)

    # Align tapers with X
    tapers = np.tile(tapers.T, [n_stations, 1, 1])
    tapers = np.swapaxes(tapers, 0, 1)

    # Compute tapered FFT of X
    # Note that X is assumed to be real, so that the negative frequencies can be discarded
    X = scipy.fft.rfft(np.multiply(tapers, x), 2 * nfft, axis=-1)

    # Multitaper power spectrum (not scaled by weights.sum()!)
    Pk = np.abs(X) ** 2
    Pxx = np.sum(Pk.T * weights, axis=-1).T
    inv_Px = 1 / np.sqrt(Pxx)

    # If a specific frequency band is given
    if frequency_band is not None:
        # Check if the sampling frequency is specified
        if sampling_rate is None:
            print("When a frequency band is selected, sampling_rate must be provided")
            return False
        # Compute the frequency range
        freqs = scipy.fft.rfftfreq(n=2 * nfft, d=1.0 / sampling_rate)
        # Select the frequency band indices
        inds = (freqs >= frequency_band[0]) & (freqs < frequency_band[1])
        # Slice the vectors
        X = X[:, :, inds]
        inv_Px = inv_Px[:, inds]
    # Compute covariance matrix
    Cxy = compute_Cxy_jit(X, weights, inv_Px)
    # Make Cxy Hermitian
    Cxy = Cxy + np.transpose(Cxy.conj(), axes=[1, 0, 2])
    for i in range(Cxy.shape[0]):
        Cxy[i, i] = 1
    return Cxy


@nb.njit(nogil=True, parallel=True)
def compute_Cxy_jit(X, weights, scale):
    tapers, n_stations, n_samples = X.shape
    Cxy = np.zeros((n_stations, n_stations, n_samples), dtype=nb.complex64)
    Xfc = X.conj()
    for i in nb.prange(n_stations):
        for j in nb.prange(i + 1):
            for k in range(tapers):  # Do not prange this one!
                Cxy[i, j] += weights[k] * X[k, i] * Xfc[k, j]
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
