import numpy as np

from .multitaper import multitaper_correlate


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

    def beamform(self, x):
        freq, C = multitaper_correlate(
            x,
            n_tapers=self.n_tapers,
            frequency_band=self.frequency_band,
            sampling_rate=self.sampling_rate,
        )
        A = self.get_steering_vector(freq)
        Pr = noise_space_projection(C, A, n_sources=1)
        P = 1.0 / Pr
        P = P.reshape((len(self.azimuth_grid), len(self.speed_grid)))
        return P

    def get_steering_vector(self, freq):
        delay = self.get_delay()
        A = np.exp(2j * np.pi * freq[:, None, None, None] * delay[None, :, :, :])
        return A

    def get_delay(self):
        x, y = self.coords.T
        sx, sy = self.get_grid()
        return sx[:, :, None] * x[None, None, :] + sy[:, :, None] * y[None, None, :]

    def get_grid(self):
        slowness_grid = 1 / self.speed_grid
        sx = -np.sin(self.azimuth_grid)[:, None] * slowness_grid[None, :]
        sy = -np.cos(self.azimuth_grid)[:, None] * slowness_grid[None, :]
        return sx, sy


def noise_space_projection(Rxx, A, n_sources=1):
    A = A.reshape(A.shape[0], -1, A.shape[-1])
    n_freqs, n_slownesses, n_stations = A.shape
    scale = 1.0 / (n_stations * n_freqs)
    Pm = np.zeros(n_slownesses, dtype=complex)
    for f in range(n_freqs):
        Af = A[f]
        _, v = np.linalg.eigh(Rxx[:, :, f])
        un = v[:, : n_stations - n_sources]
        Un = np.dot(un, np.conj(un.T))
        Pm += np.einsum("sn, nk, sk->s", Af.conj(), Un, Af, optimize=True)
    return np.real(Pm) * scale
