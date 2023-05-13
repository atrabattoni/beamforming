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
        n_sources,
    ):
        self.coords = coords
        self.azimuth_grid = azimuth_grid
        self.speed_grid = speed_grid
        self.n_samples = n_samples
        self.frequency_band = frequency_band
        self.sampling_rate = sampling_rate
        self.n_tapers = n_tapers
        self.n_sources = n_sources

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
        return P

    def get_steering_vector(self, freq):
        delay = self.get_delay()
        return np.exp(2j * np.pi * freq[:, None, None, None] * delay[None, :, :, :])

    def get_delay(self):
        x, y = self.coords.T
        sx, sy = self.get_grid()
        return sx[:, :, None] * x[None, None, :] + sy[:, :, None] * y[None, None, :]

    def get_grid(self):
        slowness_grid = 1 / self.speed_grid
        sx = -np.sin(self.azimuth_grid)[:, None] * slowness_grid[None, :]
        sy = -np.cos(self.azimuth_grid)[:, None] * slowness_grid[None, :]
        return sx, sy


def noise_space_projection(C, A, n_sources=1):
    n_freqs = A.shape[0]
    n_stations = A.shape[-1]
    scale = 1.0 / (n_stations * n_freqs)
    Pm = np.zeros(A.shape[1:-1], dtype="complex")
    for idx in range(n_freqs):
        _, v = np.linalg.eigh(C[:, :, idx])
        un = v[:, : n_stations - n_sources]
        Un = np.dot(un, np.conj(un.T))
        Pm += np.sum(np.conj(A[idx]) @ Un[None, :, :] * A[idx], axis=-1)
    return np.real(Pm) * scale
