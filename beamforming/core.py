import numpy as np
import xarray as xr


from .multitaper import multitaper_correlate


def polar_grid(azimuth, speed):
    azimuth = xr.DataArray(azimuth, {"azimuth": azimuth})
    speed = xr.DataArray(speed, {"speed": speed})
    return xr.Dataset(
        {
            "x": -np.sin(azimuth) / speed,
            "y": -np.cos(azimuth) / speed,
        }
    )


class Beamformer:
    def __init__(
        self,
        coords,
        grid,
        sampling_rate,
        frequency_band,
        n_tapers,
        n_sources,
    ):
        self.coords = coords
        self.grid = grid
        self.frequency_band = frequency_band
        self.sampling_rate = sampling_rate
        self.n_tapers = n_tapers
        self.n_sources = n_sources

    def beamform(self, da):
        C = multitaper_correlate(
            da,
            n_tapers=self.n_tapers,
            frequency_band=self.frequency_band,
            sampling_rate=self.sampling_rate,
        )
        A = self.get_steering_vector(C["frequency"])
        Pr = noise_space_projection(C, A, n_sources=1)
        P = 1.0 / Pr
        return P

    def get_steering_vector(self, freq):
        delay = (self.grid * self.coords).to_array("dimension").sum("dimension")
        return np.exp(2j * np.pi * freq * delay)


def noise_space_projection(C, A, n_sources=1):
    C = C.transpose("station_i", "station_j", "frequency")
    A = A.values
    X = C.values
    n_freqs = A.shape[0]
    n_stations = A.shape[-1]
    scale = 1.0 / (n_stations * n_freqs)
    w, v = np.linalg.eigh(np.transpose(C, [2, 0, 1]))
    un = v[:, :, : n_stations - n_sources]
    Un = un @ np.conj(np.transpose(un, [0, 2, 1]))
    P = np.sum(np.sum(np.conj(A) @ Un[:, None, :, :] * A, axis=-1), axis=0)
    return np.real(P) * scale
