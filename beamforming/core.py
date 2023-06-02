from dataclasses import dataclass

import numpy as np
import xarray as xr

from .spectral import autocorrelate, fft


@dataclass
class Beamformer:
    coords: xr.Dataset
    grid: xr.Dataset
    window: str | None = "hann"
    nperseg: int | None = None
    noverlap: int | None = None
    band: tuple[float] | None = None
    welch: bool = False
    signal_dim: str = "time"
    spectral_dim: str = "frequency"
    sensor_dim: str = "station"

    def __call__(self, x):
        X = fft(
            x,
            self.window,
            self.nperseg,
            self.noverlap,
            self.signal_dim,
            self.spectral_dim,
        )
        if self.band is not None:
            X = X.sel({self.spectral_dim: slice(*self.band)})
        v = self.get_steering_vector(X[self.spectral_dim])
        P = xr.dot(np.conj(v), X, dims=[self.sensor_dim])
        P = (np.real(np.conj(P) * P)).sum(self.spectral_dim)
        if self.welch and self.nperseg is not None:
            P = P.sum(self.signal_dim)
        return P

    def get_delay(self):
        return (self.grid * self.coords).to_array("dimension").sum("dimension")

    def get_steering_vector(self, f):
        delay = self.get_delay()
        return np.exp(-2j * np.pi * f * delay)


@dataclass
class AdaptativeBeamformer(Beamformer):
    mode: str = "capon"
    multitaper: int | None = 5

    def __call__(self, x):
        R = autocorrelate(
            x,
            self.window,
            self.nperseg,
            self.noverlap,
            self.band,
            self.multitaper,
            self.signal_dim,
            self.spectral_dim,
            self.sensor_dim,
        )
        if self.welch and self.nperseg is not None:
            R = R.sum(self.signal_dim)

        v = self.get_steering_vector(R["frequency"])
        vh = np.conj(v).rename({"station": "station_j"})
        v = v.rename({"station": "station_i"})

        if self.mode == "bartlett":
            P = np.real(xr.dot(vh, R, v, dims=("station_i", "station_j")))
        elif self.mode == "capon":
            R = R.transpose(..., "station_i", "station_j")
            data = np.linalg.pinv(R.values, hermitian=True)
            Rinv = R.copy(data=data)
            P = 1.0 / np.real(xr.dot(vh, Rinv, v, dims=("station_i", "station_j")))
        elif self.mode == "music":
            raise NotImplementedError
            # R = R.transpose(..., "station_i", "station_j")
            # s, u = np.linalg.eigh(R.values)
            # data = u @ (hermitian_transpose(u) / s[..., None])
            # Rinv = R.copy(data=data)
            # P = 1.0 / np.real(xr.dot(vh, Rinv, v, dims=("station_i", "station_j")))
            # data = Un @ hermitian_transpose(Un)
            # Y = R.copy(data=data)
            # P = 1.0 / np.real(xr.dot(vh, Y, v, dims=("station_i", "station_j")))
        return P.sum("frequency")
