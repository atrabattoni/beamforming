import numpy as np
import xarray as xr
from spectrum import dpss

from .fft import rfft, stft


class Beamformer:
    def get_delay(self):
        return (self.grid * self.coords).to_array("dimension").sum("dimension")

    def get_steering_vector(self, freq):
        delay = self.get_delay()
        return np.exp(-2j * np.pi * freq * delay)


class MultitaperBeamformer(Beamformer):
    def __init__(
        self,
        coords,
        grid,
        frequency_band,
        adaptative,
        n_tapers,
        n_sources,
    ):
        self.coords = coords
        self.grid = grid
        self.frequency_band = frequency_band
        self.adaptative = adaptative
        self.n_tapers = n_tapers
        self.n_sources = n_sources

    def __call__(self, x):
        if self.adaptative:
            R = multitaper_correlate(
                x,
                n_tapers=self.n_tapers,
                frequency_band=self.frequency_band,
            )
            v = self.get_steering_vector(R["frequency"])
            P = music(R, v, n_sources=1)
            P = xr.DataArray(P, self.grid.coords)
        else:
            X = rfft(x)
            v = self.get_steering_vector(X["frequency"])
            P = (np.abs((np.conj(v) * X).sum("station")) ** 2).sum("frequency")
        return P


class SlidingBeamformer(Beamformer):
    def __init__(self, coords, grid, frequency_band, nperseg):
        self.coords = coords
        self.grid = grid
        self.frequency_band = frequency_band
        self.nperseg = nperseg

    def __call__(self, x):
        X = stft(x, self.nperseg)
        X = X.sel(frequency=slice(*self.frequency_band))
        v = self.get_steering_vector(X["frequency"])
        Y = xr.dot(np.conj(v), X, dims=["station"])
        return (np.real(np.conj(Y) * Y)).sum("frequency")


class CorrBeamformer(Beamformer):
    def __init__(self, coords, grid, frequency_band, nperseg, mode):
        self.coords = coords
        self.grid = grid
        self.frequency_band = frequency_band
        self.nperseg = nperseg
        self.mode = mode

    def __call__(self, x):
        X = stft(x, self.nperseg)
        X = X.sel(frequency=slice(*self.frequency_band))
        R = outer(np.conj(X), X, "station")
        R = R.sum("time")

        v = self.get_steering_vector(X["frequency"])
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


# Linear algebra


def hermitian_transpose(x):
    return np.conj(np.swapaxes(x, -1, -2))


def outer(x, y, dim):
    x = x.rename({dim: dim + "_i"})
    y = y.rename({dim: dim + "_j"})
    return x * y


# Multi-taper


def multitaper_correlate(da, n_tapers, frequency_band):
    weight, da = multitaper(da, n_tapers)
    da = rfft(da)
    if frequency_band is not None:
        da = da.sel(frequency=slice(*frequency_band))
    return correlate(da, weight)


def correlate(da, weight):
    norm = (np.real(da * np.conj(da)) * weight).sum(weight.dims)
    C = (
        da.rename({"station": "station_i"})
        * np.conj(da.rename({"station": "station_j"}))
        * weight
    ).sum(weight.dims)
    C = (
        C
        * norm.rename({"station": "station_i"})
        * norm.rename({"station": "station_j"})
    )
    return C


def multitaper(da, n_tapers):
    taper, eigval = dpss(da.sizes["time"], n_tapers)
    taper = xr.DataArray(taper, dims=("time", "taper"))
    weight = xr.DataArray(eigval / (np.arange(len(eigval)) + 1), dims="taper")
    return weight, da * taper


# Music


def music(C, A, n_sources=1):
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
    P = np.real(P) * scale
    return 1.0 / P
