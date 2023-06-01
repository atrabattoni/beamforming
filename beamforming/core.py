import numpy as np
import scipy.fft
import scipy.signal as sp
import xarray as xr
from spectrum import dpss


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
        adaptative,
        n_tapers,
        n_sources,
    ):
        self.coords = coords
        self.grid = grid
        self.frequency_band = frequency_band
        self.sampling_rate = sampling_rate
        self.adaptative = adaptative
        self.n_tapers = n_tapers
        self.n_sources = n_sources

    def beamform(self, x):
        if self.adaptative:
            C = multitaper_correlate(
                x,
                n_tapers=self.n_tapers,
                frequency_band=self.frequency_band,
                sampling_rate=self.sampling_rate,
            )
            A = self.get_steering_vector(C["frequency"])
            Pr = noise_space_projection(C, A, n_sources=1)
            P = 1.0 / Pr
            return P
        else:
            X = rfft(x)
            A = self.get_steering_vector(X["frequency"])
            P = (np.abs((A.conj() * X).sum("station")) ** 2).sum("frequency")
            return P

    def get_steering_vector(self, freq):
        delay = (self.grid * self.coords).to_array("dimension").sum("dimension")
        return np.exp(2j * np.pi * freq * delay)


class SlidingBeamformer:
    def __init__(self, coords, grid, frequency_band, nperseg):
        self.coords = coords
        self.grid = grid
        self.frequency_band = frequency_band
        self.nperseg = nperseg

    def beamform(self, x):
        X = stft(x, self.nperseg)
        v = self.get_steering_vector(X)
        Y = xr.dot(np.conj(v), X, dims=["station"])
        return (np.real(np.conj(Y) * Y)).sum("frequency")

    def get_steering_vector(self, X):
        delay = (self.grid * self.coords).to_array("dimension").sum("dimension")
        v = np.exp(2j * np.pi * X["frequency"] * delay)
        return v


def stft(x, nperseg, dim="time", spectral_dim="frequency"):
    fs = get_sampling_rate(x)
    x = x.transpose(..., dim)
    f, t, data = sp.stft(x.values, fs=fs, nperseg=nperseg, padded=False, boundary=None)
    coords = {key: x[key].values for key in x.dims if not key == dim}
    coords.update({spectral_dim: f, dim: t})
    return xr.DataArray(data, coords)


def dot(x, y, dim):
    x = x.transpose(..., dim)
    y = y.transpose(..., dim)
    x, y = xr.broadcast(x, y)
    data = np.squeeze(
        np.conj(y.values[..., None, :]) @ x.values[..., :, None], axis=(-1, -2)
    )
    coords = {key: x[key] for key in x.dims if not key == dim}
    return xr.DataArray(data, coords)


def get_sampling_rate(x):
    d = np.median(np.diff(x["time"].values))
    if np.issubdtype(np.dtype(d), np.timedelta64):
        d = d / np.timedelta64(1, "s")
    return 1.0 / d


def rfft(x):
    n = scipy.fft.next_fast_len(x.sizes["time"])
    fs = get_sampling_rate(x)
    freq = scipy.fft.rfftfreq(n=n, d=1.0 / fs)
    data = scipy.fft.rfft(x.values, n, x.get_axis_num("time"))
    coords = {
        "frequency" if dim == "time" else dim: freq if dim == "time" else x.coords[dim]
        for dim in x.coords
    }
    dims = tuple("frequency" if dim == "time" else dim for dim in x.dims)
    return xr.DataArray(data, coords, dims)


def multitaper_correlate(da, n_tapers, frequency_band, sampling_rate):
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
