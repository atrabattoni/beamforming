import numpy as np
import scipy.fft
import xarray as xr
from spectrum import dpss


def rfft(da):
    n = scipy.fft.next_fast_len(da.sizes["time"])
    d = np.median(np.diff(da["time"].values))
    if np.issubdtype(np.dtype(d), np.datetime64):
        d = d / np.timedelta64(1, "s")
    freq = scipy.fft.rfftfreq(n=n, d=d)
    data = scipy.fft.rfft(da.values, n, da.get_axis_num("time"))
    coords = {
        "frequency" if dim == "time" else dim: freq if dim == "time" else coords[dim]
        for dim in da.coords
    }
    dims = tuple("frequency" if dim == "time" else dim for dim in da.dims)
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
