import numpy as np
import xarray as xr
from spectrum import dpss

from .fft import rfft


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
