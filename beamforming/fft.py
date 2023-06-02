import numpy as np
import scipy.fft
import scipy.signal as sp
import xarray as xr


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


def stft(x, nperseg, dim="time", spectral_dim="frequency"):
    fs = get_sampling_rate(x)
    x = x.transpose(..., dim)
    f, t, data = sp.stft(x.values, fs=fs, nperseg=nperseg, padded=False, boundary=None)
    coords = {key: x[key].values for key in x.dims if not key == dim}
    coords.update({spectral_dim: f, dim: t})
    return xr.DataArray(data, coords)


def get_sampling_rate(x):
    d = np.median(np.diff(x["time"].values))
    if np.issubdtype(np.dtype(d), np.timedelta64):
        d = d / np.timedelta64(1, "s")
    return 1.0 / d
