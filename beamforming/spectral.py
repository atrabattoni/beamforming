import numpy as np
import scipy.fft
import scipy.signal as sp
import xarray as xr
from scipy.signal.windows import dpss


def fft(
    x,
    window="hann",
    nperseg=None,
    noverlap=None,
    signal_dim="time",
    spectral_dim="frequency",
):
    axis = x.get_axis_num(signal_dim)
    fs = get_sampling_rate(x, signal_dim)
    if nperseg is None:
        n = scipy.fft.next_fast_len(x.sizes[signal_dim])
        f = scipy.fft.rfftfreq(n=n, d=1.0 / fs)
        win = xr.DataArray(sp.get_window(window, n), dims=signal_dim)
        data = scipy.fft.rfft((win * x).values, n, axis)
    else:
        f, t, data = sp.stft(
            x.values,
            fs,
            window,
            nperseg,
            noverlap,
            boundary=None,
            padded=False,
            axis=axis,
        )
    coords = {
        spectral_dim
        if dim == signal_dim
        else dim: f
        if dim == signal_dim
        else x.coords[dim]
        for dim in x.coords
    }
    dims = tuple(spectral_dim if dim == signal_dim else dim for dim in x.dims)
    if nperseg is not None:
        coords[signal_dim] = t
        dims = dims + (signal_dim,)
    return xr.DataArray(data, coords, dims)


def autocorrelate(
    x,
    window="hann",
    nperseg=None,
    noverlap=None,
    band=None,
    multitaper=None,
    signal_dim="time",
    spectral_dim="frequency",
    sensor_dim="station",
):
    if multitaper is None:
        X = fft(x, window, nperseg, noverlap, signal_dim, spectral_dim)
        if band is not None:
            X = X.sel(frequency=slice(*band))
        return outer(np.conj(X), X, sensor_dim)
    else:
        if nperseg is not None:
            n = nperseg
        else:
            n = scipy.fft.next_fast_len(x.sizes[signal_dim])
        tapers, eigvals = dpss(
            n,
            multitaper / 2,
            multitaper,
            sym=False,
            return_ratios=True,
        )
        return sum(
            autocorrelate(
                x,
                taper,
                nperseg,
                noverlap,
                band,
                None,
                signal_dim,
                spectral_dim,
                sensor_dim,
            )
            * eigval
            / n
            for n, (taper, eigval) in enumerate(zip(tapers, eigvals), start=1)
        )


def outer(x, y, dim):
    n = x.sizes[dim]
    x = x.rename({dim: dim + "_i"})
    y = y.rename({dim: dim + "_j"})
    return x * y / n


def get_sampling_rate(x, dim="time"):
    d = np.median(np.diff(x[dim].values))
    if np.issubdtype(np.dtype(d), np.timedelta64):
        d = d / np.timedelta64(1, "s")
    return 1.0 / d
