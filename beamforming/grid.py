import numpy as np
import xarray as xr


def polar_grid(azimuth, speed):
    azimuth = xr.DataArray(azimuth, {"azimuth": azimuth})
    speed = xr.DataArray(speed, {"speed": speed})
    return xr.Dataset(
        {
            "x": -np.sin(azimuth) / speed,
            "y": -np.cos(azimuth) / speed,
        }
    )
