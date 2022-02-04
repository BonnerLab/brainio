import os
from typing import Hashable, Union

import netCDF4
import xarray as xr
from xarray import DataArray
from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE


def extend_netcdf(da: DataArray, path: Union[str, os.PathLike], extending_dim: Hashable) -> None:
    da = _wrap_as_dataset(da)

    # If the file doesn't already exist, create it
    if not os.path.exists(path):
        da.to_netcdf(path, unlimited_dims=[extending_dim])
        return

    # Otherwise, extend the existing file
    with netCDF4.Dataset(path, mode='a') as nc:
        nc_shape = nc.dimensions[extending_dim].size
        added_size = len(da[extending_dim])
        variables, attrs = xr.conventions.encode_dataset_coordinates(da)
        for name, data in variables.items():
            if extending_dim not in data.dims:
                # Nothing to extend along this data's dimensions
                continue
            nc_variable = nc[name]
            _expand_netcdf4_variable(nc_variable, data, extending_dim, nc_shape, added_size)


def _wrap_as_dataset(da: DataArray) -> xr.Dataset:
    if da.name is None:
        # If no name is set then use a generic xarray name
        da = da.to_dataset(name=DATAARRAY_VARIABLE)
    elif da.name in da.coords or da.name in da.dims:
        # The name is the same as one of the coords names, which netCDF
        # doesn't support, so rename it but keep track of the old name
        da = da.to_dataset(name=DATAARRAY_VARIABLE)
        da.attrs[DATAARRAY_NAME] = da.name
    else:
        # No problems with the name - so we're fine!
        da = da.to_dataset()
    return da


def _expand_netcdf4_variable(nc_variable, data, expanding_dim, nc_shape, added_size):
    data_encoded = xr.conventions.encode_cf_variable(data)
    left_slices = data.dims.index(expanding_dim)
    right_slices = data.ndim - left_slices - 1
    nc_slice = (slice(None),) * left_slices + \
               (slice(nc_shape, nc_shape + added_size),) + \
               (slice(None),) * right_slices
    nc_variable[nc_slice] = data_encoded.data
