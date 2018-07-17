
import itertools
from collections import OrderedDict
import sky_anisotropy as sa
import numpy as np
import xarray as xr
import healpy as hp

from dask.diagnostics import ProgressBar
import dask.array as da
import dask.dataframe as dd
from dask import compute


def digitize_columns(partition, col_to_bins):
    for col in col_to_bins.keys():
        bins = col_to_bins[col]
        partition[col + '_digitized'] = np.digitize(partition[col], bins=bins) - 1
    return partition


def ang2pix(theta, phi, nside=64):
    return hp.ang2pix(nside=nside, theta=theta, phi=phi)


def binned_skymaps(ddf, col_to_bins, ra_col=None, dec_col=None, nside=64,
                   num_workers=1, scheduler='threading', **compute_kwargs):
    """ Calculate binned skymaps for input data

    Parameters
    ----------
    ddf : dask.dataframe.DataFrame
        Input data.
    col_to_bins : collections.OrderedDict
        Dictionary with mapping between columns in ddf and bin edges.
    ra_col : str
        Column name in ddf with right ascension values (in radians).
    dec_col : str
        Column name in ddf with declination values (in radians).
    nside : int, optional
        Number of sides used for healpix map (default is 64).
    num_workers : int, optional
        Number of processes or threads to use (default is 1).
    scheduler : str, optional
        Dask scheduler to use (default is 'threading').
    compute_kwargs : dict
        Optional keyword arguments to pass to Dask's compute() function.

    Returns
    -------
    data : xarray.DataArray
    """

    if not all([ra_col, dec_col]):
        raise ValueError('Both ra_col and dec_col must not be None')

    if not isinstance(ddf, dd.DataFrame):
        raise TypeError('ddf must be a dask DataFrame')

    if not isinstance(col_to_bins, OrderedDict):
        raise TypeError('col_to_bins must be an instance of collections.OrderedDict')

    npix = hp.nside2npix(nside)

    # Get bin bin index for each column
    ddf_digitized = ddf.map_partitions(digitize_columns, col_to_bins)
    shape = list((len(bins)-1 for bins in col_to_bins.values()))
    bin_idxs = [np.arange(l) for l in shape]

    # Compute skymaps for each unique bin combination
    cols = col_to_bins.keys()
    maps = []
    for idx in itertools.product(*bin_idxs):
        bool_masks = list(ddf_digitized['{}_digitized'.format(col)] == i
                          for col, i in zip(cols, idx))
        if len(bool_masks) == 1:
            mask = bool_masks[0]
        else:
            mask = da.logical_and(*bool_masks)
        theta, phi = sa.equatorial_to_healpy(ddf.loc[mask, ra_col],
                                             ddf.loc[mask, dec_col])
        ipix = da.map_blocks(ang2pix, theta.values, phi.values)
        m, _ = da.histogram(ipix, bins=np.arange(npix + 1))
        maps.append(m)

    with ProgressBar():
        maps = compute(*maps,
                       num_workers=num_workers,
                       scheduler=scheduler,
                       **compute_kwargs)

    # Format maps into an xarray.DataArray
    data = np.zeros(shape + [npix])
    for idx, m in zip(itertools.product(*bin_idxs), maps):
        data[idx] = m
    dims = col_to_bins.keys() + ['ipix']

    coords = {}
    for col in col_to_bins.keys():
        bins = col_to_bins[col]
        coords[col] = (bins[1:] + bins[:-1]) / 2

    data = xr.DataArray(data, dims=dims, coords=coords)

    return data
