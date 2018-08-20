
from __future__ import division, print_function
import itertools
import numpy as np
import healpy as hp


def digitize_columns(data, bins):
    digitized = np.empty_like(data, dtype=int)
    for i in range(len(bins)):
        digitized[:, i] = np.digitize(data[:, i], bins=bins[i]) - 1

    return digitized


def binned_skymaps(data, pix, bins, nside=64):
    """ Create skymap of binned data

    Parameters
    ----------
    data : array_like
        Input data to be binned.
    pix : array_like
        Corresponding healpix pixel for each item in data.
    bins : array_like, optional
        Bin edges to use when making binned maps.
    """

    data_digitized = digitize_columns(data, bins=bins)

    shape = [i - 1 for i in map(len, bins)]
    npix = hp.nside2npix(nside)
    shape.append(npix)

    maps = np.zeros(shape, dtype=np.int)
    for bin_idx in itertools.product(*map(range, shape[:-1])):
        in_bins = [data_digitized[:, i] == bin_idx[i] for i in range(data_digitized.shape[1])]
        bin_mask = np.logical_and(*in_bins)

        unique_pix, pix_counts = np.unique(pix[bin_mask], return_counts=True)
        maps[bin_idx][unique_pix] = pix_counts

    return maps
