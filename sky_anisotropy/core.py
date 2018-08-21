
from __future__ import division, print_function
from numbers import Number
import numpy as np
import pandas as pd
import healpy as hp
from scipy.special import erfcinv
from scipy import stats
from dask import delayed

from .regions import on_regions, off_regions


def on_off_chi_squared(binned_maps, pix_center, on_region='disc',
                       size=np.deg2rad(10), off_region='allsky', nside=64,
                       hist_func=None, compute=True):
    """ Calculates chi-squared for binned distributions between on and off regions on the sky

    Parameters
    ----------
    binned_maps : array_like
        Array of Healpix maps. See ``sky_anisotropy.reductions.binned_skymaps``
        for more information.
    pix_center : int, array_like
        Healpix pixels on which to center on-regions.
    on_region : {'disc', 'square'}
        Specifies the on region on the sky to use when calculating chi-squared
        distribution (default is 'disc').
    size : float, optional
        Size (in radians) of on region on sky. Size is the radius for disc on
        regions and 2*size is the size of a side for square on regions
        (default is 0.17 radians, or 10 degrees).
    off_region : {'allsky', 'theta_band'}
        Specifies the off region on the sky to use when calculating chi-squared
        distributions (default is 'allsky').
    nside : float, optional
        Number of sides used for healpix map (default is 64).
    hist_func : function, optional
        Function to map from counts histogram to counts distribution for
        chi-squared calculation. This could be, for instance, a function that
        performs a counts distribution unfolding (default is None).
    compute : boolean, optional
        Whether to compute result or return a dask delayed object instead
        (default is True).

    Returns
    -------
    results : pandas.DataFrame, dask.delayed.Delayed
        DataFrame with information about the distribution comparison between
        the on and off regions. If compute is False, then a dask delayed
        object that represents the DataFrame calculation is returned.
    """
    npix = binned_maps.shape[-1]
    if nside != hp.npix2nside(npix):
        raise ValueError('Inconsistent number of healpy pixels between '
                         'binned_maps and nside.')

    if isinstance(pix_center, Number):
        pix_center = [pix_center]

    binned_maps_delayed = delayed(binned_maps)
    records = [delayed(on_off_chi_squared_single)(binned_maps_delayed,
                                                  pix_center=p,
                                                  on_region=on_region,
                                                  size=size,
                                                  off_region=off_region,
                                                  nside=nside,
                                                  hist_func=hist_func)
               for p in pix_center]
    results = delayed(pd.DataFrame.from_records)(records)

    if compute:
        results = results.compute()

    return results


def on_off_chi_squared_single(binned_maps, pix_center, on_region='disc',
                              size=np.deg2rad(10), off_region='allsky',
                              nside=64, hist_func=None):

    counts_on, counts_on_err, counts_off, counts_off_err = on_off_distributions(
                                                                binned_maps=binned_maps,
                                                                pix_center=pix_center,
                                                                on_region=on_region,
                                                                size=size,
                                                                off_region=off_region,
                                                                nside=nside,
                                                                hist_func=hist_func)

    # Want to make sure off region histogram is scaled to the on region histogram
    alpha = np.sum(counts_on) / np.sum(counts_off)
    scaled_counts_off = alpha * counts_off
    scaled_counts_off_err = alpha * counts_off_err

    # Calculate chi-squared, p-value, and significance
    chi_squared = counts_chi_squared_uncertainties(counts_on,
                                                   counts_on_err,
                                                   scaled_counts_off,
                                                   scaled_counts_off_err)
    # chi_squared = counts_chi_squared(counts_on, scaled_counts_off)
    ndof = counts_on.shape[0]
    pval = stats.chi2.sf(chi_squared, ndof)
    sig = erfcinv(2 * pval) * np.sqrt(2)

    result = {'pix_center': pix_center,
              'alpha': alpha,
              'num_on':  np.sum(counts_on),
              'chi2': chi_squared,
              'pval': pval,
              'sig': sig,
              'ndof': ndof,
              }

    return result


def on_off_distributions(binned_maps, pix_center, on_region='disc',
                         size=np.deg2rad(10), off_region='allsky', nside=64,
                         hist_func=None):
    # Construct on region mask
    on_region_pix = on_regions[on_region](pix_center,
                                          size=size,
                                          nside=nside)
    # Construct off region mask
    off_region_pix = off_regions[off_region](on_region_pix,
                                             nside=nside)

    counts_on = binned_maps[..., on_region_pix].sum(axis=-1)
    counts_off = binned_maps[..., off_region_pix].sum(axis=-1)

    if hist_func is not None:
        counts_on, counts_on_err = hist_func(counts_on)
        counts_off, counts_off_err = hist_func(counts_off)
    else:
        # Assume Poisson counting errors
        counts_on_err = np.sqrt(counts_on)
        counts_off_err = np.sqrt(counts_off)

    if 0 in counts_on:
        raise ValueError('Binned distribution for on region centered at pixel '
                         '{} has zero counts in a bin'.format(pix_center))
    if 0 in counts_off:
        raise ValueError('Binned distribution for off region centered at pixel '
                         '{} has zero counts in a bin'.format(pix_center))

    return counts_on, counts_on_err, counts_off, counts_off_err


def counts_chi_squared(counts_on, counts_off):
    """ Calculates reduced chi-squared between two energy histograms

    Parameters
    ----------
    counts_on : numpy.ndarray
        Energy distribution inside disc.
    counts_off : numpy.ndarray
        Energy distribution outside disc. Note that counts_off should be scaled
        to have the same total number of counts as counts_on.

    Returns
    -------
    chi_squared : float
        Chi-squared between two input distributions.
    """

    assert counts_on.shape == counts_off.shape
    np.testing.assert_allclose(np.sum(counts_on), np.sum(counts_off))
    chi_squared = 2 * np.sum(counts_off - counts_on + (counts_on * np.log(counts_on / counts_off)))
    return chi_squared


def counts_chi_squared_uncertainties(counts_on, counts_on_err, counts_off,
                                     counts_off_err):
    """ Calculates reduced chi-squared between two energy histograms

    Parameters
    ----------
    counts_on : numpy.ndarray
        Energy distribution inside disc.
    counts_off : numpy.ndarray
        Energy distribution outside disc. Note that counts_off should be scaled
        to have the same total number of counts as counts_on.

    Returns
    -------
    chi_squared : float
        Chi-squared between two input distributions.
    """

    assert counts_on.shape == counts_off.shape
    np.testing.assert_allclose(np.sum(counts_on), np.sum(counts_off))
    chi_squared = np.sum((counts_on - counts_off) ** 2 / counts_on_err ** 2)

    return chi_squared
