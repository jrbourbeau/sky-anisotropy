
from __future__ import division, print_function
from numbers import Number
import numpy as np
import pandas as pd
import healpy as hp
from scipy.special import erfcinv
from scipy import stats
from dask import delayed

from .coordinates import healpy_to_equatorial


def on_off_chi_squared(values, pix, pix_center, on_region='disc',
                       size=np.deg2rad(10), off_region='allsky', nside=64,
                       bins=None, compute=True):
    """ Calculates chi-squared for binned distributions between on and off regions on the sky

    Parameters
    ----------
    values : array_like
        Input values to be binned.
    pix : array_like
        Corresponding healpix pixel for each value in values.
    pix_center : int, array_like
        Healpix pixels on which to center on-regions.
    on_region : {'disc', 'square'}
        Specifies the on region on the sky to use when calculating chi-squared
        distribution (default is 'disc').
    size : float, optional
        Size (in radians) of on region on sky. Size is the radius for disc on
        regions and 2*size is the size of a side for square on regions
        (default is 0.17 radians, or 10 degrees).
    off_region : {'allsky', 'theta_band', 'opposite'}
        Specifies the off region on the sky to use when calculating chi-squared
        distributions (default is 'allsky').
    nside : float, optional
        Number of sides used for healpix map (default is 64).
    bins : array_like, optional
        Bin edges to use when making binned values disbtritutions (default is
        numpy.linspace(values.min(), values.max(), 20)).
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

    values = np.asarray(values)
    pix = np.asarray(pix)
    if values.shape != pix.shape:
        raise ValueError('values and pix must have the same shape, but got '
                         '{} and {}'.format(values.shape, pix.shape))
    if isinstance(pix_center, Number):
        pix_center = [pix_center]
    records = [delayed(on_off_chi_squared_single)(values, pix,
                                                  pix_center=p,
                                                  on_region=on_region,
                                                  size=size,
                                                  off_region=off_region,
                                                  nside=nside,
                                                  bins=bins)
               for p in pix_center]
    results = delayed(pd.DataFrame.from_records)(records)

    if compute:
        results = results.compute()

    return results


def on_off_chi_squared_single(values, pix, pix_center, on_region='disc',
                              size=np.deg2rad(10), off_region='allsky',
                              nside=64, bins=None):
    # Construct on region mask
    in_on_region = on_region_func(on_region)(pix, pix_center,
                                             size=size,
                                             nside=nside)
    # Construct off region mask
    in_off_region = off_region_func(off_region)(pix, pix_center, in_on_region,
                                                nside=nside)
    # Value distributions for on and off regions
    if bins is None:
        bins = np.linspace(values.min(), values.max(), 20)
    counts_on, _ = np.histogram(values[in_on_region], bins=bins)
    counts_off, _ = np.histogram(values[in_off_region], bins=bins)

    if np.isin([counts_on, counts_off], 0).any():
        raise ValueError('Binned distribution has zero counts in a bin')

    # Want to make sure off region histogram is scaled to the on region histogram
    alpha = np.sum(counts_on) / np.sum(counts_off)
    scaled_counts_off = alpha * counts_off

    # Calculate chi-squared, p-value, and significance
    chi_squared = counts_chi_squared(counts_on, scaled_counts_off)
    ndof = len(bins) - 1
    pval = stats.chi2.sf(chi_squared, ndof)
    sig = erfcinv(2 * pval) * np.sqrt(2)

    result = {'pix_center': pix_center,
              'alpha': alpha,
              'num_on':  np.sum(counts_on),
              'chi2': chi_squared,
              'pval': pval,
              'sig': sig,
              }

    return result


def disc_on_region(pix, pix_center, size=np.deg2rad(10), nside=64):
    """ Circular on region
    """

    vec_disc = hp.pix2vec(nside=nside, ipix=pix_center)
    pix_in_disc = hp.query_disc(nside=nside, vec=vec_disc, radius=size)
    in_on_region = np.isin(pix, pix_in_disc)

    return in_on_region


def square_on_region(pix, pix_center, size=np.deg2rad(10), nside=64):
    """ Square on region
    """

    theta_center, phi_center = hp.pix2ang(nside=nside, ipix=pix_center)
    ra_center, dec_center = healpy_to_equatorial(theta_center, phi_center)
    theta, phi = hp.pix2ang(nside=nside, ipix=pix)
    theta_mask = np.logical_and(theta <= theta_center + size,
                                theta >= theta_center - size)
    size_phi = size / np.cos(dec_center)
    phi_mask = np.logical_and(phi <= phi_center + size_phi,
                              phi >= phi_center - size_phi)
    in_on_region = theta_mask & phi_mask

    return in_on_region


on_region_funcs = {'disc': disc_on_region,
                   'square': square_on_region,
                   }


def on_region_func(name):
    try:
        return on_region_funcs[name]
    except KeyError:
        raise ValueError('Invalid on_region entered ({}). Must be either '
                         '"disc" or "square".'.format(name))


def allsky_off_region(pix, pix_center, on_region_mask, nside=64):
    """ All sky off region
    """

    return ~on_region_mask


def theta_band_off_region(pix, pix_center, on_region_mask, nside=64):
    pix_in_disc = pix[on_region_mask]
    theta_in_disc, _ = hp.pix2ang(nside=nside, ipix=pix_in_disc)
    theta_disc_min = theta_in_disc.min()
    theta_disc_max = theta_in_disc.max()

    theta, phi = hp.pix2ang(nside=nside, ipix=pix)
    theta_band_mask = np.logical_and(theta <= theta_disc_max,
                                     theta >= theta_disc_min)
    off_region_mask = np.logical_and(~on_region_mask, theta_band_mask)

    return off_region_mask


def opposite_off_region(pix, pix_center, on_region_mask, nside=64):
    on_region_pix = pix[on_region_mask]
    theta_on, phi_on = hp.pix2ang(nside=nside, ipix=on_region_pix)
    phi_off = phi_on + np.pi
    theta_off = theta_on
    off_region_pix = hp.ang2pix(nside=nside, theta=theta_off, phi=phi_off)
    off_region_mask = np.isin(pix, off_region_pix)

    return off_region_mask


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


off_region_funcs = {'allsky': allsky_off_region,
                    'theta_band': theta_band_off_region,
                    'opposite': opposite_off_region,
                    }


def off_region_func(name):
    try:
        return off_region_funcs[name]
    except KeyError:
        raise ValueError('Invalid off_region entered ({}). Must be either '
                         '"allsky", "theta_band", or "opposite".'.format(name))
