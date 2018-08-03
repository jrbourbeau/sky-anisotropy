
from __future__ import division, print_function
from numbers import Number
import numpy as np
import pandas as pd
import healpy as hp
from scipy.special import erfcinv
from scipy import stats
from dask import delayed

from .coordinates import healpy_to_equatorial


def on_off_chi_squared(samples, pix, pix_center, on_region='disc',
                       size=np.deg2rad(10), off_region='allsky', nside=64,
                       bins=None, hist_func=None, compute=True):
    """ Calculates chi-squared for binned distributions between on and off regions on the sky

    Parameters
    ----------
    samples : array_like
        Input samples to be passed to numpy.histogramdd.
    pix : array_like
        Corresponding healpix pixel for each item in samples.
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
        Bin edges to use when making binned samples disbtritutions (default is
        numpy.linspace(samples.min(), samples.max(), 20)).
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

    samples = np.asarray(samples)
    pix = np.asarray(pix)
    if samples.shape[0] != pix.shape[0]:
        raise ValueError('samples and pix must have the same shape, but got '
                         '{} and {}'.format(samples.shape, pix.shape))
    if isinstance(pix_center, Number):
        pix_center = [pix_center]
    records = [delayed(on_off_chi_squared_single)(samples, pix,
                                                  pix_center=p,
                                                  on_region=on_region,
                                                  size=size,
                                                  off_region=off_region,
                                                  nside=nside,
                                                  bins=bins,
                                                  hist_func=hist_func)
               for p in pix_center]
    results = delayed(pd.DataFrame.from_records)(records)

    if compute:
        results = results.compute()

    return results


def on_off_chi_squared_single(samples, pix, pix_center, on_region='disc',
                              size=np.deg2rad(10), off_region='allsky',
                              nside=64, bins=None, hist_func=None):
    # Construct on region mask
    in_on_region = on_region_func(on_region)(pix, pix_center,
                                             size=size,
                                             nside=nside)
    # Construct off region mask
    in_off_region = off_region_func(off_region)(pix, pix_center, in_on_region, size,
                                                nside=nside)
    # Value distributions for on and off regions
    if bins is None:
        bins = np.linspace(samples.min(), samples.max(), 20)
    if isinstance(bins, np.ndarray) and bins.ndim == 1:
        bins = [bins]

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    counts_on, _ = np.histogramdd(samples[in_on_region], bins=bins)
    counts_off, _ = np.histogramdd(samples[in_off_region], bins=bins)

    if hist_func is not None:
        counts_on = hist_func(counts_on)
        counts_off = hist_func(counts_off)

    if 0 in counts_on:
        raise ValueError('Binned distribution for on region centered at pixel '
                         '{} has zero counts in a bin'.format(pix_center))
    if 0 in counts_off:
        raise ValueError('Binned distribution for off region centered at pixel '
                         '{} has zero counts in a bin'.format(pix_center))

    # Want to make sure off region histogram is scaled to the on region histogram
    alpha = np.sum(counts_on) / np.sum(counts_off)
    scaled_counts_off = alpha * counts_off

    # Calculate chi-squared, p-value, and significance
    chi_squared = counts_chi_squared(counts_on, scaled_counts_off)
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


def disc_on_region(pix, pix_center, size=np.deg2rad(10), nside=64):
    """ Circular on region
    """

    vec_disc = hp.pix2vec(nside=nside, ipix=pix_center)
    pix_in_disc = hp.query_disc(nside=nside, vec=vec_disc, radius=size)
    in_on_region = np.isin(pix, pix_in_disc)

    return in_on_region


def normalize_angle(x):
    x = (x + 2 * np.pi) % (2 * np.pi)
    return x


def square_on_region(pix, pix_center, size=np.deg2rad(10), nside=64):
    """ Square on region
    """
    theta_center, phi_center = hp.pix2ang(nside=nside, ipix=pix_center)
    ra_center, dec_center = healpy_to_equatorial(theta_center, phi_center)
    theta, phi = hp.pix2ang(nside=nside, ipix=pix)
    phi = normalize_angle(phi)
    theta_mask = np.logical_and(theta <= theta_center + size,
                                theta >= theta_center - size)
    size_phi = size / np.cos(dec_center)
    phi_upper = phi_center + size_phi
    phi_upper = normalize_angle(phi_upper)
    phi_lower = phi_center - size_phi
    phi_lower = normalize_angle(phi_lower)
    if (phi_center > size_phi and phi_center < 2 * np.pi - size_phi):
        phi_mask = np.logical_and(phi <= phi_upper, phi >= phi_lower)
    else:
        phi_mask = ~np.logical_and(phi >= phi_upper, phi <= phi_lower)
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


def allsky_off_region(pix, pix_center, on_region_mask, size, nside=64):
    """ All sky off region
    """

    return ~on_region_mask


def theta_band_off_region(pix, pix_center, on_region_mask, size, nside=64):
    pix_in_disc = pix[on_region_mask]
    theta_in_disc, _ = hp.pix2ang(nside=nside, ipix=pix_in_disc)
    theta_disc_min = theta_in_disc.min()
    theta_disc_max = theta_in_disc.max()

    theta, phi = hp.pix2ang(nside=nside, ipix=pix)
    theta_band_mask = np.logical_and(theta <= theta_disc_max,
                                     theta >= theta_disc_min)
    off_region_mask = np.logical_and(~on_region_mask, theta_band_mask)

    return off_region_mask


def opposite_off_region(pix, pix_center, on_region_mask, size, nside=64):
    on_region_pix = pix[on_region_mask]
    theta_on, phi_on = hp.pix2ang(nside=nside, ipix=on_region_pix)
    phi_off = phi_on + np.pi
    theta_off = theta_on
    off_region_pix = hp.ang2pix(nside=nside, theta=theta_off, phi=phi_off)
    off_region_mask = np.isin(pix, off_region_pix)

    return off_region_mask


def disc_theta_band_off_region(pix, pix_center, on_region_mask, size=np.radians(10), nside=64):
    theta_center, phi_center = hp.pix2ang(nside, pix_center)
    num_disc = np.pi * np.sin(theta_center) // size + 1
    num_disc = int(num_disc)
    phi_disc_cent = np.linspace(phi_center, phi_center + 2 * np.pi, num_disc)
    theta_disc_cent = np.full(num_disc, theta_center)

    disc_vec = hp.ang2vec(theta_disc_cent, phi_disc_cent)
    disc_list = []
    for i in range(1, len(disc_vec) - 1):
        discs = hp.query_disc(nside, disc_vec[i, :], size)
        disc_list.append(discs)
    disc_arr = np.concatenate(disc_list)
    in_off_region = np.isin(pix, disc_arr)

    return in_off_region


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
                    'disc_theta_band': disc_theta_band_off_region
                    }


def off_region_func(name):
    try:
        return off_region_funcs[name]
    except KeyError:
        raise ValueError('Invalid off_region entered ({}). Must be either '
                         '"allsky", "theta_band", or "opposite".'.format(name))
