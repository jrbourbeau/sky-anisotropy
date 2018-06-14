
from __future__ import division, print_function
import sys
from numbers import Number
import numpy as np
import pandas as pd
import healpy as hp
from scipy.special import erfcinv
from scipy import stats
from dask import delayed
from dask.diagnostics import ProgressBar


def equatorial_to_healpy(ra, dec):
    """Convert equatorial coordinates (ra, dec) to healpy coordinates (theta, phi)

    Parameters
    ----------
    ra : float, array_like
        Right ascension coordinates in radians.
    dec : float, array_like
        Declination coordinates in radians.

    Returns
    -------
    theta : float, array_like
        Healpy theta coordinates in radians.
    phi : float, array_like
        Healpy phi coodinates in radians.
    """
    theta = np.pi/2 - dec
    phi = ra

    return theta, phi


def healpy_to_equatorial(theta, phi):
    """Convert healpy coordinates (theta, phi) to equatorial coordinates (ra, dec)

    Parameters
    ----------
    theta : float, array_like
        Healpy theta coordinates in radians.
    phi : float, array_like
        Healpy phi coodinates in radians.

    Returns
    -------
    ra : float, array_like
        Right ascension coordinates in radians.
    dec : float, array_like
        Declination coordinates in radians.
    """
    dec = np.pi/2 - theta
    ra = phi

    return ra, dec


def disc_chi_squared(values, pix, pix_disc, radius=np.deg2rad(10), nside=64,
                     theta_band=False, bins=None, n_jobs=1, verbose=True):
    """Calculates chi-squared for binned distributions between on and off regions on the sky

    Parameters
    ----------
    values : array_like
        Input values to be binned.
    pix : array_like
        Corresponding healpix pixel for each value in values.
    pix_disc : int, array_like
        Healpix pixels on which to center on-region discs.
    radius : float, optional
        Radius (in radians) of on region disc on sky (default is 0.17 radians,
        or 10 degrees).
    nside : float, optional
        Number of sides used for healpix map (default is 64).
    theta_band : bool, optional
        Option to use full sky off region, or restrict the off region to be
        within the same theta band of the on region (default is False,
        will use full sky off region).
    bins : array_like, optional
        Bin edges to use when making binned values disbtritutions (default is
        numpy.linspace(values.min(), values.max(), 20)).
    n_jobs : int, optional
        Number of jobs to run in parallel (default is 1).
    verbose : bool, optional
        Option for verbose output (default is True).

    Returns
    -------
    result : pandas.DataFrame
        DataFrame with information about the distribution comparison between
        the on and off regions.
    """
    values = np.asarray(values)
    pix = np.asarray(pix)
    if values.shape != pix.shape:
        raise ValueError('values and pix must have the same shape, but got '
                         '{} and {}'.format(values.shape, pix.shape))
    if isinstance(pix_disc, Number):
        pix_disc = [pix_disc]
    records = [delayed(disc_chi_squared_single)(values, pix, pix_disc=p,
                                                radius=radius, nside=nside,
                                                theta_band=theta_band, bins=bins)
               for p in pix_disc]
    results = delayed(pd.DataFrame.from_records)(records)
    scheduler = 'threads' if n_jobs > 1 else 'sync'
    if verbose:
        msg = 'Calculating chi-squared values for {} regions\n'.format(len(pix_disc))
        sys.stdout.write(msg)
        with ProgressBar():
            results = results.compute(scheduler=scheduler, num_workers=n_jobs)
    else:
        results = results.compute(scheduler=scheduler, num_workers=n_jobs)

    return results


def disc_chi_squared_single(values, pix, pix_disc, radius=np.deg2rad(10), nside=64,
                            theta_band=False, bins=None):
    vec_disc = hp.pix2vec(nside=nside, ipix=pix_disc)
    pix_in_disc = hp.query_disc(nside=nside, vec=vec_disc, radius=radius)
    in_on_region = np.isin(pix, pix_in_disc)
    if not np.sum(in_on_region):
        raise ValueError('No events found in disc centered at pixel {} with '
                         'radius {}'.format(pix_disc, radius))

    # Construct off region mask
    if not theta_band:
        in_off_region = ~in_on_region
    else:
        theta_in_disc, _ = hp.pix2ang(nside=nside, ipix=pix_in_disc)
        pix_theta_band = hp.query_strip(nside=nside,
                                        theta1=theta_in_disc.min(),
                                        theta2=theta_in_disc.max())
        pix_off_region = np.setdiff1d(pix_theta_band, pix_in_disc)
        in_off_region = np.isin(pix, pix_off_region)

    # Energy distribution inside and outside of disc
    if bins is None:
        bins = np.linspace(values.min(), values.max(), 20)
    counts_in_disc, _ = np.histogram(values[in_on_region], bins=bins)
    counts_outside_disc, _ = np.histogram(values[in_off_region], bins=bins)

    if np.isin([counts_in_disc, counts_outside_disc], 0).any():
        raise ValueError('Energy distribution has zero counts in a bin')

    # Want to make sure off region histogram is scaled to the on region histogram
    alpha = np.sum(counts_in_disc) / np.sum(counts_outside_disc)
    scaled_counts_outside_disc = alpha * counts_outside_disc

    # Calculate chi-squared, p-value, and significance
    chi_squared = counts_chi_squared(counts_in_disc, scaled_counts_outside_disc)
    ndof = len(bins) - 1
    pval = stats.chi2.sf(chi_squared, ndof)
    sig = erfcinv(2 * pval) * np.sqrt(2)

    result = {'pix_disc': pix_disc,
              'alpha': alpha,
              'num_on':  np.sum(counts_in_disc),
              'chi2': chi_squared,
              'pval': pval,
              'sig': sig,
              }

    return result


def counts_chi_squared(counts_on, counts_off):
    """Calculates reduced chi-squared between two energy histograms

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
