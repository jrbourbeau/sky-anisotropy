
from __future__ import division, print_function
import numpy as np


def equatorial_to_healpy(ra, dec):
    """ Convert equatorial coordinates (ra, dec) to healpy coordinates (theta, phi)

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
    """ Convert healpy coordinates (theta, phi) to equatorial coordinates (ra, dec)

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
