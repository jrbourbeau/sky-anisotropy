
from __future__ import division, print_function
import numpy as np
import healpy as hp

from .coordinates import healpy_to_equatorial


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

    raise NotImplementedError()

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
    phi_mask = np.logical_and(phi <= phi_upper, phi >= phi_lower)
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
