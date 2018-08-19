
from __future__ import division, print_function
import numpy as np
import healpy as hp


def disc_on_region(pix_center, size=np.deg2rad(10), nside=64):
    """ Circular on region
    """

    vec_disc = hp.pix2vec(nside=nside, ipix=pix_center)
    pix_in_disc = hp.query_disc(nside=nside, vec=vec_disc, radius=size)

    return pix_in_disc


on_regions = {'disc': disc_on_region,
              }


def allsky_off_region(on_region_pix, nside=64):
    """ All sky off region
    """
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    off_region_mask = ~np.isin(pix, on_region_pix)
    off_region_pix = pix[off_region_mask]

    return off_region_pix


def theta_band_off_region(on_region_pix, nside=64):
    on_region_theta, _ = hp.pix2ang(nside=nside, ipix=on_region_pix)
    theta_on_min = on_region_theta.min()
    theta_on_max = on_region_theta.max()

    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    theta, phi = hp.pix2ang(nside=nside, ipix=pix)
    theta_band_mask = np.logical_and(theta <= theta_on_max,
                                     theta >= theta_on_min)
    off_region_mask = ~np.isin(pix[theta_band_mask], on_region_pix)
    off_region_pix = pix[theta_band_mask][off_region_mask]

    return off_region_pix


off_regions = {'allsky': allsky_off_region,
               'theta_band': theta_band_off_region,
               }
