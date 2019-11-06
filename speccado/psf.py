"""
Functions relating to the psf

These functions enhance the simcado.PSF class in a crude way
"""

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits

from astropy import units as u

import simcado as sim

# The contents of the two following functions should go into simcado
# psf module
def prepare_psf(psffile, fits_ext=0):
    """
    Create a SimCADO PSF object with added WCS and interpolation

    Parameters
    ----------
    psffile : path
       File with one or more PSFs in separate extensions
    fits_ext : int
       Extension that holds the desired PSF. Default: 0

    """
    # Load the psf
    psf = sim.psf.UserPSF(psffile, fits_ext=fits_ext)

    # TODO: Check the units - we have confusion between arcsec and mas
    psf.array /= psf.pix_res**2      # normiert per arcsec2

    # Add the WCS to the PSF object
    psf.wcs = WCS(fits.getheader(psffile, ext=fits_ext))

    # Add an interpolation function to the PSF object
    psf.interp = psf_interpolate(psf)

    return psf


def psf_interpolate(psf):
    """
    Create interpolation function for a simcado PSF object

    Parameters
    ----------
    psf : instance of simcado.psf.UserPSF
    """
    from scipy.interpolate import RectBivariateSpline

    # Pixel grid in python convention -> origin = 0
    xpix = np.arange(psf.shape[1])
    ypix = np.arange(psf.shape[0])

    # Convert to angular coordinates (depends on the cunits of psf.wcs)
    psf_unit_x = u.Unit(psf.wcs.wcs.cunit[0])
    psf_unit_y = u.Unit(psf.wcs.wcs.cunit[1])
    xas, yas = psf.wcs.all_pix2world(xpix, ypix, 0)
    xas *= psf_unit_x.to(u.arcsec)
    yas *= psf_unit_y.to(u.arcsec)

    return RectBivariateSpline(xas, yas, psf.array)
