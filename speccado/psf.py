'''Functions relating to the psf'''

import numpy as np
from astropy.wcs import WCS
import simcado as sim

# The contents of the two following functions should go into simcado
# psf module
def prepare_psf(psffile):
    '''Create a SimCADO PSF object with added WCS and interpolation'''
    psf = sim.psf.UserPSF(psffile)
    psf.array /= psf.pix_res**2      # normiert per arcsec2
    psf.wcs = WCS(psffile)
    psf.interp = psf_interpolate(psf)
    return psf


def psf_interpolate(psf):
    '''Create interpolation function for a simcado PSF object'''
    from scipy.interpolate import RectBivariateSpline

    # Pixel grid in python convention -> origin = 0
    xpix = np.arange(psf.shape[1])
    ypix = np.arange(psf.shape[0])
    xas, yas = psf.wcs.all_pix2world(xpix, ypix, 0)
    return RectBivariateSpline(xas, yas, psf.array)
