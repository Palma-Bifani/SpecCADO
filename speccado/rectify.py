'''Classes and functions for a preliminary registration of MICADO spectra'''
from os.path import basename, splitext

import numpy as np
from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from .layout import SpectralTrace

class SpecChip(object):
    '''Class holding information about a chip readout

    Parameters
    ----------

    hdu : Instance of ImageHDU

    Attributes
    ----------

    wcs_fp : WCS
         transforms between pixels and focal plane coordinates (in mm)
    nx, ny : int, int
         dimensions of the image
    extname : str
         name of the chip
    interp :
         spline interpolation of chip image as a function of pixel
         coordinates
'''

    def __init__(self, hdu):
        self.wcs_fp = WCS(hdu, key='A')
        self.ny, self.nx = hdu.data.shape
        iarr, jarr = np.arange(self.nx), np.arange(self.ny)
        self.hdu = hdu
        #self.interp = RectBivariateSpline(iarr, jarr, hdu.data)
        self.extname = hdu.header['EXTNAME']
        self.interp = None
        print("Done ", self.extname)

    def interpolate(self):
        '''Determine interpolation function for chip image'''
        iarr = np.arange(self.nx)
        jarr = np.arange(self.ny)
        self.interp = RectBivariateSpline(jarr, iarr, self.hdu.data)



def is_order_in_field(spectrace, chip):
    '''Determine whether an order described by layout appears on chip

    Parameters
    ----------
    spectrace : instance of sc.SpectralTrace

    imagehdu : instance of astropy.fits.ImageHDU
    '''
    # TODO Should we take filter into account?

    wcs = chip.wcs_fp
    imax = chip.nx - 1
    jmax = chip.ny - 1
    ijedges = [[0, 0], [0, jmax], [imax, 0], [imax, jmax]]
    xyedges = wcs.all_pix2world(ijedges, 0)
    xlo = xyedges[:, 0].min()
    xhi = xyedges[:, 0].max()
    ylo = xyedges[:, 1].min()
    yhi = xyedges[:, 1].max()

    xcorner = np.array([spectrace.left_edge(ylo), spectrace.right_edge(ylo),
                        spectrace.left_edge(yhi), spectrace.right_edge(yhi)])

    return np.any((xcorner > xlo) * (xcorner <= xhi))


def rectify_trace(trace, chiplist, params):
    '''Create 2D spectrum for a trace

    Parameters
    ----------

    trace : object of class SpectralTrace
    chiplist : list of SpecChip objects
    params : parameter dictionary (required: pixsize, pixscale, slit_length)

    Output
    ------
    The function writes a FITS file with the rectified spectrum.

'''

    filebase = splitext(basename(trace.name))[0]

    # Check whether the trace is on any chips at all
    goodchips = []
    for ichip, chip in enumerate(chiplist):
        if is_order_in_field(trace, chip):
            goodchips.append(ichip)
    if not goodchips:
        print("On no chips")
        return None

    # Build the xi-lambda image for this trace
    lam_min, lam_max, dlam_min = trace.analyse_lambda()
    print("   extends from ", lam_min, " to ", lam_max)
    dlam_pix = dlam_min * params['pixsize']

    # image size
    xi_min = trace.layout['xi1'].min()
    delta_xi = params['pixscale']
    n_xi = int(params['slit_length'] / delta_xi)
    n_lam = int((lam_max - lam_min) / dlam_pix)


    ## WCS for the rectified spectrum
    ## TODO: Convert xi to arcsec
    ## TODO: Define xi_min and delta_xi
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['LINEAR', 'WAVE']
    wcs.wcs.cname = ['SLITPOS', 'WAVELEN']
    wcs.wcs.cunit = ['arcsec', 'um']
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.crval = [xi_min, lam_min]
    wcs.wcs.cdelt = [delta_xi, dlam_pix]

    ## Now I could create Xi, Lam images
    Iarr, Jarr = np.meshgrid(np.arange(n_xi, dtype=np.float32),
                             np.arange(n_lam, dtype=np.float32))

    Xi, Lam = wcs.all_pix2world(Iarr, Jarr, 0)

    del Iarr
    del Jarr

    # Make sure that we do have microns
    Lam = Lam * u.Unit(wcs.wcs.cunit[1]).to(u.um)

    # Convert Xi, Lam to focal plane units
    Xarr = trace.xilam2x(Xi, Lam)
    Yarr = trace.xilam2y(Xi, Lam)

    del Xi
    del Lam

    rect_spec = np.zeros_like(Xarr, dtype=np.float32)

    print("Xarr.shape: ", Xarr.shape)
    # Convert to pixels on the chip

    ## Have to step through all chips.
    ## It would be better to create Chip objects that hold the interpolation
    ## objects, so that these do not have to be created for every trace
    for ichip in goodchips:
        chip = chiplist[ichip]

        # Determine interpolation function
        if chip.interp is None:
            chip.interpolate()

        print("  Trace is in chip ", chip.extname)

        iarr, jarr = chip.wcs_fp.all_world2pix(Xarr, Yarr, 0)
        mask = (iarr > 0) * (iarr < chip.nx) * (jarr > 0) * (jarr < chip.ny)
        if not np.any(mask):
            print("It's not on the chip after all...")
            continue

        specpart = chip.interp(jarr, iarr, grid=False)
        rect_spec += specpart * mask

    print("Writing", filebase + ".fits")
    fits.writeto(filebase + ".fits", rect_spec, header=wcs.to_header(),
                 overwrite=True)
