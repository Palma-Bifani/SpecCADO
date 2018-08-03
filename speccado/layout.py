'''Functions and classes relating to spectroscopic layout'''
import numpy as np
from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

import simcado as sim

__all__ = ['SpectralTrace', 'is_order_on_chip']

#class SpectralTraceList(list):
#    '''List of spectral traces
#
#    The class is instantiated from a multi-extension FITS file, where
#    each extension contains the description of a spectral order and is
#    read into a `SpectralTrace' object.
#
#    Parameters
#    ----------
#    layoutfile : str
#       name of a FITS file
#
#    Returns
#    -------
#    List of objects of class `SpectralTrace'
#    '''
#
#    def __init__(self, layoutfile):
#
#        self = []
#        # get number of extensions in FITS file
#        with fits.open(layoutfile) as hdul:
#            n_ext = len(hdul) - 1
#        print(n_ext)
#        for i_ext in range(n_ext):
#            self.append(SpectralTrace(layoutfile, i_ext + 1))
#            print(self[-1])


class SpectralTrace(object):
    '''Description of a spectral trace

   The class reads an order layout and fits several functions
    useful to describe the geometry of the trace.

    Parameters
    ----------
    layoutfile : str
        name of a FITS file
    ext : int
        number of FITS extension to be read with `read_spec_order`


    Attributes
    ----------
    file :
        name of the file that specified the trace
    layout :
        table of with wavelength lam and positions xi, yi for lines defining
        the trace
    xilam2x, xilam2y :
        polynomials to convert slit position xi and wavelength lam to position
        x, y in the focal plane
    xy2xi, xy2lam :
        polynomials to convert position x, y in the focal plane to slit position
        xi and wavelength lambda
    left_edge, centre_trace, right_edge :
        1D polynomials giving x as a function of y of the left edge, middle
        and right edge of the trace on the focal plane (xi = 0, 0.5, 1,
        respectively)
    dlam_by_dy :
        polynomial giving the wavelength dispersion in the y direction in um/mm
        as a function of focal plane position x, y


    Notes
    -----
    Attribute `.layout' is an `astropy.Table' with columns
        - lam : wavelength
        - x_1, x_2, ... : focal plane x-coordinates of points along the slit
        - y_1, y_2, ... : focal plane y-coordinates of points along the slit
        - r50_1, r50_2, ... : radii of 50% encircled energy
        - r80_1, r80_2, ... : radii of 80% encircled energy
    '''

    def __init__(self, layoutfile, hdu=1):
        self.file = layoutfile
        # self.layout = read_spec_order(layoutfile, ext)
        self.layout = Table.read(layoutfile, hdu)
        self.xy2xi, self.xy2lam = xy2xilam_fit(self.layout)
        self.xilam2x, self.xilam2y = xilam2xy_fit(self.layout)
        self.dlam_by_dy = sim.utils.deriv_polynomial2d(self.xy2lam)[1]
        self.left_edge = trace_fit(self.layout, 'left')
        self.centre_trace = trace_fit(self.layout, 'centre')
        self.right_edge = trace_fit(self.layout, 'right')
        try:
            self.name = self.layout.meta['EXTNAME']
        except KeyError:
            self.name = filename + '[' + hdu + ']'


class XiLamImage(object):
    '''Class to compute a rectified 2D spectrum.

    The class produces and holds an image of xi (relative position along
    the spatial slit direction) and wavelength lambda.
    '''

    def __init__(self, src, psf, lam_min, lam_max, dlam_per_pix, cmds,
                 transmission):

        # Slit dimensions: oversample with respect to detector pixel scale
        pixscale = cmds['SIM_DETECTOR_PIX_SCALE']  # arcsec/detector pixel
        xi_scale = 2
        eta_scale = 4

        # Steps in xi (along slit length) and eta (along slit width)
        delta_xi = pixscale / xi_scale      # in arcsec / xilam pixel
        delta_eta = pixscale / eta_scale

        # Slit width. Input in arcsec. The slit image has npix_eta pixels
        # in the width (dispersion) direction. This corresponds to a wavelength
        # range which is determined by the local dispersion dlam_per_pix, given
        # per *detector pixel*
        slit_width_as = cmds['SPEC_SLIT_WIDTH']         # in arcsec
        npix_eta = np.int(slit_width_as / delta_eta)
        slit_width_lam = dlam_per_pix * npix_eta / eta_scale

        # Slit length. Input in arcsec. The slit image has npix_xi pixels
        # in the length direction, as does the xilam image.
        slit_length_as = cmds['SPEC_SLIT_LENGTH']
        npix_xi = np.int(slit_length_as / delta_xi)

        # pixel coordinates of slit centre
        xi_cen = npix_xi / 2 - 0.5    # TODO: is this useful? or rather xi=0?
        eta_cen = npix_eta / 2 - 0.5

        # Step in wavelength, oversample at least 5 times with respect
        # to detector dispersion, better use the dispersion of the best source
        # spectrum.
        delta_lam = min(0.2 * dlam_per_pix, src.dlam)

        # Initialise the wavelength vector. The xilam image will have npix_lam
        # pixels in the wavelength direction.
        lam = sim.utils.seq(lam_min, lam_max, delta_lam)
        npix_lam = len(lam)

        ## Initialise image to hold the xi-lambda image
        self.image = np.zeros((npix_xi, npix_lam), dtype=np.float32)

        # hdu lists to hold slit images and xi-lambda images
        slithdul = fits.HDUList()

        # Create a wcs for the slit
        # Note that the xi coordinates are excentric for the 16 arcsec slit
        # TODO: Replace by layout['xi1'] -- but we don't have layout here?
        xi1 = -1.5
        slitwcs = WCS(naxis=2)
        slitwcs.wcs.ctype = ['LINEAR', 'LINEAR']
        slitwcs.wcs.cunit = psf.wcs.wcs.cunit
        slitwcs.wcs.crval = [xi1, 0]
        slitwcs.wcs.crpix = [1, 1 + eta_cen]
        slitwcs.wcs.cdelt = [delta_xi, delta_eta]

        xi_cen = -xi1 / delta_xi   # TODO: not correct in general

        ## Loop over all sources
        for curspec in src.spectra:
            wcs_spec = curspec.wcs
            flux_interp = curspec.interp

            ## Build slit images
            if curspec.spectype == 'src':
                # place a psf image for each src spectrum
                srcpos = curspec.srcpos
                scale = 1   # TODO: generalise
                angle = 0   # TODO: generalise

                slit_image = np.zeros((npix_eta, npix_xi), dtype=np.float32)

                # Build a WCS that allows mapping the source to the slit
                mapwcs = WCS(naxis=2)
                mapwcs.wcs.ctype = ['LINEAR', 'LINEAR']
                mapwcs.wcs.cunit = psf.wcs.wcs.cunit
                mapwcs.wcs.crval = [0, 0]
                mapwcs.wcs.crpix = [1 + xi_cen + srcpos[0] / delta_xi,
                                    1 + eta_cen + srcpos[1] / delta_eta]
                mapwcs.wcs.cdelt = np.array([delta_xi, delta_eta]) / scale
                mapwcs.wcs.pc = [[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]]

                ## Pixel and arcsec grids on the slit
                xi_arr, eta_arr = np.meshgrid(np.arange(npix_xi),
                                              np.arange(npix_eta))
                xi_as, eta_as = mapwcs.all_pix2world(xi_arr, eta_arr, 0)

                # Create psf image on the slit
                slit_image = psf.interp(xi_as, eta_as, grid=False)
                slithdul.append(fits.ImageHDU(slit_image,
                                              header=slitwcs.to_header()))

            elif curspec.spectype == 'bg':
                # bg spectra fill the slit homogeneously
                slit_image = np.ones((npix_eta, npix_xi),
                                     dtype=np.float32)
                slithdul.append(fits.ImageHDU(slit_image,
                                              header=slitwcs.to_header()))

            if npix_eta == 1:
                dlam_eta = 0
            else:
                dlam_eta = slit_width_lam / (npix_eta - 1)

            ## Each row of the slit image is tensor multiplied with the spectrum
            ## slightly shifted in wavelength to account for the finite slit
            ## width (i.e. convolution with the slit profile) and added to the
            ## xi-lambda image.
            ## TODO: Check what is happening to the units here!
            for i in range(npix_eta):
                lam0 = lam - slit_width_lam / 2 + i * dlam_eta
                nimage = np.outer(slit_image[i,], flux_interp(lam0)
                                  * transmission(lam0))
                self.image += nimage * delta_eta

        slithdul.writeto("slitimages.fits", overwrite=True)

        ## Default WCS with xi in arcsec from -1.5 to 13.5
        ## TODO: Make this default
        x1 = -1.5   # Take from elsewhere
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [1, 1]
        self.wcs.wcs.crval = [lam[0], x1]
        self.wcs.wcs.pc = [[1, 0], [0, 1]]
        self.wcs.wcs.cdelt = [delta_lam, delta_xi]  #
        self.wcs.wcs.ctype = ['LINEAR', 'LINEAR']
        self.wcs.wcs.cname = ['WAVELEN', 'SLITPOS']
        self.wcs.wcs.cunit = ['um', 'arcsec']

        ## WCS for the xi-lambda-image, i.e. the rectified 2D spectrum
        ## Alternative : xi = [0, 1], dimensionless
        self.wcsa = WCS(naxis=2)
        self.wcsa.wcs.crpix = [1, 1]
        self.wcsa.wcs.crval = [lam[0], 0]
        self.wcsa.wcs.pc = [[1, 0], [0, 1]]
        self.wcsa.wcs.cdelt = [delta_lam, 1./npix_xi]
        self.wcsa.wcs.ctype = ['LINEAR', 'LINEAR']
        self.wcsa.wcs.cname = ['WAVELEN', 'SLITPOS']
        self.wcsa.wcs.cunit = ['um', '']

        # TODO: this might be a place to restrict to the short slit
        #       The previous interpolation should be done on the long slit,
        #       actual image construction and interpolation only on relevant
        #       part of the slit
        self.xi = self.wcs.all_pix2world(lam[0], np.arange(npix_xi), 0)[1]
        self.lam = lam
        self.npix_xi = npix_xi
        self.npix_lam = npix_lam
        self.interp = RectBivariateSpline(self.xi, self.lam, self.image)


        #self.xi2 = self.wcsa.all_pix2world(lam[0], np.arange(npix_xi), 0)[1]
        #self.lam2 = lam
        #self.interp2 = RectBivariateSpline(self.xi2, self.lam2, self.image)

    def writeto(self, outfile, overwrite=True):
        header = self.wcs.to_header()
        header.extend(self.wcsa.to_header(key='A'))
        hdulist = fits.HDUList(fits.PrimaryHDU(header=header,
                                               data=self.image))
        hdulist.writeto(outfile, overwrite=overwrite)



def trace_fit(layout, side='left', degree=1):
    '''Linear fit to describe edge of a spectral trace

    Parameters
    ----------
    layout : [table]
         a table describing a spectral trace
    side : [str]
         can be 'left', 'right' or 'centre'
    '''
    from astropy.modeling.models import Polynomial1D
    from astropy.modeling.fitting import LinearLSQFitter

    if side == 'left':
        xpt = layout['x1']
        ypt = layout['y1']
        name = 'left edge'
    elif side == 'centre':
        xpt = layout['x2']
        ypt = layout['y2']
        name = 'trace centre'
    elif side == 'right':
        xpt = layout['x3']
        ypt = layout['y3']
        name = 'right edge'
    else:
        raise ValueError('Side ' + str(side) + ' not recognized')

    pinit = Polynomial1D(degree=degree)
    fitter = LinearLSQFitter()
    edge_fit = fitter(pinit, ypt, xpt)
    edge_fit.name = name
    return edge_fit


def xilam2xy_fit(layout):
    '''Determine polynomial fits of FPA position

    Fits are of degree 4 as a function of slit position and wavelength.
    '''
    from astropy.modeling import models, fitting

    xilist = []
    xlist = []
    ylist = []
    for key in layout.colnames:
        if key[:2] == 'xi':
            xilist.append(layout[key])
        elif key[:1] == 'x':
            xlist.append(layout[key])
        elif key[:1] == 'y':
            ylist.append(layout[key])

    xi = np.concatenate(xilist)
    lam = np.tile(layout['lam'], len(xlist))
    x = np.concatenate(xlist)
    y = np.concatenate(ylist)

    pinit_x = models.Polynomial2D(degree=4)
    pinit_y = models.Polynomial2D(degree=4)
    fitter = fitting.LinearLSQFitter()
    xilam2x = fitter(pinit_x, xi, lam, x)
    xilam2x.name = 'xilam2x'
    xilam2y = fitter(pinit_y, xi, lam, y)
    xilam2y.name = 'xilam2y'
    return xilam2x, xilam2y


def xy2xilam_fit(layout):
    '''Determine polynomial fits of wavelength/slit position

    Fits are of degree 4 as a function of focal plane position'''
    from astropy.modeling import models, fitting
    xilist = []
    xlist = []
    ylist = []
    for key in layout.colnames:
        if key[:2] == 'xi':
            xilist.append(layout[key])
        elif key[:1] == 'x':
            xlist.append(layout[key])
        elif key[:1] == 'y':
            ylist.append(layout[key])


    xi = np.concatenate(xilist)
    lam = np.tile(layout['lam'], len(xlist))
    x = np.concatenate(xlist)
    y = np.concatenate(ylist)

    pinit_xi = models.Polynomial2D(degree=4)
    pinit_lam = models.Polynomial2D(degree=4)
    fitter = fitting.LinearLSQFitter()
    xy2xi = fitter(pinit_xi, x, y, xi)
    xy2xi.name = 'xy2xi'
    xy2lam = fitter(pinit_lam, x, y, lam)
    xy2lam.name = 'xy2lam'
    return xy2xi, xy2lam


def is_order_on_chip(spectrace, chip, ylo=None, yhi=None):
    '''Determine whether an order described by layout appears on chip

    Parameters
    ----------

    Output
    ------
'''
    if ylo is None:
        ylo = chip.ymin_um
    if yhi is None:
        yhi = chip.ymax_um

    xlo = chip.xmin_um
    xhi = chip.xmax_um
    xcorner = np.array([spectrace.left_edge(ylo), spectrace.right_edge(ylo),
                        spectrace.left_edge(yhi), spectrace.right_edge(yhi)])

    return np.any((xcorner > xlo) * (xcorner <= xhi))


def analyse_trace(trace):
    '''Get a few numbers describing the extent of a trace'''
    lam_min = np.min(trace.layout['lam'])
    lam_max = np.max(trace.layout['lam'])
    dlam_min = np.min(trace.dlam_by_dy(trace.layout['x2'],
                                       trace.layout['y2']))
    return lam_min, lam_max, dlam_min


def read_spec_order(filename, ext=0):
    '''Read spectral order definition from a FITS file

    Parameters
    ----------
    filename : str

    Returns
    -------
    If ext == 0: a list of SpectralTrace objects (all extensions of `filename')
    If ext > 0: a SpectralTrace object loaded from extension `ext' of `filename'
    '''
    if ext == 0:
        with fits.open(filename) as hdul:
            n_ext = len(hdul)
        tablist = []
        for i_ext in np.arange(1, n_ext):
            try:
                tablist.append(SpectralTrace(filename, hdu=i_ext))
            except IndexError:
                pass

        return tablist

    else:
        return SpectralTrace(filename, hdu=ext)
