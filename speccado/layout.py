'''Functions and classes relating to spectroscopic layout'''
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

import simcado as sim

__all__ = ['SpectralTrace']

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
        self.xy2xi, self.xy2lam = self.xy2xilam_fit()
        self.xilam2x, self.xilam2y = self.xilam2xy_fit()
        self._xiy2x, self._xiy2lam = self._xiy2xlam_fit()
        self.dlam_by_dy = sim.utils.deriv_polynomial2d(self.xy2lam)[1]
        self.left_edge = self.trace_fit('left')
        self.centre_trace = self.trace_fit('centre')
        self.right_edge = self.trace_fit('right')
        try:
            self.name = self.layout.meta['EXTNAME']
        except KeyError:
            self.name = layoutfile + '[' + hdu + ']'

    def __str__(self):
        objtype = type(self).__name__
        name = self.name
        lam_min = self.layout['lam'].min()
        lam_max = self.layout['lam'].max()
        lam_unit = self.layout['lam'].unit.name
        template = "{} '{}': {:.2f} - {:.2f} {}"
        return template.format(objtype, name, lam_min, lam_max, lam_unit)

    def analyse_lambda(self):
        '''Get a few numbers describing the extent of a trace

        Returns
        -------
        a tuple with (`lam_min`, `lam_max`, `dlam_min`):
        lam_min : minimum wavelength covered by the trace
        lam_max : maximum wavelength covered by the trace
        dlam_min : minimum dispersion in wavelength range, in um/mm
        '''
        lam_min = np.min(self.layout['lam'])
        lam_max = np.max(self.layout['lam'])
        dlam_min = np.min(self.dlam_by_dy(self.layout['x2'],
                                          self.layout['y2']))
        return lam_min, lam_max, dlam_min


    def write_reg(self, filename=None, append=False, slit_length=None,
                  waveband=None):
        '''Write a regions file for ds9

        The regions file uses focal plane coordinates and has to be loaded
        with WCSA ("PIX2FP") of a SimCADO-produced image.

        Parameters
        ----------
        filename : str
            The name of the regions file to write. If `None`, the filename
            is constructed from the name of the trace.
        append : boolean
            If `True` and `filename` exists, then the region for the trace
            is appended to the file. Otherwise, a new file is created (or
            the existing file is overwritten).
        slit_length : length of the slit in arcsec
            If "None", the full extent of the layout table is used.
        waveband : str or tuple of floats
            Name of order-sorting filter or limits of the wavelength range
            to consider (in um)
        '''
        # Header line for regions file
        headline = '''# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
wcsa;
'''
        print(str(self))

        # Reformat the layout table
        # Note: This is redundant as it occurs in the fitting functions, too
        xilist = []
        for key in self.layout.colnames:
            if key[:2] == 'xi':
                xilist.append(self.layout[key])

        xi = np.concatenate(xilist)
        lam = self.layout['lam']

        xi_min = xi.min()
        xi_max = xi_min + slit_length

        # Wavelength limits: If no waveband has been specified, plot the
        # full trace. Otherwise cut to desired limits.
        if waveband is None:
            lam_min = lam.min()
            lam_max = lam.max()
        else:
            if isinstance(waveband, tuple):
                lam_min, lam_max = waveband
            else:
                order_sort = sim.spectral.TransmissionCurve(waveband)
                os_mask = order_sort.val_orig > 0
                lam_min = order_sort.lam_orig[os_mask].min()
                lam_max = order_sort.lam_orig[os_mask].max()

            # Check whether the trace is in the desired waveband
            # and find the closest value in lam to these limits
            if lam_min > lam.max() or lam_max < lam.min():
                print("   --> outside wavelength range.")
                return
            else:
                templam = self.layout['lam']
                lam_min = templam[np.abs(templam - lam_min).argmin()]
                lam_max = templam[np.abs(templam - lam_max).argmin()]

        # Limits of the trace x_ij, y_ij in the focal plane.
        x_11 = self.xilam2x(xi_min, lam_min)
        x_12 = self.xilam2x(xi_max, lam_min)

        y_11 = self.xilam2y(xi_min, lam_min)
        y_12 = self.xilam2y(xi_max, lam_min)

        x_21 = self.xilam2x(xi_min, lam_max)
        x_22 = self.xilam2x(xi_max, lam_max)

        y_21 = self.xilam2y(xi_min, lam_max)
        y_22 = self.xilam2y(xi_max, lam_max)

        polygon = "polygon({},{},{},{},{},{},{},{})\n".format(x_11, y_11,
                                                              x_12, y_12,
                                                              x_22, y_22,
                                                              x_21, y_21)

        if filename is None:
            filename = os.path.splitext(self.name)[0] + ".reg"

        if not (os.path.exists(filename) and append):
            fp1 = open(filename, 'w')
            fp1.write(headline)
        else:
            fp1 = open(filename, 'a')

        fp1.write(polygon)
        fp1.close()
        xall = np.array([x_11, x_12, x_22, x_21])
        yall = np.array([y_11, y_12, y_22, y_21])
        return np.column_stack((xall, yall))

    def is_on_chip(self, chip, slitlength=3, ylo=None, yhi=None):
        '''Determine whether the trace appears on chip

        Parameters
        ----------
        chip : simcado.detector.Chip

        slitlength : float
            Length of the slit on the sky (arcsec)

        ylo, yhi : float
            y-limits of the chip in the focal plane (um). Default is
            to obtain these values from chip.

        Output
        ------
        True if part of the trace is mapped to the chip, False otherwise.
        '''
        # chip corners in focal plane
        if ylo is None:
            ylo = chip.ymin_um
        if yhi is None:
            yhi = chip.ymax_um

        xlo = chip.xmin_um
        xhi = chip.xmax_um

        # Slit dimensions (xi in arcsec)
        xilo = -1.5
        xihi = slitlength - 1.5

        xcorner = np.array([self._xiy2x(xilo, ylo), self._xiy2x(xihi, ylo),
                            self._xiy2x(xilo, yhi), self._xiy2x(xihi, yhi)])

        return np.any((xcorner > xlo) * (xcorner <= xhi))


    def trace_fit(self, side='left', degree=1):
        '''Linear fit to describe edge of a spectral trace

        Parameters
        ----------
        side : [str]
             can be 'left', 'right' or 'centre'
        '''
        from astropy.modeling.models import Polynomial1D
        from astropy.modeling.fitting import LinearLSQFitter

        if side == 'left':
            xpt = self.layout['x1']
            ypt = self.layout['y1']
            name = 'left edge'
        elif side == 'centre':
            xpt = self.layout['x3']
            ypt = self.layout['y3']
            name = 'trace centre'
        elif side == 'right':   # HACK: This assumes 5-point layout
            xpt = self.layout['x5']
            ypt = self.layout['y5']
            name = 'right edge'
        else:
            raise ValueError('Side ' + str(side) + ' not recognized')

        pinit = Polynomial1D(degree=degree)
        fitter = LinearLSQFitter()
        edge_fit = fitter(pinit, ypt, xpt)
        #edge_fit.name = name
        return edge_fit


    def xilam2xy_fit(self):
        '''Determine polynomial fits of FPA position

        Fits are of degree 4 as a function of slit position and wavelength.
        '''
        from astropy.modeling import models, fitting

        # Build full lists (distinction in x1...x5 not necessary)
        xilist = []
        xlist = []
        ylist = []
        for key in self.layout.colnames:
            if key[:2] == 'xi':
                xilist.append(self.layout[key])
            elif key[:1] == 'x':
                xlist.append(self.layout[key])
            elif key[:1] == 'y':
                ylist.append(self.layout[key])

        xi = np.concatenate(xilist)
        lam = np.tile(self.layout['lam'], len(xlist))
        x = np.concatenate(xlist)
        y = np.concatenate(ylist)

        # Filter the lists: remove any points with x==0
        good = x != 0
        xi = xi[good]
        lam = lam[good]
        x = x[good]
        y = y[good]

        # compute the fits
        pinit_x = models.Polynomial2D(degree=4)
        pinit_y = models.Polynomial2D(degree=4)
        fitter = fitting.LinearLSQFitter()
        xilam2x = fitter(pinit_x, xi, lam, x)
        #xilam2x.name = 'xilam2x'
        xilam2y = fitter(pinit_y, xi, lam, y)
        #xilam2y.name = 'xilam2y'
        return xilam2x, xilam2y


    def xy2xilam_fit(self):
        '''Determine polynomial fits of wavelength/slit position

        Fits are of degree 4 as a function of focal plane position'''
        from astropy.modeling import models, fitting

        # Build full lists (distinction in x1...x5 not necessary)
        xilist = []
        xlist = []
        ylist = []
        for key in self.layout.colnames:
            if key[:2] == 'xi':
                xilist.append(self.layout[key])
            elif key[:1] == 'x':
                xlist.append(self.layout[key])
            elif key[:1] == 'y':
                ylist.append(self.layout[key])

        xi = np.concatenate(xilist)
        lam = np.tile(self.layout['lam'], len(xlist))
        x = np.concatenate(xlist)
        y = np.concatenate(ylist)

        # Filter the lists: remove any points with x==0
        good = x != 0
        xi = xi[good]
        lam = lam[good]
        x = x[good]
        y = y[good]

        pinit_xi = models.Polynomial2D(degree=4)
        pinit_lam = models.Polynomial2D(degree=4)
        fitter = fitting.LinearLSQFitter()
        xy2xi = fitter(pinit_xi, x, y, xi)
        #xy2xi.name = 'xy2xi'
        xy2lam = fitter(pinit_lam, x, y, lam)
        #xy2lam.name = 'xy2lam'
        return xy2xi, xy2lam


    def _xiy2xlam_fit(self):
        '''Determine polynomial fits of wavelength/slit position

        Fits are of degree 4 as a function of focal plane position'''
        # These are helper functions to allow fitting of left/right edges
        # for the purpose of checking whether a trace is on a chip or not.
        from astropy.modeling import models, fitting

        # Build full lists (distinction in x1...x5 not necessary)
        xilist = []
        xlist = []
        ylist = []
        for key in self.layout.colnames:
            if key[:2] == 'xi':
                xilist.append(self.layout[key])
            elif key[:1] == 'x':
                xlist.append(self.layout[key])
            elif key[:1] == 'y':
                ylist.append(self.layout[key])

        xi = np.concatenate(xilist)
        lam = np.tile(self.layout['lam'], len(xlist))
        x = np.concatenate(xlist)
        y = np.concatenate(ylist)

        # Filter the lists: remove any points with x==0
        good = x != 0
        xi = xi[good]
        lam = lam[good]
        x = x[good]
        y = y[good]

        pinit_x = models.Polynomial2D(degree=4)
        pinit_lam = models.Polynomial2D(degree=4)
        fitter = fitting.LinearLSQFitter()
        xiy2x = fitter(pinit_x, xi, y, x)
        #xy2xi.name = 'xy2xi'
        xiy2lam = fitter(pinit_lam, xi, y, lam)
        #xy2lam.name = 'xy2lam'
        return xiy2x, xiy2lam


class XiLamImage(object):
    '''Class to compute a rectified 2D spectrum.

    The class produces and holds an image of xi (relative position along
    the spatial slit direction) and wavelength lambda.
    '''

    def __init__(self, src, psf, lam_min, lam_max, xi_min,
                 dlam_per_pix, cmds, transmission):

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
        xi_cen = -xi_min / delta_xi
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
        slitwcs = WCS(naxis=2)
        slitwcs.wcs.ctype = ['LINEAR', 'LINEAR']
        slitwcs.wcs.cunit = psf.wcs.wcs.cunit
        slitwcs.wcs.crval = [xi_min, 0]
        slitwcs.wcs.crpix = [1, 1 + eta_cen]
        slitwcs.wcs.cdelt = [delta_xi, delta_eta]

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
            ## TODO: outer multiplication only works for background spectra.
            ##       For sources, we need to build the cube explicitely in order to take ADS
            ##       into account.
            for i in range(npix_eta):
                lam0 = lam - slit_width_lam / 2 + i * dlam_eta
                nimage = np.outer(slit_image[i,], flux_interp(lam0)
                                  * transmission(lam0))
                self.image += nimage * delta_eta

        slithdul.writeto("slitimages.fits", overwrite=True)

        ## WCS for the xi-lambda-image, i.e. the rectified 2D spectrum
        ## Default WCS with xi in arcsec from xi_min (-1.5)
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [1, 1]
        self.wcs.wcs.crval = [lam[0], xi_min]
        self.wcs.wcs.pc = [[1, 0], [0, 1]]
        self.wcs.wcs.cdelt = [delta_lam, delta_xi]
        self.wcs.wcs.ctype = ['LINEAR', 'LINEAR']
        self.wcs.wcs.cname = ['WAVELEN', 'SLITPOS']
        self.wcs.wcs.cunit = ['um', 'arcsec']

        ## Alternative : xi = [0, 1], dimensionless
        self.wcsa = WCS(naxis=2)
        self.wcsa.wcs.crpix = [1, 1]
        self.wcsa.wcs.crval = [lam[0], 0]
        self.wcsa.wcs.pc = [[1, 0], [0, 1]]
        self.wcsa.wcs.cdelt = [delta_lam, 1./npix_xi]
        self.wcsa.wcs.ctype = ['LINEAR', 'LINEAR']
        self.wcsa.wcs.cname = ['WAVELEN', 'SLITPOS']
        self.wcsa.wcs.cunit = ['um', '']

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
