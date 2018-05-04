'''Spectroscopic source object'''
import sys
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy import constants as c

from scipy.interpolate import interp1d

class SpectralSource(object):
    '''Source object for the spectroscopy mode

    Create a source object consisting of a number of point sources
    and background sources. The latter fill the slit homogeneously, the
    former will be represented by the PSF.

    Parameters
    ----------
    srcspec : list of str
       list of FITS files containing 1D spectra treated as point sources
    srcpos : list of tuples [arcsec]
       list of source positions relative to the field centre
    bgspec : list of str
       list of FITS files containing 1D spectra filling the slit
    '''

    def __init__(self, cmds, srcspec, srcpos, bgspec):

        self.spectra = []
        if srcspec is not None:
            for thespec, thepos in zip(srcspec, srcpos):
                self.spectra.append(Spectrum(cmds, thespec, 'src', thepos))
        if bgspec is not None:
            for thespec in bgspec:
                self.spectra.append(Spectrum(cmds, thespec, 'bg'))

        self.dlam = self.min_dlam()

    def min_dlam(self):
        '''Determine the minimum wavelength step of the spectra'''
        dlam = []
        for spec in self.spectra:
            dlam.append(np.min(spec.lam[1:] - spec.lam[:-1]))

        return np.min(np.asarray(dlam))


class Spectrum(object):
    '''A single spectrum

    The default method expects a one-dimensional fits image. The wavelength
    vector is constructed from the WCS.

    Parameters
    ----------
    specfile : string
        fits file name
    spectype : string
        type of spectrum ('src' or 'bg')
    srcpos : tuple of floats
        position of the source (only for spectype=='src')
    '''

    def __init__(self, cmds, specfile, spectype=None, srcpos=None):

        ## TODO: This should not be hardcoded
        area_scope = 978 * u.m**2
        exptime = cmds["OBS_EXPTIME"] * u.s

        self.spectype = spectype
        #if srcpos is not None:
        self.srcpos = srcpos

        try:
            fluxunit = u.Unit(fits.getheader(specfile)['BUNIT'])
        except KeyError:
            print("Input file ", specfile,
                  ":\n    Required keyword 'BUNIT' not found")
            sys.exit(1)

        flux = fits.getdata(specfile) * fluxunit

        self.wcs = WCS(specfile)
        lamunit = self.wcs.wcs.cunit[0]
        self.lam = self.wcs.all_pix2world(np.arange(len(flux)), 0)[0]

        # Fluxes are converted to
        #       photons / s / m2 / arcsec2 / um  for bg spectra
        #       photons / s / m2 / um            for src spectra
        if spectype == 'bg':
            if fluxunit.is_equivalent("erg / (m2 um arcsec2 s)"):
                flux *= u.ph * (self.lam * u.m) / (c.h * c.c)
            elif fluxunit.is_equivalent("1 / (m2 um arcsec2 s)"):
                flux *= u.ph
            elif not fluxunit.is_equivalent("ph / (m2 um arcsec2 s)"):
                raise TypeError("Unknown units in " + specfile)

            flux = flux * area_scope * exptime
            self.fluxunit = 'ph / (um arcsec2)'
            self.flux = flux.to(u.Unit(self.fluxunit)).value

        elif spectype == 'src':
            if fluxunit.is_equivalent("erg / (m2 um s)"):
                flux *= u.ph * (self.lam * u.m) / (c.h * c.c)
            elif fluxunit.is_equivalent("1 / (m2 um s)"):
                flux *= u.ph
            elif not fluxunit.is_equivalent("ph / (m2 um s)"):
                raise TypeError("Unknown units in " + specfile)
            flux = flux * area_scope * exptime
            self.fluxunit = 'ph / um'
            self.flux = flux.to(u.Unit(self.fluxunit)).value

        ## Wavelengths will be stored in microns
        self.lamunit = 'um'
        self.lam = (self.lam * lamunit).to(self.lamunit).value  # from m to um

        # Cubic interpolation function
        self.interp = interp1d(self.lam, self.flux, kind='cubic',
                               bounds_error=False, fill_value=0.)
