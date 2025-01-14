'''Spectroscopic source object'''
import sys
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

from scipy.interpolate import interp1d

class SpectralSource():
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

    def __init__(self, srcspec, srcpos, bgspec):

        self.spectra = []
        if srcspec is not None:
            for thespec, thepos in zip(srcspec, srcpos):
                self.spectra.append(Spectrum(thespec, 'src', thepos))
        if bgspec is not None:
            for thespec in bgspec:
                self.spectra.append(Spectrum(thespec, 'bg'))

        self.dlam = self.min_dlam()

    def min_dlam(self):
        '''Determine the minimum wavelength step of the spectra'''
        dlam = []
        for spec in self.spectra:
            dlam.append(np.min(spec.lam[1:] - spec.lam[:-1]))

        return np.min(np.asarray(dlam))


class Spectrum():
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

    def __init__(self, specfile, spectype=None, srcpos=None):

        ## TODO: This should not be hardcoded
        area_scope = 978 * u.m**2

        self.spectype = spectype
        self.srcpos = srcpos

        try:
            fluxunit = u.Unit(fits.getheader(specfile)['BUNIT'])
        except KeyError:
            print("Input file ", specfile,
                  ":\n    Required keyword 'BUNIT' not found")
            sys.exit(1)

        flux = fits.getdata(specfile) * fluxunit

        self.wcs = WCS(specfile)

        # TODO we need to be able to use cubes at some point
        lamunit = u.Unit(self.wcs.wcs.cunit[0])

        # Convert to internal units: microns (u.um)
        temp_lam = self.wcs.all_pix2world(np.arange(len(flux)), 0)[0] * lamunit
        temp_lam = convert_wav_units(temp_lam)
        self.lam = temp_lam.value
        self.lamunit = str(temp_lam.unit)

        # Fluxes are converted to
        #       photons / s / m2 / arcsec2 / um  for bg spectra
        #       photons / s / m2 / um            for src spectra
        flux = area_scope * convert_flux_units(flux, self.lam * u.Unit(self.lamunit))
        self.flux = flux.value
        self.fluxunit = str(flux.unit)

        # Cubic interpolation function
        self.interp = interp1d(self.lam, self.flux, kind='cubic',
                               bounds_error=False, fill_value=0.)




### Functions for input unit conversion
def convert_wav_units(inwav):
    """
    Convert spectral units to micrometers

    "Spectral units" refers to all units that can be assigned to the
    arguments of a spectrum, i.e. wavelength, wave number, and frequency.
    Internally, SpecCADO uses wavelengths with units um (microns).

    Parameters
    ----------
    inwav : a Quantity object
    """
    return inwav.to(u.um, equivalencies=u.spectral())


def convert_flux_units(influx, wav=None):
    """
    Convert flux units

    "Flux units" refers to both integrated fluxes for point sources
    and to surface brightnesses.

    The internal units are:
    - Flux:   ph / (m2 um s)
    - surface flux:  ph / (m2 um s arcsec2)

    Parameters
    ----------
    influx : list of astropy.unit.Quantity
        These can be energy or photon fluxes
    wav : float, nd.array
        Wavelengths at which influx is given. Default is None, which
        is okay if the conversion is independent of wavelength.
    """
    if wav is not None:
        useequivalencies = u.spectral_density(wav)
    else:
        useequivalencies = None

    # Check whether we have a surface brightness
    inunit = influx.unit
    factor = 1
    for un, power in zip(inunit.bases, inunit.powers):
        if un.is_equivalent(u.arcsec):
            conversion = (un.to(u.arcsec) / un)**power
            influx *= conversion
            factor = u.arcsec**(-2)

    outflux = influx.to(u.ph / u.m**2 / u.um / u.s,
                        useequivalencies)

    return outflux * factor
