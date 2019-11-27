"""
Unit tests for the conversion of input units
"""
import pytest
import numpy as np

from astropy import units as u

from speccado.source import convert_wav_units
#from speccado.source import convert_flux_units

class TestWavUnits:
    """Tests of function sc.source.convert_wav_units"""

    def test_wavelength(self):
        """Conversion of various wavelength units"""

        inlist = [1.234 * u.nm,
                  98765 * u.angstrom,
                  3.24 * u.um,
                  3.42e-06 * u.m]
        testlist = [0.001234 * u.um,
                    9.8765 * u.um,
                    3.24 * u.um,
                    3.42 * u.um]
        for inwav, testwav in zip(inlist, testlist):
            outwav = convert_wav_units(inwav)
            assert np.isclose(outwav.value, testwav.value)


    def test_wavenumber(self):
        """Conversion of wavenumber to wavelength"""
        testlist = np.array([0.001234,
                             9.8765,
                             3.24,
                             3.42]) * u.um
        inlist = [8103727.71474878 * u.cm**(-1),
                  101250.44297069 * u.m**(-1),
                  0.30864198 * u.um**(-1),
                  292.39766082 * u.mm**(-1)]
        for inwav, testwav in zip(inlist, testlist):
            outwav = convert_wav_units(inwav)
            assert np.isclose(outwav.value, testwav.value)


    def test_frequency(self):
        """Conversion of frequency to wavelength"""
        testlist = np.array([0.001234,
                             9.8765,
                             3.24,
                             3.42]) * u.um

        inlist = [2.42943645e+05 * u.THz,
                  3.03541192e+04 * u.GHz,
                  9.25285364e+07 * u.MHz,
                  8.76586135e+13 / u.s]

        for inwav, testwav in zip(inlist, testlist):
            outwav = convert_wav_units(inwav)
            assert np.isclose(outwav.value, testwav.value)


    def test_photon_energy(self):
        """Conversion of photon energy to wavelength"""
        testlist = np.array([0.001234,
                             9.8765,
                             3.24,
                             3.42]) * u.um
        inlist = [1.60976161e-09 * u.erg,
                  0.12553455 * u.eV,
                  6.13100563e-20 * u.J,
                  0.00036253 * u.keV]

        for inwav, testwav in zip(inlist, testlist):
            outwav = convert_wav_units(inwav)
            assert np.isclose(outwav.value, testwav.value)


    def test_incompatible_units(self):
        """Test one case of non-equivalent units"""
        inwav = 2.34 * u.m / u.s
        with pytest.raises(u.UnitConversionError):
            outwav = convert_wav_units(inwav)



#class TestFluxUnits:
#    """Tests of function sc.source,convert_flux_units"""
#
#    def test_src_si_units(self):
#        """
#        Conversion of SI input units to internal units, point source
#        """
#        si_flux = np.array([1, 3.4]) * u.J / u.m**2 / u.m / u.s
#        sc_flux = np.array([10, 34]) * u.erg / u.m**2 / u.um / u.s
#        outflux = convert_flux_units(si_flux)
#        assert outflux.unit.is_equivalent(
#            si_flux.unit,
#            equivalencies=u.spectral_density(wav=1*u.um))
#        assert np.allclose(sc_flux.value, outflux.value)
#
#    def test_src_jansky(self):
#        """
#        Conversion of Jy to internal units, point source
#
#        The test uses a reference wavelength of 3 * u.um.
#        """
#        jy_flux = np.array([1, 3.4]) * u.Jy
#        lam_ref = 3 * u.um
#        sc_flux = 3.33102731e-06 * u.erg / u.m**2 / u.s / u.um
#        outflux = convert_flux_units(jy_flux, wav=lam_ref)
#
#        assert outflux.unit.is_equivalent(
#            jy_flux.unit,
#            equivalencies=u.spectral_density(wav=lam_ref))
#        assert np.allclose(sc_flux.value, outflux.value)
