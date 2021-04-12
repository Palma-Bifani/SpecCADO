"""
Unit tests for the conversion of input units
"""
import pytest
import numpy as np

from astropy import units as u

from speccado.source import convert_wav_units, convert_flux_units

class TestWavUnits:
    """Tests of function sc.source.convert_wav_units"""

    @pytest.mark.parametrize("inwav, testwav",
                             [(1.234 * u.nm, 0.001234 * u.um),
                              (98765 * u.angstrom, 9.8765 * u.um),
                              (3.24 * u.um, 3.24 * u.um),
                              (3.42e-06 * u.m, 3.42 * u.um)])
    def test_wavelength(self, inwav, testwav):
        """Conversion of various wavelength units"""
        outwav = convert_wav_units(inwav)
        assert np.isclose(outwav.value, testwav.value)

    @pytest.mark.parametrize("inwav, testwav",
                             [(8103727.71474878 * u.cm**(-1), 0.001234 * u.um),
                              (101250.44297069 * u.m**(-1), 9.8765 * u.um),
                              (0.30864198 * u.um**(-1), 3.24 * u.um),
                              (292.39766082 * u.mm**(-1), 3.42 * u.um)])
    def test_wavenumber(self, inwav, testwav):
        """Conversion of wavenumber to wavelength"""
        outwav = convert_wav_units(inwav)
        assert np.isclose(outwav.value, testwav.value)


    @pytest.mark.parametrize("inwav, testwav",
                             [(2.42943645e+05 * u.THz, 0.001234* u.um),
                              (3.03541192e+04 * u.GHz, 9.8765 * u.um),
                              (9.25285364e+07 * u.MHz, 3.24* u.um),
                              (8.76586135e+13 / u.s, 3.42 * u.um)])
    def test_frequency(self, inwav, testwav):
        """Conversion of frequency to wavelength"""
        outwav = convert_wav_units(inwav)
        assert np.isclose(outwav.value, testwav.value)

    @pytest.mark.parametrize("inwav, testwav",
                             [(1.60976161e-09 * u.erg, 0.001234 * u.um),
                              (0.12553455 * u.eV, 9.8765 * u.um),
                              (6.13100563e-20 * u.J, 3.24 * u.um),
                              (0.00036253 * u.keV, 3.42 * u.um)])
    def test_photon_energy(self, inwav, testwav):
        """Conversion of photon energy to wavelength"""
        outwav = convert_wav_units(inwav)
        assert np.isclose(outwav.value, testwav.value)

    def test_incompatible_units(self):
        """Test one case of non-equivalent units"""
        inwav = 2.34 * u.m / u.s
        with pytest.raises(u.UnitConversionError):
            convert_wav_units(inwav)



class TestFluxUnits:
    """Tests of function sc.source,convert_flux_units"""

    def test_src_flam_si(self):
        """
        Conversion of SI input units to internal units, point source
        """
        si_flux = np.array([1, 3.4]) * u.J / u.m**2 / u.m / u.s
        wavelength = np.array([3.1, 4.3]) * u.um

        testflux = np.array([1.56057616e+13,
                             7.35987854e+13]) * u.ph / u.s / u.m**2 / u.um

        outflux = convert_flux_units(si_flux, wavelength)
        assert outflux.unit.is_equivalent(
            si_flux.unit,
            equivalencies=u.spectral_density(wavelength))
        assert np.allclose(outflux.value, testflux.value)


    def test_src_fnu(self):
        """
        Conversion of f_nu to f_lambda
        """
        f_nu = np.array([2.3e+4, 99.4e+4]) * u.erg / u.s / u.m**2 / u.Hz
        wavelength = np.array([3.1, 4.3]) * u.um

        testf_lam = np.array([1.11972176e+30,
                              3.48868619e+31]) * u.ph / u.s / u.m**2 / u.um

        f_lam = convert_flux_units(f_nu, wavelength)

        assert f_lam.unit.is_equivalent(
            f_nu.unit,
            equivalencies=u.spectral_density(wavelength))
        assert np.allclose(f_lam.value, testf_lam.value)


    def test_src_jansky(self):
        """
        Conversion of Jy to internal units, point source

        The test uses a reference wavelength of 3 * u.um.
        """
        jy_flux = np.array([1, 3.4]) * u.Jy
        lam_ref = np.array([3.0, 5.7]) * u.um
        testflux = np.array([5030634.01565452,
                             9002187.18590809]) * u.ph / u.m**2 / u.s / u.um

        outflux = convert_flux_units(jy_flux, wav=lam_ref)

        assert outflux.unit.is_equivalent(
            jy_flux.unit,
            equivalencies=u.spectral_density(wav=lam_ref))
        assert np.allclose(testflux.value, outflux.value)


    def test_bg_flam_per_deg2(self):
        """
        Test a simple conversion of angular scale

        This uses a influx per deg^2
        """
        influx = 2.34 * u.erg / u.s / u.m**2 / u.um / u.deg**2
        lam_ref = 4.2 * u.um

        testflux = 381753.84604285 * u.ph / u.s / u.m**2 / u.um / u.arcsec**2

        outflux = convert_flux_units(influx, lam_ref)
        print(testflux)
        print(outflux)
        assert np.allclose(outflux.value, testflux.value)


    def test_bg_flam_per_rad(self):
        """
        Test a simple conversion of angular scale

        This uses influx per rad^2
        """
        influx = 2.34 * u.erg / u.s / u.m**2 / u.um / u.rad**2
        lam_ref = 4.2 * u.um

        testflux = 116.2888 * u.ph / u.s / u.m**2 / u.um / u.arcsec**2

        outflux = convert_flux_units(influx, lam_ref)
        print(testflux)
        print(outflux)
        assert np.allclose(outflux.value, testflux.value)


    def test_bg_jansky_per_deg2(self):
        """
        Test conversion from Jy/deg^2
        """
        influx = 1.2e-5 * u.Jy / u.deg**2
        lam_ref = 9.2 * u.um

        testflux = 1.51891124e-06 * u.ph / u.s / u.m**2 / u.um / u.arcsec**2

        outflux = convert_flux_units(influx, lam_ref)
        assert np.allclose(outflux.value, testflux.value)
