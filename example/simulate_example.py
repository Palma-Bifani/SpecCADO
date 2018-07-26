'''Example script for SimCADO spectroscopy mode'''

import glob

import datetime

import numpy as np

from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy import constants as c

import simcado as sim
import speccado as sc


##################### MAIN ####################
def main():
    '''Main function

    Define source components, commands, psf, detector, etc. here
    '''
    ## Commands to control the simulation
    #cmds = sim.UserCommands('spectro_HK.config')
    cmds = sim.UserCommands('spectro_IJ.config')

    # Optionally set some parameters explicitely.
    cmds['OBS_EXPTIME'] = 60
    cmds['FPA_LINEARITY_CURVE'] = 'none'

    cmds['SPEC_INTERPOLATION'] = 'spline'
    #cmds['SPEC_INTERPOLATION'] = 'nearest'

    ## Define the source(s)  -- Only point sources for the moment
    specfiles = ['GW_Ori+9mag.fits', 'GW_Ori+9mag.fits']
    sourcepos = [[-1, 0], [1, 0]]

    ## Background spectra - these fill the slit
    bgfiles = ['atmo_emission.fits']

    ## Create source object. The units of the spectra are
    ##         - ph / um  for source spectra
    ##         - erg / (um arcsec2) for background spectra
    srcobj = sc.SpectralSource(cmds, specfiles, sourcepos, bgfiles)

    ## Load the psf
    psfobj = sc.prepare_psf("PSF_SCAO_120.fits")

    # Create detector
    detector = sim.Detector(cmds, small_fov=False)

    # Create transmission curve.
    # Here we take the  transmission from the simcado optical train,
    # this includes atmospheric, telescope and instrumental
    # transmissivities.
    # You can create your own transmission by suitably defining
    # tc_lam (in microns) and tc_val (as numpy arrays).
    opttrain = sim.OpticalTrain(cmds)
    tc_lam = opttrain.tc_mirror.lam_orig
    tc_val = opttrain.tc_mirror.val_orig
    transmission = interp1d(tc_lam, tc_val, kind='linear',
                            bounds_error=False, fill_value=0.)

    ## Prepare the order descriptions
    #layoutlist = glob.glob("?_?.TXT")
    layoutfile = 'specorders-160904.fits'
    tracelist = sc.layout.read_spec_order(layoutfile)

    #    tracelist = list()
    #for lfile in layoutlist:
    #    tracelist.append(sc.SpectralTrace(lfile))

    #sc.do_one_chip(detector.chips[3], srcobj, psfobj, tracelist, cmds,
    #            transmission)
    sc.do_all_chips(detector, srcobj, psfobj, tracelist, cmds, transmission)


if __name__ == "__main__":
    main()
