'''Example script for SimCADO spectroscopy mode

Call
----
    python simulate_example configfile <sim_data_dir>

Command line parameters
-----------------------
configfile : File with SimCADO and SpecCADO configuration parameters

sim_data_dir : Path to the directory where SimCADO data are stored.
               This can also be defined in the configfile (SIM_DATA_DIR)
'''

import sys

from scipy.interpolate import interp1d

import simcado as sim
import speccado as sc


##################### MAIN ####################
def main(configfile, sim_data_dir=None):
    '''Main function

    Define source components, commands, psf, detector, etc. here

    Parameters
    ----------
    configfile : [str]
    sim_data_dir : SimCADO data directory (None, if defined in configfile)
    '''

    ## Commands to control the simulation
    print("Config file: ", configfile)
    cmds = sim.UserCommands(configfile, sim_data_dir)

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
    psfobj = sc.prepare_psf(cmds['SCOPE_PSF_FILE'])

    # Create detector
    detector = sim.Detector(cmds, small_fov=False)

    # Create transmission curve.
    # Here we take the  transmission from the simcado optical train,
    # this includes atmospheric, telescope and instrumental
    # transmissivities.
    # You can create your own transmission by suitably defining
    # tc_lam (in microns) and tc_val (as numpy arrays).
    opttrain = sim.OpticalTrain(cmds)
    tc_lam = opttrain.tc_source.lam_orig
    tc_val = opttrain.tc_source.val_orig
    transmission = interp1d(tc_lam, tc_val, kind='linear',
                            bounds_error=False, fill_value=0.)

    ## Prepare the order descriptions
    tracelist = sc.layout.read_spec_order(cmds['SPEC_ORDER_LAYOUT'])

    #    tracelist = list()
    #for lfile in layoutlist:
    #    tracelist.append(sc.SpectralTrace(lfile))

    #sc.do_one_chip(detector.chips[3], srcobj, psfobj, tracelist, cmds,
    #            transmission)
    sc.do_all_chips(detector, srcobj, psfobj, tracelist, cmds, transmission)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n   sys,argv[0] configfile <sim_data_dir>")
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main(sys.argv[1], sys.argv[2])
