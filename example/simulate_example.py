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

# 2019-03-31: Test atmospheric transmission on a constant source spectrum
# 2019-04-01: Move simulation functionality to speccado.simulation

import sys
import getopt

import numpy as np

import simcado as sim
import speccado as sc


##################### MAIN ####################
#def main(configfile, chip=None, sim_data_dir=None):
def main(progname, argv):
    '''Main function

    Define source components, commands, psf, detector, etc. here

    Parameters
    ----------
    configfile : [str]
    sim_data_dir : SimCADO data directory (None, if defined in configfile)
    '''
    # defaults
    chip = None
    sim_data_dir = '.'

    try:
        opts, args = getopt.getopt(argv, 'hc:s:',
                                   ["help", "chip=", "simdatadir="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(progname, "-c <chipno> -s <simdatadir> configfile")
            sys.exit()
        elif opt in ('-c', '--chip'):
            if arg == 'all':
                chip = None
            else:
                try:
                    chip = np.int(arg)
                except ValueError:
                    print("Argument of -c/--chip need to be integer or 'all'.")
        elif opt in ('-s', '--simdatadir'):
            sim_data_dir = arg

    configfile = args[0]

    ## Commands to control the simulation
    print("Config file: ", configfile)
    if chip is None:
        print("Simulate all chips")
    else:
        print("Simulate chip ", chip)

    cmds = sim.UserCommands(configfile, sim_data_dir)

    # Optionally set some parameters explicitely.
    cmds['OBS_EXPTIME'] = 60
    cmds['FPA_LINEARITY_CURVE'] = 'none'

    cmds['SPEC_INTERPOLATION'] = 'spline'

    ## Define the source(s)  -- Only point sources for the moment
    specfiles = ['GW_Ori+9mag.fits', 'GW_Ori+9mag.fits']
    sourcepos = [[-1, 0], [1, 0]]

    # Constant source spectrum
    #specfiles = ['const.fits']
    #sourcepos = [[0, 0]]

    ## Background spectra - these fill the slit
    bgfiles = ['atmo_emission.fits']
    #bgfiles = []

    outfile = sc.simulate(cmds, specfiles, sourcepos, bgfiles, chip=chip)
    print("Simulated file is ", outfile)


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
