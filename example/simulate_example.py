'''Example script for SimCADO spectroscopy mode

Call
----
python simulate_example.py -h -c <chipno> -s <sim_data_dir> configfile <PARAMETER_NAME=VALUE>

Command line options
--------------------
-h, --help : Print help and exit

-c, --chip  : int
   Number of chip to simulate

-s, --simdatadir : Path to the directory where SimCADO data are stored.
                   This can also be defined in the configfile (SIM_DATA_DIR)

Command line parameters
-----------------------
configfile : File with SimCADO and SpecCADO configuration parameters

parameters : strings of the form "PARAMETER_NAME=VALUE", where
    PARAMETER_NAME is a configuration parameter known to SpecCADO/SimCADO
'''

# 2019-03-31: Test atmospheric transmission on a constant source spectrum
# 2019-04-01: Move simulation functionality to speccado.simulation
# 2020-07-28: Test refactoring of layout.py to enable cube readin
# 2020-09-06: Make this the offcial simulate_example.py for SpecCADO

import sys
import getopt

import numpy as np

import simcado as sim
import speccado as sc


##################### MAIN ####################
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
            print(__doc__)
            sys.exit()
        elif opt in ('-c', '--chip'):
            if arg == 'all':
                chip = None
            else:
                try:
                    chip = np.int(arg)
                except ValueError:
                    print("Argument of -c/--chip must be integer or 'all'.")
        elif opt in ('-s', '--simdatadir'):
            sim_data_dir = arg

    pardict = dict()
    for arg in args:
        if '=' in arg:
            newpar = arg.split('=')
            try:
                newpar[1] = float(newpar[1])
            except ValueError:
                pass
            pardict[newpar[0]] = newpar[1]
        else:
            configfile = arg

    ## Commands to control the simulation
    print("Config file: ", configfile)
    if chip is None:
        print("Simulate all chips")
    else:
        print("Simulate chip ", chip)

    cmds = sim.UserCommands(configfile, sim_data_dir)

    # Optionally set some parameters explicitely.
    cmds['OBS_DIT'] = 60
    cmds['OBS_NDIT'] = 1
    cmds['FPA_LINEARITY_CURVE'] = 'none'

    cmds['SPEC_INTERPOLATION'] = 'spline'

    # Update cmds from command line arguments
    cmds.update(pardict)

    ## Define the source(s)  -- Only point sources for the moment
    specfiles = []
    sourcepos = []
    #specfiles = ['GW_Ori+9mag.fits']
    #sourcepos = [[0, 0]]
    #specfiles = ['GW_Ori+9mag.fits', 'GW_Ori+9mag.fits']
    #sourcepos = [[-1, 0], [1, 0]]

    ## Background spectra - these fill the slit
    bgfiles = ['atmo_emission.fits']
    #bgfiles = ['atmo_smooth.fits']
    #bgfiles = []

    ## Spectral cubes
    cubefiles = ['testcube_nosky.fits']
    #cubefiles = []

    outfile = sc.simulate(cmds, specfiles, sourcepos, bgfiles, cubefiles,
                          chip=chip)
    print("Simulated file is ", outfile)


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
