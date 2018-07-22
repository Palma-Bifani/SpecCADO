'''Extract 1D spectrum from simulated MICADO spectrum'''

# Assume there is a single order to be extracted, this is what extract_1d
# is working on.
# The trace may be on several chips. Which ones? - From the tracefile
# we get maximum and minimum wavelength, and ranges in X and Y (in mm on
# the FP). X and Y can be located on the chips using the WCS PIX2FP and
# function sc.is_order_on_chip - unfortunately, this requires a Chip object,
# which may not be good for this purpose. What if we pass only a wcs instead
# of the full chip?

import sys
import glob

from astropy.io import fits
import simcado as sim
import speccado as sc


def main(filename):
    '''Main function'''

    ## TODO Replace with UserCommands()
    params = dict()
    params['pixsize'] = 0.015  # mm
    params['pixscale'] = 0.004  # arcsec
    params['slit_length'] = 3   # arcsec

    print("Create Chip objects ", filename)
    hdulist = fits.open(filename)
    chiplist = [sc.rectify.SpecChip(hdu) for hdu in hdulist[1:]]

    ## Prepare the order descriptions
    layoutlist = sorted(glob.glob("?_?.TXT"))
    for lfile in layoutlist:
        sc.rectify.rectify_trace(lfile, chiplist, params)

    # hdulist has to be kept open till now
    hdulist.close()


# Main script
if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit("Please provide a file name")
    else:
        print(sys.argv[1])
        main(sys.argv[1])
