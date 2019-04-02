'''Rectify simulated MICADO spectra

Call
----
   python rectify_example fitsfile

Parameters
----------
   fitsfile : Result of a SpecCADO simulation
'''


import sys
import glob

from astropy.io import fits
import simcado as sim
import speccado as sc


def main(filename):
    '''Main function'''

    # Open the FITS file. This should contain all the information needed
    hdulist = fits.open(filename)

    # Collect a few parameters
    params = dict()
    try:
        params['pixsize'] = hdulist[0].header['CDELT1A']  # mm
    except KeyError:
        params['pixsize'] = hdulist[1].header['CDELT1A']  # mm
    params['pixscale'] = hdulist[0].header['SIM_DETECTOR_PIX_SCALE']
    params['slit_length'] = hdulist[0].header['SPEC_SLIT_LENGTH']

    print("Create Chip objects ", filename)
    if hdulist[0].data is None:
        chiplist = [sc.rectify.SpecChip(hdu) for hdu in hdulist[1:]]
    else:
        chiplist = [sc.rectify.SpecChip(hdu) for hdu in hdulist]

    ## Prepare the order descriptions
    spec_order_layout = hdulist[0].header['SPEC_ORDER_LAYOUT']
    tracelist = sc.layout.read_spec_order(spec_order_layout)
    for trace in tracelist:
        sc.rectify.rectify_trace(trace, chiplist, params)

    # hdulist has to be kept open till now
    hdulist.close()


# Main script
if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit("Please provide a file name")
    else:
        print(sys.argv[1])
        main(sys.argv[1])
