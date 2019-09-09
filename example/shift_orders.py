'''Script to shift orders in the dispersion direction

Order traces have been provided for the nominal slit in MICADO. In
order to fill the gaps between detectors and achieve continuous
wavelength coverage, a second (set of) slit(s) will be provided
shifted perpendicular to the shift. This script provides a simple
means to simulate spectra from the shifted slit(s) by adding a
constant value to the y positions in the order definition files.
'''

import sys
from datetime import date
from astropy.io import fits

def usage(scriptname):
    '''Print usage string'''
    print(f'''Usage:
   {scriptname} infile shift outfile''')


def shift_orders(infile, shift, outfile):
    '''Shift y columns in infile by shift mm and write to outfile'''
    shift = float(shift)
    inhdul = fits.open(infile)

    outhdul = fits.HDUList()

    # Amend primary header
    today = date.today().isoformat()
    header = inhdul[0].header
    header['DATE'] = today
    header.add_history(f"[{today}] Shift by {shift} mm in y")

    outhdul.append(fits.PrimaryHDU(header=header))

    # Step through the table extensions
    for curhdu in inhdul[1:]:
        table = curhdu.data
        for colname in table.names:
            if colname[0] == 'y':
                table[colname] += shift

        outhdul.append(fits.TableHDU(table, name=curhdu.name))

    outhdul.writeto(outfile)
    outhdul.close()
    inhdul.close()

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] == '-h':
        usage(sys.argv[0])
    else:
        shift_orders(sys.argv[1], sys.argv[2], sys.argv[3])
