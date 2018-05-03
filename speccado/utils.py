'''Utility functions for the spectroscopy mode'''

def message(text, indent=""):
    '''Print a text message with indentation'''
    print(indent, text)


def bintab2img(infile, outfile, overwrite=False):
    '''Convert spectrum from fits table to fits image

    ESO level 3 spectra are provided as fits tables. This helper function
    converts them into 1D fits images with a WCS.
    '''
    from astropy.table import Table
    tab = Table.read(infile)
    lam = tab['WAVE'][0]
    lamunit = tab['WAVE'].unit
    flux = tab['FLUX'][0]
    fluxunit = tab['FLUX'].unit

    outwcs = WCS(naxis=1)
    outwcs.wcs.ctype = ['WAVE']
    outwcs.wcs.cunit = [lamunit.to_string()]
    outwcs.wcs.crpix = [1]
    outwcs.wcs.crval = [lam[0]]
    outwcs.wcs.cdelt = [np.mean(lam[1:] - lam[:-1])]

    pdu = fits.PrimaryHDU(flux, outwcs.to_header())
    pdu.header['BUNIT'] = fluxunit.to_string()
    pdu.writeto(outfile, overwrite=overwrite)
