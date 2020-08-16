'''Spectroscopic simulation'''
import datetime
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.io import ascii as ioascii
import simcado as sim
from .utils import message
from .layout import XiLamImage, read_spec_order
from .source import SpectralSource
from .psf import prepare_psf

def simulate(cmds, specfiles, sourcepos, bgfiles, cubefiles, chip=None):
    '''Perform SpecCADO simulation

Parameters
----------
    cmds : simcado.UserCommands
    specfiles : list of str
        1D FITS files describing source spectra
    sourcepos : list of tuples
        (x,y) position of source within slit
    bgfiles : list of str
        1D FITS files describing background spectra
    cubefiles : list of str
        3D FITS files containing spectral cubes
    chip : integer
        Number of chip in MICADO array (1..9)
        If None, the entire FPA (9 chips) is simulated.

Returns
-------
    name of output file
'''

    ## Create source object. The units of the spectra are
    ##         - ph / um  for source spectra
    ##         - erg / (um arcsec2) for background spectra
    srcobj = SpectralSource(cubespec=cubefiles,
                            pointspec=specfiles,
                            pointpos=sourcepos,
                            bgspec=bgfiles)

    ## Load the psf
    psfobj = prepare_psf(cmds['SCOPE_PSF_FILE'])

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
    tc_interp = interp1d(tc_lam, tc_val, kind='linear',
                         bounds_error=False, fill_value=0.)
    atmo_tc_tab = ioascii.read(cmds['ATMO_TC'])
    atmo_lam = atmo_tc_tab['lambda']
    atmo_val = atmo_tc_tab['X1.0']
    atmo_val = atmo_val * tc_interp(atmo_lam)

    transmission = interp1d(atmo_lam, atmo_val,
                            kind='linear', bounds_error=False, fill_value=0.)
    np.savetxt("transmission.txt", (atmo_lam, atmo_val))

    ## Prepare the order descriptions
    tracelist = read_spec_order(cmds['SPEC_ORDER_LAYOUT'])

    #    tracelist = list()
    #for lfile in layoutlist:
    #    tracelist.append(sc.SpectralTrace(lfile))
    if chip is None:
        outfile = do_all_chips(detector, srcobj, psfobj, tracelist, cmds,
                               transmission)
    else:
        outfile = do_one_chip(detector, chip, srcobj, psfobj,
                              tracelist, cmds, transmission)

    return outfile


def map_spectra_to_chip(chip, src, psf, tracelist, cmds, transmission):
    '''Map the spectra to a Micado chip

    Parameters
    ----------
    chip : [simcado.detector.Chip]
    src : instance of class SpectralSource
    psf : instance of class simcado.psf.PSF
    tracelist : [list]
        List of SpectralTrace objects
    cmds : instance of class simcado.UserCommands
    transmission : interpolation object
    '''

    indent = ""

    ## Initialise the (ideal) chip image
    chip.image = np.zeros((chip.naxis2, chip.naxis1), dtype=np.float32)

    ## Convert chip limits from arcsec to mm
    pixscale = cmds['SIM_DETECTOR_PIX_SCALE']  # arcsec
    pixsize = chip.wcs_fp.wcs.cdelt[0]         # mm
    platescale = pixscale / pixsize            # arcsec / mm

    # TODO do this via wcs_fp?
    chip.xmin_um = chip.x_min / platescale
    chip.xmax_um = chip.x_max / platescale
    chip.ymin_um = chip.y_min / platescale
    chip.ymax_um = chip.y_max / platescale
    chip.xcen_um = chip.x_cen / platescale
    chip.ycen_um = chip.y_cen / platescale

    ## arrays for the edges of the chip pixels
    pix_edges_x = chip.xmin_um + np.arange(0, chip.naxis1 + 1) * pixsize
    pix_edges_y = chip.ymin_um + np.arange(0, chip.naxis2 + 1) * pixsize

    indent += "    "
    message("Extent in x: " + str(chip.xmin_um) + " to " + str(chip.xmax_um) +
            " --> " + str(chip.xmax_um - chip.xmin_um) + " um", indent)
    message("Extent in y: " + str(chip.ymin_um) + " to " + str(chip.ymax_um) +
            " --> " + str(chip.ymax_um - chip.ymin_um) + " um", indent)
    message("Centred at:  " + str(chip.xcen_um) + " -- " + str(chip.ycen_um),
            indent)
    indent = indent[:-4]


    # Step 512 lines at a time
    nchiplines = 512
    chip_j1 = 0
    chip_j2 = np.min([chip.naxis2, nchiplines])
    while chip_j2 <= chip.naxis2:
        indent = ''
        message("")

        # range of slice in focal-plane coordinates
        ymin = pix_edges_y[chip_j1]
        ymax = pix_edges_y[chip_j2]

        message("Rows " + str(chip_j1) + " to " + str(chip_j2) + " : " +
                str(ymin) + " to " + str(ymax), indent)

        indent += "    "
        for spectrace in tracelist:

            # Does the order appear on the chip?
            slitlength = cmds['SPEC_SLIT_LENGTH']
            if not spectrace.is_on_chip(chip, slitlength, ymin, ymax):
                message(spectrace.name + " is not on chip", indent)
                continue

            # Need to know the wavelength range for chip slice
            # Maybe extend lambda range for safety?
            trace_xcen = spectrace.centre_trace((ymin + ymax) / 2)
            lam_min, lam_max = spectrace.xy2lam(np.array([trace_xcen,
                                                          trace_xcen]),
                                                np.array([ymin, ymax]))
            # CHECK extend lambda range by 30 per cent either side
            # 30 per cent seems to be sufficient, but overlaps counted twice?
            lam_range = lam_max - lam_min
            lam_max += 0.3 * lam_range
            lam_min -= 0.3 * lam_range
            xcen = spectrace.xilam2x(0., (lam_min + lam_max) / 2)
            message(spectrace.name + ":", indent)
            indent += "    "
            message("lam_min: " + str(lam_min), indent)
            message("lam_max: " + str(lam_max), indent)
            message("xcen:    " + str(xcen), indent)

            # wavelength step per detector pixel at centre of slice
            # TODO: use array instead of mean value
            dlam_per_pix = spectrace.dlam_by_dy(xcen,
                                                (ymin + ymax) / 2) * pixsize
            xi_min = spectrace.layout['xi1'].min()
            try:
                xilam = XiLamImage(src, psf, lam_min, lam_max,
                                   xi_min, dlam_per_pix,
                                   cmds, transmission)
            except ValueError:
                message(" ---> " + spectrace.file + "[" + spectrace.name +
                        "] gave ValueError", indent)
                indent = indent[:-4]
                continue

            xilam.writeto("xilam_image.fits", overwrite=True)
            xilam_img = xilam.image
            npix_xi, npix_lam = xilam.npix_xi, xilam.npix_lam
            xilam_wcs = xilam.wcs

            # Now map the source to the focal plane
            # TODO : These are linear - compute directly
            #lam = xilam_wcs.all_pix2world(np.arange(npix_lam), 0.5, 0)
            #lam = lam[0]
            #xi = xilam_wcs.all_pix2world(lam[0], np.arange(npix_xi), 0)
            #xi = xi[1]
            lam = xilam.lam
            xi = xilam.xi

            ## PRELIM: image interpolation function
            #image_interp = RectBivariateSpline(xi, lam, xilam_img)
            image_interp = xilam.interp

            # These are needed to determine xmin, xmax, ymin, ymax
            xlims = spectrace.xilam2x(xi[[0, -1, -1, 0]], lam[[0, 0, -1, -1]])
            if xlims.max() < chip.xmin_um or xlims.min() > chip.xmax_um:
                indent = indent[:-4]
                continue
            else:
                xmax = min(xlims.max(), chip.xmax_um)
                xmin = max(xlims.min(), chip.xmin_um)

            # These should only be used for tests
            ylims = spectrace.xilam2y(xi[[0, -1, -1, 0]], lam[[0, 0, -1, -1]])
            ymax_test = ylims.max()
            ymin_test = ylims.min()

            # I think I ought to adjust xmin, xmax, etc. to correspond to chip
            # pixel locations...
            #osample_x = np.int(image.shape[0] / ((xmax - xmin) / \
            #                   obs_params['pixsize']))
            # TODO: Is oversampling necessary with spline interpolation?
            if cmds['SPEC_INTERPOLATION'] == 'nearest':
                osample_x = 2
                osample_y = np.int(xilam_img.shape[1] / nchiplines)
            else:
                osample_x = 1
                osample_y = 1

            message("osample_x: " + str(osample_x), indent)
            message("osample_y: " + str(osample_y), indent)
            n_x = np.int((xmax - xmin) / pixsize) * osample_x
            n_y = np.int((ymax - ymin) / pixsize) * osample_y

            # This is where the spectrum fits into the chip image
            imax = (xmax - pix_edges_x[0]) / pixsize
            imin = (xmin - pix_edges_x[0]) / pixsize
            jmax = (ymax - pix_edges_y[0]) / pixsize
            jmin = (ymin - pix_edges_y[0]) / pixsize

            # number of fractional images covered - need to pad to integer
            n_i = min(np.ceil(imax - imin).astype(np.int),
                      chip.naxis1)
            n_j = nchiplines

            # Range in pixels : i
            istart = max(0, np.floor(imin).astype(np.int))
            iend = istart + n_i
            if iend > chip.naxis1:
                iend = chip.naxis1
                istart = max(iend - n_i, 0)

            # Range in pixels : j (should be jstart = chip_j1, jend = chip_j2)
            jstart = max(0, np.floor(jmin).astype(np.int))
            jend = jstart + n_j
            if jend > chip.naxis2:
                jend = chip.naxis2
                jstart = jend - n_j

            ## TODO : SOMETHING WRONG HERE
            xstart = pix_edges_x[istart] + 0.5 * pixsize
            xend = pix_edges_x[iend] - 0.5 * pixsize
            ystart = ymin
            yend = ymax
            XIMG_fpa, YIMG_fpa = np.meshgrid(np.linspace(xstart, xend,
                                                         n_i * osample_x,
                                                         dtype=np.float32),
                                             np.linspace(ystart, yend,
                                                         n_j * osample_y,
                                                         dtype=np.float32))

            buffer = 5e6
            nbufferlines = np.ceil(buffer / XIMG_fpa.shape[0]).astype(int)
            nlines = XIMG_fpa.shape[0]
            lower = 0
            upper = np.min([nbufferlines, nlines])
            while upper <= nlines:
                # Image mapping (xi, lambda) on the focal plane
                XI_fpa = (spectrace.xy2xi(XIMG_fpa[lower:upper, :],
                                          YIMG_fpa[lower:upper, :]).
                          astype(np.float32))
                LAM_fpa = (spectrace.xy2lam(XIMG_fpa[lower:upper, :],
                                            YIMG_fpa[lower:upper, :]).
                           astype(np.float32))

                # Mask everything outside the slit
                mask = (XI_fpa >= xi[0]) & (XI_fpa <= xi[-1])
                XI_fpa *= mask
                LAM_fpa *= mask

                # We're using XIMG_fpa as output, reset part already used
                XIMG_fpa[lower:upper, :] = 0

                # Convert to pixel images (for nearest-neighbour or something)
                # These are the pixel coordinates in image corresponding to
                # xi, lambda
                # It's much quicker to do the linear transformation by hand
                # than to use the astropy.wcs functions for conversion.
                I_IMG = ((LAM_fpa - xilam_wcs.wcs.crval[0])
                         / xilam_wcs.wcs.cdelt[0]).astype(int)
                J_IMG = ((XI_fpa - xilam_wcs.wcs.crval[1])
                         / xilam_wcs.wcs.cdelt[1]).astype(int)

                # truncate images to remove pixel coordinates outside the image
                ijmask = ((I_IMG >= 0) * (I_IMG < npix_lam)
                          * (J_IMG >= 0) * (J_IMG < npix_xi))

                # do the actual interpolation
                if cmds["SPEC_INTERPOLATION"] == "nearest":
                    I_IMG[(I_IMG < 0) | (I_IMG >= npix_lam)] = 0
                    J_IMG[(J_IMG < 0) | (J_IMG >= npix_xi)] = 0
                    XIMG_fpa[lower:upper, :] = image[J_IMG, I_IMG] * ijmask
                elif cmds["SPEC_INTERPOLATION"] == "spline":
                    XIMG_fpa[lower:upper, :] = (image_interp(XI_fpa, LAM_fpa,
                                                             grid=False)
                                                * ijmask
                                                * pixscale * dlam_per_pix)
                else:
                    raise ValueError("SPEC_INTERPOLATION unknown: " +
                                     cmds["SPEC_INTERPOLATION"])

                if upper == nlines:
                    break
                lower = upper
                upper = np.min([upper + nbufferlines, nlines])

            # Output image is XIMG_fpa. This should be mapped to chip.image
            # at the correct position
            chip_i1 = np.int((xmin - chip.xmin_um) /
                             (chip.xmax_um - chip.xmin_um) * chip.naxis1) + 1
            chip_i2 = np.int((xmax - chip.xmin_um) /
                             (chip.xmax_um - chip.xmin_um) * chip.naxis1)

            chip.image[jstart:jend, istart:iend] += \
               XIMG_fpa.reshape(n_j, osample_y,
                                n_i, osample_x).sum(axis=3).sum(axis=1)

            indent = indent[:-4]

            # End loop over spectral traces

        # Update
        if chip_j2 >= chip.naxis2:
            break
        chip_j1 = chip_j2
        chip_j2 = min(chip_j2 + nchiplines, chip.naxis2)


def do_all_chips(detector, src, psf, tracelist, cmds, transmission, write=True):
    '''Map spectra to the full MICADO detector plane'''

    for chip in detector.chips:
        message("Working on chip " + str(chip.id))
        map_spectra_to_chip(chip, src, psf, tracelist, cmds, transmission)
        chip.array = chip.image.T
        message("DONE with chip " + str(chip.id))
        message("----------------------------------------------\n")

    if write:
        timestamp = '{:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now())
        filename = "detector-" + timestamp + ".fits"
        detector.read_out(filename=filename)
        return filename
    else:
        return detector


def do_one_chip(detector, chipno, src, psf, tracelist, cmds, transmission):
    '''Map spectra onto a single chip'''
    chip = detector.chips[chipno - 1]

    # Create a new detector with a single chip
    newdetector = sim.Detector(cmds, small_fov=False)
    newdetector.chips = [chip]

    # Run simulation, returns detector
    newdetector = do_all_chips(newdetector, src, psf, tracelist, cmds,
                               transmission, write=False)

    # write out
    timestamp = '{:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now())
    filename = "chip-" + str(chipno) + "-" + timestamp + ".fits"
    newdetector.read_out(filename=filename)

    # Return file name
    return filename
