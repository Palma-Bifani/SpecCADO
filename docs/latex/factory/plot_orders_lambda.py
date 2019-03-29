import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import speccado as sc
import simcado as sim

def main(datadir):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    # plot transmissivities
    configfiles = ["spectro_IJ.config", "spectro_J.config",
                   "spectro_HK.config"]
    for cfile in configfiles:
        cmds = sim.UserCommands(cfile, datadir)
        opttrain = sim.OpticalTrain(cmds)
        tc_lam = opttrain.tc_mirror.lam_orig
        tc_val = opttrain.tc_mirror.val_orig
        plt.plot(tc_lam, tc_val * 9 *18/4, alpha=0.5)

    # plot orders
    orderfile = "specorders-180629.fits"
    tracelist = sc.layout.read_spec_order(orderfile)
    #fpa = sim.Detector(cmds, small_fov=False)
    #cmds = sim.UserCommands(configfile, datadir)

    tracelist.reverse()
    minlamarr = np.zeros(len(tracelist))
    maxlamarr = np.zeros(len(tracelist))
    for i, trace in enumerate(tracelist):
        minlamarr[i], maxlamarr[i], _ = trace.analyse_lambda()

    #minlamarr = minlamarr[::-1]
    #maxlamarr = maxlamarr[::-1]
    lam_min = np.min(minlamarr)
    lam_max = np.max(maxlamarr)
    y=0.5
    hw = 0.2
    for l1, l2, trace in zip(minlamarr, maxlamarr, tracelist):
        xy = np.array([[l1, y-hw],
                       [l1, y+hw],
                       [l2, y+hw],
                       [l2, y-hw]])
        print(l1, l2)
        ax.add_patch(Polygon(xy, closed=True, fill=True, color='k'))
        ax.text(l2+0.03, y, trace.name.split('.')[0],
                fontsize='x-small', va='center'
        )
        y = y + 1

    ax.set_xlim(0.5, 2.85)
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylim(0, 2 + len(tracelist))
    ax.get_yaxis().set_visible(False)

    #    colorlist = ['r', 'b', 'g', 'c', 'm', 'k', 'orange', 'chartreuse',
    #                 'k', 'k', 'k', 'k']
    #    colorlist.reverse()
    plt.savefig("spectral_orders_v_wavelength.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
