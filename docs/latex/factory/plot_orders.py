import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import speccado as sc
import simcado as sim

def main(configfile, datadir):
    if configfile == "spectro_HK.config":
        filters = "HK"
    elif configfile == "spectro_J.config":
        filters = "J"
    elif configfile == "spectro_IJ.config":
        filters = "IzJ"
    else:
        filters = ""
    cmds = sim.UserCommands(configfile, datadir)
    tracelist = sc.layout.read_spec_order(cmds['SPEC_ORDER_LAYOUT'])
    fpa = sim.Detector(cmds, small_fov=False)
    sim.detector.plot_detector_layout(fpa, plane='fpa', label=False,
                                      color='k', linestyle='-', linewidth=1)

    colorlist = ['r', 'b', 'g', 'c', 'm', 'k', 'orange', 'chartreuse',
                 'k', 'k', 'k', 'k']
    colorlist.reverse()
    ax = plt.gca()
    thecol = colorlist.pop()
    for trace in tracelist:
        xy = trace.write_reg("/dev/null", slit_length=cmds['SPEC_SLIT_LENGTH'],
                             waveband=cmds['INST_FILTER_TC'])
        if xy is not None:
            print("Hallo: ", trace.name, thecol)
            ax.add_patch(Polygon(xy, closed=True, fill=False, color=thecol,
                                 label=trace.name.split('.')[0]))
            thecol = colorlist.pop()
    plt.text(90, 90, filters, ha='right', va='top', fontsize=14,
             style='italic')
    #    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0., fontsize='small')
    outfile = configfile.split('.')[0] + "_layout.pdf"
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
