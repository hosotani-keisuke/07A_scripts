##################################################################
#                 **RainArea Labeling**                          #
# coded by Keisuke HOSOTANI                                      #
# last modified : 2022/05/08                                     #
# Usage: detect Rain Areas from KuPR V07A data and label them.   #
##################################################################

import numpy as np
import sys

sys.path.append("/home/hosotani/work/")
from lab_dataread import gpm

sys.setrecursionlimit(500000)

nscn = 8100  # number of scan parallel to the orbit
nang = 49  # number of scan perpendicular to the orbit
ncell1 = 176  # number of vertical grid
N = nscn * nang  # number of data per one record


# rainarea detection
def detect_pr(nscn, nang, nsp, flagPrecip, lb):
    for i in range(nscn):
        for j in range(nang):
            if (nsp[i, j] >= 0.5) and (flagPrecip[i, j] > 0):
                lb[i, j] = -1
            else:
                pass


# strong rainarea detection
def detect_pr_s(nscn, nang, nsz, flagPrecip, lb2):
    for i in range(nscn):
        for j in range(nang):
            if (nsz[i, j] >= 40) and (flagPrecip[i, j] > 0):
                lb2[i, j] = -1
            else:
                pass


# moderate rainarea detection
def detect_pr_m(nscn, nang, nsz, flagPrecip, lb2):
    for i in range(nscn):
        for j in range(nang):
            if (nsz[i, j] >= 30) and (flagPrecip[i, j] > 0):
                lb2[i, j] = -1
            else:
                pass


# rainarea labeling
def labeling(i, j, lb, nlab):
    lb[i, j] = nlab
    if j == 0:

        if lb[i - 1, j] == -1:
            labeling(i - 1, j, lb, nlab)

        if lb[i + 1, j] == -1:
            labeling(i + 1, j, lb, nlab)

        if lb[i - 1, j + 1] == -1:
            labeling(i - 1, j + 1, lb, nlab)

        if lb[i, j + 1] == -1:
            labeling(i, j + 1, lb, nlab)

        if lb[i + 1, j + 1] == -1:
            labeling(i + 1, j + 1, lb, nlab)
    elif j == 48:

        if lb[i - 1, j - 1] == -1:
            labeling(i - 1, j - 1, lb, nlab)

        if lb[i, j - 1] == -1:
            labeling(i, j - 1, lb, nlab)

        if lb[i + 1, j - 1] == -1:
            labeling(i + 1, j - 1, lb, nlab)

        if lb[i - 1, j] == -1:
            labeling(i - 1, j, lb, nlab)

        if lb[i + 1, j] == -1:
            labeling(i + 1, j, lb, nlab)
    else:

        if lb[i - 1, j - 1] == -1:
            labeling(i - 1, j - 1, lb, nlab)

        if lb[i, j - 1] == -1:
            labeling(i, j - 1, lb, nlab)

        if lb[i + 1, j - 1] == -1:
            labeling(i + 1, j - 1, lb, nlab)

        if lb[i - 1, j] == -1:
            labeling(i - 1, j, lb, nlab)

        if lb[i + 1, j] == -1:
            labeling(i + 1, j, lb, nlab)

        if lb[i - 1, j + 1] == -1:
            labeling(i - 1, j + 1, lb, nlab)

        if lb[i, j + 1] == -1:
            labeling(i, j + 1, lb, nlab)

        if lb[i + 1, j + 1] == -1:
            labeling(i + 1, j + 1, lb, nlab)


#################################
#################################
#################################

start = 144
end = 46390
for scan in range(start, end):

    ######################################################
    ##################[[READ RAINDATA]]###################
    ######################################################

    try:
        Rain_data = gpm.KuR(scan)
        nsp = Rain_data.value("nsp").reshape([nscn, nang])
        nsz = Rain_data.value("nsz").reshape([nscn, nang])
        flagPrecip = Rain_data.value("flagPrecip").reshape([nscn, nang])
    except AssertionError:
        print("input file may be broken: " + str(scan))
        continue

    ######################################################
    ################[[RAINAREA DETECTION]]################
    ######################################################

    # rainarea detection and labaling
    lb = np.zeros([nscn, nang], dtype=int)
    detect_pr(nscn, nang, nsp, flagPrecip, lb)
    nlab = 0
    for i in range(nscn):
        for j in range(nang):
            if lb[i, j] == -1:
                nlab = nlab + 1
                labeling(i, j, lb, nlab)

    # strong rainarea detection and labeling
    lb_s = np.zeros([nscn, nang], dtype=int)
    detect_pr_s(nscn, nang, nsz, flagPrecip, lb_s)
    nlab_s = 0
    for i in range(nscn):
        for j in range(nang):
            if lb_s[i, j] == -1:
                nlab_s = nlab_s + 1
                labeling(i, j, lb_s, nlab_s)

    # moderate rainarea detection and labeling
    lb_m = np.zeros([nscn, nang], dtype=int)
    detect_pr_m(nscn, nang, nsz, flagPrecip, lb_m)
    nlab_m = 0
    for i in range(nscn):
        for j in range(nang):
            if lb_m[i, j] == -1:
                nlab_m = nlab_m + 1
                labeling(i, j, lb_m, nlab_m)

    ######################################################
    ######################[[OUTPUT]]######################
    ######################################################

    np.save(
        f"/home/hosotani/work/07A_analysis/labeling_data/RainAreas/RAlabel.{scan:06}.npy",
        lb,
    )
    np.save(
        f"/home/hosotani/work/07A_analysis/labeling_data/40dBZ_Areas/40label.{scan:06}.npy",
        lb_s,
    )
    np.save(
        f"/home/hosotani/work/07A_analysis/labeling_data/30dBZ_Areas/30label.{scan:06}.npy",
        lb_m,
    )
