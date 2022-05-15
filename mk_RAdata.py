###########################################################
#                 **make RainArea data**                  #
# coded by Keisuke HOSOTANI                               #
# last modified : 2022/05/09                              #
# Note: Be sure that labeling data have been ready.       #
#       labeling data can be made by labeling.py .        #
# Usage: make RainArea data from KuR V07A                 #
#        following the labeling data.                     #
###########################################################

import numpy as np
import pandas as pd

import sys

sys.path.append("/home/hosotani/work/")
from lab_dataread import gpm

nscn = 8100  # number of scans parallel to the orbit
nang = 49  # number of scans perpendicular to the orbit
ncell1 = 176  # number of vertical grids
N = nscn * nang  # number of data per orbit


##########################################################
########################[[MAIN]]##########################
##########################################################

start = 144
end = 46390
for scan in range(start, end):

    ######################################################
    ###################[[READ RAINDATA]]##################
    ######################################################

    try:
        data = gpm.KuR(scan)
        nsp = data.value("nsp").reshape([nscn, nang])
        nsz = data.value("nsz").reshape([nscn, nang])
        flagPrecip = data.value("flagPrecip").reshape([nscn, nang])
        ptyp = data.value("ptyp").reshape([nscn, nang])
        styp = data.value("styp").reshape([nscn, nang])
        ze = data.value("ze").reshape([ncell1, nscn, nang])
        lon = data.value("lon").reshape([nscn, nang])
        lat = data.value("lat").reshape([nscn, nang])
    except AssertionError:
        print("input KuR file may be broken: " + str(scan))
        continue
    except FileNotFoundError:
        print("KuR file not found: " + str(scan))
        continue
    except PermissionError:
        print("KuR file permission denied: " + str(scan))
        continue

    ######################################################
    #################[[READ TIME DATA]]###################
    ######################################################

    try:
        data = gpm.KuT(scan)
        year = data.value("year")
        month = data.value("month")
        day = data.value("day")
        hour = data.value("hour")
        minu = data.value("minu")
        sec = data.value("sec")
        msec = data.value("msec")
    except AssertionError:
        print("input Time file may be broken: " + str(scan))
        continue
    except FileNotFoundError:
        print("Time file not found: " + str(scan))
        continue
    except PermissionError:
        print("Time file permission denied: " + str(scan))
        continue

    ######################################################
    ####################[[READ LABELS]]###################
    ######################################################

    # RA label
    ifile = "../labeling_data/RainAreas/RAlabel.{:0=6}".format(scan) + ".npy"
    label_RA = np.load(ifile)

    # 40dBZ label
    ifile = "../labeling_data/40dBZ_Areas/40label.{:0=6}".format(scan) + ".npy"
    label_40 = np.load(ifile)

    # 30dBZ label
    ifile = "../labeling_data/30dBZ_Areas/30label.{:0=6}".format(scan) + ".npy"
    label_30 = np.load(ifile)

    nlab = np.max(label_RA)
    nlab_40 = np.max(label_40)
    nlab_30 = np.max(label_30)

    # assign strong to normal
    RA_40 = np.zeros(
        nlab_40 + 1, dtype="i2"
    )  # list which returns RAlabel according to given 40label
    lab_40 = 1
    for i in range(nscn):
        for j in range(nang):
            if label_40[i, j] == lab_40:
                RA_40[lab_40] = label_RA[i, j]
                lab_40 += 1

    contain_40dBZ = np.zeros(nlab)
    for lab_40 in range(1, len(RA_40)):
        contain_40dBZ[RA_40[lab_40] - 1] += 1

    # assign moderate to normal
    RA_30 = np.zeros(
        nlab_30 + 1, dtype="i2"
    )  # list which returns RAlabel according to given 30label
    lab_30 = 1
    for i in range(nscn):
        for j in range(nang):
            if label_30[i, j] == lab_30:
                RA_30[lab_30] = label_RA[i, j]
                lab_30 += 1

    contain_30dBZ = np.zeros(nlab)
    for lab_30 in range(1, len(RA_30)):
        contain_30dBZ[RA_30[lab_30] - 1] += 1

    ######################################################
    ##################[[40dBZ PROPERTY]]##################
    ######################################################

    # initialize
    pix = np.zeros(nlab_40 + 1, dtype="i2")
    RA = np.zeros(nlab_40 + 1, dtype="i2")

    for lab_40 in range(1, nlab_40 + 1):
        pix[lab_40] = np.count_nonzero(label_40 == lab_40)
        RA[lab_40] = RA_40[lab_40]

    d_40 = pd.DataFrame(
        {
            "pix_40dBZ": pix,
            "RAlabel": RA,
        }
    )

    ######################################################
    ##################[[30dBZ PROPERTY]]##################
    ######################################################

    # initialize
    pix = np.zeros(nlab_30 + 1, dtype="i2")
    RA = np.zeros(nlab_30 + 1, dtype="i2")

    for lab_30 in range(1, nlab_30 + 1):
        pix[lab_30] = np.count_nonzero(label_30 == lab_30)
        RA[lab_30] = RA_30[lab_30]

    d_30 = pd.DataFrame(
        {
            "pix_30dBZ": pix,
            "RAlabel": RA,
        }
    )

    ######################################################
    ################[[RAINAREA PROPERTY]]#################
    ######################################################

    # initialize
    year_ra = np.zeros(nlab)
    month_ra = np.zeros(nlab)
    day_ra = np.zeros(nlab)
    hour_ra = np.zeros(nlab)
    minu_ra = np.zeros(nlab)
    sec_ra = np.zeros(nlab)
    msec_ra = np.zeros(nlab)
    nsp_max_lat = np.zeros(nlab)
    nsp_max_lon = np.zeros(nlab)
    pixel = np.zeros(nlab)
    nsp_area = np.zeros(nlab)
    nsp_max = np.zeros(nlab)
    nsp_99pt = np.zeros(nlab)
    nsp_95pt = np.zeros(nlab)
    nsp_90pt = np.zeros(nlab)
    pix_stra = np.zeros(nlab)
    nsp_stra = np.zeros(nlab)
    nsp_stra_max = np.zeros(nlab)
    pix_conv = np.zeros(nlab)
    nsp_conv = np.zeros(nlab)
    nsp_conv_max = np.zeros(nlab)
    pix_other = np.zeros(nlab)
    nsp_other = np.zeros(nlab)
    nsp_other_max = np.zeros(nlab)
    top_40dBZ = np.zeros(nlab)
    top_30dBZ = np.zeros(nlab)
    pix_40dBZ = np.zeros(nlab)
    pix_30dBZ = np.zeros(nlab)
    ocean = np.zeros(nlab)
    land = np.zeros(nlab)
    coast = np.zeros(nlab)
    other_sfc = np.zeros(nlab)
    undef_sfc = np.zeros(nlab)

    nsp_values = [[] for i in range(nlab)]

    # main loop
    for i in range(nscn):
        for j in range(nang):
            lab = label_RA[i, j]

            if lab == 0:
                continue

            lab = lab - 1
            # pointing index("lab") statrs from 0, while it was counted from 1 when labeling

            # total pixel
            pixel[lab] += 1

            # total precipitation
            nsp_area[lab] += nsp[i, j]
            # note: nsp can't be undef, as long as the pixel is labelled

            nsp_values[lab].append(nsp[i, j])

            if nsp[i, j] > nsp_max[lab]:
                nsp_max[lab] = nsp[i, j]  # max precipitaton
                # position and time are represented by those of the pixel with max precipitation
                nsp_max_lat[lab] = lat[i, j]  # lattitude
                nsp_max_lon[lab] = lon[i, j]  # longtitude
                # time
                year_ra[lab] = year[i]
                month_ra[lab] = month[i]
                day_ra[lab] = day[i]
                hour_ra[lab] = hour[i]
                minu_ra[lab] = minu[i]
                sec_ra[lab] = sec[i]
                msec_ra[lab] = msec[i]

            if (ptyp[i, j] >= 1) and (ptyp[i, j] < 2):
                pix_stra[lab] += 1
                nsp_stra[lab] += nsp[i, j]
                if nsp[i, j] > nsp_stra_max[lab]:
                    nsp_stra_max[lab] = nsp[i, j]

            elif (ptyp[i, j] >= 2) and (ptyp[i, j] < 3):
                pix_conv[lab] += 1
                nsp_conv[lab] += nsp[i, j]
                if nsp[i, j] > nsp_conv_max[lab]:
                    nsp_conv_max[lab] = nsp[i, j]
            else:
                pix_other[lab] += 1
                nsp_other[lab] += nsp[i, j]
                if nsp[i, j] > nsp_other_max[lab]:
                    nsp_other_max[lab] = nsp[i, j]

            for z in range(ncell1):
                if ze[z, i, j] > 40 and z > top_40dBZ[lab]:
                    top_40dBZ[lab] = z
                if ze[z, i, j] > 30 and z > top_30dBZ[lab]:
                    top_30dBZ[lab] = z

            if styp[i, j] >= 0 and styp[i, j] < 100:
                ocean[lab] += 1
            elif styp[i, j] >= 100 and styp[i, j] < 200:
                land[lab] += 1
            elif styp[i, j] >= 300 and styp[i, j] < 400:
                coast[lab] += 1
            elif styp[i, j] >= 400:
                other_sfc[lab] += 1
            else:
                undef_sfc[lab] += 1

    for lab in range(nlab):
        nsp_99pt[lab] = np.percentile(np.array(nsp_values[lab]), 99)
        nsp_95pt[lab] = np.percentile(np.array(nsp_values[lab]), 95)
        nsp_90pt[lab] = np.percentile(np.array(nsp_values[lab]), 90)

    tmp = ocean + land + coast + other_sfc + undef_sfc
    ocean = ocean / tmp
    land = land / tmp
    coast = coast / tmp
    other_sfc = other_sfc / tmp

    for lab in d_40[(d_40["RAlabel"] > 0)]["RAlabel"].unique():
        pix_40dBZ[lab - 1] = d_40[(d_40["RAlabel"] == lab)][
            "pix_40dBZ"
        ].max()  # RAlabel starts from 1, but "lab" in this database strats from 0
        pix_40dBZ = pix_40dBZ.astype(np.int16)

    for lab in d_30[(d_30["RAlabel"] > 0)]["RAlabel"].unique():
        pix_30dBZ[lab - 1] = d_30[(d_30["RAlabel"] == lab)][
            "pix_30dBZ"
        ].max()  # RAlabel starts from 1, but "lab" in this database strats from 0
        pix_30dBZ = pix_30dBZ.astype(np.int16)

    ########################################
    ########################################
    ########################################

    ######################################################
    #####################[[OUTPUT]]#######################
    ######################################################

    dty = np.dtype(
        [
            ("nlab", "<i2"),
            ("year", "<" + str(nlab) + "i2"),
            ("month", "<" + str(nlab) + "i2"),
            ("day", "<" + str(nlab) + "i2"),
            ("hour", "<" + str(nlab) + "i2"),
            ("minu", "<" + str(nlab) + "i2"),
            ("sec", "<" + str(nlab) + "i2"),
            ("msec", "<" + str(nlab) + "i2"),
            ("lat", "<" + str(nlab) + "f2"),
            ("lon", "<" + str(nlab) + "f2"),
            ("pixel", "<" + str(nlab) + "i2"),
            ("nsp_area", "<" + str(nlab) + "f2"),
            ("nsp_max", "<" + str(nlab) + "f2"),
            ("nsp_99pt", "<" + str(nlab) + "f2"),
            ("nsp_95pt", "<" + str(nlab) + "f2"),
            ("nsp_90pt", "<" + str(nlab) + "f2"),
            ("pix_stra", "<" + str(nlab) + "i2"),
            ("nsp_stra", "<" + str(nlab) + "f2"),
            ("nsp_stra_max", "<" + str(nlab) + "f2"),
            ("pix_conv", "<" + str(nlab) + "i2"),
            ("nsp_conv", "<" + str(nlab) + "f2"),
            ("nsp_conv_max", "<" + str(nlab) + "f2"),
            ("pix_other", "<" + str(nlab) + "i2"),
            ("nsp_other", "<" + str(nlab) + "f2"),
            ("nsp_other_max", "<" + str(nlab) + "f2"),
            ("top_40dBZ", "<" + str(nlab) + "i2"),
            ("top_30dBZ", "<" + str(nlab) + "i2"),
            ("pix_40dBZ", "<" + str(nlab) + "i2"),
            ("pix_30dBZ", "<" + str(nlab) + "i2"),
            ("contain_40dBZ", "<" + str(nlab) + "i2"),
            ("contain_30dBZ", "<" + str(nlab) + "i2"),
            ("ocean", "<" + str(nlab) + "f2"),
            ("land", "<" + str(nlab) + "f2"),
            ("coast", "<" + str(nlab) + "f2"),
            ("other_sfc", "<" + str(nlab) + "f2"),
        ]
    )

    output = np.zeros(1, dtype=dty)
    output[0]["nlab"] = nlab
    output[0]["year"] = year_ra
    output[0]["month"] = month_ra
    output[0]["day"] = day_ra
    output[0]["hour"] = hour_ra
    output[0]["minu"] = minu_ra
    output[0]["sec"] = sec_ra
    output[0]["msec"] = msec_ra
    output[0]["lat"] = nsp_max_lat
    output[0]["lon"] = nsp_max_lon
    output[0]["pixel"] = pixel
    output[0]["nsp_area"] = nsp_area
    output[0]["nsp_max"] = nsp_max
    output[0]["nsp_99pt"] = nsp_90pt
    output[0]["nsp_95pt"] = nsp_90pt
    output[0]["nsp_90pt"] = nsp_90pt
    output[0]["pix_stra"] = pix_stra
    output[0]["nsp_stra"] = nsp_stra
    output[0]["nsp_stra_max"] = nsp_stra_max
    output[0]["pix_conv"] = pix_conv
    output[0]["nsp_conv"] = nsp_conv
    output[0]["nsp_conv_max"] = nsp_conv_max
    output[0]["pix_other"] = pix_other
    output[0]["nsp_other"] = nsp_other
    output[0]["nsp_other_max"] = nsp_other_max
    output[0]["top_40dBZ"] = top_40dBZ
    output[0]["top_30dBZ"] = top_30dBZ
    output[0]["pix_40dBZ"] = pix_40dBZ
    output[0]["pix_30dBZ"] = pix_30dBZ
    output[0]["contain_40dBZ"] = contain_40dBZ
    output[0]["contain_30dBZ"] = contain_30dBZ
    output[0]["ocean"] = ocean
    output[0]["land"] = land
    output[0]["coast"] = coast
    output[0]["other_sfc"] = other_sfc

    f = open(f"../RAdata/RAdata.{scan:06}.bin", "wb")
    f.write(output)
    f.close()
