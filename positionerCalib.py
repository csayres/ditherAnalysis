from coordio.defaults import calibration
from coordio.utils import fitsTableToPandas
from astropy.io import fits
from astropy.table import Table
from coordio.transforms import FVCTransformLCO, FVCTransformAPO
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist
import numpy
from skimage.transform import EuclideanTransform
import pandas
from coordio.transforms import positionerToWok
import seaborn as sns
from scipy.optimize import minimize
import time
from coordio import defaults
import os
import glob
from multiprocessing import Pool
from astropy.table import Table
from functools import partial

# fcCols = ["site", "holeID", "id", "xWok", "yWok", "zWok", "col", "row"]
# ptCols = ["site", "holeID", "positionerID", "wokID", "alphaArmLen", "metX", "metY", "apX",
#            "apY", "bossX", "bossY", "alphaOffset", "betaOffset", "dx", "dy"]

pt = calibration.positionerTable.reset_index()
wc = calibration.wokCoords.reset_index()
fc = calibration.fiducialCoords.reset_index()

jt = pt.merge(wc, on="holeID")

# mjd = 59785
# imgStart = 4
# imgEnd = 62
centType="flex"
site = "APO"
# baseDir = "/Volumes/futa/apo/data/fcam/%i"%mjd

# 54 first robot to get to run by disabling, maybe its flux?

# disabledRobots = [608, 612, 1136, 182, 54, 1300, 565, 719]
disabledRobot = None

# apo old and new darks
olddark = numpy.array(fits.open("olddark.fits")[1].data, dtype=float)[:,::-1]
newdark = numpy.array(fits.open("newdark.fits")[1].data, dtype=float)[:,::-1]


def processImg(imgPath, name, pt, wc, fc):
    # print("imgPath", imgPath)
    imgNum = int(imgPath.strip(".fits").split("-")[-1])
    mjd = int(imgPath.split("/")[-2])
    ff = fits.open(imgPath)

    imgData = ff[1].data
    imgData = imgData + olddark - newdark
    posAngles = Table(ff["POSANGLES"].data).to_pandas()
    IPA = ff[1].header["IPA"]

    if site == "APO":
        fvct = FVCTransformAPO(
            imgData,
            posAngles,
            IPA,
            plotPathPrefix="pdf/%s.mjd%i.imgNum%s"%(name, mjd, ("%i"%imgNum).zfill(4)),
            positionerTable=pt,
            wokCoords=wc,
            fiducialCoords=fc
        )
    else:
        fvct = FVCTransformLCO(
            imgData,
            posAngles,
            IPA,
            plotPathPrefix="pdf/%s.mjd%i.imgNum%s"%(name, mjd, ("%i"%imgNum).zfill(4)),
            positionerTable=pt,
            wokCoords=wc,
            fiducialCoords=fc
        )

    fvct.extractCentroids()
    # increase max final distance to get matches easier
    # robots are a bit off (mostly trans/rot/scale)
    fvct.fit(centType=centType)
    ptm = fvct.positionerTableMeas.copy()
    ptm["fvcImgNum"] = imgNum
    ptm["mjd"] = mjd

    ff.close()

    return ptm



def processImgs(
        name, imgPaths, pt=pt, wc=wc, fc=fc
    ):
    """
    inputs
    --------
    name : for saving files and plots
    pt : positioner table
    wc : wok coordinates
    fc : fiducial coordinates
    mjd : mjd
    imgStart : first image number
    imgEnd : last image number
    maxFinalDist : max euclidean distance for centroid matching
    """

    p = Pool(10)
    _pimg = partial(processImg, name=name, pt=pt, wc=wc, fc=fc)
    pDF = p.map(_pimg, imgPaths)

    # pDF = []
    # for imgPath in imgPaths:
    #     pDF.append(processImg(imgPath, name=name, pt=pt, wc=wc, fc=fc))
    #     print("finished img", imgPath)


    return pandas.concat(pDF)
    # pDF = pandas.concat(pDF)
    # fDF = pandas.concat(fDF)

    # pDF.to_csv("pt_%s.csv"%name)
    # fDF.to_csv("fc_%s.csv"%name)


def forwardModel(x, positionerID, alphaDeg, betaDeg):
    xBeta, la, alphaOffDeg, betaOffDeg, dx, dy = x
    yBeta = 0 # by definition for metrology fiber

    _jt = jt[jt.positionerID==positionerID]
    b = numpy.array([_jt[x] for x in ["xWok", "yWok", "zWok"]]).flatten()
    iHat = numpy.array([_jt[x] for x in ["ix", "iy", "iz"]]).flatten()
    jHat = numpy.array([_jt[x] for x in ["jx", "jy", "jz"]]).flatten()
    kHat = numpy.array([_jt[x] for x in ["kx", "ky", "kz"]]).flatten()

    xw, yw, zw = positionerToWok(
        alphaDeg, betaDeg,
        xBeta, yBeta, la,
        alphaOffDeg, betaOffDeg,
        dx, dy, b, iHat, jHat, kHat
    )

    return xw, yw


def minimizeMe(x, positionerID, alphaDeg, betaDeg, xWok, yWok):
    xw, yw = forwardModel(x, positionerID, alphaDeg, betaDeg)
    return numpy.sum((xw-xWok)**2 + (yw-yWok)**2)


def fitCalibs(positionerTableIn, positionerTableMeas, disabledRobots=None, keepF2F=True):
    x0 = numpy.array([
        defaults.MET_BETA_XY[0], defaults.ALPHA_LEN,
        0, 0, 0, 0
    ])
    positionerIDs = positionerTableIn.positionerID.to_numpy()
    _xBeta = []
    _la = []
    _alphaOffDeg = []
    _betaOffDeg = []
    _dx = []
    _dy = []
    for positionerID in positionerIDs:
        if bool(disabledRobots) and positionerID in disabledRobots:
            print("hacking disabled positioner %i, keep old values"%positionerID)
            _row = positionerTableIn[positionerTableIn.positionerID==positionerID]
            _xBeta.append(float(_row["metX"]))
            _la.append(float(_row["alphaArmLen"]))
            _alphaOffDeg.append(float(_row["alphaOffset"]))
            _betaOffDeg.append(float(_row["betaOffset"]))
            _dx.append(float(_row["dx"]))
            _dy.append(float(_row["dy"]))
            continue

        print("calibrating positioner", positionerID)
        _df = positionerTableMeas[positionerTableMeas.positionerID==positionerID]
        args = (
            positionerID,
            _df.alphaReport.to_numpy(),
            _df.betaReport.to_numpy(),
            _df.xWokMeasMetrology.to_numpy(),
            _df.yWokMeasMetrology.to_numpy()
        )
        tstart = time.time()
        out = minimize(minimizeMe, x0, args, method="Powell")

        xBeta, la, alphaOffDeg, betaOffDeg, dx, dy = out.x
        _xBeta.append(xBeta)
        _la.append(la)
        _alphaOffDeg.append(alphaOffDeg)
        _betaOffDeg.append(betaOffDeg)
        _dx.append(dx)
        _dy.append(dy)
        tend = time.time()
        print("took %.2f"%(tend-tstart))

    _xBeta = numpy.array(_xBeta)
    _la = numpy.array(_la)
    _alphaOffDeg = numpy.array(_alphaOffDeg)
    _betaOffDeg = numpy.array(_betaOffDeg)
    _dx = numpy.array(_dx)
    _dy = numpy.array(_dy)

    positionerTableOut = positionerTableIn.copy()
    dxBoss = positionerTableOut.bossX.to_numpy() - positionerTableOut.metX.to_numpy()
    dxAp = positionerTableOut.apX.to_numpy() - positionerTableOut.apX.to_numpy()
    positionerTableOut["metX"] = _xBeta
    positionerTableOut["alphaArmLen"] = _la
    positionerTableOut["alphaOffset"] = _alphaOffDeg
    positionerTableOut["betaOffset"] = _betaOffDeg
    positionerTableOut["dx"] = _dx
    positionerTableOut["dy"] = _dy
    if keepF2F:
        positionerTableOut["bossX"] = positionerTableOut.metX + dxBoss
        positionerTableOut["apX"] = positionerTableOut.metX + dxAp



    return positionerTableOut

def plotDistances(pt_meas, title=""):
    pt_meas = pt_meas.copy()
    positionerID = pt_meas.positionerID.to_numpy()
    xReport = pt_meas.xWokReportMetrology
    yReport = pt_meas.yWokReportMetrology
    xMeas = pt_meas.xWokMeasMetrology
    yMeas = pt_meas.yWokMeasMetrology
    dx = xReport - xMeas
    dy = yReport - yMeas
    dr = numpy.sqrt(dx**2+dy**2)
    # cut on big errors (> 0.5 mm)
    # keep = dr < 0.5

    # xReport = xReport[keep]
    # yReport = yReport[keep]
    # xMeas = xMeas[keep]
    # yMeas = yMeas[keep]
    # dx = dx[keep]
    # dy = dy[keep]
    # dr = dr[keep]
    # positionerID = pt_meas.positionerID.to_numpy()[keep]


    rms = numpy.sqrt(numpy.mean(dr**2))*1000
    med = numpy.percentile(dr, 50)*1000
    p75 = numpy.percentile(dr, 75)*1000
    p90 = numpy.percentile(dr, 90)*1000
    rmsStr = " RMS: %.1f p50: %.1f  p75: %.1f  p90: %.1f  (um)"%(rms,med,p75,p90)
    title = title + rmsStr

    plt.figure(figsize=(13,8))
    sns.boxplot(x=positionerID, y=dr)
    plt.title(title)

    plt.figure(figsize=(8,8))
    plt.quiver(xReport, yReport, dx, dy ,angles="xy", units="xy", scale=.01, width=1)
    plt.title(title)

    bins = numpy.linspace(0,0.2,200)
    plt.figure()
    plt.hist(dr, bins=bins, cumulative=True, density=True, histtype="step")
    plt.title(title)


def getPTM(imgs):
    dfList = []
    for img in imgs:
        ff = fits.open(img)
        ptm = Table(ff["POSITIONERTABLEMEAS"].data).to_pandas()
        toks = img.strip(".fits").split("-")
        imgNum = int(toks[-1])
        mjd = int(img.split("/")[-2])
        ptm["fvcImgNum"] = imgNum
        ptm["mjd"] = mjd
        dfList.append(ptm)

    return pandas.concat(dfList)


def massage(df):
    df["xWokBlindErr"] = df.xWokReportMetrology - df.xWokMeasMetrology
    df["yWokBlindErr"] = df.yWokReportMetrology - df.yWokMeasMetrology
    df["rWokBlindErr"] = numpy.sqrt(df.xWokBlindErr**2+df.yWokBlindErr**2)
    # broken met fibers have > 5mm error on average
    df_m = df[["positionerID", "rWokBlindErr"]].groupby("positionerID").median().reset_index()
    # brokenMets = list(set(df_m[df_m.rWokBlindErr > 5]["positionerID"])) hardcode instead
    brokenMets = [802, 460, 846, 1049, 639] #985,

    # drop brokenMets
    print(brokenMets)
    df = df[~df.positionerID.isin(brokenMets)]
    fullDisable = [201, 478, 503, 535, 1231, 716, 639, 802, 985, 846, 460, 995, 1049, 1246, 1290, 1026, 1202]

    dfList = []
    imgs1231 = numpy.arange(1,28)
    # handle special robots that were disabled at different times throughout the scan
    for name, group in df.groupby("positionerID"):
        if name == 1231:
            group = group[group.mjd==60658]
            group = group[group.fvcImgNum.isin(imgs1231)]
            dfList.append(group)
        elif name == 985:
            group = group[group.mjd==60665]
            dfList.append(group)
        elif name in fullDisable:
            # only use data from first two MJDS
            for n2, g2 in group.groupby("mjd"):
                if n2 == 60663:
                    continue
                if n2 == 60658:
                    g2 = g2[~g2.fvcImgNum.isin(imgs1231)]

                dfList.append(g2)
        else:
            # use images from all mjds
            for n2, g2 in group.groupby("mjd"):
                if n2 == 60658:
                    g2 = g2[~g2.fvcImgNum.isin(imgs1231)]
                # if n2 == 60663:
                #     g2 = g2[g2.fvcImgNum < 35]

                dfList.append(g2)


    df = pandas.concat(dfList)
    return df, brokenMets

if __name__ == "__main__":

    # quick look:
    # pt = pandas.read_csv("ptorig.csv")
    # ptOut = pandas.read_csv("ptnew.csv")
    # # plot difference in xy robot positions
    # ptm = pt.merge(ptOut, on=["positionerID", "holeID"], suffixes=("_o", "_n"))
    # ptm = ptm.merge(wc, on="holeID")
    # ptm["dx"] = ptm.dx_n - ptm.dx_o
    # ptm["dy"] = ptm.dy_n - ptm.dy_o
    # ptm["dr"] = numpy.sqrt(ptm.dx**2+ptm.dy**2)
    # plt.figure()
    # plt.hist(ptm.dr)

    # brokenRobots = [1026, 1290, 523, 639, 275, 535, 1049, 1051, 671, 802, 1187, 935, 1202, 201, 460, 716, 846, 1231, 985, 478, 1246, 995, 997, 503, 1023, 984, 1043]

    # plt.figure(figsize=(8,8))
    # plt.quiver(ptm.xWok, ptm.yWok, ptm.dx, ptm.dy, angles="xy", units="xy", scale=0.001)
    # plt.axis("equal")

    # ptm = ptm[ptm.positionerID.isin(brokenRobots)]
    # plt.quiver(ptm.xWok, ptm.yWok, ptm.dx, ptm.dy, color="red", angles="xy", units="xy", scale=0.001)

    # plt.show()
    # import pdb; pdb.set_trace()

    # pt1 = pandas.read_csv("ptorig.csv")
    # pt2 = pandas.read_csv("ptnew.csv")

    # pt = pt1.merge(pt2, on="positionerID", suffixes=("_old", "_new"))

    # plt.figure()
    # dx = pt.dx_old - pt.dx_new
    # dy = pt.dy_old - pt.dy_new
    # dr = numpy.sqrt(dx**2+dy**2)

    # dalpha = pt.alphaOffset_old - pt.alphaOffset_new
    # dbeta = pt.betaOffset_old - pt.betaOffset_new
    # plt.hist(dalpha)

    # plt.figure()
    # plt.hist(dbeta)
    # plt.show()
    # import pdb; pdb.set_trace()

    ############# apo #############################
    ###############################################
    # turned on some broken robots and tried to recalibrate


    # imgs = glob.glob("/Users/csayres/Downloads/fcam/60661/proc*.fits")
    # imgs.extend(glob.glob("/Users/csayres/Downloads/fcam/60658/proc*.fits"))
    # # remove img 10 mjd 60661 (solve broke?)
    # imgs.remove("/Users/csayres/Downloads/fcam/60661/proc-fimg-fvc1n-0010.fits")
    # imgs60663 = glob.glob("/Users/csayres/Downloads/fcam/60663/proc*.fits")
    # imgs60663.sort()
    # keep = []
    # lastConfig = None
    # for img in imgs60663:
    #     imgNum = int(img.strip(".fits").split("-")[-1])
    #     if imgNum < 35:
    #         keep.append(img)
    #         continue
    #     ff = fits.open(img)
    #     thisConfig = ff[1].header["CONFIGID"]
    #     if thisConfig == lastConfig:
    #         continue
    #     lastConfig = thisConfig
    #     keep.append(img)
    # imgs.extend(keep)


    # imgs60664 = glob.glob("/Users/csayres/Downloads/fcam/60664/proc*.fits")
    # imgs60664.sort()
    # keep = []
    # lastConfig = None
    # for img in imgs60663:
    #     imgNum = int(img.strip(".fits").split("-")[-1])
    #     ff = fits.open(img)
    #     thisConfig = ff[1].header["CONFIGID"]
    #     if thisConfig == lastConfig:
    #         continue
    #     lastConfig = thisConfig
    #     keep.append(img)
    # imgs.extend(keep)


    # imgs.extend(glob.glob("/Users/csayres/Downloads/fcam/60665/proc*.fits"))
    ###############################################

    # recalib based on fvc images from 60700-73
    # have to be careful to just pick first fvc image after blind robot move
    # and handle stationary robots correctly (eg ignore them).
    # posAngles = []
    # imgs = []
    # lastConfig = None
    # for mjd in [60700, 60701, 60702, 60703]:
    #     fs = glob.glob("/Users/csayres/Downloads/fcam/%i/proc*.fits"%mjd)
    #     fs = sorted(fs)
    #     for f in fs:
    #         ff = fits.open(f)
    #         pa = Table(ff["POSANGLES"].data).to_pandas()
    #         medianAlpha = numpy.around(numpy.median(pa["alphaReport"]))
    #         medianBeta = numpy.around(numpy.median(pa["betaReport"]))
    #         if medianAlpha==10.0 and medianBeta==170.0:
    #             print("skipping folded file", f)
    #             continue

    #         configID = ff[1].header["CONFIGID"]
    #         if configID == lastConfig:
    #             print("skipping, already have image for this config", f)
    #             continue

    #         imgs.append(f)
    #         pa["configID"] = configID
    #         posAngles.append(pa)
    #         lastConfig = configID
    #         print("keeping", f)

    # posAngles = pandas.concat(posAngles)
    # pa_std = posAngles[["positionerID", "alphaReport", "betaReport"]].groupby("positionerID").std().reset_index()

    # plt.figure()
    # plt.plot(pa_std.positionerID, pa_std.alphaReport, "o-", label="std alpha")
    # plt.plot(pa_std.positionerID, pa_std.betaReport, "o-", mfc="none", mec="black", label="std beta")
    # plt.legend()
    # plt.show()

    # # list broken robots
    # pa_broken = pa_std[pa_std.alphaReport < 0.2]
    # pa_broken = pa_broken[pa_broken.betaReport < 0.2]

    # brokenRobots = list(set(pa_broken.positionerID))
    # print("broken robots", len(brokenRobots), brokenRobots)
    # # add in 984 and 1043 (recently broken metrology)
    # brokenRobots += [984, 1043]



    # df = processImgs("stage0", imgs, pt=pt)
    # # remove broken robots
    # df = df[~df.positionerID.isin(brokenRobots)]
    # df.to_csv("dforig.csv")

    # plotDistances(df, title="stage0")

    # ptOut = fitCalibs(pt, df, disabledRobots=brokenRobots)
    # dfNew = processImgs("stage2", imgs, pt=ptOut)
    # dfNew = dfNew[~dfNew.positionerID.isin(brokenRobots)]
    # plotDistances(dfNew, title="stage1")


    # pt.to_csv("ptorig.csv")
    # ptOut.to_csv("ptnew.csv")
    # df.to_csv("dforig.csv")
    # dfNew.to_csv("dfnew.csv")






    # plt.show()




    # import pdb; pdb.set_trace()



    ### old stuff ###
    ################## after alpha/beta home, safe calib ####################
    # processImgs("stage2.2", pt=pt, imgStart=imgStart, imgEnd=imgEnd, maxFinalDist=1.5)
    # df = pandas.read_csv("pt_stage2.2.csv")
    # df = df[~df.wokErrWarn] # throw out likely mismatches
    # plotDistances(df, title="2.2")

    # positionerTableOut = fitCalibs(pt, df)
    # positionerTableOut.to_csv("positionerTable.stage2.2.csv")

    # processImgs("stage2.3", pt=positionerTableOut, imgStart=imgStart, imgEnd=imgEnd, maxFinalDist=0.5)

    # positionerTableOut = pandas.read_csv("positionerTable.stage2.2.csv")

    # df = pandas.read_csv("pt_stage2.3.csv")
    # df = df[~df.wokErrWarn] # throw out likely mismatches
    # plotDistances(df, title="2.3")

    # ############# danger move calibration #########################


    ##### apo shutdown 2025 ###########
    ### safe moves ###
    # mjd = 60930
    # imgStart = 80
    # imgEnd = 115
    #################

    ### danger moves ###
    # mjd = 60931
    # imgStart = 4
    # imgEnd = 42

    dd = {
        60931: [4,42],
        60931: [58,84],
        60932: [3,107]
    }


    imgList = []
    for mjd, imgRange in dd.items():
        for imgNum in range(imgRange[0], imgRange[1]+1):
            imgStr = str(imgNum).zfill(4)
            baseDir = "/Users/csayres/Downloads/fcam/%i"%mjd
            imgList.append(baseDir+"/proc-fimg-fvc1n-%s.fits"%imgStr)

    df = processImgs("stage0", imgList, pt=pt)
    df.to_csv("dfstart.csv")
    plotDistances(df, title="start")


    positionerTableOut = fitCalibs(pt, df)
    positionerTableOut.to_csv("positionerTable.apo.danger2025.winpos.f2f.csv", index_label="id")

    df = processImgs("stage1", imgList, pt=positionerTableOut)
    df.to_csv("dfdanger.csv")
    plotDistances(df, title="danger")

    plt.show()

    import pdb; pdb.set_trace()





    plt.show()

