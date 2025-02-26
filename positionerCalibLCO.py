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
site = "LCO"
# baseDir = "/Volumes/futa/apo/data/fcam/%i"%mjd

# 54 first robot to get to run by disabling, maybe its flux?

# disabledRobots = [608, 612, 1136, 182, 54, 1300, 565, 719]


def processImg(imgPath, name, pt, wc, fc):
    # print("imgPath", imgPath)
    imgNum = int(imgPath.strip(".fits").split("-")[-1])
    mjd = int(imgPath.split("/")[-2])
    ff = fits.open(imgPath)

    imgData = ff[1].data
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


def fitCalibs(positionerTableIn, positionerTableMeas, disabledRobots=None):
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
    positionerTableOut["metX"] = _xBeta
    positionerTableOut["alphaArmLen"] = _la
    positionerTableOut["alphaOffset"] = _alphaOffDeg
    positionerTableOut["betaOffset"] = _betaOffDeg
    positionerTableOut["dx"] = _dx
    positionerTableOut["dy"] = _dy

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
    rmsStr = " RMS: %.2f um"%rms
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

    ############# lco #############################
    # post robot replace
    ###############################################

    # imgNums = list(range(10,30)) + list(range(33,51))
    # imgs = ["/Users/csayres/Downloads/fcam/lco/60690/proc-fimg-fvc1s-%s.fits"%(str(x).zfill(4)) for x in imgNums]
    # # hack robot 725 to it's old offset
    # idx = numpy.argwhere(pt.positionerID==725)[0][0]
    # alphaOffset = pt.alphaOffset.to_numpy()
    # betaOffset = pt.betaOffset.to_numpy()
    # alphaOffset[idx] = 3.948
    # betaOffset[idx] = 0.846
    # pt["alphaOffset"] = alphaOffset
    # pt["betaOffset"] = betaOffset

    # week or two post robot replecements, data taken for ap fiber measurements, but just using
    # normal fvc loop data
    imgNums = []
    for imgNumStart, rot in [[12,180],[187,60]]:
        imgNum = imgNumStart
        for cc in range(15):
            for ii in range(3):
                imgNums.append(imgNum)
                imgNum += 1
            for ii in range(6):
                imgNum += 1
    imgs = ["/Users/csayres/Downloads/fcam/lco/60711/proc-fimg-fvc1s-%s.fits"%(str(x).zfill(4)) for x in imgNums]


    df = processImgs("stage0", imgs, pt=pt)
    # df, brokenMets = massage(df) # massage for APO
    # # df = getPTM(imgs)
    #
    df.to_csv("dforig.csv")

    plotDistances(df, title="stage0")

    ptOut = fitCalibs(pt, df) #, disabledRobots=brokenMets)
    dfNew = processImgs("stage2", imgs, pt=ptOut)
    # dfNew, brokenMets = massage(dfNew) massage only for APO
    plotDistances(dfNew, title="stage1")


    pt.to_csv("ptorig.csv")
    ptOut.to_csv("ptnew.csv")
    df.to_csv("dforig.csv")
    dfNew.to_csv("dfnew.csv")

    ptOut = pandas.read_csv("ptnew.csv")
    dfNew = pandas.read_csv("dfnew.csv")

    ptOut2 = fitCalibs(ptOut, dfNew) #, disabledRobots=brokenMets)
    dfNew2 = processImgs("stage3", imgs, pt=ptOut2)
    # dfNew2, brokenMets = massage(dfNew2) massage only for APO
    plotDistances(dfNew2, title="stage3")


    idx = numpy.argwhere(ptOut2.positionerID==725)[0][0]
    alphaOffset = ptOut2.alphaOffset.to_numpy()
    betaOffset = ptOut2.betaOffset.to_numpy()
    alphaOffset[idx] = 0
    betaOffset[idx] = 0
    ptOut2["alphaOffset"] = alphaOffset
    ptOut2["betaOffset"] = betaOffset

    ptOut2.drop("Unnamed: 0", axis=1, inplace=True)
    ptOut2.to_csv("ptnew2.csv")
    dfNew2.to_csv("dfnew2.csv")

    plt.show()

    # weird robots {864, 563}
    # long mets [564, 563, 864, 1166]


    import pdb; pdb.set_trace()



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




    plt.show()

