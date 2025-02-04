import numpy
from astropy.io import fits
from astropy.table import Table
from multiprocessing import Pool
from scipy.signal import correlate
from scipy.interpolate import bisplrep, bisplev, RBFInterpolator, RegularGridInterpolator
from skimage.transform import resize, SimilarityTransform
from functools import partial
import pickle
import pandas
import os

from coordio.transforms import design_matrix, FVCTransformLCO, FVCTransformAPO, fourier_functions_outer

from compileData import getFVCPath
import time

import matplotlib.pyplot as plt

# import warnings

# def my_formatwarning(message, category, filename, lineno, line=None):
#   print(message, category)
#   # lineno is the line number you are looking for
#   print('file:', filename, 'line number:', lineno)
#   ...

# warnings.formatwarning = my_formatwarning

# lco calib data post-remount on 60521
# dataList = [
#     # site, mjd, imgStart, imgEnd, configID
#     ["lco", 60522, 3, 77, 1],
#     ["lco", 60522, 78, 152, 2],
#     ["lco", 60522, 153, 227, 3],
#     ["lco", 60522, 228, 302, 4],
#     ["lco", 60522, 303, 377, 5],
#     ["lco", 60523, 4, 78, 6],
#     ["lco", 60523, 79, 153, 7],
#     ["lco", 60523, 154, 228, 8],
#     ["lco", 60523, 229, 303, 9],
#     ["lco", 60574, 6, 124, 10],  # dense
#     ["lco", 60576, 35, 153, 11], # dense
#     ["lco", 60577, 2, 240, 12], # dense
# ]

#apo calib data post-shutdown 2024
# dataList = [
#     ["apo", 60518, 2, 73, 1],
#     ["apo", 60518, 74, 145, 2],
#     ["apo", 60518, 146, 217, 3],
#     ["apo", 60518, 218, 289, 4],
#     ["apo", 60518, 290, 361, 5],
#     ["apo", 60519, 1, 72, 6],
#     ["apo", 60519, 73, 144, 7],
#     ["apo", 60519, 145, 216, 8],
#     ["apo", 60519, 217, 288, 9],
#     ["apo", 60523, 3, 74, 10],
#     ["apo", 60523, 75, 146, 11],
#     ["apo", 60544, 4, 363, 12] # dense
# ]

#lco calib data post-engineering january 2025
dataList = [
    ["lco", 60691, 22, 257, 1], # dense sea-anemone
    ["lco", 60691, 284, 314, 2], # first half before mjd rollover
    ["lco", 60692, 1, 17, 2], # second half after mjd rollover
    ["lco", 60692, 18, 74, 3],
    ["lco", 60692, 75, 125, 4],
    ["lco", 60692, 126, 180, 5]
]


def getPSF(x, cutoutSize, upsample):
    site, mjd, imgNum, confID = x
    if site.lower() == "lco":
        _fvct = FVCTransformLCO
    else:
        _fvct = FVCTransformAPO
    f = getFVCPath(mjd,site,imgNum)
    ff = fits.open(f)
    ipa = ff[1].header["IPA"]
    fvct = _fvct(
        ff[1].data,
        Table(ff["POSANGLES"].data).to_pandas(),
        ipa,
        # plotPathPrefix="./temp"
    )
    fvct.extractCentroids()
    imgData = fvct.data_sub

    # get a box for each pixel and median combine them
    # boxwidth = 11
    pixStack = []
    for idx, row in fvct.centroids.iterrows():
        xp = int(numpy.around(row.xpeak))
        yp = int(numpy.around(row.ypeak))
        # cutout = imgData[yp-meanPsfRad:yp+meanPsfRad+1,xp-meanPsfRad:xp+meanPsfRad+1]
        # cutout = cutout / numpy.sum(cutout) # normalize by total flux
        # cutoutAll.append(cutout)

        # imgDataUp = resize(imgData, (imgData.shape[0]*upsample, imgData.shape[1]*upsample), order=0, anti_aliasing=False)

        cutout = imgData[yp-cutoutSize//2-2:yp+cutoutSize//2+3,xp-cutoutSize//2-2:xp+cutoutSize//2+3]  # extra padding for upsampling
        cutout = cutout / numpy.sum(cutout)
        cutout = resize(cutout, (cutout.shape[0]*upsample, cutout.shape[1]*upsample), order=0, anti_aliasing=False)
        # up = resize(cutout, (cutout.shape[0]*upsample, cutout.shape[1]*upsample), order=0, anti_aliasing=False)
        dxp2 = int(numpy.around((row.x-xp)*upsample))
        dyp2 = int(numpy.around((row.y-yp)*upsample))
        xp2 = cutout.shape[0]//2 + dxp2
        yp2 = cutout.shape[0]//2 + dyp2
        cutoutUp = cutout[yp2-(cutoutSize//2*upsample):yp2+(cutoutSize//2*upsample)+1, xp2-(cutoutSize//2*upsample):xp2+(cutoutSize//2*upsample)+1]
        cutoutUp = cutoutUp / numpy.sum(cutoutUp)
        # import pdb; pdb.set_trace()
        pixStack.append(cutoutUp)

        # boxwidth = 11
        # xint = int(numpy.around(row.x))
        # yint = int(numpy.around(row.y))
        # cutout = fvct.data_sub[yint-cutout//2:yint+cutout//2+1,xint-cutout//2:xint+cutout//2+1]
        # cutout = cutout / numpy.sum(cutout)
        # print("cutout shape", cutout.shape)

        # pixStack.append(cutout)

    medPix = numpy.median(pixStack, axis=0)
    # plt.figure()
    # plt.imshow(medPix, origin="lower")


    # plt.figure()
    # plt.plot(numpy.arange(len(medPix)), numpy.sum(medPix, axis=0))
    # plt.plot(numpy.arange(len(medPix)), numpy.sum(medPix, axis=1))

    # plt.show()
    # import pdb; pdb.set_trace()

    return medPix

    # pixStack = numpy.median(pixStack, axis=0)
    # plt.figure()
    # plt.imshow(pixStack, origin="lower")

    # plt.figure()
    # plt.plot(numpy.arange(cutout)-6,numpy.sum(pixStack,axis=0))

    # plt.plot(numpy.arange(cutout)-6, numpy.sum(pixStack,axis=1))

    # convStack = []
    # plt.figure(figsize=(8,8))
    # for idx, row in fvct.centroids.iterrows():
    #     # cutout = 11
    #     xint = int(numpy.around(row.x))
    #     yint = int(numpy.around(row.y))
    #     cutout = fvct.data_sub[yint-cutout//2:yint+cutout//2+1,xint-cutout//2:xint+cutout//2+1]
    #     cutout = cutout / numpy.sum(cutout)
    #     corr = correlate(pixStack, cutout, mode="same")
    #     convStack.append(corr)
    #     plt.plot(numpy.arange(cutout)-6, numpy.sum(corr, axis=0), '-b', lw=1)
    #     # plt.plot(numpy.arange(cutout)-6, numpy.sum(corr, axis=1), '-r', lw=1)

    # plt.show()

    # convStack = numpy.median(convStack, axis=0)
    # plt.plot(numpy.arange(cutout)-6,numpy.sum(convStack,axis=0))

    # plt.plot(numpy.arange(cutout)-6, numpy.sum(convStack,axis=1))

    # plt.show()
    # import pdb; pdb.set_trace()


    # fvct.fit(centType=)


def generateEPSF(cutoutSize=11, upsample=31):
    # only use a random sample of images from
    # across rotator scans
    # plt.figure()
    xList = []
    for site, mjd, start, end, config in dataList:
        # print("processing config", config)
        imgNums = numpy.arange(start, end+1)
        numpy.random.shuffle(imgNums)
        for imgNum in imgNums[:20]:
            # print("processing image", imgNum)
            x = [site, mjd, imgNum, config]
            xList.append(x)

            # plt.plot(numpy.arange(len(medPix)), numpy.sum(medPix, axis=0), '-r', alpha=0.1)
            # plt.plot(numpy.arange(len(medPix)), numpy.sum(medPix, axis=1), '-b', alpha=0.1)


    # plt.show()
    # import pdb; pdb.set_trace()
            # xList.append(x)
    _getPSF = partial(getPSF, cutoutSize=cutoutSize, upsample=upsample)

    medPSFs = []
    for x in xList:
        medPSFs.append(_getPSF(x))

    # tStart = time.time()
    # p = Pool(8)
    # medPSFs = p.map(_procOne, xList)
    # print("took", time.time()-tStart)

    _medPSFs = numpy.median(medPSFs, axis=0)
    plt.figure(figsize=(8,8))
    for psf in medPSFs:
        plt.plot(numpy.arange(len(psf)), numpy.sum(psf, axis=0), '-r', alpha=0.1)
        plt.plot(numpy.arange(len(psf)), numpy.sum(psf, axis=1), '-b', alpha=0.1)

    plt.plot(numpy.arange(len(_medPSFs)), numpy.sum(_medPSFs, axis=0), '-k', alpha=1)
    plt.plot(numpy.arange(len(_medPSFs)), numpy.sum(_medPSFs, axis=1), '-k', alpha=1)
    plt.show()

    d = {
        "site": site,
        "dataUsed": xList,
        "cutoutSize": 11,
        "upsample": 31,
        "psf": _medPSFs
    }

    outfile = "fvc_psf_%s.pkl"%site
    with open(outfile, "wb") as f:
        pickle.dump(d, f)


def combineRobotFIF(fcm, ptm):
    fcm = fcm[fcm.wokErrWarn==False]
    fcm["fiducial"] = True
    for col in ["row","col","xyVar", "index", "site"]:
        if col in fcm.columns:
            fcm.drop(col, axis=1, inplace=True)

    # delete the xyVar column to allow stacking with
    # ptm (xyVar is not in ptm table)
    keepcols = list(fcm.columns)

    ptm = ptm[ptm.wokErrWarn==False]
    ptm["id"] = ["P"+str(x) for x in ptm.positionerID.to_list()]
    ptm["xWokMeas"] = ptm.xWokMeasMetrology
    ptm["yWokMeas"] = ptm.yWokMeasMetrology
    ptm["fiducial"] = False

    if "index" in ptm.columns:
        ptm.drop(["index"], axis=1, inplace=True)

    ptm = ptm[keepcols]

    df = pandas.concat([fcm,ptm])
    return df


def procOne(x):
    if len(x) == 4:
        site, mjd, imgNum, confID = x
        calcRotCen = False
    else:
        site, mjd, imgNum, confID, xRotCenCCD, yRotCenCCD = x
        calcRotCen = True

    print("processing", x)
    if site.lower()=="lco":
        _fvct = FVCTransformLCO
    else:
        _fvct = FVCTransformAPO
    f = getFVCPath(mjd,site,imgNum)
    if not os.path.exists(f):
        return pandas.DataFrame()
    ff = fits.open(f)
    ipa = ff[1].header["IPA"]
    fvct = _fvct(
        ff[1].data,
        Table(ff["POSANGLES"].data).to_pandas(),
        ipa,
        # plotPathPrefix="./temp"
    )
    fvct.extractCentroids()
    # fvct.fit(centType="flex")
    fvct.fit(centType="sep")

    df = combineRobotFIF(fvct.fiducialCoordsMeas, fvct.positionerTableMeas)

    df["site"] = site.upper()
    df["imgNum"] = imgNum
    df["mjd"] = mjd
    df["configID"] = confID
    df["ipa"] = numpy.rint(ipa)

    if site.lower() == "lco":
        df["telAx1"] = numpy.rint(ff[1].header["HA"])
        df["telAx2"] = numpy.rint(ff[1].header["DEC"])
        df["alt"] = 90 - numpy.degrees(numpy.arccos(1/ff[1].header["AIRMASS"]))
    else:
        df["telAx1"] = numpy.rint(ff[1].header["AZ"])
        df["telAx2"] = numpy.rint(ff[1].header["ALT"])
        df["alt"] = df.telAx2
    if calcRotCen:
        # dxFlex = numpy.mean(fvct.positionerTableMeas.x - fvct.positionerTableMeas.xFlex)
        # dyFlex = numpy.mean(fvct.positionerTableMeas.y - fvct.positionerTableMeas.yFlex)
        rotCenXY = numpy.array([[xRotCenCCD,yRotCenCCD],[xRotCenCCD,yRotCenCCD]])

        xyWokRotCen = fvct.fullTransform.apply(rotCenXY)[0]
        df["xWokRotCen"] = xyWokRotCen[0]
        df["yWokRotCen"] = xyWokRotCen[1]
        print(xyWokRotCen)

    trans = [ff[1].header["FVC_TRAX"], ff[1].header["FVC_TRAY"]]
    rot = numpy.radians(ff[1].header["FVC_ROT"])
    scale = ff[1].header["FVC_SCL"]

    st = SimilarityTransform(
        translation=trans,
        rotation=rot,
        scale=scale
    )
    xyWokSim = st(df[["xFVC", "yFVC"]].to_numpy())
    df["xWokSim"] = xyWokSim[:,0]
    df["yWokSim"] = xyWokSim[:,1]
    df["fvcRot"] = rot

    return df


def processFVCimgs(xRotCenCCD=None, yRotCenCCD=None, configID=None):
    dfList = []
    procList = []
    for site, mjd, imgStart, imgEnd, confID in dataList:
        if configID is not None and configID != confID:
            continue
        for imgNum in range(imgStart,imgEnd+1):
            if xRotCenCCD is not None:
                procList.append([site,mjd,imgNum,confID,xRotCenCCD,yRotCenCCD])
            else:
                procList.append([site,mjd,imgNum,confID])

    p = Pool(8)
    dfList = p.map(procOne, procList)

    # dfList = []
    # for x in procList:
    #     dfList.append(procOne(x))

    df = pandas.concat(dfList)
    df.to_csv("fvc_nudge_data_%s.csv"%site)


def fitOuterNudge(site, mjd_dense):
    if site == "apo":
        wokCenPix = FVCTransformAPO.wokCenPix
    elif site == "lco":
        wokCenPix = FVCTransformLCO.wokCenPix
    else:
        raise RuntimeError("unrecognized site")
    df = pandas.read_csv("fvc_nudge_data_%s.csv"%site, index_col=0)
    df = df[df.outerNudge==True]
    df = df[df.fiducial==True]
    df = df[df.mjd.isin(mjd_dense)] # only use dense scans
    # df = df[df.mjd==60577] # only use dense scans

    # plt.figure(figsize=(8,8))
    # plt.plot(df.xFlex, df.yFlex, '.k')
    # plt.axis("equal")
    # plt.show()
    # import pdb; pdb.set_trace()
    dfList = []
    for name, group in df.groupby("imgNum"):
        xm = numpy.mean(group.x)
        ym = numpy.mean(group.y)
        _x = group.x - xm
        _y = group.y - ym
        # import pdb; pdb.set_trace()
        # group["thetaFVC"] = numpy.arctan2(_y,_x)
        group["thetaFVC"] = numpy.arctan2(group.yFlex-wokCenPix[1],group.xFlex-wokCenPix[0])
        dfList.append(group)

    df = pandas.concat(dfList)

    # import pdb; pdb.set_trace()
    # df = df[df.outerFIF==True]
    # df["rWok"] = numpy.sqrt(df.xWok**2+df.yWok**2)
    # df = df[df.rWok > 320]
    # df = df[df.fiducial==True]

    df_m = df[["configID", "imgNum", "id", "x", "y", "xWokMeas", "yWokMeas", "xWokSim", "yWokSim", "fvcRot", "ipa", "thetaFVC"]]
    df_m = df_m.groupby(["configID", "id"]).mean().reset_index()

    df = df.merge(df_m, on=["configID", "id"], suffixes=(None, "_m"))
    df["dxWok"] = df.xWokMeas - df.xWokMeas_m
    df["dyWok"] = df.yWokMeas - df.yWokMeas_m
    df["drWok"] = numpy.sqrt(df.dxWok**2+df.dyWok**2)
    # df["thetaWok"] = numpy.arctan2(df.yWokSim_m, df.xWokSim_m)
    # df["thetaPhase"] = (df.fvcRot - df.thetaWok)%(2*numpy.pi)


    cosRot = numpy.cos(-1*df.fvcRot)
    sinRot = numpy.sin(-1*df.fvcRot)

    df["dxFVC"] = (df.dxWok*cosRot - df.dyWok*sinRot)/.120
    df["dyFVC"] = (df.dxWok*sinRot + df.dyWok*cosRot)/.120


    # for name, group in df.groupby("id"):
    #     plt.figure(figsize=(8,8))
    #     group = group.sort_values("thetaPhase")
    #     plt.plot(group.thetaPhase, group.dxFVC, '.', label="dxFVC")
    #     plt.plot(group.thetaPhase, group.dyFVC, '.', label="dyFVC")
    #     plt.title(name)
    #     plt.legend()

    plt.figure(figsize=(8,8))
    for name, group in df.groupby("id"):
        group = group.sort_values("thetaFVC")
        plt.plot(group.thetaFVC, group.dxFVC, '-', label=name)
        plt.title("dx")
        #plt.ylim([-1,1])
    # plt.legend()

    plt.figure(figsize=(8,8))
    for name, group in df.groupby("id"):
        group = group.sort_values("thetaFVC")
        plt.plot(group.thetaFVC, group.dyFVC, '-', label=name)
        plt.title("dy")
        #plt.ylim([-1,1])





    orders = []
    xresid = []
    yresid = []
    for nOrder in range(1,20):
        orders.append(nOrder)
        X = fourier_functions_outer(df.thetaFVC.to_numpy(), nOrder=nOrder)
        dx = df.dxFVC.to_numpy()
        dy = df.dyFVC.to_numpy()
        trainInds = numpy.random.uniform(size=len(dx)) < 0.8
        testInds = ~trainInds

        Xtrain = X[trainInds,:]
        dxtrain = dx[trainInds]
        dytrain = dy[trainInds]

        coeffs_x = numpy.linalg.lstsq(Xtrain, dxtrain)[0]
        coeffs_y = numpy.linalg.lstsq(Xtrain, dytrain)[0]

        Xtest = X[testInds,:]
        dxtest = dx[testInds]
        dytest = dy[testInds]

        dxHats = Xtest @ coeffs_x
        dyHats = Xtest @ coeffs_y

        residX = dxHats - dxtest
        residY = dyHats - dytest
        xresid.append(numpy.sqrt(numpy.mean(residX**2)))
        yresid.append(numpy.sqrt(numpy.mean(residY**2)))

        print("x rms before/after", numpy.sqrt(numpy.mean(dxtest**2)), numpy.sqrt(numpy.mean(residX**2)))
        print("y rms before/after", numpy.sqrt(numpy.mean(dytest**2)), numpy.sqrt(numpy.mean(residY**2)))



    print("n order", nOrder)
    plt.figure()
    plt.plot(orders, xresid, label="xresidual")
    plt.plot(orders, yresid, label="yresidual")
    plt.legend()

    # plot full models
    dxAll = X @ coeffs_x
    dyAll = X @ coeffs_y
    plt.figure(figsize=(6,6))
    plt.plot(df.thetaFVC, dxAll, ".", label="dx fit")
    plt.plot(df.thetaFVC, dyAll, ".", label="dy fit")
    plt.legend()

    # plot residuals
    residX = dxAll - dx
    residY = dyAll - dy
    df["residX"] = residX
    df["residY"] = residY

    plt.figure(figsize=(6,6))
    for name, group in df.groupby("id"):
        group = group.sort_values("thetaFVC")
        plt.plot(group.thetaFVC, group.residX, '-', label=name)
        plt.title("resid dx")
        #plt.ylim([-1,1])
    # plt.legend()

    plt.figure(figsize=(6,6))
    for name, group in df.groupby("id"):
        group = group.sort_values("thetaFVC")
        plt.plot(group.thetaFVC, group.residY, '-', label=name)
        plt.title("resid dy")
        #plt.ylim([-1,1])


    print("coeffs_x", coeffs_x)
    print("\n")
    print("coeffs_y", coeffs_y)

    with open("beta_x_outer_%s.npy"%site, "wb") as f:
        numpy.save(f, coeffs_x)
    with open("beta_y_outer_%s.npy"%site, "wb") as f:
        numpy.save(f, coeffs_y)
    print("len coeffs", len(coeffs_x), "nOrder", nOrder)
    plt.show()

def fitDistortion(xs, ys, dxs, dys, trainFrac=0.6):
    t1 = time.time()
    X = design_matrix(xs, ys)

    n, p = X.shape

    numpy.random.seed(42)
    rands = numpy.random.uniform(size=n)

    train = rands <= trainFrac
    test = rands > trainFrac

    beta_x, resids, rank, s = numpy.linalg.lstsq(X[train], dxs[train], rcond=None)

    dxs_hat = X[test] @ beta_x

    beta_y, resids, rank, s = numpy.linalg.lstsq(X[train], dys[train], rcond=None)
    dys_hat = X[test] @ beta_y

    origRMS = numpy.sqrt(numpy.mean(dxs[test]**2 + dys[test]**2))
    fitRMS = numpy.sqrt(numpy.mean((dxs[test] - dxs_hat) ** 2 + (dys[test] - dys_hat) ** 2))

    return beta_x, beta_y, origRMS, fitRMS


def applyDistortion(xs, ys, beta_x, beta_y):
    X = design_matrix(xs, ys)
    dxs_hat = X @ beta_x
    dys_hat = X @ beta_y
    return dxs_hat, dys_hat


def fitNudgeSpline(site, fit=True):
    df = pandas.read_csv("fvc_nudge_data_%s.csv"%site, index_col=0)
    df_m = df[["configID", "id", "xWokMeas", "yWokMeas", "xWokSim", "yWokSim"]]
    df_m = df_m.groupby(["configID", "id"]).mean().reset_index()

    # df = df[df.xFlex < 4000]
    # df = df[df.xFlex > 2000]

    # df = df[df.yFlex < 4000]
    # df = df[df.yFlex > 2000]


    df = df.merge(df_m, on=["configID", "id"], suffixes=(None, "_m"))
    df["dxWok"] = df.xWokMeas - df.xWokMeas_m
    df["dyWok"] = df.yWokMeas - df.yWokMeas_m
    df["drWok"] = numpy.sqrt(df.dxWok**2+df.dyWok**2)
    cosRot = numpy.cos(-1*df.fvcRot)
    sinRot = numpy.sin(-1*df.fvcRot)

    df["dxFVC"] = (df.dxWok*cosRot - df.dyWok*sinRot)/.120
    df["dyFVC"] = (df.dxWok*sinRot + df.dyWok*cosRot)/.120

    # dfTrain = df.sample(frac=0.1)

    plt.figure(figsize=(8,8))
    plt.quiver(df.xFlex, df.yFlex, df.dxFVC, df.dyFVC, angles="xy", units="xy", width=0.2, scale=0.004)
    plt.axis("equal")

    # plt.show()

    if fit:
        ds = 5
        DX = numpy.zeros((6000//ds,6000//ds))
        DY = numpy.zeros((6000//ds,6000//ds))
        xs = df.xFlex.to_numpy()
        ys = df.yFlex.to_numpy()
        dx = df.dxFVC.to_numpy()
        dy = df.dyFVC.to_numpy()
        for idx in range(len(DX)):
            print("on idx", idx)
            for idy in range(len(DX)):
                dist = numpy.sqrt((idx*ds-xs)**2+(idy*ds-ys)**2)
                keep = dist < 50 # average over radius of 30 (true) pixels
                weights = 1 / (dist[keep]**2)
                if not True in keep:
                    continue
                # print("keeping", numpy.sum(keep))
                _dx = numpy.average(dx[keep], weights=weights)
                _dy = numpy.average(dy[keep], weights=weights)
                # print("_dxy", _dx, _dy)
                DX[idy,idx] = _dx
                DY[idy,idx] = _dy

                # if idx*ds > 1000 and idy*ds > 1000 and sum(keep) > 20:
                #     plt.figure()
                #     plt.quiver(xs[keep], ys[keep], _dx, _dy)
                #     plt.plot(idx*ds, idy*ds, '+r')
                #     plt.show()
                #     import pdb; pdb.set_trace()

        d = {
            "downsample": ds,
            "DX": DX,
            "DY": DY
        }
        with open("nudge_%s.pkl"%site, "wb") as f:
            pickle.dump(d, f)

    nudge = pickle.load(open("nudge_%s.pkl"%site, "rb"))
    xTest = df.xFlex.to_numpy()/nudge["downsample"]
    yTest = df.yFlex.to_numpy()/nudge["downsample"]

    plt.figure()
    plt.imshow(nudge["DX"], origin="lower")


    plt.figure()
    plt.imshow(nudge["DY"], origin="lower")

    plt.figure()
    plt.imshow(nudge["DY"]==0, origin="lower")

    plt.plot(xTest, yTest, '.', color="white")


    import time; tstart=time.time()

    inds = numpy.arange(nudge["DX"].shape[1])

    dxInterp = RegularGridInterpolator((inds, inds), nudge["DX"], method="cubic")
    dyInterp = RegularGridInterpolator((inds, inds), nudge["DY"], method="cubic")

    dxHats = dxInterp((yTest,xTest))
    dyHats = dyInterp((yTest,xTest))
    print("interp stuff took", tstart-time.time())

    # inds = numpy.array(df[["xFlex", "yFlex"]].to_numpy()/nudge["downsample"], dtype=int)

    # dxHats = [nudge["DX"][_y, _x] for _x,_y in inds]
    # dyHats = [nudge["DY"][_y, _x] for _x,_y in inds]

    df["dxHatFVC"] = dxHats
    df["dyHatFVC"] = dyHats

    # plt.show()

    # import pdb; pdb.set_trace()

    # X = dfTrain[["xFlex", "yFlex"]].to_numpy()
    # dx = dfTrain.dxFVC.to_numpy()
    # dy = dfTrain.dyFVC.to_numpy()

    # xMod = RBFInterpolator(X, dx)
    # yMod = RBFInterpolator(X, dy)
    # df["dxHatFVC"] = xMod(df[["xFlex", "yFlex"]].to_numpy())
    # df["dyHatFVC"] = yMod(df[["xFlex", "yFlex"]].to_numpy())

    # import time; tstart = time.time()
    # tckx = bisplrep(X[:,0], X[:,1], dx) #, s=0)
    # tcky = bisplrep(X[:,0], X[:,1], dy) #df.dyFVC.to_numpy(), s=0)
    # print("fit took", time.time()-tstart)
    # df["dxHatFVC"] = bisplev(X[:,0], X[:,1], tckx)
    # df["dxHatFVC"] = bisplev(X[:,0], X[:,1], tcky)
    # import pdb; pdb.set_trace()

    # df = df.iloc[:300]
    # print(len(df))




    plt.figure(figsize=(8,8))
    plt.quiver(df.xFlex, df.yFlex, df.dxHatFVC, df.dyHatFVC, angles="xy", units="xy", width=0.2, scale=0.004)
    plt.axis("equal")

    df["dxResid"] = df.dxHatFVC - df.dxFVC
    df["dyResid"] = df.dyHatFVC - df.dyFVC
    df["drResid"] = numpy.sqrt(df.dxResid**2+df.dyResid**2)
    plt.figure()
    plt.hist(df.drResid*.120, bins = numpy.linspace(0,0.1,300))
    plt.xlim([0,0.04])
    plt.title("dr resid lookup")

    plt.figure()
    plt.hist(numpy.sqrt(df.dxFVC**2+df.dyFVC**2)*.120, bins=300)


    plt.figure(figsize=(8,8))
    plt.quiver(df.xFlex, df.yFlex, df.dxResid, df.dyResid, angles="xy", units="xy", width=0.2, scale=0.004)
    plt.axis("equal")

    return df.drResid.to_numpy()



def fitNudge(site, fit=True):
    df = pandas.read_csv("fvc_nudge_data_%s.csv"%site, index_col=0)
    # df = df[df.outerNudge==False]
    print(set(df.configID))
    print(set(df.mjd))

    df_m = df[["configID", "id", "xWokMeas", "yWokMeas", "xWokSim", "yWokSim"]]
    df_m = df_m.groupby(["configID", "id"]).mean().reset_index()

    df = df.merge(df_m, on=["configID", "id"], suffixes=(None, "_m"))
    df["dxWok"] = df.xWokMeas - df.xWokMeas_m
    df["dyWok"] = df.yWokMeas - df.yWokMeas_m
    df["drWok"] = numpy.sqrt(df.dxWok**2+df.dyWok**2)

    bins = numpy.linspace(0,0.1,300)
    plt.figure()
    plt.hist(df.drWok,bins=bins)
    # plt.xlim([0,0.07])


    # plt.figure(figsize=(8,8))
    # plt.quiver(df.xWokMeas, df.yWokMeas, df.dxWok, df.dyWok, angles="xy", units="xy", width=0.1, scale=0.005)
    # plt.axis("equal")

    cosRot = numpy.cos(-1*df.fvcRot)
    sinRot = numpy.sin(-1*df.fvcRot)

    df["dxFVC"] = (df.dxWok*cosRot - df.dyWok*sinRot)/.120
    df["dyFVC"] = (df.dxWok*sinRot + df.dyWok*cosRot)/.120

    plt.figure(figsize=(8,8))
    plt.quiver(df.xFlex, df.yFlex, df.dxFVC, df.dyFVC, angles="xy", units="xy", width=0.2, scale=0.004)
    plt.axis("equal")

    # plt.show()
    # import pdb; pdb.set_trace()

    # plot errors bigger than 50 microns in red
    # _df = df[df.drWok > 0.06]
    # plt.quiver(_df.xFlex, _df.yFlex, _df.dxFVC, _df.dyFVC, color="red", angles="xy", units="xy", width=0.2, scale=0.004)


    # _df = df[df.drWok > 0.06]
    # plt.plot(_df.xFlex, _df.yFlex, '.k')
    # plt.show()
    # import pdb; pdb.set_trace()



    # fit and plot distortion model
    xs = df.xFlex.to_numpy()
    ys = df.yFlex.to_numpy()
    dxs = df.dxFVC.to_numpy()
    dys = df.dyFVC.to_numpy()

    if fit:
        beta_x, beta_y, origRMS, fitRMS = fitDistortion(xs, ys, dxs, dys, trainFrac=0.8)

        print("orig/fit", origRMS*.120, fitRMS*.120)

        with open("beta_x_%s.npy"%site, "wb") as f:
            numpy.save(f, beta_x)
        with open("beta_y_%s.npy"%site, "wb") as f:
            numpy.save(f, beta_y)

    with open("beta_x_%s.npy"%site, "rb") as f:
        beta_x = numpy.load(f)

    with open("beta_y_%s.npy"%site, "rb") as f:
        beta_y = numpy.load(f)

    dx_hats, dy_hats = applyDistortion(xs,ys,beta_x,beta_y)
    dxResid = dx_hats - dxs
    dyResid = dy_hats - dys
    drResid = numpy.sqrt(dxResid**2+dyResid**2)

    plt.figure()
    plt.hist(drResid*.120,bins=bins)
    plt.title("dr resid fourier")
    plt.xlim([0,0.04])

    plt.figure(figsize=(8,8))
    plt.quiver(df.xFlex, df.yFlex, dxResid, dyResid, angles="xy", units="xy", width=0.2, scale=0.004)
    plt.axis("equal")

    ### visualize models
    xs = []
    ys = []
    for x in numpy.linspace(0,6000,1000):
        for y in numpy.linspace(0,6000,1000):
            xs.append(x)
            ys.append(y)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    dx_hats, dy_hats = applyDistortion(xs,ys,beta_x,beta_y)
    r_hats = numpy.sqrt(dx_hats**2+dy_hats**2)
    keep = r_hats < 1
    _xs = xs[keep]
    _ys = ys[keep]
    _dx_hats = dx_hats[keep]
    _dy_hats = dy_hats[keep]
    plt.figure(figsize=(8,8))
    plt.quiver(_xs, _ys, _dx_hats, _dy_hats, angles="xy", units="xy", width=0.2, scale=0.004)
    plt.axis("equal")

    toss = r_hats > 1
    _xs = xs[toss]
    _ys = ys[toss]
    _dx_hats = dx_hats[toss]
    _dy_hats = dy_hats[toss]
    plt.figure(figsize=(8,8))
    plt.plot(_xs, _ys, '.k') # _dx_hats, _dy_hats, color="red", angles="xy", units="xy", width=0.2, scale=0.004)
    plt.axis("equal")

    return drResid

def calc_rotInstXY(site):
    from coordio.utils import fit_circle
    df = pandas.read_csv("fvc_nudge_data_%s.csv"%site, index_col=0)
    # import pdb; pdb.set_trace()
    # just use first config
    df = df[df.configID==1]

    _xCenCCD = []
    _yCenCCD = []
    for name, group in df.groupby(["configID","id"]):
        xCen,yCen,rad,keep = fit_circle(group.x, group.y)
        _xCenCCD.append(xCen)
        _yCenCCD.append(yCen)
        # plt.figure()
        # plt.plot(group.x, group.y, '.k')
        # plt.axis("equal")
        # plt.plot(xCen,yCen,"xr")

        # thetas = numpy.linspace(0,2*numpy.pi,1000)
        # xs = rad*numpy.cos(thetas) + xCen
        # ys = rad*numpy.sin(thetas) + yCen
        # plt.plot(xs,ys,'r')

        # plt.show()
        # import pdb; pdb.set_trace()
    xccd = numpy.array(_xCenCCD)
    yccd = numpy.array(_yCenCCD)

    plt.figure()
    plt.plot(xccd, yccd, '.k')
    plt.axis("equal")
    plt.xlabel("x (pix)")
    plt.ylabel("y (pix)")
    # plt.show()


    # import pdb; pdb.set_trace()
    xccd = numpy.mean(xccd)
    yccd = numpy.mean(yccd)

    processFVCimgs(xccd,yccd,configID=1)

    # new file has extra columns xWokRotCen, yWokRotCen
    df = pandas.read_csv("fvc_nudge_data_%s.csv"%site, index_col=0)
    df = df[df.configID==1]

    plt.figure(figsize=(5,5))
    plt.plot(df.xWokRotCen, df.yWokRotCen, '.k')
    plt.title("Rot location %.2f, %.2f (wok mm)"%(numpy.mean(df.xWokRotCen), numpy.mean(df.yWokRotCen)))
    # plt.axis("equal")
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")

    plt.figure()
    plt.plot(df.ipa, df.xWokRotCen, '.', label="x wok rot cen")
    plt.plot(df.ipa, df.yWokRotCen, '.', label="y wok rot cen")
    plt.legend()
    plt.xlabel("IPA")
    plt.ylabel("mm")

    plt.show()






if __name__ == "__main__":
    # generateEPSF() # write pickle result to fps_calibrations
    # processFVCimgs() # measure wok locations under "flex model" (no nudge)
    # fitOuterNudge("lco", [60691])
    # dr1 = fitNudge("apo", fit=True)

    # calc_rotInstXY("lco")
    # plt.show()

    xyRotCen = numpy.array([0.02,-0.26])
    pa = numpy.radians(270)
    cosp = numpy.cos(pa)
    sinp = numpy.sin(pa)
    scale = 330.275 # mm/deg
    # on sky
    rotMat = numpy.array([[cosp, -sinp],[sinp, cosp]])
    dxySky = 1/scale*rotMat@xyRotCen * 3600
    print("dxysky", dxySky)



    # x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])
    # def ff(x, y):
    #     return x**2 + y**2

    # xg, yg = np.meshgrid(x, y, indexing='ij')
    # data = ff(xg, yg)
    # interp = RegularGridInterpolator((x, y), data,
    #                                  bounds_error=False, fill_value=None)


    # dr1 = fitNudge("apo", fit=False)
    # dr2 = fitNudgeSpline("apo", fit=False)

    # dr1 = dr1*.120
    # dr2 = dr2*.120
    # print("old", numpy.mean(dr1), numpy.median(dr1), numpy.percentile(dr1, 95))
    # print("new", numpy.mean(dr2), numpy.median(dr2), numpy.percentile(dr2, 95))

    # plt.figure()
    # bins = numpy.linspace(0, 0.1, 300)
    # plt.hist(dr1, bins=bins, cumulative=True, histtype="step", density=True, label="old")
    # plt.hist(dr2, bins=bins, cumulative=True, histtype="step", density=True, label="new")
    # plt.legend()

    # plt.show()


