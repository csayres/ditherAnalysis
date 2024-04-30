from astropy.io import fits
import glob
from astropy.table import Table
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import numpy
from coordio.utils import radec2wokxy
from findFiberCenter import fitOneSet, _plotOne
import os
import socket
from multiprocessing import Pool

from parseConfSummary import parseConfSummary

mjd = 60420

_hostname = socket.gethostname()
if "Conors" in _hostname:
    LOCATION = "local"
    OUT_DIR = os.getcwd()
    CORES = 10
if "apogee" in _hostname:
    LOCATION = "utah"
    OUT_DIR = "/uufs/chpc.utah.edu/common/home/u0449727/work/ditherAnalysis"
    CORES = 30
if "sdss5" in _hostname:
    LOCATION = "mountain"
    OUT_DIR = os.getcwd()
    CORES = 1
else:
    raise RuntimeError("unrecoginzed computer, don't know where data is")


def getGFAFiles(mjd, site, location=LOCATION):
    site = site.lower()
    if location == "local":
        glbstr = "/Volumes/futa/%s/data/gcam/%i/proc*.fits"%(site,mjd)
    elif location == "mountain":
        glbstr = glbstr = "/data/gcam/%i/proc*.fits"%(mjd)
    else:
        # utah
        glbstr = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/%s/%i/proc*.fits"%(site,mjd)

    return glob.glob(glbstr)


def getBOSSPath(mjd, site, location=LOCATION):
    site = site.lower()
    if location == "local":
        bossPath = "/Volumes/futa/%s/data/boss/sos/%i/dither"%(site, mjd)
    elif location == "mountain":
        bossPath = "/data/boss/sos/%i/dither"%(mjd)
    else:
        # utah
        bossPath = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/boss/sos/%s/%i/dither"%(site,mjd)

    return bossPath


def getFVCPath(mjd, site, imgNum, location=LOCATION):
    site = site.lower()
    imgNumStr = str(imgNum).zfill(4)
    if site == "apo":
        camname = "fvc1n"
    else:
        camname = "fvc1s"
    if location == "local":
        fvcPath = "/Volumes/futa/%s/data/fcam/%i/proc-fimg-%s-%s.fits"%(site, mjd, camname, imgNumStr)
    elif location == "mountain":
        fvcPath = "/data/fcam/%i/proc-fimg-%s-%s.fits"%(mjd, camname, imgNumStr)
    else:
        # utah
        fvcPath = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/%s/%i/proc-fimg-%s-%s.fits"%(site, mjd, camname, imgNumStr)

    return fvcPath


def getConfSummPath(configID, site, location=LOCATION):
    site = site.lower()

    confStr = str(configID).zfill(6)[:-2] + "XX"

    if location == "local":
        confPath = "confSummaryF-%i.par"%configID
    elif location == "mountain":
        confPath = "/home/sdss5/software/sdsscore/main/%s/summary_files/%s/confSummaryF-%i.par"%(site, confStr, configID)
    else:
        # utah
        confPath = "/uufs/chpc.utah.edu/common/home/sdss50/software/git/sdss/sdsscore/main%s/summary_files/%s/confSummaryF-%i.par"%(site, confStr, configID)

    return confPath


def getGFATables(mjd, site):
    site = site.lower()
    files = getGFAFiles(mjd, site)
    dfList = []
    for f in files:
        ff = fits.open(f)
        toks = f.split("-")

        offra = ff[1].header["OFFRA"]
        offdec = ff[1].header["OFFDEC"]
        offpa = ff[1].header["AOFFPA"]
        bossExp = ff[1].header["SPIMGNO"]

        if site == "apo":
            gfaNum = int(toks[-2].strip("gfa").strip("n"))
            bossExp = bossExp + 1
        else:
            gfaNum = int(toks[-2].strip("gfa").strip("s"))

        if bossExp == -999:
            continue
        if offra == 0 and offdec == 0:
            continue

        t = Table(ff["GAIAMATCH"].data).to_pandas()
        # table has zps for all gfas
        t = t[t.gfaNum == gfaNum].reset_index(drop=True)
        t["gfaImgNum"] = int(toks[-1].strip(".fits"))
        t["configID"] = ff[1].header["CONFIGID"]
        t["offra"] = offra
        t["offdec"] = offdec
        t["offpa"] = offpa
        t["bossExpNum"] = bossExp
        t["taiMid"] = ff[1].header["TAI_MID"]
        t["SOL_RA"] = ff[1].header["SOL_RA"]
        t["SOL_DEC"] = ff[1].header["SOL_DEC"]
        t["SOL_PA"] = ff[1].header["SOL_PA"]
        t["SOL_SCL"] = ff[1].header["SOL_SCL"]
        t["SOL_ALT"] = ff[1].header["SOL_ALT"]
        t["SOL_AZ"] = ff[1].header["SOL_AZ"]
        t["SOLVMODE"] = ff[1].header["SOLVMODE"]
        # t["FWHM_FIT"] = ff[1].header["FWHM_FIT"]
        t["file_path"] = f
        t["gfaDateObs"] = ff[1].header["DATE-OBS"]
        t["gfaExptime"] = ff[1].header["EXPTIMEN"]
        dfList.append(t)

    dfGFA = pandas.concat(dfList)
    # drop camera column it's reduntant with gfaNum
    # and conflicts with boss camera columns
    dfGFA = dfGFA.drop("camera", axis=1)
    # dfGFA.to_csv("dither_gfa_%s_%i.csv"%(site,mjd))
    return dfGFA


def getBossFlux(mjd, site, expNum):
    site = site.lower()
    if site == "apo":
        dd = {
            "r1": numpy.mean([6910, 8500]), # mean of bandpass in nm
            "b1": numpy.mean([4000, 5500]), # mean of bandpass in nm
        }
    else:
        dd = {
            "r2": numpy.mean([6910, 8500]), # mean of bandpass in nm
            "b2": numpy.mean([4000, 5500]), # mean of bandpass in nm
        }

    basePath = getBOSSPath(mjd, site)
    expNumStr = str(expNum).zfill(8)
    dfList = []
    for camera, lambdaCen in dd.items():
        globpath = basePath + "/ditherBOSS-%s-%s*.fits"%(expNumStr, camera)
        filepath = glob.glob(globpath)
        df = pandas.DataFrame()
        if len(filepath) == 1:
            ff = fits.open(filepath[0])
            df = Table(ff[1].data).to_pandas()
            df["lambdaCen"] = lambdaCen
            df["fiberID"] = df.fiber
            df["bossExpNum"] = expNum
            df["mjd"] = mjd

            keepCols = ["bossExpNum", "fiberID", "mjd", "lambdaCen", "camera", "spectroflux", "spectroflux_ivar", "objtype"]
            df = df[keepCols]
        dfList.append(df)

    return pandas.concat(dfList)


def getFVCData(mjd, site, expNum):
    imgPath = getFVCPath(mjd, site, expNum)
    ff = fits.open(imgPath)
    ptm = Table(ff["POSITIONERTABLEMEAS"].data).to_pandas()

    fcm = Table(ff["FIDUCIALCOORDSMEAS"].data).to_pandas()
    fcm["xWokMeasMetrology"] = fcm.xWokMeas
    fcm["yWokMeasMetrology"] = fcm.yWokMeas
    fcm["positionerID"] = -1
    fcm["xWokReportMetrology"] = fcm.xWok
    fcm["yWokReportMetrology"] = fcm.yWok

    ptm = pandas.concat([ptm, fcm], axis=0)

    ptm["configID"] = ff[1].header["CONFIGID"]
    ptm["designID"] = ff[1].header["DESIGNID"]
    ptm["mjd"] = mjd
    ptm["fvcImgNum"] = expNum
    ptm["fvcRot"] = ff[1].header["FVC_ROT"]
    ptm["fvcScale"] = ff[1].header["FVC_SCL"]
    ptm["fvcTransX"] = ff[1].header["FVC_TRAX"]
    ptm["fvcTransY"] = ff[1].header["FVC_TRAY"]
    ptm["fvcIPA"] = ff[1].header["IPA"]
    ptm["fvcALT"] = ff[1].header["ALT"]
    ptm["fvcAZ"] = ff[1].header["AZ"]

    # import pdb; pdb.set_trace()
    return ptm


def getDitherTables(mjd, site):
    site = site.lower()
    dfGFA = getGFATables(mjd, site)
    # dfGFA = pandas.read_csv("dither_gfa_%i.csv"%mjd)
    configIDs = list(set(dfGFA.configID))
    dfList = []

    # get confsummary data
    for configID in configIDs:
        confpath = getConfSummPath(configID, site)
        dfList.append(parseConfSummary(confpath))
    dfConfSumm = pandas.concat(dfList)

    # get spectroflux data
    # import pdb; pdb.set_trace()
    bossExpNums = list(set(dfGFA.bossExpNum))

    dfList = []
    for bossExpNum in bossExpNums:
        df = getBossFlux(mjd, bossExpNum)
        dfList.append(df)

    dfBoss = pandas.concat(dfList)

    # remove gfa exposures that don't match boss exps
    # dfGFA = dfGFA[dfGFA.bossExpNum.isin(list(set(dfBoss.bossExpNum)))]

    # get the fvc data
    fvcImgNums = list(set(dfConfSumm.fvcImgNum))
    dfList = [getFVCData(mjd, site, x) for x in fvcImgNums]

    dfFVC = pandas.concat(dfList)

    newDir = OUT_DIR + "/" + str(mjd)
    if not os.path.exists(newDir):
        os.mkdir(newDir)

    dfGFA.to_csv(newDir + "/dither_gfa_%i_%s.csv"%(mjd, site), index=False)
    dfConfSumm.to_csv(newDir + "/dither_confsumm_%i_%s.csv"%(mjd, site), index=False)
    dfBoss.to_csv(newDir + "/dither_boss_%i_%s.csv"%(mjd, site), index=False)
    dfFVC.to_csv(newDir + "/dither_fvc_%i_%s.csv"%(mjd, site), index=False)



def _fluxNormGFA(dfGFA, plot=False):
    # adds a few normalization columns for flux measured
    dfGFA = dfGFA[dfGFA.aperflux/dfGFA.aperfluxerr > 400]
    # convert to flux / sec in case of variable gfa exptimes
    dfGFA["aperflux"] = dfGFA.aperflux / dfGFA.gfaExptime
    fluxRate = dfGFA[["source_id", "aperflux"]].groupby("source_id").median().reset_index()
    dfGFA = dfGFA.merge(fluxRate, on="source_id", suffixes=(None, "_median"))
    dfGFA["fluxCoor"] = dfGFA.aperflux / dfGFA.aperflux_median
    if plot:
        plt.figure(figsize=(8,8))
        for idx, group in dfGFA.groupby("source_id"):
            group = group.sort_values("gfaImgNum")
            plt.plot(group.gfaImgNum, group.fluxCoor, '.k', alpha=0.5)
        plt.ylim([0.975,1.025])

    dfmed = dfGFA[["gfaImgNum", "fluxCoor"]].groupby("gfaImgNum").median().reset_index()
    dfGFA = dfGFA.merge(dfmed, on="gfaImgNum", suffixes=(None, "_median")).sort_values("gfaImgNum")

    if plot:
        plt.plot(dfGFA.gfaImgNum, dfGFA.fluxCoor_median, '-r')
        plt.show()

    return dfGFA


def computeWokCoords(site, mjd):
    # dfGFA = pandas.read_csv("dither_gfa_%i.csv"%mjd)
    newDir = OUT_DIR + "/" + str(mjd)
    dfGFA = pandas.read_csv(newDir + "/dither_gfa_%i_%s.csv"%(mjd, site))
    dfConfSumm = pandas.read_csv(newDir + "/dither_confsumm_%i_%s.csv"%(mjd, site))
    dfBoss = pandas.read_csv(newDir + "/dither_boss_%i_%s.csv"%(mjd, site))
    dfFVC = pandas.read_csv(newDir + "/dither_fvc_%i_%s.csv"%(mjd, site))

    dfGFA = _fluxNormGFA(dfGFA)
    # just need summary (eg header info) from each gfa exposure
    dfGFA = dfGFA.groupby(["mjd", "gfaImgNum"]).first().reset_index()

    # dfBoss = pandas.read_csv("dither_boss_%i.csv"%mjd)

    df = dfBoss.merge(dfGFA, on=["mjd", "bossExpNum"])
    # average gfa info over the boss exposure

    df = df.merge(dfConfSumm, on=["mjd", "configID", "fiberID"], suffixes=("_gfa", "_conf")).reset_index()
    # import pdb; pdb.set_trace()

    df = df.merge(dfFVC, on=["configID", "designID", "positionerID", "mjd", "fvcImgNum"], suffixes=("_gfa", "_fvc"))
    # df["fiberID"] = df.bossSpecID
    # for col in df.columns:
    #     print(col)

    dfList = []
    for camera in list(set(df.camera)):
        _df = df[df.camera == camera]
        if camera in ["b1", "b2"]:
            wl = "Boss"
        else:
            wl = "GFA"
        for name, group in _df.groupby(["mjd", "gfaImgNum"]):
            tobs = Time(group.gfaDateObs.iloc[0], scale="tai")
            tobs += TimeDelta(group.gfaExptime.iloc[0]/2, format="sec")
            ra = group.racat.to_numpy()
            dec = group.deccat.to_numpy()
            pmra = group.pmra_conf.to_numpy()
            pmdec = group.pmdec_conf.to_numpy()
            px = group.parallax_conf.to_numpy()
            coord_epoch = Time(group.coord_epoch.iloc[0], format="jyear", scale="tai")
            # import pdb; pdb.set_trace()
            raCen = group.SOL_RA.iloc[0]
            decCen = group.SOL_DEC.iloc[0]
            paCen = group.SOL_PA.iloc[0] + group.offpa.iloc[0]/3600.
            focScale = group.SOL_SCL.iloc[0]
            darLambda = group.lambdaCen.to_numpy()
            # darLambda = None

            xwok, ywok, fw, ha, pa = radec2wokxy(
                ra, dec, coord_epoch.jd, wl,
                raCen, decCen, paCen,
                site.upper(), tobs.jd, focScale,
                pmra, pmdec, px, darLambda=darLambda
            )
            group["xWokStarPredict"] = xwok
            group["yWokStarPredict"] = ywok

            raCen2 = group.field_cen_ra.iloc[0]
            decCen2 = group.field_cen_dec.iloc[0]
            paCen2 = group.field_cen_pa.iloc[0]

            xwok, ywok, fw, ha, pa = radec2wokxy(
                ra, dec, coord_epoch.jd, wl,
                raCen2, decCen2, paCen2,
                "APO", tobs.jd, focScale,
                pmra, pmdec, px, darLambda=darLambda
            )
            group["xWokStarPredictCen"] = xwok
            group["yWokStarPredictCen"] = ywok


            dfList.append(group)

    df = pandas.concat(dfList)

    df["dxWokStar"] = df.xWokStarPredict - df.xWokMeasBOSS
    df["dyWokStar"] = df.yWokStarPredict - df.yWokMeasBOSS
    df["drWokStar"] = numpy.sqrt(df.dxWokStar**2+df.dyWokStar**2)
    df = df[df.drWokStar < 1] # throw out stars more than 1mm from their fiber

    dfList = []
    for name, group in df.groupby(["camera", "configID", "fiberID"]):
        maxFlux = numpy.max(group.spectroflux)
        group["spectrofluxNorm"] = group.spectroflux / maxFlux
        dfList.append(group)

    df = pandas.concat(dfList)
    df["site"] = site

    df.to_csv(newDir + "/dither_merged_%i_%s"%(mjd, site), index=False)


    # plt.figure()
    # plt.hist(dfGFA.aperflux/dfGFA.aperfluxerr, bins=100)

    # import pdb; pdb.set_trace()


def fitOne(name):
    configID, fiberId, camera, mjd, site = name
    csvName = OUT_DIR + "/%i"%mjd +"/ditherFit_%i_%i_%s_%i_%s.csv"%name
    if os.path.exists(csvName):
        return
    print("---------\non %i %i %s\n------"%name)

    df = pandas.read_csv(OUT_DIR + "/%i/dither_merged_%i_%s.csv"%(mjd,mjd,site))
    group = df[(df.configID==name[0]) & (df.fiberID==name[1]) & (df.camera==name[2])]
    xStar = group.xWokStarPredict.to_numpy()
    yStar = group.yWokStarPredict.to_numpy()
    dxStar = group.dxWokStar.to_numpy()
    dyStar = group.dyWokStar.to_numpy()
    # xFiber = group.xWokMeasBOSS.to_numpy()
    # xFiber = group.yWokMeasBOSS.to_numpy()
    flux = group.spectrofluxNorm.to_numpy()
    sigma = numpy.median(numpy.sqrt(group.xstd**2+group.ystd**2)) * 13.5 / 1000 # in mm
    fitAmp, fitSigma, fitFiberX, fitFiberY = fitOneSet(xStar, yStar, flux, sigma, fitSigma=True)
    _plotOne(
        fitFiberX, fitFiberY, fitSigma, fitAmp,
        xStar, yStar, flux, group.mjd.iloc[0],
        group.fiberID.iloc[0], group.configID.iloc[0],
        group.r_mag.iloc[0], group.camera.iloc[0]
    )
    # for col in df.columns:
    #     print(col)
    group["xWokDitherFit"] = fitFiberX
    group["yWokDitherFit"] = fitFiberY
    group["sigmaWokDitherFit"] = fitSigma
    group["fluxAmpDitherFit"] = fitAmp
    group.to_csv(csvName, index=False)

    plotName = csvName.strip(".csv") + ".png"
    plt.savefig(plotName, dpi=200)
    plt.close("all")


def fitFiberCenters(site, mjd): #df):
    df = pandas.read_csv(OUT_DIR + "/%i/dither_merged_%i_%s.csv"%(mjd,mjd,site))
    groupNames = []
    for name, group in df.groupby(["configID", "fiberID", "camera", "mjd", "site"]):
        groupNames.append(name)

    # for name in groupNames:
    #     fitOne(name)

    p = Pool(CORES)
    p.map(fitOne, groupNames)


def plotDitherPSFs():
    # df = getGFATables(mjd)
    dfAll = pandas.read_csv("dither_gfa_%i.csv"%mjd)
    configIDs = list(set(dfAll.configID))
    for configID in configIDs:
        df = dfAll[dfAll.configID==configID]
        dff = df[["gfaNum", "fwhm"]].groupby(["gfaNum"]).median().reset_index().sort_values("fwhm")
        gfaNum = dff.gfaNum.to_numpy()[0]

        df = df[df.gfaNum==gfaNum]
        # plt.figure()
        # plt.hist(df.peak)

        # plt.figure()
        # plt.hist(df.flux)

        # plt.show()

        df = df[df.peak > 9000]
        df = df[df.peak < 40000]
        # df = df[df.configID==13797]
        # print("n imgs", len(set(df.imgNum)))
        dfg = df.groupby(["source_id"]).count().reset_index().sort_values("x") # could sort on any column
        source_id = dfg.source_id.to_numpy()[0]

        # source_id = 1042696945687328896 # in every image

        df = df[df.source_id==source_id]
        xCen = int(numpy.mean(df.x))
        yCen = int(numpy.mean(df.y))
        cutoutSize = 13 # must be odd

        dfg = df.groupby(["offra", "offdec"])
        ditherNum = 0
        for idx, group in dfg:
            imgStack = []
            group = group.sort_values("gfaImgNum")
            xm = numpy.mean(group.x) - (xCen-cutoutSize)
            ym = numpy.mean(group.y) - (yCen-cutoutSize)
            imgNum = 0
            for fpath in group.file_path.to_list():
                data = fits.open(fpath)[1].data
                cutout = data[yCen-cutoutSize:yCen+cutoutSize+1, xCen-cutoutSize:xCen+cutoutSize+1]
                imgStack.append(cutout)
                plt.figure()
                plt.imshow(cutout, origin="lower")
                plotFiberCircle(xm,ym)
                plt.title("config %i dither %i: "%(configID, ditherNum) + str(idx) + " imgNum: %i"%(imgNum))
                plt.savefig("dither_%i_%i_img_%i.png"%(configID, ditherNum, imgNum), dpi=100)
                imgNum += 1


            imgStack = numpy.sum(imgStack, axis=0)
            plt.figure()
            plt.imshow(imgStack, origin="lower")
            plotFiberCircle(xm,ym)
            # plt.show()
            plt.title("config %i dither %i: "%(configID, ditherNum) + str(idx) + " summed")
            plt.savefig("dither_%i_%i_sum.png"%(configID, ditherNum), dpi=100)
            ditherNum += 1
            plt.close("all")
        # import pdb; pdb.set_trace()


def plot_zps():
    df = pandas.read_csv("allGFAMatches.csv")
    df = df[["gfaNum", "taiMid", "gfaImgNum", "zp"]]
    df = df.groupby(["gfaImgNum", "gfaNum"]).median().reset_index()
    plt.figure(figsize=(8,8))
    sns.lineplot(x="taiMid", y="zp", hue="gfaNum", data=df)
    plt.show()


def plotFiberCircle(xCen, yCen):
    pixelSize = 13.5 # micron
    radius = 60/pixelSize # micron
    thetas = numpy.linspace(0, numpy.pi*2,200)
    xs = radius * numpy.cos(thetas) + xCen
    ys = radius * numpy.sin(thetas) + yCen
    plt.plot(xs,ys,'-r', lw=1)

def plotFluxScatter(df):
    for name, group in df.groupby(["camera", "configID"]):
        # just keep fiber with most flux?
        # plt.figure()
        # plt.hist(df.spectrofluxNorm, bins=100)

        # for fiberID in list(set(group.fiberID)):
        #     plt.figure(figsize=(8,8))
        #     sns.scatterplot(x="dxWokFlux", y="dyWokFlux", hue="spectrofluxNorm", s=12, data=group[group.fiberID==fiberID])
        #     plt.axis("equal")
        #     plt.savefig("fiber_%i.png"%fiberID, dpi=200)
        #     plt.close("all")

        plt.figure(figsize=(6,6))
        sns.scatterplot(x="dxWokFlux", y="dyWokFlux", hue="spectrofluxNorm", s=2, data=group)
        plt.axhline(0, ls=":")
        plt.axvline(0, ls=":")
        plt.title("camera: %s  config: %i"%(name[0], name[1]))
        plt.axis("equal")

        # x,y,dx,dy = group[["xWokMeasBOSS", "yWokMeasBOSS", "dxWokFlux", "dyWokFlux"]].to_numpy().T
        # plt.figure(figsize=(8,8))
        # plt.quiver(x,y,dx,dy,angles="xy",units="xy", scale=0.01)
        # plt.axis("equal")

        # break

    plt.show()
    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    # plotDitherPSFs()
    # plot_zps()
    getDitherTables(mjd)
    # dfOut = computeWokCoords(
    #     pandas.read_csv("dither_gfa_%i.csv"%mjd),
    #     pandas.read_csv("dither_boss_%i.csv"%mjd),
    #     pandas.read_csv("dither_fvc_%i.csv"%mjd),
    #     pandas.read_csv("dither_confsumm_%i.csv"%mjd)
    # )
    # dfOut.to_csv("dither_merged.csv", index=False)
    # dfOut = pandas.read_csv("dither_merged.csv")
    # fitFiberCenters(dfOut)
    # plotFluxScatter(dfOut)



