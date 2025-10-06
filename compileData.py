from astropy.io import fits
import glob
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import numpy
from coordio.utils import radec2wokxy
from coordio.guide import SolvePointing
from coordio.transforms import FVCTransformAPO, FVCTransformLCO
from coordio.defaults import calibration
import os
import socket
from multiprocessing import Pool
import sys
from functools import partial
import seaborn as sns

# from parseConfSummary import parseConfSummary
from findFiberCenter import fitOneSet, _plotOne, fractionalFlux

# mjd = 60420
import warnings
warnings.filterwarnings("ignore", message="Warning! Coordinate far off telescope optical axis conversion may be bogus")

POLIDS=numpy.array([0, 1, 2, 3, 4, 5, 6, 9, 20, 27, 28, 29, 30])
RMAX = 310
CENTTYPE = "flex"

_hostname = socket.gethostname()

if "conor" in _hostname.lower():
    LOCATION = "local"
    OUT_DIR = os.getcwd()
    CORES = 10
elif "apogee" or "manga" in _hostname:
    LOCATION = "utah"
    OUT_DIR = "/uufs/chpc.utah.edu/common/home/u0449727/work/ditherAnalysis"
    CORES = 28
    gaia_connection_string = "postgresql://sdss_user@operations.sdss.org/sdss5db"
    gaia_connection_table = "catalogdb.gaia_dr2_source"
elif "sdss5" in _hostname:
    LOCATION = "mountain"
    OUT_DIR = os.getcwd()
    CORES = 1
elif "mako" == _hostname:
    LOCATION = "mako"
    OUT_DIR = os.getcwd()
    CORES = 26
    gaia_connection_string = "postgresql://sdss@localhost:5433/sdss5db"
    gaia_connection_table = "catalogdb.gaia_dr2_source_g19_2"
else:
    raise RuntimeError("unrecoginzed computer, don't know where data is")


def getGFAFiles(mjd, site, imgNum=None, location=LOCATION):
    site = site.lower()
    if imgNum is None:
        imgNumStr = ""
    else:
        imgNumStr = "-" + str(imgNum).zfill(4)
    if location == "local":
        glbstr = "/Volumes/futa/%s/data/gcam/%i/proc*%s.fits"%(site,mjd,imgNumStr)
    elif location == "mountain":
        glbstr = glbstr = "/data/gcam/%i/proc*%s.fits"%(mjd,imgNumStr)
    elif location == "mako":
        glbstr = "/data/gcam/%s/%i/proc*%s.fits"%(site,mjd, imgNumStr)
    else:
        # utah
        glbstr = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/%s/%i/proc*%s.fits"%(site,mjd,imgNumStr)

    return glob.glob(glbstr)


def getBOSSPath(mjd, site, location=LOCATION):
    site = site.lower()
    if location == "local":
        bossPath = "/Volumes/futa/%s/data/boss/sos/%i/dither"%(site, mjd)
    elif location == "mountain":
        bossPath = "/data/boss/sos/%i/dither"%(mjd)
    elif location == "mako":
        bossPath = "/data/boss/sos/%s/%i/dither"%(site,mjd)
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
        # fvcPath = "/Volumes/futa/%s/data/fcam/%i/proc-fimg-%s-%s.fits"%(site, mjd, camname, imgNumStr)
        subdir = ""
        if site == "lco":
            subdir = "/lco/"
        fvcPath = "/Users/csayres/Downloads/fcam%s/%i/proc-fimg-%s-%s.fits"%(subdir, mjd, camname, imgNumStr)
    elif location == "mountain":
        fvcPath = "/data/fcam/%i/proc-fimg-%s-%s.fits"%(mjd, camname, imgNumStr)
    elif location == "mako":
        fvcPath = "/data/fcam/%s/%i/proc-fimg-%s-%s.fits"%(site, mjd, camname, imgNumStr)
    else:
        # utah
        fvcPath = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam/%s/%i/proc-fimg-%s-%s.fits"%(site, mjd, camname, imgNumStr)

    return fvcPath


def getConfSummPath(configID, site, location=LOCATION):
    site = site.lower()

    confStr = str(configID).zfill(6)[:-2] + "XX"
    confStr2 = str(configID).zfill(6)[:-3] + "XXX"

    if location == "local":
        confPath = "confSummaryF-%i.par"%configID
    elif location == "mountain":
        confPath = "/home/sdss5/software/sdsscore/main/%s/summary_files/%s/confSummaryF-%i.par"%(site, confStr, configID)
    elif location == "mako":
        confPath = "confSummaryFiles/confSummaryF-%i.par"%configID
    else:
        # utah
        confPath = "/uufs/chpc.utah.edu/common/home/sdss50/software/git/sdss/sdsscore/main/%s/summary_files/%s/%s/confSummaryFS-%i.parquet"%(site, confStr2, confStr, configID)

    return confPath

def parseConfSummary(confFilePath):

    df = pandas.read_parquet(confFilePath)
    fvcImgNum = df.fvc_image_path.iloc[0]
    fvcImgNum = int(fvcImgNum.split("-")[-1].strip(".fits"))


    df["configID"] = df.configuration_id
    df["designID"] = df.design_id
    df["mjd"] = df.MJD
    df["fvcImgNum"] = fvcImgNum
    df["field_cen_ra"] = df.raCen
    df["field_cen_dec"] = df.decCen
    df["field_cen_pa"] = df.pa
    df["fiberID"] = df.fiberId
    df["positionerID"] = df.positionerId
    df["holeID"] = df.holeId
    df = df[df.firstcarton == "manual_fps_position_stars_10"].reset_index(drop=True)

    u_mag = []
    g_mag = []
    r_mag = []
    i_mag = []
    z_mag = []
    for idx, row in df.iterrows():
        u_mag.append(row.mag[0])
        g_mag.append(row.mag[1])
        r_mag.append(row.mag[2])
        i_mag.append(row.mag[3])
        z_mag.append(row.mag[4])
    df["u_mag"] = u_mag
    df["g_mag"] = g_mag
    df["r_mag"] = r_mag
    df["i_mag"] = i_mag
    df["z_mag"] = z_mag


    return df

def procOneGFA(imgNum, mjd, site):
    imgs = getGFAFiles(mjd,site,imgNum)
    sp = None
    for img in imgs:
        ff = fits.open(img)
        if sp is None:
            sp = SolvePointing(
                raCen=ff[1].header["RAFIELD"],
                decCen=ff[1].header["DECFIELD"],
                paCen=ff[1].header["FIELDPA"],
                offset_ra=ff[1].header["AOFFRA"],
                offset_dec=ff[1].header["AOFFDEC"],
                offset_pa=ff[1].header["AOFFPA"],
                db_conn_st=gaia_connection_string,
                db_tab_name=gaia_connection_table
            )
        wcs = None
        if ff[1].header["SOLVED"]:
            wcs = WCS(ff[1].header)
            # field could be solved but not every
            # gfa has a wcs
            if "CTYPE1" not in wcs.to_header():
                wcs = None
        sp.add_gimg(
            img,
            Table(ff["CENTROIDS"].data).to_pandas(),
            wcs,
            ff[1].header["GAIN"]
        )
        ff.close()

    # print("n wcs", len(sp.gfaWCS))
    if len(sp.gfaWCS) < 2:
        # skip need at least 2 wcs solns for
        # coordio solve
        print("skipping gimg with < 2 wcs solns", imgNum)
        return None
    try:
        sp.solve()
    except Exception as e:
        print("skipping gimg (solve failed)", site, mjd, imgNum)
        print(e)
        return None
    dfList = []
    for img in imgs:
        ff = fits.open(img)
        toks = img.split("-")
        offra = ff[1].header["OFFRA"]
        offdec = ff[1].header["OFFDEC"]
        imgNum = int(toks[-1].strip(".fits"))

        offpa = ff[1].header["AOFFPA"]

        if site == "apo":
            gfaNum = int(toks[-2].strip("gfa").strip("n"))
        else:
            gfaNum = int(toks[-2].strip("gfa").strip("s"))

        # table has zps for all gfas
        t = sp.matchedSources[sp.matchedSources.gfaNum == gfaNum].reset_index(drop=True)
        t["gfaImgNum"] = imgNum
        t["configID"] = ff[1].header["CONFIGID"]
        t["offra"] = offra
        t["offdec"] = offdec
        t["offpa"] = offpa
        t["bossExpNum"] = -999
        t["file_path"] = img
        t["gfaDateObs"] = ff[1].header["DATE-OBS"]
        t["gfaExptime"] = ff[1].header["EXPTIMEN"]
        t["guideFWHM"] = ff[1].header["FWHM"]


        t["taiMid"] = sp.obsTimeRef.mjd * 24 * 60 * 60
        t["SOL_RA"] = sp.raCenMeas
        t["SOL_DEC"] = sp.decCenMeas
        t["SOL_PA"] = sp.paCenMeas
        t["SOL_SCL"] = sp.scaleMeas
        t["SOL_ALT"] = sp.altCenMeas
        t["SOL_AZ"] = sp.azCenMeas
        t["SOLVMODE"] = "coordio-reproc"

        t["guideErrRA"] = sp.delta_ra
        t["guideErrDec"] = sp.delta_dec
        t["guideErrRot"] = sp.delta_rot
        t["guideRMS"] = sp.guide_rms
        t["guideFitRMS"] = sp.fit_rms

        # t["FWHM_FIT"] = ff[1].header["FWHM_FIT"]
        ff.close()
        dfList.append(t)

    return pandas.concat(dfList)


def getGFATables(mjd, site, reprocess=False):
    site = site.lower()
    files = getGFAFiles(mjd, site)

    reprocSet = set()

    dfList = []
    for f in files:
        ff = fits.open(f)
        toks = f.split("-")
        offra = ff[1].header["OFFRA"]
        offdec = ff[1].header["OFFDEC"]
        imgNum = int(toks[-1].strip(".fits"))

        if offra == 0 and offdec == 0:
            continue

        if reprocess or "coordio" not in ff[1].header["SOLVMODE"]:
            reprocSet.add(imgNum)
            continue


        offpa = ff[1].header["AOFFPA"]
        bossExp = ff[1].header["SPIMGNO"]

        if site == "apo":
            gfaNum = int(toks[-2].strip("gfa").strip("n"))
            bossExp = bossExp + 1
        else:
            gfaNum = int(toks[-2].strip("gfa").strip("s"))


        t = Table(ff["GAIAMATCH"].data).to_pandas()
        # table has zps for all gfas
        t = t[t.gfaNum == gfaNum].reset_index(drop=True)
        t["gfaImgNum"] = imgNum
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

        t["guideErrRA"] = ff[1].header["DELTARA"]
        t["guideErrDec"] = ff[1].header["DELTADEC"]
        t["guideErrRot"] = ff[1].header["DELTAROT"]
        t["guideRMS"] = ff[1].header["SOL_GRMS"]
        t["guideFitRMS"] = ff[1].header["SOL_FRMS"]
        t["guideFWHM"] = ff[1].header["FWHM"]

        dfList.append(t)
        ff.close()

    # drop camera column it's reduntant with gfaNum
    # and conflicts with boss camera columns

    # dfGFA.to_csv("dither_gfa_%s_%i.csv"%(site,mjd))
    print("reprocessing gimgs", reprocSet)
    # for imgNum in list(reprocSet):
    #     dfList.append(procOneGFA(imgNum, mjd, site))

    p = Pool(CORES)
    fo = partial(procOneGFA, mjd=mjd, site=site)
    _dfList = p.map(fo, list(reprocSet))
    # remove None's
    dfList.extend([x for x in _dfList if x is not None])

        # imgs = getGFAFiles(mjd,site,imgNum)
        # sp = None
        # for img in imgs:
        #     ff = fits.open(img)
        #     if sp is None:
        #         sp = SolvePointing(
        #             raCen=ff[1].header["RAFIELD"],
        #             decCen=ff[1].header["DECFIELD"],
        #             paCen=ff[1].header["FIELDPA"],
        #             offset_ra=ff[1].header["AOFFRA"],
        #             offset_dec=ff[1].header["AOFFDEC"],
        #             offset_pa=ff[1].header["AOFFPA"],
        #             db_conn_st=gaia_connection_string,
        #             db_tab_name=gaia_connection_table
        #         )
        #     wcs = None
        #     if ff[1].header["SOLVED"]:
        #         wcs = WCS(ff[1].header)
        #         # field could be solved but not every
        #         # gfa has a wcs
        #         if "CTYPE1" not in wcs.to_header():
        #             wcs = None
        #     sp.add_gimg(
        #         img,
        #         Table(ff["CENTROIDS"].data).to_pandas(),
        #         wcs,
        #         ff[1].header["GAIN"]
        #     )
        #     ff.close()

        # print("n wcs", len(sp.gfaWCS))
        # if len(sp.gfaWCS) < 2:
        #     # skip need at least 2 wcs solns for
        #     # coordio solve
        #     print("skipping gimg with < 2 wcs solns", imgNum)
        #     continue
        # try:
        #     sp.solve()
        # except Exception as e:
        #     print("skipping gimg (solve failed)", site, mjd, imgNum)
        #     print(e)
        #     continue
        # for img in imgs:
        #     ff = fits.open(img)
        #     toks = img.split("-")
        #     offra = ff[1].header["OFFRA"]
        #     offdec = ff[1].header["OFFDEC"]
        #     imgNum = int(toks[-1].strip(".fits"))

        #     offpa = ff[1].header["AOFFPA"]

        #     if site == "apo":
        #         gfaNum = int(toks[-2].strip("gfa").strip("n"))
        #     else:
        #         gfaNum = int(toks[-2].strip("gfa").strip("s"))

        #     # table has zps for all gfas
        #     t = sp.matchedSources[sp.matchedSources.gfaNum == gfaNum].reset_index(drop=True)
        #     t["gfaImgNum"] = imgNum
        #     t["configID"] = ff[1].header["CONFIGID"]
        #     t["offra"] = offra
        #     t["offdec"] = offdec
        #     t["offpa"] = offpa
        #     t["bossExpNum"] = -999
        #     t["file_path"] = img
        #     t["gfaDateObs"] = ff[1].header["DATE-OBS"]
        #     t["gfaExptime"] = ff[1].header["EXPTIMEN"]
        #     t["guideFWHM"] = ff[1].header["FWHM"]


        #     t["taiMid"] = sp.obsTimeRef.mjd * 24 * 60 * 60
        #     t["SOL_RA"] = sp.raCenMeas
        #     t["SOL_DEC"] = sp.decCenMeas
        #     t["SOL_PA"] = sp.paCenMeas
        #     t["SOL_SCL"] = sp.scaleMeas
        #     t["SOL_ALT"] = sp.altCenMeas
        #     t["SOL_AZ"] = sp.azCenMeas
        #     t["SOLVMODE"] = "coordio-reproc"

        #     t["guideErrRA"] = sp.delta_ra
        #     t["guideErrDec"] = sp.delta_dec
        #     t["guideErrRot"] = sp.delta_rot
        #     t["guideRMS"] = sp.guide_rms
        #     t["guideFitRMS"] = sp.fit_rms

        #     # t["FWHM_FIT"] = ff[1].header["FWHM_FIT"]
        #     dfList.append(t)
        #     ff.close()

            # print(wcs)
            # import pdb; pdb.set_trace()

    dfGFA = pandas.concat(dfList)
    dfGFA = dfGFA.drop("camera", axis=1)
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
            df["bossExptime"] = df.exptime
            df["bossExpMid"] = Time(df.mjd_obs, format="mjd", scale="tai").mjd * 24 * 60 * 60
            df["bossExpStart"] = df.bossExpMid - df.exptime/2
            df["bossExpEnd"] = df.bossExpMid + df.exptime/2
            df["configID"] = ff[0].header["CONFIGID"]
            keepCols = ["configID", "bossExpNum", "fiberID", "mjd", "lambdaCen", "camera", "spectroflux", "spectroflux_ivar", "objtype", "bossExptime", "bossExpStart", "bossExpEnd"]
            df = df[keepCols]
            ff.close()
        dfList.append(df)

    return pandas.concat(dfList)


def getFVCData(mjd, site, expNum, reprocess=False):
    imgPath = getFVCPath(mjd, site, expNum)
    ff = fits.open(imgPath)

    # olddark = numpy.array(fits.open("olddark.fits")[1].data, dtype=float)[:,::-1]
    # newdark = numpy.array(fits.open("newdark.fits")[1].data, dtype=float)[:,::-1]

    if reprocess:
        print("reprocessing fvc image", imgPath)
        pt = calibration.positionerTable.reset_index()
        wc = calibration.wokCoords.reset_index()
        fc = calibration.fiducialCoords.reset_index()
        if site.lower()=="lco":
            _fvct = FVCTransformLCO
            pt = pt[pt.site=="LCO"]
            wc = wc[wc.site=="LCO"]
            fc = fc[fc.site=="LCO"]
        else:
            _fvct = FVCTransformAPO
            pt = pt[pt.site=="APO"]
            wc = wc[wc.site=="APO"]
            fc = fc[fc.site=="APO"]

        fvct = _fvct(
            ff[1].data, #+ olddark - newdark,
            Table(ff["POSANGLES"].data).to_pandas(),
            ff[1].header["IPA"],
            positionerTable=pt,
            wokCoords=wc,
            fiducialCoords=fc,
        )
        fvct.extractCentroids()
        fvct.fit(centType=CENTTYPE)
        ptm = fvct.positionerTableMeas.copy()
        fcm = fvct.fiducialCoordsMeas.copy()
        for col in ["level_0", "index"]:
            ptm.drop(col, axis=1, inplace=True)
            fcm.drop(col, axis=1, inplace=True)

    else:
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
    if site.lower() == "apo":
        ptm["fvcALT"] = ff[1].header["ALT"]
        ptm["fvcAZ"] = ff[1].header["AZ"]

    # import pdb; pdb.set_trace()
    ff.close()
    return ptm


def getDitherTables(mjd, site, reprocGFA=False, reprocFVC=False):
    site = site.lower()
    dfGFA = getGFATables(mjd, site, reprocGFA)

    configIDs = list(set(dfGFA.configID))
    dfList = []

    # get confsummary data
    noConfF = []
    for configID in configIDs:
        try:
            confpath = getConfSummPath(configID, site)
            dfList.append(parseConfSummary(confpath))
        except:
            noConfF.append(configID)


    # remove GFA images associated with a configuration
    # without a conf-summaryF file
    dfGFA = dfGFA[~dfGFA.configID.isin(noConfF)]

    dfConfSumm = pandas.concat(dfList)

    # get spectroflux data
    # get all boss data for this mjd
    # then filter out exposures that don't belong
    bossPath = getBOSSPath(mjd, site)
    allExps = glob.glob(bossPath + "/ditherBOSS*.fits")
    bossExpNums = set()
    for exp in allExps:
        expNum = int(exp.split("-")[1])
        bossExpNums.add(expNum)
    # import pdb; pdb.set_trace()
    bossExpNums = list(bossExpNums)

    dfList = []
    for bossExpNum in bossExpNums:
        df = getBossFlux(mjd, site, bossExpNum)
        dfList.append(df)

    dfBoss = pandas.concat(dfList)
    print("len boss before", len(dfBoss))
    # find the configs that were dithered
    dfBoss = dfBoss[dfBoss.configID.isin(configIDs)].reset_index(drop=True)
    print("len boss after", len(dfBoss))


    # now only keep GFA exposures taken during a boss exposure
    # forget about the boss image number from the header
    # need to double check that this is valid for LCO as well
    dfList = []
    for gfaImgNum, gfaGroup in dfGFA.groupby("gfaImgNum"):
        # gfaGroup["bossExpNum"] = -999
        for bossExpNum, bossGroup in dfBoss.groupby("bossExpNum"):
            gfaMid = numpy.mean(gfaGroup.taiMid.to_numpy())
            bossStart = numpy.mean(bossGroup.bossExpStart.to_numpy())
            bossEnd = numpy.mean(bossGroup.bossExpEnd.to_numpy())
            # bossStart -= 60 # this is found as a descrepancy between ditherBOSS mjd_obs column and DATE-OBS in the raw spectro header
            # bossEnd -= 60
            _bossExpNum = gfaGroup.bossExpNum.iloc[0]
            # if bossExpNum == _bossExpNum:
            #     print("mjd", mjd, bossExpNum, gfaGroup.gfaImgNum.iloc[0], gfaGroup.gfaDateObs.iloc[0])
            #     print(bossExpNum, bossStart, gfaMid-bossStart)
            #     import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
            if gfaMid > bossStart and gfaMid < bossEnd:
                # warning over writing bossExpNum
                # as reported in gfa image header
                # instead relying on time-matching
                gfaGroup["bossExpNum"] = bossExpNum
                dfList.append(gfaGroup)
                # print("matched", gfaImgNum, bossExpNum, _bossExpNum)

    # # print("len gfa before", len(dfGFA))
    dfGFA = pandas.concat(dfList)

    # calcuate a flux normalization factor to apply
    # to each boss exposure in dither sequence
    dfList = []
    for name, group in dfGFA.groupby("configID"):
        group = group.copy().reset_index()
        # only keep decent detections
        group = group[group.peak > 2000]
        group = group[group.peak < 50000]
        group["aperflux"] = group.aperflux / group.gfaExptime
        _dfm = group[["source_id", "aperflux"]].groupby("source_id").median().reset_index()
        group = group.merge(_dfm, on="source_id", suffixes=(None, "_m"))
        group["fluxcorr"] = group.aperflux / group.aperflux_m # per source-gfa-exposure
        # next average over individual boss exposures
        _df = group[["bossExpNum", "fluxcorr"]]
        _dfm = _df.groupby("bossExpNum").mean().reset_index()
        _dfsig = _df.groupby("bossExpNum").std().reset_index()
        _dfm = _dfm.merge(_dfsig, on="bossExpNum", suffixes=("_mean", "_sigma"))
        group = group.merge(_dfm, on="bossExpNum")
        dfList.append(group)

    dfGFA = pandas.concat(dfList)



    # bossExpNums = list(set(dfBoss.bossExpNum))
    # dfGFA = dfGFA[dfGFA.bossExpNum.isin(bossExpNums)]

    # import pdb; pdb.set_trace()
    # print("len gfa after", len(dfGFA))

    # import pdb; pdb.set_trace()

    # remove gfa exposures that don't match boss exps
    # dfGFA = dfGFA[dfGFA.bossExpNum.isin(list(set(dfBoss.bossExpNum)))]

    # get the fvc data
    fvcImgNums = list(set(dfConfSumm.fvcImgNum))
    # next find additional "no apply fvc images where robots werent moved"
    # after the fvc loop additional fvc loop --no-apply --no-write summaries are taken
    fvcImgNumNoApply = []
    for fvcImgNum in fvcImgNums:
        _cid = fits.open(getFVCPath(mjd, site, fvcImgNum))[1].header["CONFIGID"]
        ii = 0
        while True:
            ii += 1
            nextImgNum = fvcImgNum + ii
            try:
                _cid2 = fits.open(getFVCPath(mjd, site, nextImgNum))[1].header["CONFIGID"]
            except:
                break
            if _cid2 != _cid:
                break
            fvcImgNumNoApply.append(nextImgNum)

    dfConfSumm = dfConfSumm.drop("fvcImgNum", axis=1)

    dfList = [getFVCData(mjd, site, x, reprocFVC) for x in sorted(fvcImgNums + fvcImgNumNoApply)]



    dfFVC = pandas.concat(dfList)

    newDir = OUT_DIR + "/" + str(mjd)
    if not os.path.exists(newDir):
        os.mkdir(newDir)

    dfGFA.to_csv(newDir + "/dither_gfa_%i_%s.csv"%(mjd, site), index=False)
    dfConfSumm.to_csv(newDir + "/dither_confsumm_%i_%s.csv"%(mjd, site), index=False)
    dfBoss.to_csv(newDir + "/dither_boss_%i_%s.csv"%(mjd, site), index=False)
    dfFVC.to_csv(newDir + "/dither_fvc_%i_%s.csv"%(mjd, site), index=False)


# def _fluxNormGFA(dfGFA, plot=False):
#     # adds a few normalization columns for flux measured
#     dfGFA = dfGFA[dfGFA.aperflux/dfGFA.aperfluxerr > 400].reset_index(drop=True)
#     # convert to flux / sec in case of variable gfa exptimes
#     dfGFA["aperflux"] = dfGFA.aperflux / dfGFA.gfaExptime
#     fluxRate = dfGFA[["source_id", "aperflux"]].groupby("source_id").median().reset_index()
#     dfGFA = dfGFA.merge(fluxRate, on="source_id", suffixes=(None, "_median"))
#     dfGFA["fluxCoor"] = dfGFA.aperflux / dfGFA.aperflux_median
#     if plot:
#         plt.figure(figsize=(8,8))
#         for idx, group in dfGFA.groupby("source_id"):
#             group = group.sort_values("gfaImgNum")
#             plt.plot(group.gfaImgNum, group.fluxCoor, '.k', alpha=0.5)
#         plt.ylim([0.975,1.025])

#     dfmed = dfGFA[["gfaImgNum", "fluxCoor"]].groupby("gfaImgNum").median().reset_index()
#     dfGFA = dfGFA.merge(dfmed, on="gfaImgNum", suffixes=(None, "_median")).sort_values("gfaImgNum")

#     if plot:
#         plt.plot(dfGFA.gfaImgNum, dfGFA.fluxCoor_median, '-r')
#         plt.show()

#     return dfGFA


def computeWokCoords(mjd, site):
    # dfGFA = pandas.read_csv("dither_gfa_%i.csv"%mjd)
    newDir = OUT_DIR + "/" + str(mjd)
    dfGFA = pandas.read_csv(newDir + "/dither_gfa_%i_%s.csv"%(mjd, site))
    dfConfSumm = pandas.read_csv(newDir + "/dither_confsumm_%i_%s.csv"%(mjd, site))
    dfBoss = pandas.read_csv(newDir + "/dither_boss_%i_%s.csv"%(mjd, site))
    dfFVC = pandas.read_csv(newDir + "/dither_fvc_%i_%s.csv"%(mjd, site))

    # dfGFA = _fluxNormGFA(dfGFA)
    # just need summary (eg header info) from each gfa exposure
    dfGFA = dfGFA.groupby(["mjd", "gfaImgNum"]).first().reset_index()

    # dfBoss = pandas.read_csv("dither_boss_%i.csv"%mjd)

    df = dfBoss.merge(dfGFA, on=["mjd", "bossExpNum", "configID"])
    # average gfa info over the boss exposure

    df = df.merge(dfConfSumm, on=["mjd", "configID", "fiberID"], suffixes=("_gfa", "_conf")).reset_index()
    # import pdb; pdb.set_trace()

    # df = df.merge(dfFVC, on=["configID", "designID", "positionerID", "mjd", "fvcImgNum"], suffixes=("_gfa", "_fvc"))
    df = df.merge(dfFVC, on=["configID", "designID", "positionerID", "mjd"], suffixes=("_gfa", "_fvc"))
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
            paCen2 = group.field_cen_pa.iloc[0] + group.offpa.iloc[0]/3600.

            xwok, ywok, fw, ha, pa = radec2wokxy(
                ra, dec, coord_epoch.jd, wl,
                raCen2, decCen2, paCen2,
                site.upper(), tobs.jd, focScale,
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

    df.to_csv(newDir + "/dither_merged_%i_%s.csv"%(mjd, site), index=False)


    # plt.figure()
    # plt.hist(dfGFA.aperflux/dfGFA.aperfluxerr, bins=100)

    # import pdb; pdb.set_trace()


def fitOne(name, reprocess=False):
    configID, fiberId, camera, mjd, site = name
    csvName = OUT_DIR + "/%i"%mjd +"/ditherFit_%i_%i_%s_%i_%s.csv"%name
    if os.path.exists(csvName) and not reprocess:
        return
    print("---------\non %i %i %s\n------"%(configID, fiberId, camera))

    group = pandas.read_csv(OUT_DIR + "/%i/dither_merged_%i_%s.csv"%(mjd,mjd,site))
    group = group[(group.configID==name[0]) & (group.fiberID==name[1]) & (group.camera==name[2])].reset_index(drop=True)

    group["spectroflux_corr"] = group.spectroflux * group.fluxcorr_mean
    group = group.sort_values("spectroflux_corr", ascending=False)


    # set initial guess for sigma based on sigma measured from guider psfs
    sigmaInit = numpy.median(numpy.sqrt(group.xstd**2+group.ystd**2)) * 13.5 / 1000 # (guider informed) mm

    # set initial guesses for fitter, use fiber position and flux amplitude
    # at brightest flux respose
    _maxFlux = group[group.bossExpNum==group.bossExpNum.iloc[0]][["spectroflux_corr", "xWokStarPredict", "yWokStarPredict"]].mean()  # sorted by high to low flux
    xInit = numpy.array([_maxFlux.spectroflux_corr, sigmaInit, _maxFlux.xWokStarPredict, _maxFlux.yWokStarPredict])

    fitAmp, fitSigma, fitFiberX, fitFiberY, fitSuccess = fitOneSet(
        xInit, group.xWokStarPredict, group.yWokStarPredict,
        group.spectroflux_corr, flux_ivar=group.spectroflux_ivar
    )
    # calculate flux model at each gfa image
    fluxModel = []
    for xstar, ystar in group[["xWokStarPredict", "yWokStarPredict"]].to_numpy():
        fluxModel.append(fractionalFlux(fitAmp, fitSigma, xstar, ystar, fitFiberX, fitFiberY))

    group["fluxAmpDitherFit"] = fitAmp
    group["sigmaWokDitherFit"] = fitSigma
    group["xWokDitherFit"] = fitFiberX
    group["yWokDitherFit"] = fitFiberY
    group["ditherFitSuccess"] = fitSuccess
    group["fluxModelDitherFit"] = fluxModel

    group.drop('level_0', axis=1, inplace=True)
    # next perform leave one out cross verification, throwing out
    # one boss exposure and refitting, use previous solution as
    # initial guess to speed up fitter

    xInit = numpy.array([fitAmp, fitSigma, fitFiberX, fitFiberY])

    bossExps = list(set(group.bossExpNum))
    dfList = []
    for bossExp in bossExps:
        _outgroup = group[group.bossExpNum==bossExp].copy().reset_index()
        _ingroup = group[group.bossExpNum!=bossExp]

        _fitAmp, _fitSigma, _fitFiberX, _fitFiberY, _fitSuccess = fitOneSet(
            xInit, _ingroup.xWokStarPredict, _ingroup.yWokStarPredict, _ingroup.spectroflux, flux_ivar=_ingroup.spectroflux_ivar
        )
        # calculate flux model at each gfa image
        fluxModel = []
        for xstar, ystar in _outgroup[["xWokStarPredict", "yWokStarPredict"]].to_numpy():
            fluxModel.append(fractionalFlux(_fitAmp, _fitSigma, xstar, ystar, _fitFiberX, _fitFiberY))

        _outgroup["fluxAmpDitherFit_loo"] = _fitAmp
        _outgroup["sigmaWokDitherFit_loo"] = _fitSigma
        _outgroup["xWokDitherFit_loo"] = _fitFiberX
        _outgroup["yWokDitherFit_loo"] = _fitFiberY
        _outgroup["ditherFitSuccess_loo"] = _fitSuccess
        _outgroup["fluxModelDitherFit_loo"] = fluxModel


        dfList.append(_outgroup)

    group = pandas.concat(dfList)

    # import pdb; pdb.set_trace()
    # xStar = group.xWokStarPredict.to_numpy()
    # yStar = group.yWokStarPredict.to_numpy()
    # dxStar = group.dxWokStar.to_numpy()
    # dyStar = group.dyWokStar.to_numpy()
    # # xFiber = group.xWokMeasBOSS.to_numpy()
    # # xFiber = group.yWokMeasBOSS.to_numpy()
    # flux = group.spectroflux.to_numpy()
    # sigma = numpy.median(numpy.sqrt(group.xstd**2+group.ystd**2)) * 13.5 / 1000 # in mm
    # fitAmp, fitSigma, fitFiberX, fitFiberY = fitOneSet(xStar, yStar, flux, sigma, fitSigma=True)
    camera = group.camera.iloc[0]
    if "b" in camera:
        magStr = "     g mag = %.2f"%(group.g_mag.iloc[0])
    else:
        magStr = "     r mag = %.2f"%(group.r_mag.iloc[0])

    _plotOne(
        fitFiberX, fitFiberY, fitSigma, fitAmp,
        group.xWokStarPredict, group.yWokStarPredict, group.spectroflux_corr, group.mjd.iloc[0],
        group.fiberID.iloc[0], group.configID.iloc[0],
        magStr, group.camera.iloc[0]
    )
    # for col in df.columns:
    #     print(col)
    # group["xWokDitherFit"] = fitFiberX
    # group["yWokDitherFit"] = fitFiberY
    # group["sigmaWokDitherFit"] = fitSigma
    # group["fluxAmpDitherFit"] = fitAmp
    group.to_csv(csvName, index=False)

    plotName = csvName.strip(".csv") + ".png"
    plt.savefig(plotName, dpi=200)
    plt.close("all")


def fitFiberCenters(mjd, site, reprocess=False): #df):
    df = pandas.read_csv(OUT_DIR + "/%i/dither_merged_%i_%s.csv"%(mjd,mjd,site))
    groupNames = []
    for name, group in df.groupby(["configID", "fiberID", "camera", "mjd", "site"]):
        groupNames.append(name)

    # for name in groupNames:
    #     fitOne(name, reprocess=reprocess)

    p = Pool(CORES)
    fo = partial(fitOne, reprocess=reprocess)
    p.map(fo, groupNames)


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
                ff = fits.open(fpath)
                data = ff[1].data
                cutout = data[yCen-cutoutSize:yCen+cutoutSize+1, xCen-cutoutSize:xCen+cutoutSize+1]
                imgStack.append(cutout)
                plt.figure()
                plt.imshow(cutout, origin="lower")
                plotFiberCircle(xm,ym)
                plt.title("config %i dither %i: "%(configID, ditherNum) + str(idx) + " imgNum: %i"%(imgNum))
                plt.savefig("dither_%i_%i_img_%i.png"%(configID, ditherNum, imgNum), dpi=100)
                imgNum += 1
                ff.close()


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


def fitBOSSExp(mjd, site): #df):
    fs = glob.glob(OUT_DIR + "/%i/ditherFit*_%s.csv"%(mjd,site))
    dfList = []
    for f in fs:
        dfList.append(pandas.read_csv(f))
    df = pandas.concat(dfList)

    # import pdb; pdb.set_trace()


    # divide all spectrofluxes by fit flux amp
    # overwrite spectroflux norm
    dfList = []
    for name, group in df.groupby(["configID", "fiberID", "camera"]):
        amp = group.fluxAmpDitherFit.iloc[0]
        sigma = group.sigmaWokDitherFit.iloc[0]

        fluxPeakFit = fractionalFlux(amp, sigma, 0, 0, 0, 0)
        fluxNorm = group.spectroflux_corr.to_numpy() / fluxPeakFit
        # fluxNorm[fluxNorm > 1] = 1
        group["spectrofluxNorm"] = fluxNorm
        dfList.append(group)

    df = pandas.concat(dfList)

    df["dxWok"] = df.xWokStarPredict - df.xWokDitherFit
    df["dyWok"] = df.yWokStarPredict - df.yWokDitherFit
    df["drWok"] = numpy.sqrt(df.dxWok**2+df.dyWok**2)

    # import pdb; pdb.set_trace()
    for name, group in df.groupby(["configID", "camera"]):

        plt.figure(figsize=(8,8))
        ax1 = plt.gca()
        sns.scatterplot(ax=ax1, x=group.dxWok, y=group.dyWok, hue=group.spectrofluxNorm, s=10, hue_norm=(0,1))
        ax1.set_aspect("equal")
        ax1.set_xlim([-0.15,0.15])
        ax1.set_ylim([-0.15,0.15])
        ax1.set_xlabel("dx wok (mm)")
        ax1.set_ylabel("dy wok (mm)")
        plt.savefig("bossExp_configID_%s_camera_%s.png"%(str(name[0]), str(name[1])))

        plt.figure(figsize=(8,8))
        ax1 = plt.gca()
        l1 = numpy.percentile(group.sigmaWokDitherFit, 25)
        l2 = numpy.percentile(group.sigmaWokDitherFit, 75)
        sns.scatterplot(ax=ax1, x=group.xWokDitherFit, y=group.yWokDitherFit, hue=group.sigmaWokDitherFit, s=10)
        ax1.set_aspect("equal")
        # ax1.set_xlim([-0.15,0.15])
        # ax1.set_ylim([-0.15,0.15])
        ax1.set_xlabel("x wok (mm)")
        ax1.set_ylabel("y wok (mm)")
        plt.savefig("bossExp_spatial_configID_%s_camera_%s.png"%(str(name[0]), str(name[1])))

        plt.figure(figsize=(8,8))
        ax1 = plt.gca()
        ax1.plot(group.drWok, group.spectrofluxNorm, '.', ms=10, mec="none", mfc="black", alpha=0.1)
        pp = sns.color_palette("muted") #n_colors=len(set(group.bossExpNum)))
        # sns.scatterplot(ax=ax1, x=group.drWok, y=group.spectrofluxNorm, hue=group.bossExpNum, palette=pp, s=4)
        group = group.sort_values("sigmaWokDitherFit")
        _grp = group[["fiberID", "sigmaWokDitherFit"]].groupby("fiberID").mean().reset_index()
        _grp = _grp.sort_values("sigmaWokDitherFit")
        _sigClip = sigma_clip(_grp.sigmaWokDitherFit.to_numpy(), sigma=5)
        goodIDs = _grp.fiberID.to_numpy()[~_sigClip.mask]
        badIDs = _grp.fiberID.to_numpy()[_sigClip.mask]


        for goodID in goodIDs:
            row = group[group.fiberID==goodID].iloc[0]
            s = float(row.sigmaWokDitherFit)
            amp = float(row.fluxAmpDitherFit)
            f0 = fractionalFlux(amp,s,0,0,0,0)
            fluxes = []
            drs = numpy.linspace(0, 0.15, 100)
            for _dr in drs:
                fluxes.append(fractionalFlux(amp,s,0,_dr,0,0)/f0)
            ax1.plot(drs,fluxes, ls="-", color=pp[-1], lw=0.1, alpha=1)

        for badID in badIDs:
            row = group[group.fiberID==badID].iloc[0]
            s = float(row.sigmaWokDitherFit)
            amp = float(row.fluxAmpDitherFit)
            f0 = fractionalFlux(amp,s,0,0,0,0)
            fluxes = []
            drs = numpy.linspace(0, 0.15, 100)
            for _dr in drs:
                fluxes.append(fractionalFlux(amp,s,0,_dr,0,0)/f0)
            ax1.plot(drs,fluxes, ls="-", color=pp[3], lw=1.5, alpha=1)

        ax1.set_ylim([0,1.1])
        ax1.set_xlim([0,0.15])
        ax1.set_xlabel("dr wok (mm)")
        ax1.set_ylabel("spectroflux norm")
        plt.savefig("bossExp_rad_configID_%s_camera_%s.png"%(str(name[0]), str(name[1])), dpi=300)

        # calculate mean square error of model
        dfList = []
        # maxErrX = []
        # stdErrX = []
        # maxErrY = []
        # stdErrY = []
        dx = []
        dy = []

        group["dx_loo"] = group.xWokDitherFit - group.xWokDitherFit_loo
        group["dy_loo"] = group.yWokDitherFit - group.yWokDitherFit_loo
        group["dr_loo"] = numpy.sqrt(group.dx_loo**2+group.dy_loo**2)

        errs = group[["fiberID", "dx_loo", "dy_loo", "dr_loo", "fluxAmpDitherFit"]]
        # errs = errs[errs.fiberID.isin(goodIDs)]
        errs = errs.sort_values("fiberID")
        errs_m = errs.groupby(["fiberID", "fluxAmpDitherFit"]).std().reset_index()
        # errs_max = errs.groupby("fiberID").max().reset_index()

        plt.figure()
        plt.hist(errs_m.dr_loo, bins=100, label="dr")
        plt.hist(errs_m.dx_loo, bins=100, label="dx")
        plt.hist(errs_m.dy_loo, bins=100, label="dy")
        plt.legend()
        plt.savefig("bossExp_hist_configID_%s_camera_%s.png"%(str(name[0]), str(name[1])), dpi=300)

        plt.figure()
        plt.plot(errs_m.dr_loo, errs_m.fluxAmpDitherFit, '.k')
        plt.savefig("bossExp_errs_configID_%s_camera_%s.png"%(str(name[0]), str(name[1])), dpi=300)


        # for n, g in group.groupby("fiberID"):
        #     dx = g.xWokDitherFit - g.xWokDitherFit_loo
        #     dy = g.yWokDitherFit - g.yWokDitherFit_loo

        plt.figure(figsize=(8,8))
        ax1 = plt.gca()
        ax1.plot(group.fluxAmpDitherFit, group.sigmaWokDitherFit, '.k', ms=2)
        # ax1.set_aspect("equal")
        # ax1.set_ylim([0,1.1])
        # ax1.set_xlim([0,0.15])
        ax1.set_xlabel("flux amplitude")
        ax1.set_ylabel("sigma (mm)")
        plt.savefig("bossExp_sig_configID_%s_camera_%a.png"%(str(name[0]), str(name[1])))




        plt.close("all")


if __name__ == "__main__":
    mjd = int(sys.argv[1])
    site = sys.argv[2].lower()
    getDitherTables(mjd, site, reprocGFA=True, reprocFVC=True)
    computeWokCoords(mjd, site)
    fitFiberCenters(mjd, site, reprocess=True)

    # fitBOSSExp(mjd, site)

    # plotDitherPSFs()
    # plot_zps()
    #
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



