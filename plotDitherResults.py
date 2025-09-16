import pandas
import matplotlib.pyplot as plt
import numpy
import glob
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY
from coordio import calibration
from skimage.transform import SimilarityTransform, EuclideanTransform
import seaborn as sns
from coordio.transforms import FVCTransformAPO
from astropy.io import fits
from astropy.table import Table

# cp = sns.color_palette("husl", 20)
POLIDS=numpy.array([0, 1, 2, 3, 4, 5, 6, 9, 20, 27, 28, 29, 30])
RMAX = 310


def merge_all(mjds, suffix="", site=""):
    dfList = []
    # dirs = glob.glob("60*preweight")
    for mjd in mjds:
        fs = glob.glob("%i%s/ditherFit*%s.csv"%(mjd, suffix, site))
        for f in fs:
            print("processing ", f)
            dfList.append(pandas.read_csv(f))
    df = pandas.concat(dfList)

    # remove configurations with 5 or less boss exposures
    dfList = []
    for name, group in df.groupby("configID"):
        nBossExp = len(set(group.bossExpNum))
        if nBossExp >= 5:
            dfList.append(group)

    df = pandas.concat(dfList)

    # remove individual fits without loo success
    dfList = []
    for name, group in df.groupby(["configID", "fiberID", "camera"]):
        if False in group.ditherFitSuccess_loo:
            continue
        if False in group.ditherFitSuccess:
            continue
        dfList.append(group)

    df = pandas.concat(dfList)
    # ignore configID 18717 (an apo field that seemed bad?)
    # df = df[df.configID != 18717]
    # throw out data with unsuccessful fits
    df = df[df.ditherFitSuccess==True]
    # throw out data with negative flux amplitudes
    df = df[df.fluxAmpDitherFit > 0]
    df.to_csv("ditherFit_all_merged.csv", index=False)


# def fitZBs(x,y,dx,dy):

#     polids, coeffs = fitZhaoBurge(
#         x,
#         y,
#         x+dx,
#         y+dy,
#         polids=POLIDS,
#         normFactor=RMAX
#     )

#     _dx, _dy = getZhaoBurgeXY(
#         polids,
#         coeffs,
#         x,
#         y,
#         normFactor=RMAX
#     )

#     zdx = _dx - dx
#     zdy = _dy - dy
#     zdr = numpy.sqrt(zdx**2+zdy**2)
#     return zdx, zdy


def plotOne(df, xCol,yCol,dxCol,dyCol,xlabel,ylabel):
    site = str(list(set(df.site)))
    mjds = str(list(set(df.mjd)))
    scale=0.004
    width=0.5
    if "CCD" in xlabel:
        scale=scale*0.120
        width=4
    dr = numpy.sqrt(df[dxCol]**2+df[dyCol]**2)

    plt.figure(figsize=(8,8))
    ii = 0
    rms = numpy.sqrt(numpy.mean(dr**2))
    p90 = numpy.percentile(dr, 90)
    median = numpy.median(dr)

    cp = sns.color_palette("husl", len(set(df.configID)))

    for name, group in df.groupby("configID"):
        color = cp[ii]
        ii += 1
        plt.quiver(
            group[xCol],group[yCol],group[dxCol],group[dyCol], color=color, angles="xy",units="xy", width=width, scale=scale
        )
        dx = 1.5*p90 * numpy.cos(numpy.radians(group.fvcIPA.iloc[0]))
        dy = 1.5*p90 * numpy.sin(numpy.radians(group.fvcIPA.iloc[0]))
        plt.quiver(300, 200, dx, dy, color=color, angles="xy",units="xy", width=width*4, scale=scale)

    plt.axis("equal")
    plt.title("site=%s mjd=%s\nmedian=%.1f   rms=%.1f   p90=%.1f (um)"%(site, mjds, median*1000, rms*1000, p90*1000))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plotAll(mjd=None, betaArmUpdate=None):
    df = pandas.read_csv("ditherFit_all_merged.csv")
    if mjd is not None:
        df = df[df.mjd.isin(mjd)]
    print(set(df.fvcIPA))
    nConfigs = len(set(df.configID))

    xSky = df.xWokDitherFit.to_numpy()
    ySky = df.yWokDitherFit.to_numpy()
    xFVC = df.xWokMeasBOSS.to_numpy()
    yFVC = df.yWokMeasBOSS.to_numpy()

    dx = xFVC - xSky
    dy = yFVC - ySky
    dr = numpy.sqrt(dx**2+dy**2)

    df["dx"] = dx
    df["dy"] = dy
    df["dr"] = dr

    df = df[df.dr < 0.14]

    plotOne(df, "xWokDitherFit", "yWokDitherFit", "dx", "dy", "x wok (mm)", "y wok (mm)")
    plt.savefig("wok_err.png", dpi=200)


    # now rotate to fvc frame and look for coherence
    cosFVCRot = numpy.cos(-1*numpy.radians(df.fvcRot))
    sinFVCRot = numpy.sin(-1*numpy.radians(df.fvcRot))
    df["fdx"] = (df.dx * cosFVCRot - df.dy * sinFVCRot)
    df["fdy"] = (df.dx * sinFVCRot + df.dy * cosFVCRot)

    plotOne(df, "x_fvc", "y_fvc", "fdx", "fdy", "x CCD (pix)", "y CCD (pix)")
    plt.savefig("fvc_err.png", dpi=200)


    # rotate things to beta arm frame and plot
    alpha = df.alphaMeas.to_numpy() + df.alphaOffset.to_numpy()
    beta = df.betaMeas.to_numpy() + df.betaOffset.to_numpy()
    theta = numpy.radians(alpha+beta) - numpy.pi/2
    dxRot = numpy.cos(theta)*df.dx+numpy.sin(theta)*df.dy
    dyRot = -numpy.sin(theta)*df.dx+numpy.cos(theta)*df.dy

    df["dxBetaArm"] = dxRot
    df["dyBetaArm"] = dyRot



    plotOne(df, "xWok", "yWok", "dxBetaArm", "dyBetaArm", "x wok (mm)", "y wok (mm)")

    if betaArmUpdate is not None:
        pt = calibration.positionerTable.reset_index()
        # how many measurements are we averaging over for each robots?
        for name, group in df.groupby("positionerID"):
            print(name, "averaged over", len(group))
        _dfmean = df[["positionerID", "dxBetaArm", "dyBetaArm"]].groupby("positionerID").mean().reset_index()  # warning only boss fibers here!
        cols = pt.columns.to_list()
        ptNew = pt.merge(_dfmean, how="outer", on="positionerID")
        ptNew = ptNew.fillna(value=0)
        ptNew["bossX"] = ptNew.bossX - ptNew.dxBetaArm
        ptNew["bossY"] = ptNew.bossY - ptNew.dyBetaArm
        ptNew = ptNew[cols]
        ptNew.to_csv(betaArmUpdate, index_label="id")
        # import pdb; pdb.set_trace()

    dfList = []

    for name, group in df.groupby(["fiberID"]):
        group.drop('level_0', axis=1, inplace=True)
        group = group.reset_index()
        group["dxBetaArmFit"] = group.dxBetaArm - numpy.mean(group.dxBetaArm)
        group["dyBetaArmFit"] = group.dyBetaArm - numpy.mean(group.dyBetaArm)
        dfList.append(group)

    df = pandas.concat(dfList)

    plotOne(df, "xWok", "yWok", "dxBetaArmFit", "dyBetaArmFit", "x wok (mm)", "y wok (mm)")



def plotFVCdistortion(mjd=None, fiducialOut=None, includeVar=True):
    df = pandas.read_csv("ditherFit_all_merged.csv")
    if mjd is not None:
        df = df[df.mjd.isin(mjd)]
    nConfigs = len(set(df.configID))
    print("nConfigs!!", nConfigs)

    xSky = df.xWokDitherFit.to_numpy()
    ySky = df.yWokDitherFit.to_numpy()

    # _dx = df.xWokMeasBOSS - df.xWokMeasMetrology
    xFVC = df.xWokMeasBOSS.to_numpy()
    yFVC = df.yWokMeasBOSS.to_numpy()

    dx = xFVC - xSky
    dy = yFVC - ySky
    dr = numpy.sqrt(dx**2+dy**2)

    df["dx"] = dx
    df["dy"] = dy
    df["dr"] = dr

    df = df[df.dr < 0.14]

    df["xWokMetDithFit"] = df.xWokDitherFit + (df.xWokMeasMetrology - df.xWokMeasBOSS)
    df["yWokMetDithFit"] = df.yWokDitherFit + (df.yWokMeasMetrology - df.yWokMeasBOSS)
    df["rWokMetDithFit"] = numpy.sqrt(df.xWokMetDithFit**2+df.yWokMetDithFit**2)

    # compute variances for xyDither fits
    _df = df[["fiberID", "configID", "camera", "xWokDitherFit_loo", "yWokDitherFit_loo"]]
    _df = _df.groupby(["fiberID", "configID", "camera"]).var().reset_index()
    df = df.merge(_df, on=["fiberID", "configID", "camera"], suffixes=(None, "_var"))

    dfList = []
    dfFIFList = []
    dfXYTest = pandas.DataFrame()
    # rMax = numpy.max(df.rWokMetDithFit)
    rTest = numpy.random.uniform(0, RMAX**2, size=1000)
    thetaTest = numpy.random.uniform(0, numpy.pi*2, size=1000)
    xs = numpy.sqrt(rTest)*numpy.cos(thetaTest)
    ys = numpy.sqrt(rTest)*numpy.sin(thetaTest)
    for name, group in df.groupby(["configID"]):
        if sum(numpy.isnan(group.xWokDitherFit_loo.to_numpy())) > 0:
            print("skipping due to nans", group["mjd"].iloc[0], name)
            continue
        xyFVC = group[["x_fvc", "y_fvc"]].to_numpy()
        xyWok = group[["xWokMetDithFit", "yWokMetDithFit"]].to_numpy()
        x_var = group.xWokDitherFit_loo_var.to_numpy()
        y_var = group.yWokDitherFit_loo_var.to_numpy()


        st = SimilarityTransform()
        st.estimate(xyFVC, xyWok)
        xyFit = st(xyFVC)

        dxyFit = xyWok - xyFit

        group["dxWok"] = dxyFit[:,0]
        group["dyWok"] = dxyFit[:,1]
        group["drWok"] = numpy.sqrt(group.dxWok**2+group.dyWok**2)
        group["strot"] = numpy.degrees(st.rotation)
        group["stdx"] = st.translation[0]
        group["stdy"] = st.translation[1]
        group["stscale"] = st.scale

        # also fit a zb poly
        print("\n\n")
        print("fitting ZB", name, "mjd ", group.mjd.iloc[0])
        print("\n\n")
        polids, coeffs = fitZhaoBurge(
            xyFit[:,0],
            xyFit[:,1],
            xyWok[:,0],
            xyWok[:,1],
            # polids=numpy.arange(33),
            polids=POLIDS,
            normFactor=RMAX,
            x_var = x_var,
            y_var = y_var
        )

        dx, dy = getZhaoBurgeXY(
            polids,
            coeffs,
            xyFit[:,0],
            xyFit[:,1],
            normFactor=RMAX
        )


        zdx = group.dxWok - dx
        zdy = group.dyWok - dy
        zdr = numpy.sqrt(zdx**2+zdy**2)
        group["zdx"] = zdx
        group["zdy"] = zdy
        group["zdr"] = zdr


        # now move FIFs to wok frame
        mjd = group.mjd.iloc[0]
        site = group.site.iloc[0]
        fvcpath = "%i/dither_fvc_%i_%s.csv"%(mjd,mjd,site)
        df_fvc = pandas.read_csv(fvcpath)
        df_fvc = df_fvc[df_fvc.fvcImgNum == group.fvcImgNum.iloc[0]]
        # just get fiducials
        df_fvc = df_fvc[df_fvc.positionerID==-1].reset_index(drop=True)
        # tossHoles = ["F%i"%x for x in range(20)]
        # df_fvc = df_fvc[~df_fvc.holeID.isin(tossHoles)]

        xyFVC = df_fvc[["x", "y"]].to_numpy()
        xyFit = st(xyFVC)

        dx, dy = getZhaoBurgeXY(
            polids,
            coeffs,
            xyFit[:,0],
            xyFit[:,1],
            normFactor=RMAX
        )

        df_fvc["xWokMeas"] = xyFit[:,0] + dx
        df_fvc["yWokMeas"] = xyFit[:,1] + dy

        df_fvc = df_fvc[["holeID", "xWok", "yWok", "xWokMeas", "yWokMeas"]]
        dfFIFList.append(df_fvc)
        dfList.append(group)

        dx, dy = getZhaoBurgeXY(
            polids,
            coeffs,
            xs,
            ys,
            normFactor=RMAX
        )

        ### now create test points for each zb transform
        df_test = pandas.DataFrame(
            {
            "x": xs,
            "y": ys,
            "dx": dx,
            "dy": dy,
            "id": numpy.arange(len(xs))
            }
        )
        df_test["configID"] = name[0]
        for ii, coeff in enumerate(coeffs):
            df_test["C%i"%ii] = coeff

        dfXYTest = pandas.concat([dfXYTest, df_test])

    df = pandas.concat(dfList)
    dfFIF = pandas.concat(dfFIFList)


    ###### visualize the zb models #####
    df_m = dfXYTest.groupby(["id"]).mean().reset_index()
    dfXYTest = dfXYTest.merge(df_m[["id", "dx", "dy"]], on="id", suffixes=(None, "_m"))
    dfXYTest["dxm"] = dfXYTest.dx - dfXYTest.dx_m
    dfXYTest["dym"] = dfXYTest.dy - dfXYTest.dy_m
    dfXYTest["drm"] = numpy.sqrt(dfXYTest.dxm**2+dfXYTest.dym**2)

    for name, group in dfXYTest.groupby("configID"):
        plt.figure(figsize=(8,8))
        plt.quiver(
            group.x, group.y, group.dxm, group.dym, angles="xy", units="xy", width=1, scale=0.003
        )
        rms = numpy.sqrt(numpy.mean(group.drm**2))*1000
        plt.title("configID=%i rms = %.1f um"%(name, rms))
        plt.xlabel("x wok (mm)")
        plt.ylabel("y wok (mm)")
        plt.axis("equal")



    plt.figure(figsize=(8,8))
    plt.quiver(
        df.xWokMetDithFit, df.yWokMetDithFit, df.dxWok, df.dyWok, angles="xy", units="xy", width=1, scale=0.03
    )
    rms = numpy.sqrt(numpy.mean(df.drWok**2))*1000
    plt.title("rms = %.1f um"%rms)
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")
    plt.axis("equal")
    plt.savefig("st_fvc.png", dpi=200)


    plt.figure(figsize=(8,8))
    plt.quiver(
        df.xWokMetDithFit, df.yWokMetDithFit, df.zdx, df.zdy, angles="xy", units="xy", width=1, scale=0.003
    )
    rms = numpy.sqrt(numpy.mean(df.zdr**2))*1000
    plt.title("rms = %.1f um"%rms)
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")
    plt.axis("equal")
    plt.savefig("zb_fvc.png", dpi=200)

    ### now look at fiducial errors
    dfFIF["dxWok"] = dfFIF.xWokMeas - dfFIF.xWok
    dfFIF["dyWok"] = dfFIF.yWokMeas - dfFIF.yWok
    dfFIF["drWok"] = numpy.sqrt(dfFIF.dxWok**2+dfFIF.dyWok**2)
    dfFIF = dfFIF[dfFIF.drWok < 2]
    dfm = dfFIF.groupby("holeID").mean().reset_index()
    dfFIF = dfFIF.merge(dfm[["holeID", "dxWok", "dyWok"]], on="holeID", suffixes=(None,"_m"))

    # plt.figure()
    # plt.hist(dfFIF.drWok)

    plt.figure(figsize=(8,8))
    plt.quiver(
        dfFIF.xWok, dfFIF.yWok, dfFIF.dxWok, dfFIF.dyWok, angles="xy", units="xy", width=1, scale=0.001
    )
    rms = numpy.sqrt(numpy.mean(dfFIF.drWok**2))
    plt.title("rms = %.1f um"%(rms*1000))
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")
    plt.axis("equal")
    plt.savefig("zb_fif_all.png", dpi=200)

    # tossHoles = ["F%i"%x for x in range(20)]
    # dfFIF = dfFIF[~dfFIF.holeID.isin(tossHoles)]

    # plt.figure()
    # plt.hist(dfFIF.drWok)

    plt.figure(figsize=(8,8))
    plt.quiver(
        dfFIF.xWok, dfFIF.yWok, dfFIF.dxWok, dfFIF.dyWok, angles="xy", units="xy", width=1, scale=0.001
    )
    plt.quiver(
        dfFIF.xWok, dfFIF.yWok, dfFIF.dxWok_m, dfFIF.dyWok_m, color="red", angles="xy", units="xy", width=1, scale=0.001
    )
    rms = numpy.sqrt(numpy.mean(dfFIF.drWok**2))
    plt.title("rms = %.1f um"%(rms*1000))
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")
    plt.axis("equal")
    plt.savefig("zb_fif_less.png", dpi=200)

    dfFIF["dxWokRes"] = dfFIF.dxWok - dfFIF.dxWok_m
    dfFIF["dyWokRes"] = dfFIF.dyWok - dfFIF.dyWok_m
    dfFIF["drWokRes"] = numpy.sqrt(dfFIF.dxWokRes**2+dfFIF.dyWokRes**2)
    # use same variance for x and y
    dfVAR = dfFIF[["holeID", "drWokRes"]].groupby("holeID").var().reset_index()
    dfVAR["xyVar"] = dfVAR.drWokRes
    dfm = dfm.merge(dfVAR, on="holeID", suffixes=(None, "_var"))

    plt.figure(figsize=(8,8))
    plt.quiver(
        dfFIF.xWok, dfFIF.yWok, dfFIF.dxWokRes, dfFIF.dyWokRes, angles="xy", units="xy", width=1, scale=0.001
    )

    rms = numpy.sqrt(numpy.mean(dfFIF.drWokRes**2))
    plt.title("rms = %.1f um"%(rms*1000))
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")
    plt.axis("equal")
    plt.savefig("zb_fif_shift.png", dpi=200)

    # write updated fiducial coords
    keepCols = ['site', 'holeID', 'id', 'xWok', 'yWok', 'zWok', 'col', 'row', 'xyVar']
    fcm = calibration.fiducialCoords.reset_index()
    # fcm["xyVar"] = numpy.max(dfFIF.xyVar) # initialize variances to max measured variance (for missing fiducial measurements)
    # import pdb; pdb.set_trace()
    print(len(fcm), len(dfm))
    # fiducial F7 doesn't exist in measurements (faint)
    fcm = fcm.merge(dfm, how="left", on="holeID", suffixes=(None, "_m"))
    # not all fiducials have measurements some are broken (NaN values in xWokMeas)
    xWokSaved = fcm.xWok.to_numpy()
    yWokSaved = fcm.yWok.to_numpy()
    xyVarSaved = numpy.nanmax(fcm.xyVar)
    xWokNew = fcm.xWokMeas.to_numpy()
    yWokNew = fcm.yWokMeas.to_numpy()
    xyVarNew = fcm.xyVar.to_numpy()
    brokenInds = numpy.argwhere(numpy.isnan(xWokNew)).flatten()
    for idx in brokenInds:
        xWokNew[idx] = xWokSaved[idx]
        yWokNew[idx] = yWokSaved[idx]
        xyVarNew[idx] = xyVarSaved

    fcm["xWok"] = xWokNew
    fcm["yWok"] = yWokNew
    fcm["xyVar"] = xyVarNew

    if fiducialOut:
        fcm = fcm[keepCols]
        if not includeVar:
            fcm = fcm.drop("xyVar", axis=1)
        fcm.to_csv(fiducialOut)
    plt.show()
    # import pdb; pdb.set_trace()


# def reprocessFVC(centType="sep"):
#     dfAll = pandas.read_csv("ditherFit_all_merged.csv")
#     fcm = pandas.read_csv("fiducialCoords_dither_updated.csv")
#     fvcImg = dfAll[["fvcImgNum", "mjd"]].groupby(["fvcImgNum", "mjd"]).first().reset_index().sort_values(["mjd","fvcImgNum"])
#     fvcImgNums = fvcImg.fvcImgNum.to_numpy()
#     mjds = fvcImg.mjd.to_numpy()

#     dfList = []
#     for imgNum, mjd in zip(fvcImgNums,mjds):
#         imgNumStr = str(imgNum).zfill(4)
#         ff = fits.open("/Volumes/futa/apo/data/fcam/%i/proc-fimg-fvc1n-%s.fits"%(mjd, imgNumStr))
#         posCoords = Table(ff["POSANGLES"].data).to_pandas()

#         fvct = FVCTransformAPO(
#             ff[1].data,
#             posCoords,
#             ff[1].header["IPA"],
#             fiducialCoords=fcm,
#             polids=POLIDS,
#             zbNormFactor=RMAX
#         )
#         fvct.extractCentroids()
#         fvct.fit(
#             centType=centType
#         )

#         _df = dfAll[(dfAll.fvcImgNum==imgNum) & (dfAll.mjd==mjd)]
#         _df = _df.groupby(["positionerID"]).first().reset_index()

#         _df = _df.merge(fvct.positionerTableMeas, on="positionerID", suffixes=(None, "_newFVC"))
#         _df["x_newFVC"] = _df.x
#         _df["y_newFVC"] = _df.y

#         for kw in ["", "_newFVC"]:
#             x = _df.xWokDitherFit.to_numpy()
#             y = _df.yWokDitherFit.to_numpy()
#             dx = _df["xWokMeasBOSS%s"%kw] - x
#             dy = _df["yWokMeasBOSS%s"%kw] - y
#             _df["dxWok%s"%kw] = dx
#             _df["dyWok%s"%kw] = dy
#         dfList.append(_df)

#     df = pandas.concat(dfList)
#     df.to_csv("dither_reprocess_fvc.csv", index=False)
            # import pdb; pdb.set_trace()
        #     dr = numpy.sqrt(dx**2+dy**2)
        #     rms = numpy.sqrt(numpy.mean(dr**2))
        #     med = numpy.median(dr)
        #     p95 = numpy.percentile(dr, 95)
        #     # print("rms", rms)
        #     plt.figure(figsize=(8,8))
        #     plt.quiver(x,y,dx,dy,angles="xy", units="xy", scale=0.002)
        #     plt.title("%i %.3f %.3f %.3f"%(imgNum, rms, med, p95))
        #     plt.axis("equal")

        # plt.show()
    # import pdb; pdb.set_trace()

def plotReprocessFVC():
    df = pandas.read_csv("dither_reprocess_fvc.csv")

    for suffix in ["", "_newFVC"]:
        x = df["xWokDitherFit"].to_numpy()
        y = df["yWokDitherFit"].to_numpy()
        dx = df["dxWok%s"%suffix].to_numpy()
        dy = df["dyWok%s"%suffix].to_numpy()
        dr = numpy.sqrt(dx**2+dy**2)
        plt.figure(figsize=(7,7))
        plt.quiver(x,y,dx,dy,angles="xy",units="xy",width=0.4,scale=0.003)
        plt.axis("equal")
        plt.xlabel("x wok (mm)")
        plt.ylabel("y wok (mm)")
        plt.axis("equal")

        rms = numpy.sqrt(numpy.mean(dr**2))
        median = numpy.median(dr)
        p90 = numpy.percentile(dr, 90)
        if "new" in suffix:
            title = "new p50=%.3f rms=%.3f p90=%.3f (mm)"%(median, rms, p90)
        else:
            title = "old p50=%.3f rms=%.3f p90=%.3f (mm)"%(median, rms, p90)
        plt.title(title)

        # rotate into ccd frame
        if "new" in suffix:
            x = df["x_newFVC"].to_numpy()
            y = df["y_newFVC"].to_numpy()
        else:
            x = df.x_fvc.to_numpy()
            y = df.y_fvc.to_numpy()

        dxCCD = dx * numpy.cos(-1*numpy.radians(df.fvcRot.to_numpy())) - dy * numpy.sin(-1*numpy.radians(df.fvcRot.to_numpy()))
        dyCCD = dx * numpy.sin(-1*numpy.radians(df.fvcRot.to_numpy())) + dy * numpy.cos(-1*numpy.radians(df.fvcRot.to_numpy()))

        plt.figure(figsize=(7,7))
        plt.quiver(x,y,dxCCD,dyCCD,angles="xy",units="xy", width=4,scale=0.0004)
        plt.axis("equal")
        plt.xlabel("x CCD (pix)")
        plt.ylabel("y CCD (pix)")
        plt.title(title)
        plt.axis("equal")

    plt.show()

def plotGFADistortion(mjd=None, preNudge=False, filename=None):
    dfList = []
    if mjd is not None:
        for _m in mjd:
            try:
                if preNudge:
                    pn = ".preGFA"
                else:
                    pn = ""
                dfList.append(pandas.read_csv("%i%s/dither_gfa_%i_lco.csv"%(_m,pn,_m)).sort_values("gfaNum"))
            except:
                dfList.append(pandas.read_csv("%i/dither_gfa_%i_apo.csv"%(_m,_m)).sort_values("gfaNum"))

    df = pandas.concat(dfList)
    # plt.figure()
    # sns.histplot(x=df.x2, hue=df.gfaNum, palette="Set2")
    # plt.show()
    df["dxWok"] = df.xWokMeas - df.xWokPred
    df["dyWok"] = df.yWokMeas - df.yWokPred
    df["drWok"] = numpy.sqrt(df.dxWok**2+df.dyWok**2)
    df = df[df.drWok < 0.1]

    gfaCoords = calibration.gfaCoords.reset_index().sort_values("id")

    dfList = []
    xGFAoff = []
    yGFAoff = []
    gfaNum = []
    for name, group in df.groupby("gfaNum"):
        plt.figure()
        plt.hist(group.drWok, bins=100)
        plt.title(str(name))

        plt.figure(figsize=(8,8))
        plt.quiver(group.xWokPred, group.yWokPred, group.dxWok, group.dyWok, angles="xy", units="xy", scale=0.03)
        plt.axis("equal")
        plt.title(str(name))

        # st = SimilarityTransform()
        # # st = EuclideanTransform()
        # xyPred = group[["xWokPred", "yWokPred"]].to_numpy()
        # xyMeas = group[["xWokMeas", "yWokMeas"]].to_numpy()
        # st.estimate(xyMeas, xyPred)

        # print("gfa", name, st.translation, numpy.degrees(st.rotation)) #, st.scale)

        # xyFit = st(xyMeas)

        # dxyWokFit = xyFit - xyPred
        # group["dxWokFit"] = dxyWokFit[:,0]
        # group["dyWokFit"] = dxyWokFit[:,1]
        # group["drWokFit"] = numpy.linalg.norm(dxyWokFit,axis=1)

        dxOff = numpy.mean(group.dxWok)
        dyOff = numpy.mean(group.dyWok)
        # print("mean xy offset", dxOff, dyOff)
        xGFAoff.append(dxOff)
        yGFAoff.append(dyOff)
        gfaNum.append(name)

        group["dxWokFit"] = group.dxWok - dxOff
        group["dyWokFit"] = group.dyWok - dyOff
        group["drWokFit"] = numpy.sqrt(group.dxWokFit**2+group.dyWokFit**2)

        plt.figure()
        plt.hist(group.drWokFit, bins=100)
        plt.title(str(name))

        plt.figure(figsize=(8,8))
        plt.quiver(group.xWokPred, group.yWokPred, group.dxWokFit, group.dyWokFit, angles="xy", units="xy", scale=0.08)
        plt.axis("equal")
        plt.title(str(name))

        dfList.append(group)

    df = pandas.concat(dfList)


    ### apply dc xy offset to current gfaCoords
    # import pdb; pdb.set_trace()
    # missingGFAs = set(range(1,7)) - set(gfaNum)
    # gfaCoords["xWok"] = gfaCoords.xWok - numpy.array(xGFAoff)
    # gfaCoords["yWok"] = gfaCoords.yWok - numpy.array(yGFAoff)

    for _gfaNum, _xOff, _yOff in zip(gfaNum, xGFAoff, yGFAoff):
        gfaCoords.loc[gfaCoords.id==_gfaNum, "xWok"] = gfaCoords.loc[gfaCoords.id==_gfaNum, "xWok"] - _xOff
        gfaCoords.loc[gfaCoords.id==_gfaNum, "yWok"] = gfaCoords.loc[gfaCoords.id==_gfaNum, "yWok"] - _yOff

    if filename is not None:
        gfaCoords.to_csv(filename)

    plt.figure()
    plt.hist(df.drWok, bins=100)
    plt.xlabel("dr (mm)")

    rms = numpy.sqrt(numpy.mean(df.drWok**2))*1000
    plt.figure(figsize=(8,8))
    plt.quiver(df.xWokPred, df.yWokPred, df.dxWok, df.dyWok, angles="xy", units="xy", scale=0.002, width=0.1)
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")
    plt.title("rms err: %.1f micron"%rms)
    plt.axis("equal")

    rms = numpy.sqrt(numpy.mean(df.drWokFit**2))*1000
    plt.figure(figsize=(8,8))
    plt.quiver(df.xWokPred, df.yWokPred, df.dxWokFit, df.dyWokFit, angles="xy", units="xy", scale=0.002, width=0.1)
    plt.title("rms err: %.1f micron"%rms)
    plt.xlabel("x wok (mm)")
    plt.ylabel("y wok (mm)")
    plt.axis("equal")

    plt.figure()
    plt.hist(df.drWokFit, bins=100)
    plt.xlabel("dr (mm)")

    # plt.show()

    # import pdb; pdb.set_trace()


def plotPAvsDec():
    df = pandas.read_csv("ditherFit_all_merged.csv")
    plt.figure()
    plt.plot(df.SOL_PA, df.SOL_DEC, '.', ms=10)
    plt.ylabel("DEC (deg)")
    plt.xlabel("PA (deg)")
    plt.show()

    import pdb; pdb.set_trace()

def plotScale():
    df = pandas.read_csv("ditherFit_all_merged.csv")
    plt.figure()
    plt.plot(df.taiMid, df.focal_scale, '.k')
    # plt.plot(df.taiMid, df.focal_scale*1.0003, 'xk')
    plt.plot(df.taiMid, df.SOL_SCL, '.r')
    plt.show()

    import pdb; pdb.set_trace()

    # plt.show()


def plotStars():
    df = pandas.read_csv("ditherFit_all_merged.csv")
    df = df[["configID", "mjd", "camera", "fiberID", "xWokStarPredict", "yWokStarPredict", "xWokDitherFit", "yWokDitherFit"]]
    df = df.groupby(["configID", "mjd", "fiberID", "camera"]).mean().reset_index()
    df["dx"] = df.xWokDitherFit - df.xWokStarPredict
    df["dy"] = df.yWokDitherFit - df.yWokStarPredict
    df["dr"] = numpy.sqrt(df.dx**2+df.dy**2)
    df = df[df.dr<0.5]

    for name, group in df.groupby("mjd"):
        plt.figure()
        plt.hist(df.dr)
        plt.title(str(name))

        plt.figure(figsize=(8,8))
        plt.quiver(df.xWokStarPredict, df.yWokStarPredict, df.dx, df.dy, angles="xy", units="xy", scale=0.005)
        plt.axis("equal")
        plt.title(str(name))
    plt.show()


def plotFWHMs():
    df = pandas.read_csv("ditherFit_all_merged.csv")
    plt.figure()
    for name, group in df.groupby("configID"):
        fwhmGFA = group.fwhm.to_numpy()
        fwhmWok = 2.355*group.sigmaWokDitherFit.to_numpy()/217.7358*3600
        medGFA = numpy.median(fwhmGFA)
        lowGFA = numpy.percentile(fwhmGFA, 25)
        highGFA = numpy.percentile(fwhmGFA, 75)
        medWok = numpy.median(fwhmWok)
        lowWok = numpy.percentile(fwhmWok, 25)
        highWok = numpy.percentile(fwhmWok, 75)
        xerr = numpy.array([[lowGFA, highGFA]]).T
        yerr = numpy.array([[lowWok, highWok]]).T


        meanGFA = numpy.mean(fwhmGFA)
        xerr = numpy.std(fwhmGFA)

        meanWok = numpy.mean(fwhmWok)
        yerr = numpy.std(fwhmWok)
        # print(fwhm, sigWok)
        plt.errorbar(meanGFA, meanWok, xerr=xerr, yerr=yerr, color="black")

    plt.xlabel("fwhm (GFA)")
    plt.ylabel("fwhm (dither)")
    plt.axis("equal")
    plt.xlim([1,3])
    # plt.ylim([1,3])
    plt.show()

        # for col in list(df.columns):
        #     if col.endswith("_gfa"):
        #         print(col)
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    # look at GFA calib errors
    # plt.show()

    # merge_all(mjds=[60521,60528])
    # # plotStars()

    # # import pdb; pdb.set_trace()

    # plotGFADistortion(mjd=[60521,60528], preNudge=True)
    # plt.show()
    # import pdb; pdb.set_trace()
    # # plotAll()
    # # plotAll(mjd=60448) # apo, good after fiducial fixes

    # # plotAll(mjd=60229) # after baffle rotation
    # # plotAll(mjd=60371) # after IMB change
    # # plotAll(mjd=60520) # new (bad) mount lco

    #### LAST LCO DITHERS (pre robot replacement in jan 2025) ############
    # merge_all(mjds=[60521,60528, 60573, 60575, 60576])
    # plotGFADistortion(mjd=[60521,60528, 60573, 60575, 60576])
    # plotAll(mjd=[60521,60528, 60573, 60575, 60576]) # mount loosened
    # plotFVCdistortion(mjd=[60521,60528, 60573, 60575, 60576], fiducialOut="fiducial_coords_lco_60576.csv") # writes new file for fiducial positions
    ####################################

    #### LAST APO DITHERS? ##########################
    # merge_all(mjds=[60629, 60537, 60572, 60558, 60529]) #, suffix=".prePsfNudge") #, 60606])
    # plotAll(mjd=[60629, 60537, 60572, 60558, 60529])#, betaArmUpdate="positionerTable_apo_weighted2.csv") #, 60606]) #, 60537, 60572, 60558], betaArmUpdate="apo_positionerTable_barm_fixed.csv") # apo post shutdown
    # plotGFADistortion(mjd=[60629, 60537, 60572, 60558, 60529])#, filename="gfaCoords_apo_weighted2.csv") #, 60537, 60572, 60558])
    # plotFVCdistortion(mjd=[60629, 60537, 60572, 60558, 60529])#, fiducialOut="fiducialCoords_apo_weighted2.csv") #, 60606], fiducialOut="fiducialCoords_apo_weighted.csv") #, 60537, 60572, 60558], fiducialOut="junk_coords.csv")
    # plotFWHMs()
    ################################################


    # #### LCO Jan 2025 eng ###########
    # merge_all(mjds=[60691, 60692, 60693], site="lco") # 60690 was a poor dither sequence no robot dithers and clouds came in
    # plotAll(mjd=[60691, 60692, 60693], betaArmUpdate="positionerTable_lco_jan_25_4.csv")
    # plotGFADistortion(mjd=[60691, 60692, 60693], filename="gfaCoords_lco_jan_25_4.csv")
    # plotFVCdistortion(mjd=[60691, 60692, 60693], fiducialOut="fiducialCoords_lco_jan_25_4.csv", includeVar=True)


    #### APO revisit  Jan 2025 using flex model (no nudge) ###########
    # merge_all(mjds=[60693, 60661, 60712], site="apo") # 60606, 60629,
    # plotAll(mjd=[60693, 60661, 60712], betaArmUpdate="positionerTable_apo_feb_7_1.csv")
    # plotGFADistortion(mjd=[60693, 60661, 60712], filename="gfaCoords_apo_feb_7_1.csv")
    # plotFVCdistortion(mjd=[60693, 60661, 60712], fiducialOut="fiducialCoords_apo_feb_7_1.csv", includeVar=True)

    # #### LCO Feb 2025 dither (pointing looks bad)
    # merge_all(mjds=[60714], site="lco") # 60606, 60629,
    # plotAll(mjd=[60714], betaArmUpdate="junk.csv")
    # plotGFADistortion(mjd=[60714], filename="junk.csv")
    # plotFVCdistortion(mjd=[60714], fiducialOut="junk.csv", includeVar=True)

    #### APO Feb 2025 dither post attempted fix
    # merge_all(mjds=[60715], site="apo") # 60606, 60629,
    # plotAll(mjd=[60715], betaArmUpdate="junk.csv")
    # plotGFADistortion(mjd=[60715], filename="junk.csv")
    # plotFVCdistortion(mjd=[60715], fiducialOut="junk.csv", includeVar=True)

    #### LCO Feb 2025 dither stui script blind tracking
    # merge_all(mjds=[60717], site="lco") # 60606, 60629,
    # plotAll(mjd=[60717], betaArmUpdate="junk.csv")
    # plotGFADistortion(mjd=[60717], filename="junk.csv")
    # plotFVCdistortion(mjd=[60717], fiducialOut="junk.csv", includeVar=True)

    ### LCO July 2025 ### back to junk data?
    merge_all(mjds=[60867, 60815], site="lco") # 60606, 60629,
    plotAll(mjd=[60867, 60815], betaArmUpdate="junk.csv")
    plotGFADistortion(mjd=[60867, 60815], filename="junk.csv")
    plotFVCdistortion(mjd=[60867, 60815], fiducialOut="junk.csv", includeVar=True)


    plt.show()
