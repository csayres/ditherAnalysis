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

cp = sns.color_palette("husl", 11)
POLIDS=numpy.array([0, 1, 2, 3, 4, 5, 6, 9, 20, 27, 28, 29, 30])
RMAX = 310


def merge_all():
    dfList = []
    dirs = glob.glob("60*")
    for d in dirs:
        fs = glob.glob(d + "/ditherFit*.csv")
        for f in fs:
            print("processing ", f)
            dfList.append(pandas.read_csv(f))
    df = pandas.concat(dfList)
    df.to_csv("ditherFit_all_merged.csv", index=False)


def fitZBs(x,y,dx,dy):

    polids, coeffs = fitZhaoBurge(
        x,
        y,
        x+dx,
        y+dy,
        polids=POLIDS,
        normFactor=RMAX
    )

    _dx, _dy = getZhaoBurgeXY(
        polids,
        coeffs,
        x,
        y,
        normFactor=RMAX
    )

    zdx = _dx - dx
    zdy = _dy - dy
    zdr = numpy.sqrt(zdx**2+zdy**2)
    return zdx, zdy


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
    plt.title("site=%s mjd=%s\nmedian=%.1f   rms=%.1f   p90=%.1f (arcsec)"%(site, mjds, median/.06, rms/.06, p90/.06))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plotAll(mjd=None):
    df = pandas.read_csv("ditherFit_all_merged.csv")
    if mjd is not None:
        df = df[df.mjd==mjd]
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

    dx, dy = fitZBs(
        df.xWokDitherFit.to_numpy(),
        df.yWokDitherFit.to_numpy(),
        df.dx.to_numpy(),
        df.dy.to_numpy()
    )

    df["zdx"] = dx
    df["zdy"] = dy

    plotOne(df, "xWokDitherFit", "yWokDitherFit", "zdx", "zdy", "x wok (mm)", "y wok (mm)")
    plt.savefig("wok_zb_err.png", dpi=200)


    dx, dy = fitZBs(
        df.x_fvc.to_numpy(),
        df.y_fvc.to_numpy(),
        df.fdx.to_numpy(),
        df.fdy.to_numpy()
    )

    df["zfdx"] = dx
    df["zfdy"] = dy

    plotOne(df, "x_fvc", "y_fvc", "zfdx", "zfdy", "x CCD (pix)", "y CCD (pix)")
    plt.savefig("fvc_zb_err.png", dpi=200)
    # plt.show()
    # import pdb; pdb.set_trace()


    df["fdx"] = (df.zdx * cosFVCRot - df.zdy * sinFVCRot)
    df["fdy"] = (df.zdx * sinFVCRot + df.zdy * cosFVCRot)

    plotOne(df, "x_fvc", "y_fvc", "fdx", "fdy", "x CCD (pix)", "y CCD (pix)")

    dx, dy = fitZBs(
        df.x_fvc.to_numpy(),
        df.y_fvc.to_numpy(),
        df.fdx.to_numpy(),
        df.fdy.to_numpy()
    )

    df["zfdx"] = dx
    df["zfdy"] = dy

    plotOne(df, "x_fvc", "y_fvc", "zfdx", "zfdy", "x CCD (pix)", "y CCD (pix)")



    # plt.show()
    # import pdb; pdb.set_trace()



    # polids, coeffs = fitZhaoBurge(
    #     df.xWokMeasBOSS.to_numpy(),
    #     df.yWokMeasBOSS.to_numpy(),
    #     df.xWokDitherFit.to_numpy(),
    #     df.yWokDitherFit.to_numpy(),
    #     polids=POLIDS,
    #     normFactor=RMAX
    # )

    # dx, dy = getZhaoBurgeXY(
    #     polids,
    #     coeffs,
    #     df.xWokMeasBOSS.to_numpy(),
    #     df.yWokMeasBOSS.to_numpy(),
    #     normFactor=RMAX
    # )

    # zxfit = df.xWokMeasBOSS.to_numpy() + dx
    # zyfit = df.yWokMeasBOSS.to_numpy() + dy

    # zdx = zxfit - df.xWokDitherFit.to_numpy()
    # zdy = zyfit - df.yWokDitherFit.to_numpy()
    # zdr = numpy.sqrt(zdx**2+zdy**2)

    # print("rms all", numpy.sqrt(numpy.mean(zdr**2)))
    # plt.figure()
    # plt.hist(zdr,bins=numpy.linspace(0,0.14,100))

    # plt.figure(figsize=(10,10))
    # plt.quiver(
    #     df.xWokDitherFit.to_numpy(),df.yWokDitherFit.to_numpy(),zdx,zdy,angles="xy",units="xy", width=0.5, scale=0.003
    # )

    # df["zdx"] = zdx
    # df["zdy"] = zdy
    # df["zdr"] = zdr


    # # now group by robots
    # dfList = []
    # for name, group in df.groupby("positionerID"):
    #     # if len(set(group.designID)) < 4:
    #     #     continue
    #     group = group[["configID", "designID", "xWok", "yWok", "zdx", "zdy", "alphaMeas", "betaMeas", "alphaOffset", "betaOffset"]].groupby("designID").mean().reset_index()
    #     group["totalRot"] = -1*numpy.radians(group.alphaMeas+group.alphaOffset+group.betaMeas+group.betaOffset) + numpy.pi/2.
    #     group["bdx"] = group.zdx * numpy.cos(group.totalRot) - group.zdy*numpy.sin(group.totalRot)
    #     group["bdy"] = group.zdx * numpy.sin(group.totalRot) + group.zdy*numpy.cos(group.totalRot)

    #     # subtract the mean beta arm offset
    #     group["bdx2"] = group.bdx - numpy.mean(group.bdx)
    #     group["bdy2"] = group.bdy - numpy.mean(group.bdy)
    #     group["bdr2"] = numpy.sqrt(group.bdx2**2+group.bdy2**2)
    #     dfList.append(group)

    # dfBeta = pandas.concat(dfList)

    # plt.figure(figsize=(10,10))
    # plt.quiver(
    #     dfBeta.xWok.to_numpy(),dfBeta.yWok.to_numpy(),dfBeta.bdx,dfBeta.bdy,angles="xy",units="xy", width=0.5, scale=0.003
    # )

    # print("rms all", numpy.sqrt(numpy.mean(dfBeta.bdr2**2)))
    # plt.figure()
    # plt.hist(dfBeta.bdr2,bins=numpy.linspace(0,0.14,100))

    # plt.figure(figsize=(10,10))
    # plt.quiver(
    #     dfBeta.xWok.to_numpy(),dfBeta.yWok.to_numpy(),dfBeta.bdx2,dfBeta.bdy2,angles="xy",units="xy", width=0.5, scale=0.003
    # )

    # plt.figure(figsize=(10,10))
    # plt.quiver(
    #     df.xWokDitherFit,df.yWokDitherFit,df.dxBeta,df.dyBeta,angles="xy",units="xy", width=0.5, scale=0.1
    # )

    #



    # plt.show()
    # import pdb; pdb.set_trace()


def plotFVCdistortion(mjd=None):
    df = pandas.read_csv("ditherFit_all_merged.csv")
    if mjd is not None:
        df = df[df.mjd==mjd]
    nConfigs = len(set(df.configID))

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

    dfList = []
    dfFIFList = []
    dfXYTest = pandas.DataFrame()
    # rMax = numpy.max(df.rWokMetDithFit)
    rTest = numpy.random.uniform(0, RMAX**2, size=1000)
    thetaTest = numpy.random.uniform(0, numpy.pi*2, size=1000)
    xs = numpy.sqrt(rTest)*numpy.cos(thetaTest)
    ys = numpy.sqrt(rTest)*numpy.sin(thetaTest)
    for name, group in df.groupby(["configID"]):
        xyFVC = group[["x_fvc", "y_fvc"]].to_numpy()
        xyWok = group[["xWokMetDithFit", "yWokMetDithFit"]].to_numpy()

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
        polids, coeffs = fitZhaoBurge(
            xyFit[:,0],
            xyFit[:,1],
            xyWok[:,0],
            xyWok[:,1],
            # polids=numpy.arange(33),
            polids=POLIDS,
            normFactor=RMAX
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
        plt.title("rms = %.1f um"%rms)
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

    # df["rWok"] = numpy.sqrt(df.xWokMetDithFit**2+df.yWokMetDithFit**2)
    # df["thetaWok"] = numpy.arctan2(df.yWokMetDithFit, df.xWokMetDithFit)
    # df["drWok"] = numpy.cos(-1*df.thetaWok)*df.dxWok - numpy.sin(-1*df.thetaWok)*df.dyWok

    # plt.figure()
    # plt.plot(df.rWok, df.drWok, '.k', ms=1, alpha=0.2)
    # plt.xlabel("r wok (mm)")
    # plt.ylabel("dr wok (mm)")

    # X = numpy.array([df.rWok, df.rWok**2, df.rWok**3, df.rWok**4]).T

    # out = numpy.linalg.lstsq(X,df.drWok)

    # drFit = X @ out[0]
    # df["drFit"] = drFit
    # df = df.sort_values("rWok")

    # plt.plot(df.rWok, df.drFit, '-r')

    # df["dxWok2"] = df.dxWok - df.drFit * df.dxWok
    # df["dyWok2"] = df.dyWok - df.drFit * df.dyWok
    # df["drWok2"] = numpy.sqrt(df.dxWok2**2+df.dyWok2**2)

    # plt.figure(figsize=(8,8))
    # plt.quiver(
    #     df.xWokMetDithFit, df.yWokMetDithFit, df.dxWok2, df.dyWok2, angles="xy", units="xy", width=1, scale=0.03
    # )
    # rms = numpy.sqrt(numpy.mean(df.drWok2**2))/.06
    # plt.title("rms = %.1f arcsec"%rms)
    # plt.xlabel("x wok (mm)")
    # plt.ylabel("y wok (mm)")
    # plt.axis("equal")

    # dx, dy = fitZBs(
    #     df.xWokMetDithFit.to_numpy(),
    #     df.yWokMetDithFit.to_numpy(),
    #     df.dxWok.to_numpy(),
    #     df.dyWok.to_numpy()
    # )

    # df["zdx"] = dx
    # df["zdy"] = dy
    # df["zdr"] = numpy.sqrt(dx**2+dy**2)


    # plt.figure(figsize=(8,8))
    # plt.quiver(
    #     df.xWokMetDithFit, df.yWokMetDithFit, df.zdx, df.zdy, angles="xy", units="xy", width=1, scale=0.03
    # )
    # rms = numpy.sqrt(numpy.mean(df.zdr**2))/.06
    # plt.title("rms = %.1f arcsec"%rms)
    # plt.xlabel("x wok (mm)")
    # plt.ylabel("y wok (mm)")
    # plt.axis("equal")


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
    keepCols = ['site', 'holeID', 'id', 'xWok', 'yWok', 'zWok', 'col', 'row']
    fcm = calibration.fiducialCoords.reset_index()
    print(len(fcm), len(dfm))
    # fiducial F7 doesn't exist in measurements (faint)
    fcm = fcm.merge(dfm, how="left", on="holeID", suffixes=(None, "_m"))
    # not all fiducials have measurements some are broken (NaN values in xWokMeas)
    xWokSaved = fcm.xWok.to_numpy()
    yWokSaved = fcm.yWok.to_numpy()
    xWokNew = fcm.xWokMeas.to_numpy()
    yWokNew = fcm.yWokMeas.to_numpy()
    brokenInds = numpy.argwhere(numpy.isnan(xWokNew)).flatten()
    for idx in brokenInds:
        xWokNew[idx] = xWokSaved[idx]
        yWokNew[idx] = yWokSaved[idx]

    fcm["xWok"] = xWokNew
    fcm["yWok"] = yWokNew

    fcm = fcm[keepCols]
    fcm.to_csv("fiducialCoords_lco_july_2024.csv")
    plt.show()
    # import pdb; pdb.set_trace()


def reprocessFVC():
    dfAll = pandas.read_csv("ditherFit_all_merged.csv")
    fcm = pandas.read_csv("fiducialCoords_dither_updated.csv")
    fvcImg = dfAll[["fvcImgNum", "mjd"]].groupby(["fvcImgNum", "mjd"]).first().reset_index().sort_values(["mjd","fvcImgNum"])
    fvcImgNums = fvcImg.fvcImgNum.to_numpy()
    mjds = fvcImg.mjd.to_numpy()

    dfList = []
    for imgNum, mjd in zip(fvcImgNums,mjds):
        imgNumStr = str(imgNum).zfill(4)
        ff = fits.open("/Volumes/futa/apo/data/fcam/%i/proc-fimg-fvc1n-%s.fits"%(mjd, imgNumStr))
        posCoords = Table(ff["POSANGLES"].data).to_pandas()

        fvct = FVCTransformAPO(
            ff[1].data,
            posCoords,
            ff[1].header["IPA"],
            fiducialCoords=fcm,
            polids=POLIDS,
            zbNormFactor=RMAX
        )
        fvct.extractCentroids()
        fvct.fit(
            centType="sep"
        )

        _df = dfAll[(dfAll.fvcImgNum==imgNum) & (dfAll.mjd==mjd)]
        _df = _df.groupby(["positionerID"]).first().reset_index()

        _df = _df.merge(fvct.positionerTableMeas, on="positionerID", suffixes=(None, "_newFVC"))
        _df["x_newFVC"] = _df.x
        _df["y_newFVC"] = _df.y

        for kw in ["", "_newFVC"]:
            x = _df.xWokDitherFit.to_numpy()
            y = _df.yWokDitherFit.to_numpy()
            dx = _df["xWokMeasBOSS%s"%kw] - x
            dy = _df["yWokMeasBOSS%s"%kw] - y
            _df["dxWok%s"%kw] = dx
            _df["dyWok%s"%kw] = dy
        dfList.append(_df)

    df = pandas.concat(dfList)
    df.to_csv("dither_reprocess_fvc.csv", index=False)
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

def plotGFADistortion(mjd=None):
    df = pandas.read_csv("%i/dither_gfa_%i_lco.csv"%(mjd,mjd)).sort_values("gfaNum")

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
    for name, group in df.groupby("gfaNum"):
        print("name", name)
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

        group["dxWokFit"] = group.dxWok - dxOff
        group["dyWokFit"] = group.dyWok - dyOff
        group["drWokFit"] = numpy.sqrt(group.dxWokFit**2+group.dyWokFit**2)

        plt.figure()
        plt.hist(group.drWokFit, bins=100)
        plt.title(str(name))

        plt.figure(figsize=(8,8))
        plt.quiver(group.xWokPred, group.yWokPred, group.dxWokFit, group.dyWokFit, angles="xy", units="xy", scale=0.03)
        plt.axis("equal")
        plt.title(str(name))

        dfList.append(group)

    df = pandas.concat(dfList)

    ### apply dc xy offset to current gfaCoords
    # import pdb; pdb.set_trace()
    gfaCoords["xWok"] = gfaCoords.xWok - numpy.array(xGFAoff)
    gfaCoords["yWok"] = gfaCoords.yWok - numpy.array(yGFAoff)

    gfaCoords.to_csv("gfaCoords_lco_july_2024.csv")

    plt.figure()
    plt.hist(df.drWok, bins=100)

    plt.figure(figsize=(8,8))
    plt.quiver(df.xWokPred, df.yWokPred, df.dxWok, df.dyWok, angles="xy", units="xy", scale=0.001, width=0.1)
    plt.axis("equal")

    plt.figure(figsize=(8,8))
    plt.quiver(df.xWokPred, df.yWokPred, df.dxWokFit, df.dyWokFit, angles="xy", units="xy", scale=0.001, width=0.1)
    plt.axis("equal")

    plt.figure()
    plt.hist(df.drWokFit, bins=100)

    plt.show()

    import pdb; pdb.set_trace()


def plotPAvsDec():
    df = pandas.read_csv("ditherFit_all_merged.csv")
    plt.figure()
    plt.plot(df.SOL_PA, df.SOL_DEC, '.', ms=10)
    plt.ylabel("DEC (deg)")
    plt.xlabel("PA (deg)")
    plt.show()

    import pdb; pdb.set_trace()



    # plt.show()



if __name__ == "__main__":

    # merge_all()
    # plotAll()
    # plotAll(mjd=60448) # apo, good after fiducial fixes

    # plotAll(mjd=60229) # after baffle rotation
    # plotAll(mjd=60371) # after IMB change
    # plotAll(mjd=60520) # new (bad) mount lco
    # plotAll(mjd=60521) # mount loosened

    # plt.show()
    # update FIF locations
    # warning make sure correct wok calibs are setup (for the site)
    # plotFVCdistortion(mjd=60521) # writes new file for fiducial positions
    # reprocessFVC()
    # plotReprocessFVC()
    # plotPAvsDec()


    # look at GFA calib errors
    plotGFADistortion(mjd=60521)

    plt.show()
