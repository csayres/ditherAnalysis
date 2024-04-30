import numpy
from scipy.integrate import dblquad
from functools import partial
from scipy.optimize import minimize
import pandas
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from coordio.defaults import PLATE_SCALE
import time
from multiprocessing import Pool
import matplotlib as mpl
import os
from scipy import stats

import ctypes
from scipy import LowLevelCallable
lib = ctypes.CDLL(os.path.abspath("gausslib.so"))
lib.f.restype = ctypes.c_double
lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

# cIntegrand = LowLevelCallable(lib.f)

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
cp = sns.color_palette("flare_r")
cpMap = sns.color_palette("flare_r", as_cmap=True)

FIBER_RAD = 60/1000 # mm

# MM_PER_AS = PLATE_SCALE["APO"] / 3600. # mm/arcsec


def bivariateGaussian(x, y, amp, sigma, starx, stary, fibx, fiby):
    A = amp*(1/(2*numpy.pi*sigma*sigma))
    B = ((x + starx - fibx)/sigma)**2
    C = ((y + stary - fiby)/sigma)**2
    return A*numpy.exp(-0.5*(B+C))


def integrand(r, theta, amp, sigma, starx, stary, fibx, fiby):
    _x = r*numpy.cos(theta)
    _y = r*numpy.sin(theta)
    return r * bivariateGaussian(_x, _y, amp, sigma, starx, stary, fibx, fiby)


def fractionalFlux(amp, sigma, starx, stary, fibx, fiby):
    ##### python implementation (slower)#########
    # https://scipython.com/book/chapter-8-scipy/questions/dblquad-gotcha/
    # A, err = dblquad(integrand, 0, 2*numpy.pi, 0, FIBER_RAD, args=(amp, sigma, starx, stary, fibx, fiby))

    #### ctyptes implementation ########
    ptr_to_buffer = (ctypes.c_double * 6)(amp, sigma, starx, stary, fibx, fiby)
    user_data = ctypes.cast(ptr_to_buffer, ctypes.c_void_p)
    cIntegrand = LowLevelCallable(lib.f, user_data)
    A, err = dblquad(cIntegrand, 0, 2*numpy.pi, 0, FIBER_RAD) #, args=(amp, sigma, starx, stary, fibx, fiby))
    return A


def minimizeMe1(x, starx, stary, flux):
    # fit sigma
    amp, sigma, fibx, fiby = x
    fHats = []
    for _xs,_ys in zip(starx, stary):
        fHats.append(fractionalFlux(amp, sigma, _xs, _ys, fibx, fiby))
    fHats = numpy.array(fHats)
    return numpy.mean((fHats-flux)**2)


def minimizeMe2(x, sigma, starx, stary, flux):
    # pass sigma
    amp, fibx, fiby = x
    fHats = []
    for _xs,_ys in zip(starx, stary):
        fHats.append(fractionalFlux(amp, sigma, _xs, _ys, fibx, fiby))
    fHats = numpy.array(fHats)
    return numpy.mean((fHats-flux)**2)


# def minimizeMe3(x, starx, stary, flux):
#     amp, sigma, fibx, fiby = x
#     fHats = []
#     for _xs,_ys in zip(starx, stary):
#         loc = (_xs-fibx)**2/sigma**2 + (_ys-fiby)**2/sigma**2
#         f = stats.ncx2.cdf(amp**2/sigma**2, nc=loc, df=2)
#         fHats.append(f)
#     fHats = numpy.array(fHats)
#     return numpy.mean((fHats-flux)**2)


def weightedCenter(starx, stary, flux):
    asort = numpy.argsort(flux)
    flux = flux[asort][-4:]
    fluxNorm = flux/numpy.sum(flux)
    starx = starx[asort][-4:]
    stary = stary[asort][-4:]

    meanx = numpy.sum(starx*fluxNorm)
    meany = numpy.sum(stary*fluxNorm)

    return starx[-1], stary[-1]
    # return meanx, meany


def fitOneSet(starx, stary, flux, sigma, fitSigma=False, method="Powell"):
    # initial guess for fitter pick spot with most flux
    amaxFlux = numpy.argmax(flux)
    ampInit = flux[amaxFlux]
    xFibInit = starx[amaxFlux]
    yFibInit = stary[amaxFlux]

    tstart = time.time()
    if fitSigma:
        xInit = numpy.array([ampInit, sigma, xFibInit, yFibInit])
        _minimizeMe = partial(minimizeMe1, starx=starx, stary=stary, flux=flux)
        minOut = minimize(_minimizeMe, xInit, method=method, options={"disp":True})
        fitAmp, fitSigma, fitFiberX, fitFiberY = minOut.x
    else:
        xInit = numpy.array([ampInit, xFibInit, yFibInit])
        _minimizeMe = partial(minimizeMe2, sigma=sigma, starx=starx, stary=stary, flux=flux)
        minOut = minimize(_minimizeMe, xInit, method=method, options={"disp":True})
        fitAmp, fitFiberX, fitFiberY = minOut.x
        fitSigma = sigma
    print("minimize took", time.time()-tstart)
    print(minOut)
    return fitAmp, fitSigma, fitFiberX, fitFiberY


def plotFlux(positionerId, configID, df):
    plt.figure()
    _df = df[(df.positionerId==positionerId)&(df.configID==configID)]
    sns.scatterplot(x="xOff", y="yOff", hue="spectroflux", data=_df, palette=cpMap)
    plt.axis("equal")


def plotContour(ax, xStar, yStar, sigma, amp, xMin=-.23, xMax=.23, yMin=-.23, yMax=.23, npts=75):

    x = numpy.linspace(xMin, xMax, npts) + xStar
    y = numpy.linspace(yMin, yMax, npts) + yStar
    xx, yy = numpy.meshgrid(x,y)
    g = []

    tstart = time.time()
    for _xx, _yy in zip(xx.flatten(),yy.flatten()):
        g.append(fractionalFlux(amp, sigma, _xx, _yy, xStar, yStar))
    g = numpy.array(g).reshape(npts,npts)
    vmax=numpy.max(g)
    vmin=numpy.min(g)
    ax.contourf(x, y, g, levels=75, vmin=vmin, vmax=vmax, cmap=cpMap)
    print("surface plot took", time.time()-tstart)
    xMin = numpy.min(x) + 0.02
    xMax = numpy.max(x) - 0.02
    yMin = numpy.min(y) + 0.02
    yMax = numpy.max(y) - 0.02
    return (vmin, vmax), (xMin,xMax), (yMin,yMax)

    # plt.axis("equal")


# def fitOne(posIdConfigID, returnVals=False):
#     positionerId, configID = posIdConfigID
#     try:
#         os.nice(5)

#         df = pandas.read_csv("holtzScrapeMerged.csv")
#         df = df[(df.positionerId==positionerId)&(df.configID==configID)]
#         # make sure we actually have dithers to fit!
#         if len(df) < 5:
#             return
#         xstar = df.xOff.to_numpy()
#         ystar = df.yOff.to_numpy()
#         flux = df.spectroflux.to_numpy()
#         sigma = numpy.median(df.sigmaGFA)
#         fitAmp, fitSigma, fitFiberX, fitFiberY = fitOneSet(xstar,ystar,flux,sigma,fitSigma=True)
#         if returnVals:
#             return fitAmp, fitSigma, fitFiberX, fitFiberY
#         else:
#             # write a new csv
#             df["fitAmp"] = fitAmp
#             df["fitSigma"] = fitSigma
#             df["fitFiberX"] = fitFiberX
#             df["fitFiberY"] = fitFiberY
#             df.to_csv("fitPositioner_%i_%i.csv"%(positionerId, configID))
#     except:
#         print(positionerId, configID, "fit failed")


# def fitAll():
#     df = pandas.read_csv("holtzScrapeMerged.csv")

#     _posIdConfigID = []
#     confList = list(set(df.configID))
#     for conf in confList:
#         _df = df[df.configID==conf]
#         posList = list(set(_df.positionerId))
#         for posId in posList:
#             _posIdConfigID.append([posId,conf])

#     p = Pool(30)
#     p.map(fitOne, _posIdConfigID)
#     p.close()


def _plotOne(fitFiberX, fitFiberY, fitSigma, fitAmp, xStar, yStar, spectroflux, mjd, fiberID, configID, r_mag, camera):
    plt.figure(figsize=(8,8))
    ax1 = plt.gca()

    hueNorm,xMinMax,yMinMax = plotContour(ax1, fitFiberX, fitFiberY, fitSigma, fitAmp)
    sns.scatterplot(ax=ax1, x=xStar, y=yStar, hue=spectroflux, s=100, palette=cpMap, hue_norm=hueNorm)
    ax1.set_aspect("equal")

    ax1.set_xlabel("x wok (mm)")

    ax1.set_ylabel("y wok (mm)")


    ax1.set_ylim(yMinMax)
    ax1.set_xlim(xMinMax)
    ax1.axhline(fitFiberY, ls="--", color="white", alpha=0.5)
    ax1.axvline(fitFiberX, ls="--", color="white", alpha=0.5)
    # dxy = ((xStar-fitFiberX))
    # plt.title("%i %i sigma=%.3f flux=%.0f"%(TEST_ROBOT, TEST_CONFIG, fitSigma, fitAmp))
    # dxyStr = "fit results\n-----------\nstar dxy = [%.0f, %.0f]"%(fitFiberX*1000,fitFiberY*1000) + r" $\mu$m"
    sigStr = r"$\sigma$" + " = %.3f "%(fitSigma*1000) + r"$\mu$m"
    seeingStr = "FWHM = %.1f arcsec"%(fitSigma*1000/60*2.355)
    fluxStr = r"$f_o$" + " = %.0f "%fitAmp + r"e$^-$/sec"


    fitText = "\n".join([fluxStr,sigStr,seeingStr])

    mjdStr = "MJD = %i"%mjd
    camStr = "camera = %s"%(camera)
    fiberIDStr = "fiber id = %i"%fiberID
    confIDStr = "configuration id = %i"%configID
    magStr = "star magnitude = %.2f"%(r_mag)
    infoText = "\n".join([mjdStr, camStr, fiberIDStr, confIDStr, magStr])

    ax1.legend(loc="upper right", title="flux\n" +r"(e$^-$/sec)")
    ax1.text(xMinMax[0]+.01, yMinMax[1]-.01, infoText, ha="left", va="top", color="white")
    ax1.text(xMinMax[0]+.01, yMinMax[0]+.01, fitText, ha="left", va="bottom", color="white")

    # without the color background
    # plt.savefig("fluxSolve_%i_%i.png"%(TEST_ROBOT,TEST_CONFIG), dpi=200)

    # plt.figure(figsize=(8,8))
    # ax = plt.gca()
    # hueNorm = plotContour(ax, fitFiberX, fitFiberY, fitSigma, fitAmp)
    # sns.scatterplot(ax=ax2, x="xOff", y="yOff", hue="spectroflux", s=100, data=_hc, palette=cpMap, hue_norm=hueNorm)
    # ax2.set_aspect("equal")
    # if xaxis:
    #     ax2.set_xlabel("dx (mm)")
    # else:
    #     ax2.set_xticks([])
    #     x_axis = ax2.axes.get_xaxis()
    #     label = x_axis.get_label()
    #     label.set_visible(False)
    # if yaxis:
    #     ax2.set_ylabel("dy (mm)")
    # else:
    #     ax2.set_yticks([])
    #     y_axis = ax2.axes.get_yaxis()
    #     label = y_axis.get_label()
    #     label.set_visible(False)

    # ax2.set_xlim(xMinMax)
    # ax2.set_ylim(yMinMax)
    # ax2.legend(loc="upper right", title="flux\n"+r"(e$^-$/sec)")
    # ax2.text(xMinMax[0]+.01, yMinMax[1]-.01, infoText, ha="left", va="top", color="black")
    # # ax2.set_savefig("fluxData_%i_%i.png"%(TEST_ROBOT,TEST_CONFIG), dpi=200)

# def testOne(TEST_ROBOT, TEST_CONFIG, ax1, ax2,xaxis=True,yaxis=True):
#     hc = pandas.read_csv("holtzScrape.csv")
#     hc["configID"] = hc.configurationId
#     # convert cherno offsets to mm offsets
#     xOff = hc.dChernoDec * MM_PER_AS
#     yOff = hc.dChernoRA * MM_PER_AS
#     hc["xOff"] = xOff
#     hc["yOff"] = yOff

#     files = glob.glob("stage1/configImgNum/dither*.csv")

#     mc = pandas.concat([pandas.read_csv(x) for x in files])
#     mc = mc.drop_duplicates()
#     mc["fiberID"] = mc.fiberId

#     validConfigs = list(set(hc.configID))
#     mc = mc[mc.configID.isin(validConfigs)]

#     # print("len before", len(hc))
#     _dfList = []
#     for config in validConfigs:
#         _a = hc[hc.configID==config].copy()
#         _b = mc[mc.configID==config]

#         # keep only fibers on target
#         sb = set(_b.fiberID)
#         _a = _a[_a.fiberID.isin(list(sb))]
#         _dfList.append(_a)

#     hc = pandas.concat(_dfList)


#     hc = hc.merge(mc, on=["configID", "sciImgNum", "fiberID"], suffixes=(None, "_mc"))
#     hc.to_csv("holtzScrapeMerged.csv", index=False)

#     # print("robot options", set(hc.positionerId))

#     # hc1 = hc[hc.positionerId==TEST_ROBOT]
#     # hc1 = hc1[hc1.configID==TEST_CONFIG]

#     # plotFlux(TEST_ROBOT, TEST_CONFIG, hc)

#     _hc = hc[(hc.positionerId == TEST_ROBOT) & (hc.configID == TEST_CONFIG)]

#     vmin = numpy.min(_hc.spectroflux.to_numpy())
#     vmax = numpy.max(_hc.spectroflux.to_numpy())
#     hueNorm = [vmin, vmax]


#     xstar = _hc.xOff.to_numpy()
#     ystar = _hc.yOff.to_numpy()
#     flux = _hc.spectroflux.to_numpy()
#     sigma = numpy.median(_hc.sigmaGFA)

#     tstart = time.time()
#     out = fitOne([TEST_ROBOT, TEST_CONFIG], returnVals=True)
#     if out is None:
#         return
#     fitAmp, fitSigma, fitFiberX, fitFiberY = out
#     print("fitSigma", TEST_ROBOT, TEST_CONFIG, fitSigma, fitAmp)
#     print("minimize took", time.time()-tstart)

#     # plt.figure(figsize=(8,8))
#     # ax = plt.gca()
#     hueNorm,xMinMax,yMinMax = plotContour(ax1, fitFiberX, fitFiberY, fitSigma, fitAmp)
#     sns.scatterplot(ax=ax1, x="xOff", y="yOff", hue="spectroflux", s=100, data=_hc, palette=cpMap, hue_norm=hueNorm)
#     ax1.set_aspect("equal")
#     if xaxis:
#         ax1.set_xlabel("dx (mm)")
#     else:
#         ax1.set_xticks([])
#         x_axis = ax1.axes.get_xaxis()
#         label = x_axis.get_label()
#         label.set_visible(False)
#     if yaxis:
#         ax1.set_ylabel("dy (mm)")
#     else:
#         ax1.set_yticks([])
#         y_axis = ax1.axes.get_yaxis()
#         label = y_axis.get_label()
#         label.set_visible(False)

#     ax1.set_ylim(yMinMax)
#     ax1.set_xlim(xMinMax)
#     ax1.axhline(fitFiberY, ls="--", color="white", alpha=0.5)
#     ax1.axvline(fitFiberX, ls="--", color="white", alpha=0.5)
#     # plt.title("%i %i sigma=%.3f flux=%.0f"%(TEST_ROBOT, TEST_CONFIG, fitSigma, fitAmp))
#     dxyStr = "fit results\n-----------\nstar dxy = [%.0f, %.0f]"%(fitFiberX*1000,fitFiberY*1000) + r" $\mu$m"
#     sigStr = r"$\sigma$" + " = %.3f "%(fitSigma*1000) + r"$\mu$m"
#     seeingStr = "FWHM = %.1f arcsec"%(fitSigma*1000/60*2.355)
#     fluxStr = r"$f_o$" + " = %.0f "%fitAmp + r"e$^-$/sec"


#     fitText = "\n".join([dxyStr,fluxStr,sigStr,seeingStr])

#     mjdStr = "MJD = %i"%_hc.mjd.to_numpy()[0]
#     fiberIDStr = "fiber id = %i"%_hc.fiberID.to_numpy()[0]
#     confIDStr = "configuration id = %i"%TEST_CONFIG
#     magStr = "star magnitude = %.2f"%(_hc.mag_r.to_numpy()[0])
#     infoText = "\n".join([mjdStr, fiberIDStr, confIDStr, magStr])

#     ax1.legend(loc="upper right", title="flux\n" +r"(e$^-$/sec)")
#     ax1.text(xMinMax[0]+.01, yMinMax[1]-.01, infoText, ha="left", va="top", color="white")
#     ax1.text(xMinMax[0]+.01, yMinMax[0]+.01, fitText, ha="left", va="bottom", color="white")
#     # plt.savefig("fluxSolve_%i_%i.png"%(TEST_ROBOT,TEST_CONFIG), dpi=200)

#     # plt.figure(figsize=(8,8))
#     # ax = plt.gca()
#     # hueNorm = plotContour(ax, fitFiberX, fitFiberY, fitSigma, fitAmp)
#     sns.scatterplot(ax=ax2, x="xOff", y="yOff", hue="spectroflux", s=100, data=_hc, palette=cpMap, hue_norm=hueNorm)
#     ax2.set_aspect("equal")
#     if xaxis:
#         ax2.set_xlabel("dx (mm)")
#     else:
#         ax2.set_xticks([])
#         x_axis = ax2.axes.get_xaxis()
#         label = x_axis.get_label()
#         label.set_visible(False)
#     if yaxis:
#         ax2.set_ylabel("dy (mm)")
#     else:
#         ax2.set_yticks([])
#         y_axis = ax2.axes.get_yaxis()
#         label = y_axis.get_label()
#         label.set_visible(False)

#     ax2.set_xlim(xMinMax)
#     ax2.set_ylim(yMinMax)
#     ax2.legend(loc="upper right", title="flux\n"+r"(e$^-$/sec)")
#     ax2.text(xMinMax[0]+.01, yMinMax[1]-.01, infoText, ha="left", va="top", color="black")
#     # ax2.set_savefig("fluxData_%i_%i.png"%(TEST_ROBOT,TEST_CONFIG), dpi=200)

#     # ax.plot(fitFiberX, fitFiberY, 'xr')

#     # import pdb; pdb.set_trace()

#     # plt.show()
#     # plt.close("all")
#     # df = hc.merge(mc, on=["configID", "fiberID"])

#     # import pdb; pdb.set_trace()

# def plotOneFiber():
#     fig1, axs1 = plt.subplots(2,2,figsize=(10,10))
#     fig2, axs2 = plt.subplots(2,2,figsize=(10,10))

#     axs1 = axs1.flatten()
#     axs2 = axs2.flatten()

#     for ii,config in enumerate([5287,5301,5288,5316]):
#         if ii == 0:
#             xaxis=False
#             yaxis=True
#         elif ii == 1:
#             xaxis=False
#             yaxis=False
#         elif ii == 2:
#             xaxis=True
#             yaxis=True
#         else:
#             xaxis=True
#             yaxis=False
#         ax1 = axs1[ii]
#         ax2 = axs2[ii]
#         testOne(1037, config,ax1,ax2,xaxis,yaxis)

#     fig1.tight_layout()
#     fig2.tight_layout()
#     fig1.savefig("indDitherFit.pdf", bbox_inches="tight")
#     fig2.savefig("indDitherData.pdf", bbox_inches="tight")

#     plt.show()

if __name__ == "__main__":
    # fitAll()
    # testOne(1037, 5301)
    # config options for 1037
    # {5315, 5316, 5189, 5192, 5194, 5259, 5260, 5144, 5146, 5148, 5286, 5287, 5288, 5165, 5167, 5169, 5300, 5302}
    # testOne(1037, 5912)
    # bad ones [5189, 5144]
    pass


