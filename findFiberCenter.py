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


def minimizeMe1(x, starx, stary, flux, flux_ivar=1):
    # fit sigma
    amp, sigma, fibx, fiby = x
    fHats = []
    for _xs,_ys in zip(starx, stary):
        fHats.append(fractionalFlux(amp, sigma, _xs, _ys, fibx, fiby))
    fHats = numpy.array(fHats)
    return numpy.mean(flux_ivar*(fHats-flux)**2)


def minimizeMe2(x, amp, sigma, starx, stary, flux, flux_ivar=1):
    # pass sigma and amplitude
    fibx, fiby = x
    fHats = []
    for _xs,_ys in zip(starx, stary):
        fHats.append(fractionalFlux(amp, sigma, _xs, _ys, fibx, fiby))
    fHats = numpy.array(fHats)
    return numpy.mean(flux_ivar*(fHats-flux)**2)


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


def fitOneSet(xInit, starx, stary, flux, method="Powell", flux_ivar=1):
    # initial guess for fitter pick spot with most flux
    # amaxFlux = numpy.argmax(flux)
    # ampInit = flux[amaxFlux]
    # xFibInit = starx[amaxFlux]
    # yFibInit = stary[amaxFlux]


    # tstart = time.time()
    # # fit all parameters
    # xInit = numpy.array([ampInit, sigma, xFibInit, yFibInit])

    tstart = time.time()
    _minimizeMe = partial(minimizeMe1, starx=starx, stary=stary, flux=flux, flux_ivar=flux_ivar)
    minOut = minimize(_minimizeMe, xInit, method=method) #, options={"disp":True})
    fitAmp, fitSigma, fitFiberX, fitFiberY = minOut.x
    print("minimize took", time.time()-tstart)


    # _fitFiberX = []
    # _fitFiberY = []
    # if bossExpNums is not None:
    #     # LOOCV throwing out boss exposures one at a time
    #     _bossExps = []
    #     for bossExp in list(set(bossExpNums)):
    #         _bossExps.append(bossExp)
    #         keep = bossExpNums != bossExp
    #         _starx = starx[keep]
    #         _stary = stary[keep]
    #         _flux = flux[keep]
    #         if hasattr(flux_ivar, "__len__"):
    #             _flux_ivar = flux_ivar[keep]
    #         else:
    #             _flux_ivar = flux_ivar
    #         tstart = time.time()

    #         # xInit = numpy.array([fitFiberX, fitFiberY])
    #         # _minimizeMe = partial(minimizeMe2, amp=fitAmp, sigma=fitSigma, starx=_starx, stary=_stary, flux=_flux, flux_ivar=_flux_ivar)

    #         xInit = numpy.array([fitAmp, fitSigma, fitFiberX, fitFiberY])
    #         _minimizeMe = partial(minimizeMe1, starx=_starx, stary=_stary, flux=_flux, flux_ivar=_flux_ivar)

    #         minOut = minimize(_minimizeMe, xInit, method=method) #, options={"disp":True})
    #         print("quick minimize took", time.time()-tstart)
    #         _junk1, _junk2, x, y = minOut.x
    #         _fitFiberX.append(x)
    #         _fitFiberY.append(y)
            # import pdb; pdb.set_trace()

    # else:
    #     xInit = numpy.array([ampInit, xFibInit, yFibInit])
    #     _minimizeMe = partial(minimizeMe2, sigma=sigma, starx=starx, stary=stary, flux=flux)
    #     minOut = minimize(_minimizeMe, xInit, method=method) #, options={"disp":True})
    #     fitAmp, fitFiberX, fitFiberY = minOut.x
    #     fitSigma = sigma

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



def _plotOne(fitFiberX, fitFiberY, fitSigma, fitAmp, xStar, yStar, spectroflux, mjd, fiberID, configID, magStr, camera, site="apo"):
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

    if site == "apo":
        _scale = 60 # microns per arcsec
    elif site == "lco":
        _scale = 92.3 # microns per arcsec
    else:
        raise RuntimeError("must specify apo or lco as site")
    seeingStr = "      FWHM = %.1f arcsec"%(fitSigma*1000/_scale*2.355)
    sigStr = r"                $\sigma$" + " = %.0f "%(fitSigma*1000) + r"$\mu$m"
    fluxStr = r"               $f_o$" + " = %.2e "%fitAmp + r"e$^-$/sec"
    ctrStr = "fiber center = (%.3f, %.3f) mm"%(fitFiberX, fitFiberY)


    fitText = "\n".join([ctrStr, fluxStr,sigStr,seeingStr])

    mjdStr =     "      MJD = %i"%mjd
    camStr =     "   camera = %s"%(camera)
    fiberIDStr = "   fiber id = %i"%fiberID
    confIDStr =  "config id = %i"%configID
    # magStr = "star magnitude = %.2f"%(r_mag)
    infoText = "\n".join([mjdStr, camStr, fiberIDStr, confIDStr, magStr])

    ax1.legend(loc="upper right", title="flux\n" +r"(e$^-$/sec)")
    ax1.text(xMinMax[0]+.01, yMinMax[1]-.01, infoText, ha="left", va="top", color="white")
    ax1.text(xMinMax[0]+.01, yMinMax[0]+.01, fitText, ha="left", va="bottom", color="white")



def createIntegrationGrid(minSig, maxSig, nStepSig, minOff, maxOff, nStepOff):
    # sigma is gaussian sigma in mm
    # offset is star to fiber offset in mm
    sigs = numpy.linspace(minSig, maxSig, nStepSig)
    offs = numpy.linspace(minOff, maxOff, nStepOff)

    outArr = numpy.zeros((nStepSig,nStepOff))
    _sigs = []
    _offs = []
    _flux = []

    for ii, s in enumerate(sigs):
        for jj, o in enumerate(offs):
            amp = 1
            sigma = s
            starx = 0
            stary = 0
            fibx = 0
            fiby = o
            ptr_to_buffer = (ctypes.c_double * 6)(amp, sigma, starx, stary, fibx, fiby)
            user_data = ctypes.cast(ptr_to_buffer, ctypes.c_void_p)
            cIntegrand = LowLevelCallable(lib.f, user_data)
            A, err = dblquad(cIntegrand, 0, 2*numpy.pi, 0, FIBER_RAD)
            # _sigs.append(s)
            # _offs.append(o)
            # _flux.append(A)
            outArr[ii,jj] = A

    return sigs, offs, outArr




if __name__ == "__main__":
    # fitAll()
    # testOne(1037, 5301)
    # config options for 1037
    # {5315, 5316, 5189, 5192, 5194, 5259, 5260, 5144, 5146, 5148, 5286, 5287, 5288, 5165, 5167, 5169, 5300, 5302}
    # testOne(1037, 5912)
    # bad ones [5189, 5144]
    nsteps= 1000
    import time; tstart = time.time()
    sigs, offs, flux = createIntegrationGrid(0, .25, nsteps, 0, 1.5, nsteps)
    print("time took", time.time()-tstart)
    from scipy.interpolate import RegularGridInterpolator
    rgi = RegularGridInterpolator((sigs,offs), flux)

    testsig = 0.060
    testOff = 0.025

    tstart = time.time()
    Ahat = rgi((testsig,testOff))
    print("interp took", time.time()-tstart)

    tstart = time.time()
    ptr_to_buffer = (ctypes.c_double * 6)(1, testsig, 0, 0, 0, testOff)
    user_data = ctypes.cast(ptr_to_buffer, ctypes.c_void_p)
    cIntegrand = LowLevelCallable(lib.f, user_data)
    Aother, err = dblquad(cIntegrand, 0, 2*numpy.pi, 0, FIBER_RAD)
    print("integrate took", time.time()-tstart)

    print(Ahat, Aother, (Aother-Ahat)/Aother*100)



