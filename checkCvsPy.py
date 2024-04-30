import pandas
import glob
import matplotlib.pyplot as plt
import numpy
from skimage.transform import SimilarityTransform
from coordio.zhaoburge import fitZhaoBurge, getZhaoBurgeXY

dfPy = pandas.concat([pandas.read_csv(x) for x in glob.glob("ditherAnalysis/*.csv")])
dfC = pandas.concat([pandas.read_csv(x) for x in glob.glob("ctype_ditherAnalysis/*.csv")])

dfPy = dfPy.groupby(["mjd", "configID", "camera", "fiberID"]).first().reset_index()
dfC = dfC.groupby(["mjd", "configID", "camera", "fiberID"]).first().reset_index()

df = dfPy.merge(dfC, on=["mjd", "configID", "camera", "fiberID"], suffixes=("_py", "_c"))
df["dSig"] = df.sigmaWokDitherFit_py - df.sigmaWokDitherFit_c
df["dx"] = df.xWokDitherFit_py - df.xWokDitherFit_c
df["dy"] = df.yWokDitherFit_py - df.yWokDitherFit_c
df["dr"] = numpy.sqrt(df.dx**2+df.dy**2)
df["dAmp"] = df.fluxAmpDitherFit_py - df.fluxAmpDitherFit_c

# plt.figure()
# plt.hist(df.dSig, bins=100)
# plt.figure()
# plt.plot(df.sigmaWokDitherFit_c, df.sigmaWokDitherFit_py, '.k')

# plt.figure()
# plt.hist(df.dx, bins=100)
# plt.figure()
# plt.hist(df.dy, bins=100)


# plt.show()
# plt.plot(df.sigmaWokDitherFit_c, df.sigmaWokDitherFit_py, '.k')

dfC["dxBOSS"] = dfC.xWokMeasBOSS - dfC.xWokDitherFit
dfC["dyBOSS"] = dfC.yWokMeasBOSS - dfC.yWokDitherFit

# calcuate offset to metrology fiber
dxmet = dfC.xWokMeasMetrology - dfC.xWokMeasBOSS
dymet = dfC.yWokMeasMetrology - dfC.yWokMeasBOSS
dfC["xWokDitherFitMetrology"] = dfC.xWokDitherFit + dxmet
dfC["yWokDitherFitMetrology"] = dfC.yWokDitherFit + dymet

dfC["drBOSS"] = numpy.sqrt(dfC.dxBOSS**2+dfC.dyBOSS**2)
# plt.figure()
# plt.hist(dfC.drBOSS * 1000, bins=numpy.linspace(0,200,50))
# plt.show()

print(set(dfC.field_cen_ra), set(dfC.field_cen_dec), set(dfC.field_cen_pa))


dfFVC = pandas.read_csv("dither_fvc_60420.csv")[["fvcImgNum", "configID", "fvcRot", "fvcTransX", "fvcTransY", "fvcScale", "fvcIPA", "fvcALT", "fvcAZ"]]
dfFVC = dfFVC.groupby(["fvcImgNum", "configID", "fvcRot", "fvcTransX", "fvcTransY", "fvcScale", "fvcIPA", "fvcALT", "fvcAZ"]).first().reset_index()
print(dfFVC)

dfC = dfC.merge(dfFVC, on=["fvcImgNum", "configID"])

# for name, group in dfC.groupby(["configID", "camera"]):
#     plt.figure(figsize=(7,7))
#     plt.quiver(group.xWokDitherFit, group.yWokDitherFit, group.dxBOSS, group.dyBOSS, angles="xy", units="xy") #, scale=0.01)
#     plt.axis("equal")
#     plt.title("wok " + str(name))

#     st = SimilarityTransform(
#         translation=group[["fvcTransX", "fvcTransY"]].iloc[0],
#         rotation=numpy.radians(group.fvcRot.iloc[0]),
#         scale=group.fvcScale.iloc[0]
#     )

#     # xyWok1 = st(group[["x_fvc", "y_fvc"]].to_numpy())
#     # dxyWok = xyWok1 - group[["xWokDitherFit", "yWokDitherFit"]].to_numpy()
#     # plt.figure(figsize=(7,7))
#     # plt.quiver(group.xWokDitherFit, group.yWokDitherFit, dxyWok[:,0], dxyWok[:,1], angles="xy", units="xy", scale=0.01)
#     # plt.axis("equal")
#     # plt.title("wok 2 " + str(name))
#     print("Alt Az IPA", group.fvcALT.iloc[0], group.fvcAZ.iloc[0], group.fvcIPA.iloc[0])

#     # de-rotate dxy vectors and scale into ccd units
#     cosRot = numpy.cos(-1*numpy.radians(group.fvcRot.iloc[0]))
#     sinRot = numpy.sin(-1*numpy.radians(group.fvcRot.iloc[0]))
#     dxCCD = group.dxBOSS.to_numpy()*cosRot - group.dyBOSS.to_numpy()*sinRot
#     dyCCD = group.dxBOSS.to_numpy()*sinRot + group.dyBOSS.to_numpy()*cosRot
#     mm2pix = 1000/120


#     dxNudge = group.xNudge - group.x_fvc
#     dyNudge = group.yNudge - group.y_fvc
#     plt.figure(figsize=(7,7))
#     plt.quiver(group.x_fvc, group.y_fvc, dxCCD*mm2pix, dyCCD*mm2pix, color="black", angles="xy", units="xy")
#     plt.quiver(group.x_fvc, group.y_fvc, dxNudge, dyNudge, color="red", angles="xy", units="xy")
#     plt.axis("equal")
#     plt.title("fvc " + str(name))


    # xyFVCDitherFit = st.inverse(group[["xWokDitherFit", "yWokDitherFit"]].to_numpy())

    # import pdb; pdb.set_trace()

dfList = []
for name, group in dfC.groupby(["configID", "camera"]):
    xyWok = group[["xWokDitherFitMetrology", "yWokDitherFitMetrology"]].to_numpy()
    xyCCD = group[["x_fvc", "y_fvc"]].to_numpy()
    # xyCCD = group[["xNudge", "yNudge"]].to_numpy()

    st = SimilarityTransform()
    st.estimate(xyWok, xyCCD)
    xyFit = st(xyWok)
    dxy = xyFit - xyCCD

    group["xFitCCD"] = xyFit[:,0]
    group["yFitCCD"] = xyFit[:,1]
    group["dxCCD"] = dxy[:,0]
    group["dyCCD"] = dxy[:,1]
    group["drCCD"] = numpy.linalg.norm(dxy, axis=1)

    st = SimilarityTransform()
    st.estimate(xyCCD, xyWok)
    xyFit = st(xyCCD)
    dxy = xyFit - xyWok

    group["xFitWok"] = xyFit[:,0]
    group["yFitWok"] = xyFit[:,1]
    group["dxWok"] = dxy[:,0]
    group["dyWok"] = dxy[:,1]
    group["drWok"] = numpy.linalg.norm(dxy, axis=1)

    polids, coeffs = fitZhaoBurge(
        group.xFitWok.to_numpy(),
        group.yFitWok.to_numpy(),
        group.xWokDitherFit.to_numpy(),
        group.yWokDitherFit.to_numpy(),
        polids=numpy.arange(33),
        normFactor=1
    )

    dx, dy = getZhaoBurgeXY(
        polids,
        coeffs,
        group.xFitWok.to_numpy(),
        group.yFitWok.to_numpy(),
        normFactor=1
    )
    zxfit = group.xFitWok.to_numpy() + dx
    zyfit = group.yFitWok.to_numpy() + dy

    dx = zxfit - group.xWokDitherFit.to_numpy()
    dy = zyfit - group.yWokDitherFit.to_numpy()
    dr = numpy.sqrt(dx**2+dy**2)

    group["dxZBWok"] = dx
    group["dyZBWok"] = dy
    group["drZBWok"] = dr

    dfList.append(group)

df = pandas.concat(dfList)

plt.figure(figsize=(8,8))
plt.quiver(df.xWokDitherFit, df.yWokDitherFit, df.dxWok, df.dyWok, angles="xy", units="xy", scale=0.05)
plt.axis("equal")


plt.figure()
plt.hist(df.drWok, bins=50)

# polids, coeffs = fitZhaoBurge(
#     df.xFitWok.to_numpy(),
#     df.yFitWok.to_numpy(),
#     df.xWokDitherFit.to_numpy(),
#     df.yWokDitherFit.to_numpy(),
#     polids=numpy.arange(33),
#     normFactor=1
# )

# dx, dy = getZhaoBurgeXY(
#     polids,
#     coeffs,
#     df.xFitWok.to_numpy(),
#     df.yFitWok.to_numpy(),
#     normFactor=1
# )
# zxfit = df.xFitWok.to_numpy() + dx
# zyfit = df.yFitWok.to_numpy() + dy

# dx = zxfit - df.xWokDitherFit.to_numpy()
# dy = zyfit - df.yWokDitherFit.to_numpy()
# dr = numpy.sqrt(dx**2+dy**2)

plt.figure()
plt.hist(df.drZBWok, bins=50)

plt.figure(figsize=(8,8))
plt.quiver(df.xWokDitherFit, df.yWokDitherFit, df.dxZBWok, df.dyZBWok, angles="xy", units="xy", scale=0.05)
plt.axis("equal")


plt.show()


import pdb; pdb.set_trace()