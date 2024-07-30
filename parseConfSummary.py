import pandas

def parseFiberMapLine(line):
    # remove brackets from string (for ugriz mags)
    line = line.strip("\n")
    toks = line.replace("{", "").replace("}", "").split()

    # todo: parse these names from the file
    # rather than hardcode them in
    colsOld = [
        "FIBERMAP",
        "positionerId",
        "holeId",
        "fiberType",
        "assigned",
        "on_target",
        "valid",
        "decollided",
        "xwok",
        "ywok",
        "zwok",
        "xFocal",
        "yFocal",
        "alpha",
        "beta",
        "racat",
        "deccat",
        "pmra",
        "pmdec",
        "parallax",
        "ra",
        "dec",
        "lambda_design",
        "lambda_eff",
        "coord_epoch",
        "spectrographId",
        "fiberId",
        "u_mag",
        "g_mag",
        "r_mag",
        "i_mag",
        "z_mag",
        "optical_prov",
        "bp_mag",
        "gaia_g_mag",
        "rp_mag",
        "h_mag",
        "catalogid",
        "carton_to_target_pk",
        "cadence",
        "firstcarton",
        "program",
        "category",
        "sdssv_boss_target0",
        "sdssv_apogee_target0",
        "double delta_ra",
        "double delta_dec",
    ]

    colsNew = [
        "FIBERMAP",
        "positionerId",
        "holeId",
        "fiberType",
        "assigned",
        "on_target",
        "valid",
        "decollided",
        "too",
        "xwok",
        "ywok",
        "zwok",
        "xFocal",
        "yFocal",
        "alpha",
        "beta",
        "racat",
        "deccat",
        "pmra",
        "pmdec",
        "parallax",
        "ra",
        "dec",
        "double ra_observed",
        "double dec_observed",
        "double alt_observed",
        "double az_observed",
        "lambda_design",
        "lambda_eff",
        "coord_epoch",
        "spectrographId",
        "fiberId",
        "u_mag",
        "g_mag",
        "r_mag",
        "i_mag",
        "z_mag",
        "optical_prov",
        "bp_mag",
        "gaia_g_mag",
        "rp_mag",
        "h_mag",
        "catalogid",
        "carton_to_target_pk",
        "cadence",
        "firstcarton",
        "program",
        "category",
        "sdssv_boss_target0",
        "sdssv_apogee_target0",
        "double delta_ra",
        "double delta_dec",
    ]
    # import pdb; pdb.set_trace()
    if len(toks) == len(colsNew):
        cols = colsNew
    else:
        cols = colsOld

    castDict = {
        "positionerId": int,
        "holeId": str,
        "fiberId": int,
        "decollided": bool,
        "fiberType": str,
        "firstcarton": str,
        "racat": float,
        "deccat": float,
        "pmra": float,
        "pmdec": float,
        "parallax": float,
        "coord_epoch": float,
        "gaia_g_mag": float,
        "bp_mag": float,
        "rp_mag": float,
        "u_mag": float,
        "g_mag": float,
        "r_mag": float,
        "i_mag": float,
        "z_mag": float,
    }

    dd = {}
    # print(len(cols), len(toks))
    for col, val in zip(cols, toks):
        # print(col, val)
        # import pdb; pdb.set_trace()
        if col in castDict.keys():
            colName = col
            if col == "fiberId":
                colName = "fiberID"
            if col == "positionerId":
                colName = "positionerID"
            if col == "holeId":
                colName = "holeID"
            dd[colName] = castDict[col](val)

    return dd


def parseConfSummary(confFilePath):
    with open(confFilePath, "r") as f:
        lines = f.readlines()

    dictList = []
    for line in lines:
        line = line.strip("\n")
        if line.startswith("fvc_image_path"):
            fvcImgNum = int(line.split("-")[-1].strip(".fits"))
        if line.startswith("configuration_id"):
            confid = int(line.split()[-1])
        if line.startswith("design_id"):
            designid = int(line.split()[-1])
        if line.startswith("MJD"):
            mjd = int(line.split()[-1])
        if line.startswith("focal_scale"):
            focal_scale = float(line.split()[-1])
        if line.startswith("raCen"):
            field_cen_ra = float(line.split()[-1])
        if line.startswith("decCen"):
            field_cen_dec = float(line.split()[-1])
        if line.startswith("pa"):
            field_cen_pa = float(line.split()[-1])

        if line.startswith("FIBERMAP"):
            dictList.append(parseFiberMapLine(line))

    df = pandas.DataFrame(dictList)
    df["configID"] = confid
    df["designID"] = designid
    df["mjd"] = mjd
    df["focal_scale"] = focal_scale
    df["fvcImgNum"] = fvcImgNum
    df["field_cen_ra"] = field_cen_ra
    df["field_cen_dec"] = field_cen_dec
    df["field_cen_pa"] = field_cen_pa
    df = df[df.firstcarton == "manual_fps_position_stars_10"].reset_index(drop=True)
    # import pdb; pdb.set_trace()

    return df