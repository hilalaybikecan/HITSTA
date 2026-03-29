"""Utilities to parse HITSTA measurement files and compute performance metrics."""
import pandas as pd
from io import StringIO
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import ruptures as rpt
from scipy.special import gammainc, gamma

__all__ = ["import_HITSTA"]


def import_HITSTA(filenames):
    """
    Parse one or more HITSTA measurement files and compute derived metrics.

    Parameters
    ----------
    filenames : str or list[str]
        Path(s) to the HITSTA data files.

    Returns
    -------
    dict
        Nested dictionary keyed by sample ID with computed metrics.
    """

    def bandedge_func(x, a_const, b_const, l_const):
        return a_const * (0.5 - 0.5 * np.tanh((x - b_const) / l_const))

    def reflectance_func(
        x,
        a_const,
        b_const,
        l_const,
        a2_const,
        b2_const,
        l2_const,
        c_const,
    ):
        return bandedge_func(x, a_const, b_const, l_const) + bandedge_func(
            x, a2_const, b2_const, l2_const
        ) + c_const

    def exp_func(x, a_const, t_const, c_const):
        return a_const * np.exp(-x / t_const) + c_const

    def ex_func(x, a_const, t_const, c_const):
        return -a_const * np.exp(-x / t_const) + a_const + c_const

    def expstretch_func(x, a_const, t_const, c_const, r_const):
        return a_const * np.exp(-(x / t_const) ** r_const) + c_const

    def stretched_exp_definite_integral(t0, t1, tau, beta, A=1.0):
        """Definite integral of A * exp(-(t/tau)**beta) from t0 to t1."""
        x0 = (np.array(t0) / tau) ** beta
        x1 = (np.array(t1) / tau) ** beta
        pre = tau / beta
        gamma_term = gamma(1 / beta)
        return A * pre * gamma_term * (gammainc(1 / beta, x1) - gammainc(1 / beta, x0))

    def linear_func(x, t_const, b_const):
        return x / t_const + b_const

    def plfit_func(x, ashort_const, tshort_const, along_const, tlong_const):
        return ashort_const * np.exp(-x / tshort_const) + along_const * (x**0.5) * np.exp(
            -x / tlong_const
        )

    def smooth(y, N):
        ysmooth = []
        for row in y:
            ysmooth.append(np.convolve(row, np.ones(N) / N, mode="same"))
        return np.array(ysmooth)

    def gaussian_func(wavelengths, A, mean, std, constant):
        return A * np.exp(-(wavelengths - mean) ** 2 / std**2) + constant

    def _parse_section(section):
        section = section.strip("\t\n")
        section = section.replace(",", ".")
        ID = section.split("\n")[0]
        csvStringIO = StringIO(section)
        dataframe = pd.read_csv(csvStringIO, delimiter="\t", header=1)
        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
        return ID, dataframe

    if isinstance(filenames, str):
        filenames = [filenames]

    if not filenames:
        raise ValueError("No filenames provided.")

    all_sections = []
    for filename in filenames:
        with open(filename, errors="ignore") as f:
            contents = f.read()
        sections = contents.split("#")
        all_sections.append(sections)

    sections_merged = [all_sections[0][0]]  # header from first file
    for i in range(1, len(all_sections[0])):  # skip header
        for j in range(0, len(filenames)):
            if j == 0:
                IDstring, df = _parse_section(all_sections[0][i])
            else:
                _, df_add = _parse_section(all_sections[j][i])
                df_add["Time (h)"] = df_add["Time (h)"] + df["Time (h)"].iloc[-1]
                df = pd.concat([df, df_add])
        sections_merged.append(IDstring + "\n" + df.to_csv(sep="\t"))

    exp = {}
    for section in sections_merged[1:50]:
        IDstring, df = _parse_section(section)
        wavelengths = df.columns[4:].to_numpy(dtype=float)
        wavelengths = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths))
        exp[IDstring] = {
            "Rounds": df[df["Type (D/W/L)"] == "D"].to_numpy()[:, 0] - 1,
            "Times": df[df["Type (D/W/L)"] == "D"]["Time (h)"].to_numpy(),
            "Wavelengths": wavelengths,
            "Dark Raw": df[df["Type (D/W/L)"] == "D"].iloc[:, 4:].to_numpy(dtype=float)
            / np.tile(df[df["Type (D/W/L)"] == "D"].iloc[:, 3].to_numpy(dtype=float), (1599, 1)).transpose(),
            "Reflectance Raw": df[df["Type (D/W/L)"] == "W"].iloc[:, 4:].to_numpy(dtype=float)
            / np.tile(df[df["Type (D/W/L)"] == "W"].iloc[:, 3].to_numpy(dtype=float), (1599, 1)).transpose(),
            "Laser Raw": df[df["Type (D/W/L)"] == "L"].iloc[:, 4:].to_numpy(dtype=float)
            / np.tile(df[df["Type (D/W/L)"] == "L"].iloc[:, 3].to_numpy(dtype=float), (1599, 1)).transpose(),
        }

    for IDstring in exp.keys():
        wavelengths = exp[IDstring]["Wavelengths"]
        inds = (wavelengths > 650) & (wavelengths < 850)
        wavelengths_cut = wavelengths[inds]

        exp[IDstring]["PL"] = exp[IDstring]["Laser Raw"] - exp[IDstring]["Dark Raw"]
        inds_PLsubtr = wavelengths > 920
        exp[IDstring]["PL Subtr"] = exp[IDstring]["PL"] - np.tile(
            np.expand_dims(np.mean(exp[IDstring]["PL"][:, inds_PLsubtr], axis=1), 1),
            (1, 1599),
        )
        exp[IDstring]["PL Peak Intensity"] = np.max(exp[IDstring]["PL Subtr"], axis=1)
        exp[IDstring]["PL Peak Wavelength"] = exp[IDstring]["Wavelengths"][
            np.argmax(exp[IDstring]["PL Subtr"], axis=1)
        ]
        exp[IDstring]["PL Fitted"] = []
        exp[IDstring]["PL Fit Parameters"] = []

        for idt, PL_measurement in enumerate(exp[IDstring]["PL Subtr"]):
            try:
                if idt == 0:
                    popt, pcov = curve_fit(
                        gaussian_func,
                        wavelengths_cut,
                        PL_measurement[inds],
                        maxfev=20000,
                        p0=[2, 750, 100, 0],
                        bounds=([0, 500, 10, -0.005], [200, 900, 150, 0.005]),
                    )
                else:
                    popt, pcov = curve_fit(
                        gaussian_func,
                        wavelengths_cut,
                        PL_measurement[inds],
                        maxfev=20000,
                        p0=popt,
                        bounds=([0, 500, 10, -0.005], [200, 900, 150, 0.005]),
                    )
                exp[IDstring]["PL Fitted"].append(gaussian_func(wavelengths, *popt))
                exp[IDstring]["PL Fit Parameters"].append(popt)
            except Exception:
                print([IDstring, " failed PL fit"])
        exp[IDstring]["PL Fitted"] = np.array(exp[IDstring]["PL Fitted"])
        exp[IDstring]["PL Fit Parameters"] = np.array(exp[IDstring]["PL Fit Parameters"])

        p = np.array(exp[IDstring]["PL Fit Parameters"], dtype=object)
        first = np.array(p[0], dtype=float) if p.size else np.array([])
        exp[IDstring]["Bandgap (initial PL)"] = 1240 / first[1] if first.size >= 2 and first[0] > 1 else np.nan

        PL = exp[IDstring]["PL Subtr"]
        PL0 = np.tile(exp[IDstring]["PL Subtr"][0, :], (len(exp[IDstring]["Times"]), 1))
        exp[IDstring]["PL Self-Similarity"] = np.sum(PL * PL0, axis=1) / (
            np.sqrt(np.sum(PL0 * PL0, axis=1)) * np.sqrt(np.sum(PL * PL, axis=1))
        )

        exp[IDstring]["Reflectance Raw Subtr"] = exp[IDstring]["Reflectance Raw"] - exp[IDstring]["Dark Raw"]
        Refl_denominator = np.abs(exp["ID2"]["Reflectance Raw"] - exp["ID2"]["Dark Raw"] + 0.01)
        exp[IDstring]["Reflectance"] = exp[IDstring]["Reflectance Raw Subtr"] / Refl_denominator
        Reflectance = exp[IDstring]["Reflectance"][0]
        Rmid = 0.5 * (np.max(Reflectance[inds]) + np.min(Reflectance[inds]))
        index_bandedge = np.argmin(np.abs(Reflectance[inds] - Rmid))
        wavelength_bandedge = wavelengths_cut[index_bandedge]
        R0_bandedge = Reflectance[inds][index_bandedge]
        exp[IDstring]["Bandedge"] = (wavelength_bandedge, R0_bandedge)
        exp[IDstring]["Rmid"] = Rmid
        bandedge_interval = 100
        R_slopes = []
        for Reflectance in exp[IDstring]["Reflectance"]:
            x = wavelengths_cut[
                max([(index_bandedge - bandedge_interval), 0]) : min(
                    [(index_bandedge + bandedge_interval), len(wavelengths_cut)]
                )
            ]
            y = Reflectance[inds][
                max([(index_bandedge - bandedge_interval), 0]) : min(
                    [(index_bandedge + bandedge_interval), len(wavelengths_cut)]
                )
            ]
            y = np.array(y, dtype=float)
            regression = stats.linregress(x, y)
            if np.isnan(regression.slope):
                print("nan value")
            R_slopes.append(regression.slope)
        exp[IDstring]["R_slopes (raw)"] = np.array(R_slopes)
        exp[IDstring]["R_slopes (norm.)"] = np.array(R_slopes) / R_slopes[0]
        exp[IDstring]["Eg (estimate)"] = 1240 / (wavelength_bandedge - Rmid / R_slopes[0])
        exp[IDstring]["Eg (estimate)"] = 1240 / (wavelength_bandedge - 50)

        R = exp[IDstring]["Reflectance"][:, inds]
        R0 = np.tile(exp[IDstring]["Reflectance"][0, inds], (len(exp[IDstring]["Times"]), 1))
        exp[IDstring]["R Self-Similarity"] = np.sum(R * R0, axis=1) / (
            np.sqrt(np.sum(R0 * R0, axis=1)) * np.sqrt(np.sum(R * R, axis=1))
        )

        ind_min = np.argmax(exp[IDstring]["Wavelengths"] > 520)
        ind_max = np.argmax(exp[IDstring]["Wavelengths"] > 560)
        exp[IDstring]["Short-Wavelength Step"] = exp[IDstring]["Reflectance"][:, ind_max] - exp[IDstring][
            "Reflectance"
        ][:, ind_min]

    for IDstring in exp.keys():
        X = exp[IDstring]["Times"]
        Y = exp[IDstring]["Short-Wavelength Step"]
        try:
            popt, pcov = curve_fit(ex_func, X, Y, p0=[0.1, 5, 0.05], bounds=([0, 0.05, 0], [0.5, 1000, 0.5]))
            exp[IDstring]["SWS Fit Covariance"] = pcov
            exp[IDstring]["SWS Fit Parameters"] = popt
            exp[IDstring]["SWS Fit"] = (X, ex_func(X, *popt))
        except Exception as e:
            exp[IDstring]["SWS Fit Covariance"] = np.nan
            exp[IDstring]["SWS Fit Parameters"] = [np.nan] * 2
            exp[IDstring]["SWS Fit"] = [np.nan] * 2
            print("SWS fit failed on", IDstring, "with exception:", e)
        SWS_score = 100 * (1 - Y / 0.6)
        if SWS_score[0] < 80:
            exp[IDstring]["T80_SWS_score"] = 0
        elif all(SWS_score > 80):
            popt, pcov = curve_fit(linear_func, X, SWS_score, p0=[-100, 100], bounds=([-1000, 90], [-50, 100]))
            exp[IDstring]["T80_SWS_score"] = (80 - popt[1]) * popt[0]
        else:
            exp[IDstring]["T80_SWS_score"] = X[np.argmax(SWS_score < 80)]

        PL = exp[IDstring]["PL Subtr"]
        P = np.atleast_2d(np.asarray(exp[IDstring]["PL Fit Parameters"], dtype=float))
        PL_arr = np.asarray(PL)
        skip = int(np.argmax(PL_arr)) if PL_arr.dtype == bool else int(PL_arr.ravel()[0])

        P = np.atleast_2d(np.asarray(exp[IDstring]["PL Fit Parameters"], dtype=float))
        skip = min(skip, P.shape[0] - 1)
        ind_min = int(np.nanargmax(P[skip:, 0]) + skip)

        ind_max = len(exp[IDstring]["Times"])
        X = exp[IDstring]["Times"][ind_min:ind_max] - exp[IDstring]["Times"][ind_min]
        Y = exp[IDstring]["PL Fit Parameters"][ind_min:ind_max, 0]
        Y_log = np.log(exp[IDstring]["PL Fit Parameters"][:, 0]).reshape(-1, 1)
        try:
            model = "l2"
            algo = rpt.Pelt(model=model).fit(Y_log)
            sigma2 = np.var(Y_log[-len(Y_log) // 4 :])
            pen = 3 * sigma2 * np.log(len(Y_log))
            bkps = algo.predict(pen=pen)
            ind_min = bkps[0]

            X = exp[IDstring]["Times"][ind_min:ind_max] - exp[IDstring]["Times"][ind_min]
            Y = exp[IDstring]["PL Fit Parameters"][ind_min:ind_max, 0]
            popt, pcov = curve_fit(
                expstretch_func,
                X,
                Y,
                p0=[10, 10, 0.5, 0.75],
                bounds=([0, 0, 0.49, 0.5], [100, 1000, 0.51, 1]),
            )
            exp[IDstring]["PL Lifetime Fit Parameters"] = popt
            exp[IDstring]["PL Lifetime Fit"] = (X + exp[IDstring]["Times"][ind_min], expstretch_func(X, *popt))
            tmin = exp[IDstring]["Times"][ind_min]
            tmax = 1000
            exp[IDstring]["PL Avg. 1000h"] = 1 / 1000 * (
                np.trapz(exp[IDstring]["PL Fit Parameters"][:ind_min, 0], x=exp[IDstring]["Times"][:ind_min])
                + stretched_exp_definite_integral(tmin, tmax, popt[1], popt[3], A=popt[0])
            )

        except Exception as e:
            exp[IDstring]["PL Lifetime Fit Parameters"] = [np.nan] * 4
            exp[IDstring]["PL Lifetime Fit"] = [np.nan] * 2
            print("PL lifetime fit failed on", IDstring, "with exception:", e)

        ind_min = 0
        X = exp[IDstring]["Times"][ind_min:ind_max] - exp[IDstring]["Times"][ind_min]
        Y = exp[IDstring]["R_slopes (norm.)"][ind_min:ind_max]
        try:
            popt, pcov = curve_fit(bandedge_func, X, Y, p0=[1, 10, 20], bounds=([0.9, 0, 1], [1.2, 150, 100]))
            exp[IDstring]["R_slopes Fit Parameters"] = popt
            exp[IDstring]["R_slopes Fit"] = (X + exp[IDstring]["Times"][ind_min], bandedge_func(X, *popt))
            exp[IDstring]["R_slopes Fit T80"] = popt[1] + popt[2] * np.arctanh(-0.3 / 0.5)
        except Exception as e:
            exp[IDstring]["R_slopes Fit Parameters"] = [np.nan] * 4
            exp[IDstring]["R_slopes Fit"] = [np.nan] * 2
            exp[IDstring]["R_slopes Fit T80"] = np.nan
            print("R_slopes Fit failed on", IDstring, "with exception:", e)

        try:
            X = exp[IDstring]["Times"]
            Y = exp[IDstring]["R_slopes (norm.)"]
            inds = X > 1
            popt, pcov = curve_fit(
                linear_func, X[inds], Y[inds], maxfev=20000, p0=[1, 0.9], bounds=([0.01, 0], [1e3, 1])
            )
            exp[IDstring]["BES Fit"] = (X, linear_func(X, *popt))
            exp[IDstring]["BES Fit Parameters"] = popt
        except Exception as e:
            print("Failed deg. rate fitting on ", IDstring, "with exception: ", e)
            exp[IDstring]["BES Fit"] = np.nan
            exp[IDstring]["BES Fit Parameters"] = np.nan
        exp[IDstring]["BES Final"] = exp[IDstring]["R_slopes (norm.)"][-1]
        exp[IDstring]["SWS Final"] = exp[IDstring]["Short-Wavelength Step"][-1]

    return exp


if __name__ == "__main__":
    # Example usage: replace with your own file paths
    example_file = r"G:\My Drive\Local Experiments\chiara_hitsta\Chiara021025_20_60_100FaBr.txt"
    try:
        results = import_HITSTA([example_file])
        print("Parsed samples:", list(results.keys()))
    except FileNotFoundError:
        print("Update example_file with a valid path before running directly.")
