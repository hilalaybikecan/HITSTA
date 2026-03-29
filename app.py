# -*- coding: utf-8 -*-
"""
HITSTA Interactive Analysis App
Streamlit-based interactive viewer for HITSTA optical data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import StringIO
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.special import gammainc, gamma
import plotly.graph_objects as go
import plotly.express as px

# ── Fitting functions ──────────────────────────────────────────────────────────

def bandedge_func(x, a_const, b_const, l_const):
    return a_const * (0.5 - 0.5 * np.tanh((x - b_const) / l_const))

def exp_func(x, a_const, t_const, c_const):
    return a_const * np.exp(-x / t_const) + c_const

def ex_func(x, a_const, t_const, c_const):
    return -a_const * np.exp(-x / t_const) + a_const + c_const

def expstretch_func(x, a_const, t_const, c_const, r_const):
    return a_const * np.exp(-(x / t_const) ** r_const) + c_const

def linear_func(x, t_const, b_const):
    return x / t_const + b_const

def gaussian_func(wavelengths, A, mean, std, constant):
    return A * np.exp(-(wavelengths - mean) ** 2 / std ** 2) + constant

def stretched_exp_definite_integral(t0, t1, tau, beta, A=1.0):
    x0 = (np.array(t0) / tau) ** beta
    x1 = (np.array(t1) / tau) ** beta
    pre = tau / beta
    gamma_term = gamma(1 / beta)
    return A * pre * gamma_term * (gammainc(1 / beta, x1) - gammainc(1 / beta, x0))


# ── Data Import ────────────────────────────────────────────────────────────────

def parse_section(section):
    def sanitize_csv_string(csv_string):
        lines = csv_string.strip().splitlines()
        sanitized_lines = []
        for line in lines:
            parts = line.split("\t")
            if parts[-1] == "":
                parts = parts[:-1]
            sanitized_lines.append("\t".join(parts))
        return "\n".join(sanitized_lines)

    section = section.strip("\t\n")
    section = section.replace(",", ".")
    ID = section.split("\n")[0].strip()
    section = sanitize_csv_string(section)
    csvStringIO = StringIO(section)
    dataframe = pd.read_csv(csvStringIO, delimiter="\t", header=1)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')]
    return ID, dataframe


@st.cache_data(show_spinner="Importing HITSTA data...", ttl=60)
def import_HITSTA(file_contents_list, _version=3):
    """
    Accepts a list of (filename, file_content_string) tuples.
    Returns a dictionary with keys "ID1", "ID2", etc.
    """

    all_sections = []
    for fname, contents in file_contents_list:
        sections = contents.split("#")
        all_sections.append(sections)

    # Merge sections from multiple files
    sections_merged = [all_sections[0][0]]
    for i in range(1, len(all_sections[0])):
        for j in range(len(file_contents_list)):
            if j == 0:
                IDstring, df = parse_section(all_sections[0][i])
            else:
                _, df_add = parse_section(all_sections[j][i])
                df_add["Time (h)"] = df_add["Time (h)"] + df["Time (h)"].iloc[-1]
                df = pd.concat([df, df_add])
        sections_merged.append(IDstring + "\n" + df.to_csv(sep="\t"))

    exp = {}
    for section in sections_merged[1:50]:
        IDstring, df = parse_section(section)
        wavelengths = df.columns[4:].to_numpy(dtype=float)
        wavelengths = np.linspace(min(wavelengths), max(wavelengths), len(wavelengths))

        rounds = df[df["Type (D/W/L)"] == "D"].to_numpy()[:, 0] - 1
        n_wavelengths = len(wavelengths)

        dark_raw = df[df["Type (D/W/L)"] == "D"].iloc[:, 4:].to_numpy(dtype=float) / np.tile(
            df[df["Type (D/W/L)"] == "D"].iloc[:, 3].to_numpy(dtype=float), (n_wavelengths, 1)).transpose()
        refl_raw = df[df["Type (D/W/L)"] == "W"].iloc[:, 4:].to_numpy(dtype=float) / np.tile(
            df[df["Type (D/W/L)"] == "W"].iloc[:, 3].to_numpy(dtype=float), (n_wavelengths, 1)).transpose()
        laser_raw = df[df["Type (D/W/L)"] == "L"].iloc[:, 4:].to_numpy(dtype=float) / np.tile(
            df[df["Type (D/W/L)"] == "L"].iloc[:, 3].to_numpy(dtype=float), (n_wavelengths, 1)).transpose()

        exp[IDstring] = {
            "Rounds": rounds,
            "Times": df[df["Type (D/W/L)"] == "D"]["Time (h)"].to_numpy(),
            "Wavelengths": wavelengths,
            "Dark Raw": dark_raw,
            "Reflectance Raw": refl_raw,
            "Laser Raw": laser_raw,
        }

    # ── Time-dependent quantities ──
    ref_id = "ID2"  # reference cell for reflectance normalization
    if ref_id not in exp:
        ref_id = list(exp.keys())[0]

    for IDstring in exp.keys():
        wavelengths = exp[IDstring]["Wavelengths"]
        n_wavelengths = len(wavelengths)
        # Align arrays to the minimum number of rounds across D/W/L
        n_dark = exp[IDstring]["Dark Raw"].shape[0]
        n_refl = exp[IDstring]["Reflectance Raw"].shape[0]
        n_laser = exp[IDstring]["Laser Raw"].shape[0]
        n_min = min(n_dark, n_refl, n_laser)
        exp[IDstring]["Dark Raw"] = exp[IDstring]["Dark Raw"][:n_min]
        exp[IDstring]["Reflectance Raw"] = exp[IDstring]["Reflectance Raw"][:n_min]
        exp[IDstring]["Laser Raw"] = exp[IDstring]["Laser Raw"][:n_min]
        exp[IDstring]["Rounds"] = exp[IDstring]["Rounds"][:n_min]
        exp[IDstring]["Times"] = exp[IDstring]["Times"][:n_min]

        rounds_count = len(exp[IDstring]["Rounds"])
        inds = (wavelengths > 650) & (wavelengths < 850)
        wavelengths_cut = wavelengths[inds]

        # PL
        exp[IDstring]["PL"] = exp[IDstring]["Laser Raw"] - exp[IDstring]["Dark Raw"]
        inds_PLsubtr = wavelengths > 920
        exp[IDstring]["PL Subtr"] = exp[IDstring]["PL"] - np.tile(
            np.expand_dims(np.mean(exp[IDstring]["PL"][:, inds_PLsubtr], axis=1), 1), (1, n_wavelengths))
        exp[IDstring]["PL Peak Intensity"] = np.max(exp[IDstring]["PL Subtr"], axis=1)
        exp[IDstring]["PL Peak Wavelength"] = exp[IDstring]["Wavelengths"][
            np.argmax(exp[IDstring]["PL Subtr"], axis=1)]

        # PL fit
        exp[IDstring]["PL Fitted"] = [None] * rounds_count
        pl_fit_params_list = []
        last_popt = None
        for idt, PL_measurement in enumerate(exp[IDstring]["PL Subtr"]):
            try:
                p0 = last_popt if last_popt is not None else [2, 750, 100, 0]
                popt, pcov = curve_fit(gaussian_func, wavelengths_cut, PL_measurement[inds],
                                       maxfev=20000, p0=p0,
                                       bounds=([0, 500, 10, -0.005], [200, 900, 150, 0.005]))
                last_popt = popt
                exp[IDstring]["PL Fitted"][idt] = gaussian_func(wavelengths, *popt)
                pl_fit_params_list.append(popt)
            except:
                pl_fit_params_list.append([np.nan, np.nan, np.nan, np.nan])

        if any(not np.isnan(p[0]) for p in pl_fit_params_list):
            exp[IDstring]["PL Fit Parameters"] = np.array(pl_fit_params_list)
        else:
            exp[IDstring]["PL Fit Parameters"] = np.array([[0, 0, 0, 0]])

        # PL self-similarity
        PL = exp[IDstring]["PL Subtr"]
        PL0 = np.tile(exp[IDstring]["PL Subtr"][0, :], (len(exp[IDstring]["Times"]), 1))
        denom = np.sqrt(np.sum(PL0 * PL0, axis=1)) * np.sqrt(np.sum(PL * PL, axis=1))
        denom[denom == 0] = 1e-10
        exp[IDstring]["PL Self-Similarity"] = np.sum(PL * PL0, axis=1) / denom

        # Reflectance
        exp[IDstring]["Reflectance Raw Subtr"] = exp[IDstring]["Reflectance Raw"] - exp[IDstring]["Dark Raw"]
        n_rounds = exp[IDstring]["Reflectance Raw Subtr"].shape[0]
        n_ref = min(n_rounds, exp[ref_id]["Reflectance Raw"].shape[0])
        Refl_denominator = np.abs(
            exp[ref_id]["Reflectance Raw"][:n_ref] - exp[ref_id]["Dark Raw"][:n_ref] + 0.01)
        # For rounds beyond ref's range, tile the last available ref round
        if n_rounds > n_ref:
            extra = np.tile(Refl_denominator[-1], (n_rounds - n_ref, 1))
            Refl_denominator = np.concatenate([Refl_denominator, extra], axis=0)
        # Align wavelength axis between cell and reference
        n_wl = min(exp[IDstring]["Reflectance Raw Subtr"].shape[1], Refl_denominator.shape[1])
        if n_wl != n_wavelengths:
            wavelengths = wavelengths[:n_wl]
            n_wavelengths = n_wl
            exp[IDstring]["Wavelengths"] = wavelengths
            inds = (wavelengths > 650) & (wavelengths < 850)
            wavelengths_cut = wavelengths[inds]
            inds_PLsubtr = wavelengths > 920
            PL_trimmed = exp[IDstring]["PL"][:, :n_wl]
            exp[IDstring]["PL Subtr"] = PL_trimmed - np.tile(
                np.expand_dims(np.mean(PL_trimmed[:, inds_PLsubtr], axis=1), 1), (1, n_wl))
        exp[IDstring]["Reflectance"] = exp[IDstring]["Reflectance Raw Subtr"][:, :n_wl] / Refl_denominator[:, :n_wl]

        Reflectance = exp[IDstring]["Reflectance"][0]
        Rmid = 0.5 * (np.max(Reflectance[inds]) + np.min(Reflectance[inds]))
        index_bandedge = np.argmin(np.abs(Reflectance[inds] - Rmid))
        wavelength_bandedge = wavelengths_cut[index_bandedge]
        R0_bandedge = Reflectance[inds][index_bandedge]
        exp[IDstring]["Bandedge"] = (wavelength_bandedge, R0_bandedge)
        exp[IDstring]["Rmid"] = Rmid

        bandedge_interval = 100
        exp[IDstring]["Bandedge Fit WL Range"] = (
            wavelengths_cut[max(index_bandedge - bandedge_interval, 0)],
            wavelengths_cut[min(index_bandedge + bandedge_interval, len(wavelengths_cut)) - 1]
        )
        R_slopes = []
        for Refl in exp[IDstring]["Reflectance"]:
            x = wavelengths_cut[max([(index_bandedge - bandedge_interval), 0]):min(
                [(index_bandedge + bandedge_interval), len(wavelengths_cut)])]
            y = Refl[inds][max([(index_bandedge - bandedge_interval), 0]):min(
                [(index_bandedge + bandedge_interval), len(wavelengths_cut)])]
            y = np.array(y, dtype=float)
            regression = stats.linregress(x, y)
            R_slopes.append(regression.slope)
        exp[IDstring]["R_slopes (raw)"] = np.array(R_slopes)
        if R_slopes[0] != 0:
            exp[IDstring]["R_slopes (norm.)"] = np.array(R_slopes) / R_slopes[0]
        else:
            exp[IDstring]["R_slopes (norm.)"] = np.array(R_slopes)

        # R self-similarity
        R = exp[IDstring]["Reflectance"][:, inds]
        R0 = np.tile(exp[IDstring]["Reflectance"][0, inds], (len(exp[IDstring]["Times"]), 1))
        denom = np.sqrt(np.sum(R0 * R0, axis=1)) * np.sqrt(np.sum(R * R, axis=1))
        denom[denom == 0] = 1e-10
        exp[IDstring]["R Self-Similarity"] = np.sum(R * R0, axis=1) / denom

        # Short-wavelength step
        ind_min = np.argmax(exp[IDstring]["Wavelengths"] > 520)
        ind_max = np.argmax(exp[IDstring]["Wavelengths"] > 560)
        exp[IDstring]["Short-Wavelength Step"] = (
            exp[IDstring]["Reflectance"][:, ind_max] - exp[IDstring]["Reflectance"][:, ind_min])

    # ── Metrics ──
    for IDstring in exp.keys():
        X = exp[IDstring]["Times"]
        Y = exp[IDstring]["Short-Wavelength Step"]
        try:
            popt, pcov = curve_fit(ex_func, X, Y, p0=[0.1, 5, 0.05],
                                    bounds=([0, 0.05, 0], [0.5, 1000, 0.5]))
            exp[IDstring]["SWS Fit Parameters"] = popt
            exp[IDstring]["SWS Fit"] = (X, ex_func(X, *popt))
        except:
            exp[IDstring]["SWS Fit Parameters"] = [np.nan] * 3
            exp[IDstring]["SWS Fit"] = [np.nan] * 2

        exp[IDstring]["BES Final"] = exp[IDstring]["R_slopes (norm.)"][-1]
        exp[IDstring]["SWS Final"] = exp[IDstring]["Short-Wavelength Step"][-1]

    return exp


# ── Streamlit App ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="HITSTA Analysis", layout="wide")
st.title("HITSTA Interactive Analysis")

# ── Sidebar: Data Loading ──
st.sidebar.header("1. Load Data Files")

uploaded_files = []

uploaded = st.sidebar.file_uploader(
    "Upload HITSTA .txt data file(s)", type=["txt"], accept_multiple_files=True,
    help="Select one or more HITSTA measurement .txt files")
if uploaded:
    uploaded_files = [(f.name, f.read().decode("utf-8", errors="ignore")) for f in uploaded]

# ── Sidebar: Condition mapping ──
st.sidebar.header("2. Runsheet / Condition Mapping (optional)")
condition_file = st.sidebar.file_uploader(
    "Upload runsheet (Excel) with sample conditions",
    type=["xlsx", "xls"],
    help="Excel file mapping HITSTA cell IDs to experimental conditions (e.g. composition, treatment)")
df_conditions = None
if condition_file:
    df_conditions = pd.read_excel(condition_file)
    st.sidebar.success(f"Loaded {len(df_conditions)} rows")

# ── Process data ──
if not uploaded_files:
    st.info("Upload HITSTA .txt data file(s) in the sidebar to get started. "
            "Optionally upload a runsheet (Excel) to label cells by experimental condition.")
    st.stop()

exp = import_HITSTA(uploaded_files)
all_ids = list(exp.keys())
default_ids = [sid for sid in all_ids if sid not in ("ID1", "ID2")][:5]

st.sidebar.success(f"Loaded {len(all_ids)} cells: {', '.join(all_ids)}")

# ── Build condition lookup ──
condition_map = {}
cond_col = "Condition"
if df_conditions is not None:
    cols = list(df_conditions.columns)

    # Auto-detect defaults
    hitsta_default = 0
    cond_default = min(1, len(cols) - 1)
    for i, col in enumerate(cols):
        if "hitsta" in col.lower() or "id" in col.lower():
            hitsta_default = i
        if "condition" in col.lower() or "group" in col.lower() or "sample" in col.lower():
            cond_default = i

    hitsta_col = st.sidebar.selectbox("Column with HITSTA cell ID (number)", cols, index=hitsta_default)
    cond_col = st.sidebar.selectbox("Column with condition / group", cols, index=cond_default)
    st.sidebar.dataframe(df_conditions[[hitsta_col, cond_col]].head(8), use_container_width=True)

    for _, row in df_conditions.iterrows():
        if pd.isna(row[hitsta_col]):
            continue
        id_key = f"ID{int(row[hitsta_col])}" if not str(row[hitsta_col]).startswith("ID") else str(row[hitsta_col])
        condition_map[id_key] = str(row[cond_col])
    st.sidebar.success(f"Mapped {len(condition_map)} cells to conditions")


def get_label(id_str):
    """Return condition label if available, else the ID string."""
    if id_str in condition_map:
        return f"{id_str} ({condition_map[id_str]})"
    return id_str


# ── Plot Selection ──
st.sidebar.header("3. Plot Category")
plot_category = st.sidebar.radio("Category", ["Reflectance", "PL", "Conditions", "Correlations"], horizontal=True)


# ── Color palette ──
def get_colors(n):
    cmap = plt.cm.tab20b(np.linspace(0, 1, max(n, 2)))
    return [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in cmap]

def get_sequential_colors(n):
    cmap = plt.cm.coolwarm(np.linspace(0, 1, max(n, 2)))
    return [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in cmap]


def apply_time_skip(times, *arrays, skip_range=None):
    """Return times and value arrays with points inside skip_range removed."""
    if skip_range is None or skip_range[0] >= skip_range[1]:
        return (times,) + arrays
    t_start, t_end = skip_range
    mask = ~((times >= t_start) & (times <= t_end))
    return (times[mask],) + tuple(arr[mask] for arr in arrays)


def _time_skip_ui(key_prefix):
    """Render compact time-skip controls inline; return skip_range tuple or None."""
    enable = st.checkbox("Skip time range", key=f"{key_prefix}_ts_en")
    if enable:
        c1, c2 = st.columns(2)
        t_start = c1.number_input("From (h)", value=0.0, step=0.5, format="%.2f", key=f"{key_prefix}_ts_s")
        t_end = c2.number_input("To (h)", value=1.0, step=0.5, format="%.2f", key=f"{key_prefix}_ts_e")
        if t_start < t_end:
            return (t_start, t_end)
        st.warning("'From' must be less than 'To'.")
    return None


_FONT = dict(font=dict(size=14), title_font_size=16)

# ── Plot Rendering ──
from plotly.subplots import make_subplots

skip_range = None  # default; overridden inline per-tab where relevant

if plot_category == "Reflectance":
    tab_single, tab_multi, tab_bes, tab_rss = st.tabs([
        "Single Cell", "Multi Cell", "Band-edge Slope vs Time",
        "R Self-Similarity vs Time"])

    with tab_single:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_id = st.selectbox("Cell", all_ids, index=min(2, len(all_ids)-1), key="r_single_cell")
            max_round = int(exp[selected_id]["Rounds"][-1]) if len(exp[selected_id]["Rounds"]) > 0 else 0
            round_range = st.slider("Round range", 0, max_round, (0, max_round), key="r_single_rr")
            wl_min = st.number_input("WL min (nm)", value=700, step=10, key="r_single_wl_min")
            wl_max = st.number_input("WL max (nm)", value=850, step=10, key="r_single_wl_max")
            wl_range = (wl_min, wl_max)
            y_min = st.number_input("Y min", value=0.0, step=0.05, format="%.2f", key="r_single_yr_min")
            y_max_val = st.number_input("Y max", value=0.8, step=0.05, format="%.2f", key="r_single_yr_max")
            y_range = (y_min, y_max_val)
            smooth_sigma = st.number_input("Smoothing (σ)", min_value=0, max_value=50, value=5, step=1, key="r_single_smooth")
            show_bandedge = st.checkbox("Show band-edge points", value=False, key="r_single_be")
        with col2:
            rounds_to_plot = range(round_range[0], round_range[1] + 1)
            colors = get_sequential_colors(len(rounds_to_plot))
            fig = go.Figure()
            be_wl = exp[selected_id]["Bandedge"][0]
            wl_arr = exp[selected_id]["Wavelengths"]
            be_idx = int(np.argmin(np.abs(wl_arr - be_wl)))
            for i, rnd in enumerate(rounds_to_plot):
                if rnd < len(exp[selected_id]["Reflectance"]):
                    refl_y = exp[selected_id]["Reflectance"][rnd].copy()
                    if smooth_sigma > 0:
                        refl_y = gaussian_filter1d(refl_y, sigma=smooth_sigma)
                    fig.add_trace(go.Scatter(
                        x=wl_arr,
                        y=refl_y,
                        mode='lines', name=f"Round {int(rnd)}",
                        line=dict(color=colors[i], width=2.5)))
                    if show_bandedge:
                        fig.add_trace(go.Scatter(
                            x=[be_wl], y=[exp[selected_id]["Reflectance"][rnd][be_idx]],
                            mode='markers',
                            marker=dict(color='black', size=9, symbol='circle-open',
                                        line=dict(width=2.5)),
                            showlegend=False,
                            hovertemplate=f"Round {int(rnd)}: {exp[selected_id]['Reflectance'][rnd][be_idx]:.3f}<extra></extra>"))
            fig.update_layout(
                xaxis_title="Wavelength (nm)", yaxis_title="Transflectance",
                xaxis_range=list(wl_range), yaxis_range=list(y_range),
                height=500, template="plotly_white",
                title=f"Reflectance - {get_label(selected_id)}", **_FONT)
            st.plotly_chart(fig, use_container_width=True)

    with tab_multi:
        col1, col2 = st.columns([1, 3])
        with col1:
            select_all_r_multi = st.checkbox("Select All", key="r_multi_all")
            if select_all_r_multi and not st.session_state.get("r_multi_all_prev", False):
                st.session_state["r_multi_cells"] = all_ids
            st.session_state["r_multi_all_prev"] = select_all_r_multi
            selected_ids = st.multiselect("Cells", all_ids, default=default_ids, key="r_multi_cells")
            round_num = st.number_input("Round", min_value=0, value=0, key="r_multi_rnd")
            wl_min = st.number_input("WL min (nm)", value=700, step=10, key="r_multi_wl_min")
            wl_max = st.number_input("WL max (nm)", value=850, step=10, key="r_multi_wl_max")
            wl_range = (wl_min, wl_max)
            y_min = st.number_input("Y min", value=0.0, step=0.05, format="%.2f", key="r_multi_yr_min")
            y_max_val = st.number_input("Y max", value=1.0, step=0.05, format="%.2f", key="r_multi_yr_max")
            y_range = (y_min, y_max_val)
            smooth_sigma = st.number_input("Smoothing (σ)", min_value=0, max_value=50, value=5, step=1, key="r_multi_smooth")
        with col2:
            colors = get_colors(len(selected_ids))
            fig = go.Figure()
            for i, sid in enumerate(selected_ids):
                rnd = min(int(round_num), len(exp[sid]["Reflectance"]) - 1)
                refl_y = exp[sid]["Reflectance"][rnd].copy()
                if smooth_sigma > 0:
                    refl_y = gaussian_filter1d(refl_y, sigma=smooth_sigma)
                fig.add_trace(go.Scatter(
                    x=exp[sid]["Wavelengths"],
                    y=refl_y,
                    mode='lines', name=get_label(sid),
                    line=dict(color=colors[i], width=2.5)))
            fig.update_layout(
                xaxis_title="Wavelength (nm)", yaxis_title="Transflectance",
                xaxis_range=list(wl_range), yaxis_range=list(y_range),
                height=500, template="plotly_white",
                title=f"Reflectance comparison - Round {int(round_num)}", **_FONT)
            st.plotly_chart(fig, use_container_width=True)

    with tab_bes:
        col1, col2 = st.columns([1, 3])
        with col1:
            select_all_bes = st.checkbox("Select All", key="r_bes_all")
            if select_all_bes and not st.session_state.get("r_bes_all_prev", False):
                st.session_state["r_bes_cells"] = all_ids
            st.session_state["r_bes_all_prev"] = select_all_bes
            selected_ids = st.multiselect("Cells", all_ids, default=default_ids, key="r_bes_cells")
            normalize = st.checkbox("Normalized", value=True, key="r_bes_norm")
            skip_range = _time_skip_ui("r_bes")
        with col2:
            colors = get_colors(len(selected_ids))
            fig = go.Figure()
            key = "R_slopes (norm.)" if normalize else "R_slopes (raw)"
            for i, sid in enumerate(selected_ids):
                t, y = apply_time_skip(exp[sid]["Times"], exp[sid][key], skip_range=skip_range)
                fig.add_trace(go.Scatter(
                    x=t, y=y,
                    mode='lines+markers', name=get_label(sid),
                    line=dict(color=colors[i], width=1.75),
                    marker=dict(size=4)))
            fig.update_layout(
                xaxis_title="Time (h)", yaxis_title="Band-edge slope",
                height=500, template="plotly_white",
                title="Band-edge slope vs Time", **_FONT)
            st.plotly_chart(fig, use_container_width=True)

    with tab_rss:
        col1, col2 = st.columns([1, 3])
        with col1:
            select_all_rss = st.checkbox("Select All", key="r_rss_all")
            if select_all_rss and not st.session_state.get("r_rss_all_prev", False):
                st.session_state["r_rss_cells"] = all_ids
            st.session_state["r_rss_all_prev"] = select_all_rss
            selected_ids = st.multiselect("Cells", all_ids, default=default_ids, key="r_rss_cells")
            skip_range = _time_skip_ui("r_rss")
        with col2:
            colors = get_colors(len(selected_ids))
            fig = go.Figure()
            for i, sid in enumerate(selected_ids):
                t, y = apply_time_skip(exp[sid]["Times"], exp[sid]["R Self-Similarity"], skip_range=skip_range)
                fig.add_trace(go.Scatter(
                    x=t, y=y,
                    mode='lines+markers', name=get_label(sid),
                    line=dict(color=colors[i], width=1.75), marker=dict(size=3)))
            fig.update_layout(
                xaxis_title="Time (h)", yaxis_title="R Self-Similarity",
                height=500, template="plotly_white",
                title="Reflectance Self-Similarity vs Time", **_FONT)
            st.plotly_chart(fig, use_container_width=True)

elif plot_category == "PL":
    tab_single, tab_multi, tab_intensity, tab_twin, tab_pss, tab_bandgap = st.tabs([
        "Single Cell", "Multi Cell", "PL Peak Intensity vs Time",
        "PL + Band-edge Slope", "PL Self-Similarity vs Time", "PL Bandgap vs Time"])

    with tab_single:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_id = st.selectbox("Cell", all_ids, index=min(2, len(all_ids)-1), key="pl_single_cell")
            wl_min = st.number_input("WL min (nm)", value=600, step=10, key="pl_single_wl_min")
            wl_max = st.number_input("WL max (nm)", value=900, step=10, key="pl_single_wl_max")
            wl_range = (wl_min, wl_max)
            auto_y = st.checkbox("Auto Y-axis", value=True, key="pl_auto_y")
            if not auto_y:
                y_min_pl = st.number_input("Y min", value=-0.5, step=0.5, format="%.1f", key="pl_y_min")
                y_max_pl = st.number_input("Y max", value=5.0, step=0.5, format="%.1f", key="pl_y_max")
                y_range = (y_min_pl, y_max_pl)
            show_fit = st.checkbox("Show Gaussian fit", value=False, key="pl_fit")
            show_peak_wl = st.checkbox("Show peak wavelength", value=False, key="pl_show_peak")
        with col2:
            wl = exp[selected_id]["Wavelengths"]
            times = exp[selected_id]["Times"]
            n_rounds = len(times)
            wl_mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
            peak_wls = exp[selected_id]["PL Peak Wavelength"]

            # ── Single slider: show rounds 0 → selected ──
            max_round = int(exp[selected_id]["Rounds"][-1]) if len(exp[selected_id]["Rounds"]) > 0 else 0
            round_idx = st.slider("Round", 0, max_round, max_round, key="pl_all_rr")
            t_current = times[min(round_idx, n_rounds - 1)]
            st.caption(f"Round {round_idx}  →  {t_current:.2f} h  |  "
                       f"peak = {peak_wls[min(round_idx, n_rounds - 1)]:.1f} nm")

            # ── PL spectra plot (rounds 0 … round_idx) ──
            rounds_to_plot = range(0, round_idx + 1)
            colors = get_sequential_colors(max(round_idx + 1, 2))
            fig = go.Figure()
            y_max = 0
            for i, rnd in enumerate(rounds_to_plot):
                if rnd < len(exp[selected_id]["PL Subtr"]):
                    y_data = exp[selected_id]["PL Subtr"][rnd]
                    if np.any(wl_mask):
                        y_max = max(y_max, np.max(y_data[wl_mask]))
                    fig.add_trace(go.Scatter(
                        x=wl, y=y_data,
                        mode='lines', name=f"Round {int(rnd)}",
                        line=dict(color=colors[i], width=1.5)))
                    if show_fit and rnd < len(exp[selected_id]["PL Fitted"]) and exp[selected_id]["PL Fitted"][rnd] is not None:
                        fig.add_trace(go.Scatter(
                            x=wl, y=exp[selected_id]["PL Fitted"][rnd],
                            mode='lines', name=f"Fit {int(rnd)}",
                            line=dict(color=colors[i], width=1.5, dash='dash'),
                            showlegend=False))
            cur = min(round_idx, n_rounds - 1)
            if show_peak_wl and peak_wls[cur] > 0:
                fig.add_vline(x=peak_wls[cur], line_dash="dash",
                              line_color="red", line_width=1.5,
                              annotation_text=f"{peak_wls[cur]:.1f} nm",
                              annotation_position="top right")
            layout_kwargs = dict(
                xaxis_title="Wavelength (nm)", yaxis_title="PL intensity (counts)",
                xaxis_range=list(wl_range),
                height=480, template="plotly_white",
                title=f"PL Spectra - {get_label(selected_id)}", **_FONT)
            if auto_y:
                layout_kwargs["yaxis_range"] = [-y_max * 0.05, y_max * 1.1] if y_max > 0 else None
            else:
                layout_kwargs["yaxis_range"] = list(y_range)
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)

            # ── Peak wavelength vs time ──
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=times, y=peak_wls, mode='lines+markers',
                line=dict(color='#555', width=1.5), marker=dict(size=4),
                name="PL Peak Wavelength"))
            fig2.add_trace(go.Scatter(
                x=[t_current], y=[peak_wls[cur]],
                mode='markers', marker=dict(size=12, color='red', symbol='circle'),
                showlegend=False,
                hovertemplate=f"Round {int(exp[selected_id]['Rounds'][cur])}: "
                              f"{peak_wls[cur]:.1f} nm<extra></extra>"))
            fig2.update_layout(
                xaxis_title="Time (h)", yaxis_title="Peak Wavelength (nm)",
                height=250, template="plotly_white",
                title="PL Peak Wavelength vs Time",
                margin=dict(t=40, b=40), **_FONT)
            st.plotly_chart(fig2, use_container_width=True)

    with tab_multi:
        col1, col2 = st.columns([1, 3])
        with col1:
            select_all_pl_multi = st.checkbox("Select All", key="pl_multi_all")
            if select_all_pl_multi and not st.session_state.get("pl_multi_all_prev", False):
                st.session_state["pl_multi_cells"] = all_ids
            st.session_state["pl_multi_all_prev"] = select_all_pl_multi
            selected_ids = st.multiselect("Cells", all_ids, default=default_ids, key="pl_multi_cells")
            max_round_multi = max((int(exp[sid]["Rounds"][-1]) for sid in selected_ids if len(exp[sid]["Rounds"]) > 0), default=0)
            round_num = st.slider("Round", 0, max(max_round_multi, 1), 0, key="pl_multi_rnd")
            if selected_ids:
                _ref = selected_ids[0]
                _ref_rnd = min(round_num, len(exp[_ref]["Times"]) - 1)
                st.caption(f"t ≈ {exp[_ref]['Times'][_ref_rnd]:.2f} h  (ref: {get_label(_ref)})")
            wl_min = st.number_input("WL min (nm)", value=600, step=10, key="pl_multi_wl_min")
            wl_max = st.number_input("WL max (nm)", value=900, step=10, key="pl_multi_wl_max")
            wl_range = (wl_min, wl_max)
            auto_y = st.checkbox("Auto Y-axis", value=True, key="pl_multi_auto_y")
            if not auto_y:
                y_min_pl = st.number_input("Y min", value=-0.5, step=0.5, format="%.1f", key="pl_multi_y_min")
                y_max_pl = st.number_input("Y max", value=5.0, step=0.5, format="%.1f", key="pl_multi_y_max")
                y_range = (y_min_pl, y_max_pl)
            show_fit = st.checkbox("Show Gaussian fit", value=False, key="pl_multi_fit")
            show_fit_peak = st.checkbox("Show fit peak wavelength", value=False, key="pl_multi_show_peak")
            skip_range = _time_skip_ui("pl_multi")
        with col2:
            colors = get_colors(len(selected_ids))
            fig = go.Figure()
            y_max = 0
            for i, sid in enumerate(selected_ids):
                rnd = min(int(round_num), len(exp[sid]["PL Subtr"]) - 1)
                y_data = exp[sid]["PL Subtr"][rnd]
                wl_mask = (exp[sid]["Wavelengths"] >= wl_range[0]) & (exp[sid]["Wavelengths"] <= wl_range[1])
                if np.any(wl_mask):
                    y_max = max(y_max, np.max(y_data[wl_mask]))
                fig.add_trace(go.Scatter(
                    x=exp[sid]["Wavelengths"],
                    y=y_data,
                    mode='lines', name=get_label(sid),
                    line=dict(color=colors[i], width=1.5)))
                if show_fit and rnd < len(exp[sid]["PL Fitted"]) and exp[sid]["PL Fitted"][rnd] is not None:
                    fig.add_trace(go.Scatter(
                        x=exp[sid]["Wavelengths"],
                        y=exp[sid]["PL Fitted"][rnd],
                        mode='lines', name=f"{get_label(sid)} fit",
                        line=dict(color=colors[i], width=1.5, dash='dash'),
                        showlegend=False))
                if show_fit_peak and rnd < len(exp[sid]["PL Fit Parameters"]):
                    peak_nm = exp[sid]["PL Fit Parameters"][rnd, 1]
                    if not np.isnan(peak_nm) and peak_nm > 0:
                        fig.add_vline(x=peak_nm, line_color=colors[i],
                                      line_dash="dot", line_width=1.5,
                                      annotation_text=f"{peak_nm:.1f}",
                                      annotation_font_color=colors[i],
                                      annotation_position="top left")
            layout_kwargs = dict(
                xaxis_title="Wavelength (nm)", yaxis_title="PL intensity",
                xaxis_range=list(wl_range),
                height=460, template="plotly_white",
                title=f"PL comparison - Round {int(round_num)}", **_FONT)
            if auto_y:
                layout_kwargs["yaxis_range"] = [-y_max * 0.05, y_max * 1.1]
            else:
                layout_kwargs["yaxis_range"] = list(y_range)
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, use_container_width=True)

            # ── Peak wavelength vs time ──
            fig2 = go.Figure()
            for i, sid in enumerate(selected_ids):
                fit_amps = exp[sid]["PL Fit Parameters"][:, 0]
                cutoff = next((j for j, a in enumerate(fit_amps) if a < 0.5), len(fit_amps))
                t, y = apply_time_skip(exp[sid]["Times"][:cutoff], exp[sid]["PL Peak Wavelength"][:cutoff],
                                       skip_range=skip_range)
                fig2.add_trace(go.Scatter(
                    x=t, y=y, mode='lines+markers', name=get_label(sid),
                    line=dict(color=colors[i], width=1.5), marker=dict(size=3)))
                if show_fit_peak:
                    fit_peak = exp[sid]["PL Fit Parameters"][:cutoff, 1]
                    t_fit, y_fit = apply_time_skip(exp[sid]["Times"][:cutoff], fit_peak,
                                                   skip_range=skip_range)
                    valid = ~np.isnan(y_fit)
                    if np.any(valid):
                        fig2.add_trace(go.Scatter(
                            x=t_fit[valid], y=y_fit[valid],
                            mode='lines', name=f"{get_label(sid)} fit peak",
                            line=dict(color=colors[i], width=1.5, dash='dot'),
                            showlegend=False))
            fig2.update_layout(
                xaxis_title="Time (h)", yaxis_title="Peak Wavelength (nm)",
                height=270, template="plotly_white",
                title="PL Peak Wavelength vs Time",
                margin=dict(t=40, b=40), **_FONT)
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Note: data is cut when PL signal drops too low — the Gaussian fit fails below an amplitude of 0.5, so no valid peak position can be extracted beyond that point.")

    with tab_intensity:
        col1, col2 = st.columns([1, 3])
        with col1:
            select_all_pl_int = st.checkbox("Select All", key="pl_int_all")
            if select_all_pl_int and not st.session_state.get("pl_int_all_prev", False):
                st.session_state["pl_int_cells"] = all_ids
            st.session_state["pl_int_all_prev"] = select_all_pl_int
            selected_ids = st.multiselect("Cells", all_ids, default=default_ids, key="pl_int_cells")
            skip_range = _time_skip_ui("pl_int")
        with col2:
            colors = get_colors(len(selected_ids))
            fig = go.Figure()
            for i, sid in enumerate(selected_ids):
                if exp[sid]["PL Fit Parameters"][0, 0] > 0:
                    t, y = apply_time_skip(exp[sid]["Times"], exp[sid]["PL Fit Parameters"][:, 0], skip_range=skip_range)
                    fig.add_trace(go.Scatter(
                        x=t, y=y,
                        mode='lines+markers', name=get_label(sid),
                        line=dict(color=colors[i], width=2),
                        marker=dict(size=4)))
            fig.update_layout(
                xaxis_title="Time (h)", yaxis_title="PL Peak Intensity",
                height=500, template="plotly_white",
                title="PL Peak Intensity vs Time", **_FONT)
            st.plotly_chart(fig, use_container_width=True)

    with tab_twin:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_id = st.selectbox("Cell", all_ids, index=min(2, len(all_ids)-1), key="pl_twin_cell")
            skip_range = _time_skip_ui("pl_twin")
        with col2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            t_bes, y_bes = apply_time_skip(exp[selected_id]["Times"], exp[selected_id]["R_slopes (norm.)"], skip_range=skip_range)
            fig.add_trace(go.Scatter(
                x=t_bes, y=y_bes,
                mode='lines+markers', name="Band-edge slope",
                line=dict(color='#800000', width=1.75), marker=dict(size=3)),
                secondary_y=False)
            if exp[selected_id]["PL Fit Parameters"][0, 0] > 0:
                t_pl, y_pl = apply_time_skip(exp[selected_id]["Times"], exp[selected_id]["PL Fit Parameters"][:, 0], skip_range=skip_range)
                fig.add_trace(go.Scatter(
                    x=t_pl, y=y_pl,
                    mode='lines+markers', name="PL Peak Intensity",
                    line=dict(color='#01153E', width=1.5), marker=dict(size=3)),
                    secondary_y=True)
            fig.update_yaxes(title_text="Band-edge slope", secondary_y=False, color='#800000')
            fig.update_yaxes(title_text="PL Peak Intensity", secondary_y=True, color='#01153E')
            fig.update_xaxes(title_text="Time (h)")
            fig.update_layout(height=500, template="plotly_white",
                              title=f"PL + Band-edge slope - {get_label(selected_id)}", **_FONT)
            st.plotly_chart(fig, use_container_width=True)

    with tab_pss:
        col1, col2 = st.columns([1, 3])
        with col1:
            select_all_pl_pss = st.checkbox("Select All", key="pl_pss_all")
            if select_all_pl_pss and not st.session_state.get("pl_pss_all_prev", False):
                st.session_state["pl_pss_cells"] = all_ids
            st.session_state["pl_pss_all_prev"] = select_all_pl_pss
            selected_ids = st.multiselect("Cells", all_ids, default=default_ids, key="pl_pss_cells")
            skip_range = _time_skip_ui("pl_pss")
        with col2:
            colors = get_colors(len(selected_ids))
            fig = go.Figure()
            for i, sid in enumerate(selected_ids):
                t, y = apply_time_skip(exp[sid]["Times"], exp[sid]["PL Self-Similarity"], skip_range=skip_range)
                fig.add_trace(go.Scatter(
                    x=t, y=y,
                    mode='lines+markers', name=get_label(sid),
                    line=dict(color=colors[i], width=1.75), marker=dict(size=3)))
            fig.update_layout(
                xaxis_title="Time (h)", yaxis_title="PL Self-Similarity",
                height=500, template="plotly_white",
                title="PL Self-Similarity vs Time", **_FONT)
            st.plotly_chart(fig, use_container_width=True)

    with tab_bandgap:
        col1, col2 = st.columns([1, 3])
        with col1:
            select_all_pl_bg = st.checkbox("Select All", key="pl_bg_all")
            if select_all_pl_bg and not st.session_state.get("pl_bg_all_prev", False):
                st.session_state["pl_bg_cells"] = all_ids
            st.session_state["pl_bg_all_prev"] = select_all_pl_bg
            selected_ids = st.multiselect("Cells", all_ids, default=default_ids, key="pl_bg_cells")
            skip_range = _time_skip_ui("pl_bg")
        with col2:
            colors = get_colors(len(selected_ids))
            fig = go.Figure()
            for i, sid in enumerate(selected_ids):
                peak_wls = exp[sid]["PL Peak Wavelength"]
                fit_amps = exp[sid]["PL Fit Parameters"][:, 0]
                cutoff = next((j for j, a in enumerate(fit_amps) if a < 0.5), len(fit_amps))
                peak_wls = peak_wls[:cutoff]
                times_sid = exp[sid]["Times"][:cutoff]
                valid = peak_wls > 0
                bandgaps = np.where(valid, 1240.0 / np.where(valid, peak_wls, 1), np.nan)
                t, bg = apply_time_skip(times_sid, bandgaps, skip_range=skip_range)
                fig.add_trace(go.Scatter(
                    x=t, y=bg,
                    mode='lines+markers', name=get_label(sid),
                    line=dict(color=colors[i], width=1.75),
                    marker=dict(size=3)))
            fig.update_layout(
                xaxis_title="Time (h)", yaxis_title="Bandgap (eV)",
                height=500, template="plotly_white",
                title="PL Bandgap vs Time  (E = 1240 / λ_peak)", **_FONT)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Note: data is cut when PL signal drops too low — the Gaussian fit fails below an amplitude of 0.5, so no valid peak position (and therefore no valid bandgap) can be extracted beyond that point.")

elif plot_category == "Conditions":
    if not condition_map:
        st.warning("Upload a runsheet (Excel) with ID-to-condition mapping in the sidebar to use this section.")
    else:
        tab_box, tab_summary = st.tabs(["Distribution Plot", "Summary Table"])

        with tab_box:
            col1, col2 = st.columns([1, 3])
            with col1:
                plot_type = st.radio("Plot type", ["Box", "Scatter"], horizontal=True, key="cond_plot_type")
                metric = st.selectbox("Metric", [
                    "Band-edge slope (last)",
                    "PL Peak Intensity (initial)",
                    "PL Peak Intensity (last)",
                    "Bandgap — Initial (eV)",
                    "Bandgap — Final (eV)",
                ], key="cond_metric")
                excluded = st.multiselect("Exclude IDs", all_ids, key="cond_exclude")
            with col2:
                data_list = []
                for sid in all_ids:
                    if sid in excluded or sid not in condition_map:
                        continue
                    if metric == "Band-edge slope (last)":
                        val = exp[sid]["R_slopes (norm.)"][-1]
                    elif metric == "PL Peak Intensity (initial)":
                        val = exp[sid]["PL Peak Intensity"][0]
                    elif metric == "PL Peak Intensity (last)":
                        val = exp[sid]["PL Peak Intensity"][-1]
                    elif metric == "Bandgap — Initial (eV)":
                        wl0 = exp[sid]["PL Peak Wavelength"][0]
                        val = 1240.0 / wl0 if wl0 > 0 else np.nan
                    elif metric == "Bandgap — Final (eV)":
                        fit_amps = exp[sid]["PL Fit Parameters"][:, 0]
                        cutoff = next((j for j, a in enumerate(fit_amps) if a < 0.5), len(fit_amps))
                        peak_wls_v = exp[sid]["PL Peak Wavelength"][:cutoff]
                        val = (1240.0 / peak_wls_v[-1]) if len(peak_wls_v) > 0 and peak_wls_v[-1] > 0 else np.nan
                    data_list.append({"Condition": condition_map[sid], metric: val, "ID": sid})

                if data_list:
                    df_plot = pd.DataFrame(data_list)
                    if plot_type == "Box":
                        fig = px.box(df_plot, x="Condition", y=metric, points="all",
                                     hover_data=["ID"], template="plotly_white")
                    else:
                        fig = px.strip(df_plot, x="Condition", y=metric,
                                       hover_data=["ID"], color="Condition",
                                       template="plotly_white")
                        fig.update_traces(marker=dict(size=10, opacity=0.8))
                    fig.update_layout(height=500, title=f"{metric} by {cond_col}",
                                      xaxis_title=cond_col, **_FONT)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No matching data for plot.")

        with tab_summary:
            summary_rows = []
            for sid in all_ids:
                if sid not in condition_map:
                    continue
                row = {
                    "ID": sid,
                    "Condition": condition_map.get(sid, ""),
                    "Rounds": len(exp[sid]["Rounds"]),
                    "Max Time (h)": f"{exp[sid]['Times'][-1]:.1f}" if len(exp[sid]['Times']) > 0 else "",
                    "PL Peak Init.": f"{exp[sid]['PL Peak Intensity'][0]:.2f}",
                    "PL Peak Final": f"{exp[sid]['PL Peak Intensity'][-1]:.2f}",
                    "BES Final": f"{exp[sid]['R_slopes (norm.)'][-1]:.3f}",
                    "SWS Final": f"{exp[sid]['Short-Wavelength Step'][-1]:.3f}",
                }
                summary_rows.append(row)
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            else:
                st.info("No cells matched to conditions.")

elif plot_category == "Correlations":
    def _cell_metrics(sid):
        fit_amps = exp[sid]["PL Fit Parameters"][:, 0]
        cutoff = next((j for j, a in enumerate(fit_amps) if a < 0.5), len(fit_amps))
        peak_wls_valid = exp[sid]["PL Peak Wavelength"][:cutoff]
        bg_init = 1240.0 / exp[sid]["PL Peak Wavelength"][0] if exp[sid]["PL Peak Wavelength"][0] > 0 else np.nan
        bg_final = (1240.0 / peak_wls_valid[-1]) if len(peak_wls_valid) > 0 and peak_wls_valid[-1] > 0 else np.nan
        amp_init = float(fit_amps[0]) if len(fit_amps) > 0 else np.nan
        amp_final = float(fit_amps[cutoff - 1]) if cutoff > 0 else np.nan
        return {
            "PL Peak WL — Initial (nm)":        float(exp[sid]["PL Peak Wavelength"][0]),
            "PL Peak WL — Final (nm)":           float(exp[sid]["PL Peak Wavelength"][-1]),
            "PL Peak WL Shift (nm)":             float(exp[sid]["PL Peak Wavelength"][-1] - exp[sid]["PL Peak Wavelength"][0]),
            "Bandgap — Initial (eV)":            bg_init,
            "Bandgap — Final (eV)":              bg_final,
            "Bandgap Shift (eV)":                (bg_final - bg_init) if not (np.isnan(bg_init) or np.isnan(bg_final)) else np.nan,
            "PL Amplitude — Initial":            amp_init,
            "PL Amplitude — Final (valid)":      amp_final,
            "PL Amplitude Ratio (final/initial)": amp_final / amp_init if (amp_init and amp_init != 0) else np.nan,
            "PL Self-Similarity — Final":        float(exp[sid]["PL Self-Similarity"][-1]),
            "Band-edge Slope — Initial (raw)":   float(exp[sid]["R_slopes (raw)"][0]),
            "Band-edge Slope — Final (norm.)":   float(exp[sid]["R_slopes (norm.)"][-1]),
            "R Self-Similarity — Final":         float(exp[sid]["R Self-Similarity"][-1]),
            "Duration (h)":                      float(exp[sid]["Times"][-1]) if len(exp[sid]["Times"]) > 0 else np.nan,
        }

    metrics_df = pd.DataFrame(
        [{"ID": sid, "Condition": condition_map.get(sid, sid), **_cell_metrics(sid)} for sid in all_ids]
    )
    metric_cols = [c for c in metrics_df.columns if c not in ("ID", "Condition")]

    col1, col2 = st.columns([1, 3])
    with col1:
        x_metric = st.selectbox("X axis", metric_cols, index=0, key="corr_x")
        y_metric = st.selectbox("Y axis", metric_cols, index=1, key="corr_y")
        color_by = st.radio("Color by", ["Cell ID", "Condition"], key="corr_color", horizontal=True)
        excluded_corr = st.multiselect("Exclude IDs", all_ids, key="corr_exclude")
    with col2:
        df_plot = metrics_df[~metrics_df["ID"].isin(excluded_corr)].dropna(subset=[x_metric, y_metric])
        color_col = "Condition" if color_by == "Condition" else "ID"
        fig = px.scatter(
            df_plot, x=x_metric, y=y_metric,
            color=color_col, text="ID",
            hover_data={"ID": True, "Condition": True,
                        x_metric: ":.4f", y_metric: ":.4f"},
            template="plotly_white", height=550,
            title=f"{y_metric}  vs  {x_metric}"
        )
        fig.update_traces(textposition="top center", marker=dict(size=10))
        fig.update_layout(xaxis_title=x_metric, yaxis_title=y_metric, **_FONT)
        st.plotly_chart(fig, use_container_width=True)

        # Pearson correlation of every metric vs the chosen X
        st.subheader(f"Pearson correlations vs  {x_metric}")
        corr_rows = []
        for col in metric_cols:
            if col == x_metric:
                continue
            clean = metrics_df[~metrics_df["ID"].isin(excluded_corr)].dropna(subset=[x_metric, col])
            if len(clean) > 2 and clean[x_metric].std() > 0 and clean[col].std() > 0:
                r, p = stats.pearsonr(clean[x_metric], clean[col])
                corr_rows.append({"Parameter": col, "r": round(r, 3), "p": round(p, 4), "n": len(clean)})
        if corr_rows:
            corr_table = pd.DataFrame(corr_rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
            st.dataframe(corr_table, use_container_width=True, hide_index=True)

# ── Data Table ──
with st.expander("Summary Table"):
    summary_rows = []
    for sid in all_ids:
        row = {
            "ID": sid,
            "Condition": condition_map.get(sid, ""),
            "Rounds": len(exp[sid]["Rounds"]),
            "Max Time (h)": f"{exp[sid]['Times'][-1]:.1f}" if len(exp[sid]['Times']) > 0 else "",
            "PL Peak Init.": f"{exp[sid]['PL Peak Intensity'][0]:.2f}",
            "PL Peak Final": f"{exp[sid]['PL Peak Intensity'][-1]:.2f}",
            "BES Final": f"{exp[sid]['R_slopes (norm.)'][-1]:.3f}",
        }
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
