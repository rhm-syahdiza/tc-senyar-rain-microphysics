'''
Fig. S5
'''

import os
import csv
import warnings
import datetime as dtm
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

# Silence harmless empty-slice warnings from the SNR masking
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# ============================================================
# Settings
# ============================================================
EVENT_FILES = [
    "../EAR/20251125.nc",
    "../EAR/20251126.nc",
    "../EAR/20251127.nc",
]

SNR_MIN = 5.0    # dB; gates with SNR <= SNR_MIN are excluded from the width analysis

# Directory setup
script_dir  = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
DISDRO_ROOT = os.path.join(script_dir, "..", "Disdrometer", "Kototabang")

# ============================================================
# Open EAR files
# ============================================================
def open_ear(paths):
    return xr.open_mfdataset(
        paths,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        preprocess=lambda d: d.drop_vars(["date"], errors="ignore"),
    )

files = [f for f in EVENT_FILES if os.path.exists(f)]
if not files:
    raise FileNotFoundError("NetCDF source files missing from the active directory.")

ds = open_ear(files)

# ============================================================
# TC Senyar time window from obs data
# ============================================================
t_start = pd.Timestamp("2025-11-25 12:00")
t_end   = pd.Timestamp("2025-11-26 21:00")

ds = ds.sel(time=slice(t_start, t_end)).load()
time = ds["time"]
z = ds["range"]   # km above radar level

# ============================================================
# Variables, SNR, spectral width
# ============================================================
W = ds["vwind"].transpose("range", "time")

pwr = ds["pwr"].sel(beam=0).where(ds["pwr"].sel(beam=0) < 1e9)
pnoise = ds["pnoise"].sel(beam=0)
snr = pwr - pnoise

width = ds["width"].sel(beam=0).where(ds["width"].sel(beam=0) < 1e9)
width = width.where(snr > SNR_MIN)          # SNR mask

print(f"EAR period used: {str(time.values[0])[:16]} to {str(time.values[-1])[:16]} UTC")

# ============================================================
# Does sigma_v track rain or air motion?  (disdrometer R, |w|, SNR)
# ============================================================
def read_disdro(root):
    if not os.path.isdir(root):
        return None
    ts, rr = [], []
    for month in sorted(os.listdir(root)):
        mpath = os.path.join(root, month)
        if not os.path.isdir(mpath): continue
        for day in sorted(os.listdir(mpath)):
            dpath = os.path.join(mpath, day)
            if not os.path.isdir(dpath): continue
            for fname in sorted(os.listdir(dpath)):
                if not fname.endswith(".csv"): continue
                with open(os.path.join(dpath, fname), newline="") as f:
                    for row in csv.reader(f):
                        if not row or not row[0].strip(): continue
                        try:
                            t_lt = dtm.datetime.strptime(f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S")
                            t_utc = pd.Timestamp(t_lt - dtm.timedelta(hours=7))
                            if t_start <= t_utc <= t_end:
                                ts.append(t_utc); rr.append(float(row[2]))
                        except Exception:
                            continue
    return (ts, rr) if ts else None

def layer_mean(da_tr, zz, z0, z1, min_gates=3):
    sel = da_tr.where((zz >= z0) & (zz < z1))
    return sel.mean("range").where(sel.count("range") >= min_gates).values

def rho(a, b, nmin=20):
    m = np.isfinite(a) & np.isfinite(b)
    return (spearmanr(a[m], b[m])[0] if m.sum() >= nmin else np.nan), int(m.sum())

dis = read_disdro(DISDRO_ROOT)
if dis is None:
    print("\n[corr] Disdrometer data not found at DISDRO_ROOT; skipping sigma_v vs rain analysis.")
else:
    t_d = pd.DatetimeIndex(dis[0]); r_d = np.array(dis[1], dtype=float)
    o = np.argsort(t_d.values); t_d, r_d = t_d[o], r_d[o]
    t_ear = pd.DatetimeIndex(time.values)
    half = pd.Series(t_ear).diff().median() / 2
    R_ear = np.full(len(t_ear), np.nan)
    for i, t0 in enumerate(t_ear):
        m = (t_d >= t0 - half) & (t_d < t0 + half)
        if m.any():
            R_ear[i] = r_d[m].mean()

    ok = width.notnull()
    absW = np.abs(W.transpose("time", "range")).where(ok)
    
    # --------------------------------------------------------
    # Scatter Plots with Regression Lines & CIs
    # --------------------------------------------------------
    sv26 = layer_mean(width, z, 2, 6)
    aw26 = layer_mean(absW, z, 2, 6)
    figc, axc = plt.subplots(1, 2, figsize=(9, 4), facecolor="white")
    
    # Panel (a): Rain Rate vs Spectral Width
    m = np.isfinite(sv26) & np.isfinite(R_ear) & (R_ear > 0)
    sns.regplot(
        x=R_ear[m], y=sv26[m], ax=axc[0],
        scatter_kws={"s": 15, "alpha": 0.5, "color": "tab:blue", "edgecolor": "none"},
        line_kws={"color": "black", "linewidth": 1.5}
    )
    axc[0].set_xscale("log")
    axc[0].set_xlabel("Parsivel rain rate (mm h$^{-1}$)", fontweight="bold")
    axc[0].set_ylabel(r"Layer-mean $\sigma_v$, 2–6 km (m s$^{-1}$)", fontweight="bold")
    axc[0].set_title(f"(a) rho = {rho(sv26, R_ear)[0]:.2f}", loc="left", fontweight="bold")
    
    # Panel (b): Vertical Air Speed vs Spectral Width
    m = np.isfinite(sv26) & np.isfinite(aw26)
    sns.regplot(
        x=aw26[m], y=sv26[m], ax=axc[1],
        scatter_kws={"s": 15, "alpha": 0.5, "color": "tab:blue", "edgecolor": "none"},
        line_kws={"color": "black", "linewidth": 1.5}
    )
    axc[1].set_xlabel(r"Layer-mean $\vert{}w\vert{}$, 2–6 km (m s$^{-1}$)", fontweight="bold")
    axc[1].set_title(f"(b) rho = {rho(sv26, aw26)[0]:.2f}", loc="left", fontweight="bold")
    
    for a in axc:
        a.grid(True, ls=":", alpha=0.5)
        a.tick_params(direction="in", top=True, right=True)
        
    figc.tight_layout()
    figc.savefig(os.path.join(script_dir, "FigS5.png"), dpi=300, facecolor="white")
    figc.savefig(os.path.join(script_dir, "FigS5.tif"), dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Saved scatter plots to {script_dir}")
    plt.close(figc)
