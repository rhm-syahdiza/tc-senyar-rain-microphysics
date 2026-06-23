"""
Time-series DSD plot: D (mm) vs time, colorbar log10(N(D)) [mm^-1 m^-3]
Overlaid with Rain Rate (R) on secondary y-axis.
"""

import os, csv, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

# ── rcParams ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":     1.2,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "figure.dpi":         150,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
})

LABEL_FS = 16
TICK_FS  = 14

# ── Parsivel bin centers and widths ───────────────────────────────────────────
D_bins = np.array([
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937,
    1.062, 1.187, 1.375, 1.625, 1.875, 2.125, 2.375, 2.750,
    3.250, 3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500,
    9.500, 11.00, 13.00, 15.00, 17.00, 19.00, 21.50, 24.50
])
V_bins = np.array([
    0.050, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750,
    0.850, 0.950, 1.100, 1.300, 1.500, 1.700, 1.900, 2.200,
    2.600, 3.000, 3.400, 3.800, 4.400, 5.200, 6.000, 6.800,
    7.600, 8.800, 10.40, 12.00, 13.60, 15.20, 17.60, 20.80
])
dD = np.array([
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
    0.125, 0.125, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500,
    0.500, 0.500, 0.500, 0.500, 1.000, 1.000, 1.000, 1.000,
    1.000, 2.000, 2.000, 2.000, 2.000, 2.000, 3.000, 3.000
])

# ── Data QC ───────────────────────────────────────────────────
def atlas_v_corrected(d, altitude_m=865.0):
    """Density-corrected terminal velocity (m/s) for elevated sites."""
    v_sea_level = 9.65 - 10.3 * np.exp(-0.6 * d)
    
    # Standard Atmosphere constants
    rho_0, T_0, L, R, g = 1.225, 288.15, 0.0065, 287.05, 9.80665
    T_z = T_0 - (L * altitude_m)
    rho_z = rho_0 * (T_z / T_0) ** ((g / (L * R)) - 1)
    
    correction_factor = (rho_0 / rho_z) ** 0.4
    return np.maximum(v_sea_level * correction_factor, 0.0)

# Apply the density-corrected velocity to the Parsivel bins
V_term = atlas_v_corrected(D_bins, altitude_m=865.0)
V_FRAC = 0.60  # ±60 % tolerance

# Dynamic Effective Sampling Area: A_eff(D_i) = (L * (B - D_i/2)) * 10^-6 m^2
L_beam, B_beam = 180.0, 30.0
A_eff = (L_beam * (B_beam - (D_bins / 2.0))) * 1e-6 # shape (32,)
t_int = 60.0  # seconds

# QC
D_valid_mask = (D_bins >= 0.312) & (D_bins <= 10.0) #drops lowest 2 bins
MIN_DROPS = 10
R_MIN     = 0.1

# ── Time window ───────────────────────────────────────────────────────────────
T_START_LT = dt.datetime(2025, 11, 25, 12, 0, 0) + dt.timedelta(hours=7)
T_END_LT   = dt.datetime(2025, 11, 28, 12, 0, 0) + dt.timedelta(hours=7)

# ── Read & QC data ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root  = os.path.join(script_dir, "Kototabang")

times_utc = []
rain_list = []
nd_list   = []
valid_sample_count = 0 

for month in sorted(os.listdir(data_root)):
    mpath = os.path.join(data_root, month)
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
                        t_lt = dt.datetime.strptime(f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S")
                        if not (T_START_LT <= t_lt <= T_END_LT): continue

                        r_raw = float(row[2])

                        # ── Raw spectrum ──────────────────────────────────────
                        spec_raw = row[15:1038]
                        vals = [float(x) if x.strip() else 0.0 for x in spec_raw]
                        vals += [0.0] * (1024 - len(vals))
                        spec = np.array(vals).reshape(32, 32)   # (v, D)

                        # ── QC-1: ±60% Velocity Filter ────────────────────────
                        for d_i in range(32):
                            V_lo, V_hi = V_term[d_i] * (1 - V_FRAC), V_term[d_i] * (1 + V_FRAC)
                            for v_i in range(32):
                                if not (V_lo <= V_bins[v_i] <= V_hi):
                                    spec[v_i, d_i] = 0.0

                        # ── QC-2: Minimum drop count ──────────────────────────
                        if spec.sum() < MIN_DROPS: continue

                        # ── QC-3: D-axis mask ─────────────────────────────────
                        spec[:, ~D_valid_mask] = 0.0

                        # ── N(D) Retrieval ────────
                        n_d = np.zeros(32)
                        for d_i in range(32):
                            if not D_valid_mask[d_i]: continue
                            col = spec[:, d_i]
                            with np.errstate(divide='ignore', invalid='ignore'):
                                n_d[d_i] = np.sum(np.where(
                                    V_bins > 0, 
                                    col / (V_bins * A_eff[d_i] * t_int * dD[d_i]), 
                                    0.0
                                ))

                        # ── Rainrate────────────────
                        r_dsd = 6 * np.pi * 1e-4 * np.sum(V_term * (D_bins**3) * n_d * dD)

                        # ── LT to UTC ────────────────────────────────
                        t_utc = t_lt - dt.timedelta(hours=7)
                        times_utc.append(t_utc)
                        
                        # rain threshold 
                        if r_raw < R_MIN or r_dsd == 0:
                            rain_list.append(np.nan)
                            nd_list.append(np.full(32, np.nan))
                        else:
                            rain_list.append(r_dsd) 
                            nd_list.append(n_d)
                            valid_sample_count += 1 

                    except Exception:
                        continue

times_utc = np.array(times_utc)
rain_arr  = np.array(rain_list, dtype=float)
nd_matrix = np.array(nd_list, dtype=float)

idx       = np.argsort(times_utc)
times_utc = times_utc[idx]
rain_arr  = rain_arr[idx]
nd_matrix = nd_matrix[idx]

# ── log10(N(D)) array ────────────────────────────────────────────────
D_plot = D_bins[D_valid_mask]
ND     = nd_matrix[:, D_valid_mask] 
with np.errstate(divide='ignore', invalid='ignore'):
    ND_log = np.where(ND > 0, np.log10(ND), np.nan)

Z = ND_log.T 

# ── Bin edges ─────────────────────────────────────────────────────────────────
def edges(c):
    e = np.zeros(len(c) + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0]    = max(0, c[0]  - (c[1]  - c[0])  / 2)
    e[-1]   =        c[-1] + (c[-1] - c[-2]) / 2
    return e

D_edges = edges(D_plot)
t_num   = mdates.date2num(times_utc)
t_edges = edges(t_num)

# ── Plot layout ───────────────────────────────────────────────────────────────
D_MAX         = 8.0  
D_RAIN0       = 5.5
R_MAX_DISPLAY = 120.0

fig, ax1 = plt.subplots(figsize=(14, 5))

# ── DSD colour mesh ───────────────────────────────────────────────────────────
cmap = plt.cm.viridis
cmap.set_bad(color='white')

pcm = ax1.pcolormesh(t_edges, D_edges, Z,
                     cmap=cmap, vmin=0, vmax=4,
                     shading='flat', zorder=1)

cbar = plt.colorbar(pcm, ax=ax1, pad=0.08, aspect=25, fraction=0.03)
cbar.set_label("log$_{10}[N(D)]$ (mm$^{-1}$ m$^{-3}$)", fontsize=12, fontweight="bold")
cbar.set_ticks(np.arange(0, 5, 1))
cbar.ax.tick_params(labelsize=TICK_FS)
cbar.outline.set_linewidth(1.2)

# ── Rain rate on secondary y-axis───────────────────────────
R_scale = (D_MAX - D_RAIN0) / R_MAX_DISPLAY
rain_D  = D_RAIN0 + rain_arr * R_scale  

ax1.plot(t_num, rain_D, color='magenta', linewidth=1.5, alpha=1.0, zorder=5)
ax1.axhline(D_RAIN0, color='magenta', linewidth=1.0, linestyle='--', alpha=0.6, zorder=4)

# ── Right y-axis: R ticks ─────────────────────────────────────────────────────
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())

R_ticks_val = np.array([0, 20, 40, 60, 80, 100, 120])
R_ticks_D   = D_RAIN0 + R_ticks_val * R_scale
mask        = (R_ticks_D >= D_RAIN0) & (R_ticks_D <= D_MAX)

ax2.set_yticks(R_ticks_D[mask])
ax2.set_yticklabels([str(int(v)) for v in R_ticks_val[mask]], color='magenta', fontweight="bold")
ax2.set_ylabel("Rain Rate, $R$ (mm h$^{-1}$)", color='magenta', fontsize=12, fontweight="bold", labelpad=5)
ax2.tick_params(axis='y', colors='magenta', direction='in', labelsize=TICK_FS)
ax2.spines["right"].set_color("magenta")
ax2.spines["right"].set_linewidth(1.5)
ax2.set_ylim(0, D_MAX)

# ── Left y-axis ───────────────────────────────────────────────────────────────
ax1.set_ylim(0, D_MAX)
ax1.set_ylabel("Diameter, $D$ (mm)", fontsize=LABEL_FS, fontweight="bold")
ax1.set_yticks(np.arange(0, D_MAX+1, 1))
ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
ax1.tick_params(which="both", direction="in", labelsize=TICK_FS)
ax1.tick_params(which="major", length=6)
ax1.tick_params(which="minor", length=3)

lv = ax2.axvline(pd.Timestamp("2025-11-26 21:26"), color="limegreen", ls=":", lw=2.5, zorder=3)
   
# ── Time axis ─────────────────────────────────────────────────────────────────
T_PLOT_START = dt.datetime(2025, 11, 25, 12, 0, 0)
T_PLOT_END   = dt.datetime(2025, 11, 27, 00, 0, 0)
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d%H")) 
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
ax1.set_xlabel("Time (DDHH UTC)", fontsize=LABEL_FS, fontweight="bold")
ax1.set_xlim(mdates.date2num(T_PLOT_START), mdates.date2num(T_PLOT_END))

ax1.grid(True, which='major', axis='x', linestyle=':', linewidth=0.8, color='black', alpha=0.5, zorder=0)

plt.tight_layout()
out = os.path.join(script_dir, "Fig4.png")
plt.savefig(out)


print("\n─── TC Senyar Event Statistics ───")
print(f"✅ Total valid 1-minute spectra : {valid_sample_count}")
#if np.any(~np.isnan(rain_arr)):
#    print(f"✅ Maximum Derived Rain Rate (R): {max_rain:.2f} mm/h")
#    print(f"✅ Maximum Drop Diameter (D_max): {max_D:.3f} mm")
print(f"Saved → {out}\n")

plt.show()