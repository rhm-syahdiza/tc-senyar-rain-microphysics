"""
EAR Plot
a.SNR plot + mean
b.Vertical velocity + horizontal wind + mean
c.Spectral width + mean
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator

# ============================================================
# Open and merge EAR files
# ============================================================
files = [
    "20251125.nc",
    "20251126.nc",
    "20251127.nc",
]

files = [f for f in files if os.path.exists(f)]
if not files:
    raise FileNotFoundError("NetCDF source files missing from the active directory.")

ds = xr.open_mfdataset(files, combine="by_coords")

# ============================================================
# TC Senyar time window from obs data
# ============================================================
t_start = pd.Timestamp("2025-11-25 12:00")
t_end   = pd.Timestamp("2025-11-26 21:00")

ds = ds.sel(time=slice(t_start, t_end))

time = ds["time"]
z = ds["range"]   # in km

# ============================================================
# Variables & SNR Calculation
# ============================================================
# Winds
W = ds["vwind"].transpose("range", "time")
U = ds["zwind"].transpose("range", "time")
V = ds["mwind"].transpose("range", "time")

dpl = ds["dpl"].sel(beam=0)
dpl = dpl.where(dpl < 1e9)

# Beam-0 quantities + SNR
pwr = ds["pwr"].sel(beam=0)
pwr = pwr.where(pwr < 1e9)
pnoise = ds["pnoise"].sel(beam=0)
snr = pwr - pnoise 

width = ds["width"].sel(beam=0)
width = width.where(width < 1e9)

# ============================================================
# Mean value
# ============================================================
snr_mean = snr.mean("time")
w_mean   = W.mean("time")
width_mean = width.mean("time")
dpl_mean = dpl.mean("time")

# ============================================================
# Figure & layout (3 × 2)
# ============================================================
fig = plt.figure(figsize=(8.5, 10.5), facecolor="white")

gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[5.2, 1.1],
    height_ratios=[1.0, 1.0, 1.0],
    hspace=0.18,
    wspace=0.22
)

ax_snr      = fig.add_subplot(gs[0, 0])
ax_snr_prof = fig.add_subplot(gs[0, 1], sharey=ax_snr)

ax_w        = fig.add_subplot(gs[1, 0], sharex=ax_snr)
ax_w_prof   = fig.add_subplot(gs[1, 1], sharey=ax_w)

ax_width      = fig.add_subplot(gs[2, 0], sharex=ax_snr)
ax_width_prof = fig.add_subplot(gs[2, 1], sharey=ax_width)

# ============================================================
# --- TOP: Echo Power SNR (beam 0)
# ============================================================
levels_Z = np.arange(0, 55, 5)

cf_snr = ax_snr.contourf(
    time, z,
    snr.transpose("range", "time"),
    levels=levels_Z,
    cmap="viridis", 
    extend="max"
)

ax_snr.set_ylabel("Altitude (km)", fontweight="bold")
ax_snr.set_title("(a) Echo Power SNR", loc="left", fontweight="bold", fontsize=11)
ax_snr.set_xlim(t_start, t_end)

# Colorbar
cax_snr = fig.add_axes([0.71, 0.68, 0.018, 0.16])
cbar_snr = fig.colorbar(cf_snr, cax=cax_snr)
cbar_snr.ax.set_xlabel("(dB)", labelpad=6) 
cbar_snr.ax.xaxis.set_label_coords(1.5, 1.15)
cbar_snr.ax.xaxis.label.set_fontsize(9)
cbar_snr.ax.tick_params(labelsize=8)

# Mean plot
ax_snr_prof.plot(snr_mean, z, color="black", label="SNR mean")
ax_snr_prof.set_xlim(0, 50)
ax_snr_prof.legend(frameon=False, fontsize=8)
ax_snr_prof.legend(frameon=False, fontsize=8, loc="lower left")
ax_snr_prof.tick_params(labelleft=False)

# ============================================================
# --- MIDDLE: Vertical velocity + horizontal wind 
# ============================================================
levels_w = np.array([-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5])

cf_w = ax_w.contourf(
    time, z,
    W,
    levels=levels_w,
    cmap="RdBu_r", 
    extend="both"
)

ax_w.contour(time, z, W, levels=levels_w, colors="black", linewidths=0.2, alpha=0.5)

# Wind vectors
t_skip, z_skip = 4, 3 
wind_speed = np.sqrt(U**2 + V**2)
ref_wind = 20 if np.nanmax(wind_speed) > 10 else 10

Q = ax_w.quiver(time[::t_skip], z[::z_skip], U[::z_skip, ::t_skip], V[::z_skip, ::t_skip],
                scale=800, width=0.0027, color="black", zorder=3)
ax_w.set_ylabel("Altitude (km)", fontweight="bold")
ax_w.set_title(r"(b) Vertical Velocity ($w$) & Horizontal Wind", loc="left", fontweight="bold", fontsize=11)

# Colorbar
cax_w = fig.add_axes([0.71, 0.39, 0.018, 0.18])
cbar_w = fig.colorbar(cf_w, cax=cax_w)
cbar_w.ax.set_xlabel("(m s$^{-1}$)", labelpad=6)
cbar_w.ax.xaxis.set_label_coords(1.5, 1.15)
cbar_w.ax.xaxis.label.set_fontsize(9)
cbar_w.ax.tick_params(labelsize=8)

# Vector legend 
rect = mpatches.Rectangle((0.1, 0.8), 0.18, 0.12, transform=ax_w.transAxes, 
                         facecolor='white', edgecolor='gray', linewidth=0.8, alpha=0.9, zorder=4)
ax_w.add_patch(rect)

qk = ax_w.quiverkey(
    Q,
    X=0.13,
    Y=0.85,
    U=ref_wind,
    label=f"{ref_wind} m s$^{{-1}}$",
    labelpos="E",
    coordinates="axes", 
    fontproperties={'size': 9},
    zorder=5
)

# Overlaid Profile Panel
ax_w_prof.plot(w_mean, z, color="black", linestyle="-", label=r"$w$ mean")
ax_w_prof.set_xlim(-1.0, 1.0)  
ax_w_prof.legend(frameon=False, fontsize=8, loc="lower left")
ax_w_prof.tick_params(labelleft=False)

# ============================================================
# --- BOTTOM: Spectral Width
# ============================================================
levels_width = np.arange(0, 4, 0.4)

cf_width = ax_width.contourf(
    time, z,
    width.transpose("range", "time"),
    levels=levels_width,
    cmap="viridis",
    extend="both"
)

ax_width.set_title(r"(c) Spectral Width ($\sigma_v$)", loc="left", fontweight="bold", fontsize=11)

# Colorbar
cax_width = fig.add_axes([0.71, 0.13, 0.018, 0.16])
cbar_width = fig.colorbar(cf_width, cax=cax_width)
cbar_width.ax.set_xlabel("(m s$^{-1}$)", labelpad=6)
cbar_width.ax.xaxis.set_label_coords(1.5, 1.15)
cbar_width.ax.xaxis.label.set_fontsize(9)
cbar_width.ax.tick_params(labelsize=8)

# Mean Plot
ax_width_prof.axvline(0, color="gray", linestyle="--", lw=0.8)
ax_width_prof.plot(width_mean, z, color="black", label=r"$\sigma_v$ mean")
ax_width_prof.set_xlim(0, 2)
ax_width_prof.legend(frameon=False, fontsize=8)
ax_width_prof.tick_params(labelleft=False)

ax_width.set_ylabel("Altitude (km)", fontweight="bold")
ax_width.set_xlabel("Time (DDHH UTC)", fontweight="bold", labelpad=8)

# ============================================================
# Formatting
# ============================================================
for ax in [ax_snr, ax_w, ax_width]:
    ax.set_ylim(0, 10)
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction="out", top=True, right=True)

ax_snr.tick_params(labelbottom=False)
ax_w.tick_params(labelbottom=False)

ax_width.xaxis.set_major_formatter(mdates.DateFormatter("%d%H"))

plt.tight_layout(rect=[0.0, 0.07, 0.83, 1.0])

# ============================================================
# Save
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
plt.savefig(
    os.path.join(script_dir, "Fig3.png"),
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)
plt.close()

print(f"Saved cleanly → {os.path.join(script_dir, 'Fig3.png')}")