'''
Fig S1
'''

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

# ── 1. Configuration ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":    1.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size":  6,
    "ytick.major.size":  6,
    "figure.dpi":        120,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
})

LABEL_FS = 15
TICK_FS  = 13
TITLE_FS = 18
ANNOT_FS = 11

# =========================
# File paths
# =========================
file_sl_inst = "era5_senyar_singlelevel_instant.nc"
file_850_new = "era5_senyar_850mb_new.nc"

# =========================
# Open datasets
# =========================
ds_sl = xr.open_dataset(file_sl_inst)
ds_850 = xr.open_dataset(file_850_new)

# =========================
# Select time range
# =========================
time_start = np.datetime64("2025-11-25T12:00")
time_end = np.datetime64("2025-11-26T21:00")

ds_sl = ds_sl.sel(valid_time=slice(time_start, time_end))
ds_850 = ds_850.sel(valid_time=slice(time_start, time_end))

# Select 850 hPa if pressure_level dimension exists
if "pressure_level" in ds_850.dims:
    ds_850 = ds_850.sel(pressure_level=850)

# =========================
# Calculate time means
# =========================
# CAPE and CIN (J kg^-1)
cape_mean = ds_sl["cape"].mean(dim="valid_time")
cin_mean = ds_sl["cin"].mean(dim="valid_time")

# Vertical velocity w (Pa s^-1) and horizontal wind components (m s^-1)
w_mean = ds_850["w"].mean(dim="valid_time")
u_mean = ds_850["u"].mean(dim="valid_time")
v_mean = ds_850["v"].mean(dim="valid_time")

lats_sl = ds_sl["latitude"]
lons_sl = ds_sl["longitude"]
lats_850 = ds_850["latitude"]
lons_850 = ds_850["longitude"]

# Site location: Kototabang (-0.20424, 100.32004)
site_lat = -0.20424
site_lon = 100.32004

# =========================
# Plotting setup
# =========================
proj = ccrs.PlateCarree()

skip = 3
scale_val = 400
shaft_width = 0.003
head_width = 3.5

xticks = [88, 92, 96, 100, 104]
yticks = [-4, 0, 4, 8, 12]
extent = [xticks[0], xticks[-1], yticks[0], yticks[-1]]

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(20, 7),
    subplot_kw={"projection": proj},
    constrained_layout=True,
)

# -------------------------------------------------------------
# Panel 1: CAPE + 850 hPa Winds
# -------------------------------------------------------------
ax1 = axes[0]
levels_cape = np.arange(0, 2201, 100)
cf1 = ax1.contourf(
    lons_sl,
    lats_sl,
    cape_mean,
    levels=levels_cape,
    cmap="turbo",
    extend="max",
    transform=proj,
)
ax1.set_title(
    "(a) Mean CAPE & 850 hPa Wind",
    fontsize=LABEL_FS,
    loc="left",
    fontweight="bold",
)
cbar1 = fig.colorbar(
    cf1, ax=ax1, orientation="horizontal", pad=0.06, shrink=0.9
)
cbar1.set_label(r"CAPE (J kg$^{-1}$)", fontsize=LABEL_FS)
cbar1.set_ticks(np.arange(0, 2400, 400))
cbar1.ax.tick_params(labelsize=TICK_FS)

# -------------------------------------------------------------
# Panel 2: CIN + 850 hPa Winds
# -------------------------------------------------------------
ax2 = axes[1]
levels_cin = np.arange(0, 120, 5)
cf2 = ax2.contourf(
    lons_sl,
    lats_sl,
    cin_mean,
    levels=levels_cin,
    cmap="viridis",
    extend="min",
    transform=proj,
)
ax2.set_title(
    "(b) Mean CIN & 850 hPa Wind",
    fontsize=LABEL_FS,
    loc="left",
    fontweight="bold",
)
cbar2 = fig.colorbar(
    cf2, ax=ax2, orientation="horizontal", pad=0.06, shrink=0.9
)
cbar2.set_label(r"CIN (J kg$^{-1}$)", fontsize=LABEL_FS)
cbar2.set_ticks(np.arange(0, 120, 20))
cbar2.ax.tick_params(labelsize=TICK_FS)

# -------------------------------------------------------------
# Panel 3: 850 hPa Vertical Velocity (w) + Winds
# -------------------------------------------------------------
ax3 = axes[2]
norm_w = TwoSlopeNorm(vmin=-1.2, vcenter=0.0, vmax=1.2)
levels_w = np.linspace(-1.2, 1.2, 25)
cf3 = ax3.contourf(
    lons_850,
    lats_850,
    w_mean,
    levels=levels_w,
    cmap="bwr_r",
    norm=norm_w,
    extend="both",
    transform=proj,
)
ax3.set_title(
    r"(c) Mean 850 hPa $\omega$ & Wind",
    fontsize=LABEL_FS,
    loc="left",
    fontweight="bold",
)
cbar3 = fig.colorbar(
    cf3, ax=ax3, orientation="horizontal", pad=0.06, shrink=0.9
)
cbar3.set_label(r"850 hPa $\omega$ (Pa s$^{-1}$)", fontsize=LABEL_FS)
cbar3.set_ticks([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2])
cbar3.ax.tick_params(labelsize=TICK_FS)

# =========================
# Common map features, ticks & quivers
# =========================
for ax in axes:
    ax.set_extent(extent, crs=proj)

    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="gray", alpha=0.15, zorder=1)

    # 850 hPa Wind vectors
    q = ax.quiver(
        lons_850[::skip],
        lats_850[::skip],
        u_mean[::skip, ::skip],
        v_mean[::skip, ::skip],
        scale=scale_val,
        width=shaft_width,
        headwidth=head_width,
        headlength=4.5,
        color="black",
        transform=proj,
        zorder=3,
    )

    # Filled magenta square marker for Kototabang
    ax.plot(
        site_lon,
        site_lat,
        marker="s",
        markerfacecolor="magenta",
        markeredgecolor="black",
        markersize=9,
        transform=proj,
        zorder=5,
    )

    # Draw both xticks and yticks with labels across ALL panels
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    ax.set_xticklabels([f"{x}°E" for x in xticks], fontsize=TICK_FS)
    ax.set_yticklabels([f"{abs(y)}°S" if y < 0 else f"{y}°N" if y > 0 else "0°" for y in yticks], fontsize=TICK_FS)
    ax.tick_params(labelsize=TICK_FS, top=True, right=True)

    # Boxed Reference Vector Key
    rect = mpatches.Rectangle(
        (0.02, 0.02),
        0.20,
        0.08,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=10,
    )
    ax.add_patch(rect)

    ax.quiverkey(
        q,
        X=0.06,
        Y=0.06,
        U=15,
        label=r"15 m s$^{-1}$",
        labelpos="E",
        coordinates="axes",
        labelsep=0.04,
        fontproperties={"size": ANNOT_FS, "weight": "bold"},
        zorder=11,
    )

fig.suptitle(
    "(12 UTC 25 – 21 UTC 26 November 2025)",
    fontsize=TITLE_FS,
    fontweight="bold",
    y=1.05,
)

# =========================
# Save figure
# =========================
plt.savefig("Fig_S1.png")
plt.savefig("Fig_S1.tif", dpi=300, bbox_inches="tight")

plt.close()

print("Saved: Fig_S1.png")
print("Saved: Fig_S1.tif")
