"""
Panel (a): Topography (DEMNAS) + TC Senyar track
Panel (b): Time series — pressure, wind speed, distance to Kototabang
Panel (c): ERA5 Mean 850 hPa Wind + Acc VIMD
"""

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ── 1. Configuration & rcParams ────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"], 
    "axes.linewidth":     1.5,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.width":  1.5,
    "ytick.major.width":  1.5,
    "xtick.major.size":   6,
    "ytick.major.size":   6,
    "figure.dpi":         150,
    "savefig.dpi":        600,    
    "savefig.bbox":       "tight",
})

LABEL_FS = 14
TICK_FS  = 12
TITLE_FS = 16

SITE_LAT, SITE_LON = -0.20424, 100.32004
KT_PER_MS          = 0.514444
PROJ               = ccrs.PlateCarree()
MAP_EXTENT_A       = [94.5, 105.5, -1.5, 6.5]   # [W, E, S, N] for Panel A

# Time window for ERA5
T_START = np.datetime64("2025-11-25T12:00")
T_END   = np.datetime64("2025-11-28T12:00")

# ── 2. Functions ───────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance in km."""
    R = 6371.0
    phi1, lam1, phi2, lam2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * R * np.arcsin(np.sqrt(
        np.sin((phi2 - phi1) / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2)**2
    ))

def stroke(lw=3.0, fg="white"):
    """Returns a PathEffect for text outlining."""
    return [pe.withStroke(linewidth=lw, foreground=fg)]

# ── 3. Data ─────────────────────────────────────────────────
print("Loading IBTrACS data...")
track = pd.read_csv(
    "../Track/ibtracs.since1980.list.v04r01.csv",
    low_memory=False, header=0, skiprows=[1]
)
track.columns = track.columns.str.strip().str.lower()
track["datetime"] = pd.to_datetime(track["iso_time"], format="%Y-%m-%d %H:%M:%S", utc=True)

for col in ("lat", "lon", "usa_wind", "usa_pres"):
    track[col] = pd.to_numeric(track[col], errors="coerce")

# Filter for Senyar
track = (
    track[track["name"].str.strip().str.upper() == "SENYAR"]
    .dropna(subset=["datetime", "lat", "lon"])
    .sort_values("datetime")
    .reset_index(drop=True)
)

track["dist_km"]      = haversine(track["lat"], track["lon"], SITE_LAT, SITE_LON)
track["wind_ms"]      = track["usa_wind"] * KT_PER_MS
track["datetime_utc"] = track["datetime"].dt.tz_localize(None)
track["dt_str"]       = track["datetime_utc"].dt.strftime("%d%H")

print("Loading ERA5 data...")
ds_850 = xr.open_dataset("era5_senyar_850mb.nc").sel(valid_time=slice(T_START, T_END), pressure_level=850)
ds_sl  = xr.open_dataset("era5_senyar_singlelevel.nc").sel(valid_time=slice(T_START, T_END))

u_mean   = ds_850["u"].mean("valid_time")
v_mean   = ds_850["v"].mean("valid_time")
vimd_sum = ds_sl["vimd"].sum("valid_time")
lats_e5, lons_e5 = ds_850["latitude"], ds_850["longitude"]

print("Loading DEMNAS Topography...")
dem_path = "../DEM_SRTM/dem_sumatera_a_1.jp2"
with rasterio.open(dem_path) as src:
    W, E, S, N = MAP_EXTENT_A
    fb = src.bounds
    cW, cE = max(W, fb.left), min(E, fb.right)
    cS, cN = max(S, fb.bottom), min(N, fb.top)
    
    window = from_bounds(cW, cS, cE, cN, transform=src.transform)
    nx = 2500
    ny = int(nx * (cN - cS) / (cE - cW))
    dem = src.read(1, window=window, out_shape=(ny, nx), resampling=Resampling.bilinear).astype(float)
    if src.nodata is not None:
        dem = np.where(dem == src.nodata, np.nan, dem)

dem_land = np.where(dem <= 0, np.nan, dem)
terrain_cmap = plt.cm.terrain.copy()
terrain_cmap.set_bad(alpha=0.0)

ls = LightSource(azdeg=315, altdeg=45)
rgb_dem = ls.shade(np.ma.masked_invalid(dem_land), cmap=terrain_cmap, blend_mode="overlay", vmin=0, vmax=3000, vert_exag=0.1)

# ── 4. Figure Layout setup ────────────────────────────────────────────────────
print("Plotting Figure...")
fig = plt.figure(figsize=(22, 4)) 
gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.15, 1.15], wspace=0.25)
gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0], width_ratios=[1, 1.15], wspace=0.2)

ax1 = fig.add_subplot(gs_inner[0], projection=PROJ)
ax2 = fig.add_subplot(gs_inner[1])
ax3 = fig.add_subplot(gs_outer[1], projection=PROJ)

# ══════════════════════════════════════════════════════════════════════════════
# Panel (a): Topography + Track
# ══════════════════════════════════════════════════════════════════════════════
ax1.set_extent(MAP_EXTENT_A, crs=PROJ)
ax1.add_feature(cfeature.OCEAN, facecolor="#D1E5F0", zorder=0) 
ax1.add_feature(cfeature.LAND,  facecolor="#E8E8E8", zorder=0)
ax1.imshow(rgb_dem, extent=[cW, cE, cS, cN], transform=PROJ, origin="upper", zorder=1)

ax1.coastlines(resolution="10m", linewidth=1.2, color="black", zorder=4)
ax1.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle="-", edgecolor="black", zorder=4)

gl = ax1.gridlines(draw_labels=False, crs=PROJ, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, zorder=2)
ax1.set_xticks([96, 99, 102, 105], crs=PROJ)
ax1.set_yticks([0, 2, 4, 6], crs=PROJ)
ax1.xaxis.set_major_formatter(LongitudeFormatter())
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.tick_params(labelsize=TICK_FS)

# Kototabang
ax1.plot(SITE_LON, SITE_LAT, "rs", ms=9, mec="black", mew=1.2, transform=PROJ, zorder=10)
ax1.text(SITE_LON + 0.2, SITE_LAT + 0.15, "KT", fontsize=TICK_FS, fontweight="bold", color="white", 
         transform=PROJ, zorder=11, path_effects=stroke(lw=3.0, fg="black"))

# Track
ax1.plot(track["lon"], track["lat"], color="black", lw=2.5, transform=PROJ, zorder=5)
label_idx = set(track.loc[track["datetime_utc"].dt.hour == 0, "dt_str"])
label_idx.update([track["dt_str"].iloc[0], track["dt_str"].iloc[-1]])

for _, row in track.iterrows():
    ax1.plot(row["lon"], row["lat"], "o", ms=5, mfc="black", mec="white", mew=0.5, transform=PROJ, zorder=6)
    if row["dt_str"] in label_idx:
        ax1.text(row["lon"] + 0.15, row["lat"] + 0.15, row["dt_str"], fontsize=10, fontweight="bold", 
                 transform=PROJ, zorder=12, path_effects=stroke(lw=2.5))

# Disdrometer Obs Highlight
disdro_pt = track[track["dt_str"] == "2621"]
if not disdro_pt.empty:
    ax1.plot(disdro_pt["lon"].iloc[0], disdro_pt["lat"].iloc[0], "o", ms=8, mfc="limegreen", mec="black", mew=1.2, transform=PROJ, zorder=9)
    ax1.text(disdro_pt["lon"].iloc[0] + 0.15, disdro_pt["lat"].iloc[0] - 0.35, "2621", fontsize=10, fontweight="bold", 
             color="limegreen", transform=PROJ, zorder=12, path_effects=stroke(lw=2.5, fg="black"))

ax1.set_title("(a) Topography & Track", loc="left", fontsize=TITLE_FS, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Panel (b): Time Series
# ══════════════════════════════════════════════════════════════════════════════
t = track["datetime_utc"]
c_pres, c_wind, c_dist = "#1565C0", "#C62828", "black"

# Left: Pressure
lp, = ax2.plot(t, track["usa_pres"], color=c_pres, lw=2.5, label="Min. Pressure")
ax2.set_ylabel("Minimum Pressure (hPa)", color=c_pres, fontsize=LABEL_FS, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=c_pres, labelsize=TICK_FS)
ax2.spines["left"].set_color(c_pres)
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(False)

# Right 1: Wind
ax_w = ax2.twinx()
lw_, = ax_w.plot(t, track["wind_ms"], color=c_wind, lw=2.5, label="Max. Wind Speed")
ax_w.set_ylabel("Maximum Wind Speed (ms$^{-1}$)", color=c_wind, fontsize=LABEL_FS, fontweight="bold")
ax_w.tick_params(axis="y", labelcolor=c_wind, labelsize=TICK_FS)
ax_w.spines["right"].set_color(c_wind)
ax_w.spines["top"].set_visible(True)
ax_w.spines["left"].set_visible(False)

# Right 2: Distance
ax_d = ax2.twinx()
ax_d.spines["right"].set_position(("outward", 55))
ax_d.spines["right"].set_color(c_dist)
ax_d.spines["top"].set_visible(True)
ax_d.spines["left"].set_visible(False)
ld, = ax_d.plot(t, track["dist_km"], color=c_dist, lw=2.0, ls="--", label="Distance to KT")
ax_d.set_ylabel("Distance to Kototabang (km)", fontsize=LABEL_FS, color=c_dist, fontweight="bold")
ax_d.tick_params(axis="y", labelsize=TICK_FS, labelcolor=c_dist)

# X-axis
ax2.set_xlim(t.iloc[0], t.iloc[-1])
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d%H"))
ax2.tick_params(axis="x", labelsize=TICK_FS, rotation=0)
ax2.set_xlabel("Time (DDHH UTC)", fontsize=LABEL_FS, fontweight="bold")

lv = ax2.axvline(pd.Timestamp("2025-11-26 21:00"), color="limegreen", ls=":", lw=2.5, zorder=3)

ax2.legend(handles=[lp, lw_, ld], loc="upper right", fontsize=10, frameon=True, framealpha=0.9, edgecolor="gray")
ax2.set_title("(b) Cyclone Parameters & Distance", loc="left", fontsize=TITLE_FS, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Panel (c): ERA5 Mean 850 hPa Wind & Acc VIMD
# ══════════════════════════════════════════════════════════════════════════════
levels = [-50, -35, -20, -5, 5, 20, 35, 50]
cmap = plt.get_cmap("RdBu_r").resampled(len(levels) - 1)
norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cf = ax3.contourf(lons_e5, lats_e5, vimd_sum, levels=levels, cmap=cmap, norm=norm, extend="both", transform=PROJ)

skip = 3
Q = ax3.quiver(lons_e5[::skip], lats_e5[::skip], u_mean[::skip, ::skip], v_mean[::skip, ::skip],
               scale=400, width=0.004, headwidth=4, transform=PROJ, zorder=3)

# Quiver Box
rect = mpatches.Rectangle((0.02, 0.02), 0.17, 0.08, transform=ax3.transAxes, facecolor='white', edgecolor='black', zorder=10)
ax3.add_patch(rect)
Q_key = ax3.quiverkey(Q, X=0.06, Y=0.06, U=15, label='15 ms$^{-1}$', labelpos='E', coordinates='axes', 
                      labelsep=0.04, fontproperties={'size': 11, 'weight': 'bold'}, zorder=11)

# Coastlines & Map formatting
ax3.coastlines(resolution="10m", linewidth=1.2, color='black', zorder=4)
ax3.add_feature(cfeature.BORDERS, linestyle="-", linewidth=0.8, edgecolor='black', zorder=4)

ax3.set_xticks([90, 95, 100, 105], crs=PROJ)
ax3.set_yticks([0, 2, 4, 6], crs=PROJ)
ax3.xaxis.set_major_formatter(LongitudeFormatter())
ax3.yaxis.set_major_formatter(LatitudeFormatter())
ax3.tick_params(labelsize=TICK_FS)

# Kototabang
ax3.plot(SITE_LON, SITE_LAT, marker="s", color="red", markersize=6, transform=PROJ, zorder=10)
ax3.text(SITE_LON + 0.15, SITE_LAT, "KT", size=TICK_FS, weight='bold', color='white', transform=PROJ, zorder=13, path_effects=stroke(lw=2, fg='black'))

# TC Track on ERA5
ax3.plot(track["lon"], track["lat"], color="black", linewidth=2.0, transform=PROJ, zorder=5)
ax3.scatter(track["lon"], track["lat"], color="black", s=25, transform=PROJ, zorder=6)

ax3.scatter(track["lon"].iloc[0], track["lat"].iloc[0], c="black", s=60, marker="o", ec="white", transform=PROJ, zorder=7)
ax3.scatter(track["lon"].iloc[-1], track["lat"].iloc[-1], c="black", s=60, marker="o", ec="white", transform=PROJ, zorder=7)

ax3.text(track["lon"].iloc[0] + 0.15, track["lat"].iloc[0] + 0.15, track["dt_str"].iloc[0], fontsize=10, fontweight='bold', color='black', transform=PROJ, zorder=10, path_effects=stroke(lw=2.5))
ax3.text(track["lon"].iloc[-1] + 0.15, track["lat"].iloc[-1] + 0.15, track["dt_str"].iloc[-1], fontsize=10, fontweight='bold', color='black', transform=PROJ, zorder=10, path_effects=stroke(lw=2.5))

# Colorbar
cbar = plt.colorbar(cf, ax=ax3, orientation="vertical", pad=0.02, shrink=0.8, extendfrac=0, ticks=levels)
cbar.set_label("Acc VIMD (kg m$^{-2}$)", fontsize=LABEL_FS, fontweight="bold")
cbar.ax.tick_params(labelsize=TICK_FS)

ax3.set_title("(c) ERA5 Mean 850 hPa Wind & Acc VIMD", loc="left", fontsize=TITLE_FS, fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "Fig1abc.png"
plt.savefig(out_path)
print(f"Saved: {out_path}")
plt.show()