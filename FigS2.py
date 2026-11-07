"""
Figure S2: Merged Hovmöller Diagram (Time-Longitude)
Panel (a): Himawari-9 Cloud Top Temperature (TBB)
Panel (b): BMKG Radar Composite Reflectivity (CMAX)

Averaged over latitudes -0.5 to 0.5. 
"""

import os
import re
import bz2
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. Configuration & Paths
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."

# Data directories 
HIM_DIR   = os.path.join(script_dir, "..", "HIMAWARI-9")
HIM_LUT   = os.path.join(HIM_DIR, "2025", "tir.01")

# Robust radar folder detection
RADAR_CANDIDATES = [
    os.path.join(script_dir, "..", "Radar_BMKG", "radar_sumatera")
]
RADAR_DIR = next((d for d in RADAR_CANDIDATES if os.path.exists(d)), RADAR_CANDIDATES[0])

# Time window & Domain
START_DT = datetime(2025, 11, 25, 12, 0)
END_DT   = datetime(2025, 11, 26, 21, 0)
STEP_MIN = 10  # Both datasets use 10-minute intervals

LAT_MIN, LAT_MAX = -0.5, 0.5
LON_MIN, LON_MAX = 95.0, 105.0
SITE_LON = 100.32004  # Kototabang

# ============================================================
# 2. Himawari-9 Data Processing
# ============================================================
def process_himawari():
    print("--- Starting Himawari-9 Processing ---")
    lut = np.loadtxt(HIM_LUT)
    lut_table = np.full(65536, np.nan, dtype=np.float32)
    lut_table[lut[:, 0].astype(np.int32)] = lut[:, 1].astype(np.float32) - 273.15

    lon_full = np.linspace(85.0, 205.0, 6000)
    lat_full = np.linspace(60.0, -60.0, 6000)

    ix = np.where((lon_full >= LON_MIN) & (lon_full <= LON_MAX))[0]
    iy = np.where((lat_full >= LAT_MIN) & (lat_full <= LAT_MAX))[0]
    lon_sub = lon_full[ix]

    time_list = []
    hov_list = []
    cur = START_DT

    while cur <= END_DT:
        ts = cur.strftime("%Y%m%d%H%M")
        patterns = [
            os.path.join(HIM_DIR, cur.strftime("%Y"), cur.strftime("%m"), f"{ts}.tir.01.fld.geoss.bz2"),
            os.path.join(HIM_DIR, f"{ts}.tir.01.fld.geoss.bz2")
        ]
        
        file_path = next((p for p in patterns if os.path.exists(p) and os.path.getsize(p) > 0), None)

        if file_path:
            try:
                with bz2.open(file_path, "rb") as f:
                    raw = np.frombuffer(f.read(), dtype=">u2").reshape((6000, 6000))
                sub = raw[np.ix_(iy, ix)]
                tbb_sub = np.where(sub == 65535, np.nan, lut_table[sub])
                tbb_profile = np.nanmean(tbb_sub, axis=0)
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
                tbb_profile = np.full(len(lon_sub), np.nan)
        else:
            tbb_profile = np.full(len(lon_sub), np.nan)

        time_list.append(cur)
        hov_list.append(tbb_profile)
        cur += timedelta(minutes=STEP_MIN)

    print(f"Processed {len(time_list)} Himawari timesteps.\n")
    return lon_sub, time_list, np.array(hov_list)

# ============================================================
# 3. BMKG Radar Data Processing
# ============================================================
def extract_radar_datetime(filename):
    match = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    return None

def get_closest_radar_file(folder_path, target_dt, max_tol=10):
    if not os.path.exists(folder_path): return None
    best_file, min_diff = None, timedelta(days=999)
    for entry in os.listdir(folder_path):
        file_path = os.path.join(folder_path, entry)
        if not os.path.isfile(file_path): continue
        file_dt = extract_radar_datetime(entry)
        if not file_dt: continue
        diff = abs(file_dt - target_dt)
        if diff < min_diff:
            min_diff, best_file = diff, file_path
    if min_diff <= timedelta(minutes=max_tol):
        return best_file
    return None

def extract_meridional_avg_profile(base_dir, target_dt, grid_params):
    folders = ["Aceh", "Medan", "Nias", "Padang", "Pekanbaru", "Batam", "Jambi"]
    tif_files = []
    for folder in folders:
        fp = get_closest_radar_file(os.path.join(base_dir, folder), target_dt)
        if fp: tif_files.append(fp)

    if not tif_files:
        return np.full(grid_params["width"], np.nan)

    src_files = []
    for fp in tif_files:
        try:
            src_files.append(rasterio.open(fp))
        except: pass

    if not src_files:
        return np.full(grid_params["width"], np.nan)

    mosaic, out_trans = merge(src_files, method="max")
    src_crs = src_files[0].crs or "EPSG:4326"
    for src in src_files: src.close()

    mosaic = np.squeeze(mosaic).astype(np.float32)
    mosaic = np.where(mosaic < 5.0, np.nan, mosaic)

    resampled = np.full((grid_params["height"], grid_params["width"]), np.nan, dtype=np.float32)
    reproject(
        source=mosaic, destination=resampled,
        src_transform=out_trans, src_crs=src_crs,
        dst_transform=grid_params["transform"], dst_crs="EPSG:4326",
        resampling=Resampling.bilinear
    )

    with np.errstate(all="ignore"):
        return np.nanmean(resampled, axis=0)

def process_radar():
    print("--- Starting Radar CMAX Processing ---")
    width = int((LON_MAX - LON_MIN) / 0.01)
    height = int((LAT_MAX - LAT_MIN) / 0.01)
    grid_params = {
        "width": width, "height": height,
        "transform": from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, width, height),
        "lon_coords": np.linspace(LON_MIN, LON_MAX, width)
    }

    time_series, hovmoller_matrix = [], []
    curr = START_DT
    
    # Process radar at the exact same timesteps as Himawari
    while curr <= END_DT:
        time_series.append(curr)
        profile = extract_meridional_avg_profile(RADAR_DIR, curr, grid_params)
        hovmoller_matrix.append(profile)
        curr += timedelta(minutes=STEP_MIN)

    print(f"Processed {len(time_series)} Radar timesteps.\n")
    return grid_params["lon_coords"], time_series, np.array(hovmoller_matrix)

# ============================================================
# 4. Plotting & Layout
# ============================================================
print("Extracting data (this may take a few minutes)...")
lon_him, time_him, hov_him = process_himawari()
lon_rad, time_rad, hov_rad = process_radar()

print("Generating combined Figure...")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight"
})

fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True, facecolor="white")
ax1, ax2 = axes

# --- Panel (a): Himawari-9 ---
levels_h = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10]
colors_h = ["#4d004b", "#b30000", "#e34a33", "#fc8d59", "#fdcc8a", 
            "#ffffbf", "#c7e9b4", "#7fcdbb", "#41b6c4", "#d0d0d0"]
cmap_h = mcolors.ListedColormap(colors_h)
norm_h = mcolors.BoundaryNorm(levels_h, cmap_h.N, clip=True)

X_h, Y_h = np.meshgrid(lon_him, mdates.date2num(time_him))
pcm_h = ax1.pcolormesh(X_h, Y_h, hov_him, cmap=cmap_h, norm=norm_h, shading="auto", rasterized=True)

ax1.axvline(SITE_LON, color="black", linestyle="--", linewidth=1.5, label="Kototabang")
ax1.set_xlim(LON_MIN, LON_MAX)
ax1.set_xticks(np.arange(96, 106, 2))
ax1.set_ylabel("Time (UTC)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Longitude", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.5, color="gray")
ax1.set_title("(a) Himawari-9 Cloud Top Temperature (T$_{BB}$)", loc="left", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.9, edgecolor="gray")

cb_h = fig.colorbar(pcm_h, ax=ax1, orientation="vertical", pad=0.02, shrink=0.85, ticks=levels_h)
cb_h.set_label("Cloud Top T$_{BB}$ (°C)", fontsize=12, fontweight="bold")

# --- Panel (b): Radar CMAX ---
levels_r = [5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 65]
colors_r = ["#d0d0d0", "#41b6c4", "#7fcdbb", "#c7e9b4", "#ffffbf", 
            "#fdcc8a", "#fc8d59", "#e34a33", "#b30000", "#4d004b"]
cmap_r = mcolors.ListedColormap(colors_r)
norm_r = mcolors.BoundaryNorm(levels_r, cmap_r.N)

X_r, Y_r = np.meshgrid(lon_rad, mdates.date2num(time_rad))
pcm_r = ax2.pcolormesh(X_r, Y_r, hov_rad, cmap=cmap_r, norm=norm_r, shading="auto", rasterized=True)

ax2.axvline(SITE_LON, color="black", linestyle="--", linewidth=1.5, label="Kototabang")
ax2.set_xlim(LON_MIN, LON_MAX)
ax2.set_xticks(np.arange(96, 106, 2))
ax2.set_xlabel("Longitude", fontsize=13, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.5, color="gray")
ax2.set_title("(b) Radar Composite Reflectivity (CMAX)", loc="left", fontsize=14, fontweight="bold")
ax2.legend(loc="upper right", framealpha=0.9, edgecolor="gray")

cb_r = fig.colorbar(pcm_r, ax=ax2, orientation="vertical", pad=0.02, shrink=0.85, ticks=levels_r)
cb_r.set_label("Mean Reflectivity (dBZ)", fontsize=12, fontweight="bold")

# --- Final Time Axis Formatting ---
ax1.yaxis_date()
ax1.yaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
ax1.yaxis.set_major_locator(mdates.HourLocator(interval=3))
ax1.invert_yaxis() # Put earliest time at the top

fig.subplots_adjust(wspace=0.15)

# Save
png_path = os.path.join(script_dir, "Fig5_Hovmoller.png")
tif_path = os.path.join(script_dir, "Fig5_Hovmoller.tif")

plt.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(tif_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved cleanly -> {png_path} and {tif_path}")
