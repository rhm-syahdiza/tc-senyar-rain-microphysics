import bz2
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mticker
import pandas as pd

from datetime import datetime, timedelta

"""
Himawari9 TBB
"""

# =========================
# Setup 
# =========================
base_time = datetime(2025, 11, 25, 12, 0)
times = [base_time + timedelta(hours=12*i) for i in range(3)]
print(times)

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

panel_labels = ["(d)", "(e)", "(f)"]

for idx, current_time in enumerate(times):

    ax = axes[idx]

    yyyy = current_time.strftime("%Y")
    mm   = current_time.strftime("%m")
    dd   = current_time.strftime("%d")
    hh   = current_time.strftime("%H")

    # ===============================
    # File
    # ===============================
    bz2_file = f"./{yyyy}/{mm}/{yyyy}{mm}{dd}{hh}00.tir.01.fld.geoss.bz2"
    bin_file = bz2_file.replace(".bz2", "")

    if not os.path.exists(bin_file):
        with bz2.open(bz2_file, "rb") as f_in, open(bin_file, "wb") as f_out:
            f_out.write(f_in.read())

    tir = np.fromfile(bin_file, dtype=">u2").reshape((6000, 6000))
    tir = np.where(tir == 65535, np.nan, tir)

    # ===============================
    # LUT
    # ===============================
    lut = np.loadtxt("./2025/tir.01")
    hs = lut[:, 0].astype(np.int32)
    tbb = lut[:, 1].astype(np.float32)

    lut_dict = dict(zip(hs, tbb))
    tir_tbb = np.full(tir.shape, np.nan, dtype=np.float32)

    valid = ~np.isnan(tir)
    unique_vals = np.unique(tir[valid]).astype(np.int32)

    for v in unique_vals:
        tir_tbb[tir == v] = lut_dict[v] - 273

    # ===============================
    # Lat and Lon
    # ===============================
    lon = np.linspace(85.0, 205.0, 6000)
    lat = np.linspace(60.0, -60.0, 6000)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # ===============================
    # Plot
    # ===============================
    ax.set_extent([88, 106, -5, 15])


    # ==========================
    # Cloud Top Temperature (°C)
    # ==========================
    levels = [-95, -90, -85, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10]

    colors = [
        "#cc0000", # -60
        "#ff5500",
        "#ffb000",
        "#ffff00",
        "#aaff00",
        "#00ff99",    
        "#00bfbf",
        "#004bff",
        "#4f8bff",
        "#a6c8ff",
        "#dce6ff",
        "#ffffff" #0
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)

    pcm = ax.pcolormesh(
        lon2d, lat2d, tir_tbb,
        cmap=cmap, norm=norm,
        shading="auto"
    )

    ax.coastlines(resolution="10m", linewidth=1.2)

    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

    gl.xlocator = mticker.FixedLocator([88,92,96,100,104])

    gl.ylocator = mticker.FixedLocator([-4,0,4,8,12])
    
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    gl.top_labels = False
    gl.right_labels = False

    # ===============================
    # Kototabang
    # ===============================
    ear_lat = -0.20424
    ear_lon = 100.32004
    ax.scatter(
        ear_lon,
        ear_lat,
        s=70,                 # ukuran marker
        marker="s",           # ← square / rectangle
        color="magenta",
        edgecolor="black",
        linewidth=0.6,
        transform=ccrs.PlateCarree(),
        zorder=7
    )

    ax.text(
        ear_lon + 0.08,
        ear_lat + 0.08,
        "KT",
        fontsize=14,
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        zorder=7
        )

    # ===============================
    # TRACK
    # ===============================
    # ===============================
    # TRACK
    # ===============================
    # ===============================
    # TRACK
    # ===============================
    print("Loading IBTrACS data...")
    
    # 1. Read the file, skipping the second row (index 1) which contains units
    track = pd.read_csv(
        "../Track/ibtracs.since1980.list.v04r01.csv", 
        skiprows=[1],
        low_memory=False # Prevents memory warnings due to file size
    )

    # 2. Convert IBTrACS 'ISO_TIME' to actual datetime objects
    track["datetime"] = pd.to_datetime(track["ISO_TIME"])

    # 3. Rename LAT and LON to lowercase so your existing ax.plot works
    track = track.rename(columns={"LAT": "lat", "LON": "lon"})

    # 4. FILTER FOR YOUR SPECIFIC CYCLONE
    # Replace 'YOUR_STORM_NAME' with the actual name in ALL CAPS (e.g., 'PADDY', 'SEROJA')
    # If it is an unnamed tropical depression, use the SID column instead.
    storm_name = "YOUR_STORM_NAME" 
    track = track[track["NAME"] == storm_name]

    # Plot the full track for this specific cyclone
    ax.plot(track["lon"], track["lat"],
            "-o", color="black",
            linewidth=1.5, markersize=4)

    # Plot the specific location at the current loop time
    peak = track.loc[
        track["datetime"] == current_time.strftime("%Y-%m-%d %H:00")
    ]

    ax.scatter(
        peak["lon"], peak["lat"],
        marker="*", s=160,
        color="magenta", edgecolor="black",
        zorder=6
    )

    # ===============================
    # TITLE (DDHH + panel label)
    # ===============================
    time_ddhh = current_time.strftime("%d%H")

    ax.set_title(
        f"{panel_labels[idx]}  {time_ddhh} UTC",
        fontsize=16,
        loc="left",
        fontweight="bold"
    )

# ===============================
# COLORBAR
# ===============================
fig.subplots_adjust(right=0.88)  
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  

cbar = fig.colorbar(pcm, cax=cbar_ax)

cbar.set_label("Cloud Top Temperature (°C)", fontsize=14)
cbar.ax.tick_params(labelsize=12)


plt.savefig(
    "Fig1_def.png",
    dpi=300,
    bbox_inches="tight"
)
    
