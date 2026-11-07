"""Figure 1 (revised): TC Senyar overview, 3 rows x 3 columns

  Row 1: (a) Topography + track | (b) Cyclone parameters & distance | (c) ERA5
  850 hPa wind + Acc VIMD
  Row 2: (d)-(f) Himawari-9 TBB at 2512, 2600, 2612 UTC          (shared
  horizontal colorbar)
  Row 3: (g)-(i) BMKG radar CMAX composite at the same times     (shared
  horizontal colorbar)
         Radar = nearest scan of each radar within +-RADAR_TOL of the panel time
         (no 1-h composite).
         White = inside radar coverage, grey = no radar coverage (Nias is
         unavailable on 26 Nov).
"""

import bz2
from datetime import datetime, timedelta
import os
import re
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.windows import from_bounds
import xarray as xr

# ── 1. Configuration ───────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

LABEL_FS = 15
TICK_FS = 13
TITLE_FS = 18
ANNOT_FS = 11

SITE_LAT, SITE_LON = -0.20424, 100.32004
KT_PER_MS = 0.514444
K0 = 273.15  # K -> degC 
PROJ = ccrs.PlateCarree()

EXTENT_A = [88, 106, -5, 15]  # map window  [W, E, S, N]

T_START = np.datetime64("2025-11-25T12:00")
T_END = np.datetime64("2025-11-26T21:00")
OBS_START = pd.Timestamp("2025-11-25 12:00")  # disdrometer window (UTC)
OBS_END = pd.Timestamp("2025-11-26 21:26")
TBB_TIMES = [
    datetime(2025, 11, 25, 12) + timedelta(hours=12 * i) for i in range(3)
]

DISDRO_STR = "2621"  # DDHH of disdrometer observation

# Radar settings
RADAR_DIR_CANDIDATES = [
    "../Radar_BMKG/radar_sumatera",
]
RADAR_DIR = next(
    (
        d
        for d in RADAR_DIR_CANDIDATES
        if any(
            os.path.isdir(os.path.join(d, s))
            for s in ("Aceh", "Medan", "Nias", "Padang", "Pekanbaru", "Batam", "Jambi")
        )
    ),
    RADAR_DIR_CANDIDATES[0],
)
print(f"Radar root: {os.path.abspath(RADAR_DIR)}")

RADAR_SITES = ["Aceh", "Medan", "Nias", "Padang", "Pekanbaru", "Batam", "Jambi"]
RADAR_TOL = timedelta(minutes=10)  # max |scan time - panel time|
RADAR_DS = 4  # block factor used only for the coverage mask
NOCOV_GRAY = "#D9D9D9"  # background where no radar covers


# ── 2. Helpers ─────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
  R = 6371.0
  phi1, lam1, phi2, lam2 = map(np.radians, [lat1, lon1, lat2, lon2])
  return (
      2
      * R
      * np.arcsin(
          np.sqrt(
              np.sin((phi2 - phi1) / 2) ** 2
              + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2) ** 2
          )
      )
  )


def stroke(lw=3.0, fg="white"):
  return [pe.withStroke(linewidth=lw, foreground=fg)]


def format_map_axes(ax, xticks, yticks, grid=True):
  ax.set_xticks(xticks, crs=PROJ)
  ax.set_yticks(yticks, crs=PROJ)
  ax.xaxis.set_major_formatter(LongitudeFormatter())
  ax.yaxis.set_major_formatter(LatitudeFormatter())
  ax.tick_params(labelsize=TICK_FS)
  if grid:
    ax.gridlines(
        draw_labels=False,
        crs=PROJ,
        color="gray",
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
        zorder=2,
    )


def draw_kt(ax, ms=10, fc="red", fs=TICK_FS):
  ax.plot(
      SITE_LON,
      SITE_LAT,
      "s",
      ms=ms,
      mfc=fc,
      mec="black",
      mew=1.2,
      transform=PROJ,
      zorder=10,
  )
  ax.text(
      SITE_LON + 0.2,
      SITE_LAT + 0.2,
      "KT",
      fontsize=fs,
      fontweight="bold",
      color="white",
      transform=PROJ,
      zorder=11,
      path_effects=stroke(3.0, "black"),
  )


def draw_track(ax, track, lw=2.5, ms=5, label_all_00=False):
  """Track with white halo and strict whitelist label placement (guaranteed no 2700)."""
  ax.plot(
      track["lon"],
      track["lat"],
      color="white",
      lw=lw + 2.0,
      transform=PROJ,
      zorder=4.8,
  )
  ax.plot(
      track["lon"], track["lat"], color="black", lw=lw, transform=PROJ, zorder=5
  )
  ax.plot(
      track["lon"],
      track["lat"],
      "o",
      ms=ms,
      mfc="black",
      mec="white",
      mew=0.6,
      ls="none",
      transform=PROJ,
      zorder=6,
  )
  ax.plot(
      track["lon"].iloc[[0, -1]],
      track["lat"].iloc[[0, -1]],
      "o",
      ms=ms + 4,
      mfc="black",
      mec="white",
      mew=1.0,
      ls="none",
      transform=PROJ,
      zorder=7,
  )

  # Whitelist ONLY the specific timestamps to display
  if label_all_00:
    target_labels = {"2512", "2600", "2800", "2812"}
  else:
    target_labels = {
        str(track["dt_str"].iloc[0]),
        str(track["dt_str"].iloc[-1]),
    }

  label_offsets = {
      "2512": (0.18, 0.18, "left", "bottom"),  # Up into Andaman Sea
      "2600": (-0.18, 0.14, "right", "bottom"),  # Out to the northwest
      "2800": (0.15, 0.18, "left", "bottom"),  # Above track
      "2812": (0.20, 0.00, "left", "center"),  # East of track terminus
  }

  drawn = set()
  for _, row in track.iterrows():
    dt_key = str(row["dt_str"]).strip()

    # Hard stop: never draw 2700 or any label from Nov 27
    if dt_key.startswith("27") or (dt_key not in target_labels):
      continue

    # De-duplicate across multi-agency IBTrACS records
    if dt_key in drawn:
      continue
    drawn.add(dt_key)

    dx, dy, ha, va = label_offsets.get(dt_key, (0.14, 0.14, "left", "bottom"))
    ax.text(
        row["lon"] + dx,
        row["lat"] + dy,
        dt_key,
        fontsize=ANNOT_FS,
        fontweight="bold",
        ha=ha,
        va=va,
        transform=PROJ,
        zorder=12,
        path_effects=stroke(2.5),
    )


def draw_disdro_point(ax, track):
  pt = track[track["dt_str"] == DISDRO_STR]
  if pt.empty:
    return
  lon_pt = pt["lon"].iloc[0]
  lat_pt = pt["lat"].iloc[0]

  ax.plot(
      lon_pt,
      lat_pt,
      "o",
      ms=10,
      mfc="limegreen",
      mec="black",
      mew=1.2,
      transform=PROJ,
      zorder=9,
  )
  ax.text(
      lon_pt + 0.12,
      lat_pt - 0.35,
      DISDRO_STR,
      fontsize=ANNOT_FS,
      fontweight="bold",
      ha="left",
      va="top",
      color="limegreen",
      transform=PROJ,
      zorder=12,
      path_effects=stroke(2.5, "black"),
  )


def interp_track(track, t):
  """TC center (lon, lat) interpolated to datetime t."""
  x = track["datetime_utc"].astype("int64").to_numpy().astype(float)
  xt = float(pd.Timestamp(t).value)
  return (
      np.interp(xt, x, track["lon"].to_numpy()),
      np.interp(xt, x, track["lat"].to_numpy()),
  )


def decorate_snapshot(ax, track, cur, label, kt_fc="magenta"):
  """Common decoration for snapshot maps: grid, KT, track, TC center, title."""
  gl = ax.gridlines(
      draw_labels=True, crs=PROJ, linestyle="--", color="gray", alpha=0.5
  )
  gl.xlocator = mticker.FixedLocator([88, 92, 96, 100, 104])
  gl.ylocator = mticker.FixedLocator([-4, 0, 4, 8, 12])
  gl.xlabel_style = {"size": TICK_FS}
  gl.ylabel_style = {"size": TICK_FS}
  gl.top_labels = False
  gl.right_labels = False

  draw_kt(ax, ms=10, fc=kt_fc)
  ax.plot(
      track["lon"],
      track["lat"],
      color="white",
      lw=3.5,
      transform=PROJ,
      zorder=4.8,
  )
  ax.plot(
      track["lon"],
      track["lat"],
      "-o",
      color="black",
      lw=1.8,
      ms=4,
      transform=PROJ,
      zorder=5,
  )
  lon_now, lat_now = interp_track(track, cur)
  ax.scatter(
      [lon_now],
      [lat_now],
      marker="*",
      s=380,
      color="magenta",
      edgecolor="black",
      linewidth=1.0,
      transform=PROJ,
      zorder=8,
  )
  ax.set_title(
      f"{label} {cur:%d%H} UTC",
      loc="left",
      fontsize=TITLE_FS,
      fontweight="bold",
      pad=12,
  )


# ── 2b. Radar helpers ──────────────────────────────────────────────────────
def build_radar_index():
  """{site: sorted [(datetime, path), ...]}; time = first 14 digits of the filename (UTC)."""
  idx = {}
  for site in RADAR_SITES:
    folder, items = os.path.join(RADAR_DIR, site), []
    if os.path.isdir(folder):
      for fn in os.listdir(folder):
        m = re.search(r"(\d{14})", fn)
        if m:
          try:
            items.append((
                datetime.strptime(m.group(1), "%Y%m%d%H%M%S"),
                os.path.join(folder, fn),
            ))
          except ValueError:
            continue
    idx[site] = sorted(items)
    if not os.path.isdir(folder):
      print(f"  WARNING: radar folder not found: {os.path.abspath(folder)}")
    elif not items:
      print(
          f"  WARNING: no parsable radar files in {os.path.abspath(folder)}"
      )
    else:
      print(
          f"  {site}: {len(items)} files, {idx[site][0][0]:%d %b %H:%M} to"
          f" {idx[site][-1][0]:%d %b %H:%M} UTC"
      )
  return idx


def nearest_scans(idx, t):
  """Per radar: the single nearest scan within RADAR_TOL of t."""
  picks = {}
  for site, items in idx.items():
    if not items:
      continue
    d, p = min(items, key=lambda x: abs(x[0] - t))
    if abs(d - t) <= RADAR_TOL:
      picks[site] = (d, p)
  return picks


def _block(a, k, func):
  h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
  return func(a[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3))


def radar_composite(picks):
  """Max-mosaic of the picked scans with bounding-box coverage mask."""
  srcs = []
  bounds_list = []
  for site, (_, p) in picks.items():
    try:
      src = rasterio.open(p)
      srcs.append(src)
      bounds_list.append(src.bounds)
    except Exception as e:
      print(f"  cannot open {p}: {e}")
  if not srcs:
    return None

  try:
    mosaic, tr = merge(
        srcs,
        method="max",
        nodata=-9999.0,
        dtype="float32",
        bounds=(EXTENT_A[0], EXTENT_A[2], EXTENT_A[1], EXTENT_A[3]),
    )
  except Exception as e:
    print(f"  merge failed: {e}")
    return None
  finally:
    for s in srcs:
      s.close()

  z = np.squeeze(mosaic)
  h, w = z.shape

  lon_arr = tr.c + (np.arange(w) + 0.5) * tr.a
  lat_arr = tr.f + (np.arange(h) + 0.5) * tr.e

  cover = z > -9990
  for b in bounds_list:
    ix = (lon_arr >= b.left) & (lon_arr <= b.right)
    iy = (lat_arr >= b.bottom) & (lat_arr <= b.top)
    if np.any(iy) and np.any(ix):
      cover[np.ix_(iy, ix)] = True

  z = np.where(cover & (z >= 5.0), z, np.nan)

  extent = [tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f]

  k = RADAR_DS
  cc = (_block(cover.astype(float), k, np.mean) > 0.5).astype(float)
  hc, wc = cc.shape
  lon = (tr.c + (np.arange(wc * k) + 0.5) * tr.a).reshape(wc, k).mean(1)
  lat = (tr.f + (np.arange(hc * k) + 0.5) * tr.e).reshape(hc, k).mean(1)

  zmax = np.nanmax(z) if np.isfinite(z).any() else float("nan")
  print(
      f"  composite max {zmax:.1f} dBZ, covered fraction of domain"
      f" {cover.mean():.2%}"
  )
  return dict(z=z, extent=extent, cov=(lon, lat, cc))


def draw_radar_panel(ax, rad, cmap, norm):
  """Grey = no radar coverage, white = covered, colours = reflectivity."""
  ax.set_facecolor(NOCOV_GRAY)
  if rad is None:
    ax.text(
        0.5,
        0.5,
        "No radar data",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="0.35",
        zorder=3,
    )
    return
  lon, lat, cc = rad["cov"]
  if cc.max() >= 0.5:
    ax.contourf(
        lon,
        lat,
        cc,
        levels=[0.5, 1.5],
        colors="white",
        transform=PROJ,
        zorder=1,
    )
  ax.imshow(
      np.ma.masked_invalid(rad["z"]),
      origin="upper",
      extent=rad["extent"],
      transform=PROJ,
      cmap=cmap,
      norm=norm,
      interpolation="nearest",
      zorder=2,
  )
  ax.set_extent(EXTENT_A, crs=PROJ)


# ── 3. Data Loading ────────────────────────────────────────────────────────
print("Loading IBTrACS data...")
track = pd.read_csv(
    "../Track/ibtracs.since1980.list.v04r01.csv",
    low_memory=False,
    header=0,
    skiprows=[1],
)
track.columns = track.columns.str.strip().str.lower()
track["datetime"] = pd.to_datetime(
    track["iso_time"], format="%Y-%m-%d %H:%M:%S", utc=True
)
for col in ("lat", "lon", "usa_wind", "usa_pres"):
  track[col] = pd.to_numeric(track[col], errors="coerce")

track = (
    track[track["name"].str.strip().str.upper() == "SENYAR"]
    .dropna(subset=["datetime", "lat", "lon"])
    .sort_values("datetime")
    .reset_index(drop=True)
)
track["dist_km"] = haversine(track["lat"], track["lon"], SITE_LAT, SITE_LON)
track["wind_ms"] = track["usa_wind"] * KT_PER_MS
track["datetime_utc"] = track["datetime"].dt.tz_localize(None)
track["dt_str"] = track["datetime_utc"].dt.strftime("%d%H")

print("Loading ERA5 data...")
ds_850 = xr.open_dataset("../ERA5/era5_senyar_850mb_new.nc").sel(
    valid_time=slice(T_START, T_END), pressure_level=850
)
ds_sl = xr.open_dataset("../ERA5/era5_senyar_singlelevel_accum.nc").sel(
    valid_time=slice(T_START, T_END)
)
u_mean = ds_850["u"].mean("valid_time")
v_mean = ds_850["v"].mean("valid_time")
vimd_sum = ds_sl["vimd"].sum("valid_time")
lats_e5, lons_e5 = ds_850["latitude"], ds_850["longitude"]

print("Loading DEM...")
with rasterio.open("../DEM_SRTM/dem_sumatera_a_1.jp2") as src:
  W, E, S, N = EXTENT_A
  fb = src.bounds
  cW, cE = max(W, fb.left), min(E, fb.right)
  cS, cN = max(S, fb.bottom), min(N, fb.top)
  window = from_bounds(cW, cS, cE, cN, transform=src.transform)
  nx = 2500
  ny = int(nx * (cN - cS) / (cE - cW))
  dem = src.read(
      1, window=window, out_shape=(ny, nx), resampling=Resampling.bilinear
  ).astype(float)
  if src.nodata is not None:
    dem = np.where(dem == src.nodata, np.nan, dem)

DEM_MAX = 3000
dem_land = np.where(dem <= 0, np.nan, dem)

terrain_land = mcolors.LinearSegmentedColormap.from_list(
    "terrain_land", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
)
terrain_land.set_bad(alpha=0.0)
ls = LightSource(azdeg=315, altdeg=45)
rgb_dem = ls.shade(
    np.ma.masked_invalid(dem_land),
    cmap=terrain_land,
    blend_mode="overlay",
    vmin=0,
    vmax=DEM_MAX,
    vert_exag=0.1,
)

print("Loading Himawari-9 TBB...")
lut = np.loadtxt("../HIMAWARI-9/2025/tir.01")
lut_arr = np.full(65536, np.nan, dtype=np.float32)
lut_arr[lut[:, 0].astype(np.int32)] = lut[:, 1].astype(np.float32) - K0

lon_full = np.linspace(85.0, 205.0, 6000)
lat_full = np.linspace(60.0, -60.0, 6000)
ix = (lon_full >= EXTENT_A[0] - 1) & (lon_full <= EXTENT_A[1] + 1)
iy = (lat_full >= EXTENT_A[2] - 1) & (lat_full <= EXTENT_A[3] + 1)
lon_c, lat_c = lon_full[ix], lat_full[iy]

tbb_fields = []
for t in TBB_TIMES:
  bz2_file = (
      f"../HIMAWARI-9/{t:%Y}/{t:%m}/{t:%Y%m%d%H%M}.tir.01.fld.geoss.bz2"
  )
  with bz2.open(bz2_file, "rb") as f_in:
    raw = np.frombuffer(f_in.read(), dtype=">u2").reshape((6000, 6000))
  raw = raw[np.ix_(iy, ix)].astype(np.int64)
  field = lut_arr[raw]
  field[raw == 65535] = np.nan
  tbb_fields.append(field)

print("Building radar composites...")
radar_idx = build_radar_index()
radar_fields, radar_sites = [], []
for t_ in TBB_TIMES:
  picks = nearest_scans(radar_idx, t_)
  desc = ", ".join(f"{s} ({d:%H:%M:%S})" for s, (d, _) in picks.items())
  print(f"{t_:%d%H} UTC radars: {desc if desc else 'NONE'}")
  radar_fields.append(radar_composite(picks) if picks else None)
  radar_sites.append([f"{s} {d:%H:%M}" for s, (d, _) in picks.items()])

# ── 4. Figure layout: 3 rows x 3 columns ───────────────────────────────────
print("Plotting Figure 1...")
fig = plt.figure(figsize=(22, 24))

gs = gridspec.GridSpec(
    3,
    3,
    figure=fig,
    height_ratios=[1, 1, 1],
    hspace=0.01,
    wspace=0.40,
    top=0.96,
    bottom=0.07,
)

ax_a = fig.add_subplot(gs[0, 0], projection=PROJ)
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2], projection=PROJ)
ax_d = [fig.add_subplot(gs[1, i], projection=PROJ) for i in range(3)]
ax_r = [fig.add_subplot(gs[2, i], projection=PROJ) for i in range(3)]

# Match the box aspect of (b) to the maps
map_aspect = (EXTENT_A[3] - EXTENT_A[2]) / (EXTENT_A[1] - EXTENT_A[0])
ax_b.set_box_aspect(map_aspect)

# ══════════════════════════════════════════════════════════════════════════
# (a) Topography + track
# ══════════════════════════════════════════════════════════════════════════
ax_a.set_extent(EXTENT_A, crs=PROJ)
ax_a.add_feature(cfeature.OCEAN, facecolor="#CFE3F1", zorder=0)
ax_a.add_feature(cfeature.LAND, facecolor="#EDEDED", zorder=0)
ax_a.imshow(
    rgb_dem, extent=[cW, cE, cS, cN], transform=PROJ, origin="upper", zorder=1
)
ax_a.coastlines(resolution="10m", linewidth=1.2, color="black", zorder=4)
ax_a.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black", zorder=4)
format_map_axes(ax_a, [88, 92, 96, 100, 104], [-4, 0, 4, 8, 12])

draw_track(ax_a, track, label_all_00=True)
draw_disdro_point(ax_a, track)
draw_kt(ax_a)

sm_dem = mcm.ScalarMappable(
    norm=mcolors.Normalize(0, DEM_MAX), cmap=terrain_land
)
#cax_a = ax_a.inset_axes([0.045, -0.20, 0.92, 0.05])
cax_a = ax_a.inset_axes([0.045, -0.15, 0.92, 0.05])
cb_a = fig.colorbar(
    sm_dem,
    cax=cax_a,
    orientation="horizontal",
    extend="max",
    ticks=np.arange(0, DEM_MAX + 1, 500),
)
cb_a.set_label("Elevation (m)", fontsize=LABEL_FS, fontweight="bold")
cb_a.ax.tick_params(labelsize=TICK_FS)
ax_a.set_title(
    "(a) Topography & Track",
    loc="left",
    fontsize=TITLE_FS,
    fontweight="bold",
    pad=12,
)

# ══════════════════════════════════════════════════════════════════════════
# (b) Time series (cyclone parameters) + snapshot markers
# ══════════════════════════════════════════════════════════════════════════
t = track["datetime_utc"]
c_pres, c_wind, c_dist = "#1565C0", "#C62828", "black"

ax_b.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
ax_b.grid(True, axis="x", linestyle="--", alpha=0.4, zorder=0)

(lp,) = ax_b.plot(
    t, track["usa_pres"], color=c_pres, lw=3.0, zorder=4, label="Min. Pressure"
)
ax_b.set_ylabel(
    "Minimum Pressure (hPa)", color=c_pres, fontsize=LABEL_FS, fontweight="bold"
)
ax_b.tick_params(axis="y", labelcolor=c_pres, labelsize=TICK_FS)
ax_b.spines["left"].set_color(c_pres)
ax_b.spines["left"].set_linewidth(2)
ax_b.spines["right"].set_visible(False)
ax_b.spines["top"].set_visible(False)

ax_w = ax_b.twinx()
ax_w.set_box_aspect(map_aspect)
(lw_,) = ax_w.plot(
    t,
    track["wind_ms"],
    color=c_wind,
    lw=3.0,
    zorder=4,
    label="Max. Wind Speed",
)
ax_w.set_ylabel(
    "Max. Wind Speed (m s$^{-1}$)",
    color=c_wind,
    fontsize=LABEL_FS,
    fontweight="bold",
)
ax_w.tick_params(axis="y", labelcolor=c_wind, labelsize=TICK_FS)
ax_w.spines["right"].set_color(c_wind)
ax_w.spines["right"].set_linewidth(2)
ax_w.spines["left"].set_visible(False)
ax_w.spines["top"].set_visible(False)

ax_dist = ax_b.twinx()
ax_dist.set_box_aspect(map_aspect)
ax_dist.spines["right"].set_position(("outward", 45))
ax_dist.spines["right"].set_color(c_dist)
ax_dist.spines["right"].set_linewidth(2)
ax_dist.spines["left"].set_visible(False)
ax_dist.spines["top"].set_visible(False)

(ld,) = ax_dist.plot(
    t,
    track["dist_km"],
    color=c_dist,
    lw=2.5,
    ls="--",
    zorder=3,
    label="Distance to KT",
)
ax_dist.set_ylabel(
    "Distance to Kototabang (km)",
    fontsize=LABEL_FS,
    color=c_dist,
    fontweight="bold",
)
ax_dist.tick_params(axis="y", labelsize=TICK_FS, labelcolor=c_dist)

ax_b.set_xlim(t.iloc[0], t.iloc[-1])
ax_b.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%d%H"))
ax_b.tick_params(axis="x", labelsize=TICK_FS)
ax_b.set_xlabel("Time (DDHH UTC)", fontsize=LABEL_FS, fontweight="bold")

# End of disdrometer record (power outage)
ax_b.axvline(OBS_END, color="limegreen", ls=":", lw=3, zorder=2)

# Snapshot times of panels (d)-(f) / (g)-(i)
for lbl, tt in zip(["(d,g)", "(e,h)", "(f,i)"], TBB_TIMES):
  ax_b.axvline(pd.Timestamp(tt), color="gray", ls=":", lw=1.8, zorder=1)
  ax_b.text(
      pd.Timestamp(tt),
      0.03,
      lbl,
      transform=ax_b.get_xaxis_transform(),
      ha="left",
      va="bottom",
      fontsize=ANNOT_FS,
      fontweight="bold",
      bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85),
      zorder=6,
  )

leg_b = ax_dist.legend(
    handles=[lp, lw_, ld],
    loc="upper right",
    bbox_to_anchor=(0.93, 0.88),
    ncol=1,
    fontsize=11,
    frameon=True,
    framealpha=0.9,
    edgecolor="gray",
)
leg_b.set_zorder(30)
ax_b.set_title(
    "(b) Cyclone Parameters & Distance",
    loc="left",
    fontsize=TITLE_FS,
    fontweight="bold",
    pad=12,
)

# ══════════════════════════════════════════════════════════════════════════
# (c) ERA5 mean 850 hPa wind + accumulated VIMD
# ══════════════════════════════════════════════════════════════════════════
ax_c.set_extent(EXTENT_A, crs=PROJ)
levels_v = [-50, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 50]
cmap_v = plt.get_cmap("RdBu_r").resampled(len(levels_v) - 1)
norm_v = mcolors.BoundaryNorm(levels_v, ncolors=cmap_v.N, clip=True)

cf = ax_c.contourf(
    lons_e5,
    lats_e5,
    vimd_sum,
    levels=levels_v,
    cmap=cmap_v,
    norm=norm_v,
    extend="both",
    transform=PROJ,
    zorder=1,
)

skip = 3
Q = ax_c.quiver(
    lons_e5[::skip],
    lats_e5[::skip],
    u_mean[::skip, ::skip],
    v_mean[::skip, ::skip],
    scale=400,
    width=0.004,
    headwidth=4,
    color="black",
    transform=PROJ,
    zorder=3,
)
ax_c.add_patch(
    mpatches.Rectangle(
        (0.02, 0.02),
        0.30,
        0.09,
        transform=ax_c.transAxes,
        facecolor="white",
        edgecolor="black",
        zorder=10,
    )
)
ax_c.quiverkey(
    Q,
    X=0.06,
    Y=0.065,
    U=15,
    label="15 m s$^{-1}$",
    labelpos="E",
    coordinates="axes",
    labelsep=0.05,
    fontproperties={"size": 12, "weight": "bold"},
    zorder=11,
)

ax_c.coastlines(resolution="10m", linewidth=1.2, color="black", zorder=4)
ax_c.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black", zorder=4)
format_map_axes(ax_c, [88, 92, 96, 100, 104], [-4, 0, 4, 8, 12], grid=False)

draw_track(ax_c, track, lw=2.2, ms=4)
draw_kt(ax_c, ms=9)

#cax_c = ax_c.inset_axes([0.045, -0.20, 0.92, 0.05])
cax_c = ax_c.inset_axes([0.045, -0.15, 0.92, 0.05])
cb_c = fig.colorbar(
    cf,
    cax=cax_c,
    orientation="horizontal",
    extendfrac=0,
    ticks=[-50, -30, -10, 10, 30, 50],
)
cb_c.set_label("Acc. VIMD (kg m$^{-2}$)", fontsize=LABEL_FS, fontweight="bold")
cb_c.ax.tick_params(labelsize=TICK_FS - 1)
ax_c.set_title(
    "(c) ERA5 850 hPa Wind & Acc. VIMD",
    loc="left",
    fontsize=TITLE_FS,
    fontweight="bold",
    pad=12,
)

# ══════════════════════════════════════════════════════════════════════════
# (d)-(f) Himawari-9 TBB
# ══════════════════════════════════════════════════════════════════════════
levels_t = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10]
colors_t = [
    "#4d004b",
    "#b30000",
    "#e34a33",
    "#fc8d59",
    "#fdcc8a",
    "#ffffbf",
    "#c7e9b4",
    "#7fcdbb",
    "#41b6c4",
    "#d0d0d0",
]
cmap_t = ListedColormap(colors_t)
norm_t = BoundaryNorm(levels_t, cmap_t.N, clip=True)

for i, (ax, cur, field) in enumerate(zip(ax_d, TBB_TIMES, tbb_fields)):
  ax.set_extent(EXTENT_A, crs=PROJ)
  pcm = ax.pcolormesh(
      lon_c,
      lat_c,
      field,
      cmap=cmap_t,
      norm=norm_t,
      shading="auto",
      transform=PROJ,
      rasterized=True,
  )
  ax.coastlines(resolution="10m", linewidth=1.2, color="black")
  decorate_snapshot(ax, track, cur, ["(d)", "(e)", "(f)"][i])

# Shared horizontal colorbar for (d)-(f)
#cax_t = ax_d[1].inset_axes([-0.9, -0.21, 2.8, 0.04])
cax_t = ax_d[1].inset_axes([-0.9, -0.15, 2.8, 0.04])
cbar_t = fig.colorbar(pcm, cax=cax_t, orientation="horizontal", ticks=levels_t)
cbar_t.set_label(
    "Cloud Top Temperature (°C)", fontsize=LABEL_FS, fontweight="bold"
)
cbar_t.ax.tick_params(labelsize=TICK_FS)

# ══════════════════════════════════════════════════════════════════════════
# (g)-(i) BMKG radar composite reflectivity (CMAX)
# ══════════════════════════════════════════════════════════════════════════
levels_r = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
colors_r = [
    "#00FFFF",
    "#00A5FF",
    "#0000FF",
    "#00FF00",
    "#00C800",
    "#008C00",
    "#FFFF00",
    "#FFC800",
    "#FF9600",
    "#FF0000",
    "#C80000",
    "#FF00FF",
    "#9600B4",
]
cmap_r = ListedColormap(colors_r)
norm_r = BoundaryNorm(levels_r, cmap_r.N)

for i, (ax, cur) in enumerate(zip(ax_r, TBB_TIMES)):
  ax.set_extent(EXTENT_A, crs=PROJ)
  draw_radar_panel(ax, radar_fields[i], cmap_r, norm_r)
  ax.coastlines(resolution="10m", linewidth=1.2, color="black", zorder=3)
  decorate_snapshot(ax, track, cur, ["(g)", "(h)", "(i)"][i])

  if radar_sites[i]:
    ax.text(
        0.98,
        0.72,
        "Scan times (UTC):\n" + "\n".join(radar_sites[i]),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        zorder=12,
        bbox=dict(
            facecolor="white", edgecolor="gray", pad=2.5, alpha=0.9
        ),
    )

# Legend for the background shading
cov_handles = [
    mpatches.Patch(
        facecolor="white", edgecolor="gray", label="Radar coverage"
    ),
    mpatches.Patch(
        facecolor=NOCOV_GRAY, edgecolor="gray", label="No radar coverage"
    ),
]
ax_r[0].legend(
    handles=cov_handles,
    loc="lower left",
    bbox_to_anchor=(0.0, 0.05),
    fontsize=11,
    framealpha=0.95,
)

# Shared horizontal colorbar for (g)-(i)
sm_r = mcm.ScalarMappable(norm=norm_r, cmap=cmap_r)
#cax_r = ax_r[1].inset_axes([-0.9, -0.21, 2.8, 0.04])
cax_r = ax_r[1].inset_axes([-0.9, -0.15, 2.8, 0.04])
cbar_r = fig.colorbar(sm_r, cax=cax_r, orientation="horizontal", ticks=levels_r)
cbar_r.set_label("Radar Reflectivity (dBZ)", fontsize=LABEL_FS, fontweight="bold")
cbar_r.ax.tick_params(labelsize=TICK_FS)

# ── Save ───────────────────────────────────────────────────────────────────
out_path_png = "Fig1_rev.png"
out_path_tif = "Fig1_rev.tif"

plt.savefig(out_path_png)
print(f"Saved: {out_path_png}")

# TIFFs are usually required for publications, so adding dpi=300 is recommended
plt.savefig(out_path_tif, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path_tif}")

plt.show()
