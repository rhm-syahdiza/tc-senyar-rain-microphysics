"""
Mean fall speed vs. drop diameter from Parsivel disdrometer.
"""

import os, csv, datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

# ── rcParams───────────────────────────────────────────────
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

LABEL_FS = 14
TICK_FS  = 12

# ── Parsivel bin centers ───────────────────────────────────────────────────────
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

# ── Atlas et al. (1973) empirical fall speed v(D) = 9.65 − 10.3e(−0.6D) (ρ/ρ0)0.4 ,
def atlas_v_corrected(d, altitude_m=865.0):
    """
    Parameters:
    d: Drop diameter in mm.
    altitude_m: Altitude of the site in meters ASL. Kototabang (865 m).
    """
    # 1. Terminal velocity at sea level
    v_sea_level = 9.65 - 10.3 * np.exp(-0.6 * d)
    
    # 2. Standard Atmosphere constants
    rho_0 = 1.225    # Sea-level air density (kg/m^3)
    T_0   = 288.15   # Sea-level temperature (K)
    L     = 0.0065   # Temperature lapse rate (K/m)
    R     = 287.05   # Gas constant for dry air (J/(kg*K))
    g     = 9.80665  # Gravity (m/s^2)

    # 3. Barometric calculation for density at altitude (rho)
    T_z = T_0 - (L * altitude_m)
    rho_z = rho_0 * (T_z / T_0) ** ((g / (L * R)) - 1)

    # 4. Density correction factor: (rho_0 / rho)^0.4
    correction_factor = (rho_0 / rho_z) ** 0.4

    # 5. Apply correction
    v_corrected = v_sea_level * correction_factor
    return np.maximum(v_corrected, 0.0)

V_emp = atlas_v_corrected(D_bins, altitude_m=865.0)

# ── QC valid diameter bins ──
D_valid_mask = (D_bins <= 0.312) & (D_bins <= 10.0)

# ── Time window in LOCAL time (GMT+7) ─────────────────────────────────────────
T_START = dt.datetime(2025, 11, 25, 12, 0, 0) + dt.timedelta(hours=7)
T_END   = dt.datetime(2025, 11, 28, 12, 0, 0) + dt.timedelta(hours=7)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_root  = os.path.join(script_dir, "Kototabang")
N = np.zeros((32, 32), dtype=float)

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
                        t_local = dt.datetime.strptime(f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S")
                        if not (T_START <= t_local <= T_END): continue

                        # QC: Rain rate >= 0.1 mm/h
                        rain_rate = float(row[2])
                        if rain_rate < 0.1: continue

                        spec_raw = row[15:1038]
                        vals = [float(x) if x.strip() else 0.0 for x in spec_raw]
                        vals += [0.0] * (1024 - len(vals))
                        spec = np.array(vals).reshape(32, 32)

                        # QC: invalid diameter bins (> 0.2 mm and > 10 mm)
                        spec[:, ~D_valid_mask] = 0.0

                        # QC: Atlas velocity filter (±60% of V_emp)
                        for d_i in range(32):
                            if not D_valid_mask[d_i]: continue
                            v_emp = V_emp[d_i]
                            for v_i in range(32):
                                if not (0.4 * v_emp <= V_bins[v_i] <= 1.6 * v_emp):
                                    spec[v_i, d_i] = 0.0

                        # QC: drop count >= 10 
                        if spec.sum() < 10: continue

                        N += spec
                        valid_sample_count += 1
                        
                    except Exception:
                        continue

# ── Mean & std fall speed per diameter ───────────────
mean_v = np.full(32, np.nan)
std_v  = np.full(32, np.nan)

for d in range(32):
    if not D_valid_mask[d]: continue
    col   = N[:, d]
    total = col.sum()
    if total < 20: continue
    mean_v[d] = np.sum(V_bins * col) / total
    std_v[d]  = np.sqrt(np.sum(col * (V_bins - mean_v[d])**2) / total)

valid = ~np.isnan(mean_v)

# ── Plot ──────────────────────────────────────────────────────────────────────
D_line = np.linspace(0.3, 8.0, 200)

fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
ax.set_facecolor("white")

def make_edges(centers):
    mids = 0.5 * (centers[:-1] + centers[1:])
    left  = centers[0]  - (centers[1]  - centers[0])  / 2
    right = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[left], mids, [right]])

D_edges = make_edges(D_bins)
V_edges = make_edges(V_bins)

N_plot = np.where(N > 0, N, np.nan)
norm = mcolors.LogNorm(vmin=1, vmax=np.nanmax(N_plot))
mesh = ax.pcolormesh(
    D_edges, V_edges, N_plot,
    cmap="viridis", norm=norm, shading="flat", zorder=1
)

cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label("Drop count", fontsize=LABEL_FS, fontweight="bold")
cbar.ax.tick_params(labelsize=TICK_FS)
cbar.outline.set_linewidth(1.2)

# ±60 % Atlas threshold lines 
ax.plot(D_line, 0.4 * atlas_v_corrected(D_line), color="black",
        linewidth=1.5, linestyle="--", zorder=3, label="±60 % Atlas (1973)")
ax.plot(D_line, 1.6 * atlas_v_corrected(D_line), color="black",
        linewidth=1.5, linestyle="--", zorder=3)

# Atlas et al. reference curve 
ax.plot(D_line, atlas_v_corrected(D_line), color="black",
        linewidth=2.0, linestyle="-", zorder=4, label="Atlas et al. (1973)")

# Mean fall speed with ±1 σ error bars
ax.errorbar(
    D_bins[valid], mean_v[valid], yerr=std_v[valid],
    fmt="s", mfc="red", mec="white", ecolor="red",
    markersize=7, capsize=4, linewidth=1.5,
    zorder=5, label="Mean ± σ (obs.)"
)

# Axes Formatting
ax.set_xlabel("Diameter (mm)", fontsize=LABEL_FS, fontweight="bold")
ax.set_ylabel("Fall Speed (m s$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
ax.set_xlim(0, 8)  
ax.set_ylim(0, 15)

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.tick_params(which="both", top=True, right=True)
ax.tick_params(which="major", length=6)
ax.tick_params(which="minor", length=3)
ax.tick_params(axis='both', labelsize=TICK_FS)

#legend
ax.legend(fontsize=11, loc="upper left", frameon=True, framealpha=0.9, edgecolor="gray")

plt.tight_layout()

out = os.path.join(script_dir, "Fig2.png")
plt.savefig(out)

# Print the sample size
print(f"✅ Total valid 1-minute spectra (sample size): {valid_sample_count}")
print(f"Saved → {out}")

plt.show()
