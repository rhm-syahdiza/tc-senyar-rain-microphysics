"""
Scatter plot of log10(Nw) vs Dm, colored by Liquid Water Content (LWC), 
with marker shapes indicating Rain Rate (R) categories.
Overlaid with Bringi et al. (2003,2009) and Thompson et al. (2015) reference boundaries.
"""

import os, csv, datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── rcParams ────────────────────────────────────────────────────
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

LABEL_FS, TICK_FS = 14, 12

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

# QC───────────────────
def atlas_v_corrected(d, altitude_m=865.0):
    """Density-corrected terminal velocity (m/s) for elevated sites."""
    v_sea_level = 9.65 - 10.3 * np.exp(-0.6 * d)
    rho_0, T_0, L, R, g = 1.225, 288.15, 0.0065, 287.05, 9.80665
    T_z = T_0 - (L * altitude_m)
    rho_z = rho_0 * (T_z / T_0) ** ((g / (L * R)) - 1)
    correction_factor = (rho_0 / rho_z) ** 0.4
    return np.maximum(v_sea_level * correction_factor, 0.0)

D_valid_mask = (D_bins >= 0.312) & (D_bins <= 10.0) 
V_term = atlas_v_corrected(D_bins, altitude_m=865.0)
V_FRAC, MIN_DROPS, R_MIN = 0.60, 10, 0.1
A_eff = (180.0 * (30.0 - (D_bins / 2.0))) * 1e-6
t_int = 60.0

# ── Rainrate Categories ───────────────────────────────────────────────────────────
rr_cats = [
    ("R < 1",  0.1,   1, "o", 30),
    ("1–5",    1,   5.0, "s", 30),
    ("5–10",     5.0,  10.0, "^", 40),
    ("10–20",   10.0,  20.0, "D", 40),
    ("20–50",   20.0,  50.0, "v", 50),
    ("R > 50",  50.0, 9999., "P", 60),
]

# TC Senyar period
T_START = dt.datetime(2025, 11, 25, 12, 0, 0) + dt.timedelta(hours=7)
T_END   = dt.datetime(2025, 11, 28, 12, 0, 0) + dt.timedelta(hours=7)

# ── Data Processing ───────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root  = os.path.join(script_dir, "Kototabang")

results = {c[0]: {"Dm": [], "logNw": [], "LWC": []} for c in rr_cats}

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
                        if not (T_START <= t_lt <= T_END): continue
                        
                        rr = float(row[2])
                        if rr < R_MIN: continue

                        spec_raw = row[15:1038]
                        vals = [float(x) if x.strip() else 0.0 for x in spec_raw]
                        vals += [0.0] * (1024 - len(vals))
                        spec = np.array(vals).reshape(32, 32)

                        # QC: Velocity & Drop Count
                        for d_i in range(32):
                            v_lo, v_hi = V_term[d_i]*(1-V_FRAC), V_term[d_i]*(1+V_FRAC)
                            for v_i in range(32):
                                if not (v_lo <= V_bins[v_i] <= v_hi): spec[v_i, d_i] = 0.0
                        
                        spec[:, ~D_valid_mask] = 0.0
                        if spec.sum() < MIN_DROPS: continue

                        # Calculate N(D)
                        n_d = np.zeros(32)
                        for d_i in range(32):
                            if not D_valid_mask[d_i]: continue
                            with np.errstate(divide='ignore', invalid='ignore'):
                                n_d[d_i] = np.sum(np.where(V_bins > 0, spec[:, d_i] / (V_bins * A_eff[d_i] * t_int * dD[d_i]), 0.0))

                        # Apply Physics Equations
                        nd_valid = n_d[D_valid_mask]
                        D_valid  = D_bins[D_valid_mask]
                        dw_valid = dD[D_valid_mask]

                        M3 = np.sum(nd_valid * (D_valid**3) * dw_valid)
                        M4 = np.sum(nd_valid * (D_valid**4) * dw_valid)
                        if M3 <= 0 or M4 <= 0: continue

                        Dm    = M4 / M3
                        LWC   = (np.pi / 6000) * M3          
                        Nw    = (4**4 / 6.0) * (M3**5 / M4**4)
                        if Nw <= 0: continue
                        
                        logNw = np.log10(Nw)

                        # Store in appropriate category
                        for name, lo, hi, *_ in rr_cats:
                            if lo <= rr < hi:
                                results[name]["Dm"].append(Dm)
                                results[name]["logNw"].append(logNw)
                                results[name]["LWC"].append(LWC)
                                break
                    except Exception:
                        continue

# ── Sample Distribution Statistics (Bringi et al. 2003 Clusters) ─────────────
all_Dm_points = []
all_logNw_points = []

for cat in results:
    all_Dm_points.extend(results[cat]["Dm"])
    all_logNw_points.extend(results[cat]["logNw"])

total_qc_samples = len(all_Dm_points)

print("\n─── CLIMATOLOGICAL CLUSTER ANALYSIS ───")
if total_qc_samples > 0:
    Dm_array = np.array(all_Dm_points)
    logNw_array = np.array(all_logNw_points)
    
    # Continental Convective Cluster
    continental_mask = (Dm_array >= 2.0) & (Dm_array <= 2.75) & (logNw_array >= 3.0) & (logNw_array <= 3.5)
    continental_counts = np.sum(continental_mask)
    continental_pct = (continental_counts / total_qc_samples) * 100
    
    # Maritime Convective Cluster
    maritime_mask = (Dm_array >= 1.5) & (Dm_array <= 1.75) & (logNw_array >= 4.0) & (logNw_array <= 4.5)
    maritime_counts = np.sum(maritime_mask)
    maritime_pct = (maritime_counts / total_qc_samples) * 100

    print(f"Total QC Passed Samples Evaluated: {total_qc_samples}")
    print(f"Continental Cluster Counts       : {continental_counts} samples")
    print(f"Continental Cluster Percentage   : {continental_pct:.2f}%")
    print(f"Maritime Cluster Counts          : {maritime_counts} samples")
    print(f"Maritime Cluster Percentage      : {maritime_pct:.2f}%\n")
else:
    print("❌ Stats Error: No validated samples available to compute cluster density.\n")

# ── Figure ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")

all_lwc = np.concatenate([results[c[0]]["LWC"] for c in rr_cats if results[c[0]]["LWC"]])
vmin, vmax = 0, np.percentile(all_lwc, 99) if len(all_lwc) else 5.0
cmap = plt.cm.viridis

# Scatter points by category
for name, lo, hi, marker, ms in rr_cats:
    d = results[name]
    if not d["Dm"]: continue
    
    # Sorting arrays by LWC ensures brighter (higher LWC) dots plot on top of darker ones
    sort_idx = np.argsort(d["LWC"])
    Dm_arr    = np.array(d["Dm"])[sort_idx]
    logNw_arr = np.array(d["logNw"])[sort_idx]
    lwc_arr   = np.array(d["LWC"])[sort_idx]
    
    sc = ax.scatter(Dm_arr, logNw_arr, c=lwc_arr, cmap=cmap,
                    vmin=vmin, vmax=vmax, marker=marker, s=ms, 
                    alpha=0.85, linewidths=0.5, edgecolors='black',
                    label=f"{name} mm h$^{{-1}}$", zorder=3)

# Add Colorbar
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Liquid Water Content, LWC (g m$^{-3}$)", fontsize=LABEL_FS, fontweight="bold")
cbar.ax.tick_params(labelsize=TICK_FS)
cbar.outline.set_linewidth(1.2)

# (a) Maritime Convective Cluster (Dm: 1.5 to 1.75 mm, log10(Nw): 4.0 to 4.5)
ax.add_patch(mpatches.Rectangle(
    (1.5, 4.0), 0.25, 0.5,  
    linewidth=1.8, edgecolor='dimgray', facecolor='none', linestyle='-', zorder=4))
ax.text(1.625, 4.55, "Maritime\nConvection", fontsize=10, color='dimgray', 
        fontweight='bold', ha='center', va='bottom', zorder=5)

# (b) Continental Convective Cluster (Dm: 2.0 to 2.75 mm, log10(Nw): 3.0 to 3.5)
ax.add_patch(mpatches.Rectangle(
    (2.0, 3.0), 0.75, 0.5,  
    linewidth=1.8, edgecolor='dimgray', facecolor='none', linestyle='-', zorder=4))
ax.text(3.1, 3.25, "Continental\nConvection", fontsize=10, color='dimgray', 
        fontweight='bold', ha='center', va='top', zorder=5)

# (c) Bringi et al. (2009) Stratiform/Convective Separator: log10(Nw) = -1.6*Dm + 6.3
Dm_line = np.linspace(0.0, 5.0, 200)
ax.plot(Dm_line, -1.6 * Dm_line + 6.3, color='black', linewidth=2.0, linestyle='-', zorder=4)
ax.text(0.15, 4.8, "Bringi (2009)", fontsize=10, fontweight="bold", color='black', zorder=5)

# (d) Thompson et al. (2015) Maritime Stratiform/Convective Separator: log10(Nw) = 3.85
ax.axhline(3.85, color='black', linestyle='--', linewidth=1.5, zorder=4)
ax.text(4.8, 3.88, "Thompson (2015)", fontsize=10, fontweight="bold", color='black', ha='right', va='bottom', zorder=5)

# ── Axes Formatting ───────────────────────────────────────────────────────────
ax.set_xlabel("Mass-weighted Mean Diameter, $D_m$ (mm)", fontsize=LABEL_FS, fontweight="bold")
ax.set_ylabel("Normalized Intercept, $\log_{10}N_w$ (m$^{-3}$ mm$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")

ax.set_xlim(0, 5)
ax.set_ylim(2, 5)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))

ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=TICK_FS)
ax.tick_params(which='major', length=6)
ax.tick_params(which='minor', length=3)
ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)

#Legend
handles = []
for name, lo, hi, marker, ms in rr_cats:
    if results[name]["Dm"]:
        handles.append(plt.scatter([], [], marker=marker, s=ms*1.2,
                                   color='white', label=f"{name}",
                                   edgecolors='k', linewidths=1.2))

ax.legend(handles=handles, fontsize=10, loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray', title="Rain Rate (mm h$^{-1}$)", title_fontsize=10)

# ── Output ────────────────────────────────────────────────────────────────────
out = os.path.join(script_dir, "Fig6.png")  
plt.savefig(out)
print(f"✅ Success! Saved to: {out}")
plt.show()