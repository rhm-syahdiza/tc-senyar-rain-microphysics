"""
Parsivel DSD Analysis:
(a) Mean N(D) vs D for different rain rate categories
(b) Rain rate occurrence percentage (Using CSV raw RR)
(c) Contribution to total drop count (Nt) and Rainfall (R) by Diameter Class
"""

import os, csv, datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

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

LABEL_FS, TICK_FS, TITLE_FS = 14, 14, 16

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

# ── QC ───────────────────
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

# Rainrate Categories 
rr_cats = [
    ("$R$ < 1",  0.1,   1.0, "tab:blue"),
    ("1–5",    1,   5.0, "tab:green"),
    ("5–10",     5.0,  10.0, "tab:orange"),
    ("10–20",   10.0,  20.0, "tab:red"),
    ("20–50",   20.0,  50.0, "tab:pink"),
    ("$R$ > 50",  50.0, 9999., "tab:purple"),
]
D_classes = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,np.inf)]
labels_c  = ['<1', '1–2', '2–3', '3–4', '4–5', '>5']

# TC Senyar period 
T_START = dt.datetime(2025, 11, 25, 12, 0, 0) + dt.timedelta(hours=7)
T_END   = dt.datetime(2025, 11, 28, 12, 0, 0) + dt.timedelta(hours=7)

# ── Data Processing ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root  = os.path.join(script_dir, "Kototabang")
data_records = [] 

for month in sorted(os.listdir(data_root)):
    mpath = os.path.join(data_root, month)
    if not os.path.isdir(mpath): continue
    for day in sorted(os.listdir(mpath)):
        dpath = os.path.join(mpath, day)
        if not os.path.isdir(dpath): continue
        for fname in sorted(os.listdir(dpath)):
            if not fname.endswith(".csv"): continue
            with open(os.path.join(dpath, fname), newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or not row[0].strip(): continue
                    try:
                        t_lt = dt.datetime.strptime(f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S")
                        if not (T_START <= t_lt <= T_END): continue

                        rr_csv = float(row[2]) 
                        if rr_csv < R_MIN: continue

                        spec_raw = row[15:1038]
                        vals = [float(x) if x.strip() else 0.0 for x in spec_raw]
                        vals += [0.0] * (1024 - len(vals))
                        spec = np.array(vals).reshape(32, 32)

                        # QC: Velocity & Drop Count (Using Corrected V_term)
                        for d_i in range(32):
                            v_lo, v_hi = V_term[d_i]*(1-V_FRAC), V_term[d_i]*(1+V_FRAC)
                            for v_i in range(32):
                                if not (v_lo <= V_bins[v_i] <= v_hi): spec[v_i, d_i] = 0.0
                        
                        spec[:, ~D_valid_mask] = 0.0
                        if spec.sum() < MIN_DROPS: continue

                        # Retrieve N(D)
                        n_d = np.zeros(32)
                        for d_i in range(32):
                            if not D_valid_mask[d_i]: continue
                            with np.errstate(divide='ignore', invalid='ignore'):
                                n_d[d_i] = np.sum(np.where(V_bins > 0, spec[:, d_i] / (V_bins * A_eff[d_i] * t_int * dD[d_i]), 0.0))

                        cat_idx = next((i for i, c in enumerate(rr_cats) if c[1] <= rr_csv < c[2]), None)
                        if cat_idx is not None:
                            data_records.append({'R': rr_csv, 'ND': n_d, 'cat': cat_idx})
                    except: continue

rr_arrays = {i: [] for i in range(len(rr_cats))}
total_ND_all = []
Nt_contrib, R_contrib = np.zeros(len(D_classes)), np.zeros(len(D_classes))

for r in data_records:
    rr_arrays[r['cat']].append(r)
    total_ND_all.append(r['ND'])
    
    for i, (dmin, dmax) in enumerate(D_classes):
        m = (D_bins >= dmin) & (D_bins < dmax) & D_valid_mask
        Nt_contrib[i] += np.sum(r['ND'][m] * dD[m])
        R_contrib[i]  += np.sum(r['ND'][m] * V_term[m] * (D_bins[m]**3) * dD[m])

mean_nd_all = np.array(total_ND_all).mean(axis=0) if total_ND_all else np.zeros(32)

Nt_pct = Nt_contrib / Nt_contrib.sum() * 100 if Nt_contrib.sum() > 0 else Nt_contrib
R_pct  = R_contrib / R_contrib.sum() * 100 if R_contrib.sum() > 0 else R_contrib

mask_all = (mean_nd_all > 0) & D_valid_mask
N_total_all = np.sum(mean_nd_all[mask_all] * dD[mask_all])

# ── Figure Layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax_dsd, ax_hist, ax_c = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])

# (a) Mean DSD plotting
for i, (name, _, _, color) in enumerate(rr_cats):
    cat_recs = rr_arrays[i]
    if not cat_recs: continue
    nd_matrix = np.array([r['ND'] for r in cat_recs])
    mean_nd = nd_matrix.mean(axis=0)
    
    mask = (mean_nd > 0) & D_valid_mask
    ax_dsd.semilogy(D_bins[mask], mean_nd[mask], color=color, lw=1.8, label=name)

if np.any(mask_all):
    ax_dsd.semilogy(D_bins[mask_all], mean_nd_all[mask_all], color='black', lw=3.0, 
                    linestyle='-', marker='o', mfc='black', markersize=4, label='All', zorder=10)

ax_dsd.set_xlabel("Diameter, $D$ (mm)", fontsize=LABEL_FS, fontweight="bold")
ax_dsd.set_ylabel("Number Concentration, $N(D)$ (m$^{-3}$ mm$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
ax_dsd.set_xlim(0, 8)
ax_dsd.set_ylim(1e-4, 1e4) 
ax_dsd.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax_dsd.tick_params(axis='both', which='major', labelsize=TICK_FS)
ax_dsd.grid(True, which='major', linestyle=':', alpha=0.6)
ax_dsd.legend(fontsize=LABEL_FS, loc='upper right', framealpha=0.9, edgecolor='gray')
ax_dsd.set_title("(a) Mean Drop Size Distribution", loc="left", fontsize=TITLE_FS, fontweight="bold")

# (b) RR Histogram
all_rs = np.array([r['R'] for r in data_records])
counts, _ = np.histogram(all_rs, bins=[0.1, 1, 5, 10, 20, 50, np.inf])
pcts = (counts / counts.sum()) * 100 if counts.sum() > 0 else counts
bars = ax_hist.bar([c[0] for c in rr_cats], pcts, color=[c[3] for c in rr_cats], edgecolor='black')
ax_hist.set_title("(b) Rain Rate Distribution", loc="left", fontsize=TITLE_FS, fontweight="bold")
ax_hist.set_ylabel("Frequency (%)", fontsize=LABEL_FS, fontweight="bold")
ax_hist.set_ylim(0, 45) 
ax_hist.set_xlabel("Rainrate Class, $R$ (mm h$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
ax_hist.tick_params(axis='both', which='major', labelsize=TICK_FS)
for b in bars: ax_hist.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}%', ha='center', fontsize=12)

# (c) Contribution
x_c = np.arange(len(labels_c)); w = 0.35

bars_Nt = ax_c.bar(x_c - w/2, Nt_pct, w, color='tab:red', edgecolor='black', label='Contribution to $N_t$')
bars_R  = ax_c.bar(x_c + w/2, R_pct,  w, color='tab:blue', edgecolor='black', label='Contribution to $R$')

ax_c.set_title("(c) Contribution to Total Drop Count and Rainfall", loc="left", fontsize=TITLE_FS, fontweight="bold")
ax_c.set_xticks(x_c); ax_c.set_xticklabels(labels_c)
ax_c.tick_params(axis='both', which='major', labelsize=TICK_FS)
ax_c.set_ylabel("Percentage (%)", fontsize=LABEL_FS, fontweight="bold") 
ax_c.set_xlabel("Diameter Class, $D$ (mm)", fontsize=LABEL_FS, fontweight="bold")

#Plot properties
for b in bars_Nt:
    height = b.get_height()
    if height > 0.00001:
        if height < 0.01:
            label_text = f'{height:.4f}%'
        elif height < 0.1:
            label_text = f'{height:.3f}%'
        else:
            label_text = f'{height:.1f}%'
            
        ax_c.text(b.get_x() + b.get_width()/2.0, height + 1.0, label_text, 
                  ha='center', va='bottom', fontsize=LABEL_FS, color='black')

for b in bars_R:
    height = b.get_height()
    if height > 0.00001: 
        if height < 0.01:
            label_text = f'{height:.4f}%'
        elif height < 0.1:
            label_text = f'{height:.3f}%'
        else:
            label_text = f'{height:.1f}%'
            
        ax_c.text(b.get_x() + b.get_width()/2.0, height + 1.0, label_text, 
                  ha='center', va='bottom', fontsize=LABEL_FS, color='black')

ax_c.set_ylim(0, max(max(Nt_pct), max(R_pct)) * 1.15)

ax_c.legend(fontsize=11, loc='upper right')
ax_c.set_facecolor("#f9f9f9")

for ax in [ax_dsd, ax_hist, ax_c]: ax.tick_params(direction='in', top=True, right=True)
plt.savefig(os.path.join(script_dir, "Fig5.png"))
plt.show()