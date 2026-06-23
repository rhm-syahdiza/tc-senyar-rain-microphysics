"""
(a) mu-Lambda scatter and empirical fit
(b) mu vs Rain Rate (Scatter + PDF)
(c) Lambda vs Rain Rate (Scatter + PDF)
"""

import os, csv, datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

# ── rcParams────────────────────────────────────────────────────
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

LABEL_FS, TICK_FS, TITLE_FS = 14, 12, 16

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

# ── QC───────────────────
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

# ── Reference mu-Lambda─────────────────────────────────────────────
lam_chang = lambda mu: 0.0136*mu**2 + 0.6984*mu + 1.5131   
lam_bao   = lambda mu: 0.0222*mu**2 + 0.6673*mu + 2.3453   
lam_bao2  = lambda mu: 0.0264*mu**2 + 0.8169*mu + 2.0048   
lam_jan   = lambda mu: 0.0129*mu**2 + 0.8360*mu + 2.2260   
mu_wen    = lambda lam: -0.0227*lam**2 + 1.317*lam - 2.232 

mu_ref  = np.linspace(-2, 20, 300)
lam_ref = np.linspace(0, 25, 300)

# TC Senyar time period 
T_START = dt.datetime(2025, 11, 25, 12, 0, 0) + dt.timedelta(hours=7)
T_END   = dt.datetime(2025, 11, 28, 12, 0, 0) + dt.timedelta(hours=7)

# ── Data Processing ───────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root  = os.path.join(script_dir, "Kototabang")

rr_list, mu_list, lam_list = [], [], []

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

                        # QC Filters
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

                        # M3-M4-M6 Moment Method
                        nd_v, D_v, dw_v = n_d[D_valid_mask], D_bins[D_valid_mask], dD[D_valid_mask]
                        M3 = np.sum(nd_v * (D_v**3) * dw_v)
                        M4 = np.sum(nd_v * (D_v**4) * dw_v)
                        M6 = np.sum(nd_v * (D_v**6) * dw_v)
                        
                        if M3 <= 0 or M4 <= 0 or M6 <= 0: continue

                        G  = (M4**3) / ((M3**2) * M6)
                        if G >= 0.95: continue
                        
                        mu  = ((11*G - 8) + np.sqrt(G * (G + 8))) / (2*(1 - G))
                        lam = (mu + 4) * M3 / M4 

                        if not (-2 <= mu <= 40) or not (0 <= lam <= 40): continue

                        rr_list.append(rr)
                        mu_list.append(mu)
                        lam_list.append(lam)

                    except Exception:
                        continue

rr, mu, lam = np.array(rr_list), np.array(mu_list), np.array(lam_list)

# Best-fit quadratic
def quad(x, a, b, c): return a*x**2 + b*x + c
popt, _ = curve_fit(quad, mu, lam, p0=[0.02, 0.7, 2.0])
a_fit, b_fit, c_fit = popt

fig = plt.figure(figsize=(14, 10))
gs_outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.25, wspace=0.15)

params = [
    (mu,    r'Shape Parameter, $\mu$',           -5, 40, '(b)', gs_outer[1, 0]),
    (lam,   r'Slope Parameter, $\Lambda$ (mm$^{-1}$)', 0, 40, '(c)', gs_outer[1, 1]),
]

RR_MAX, RR_TICK = 125, 25

# === Panel (a): mu-Lambda===
ax_a = fig.add_subplot(gs_outer[0, :])
ax_a.scatter(lam, mu, c="tab:blue", s=10, alpha=0.3, linewidths=0, zorder=2)

mu_fit = np.linspace(-2, 20, 300)
ax_a.plot(quad(mu_fit, *popt), mu_fit, color="black", lw=2.5, zorder=4,
          label=f"TC Senyar (KT-OR): $\\Lambda = {a_fit:.4f}\\mu^2 + {b_fit:.4f}\\mu + {c_fit:.4f}$")

ax_a.plot(lam_chang(mu_ref), mu_ref, color="tab:cyan", ls="--", lw=1.5, 
          label="Chang et al. (2009): $\\Lambda = 0.0136\\mu^2 + 0.6984\\mu + 1.5131$")

ax_a.plot(lam_bao(mu_ref), mu_ref, color="dimgray", ls="--", lw=1.5, 
          label="Bao et al. (2019): $\\Lambda = 0.0222\\mu^2 + 0.6673\\mu + 2.3453$")

ax_a.plot(lam_bao2(mu_ref), mu_ref, color="tab:red", ls="--", lw=1.5, 
          label="Bao et al. (2020): $\\Lambda = 0.0264\\mu^2 + 0.8169\\mu + 2.0048$")

ax_a.plot(lam_jan(mu_ref), mu_ref, color="tab:green", ls="--", lw=1.5, 
          label="Janapati et al. (2020): $\\Lambda = 0.0129\\mu^2 + 0.8360\\mu + 2.2260$")

ax_a.plot(lam_ref, mu_wen(lam_ref), color="tab:purple", ls="--", lw=1.5, 
          label="Wen et al. (2018): $\\mu = -0.0227\\Lambda^2 + 1.317\\Lambda - 2.232$")


# Formatting
ax_a.set_xlabel(r"Slope Parameter, $\Lambda$ (mm$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
ax_a.set_ylabel(r"Shape Parameter, $\mu$", fontsize=LABEL_FS, fontweight="bold")
ax_a.set_xlim(0, 20)
ax_a.set_ylim(-5, 20)
ax_a.xaxis.set_major_locator(MultipleLocator(2))
ax_a.yaxis.set_major_locator(MultipleLocator(5))
ax_a.grid(True, ls=':', alpha=0.6)

# Legend
ax_a.legend(fontsize=12, loc="upper left", framealpha=0.9, edgecolor='gray', ncol=1)
ax_a.set_title("(a) $\mu$-$\Lambda$ Empirical Relationship", loc="left", fontsize=TITLE_FS, fontweight="bold")

# === Panels (b)-(c): Scatter + PDF ===
for data, ylabel, ymin, ymax, label, gs_pos in params:
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_pos, width_ratios=[5, 1], wspace=0.07)
    ax_s, ax_p = fig.add_subplot(gs_inner[0]), fig.add_subplot(gs_inner[1])

    ax_s.scatter(rr, data, c='tab:blue', s=8, alpha=0.4, linewidths=0, zorder=2)
    ax_s.set_xlim(0, RR_MAX); ax_s.set_ylim(ymin, ymax)
    
    ax_s.xaxis.set_major_locator(MultipleLocator(RR_TICK))
    ax_s.xaxis.set_minor_locator(MultipleLocator(5.0))  
    
    ax_s.yaxis.set_major_locator(MultipleLocator((ymax - ymin) / 5))
    ax_s.grid(True, ls=':', color='gray', alpha=0.5)
    ax_s.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    ax_s.set_xlabel("Rain Rate, $R$ (mm h$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
    ax_s.set_title(f"{label}", loc="left", fontsize=TITLE_FS, fontweight="bold")

    # PDF Panel
    y_range = np.linspace(ymin, ymax, 300)
    kde = gaussian_kde(data, bw_method=0.2)
    pdf_norm = (kde(y_range) / kde(y_range).max()) * 0.8
    
    ax_p.plot(pdf_norm, y_range, color='black', lw=1.5)
    ax_p.fill_betweenx(y_range, 0, pdf_norm, color='tab:blue', alpha=0.4)
    ax_p.set_xlim(0, 1); ax_p.set_ylim(ymin, ymax)
    
    ax_p.set_xticks([0, 0.5, 1.0])
    ax_p.set_xticklabels(['', '0.5', '1.0'], fontsize=11)
    ax_p.set_xlabel("PDF", fontsize=LABEL_FS, fontweight="bold")
    ax_p.yaxis.set_visible(False); ax_p.axvline(0, color='black', lw=1.5)

for ax in fig.get_axes():
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=TICK_FS)
    ax.tick_params(which='minor', length=3)

plt.savefig(os.path.join(script_dir, "Fig7.png"))
plt.show()