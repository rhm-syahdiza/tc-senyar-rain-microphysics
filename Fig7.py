"""
Fig7
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
# Validity limits of the M3-M4-M6 gamma retrieval (reported in the caption / Methods)
G_MAX, MU_MIN, MU_MAX, LAM_MIN, LAM_MAX = 0.95, -2, 40, 0, 40
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

# ── Data (shared QC from dsd_common: same 1,452 spectra as every other figure) ──
import dsd_common as C

script_dir = str(C.OUT_DIR)                                   # figure and csv outputs go to Fig_outputs
data = C.get_qc_dataset()
d = data["dsd"]
n_qc = len(data["idx"])

# moment-method validity filters, counted one by one (R3 minor 19)
mu_all, lam_all, G_all = C.mu_lambda(d["M3"], d["M4"], d["M6"])
nonpos = (d["M3"] <= 0) | (d["M4"] <= 0) | (d["M6"] <= 0)
badG = ~nonpos & (G_all >= G_MAX)
badmu = ~nonpos & ~badG & ~((mu_all >= MU_MIN) & (mu_all <= MU_MAX))
badlam = ~nonpos & ~badG & ~badmu & ~((lam_all >= LAM_MIN) & (lam_all <= LAM_MAX))
keep = ~(nonpos | badG | badmu | badlam)
cut = {"M3, M4 or M6 <= 0": int(nonpos.sum()), f"G >= {G_MAX}": int(badG.sum()),
       f"mu outside [{MU_MIN}, {MU_MAX}]": int(badmu.sum()), f"Lambda outside [{LAM_MIN}, {LAM_MAX}]": int(badlam.sum())}

Rc = data["R_class"]                                          # rain rate for the scatter panels (see dsd_common.CLASS_RATE)
rej = [("mu" if badmu[i] else "Lambda", data["t"][i].strftime("%Y-%m-%d %H:%M"), Rc[i], d["counts"][i].sum(),
        d["Dm"][i], mu_all[i], lam_all[i]) for i in np.where(badmu | badlam)[0]]
rr_list, mu_list, lam_list, dm_list = list(Rc[keep]), list(mu_all[keep]), list(lam_all[keep]), list(d["Dm"][keep])

rr, mu, lam = np.array(rr_list), np.array(mu_list), np.array(lam_list)

# Best-fit quadratic
def quad(x, a, b, c): return a*x**2 + b*x + c
popt, _ = curve_fit(quad, mu, lam, p0=[0.02, 0.7, 2.0])
a_fit, b_fit, c_fit = popt

# ── mu-Lambda sample cascade and coefficient uncertainty (R3 minor 19, 34; R3 major 12) ──
print(f"\nSpectra passing Parsivel QC            : {n_qc}")
for k, v in cut.items():
    print(f"  removed, {k:<28}: {v}")
print(f"Spectra used for mu-Lambda (final N)   : {len(mu)}  ({100*len(mu)/max(n_qc,1):.1f} % of QC spectra)")

# What do the rejected spectra look like? (Dm = M4/M3, mm)
for reason in ("mu", "Lambda"):
    sub = [r for r in rej if r[0] == reason]
    if sub:
        a = np.array([r[2:] for r in sub], dtype=float)      # rain rate, drops, Dm, mu, Lambda
        print(f"Rejected by {reason:<6}: n={len(sub):3d} | Dm median {np.median(a[:,2]):.2f} mm (range {a[:,2].min():.2f}-{a[:,2].max():.2f})"
              f" | drops median {np.median(a[:,1]):.0f} | rain rate median {np.median(a[:,0]):.2f} mm/h"
              f" | mu median {np.median(a[:,3]):.1f} | Lambda median {np.median(a[:,4]):.1f}")
with open(os.path.join(script_dir, "Fig7_rejected_spectra.csv"), "w", newline="") as _f:
    _w = csv.writer(_f); _w.writerow(["reason", "time_local", "rain_rate_mmh", "drops", "Dm_mm", "mu", "Lambda"]); _w.writerows(rej)
dm_med = np.median(dm_list)
print(f"Retained spectra: median Dm = {dm_med:.2f} mm, so 1/median(Dm) = {1/dm_med:.2f} mm-1 (the fitted slope b is expected to be of this size)")

mu_fit = np.linspace(-2, 20, 300)

def block_boot_quad(x, y, grid, B=1000, L=15, seed=1):
    """95 % moving-block-bootstrap intervals of (a, b, c) and of the fitted curve on `grid`."""
    rng = np.random.default_rng(seed); n = len(x); L = max(1, min(L, n)); nb = int(np.ceil(n / L))
    coefs = []
    for _ in range(B):
        starts = rng.integers(0, n - L + 1, size=nb)
        idx = (starts[:, None] + np.arange(L)).ravel()[:n]
        coefs.append(np.polyfit(x[idx], y[idx], 2))
    coefs = np.array(coefs)
    ci = np.percentile(coefs, [2.5, 97.5], axis=0)
    band = np.percentile([np.polyval(c, grid) for c in coefs], [2.5, 97.5], axis=0)
    return ci, band

(ci_lo, ci_hi), (band_lo, band_hi) = block_boot_quad(mu, lam, mu_fit)
for nm, v, lo, hi in zip(("a", "b", "c"), popt, ci_lo, ci_hi):
    print(f"  Lambda = a*mu^2 + b*mu + c : {nm} = {v:.4f}  (95 % CI {lo:.4f} to {hi:.4f})")
pd_out = {"qc_spectra": n_qc, "final_N": len(mu), **{f"removed_{k}": v for k, v in cut.items()},
          **{f"{nm}": v for nm, v in zip(("a", "b", "c"), popt)},
          **{f"{nm}_lo": v for nm, v in zip(("a", "b", "c"), ci_lo)},
          **{f"{nm}_hi": v for nm, v in zip(("a", "b", "c"), ci_hi)}}
with open(os.path.join(script_dir, "Fig7_sample_and_fit.csv"), "w") as _f:
    _f.write("item,value\n" + "\n".join(f"{k},{v}" for k, v in pd_out.items()) + "\n")

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

ax_a.fill_betweenx(mu_fit, band_lo, band_hi, color="0.5", alpha=0.35, lw=0, zorder=3,
                   label="95 % Confidence Interval of the fit")
ax_a.plot(quad(mu_fit, *popt), mu_fit, color="black", lw=2.5, zorder=4,
          label=f"TC Senyar (KT-OR): $\\Lambda = {a_fit:.4f}\\mu^2 + {b_fit:.4f}\\mu + {c_fit:.4f}$")
ax_a.text(0.98, 0.12, "95 % CI for coefficients:\n"
          f"$a$ = {ci_lo[0]:.4f} to {ci_hi[0]:.4f}\n$b$ = {ci_lo[1]:.3f} to {ci_hi[1]:.3f}\n$c$ = {ci_lo[2]:.3f} to {ci_hi[2]:.3f}",
          transform=ax_a.transAxes, ha="right", va="bottom", fontsize=11, zorder=6,
          bbox=dict(fc="white", ec="gray", alpha=0.9))

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
ax_a.set_title(r"(a) $\mu$-$\Lambda$ Empirical Relationship", loc="left", fontsize=TITLE_FS, fontweight="bold")
ax_a.text(0.98, 0.04, f"N = {len(mu):,} of {n_qc:,} QC spectra", transform=ax_a.transAxes, ha="right",
          fontsize=12, bbox=dict(fc="white", ec="gray", alpha=0.9))

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
plt.savefig(os.path.join(script_dir, "Fig7.tif"), dpi=300, bbox_inches="tight")
plt.show()
