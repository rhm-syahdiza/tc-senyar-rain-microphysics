"""
Fig8 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dsd_common as C

BOOT_B, BLOCK_LEN, K_FOLD = 2000, 15, 5

REFS = [  # label, A, b, colour, linestyle
    ("Marshall-Palmer (1948)", 200.0, 1.60, "#7F7F7F", "--"),
    ("Rosenfeld (1993)", 250.0, 1.20, "#FF7F0E", "--"),
    ("Marzuki (2018)", 279.0, 1.43, "#9467BD", "-"),
    ("Chang (2009)", 206.83, 1.45, "#D62728", "-"),
    ("Deo & Walsh (2016)", 250.0, 1.30, "#1F77B4", "-"),
    ("Wen (2018)", 147.28, 1.38, "#2CA02C", "-"),
    ("Janapati (2020)", 148.44, 1.30, "#17BECF", "-"),   
    ("Bao (2020)", 237.58, 1.43, "#8C564B", "-"),
]


def hist2d(X, Y, scaling=2):
    Xr = 100 * (X - X.min()) / (X.max() - X.min()); Yr = 100 * (Y - Y.min()) / (Y.max() - Y.min())
    return np.array([np.sum(np.hypot(Xr[i] - Xr, Yr[i] - Yr) < scaling) for i in range(len(Xr))])


def fit_dB(R, Z):
    """Regression of 10log10 Z on 10log10 R. Returns A, b, r2 (of the dB regression)."""
    x, y = 10 * np.log10(R), 10 * np.log10(Z)
    b, c = np.polyfit(x, y, 1)
    return 10 ** (c / 10), b, np.corrcoef(x, y)[0, 1] ** 2


def r_from_z(Z, A, b):
    return (Z / A) ** (1.0 / b)


def qpe_metrics(R_obs, R_est):
    e = 10 * np.log10(R_est) - 10 * np.log10(R_obs)
    out = dict(RMSE_dBR=np.sqrt(np.mean(e ** 2)), bias_dBR=e.mean(),
               NB_pct=100 * (R_est - R_obs).sum() / R_obs.sum(),
               NRMSE_pct=100 * np.sqrt(np.mean((R_est - R_obs) ** 2)) / R_obs.mean(),
               median_rel_err_pct=np.median(100 * (R_est - R_obs) / R_obs))
    for thr in (20, 50):
        m = R_obs >= thr
        out[f"NB_pct_R>={thr}"] = 100 * (R_est[m] - R_obs[m]).sum() / R_obs[m].sum() if m.any() else np.nan
        out[f"n_R>={thr}"] = int(m.sum())
    return out


# ── Data ───────────────────────────────────────────────────────────────────────
data = C.get_qc_dataset()
d = data["dsd"]
ok = (d["R"] >= C.RR_MIN) & (d["Z"] > 0)
R_obs, Z_obs = d["R"][ok], d["Z"][ok]
n = len(R_obs)
dBR, dBZ = 10 * np.log10(R_obs), 10 * np.log10(Z_obs)
print(f"\nZ-R sample: {n} of {len(d['R'])} QC spectra (derived R >= {C.RR_MIN} mm/h required)")

# ── Fit, bootstrap, cross-validation ───────────────────────────────────────────
A_fit, b_fit, r2 = fit_dB(R_obs, Z_obs)
rng = np.random.default_rng(1)
bs = np.array([fit_dB(R_obs[i], Z_obs[i])[:2] for i in (C.block_idx(n, BLOCK_LEN, rng) for _ in range(BOOT_B))])
A_lo, A_hi = np.percentile(bs[:, 0], [2.5, 97.5]); b_lo, b_hi = np.percentile(bs[:, 1], [2.5, 97.5])

R_cv = np.empty(n)
for te in np.array_split(np.arange(n), K_FOLD):
    tr = np.setdiff1d(np.arange(n), te)
    A_k, b_k, _ = fit_dB(R_obs[tr], Z_obs[tr])
    R_cv[te] = r_from_z(Z_obs[te], A_k, b_k)

s_inv, t_inv = np.polyfit(dBZ, dBR, 1)                       # QPE direction: dBR = s*dBZ + t
A_inv, b_inv = 10 ** (-t_inv / s_inv / 10), 1.0 / s_inv

rows = [dict(relation="TC Senyar fit (in-sample)", A=A_fit, b=b_fit, **qpe_metrics(R_obs, r_from_z(Z_obs, A_fit, b_fit))),
        dict(relation=f"TC Senyar fit ({K_FOLD}-fold blocked CV)", A=A_fit, b=b_fit, **qpe_metrics(R_obs, R_cv))]
for lab, A, b, *_ in REFS:
    rows.append(dict(relation=lab, A=A, b=b, **qpe_metrics(R_obs, r_from_z(Z_obs, A, b))))
metrics = pd.DataFrame(rows)
metrics.to_csv(C.OUT_DIR / "Fig8_QPE_metrics.csv", index=False)
pd.set_option("display.width", 220, "display.float_format", "{:,.2f}".format)
print(f"\nZ = A R^b (dB regression): A = {A_fit:.1f} (95% CI {A_lo:.1f}-{A_hi:.1f}), b = {b_fit:.3f} ({b_lo:.3f}-{b_hi:.3f}), r2 = {r2:.3f}")
print(f"R-on-Z (QPE direction) fit: A = {A_inv:.1f}, b = {b_inv:.3f}")
print("\n=== QPE metrics (R estimated from Z; every row scored on the same spectra) ===")
print(metrics.to_string(index=False))
print("NB = accumulated-rainfall error (each sample is 1 min). RMSE in dBR = sqrt(mean[(10log10 R_est - 10log10 R)^2]).")

# ── Error range implied by the RMSE (dBR) ─────────────────────────────────────
# An error of e dBR means R_est/R = 10^(e/10), so +/- RMSE gives:  lower = 10^(-RMSE/10) - 1,  upper = 10^(+RMSE/10) - 1
err = metrics[["relation", "A", "b", "RMSE_dBR"]].copy()
err["lower_error_pct"] = 100 * (10 ** (-err["RMSE_dBR"] / 10) - 1)
err["upper_error_pct"] = 100 * (10 ** (err["RMSE_dBR"] / 10) - 1)
err = err.sort_values("RMSE_dBR", ascending=False).reset_index(drop=True)
err.round(2).to_csv(C.OUT_DIR / "Fig8_RMSE_error_range.csv", index=False)
print("\n=== QPE error range implied by RMSE (typical +/- 1 RMSE range, not a bound) ===")
print(err.to_string(index=False))
worst = err[~err["relation"].str.startswith("TC Senyar")].iloc[0]
print(f"\nSentence: the {worst['relation']} relation produced the highest error with RMSE = {worst['RMSE_dBR']:.1f} dBR, "
      f"corresponding to QPE uncertainties from {worst['lower_error_pct']:.1f}% to +{worst['upper_error_pct']:.1f}%.")

# ── Measured vs Atlas fall speed (R3 major 7) ──────────────────────────────────
variants = [("1 manuscript: N(D) measured v, R with Atlas v", C.dsd(data["spec"], "measured", "atlas")),
            ("2 measured v for N(D) and R", C.dsd(data["spec"], "measured", "consistent")),
            ("3 Atlas v for N(D) and R", C.dsd(data["spec"], "atlas", "consistent"))]
d1 = variants[0][1]
vrows = []
for name, dv in variants:
    okv = (dv["R"] >= C.RR_MIN) & (dv["Z"] > 0)
    A_v, b_v, r2_v = fit_dB(dv["R"][okv], dv["Z"][okv])
    vrows.append(dict(variant=name, N_ZR=int(okv.sum()), A=A_v, b=b_v, r2=r2_v, median_R=np.median(dv["R"]),
                      R_ratio=np.median(dv["R"] / d1["R"]), Dm_ratio=np.median(dv["Dm"] / d1["Dm"]),
                      Nw_ratio=np.median(dv["Nw"] / d1["Nw"]), Z_ratio=np.median(dv["Z"] / d1["Z"])))
vtab = pd.DataFrame(vrows); vtab.to_csv(C.OUT_DIR / "Fig8_velocity_sensitivity.csv", index=False)
print("\n=== Sensitivity to the fall speed used (ratios are medians relative to variant 1) ===")
print(vtab.to_string(index=False))

# ── Figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
sc = ax.scatter(dBR, dBZ, s=10, c=hist2d(dBR, dBZ), cmap="viridis", zorder=2)
cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.ax.set_ylabel("Point Density", size=11, fontweight="bold"); cb.outline.set_linewidth(1.2)

line = np.linspace(-10, 20, 500)
rm = metrics.set_index("relation")["RMSE_dBR"]
ax.plot(line, b_fit * line + 10 * np.log10(A_fit), color="black", lw=2.5, zorder=5,
        label=f"TC Senyar (KT-OR): $Z={A_fit:.0f}R^{{{b_fit:.2f}}}$; $r^2$={r2:.2f}; RMSE={rm['TC Senyar fit (in-sample)']:.2f} dBR")
for lab, A, b, col, ls in REFS:
    ax.plot(line, b * line + 10 * np.log10(A), color=col, ls=ls, lw=1.8 if ls == "--" else 1.5, zorder=3,
            label=f"{lab}: $Z={A:g}R^{{{b:g}}}$; RMSE={rm[lab]:.2f} dBR")
ax.set_xticks([-10, 0, 10, 20])
ax.set_xticklabels([f"{v}\n({r} mm h$^{{-1}}$)" for v, r in zip([-10, 0, 10, 20], ["0.1", "1", "10", "100"])], fontsize=9.5)
ax.set_xlim(-10, 20); ax.set_ylim(10, 80)
ax.set_xlabel(r"Rain Rate, $10\log_{10}R$ (dBR)", fontsize=12, fontweight="bold")
ax.set_ylabel(r"Radar Reflectivity, $10\log_{10}Z$ (dBZ)", fontsize=12, fontweight="bold")
ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
ax.grid(True, ls=":", lw=0.8, alpha=0.5)
ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=1,
          title=f"N = {n:,} 1-min spectra; RMSE in dBR (same data as the fit)", title_fontsize=10)
'''
ax.text(0.98, 0.04, f"95% CI: A = {A_lo:.0f}–{A_hi:.0f}, b = {b_lo:.2f}–{b_hi:.2f}\n"
        f"{K_FOLD}-fold blocked CV RMSE = {rm[f'TC Senyar fit ({K_FOLD}-fold blocked CV)']:.2f} dBR",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, bbox=dict(fc="white", ec="gray", alpha=0.9))
'''
ax.text(0.98, 0.04, f"95% CI: A = {A_lo:.0f}–{A_hi:.0f}, b = {b_lo:.2f}–{b_hi:.2f}\n"
        f"Blocked Cross Validation RMSE = {rm[f'TC Senyar fit ({K_FOLD}-fold blocked CV)']:.2f} dBR",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, bbox=dict(fc="white", ec="gray", alpha=0.9))
plt.tight_layout()
plt.savefig(C.OUT_DIR / "Fig8.png", bbox_inches="tight")
plt.savefig(C.OUT_DIR / "Fig8.tif", dpi=300, bbox_inches="tight")
print(f"Saved -> {C.OUT_DIR / 'Fig8.png'} and {C.OUT_DIR / 'Fig8.tif'}")

#print(f"\nSaved -> {C.OUT_DIR / 'Fig8.png'}")
plt.show()
