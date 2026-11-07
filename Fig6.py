"""
Fig6
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

import dsd_common as C

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
LABEL_FS, TICK_FS = 14, 12

rr_cats = [("R < 1", 0.0, 1.0, "o", 30), ("1–5", 1.0, 5.0, "s", 30), ("5–10", 5.0, 10.0, "^", 40),
           ("10–20", 10.0, 20.0, "D", 40), ("20–50", 20.0, 50.0, "v", 50), ("R > 50", 50.0, 9999.0, "P", 60)]

data = C.get_qc_dataset()
d, Rc = data["dsd"], data["R_class"]
Dm, LWC = d["Dm"], d["LWC"]
ok = (d["Nw"] > 0) & np.isfinite(Dm)
logNw = np.full(len(Dm), np.nan); logNw[ok] = np.log10(d["Nw"][ok])
N = int(ok.sum())

# --- reference-line statistics (Dm and D0) ---
mu, lam, G = C.mu_lambda(d["M3"], d["M4"], d["M6"])
okmu = ok & (G < 0.95) & (mu >= -2) & (mu <= 40) & (lam > 0)
D0 = np.full(len(Dm), np.nan); D0[okmu] = (3.67 + mu[okmu]) / lam[okmu]
print(f"\n--- Fig. 6 statistics (N = {N} spectra) ---")
print("Bringi et al. (2009) line log10Nw = -1.6*D + 6.3 (above the line = convective side; check orientation vs paper)")
for name, X, sel in (("Dm (all spectra)", Dm, ok), ("Dm (spectra with valid mu-Lambda)", Dm, okmu), ("D0 (spectra with valid mu-Lambda)", D0, okmu)):
    above = logNw[sel] >= -1.6 * X[sel] + 6.3
    print(f"  {name:<36}: above line {100*above.mean():5.1f} % | below line {100*(~above).mean():5.1f} % (n = {sel.sum()})")
print(f"Thompson et al. (2015) line log10Nw = 3.85: above {100*(logNw[ok] >= 3.85).mean():.1f} % | below {100*(logNw[ok] < 3.85).mean():.1f} %")
cont = ok & (Dm >= 2.0) & (Dm <= 2.75) & (logNw >= 3.0) & (logNw <= 3.5)
mari = ok & (Dm >= 1.5) & (Dm <= 1.75) & (logNw >= 4.0) & (logNw <= 4.5)
print(f"Bringi (2003) clusters: continental box {cont.sum()} ({100*cont.sum()/N:.2f} %), maritime box {mari.sum()} ({100*mari.sum()/N:.2f} %)")

# --- figure ---
fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
vmax = np.percentile(LWC[ok], 99)
sc = None
for name, lo, hi, marker, ms in rr_cats:
    sel = ok & (Rc >= lo) & (Rc < hi)
    if not sel.any():
        continue
    o = np.argsort(LWC[sel])
    sc = ax.scatter(Dm[sel][o], logNw[sel][o], c=LWC[sel][o], cmap="viridis", vmin=0, vmax=vmax, marker=marker,
                    s=ms, alpha=0.85, linewidths=0.5, edgecolors="black", zorder=3)
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Liquid Water Content, LWC (g m$^{-3}$)", fontsize=LABEL_FS, fontweight="bold")
cbar.ax.tick_params(labelsize=TICK_FS); cbar.outline.set_linewidth(1.2)

ax.add_patch(mpatches.Rectangle((1.5, 4.0), 0.25, 0.5, lw=1.8, ec="dimgray", fc="none", zorder=4))
ax.text(1.625, 4.55, "Maritime\nConvection", fontsize=10, color="dimgray", fontweight="bold", ha="center", va="bottom", zorder=5)
ax.add_patch(mpatches.Rectangle((2.0, 3.0), 0.75, 0.5, lw=1.8, ec="dimgray", fc="none", zorder=4))
ax.text(3.1, 3.25, "Continental\nConvection", fontsize=10, color="dimgray", fontweight="bold", ha="center", va="top", zorder=5)
Dl = np.linspace(0.0, 5.0, 200)
ax.plot(Dl, -1.6 * Dl + 6.3, color="black", lw=2.0, zorder=4)
ax.text(0.15, 4.8, "Bringi (2009)", fontsize=10, fontweight="bold", zorder=5)
ax.axhline(3.85, color="black", ls="--", lw=1.5, zorder=4)
ax.text(4.8, 3.88, "Thompson (2015)", fontsize=10, fontweight="bold", ha="right", va="bottom", zorder=5)

ax.set_xlabel("Mass-weighted Mean Diameter, $D_m$ (mm)", fontsize=LABEL_FS, fontweight="bold")
ax.set_ylabel(r"Normalized Intercept, $\log_{10}N_w$ ($N_w$ in mm$^{-1}$ m$^{-3}$)", fontsize=LABEL_FS, fontweight="bold")
ax.set_xlim(0, 5); ax.set_ylim(2, 5)
ax.xaxis.set_major_locator(MultipleLocator(0.5)); ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=TICK_FS)
ax.tick_params(which="major", length=6); ax.tick_params(which="minor", length=3)
ax.grid(True, ls=":", lw=0.8, alpha=0.5)
handles = [plt.scatter([], [], marker=m, s=s_ * 1.2, color="white", edgecolors="k", linewidths=1.2, label=n)
           for n, lo, hi, m, s_ in rr_cats if (ok & (Rc >= lo) & (Rc < hi)).any()]
ax.legend(handles=handles, fontsize=10, loc="upper right", frameon=True, framealpha=0.9, edgecolor="gray",
          title="Rain Rate (mm h$^{-1}$)", title_fontsize=10)
ax.text(0.02, 0.03, f"N = {N:,} spectra", transform=ax.transAxes, fontsize=11, bbox=dict(fc="white", ec="gray", alpha=0.9))
plt.savefig(C.OUT_DIR / "Fig6.png")
plt.savefig(C.OUT_DIR / "Fig6.tif", dpi=300, bbox_inches="tight")
print(f"Saved -> {C.OUT_DIR / 'Fig6.png'} and {C.OUT_DIR / 'Fig6.tif'}")
plt.show()
