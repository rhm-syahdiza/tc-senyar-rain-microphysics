"""
Fig5 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import dsd_common as C

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
LABEL_FS, TICK_FS, TITLE_FS = 14, 14, 16

rr_cats = [("$R$ < 1", "tab:blue"), ("1–5", "tab:green"), ("5–10", "tab:orange"),
           ("10–20", "tab:red"), ("20–50", "tab:pink"), ("$R$ > 50", "tab:purple")]
cls_edges = [1.0, 5.0, 10.0, 20.0, 50.0]
D_classes = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, np.inf)]
labels_c = ["<1", "1–2", "2–3", "3–4", "4–5", ">5"]

data = C.get_qc_dataset()
d, nd, Rc = data["dsd"], data["dsd"]["nd"], data["R_class"]
N = len(Rc)
cls = np.digitize(Rc, cls_edges)                       # 0 ... 5
counts = np.array([(cls == i).sum() for i in range(6)])
pcts = 100.0 * counts / N

mean_nd_all = nd.mean(axis=0)
mask_all = (mean_nd_all > 0) & C.D_MASK

Nt_contrib, R_contrib = np.zeros(6), np.zeros(6)
for i, (lo, hi) in enumerate(D_classes):
    m = (C.D_bins >= lo) & (C.D_bins < hi) & C.D_MASK
    Nt_contrib[i] = (nd[:, m] * C.dD[m]).sum()
    R_contrib[i] = d["R_bin"][:, m].sum()
Nt_pct, R_pct = 100 * Nt_contrib / Nt_contrib.sum(), 100 * R_contrib / R_contrib.sum()

fig = plt.figure(figsize=(14, 10), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax_dsd, ax_hist, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# (a)
for i, (name, color) in enumerate(rr_cats):
    if counts[i] == 0:
        continue
    mean_nd = nd[cls == i].mean(axis=0)
    m = (mean_nd > 0) & C.D_MASK
    ax_dsd.semilogy(C.D_bins[m], mean_nd[m], color=color, lw=1.8, label=f"{name} (n={counts[i]})")
ax_dsd.semilogy(C.D_bins[mask_all], mean_nd_all[mask_all], color="black", lw=3.0, marker="o", mfc="black",
                markersize=4, label=f"All (n={N})", zorder=10)
ax_dsd.set_xlabel("Diameter, $D$ (mm)", fontsize=LABEL_FS, fontweight="bold")
ax_dsd.set_ylabel("Number Concentration, $N(D)$ (m$^{-3}$ mm$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
ax_dsd.set_xlim(0, 8); ax_dsd.set_ylim(1e-4, 1e4)
ax_dsd.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax_dsd.tick_params(axis="both", which="major", labelsize=TICK_FS)
ax_dsd.grid(True, which="major", ls=":", alpha=0.6)
ax_dsd.legend(fontsize=11, loc="upper right", framealpha=0.9, edgecolor="gray")
ax_dsd.set_title("(a) Mean Drop Size Distribution", loc="left", fontsize=TITLE_FS, fontweight="bold")

# (b)
bars = ax_hist.bar([c[0] for c in rr_cats], pcts, color=[c[1] for c in rr_cats], edgecolor="black")
ax_hist.set_title("(b) Rain Rate Distribution", loc="left", fontsize=TITLE_FS, fontweight="bold")
ax_hist.set_ylabel("Frequency (% of 1-min spectra)", fontsize=LABEL_FS, fontweight="bold")
ax_hist.set_ylim(0, max(pcts) * 1.3)
ax_hist.set_xlabel("Rainrate Class, $R$ (mm h$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
ax_hist.tick_params(axis="both", which="major", labelsize=TICK_FS)
for b, p, n in zip(bars, pcts, counts):
    ax_hist.text(b.get_x() + b.get_width() / 2, b.get_height() + max(pcts) * 0.02, f"{p:.1f}%\n(n={n})",
                 ha="center", fontsize=11)

# (c)
x_c, w = np.arange(6), 0.35
bars_Nt = ax_c.bar(x_c - w / 2, Nt_pct, w, color="tab:red", edgecolor="black", label="Contribution to $N_t$")
bars_R = ax_c.bar(x_c + w / 2, R_pct, w, color="tab:blue", edgecolor="black", label="Contribution to $R$")
ax_c.set_title("(c) Contribution to Total Drop Count and Rainfall", loc="left", fontsize=TITLE_FS, fontweight="bold")
ax_c.set_xticks(x_c); ax_c.set_xticklabels(labels_c)
ax_c.tick_params(axis="both", which="major", labelsize=TICK_FS)
ax_c.set_ylabel("Percentage (%)", fontsize=LABEL_FS, fontweight="bold")
ax_c.set_xlabel("Diameter Class, $D$ (mm)", fontsize=LABEL_FS, fontweight="bold")
for bars_ in (bars_Nt, bars_R):
    for b in bars_:
        h = b.get_height()
        if h > 1e-5:
            txt = f"{h:.4f}%" if h < 0.01 else f"{h:.3f}%" if h < 0.1 else f"{h:.1f}%"
            ax_c.text(b.get_x() + b.get_width() / 2.0, h + 1.0, txt, ha="center", va="bottom", fontsize=LABEL_FS)
ax_c.set_ylim(0, max(Nt_pct.max(), R_pct.max()) * 1.15)
ax_c.legend(fontsize=11, loc="upper right"); ax_c.set_facecolor("#f9f9f9")
for ax in (ax_dsd, ax_hist, ax_c):
    ax.tick_params(direction="in", top=True, right=True)
    
plt.savefig(C.OUT_DIR / "Fig5.png")
plt.savefig(C.OUT_DIR / "Fig5.tif", dpi=300, bbox_inches="tight")

tbl = pd.DataFrame({"class": [c[0].replace("$", "") for c in rr_cats], "n_spectra": counts, "pct_of_spectra": pcts})
tbl2 = pd.DataFrame({"D_class_mm": labels_c, "pct_of_Nt": Nt_pct, "pct_of_R": R_pct})
tbl.to_csv(C.OUT_DIR / "Fig5_class_frequency.csv", index=False)
tbl2.to_csv(C.OUT_DIR / "Fig5_contributions.csv", index=False)
print(f"\n--- Fig. 5 (for the caption): N = {N} spectra, classes by '{C.CLASS_RATE}' rain rate ---")
print(tbl.round(2).to_string(index=False)); print(); print(tbl2.round(3).to_string(index=False))
plt.show()
