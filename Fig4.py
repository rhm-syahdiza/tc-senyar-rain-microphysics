"""
Fig4: DSD time series, D (mm) vs time, colour = log10 N(D), with the 1-min rain rate overlaid.
"""
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

import dsd_common as C

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
LABEL_FS, TICK_FS = 16, 14
D_MAX_PLOT, D_RAIN0, R_MAX_DISPLAY = 8.0, 5.5, 120.0
T_PLOT_START = dt.datetime(2025, 11, 25, 12, 0)          # UTC
T_PLOT_END   = dt.datetime(2025, 11, 27, 0, 0)
OBS_END      = dt.datetime(2025, 11, 26, 21, 26)         # Matches Fig 1 power outage time

data = C.get_qc_dataset()
d = data["dsd"]
t_all_utc = np.array([t - C.UTC_OFFSET for t in data["times_all"]], dtype="datetime64[m]")
t_qc_utc = np.array([t - C.UTC_OFFSET for t in data["t"]], dtype="datetime64[m]")
t0 = t_all_utc.min()
n_min = int((t_all_utc.max() - t0) / np.timedelta64(1, "m")) + 1
k = ((t_qc_utc - t0) / np.timedelta64(1, "m")).astype(int)

with np.errstate(divide="ignore"):
    logND = np.where(d["nd"] > 0, np.log10(d["nd"]), np.nan)
grid_nd = np.full((n_min, 32), np.nan); grid_nd[k] = logND
grid_R = np.full(n_min, np.nan);        grid_R[k] = d["R"]

D_plot = C.D_bins[C.D_MASK]
D_edges = C.make_edges(D_plot, lower=0.0)
t0_num = mdates.date2num(t0.astype("datetime64[s]").astype(dt.datetime))
t_edges = t0_num + (np.arange(n_min + 1) - 0.5) / 1440.0
t_cent = t0_num + np.arange(n_min) / 1440.0

fig, ax1 = plt.subplots(figsize=(14, 5))
cmap = plt.cm.viridis.copy(); cmap.set_bad("white")
pcm = ax1.pcolormesh(t_edges, D_edges, grid_nd[:, C.D_MASK].T, cmap=cmap, vmin=0, vmax=4, shading="flat", zorder=1)
cbar = plt.colorbar(pcm, ax=ax1, pad=0.08, aspect=25, fraction=0.03)
cbar.set_label("log$_{10}[N(D)]$ (mm$^{-1}$ m$^{-3}$)", fontsize=12, fontweight="bold")
cbar.set_ticks(np.arange(0, 5, 1)); cbar.ax.tick_params(labelsize=TICK_FS); cbar.outline.set_linewidth(1.2)

R_scale = (D_MAX_PLOT - D_RAIN0) / R_MAX_DISPLAY
ax1.plot(t_cent, D_RAIN0 + grid_R * R_scale, color="magenta", lw=1.5, zorder=5)
ax1.axhline(D_RAIN0, color="magenta", lw=1.0, ls="--", alpha=0.6, zorder=4)

# End of record -> shaded "no data" (Hardcoded to 21:26 UTC to match Fig 1)
last_rec = mdates.date2num(OBS_END)
if last_rec < mdates.date2num(T_PLOT_END):
    ax1.axvspan(last_rec, mdates.date2num(T_PLOT_END), color="0.85", zorder=2)
    ax1.text(0.5 * (last_rec + mdates.date2num(T_PLOT_END)), 2.5, "No data", rotation=90, ha="center", va="center",
             fontsize=13, fontweight="bold", color="0.35", zorder=6)
ax1.axvline(last_rec, color="limegreen", ls=":", lw=2.5, zorder=6)

ax2 = ax1.twinx()
R_ticks = np.array([0, 20, 40, 60, 80, 100, 120]); R_pos = D_RAIN0 + R_ticks * R_scale
m = (R_pos >= D_RAIN0) & (R_pos <= D_MAX_PLOT)
ax2.set_yticks(R_pos[m]); ax2.set_yticklabels([str(int(v)) for v in R_ticks[m]], color="magenta", fontweight="bold")
ax2.set_ylabel("1-min Rain Rate, $R$ (mm h$^{-1}$)", color="magenta", fontsize=12, fontweight="bold", labelpad=5)
ax2.tick_params(axis="y", colors="magenta", direction="in", labelsize=TICK_FS)
ax2.spines["right"].set_color("magenta"); ax2.spines["right"].set_linewidth(1.5)
ax2.set_ylim(0, D_MAX_PLOT)

ax1.set_ylim(0, D_MAX_PLOT)
ax1.set_ylabel("Diameter, $D$ (mm)", fontsize=LABEL_FS, fontweight="bold")
ax1.set_yticks(np.arange(0, D_MAX_PLOT + 1, 1)); ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
ax1.tick_params(which="both", direction="in", labelsize=TICK_FS)
ax1.tick_params(which="major", length=6); ax1.tick_params(which="minor", length=3)
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d%H"))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6)); ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
ax1.set_xlabel("Time (DDHH UTC)", fontsize=LABEL_FS, fontweight="bold")
ax1.set_xlim(mdates.date2num(T_PLOT_START), mdates.date2num(T_PLOT_END))
ax1.grid(True, which="major", axis="x", ls=":", lw=0.8, color="black", alpha=0.5, zorder=0)

plt.tight_layout()
plt.savefig(C.OUT_DIR / "Fig4.png")
plt.savefig(C.OUT_DIR / "Fig4.tif", dpi=300, bbox_inches="tight")

i = int(np.nanargmax(d["R"]))
print("\n--- Fig. 4 statistics (for the caption) ---")
print(f"1-min spectra plotted: {len(d['R']):,}; first/last record {t_all_utc.min()} / {t_all_utc.max()} UTC")
print(f"Maximum 1-min rain rate (R, {C.RAIN_RATE_METHOD} method): {d['R'][i]:.1f} mm/h at {t_qc_utc[i]} UTC")
print(f"Largest diameter class with drops (D, mm): {C.D_bins[C.D_MASK][(d['nd'][:, C.D_MASK] > 0).any(axis=0)].max():.3f}")
print(f"Saved -> {C.OUT_DIR / 'Fig4.png'} and {C.OUT_DIR / 'Fig4.tif'}")
plt.show()
