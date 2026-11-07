"""
Fig. 2 (mean fall speed vs. drop diameter, Parsivel, Kototabang) + analyses
--------------------------------------------------------------------------------------------
QC cascade (samples AND drops removed at each step)
sensitivity to the +/-60 % Atlas velocity filter
drop counts per diameter bin (in Fig. 2 and in CSV)
sensitivity to the minimum-drop-count threshold
prints the density correction actually applied
per-minute large-drop velocity anomaly time series (CSV)

All loading, constants, Atlas speed and the QC settings come from dsd_common.py (same as Fig4-Fig8).
Outputs (in dsd_common.OUT_DIR = ./Fig_outputs):
  Fig2.png / Fig2.tif                    main figure: (a) heat map + mean +/- sigma, (b) drop counts per bin
  FigS4.png/FigS4.tif                    supplementary figure
  QC_cascade_samples.csv, QC_cascade_drops.csv
  QC_sensitivity_velocity_filter.csv, QC_sensitivity_min_drops.csv
  Fig2_bin_statistics.csv
  large_drop_anomaly_timeseries.csv
"""

import csv, warnings
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

import dsd_common as C

# ── Settings taken from dsd_common (do NOT redefine them here) ────────────────
OUT_DIR    = C.OUT_DIR
ALTITUDE_M = C.ALTITUDE_M
UTC_OFFSET, T_START, T_END = C.UTC_OFFSET, C.T_START, C.T_END           # window is in LOCAL time (UTC+7)
RR_MIN, MIN_DROPS, D_MIN, D_MAX = C.RR_MIN, C.MIN_DROPS, C.D_MIN, C.D_MAX
BASE_FRAC = C.VEL_FRAC                 # baseline velocity envelope: +/-60 % of Atlas

# ── Settings specific to Fig. 2 ────────────────────────────────────────────────
FRAC_LIST      = [0.20, 0.30, 0.40, 0.50, 0.60, None]   # sensitivity set (None = no velocity filter)
MIN_DROPS_LIST = [5, 10, 20, 30]
 
D_LARGE        = 2.5   # mm, "large drop" threshold used in the manuscript
MIN_BIN_DROPS  = 20    
FLAG_BIN_DROPS = 100   # bins with fewer drops are drawn with hollow markers in Fig. 2
 
BOOT_B    = 2000       # bootstrap replicates
BLOCK_LEN = 15         # block length (1-min samples) -> respects temporal autocorrelation
 
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
})
LABEL_FS, TICK_FS = 14, 12
 
# ── Parsivel classes and Atlas speed (from dsd_common) ─────────────────────────
D_bins, V_bins = C.D_bins, C.V_bins
density_factor = C.density_factor
atlas_v_corrected = C.v_atlas
V_emp     = C.V_TERM
D_MASK    = C.D_MASK                                            # centre in [D_MIN, D_MAX]
LARGE_IDX = np.where(D_MASK & (D_bins >= D_LARGE))[0]


def velocity_mask(frac):
    """[v, d] boolean: True where the velocity-bin centre lies within +/-frac of Atlas(D)."""
    if frac is None:
        return np.ones((32, 32), dtype=bool)
    lo = (1 - frac) * V_emp[None, :]
    hi = (1 + frac) * V_emp[None, :]
    return (V_bins[:, None] >= lo) & (V_bins[:, None] <= hi)
 
 
# ── Data loading (dsd_common.load_spectra; diagnostics only for the QC cascade counters) ─
def load_diagnostics():
    """Row counters for the cascade table (same parsing rules as dsd_common.load_spectra)."""
    c = dict(rows=0, bad_time=0, in_window=0, bad_values=0)
    for path in C.iter_csv_files():
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                c["rows"] += 1
                try:
                    t = dt.datetime.strptime(f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S")
                except Exception:
                    c["bad_time"] += 1
                    continue
                if not (T_START <= t <= T_END):
                    continue
                c["in_window"] += 1
                try:
                    float(row[2])
                    vals = [float(x) if x.strip() else 0.0 for x in row[15:1038]]
                    vals += [0.0] * (1024 - len(vals))
                    np.array(vals[:1024], dtype=float).reshape(32, 32)
                except Exception as e:
                    c["bad_values"] += 1
                    c.setdefault("first_error", f"{path} | {type(e).__name__}: {e} | row length {len(row)}")
    return c


def load_spectra():
    times, rr, specs = C.load_spectra()
    return times, rr, specs, load_diagnostics()


# ── QC cascade ─────────────────────────────────────────────────────────────────
def run_qc(specs, rr, frac=BASE_FRAC, min_drops=MIN_DROPS, rr_min=RR_MIN):
    """
    Order: rain rate -> diameter filter -> velocity filter -> minimum drop count.
    Because the diameter and velocity filters only remove drops, the final sample set is identical
    to testing the drop count last. Samples lost to the drop-count criterion are attributed to the
    first stage that pushed them below `min_drops`.
    """
    idx_rr = np.where(rr >= rr_min)[0]
    s0 = specs[idx_rr]                              # rain-rate-passing, raw spectra
    s1 = s0 * D_MASK[None, None, :]                 # after diameter filter
    s2 = s1 * velocity_mask(frac)[None, :, :]       # after velocity filter
    c0, c1, c2 = (x.sum(axis=(1, 2)) for x in (s0, s1, s2))
 
    fail_raw = c0 < min_drops
    fail_d   = (~fail_raw) & (c1 < min_drops)
    fail_v   = (~fail_raw) & (~fail_d) & (c2 < min_drops)
    keep     = c2 >= min_drops
 
    return dict(
        idx=idx_rr[keep], spec=s2[keep], n_rr_pass=len(idx_rr),
        n_fail_raw=int(fail_raw.sum()), n_fail_d=int(fail_d.sum()), n_fail_v=int(fail_v.sum()),
        drops_in=s0.sum(),
        drops_lo=s0[:, :, D_bins < D_MIN].sum(), drops_hi=s0[:, :, D_bins > D_MAX].sum(),
        drops_after_d=s1.sum(), drops_after_v=s2.sum(), drops_final=s2[keep].sum())
 
 
def cascade_tables(counts, q):
    N0, nbad = counts["in_window"], counts["bad_values"]
    n_readable = N0 - nbad
    steps = [
        ("Unreadable rain-rate / spectrum fields", nbad),
        (f"Rain rate < {RR_MIN} mm/h", n_readable - q["n_rr_pass"]),
        (f"< {MIN_DROPS} drops in raw spectrum", q["n_fail_raw"]),
        (f"< {MIN_DROPS} drops after diameter filter ({D_MIN}-{D_MAX} mm)", q["n_fail_d"]),
        (f"< {MIN_DROPS} drops after velocity filter (±{int(BASE_FRAC*100)} % Atlas)", q["n_fail_v"]),
    ]
    rows, n = [dict(step="All 1-min records in time window", removed=0, remaining=N0,
                    pct_removed_of_initial=0.0, pct_remaining=100.0)], N0
    for name, rem in steps:
        n -= rem
        rows.append(dict(step=name, removed=int(rem), remaining=int(n),
                         pct_removed_of_initial=100 * rem / N0, pct_remaining=100 * n / N0))
    assert n == len(q["idx"]), "cascade does not add up"
    samples = pd.DataFrame(rows)
 
    din = q["drops_in"]
    drops = pd.DataFrame([
        ("Drops in rain-rate-passing spectra (all bins)", din),
        (f"Removed: D < {D_MIN} mm", q["drops_lo"]),
        (f"Removed: D > {D_MAX} mm", q["drops_hi"]),
        (f"Removed: outside ±{int(BASE_FRAC*100)} % Atlas velocity envelope", q["drops_after_d"] - q["drops_after_v"]),
        (f"Removed: lost with spectra rejected for < {MIN_DROPS} drops", q["drops_after_v"] - q["drops_final"]),
        ("Drops in final retained spectra", q["drops_final"]),
    ], columns=["item", "drops"])
    drops["pct_of_rain_passing_drops"] = 100 * drops["drops"] / din
    return samples, drops
 
 
# ── Statistics ─────────────────────────────────────────────────────────────────
def bin_stats(spec):
    N   = spec.sum(axis=0)                    # [v, d]
    n_d = N.sum(axis=0)
    mean, std = np.full(32, np.nan), np.full(32, np.nan)
    for d in range(32):
        if not D_MASK[d] or n_d[d] < MIN_BIN_DROPS:
            continue
        mean[d] = (V_bins * N[:, d]).sum() / n_d[d]
        std[d]  = np.sqrt((N[:, d] * (V_bins - mean[d]) ** 2).sum() / n_d[d])
    return N, n_d, mean, std
 
 
def block_bootstrap(sum_n, sum_nv, B=BOOT_B, L=BLOCK_LEN, seed=42):
    """Moving-block bootstrap over 1-min samples (in time order).
    Returns per-bin mean velocity [B,32] and the drop-weighted mean anomaly of large drops [B]."""
    rng = np.random.default_rng(seed)
    n = sum_n.shape[0]
    boot_bin, boot_large = np.full((B, 32), np.nan), np.full(B, np.nan)
    for b in range(B):
        idx = C.block_idx(n, L, rng)                      # same moving-block resampling as Fig7/Fig8
        sn, snv = sum_n[idx].sum(0), sum_nv[idx].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_bin[b] = np.where(sn > 0, snv / sn, np.nan)
        tot = sn[LARGE_IDX].sum()
        if tot > 0:
            boot_large[b] = (snv[LARGE_IDX].sum() - (V_emp[LARGE_IDX] * sn[LARGE_IDX]).sum()) / tot
    return boot_bin, boot_large
 
 
def large_drop_bias(spec, B=BOOT_B):
    """Drop-weighted mean (V_obs - V_Atlas) over D >= D_LARGE with block-bootstrap 95 % CI."""
    sum_n  = spec.sum(axis=1)                                   # [sample, d]
    sum_nv = (spec * V_bins[None, :, None]).sum(axis=1)
    tot = sum_n[:, LARGE_IDX].sum()
    bias = (sum_nv[:, LARGE_IDX].sum() - (V_emp[LARGE_IDX] * sum_n[:, LARGE_IDX].sum(0)).sum()) / tot
    boot_bin, boot_large = block_bootstrap(sum_n, sum_nv, B=B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_lo = np.nanpercentile(boot_bin, 2.5, axis=0) - V_emp
        bin_hi = np.nanpercentile(boot_bin, 97.5, axis=0) - V_emp
    return dict(bias=bias, lo=np.nanpercentile(boot_large, 2.5), hi=np.nanpercentile(boot_large, 97.5),
                n_large=int(tot), bin_lo=bin_lo, bin_hi=bin_hi)
 
 
def flabel(f):
    return "No filter" if f is None else f"±{int(round(f * 100))} %"
 
 
# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_fig2(N, n_d, mean_v, std_v, n_samples):
    D_edges, V_edges = C.make_edges(D_bins), C.make_edges(V_bins)
    D_line = np.linspace(0.3, 8.0, 200)
 
    fig = plt.figure(figsize=(8.5, 8.2), facecolor="white")
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.035], height_ratios=[3.4, 1.2], hspace=0.07, wspace=0.03)
    ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axn = fig.add_subplot(gs[1, 0], sharex=ax)
 
 
    N_plot = np.where(N > 0, N, np.nan)
    mesh = ax.pcolormesh(D_edges, V_edges, N_plot, cmap="viridis",
                         norm=mcolors.LogNorm(vmin=1, vmax=np.nanmax(N_plot)), shading="flat", zorder=1)
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("Drop count", fontsize=LABEL_FS, fontweight="bold")
    cb.ax.tick_params(labelsize=TICK_FS); cb.outline.set_linewidth(1.2)
 
    ax.plot(D_line, (1 - BASE_FRAC) * atlas_v_corrected(D_line), "k--", lw=1.5, zorder=3,
            label=f"±{int(BASE_FRAC*100)} % Atlas (1973)")
    ax.plot(D_line, (1 + BASE_FRAC) * atlas_v_corrected(D_line), "k--", lw=1.5, zorder=3)
    ax.plot(D_line, atlas_v_corrected(D_line), "k-", lw=2.0, zorder=4, label="Atlas et al. (1973)")
 
    valid = ~np.isnan(mean_v)
    hi, lo = valid & (n_d >= FLAG_BIN_DROPS), valid & (n_d < FLAG_BIN_DROPS)
    ax.errorbar(D_bins[valid], mean_v[valid], yerr=std_v[valid], fmt="none", ecolor="red",
                capsize=4, lw=1.5, zorder=5)
    ax.plot(D_bins[hi], mean_v[hi], "s", mfc="red", mec="white", ms=7, zorder=6,
            label=f"Mean ± σ (n ≥ {FLAG_BIN_DROPS} drops)")
    ax.plot(D_bins[lo], mean_v[lo], "s", mfc="white", mec="red", mew=1.5, ms=7, zorder=6,
            label=f"Mean ± σ ({MIN_BIN_DROPS} ≤ n < {FLAG_BIN_DROPS} drops)")
 
    ax.set_ylabel("Fall Speed (m s$^{-1}$)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_xlim(0, 8); ax.set_ylim(0, 15)
    ax.yaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which="both", top=True, right=True, labelsize=TICK_FS)
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=10, loc="upper left", frameon=True, framealpha=0.9, edgecolor="gray")
    ax.text(0.98, 0.04, f"N = {n_samples:,} 1-min spectra", transform=ax.transAxes, ha="right",
            fontsize=11, bbox=dict(fc="white", ec="gray", alpha=0.9))
    ax.text(0.01, 1.01, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold")
 
    # (b) drop counts per diameter bin
    colors = ["#4C72B0" if n >= FLAG_BIN_DROPS else "#E69F00" if n >= MIN_BIN_DROPS else "#B0B0B0" for n in n_d]
    axn.bar(D_bins, n_d, width=np.diff(D_edges) * 0.9, color=colors, edgecolor="black", lw=0.6, zorder=2)
    axn.set_yscale("log"); axn.set_ylim(0.7, max(n_d.max(), 10) * 12)
    axn.axhline(MIN_BIN_DROPS, color="k", ls=":", lw=1.2, zorder=3)
    axn.text(7.95, MIN_BIN_DROPS * 1.15, f"n = {MIN_BIN_DROPS}", ha="right", fontsize=9)
    for d in range(32):
        if D_MASK[d] and n_d[d] > 0 and 2.0 <= D_bins[d] <= 8.0:
            axn.text(D_bins[d], n_d[d] * 1.25, f"{int(n_d[d])}", ha="center", va="bottom", fontsize=9, rotation=90)
    axn.set_xlim(0, 8); axn.set_xlabel("Diameter (mm)", fontsize=LABEL_FS, fontweight="bold")
    axn.set_ylabel("Drops per bin", fontsize=LABEL_FS - 1, fontweight="bold")
    axn.xaxis.set_major_locator(MultipleLocator(1)); axn.xaxis.set_minor_locator(MultipleLocator(0.5))
    axn.tick_params(which="both", top=True, right=True, labelsize=TICK_FS)
    axn.text(0.01, 0.88, "(b)", transform=axn.transAxes, fontsize=13, fontweight="bold")
    fig.savefig(OUT_DIR / "Fig2.png")
    fig.savefig(OUT_DIR / "Fig2.tif", dpi=300, bbox_inches="tight")
    plt.close(fig)
 
 
def plot_sensitivity(runs, results):
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(FRAC_LIST)))
    D_line = np.linspace(0.3, 8.0, 200)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), facecolor="white")
 
    for f, col in zip(FRAC_LIST, colors):
        r = runs[f]; v = ~np.isnan(r["mean"])
        axs[0].plot(D_bins[v], r["mean"][v], "-o", color=col, ms=4, lw=1.4, label=flabel(f))
        axs[1].plot(D_bins[v], (r["mean"] - V_emp)[v], "-o", color=col, ms=4, lw=1.4, label=flabel(f))
    axs[0].plot(D_line, atlas_v_corrected(D_line), "k-", lw=2.2, label="Atlas (1973)")
 
    b = runs[BASE_FRAC]; v = ~np.isnan(b["mean"])
    axs[1].fill_between(D_bins[v], b["bin_lo"][v], b["bin_hi"][v], color="0.6", alpha=0.3, zorder=0,
                        label=f"95 % CI ({flabel(BASE_FRAC)})")
    axs[1].axhline(0, color="k", lw=1.2)
 
    for a, yl in zip(axs[:2], ["Mean fall speed (m s$^{-1}$)", "Mean fall speed − Atlas (m s$^{-1}$)"]):
        a.axvspan(D_LARGE, 8.5, color="0.92", zorder=0)
        a.set_xlim(0, 8); a.set_xlabel("Diameter (mm)", fontsize=LABEL_FS, fontweight="bold")
        a.set_ylabel(yl, fontsize=LABEL_FS, fontweight="bold")
        a.tick_params(which="both", top=True, right=True, labelsize=TICK_FS)
        a.xaxis.set_major_locator(MultipleLocator(1))
    axs[0].set_ylim(0, 12); axs[0].legend(title="Velocity filter", fontsize=10, loc="upper left")
    axs[1].legend(fontsize=9, loc="upper left", ncol=2)
 
    x = np.arange(len(FRAC_LIST))
    bias = np.array([results[f]["bias"] for f in FRAC_LIST])
    lo = np.array([results[f]["lo"] for f in FRAC_LIST]); hi = np.array([results[f]["hi"] for f in FRAC_LIST])
    axs[2].errorbar(x, bias, yerr=[bias - lo, hi - bias], fmt="s", color="crimson", ms=8, capsize=5, lw=1.8)
    axs[2].axhline(0, color="k", lw=1.2)
    axs[2].set_xticks(x); axs[2].set_xticklabels([flabel(f) for f in FRAC_LIST], fontsize=TICK_FS)
    for xi, f in zip(x, FRAC_LIST):
        axs[2].text(xi, axs[2].get_ylim()[0], f"n={results[f]['n_large']:,}", ha="center", va="bottom", fontsize=9)
    axs[2].set_xlabel("Velocity filter half-width", fontsize=LABEL_FS, fontweight="bold")
    axs[2].set_ylabel(f"Mean (V$_{{obs}}$ − V$_{{Atlas}}$), D ≥ {D_LARGE} mm (m s$^{{-1}}$)", fontsize=LABEL_FS - 1, fontweight="bold")
    axs[2].tick_params(which="both", top=True, right=True, labelsize=TICK_FS)
    for a, lab in zip(axs, "abc"):
        a.text(0.01, 1.02, f"({lab})", transform=a.transAxes, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "FigS4.png")
    fig.savefig(OUT_DIR / "FigS4.tif", dpi=300, bbox_inches="tight")
    plt.close(fig)
 
 
# ── Reporting helpers ──────────────────────────────────────────────────────────
def report_coverage(times, t_keep):
    ts = pd.Series(pd.to_datetime(list(times))).sort_values().reset_index(drop=True)
    nominal = int((T_END - T_START).total_seconds() // 60) + 1
    print(f"\nCoverage (local time = UTC+7)")
    print(f"  Window         : {T_START} to {T_END}  (= {T_START - UTC_OFFSET} to {T_END - UTC_OFFSET} UTC)")
    print(f"  Records        : {len(ts):,} of {nominal:,} nominal minutes; duplicates: {len(ts) - ts.nunique():,}")
    print(f"  First / last   : {ts.iloc[0]} / {ts.iloc[-1]} local")
    t_obs_end = dt.datetime(2025, 11, 26, 21, 0, 0) + UTC_OFFSET      # end of the 33-h observation period (local)
    n_after = int((ts > t_obs_end).sum()); n_obs = int((ts <= t_obs_end).sum())
    print(f"  33-h period    : {n_obs:,} records up to 21 UTC 26 Nov (max possible {33*60+1:,}); {n_after:,} records AFTER it")
    gaps = ts.diff(); big = gaps[gaps > pd.Timedelta(minutes=5)].sort_values(ascending=False).head(5)
    for i, g in big.items():
        print(f"  Gap {g}: {ts[i-1]} -> {ts[i]} local")
    tk = pd.Series(pd.to_datetime(list(t_keep))).sort_values()
    print(f"  Retained       : {tk.iloc[0]} to {tk.iloc[-1]} local  ({tk.iloc[0]-UTC_OFFSET} to {tk.iloc[-1]-UTC_OFFSET} UTC)")
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    rho_z, corr = density_factor(ALTITUDE_M)
    print(f"Density correction: standard atmosphere at {ALTITUDE_M:.0f} m -> rho = {rho_z:.4f} kg m-3, "
          f"(rho0/rho)^0.4 = {corr:.4f}")
 
    times, rr, specs, counts = load_spectra()
    N0 = counts["in_window"]
    print(f"\nRows with a timestamp: {counts['rows']:,}; unparseable timestamp/header rows: {counts['bad_time']:,}; "
          f"in time window: {N0:,}; unreadable values: {counts['bad_values']:,}")
 
    # ---- baseline QC cascade ----
    if len(times) != N0 - counts["bad_values"]:
        print(f"WARNING: loader/diagnostic mismatch ({len(times)} vs {N0 - counts['bad_values']}) - check parsing rules")
    q = run_qc(specs, rr)
    idx_c, spec_c, _ = C.qc(specs, rr)                      # baseline must reproduce dsd_common.qc exactly
    same = np.array_equal(idx_c, q["idx"]) and np.allclose(spec_c, q["spec"])
    print(f"Baseline QC identical to dsd_common.qc: {same} ({len(idx_c):,} spectra)")
    assert same, "Fig2 QC differs from dsd_common.qc - the figures would no longer share one dataset"
    samples_tbl, drops_tbl = cascade_tables(counts, q)
    pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.float_format", "{:,.2f}".format)
    print("\n=== QC cascade: 1-min samples ===");  print(samples_tbl.to_string(index=False))
    print("\n=== QC cascade: drops ===");           print(drops_tbl.to_string(index=False))
    samples_tbl.to_csv(OUT_DIR / "QC_cascade_samples.csv", index=False)
    drops_tbl.to_csv(OUT_DIR / "QC_cascade_drops.csv", index=False)
    if "first_error" in counts:
        print(f"\nFirst unreadable record: {counts['first_error']}")
    if len(q["idx"]) == 0:
        print("\nNo spectra survived QC - fix the parsing problem above before using any output.")
        return
    report_coverage(times, times[q["idx"]])
 
    # ---- baseline bin statistics + Fig. 2 ----
    N, n_d, mean_v, std_v = bin_stats(q["spec"])
    base = large_drop_bias(q["spec"])
    stats = pd.DataFrame(dict(D_mm=D_bins, n_drops=n_d.astype(int), mean_v=mean_v, std_v=std_v,
                              atlas_v=V_emp, bias=mean_v - V_emp,
                              bias_ci_lo=base["bin_lo"], bias_ci_hi=base["bin_hi"]))
    stats[D_MASK & (D_bins <= 8.5)].to_csv(OUT_DIR / "Fig2_bin_statistics.csv", index=False)
    print(f"\nFinal sample size: {len(q['idx']):,}  ({100*len(q['idx'])/N0:.1f} % of {N0:,} records in window)")
    print("(The manuscript reports 1,452 -- if this number differs, check D_MIN / paths before using any table.)")
    plot_fig2(N, n_d, mean_v, std_v, len(q["idx"]))
 
    # ---- velocity-filter sensitivity ----
    runs, results, rows = {}, {}, []
    for f in FRAC_LIST:
        qf = run_qc(specs, rr, frac=f)
        _, nd_f, mean_f, std_f = bin_stats(qf["spec"])
        res = base if f == BASE_FRAC else large_drop_bias(qf["spec"])
        runs[f] = dict(mean=mean_f, std=std_f, n_d=nd_f, bin_lo=res["bin_lo"], bin_hi=res["bin_hi"])
        results[f] = res
        rows.append(dict(velocity_filter=flabel(f), samples=len(qf["idx"]),
                         pct_of_all_records=100 * len(qf["idx"]) / N0, drops=int(qf["drops_final"]),
                         large_drops=res["n_large"], bias_large_ms=res["bias"],
                         ci95_lo=res["lo"], ci95_hi=res["hi"], bias_positive_at_95=res["lo"] > 0))
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT_DIR / "QC_sensitivity_velocity_filter.csv", index=False)
    print(f"\n=== Velocity-filter sensitivity (large drops, D >= {D_LARGE} mm; block bootstrap, L = {BLOCK_LEN} min) ===")
    print(sens.to_string(index=False))
 
    print("\nVelocity bins retained per large-drop diameter bin (the filter is bin-limited when this is 1-2):")
    retained = pd.DataFrame({flabel(f): [int(velocity_mask(f)[:, d].sum()) for d in LARGE_IDX] for f in FRAC_LIST},
                            index=[f"D={D_bins[d]:.2f}" for d in LARGE_IDX]).T
    print(retained.to_string())
    plot_sensitivity(runs, results)
 
    # ---- minimum-drop-count sensitivity ----
    rows = []
    for m in MIN_DROPS_LIST:
        qm = run_qc(specs, rr, min_drops=m)
        res = large_drop_bias(qm["spec"], B=500)
        rows.append(dict(min_drops=m, samples=len(qm["idx"]), pct_of_all_records=100 * len(qm["idx"]) / N0,
                         large_drops=res["n_large"], bias_large_ms=res["bias"], ci95_lo=res["lo"], ci95_hi=res["hi"]))
    md = pd.DataFrame(rows); md.to_csv(OUT_DIR / "QC_sensitivity_min_drops.csv", index=False)
    print("\n=== Minimum-drop-count sensitivity (baseline velocity filter) ==="); print(md.to_string(index=False))
 
    # ---- per-minute large-drop anomaly time series (for comparison with EAR) ----
    s, idx = q["spec"], q["idx"]
    n_large = s[:, :, LARGE_IDX].sum(axis=(1, 2))
    dv = (s[:, :, LARGE_IDX] * (V_bins[None, :, None] - V_emp[LARGE_IDX][None, None, :])).sum(axis=(1, 2))
    ts = pd.DataFrame(dict(time_local=pd.to_datetime(list(times[idx])), rain_rate_mmh=rr[idx],
                           n_large_drops=n_large, mean_dV_large_ms=np.where(n_large > 0, dv / np.maximum(n_large, 1), np.nan)))
    ts.insert(1, "time_utc", ts["time_local"] - UTC_OFFSET)
    ts.to_csv(OUT_DIR / "large_drop_anomaly_timeseries.csv", index=False)
    print(f"\nAll outputs written to: {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()
