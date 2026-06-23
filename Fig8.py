"""
Z-R Scatter Plot for TC Senyar.
"""
import os, csv, datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =====================================================
# Density function
# =====================================================
def hist2d(X, Y, scaling=2):
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    Xremap = 100 * (X - xmin) / (xmax - xmin)
    Yremap = 100 * (Y - ymin) / (ymax - ymin)
    density = Xremap * np.nan
    for i in range(len(Xremap)):
        density[i] = np.sum(np.sqrt((Xremap[i] - Xremap)**2 + (Yremap[i] - Yremap)**2) < scaling)
    return density

# =====================================================
# Data Processing
# =====================================================
D_bins = np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187,
                   1.375, 1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750,
                   5.500, 6.500, 7.500, 8.500, 9.500, 11.00, 13.00, 15.00, 17.00, 19.00, 21.50, 24.50])
V_bins = np.array([0.050, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750, 0.850, 0.950,
                   1.100, 1.300, 1.500, 1.700, 1.900, 2.200, 2.600, 3.000, 3.400, 3.800,
                   4.400, 5.200, 6.000, 6.800, 7.600, 8.800, 10.40, 12.00, 13.60, 15.20, 17.60, 20.80])
dD = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
               0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 0.500, 0.500, 0.500, 0.500,
               1.000, 1.000, 1.000, 1.000, 1.000, 2.000, 2.000, 2.000, 2.000, 2.000, 3.000, 3.000])
def atlas_v_corrected(d, altitude_m=865.0):
    v_sea_level = 9.65 - 10.3 * np.exp(-0.6 * d)
    rho_0, T_0, L, R_gas, g = 1.225, 288.15, 0.0065, 287.05, 9.80665
    T_z = T_0 - (L * altitude_m)
    rho_z = rho_0 * (T_z / T_0) ** ((g / (L * R_gas)) - 1)
    return np.maximum(v_sea_level * ((rho_0 / rho_z) ** 0.4), 0.0)
D_valid_mask = (D_bins >= 0.312) & (D_bins <= 10.0)
V_term = atlas_v_corrected(D_bins, altitude_m=865.0)
A_eff = (180.0 * (30.0 - (D_bins / 2.0))) * 1e-6
t_int, V_FRAC, R_MIN = 60.0, 0.60, 0.1
t_start = dt.datetime(2025, 11, 25, 12, 0, 0) + dt.timedelta(hours=7)
t_end   = dt.datetime(2025, 11, 28, 12, 0, 0) + dt.timedelta(hours=7)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_root  = os.path.join(script_dir, "Kototabang")
R_derived_list = []
Z_derived_list = []
total_rows_read = 0
passed_time_window = 0
passed_csv_rain_filter = 0
passed_velocity_filter = 0
# =====================================================
# QC
# =====================================================
for month in sorted(os.listdir(data_root)):
    mpath = os.path.join(data_root, month)
    if not os.path.isdir(mpath): continue
    for day in sorted(os.listdir(mpath)):
        dpath = os.path.join(mpath, day)
        if not os.path.isdir(dpath): continue
        for fname in sorted(os.listdir(dpath)):
            if not fname.endswith(".csv"): continue
            with open(os.path.join(dpath, fname), "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if not row or not row[0].strip(): continue
                    total_rows_read += 1
                    try:
                        t = dt.datetime.strptime(f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S")
                        if not (t_start <= t <= t_end): continue
                        passed_time_window += 1
                        r_csv = float(row[2])
                        if r_csv < R_MIN: continue
                        passed_csv_rain_filter += 1
                        spec_raw = row[15:1038]
                        vals = [float(x) if (x and x.strip()) else 0.0 for x in spec_raw]
                        vals += [0.0] * (1024 - len(vals))
                        spec = np.array(vals[:1024]).reshape(32, 32)
                        for col_idx in range(32):
                            if not D_valid_mask[col_idx]:
                                spec[:, col_idx] = 0.0
                                continue
                            v_low  = V_term[col_idx] * (1.0 - V_FRAC)
                            v_high = V_term[col_idx] * (1.0 + V_FRAC)
                            invalid = (V_bins < v_low) | (V_bins > v_high)
                            spec[invalid, col_idx] = 0.0
                        if spec.sum() < 1: continue
                        passed_velocity_filter += 1
                        n_d = np.zeros(32)
                        for d_i in range(32):
                            if not D_valid_mask[d_i]: continue
                            n_d[d_i] = np.sum(np.where(V_bins > 0,
                                spec[:, d_i] / (V_bins * A_eff[d_i] * t_int * dD[d_i]), 0.0))
                        nd_v = n_d[D_valid_mask]
                        D_v  = D_bins[D_valid_mask]
                        dw_v = dD[D_valid_mask]
                        r_dsd = 6 * np.pi * 1e-4 * np.sum(V_term[D_valid_mask] * (D_v**3) * nd_v * dw_v)
                        M6    = np.sum(nd_v * (D_v**6) * dw_v)
                        if r_dsd >= R_MIN and M6 > 0:
                            R_derived_list.append(r_dsd)
                            Z_derived_list.append(M6)
                    except Exception as e:
                        print(f"Row Exception: {e}")
                        continue
                    
# =====================================================
# Diagnostics
# =====================================================
print("\n─── PIPELINE FILTER DIAGNOSTICS ───")
print(f" Raw rows processed           : {total_rows_read}")
print(f" Rows matching Time Window    : {passed_time_window}")
print(f" Rows passing CSV R >= 0.1    : {passed_csv_rain_filter}")
print(f" Rows passing Velocity & Drops: {passed_velocity_filter}")
print(f" Final Validated Data Points  : {len(R_derived_list)}\n")
if len(R_derived_list) == 0:
    raise ValueError("No data points survived QC pipeline.")
R_obs = np.array(R_derived_list)
Z_obs = np.array(Z_derived_list)
# =====================================================
# Transform to 10log10 space
# x = 10*log10(R) [dBR], y = 10*log10(Z) [dBZ]
# Z = A*R^b therefore y = b*x + 10*log10(A)
# =====================================================
dBR_obs = 10 * np.log10(R_obs)
dBZ_obs = 10 * np.log10(Z_obs)

# =====================================================
# Stats
# =====================================================
res         = stats.linregress(dBR_obs, dBZ_obs)
b_fit       = res.slope
A_fit       = 10 ** (res.intercept / 10.0)
n_samples   = len(dBR_obs)

# =====================================================
# RMSE in dBR = 10*log10(R)
# =====================================================
def calc_rmse_dBR(A, b):
    dBR_pred = (dBZ_obs - 10 * np.log10(A)) / b
    return np.sqrt(np.nanmean((dBR_obs - dBR_pred) ** 2))
RMSE_fit   = calc_rmse_dBR(A_fit,   b_fit)
RMSE_mp    = calc_rmse_dBR(200.0,   1.60)
RMSE_ros   = calc_rmse_dBR(250.0,   1.20)
RMSE_mzk   = calc_rmse_dBR(279.0,   1.43)
RMSE_chang = calc_rmse_dBR(206.83,  1.45)
RMSE_deo   = calc_rmse_dBR(250.0,   1.30)
RMSE_wen   = calc_rmse_dBR(147.28,  1.38)
RMSE_jan   = calc_rmse_dBR(148.44,  1.30)
RMSE_bao   = calc_rmse_dBR(237.58,  1.43)
print("\n─── RMSE SUMMARY (dBR) ───")
eq_labels = ['TC Senyar fit', 'Marshall-Palmer', 'Rosenfeld', 'Marzuki 2018',
             'Chang 2009', 'Deo & Walsh 2016', 'Wen 2018', 'Janapati 2020', 'Bao 2020']
rmses = [RMSE_fit, RMSE_mp, RMSE_ros, RMSE_mzk,
         RMSE_chang, RMSE_deo, RMSE_wen, RMSE_jan, RMSE_bao]
for lbl, rmse in zip(eq_labels, rmses):
    print(f"  {lbl:<22}: {rmse:.2f} dBR")
# =====================================================
# Figure Generation
# =====================================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

point_density = hist2d(dBR_obs, dBZ_obs, scaling=2)
sc = ax.scatter(dBR_obs, dBZ_obs, s=10, c=point_density, cmap='viridis', zorder=2)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.ax.set_ylabel('Point Density', size=11, fontweight='bold')
cb.outline.set_linewidth(1.2)

dBR_line     = np.linspace(-10, 20, 500)
dBZ_line_fit = b_fit * dBR_line + 10 * np.log10(A_fit)


def zr_dBZ(A, b, dBR):
    return b * dBR + 10 * np.log10(A)
colors = ['#000000', '#7F7F7F', '#FF7F0E', '#9467BD', '#D62728',
          '#1F77B4', '#2CA02C', '#17BECF', '#8C564B']

# TC Senyar fit line
ax.plot(dBR_line, dBZ_line_fit, color=colors[0], ls='-', lw=2.5, zorder=5,
        label=f"TC Senyar (KT-OR): $Z={A_fit:.0f}R^{{{b_fit:.2f}}}$; "
              f"$r^2$={res.rvalue**2:.2f}; RMSE={RMSE_fit:.1f} dBR")

# Reference Z-R equations
ax.plot(dBR_line, zr_dBZ(200.0,  1.60,  dBR_line), color=colors[1], ls='--', lw=1.8, zorder=3,
        label=f"Marshall-Palmer (1948): $Z=200R^{{1.6}}$; RMSE={RMSE_mp:.1f} dBR")
ax.plot(dBR_line, zr_dBZ(250.0,  1.20,  dBR_line), color=colors[2], ls='--', lw=1.8, zorder=3,
        label=f"Rosenfeld (1993): $Z=250R^{{1.2}}$; RMSE={RMSE_ros:.1f} dBR")
ax.plot(dBR_line, zr_dBZ(279.0,  1.43,  dBR_line), color=colors[3], ls='-',  lw=1.5, zorder=3,
        label=f"Marzuki (2018): $Z=279R^{{1.43}}$; RMSE={RMSE_mzk:.1f} dBR")
ax.plot(dBR_line, zr_dBZ(206.83, 1.45,  dBR_line), color=colors[4], ls='-',  lw=1.5, zorder=3,
        label=f"Chang (2009): $Z=206.83R^{{1.45}}$; RMSE={RMSE_chang:.1f} dBR")
ax.plot(dBR_line, zr_dBZ(250.0,  1.30,  dBR_line), color=colors[5], ls='-',  lw=1.5, zorder=3,
        label=f"Deo & Walsh (2016): $Z=250R^{{1.3}}$; RMSE={RMSE_deo:.1f} dBR")
ax.plot(dBR_line, zr_dBZ(147.28, 1.38,  dBR_line), color=colors[6], ls='-',  lw=1.5, zorder=3,
        label=f"Wen (2018): $Z=147.28R^{{1.38}}$; RMSE={RMSE_wen:.1f} dBR")
ax.plot(dBR_line, zr_dBZ(148.44, 1.30,  dBR_line), color=colors[7], ls='-',  lw=1.5, zorder=3,
        label=f"Janapati (2020): $Z=148.44R^{{1.33}}$; RMSE={RMSE_jan:.1f} dBR")
ax.plot(dBR_line, zr_dBZ(237.58, 1.43,  dBR_line), color=colors[8], ls='-',  lw=1.5, zorder=3,
        label=f"Bao (2020): $Z=237.58R^{{1.43}}$; RMSE={RMSE_bao:.1f} dBR")
# Axes
dBR_ticks = [-10, 0, 10, 20]
R_labels  = ['0.1', '1', '10', '100']
ax.set_xticks(dBR_ticks)
ax.set_xticklabels([f"{dbr}\n({r} mm h$^{{-1}}$)" for dbr, r in
                    zip(dBR_ticks, R_labels)], fontsize=9.5)
ax.set_xlim(-10, 20)
ax.set_ylim(10, 80)
ax.set_xlabel(r'Rain Rate, $10\log_{10}R$ (dBR)', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Radar Reflectivity, $10\log_{10}Z$ (dBZ)', fontsize=12, fontweight='bold')
ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=10)
ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)
ax.legend(loc='upper left', fontsize=11, frameon=False, ncol=1,
          title='RMSE in $10\log_{10}(R)$ (dBR)', title_fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "Fig8.png"), bbox_inches="tight")
plt.show()