"""
dsd_common.py
=============
Shared by Fig2, Fig4, Fig5, Fig6, Fig7 and Fig8, so that EVERY figure uses the
same Parsivel loading, quality control and DSD formulas. 

Quality control (same as the Methods):
  1. sensor-reported 1-min rain rate >= RR_MIN
  2. keep diameter classes with centre in [D_MIN, D_MAX]
  3. keep drops within +/- VEL_FRAC of the density-corrected Atlas et al. (1973) fall speed
  4. keep spectra with >= MIN_DROPS drops remaining AFTER steps 2-3   (must be last)
Steps 1-3 can be applied in any order (the final set is identical); step 4 must come last.
"""
import os, csv
import datetime as dt
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
_candidates = [SCRIPT_DIR.parent / "Disdrometer" / "Kototabang"]   
DATA_ROOT = next((p for p in _candidates if p.is_dir()), _candidates[0])
OUT_DIR = SCRIPT_DIR / "Fig_outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
ALTITUDE_M = 865.0
UTC_OFFSET = dt.timedelta(hours=7)
T_START = dt.datetime(2025, 11, 25, 12, 0, 0) + UTC_OFFSET          # local time (UTC+7)
T_END   = dt.datetime(2025, 11, 28, 12, 0, 0) + UTC_OFFSET
RR_MIN, MIN_DROPS, D_MIN, D_MAX, VEL_FRAC = 0.1, 10, 0.312, 10.0, 0.60
CLASS_RATE = "derived"
RAIN_RATE_METHOD = "atlas"

# ── Parsivel classes ───────────────────────────────────────────────────────────
D_bins = np.array([0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375, 1.625,
                   1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750, 5.500, 6.500, 7.500, 8.500,
                   9.500, 11.00, 13.00, 15.00, 17.00, 19.00, 21.50, 24.50])
V_bins = np.array([0.050, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750, 0.850, 0.950, 1.100, 1.300,
                   1.500, 1.700, 1.900, 2.200, 2.600, 3.000, 3.400, 3.800, 4.400, 5.200, 6.000, 6.800,
                   7.600, 8.800, 10.40, 12.00, 13.60, 15.20, 17.60, 20.80])
dD = np.array([0.125] * 10 + [0.25] * 5 + [0.5] * 5 + [1.0] * 5 + [2.0] * 5 + [3.0] * 2)
A_eff = (180.0 * (30.0 - D_bins / 2.0)) * 1e-6          # effective sampling area, m^2
T_INT = 60.0                                           # s


def density_factor(altitude_m=ALTITUDE_M):
    """Standard-atmosphere density and Atlas correction factor (rho_0/rho)^0.4 (constant, not event-specific)."""
    rho_0, T_0, L, R, g = 1.225, 288.15, 0.0065, 287.05, 9.80665
    T_z = T_0 - L * altitude_m
    rho_z = rho_0 * (T_z / T_0) ** ((g / (L * R)) - 1)
    return rho_z, (rho_0 / rho_z) ** 0.4


def v_atlas(d, altitude_m=ALTITUDE_M):
    """Density-corrected Atlas et al. (1973) terminal fall speed (m/s), D in mm."""
    return np.maximum((9.65 - 10.3 * np.exp(-0.6 * d)) * density_factor(altitude_m)[1], 0.0)


V_TERM = v_atlas(D_bins)
D_MASK = (D_bins >= D_MIN) & (D_bins <= D_MAX)
V_MASK = ((V_bins[:, None] >= (1 - VEL_FRAC) * V_TERM[None, :]) &
          (V_bins[:, None] <= (1 + VEL_FRAC) * V_TERM[None, :]))          # [v, d]


# ── Loading ────────────────────────────────────────────────────────────────────
def iter_csv_files(root=DATA_ROOT):
    for month in sorted(os.listdir(root)):
        mp = os.path.join(root, month)
        if not os.path.isdir(mp):
            continue
        for day in sorted(os.listdir(mp)):
            dp = os.path.join(mp, day)
            if not os.path.isdir(dp):
                continue
            for fn in sorted(os.listdir(dp)):
                if fn.endswith(".csv"):
                    yield os.path.join(dp, fn)


def load_spectra():
    """All readable 1-min records inside the time window: local times, sensor rain rate, spectra [n, v, d]."""
    times, rr, specs = [], [], []
    for path in iter_csv_files():
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                try:
                    t = dt.datetime.strptime(f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S")
                except Exception:
                    continue
                if not (T_START <= t <= T_END):
                    continue
                try:
                    r = float(row[2])
                    vals = [float(x) if x.strip() else 0.0 for x in row[15:1038]]
                    vals += [0.0] * (1024 - len(vals))
                    s = np.array(vals[:1024], dtype=float).reshape(32, 32)      # [v, d]
                except Exception:
                    continue
                times.append(t); rr.append(r); specs.append(s)
    return np.array(times, dtype=object), np.array(rr), np.array(specs)


def qc(specs, rr, frac=VEL_FRAC, min_drops=MIN_DROPS, rr_min=RR_MIN):
    """Master QC. Returns indices into the input arrays, cleaned spectra [n, v, d] and drops per spectrum."""
    idx = np.where(rr >= rr_min)[0]
    vmask = V_MASK if frac == VEL_FRAC else ((V_bins[:, None] >= (1 - frac) * V_TERM[None, :]) &
                                              (V_bins[:, None] <= (1 + frac) * V_TERM[None, :]))
    s = specs[idx] * D_MASK[None, None, :] * vmask[None, :, :]
    n = s.sum(axis=(1, 2))
    keep = n >= min_drops
    return idx[keep], s[keep], n[keep]


# ── DSD products ───────────────────────────────────────────────────────────────
def dsd(spec, nd_velocity="measured", rain_rate=None):
    """DSD quantities for QC'd spectra [n, v, d].

    nd_velocity : "measured" -> N(D) = sum_v n / (V_v A t dD)   (as in the manuscript)
                  "atlas"    -> N(D) = n_d / (V_Atlas(D) A t dD)
    rain_rate   : "atlas" (Eq. 3: Atlas fall speed with N(D)) or "consistent" (sum n D^3 / (A t), no fall speed)
    Units: N(D) mm^-1 m^-3, Dm mm, Nw mm^-1 m^-3, LWC g m^-3, Z = M6 mm^6 m^-3, R mm h^-1.
    """
    rain_rate = rain_rate or RAIN_RATE_METHOD
    counts = spec.sum(axis=1)                                           # [n, d]
    if nd_velocity == "measured":
        nd = (spec / (V_bins[None, :, None] * A_eff[None, None, :] * T_INT * dD[None, None, :])).sum(axis=1)
    else:
        vt = np.where(V_TERM > 0, V_TERM, np.inf)
        nd = counts / (vt[None, :] * A_eff[None, :] * T_INT * dD[None, :])
    nd = nd * D_MASK[None, :]
    M3, M4, M6 = ((nd * D_bins[None, :] ** k * dD[None, :]).sum(axis=1) for k in (3, 4, 6))
    if rain_rate == "atlas":
        r_bin = 6 * np.pi * 1e-4 * nd * V_TERM[None, :] * D_bins[None, :] ** 3 * dD[None, :]
    else:
        r_bin = 6 * np.pi * 1e-4 * counts * D_bins[None, :] ** 3 / (A_eff[None, :] * T_INT) * D_MASK[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        Dm = M4 / M3
        Nw = (4 ** 4 / 6.0) * M3 ** 5 / M4 ** 4
    return dict(nd=nd, counts=counts, M3=M3, M4=M4, M6=M6, Z=M6, Dm=Dm, Nw=Nw, LWC=np.pi / 6000.0 * M3,
                Nt=(nd * dD[None, :]).sum(axis=1), R_bin=r_bin, R=r_bin.sum(axis=1))


def mu_lambda(M3, M4, M6):
    """Three-moment (M3, M4, M6) gamma retrieval. Returns mu, Lambda (mm^-1), G; no range filtering here."""
    with np.errstate(divide="ignore", invalid="ignore"):
        G = M4 ** 3 / (M3 ** 2 * M6)
        mu = ((11 * G - 8) + np.sqrt(G * (G + 8))) / (2 * (1 - G))
        lam = (mu + 4) * M3 / M4
    return mu, lam, G


def get_qc_dataset(verbose=True):
    """Load, quality-control and compute the DSD once. Every figure script starts from this."""
    times, rr, specs = load_spectra()
    idx, spec, drops = qc(specs, rr)
    d = dsd(spec)
    out = dict(times_all=times, rr_all=rr, specs_all=specs, idx=idx, t=times[idx], rr_sensor=rr[idx],
               spec=spec, drops=drops, dsd=d, R=d["R"],
               R_class=d["R"] if CLASS_RATE == "derived" else rr[idx])
    if verbose:
        t_utc = out["t"] - UTC_OFFSET
        print(f"[dsd_common] records in window: {len(specs):,} | QC spectra: {len(idx):,} (manuscript: 1,452) | "
              f"drops: {int(drops.sum()):,}")
        print(f"[dsd_common] retained record: {t_utc.min():%Y-%m-%d %H:%M} to {t_utc.max():%Y-%m-%d %H:%M} UTC | "
              f"rain-rate classes by '{CLASS_RATE}' R, R computed with '{RAIN_RATE_METHOD}' method")
    return out


def block_idx(n, L, rng):
    """Indices for one moving-block-bootstrap resample of n time-ordered samples."""
    L = max(1, min(L, n))
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=nb)
    return (starts[:, None] + np.arange(L)).ravel()[:n]


def make_edges(c, lower=None):
    """Cell edges from cell centres (for pcolormesh)."""
    e = np.concatenate([[c[0] - (c[1] - c[0]) / 2], 0.5 * (c[:-1] + c[1:]), [c[-1] + (c[-1] - c[-2]) / 2]])
    if lower is not None:
        e[0] = max(lower, e[0])
    return e
