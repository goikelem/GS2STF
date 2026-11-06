"""Pure NumPy/SciPy kernels for SPI, SPEI, PET, classification and event statistics.

Origin: index_kernels.py."""
from __future__ import annotations

import calendar
import warnings

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc, ndtri


CLASS_NAMES = {
    1: "Extremely Dry", 2: "Severely Dry", 3: "Moderately Dry", 4: "Near Normal",
    5: "Moderately Wet", 6: "Severely Wet", 7: "Extremely Wet",
}


def classify(z: np.ndarray) -> np.ndarray:
    """Classify standardized index values into codes 1..7 (0 = missing), uint8.

    Exact boundaries (no gaps, no overlaps)::

        Z >=  2.00            -> 7 Extremely Wet
        1.50 <= Z <  2.00     -> 6 Severely Wet
        1.00 <= Z <  1.50     -> 5 Moderately Wet
       -1.00 <  Z <  1.00     -> 4 Near Normal
       -1.50 <  Z <= -1.00    -> 3 Moderately Dry
       -2.00 <  Z <= -1.50    -> 2 Severely Dry
                Z <= -2.00    -> 1 Extremely Dry
    """
    z = np.asarray(z, dtype=float)
    out = np.select(
        [z >= 2.0, z >= 1.5, z >= 1.0, z > -1.0, z > -1.5, z > -2.0, z <= -2.0],
        [7, 6, 5, 4, 3, 2, 1],
        default=0,           # NaN falls through every condition
    )
    return out.astype(np.uint8)


def accumulate(x: np.ndarray, scale: int) -> np.ndarray:
    """Rolling sum over ``scale`` consecutive months (trailing window).

    The result at time t is the sum of x[t-scale+1 .. t].  It is NaN when any value in
    the window is missing (so the first ``scale-1`` steps are NaN by construction).
    """
    x = np.asarray(x, dtype=np.float64)
    if scale < 1:
        raise ValueError("scale must be >= 1")
    if scale == 1:
        return x.copy()
    valid = np.isfinite(x)
    c = np.cumsum(np.where(valid, x, 0.0), axis=0)
    n = np.cumsum(valid, axis=0)
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), c], axis=0)
    n = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=n.dtype), n], axis=0)
    out = np.full_like(x, np.nan)
    s = c[scale:] - c[:-scale]
    k = n[scale:] - n[:-scale]
    out[scale - 1:] = np.where(k == scale, s, np.nan)
    return out


def gamma_params_thom(x: np.ndarray, min_pos: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell gamma shape/scale (Thom 1958 maximum-likelihood approximation) and P(zero).

    ``x`` has shape (n, cells).  Returns (alpha, beta, q); alpha/beta are NaN where fewer
    than ``min_pos`` positive values exist or the sample is degenerate.
    """
    fin = np.isfinite(x)
    n = fin.sum(axis=0)
    pos = fin & (x > 0)
    npos = pos.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = np.where(pos, x, 1.0)
        mean_pos = np.where(pos, x, 0.0).sum(axis=0) / npos
        mean_log = np.where(pos, np.log(xs), 0.0).sum(axis=0) / npos
        a = np.log(mean_pos) - mean_log
        alpha = (1.0 + np.sqrt(1.0 + 4.0 * a / 3.0)) / (4.0 * a)
        beta = mean_pos / alpha
        q = (n - npos) / n
    bad = (npos < min_pos) | ~np.isfinite(alpha) | (a <= 0) | (alpha <= 0) | (beta <= 0)
    alpha = np.where(bad, np.nan, alpha)
    beta = np.where(bad, np.nan, beta)
    q = np.where(n > 0, q, np.nan)
    return alpha, beta, q


def spi_from_accumulated(acc: np.ndarray, months: np.ndarray, ref_mask: np.ndarray,
                         min_pos: int = 10, min_ref: int = 20, clip: float = 3.09) -> np.ndarray:
    """SPI from accumulated precipitation ``acc`` (time, cells).

    A gamma distribution is fitted separately for each cell and calendar month on the
    reference period.  Zeros are handled with the mixed distribution
    ``H(x) = q + (1 - q) G(x)`` (McKee et al. 1993) and SPI = Phi^-1(H).
    """
    out = np.full(acc.shape, np.nan)
    p_lo = float(np.clip(1 - _norm_cdf(clip), 1e-12, 0.5))
    for m in range(1, 13):
        t_all = months == m
        t_ref = t_all & ref_mask
        ref = acc[t_ref]
        if ref.shape[0] == 0:
            continue
        alpha, beta, q = gamma_params_thom(ref, min_pos)
        nref = np.isfinite(ref).sum(axis=0)
        x = acc[t_all]
        with np.errstate(invalid="ignore", divide="ignore"):
            g = gammainc(alpha[None, :], np.where(x > 0, x, 0.0) / beta[None, :])
            h = q[None, :] + (1.0 - q[None, :]) * np.where(x > 0, g, 0.0)
            h = np.where(np.isfinite(x), h, np.nan)
            h = np.clip(h, p_lo, 1.0 - p_lo)
            z = ndtri(h)
        z[:, nref < min_ref] = np.nan
        out[t_all] = z
    return out


def _norm_cdf(z: float) -> float:
    from scipy.special import ndtr
    return float(ndtr(z))


def loglogistic_params_pwm(x: np.ndarray, min_n: int = 20):
    """Fit log-logistic (alpha scale, beta shape, gamma origin) per cell by probability-
    weighted moments (Vicente-Serrano et al. 2010).  ``x`` has shape (n, cells)."""
    xs = np.sort(x, axis=0)                       # NaN sorted to the end
    n = np.isfinite(x).sum(axis=0).astype(float)
    j = np.arange(1, x.shape[0] + 1, dtype=float)[:, None]
    valid = j <= n[None, :]
    xv = np.where(valid, xs, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = xv.sum(axis=0) / n
        b1 = (np.where(valid, (j - 1) / (n - 1), 0.0) * xv).sum(axis=0) / n
        b2 = (np.where(valid, (j - 1) * (j - 2) / ((n - 1) * (n - 2)), 0.0) * xv).sum(axis=0) / n
        w0, w1, w2 = b0, b0 - b1, b0 - 2 * b1 + b2
        beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
        g1 = gamma_fn(1 + 1 / beta) * gamma_fn(1 - 1 / beta)
        alpha = (w0 - 2 * w1) * beta / g1
        gam = w0 - alpha * g1
    bad = (n < min_n) | ~np.isfinite(beta) | (beta <= 1) | ~np.isfinite(alpha) | (alpha <= 0)
    return (np.where(bad, np.nan, alpha), np.where(bad, np.nan, beta), np.where(bad, np.nan, gam))


def spei_from_accumulated(acc: np.ndarray, months: np.ndarray, ref_mask: np.ndarray,
                          min_ref: int = 20, clip: float = 3.09, normal_fallback: bool = True) -> np.ndarray:
    """SPEI from accumulated climatic water balance (P - PET), log-logistic per cell/month.

    The log-logistic cannot represent samples with non-positive L-skewness (common for
    12-month sums, which are close to symmetric).  With ``normal_fallback`` such cell-months
    are standardized with a normal distribution fitted to the same reference sample instead
    of being dropped; ``spei_fit_failure_fraction`` reports how often this happens."""
    out = np.full(acc.shape, np.nan)
    p_lo = 1 - _norm_cdf(clip)
    for m in range(1, 13):
        t_all = months == m
        t_ref = t_all & ref_mask
        if not t_ref.any():
            continue
        alpha, beta, gam = loglogistic_params_pwm(acc[t_ref], min_ref)
        x = acc[t_all]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            d = x - gam[None, :]
            f = np.where(d > 0, 1.0 / (1.0 + (alpha[None, :] / d) ** beta[None, :]), 0.0)
            f = np.where(np.isfinite(x) & np.isfinite(alpha)[None, :], f, np.nan)
            f = np.clip(f, p_lo, 1.0 - p_lo)
            z = ndtri(f)
            if normal_fallback:
                ref = acc[t_ref]
                nn = np.isfinite(ref).sum(axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN (masked) cells
                    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0, ddof=1)
                zn = np.clip((x - mu[None, :]) / sd[None, :], -clip, clip)
                use = (~np.isfinite(alpha)) & (nn >= min_ref) & (sd > 0)
                z = np.where(use[None, :] & np.isfinite(x), zn, z)
        out[t_all] = z
    return out


def spei_fit_failure_fraction(acc: np.ndarray, months: np.ndarray, ref_mask: np.ndarray,
                              min_ref: int = 20) -> np.ndarray:
    """Per-cell fraction of calendar months (0..1) whose log-logistic fit failed.

    A fit fails when the reference sample is too short, or its L-skewness is not positive
    (the log-logistic cannot represent negatively skewed samples).  Failed fits give NaN
    SPEI; this diagnostic lets the workflow report them instead of discarding silently."""
    failed = np.zeros(acc.shape[1])
    nmon = 0
    for m in range(1, 13):
        t_ref = (months == m) & ref_mask
        if not t_ref.any():
            continue
        nmon += 1
        a, _, _ = loglogistic_params_pwm(acc[t_ref], min_ref)
        has_data = np.isfinite(acc[t_ref]).any(axis=0)
        failed += (~np.isfinite(a)) & has_data
    return failed / max(nmon, 1)


def extraterrestrial_radiation(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Daily extraterrestrial radiation Ra in mm/day of evaporation equivalent (FAO-56 eq. 21).

    Returns shape (len(doy), len(lat))."""
    phi = np.deg2rad(np.asarray(lat_deg, dtype=float))[None, :]
    j = np.asarray(doy, dtype=float)[:, None]
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * j / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta)
                                            + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return 0.408 * np.maximum(ra, 0.0)


def monthly_mean_ra(lat_deg: np.ndarray) -> np.ndarray:
    """Mean daily Ra for each calendar month (non-leap year): shape (12, nlat)."""
    out = np.empty((12, len(lat_deg)))
    start = 0
    for m in range(12):
        nd = calendar.monthrange(2001, m + 1)[1]
        doy = np.arange(start + 1, start + nd + 1)
        out[m] = extraterrestrial_radiation(lat_deg, doy).mean(axis=0)
        start += nd
    return out


def hargreaves_pet(tmin: np.ndarray, tmax: np.ndarray, ra_mm_day: np.ndarray,
                   days_in_month: np.ndarray) -> np.ndarray:
    """Monthly PET (mm/month), Hargreaves-Samani (1985).

    tmin, tmax: degC, shape (time, cells).  ra_mm_day: (time, cells) or broadcastable,
    mean daily Ra of each month.  days_in_month: (time,).

        PET_day = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)
    """
    tmean = 0.5 * (tmin + tmax)
    rng = np.maximum(tmax - tmin, 0.0)
    pet_day = 0.0023 * ra_mm_day * (tmean + 17.8) * np.sqrt(rng)
    return np.maximum(pet_day, 0.0) * np.asarray(days_in_month, dtype=float)[:, None]


def cell_area_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Grid-cell areas (km^2), shape (nlat, nlon), for a regular lat/lon grid on a sphere."""
    R = 6371.0088
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat = np.abs(np.gradient(lat)) if lat.size > 1 else np.array([1.0])
    dlon = np.abs(np.gradient(lon)) if lon.size > 1 else np.array([1.0])
    lat_n = np.deg2rad(np.clip(lat + dlat / 2, -90, 90))
    lat_s = np.deg2rad(np.clip(lat - dlat / 2, -90, 90))
    a_lat = np.abs(np.sin(lat_n) - np.sin(lat_s))              # (nlat,)
    return (R ** 2) * np.deg2rad(dlon)[None, :] * a_lat[:, None]


def spell_statistics(z: np.ndarray, threshold: float, below: bool = True) -> dict[str, np.ndarray]:
    """Per-cell spell statistics for index series ``z`` (time, cells).

    A spell is a run of consecutive months with z <= threshold (``below``) or
    z >= threshold.  Severity of a spell = sum of |z| over the spell.

    Returns arrays over cells: frequency_pct (share of valid months in a spell),
    n_events, mean_duration, max_duration, mean_severity, max_severity, mean_intensity.
    """
    z = np.asarray(z, dtype=float)
    nt, nc = z.shape
    valid = np.isfinite(z)
    hit = (z <= threshold) if below else (z >= threshold)
    hit &= valid
    run = np.zeros(nc, dtype=np.int32)
    sev = np.zeros(nc)
    n_ev = np.zeros(nc, dtype=np.int32)
    tot_dur = np.zeros(nc)
    max_dur = np.zeros(nc)
    tot_sev = np.zeros(nc)
    max_sev = np.zeros(nc)

    def close(mask):
        n_ev[mask] += 1
        tot_dur[mask] += run[mask]
        tot_sev[mask] += sev[mask]
        max_dur[mask] = np.maximum(max_dur[mask], run[mask])
        max_sev[mask] = np.maximum(max_sev[mask], sev[mask])
        run[mask] = 0
        sev[mask] = 0.0

    for t in range(nt):
        h = hit[t]
        ended = (~h) & (run > 0)
        if ended.any():
            close(ended)
        run[h] += 1
        sev[h] += np.abs(z[t, h])
    close(run > 0)

    nvalid = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = 100.0 * hit.sum(axis=0) / nvalid
        mean_dur = np.where(n_ev > 0, tot_dur / n_ev, np.nan)
        mean_sev = np.where(n_ev > 0, tot_sev / n_ev, np.nan)
        mean_int = np.where(tot_dur > 0, tot_sev / tot_dur, np.nan)
    nan = nvalid == 0
    out = {"frequency_pct": freq, "n_events": n_ev.astype(float), "mean_duration": mean_dur,
           "max_duration": max_dur, "mean_severity": mean_sev, "max_severity": max_sev,
           "mean_intensity": mean_int}
    for k, v in out.items():
        out[k] = np.where(nan, np.nan, v)
    return out
