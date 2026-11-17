"""Variant A -- empirical comparable-quantile forecast for GTA VI Twitch CCU.

Method (short): for each forecast hour h = 2..24, take every comparable
launch's release-day curve from inside the case file, fit a weighted
log-log regression of that hour's viewers on the launch's own hour-1
print (weights = similarity to the case: hour-1 scale, viewers-per-channel,
unlock pattern, pre-release level), and report the weighted empirical
quantiles of the regression residuals around the fitted centre for the
case's hour-1 print.  All fitting is done at run time from the comparables
inside each case file; nothing is tuned to any particular case.

Stdlib only.  Deterministic: no randomness, no dict/set iteration order
dependence (inputs are sorted before use).
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Coarse, principled hyperparameters (see NOTES.md).
BANDWIDTH_LOG_P1 = 1.5    # Gaussian kernel width on |ln p1_case - ln p1_comp|
BANDWIDTH_LOG_VPC = 1.0   # kernel width on log viewers-per-channel
BANDWIDTH_LOG_PRE = 1.0   # kernel width on log(pre-release mean / hour-1)
UNLOCK_MIDNIGHT = 1.0     # comparable unlocks at rolling local midnight
UNLOCK_SIMULT = 0.35      # single simultaneous global unlock (different shape)
UNLOCK_OTHER = 0.75       # unlock text missing / unverified
SLOPE_MIN = 0.0
SLOPE_MAX = 1.2
SPREAD_FLOOR = 0.35       # min log distance median -> q05 / q95
MIN_POINTS = 5            # points needed for the regression path


def parse_hour(text):
    """ISO-8601 UTC timestamp -> epoch seconds (stdlib, no dateutil)."""
    if not text:
        return None
    try:
        s = text.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def safe_log(x):
    return math.log(x) if x and x > 0 else None


def kernel(dist, width):
    z = dist / width
    return math.exp(-0.5 * z * z)


def unlock_factor(text):
    low = (text or "").lower()
    if "midnight" in low:
        return UNLOCK_MIDNIGHT
    if "simultaneous" in low or "single global" in low:
        return UNLOCK_SIMULT
    return UNLOCK_OTHER


def is_launch(comp):
    """True only for a real launch curve.  The GTA VI category's own
    pre-release history is explicitly marked 'Not a launch' in its unlock
    text and its name says 'before release'; it has no release day yet,
    so it must never be treated as a comparable launch curve."""
    name = (comp.get("name") or "").lower()
    unlock = comp.get("unlock") or ""
    if "before release" in name:
        return False
    if unlock.startswith("Not a launch"):
        return False
    return True


def extract_curve(comp):
    """Return (by_offset, p1, vpc, pre_ratio) for one comparable, or None.

    by_offset maps hour offset from the comparable's day_start (0 = hour 1)
    to (viewers, channels).  Pre-release offsets -6..-1 are kept for the
    pre-release similarity weight.
    """
    day_start = parse_hour(comp.get("day_start"))
    if day_start is None:
        return None
    by_offset = {}
    for point in comp.get("hourly") or []:
        t = parse_hour(point.get("from"))
        if t is None:
            continue
        off = int((t - day_start) // 3600)
        if -6 <= off < 24:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                by_offset[off] = (float(viewers), point.get("channels"))
    if 0 not in by_offset:
        return None
    p1 = by_offset[0][0]
    if p1 <= 0:
        return None
    channels = by_offset[0][1]
    vpc = None
    if isinstance(channels, (int, float)) and channels > 0:
        vpc = p1 / float(channels)
    pre_vals = [by_offset[o][0] for o in range(-6, 0) if o in by_offset]
    pre_ratio = None
    if len(pre_vals) >= 3:
        pre_ratio = (sum(pre_vals) / len(pre_vals)) / p1
    return by_offset, p1, vpc, pre_ratio


def case_pre_ratio(case, p1):
    """Mean viewers over the 6 whole hours before the case's day, / p1."""
    day_start = parse_hour(case.get("day_start"))
    if day_start is None or p1 <= 0:
        return None
    vals = []
    for row in case.get("history_hourly") or []:
        t = parse_hour(row.get("from"))
        if t is None:
            continue
        off = int((t - day_start) // 3600)
        if -6 <= off < 0:
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v > 0:
                vals.append(float(v))
    if len(vals) >= 3:
        return (sum(vals) / len(vals)) / p1
    return None


def weighted_quantile(pairs, tau):
    """Interpolated weighted empirical quantile of [(value, weight)]."""
    pairs = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        return pairs[len(pairs) // 2][0] if pairs else 0.0
    target = tau * total
    acc = 0.0
    for i, (value, w) in enumerate(pairs):
        prev = acc
        acc += w
        if acc >= target:
            if i == 0 or w <= 0:
                return value
            frac = (target - prev) / w
            return pairs[i - 1][0] * (1.0 - frac) + value * frac
    return pairs[-1][0]


def forecast_case(case):
    known = case.get("known") or []
    p1 = 1.0
    if known:
        v = known[0].get("viewers")
        if isinstance(v, (int, float)) and v > 0:
            p1 = float(v)
    log_p1 = math.log(p1)

    channels_case = known[0].get("channels") if known else None
    vpc_case = None
    if isinstance(channels_case, (int, float)) and channels_case > 0:
        vpc_case = p1 / float(channels_case)
    pre_case = case_pre_ratio(case, p1)

    # Build the comparable pool once, with per-comparable weights.
    pool = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name") or ""):
        if not is_launch(comp):
            continue
        got = extract_curve(comp)
        if got is None:
            continue
        by_offset, p1c, vpc_c, pre_c = got
        w = kernel(math.log(p1) - math.log(p1c), BANDWIDTH_LOG_P1)
        w *= unlock_factor(comp.get("unlock"))
        if vpc_case is not None and vpc_c:
            w *= kernel(math.log(vpc_case) - math.log(vpc_c), BANDWIDTH_LOG_VPC)
        if pre_case is not None and pre_c:
            w *= kernel(math.log(pre_case) - math.log(pre_c), BANDWIDTH_LOG_PRE)
        if w > 0:
            pool.append((by_offset, p1c, w))

    rows = []
    for hour in range(2, 25):
        off = hour - 1
        points = []  # (log p1_comp, log viewers_h, weight)
        for by_offset, p1c, w in pool:
            if off in by_offset:
                points.append((math.log(p1c), math.log(by_offset[off][0]), w))
        quantiles = None
        if len(points) >= MIN_POINTS:
            sw = sum(w for _, _, w in points)
            mx = sum(w * x for x, _, w in points) / sw
            my = sum(w * y for _, y, w in points) / sw
            var = sum(w * (x - mx) ** 2 for x, _, w in points) / sw
            cov = sum(w * (x - mx) * (y - my) for x, y, w in points) / sw
            slope = cov / var if var > 0 else 1.0
            slope = min(SLOPE_MAX, max(SLOPE_MIN, slope))
            intercept = my - slope * mx
            centre = intercept + slope * log_p1
            residuals = [(y - (intercept + slope * x), w) for x, y, w in points]
            logs = [centre + weighted_quantile(residuals, tau) for tau in LEVELS]
            quantiles = logs
        if quantiles is None:
            # Fallback: weighted empirical quantiles of the log-ratio to p1,
            # or flat persistence if even that is unavailable.
            pairs = []
            for by_offset, p1c, w in pool:
                if off in by_offset:
                    pairs.append((math.log(by_offset[off][0] / p1c), w))
            if pairs:
                quantiles = [log_p1 + weighted_quantile(pairs, tau) for tau in LEVELS]
            else:
                z = (-1.645, -0.674, 0.0, 0.674, 1.645)
                quantiles = [log_p1 + zi * 0.15 * math.sqrt(hour - 1) for zi in z]
        # Enforce a minimum honest spread around the median.
        med = quantiles[2]
        quantiles[0] = min(quantiles[0], med - SPREAD_FLOOR)
        quantiles[4] = max(quantiles[4], med + SPREAD_FLOOR)
        quantiles = sorted(quantiles)
        out_q = []
        prev = 0.0
        for q in quantiles:
            val = math.exp(q)
            if not math.isfinite(val):
                val = p1
            val = min(1e9, max(1.0, val))
            val = max(val, prev)
            out_q.append(round(val, 2))
            prev = val
        rows.append({"hour": hour, "q": out_q})
    return {"hours": rows}


def fallback_forecast(case):
    """Flat-persistence forecast, used only if forecast_case fails on a
    malformed case: median = the case's hour-1 print (1.0 if unreadable),
    spread widening as 0.15 * sqrt(h - 1) in log space.  Same output
    shape and clamps as forecast_case."""
    p1 = 1.0
    try:
        known = case.get("known") or []
        v = known[0].get("viewers") if known else None
        if isinstance(v, (int, float)) and v > 0:
            p1 = float(v)
    except Exception:
        p1 = 1.0
    z = (-1.645, -0.674, 0.0, 0.674, 1.645)
    rows = []
    for hour in range(2, 25):
        out_q = []
        prev = 0.0
        for zi in z:
            val = p1 * math.exp(zi * 0.15 * math.sqrt(hour - 1))
            if not math.isfinite(val):
                val = p1
            val = min(1e9, max(1.0, val))
            val = max(val, prev)
            out_q.append(round(val, 2))
            prev = val
        rows.append({"hour": hour, "q": out_q})
    return {"hours": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(in_dir.glob("*.in")):
        case = json.loads(in_path.read_text(encoding="utf-8"))
        try:
            result = forecast_case(case)
        except Exception:
            result = fallback_forecast(case)
        out_path = out_dir / (in_path.stem + ".out")
        out_path.write_text(
            json.dumps(result, separators=(",", ":"), sort_keys=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
