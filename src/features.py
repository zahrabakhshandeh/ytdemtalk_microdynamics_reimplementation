"""Clip features (paper section 4.1; every value not given in the paper is marked [ours] in config.yaml).

Per clip:
  1. frames with the feature-time gate, six channels (src.channels.compute_channels)
  2. cut zone: +-invalidate_margin_s around cut events of the chosen rule
  3. baseline = first baseline_s seconds of USABLE frames (valid and outside the cut zone)
  4. per-channel robust z-score with the baseline median / MAD (MAD x 1.4826)
  5. windows of length_s, hop hop_s, starting at the end of the baseline; a window is dropped if it touches the
     cut zone or has < min_valid_frac valid frames
  6. activity a_c = mean |first difference| (per second) of the normalised channel inside the window
  7. u = (a + eps) / sum(a + eps)  (6-part composition) -> D-1 pivot ILR coordinates
Clip vector phi(V) = [mean of ILR over windows, std of ILR over windows].
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import channels as CH
from src import cuts as C
from src import validity as V

CHANNELS = ["blink", "eyelid", "mouth", "brow", "gaze", "head"]


def pivot_ilr(u):
    """Pivot (balance) ILR coordinates of compositions u (n, D) -> (n, D-1).
    z_i = sqrt((D-i)/(D-i+1)) * ln(u_i / gmean(u_{i+1..D})), i = 1..D-1 (Hron/Filzmoser pivot coordinates)."""
    u = np.asarray(u, float)
    D = u.shape[1]
    lu = np.log(u)
    z = np.empty((u.shape[0], D - 1))
    for i in range(D - 1):
        k = D - i - 1                       # number of remaining parts
        z[:, i] = np.sqrt(k / (k + 1)) * (lu[:, i] - lu[:, i + 1:].mean(1))
    return z


def ilr_names(parts):
    return [f"ilr{i + 1}_{p}_vs_rest" for i, p in enumerate(parts[:-1])]


def robust_z(x, base_mask):
    b = x[base_mask]
    b = b[~np.isnan(b)]
    if len(b) < 3:
        return np.full_like(x, np.nan), np.nan, np.nan, "no_baseline"
    med = np.median(b)
    mad = 1.4826 * np.median(np.abs(b - med))
    how = "mad"
    if not mad > 0:
        mad, how = np.std(b), "std"
        if not mad > 0:
            mad, how = 1.0, "unit"
    return (x - med) / mad, med, mad, how


def clip_windows(vid, cfg, root, rule=None, brow_max_abs_yaw=None, variants=None):
    """Per-window table for one clip (only USABLE windows) + clip-level info dict.
    brow_max_abs_yaw : brow channel only on frames with |raw yaw| < this (deg); other frames are NaN
    for the brow channel only (baseline and window activity use the frontal brow frames).
    variants (optional): list of (baseline_norm, activity) pairs; then returns {pair: (win, clip)}.
      baseline_norm: robust_z (default) | subtract_median | none;  activity: mean_abs_diff (default) | std."""
    root = Path(root)
    fps = cfg["extraction"]["target_fps"]
    w = cfg["windows"]
    rule = rule or cfg["cuts"]["feature_rule"]
    fr, raw, info = V.load_frames(vid, cfg, root)
    cuts = json.loads((root / cfg["paths"]["cache_dir"] / "cuts" / f"{vid}.json").read_text())
    js = C.jump_scores(fr)
    types = [C.classify_event(ev, js, cfg) for ev in cuts["cuts"]]
    t = fr["t"].to_numpy()
    zone = C.cut_zone(t, cuts["cuts"], types, C.rule_types(rule), cfg["cuts"]["invalidate_margin_s"])
    chs = CH.compute_channels(fr, fps, cfg, mouth="lip_ap")
    valid = chs["valid"].to_numpy(bool)
    usable = valid & ~zone
    frontal_frac = np.nan
    if brow_max_abs_yaw is not None:
        nonfrontal = ~(np.abs(fr["yaw"].to_numpy(float)) < brow_max_abs_yaw)
        chs.loc[nonfrontal, "ch_brow"] = np.nan
        frontal_frac = float((~nonfrontal[valid]).mean()) if valid.any() else np.nan
    dur = len(fr) / fps

    need = int(round(w["baseline_s"] * fps))
    b_idx = int(np.searchsorted(np.cumsum(usable), need))
    clip = {"video_id": vid, "n_cut_events": len(types), "baseline_end_s": np.nan, "n_windows_total": 0,
            "n_windows_cut": 0, "n_windows_quality": 0, "n_windows": 0, "valid_frac": valid.mean(),
            "iod_px_median": float(np.nanmedian(fr["iod_px"])) if fr["iod_px"].notna().any() else np.nan,
            "n_blinks": chs.attrs["n_blinks"], "brow_frontal_frac_of_valid": frontal_frac,
            "n_windows_no_brow": 0}
    if b_idx >= len(fr):
        return (pd.DataFrame(), clip) if variants is None else {v: (pd.DataFrame(), dict(clip)) for v in variants}
    base = np.zeros(len(fr), bool)
    base[:b_idx + 1] = True
    base &= usable
    b_end = t[b_idx] + 1 / fps
    clip["baseline_end_s"] = b_end

    if variants is None:
        return _windows(vid, chs, t, valid, zone, base, b_end, dur, fps, w, dict(clip), "robust_z", "mean_abs_diff")
    return {v: _windows(vid, chs, t, valid, zone, base, b_end, dur, fps, w, dict(clip), *v) for v in variants}


def _windows(vid, chs, t, valid, zone, base, b_end, dur, fps, w, clip, baseline_norm, activity):
    Z = {}
    for c in CHANNELS:
        x = chs[f"ch_{c}"].to_numpy(float)
        Z[c], med, mad, how = robust_z(x, base)
        if baseline_norm == "subtract_median":
            Z[c], mad, how = x - med, 1.0, "none"
        elif baseline_norm == "none":
            Z[c], mad, how = x, 1.0, "none"
        elif baseline_norm != "robust_z":
            raise ValueError(baseline_norm)
        clip[f"base_med_{c}"], clip[f"base_scale_{c}"], clip[f"base_scale_how_{c}"] = med, mad, how
    if activity == "mean_abs_diff":
        dZ = {c: np.abs(np.diff(Z[c], prepend=np.nan)) * fps for c in CHANNELS}
    elif activity == "std":
        dZ = Z
    else:
        raise ValueError(activity)

    rows = []
    s0, k = b_end, 0
    while s0 + w["length_s"] <= dur + 1e-9:
        m = (t >= s0 - 1e-9) & (t < s0 + w["length_s"] - 1e-9)
        clip["n_windows_total"] += 1
        if zone[m].any():
            clip["n_windows_cut"] += 1
        elif valid[m].mean() < w["min_valid_frac"]:
            clip["n_windows_quality"] += 1
        else:
            r = {"video_id": vid, "win": k, "t0": round(s0, 4), "t1": round(s0 + w["length_s"], 4),
                 "valid_frac": valid[m].mean()}
            for c in CHANNELS:
                d = dZ[c][m]
                fin = np.isfinite(d)
                if activity == "mean_abs_diff":
                    r[f"act_{c}"] = np.nanmean(d) if fin.any() else np.nan
                else:
                    r[f"act_{c}"] = float(np.std(d[fin])) if fin.sum() >= 2 else np.nan
                r[f"nfin_{c}"] = int(fin.sum())
            rows.append(r)
        s0 += w["hop_s"]
        k += 1
    win = pd.DataFrame(rows)
    if len(win):
        clip["n_windows_no_brow"] = int(win["act_brow"].isna().sum())
        win = win.dropna(subset=[f"act_{c}" for c in CHANNELS])
    clip["n_windows"] = len(win)
    return win, clip


def add_composition(win, parts, eps):
    """Add u_* and ILR columns for the given channel subset (rebuilt composition for ablation)."""
    a = win[[f"act_{c}" for c in parts]].to_numpy(float) + eps
    u = a / a.sum(1, keepdims=True)
    z = pivot_ilr(u)
    out = win.copy()
    for j, c in enumerate(parts):
        out[f"u_{c}"] = u[:, j]
    for j, n in enumerate(ilr_names(parts)):
        out[n] = z[:, j]
    return out


def clip_features(win, parts, eps, min_windows):
    """phi(V) per clip: mean and std (ddof=0) of each ILR coordinate over the clip's windows."""
    w = add_composition(win, parts, eps)
    names = ilr_names(parts)
    g = w.groupby("video_id")
    n = g.size().rename("n_windows_used")
    mean = g[names].mean().add_prefix("mean_")
    std = g[names].std(ddof=0).add_prefix("std_")
    F = pd.concat([n, mean, std], axis=1)
    F = F[F.n_windows_used >= min_windows]
    return F, [f"mean_{c}" for c in names] + [f"std_{c}" for c in names]
