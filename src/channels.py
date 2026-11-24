"""Smoothing and the six micro-dynamic channels on the fixed-fps grid (cache/frames/<id>.parquet)."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _segments(mask):
    """Start/stop indices of contiguous True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def smooth(x, fps, cfg):
    """Savitzky-Golay then EMA, both defined in seconds, per contiguous non-NaN segment."""
    s = cfg["smoothing"]
    if s.get("savgol_window_frames") is not None:
        win = int(s["savgol_window_frames"])
        if win <= s["savgol_poly"] + 1:
            win = 10 ** 9
    else:
        win = int(round(s["savgol_window_s"] * fps))
        win = win + 1 if win % 2 == 0 else win
        win = max(win, s["savgol_poly"] + 3 if (s["savgol_poly"] + 3) % 2 else s["savgol_poly"] + 2)
    alpha = 1 - np.exp(-1.0 / (fps * s["ema_tau_s"]))
    y = np.full_like(x, np.nan, dtype=float)
    for a, b in _segments(~np.isnan(x)):
        seg = x[a:b].astype(float)
        if len(seg) >= win:
            seg = savgol_filter(seg, win, s["savgol_poly"])
        e = np.empty_like(seg)
        e[0] = seg[0]
        for i in range(1, len(seg)):
            e[i] = alpha * seg[i] + (1 - alpha) * e[i - 1]
        y[a:b] = e
    return y


def blink_mask(ear_raw, cfg, fps):
    """Blink = run of raw EAR < rel_threshold x clip median lasting <= blink_max_s; dilated by k frames.
    Longer low-EAR runs (looking down, smiling, squinting) are returned separately as 'narrow'."""
    c = cfg["channels"]
    med = np.nanmedian(ear_raw)
    thr = c["blink_rel_threshold"] * med
    low = np.nan_to_num(ear_raw, nan=np.inf) < thr
    blink = np.zeros_like(low)
    narrow = np.zeros_like(low)
    max_len = int(np.floor(c["blink_max_s"] * fps))
    for a, b in _segments(low):
        (blink if b - a <= max_len else narrow)[a:b] = True
    n_blinks = int(sum(1 for a, b in _segments(blink)))
    k = c["blink_dilate_frames"]
    if k:
        blink = np.convolve(blink.astype(int), np.ones(2 * k + 1, int), mode="same") > 0
    return blink, narrow, thr, n_blinks


def highpass(x, fps, window_s):
    lo = pd.Series(x).rolling(int(round(window_s * fps)) | 1, center=True, min_periods=3).mean().to_numpy()
    return x - lo


def compute_channels(fr, fps, cfg, mouth="lip_ap"):
    """
    Duplicate frames (ffmpeg fps filter on <15 fps sources) are treated as missing and interpolated over.
    Returns smoothed components plus ch_* columns:
      ch_blink  = mean EAR (L, R)                    ch_eyelid = |dEAR/dt| (1/s), blink frames excluded
      ch_mouth  = inner-lip aperture / IOD (variant: mouth="jaw")
      ch_brow   = |brow_L - brow_R| / IOD
      ch_gaze   = speed of (gaze_x, gaze_y), eye-widths/s (gaze = mean of both eyes)
      ch_head   = norm of angular velocity of (yaw, pitch, roll), deg/s, after removing slow drift (high-pass)
    """
    fr = fr.copy()
    num = fr.select_dtypes("number").columns.drop(["frame", "t"], errors="ignore")
    if "dup" in fr and fr["dup"].any():
        fr.loc[fr["dup"].to_numpy(bool), num] = np.nan
        fr[num] = fr[num].interpolate(limit=2, limit_area="inside")
    valid = fr["valid"].to_numpy(bool)
    fr.loc[~valid, num] = np.nan
    fr["ear"] = (fr["ear_l"] + fr["ear_r"]) / 2
    fr["brow_asym"] = (fr["brow_l"] - fr["brow_r"]).abs()
    fr["gaze_x"] = (fr["gaze_x_l"] + fr["gaze_x_r"]) / 2
    fr["gaze_y"] = (fr["gaze_y_l"] + fr["gaze_y_r"]) / 2

    out = pd.DataFrame({"t": fr["t"], "valid": valid})
    for c in ["ear", "lip_ap", "jaw", "brow_l", "brow_r", "brow_asym", "gaze_x", "gaze_y", "yaw", "pitch", "roll"]:
        out[c + "_s"] = smooth(fr[c].to_numpy(float), fps, cfg)
    blink, narrow, thr, n_blinks = blink_mask(fr["ear"].to_numpy(float), cfg, fps)
    out["blink"] = blink & valid
    out["eye_narrow"] = narrow
    out.attrs["blink_thr"] = thr
    out.attrs["n_blinks"] = n_blinks
    d = lambda x: np.diff(x, prepend=np.nan) * fps
    dear = np.abs(d(out["ear_s"].to_numpy()))
    dear[out["blink"].to_numpy()] = np.nan
    out["eyelid_dear"] = dear
    hw = cfg["channels"]["head_highpass_window_s"]
    for a in ["yaw", "pitch", "roll"]:
        out[a + "_hp"] = highpass(out[a + "_s"].to_numpy(), fps, hw)

    out["ch_blink"] = out["ear_s"]
    out["ch_eyelid"] = out["eyelid_dear"]
    out["ch_mouth"] = out[f"{mouth}_s"]
    out["ch_brow"] = out["brow_asym_s"]
    out["ch_gaze"] = np.hypot(d(out["gaze_x_s"].to_numpy()), d(out["gaze_y_s"].to_numpy()))
    out["ch_head"] = np.sqrt(sum(d(out[a + "_hp"].to_numpy()) ** 2 for a in ["yaw", "pitch", "roll"]))
    return out
