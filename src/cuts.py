"""Cut / splice detection: PySceneDetect content detector + landmark jump spikes."""
import numpy as np
import pandas as pd


def scenedetect_cuts(video_path, cfg):
    from scenedetect import ContentDetector, detect
    c = cfg["cuts"]
    scenes = detect(str(video_path), ContentDetector(threshold=c["scenedetect_threshold"],
                                                     min_scene_len=f"{c['scenedetect_min_scene_s']}s"),
                    show_progress=False)
    return [float(s[0].seconds) for s in scenes[1:]]


def _robust_z(x):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else np.nan)


def jump_scores(fr):
    """Per-step jump magnitudes on the fixed grid (raw, unsmoothed). Index i = change from i-1 to i."""
    iod = fr["iod_px"].to_numpy(float)
    mx, my = fr["eye_mid_x"].to_numpy(float), fr["eye_mid_y"].to_numpy(float)
    ref = np.fmin(iod, np.roll(iod, 1))
    d_center = np.hypot(np.diff(mx, prepend=np.nan), np.diff(my, prepend=np.nan)) / ref
    d_scale = np.abs(np.diff(np.log(iod), prepend=np.nan))
    pose = fr[["yaw", "pitch", "roll"]].to_numpy(float)
    d_pose = np.linalg.norm(np.diff(pose, axis=0, prepend=np.full((1, 3), np.nan)), axis=1)
    return pd.DataFrame({"t": fr["t"], "d_center": d_center, "d_scale": d_scale, "d_pose": d_pose})


def landmark_jump_cuts(fr, cfg):
    c = cfg["cuts"]
    js = jump_scores(fr)
    flags = np.zeros(len(js), bool)
    for col, thr in [("d_center", c["jump_center_iod"]), ("d_scale", c["jump_log_scale"]), ("d_pose", c["jump_pose_deg"])]:
        x = js[col].to_numpy()
        flags |= np.nan_to_num((x > thr) & (_robust_z(x) > c["jump_min_robust_z"]), nan=False).astype(bool)
    return js["t"].to_numpy()[flags].tolist(), js


def merge_events(events, within):
    """events: list of (t, source). Merge events closer than `within` seconds."""
    out = []
    for t, s in sorted(events):
        if out and t - out[-1]["t_last"] <= within:
            out[-1]["t_last"] = t
            out[-1]["sources"].add(s)
        else:
            out.append({"t": t, "t_last": t, "sources": {s}})
    return [{"t": round(e["t"], 3), "t_end": round(e["t_last"], 3), "sources": "+".join(sorted(e["sources"]))} for e in out]


SPLICE_TYPES = {"scenedetect", "position_scale_jump"}
ALL_TYPES = {"scenedetect", "position_scale_jump", "pose_only_jump"}

def classify_event(ev, js, cfg):
    """Which criterion fired inside the event span: scenedetect / position_scale_jump / pose_only_jump."""
    if "scenedetect" in ev["sources"]:
        return "scenedetect"
    c = cfg["cuts"]
    m = (js["t"] >= ev["t"] - 1e-6) & (js["t"] <= ev["t_end"] + 1e-6)
    pos = ((js.loc[m, "d_center"] > c["jump_center_iod"]) | (js.loc[m, "d_scale"] > c["jump_log_scale"])).any()
    return "position_scale_jump" if pos else "pose_only_jump"


def cut_zone(t, events, types, keep, margin):
    """Bool mask of frames within +-margin s of any event whose type is in `keep`."""
    z = np.zeros(len(t), bool)
    for ev, ty in zip(events, types):
        if ty in keep:
            z |= (t >= ev["t"] - margin) & (t <= ev["t_end"] + margin)
    return z


def rule_types(rule):
    return {"splices_only": SPLICE_TYPES, "all_events": ALL_TYPES, "none": set()}[rule]
