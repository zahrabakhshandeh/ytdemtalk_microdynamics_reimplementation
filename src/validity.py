"""Frame validity, recomputed at feature time from the raw landmark cache.

A frame is valid if a face was found, IOD >= min_iod_px, and every landmark that the six channels or the PnP pose
actually use is inside the frame. Landmarks used only by optional components (jaw: 2 + chin 152; mouth width:
78, 308) do NOT invalidate the frame; that component alone becomes NaN when its landmarks are outside.

Use load_frames() everywhere frames are read for features / reports, so there is one definition.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import extract as X

# landmarks per component; a component is NaN in frames where any of its landmarks is outside the frame
COMPONENT_LMK = {
    "jaw": [2, 152],
    "mouth_w": [78, 308],
}


PNP_EDGE = [234, 454, 127, 356]             # temple / cheek-edge PnP points (may be outside the frame)


def gate_landmarks(cfg=None):
    """Landmarks whose presence inside the frame makes a frame valid (six channels + PnP points)."""
    eyes = X.R_EYE + X.L_EYE                 # EAR, eyelid stability; eye corners also define gaze and IOD
    brows = X.R_BROW + X.L_BROW              # brow asymmetry
    iris = [468, 473]                        # gaze: iris centres only
    lips = [13, 14]                          # mouth channel = inner-lip aperture
    pnp = X.PNP_IDX
    if cfg is None or cfg.get("quality", {}).get("pnp_gate_points", "central15") == "central15":
        pnp = [i for i in X.PNP_IDX if i not in PNP_EDGE]
    return sorted(set(eyes + brows + iris + lips + pnp))


def outside_mask(raw):
    """(T, 478) bool: landmark outside the original frame (NaN landmarks -> False)."""
    L = X.landmarks_px(raw)
    W, H = int(raw["width"]), int(raw["height"])
    with np.errstate(invalid="ignore"):
        return (L[:, :, 0] < 0) | (L[:, :, 0] >= W) | (L[:, :, 1] < 0) | (L[:, :, 1] >= H)


def apply_gate(fr, raw, cfg):
    """Return a copy of fr with `valid` recomputed by the feature-time rule and component NaNs applied."""
    fr = fr.copy()
    out = outside_mask(raw)
    found = fr["face_found"].to_numpy(bool)
    gate = gate_landmarks(cfg)
    with np.errstate(invalid="ignore"):
        big = fr["iod_px"].to_numpy(float) >= cfg["extraction"]["min_iod_px"]
    fr["valid_extract"] = fr["valid"].astype(bool)
    fr["gate_inside"] = found & ~out[:, gate].any(1)
    fr["valid"] = fr["gate_inside"] & big
    for comp, idx in COMPONENT_LMK.items():
        bad = out[:, idx].any(1)
        fr[f"{comp}_inside"] = found & ~bad
        fr.loc[bad, comp] = np.nan
    return fr


def load_frames(vid, cfg, root):
    cache = Path(root) / cfg["paths"]["cache_dir"]
    fr = pd.read_parquet(cache / "frames" / f"{vid}.parquet")
    raw = dict(np.load(cache / "raw" / f"{vid}.npz"))
    info = json.loads((cache / "frames" / f"{vid}.info.json").read_text())
    if cfg["quality"]["frame_gate"] == "used_landmarks":
        fr = apply_gate(fr, raw, cfg)
    else:                                    # "extract": the original all-USED-landmarks rule
        fr["valid_extract"] = fr["valid"].astype(bool)
    return fr, raw, info
