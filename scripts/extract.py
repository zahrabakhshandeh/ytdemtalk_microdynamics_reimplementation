"""Extract raw per-frame data + cuts. Resumable (skips videos whose outputs exist), shardable for parallel workers.

usage: python scripts/extract.py --ids ID [ID ...]
       python scripts/extract.py --all --shard 0 --nshards 2      (one worker per shard)
       --keep-f32 DIR : also save float32 landmarks to DIR (precision check only)
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cuts as C
from src import extract as X

TAG = ""


def log(msg):
    print(time.strftime("%H:%M:%S"), TAG, msg, flush=True)


def local_copy(src, dst):
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def border_box(vid, path, cfg):
    """(x, y, w, h) of the content area if the clip has a black border (content < max_content_frac), else None."""
    from scripts.detect_borders import detect
    p = ROOT / cfg["paths"]["cache_dir"] / "borders.csv"
    b = None
    try:
        d = pd.read_csv(p)
        d = d[d.video_id == vid]
        b = d.iloc[0].to_dict() if len(d) else None
    except Exception:
        b = None
    if b is None:
        b = detect(path)
    if b["content_frac"] >= cfg["extraction"]["border_crop"]["max_content_frac"]:
        return None
    return int(b["crop_x"]), int(b["crop_y"]), int(b["crop_w"]), int(b["crop_h"])


class DriveGone(RuntimeError):
    pass


def process(row, cfg, keep_f32=None):
    vid = row.video_id
    cache = ROOT / cfg["paths"]["cache_dir"]
    p_raw = cache / "raw" / f"{vid}.npz"
    p_frames = cache / "frames" / f"{vid}.parquet"
    p_info = cache / "frames" / f"{vid}.info.json"
    p_cuts = cache / "cuts" / f"{vid}.json"
    for p in [p_raw, p_frames, p_cuts]:
        p.parent.mkdir(parents=True, exist_ok=True)
    if p_raw.exists() and p_frames.exists() and p_cuts.exists() and p_info.exists():
        log(f"{vid}: done already, skip")
        return
    t0 = time.time()
    src = Path(cfg["paths"]["video_root"]) / row.video_path
    if not Path(cfg["paths"]["video_root"]).is_dir() or not cache.is_dir():
        raise DriveGone("video_root or cache dir not accessible (Drive unmounted?)")
    loc = local_copy(src, Path(cfg["paths"]["local_video_cache"]) / row.video_path)
    t_copy = time.time() - t0

    fr, raw, info = X.extract(loc, cfg, ROOT, log=log, keep_f32=keep_f32 is not None)
    bc = cfg["extraction"]["border_crop"]
    if info["face_found_frac"] < bc["fallback_if_face_found_below"]:
        crop = border_box(vid, loc, cfg)
        if crop is not None:
            log(f"{vid}: face found in {info['face_found_frac']:.1%} of frames on the full frame; "
                f"black border detected, redoing with crop {crop}")
            ff_full = info["face_found_frac"]
            fr, raw, info = X.extract(loc, cfg, ROOT, log=log, keep_f32=keep_f32 is not None, crop=crop)
            info["face_found_frac_full_frame"] = ff_full
    if keep_f32 is not None:
        Path(keep_f32).mkdir(parents=True, exist_ok=True)
        np.save(Path(keep_f32) / f"{vid}_f32.npy", raw.pop("lmk_px_f32_debug"))

    sd = fr["t"][fr["sd_cut"]].tolist()
    lj, _ = C.landmark_jump_cuts(fr, cfg)
    events = C.merge_events([(t, "scenedetect") for t in sd] + [(t, "landmark_jump") for t in lj],
                            cfg["cuts"]["merge_within_s"])
    X.atomic_write_parquet(fr, p_frames)
    X.atomic_savez(p_raw, **raw)
    rec = {"video_id": vid, "scenedetect": sd, "landmark_jump": lj, "cuts": events}
    tmp = str(p_cuts) + ".tmp"
    Path(tmp).write_text(json.dumps(rec, indent=1))
    os.replace(tmp, p_cuts)
    info.update(copy_s=round(t_copy, 1), total_s=round(time.time() - t0, 1),
                raw_mb=round(p_raw.stat().st_size / 1e6, 2), frames_mb=round(p_frames.stat().st_size / 1e6, 2),
                valid_frac=round(float(fr["valid"].mean()), 4))
    tmp = str(p_info) + ".tmp"
    Path(tmp).write_text(json.dumps(info))
    os.replace(tmp, p_info)
    log(f"{vid}: {info['frames']} fr ({info['width']}x{info['height']}, native {info['native_fps']:.2f} fps, "
        f"{info['dup_frames']} dup) in {info['total_s']} s = {info['frames'] / info['total_s']:.1f} fr/s; "
        f"valid {info['valid_frac']:.1%}; cuts sd {len(sd)} lj {len(lj)} merged {len(events)}; "
        f"raw {info['raw_mb']} MB")
def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--keep-f32", default=None)
    a = ap.parse_args()
    TAG = f"[w{a.shard}]"
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    meta = pd.read_csv(ROOT / cfg["paths"]["metadata"])
    todo = meta if a.all else meta[meta.video_id.isin(a.ids)]
    todo = todo.iloc[a.shard::a.nshards]
    log(f"{len(todo)} videos in this shard")
    n_fail = 0
    for k, row in enumerate(todo.itertuples(), 1):
        try:
            process(row, cfg, a.keep_f32)
        except DriveGone as e:
            log(f"{row.video_id}: STOP, {e}. Rerun the same command to resume.")
            sys.exit(2)
        except Exception:
            n_fail += 1
            log(f"{row.video_id}: FAILED\n{traceback.format_exc()}")
        log(f"progress {k}/{len(todo)}")
    log(f"finished ({n_fail} failed)")


if __name__ == "__main__":
    main()
