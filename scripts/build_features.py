"""Per-window table and clip features.

Writes features/windows.parquet (usable windows, activities, composition, ILR), features/clip_info.csv
(window bookkeeping for every clip), features/clips.csv (phi(V) for clips with >= min_windows, + metadata).
Resumable: per-clip results cached in features/per_clip/<rule>/<id>.{parquet,json}. --force recomputes.
Usage: python scripts/build_features.py [--rule splices_only|all_events] [--force]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import features as FT
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--brow-frontal", action="store_true", help="variant: brow only on |yaw| < features.brow_frontal_max_abs_yaw")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    rule = a.rule or cfg["cuts"]["feature_rule"]
    meta = pd.read_csv(ROOT / cfg["paths"]["metadata"])
    fdir = ROOT / cfg["paths"]["features_dir"]
    by = cfg["features"]["brow_frontal_max_abs_yaw"] if a.brow_frontal else None
    tag = rule + ("_browfrontal" if a.brow_frontal else "")
    pc = fdir / "per_clip" / tag
    pc.mkdir(parents=True, exist_ok=True)
    wins, infos = [], []
    t0 = time.time()
    for i, vid in enumerate(meta.video_id):
        pw, pj = pc / f"{vid}.parquet", pc / f"{vid}.json"
        if pj.exists() and not a.force:
            info = json.loads(pj.read_text())
            w = pd.read_parquet(pw) if pw.exists() else pd.DataFrame()
        else:
            w, info = FT.clip_windows(vid, cfg, ROOT, rule=rule, brow_max_abs_yaw=by)
            if len(w):
                w.to_parquet(pw, index=False)
            pj.write_text(json.dumps(info, default=float))
        wins.append(w)
        infos.append(info)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(meta)} clips, {time.time() - t0:.0f} s", flush=True)
    win = pd.concat([w for w in wins if len(w)], ignore_index=True)
    info = pd.DataFrame(infos)
    c = cfg["composition"]
    win_full = FT.add_composition(win, FT.CHANNELS, c["eps"])
    F, cols = FT.clip_features(win, FT.CHANNELS, c["eps"], cfg["features"]["min_windows_per_clip"])
    suffix = ("" if rule == cfg["cuts"]["feature_rule"] else f"_{rule}") + ("_browfrontal" if a.brow_frontal else "")
    win_full.to_parquet(fdir / f"windows{suffix}.parquet", index=False)
    keep = ["video_id", "label", "y", "person_id", "orig_split", "age", "gender", "race", "glasses"]
    info = meta[keep].merge(info, on="video_id")
    info["included"] = info.video_id.isin(F.index)
    info.to_csv(fdir / f"clip_info{suffix}.csv", index=False)
    out = meta[keep].merge(F.reset_index(), on="video_id")
    out.to_csv(fdir / f"clips{suffix}.csv", index=False)
    print(f"rule={tag}: {len(win)} usable windows; {len(out)} / {len(meta)} clips with >= "
          f"{cfg['features']['min_windows_per_clip']} windows")
    ex = info[~info.included][["video_id", "label", "person_id", "orig_split", "n_windows", "valid_frac"]]
    print("excluded clips:\n" + ex.to_string(index=False))
    print(out.groupby(["orig_split", "label"]).agg(clips=("video_id", "size"), persons=("person_id", "nunique")))


if __name__ == "__main__":
    main()
