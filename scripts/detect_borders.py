"""Detect static black borders (letterbox / pillarbox) per clip with ffmpeg cropdetect.

Samples 3 s at 5 positions (10, 30, 50, 70, 85 % of duration), takes the most frequent crop per sample and the
union (largest box) over samples, so a border is only reported if it is present at every sample.
Output: cache/borders.csv  (video_id, W, H, crop_w, crop_h, crop_x, crop_y, content_frac). Resumable.
usage: python scripts/detect_borders.py [--ids ID ...]
"""
import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import extract as X

POS = [0.10, 0.30, 0.50, 0.70, 0.85]
def crop_at(path, ss, limit=24, round_to=2):
    out = subprocess.run(["ffmpeg", "-v", "info", "-ss", f"{ss:.2f}", "-i", str(path), "-t", "3",
                          "-vf", f"cropdetect={limit}:{round_to}:0", "-an", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    c = collections.Counter(re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out))
    return tuple(map(int, c.most_common(1)[0][0])) if c else None


def detect(path):
    W, H, _ = X.probe_video(path)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(path)], capture_output=True, text=True).stdout.strip())
    boxes = [b for b in (crop_at(path, p * dur) for p in POS) if b]
    if not boxes:
        return dict(W=W, H=H, crop_w=W, crop_h=H, crop_x=0, crop_y=0, content_frac=1.0, n_samples=0)
    x0 = min(b[2] for b in boxes)
    y0 = min(b[3] for b in boxes)
    x1 = max(b[2] + b[0] for b in boxes)
    y1 = max(b[3] + b[1] for b in boxes)
    w, h = x1 - x0, y1 - y0
    return dict(W=W, H=H, crop_w=w, crop_h=h, crop_x=x0, crop_y=y0, content_frac=round(w * h / (W * H), 4),
                n_samples=len(boxes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    meta = pd.read_csv(ROOT / cfg["paths"]["metadata"])
    if a.ids:
        meta = meta[meta.video_id.isin(a.ids)]
    out = ROOT / cfg["paths"]["cache_dir"] / "borders.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["video_id"])
    rows = done.to_dict("records")
    for k, r in enumerate(meta.itertuples(), 1):
        if r.video_id in set(done.video_id):
            continue
        local = Path(cfg["paths"]["local_video_cache"]) / r.video_path
        path = local if local.exists() else Path(cfg["paths"]["video_root"]) / r.video_path
        rows.append({"video_id": r.video_id, **detect(path)})
        print(k, rows[-1], flush=True)
        if k % 10 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
