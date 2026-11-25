# Facial micro-dynamics dementia screening on YT-DemTalk: an independent implementation

This is an independent implementation, written from the paper text, of:

> Cenacchi F, Cao L, McEwan M, Richards D. *Passive dementia screening via facial temporal micro-dynamics analysis of in-the-wild talking-head video.* arXiv:2511.13802v2, 2025.

The paper has no public code. We follow every step it states. Every value it does not give was fixed before any test result was seen, and is marked `[ours]` in `config.yaml`.

The code runs one protocol: Stats + PCA + Random Forest (plus GBDT and AdaBoost, as in the paper's Table 1) on the **released YT-DemTalk train / test split**.

## Result

Test set: 26 clips (9 dementia clips from 8 persons, 17 controls).

| Stats + PCA + RF | AUROC [95% CI] | AP | F1 | Acc |
|---|---|---|---|---|
| Paper | 0.953 [0.912–0.984] | 0.961 | 0.851 | 0.857 |
| **Ours, seed 42** | **0.817** [0.634–0.948] | **0.717** | 0.621 | 0.577 |
| Ours, RF seeds 0–9 | 0.769 ± 0.018 | 0.652 ± 0.020 | | |

GBDT: 0.712 (paper 0.893). AdaBoost: 0.837 (paper 0.902).

Four independent runs from the raw videos gave identical numbers.

## Data

The videos are **not** included. `metadata.csv` lists the 199 clips used, with these columns:
- the YouTube id and link;
- the label and train / test split, from the released YT-DemTalk CSV;
- `person_id`, assigned by us;
- the released metadata (age, gender, glasses, ...).

How the 199 clips were chosen and prepared:
- **Selection:** from the 300 released clips, 100 were removed by manual review, 50 per class:
  - busy scenes;
  - very low quality;
  - sunglasses.
- **Duplicate:** 1 duplicate was then removed, which leaves 99 dementia and 100 control clips.
- **About 90 s per clip:** each clip contains only segments in which the person talks to the camera, joined together. The pipeline detects the join points and drops the windows around them.

## Folder layout

```
.
├── config.yaml
├── metadata.csv
├── models
│   ├── canonical_face_model.obj
│   ├── face_landmarker.task
│   └── NOTICE.md
├── requirements.txt
├── scripts
│   ├── build_features.py
│   ├── check_setup.py
│   ├── detect_borders.py
│   ├── extract.py
│   ├── __init__.py
│   └── protocol_a.py
├── src
│   ├── channels.py
│   ├── cuts.py
│   ├── evaluate.py
│   ├── extract.py
│   ├── features.py
│   └── validity.py
└── videos                      <- ADD THIS FOLDER YOURSELF (not in the repository)
    ├── Dementia
    │   └── <video_id>.mp4      99 clips
    └── Normal
        └── <video_id>.mp4      100 clips
```

**Add the videos yourself.** Put a folder named `videos` in the repository root, next to `config.yaml`, with the two subfolders `Dementia` and `Normal`.
- Each file is named `<video_id>.mp4`, exactly as in the `video_path` column of `metadata.csv` (for example `videos/Dementia/uAlkCMfTASQ.mp4`).
- `videos/` is listed in `.gitignore`, so it is never committed.

The run creates four more folders in the repository root. They are not committed either:
- `cache/`: per-frame data, about 0.8 GB;
- `features/`: clip features;
- `results/`: `results/protocol_A/` holds tables, metrics and figures;
- `logs/`: extraction logs.

## Setup

The reference environment is Python 3.13.18, ffmpeg 6.1.1, CPU only (the Google Colab defaults).

```bash
pip install -r requirements.txt
pip install --no-deps mediapipe==1.1.1 scenedetect==0.7.2
pip install certifi
```

Next, check the two paths in `config.yaml`:

```yaml
video_root: .                              # "." = the repository root, which contains videos/
local_video_cache: /content/videos_local   # each video is copied here first (fast local disk on Colab)
```

If your videos are somewhere else, set `video_root` to the folder that **contains** `videos/`.

Finally, check the setup. It should end with `videos found: 199 / 199` and `all checks passed`:

```bash
PYTHONPATH=. python scripts/check_setup.py
```

## Run

Run these commands from the repository folder. In a Colab or Jupyter notebook cell, put `!` in front of each command.

```bash
# 1. output folders and black-border detection (a few minutes)
mkdir -p cache/frames cache/raw cache/cuts features results logs
PYTHONPATH=. python scripts/detect_borders.py

# 2. per-frame extraction: 2 background workers, about 4 h; resumable, rerun the same commands if interrupted
nohup env PYTHONPATH=. python scripts/extract.py --all --shard 0 --nshards 2 > logs/extract_0.log 2>&1 &
nohup env PYTHONPATH=. python scripts/extract.py --all --shard 1 --nshards 2 > logs/extract_1.log 2>&1 &

#    progress: finished clips (199 at the end) and running workers (0 = done)
ls cache/frames/*.parquet | wc -l
ps aux | grep "scripts/extract.py" | grep -v grep | wc -l

# 3. clip features (about 1 min), then training and test (a few minutes)
PYTHONPATH=. python scripts/build_features.py
PYTHONPATH=. python scripts/protocol_a.py
```

Expected output:
- `build_features.py` reports `5782 usable windows; 194 / 199 clips`.
- `protocol_a.py` prints `RF {'AUROC': 0.817, 'AP': 0.717, ...}`.

All results go to `results/protocol_A/`: `table1.md`, `metrics.json`, `predictions.csv` and the figures.
