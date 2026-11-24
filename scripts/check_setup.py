"""Check the environment before a run: package versions, ffmpeg, model hashes, video folder.
usage: PYTHONPATH=. python scripts/check_setup.py"""
import hashlib, importlib, shutil, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
WANT = {"numpy": "2.4.6", "pandas": "2.2.3", "scipy": "1.16.3", "sklearn": "1.6.1", "cv2": "5.0.0",
        "mediapipe": "1.1.1", "scenedetect": "0.7.2", "tabulate": "0.9.0"}
SHA = {"models/face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
       "models/canonical_face_model.obj": "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618"}
bad = 0
print("python", sys.version.split()[0], "(reference 3.13.18)")
for mod, v in WANT.items():
    try:
        got = importlib.import_module(mod).__version__
    except Exception as e:
        got = f"MISSING ({e.__class__.__name__})"
    ok = got.startswith(v)
    bad += not ok
    print(f"{'ok ' if ok else 'DIFF'} {mod:12s} {got}  (reference {v})")
for tool in ["ffmpeg", "ffprobe"]:
    p = shutil.which(tool)
    v = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout.split("\n")[0] if p else "MISSING"
    print(f"{'ok ' if p else 'DIFF'} {v}  (reference 6.1.1)"); bad += not p
for f, h in SHA.items():
    got = hashlib.sha256((ROOT / f).read_bytes()).hexdigest() if (ROOT / f).exists() else "missing"
    ok = got == h; bad += not ok
    print(f"{'ok ' if ok else 'DIFF'} {f} sha256 {got[:16]}...")
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
import pandas as pd
meta = pd.read_csv(ROOT / cfg["paths"]["metadata"])
vr = Path(cfg["paths"]["video_root"])
found = sum((vr / p).exists() for p in meta.video_path) if vr.is_dir() else 0
print(f"{'ok ' if found == len(meta) else 'DIFF'} videos found: {found} / {len(meta)} under {vr}"); bad += found != len(meta)
print("\nall checks passed" if not bad else f"\n{bad} check(s) differ from the reference run")
