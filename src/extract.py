"""Per-frame extraction at the target fps (ffmpeg decodes straight to 15 fps, native resolution).

Outputs per video (resumable, written atomically):
  cache/raw/<id>.npz        RAW: all 478 landmarks (float16 pixel offsets from a float32 per-frame anchor),
                            52 blendshapes (float16), 4x4 facial transformation matrix (float32), t, anchor
  cache/frames/<id>.parquet per-frame components (EAR L/R, brow L/R, gaze x/y per eye, yaw/pitch/roll, lip
                            aperture, jaw, ...), quality flags, IOD px, duplicate flag, PySceneDetect frame score
No combination of components into channels here; that happens at feature time (src/channels.py).
"""
import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

N_LMK = 478
ANCHOR = 168                              # nose bridge between the eyes; stable, central
R_EYE = [33, 160, 158, 133, 153, 144]     # p1..p6, image-left eye (subject's right)
L_EYE = [362, 385, 387, 263, 373, 380]    # image-right eye; corners 362 (inner), 263 (outer)
R_BROW = [70, 63, 105, 66, 107]
L_BROW = [300, 293, 334, 296, 336]
IRIS_A = list(range(468, 473))            # centre first
IRIS_B = list(range(473, 478))
MOUTH = [13, 14, 78, 308, 61, 291, 0, 17]
USED = sorted(set(R_EYE + L_EYE + R_BROW + L_BROW + IRIS_A + IRIS_B + MOUTH + [1, 2, 152]))

# Head pose: SolvePnP with MediaPipe's canonical 3D face model (generic, cm)
PNP_IDX = [1, 4, 5, 6, 168, 197, 195, 10, 151, 9, 8, 33, 133, 362, 263, 234, 454, 127, 356]


def _load_canonical(path=Path(__file__).resolve().parents[1] / "models" / "canonical_face_model.obj"):
    V = np.array([list(map(float, l.split()[1:4])) for l in open(path) if l.startswith("v ")])
    return V * np.array([1.0, -1.0, -1.0])

PNP_MODEL = _load_canonical()[PNP_IDX].astype(np.float64)


def probe_video(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return s["width"], s["height"], float(Fraction(s["avg_frame_rate"]))


def iter_frames(path, width, height, fps_out, crop=None):
    """ffmpeg decodes and outputs exactly fps_out frames/s (fps filter, nearest-frame selection).
    crop = (x, y, w, h) in pixels: frames are cropped to that box (used to remove black borders).
    Yields (index, t_seconds, rgb_frame). Sequential, no seeking."""
    vf = f"fps={fps_out}"
    if crop is not None:
        x, y, width, height = crop
        vf += f",crop={width}:{height}:{x}:{y}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    n = width * height * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield i, i / fps_out, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def make_landmarker(cfg, root):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    e = cfg["extraction"]
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(root / e["landmarker_model"])),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=e["num_faces"],
        min_face_detection_confidence=e["min_face_detection_confidence"],
        min_face_presence_confidence=e["min_face_presence_confidence"],
        min_tracking_confidence=e["min_tracking_confidence"],
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp, vision.FaceLandmarker.create_from_options(opts)


def make_scene_detector(cfg, fps):
    from scenedetect import ContentDetector
    c = cfg["cuts"]
    return ContentDetector(threshold=c["scenedetect_threshold"],
                           min_scene_len=max(1, int(round(c["scenedetect_min_scene_s"] * fps))))


def euler_from_R(R):
    """Pitch (x), yaw (y), roll (z) in degrees via RQ decomposition."""
    angles = cv2.RQDecomp3x3(R)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


def _d(a, b):
    return float(np.linalg.norm(a - b))


def components(F, W, H):
    """F: (478, 3) landmarks in pixels (x, y) + z. Per-frame components only (no channel combination)."""
    g = lambda k: F[k, :2]
    rc = (g(33) + g(133)) / 2
    lc = (g(362) + g(263)) / 2
    iod = _d(lc, rc)
    r = {"iod_px": iod, "eye_mid_x": (rc[0] + lc[0]) / 2, "eye_mid_y": (rc[1] + lc[1]) / 2}

    def ear(idx):
        p1, p2, p3, p4, p5, p6 = (g(k) for k in idx)
        return (_d(p2, p6) + _d(p3, p5)) / (2 * _d(p1, p4))
    r["ear_r"], r["ear_l"] = ear(R_EYE), ear(L_EYE)
    r["lip_ap"] = _d(g(13), g(14)) / iod          # inner-lip aperture
    r["mouth_w"] = _d(g(78), g(308)) / iod
    r["jaw"] = _d(g(2), g(152)) / iod              # subnasale to chin
    br = np.mean([g(k) for k in R_BROW], axis=0)
    bl = np.mean([g(k) for k in L_BROW], axis=0)
    r["brow_r"] = _d(br, rc) / iod
    r["brow_l"] = _d(bl, lc) / iod

    ia, ib = g(468), g(473)                         # iris assignment: nearest eye centre
    if _d(ia, rc) + _d(ib, lc) <= _d(ia, lc) + _d(ib, rc):
        ir, il = ia, ib
    else:
        ir, il = ib, ia

    def gaze(iris, left_corner, right_corner):
        ax = right_corner - left_corner
        w2 = float(ax @ ax)
        v = iris - left_corner
        perp = np.array([-ax[1], ax[0]])
        return float(v @ ax) / w2, float(v @ perp) / w2
    r["gaze_x_r"], r["gaze_y_r"] = gaze(ir, g(33), g(133))
    r["gaze_x_l"], r["gaze_y_l"] = gaze(il, g(362), g(263))

    img_pts = np.ascontiguousarray(F[PNP_IDX, :2], dtype=np.float64)
    f = float(max(W, H))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
    ok, rv, tv = cv2.solvePnP(PNP_MODEL, img_pts, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if ok and tv[2, 0] > 0:
        R, _ = cv2.Rodrigues(rv)
        r["pitch"], r["yaw"], r["roll"] = euler_from_R(R)
        r["pnp_tx"], r["pnp_ty"], r["pnp_tz"] = (float(v) for v in tv[:, 0])
        proj, _ = cv2.projectPoints(PNP_MODEL, rv, tv, K, None)
        r["pnp_reproj_iod"] = float(np.mean(np.linalg.norm(proj[:, 0] - img_pts, axis=1))) / iod
    return r


COMPONENT_COLS = ["iod_px", "eye_mid_x", "eye_mid_y", "ear_r", "ear_l", "lip_ap", "mouth_w", "jaw", "brow_r",
                  "brow_l", "gaze_x_r", "gaze_y_r", "gaze_x_l", "gaze_y_l", "pitch", "yaw", "roll",
                  "pnp_tx", "pnp_ty", "pnp_tz", "pnp_reproj_iod", "mp_pitch", "mp_yaw", "mp_roll",
                  "frac_lmk_inside", "face_idx", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]


def extract(video_path, cfg, root, log=print, keep_f32=False, crop=None):
    """Landmark every frame of the fps-filtered stream. Returns (frames DataFrame, raw dict, info).
    crop = (x, y, w, h): the landmarker sees only that box, but every coordinate is mapped back to the ORIGINAL
    frame (pixels), and W, H, the in-frame check and the PnP intrinsics stay those of the original frame."""
    W, H, native_fps = probe_video(video_path)
    cx, cy, cw, ch = crop if crop is not None else (0, 0, W, H)
    fps = cfg["extraction"]["target_fps"]
    mp, lmk = make_landmarker(cfg, root)
    sd = make_scene_detector(cfg, fps)
    from scenedetect.common import FrameTimecode
    sd_w = cfg["cuts"]["scenedetect_downscale_width"]
    sd_size = (sd_w, max(2, int(round(ch * sd_w / cw))))
    min_iod = cfg["extraction"]["min_iod_px"]

    rows, LM, BS, TM, ANC, F32 = [], [], [], [], [], []
    bs_names = None
    prev_mid, prev_frame = None, None
    t0 = time.time()
    try:
        for i, t, frame in iter_frames(video_path, W, H, fps, crop):
            dup = prev_frame is not None and np.array_equal(frame, prev_frame)
            prev_frame = frame
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), sd_size, interpolation=cv2.INTER_AREA)
            sd_cuts = sd.process_frame(FrameTimecode(i, fps=float(fps)), small)
            row = {"frame": i, "t": t, "dup": dup, "sd_score": sd._frame_score,
                   "sd_cut": bool(sd_cuts), "n_faces": 0, "face_found": False, "in_frame": False, "valid": False}
            lm = np.full((N_LMK, 3), np.nan, np.float32)
            bs = np.full(52, np.nan, np.float32)
            tm = np.full((4, 4), np.nan, np.float32)
            anc = np.full(2, np.nan, np.float32)

            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            res = lmk.detect_for_video(img, int(round(t * 1000)))
            row["n_faces"] = len(res.face_landmarks)
            if res.face_landmarks:
                faces = [np.array([[p.x * cw + cx, p.y * ch + cy, p.z * cw] for p in fl]) for fl in res.face_landmarks]
                mids = [(f[33, :2] + f[263, :2]) / 2 for f in faces]
                if prev_mid is not None and len(faces) > 1:
                    k = int(np.argmin([np.linalg.norm(m - prev_mid) for m in mids]))
                else:
                    k = int(np.argmax([np.linalg.norm(f[33, :2] - f[263, :2]) for f in faces]))
                F = faces[k]
                prev_mid = mids[k]
                lm[:] = F
                anc[:] = F[ANCHOR, :2]
                U = F[USED]
                inside = bool(np.all((U[:, 0] >= 0) & (U[:, 0] < W) & (U[:, 1] >= 0) & (U[:, 1] < H)))
                all_inside = float(np.mean((F[:, 0] >= 0) & (F[:, 0] < W) & (F[:, 1] >= 0) & (F[:, 1] < H)))
                row.update(face_found=True, in_frame=inside, frac_lmk_inside=all_inside, face_idx=k,
                           bbox_x0=F[:, 0].min(), bbox_y0=F[:, 1].min(), bbox_x1=F[:, 0].max(), bbox_y1=F[:, 1].max())
                row.update(components(F, W, H))
                if res.facial_transformation_matrixes and k < len(res.facial_transformation_matrixes):
                    tm[:] = np.array(res.facial_transformation_matrixes[k])
                    row["mp_pitch"], row["mp_yaw"], row["mp_roll"] = euler_from_R(tm[:3, :3].astype(np.float64))
                if res.face_blendshapes and k < len(res.face_blendshapes):
                    bs[:] = [c.score for c in res.face_blendshapes[k]]
                    if bs_names is None:
                        bs_names = [c.category_name for c in res.face_blendshapes[k]]
                row["valid"] = inside and row["iod_px"] >= min_iod
            rows.append(row)
            LM.append(np.where(np.isnan(lm), np.nan, lm - np.array([*anc, 0.0], np.float32)).astype(np.float16))
            BS.append(bs.astype(np.float16))
            TM.append(tm)
            ANC.append(anc)
            if keep_f32:
                F32.append(lm)
            if i and i % 300 == 0:
                log(f"    frame {i}  t={t:.0f}s  {i / (time.time() - t0):.1f} fr/s")
    finally:
        lmk.close()
    df = pd.DataFrame(rows)
    for c in COMPONENT_COLS:
        if c not in df:
            df[c] = np.nan
    raw = {"t": df["t"].to_numpy(np.float32), "anchor_px": np.stack(ANC), "lmk_offset_f16": np.stack(LM),
           "blendshapes_f16": np.stack(BS), "transform_matrix": np.stack(TM),
           "blendshape_names": np.array(bs_names or []), "width": W, "height": H, "fps": fps,
           "anchor_index": ANCHOR, "crop_xywh": np.array(crop if crop is not None else (0, 0, W, H))}
    if keep_f32:
        raw["lmk_px_f32_debug"] = np.stack(F32)
    info = {"width": W, "height": H, "native_fps": native_fps, "fps": fps, "frames": len(df),
            "dup_frames": int(df["dup"].sum()), "seconds": round(time.time() - t0, 1),
            "crop_xywh": list(map(int, crop)) if crop is not None else None,
            "face_found_frac": round(float(df["face_found"].mean()), 4)}
    return df, raw, info


def landmarks_px(raw):
    """Rebuild (T, 478, 3) float32 pixel landmarks from the stored raw npz."""
    L = raw["lmk_offset_f16"].astype(np.float32)
    L[:, :, :2] += raw["anchor_px"][:, None, :]
    return L


def atomic_write_parquet(df, path):
    tmp = str(path) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_savez(path, **arrays):
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
