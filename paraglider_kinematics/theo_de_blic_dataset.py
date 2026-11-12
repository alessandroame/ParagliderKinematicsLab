from __future__ import annotations

import math
from pathlib import Path
import shutil
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .schema import ManeuverKinematicsTrace, WingGeometrySpec
from .vision_tracker import DualViewParagliderTracker


def decode_real_video_frames_rgb(
    video_path: str | Path,
    fps: float = 30.0,
    width: int = 960,
    height: int = 540,
) -> List[np.ndarray]:
    """
    Decodes RGB frames strictly from an existing real video file on disk (`.mp4`, `.mov`, `.mkv`).
    Raises `FileNotFoundError` immediately when the video file is absent — synthetic frame generation is forbidden.
    """
    if video_path is None:
        raise FileNotFoundError("Real video path is required (None given); synthetic frame generation is disabled.")
    src_path = Path(video_path)
    if not src_path.is_file():
        raise FileNotFoundError(
            f"Real video file not found at '{src_path}'. Synthetic fallback frames are strictly forbidden."
        )

    if shutil.which("ffmpeg") is not None:
        return DualViewParagliderTracker.extract_video_frames_ffmpeg(
            src_path, fps=fps, width=width, height=height
        )

    import cv2

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open real video file via OpenCV: {src_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 1e-3:
        src_fps = fps

    frames_rgb: List[np.ndarray] = []
    raw_idx = 0
    next_sample_time = 0.0
    dt_out = 1.0 / max(1.0, fps)

    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            t_raw = raw_idx / src_fps
            raw_idx += 1
            if t_raw + 0.5 / src_fps < next_sample_time:
                continue
            if bgr.shape[1] != width or bgr.shape[0] != height:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)
            next_sample_time += dt_out
    finally:
        cap.release()

    if not frames_rgb:
        raise RuntimeError(f"Decoded 0 frames from real video file: {src_path}")
    return frames_rgb


def extract_trace_from_real_video(
    video_path: str | Path,
    maneuver_key: str,
    maneuver_name: str,
    timestamp_range: str,
    fps: float = 30.0,
    width: int = 960,
    height: int = 540,
    camera_mode: str = "side_chase",
    source_video_url: str = "https://www.youtube.com/watch?v=ykuaAYzB2qU",
    source_video_title: str = "BASIC PARAGLIDING EXERCISES - THEO DE BLIC's TUTORIALS",
    wing: Optional[WingGeometrySpec] = None,
) -> ManeuverKinematicsTrace:
    """
    Decodes a real `.mp4` clip on disk and runs frame-by-frame Computer Vision recognition
    to extract the `ManeuverKinematicsTrace`. Raises `FileNotFoundError` if `video_path` is missing.
    """
    decoded_frames = decode_real_video_frames_rgb(video_path, fps=fps, width=width, height=height)
    tracker = DualViewParagliderTracker(wing=wing or WingGeometrySpec())
    return tracker.analyze_frames(
        frames_rgb=decoded_frames,
        fps=fps,
        source_video_url=source_video_url,
        source_video_title=source_video_title,
        maneuver_id=maneuver_key,
        maneuver_name=maneuver_name,
        video_timestamp_range=timestamp_range,
        camera_mode=camera_mode,
    )


def render_cv_overlay_keyframe_png(
    video_path: str | Path,
    trace: ManeuverKinematicsTrace,
    frame_index: int,
    out_png_path: str | Path,
    width: int = 960,
    height: int = 540,
) -> Path:
    """
    Renders a diagnostic Computer Vision keyframe PNG directly onto a real decoded video frame
    (or real extracted PNG/JPEG frame). Raises `FileNotFoundError` when `video_path` does not exist.
    """
    if video_path is None:
        raise FileNotFoundError("Real video or frame path is required for keyframe overlay.")
    src_path = Path(video_path)
    if not src_path.is_file():
        raise FileNotFoundError(
            f"Real video or image file required for CV keyframe overlay (not found: '{src_path}')."
        )

    if not trace.frames:
        raise ValueError("Trace contains zero frames.")

    idx = max(0, min(len(trace.frames) - 1, frame_index))
    obs = trace.frames[idx]

    if src_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        pil_src = Image.open(src_path).convert("RGB")
        if pil_src.size != (width, height):
            pil_src = pil_src.resize((width, height), Image.Resampling.BILINEAR)
        rgb = np.asarray(pil_src, dtype=np.uint8)
    else:
        frames = decode_real_video_frames_rgb(src_path, fps=trace.fps, width=width, height=height)
        f_idx = max(0, min(len(frames) - 1, frame_index))
        rgb = frames[f_idx]

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    chase_w = int(0.68 * width)

    mid_y = 0.72 * height
    slope = math.tan(math.radians(obs.horizon_tilt_deg))
    y0 = mid_y - slope * (0.5 * chase_w)
    y1 = mid_y + slope * (0.5 * chase_w)
    draw.line([(0, y0), (chase_w, y1)], fill=(0, 255, 255), width=2)

    px, py = obs.pilot_centroid_px
    cx, cy = obs.canopy_centroid_px
    grav_rad = math.radians(obs.horizon_tilt_deg)
    gx = px - 200.0 * math.sin(grav_rad)
    gy = py - 200.0 * math.cos(grav_rad)
    draw.line([(px, py), (gx, gy)], fill=(80, 240, 120), width=2)
    draw.line([(px, py), (cx, cy)], fill=(255, 230, 0), width=3)

    if trace.camera_mode == "rear_chase":
        span_r = max(36.0, 0.5 * obs.canopy_span_px)
        bank_rad = math.radians(obs.roll_deg + obs.horizon_tilt_deg)
        cos_b, sin_b = math.cos(bank_rad), math.sin(bank_rad)
        oval_pts = [
            (
                cx + (span_r + 6.0) * math.cos(math.radians(a)) * cos_b
                - 18.0 * math.sin(math.radians(a)) * sin_b,
                cy + (span_r + 6.0) * math.cos(math.radians(a)) * sin_b
                + 18.0 * math.sin(math.radians(a)) * cos_b,
            )
            for a in range(0, 360, 15)
        ]
        draw.polygon(oval_pts, outline=(0, 255, 120), width=2)
    else:
        draw.ellipse([cx - 38, cy - 18, cx + 38, cy + 18], outline=(0, 255, 120), width=2)
    draw.ellipse([px - 18, py - 16, px + 18, py + 16], outline=(255, 90, 90), width=2)

    from .video_annotator import draw_pilot_skeleton_overlay

    draw_pilot_skeleton_overlay(draw, obs)

    draw.rectangle([8, 8, chase_w - 8, 62], fill=(12, 18, 28))
    if trace.camera_mode == "rear_chase":
        hud_line1 = (
            f"FRAME #{obs.frame_index:03d} | t = {obs.time_sec:5.2f}s | "
            f"BRAKE L/R = {obs.brake_left_norm*100:3.0f}%/{obs.brake_right_norm*100:3.0f}% | "
            f"WS = {obs.weight_shift_norm*100:+4.0f}% | EARS = {obs.big_ears_norm*100:3.0f}%"
        )
        hud_line2 = (
            f"ROLL = {obs.roll_deg:+6.2f} deg ({obs.roll_rate_deg_sec:+6.1f} deg/s) | "
            f"PITCH = {obs.pitch_deg:+5.1f} deg | SPAN = {obs.canopy_span_px:5.1f} px"
        )
    else:
        sym_pct = 50.0 * (obs.brake_left_norm + obs.brake_right_norm)
        hud_line1 = (
            f"FRAME #{obs.frame_index:03d} | t = {obs.time_sec:5.2f}s | "
            f"BRAKE L/R = {sym_pct:4.1f}% (d/dt = {obs.brake_rate_norm_per_sec * 100:+5.0f}%/s)"
        )
        hud_line2 = (
            f"CANOPY PITCH = {obs.pitch_deg:+6.2f} deg (rate = {obs.pitch_rate_deg_sec:+6.1f} deg/s) | "
            f"SURGE = {obs.canopy_forward_offset_m:+5.2f} m"
        )
    draw.text((16, 14), hud_line1, fill=(240, 248, 255))
    draw.text((16, 36), hud_line2, fill=(255, 215, 80))

    out = Path(out_png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
