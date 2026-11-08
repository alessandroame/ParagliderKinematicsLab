from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .physics_replay import ParagliderPhysicsComparator
from .schema import ManeuverKinematicsTrace, WingGeometrySpec
from .vision_tracker import DualViewParagliderTracker


def _min_jerk(t0: float, t1: float, t: float) -> float:
    span = max(1e-6, t1 - t0)
    u = max(0.0, min(1.0, (t - t0) / span))
    return (u**3) * (10.0 - 15.0 * u + 6.0 * (u**2))


def _pulse_envelope(
    t: float, t_start: float, t_end: float, attack_s: float = 0.42, release_s: float = 0.28
) -> float:
    if t <= t_start or t >= t_end:
        return 0.0
    up = _min_jerk(t_start, t_start + attack_s, t)
    down = 1.0 - _min_jerk(t_end - release_s, t_end, t)
    return up * down


def build_theo_pitch_command_law(maneuver_key: str, t: float) -> Tuple[float, float, float]:
    """
    Returns (brake_left_norm, brake_right_norm, weight_shift_norm) at time t (sec)
    for the 3 Pitch Pendulum exercises in Theo de Blic's tutorial (00:53 - 10:43).
    """
    if maneuver_key == "pitch_pendulum_uncaught":
        # 03:09 - How to Pitch (Single symmetric pull -> brisk release at aft apex -> uncaught forward surge)
        sym = 0.78 * _pulse_envelope(t, 0.50, 1.88, attack_s=0.45, release_s=0.26)
        return sym, sym, 0.0

    if maneuver_key == "pitch_pendulum_active_catch":
        # 08:19 - Catching the Dive (Pump 1 -> Surge 1 -> In-phase Pump 2 -> Release -> Active Catch after wing crosses overhead)
        pump1 = 0.72 * _pulse_envelope(t, 0.45, 1.75, attack_s=0.42, release_s=0.26)
        pump2 = 0.78 * _pulse_envelope(t, 3.60, 4.72, attack_s=0.38, release_s=0.24)
        catch_pulse = 0.74 * _pulse_envelope(t, 5.48, 6.22, attack_s=0.18, release_s=0.32)
        sym = min(1.0, pump1 + pump2 + catch_pulse)
        return sym, sym, 0.0

    if maneuver_key == "pitch_mistake_late_brake_stall":
        # 09:41 - Risk of Stalling (Braking again too early and too deep while canopy is still pitching aft at low airspeed)
        pump1 = 0.76 * _pulse_envelope(t, 0.45, 1.75, attack_s=0.42, release_s=0.26)
        early_overbrake = 0.96 * _pulse_envelope(t, 3.15, 4.95, attack_s=0.22, release_s=0.40)
        sym = min(1.0, pump1 + early_overbrake)
        return sym, sym, 0.0

    raise ValueError(f"Unknown maneuver_key: {maneuver_key}")


def render_dual_view_frame_rgb(
    width: int,
    height: int,
    t_sec: float,
    brake_l: float,
    brake_r: float,
    weight_shift: float,
    pitch_deg: float,
    roll_deg: float,
    horizon_tilt_deg: float = 1.5,
) -> np.ndarray:
    """
    Renders a Dual-View (Left 68%: External Side-Chase View; Right 32%: Onboard Pilot GoPro POV)
    RGB frame encoding the physical attitude (pitch_deg, roll_deg, horizon_tilt_deg) and
    pilot hand positions (brake_l, brake_r, weight_shift) for CV pipeline verification.
    """
    img = Image.new("RGB", (width, height), (36, 52, 78))
    draw = ImageDraw.Draw(img)

    chase_w = int(0.68 * width)
    onboard_x0 = chase_w
    onboard_w = width - chase_w

    mid_y = 0.72 * height
    slope = math.tan(math.radians(horizon_tilt_deg))
    y_left = mid_y - slope * (0.5 * chase_w)
    y_right = mid_y + slope * (0.5 * chase_w)

    draw.rectangle([0, 0, chase_w, height], fill=(115, 162, 218))
    draw.polygon(
        [(0, y_left), (chase_w, y_right), (chase_w, height), (0, height)],
        fill=(62, 78, 68),
    )

    px = 0.48 * chase_w
    py = 0.56 * height
    line_px = 0.34 * height

    total_angle_rad = math.radians(pitch_deg + horizon_tilt_deg)
    cx = px - line_px * math.sin(total_angle_rad)
    cy = py - line_px * math.cos(total_angle_rad)

    draw.line([(px, py), (cx - 18, cy + 4)], fill=(195, 205, 220), width=2)
    draw.line([(px, py), (cx + 18, cy - 4)], fill=(195, 205, 220), width=2)
    draw.line([(px, py), (cx, cy)], fill=(225, 235, 245), width=2)

    chord_half = 34.0
    thick_half = 11.0
    cos_a = math.cos(total_angle_rad)
    sin_a = math.sin(total_angle_rad)
    canopy_pts = []
    for deg in range(0, 360, 20):
        rad = math.radians(deg)
        lx = chord_half * math.cos(rad)
        ly = thick_half * math.sin(rad)
        rx = cx + lx * cos_a - ly * sin_a
        ry = cy + lx * sin_a + ly * cos_a
        canopy_pts.append((rx, ry))
    draw.polygon(canopy_pts, fill=(245, 112, 22), outline=(255, 185, 55))

    draw.ellipse([px - 14, py - 10, px + 14, py + 12], fill=(12, 14, 18))

    draw.rectangle([onboard_x0, 0, width, height], fill=(42, 56, 72))
    draw.line([(onboard_x0, 0), (onboard_x0, height)], fill=(110, 130, 155), width=3)

    y_pulley = 0.20 * height
    y_carabiner = 0.78 * height
    travel_px = y_carabiner - y_pulley

    riser_lx = onboard_x0 + 0.28 * onboard_w
    riser_rx = onboard_x0 + 0.72 * onboard_w

    draw.line([(riser_lx, 0.08 * height), (riser_lx, y_carabiner)], fill=(90, 105, 120), width=4)
    draw.line([(riser_rx, 0.08 * height), (riser_rx, y_carabiner)], fill=(90, 105, 120), width=4)

    draw.line(
        [(onboard_x0 + 15, y_pulley), (width - 15, y_pulley)], fill=(90, 190, 120), width=1
    )
    draw.line(
        [(onboard_x0 + 15, y_carabiner), (width - 15, y_carabiner)], fill=(210, 95, 95), width=1
    )

    hand_ly = y_pulley + brake_l * travel_px
    hand_ry = y_pulley + brake_r * travel_px
    r_hand = 11
    draw.ellipse(
        [riser_lx - r_hand, hand_ly - r_hand, riser_lx + r_hand, hand_ly + r_hand],
        fill=(248, 220, 40),
    )
    draw.ellipse(
        [riser_rx - r_hand, hand_ry - r_hand, riser_rx + r_hand, hand_ry + r_hand],
        fill=(248, 220, 40),
    )

    torso_x = onboard_x0 + (0.5 + 0.22 * weight_shift) * onboard_w
    torso_y = 0.88 * height
    draw.ellipse([torso_x - 14, torso_y - 10, torso_x + 14, torso_y + 10], fill=(35, 200, 225))

    return np.array(img, dtype=np.uint8)


def generate_video_and_extract_trace(
    maneuver_key: str,
    maneuver_name: str,
    timestamp_range: str,
    duration_sec: float,
    output_mp4_path: str | Path,
    fps: float = 30.0,
    width: int = 960,
    height: int = 540,
) -> ManeuverKinematicsTrace:
    """
    1. Integrates the physical reference trajectory of the NOVA Mentor 7 Light S under Theo de Blic's
       command law for `maneuver_key`,
    2. Encodes the Dual-View `.mp4` video via `ffmpeg`,
    3. Decodes the `.mp4` via `DualViewParagliderTracker.extract_video_frames_ffmpeg` and runs
       frame-by-frame Computer Vision recognition to extract the `ManeuverKinematicsTrace`.
    """
    wing = WingGeometrySpec()
    n_frames = int(round(duration_sec * fps)) + 1
    dt = 1.0 / fps

    from .schema import FrameObservation, PhaseLagMetrics

    seed_frames = []
    for i in range(n_frames):
        t = i * dt
        bl, br, ws = build_theo_pitch_command_law(maneuver_key, t)
        seed_frames.append(
            FrameObservation(
                frame_index=i,
                time_sec=round(t, 4),
                brake_left_norm=bl,
                brake_right_norm=br,
                brake_rate_norm_per_sec=0.0,
                weight_shift_norm=ws,
                big_ears_norm=0.0,
                horizon_tilt_deg=1.5,
                pitch_deg=0.0,
                pitch_rate_deg_sec=0.0,
                roll_deg=0.0,
                roll_rate_deg_sec=0.0,
                canopy_forward_offset_m=0.0,
                canopy_lateral_offset_m=0.0,
            )
        )

    dummy_metrics = PhaseLagMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    seed_trace = ManeuverKinematicsTrace(
        schema_version="1.0.0",
        source_video_url="https://www.youtube.com/watch?v=ykuaAYzB2qU",
        source_video_title="BASIC PARAGLIDING EXERCISES - THEO DE BLIC's TUTORIALS",
        maneuver_id=maneuver_key,
        maneuver_name=maneuver_name,
        video_timestamp_range=timestamp_range,
        fps=fps,
        wing=wing,
        phase_metrics=dummy_metrics,
        events=[],
        frames=seed_frames,
    )

    ref_sim = ParagliderPhysicsComparator.run_first_principles_two_body_engine(seed_trace)
    ref_times = np.array([p.time_sec for p in ref_sim])
    ref_pitches = np.array([p.pitch_deg for p in ref_sim])

    mp4_path = Path(output_mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "mpeg4",
        "-q:v",
        "2",
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(n_frames):
        t = i * dt
        bl, br, ws = build_theo_pitch_command_law(maneuver_key, t)
        pitch_deg = float(np.interp(t, ref_times, ref_pitches))
        horizon_deg = 1.5 * math.sin(0.7 * t)
        rgb = render_dual_view_frame_rgb(
            width, height, t, bl, br, ws, pitch_deg, 0.0, horizon_deg
        )
        proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    err_bytes = proc.stderr.read() if proc.stderr else b""
    if proc.stderr:
        proc.stderr.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed: {err_bytes.decode('utf-8', errors='replace')}")

    tracker = DualViewParagliderTracker(wing=wing)
    decoded_frames = tracker.extract_video_frames_ffmpeg(mp4_path, fps=fps, width=width, height=height)
    trace = tracker.analyze_frames(
        frames_rgb=decoded_frames,
        fps=fps,
        source_video_url="https://www.youtube.com/watch?v=ykuaAYzB2qU",
        source_video_title="BASIC PARAGLIDING EXERCISES - THEO DE BLIC's TUTORIALS",
        maneuver_id=maneuver_key,
        maneuver_name=maneuver_name,
        video_timestamp_range=timestamp_range,
    )
    return trace


def render_cv_overlay_keyframe_png(
    video_path: str | Path,
    trace: ManeuverKinematicsTrace,
    frame_index: int,
    out_png_path: str | Path,
    width: int = 960,
    height: int = 540,
) -> Path:
    """
    Renders a diagnostic Computer Vision keyframe PNG showing the detected canopy centroid,
    pilot centroid, suspension pendulum angle vector, horizon line, and brake hand gauges.
    """
    frames = DualViewParagliderTracker.extract_video_frames_ffmpeg(
        video_path, fps=trace.fps, width=width, height=height
    )
    idx = max(0, min(len(frames) - 1, frame_index))
    rgb = frames[idx]
    obs = trace.frames[idx]

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

    draw.ellipse([cx - 38, cy - 18, cx + 38, cy + 18], outline=(0, 255, 120), width=2)
    draw.ellipse([px - 18, py - 16, px + 18, py + 16], outline=(255, 90, 90), width=2)

    lx, ly = obs.left_hand_px
    rx, ry = obs.right_hand_px
    draw.rectangle([lx - 15, ly - 15, lx + 15, ly + 15], outline=(0, 255, 120), width=2)
    draw.rectangle([rx - 15, ry - 15, rx + 15, ry + 15], outline=(0, 255, 120), width=2)

    draw.rectangle([8, 8, chase_w - 8, 62], fill=(12, 18, 28))
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
