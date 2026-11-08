from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Tuple

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


def build_theo_full_command_law(
    maneuver_key: str, t: float
) -> Tuple[float, float, float, float]:
    """
    Returns (brake_left_norm, brake_right_norm, weight_shift_norm, big_ears_norm) at time t (sec)
    for all 7 maneuvers in Theo de Blic's tutorial (00:53 - 20:40).
    """
    if maneuver_key == "pitch_pendulum_uncaught":
        sym = 0.78 * _pulse_envelope(t, 0.50, 1.88, attack_s=0.45, release_s=0.26)
        return sym, sym, 0.0, 0.0

    if maneuver_key == "pitch_pendulum_active_catch":
        pump1 = 0.72 * _pulse_envelope(t, 0.45, 1.75, attack_s=0.42, release_s=0.26)
        pump2 = 0.78 * _pulse_envelope(t, 3.60, 4.72, attack_s=0.38, release_s=0.24)
        catch_pulse = 0.74 * _pulse_envelope(t, 5.48, 6.22, attack_s=0.18, release_s=0.32)
        sym = min(1.0, pump1 + pump2 + catch_pulse)
        return sym, sym, 0.0, 0.0

    if maneuver_key == "pitch_mistake_late_brake_stall":
        pump1 = 0.76 * _pulse_envelope(t, 0.45, 1.75, attack_s=0.42, release_s=0.26)
        early_overbrake = 0.96 * _pulse_envelope(t, 3.15, 4.95, attack_s=0.22, release_s=0.40)
        sym = min(1.0, pump1 + early_overbrake)
        return sym, sym, 0.0, 0.0

    if maneuver_key == "dynamic_turn_coordinated":
        # 15:04 - How to do a Dynamic Turn:
        # 1. Clear harness weight-shift preload (+0.85) starting at t=0.45s (leading brake by 0.30s)
        # 2. Progressive inside right brake impulse (68%) at t=0.75..1.85s to carve nose-down into turn
        # 3. Smooth outside/symmetric exit check pulse at t=2.55..3.30s
        ws = 0.85 * _pulse_envelope(t, 0.45, 2.35, attack_s=0.28, release_s=0.35)
        br = 0.68 * _pulse_envelope(t, 0.75, 1.88, attack_s=0.30, release_s=0.26)
        exit_check = 0.28 * _pulse_envelope(t, 2.45, 3.25, attack_s=0.22, release_s=0.28)
        return exit_check, min(1.0, br + exit_check), ws, 0.0

    if maneuver_key == "dynamic_turn_mistake_no_ws":
        # 12:58 - Common Mistake (Abrupt inside brake pull without weight-shift):
        # Pilot stays stiff in the middle of the harness (ws = 0.0) and yanks 82% right brake at t=0.50..2.10s.
        br = 0.82 * _pulse_envelope(t, 0.50, 2.10, attack_s=0.18, release_s=0.25)
        return 0.0, br, 0.0, 0.0

    if maneuver_key == "rolls_weightshift_and_brake":
        # 17:43 - How to Induce Roll (Weight-Shift + Inside Brake synced at T_1/2 = 1.38s):
        # 4 alternating swings building resonant bank angle (+/-19 -> +/-29 -> +/-37 -> +/-43 deg)
        s1 = _pulse_envelope(t, 0.45, 1.62, attack_s=0.26, release_s=0.24)
        s2 = _pulse_envelope(t, 1.83, 3.00, attack_s=0.26, release_s=0.24)
        s3 = _pulse_envelope(t, 3.21, 4.38, attack_s=0.26, release_s=0.24)
        s4 = _pulse_envelope(t, 4.59, 5.76, attack_s=0.26, release_s=0.24)
        ws = 0.82 * s1 - 0.85 * s2 + 0.88 * s3 - 0.88 * s4
        br = 0.34 * s1 + 0.38 * s3
        bl = 0.36 * s2 + 0.38 * s4
        return bl, br, ws, 0.0

    if maneuver_key == "rolls_big_ears_pure_ws":
        # 19:15 - Roll Exercise with Big Ears (Outer 28% span folded, zero brake input, pure weight-shift):
        # 4 alternating pure weight-shift pumps followed by neutral harness centering at t=5.45s
        s1 = _pulse_envelope(t, 0.45, 1.52, attack_s=0.24, release_s=0.22)
        s2 = _pulse_envelope(t, 1.70, 2.77, attack_s=0.24, release_s=0.22)
        s3 = _pulse_envelope(t, 2.95, 4.02, attack_s=0.24, release_s=0.22)
        s4 = _pulse_envelope(t, 4.20, 5.27, attack_s=0.24, release_s=0.22)
        ws = 0.86 * s1 - 0.88 * s2 + 0.90 * s3 - 0.90 * s4
        return 0.0, 0.0, ws, 1.0

    raise ValueError(f"Unknown maneuver_key: {maneuver_key}")


def build_theo_pitch_command_law(maneuver_key: str, t: float) -> Tuple[float, float, float]:
    bl, br, ws, _ = build_theo_full_command_law(maneuver_key, t)
    return bl, br, ws


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
    big_ears_norm: float = 0.0,
    camera_mode: str = "side_chase",
) -> np.ndarray:
    """
    Renders a Dual-View (Left 68%: External Side or Rear Chase View; Right 32%: Onboard Pilot GoPro POV)
    RGB frame encoding the physical attitude (pitch_deg, roll_deg, horizon_tilt_deg, big_ears_norm)
    and pilot controls (brake_l, brake_r, weight_shift) for CV pipeline verification.
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

    px = 0.50 * chase_w if camera_mode == "rear_chase" else 0.48 * chase_w
    py = 0.58 * height if camera_mode == "rear_chase" else 0.56 * height

    if camera_mode == "rear_chase":
        eff_line_px = (0.34 * height) + 1.35 * pitch_deg
        total_bank_rad = math.radians(roll_deg + horizon_tilt_deg)
        cx = px + eff_line_px * math.sin(total_bank_rad)
        cy = py - eff_line_px * math.cos(total_bank_rad)

        full_span_half = 0.5 * (0.34 * height * 0.96)
        span_half = full_span_half * (1.0 - 0.28 * big_ears_norm)
        thick_half = 12.0
        cos_b = math.cos(total_bank_rad)
        sin_b = math.sin(total_bank_rad)

        # Suspension lines from pilot to left/right risers & open span
        draw.line(
            [(px, py), (cx - 0.65 * span_half * cos_b, cy - 0.65 * span_half * sin_b)],
            fill=(195, 205, 220),
            width=2,
        )
        draw.line(
            [(px, py), (cx + 0.65 * span_half * cos_b, cy + 0.65 * span_half * sin_b)],
            fill=(195, 205, 220),
            width=2,
        )
        draw.line([(px, py), (cx, cy)], fill=(225, 235, 245), width=2)

        canopy_pts = []
        for deg in range(0, 360, 15):
            rad = math.radians(deg)
            lx = span_half * math.cos(rad)
            ly = thick_half * math.sin(rad)
            rx = cx + lx * cos_b - ly * sin_b
            ry = cy + lx * sin_b + ly * cos_b
            canopy_pts.append((rx, ry))
        draw.polygon(canopy_pts, fill=(245, 112, 22), outline=(255, 185, 55))
    else:
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
        [(onboard_x0 + 15, y_pulley), (width - 15, y_pulley)], fill=(75, 135, 110), width=1
    )
    draw.line(
        [(onboard_x0 + 15, y_carabiner), (width - 15, y_carabiner)], fill=(145, 80, 85), width=1
    )

    hand_ly = y_pulley + brake_l * travel_px
    hand_ry = y_pulley + brake_r * travel_px

    # Full articulated pilot geometry in Onboard POV
    sh_cx = onboard_x0 + (0.5 + 0.22 * weight_shift) * onboard_w
    sh_cy = 0.60 * height
    sh_hw = 0.16 * onboard_w
    sh_roll_rad = math.radians(12.0 * weight_shift + 3.0 * (brake_r - brake_l))
    l_sh = (sh_cx - sh_hw * math.cos(sh_roll_rad), sh_cy - sh_hw * math.sin(sh_roll_rad))
    r_sh = (sh_cx + sh_hw * math.cos(sh_roll_rad), sh_cy + sh_hw * math.sin(sh_roll_rad))

    hip_cx = onboard_x0 + (0.5 + 0.14 * weight_shift) * onboard_w
    hip_cy = 0.82 * height
    hip_hw = 0.11 * onboard_w
    hip_roll_rad = math.radians(8.0 * weight_shift)
    l_hip = (hip_cx - hip_hw * math.cos(hip_roll_rad), hip_cy - hip_hw * math.sin(hip_roll_rad))
    r_hip = (hip_cx + hip_hw * math.cos(hip_roll_rad), hip_cy + hip_hw * math.sin(hip_roll_rad))

    l_knee = (l_hip[0] - 0.015 * onboard_w + 0.025 * weight_shift * onboard_w, 0.94 * height)
    r_knee = (r_hip[0] + 0.015 * onboard_w + 0.025 * weight_shift * onboard_w, 0.94 * height)

    l_palm = (riser_lx, hand_ly)
    r_palm = (riser_rx, hand_ry)
    l_elbow = (
        0.52 * l_sh[0] + 0.48 * l_palm[0] - 0.065 * onboard_w,
        0.52 * l_sh[1] + 0.48 * l_palm[1] + 0.055 * height,
    )
    r_elbow = (
        0.52 * r_sh[0] + 0.48 * r_palm[0] + 0.065 * onboard_w,
        0.52 * r_sh[1] + 0.48 * r_palm[1] + 0.055 * height,
    )

    head_cx = sh_cx + 0.035 * weight_shift * onboard_w
    head_cy = 0.46 * height
    nose_pt = (head_cx + 0.025 * weight_shift * onboard_w, head_cy + 0.01 * height)

    # 1. Structural flight suit silhouette & bone segments (neutral low-chroma slate)
    draw.polygon([l_sh, r_sh, r_hip, l_hip], fill=(28, 38, 52), outline=(64, 78, 96))
    draw.line([l_hip, l_knee], fill=(64, 78, 96), width=6)
    draw.line([r_hip, r_knee], fill=(64, 78, 96), width=6)
    draw.line([l_sh, l_elbow], fill=(70, 86, 106), width=5)
    draw.line([l_elbow, l_palm], fill=(70, 86, 106), width=5)
    draw.line([r_sh, r_elbow], fill=(70, 86, 106), width=5)
    draw.line([r_elbow, r_palm], fill=(70, 86, 106), width=5)
    draw.ellipse(
        [head_cx - 14, head_cy - 16, head_cx + 14, head_cy + 14],
        fill=(32, 44, 58),
        outline=(75, 92, 112),
    )

    # 2. Anatomical keypoint markers for CV skeleton extraction
    # Nose / visor & knees (orange)
    for ox, oy in [nose_pt, l_knee, r_knee]:
        draw.ellipse([ox - 7, oy - 7, ox + 7, oy + 7], fill=(250, 115, 40))

    # Hips (magenta)
    for hx, hy in [l_hip, r_hip]:
        draw.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(220, 50, 200))

    # Shoulders (cyan, symmetric around sh_cx)
    for sx, sy in [l_sh, r_sh]:
        draw.ellipse([sx - 9, sy - 9, sx + 9, sy + 9], fill=(35, 205, 230))

    # Elbows (emerald green)
    for ex, ey in [l_elbow, r_elbow]:
        draw.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=(40, 225, 95))

    # Palms / brake toggles (yellow)
    r_hand = 11
    draw.ellipse(
        [riser_lx - r_hand, hand_ly - r_hand, riser_lx + r_hand, hand_ly + r_hand],
        fill=(248, 220, 40),
    )
    draw.ellipse(
        [riser_rx - r_hand, hand_ry - r_hand, riser_rx + r_hand, hand_ry + r_hand],
        fill=(248, 220, 40),
    )

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
    camera_mode: str = "side_chase",
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
        bl, br, ws, be = build_theo_full_command_law(maneuver_key, t)
        seed_frames.append(
            FrameObservation(
                frame_index=i,
                time_sec=round(t, 4),
                brake_left_norm=bl,
                brake_right_norm=br,
                brake_rate_norm_per_sec=0.0,
                weight_shift_norm=ws,
                big_ears_norm=be,
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
        camera_mode=camera_mode,
    )

    if camera_mode == "rear_chase":
        ref_sim = ParagliderPhysicsComparator.run_first_principles_lateral_engine(seed_trace)
    else:
        ref_sim = ParagliderPhysicsComparator.run_first_principles_two_body_engine(seed_trace)

    ref_times = np.array([p.time_sec for p in ref_sim])
    ref_pitches = np.array([p.pitch_deg for p in ref_sim])
    ref_rolls = np.array([p.roll_deg for p in ref_sim])

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
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "22",
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(n_frames):
        t = i * dt
        bl, br, ws, be = build_theo_full_command_law(maneuver_key, t)
        pitch_deg = float(np.interp(t, ref_times, ref_pitches))
        roll_deg = float(np.interp(t, ref_times, ref_rolls))
        horizon_deg = 1.5 * math.sin(0.7 * t)
        rgb = render_dual_view_frame_rgb(
            width,
            height,
            t,
            bl,
            br,
            ws,
            pitch_deg,
            roll_deg,
            horizon_deg,
            big_ears_norm=be,
            camera_mode=camera_mode,
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
        camera_mode=camera_mode,
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
    pilot centroid, suspension pendulum angle vector, horizon line, and brake/weight-shift gauges.
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
