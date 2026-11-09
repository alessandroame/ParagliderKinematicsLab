from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .physics_replay import ParagliderPhysicsComparator
from .schema import ManeuverKinematicsTrace, WingGeometrySpec
from .theo_de_blic_dataset import (
    build_theo_full_command_law,
    render_dual_view_frame_rgb,
)
from .vision_tracker import DualViewParagliderTracker


def draw_pilot_skeleton_overlay(draw: ImageDraw.ImageDraw, obs) -> None:
    """
    Draws the complete articulated pilot skeleton (`nose`, `l_ear`, `r_ear`, `l_shoulder`, `r_shoulder`,
    `l_elbow`, `r_elbow`, `l_wrist`, `r_wrist`, `l_palm`, `r_palm`, `l_hip`, `r_hip`, `l_knee`, `r_knee`)
    with confidence filtering (`conf >= 0.5`), risers + taut brake cables, torso/pelvis axes, and turn gaze vector.
    """
    sk = getattr(obs, "pilot_skeleton_px", None) or {}

    def _pt(key: str):
        val = sk.get(key)
        if val and len(val) >= 3 and val[2] >= 0.5:
            return (float(val[0]), float(val[1]))
        return None

    l_sh, r_sh = _pt("l_shoulder"), _pt("r_shoulder")
    l_el, r_el = _pt("l_elbow"), _pt("r_elbow")
    l_wr, r_wr = _pt("l_wrist"), _pt("r_wrist")
    l_pa, r_pa = _pt("l_palm"), _pt("r_palm")
    l_hi, r_hi = _pt("l_hip"), _pt("r_hip")
    l_kn, r_kn = _pt("l_knee"), _pt("r_knee")
    nose, l_ear, r_ear = _pt("nose"), _pt("l_ear"), _pt("r_ear")

    if l_sh and r_sh:
        mid_sh = (0.5 * (l_sh[0] + r_sh[0]), 0.5 * (l_sh[1] + r_sh[1]))
        # Shoulder axis
        draw.line([l_sh, r_sh], fill=(245, 190, 40), width=3)
        if l_hi and r_hi:
            mid_hi = (0.5 * (l_hi[0] + r_hi[0]), 0.5 * (l_hi[1] + r_hi[1]))
            # Torso sides, hip axis, and spine
            draw.line([l_sh, l_hi], fill=(148, 163, 184), width=2)
            draw.line([r_sh, r_hi], fill=(148, 163, 184), width=2)
            draw.line([l_hi, r_hi], fill=(192, 132, 252), width=3)
            draw.line([mid_sh, mid_hi], fill=(250, 204, 21), width=2)
        if nose:
            draw.line([mid_sh, nose], fill=(226, 232, 240), width=2)
            draw.ellipse([nose[0] - 12, nose[1] - 13, nose[0] + 12, nose[1] + 11], outline=(56, 189, 248), width=2)
            if l_ear and r_ear:
                draw.line([l_ear, r_ear], fill=(148, 163, 184), width=1)
            # Gaze direction vector in turn
            ws = getattr(obs, "weight_shift_norm", 0.0) or 0.0
            gx = nose[0] + 22.0 * ws
            gy = nose[1] - 6.0
            draw.line([nose, (gx, gy)], fill=(250, 204, 21), width=2)

    # Legs (`hip -> knee`): only draw when real knee keypoint is confident (`conf >= 0.5`);
    # in ONBOARD_360_KNEE mode, highlight the 360 camera knee mount instead of fake vertical legs.
    shot_mode = getattr(obs, "shot_mode", "") or ""
    if l_hi and l_kn:
        draw.line([l_hi, l_kn], fill=(168, 85, 247), width=2)
        if shot_mode == "ONBOARD_360_KNEE":
            draw.ellipse([l_kn[0] - 6, l_kn[1] - 6, l_kn[0] + 6, l_kn[1] + 6], outline=(56, 189, 248), width=2)
    if r_hi and r_kn and shot_mode != "ONBOARD_360_KNEE":
        draw.line([r_hi, r_kn], fill=(168, 85, 247), width=2)

    # Risers (`carabiner -> pulley`) & Taut Brake Cables (`pulley -> palm`)
    lx, ly = l_pa if l_pa else obs.left_hand_px
    rx, ry = r_pa if r_pa else obs.right_hand_px
    if l_sh and r_sh:
        pulley_l = (l_sh[0] - 8.0, min(l_sh[1] - 28.0, ly - 4.0))
        pulley_r = (r_sh[0] + 8.0, min(r_sh[1] - 28.0, ry - 4.0))
        carab_l = (l_sh[0] - 4.0, l_sh[1] + 18.0)
        carab_r = (r_sh[0] + 4.0, r_sh[1] + 18.0)
        draw.line([carab_l, pulley_l], fill=(148, 163, 184), width=2)
        draw.line([carab_r, pulley_r], fill=(148, 163, 184), width=2)
        draw.ellipse([pulley_l[0] - 3, pulley_l[1] - 3, pulley_l[0] + 3, pulley_l[1] + 3], fill=(250, 204, 21))
        draw.ellipse([pulley_r[0] - 3, pulley_r[1] - 3, pulley_r[0] + 3, pulley_r[1] + 3], fill=(250, 204, 21))
        draw.line([pulley_l, (lx, ly)], fill=(250, 204, 21), width=2)
        draw.line([pulley_r, (rx, ry)], fill=(250, 204, 21), width=2)

    # Left arm kinematic chain (`l_shoulder -> l_elbow -> l_wrist -> l_palm`) in Cyan
    l_chain = [p for p in (l_sh, l_el, l_wr, l_pa) if p is not None]
    for i in range(len(l_chain) - 1):
        draw.line([l_chain[i], l_chain[i + 1]], fill=(0, 240, 255), width=3)

    # Right arm kinematic chain (`r_shoulder -> r_elbow -> r_wrist -> r_palm`) in Green
    r_chain = [p for p in (r_sh, r_el, r_wr, r_pa) if p is not None]
    for i in range(len(r_chain) - 1):
        draw.line([r_chain[i], r_chain[i + 1]], fill=(16, 185, 129), width=3)

    # Joint nodes (`conf >= 0.5`)
    for k, color in [
        ("l_shoulder", (0, 240, 255)),
        ("r_shoulder", (16, 185, 129)),
        ("l_elbow", (0, 240, 255)),
        ("r_elbow", (16, 185, 129)),
        ("l_wrist", (0, 240, 255)),
        ("r_wrist", (16, 185, 129)),
        ("l_hip", (192, 132, 252)),
        ("r_hip", (192, 132, 252)),
        ("l_knee", (251, 146, 60)),
        ("r_knee", (251, 146, 60)),
        ("nose", (250, 204, 21)),
    ]:
        pt = _pt(k)
        if pt:
            draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill=color, outline=(15, 23, 42))

    # Brake toggle palm boxes
    draw.rectangle([lx - 14, ly - 14, lx + 14, ly + 14], outline=(0, 240, 255), width=2)
    draw.rectangle([rx - 14, ry - 14, rx + 14, ry + 14], outline=(16, 185, 129), width=2)


def render_annotated_frame(
    base_rgb: np.ndarray,
    obs,
    trace: ManeuverKinematicsTrace,
    active_event_text: Optional[str] = None,
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    """
    Overlays computer vision detection geometry, full articulated pilot skeleton,
    flight instrumentation HUD, and event badges directly onto an RGB video frame.
    """
    img = Image.fromarray(base_rgb)
    draw = ImageDraw.Draw(img)
    chase_w = int(0.68 * width)
    onboard_w = width - chase_w

    # 1. Horizon reference line (Cyan)
    mid_y = 0.72 * height
    slope = math.tan(math.radians(obs.horizon_tilt_deg))
    y0 = mid_y - slope * (0.5 * chase_w)
    y1 = mid_y + slope * (0.5 * chase_w)
    draw.line([(0, y0), (chase_w, y1)], fill=(0, 240, 255), width=2)

    # 2. Pilot and Canopy Centroids & Pendulum Axis
    px, py = obs.pilot_centroid_px
    cx, cy = obs.canopy_centroid_px
    grav_rad = math.radians(obs.horizon_tilt_deg)
    gx = px - 180.0 * math.sin(grav_rad)
    gy = py - 180.0 * math.cos(grav_rad)

    # Gravity plumb-line (Green)
    draw.line([(px, py), (gx, gy)], fill=(60, 230, 100), width=2)
    # Suspension pendulum line (Yellow)
    draw.line([(px, py), (cx, cy)], fill=(255, 230, 40), width=3)

    # Canopy contour / span bounds
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
        draw.polygon(oval_pts, outline=(0, 255, 140), width=2)
    else:
        draw.ellipse([cx - 40, cy - 20, cx + 40, cy + 20], outline=(0, 255, 140), width=2)

    # Pilot centroid box (Red)
    draw.ellipse([px - 16, py - 14, px + 16, py + 14], outline=(255, 80, 80), width=2)
    draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 80, 80))
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 255, 140))

    # 3. Full articulated pilot skeleton & hands in Onboard POV
    draw_pilot_skeleton_overlay(draw, obs)

    # Onboard ROI border
    draw.line([(chase_w, 0), (chase_w, height)], fill=(80, 90, 110), width=2)

    # 4. Top Diagnostic HUD Bar
    hud_bg = [(8, 8), (chase_w - 8, 64)]
    draw.rectangle(hud_bg, fill=(10, 16, 26, 230), outline=(50, 70, 100), width=1)

    if trace.camera_mode == "rear_chase":
        l1 = (
            f"FRAME #{obs.frame_index:03d} | t = {obs.time_sec:5.2f}s | "
            f"BRAKE L/R = {obs.brake_left_norm*100:3.0f}%/{obs.brake_right_norm*100:3.0f}% | "
            f"WS = {obs.weight_shift_norm*100:+4.0f}% | EARS = {obs.big_ears_norm*100:3.0f}%"
        )
        l2 = (
            f"ROLL = {obs.roll_deg:+6.1f} deg (rate={obs.roll_rate_deg_sec:+6.1f} deg/s) | "
            f"PITCH = {obs.pitch_deg:+5.1f} deg | SPAN = {obs.canopy_span_px:5.1f} px"
        )
    else:
        sym_pct = 50.0 * (obs.brake_left_norm + obs.brake_right_norm)
        l1 = (
            f"FRAME #{obs.frame_index:03d} | t = {obs.time_sec:5.2f}s | "
            f"BRAKE SYM = {sym_pct:4.1f}% (rate={obs.brake_rate_norm_per_sec*100:+5.0f}%/s)"
        )
        l2 = (
            f"CANOPY PITCH = {obs.pitch_deg:+6.1f} deg (rate={obs.pitch_rate_deg_sec:+6.1f} deg/s) | "
            f"SURGE = {obs.canopy_forward_offset_m:+5.2f} m"
        )

    draw.text((16, 14), l1, fill=(235, 245, 255))
    draw.text((16, 38), l2, fill=(255, 215, 60))

    # 5. Onboard POV Overlay Banner
    draw.rectangle([(chase_w + 8, 8), (width - 8, 38)], fill=(10, 16, 26), outline=(50, 70, 100))
    pov_title = "ONBOARD POV | SKELETON & BRAKES"
    draw.text((chase_w + 14, 16), pov_title, fill=(120, 210, 255))

    # Brake travel mini-gauges in Onboard ROI
    bl_h = int(obs.brake_left_norm * 140)
    br_h = int(obs.brake_right_norm * 140)
    # Left brake vertical bar
    draw.rectangle([(chase_w + 18, 120), (chase_w + 30, 260)], outline=(80, 90, 110))
    draw.rectangle([(chase_w + 19, 260 - bl_h), (chase_w + 29, 260)], fill=(0, 200, 255))
    draw.text((chase_w + 14, 265), f"L:{obs.brake_left_norm*100:.0f}%", fill=(200, 220, 255))

    # Right brake vertical bar
    draw.rectangle([(width - 32, 120), (width - 20, 260)], outline=(80, 90, 110))
    draw.rectangle([(width - 31, 260 - br_h), (width - 21, 260)], fill=(0, 220, 120))
    draw.text((width - 38, 265), f"R:{obs.brake_right_norm*100:.0f}%", fill=(200, 220, 255))

    # 6. Active Physical Event Banner (if triggered)
    if active_event_text:
        box_w = 480
        bx0 = (chase_w - box_w) // 2
        bx1 = bx0 + box_w
        by0 = height - 60
        by1 = height - 12
        draw.rectangle([(bx0, by0), (bx1, by1)], fill=(200, 30, 30), outline=(255, 230, 80), width=2)
        draw.text((bx0 + 16, by0 + 8), f"EVENT DETECTED: {active_event_text}", fill=(255, 255, 255))

    return np.asarray(img)


def export_annotated_mp4(
    trace: ManeuverKinematicsTrace,
    output_path: str | Path,
    width: int = 960,
    height: int = 540,
    fps: float = 30.0,
) -> Path:
    """
    Renders and encodes a full annotated MP4 video with burnt-in Computer Vision
    detection geometry, telemetry HUD, and physical event alerts.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Map events to frame indices and descriptions
    event_by_frame: Dict[int, str] = {}
    for ev in trace.events:
        event_by_frame[ev.frame_index] = f"[{ev.event_type.upper()}] {ev.description[:42]}"

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "22",
        str(out_file),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    active_event_banner: Optional[str] = None
    event_display_counter = 0

    for idx, obs in enumerate(trace.frames):
        if idx in event_by_frame:
            active_event_banner = event_by_frame[idx]
            event_display_counter = int(fps * 0.9)  # display event banner for ~0.9s
        elif event_display_counter > 0:
            event_display_counter -= 1
        else:
            active_event_banner = None

        bl, br, ws, be = build_theo_full_command_law(trace.maneuver_id, obs.time_sec)
        base_rgb = render_dual_view_frame_rgb(
            width=width,
            height=height,
            t_sec=obs.time_sec,
            brake_l=bl,
            brake_r=br,
            weight_shift=ws,
            pitch_deg=obs.pitch_deg,
            roll_deg=obs.roll_deg,
            horizon_tilt_deg=obs.horizon_tilt_deg,
            big_ears_norm=be,
            camera_mode=trace.camera_mode,
        )
        annotated_rgb = render_annotated_frame(
            base_rgb=base_rgb,
            obs=obs,
            trace=trace,
            active_event_text=active_event_banner,
            width=width,
            height=height,
        )
        proc.stdin.write(annotated_rgb.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"FFmpeg encoding failed: {err}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Export annotated flight video with CV & telemetry HUD")
    parser.add_argument("--trace", type=str, required=True, help="Path to maneuver trace JSON")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 path")
    args = parser.parse_args()

    trace_path = Path(args.trace)
    trace = ManeuverKinematicsTrace.load_json(trace_path)
    out_path = Path(args.out) if args.out else trace_path.parent.parent / "videos" / f"{trace.maneuver_id}_annotated.mp4"

    print(f"Rendering annotated video for '{trace.maneuver_name}' ({len(trace.frames)} frames)...")
    exported = export_annotated_mp4(trace, out_path)
    print(f"[OK] Exported annotated video: {exported} ({exported.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
