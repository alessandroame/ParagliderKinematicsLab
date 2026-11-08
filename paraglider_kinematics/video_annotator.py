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


def render_annotated_frame(
    base_rgb: np.ndarray,
    obs,
    trace: ManeuverKinematicsTrace,
    active_event_text: Optional[str] = None,
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    """
    Overlays computer vision detection geometry, flight instrumentation HUD,
    and event badges directly onto an RGB video frame.
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

    # 3. Pilot hands bounding boxes in Onboard POV
    lx, ly = obs.left_hand_px
    rx, ry = obs.right_hand_px
    draw.rectangle([lx - 16, ly - 16, lx + 16, ly + 16], outline=(60, 220, 255), width=2)
    draw.rectangle([rx - 16, ry - 16, rx + 16, ry + 16], outline=(60, 220, 255), width=2)

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
    pov_title = f"ONBOARD POV | BRAKE TRAVEL"
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
