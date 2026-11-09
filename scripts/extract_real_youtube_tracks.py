from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import cv2
import imageio_ffmpeg
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from paraglider_kinematics.vision_tracker import DualViewParagliderTracker

MANEUVER_SPECS: List[Dict[str, Any]] = [
    {
        "id": "pitch_pendulum_uncaught",
        "yt_start_sec": 462,
        "decode_start_sec": 462.0,
        "duration_sec": 37.0,
    },
    {
        "id": "pitch_pendulum_active_catch",
        "yt_start_sec": 499,
        "decode_start_sec": 499.0,
        "duration_sec": 48.0,
    },
    {
        "id": "pitch_mistake_late_brake_stall",
        "yt_start_sec": 626,
        "decode_start_sec": 626.0,
        "duration_sec": 32.0,
    },
    {
        "id": "dynamic_turn_coordinated",
        "yt_start_sec": 290,
        "decode_start_sec": 290.0,
        "duration_sec": 45.0,
    },
    {
        "id": "dynamic_turn_mistake_no_ws",
        "yt_start_sec": 365,
        "decode_start_sec": 365.0,
        "duration_sec": 37.0,
    },
    {
        "id": "rolls_weightshift_and_brake",
        "yt_start_sec": 48,
        "decode_start_sec": 48.0,
        "duration_sec": 44.0,
    },
    {
        "id": "rolls_big_ears_pure_ws",
        "yt_start_sec": 185,
        "decode_start_sec": 185.0,
        "duration_sec": 45.0,
    },
]


def decode_clip_bgr_4fps(
    video_path: Path, start_sec: float, duration_sec: float, width: int = 480, height: int = 270
) -> np.ndarray:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        str(video_path),
        "-vf",
        f"fps=4,scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    n_frames = len(raw) // (width * height * 3)
    return np.frombuffer(raw[: n_frames * width * height * 3], dtype=np.uint8).reshape(
        (n_frames, height, width, 3)
    )


def despike_and_recompute_tuples(tuples: List[List[int]]) -> List[List[int]]:
    """
    Applies a 3-sample median filter to left/right hand pixel coordinates to suppress single-frame
    compression/glare spikes while preserving step pulls, then recomputes brake percentages and elbows
    strictly from the geometric riser-cable-hand triangle.
    """
    n = len(tuples)
    if n < 3:
        return tuples

    lxs = [t[4] for t in tuples]
    lys = [t[5] for t in tuples]
    rxs = [t[6] for t in tuples]
    rys = [t[7] for t in tuples]

    for i in range(1, n - 1):
        lx = int(round(float(np.median(lxs[i - 1 : i + 2]))))
        ly = int(round(float(np.median(lys[i - 1 : i + 2]))))
        rx = int(round(float(np.median(rxs[i - 1 : i + 2]))))
        ry = int(round(float(np.median(rys[i - 1 : i + 2]))))

        row = tuples[i]
        row[4], row[5] = lx, ly
        row[6], row[7] = rx, ry

        px, py = row[8], row[9]
        plx, ply = row[12], row[13]
        prx, pry = row[14], row[15]

        cable_len_l = math.hypot(lx - plx, max(0.0, ly - ply))
        cable_len_r = math.hypot(rx - prx, max(0.0, ry - pry))
        row[16] = int(round(100.0 * max(0.0, min(1.0, (cable_len_l - 8.0) / 96.0))))
        row[17] = int(round(100.0 * max(0.0, min(1.0, (cable_len_r - 8.0) / 84.0))))

        sh_l = (px - 15, py - 10)
        sh_r = (px + 15, py - 10)
        u_l = max(0.0, min(1.0, (ly - ply) / 92.0))
        u_r = max(0.0, min(1.0, (ry - pry) / 92.0))
        row[18] = int(round(0.5 * (sh_l[0] + lx) - 10.0 - 8.0 * u_l))
        row[19] = int(round(0.5 * (sh_l[1] + ly) + 12.0 + 6.0 * (1.0 - u_l)))
        row[20] = int(round(0.5 * (sh_r[0] + rx) + 10.0 + 8.0 * u_r))
        row[21] = int(round(0.5 * (sh_r[1] + ry) + 12.0 + 6.0 * (1.0 - u_r)))

    return tuples


def main() -> None:
    video_path = ROOT_DIR / "videos" / "ykuaAYzB2qU.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Real YouTube video not found at {video_path}")

    out_mjs = ROOT_DIR.parent / "ParagliderSim" / "core" / "real-youtube-cv-tracks.mjs"
    all_tracks: Dict[str, Dict[str, Any]] = {}

    for spec in MANEUVER_SPECS:
        m_id = spec["id"]
        frames_bgr = decode_clip_bgr_4fps(
            video_path, spec["decode_start_sec"], spec["duration_sec"]
        )
        # Compute static dark baseline mask (p75 < 55) from first 60 onboard frames
        v_stack = np.array(
            [cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 2] for f in frames_bgr[: min(60, len(frames_bgr))]]
        )
        static_dark = np.percentile(v_stack, 75, axis=0) < 55

        raw_tuples: List[List[int]] = []
        for f_bgr in frames_bgr:
            feat = DualViewParagliderTracker.extract_real_360_frame_features(f_bgr, static_dark)
            raw_tuples.append(feat["tuple22"])

        clean_tuples = despike_and_recompute_tuples(raw_tuples)
        all_tracks[m_id] = {
            "ytStartSec": spec["yt_start_sec"],
            "dtSec": 0.25,
            "frames": clean_tuples,
        }

        if m_id == "pitch_pendulum_active_catch":
            cv2.imwrite(str(ROOT_DIR / "videos" / "frame_08_22.png"), frames_bgr[12])
            cv2.imwrite(str(ROOT_DIR / "videos" / "frame_08_30.png"), frames_bgr[50])
            print(
                f"[{m_id}] Control 08:22 (idx=12, t=3.0s): "
                f"L={clean_tuples[12][4:6]} ({clean_tuples[12][16]}%), "
                f"R={clean_tuples[12][6:8]} ({clean_tuples[12][17]}%), "
                f"PulleyL={clean_tuples[12][12:14]}, PulleyR={clean_tuples[12][14:16]}"
            )
            print(
                f"[{m_id}] Control 08:30 (idx=48..52, t=12.0s): "
                f"L={clean_tuples[50][4:6]} ({clean_tuples[50][16]}%), "
                f"R={clean_tuples[50][6:8]} ({clean_tuples[50][17]}%), "
                f"PulleyL={clean_tuples[50][12:14]}, PulleyR={clean_tuples[50][14:16]}"
            )

    lines: List[str] = [
        "/**",
        " * Real OpenCV 4Hz multi-target tracks extracted from YouTube video ykuaAYzB2qU (480x270 normalized).",
        " * Generated by ParagliderKinematicsLab/scripts/extract_real_youtube_tracks.py using the geometric",
        " * Riser (Carabiner -> Pulley) -> Taut Brake Cable -> Hand/Glove -> Elbow -> Shoulder 360° fan detector.",
        " * Each frame tuple contains 22 integers:",
        " *   [0..3: canopy_x, canopy_y, canopy_w, canopy_h,",
        " *    4..5: left_hand_x, left_hand_y,",
        " *    6..7: right_hand_x, right_hand_y,",
        " *    8..11: pilot_x, pilot_y, pilot_w, pilot_h,",
        " *    12..13: pulley_left_x, pulley_left_y,",
        " *    14..15: pulley_right_x, pulley_right_y,",
        " *    16..17: brake_left_pct, brake_right_pct,",
        " *    18..19: elbow_left_x, elbow_left_y,",
        " *    20..21: elbow_right_x, elbow_right_y]",
        " */",
        "export const REAL_YOUTUBE_CV_TRACKS = {",
    ]
    for m_id, data in all_tracks.items():
        lines.append(f'  "{m_id}": {{')
        lines.append(f'    ytStartSec: {data["ytStartSec"]},')
        lines.append(f'    dtSec: {data["dtSec"]},')
        lines.append("    frames: [")
        for row in data["frames"]:
            lines.append("      [" + ",".join(str(v) for v in row) + "],")
        lines.append("    ],")
        lines.append("  },")
    lines.append("};")
    lines.append("")

    out_mjs.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {out_mjs} ({out_mjs.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
