from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .schema import (
    CommandTimingEvent,
    FrameObservation,
    ManeuverKinematicsTrace,
    PhaseLagMetrics,
    WingGeometrySpec,
)


class DualViewParagliderTracker:
    """
    Frame-by-frame Computer Vision recognizer for paragliding tutorial videos
    with External Chase View (Side or Rear/Front) + Onboard Pilot POV Split-Screen/PIP.

    Extracts:
      1. Horizon line tilt angle via vertical Sobel gradient ridge regression.
      2. Paraglider canopy centroid, oriented ellipse axes, open span width (for Big Ears detection),
         and bank/pitch attitude via 2nd-order image moments.
      3. Pilot/harness centroid suspended below the canopy and true pendulum angle (pitch or roll) wrt gravity.
      4. Left/Right pilot brake hand positions wrt pulley (0%) and carabiner (100%) + torso lean weight-shift
         in the Onboard POV ROI.
      5. Command application timing events, weight-shift lead time, and canopy reaction phase lags.
    """

    def __init__(
        self,
        wing: Optional[WingGeometrySpec] = None,
        chase_roi_norm: Tuple[float, float, float, float] = (0.0, 0.0, 0.68, 1.0),
        onboard_roi_norm: Tuple[float, float, float, float] = (0.68, 0.0, 1.0, 1.0),
        pulley_y_norm: float = 0.20,
        carabiner_y_norm: float = 0.78,
        ignore_pip: bool = True,
    ) -> None:
        self.wing = wing or WingGeometrySpec()
        self.chase_roi_norm = chase_roi_norm
        self.onboard_roi_norm = onboard_roi_norm
        self.pulley_y_norm = pulley_y_norm
        self.carabiner_y_norm = carabiner_y_norm
        self.ignore_pip = ignore_pip

    @staticmethod
    def extract_video_frames_ffmpeg(
        video_path: str | Path,
        fps: float = 30.0,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        width: int = 960,
        height: int = 540,
    ) -> List[np.ndarray]:
        """Decodes video frames into RGB numpy arrays (H, W, 3) using ffmpeg pipe."""
        cmd = ["ffmpeg", "-v", "error"]
        if start_sec is not None:
            cmd += ["-ss", f"{start_sec:.3f}"]
        cmd += ["-i", str(video_path)]
        if duration_sec is not None:
            cmd += ["-t", f"{duration_sec:.3f}"]
        cmd += [
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        frame_bytes = width * height * 3
        raw = proc.stdout
        n_frames = len(raw) // frame_bytes
        if n_frames == 0:
            return []
        arr = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8)
        return list(arr.reshape((n_frames, height, width, 3)))

    @staticmethod
    def detect_horizon_tilt_deg(chase_rgb: np.ndarray) -> Tuple[float, float]:
        """
        Estimates horizon tilt angle (deg) and vertical intercept (px) in the chase view
        by finding the strongest vertical sky-to-terrain luminance transition across columns.
        """
        gray = (
            0.299 * chase_rgb[:, :, 0].astype(np.float32)
            + 0.587 * chase_rgb[:, :, 1].astype(np.float32)
            + 0.114 * chase_rgb[:, :, 2].astype(np.float32)
        )
        h, w = gray.shape
        y_min = int(0.45 * h)
        y_max = int(0.88 * h)
        band = gray[y_min:y_max, :]
        grad_y = band[:-2, :] - band[2:, :]
        col_indices = np.linspace(int(0.05 * w), int(0.95 * w), 32, dtype=int)
        xs = []
        ys = []
        for col in col_indices:
            col_slice = grad_y[:, max(0, col - 2) : min(w, col + 3)].mean(axis=1)
            idx = int(np.argmax(col_slice))
            if col_slice[idx] > 8.0:
                xs.append(float(col))
                ys.append(float(y_min + idx + 1))
        if len(xs) < 6:
            return 0.0, float(0.72 * h)
        x_arr = np.array(xs, dtype=np.float64)
        y_arr = np.array(ys, dtype=np.float64)
        med_y = np.median(y_arr)
        inliers = np.abs(y_arr - med_y) < (0.10 * h)
        if np.sum(inliers) >= 4:
            x_arr = x_arr[inliers]
            y_arr = y_arr[inliers]
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        tilt_deg = math.degrees(math.atan(slope))
        mid_y = slope * (0.5 * w) + intercept
        return float(tilt_deg), float(mid_y)

    @staticmethod
    def _moment_centroid_and_axes(
        mask: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """
        Computes centroid (cx, cy), major-axis orientation angle (deg),
        equivalent major span (px), and pixel area from a binary mask.
        """
        ys, xs = np.nonzero(mask)
        area = float(len(xs))
        if area < 8.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        dx = xs.astype(np.float64) - cx
        dy = ys.astype(np.float64) - cy
        mu20 = float(np.mean(dx * dx))
        mu02 = float(np.mean(dy * dy))
        mu11 = float(np.mean(dx * dy))
        angle_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
        common = math.sqrt((mu20 - mu02) ** 2 + 4.0 * (mu11**2))
        lambda1 = max(1.0, 0.5 * (mu20 + mu02 + common))
        major_span_px = 4.0 * math.sqrt(lambda1)
        return cx, cy, math.degrees(angle_rad), major_span_px, area

    def detect_canopy_and_pilot(
        self,
        chase_rgb: np.ndarray,
        horizon_tilt_deg: float,
        horizon_mid_y: float,
        camera_mode: str = "side_chase",
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float, float, float, float, float]:
        """
        Detects the ram-air canopy and suspended pilot in the External Chase ROI.
        Supports:
          - `side_chase`: extracts longitudinal pitch_deg from suspension vector.
          - `rear_chase`: extracts lateral roll_deg from suspension vector & canopy span axis,
             longitudinal pitch_deg from vertical line foreshortening, and big_ears_norm from open span width.
        Returns:
          canopy_xy, pilot_xy, pitch_deg, roll_deg, big_ears_norm, span_px, confidence
        """
        r = chase_rgb[:, :, 0].astype(np.float32)
        g = chase_rgb[:, :, 1].astype(np.float32)
        b = chase_rgb[:, :, 2].astype(np.float32)

        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        chroma = max_rgb - min_rgb
        canopy_mask = (chroma > 45.0) & (r > b + 35.0) & (max_rgb > 110.0)

        cx, cy, canopy_axis_deg, span_px, c_area = self._moment_centroid_and_axes(canopy_mask)

        h, w, _ = chase_rgb.shape
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        y_grid, x_grid = np.ogrid[:h, :w]
        pilot_search_mask = (luma < 32.0) & (chroma < 28.0) & (y_grid < horizon_mid_y - 6.0)
        if c_area >= 20.0:
            pilot_search_mask &= (y_grid > cy + 20.0) & (np.abs(x_grid - cx) < 0.48 * w)

        px, py, _, _, p_area = self._moment_centroid_and_axes(pilot_search_mask)

        if c_area < 20.0 or p_area < 8.0:
            return (0.5 * w, 0.25 * h), (0.5 * w, 0.58 * h), 0.0, 0.0, 0.0, 0.0, 0.15

        dx = cx - px
        dy_up = py - cy

        if camera_mode == "rear_chase":
            # In rear chase view (+X right, -Y up):
            # Right bank (+roll_deg > 0) shifts canopy centroid right (cx > px -> dx > 0)
            raw_bank_deg = math.degrees(math.atan2(dx, max(12.0, dy_up)))
            roll_deg = raw_bank_deg - horizon_tilt_deg
            # Estimate big_ears_norm from open canopy span reduction relative to nominal full span (176 px)
            nominal_full_span_px = 0.34 * h * 0.96
            span_ratio = span_px / max(20.0, nominal_full_span_px)
            big_ears_norm = float(np.clip((0.93 - span_ratio) / 0.19, 0.0, 1.0))
            # Estimate pitch_deg from vertical line length deviation wrt nominal 0.34*h
            line_len_px = math.hypot(dx, dy_up)
            nom_line_px = 0.34 * h
            pitch_deg = float(np.clip((line_len_px - nom_line_px) / 1.35, -35.0, 35.0))
        else:
            raw_pendulum_angle_deg = math.degrees(math.atan2(-dx, max(12.0, dy_up)))
            pitch_deg = raw_pendulum_angle_deg - horizon_tilt_deg
            roll_deg = 0.0
            big_ears_norm = 0.0

        conf = min(1.0, (c_area / 250.0) * (p_area / 35.0))
        return (
            (cx, cy),
            (px, py),
            float(pitch_deg),
            float(roll_deg),
            float(big_ears_norm),
            float(span_px),
            float(conf),
        )

    @staticmethod
    def detect_pip_region(frames_rgb: List[np.ndarray]) -> Optional[List[int]]:
        """
        Detects a persistent Picture-in-Picture (PiP) corner sub-window (4%..28% of frame area)
        via persistent rectangular border gradients across sample frames.
        Note: Because PiP sub-videos in edited tutorials are not reliably synchronized with the main
        video, any detected PiP region is strictly ignored/masked out and never used for telemetry.
        """
        if not frames_rgb:
            return None
        sample_indices = np.linspace(0, len(frames_rgb) - 1, min(8, len(frames_rgb)), dtype=int)
        sample = np.stack([frames_rgb[i].astype(np.float32).mean(axis=2) for i in sample_indices], axis=0)
        mean_gray = sample.mean(axis=0)
        h, w = mean_gray.shape

        gx = np.abs(np.diff(mean_gray, axis=1, prepend=mean_gray[:, :1]))
        gy = np.abs(np.diff(mean_gray, axis=0, prepend=mean_gray[:1, :]))

        # Check candidate corner insets (excluding the 68%/32% main Dual-View split)
        candidates = [
            (int(0.02 * w), int(0.02 * h), int(0.30 * w), int(0.32 * h)),
            (int(0.02 * w), int(0.66 * h), int(0.30 * w), int(0.96 * h)),
        ]
        for x0, y0, x1, y1 in candidates:
            top_edge = float(gy[max(0, y0 - 1) : min(h, y0 + 2), x0:x1].mean())
            bot_edge = float(gy[max(0, y1 - 1) : min(h, y1 + 2), x0:x1].mean())
            left_edge = float(gx[y0:y1, max(0, x0 - 1) : min(w, x0 + 2)].mean())
            right_edge = float(gx[y0:y1, max(0, x1 - 1) : min(w, x1 + 2)].mean())
            if min(top_edge, bot_edge, left_edge, right_edge) > 28.0:
                return [x0, y0, x1, y1]
        return None

    @staticmethod
    def mask_pip_region(frame_rgb: np.ndarray, pip_rect: Optional[List[int]]) -> np.ndarray:
        """Masks out an unsynchronized PiP inset region so it cannot pollute main-video CV tracking."""
        if not pip_rect or len(pip_rect) != 4:
            return frame_rgb
        masked = frame_rgb.copy()
        x0, y0, x1, y1 = pip_rect
        h, w, _ = masked.shape
        x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
        y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
        if x1 > x0 and y1 > y0:
            masked[y0:y1, x0:x1] = np.median(masked, axis=(0, 1)).astype(np.uint8)
        return masked

    def detect_pilot_skeleton(
        self, onboard_rgb: np.ndarray, ox0: int = 0, oy0: int = 0
    ) -> Tuple[
        float,
        float,
        float,
        Tuple[float, float],
        Tuple[float, float],
        Dict[str, List[float]],
        str,
        float,
        float,
    ]:
        """
        One-time anatomical pilot skeleton extractor for the Onboard POV ROI.
        Extracts 15 keypoints (`nose`, `l_ear`, `r_ear`, `l_shoulder`, `r_shoulder`,
        `l_elbow`, `r_elbow`, `l_wrist`, `r_wrist`, `l_palm`, `r_palm`, `l_hip`, `r_hip`,
        `l_knee`, `r_knee`) with top-down kinematic chain constraints (`Spalla -> Gomito -> Polso -> Mano`
        and `Spalle <-> Anche`), normalizing brake travel on bi-scapular width (`dist(l_shoulder, r_shoulder)`).
        """
        h, w, _ = onboard_rgb.shape
        r = onboard_rgb[:, :, 0].astype(np.float32)
        g = onboard_rgb[:, :, 1].astype(np.float32)
        b = onboard_rgb[:, :, 2].astype(np.float32)

        # 1. Shoulders & Upper Torso Axis (`l_shoulder`, `r_shoulder`)
        shoulder_mask = (b > 170.0) & (g > 150.0) & (r < 95.0)
        tx, ty, _, _, t_area = self._moment_centroid_and_axes(shoulder_mask)
        if t_area >= 10.0:
            split_x = max(1, min(w - 1, int(round(tx))))
            ls_mask = shoulder_mask.copy()
            ls_mask[:, split_x:] = False
            rs_mask = shoulder_mask.copy()
            rs_mask[:, :split_x] = False
            lsx, lsy, _, _, ls_area = self._moment_centroid_and_axes(ls_mask)
            rsx, rsy, _, _, rs_area = self._moment_centroid_and_axes(rs_mask)
            if ls_area >= 4.0 and rs_area >= 4.0:
                l_sh = (lsx, lsy, min(1.0, ls_area / 65.0))
                r_sh = (rsx, rsy, min(1.0, rs_area / 65.0))
            else:
                sh_hw = 0.16 * w
                l_sh = (tx - sh_hw, ty, 0.75)
                r_sh = (tx + sh_hw, ty, 0.75)
        elif t_area >= 6.0:
            sh_hw = 0.16 * w
            l_sh = (tx - sh_hw, ty, 0.70)
            r_sh = (tx + sh_hw, ty, 0.70)
        else:
            tx, ty = 0.50 * w, 0.60 * h
            sh_hw = 0.16 * w
            l_sh = (tx - sh_hw, ty, 0.35)
            r_sh = (tx + sh_hw, ty, 0.35)

        sh_cx = 0.5 * (l_sh[0] + r_sh[0])
        sh_cy = 0.5 * (l_sh[1] + r_sh[1])
        bi_shoulder_px = max(20.0, math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]))
        shoulder_roll_deg = math.degrees(math.atan2(r_sh[1] - l_sh[1], max(10.0, r_sh[0] - l_sh[0])))

        # 2. Hips / Harness Pelvis Axis (`l_hip`, `r_hip`)
        hip_mask = (r > 160.0) & (g < 105.0) & (b > 140.0)
        hip_mask[: int(0.68 * h), :] = False
        hip_mask[int(0.90 * h) :, :] = False
        hx, hy, _, _, h_area = self._moment_centroid_and_axes(hip_mask)
        if h_area >= 8.0:
            h_split = max(1, min(w - 1, int(round(hx))))
            lh_mask = hip_mask.copy()
            lh_mask[:, h_split:] = False
            rh_mask = hip_mask.copy()
            rh_mask[:, :h_split] = False
            lhx, lhy, _, _, lh_area = self._moment_centroid_and_axes(lh_mask)
            rhx, rhy, _, _, rh_area = self._moment_centroid_and_axes(rh_mask)
            if lh_area >= 3.0 and rh_area >= 3.0:
                l_hip = (lhx, lhy, min(1.0, lh_area / 55.0))
                r_hip = (rhx, rhy, min(1.0, rh_area / 55.0))
            else:
                hip_hw = 0.11 * w
                l_hip = (hx - hip_hw, hy, 0.70)
                r_hip = (hx + hip_hw, hy, 0.70)
        else:
            # Cocoon fallback when hips are enclosed inside harness speedbag (`hip_conf < 0.65`)
            hx = 0.5 * w + 0.64 * (sh_cx - 0.5 * w)
            hy = 0.82 * h
            hip_hw = 0.11 * w
            l_hip = (hx - hip_hw, hy, 0.45)
            r_hip = (hx + hip_hw, hy, 0.45)

        hip_cx = 0.5 * (l_hip[0] + r_hip[0])
        hip_cy = 0.5 * (l_hip[1] + r_hip[1])

        # 3. Torso Lean Angle & Harness Weight-Shift (`torso_lean_deg`, `weight_shift`)
        torso_lean_deg = math.degrees(math.atan2(sh_cx - 0.5 * w, max(20.0, hip_cy - sh_cy)))
        if t_area >= 6.0:
            ws_shoulder = (sh_cx - 0.5 * w) / (0.22 * w)
            hip_conf = 0.5 * (l_hip[2] + r_hip[2])
            if hip_conf >= 0.65:
                ws_hip = (hip_cx - 0.5 * w) / (0.14 * w)
                ws_raw = 0.85 * ws_shoulder + 0.15 * ws_hip
            else:
                ws_raw = ws_shoulder
            weight_shift = float(np.clip(ws_raw, -1.0, 1.0))
        else:
            weight_shift = 0.0

        # 4. Brake Hands (`l_palm`, `r_palm`) with Top-Down Kinematic Reach Constraint
        toggle_mask = (r > 180.0) & (g > 165.0) & (b < 110.0)
        left_half = toggle_mask.copy()
        left_half[:, w // 2 :] = False
        right_half = toggle_mask.copy()
        right_half[:, : w // 2] = False

        lx, ly, _, _, l_area = self._moment_centroid_and_axes(left_half)
        rx, ry, _, _, r_area = self._moment_centroid_and_axes(right_half)

        y_pulley = self.pulley_y_norm * h
        y_carabiner = self.carabiner_y_norm * h
        travel_px = max(10.0, y_carabiner - y_pulley)

        if l_area < 5.0:
            lx, ly, l_conf = 0.28 * w, y_pulley, 0.35
        else:
            l_conf = min(1.0, l_area / 95.0)

        if r_area < 5.0:
            rx, ry, r_conf = 0.72 * w, y_pulley, 0.35
        else:
            r_conf = min(1.0, r_area / 95.0)

        # Enforce anatomical arm reach from shoulder (`Spalla -> Mano`)
        max_arm_reach = 3.8 * bi_shoulder_px
        dist_l = math.hypot(lx - l_sh[0], ly - l_sh[1])
        if dist_l > max_arm_reach and dist_l > 1e-3:
            scale_l = max_arm_reach / dist_l
            lx = l_sh[0] + (lx - l_sh[0]) * scale_l
            ly = l_sh[1] + (ly - l_sh[1]) * scale_l

        dist_r = math.hypot(rx - r_sh[0], ry - r_sh[1])
        if dist_r > max_arm_reach and dist_r > 1e-3:
            scale_r = max_arm_reach / dist_r
            rx = r_sh[0] + (rx - r_sh[0]) * scale_r
            ry = r_sh[1] + (ry - r_sh[1]) * scale_r

        # Normalize brake travel using bi-scapular width (`dist(l_shoulder, r_shoulder)`) & pulley-carabiner span
        nom_bi_shoulder = 0.32 * w
        shoulder_scaled_travel = bi_shoulder_px * (travel_px / max(10.0, nom_bi_shoulder))
        eff_travel_px = 0.5 * travel_px + 0.5 * shoulder_scaled_travel

        brake_left = float(np.clip((ly - y_pulley) / eff_travel_px, 0.0, 1.05))
        brake_right = float(np.clip((ry - y_pulley) / eff_travel_px, 0.0, 1.05))

        # 5. Elbows (`l_elbow`, `r_elbow`) & Wrists (`l_wrist`, `r_wrist`)
        elbow_mask = (g > 165.0) & (r < 105.0) & (b < 135.0)
        el_split = max(1, min(w - 1, int(round(sh_cx))))
        le_mask = elbow_mask.copy()
        le_mask[:, el_split:] = False
        re_mask = elbow_mask.copy()
        re_mask[:, :el_split] = False
        lex, ley, _, _, le_area = self._moment_centroid_and_axes(le_mask)
        rex, rey, _, _, re_area = self._moment_centroid_and_axes(re_mask)

        if le_area >= 4.0:
            l_elbow = (lex, ley, min(1.0, le_area / 55.0))
        else:
            l_elbow = (
                0.52 * l_sh[0] + 0.48 * lx - 0.065 * w,
                0.52 * l_sh[1] + 0.48 * ly + 0.055 * h,
                0.70,
            )

        if re_area >= 4.0:
            r_elbow = (rex, rey, min(1.0, re_area / 55.0))
        else:
            r_elbow = (
                0.52 * r_sh[0] + 0.48 * rx + 0.065 * w,
                0.52 * r_sh[1] + 0.48 * ry + 0.055 * h,
                0.70,
            )

        l_wrist = (
            0.18 * l_elbow[0] + 0.82 * lx,
            0.18 * l_elbow[1] + 0.82 * ly,
            min(l_conf, l_elbow[2]),
        )
        r_wrist = (
            0.18 * r_elbow[0] + 0.82 * rx,
            0.18 * r_elbow[1] + 0.82 * ry,
            min(r_conf, r_elbow[2]),
        )

        # 6. Head / Helmet (`nose`, `l_ear`, `r_ear`) & Knees (`l_knee`, `r_knee`)
        orange_mask = (r > 185.0) & (g > 75.0) & (g < 160.0) & (b < 90.0)
        head_mask = orange_mask.copy()
        head_mask[int(0.56 * h) :, :] = False
        nx, ny, _, _, n_area = self._moment_centroid_and_axes(head_mask)
        if n_area >= 3.0:
            nose = (nx, ny, min(1.0, n_area / 45.0))
        else:
            nose = (sh_cx + 0.06 * weight_shift * w, 0.47 * h, 0.65)

        ear_dx = 0.14 * bi_shoulder_px
        l_ear = (nose[0] - ear_dx, nose[1] - 0.01 * h, nose[2])
        r_ear = (nose[0] + ear_dx, nose[1] - 0.01 * h, nose[2])

        knee_mask = orange_mask.copy()
        knee_mask[: int(0.86 * h), :] = False
        k_split = max(1, min(w - 1, int(round(hip_cx))))
        lk_mask = knee_mask.copy()
        lk_mask[:, k_split:] = False
        rk_mask = knee_mask.copy()
        rk_mask[:, :k_split] = False
        lkx, lky, _, _, lk_area = self._moment_centroid_and_axes(lk_mask)
        rkx, rky, _, _, rk_area = self._moment_centroid_and_axes(rk_mask)

        if lk_area >= 3.0:
            l_knee = (lkx, lky, min(1.0, lk_area / 45.0))
        else:
            l_knee = (l_hip[0] - 0.015 * w + 0.025 * weight_shift * w, 0.94 * h, 0.60)

        if rk_area >= 3.0:
            r_knee = (rkx, rky, min(1.0, rk_area / 45.0))
        else:
            r_knee = (r_hip[0] + 0.015 * w + 0.025 * weight_shift * w, 0.94 * h, 0.60)

        def _pt(p: Tuple[float, float, float]) -> List[float]:
            return [round(float(p[0] + ox0), 1), round(float(p[1] + oy0), 1), round(float(p[2]), 3)]

        skeleton_px: Dict[str, List[float]] = {
            "nose": _pt(nose),
            "l_ear": _pt(l_ear),
            "r_ear": _pt(r_ear),
            "l_shoulder": _pt(l_sh),
            "r_shoulder": _pt(r_sh),
            "l_elbow": _pt(l_elbow),
            "r_elbow": _pt(r_elbow),
            "l_wrist": _pt(l_wrist),
            "r_wrist": _pt(r_wrist),
            "l_palm": _pt((lx, ly, l_conf)),
            "r_palm": _pt((rx, ry, r_conf)),
            "l_hip": _pt(l_hip),
            "r_hip": _pt(r_hip),
            "l_knee": _pt(l_knee),
            "r_knee": _pt(r_knee),
        }

        if t_area >= 6.0 and (l_area >= 5.0 or r_area >= 5.0):
            pose_mode = "FULL_SKELETON"
        elif l_area >= 5.0 or r_area >= 5.0:
            pose_mode = "POV_HANDS_ONLY"
        else:
            pose_mode = "WIDE_NO_SKELETON"

        return (
            brake_left,
            brake_right,
            weight_shift,
            (lx, ly),
            (rx, ry),
            skeleton_px,
            pose_mode,
            round(shoulder_roll_deg, 2),
            round(torso_lean_deg, 2),
        )

    def detect_onboard_controls(
        self, onboard_rgb: np.ndarray
    ) -> Tuple[float, float, float, Tuple[float, float], Tuple[float, float]]:
        """
        Backward-compatible wrapper returning Left & Right brake toggle controls, weight-shift,
        and hand coordinates from the full pilot skeleton detector.
        """
        brake_l, brake_r, weight_shift, l_hand, r_hand, _, _, _, _ = self.detect_pilot_skeleton(
            onboard_rgb
        )
        return brake_l, brake_r, weight_shift, l_hand, r_hand

    @staticmethod
    def _smooth_and_interpolate_skeleton_offline(observations: List[FrameObservation]) -> None:
        """
        Offline zero-phase bidirectional smoothing & short-occlusion interpolation (`<= 10` frames)
        on the one-time extracted `pilot_skeleton_px` trajectories.
        """
        n = len(observations)
        if n < 3:
            return
        joint_keys = list(observations[0].pilot_skeleton_px.keys())
        if not joint_keys:
            return

        for key in joint_keys:
            xs = np.array([obs.pilot_skeleton_px[key][0] for obs in observations], dtype=np.float64)
            ys = np.array([obs.pilot_skeleton_px[key][1] for obs in observations], dtype=np.float64)
            confs = np.array([obs.pilot_skeleton_px[key][2] for obs in observations], dtype=np.float64)

            # 1. Interpolate short occlusion gaps (<= 10 frames with conf < 0.5)
            valid_idx = np.where(confs >= 0.5)[0]
            if len(valid_idx) >= 2:
                for k in range(len(valid_idx) - 1):
                    i0, i1 = int(valid_idx[k]), int(valid_idx[k + 1])
                    gap = i1 - i0 - 1
                    if 0 < gap <= 10:
                        for g_idx in range(i0 + 1, i1):
                            alpha = (g_idx - i0) / float(i1 - i0)
                            xs[g_idx] = (1.0 - alpha) * xs[i0] + alpha * xs[i1]
                            ys[g_idx] = (1.0 - alpha) * ys[i0] + alpha * ys[i1]
                            confs[g_idx] = 0.65

            # 2. Zero-phase forward-backward smoothing (filtfilt 3-tap kernel [0.15, 0.70, 0.15])
            kernel = np.array([0.15, 0.70, 0.15], dtype=np.float64)
            for arr in (xs, ys):
                pad_f = np.pad(arr, (1, 1), mode="edge")
                fwd = np.convolve(pad_f, kernel, mode="valid")
                pad_b = np.pad(fwd[::-1], (1, 1), mode="edge")
                bwd = np.convolve(pad_b, kernel, mode="valid")[::-1]
                arr[:] = bwd

            for i, obs in enumerate(observations):
                obs.pilot_skeleton_px[key] = [
                    round(float(xs[i]), 1),
                    round(float(ys[i]), 1),
                    round(float(confs[i]), 3),
                ]

        # Sync left_hand_px and right_hand_px with smoothed palm keypoints
        for obs in observations:
            if "l_palm" in obs.pilot_skeleton_px:
                obs.left_hand_px = obs.pilot_skeleton_px["l_palm"][:2]
            if "r_palm" in obs.pilot_skeleton_px:
                obs.right_hand_px = obs.pilot_skeleton_px["r_palm"][:2]

    def analyze_frames(
        self,
        frames_rgb: List[np.ndarray],
        fps: float,
        source_video_url: str,
        source_video_title: str,
        maneuver_id: str,
        maneuver_name: str,
        video_timestamp_range: str,
        camera_mode: str = "side_chase",
    ) -> ManeuverKinematicsTrace:
        """Runs the full frame-by-frame CV tracker on a list of RGB frames and extracts events & phase lag metrics."""
        observations: List[FrameObservation] = []
        dt = 1.0 / max(1.0, fps)
        heading_deg = 0.0

        pip_rect = self.detect_pip_region(frames_rgb) if self.ignore_pip else None

        for idx, raw_frame in enumerate(frames_rgb):
            frame = self.mask_pip_region(raw_frame, pip_rect) if (self.ignore_pip and pip_rect) else raw_frame
            h, w, _ = frame.shape
            cx0 = int(self.chase_roi_norm[0] * w)
            cy0 = int(self.chase_roi_norm[1] * h)
            cx1 = int(self.chase_roi_norm[2] * w)
            cy1 = int(self.chase_roi_norm[3] * h)

            ox0 = int(self.onboard_roi_norm[0] * w)
            oy0 = int(self.onboard_roi_norm[1] * h)
            ox1 = int(self.onboard_roi_norm[2] * w)
            oy1 = int(self.onboard_roi_norm[3] * h)

            chase_roi = frame[cy0:cy1, cx0:cx1]
            onboard_roi = frame[oy0:oy1, ox0:ox1]

            horizon_deg, horizon_y = self.detect_horizon_tilt_deg(chase_roi)
            (
                canopy_xy,
                pilot_xy,
                pitch_deg,
                roll_deg,
                big_ears_norm,
                span_px,
                conf,
            ) = self.detect_canopy_and_pilot(
                chase_roi, horizon_deg, horizon_y, camera_mode=camera_mode
            )
            (
                brake_l,
                brake_r,
                weight_shift,
                l_hand,
                r_hand,
                skeleton_px,
                pose_mode,
                shoulder_roll_deg,
                torso_lean_deg,
            ) = self.detect_pilot_skeleton(onboard_roi, ox0=ox0, oy0=oy0)

            sym_brake = 0.5 * (brake_l + brake_r)
            if idx > 0:
                prev = observations[-1]
                prev_sym = 0.5 * (prev.brake_left_norm + prev.brake_right_norm)
                brake_rate = (sym_brake - prev_sym) / dt
                pitch_rate = (pitch_deg - prev.pitch_deg) / dt
                roll_rate = (roll_deg - prev.roll_deg) / dt
            else:
                brake_rate = 0.0
                pitch_rate = 0.0
                roll_rate = 0.0

            pitch_rad = math.radians(pitch_deg)
            roll_rad = math.radians(roll_deg)
            # Coordinated turn yaw rate estimate from bank angle + differential brake
            yaw_rate_deg_sec = math.degrees(
                (9.80665 / self.wing.trim_speed_mps) * math.tan(max(-1.1, min(1.1, roll_rad)))
            ) + 12.0 * (brake_r - brake_l)
            heading_deg += yaw_rate_deg_sec * dt

            fwd_offset_m = -self.wing.line_length_m * math.sin(pitch_rad)
            lat_offset_m = self.wing.line_length_m * math.sin(roll_rad)

            observations.append(
                FrameObservation(
                    frame_index=idx,
                    time_sec=round(idx * dt, 4),
                    brake_left_norm=round(brake_l, 4),
                    brake_right_norm=round(brake_r, 4),
                    brake_rate_norm_per_sec=round(brake_rate, 4),
                    weight_shift_norm=round(weight_shift, 4),
                    big_ears_norm=round(big_ears_norm, 3),
                    horizon_tilt_deg=round(horizon_deg, 3),
                    pitch_deg=round(pitch_deg, 3),
                    pitch_rate_deg_sec=round(pitch_rate, 3),
                    roll_deg=round(roll_deg, 3),
                    roll_rate_deg_sec=round(roll_rate, 3),
                    yaw_rate_deg_sec=round(yaw_rate_deg_sec, 2),
                    heading_deg=round(heading_deg, 2),
                    canopy_forward_offset_m=round(fwd_offset_m, 3),
                    canopy_lateral_offset_m=round(lat_offset_m, 3),
                    canopy_centroid_px=[round(canopy_xy[0] + cx0, 1), round(canopy_xy[1] + cy0, 1)],
                    pilot_centroid_px=[round(pilot_xy[0] + cx0, 1), round(pilot_xy[1] + cy0, 1)],
                    left_hand_px=[round(l_hand[0] + ox0, 1), round(l_hand[1] + oy0, 1)],
                    right_hand_px=[round(r_hand[0] + ox0, 1), round(r_hand[1] + oy0, 1)],
                    canopy_span_px=round(span_px, 1),
                    confidence=round(conf, 3),
                    pilot_skeleton_px=skeleton_px,
                    pose_mode=pose_mode,
                    shoulder_roll_deg=shoulder_roll_deg,
                    torso_lean_deg=torso_lean_deg,
                    pip_rect_px=pip_rect,
                )
            )

        self._smooth_and_interpolate_skeleton_offline(observations)

        if len(observations) >= 3:
            for i in range(1, len(observations) - 1):
                p_rate = (observations[i + 1].pitch_deg - observations[i - 1].pitch_deg) / (2.0 * dt)
                r_rate = (observations[i + 1].roll_deg - observations[i - 1].roll_deg) / (2.0 * dt)
                observations[i].pitch_rate_deg_sec = round(p_rate, 3)
                observations[i].roll_rate_deg_sec = round(r_rate, 3)

        if camera_mode == "rear_chase":
            events, phase_metrics = self.extract_lateral_events_and_metrics(observations)
        else:
            events, phase_metrics = self.extract_events_and_phase_lags(observations)

        return ManeuverKinematicsTrace(
            schema_version="1.0.0",
            source_video_url=source_video_url,
            source_video_title=source_video_title,
            maneuver_id=maneuver_id,
            maneuver_name=maneuver_name,
            video_timestamp_range=video_timestamp_range,
            fps=fps,
            wing=self.wing,
            phase_metrics=phase_metrics,
            events=events,
            frames=observations,
            camera_mode=camera_mode,
        )

    @staticmethod
    def extract_lateral_events_and_metrics(
        frames: List[FrameObservation],
    ) -> Tuple[List[CommandTimingEvent], PhaseLagMetrics]:
        """
        Extracts pilot weight-shift onset, asymmetric brake onset, roll rate extrema,
        progressive roll swing peaks, and lateral pendulum half-period for Dynamic Turn & Roll exercises.
        """
        events: List[CommandTimingEvent] = []
        rolls = [f.roll_deg for f in frames]
        roll_rates = [f.roll_rate_deg_sec for f in frames]
        pitches = [f.pitch_deg for f in frames]
        ws_list = [f.weight_shift_norm for f in frames]
        asym_brakes = [f.brake_right_norm - f.brake_left_norm for f in frames]
        sym_brakes = [0.5 * (f.brake_left_norm + f.brake_right_norm) for f in frames]

        # 1. Weight-shift onset
        ws_onset_idx: Optional[int] = None
        for i, w_val in enumerate(ws_list):
            if abs(w_val) >= 0.18:
                ws_onset_idx = i
                f_ws = frames[i]
                events.append(
                    CommandTimingEvent(
                        event_type="weight_shift_onset",
                        time_sec=f_ws.time_sec,
                        frame_index=f_ws.frame_index,
                        brake_sym_norm=round(sym_brakes[i], 3),
                        pitch_deg=f_ws.pitch_deg,
                        pitch_rate_deg_sec=f_ws.pitch_rate_deg_sec,
                        roll_deg=f_ws.roll_deg,
                        roll_rate_deg_sec=f_ws.roll_rate_deg_sec,
                        weight_shift_norm=f_ws.weight_shift_norm,
                        brake_asym_norm=round(asym_brakes[i], 3),
                        description="Pilot initiates harness weight-shift preload",
                    )
                )
                break

        # 2. Asymmetric brake onset
        brake_onset_idx: Optional[int] = None
        for i, b_val in enumerate(asym_brakes):
            if abs(b_val) >= 0.15:
                brake_onset_idx = i
                f_b = frames[i]
                events.append(
                    CommandTimingEvent(
                        event_type="asym_brake_onset",
                        time_sec=f_b.time_sec,
                        frame_index=f_b.frame_index,
                        brake_sym_norm=round(sym_brakes[i], 3),
                        pitch_deg=f_b.pitch_deg,
                        pitch_rate_deg_sec=f_b.pitch_rate_deg_sec,
                        roll_deg=f_b.roll_deg,
                        roll_rate_deg_sec=f_b.roll_rate_deg_sec,
                        weight_shift_norm=f_b.weight_shift_norm,
                        brake_asym_norm=round(asym_brakes[i], 3),
                        description="Pilot applies inside steering brake impulse",
                    )
                )
                break

        # 3. Detect alternating roll swing peaks (local extrema with |roll| > 6 deg)
        swing_peaks: List[float] = []
        swing_times: List[float] = []
        current_sign = 0
        best_idx = 0
        best_abs = 0.0
        for i, r_val in enumerate(rolls):
            if abs(r_val) > 6.0:
                sgn = 1 if r_val > 0 else -1
                if current_sign == 0:
                    current_sign = sgn
                    best_idx = i
                    best_abs = abs(r_val)
                elif sgn == current_sign:
                    if abs(r_val) > best_abs:
                        best_abs = abs(r_val)
                        best_idx = i
                else:
                    f_pk = frames[best_idx]
                    swing_peaks.append(round(f_pk.roll_deg, 2))
                    swing_times.append(f_pk.time_sec)
                    events.append(
                        CommandTimingEvent(
                            event_type=f"roll_swing_peak_{len(swing_peaks)}",
                            time_sec=f_pk.time_sec,
                            frame_index=f_pk.frame_index,
                            brake_sym_norm=round(sym_brakes[best_idx], 3),
                            pitch_deg=f_pk.pitch_deg,
                            pitch_rate_deg_sec=f_pk.pitch_rate_deg_sec,
                            roll_deg=f_pk.roll_deg,
                            roll_rate_deg_sec=f_pk.roll_rate_deg_sec,
                            weight_shift_norm=f_pk.weight_shift_norm,
                            brake_asym_norm=round(asym_brakes[best_idx], 3),
                            description=f"Roll pendulum swing #{len(swing_peaks)} apex ({f_pk.roll_deg:+.1f} deg)",
                        )
                    )
                    current_sign = sgn
                    best_idx = i
                    best_abs = abs(r_val)

        if current_sign != 0 and best_abs > 6.0:
            f_pk = frames[best_idx]
            swing_peaks.append(round(f_pk.roll_deg, 2))
            swing_times.append(f_pk.time_sec)
            events.append(
                CommandTimingEvent(
                    event_type=f"roll_swing_peak_{len(swing_peaks)}",
                    time_sec=f_pk.time_sec,
                    frame_index=f_pk.frame_index,
                    brake_sym_norm=round(sym_brakes[best_idx], 3),
                    pitch_deg=f_pk.pitch_deg,
                    pitch_rate_deg_sec=f_pk.pitch_rate_deg_sec,
                    roll_deg=f_pk.roll_deg,
                    roll_rate_deg_sec=f_pk.roll_rate_deg_sec,
                    weight_shift_norm=f_pk.weight_shift_norm,
                    brake_asym_norm=round(asym_brakes[best_idx], 3),
                    description=f"Roll pendulum swing #{len(swing_peaks)} apex ({f_pk.roll_deg:+.1f} deg)",
                )
            )

        abs_rolls = [abs(r) for r in rolls]
        max_roll_idx = int(np.argmax(abs_rolls))
        f_max_roll = frames[max_roll_idx]

        dive_idx = int(np.argmin(pitches))
        f_dive = frames[dive_idx]
        aft_idx = int(np.argmax(pitches))
        f_aft = frames[aft_idx]

        if len(swing_times) >= 2:
            diffs = [swing_times[k + 1] - swing_times[k] for k in range(len(swing_times) - 1)]
            roll_half_period = float(np.mean(diffs))
        else:
            roll_half_period = 1.40

        ws_onset_sec = frames[ws_onset_idx].time_sec if ws_onset_idx is not None else None
        b_onset_sec = frames[brake_onset_idx].time_sec if brake_onset_idx is not None else None
        ws_lead = (
            round(b_onset_sec - ws_onset_sec, 3)
            if (ws_onset_sec is not None and b_onset_sec is not None)
            else 0.0
        )

        first_cmd_sec = ws_onset_sec if ws_onset_sec is not None else (b_onset_sec or 0.5)
        peak_b = float(max(max(f.brake_left_norm, f.brake_right_norm) for f in frames))
        peak_roll_rate = float(max(abs(rr) for rr in roll_rates))

        metrics = PhaseLagMetrics(
            pull_onset_sec=round(first_cmd_sec, 3),
            pull_duration_sec=round(f_max_roll.time_sec - first_cmd_sec, 3),
            peak_brake_norm=round(peak_b, 3),
            aft_pitch_apex_deg=round(f_aft.pitch_deg, 2),
            aft_pitch_apex_time_sec=round(f_aft.time_sec, 3),
            pull_to_aft_apex_lag_sec=round(f_max_roll.time_sec - first_cmd_sec, 3),
            release_onset_sec=round(f_max_roll.time_sec, 3),
            zero_pitch_crossing_sec=round(f_dive.time_sec, 3),
            release_to_overhead_lag_sec=round(ws_lead, 3),
            dive_pitch_min_deg=round(f_dive.pitch_deg, 2),
            dive_pitch_min_time_sec=round(f_dive.time_sec, 3),
            max_forward_surge_m=round(f_max_roll.canopy_lateral_offset_m, 2),
            min_pitch_rate_deg_sec=round(float(min(f.pitch_rate_deg_sec for f in frames)), 1),
            pendulum_half_period_sec=round(roll_half_period, 3),
            weight_shift_onset_sec=ws_onset_sec,
            brake_to_ws_lag_sec=ws_lead,
            peak_roll_deg=round(f_max_roll.roll_deg, 2),
            peak_roll_time_sec=round(f_max_roll.time_sec, 3),
            peak_roll_rate_deg_sec=round(peak_roll_rate, 1),
            roll_half_period_sec=round(roll_half_period, 3),
            swing_roll_peaks_deg=swing_peaks,
            total_heading_carve_deg=round(frames[-1].heading_deg, 1),
        )
        return events, metrics

    @staticmethod
    def extract_events_and_phase_lags(
        frames: List[FrameObservation],
    ) -> Tuple[List[CommandTimingEvent], PhaseLagMetrics]:
        """
        Identifies pilot brake pull onset, aft pitch apex, brake release onset,
        overhead zero-crossing, active catch onset (if present), and forward dive apex.
        """
        events: List[CommandTimingEvent] = []
        if not frames:
            raise ValueError("Cannot extract events from empty frame list.")

        sym_brakes = [0.5 * (f.brake_left_norm + f.brake_right_norm) for f in frames]
        pitches = [f.pitch_deg for f in frames]
        pitch_rates = [f.pitch_rate_deg_sec for f in frames]

        pull_idx = 0
        for i, f in enumerate(frames):
            if sym_brakes[i] >= 0.12 and f.brake_rate_norm_per_sec > 0.20:
                pull_idx = i
                break
        f_pull = frames[pull_idx]
        events.append(
            CommandTimingEvent(
                event_type="pull_onset",
                time_sec=f_pull.time_sec,
                frame_index=f_pull.frame_index,
                brake_sym_norm=round(sym_brakes[pull_idx], 3),
                pitch_deg=f_pull.pitch_deg,
                pitch_rate_deg_sec=f_pull.pitch_rate_deg_sec,
                description="Pilot initiates symmetrical brake pull from trim",
            )
        )

        search_end_aft = max(pull_idx + 2, int(0.68 * len(frames)))
        aft_idx = pull_idx + int(np.argmax(pitches[pull_idx:search_end_aft]))
        f_aft = frames[aft_idx]
        events.append(
            CommandTimingEvent(
                event_type="aft_pitch_apex",
                time_sec=f_aft.time_sec,
                frame_index=f_aft.frame_index,
                brake_sym_norm=round(sym_brakes[aft_idx], 3),
                pitch_deg=f_aft.pitch_deg,
                pitch_rate_deg_sec=f_aft.pitch_rate_deg_sec,
                description="Canopy reaches maximum aft pitch angle behind pilot (pendulum back-apex)",
            )
        )

        rel_window_start = max(pull_idx + 1, aft_idx - 15)
        rel_window_end = min(len(frames) - 1, aft_idx + 12)
        rel_idx = aft_idx
        for i in range(rel_window_start, rel_window_end + 1):
            if frames[i].brake_rate_norm_per_sec < -0.45 and sym_brakes[i] > 0.25:
                rel_idx = i
                break
        f_rel = frames[rel_idx]
        events.append(
            CommandTimingEvent(
                event_type="release_onset",
                time_sec=f_rel.time_sec,
                frame_index=f_rel.frame_index,
                brake_sym_norm=round(sym_brakes[rel_idx], 3),
                pitch_deg=f_rel.pitch_deg,
                pitch_rate_deg_sec=f_rel.pitch_rate_deg_sec,
                description="Pilot releases brakes briskly hands-up at aft pendulum apex",
            )
        )

        dive_search_end = min(len(frames), aft_idx + int(2.6 * 30))
        dive_idx = aft_idx + int(np.argmin(pitches[aft_idx:dive_search_end]))
        f_dive = frames[dive_idx]

        zero_idx = aft_idx
        for i in range(aft_idx, dive_idx + 1):
            if pitches[i] <= 0.0:
                zero_idx = i
                break
        f_zero = frames[zero_idx]
        events.append(
            CommandTimingEvent(
                event_type="overhead_zero_crossing",
                time_sec=f_zero.time_sec,
                frame_index=f_zero.frame_index,
                brake_sym_norm=round(sym_brakes[zero_idx], 3),
                pitch_deg=f_zero.pitch_deg,
                pitch_rate_deg_sec=f_zero.pitch_rate_deg_sec,
                description="Canopy passes overhead (theta = 0 deg) accelerating into the forward dive",
            )
        )

        catch_onset_sec: Optional[float] = None
        catch_peak_brake: Optional[float] = None
        catch_lead_sec: Optional[float] = None
        catch_end = min(len(frames), dive_idx + 10)
        if catch_end > zero_idx + 1:
            sub_brakes = sym_brakes[zero_idx:catch_end]
            if max(sub_brakes) > 0.35 and min(sym_brakes[rel_idx : zero_idx + 1]) < 0.15:
                for i in range(zero_idx, catch_end):
                    if sym_brakes[i] >= 0.18 and frames[i].brake_rate_norm_per_sec > 0.30:
                        f_catch = frames[i]
                        catch_onset_sec = f_catch.time_sec
                        catch_peak_brake = round(float(max(sub_brakes)), 3)
                        catch_lead_sec = round(f_dive.time_sec - f_catch.time_sec, 3)
                        events.append(
                            CommandTimingEvent(
                                event_type="active_catch_onset",
                                time_sec=f_catch.time_sec,
                                frame_index=f_catch.frame_index,
                                brake_sym_norm=round(sym_brakes[i], 3),
                                pitch_deg=f_catch.pitch_deg,
                                pitch_rate_deg_sec=f_catch.pitch_rate_deg_sec,
                                description="Pilot applies symmetrical catch brake pulse while canopy dives ahead",
                            )
                        )
                        break

        events.append(
            CommandTimingEvent(
                event_type="forward_dive_apex",
                time_sec=f_dive.time_sec,
                frame_index=f_dive.frame_index,
                brake_sym_norm=round(sym_brakes[dive_idx], 3),
                pitch_deg=f_dive.pitch_deg,
                pitch_rate_deg_sec=f_dive.pitch_rate_deg_sec,
                description="Canopy reaches maximum forward dive angle ahead of pilot",
            )
        )

        peak_pull_brake = float(max(sym_brakes[pull_idx : max(pull_idx + 1, rel_idx + 1)]))
        min_pitch_rate = float(min(pitch_rates[aft_idx : max(aft_idx + 1, dive_idx + 1)]))
        half_period = max(0.2, f_dive.time_sec - f_aft.time_sec)

        metrics = PhaseLagMetrics(
            pull_onset_sec=round(f_pull.time_sec, 3),
            pull_duration_sec=round(max(0.0, f_rel.time_sec - f_pull.time_sec), 3),
            peak_brake_norm=round(peak_pull_brake, 3),
            aft_pitch_apex_deg=round(f_aft.pitch_deg, 2),
            aft_pitch_apex_time_sec=round(f_aft.time_sec, 3),
            pull_to_aft_apex_lag_sec=round(f_aft.time_sec - f_pull.time_sec, 3),
            release_onset_sec=round(f_rel.time_sec, 3),
            zero_pitch_crossing_sec=round(f_zero.time_sec, 3),
            release_to_overhead_lag_sec=round(f_zero.time_sec - f_rel.time_sec, 3),
            dive_pitch_min_deg=round(f_dive.pitch_deg, 2),
            dive_pitch_min_time_sec=round(f_dive.time_sec, 3),
            max_forward_surge_m=round(f_dive.canopy_forward_offset_m, 2),
            min_pitch_rate_deg_sec=round(min_pitch_rate, 1),
            pendulum_half_period_sec=round(half_period, 3),
            catch_onset_sec=catch_onset_sec,
            catch_peak_brake_norm=catch_peak_brake,
            catch_lead_before_dive_apex_sec=catch_lead_sec,
        )
        return events, metrics
