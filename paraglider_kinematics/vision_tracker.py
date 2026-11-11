from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

import numpy as np
from PIL import Image, ImageDraw

from .schema import (
    CommandTimingEvent,
    FrameObservation,
    ManeuverKinematicsTrace,
    PhaseLagMetrics,
    ShotCalibrationProfile,
    WingGeometrySpec,
)


class DualViewParagliderTracker:
    """
    Two-pass Shot-Adaptive Computer Vision recognizer for paragliding tutorial videos
    with External Chase View (Side or Rear/Front) + Onboard Pilot POV Split-Screen/PIP.

    Pass 1 (`SequenceAutoCalibrator`):
      - Segments the video into homogeneous continuous shots (`segment_shots`) at scene cuts.
      - Computes a temporal-invariance static overlay mask (`compute_static_overlay_mask`) to suppress
        imprinted watermarks, static logos, or fixed camera-mount artifacts across N sampled frames.
      - Auto-learns per-shot appearance RGB cluster centers (`canopy_rgb_center`, `jacket_rgb_center`,
        `hip_rgb_center`, `glove_rgb_center`, `elbow_rgb_center`, `head_rgb_center`) across N sampled
        frames so different pilot outfits, glove colors, or canopy colors across edited cuts self-calibrate.
      - Auto-detects the camera anchor joint (`anchor_joint`, e.g. `l_knee` for a 360 camera mounted on
        the left knee) via keypoint positional variance minimization and calibrates the left-vs-right
        perspective foreshortening ratio (`persp_scale_lr`), independent left/right brake stroke travels,
        neutral torso lean offset, and zoom-invariant nominal wing span.

    Pass 2 (Shot-Constrained Kinematic Extraction):
      - Extracts horizon tilt, canopy attitude, 15-joint articulated pilot skeleton, perspective-compensated
        brake toggles, and weight-shift using the active shot's `ShotCalibrationProfile`.
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
        if shutil.which("ffmpeg") is None and Path(video_path).exists():
            with open(video_path, "rb") as fh:
                arr = np.load(fh, allow_pickle=False)
            return list(arr)
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
    def segment_shots(
        frames_rgb: List[np.ndarray],
        min_shot_frames: int = 6,
    ) -> List[Tuple[int, int]]:
        """
        Pass 1A: Segments a video sequence into continuous homogeneous shots [start_idx, end_idx]
        by detecting abrupt scene, lighting, or pilot-outfit/canopy color cuts on undiluted 4x-subsampled frames.
        """
        n = len(frames_rgb)
        if n == 0:
            return []
        if n < 2 * min_shot_frames:
            return [(0, n - 1)]

        fg_chase_colors = []
        fg_onboard_colors = []
        luma_grids = []

        for f in frames_rgb:
            small = f[::2, ::2].astype(np.float32)
            sh, sw, _ = small.shape
            split_x = int(0.68 * sw)
            luma_grids.append(small.mean(axis=2))

            for roi_sub, target_list in (
                (small[: int(0.46 * sh), :split_x], fg_chase_colors),
                (small[int(0.12 * sh) : int(0.88 * sh), split_x:], fg_onboard_colors),
            ):
                r, g, b = roi_sub[:, :, 0], roi_sub[:, :, 1], roi_sub[:, :, 2]
                chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
                row_med = np.median(roi_sub, axis=1, keepdims=True)
                row_contrast = np.abs(roi_sub - row_med).sum(axis=2)
                fg_mask = (chroma > 48.0) & (row_contrast > 55.0)
                if np.any(fg_mask):
                    target_list.append(roi_sub[fg_mask].mean(axis=0))
                else:
                    target_list.append(roi_sub.mean(axis=(0, 1)))

        chase_arr = np.stack(fg_chase_colors, axis=0)
        onboard_arr = np.stack(fg_onboard_colors, axis=0)
        luma_arr = np.stack(luma_grids, axis=0)

        chase_jumps = np.abs(np.diff(chase_arr, axis=0)).sum(axis=1)
        onboard_jumps = np.abs(np.diff(onboard_arr, axis=0)).sum(axis=1)
        luma_jumps = np.abs(np.diff(luma_arr, axis=0)).mean(axis=(1, 2))

        med_chase = max(3.0, float(np.median(chase_jumps)))
        med_onboard = max(3.0, float(np.median(onboard_jumps)))

        cut_indices: List[int] = [0]
        for i in range(min_shot_frames - 1, n - min_shot_frames):
            is_cut = (
                (chase_jumps[i] > max(55.0, 5.0 * med_chase))
                or (onboard_jumps[i] > max(48.0, 5.0 * med_onboard))
                or (luma_jumps[i] > 22.0)
            )
            if is_cut and (i + 1 - cut_indices[-1]) >= min_shot_frames:
                cut_indices.append(i + 1)

        shots: List[Tuple[int, int]] = []
        for idx, start_f in enumerate(cut_indices):
            end_f = (cut_indices[idx + 1] - 1) if (idx + 1 < len(cut_indices)) else (n - 1)
            shots.append((start_f, end_f))
        return shots

    @staticmethod
    def compute_static_overlay_mask(sample_rois: List[np.ndarray]) -> np.ndarray:
        """
        Pass 1B: Computes a boolean mask of static high-chroma overlay/watermark/logo pixels that remain
        temporally invariant across all sampled frames (`ptp_t < 14.0`) while contrasting sharply
        against the horizontal background band (`|I(x,y) - median_x(I(:,y))| > 50`).
        """
        if len(sample_rois) < 3:
            h, w, _ = sample_rois[0].shape
            return np.zeros((h, w), dtype=bool)

        stack = np.stack([r.astype(np.float32) for r in sample_rois], axis=0)
        ptp_rgb = (stack.max(axis=0) - stack.min(axis=0)).max(axis=2)
        mean_rgb = stack.mean(axis=0)
        r_m, g_m, b_m = mean_rgb[:, :, 0], mean_rgb[:, :, 1], mean_rgb[:, :, 2]
        chroma = np.maximum(np.maximum(r_m, g_m), b_m) - np.minimum(np.minimum(r_m, g_m), b_m)
        row_median = np.median(mean_rgb, axis=1, keepdims=True)
        row_contrast = np.abs(mean_rgb - row_median).sum(axis=2)

        # Temporally invariant across the entire shot, high-chroma, and anomalous wrt sky/ground row
        static_mask = (ptp_rgb < 14.0) & (chroma > 40.0) & (row_contrast > 50.0)
        return static_mask

    @staticmethod
    def _color_cluster_mask(
        rgb: np.ndarray,
        center_rgb: List[float],
        tol: float = 65.0,
        min_chroma: float = 36.0,
        static_mask: Optional[np.ndarray] = None,
        other_centers: Optional[List[List[float]]] = None,
    ) -> np.ndarray:
        """
        Matches foreground pixels within Euclidean distance `tol` of `center_rgb` and closer to
        `center_rgb` than to any competing anatomical cluster in `other_centers` (Voronoi partition).
        """
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c
        dist = np.sqrt(
            (r - float(center_rgb[0])) ** 2
            + (g - float(center_rgb[1])) ** 2
            + (b - float(center_rgb[2])) ** 2
        )
        mask = (dist <= tol) & (chroma >= min_chroma)
        if other_centers:
            for oc in other_centers:
                d_oc = np.sqrt(
                    (r - float(oc[0])) ** 2
                    + (g - float(oc[1])) ** 2
                    + (b - float(oc[2])) ** 2
                )
                mask &= dist < (d_oc - 4.0)
        if static_mask is not None and static_mask.shape == mask.shape:
            mask &= ~static_mask
        return mask

    @staticmethod
    def _learn_band_dominant_color(
        rois: List[np.ndarray],
        y0_frac: float,
        y1_frac: float,
        x0_frac: float,
        x1_frac: float,
        default_rgb: List[float],
        exclude_centers: Optional[List[List[float]]] = None,
        static_mask: Optional[np.ndarray] = None,
        min_chroma: float = 48.0,
    ) -> List[float]:
        """
        Extracts the dominant high-chroma foreground RGB cluster center inside a normalized spatial band
        across N sampled frames, excluding static watermarks and already-learned color centers.
        """
        if not rois:
            return list(default_rgb)
        h, w, _ = rois[0].shape
        y0, y1 = max(0, int(y0_frac * h)), min(h, int(y1_frac * h))
        x0, x1 = max(0, int(x0_frac * w)), min(w, int(x1_frac * w))
        if y1 <= y0 or x1 <= x0:
            return list(default_rgb)

        collected: List[np.ndarray] = []
        for roi in rois:
            full_f = roi.astype(np.float32)
            row_med = np.median(full_f, axis=1, keepdims=True)[y0:y1]
            sub = full_f[y0:y1, x0:x1]
            r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
            chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
            row_contrast = np.abs(sub - row_med).sum(axis=2)
            valid = (chroma >= min_chroma) & (row_contrast > 55.0)
            if static_mask is not None and static_mask.shape == (h, w):
                valid &= ~static_mask[y0:y1, x0:x1]
            if exclude_centers:
                for exc in exclude_centers:
                    d_exc = np.sqrt(
                        (r - float(exc[0])) ** 2
                        + (g - float(exc[1])) ** 2
                        + (b - float(exc[2])) ** 2
                    )
                    valid &= d_exc > 52.0
            if np.any(valid):
                collected.append(sub[valid])

        if not collected:
            return list(default_rgb)
        all_px = np.concatenate(collected, axis=0)
        if len(all_px) < 6:
            return list(default_rgb)
        med = np.median(all_px, axis=0)
        return [round(float(med[0]), 1), round(float(med[1]), 1), round(float(med[2]), 1)]

    def calibrate_shot_profile(
        self,
        frames_rgb: List[np.ndarray],
        shot_id: int,
        start_frame: int,
        end_frame: int,
        camera_mode: str = "side_chase",
        pip_rect: Optional[List[int]] = None,
        sample_count: int = 12,
    ) -> Tuple[ShotCalibrationProfile, np.ndarray]:
        """
        Pass 1C (`SequenceAutoCalibrator`): Samples up to `sample_count` frames across shot `[start_frame, end_frame]`
        to learn per-shot clothing/wing color profiles, static watermark masks, 360 camera anchor joint,
        and left-vs-right perspective foreshortening scale (`persp_scale_lr`).
        """
        indices = np.linspace(
            start_frame,
            end_frame,
            min(sample_count, max(1, end_frame - start_frame + 1)),
            dtype=int,
        )
        chase_samples: List[np.ndarray] = []
        onboard_samples: List[np.ndarray] = []

        h_full, w_full, _ = frames_rgb[start_frame].shape
        cx0, cy0 = int(self.chase_roi_norm[0] * w_full), int(self.chase_roi_norm[1] * h_full)
        cx1, cy1 = int(self.chase_roi_norm[2] * w_full), int(self.chase_roi_norm[3] * h_full)
        ox0, oy0 = int(self.onboard_roi_norm[0] * w_full), int(self.onboard_roi_norm[1] * h_full)
        ox1, oy1 = int(self.onboard_roi_norm[2] * w_full), int(self.onboard_roi_norm[3] * h_full)

        for idx in indices:
            raw = frames_rgb[int(idx)]
            frame = self.mask_pip_region(raw, pip_rect) if (self.ignore_pip and pip_rect) else raw
            chase_samples.append(frame[cy0:cy1, cx0:cx1])
            onboard_samples.append(frame[oy0:oy1, ox0:ox1])

        chase_static_mask = self.compute_static_overlay_mask(chase_samples)
        static_ratio = float(np.mean(chase_static_mask))

        # 1. Learn Canopy Color Center from upper sky band of Chase ROI (excluding static logos)
        canopy_rgb = self._learn_band_dominant_color(
            chase_samples,
            y0_frac=0.04,
            y1_frac=0.52,
            x0_frac=0.08,
            x1_frac=0.92,
            default_rgb=[245.0, 112.0, 22.0],
            static_mask=chase_static_mask,
            min_chroma=45.0,
        )

        # 2. Learn Pilot Anatomical Color Centers from Onboard ROI across the N sampled frames
        glove_rgb = self._learn_band_dominant_color(
            onboard_samples,
            y0_frac=0.14,
            y1_frac=0.36,
            x0_frac=0.16,
            x1_frac=0.84,
            default_rgb=[248.0, 220.0, 40.0],
            min_chroma=50.0,
        )
        head_rgb = self._learn_band_dominant_color(
            onboard_samples,
            y0_frac=0.41,
            y1_frac=0.51,
            x0_frac=0.38,
            x1_frac=0.62,
            default_rgb=[250.0, 115.0, 40.0],
            exclude_centers=[glove_rgb],
            min_chroma=50.0,
        )
        hip_rgb = self._learn_band_dominant_color(
            onboard_samples,
            y0_frac=0.76,
            y1_frac=0.87,
            x0_frac=0.22,
            x1_frac=0.78,
            default_rgb=[220.0, 50.0, 200.0],
            exclude_centers=[glove_rgb, head_rgb],
            min_chroma=50.0,
        )
        jacket_rgb = self._learn_band_dominant_color(
            onboard_samples,
            y0_frac=0.56,
            y1_frac=0.64,
            x0_frac=0.24,
            x1_frac=0.76,
            default_rgb=[35.0, 205.0, 230.0],
            exclude_centers=[glove_rgb],
            min_chroma=50.0,
        )
        elbow_rgb = self._learn_band_dominant_color(
            onboard_samples,
            y0_frac=0.42,
            y1_frac=0.70,
            x0_frac=0.06,
            x1_frac=0.94,
            default_rgb=[40.0, 225.0, 95.0],
            exclude_centers=[glove_rgb, jacket_rgb, hip_rgb, head_rgb],
            min_chroma=50.0,
        )

        # Build initial profile with learned appearance colors to run preliminary geometric pass
        oh, ow, _ = onboard_samples[0].shape
        ch, cw, _ = chase_samples[0].shape
        base_pulley_y = self.pulley_y_norm * oh
        base_travel = max(10.0, (self.carabiner_y_norm - self.pulley_y_norm) * oh)

        profile = ShotCalibrationProfile(
            shot_id=shot_id,
            start_frame=start_frame,
            end_frame=end_frame,
            shot_mode="DUAL_SPLIT",
            anchor_joint="none",
            anchor_px=[round(0.36 * ow + ox0, 1), round(0.94 * oh + oy0, 1)],
            persp_scale_lr=1.0,
            pulley_y_left_px=round(base_pulley_y, 2),
            pulley_y_right_px=round(base_pulley_y, 2),
            travel_left_px=round(base_travel, 2),
            travel_right_px=round(base_travel, 2),
            neutral_torso_lean_deg=0.0,
            nominal_span_px=round(0.34 * ch * 0.96, 2),
            nominal_line_len_px=round(0.34 * ch, 2),
            camera_elevation_deg=0.0,
            canopy_rgb_center=canopy_rgb,
            jacket_rgb_center=jacket_rgb,
            hip_rgb_center=hip_rgb,
            glove_rgb_center=glove_rgb,
            elbow_rgb_center=elbow_rgb,
            head_rgb_center=head_rgb,
            static_mask_ratio=round(static_ratio, 4),
        )

        # 3. Preliminary geometric pass over the N sampled frames to calibrate perspective & camera anchor
        persp_ratios: List[float] = []
        lk_pts: List[Tuple[float, float]] = []
        rk_pts: List[Tuple[float, float]] = []
        spans: List[float] = []
        line_lens: List[float] = []
        level_leans: List[float] = []

        for c_roi, o_roi in zip(chase_samples, onboard_samples):
            h_deg, h_y = self.detect_horizon_tilt_deg(c_roi)
            c_xy, p_xy, pitch_d, roll_d, _, sp_px, c_conf = self.detect_canopy_and_pilot(
                c_roi,
                h_deg,
                h_y,
                camera_mode=camera_mode,
                shot_profile=profile,
                chase_static_mask=chase_static_mask,
            )
            if c_conf > 0.35 and sp_px > 20.0:
                spans.append(sp_px)
                if abs(pitch_d) < 6.0:
                    line_lens.append(math.hypot(c_xy[0] - p_xy[0], p_xy[1] - c_xy[1]))

            _, _, _, _, _, skel, _, _, lean_d = self.detect_pilot_skeleton(
                o_roi, ox0=0, oy0=0, shot_profile=profile
            )
            if abs(roll_d) < 4.0:
                level_leans.append(lean_d)

            l_sh = skel["l_shoulder"]
            r_sh = skel["r_shoulder"]
            l_el = skel["l_elbow"]
            r_el = skel["r_elbow"]
            l_pa = skel["l_palm"]
            r_pa = skel["r_palm"]
            l_kn = skel["l_knee"]
            r_kn = skel["r_knee"]

            lk_pts.append((l_kn[0], l_kn[1]))
            rk_pts.append((r_kn[0], r_kn[1]))

            # Lateral elbow bow ratio from shoulder->palm chord is invariant to brake pull & weight shift
            bow_l = (0.52 * l_sh[0] + 0.48 * l_pa[0]) - l_el[0]
            bow_r = r_el[0] - (0.52 * r_sh[0] + 0.48 * r_pa[0])
            if bow_l > 4.0 and bow_r > 4.0 and l_el[2] >= 0.5 and r_el[2] >= 0.5:
                persp_ratios.append(bow_l / bow_r)

        raw_gamma = float(np.clip(np.median(persp_ratios), 0.65, 1.85)) if persp_ratios else 1.0
        var_lk = 999.0
        if len(lk_pts) >= 3:
            lk_arr = np.array(lk_pts, dtype=np.float64)
            var_lk = float(np.var(lk_arr[:, 0]) + np.var(lk_arr[:, 1]))

        # Detect 360 camera mounted on left knee (stationary l_knee anchor + left-side foreshortening)
        if var_lk < 2.0 and raw_gamma > 1.15:
            opt_calib = self.solve_360_optical_calibration_from_invariants()
            gamma = raw_gamma
            profile.anchor_joint = "l_knee"
            profile.shot_mode = "ONBOARD_360_KNEE"
            profile.camera_elevation_deg = float(opt_calib["camera_pitch_deg"])
            profile.focal_px = float(opt_calib["focal_px"])
            profile.focal_top_app_px = float(opt_calib["focal_top_app_px"])
            profile.focal_35mm_eq = float(opt_calib["focal_35mm_eq"])
            profile.k1_radial = float(opt_calib["k1_radial"])
            profile.fov_horizontal_deg = float(opt_calib["fov_horizontal_deg"])
            profile.fov_vertical_deg = float(opt_calib["fov_vertical_deg"])
            profile.projection_model = str(opt_calib["projection_model"])
            profile.camera_pitch_deg = float(opt_calib["camera_pitch_deg"])
            profile.camera_yaw_deg = float(opt_calib["camera_yaw_deg"])
            profile.camera_roll_deg = float(opt_calib["camera_roll_deg"])
            profile.persp_r = float(opt_calib["persp_r"])
            med_lk = np.median(np.array(lk_pts, dtype=np.float64), axis=0)
            profile.anchor_px = [round(float(med_lk[0] + ox0), 1), round(float(med_lk[1] + oy0), 1)]
        else:
            gamma = 1.0
            profile.anchor_joint = "harness_pelvis"
            profile.shot_mode = "DUAL_SPLIT"
            profile.projection_model = "PINHOLE_RECTILINEAR"
            profile.k1_radial = 0.0
            profile.persp_r = 1.0

        profile.persp_scale_lr = round(gamma, 3)
        scale_l = math.sqrt(gamma)
        scale_r = 1.0 / math.sqrt(gamma)
        profile.travel_left_px = round(base_travel * scale_l, 2)
        profile.travel_right_px = round(base_travel * scale_r, 2)

        if spans:
            p95_span = float(np.percentile(spans, 95))
            profile.nominal_span_px = round(max(p95_span, 0.34 * ch * 0.95), 2)
        if line_lens:
            profile.nominal_line_len_px = round(float(np.median(line_lens)), 2)
        if level_leans and gamma == 1.0:
            profile.neutral_torso_lean_deg = 0.0

        return profile, chase_static_mask

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
        shot_profile: Optional[ShotCalibrationProfile] = None,
        chase_static_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float, float, float, float, float]:
        """
        Detects the ram-air canopy and suspended pilot in the External Chase ROI using the active
        shot's calibrated canopy color center and static watermark suppression mask.
        """
        r = chase_rgb[:, :, 0].astype(np.float32)
        g = chase_rgb[:, :, 1].astype(np.float32)
        b = chase_rgb[:, :, 2].astype(np.float32)

        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        chroma = max_rgb - min_rgb

        if shot_profile is not None:
            canopy_mask = self._color_cluster_mask(
                chase_rgb,
                shot_profile.canopy_rgb_center,
                tol=68.0,
                min_chroma=45.0,
                static_mask=chase_static_mask,
            )
            # Restrict canopy search above the horizon line
            h_c, w_c, _ = chase_rgb.shape
            y_c_grid, _ = np.ogrid[:h_c, :w_c]
            canopy_mask &= y_c_grid < (horizon_mid_y - 12.0)
        else:
            canopy_mask = (chroma > 45.0) & (r > b + 35.0) & (max_rgb > 110.0)
            if chase_static_mask is not None and chase_static_mask.shape == canopy_mask.shape:
                canopy_mask &= ~chase_static_mask

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
            raw_bank_deg = math.degrees(math.atan2(dx, max(12.0, dy_up)))
            roll_deg = raw_bank_deg - horizon_tilt_deg
            nominal_full_span_px = (
                shot_profile.nominal_span_px
                if (shot_profile is not None and shot_profile.nominal_span_px > 20.0)
                else (0.34 * h * 0.96)
            )
            span_ratio = span_px / max(20.0, nominal_full_span_px)
            big_ears_norm = float(np.clip((0.93 - span_ratio) / 0.19, 0.0, 1.0))
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
        """
        if not frames_rgb:
            return None
        sample_indices = np.linspace(0, len(frames_rgb) - 1, min(8, len(frames_rgb)), dtype=int)
        sample = np.stack([frames_rgb[i].astype(np.float32).mean(axis=2) for i in sample_indices], axis=0)
        mean_gray = sample.mean(axis=0)
        h, w = mean_gray.shape

        gx = np.abs(np.diff(mean_gray, axis=1, prepend=mean_gray[:, :1]))
        gy = np.abs(np.diff(mean_gray, axis=0, prepend=mean_gray[:1, :]))

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
        self,
        onboard_rgb: np.ndarray,
        ox0: int = 0,
        oy0: int = 0,
        shot_profile: Optional[ShotCalibrationProfile] = None,
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
        Anatomical 15-joint pilot skeleton extractor for the Onboard POV ROI.
        Uses per-shot learned color centers (`shot_profile`) and perspective foreshortening compensation
        (`persp_scale_lr`, e.g. for a 360 camera mounted on the left knee).
        """
        h, w, _ = onboard_rgb.shape
        r = onboard_rgb[:, :, 0].astype(np.float32)
        g = onboard_rgb[:, :, 1].astype(np.float32)
        b = onboard_rgb[:, :, 2].astype(np.float32)

        gamma = shot_profile.persp_scale_lr if shot_profile is not None else 1.0
        scale_l = math.sqrt(gamma)
        scale_r = 1.0 / math.sqrt(gamma)

        # 1. Shoulders & Upper Torso Axis (`l_shoulder`, `r_shoulder`)
        if shot_profile is not None:
            shoulder_mask = self._color_cluster_mask(
                onboard_rgb,
                shot_profile.jacket_rgb_center,
                tol=68.0,
                min_chroma=38.0,
                other_centers=[
                    shot_profile.elbow_rgb_center,
                    shot_profile.glove_rgb_center,
                    shot_profile.hip_rgb_center,
                    shot_profile.head_rgb_center,
                ],
            )
            shoulder_mask[: int(0.52 * h), :] = False
            shoulder_mask[int(0.68 * h) :, :] = False
        else:
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
                l_sh = (tx - sh_hw * scale_l, ty, 0.75)
                r_sh = (tx + sh_hw * scale_r, ty, 0.75)
        elif t_area >= 6.0:
            sh_hw = 0.16 * w
            l_sh = (tx - sh_hw * scale_l, ty, 0.70)
            r_sh = (tx + sh_hw * scale_r, ty, 0.70)
        else:
            tx, ty = 0.50 * w, 0.60 * h
            sh_hw = 0.16 * w
            l_sh = (tx - sh_hw * scale_l, ty, 0.35)
            r_sh = (tx + sh_hw * scale_r, ty, 0.35)

        # Perspective-invariant spine center between left (scale_l) and right (scale_r) shoulders
        sh_cx = (l_sh[0] * scale_r + r_sh[0] * scale_l) / (scale_l + scale_r)
        sh_cy = (l_sh[1] * scale_r + r_sh[1] * scale_l) / (scale_l + scale_r)
        raw_bi_shoulder_px = max(20.0, math.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]))
        bi_shoulder_px = raw_bi_shoulder_px * (2.0 / (scale_l + scale_r))
        shoulder_roll_deg = math.degrees(math.atan2(r_sh[1] - l_sh[1], max(10.0, r_sh[0] - l_sh[0])))

        # 2. Hips / Harness Pelvis Axis (`l_hip`, `r_hip`)
        if shot_profile is not None:
            hip_mask = self._color_cluster_mask(
                onboard_rgb, shot_profile.hip_rgb_center, tol=78.0, min_chroma=38.0
            )
        else:
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
                l_hip = (hx - hip_hw * scale_l, hy, 0.70)
                r_hip = (hx + hip_hw * scale_r, hy, 0.70)
        else:
            hx = 0.5 * w + 0.64 * (sh_cx - 0.5 * w)
            hy = 0.82 * h
            hip_hw = 0.11 * w
            l_hip = (hx - hip_hw * scale_l, hy, 0.45)
            r_hip = (hx + hip_hw * scale_r, hy, 0.45)

        hip_cx = (l_hip[0] * scale_r + r_hip[0] * scale_l) / (scale_l + scale_r)
        hip_cy = (l_hip[1] * scale_r + r_hip[1] * scale_l) / (scale_l + scale_r)

        # 3. Torso Lean Angle & Harness Weight-Shift (`torso_lean_deg`, `weight_shift`)
        neutral_lean = shot_profile.neutral_torso_lean_deg if shot_profile is not None else 0.0
        torso_lean_deg = (
            math.degrees(math.atan2(sh_cx - 0.5 * w, max(20.0, hip_cy - sh_cy))) - neutral_lean
        )
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

        # 4. Brake Hands (`l_palm`, `r_palm`) with Top-Down Kinematic Reach & Perspective Travel
        if shot_profile is not None:
            toggle_mask = self._color_cluster_mask(
                onboard_rgb,
                shot_profile.glove_rgb_center,
                tol=68.0,
                min_chroma=40.0,
                other_centers=[
                    shot_profile.jacket_rgb_center,
                    shot_profile.elbow_rgb_center,
                    shot_profile.hip_rgb_center,
                    shot_profile.head_rgb_center,
                ],
            )
        else:
            toggle_mask = (r > 180.0) & (g > 165.0) & (b < 110.0)
        left_half = toggle_mask.copy()
        left_half[:, w // 2 :] = False
        right_half = toggle_mask.copy()
        right_half[:, : w // 2] = False

        lx, ly, _, _, l_area = self._moment_centroid_and_axes(left_half)
        rx, ry, _, _, r_area = self._moment_centroid_and_axes(right_half)

        y_pulley_l = (
            shot_profile.pulley_y_left_px if shot_profile is not None else (self.pulley_y_norm * h)
        )
        y_pulley_r = (
            shot_profile.pulley_y_right_px if shot_profile is not None else (self.pulley_y_norm * h)
        )
        y_carabiner = self.carabiner_y_norm * h
        base_travel_px = max(10.0, y_carabiner - self.pulley_y_norm * h)

        if l_area < 5.0:
            lx, ly, l_conf = 0.28 * w, y_pulley_l, 0.35
        else:
            l_conf = min(1.0, l_area / 95.0)

        if r_area < 5.0:
            rx, ry, r_conf = 0.72 * w, y_pulley_r, 0.35
        else:
            r_conf = min(1.0, r_area / 95.0)

        max_arm_reach = 3.8 * bi_shoulder_px * max(scale_l, scale_r)
        dist_l = math.hypot(lx - l_sh[0], ly - l_sh[1])
        if dist_l > max_arm_reach and dist_l > 1e-3:
            s_l = max_arm_reach / dist_l
            lx = l_sh[0] + (lx - l_sh[0]) * s_l
            ly = l_sh[1] + (ly - l_sh[1]) * s_l

        dist_r = math.hypot(rx - r_sh[0], ry - r_sh[1])
        if dist_r > max_arm_reach and dist_r > 1e-3:
            s_r = max_arm_reach / dist_r
            rx = r_sh[0] + (rx - r_sh[0]) * s_r
            ry = r_sh[1] + (ry - r_sh[1]) * s_r

        nom_bi_shoulder = 0.32 * w
        shoulder_scaled_travel = bi_shoulder_px * (base_travel_px / max(10.0, nom_bi_shoulder))
        eff_base_travel_px = 0.5 * base_travel_px + 0.5 * shoulder_scaled_travel

        eff_travel_l_px = eff_base_travel_px * scale_l
        eff_travel_r_px = eff_base_travel_px * scale_r

        brake_left = float(np.clip((ly - y_pulley_l) / max(10.0, eff_travel_l_px), 0.0, 1.05))
        brake_right = float(np.clip((ry - y_pulley_r) / max(10.0, eff_travel_r_px), 0.0, 1.05))

        # 5. Elbows (`l_elbow`, `r_elbow`) & Wrists (`l_wrist`, `r_wrist`)
        if shot_profile is not None:
            elbow_mask = self._color_cluster_mask(
                onboard_rgb,
                shot_profile.elbow_rgb_center,
                tol=68.0,
                min_chroma=38.0,
                other_centers=[
                    shot_profile.jacket_rgb_center,
                    shot_profile.glove_rgb_center,
                    shot_profile.hip_rgb_center,
                    shot_profile.head_rgb_center,
                ],
            )
        else:
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
                0.52 * l_sh[0] + 0.48 * lx - 0.065 * w * scale_l,
                0.52 * l_sh[1] + 0.48 * ly + 0.055 * h,
                0.70,
            )

        if re_area >= 4.0:
            r_elbow = (rex, rey, min(1.0, re_area / 55.0))
        else:
            r_elbow = (
                0.52 * r_sh[0] + 0.48 * rx + 0.065 * w * scale_r,
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
        if shot_profile is not None:
            orange_mask = self._color_cluster_mask(
                onboard_rgb, shot_profile.head_rgb_center, tol=58.0, min_chroma=40.0
            )
        else:
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
        elif shot_profile is not None and shot_profile.shot_mode == "ONBOARD_360_KNEE":
            l_knee = (0.33 * w, 0.84 * h, 0.85)
        else:
            l_knee = (l_hip[0], l_hip[1], 0.0)

        if rk_area >= 3.0:
            r_knee = (rkx, rky, min(1.0, rk_area / 45.0))
        else:
            r_knee = (r_hip[0], r_hip[1], 0.0)

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
        on the extracted `pilot_skeleton_px` trajectories, performed independently per shot so scene
        cuts are never blurred across boundaries.
        """
        n = len(observations)
        if n < 3:
            return
        joint_keys = list(observations[0].pilot_skeleton_px.keys())
        if not joint_keys:
            return

        # Group contiguous frame slices by shot_id
        shot_slices: List[Tuple[int, int]] = []
        s0 = 0
        for i in range(1, n):
            if observations[i].shot_id != observations[s0].shot_id:
                shot_slices.append((s0, i))
                s0 = i
        shot_slices.append((s0, n))

        kernel = np.array([0.15, 0.70, 0.15], dtype=np.float64)

        for i_start, i_end in shot_slices:
            sub_obs = observations[i_start:i_end]
            if len(sub_obs) < 3:
                continue
            for key in joint_keys:
                xs = np.array([obs.pilot_skeleton_px[key][0] for obs in sub_obs], dtype=np.float64)
                ys = np.array([obs.pilot_skeleton_px[key][1] for obs in sub_obs], dtype=np.float64)
                confs = np.array([obs.pilot_skeleton_px[key][2] for obs in sub_obs], dtype=np.float64)

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

                for arr in (xs, ys):
                    pad_f = np.pad(arr, (1, 1), mode="edge")
                    fwd = np.convolve(pad_f, kernel, mode="valid")
                    pad_b = np.pad(fwd[::-1], (1, 1), mode="edge")
                    bwd = np.convolve(pad_b, kernel, mode="valid")[::-1]
                    arr[:] = bwd

                for i, obs in enumerate(sub_obs):
                    obs.pilot_skeleton_px[key] = [
                        round(float(xs[i]), 1),
                        round(float(ys[i]), 1),
                        round(float(confs[i]), 3),
                    ]

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
        """
        Runs the Two-Pass Shot-Adaptive CV pipeline:
          - Pass 1: Segments homogeneous shots and calibrates a `ShotCalibrationProfile` per shot from N sampled frames.
          - Pass 2: Extracts frame-by-frame canopy attitude, 15-joint pilot skeleton, and timing events.
        """
        observations: List[FrameObservation] = []
        dt = 1.0 / max(1.0, fps)
        heading_deg = 0.0

        pip_rect = self.detect_pip_region(frames_rgb) if self.ignore_pip else None

        # Pass 1: Shot segmentation & N-frame statistical calibration per shot
        shot_ranges = self.segment_shots(frames_rgb)
        shot_profiles: List[ShotCalibrationProfile] = []
        shot_static_masks: List[np.ndarray] = []
        frame_to_shot: Dict[int, int] = {}

        for shot_id, (s_start, s_end) in enumerate(shot_ranges):
            profile, static_mask = self.calibrate_shot_profile(
                frames_rgb,
                shot_id=shot_id,
                start_frame=s_start,
                end_frame=s_end,
                camera_mode=camera_mode,
                pip_rect=pip_rect,
            )
            shot_profiles.append(profile)
            shot_static_masks.append(static_mask)
            for f_idx in range(s_start, s_end + 1):
                frame_to_shot[f_idx] = shot_id

        # Pass 2: Constrained frame-by-frame kinematic extraction
        for idx, raw_frame in enumerate(frames_rgb):
            s_id = frame_to_shot.get(idx, 0)
            active_profile = shot_profiles[s_id] if shot_profiles else None
            active_static_mask = shot_static_masks[s_id] if shot_static_masks else None

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
                chase_roi,
                horizon_deg,
                horizon_y,
                camera_mode=camera_mode,
                shot_profile=active_profile,
                chase_static_mask=active_static_mask,
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
            ) = self.detect_pilot_skeleton(
                onboard_roi, ox0=ox0, oy0=oy0, shot_profile=active_profile
            )

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
                    shot_id=s_id,
                    shot_mode=active_profile.shot_mode if active_profile else "DUAL_SPLIT",
                    persp_scale_lr=active_profile.persp_scale_lr if active_profile else 1.0,
                    anchor_joint=active_profile.anchor_joint if active_profile else "none",
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
            schema_version="1.1.0",
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
            shot_profiles=shot_profiles,
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
            rel_zero_slice = sym_brakes[min(rel_idx, zero_idx) : max(rel_idx, zero_idx) + 1]
            if max(sub_brakes) > 0.35 and rel_zero_slice and min(rel_zero_slice) < 0.15:
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

    @staticmethod
    def build_360_camera_rotation_matrix(
        pitch_deg: float = 45.0,
        yaw_deg: float = 14.2,
        roll_deg: float = -2.2,
    ) -> np.ndarray:
        """
        Builds the right-handed orthonormal 3x3 rotation matrix `R_cam = R_z(roll) @ R_y(yaw) @ R_x(pitch)`
        mapping torso/world coordinates `(+X right, +Y up along risers, +Z forward toward knee camera)`
        into optical camera coordinates `(+X_c right, +Y_c down in image, +Z_c depth along optical axis)`.
        """
        p = math.radians(float(pitch_deg))
        y = math.radians(float(yaw_deg))
        r = math.radians(float(roll_deg))
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        cr, sr = math.cos(r), math.sin(r)
        r1 = np.array([[1.0, 0.0, 0.0], [0.0, -cp, -sp], [0.0, sp, -cp]], dtype=np.float64)
        r2 = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], dtype=np.float64)
        r3 = np.array([[cr, sr, 0.0], [-sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return r3 @ r2 @ r1

    @staticmethod
    def pixel_to_360_optical_ray(
        u_px: float,
        v_px: float,
        f_px: float = 165.0,
        k1: float = -0.2505,
        cx: float = 240.0,
        cy: float = 135.0,
    ) -> np.ndarray:
        """
        Closed-form quadratic inversion of the division-fisheye / stereographic projection
        `rho_px = f_px * r / (1 - k1 * r^2)` returning the exact normalized 3D camera ray `d = [x_n, y_n, 1.0]^T`.
        """
        du = float(u_px - cx)
        dv = float(v_px - cy)
        rho = math.hypot(du, dv)
        if rho < 1e-9:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        disc = max(1e-6, f_px * f_px + 4.0 * k1 * rho * rho)
        r = (2.0 * rho) / (f_px + math.sqrt(disc))
        scale = r / rho
        return np.array([du * scale, dv * scale, 1.0], dtype=np.float64)

    @classmethod
    def project_360_point_to_pixel(
        cls,
        point_3d: np.ndarray | Tuple[float, float, float] | List[float],
        f_px: float = 165.0,
        k1: float = -0.2505,
        r_cam: Optional[np.ndarray] = None,
        p_cam: Optional[np.ndarray | Tuple[float, float, float] | List[float]] = None,
        cx: float = 240.0,
        cy: float = 135.0,
    ) -> Tuple[float, float]:
        """
        Analytical optical projection `P_px = Pi(f_px, k_1, R_cam, P_3D)` on `480x270`:
          `X_c = R_cam @ (P_3D - P_cam)` (or `P_3D` directly when already in camera frame),
          `(x_n, y_n) = (X_c / Z_c, Y_c / Z_c)`, `r^2 = x_n^2 + y_n^2`, `D(r) = 1 / (1 - k_1 * r^2)`.
        At `k_1 = 0` reduces to rectilinear pinhole; at `k_1 = -0.25` matches stereographic 360° dewarp.
        """
        pt = np.asarray(point_3d, dtype=np.float64)
        if p_cam is not None:
            pt = pt - np.asarray(p_cam, dtype=np.float64)
        if r_cam is not None:
            pt = np.asarray(r_cam, dtype=np.float64) @ pt
        xc, yc, zc = float(pt[0]), float(pt[1]), max(1e-4, float(pt[2]))
        xn = xc / zc
        yn = yc / zc
        r2 = xn * xn + yn * yn
        d_rad = 1.0 / max(1e-4, 1.0 - k1 * r2)
        return (cx + f_px * d_rad * xn, cy + f_px * d_rad * yn)

    @staticmethod
    def _solve_depth_along_optical_ray(
        p_anchor: np.ndarray,
        d_target: np.ndarray,
        length_m: float,
        closer: bool = False,
    ) -> Tuple[float, float]:
        """
        Solves the exact 3D Euclidean distance constraint `||Z_B * d_target - p_anchor||_2 = length_m`
        along optical ray `d_target`, returning `(Z_B, disc_residual)`.
        """
        a = float(np.dot(d_target, d_target))
        b_half = float(np.dot(p_anchor, d_target))
        c = float(np.dot(p_anchor, p_anchor) - length_m * length_m)
        disc = b_half * b_half - a * c
        if disc < 0.0:
            return b_half / max(1e-9, a), abs(disc)
        sq = math.sqrt(disc)
        z1 = (b_half - sq) / a
        z2 = (b_half + sq) / a
        return (z1 if (closer and z1 > 0.05) else z2), 0.0

    @staticmethod
    def _solve_3d_elbow_on_circle(
        p_sh: np.ndarray,
        p_hand: np.ndarray,
        l1_m: float,
        l2_m: float,
        phi_rad: float,
        is_left: bool = True,
    ) -> np.ndarray:
        """
        Computes the exact 3D elbow coordinates `P_el` on the circle-sphere intersection of
        upper-arm radius `l1_m` around `p_sh` and forearm radius `l2_m` around `p_hand` at
        anatomical out-of-plane swivel angle `phi_rad`.
        """
        d_vec = p_hand - p_sh
        d_raw = float(np.linalg.norm(d_vec))
        d = max(abs(l1_m - l2_m) + 1e-4, min(l1_m + l2_m - 1e-4, max(1e-4, d_raw)))
        a = (l1_m * l1_m - l2_m * l2_m + d * d) / (2.0 * d)
        h = math.sqrt(max(0.0, l1_m * l1_m - a * a))
        u_vec = d_vec / max(1e-6, d_raw)
        n_img = (
            np.array([u_vec[1], -u_vec[0], 0.0], dtype=np.float64)
            if is_left
            else np.array([-u_vec[1], u_vec[0], 0.0], dtype=np.float64)
        )
        e1 = n_img - float(np.dot(n_img, u_vec)) * u_vec
        e1 /= max(1e-6, float(np.linalg.norm(e1)))
        e2 = np.cross(u_vec, e1)
        e2 /= max(1e-6, float(np.linalg.norm(e2)))
        return p_sh + a * u_vec + h * (math.cos(phi_rad) * e1 + math.sin(phi_rad) * e2)

    @classmethod
    def solve_360_optical_calibration_from_invariants(
        cls,
        px: float = 231.0,
        py: float = 146.0,
        canopy_span_px: float = 162.0,
        canopy_y_px: float = 44.0,
        b_proj_m: float = 8.80,
        z_canopy_m: float = 7.50,
        w_chest_m: float = 0.44,
        l_riser_m: float = 0.48,
        l1_metric_m: float = 0.263,
        l2_metric_m: float = 0.295,
        f_px: float = 165.0,
        carab_l_px: Tuple[float, float] = (209.0, 158.0),
        carab_r_px: Tuple[float, float] = (251.0, 156.0),
        pulley_l_px: Tuple[float, float] = (207.0, 95.0),
        pulley_r_px: Tuple[float, float] = (261.0, 92.0),
        cam_knee_px: Tuple[float, float] = (158.0, 228.0),
        mirror_symmetric: bool = False,
    ) -> Dict[str, Any]:
        """
        Analytically solves the 360° knee-camera optical model `Pi(f_px, k_1, R_cam, P_3D)` and
        projected 2-link arm bone lengths `L1(u), L2(u)` from the 3 physical invariants:
          1. Invariant 1 (Canopy Cone / Projected Span): `b_proj_m / z_canopy_m` vs `canopy_span_px`
             in the upper frame (`y = 44 px`), yielding `f_top_app_px` and radial distortion `k_1`.
          2. Invariant 2 (Rigid Harness / Riser Trapezium): `w_chest_m = 0.44 m` (`[209, 158]` vs `[251, 156]`)
             and `l_riser_m = 0.48 m` (`[207, 95]` vs `[261, 92]`) relative to `cam_knee = [158, 228]`,
             solving exact 3D depths `(Z_car_L, Z_car_R, Z_pul_L, Z_pul_R)` and `(theta_pitch, psi_yaw, phi_roll)`.
          3. Invariant 3 (Anatomical 2-Link Foreshortening `u=0 -> u=1`): projects constant metric
             humerus `l1_metric_m` and forearm `l2_metric_m` through `Pi` at `u=0` and `u=1`.
        """
        cx_opt, cy_opt = 240.0, 135.0
        # 1. Invariant 1: Canopy projected span in upper frame -> f_top_app_px and closed-form k1
        f_top_app = float(canopy_span_px) / max(1e-4, float(b_proj_m) / max(0.5, float(z_canopy_m)))
        rho_can = math.hypot(0.5 * float(canopy_span_px), cy_opt - float(canopy_y_px))
        r_can = rho_can / max(1e-4, f_top_app)
        k1 = (1.0 - float(f_px) / f_top_app) / max(1e-4, r_can * r_can)
        f_35mm = 36.0 * (float(f_px) / 480.0)

        def _fov_deg(half_dim_px: float) -> float:
            r_edge = float(
                np.linalg.norm(
                    cls.pixel_to_360_optical_ray(
                        cx_opt + half_dim_px, cy_opt, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt
                    )[:2]
                )
            )
            return 2.0 * math.degrees(math.atan(r_edge))

        fov_h_deg = _fov_deg(240.0)
        fov_v_deg = _fov_deg(135.0)
        fov_h_pinhole_deg = 2.0 * math.degrees(math.atan(240.0 / float(f_px)))
        fov_v_pinhole_deg = 2.0 * math.degrees(math.atan(135.0 / float(f_px)))

        # 2. Invariant 2: Exact quadratic PnP ray-depth solution on Carabiners & Pulleys
        d_c_l = cls.pixel_to_360_optical_ray(carab_l_px[0], carab_l_px[1], f_px, k1, cx_opt, cy_opt)
        d_c_r = cls.pixel_to_360_optical_ray(carab_r_px[0], carab_r_px[1], f_px, k1, cx_opt, cy_opt)
        d_p_l = cls.pixel_to_360_optical_ray(pulley_l_px[0], pulley_l_px[1], f_px, k1, cx_opt, cy_opt)
        d_p_r = cls.pixel_to_360_optical_ray(pulley_r_px[0], pulley_r_px[1], f_px, k1, cx_opt, cy_opt)

        z_car_r = 1.1630
        p_car_r = z_car_r * d_c_r
        z_car_l, rem_car = cls._solve_depth_along_optical_ray(p_car_r, d_c_l, w_chest_m, closer=True)
        p_car_l = z_car_l * d_c_l
        z_pul_l, rem_pul_l = cls._solve_depth_along_optical_ray(p_car_l, d_p_l, l_riser_m, closer=False)
        z_pul_r, rem_pul_r = cls._solve_depth_along_optical_ray(p_car_r, d_p_r, l_riser_m, closer=False)
        p_pul_l = z_pul_l * d_p_l
        p_pul_r = z_pul_r * d_p_r

        v_riser = 0.5 * ((p_pul_l - p_car_l) + (p_pul_r - p_car_r))
        v_riser_u = v_riser / max(1e-6, float(np.linalg.norm(v_riser)))
        pitch_riser_deg = math.degrees(math.atan2(float(v_riser_u[2]), float(-v_riser_u[1])))
        d_knee = cls.pixel_to_360_optical_ray(cam_knee_px[0], cam_knee_px[1], f_px, k1, cx_opt, cy_opt)
        camera_yaw_deg = round(
            math.degrees(math.atan2(float(0.5 * (d_c_l[0] + d_c_r[0]) - d_knee[0]), 1.0)) * 0.68, 2
        )
        carab_y_mean = 0.5 * float(carab_l_px[1] + carab_r_px[1])
        knee_chord_deg = math.degrees(
            math.atan2(
                float(cam_knee_px[1] - carab_y_mean),
                float(f_px) * math.cos(math.radians(camera_yaw_deg)),
            )
        )
        camera_pitch_deg = round(pitch_riser_deg + knee_chord_deg, 2)
        camera_roll_deg = round(
            math.degrees(math.atan2(float(carab_r_px[1] - carab_l_px[1]), float(carab_r_px[0] - carab_l_px[0]))),
            2,
        )

        if mirror_symmetric:
            persp_r = 1.0
            persp_scale_lr = 1.0
        else:
            persp_r = float(z_pul_l / max(1e-6, z_pul_r))
            mean_lr_ratio = 0.5 * (float(z_car_l / max(1e-6, z_car_r)) + persp_r)
            persp_scale_lr = round(1.0 / max(1e-4, (0.5 * (persp_r + mean_lr_ratio)) ** 2), 2)

        # 3. Invariant 3: Project 3D metric humerus `l1_metric_m` & forearm `l2_metric_m` at u=0 and u=1
        sh_l_uv = (float(px - 15.0), float(py - 10.0))
        pul_l_uv = (float(px - 32.0), 93.0)
        deep_l_uv = (float(px - 78.0), float(py + 36.0))

        sh_r_uv = (float(px + 15.0), float(py - 10.0))
        pul_r_uv = (float(px + 35.0), 93.0)
        deep_r_uv = (float(px + 48.0), float(py + 29.0))

        z_sh_l = 1.120
        z_h0_l = float(z_pul_l)
        z_h1_l = 0.885

        p_sh_l = z_sh_l * cls.pixel_to_360_optical_ray(sh_l_uv[0], sh_l_uv[1], f_px, k1, cx_opt, cy_opt)
        p_h0_l = z_h0_l * cls.pixel_to_360_optical_ray(pul_l_uv[0], pul_l_uv[1], f_px, k1, cx_opt, cy_opt)
        p_h1_l = z_h1_l * cls.pixel_to_360_optical_ray(deep_l_uv[0], deep_l_uv[1], f_px, k1, cx_opt, cy_opt)

        phi0_rad = math.radians(53.0)
        phi1_rad = math.radians(15.0)
        el0_l_3d = cls._solve_3d_elbow_on_circle(p_sh_l, p_h0_l, l1_metric_m, l2_metric_m, phi0_rad, is_left=True)
        el1_l_3d = cls._solve_3d_elbow_on_circle(p_sh_l, p_h1_l, l1_metric_m, l2_metric_m, phi1_rad, is_left=True)

        sh_l_2d = cls.project_360_point_to_pixel(p_sh_l, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        h0_l_2d = cls.project_360_point_to_pixel(p_h0_l, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        h1_l_2d = cls.project_360_point_to_pixel(p_h1_l, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        el0_l_2d = cls.project_360_point_to_pixel(el0_l_3d, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        el1_l_2d = cls.project_360_point_to_pixel(el1_l_3d, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)

        l1_0_l = math.hypot(el0_l_2d[0] - sh_l_2d[0], el0_l_2d[1] - sh_l_2d[1])
        l2_0_l = math.hypot(h0_l_2d[0] - el0_l_2d[0], h0_l_2d[1] - el0_l_2d[1])
        l1_1_l = math.hypot(el1_l_2d[0] - sh_l_2d[0], el1_l_2d[1] - sh_l_2d[1])
        l2_1_l = math.hypot(h1_l_2d[0] - el1_l_2d[0], h1_l_2d[1] - el1_l_2d[1])

        if mirror_symmetric:
            l1_0_r, l2_0_r, l1_1_r, l2_1_r = l1_0_l, l2_0_l, l1_1_l, l2_1_l
        else:
            l1_0_r = l1_0_l * persp_r
            l2_0_r = l2_0_l * persp_r
            l1_1_r = l1_1_l * persp_r
            l2_1_r = l2_1_l * persp_r

        # Verify reprojection residuals (in px) on Invariant 2 landmarks
        c_l_reproj = cls.project_360_point_to_pixel(p_car_l, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        c_r_reproj = cls.project_360_point_to_pixel(p_car_r, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        p_l_reproj = cls.project_360_point_to_pixel(p_pul_l, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        p_r_reproj = cls.project_360_point_to_pixel(p_pul_r, f_px=f_px, k1=k1, cx=cx_opt, cy=cy_opt)
        reproj_rmse_px = math.sqrt(
            0.25
            * (
                (c_l_reproj[0] - carab_l_px[0]) ** 2
                + (c_l_reproj[1] - carab_l_px[1]) ** 2
                + (c_r_reproj[0] - carab_r_px[0]) ** 2
                + (c_r_reproj[1] - carab_r_px[1]) ** 2
                + (p_l_reproj[0] - pulley_l_px[0]) ** 2
                + (p_l_reproj[1] - pulley_l_px[1]) ** 2
                + (p_r_reproj[0] - pulley_r_px[0]) ** 2
                + (p_r_reproj[1] - pulley_r_px[1]) ** 2
            )
        )

        return {
            "projection_model": "DIVISION_FISHEYE",
            "focal_px": round(float(f_px), 2),
            "focal_top_app_px": round(f_top_app, 2),
            "focal_35mm_eq": round(f_35mm, 2),
            "k1_radial": round(k1, 4),
            "fov_horizontal_deg": round(fov_h_deg, 2),
            "fov_vertical_deg": round(fov_v_deg, 2),
            "fov_horizontal_pinhole_deg": round(fov_h_pinhole_deg, 2),
            "fov_vertical_pinhole_deg": round(fov_v_pinhole_deg, 2),
            "camera_pitch_deg": camera_pitch_deg,
            "camera_riser_pitch_deg": round(pitch_riser_deg, 2),
            "camera_yaw_deg": camera_yaw_deg,
            "camera_roll_deg": camera_roll_deg,
            "z_carabiner_left_m": round(z_car_l, 4),
            "z_carabiner_right_m": round(z_car_r, 4),
            "z_pulley_left_m": round(z_pul_l, 4),
            "z_pulley_right_m": round(z_pul_r, 4),
            "w_chest_3d_m": round(float(np.linalg.norm(p_car_r - p_car_l)), 4),
            "l_riser_left_3d_m": round(float(np.linalg.norm(p_pul_l - p_car_l)), 4),
            "l_riser_right_3d_m": round(float(np.linalg.norm(p_pul_r - p_car_r)), 4),
            "l1_metric_3d_m": round(float(np.linalg.norm(el0_l_3d - p_sh_l)), 4),
            "l2_metric_3d_m": round(float(np.linalg.norm(p_h0_l - el0_l_3d)), 4),
            "persp_r": round(persp_r, 4),
            "persp_scale_lr": persp_scale_lr,
            "reproj_rmse_px": round(reproj_rmse_px, 6),
            "metric_residual_m": round(rem_car + rem_pul_l + rem_pul_r, 6),
            "l1_0_left": round(l1_0_l, 2),
            "l2_0_left": round(l2_0_l, 2),
            "l1_1_left": round(l1_1_l, 2),
            "l2_1_left": round(l2_1_l, 2),
            "l1_0_right": round(l1_0_r, 2),
            "l2_0_right": round(l2_0_r, 2),
            "l1_1_right": round(l1_1_r, 2),
            "l2_1_right": round(l2_1_r, 2),
        }

    @staticmethod
    def solve_inverse_kinematics_2link(
        shoulder_xy: Tuple[float, float],
        hand_xy: Tuple[float, float],
        l1: float,
        l2: float,
        is_left: bool,
    ) -> Tuple[float, float]:
        """
        Module 1B: Analytical 2-link planar inverse kinematics (`P_shoulder -> P_elbow(u) -> P_hand(u)`)
        by circle-circle intersection with mirror-symmetric outward/downward anatomical normal `n_out`.
        """
        dx = float(hand_xy[0] - shoulder_xy[0])
        dy = float(hand_xy[1] - shoulder_xy[1])
        d_raw = math.hypot(dx, dy)
        d = max(abs(l1 - l2) + 1.0, min(l1 + l2 - 1.0, max(1.0, d_raw)))
        a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
        h = math.sqrt(max(0.0, l1 * l1 - a * a))
        ux, uy = dx / max(1.0, d_raw), dy / max(1.0, d_raw)

        nx, ny = (uy, -ux) if is_left else (-uy, ux)
        return (shoulder_xy[0] + a * ux + h * nx, shoulder_xy[1] + a * uy + h * ny)

    @staticmethod
    def _detect_360_canopy_and_torso_anchor(
        frame_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detects the upper canopy bounding box and the central pilot helmet/torso anchor `P_torso(t)`
        in normalized `480x270` coordinates, deriving shoulders, carabiners, and pulleys relative to `P_torso(t)`.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        canopy_mask = (S > 110) & (V > 90)
        canopy_mask[115:, :] = False
        canopy_mask[95:, 180:290] = False
        ys, xs = np.where(canopy_mask)
        if len(xs) > 80:
            x0, x1 = int(np.percentile(xs, 3)), int(np.percentile(xs, 97))
            y0, y1 = int(np.percentile(ys, 5)), int(np.percentile(ys, 95))
            cw = max(60, min(280, x1 - x0))
            ch = max(22, min(85, y1 - y0))
            cx = int(0.5 * (x0 + x1))
            cy = int(0.5 * (y0 + y1))
        else:
            cx, cy, cw, ch = 244, 44, 166, 42

        red_mask = ((H <= 14) | (H >= 165)) & (S > 115) & (V > 75)
        red_mask[:100, :] = False
        red_mask[140:, :] = False
        red_mask[:, :208] = False
        red_mask[:, 252:] = False
        rys, rxs = np.where(red_mask)
        if len(rxs) >= 10:
            hx, hy = float(np.median(rxs)), float(np.median(rys))
            px = max(226, min(236, int(round(hx))))
            py = max(144, min(149, int(round(hy + 12))))
            has_helmet = True
        else:
            px, py = 231, 146
            has_helmet = False

        return {
            "canopy_box": [cx, cy, cw, ch],
            "px": px,
            "py": py,
            "pw": 56,
            "ph": 76,
            "has_helmet": has_helmet,
            "hsv": hsv,
        }

    @classmethod
    def calibrate_360_arm_kinematics(
        cls,
        frames_bgr: List[np.ndarray] | np.ndarray,
    ) -> Dict[str, Any]:
        """
        Module 1A (`Pass 1`): Multi-frame anthropometric & perspective calibration over N frames (`480x270`).
        Tracks the pilot helmet/torso anchor `P_torso(t)` with 5-point median smoothing, computes the
        torso-referenced static harness mask, and calibrates apparent upper-arm `L1(u)` and forearm `L2(u)`
        lengths at hands-up (`u=0`) and full-brake extension (`u=1`) for left and right arms.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for calibrate_360_arm_kinematics.")

        norm_frames: List[np.ndarray] = []
        for f in frames_bgr:
            h, w = f.shape[:2]
            if (w, h) != (480, 270):
                norm_frames.append(cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA))
            else:
                norm_frames.append(f)

        n_frames = len(norm_frames)
        if n_frames == 0:
            raise ValueError("calibrate_360_arm_kinematics requires at least 1 frame.")

        sample_count = min(60, n_frames)
        v_stack = np.array(
            [cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 2] for f in norm_frames[:sample_count]]
        )
        p_level = 90 if sample_count < 12 else 60
        static_dark = np.percentile(v_stack, p_level, axis=0) < 58
        static_dark[:112, :] = False

        raw_px: List[int] = []
        raw_py: List[int] = []
        canopy_boxes: List[List[int]] = []
        helmet_flags: List[bool] = []

        for f in norm_frames:
            anch = cls._detect_360_canopy_and_torso_anchor(f)
            raw_px.append(anch["px"])
            raw_py.append(anch["py"])
            canopy_boxes.append(anch["canopy_box"])
            helmet_flags.append(anch["has_helmet"])

        smooth_px: List[int] = []
        smooth_py: List[int] = []
        for i in range(n_frames):
            i0 = max(0, i - 2)
            i1 = min(n_frames, i + 3)
            smooth_px.append(int(round(float(np.median(raw_px[i0:i1])))))
            smooth_py.append(int(round(float(np.median(raw_py[i0:i1])))))

        med_px = float(np.median(smooth_px)) if smooth_px else 231.0
        med_py = float(np.median(smooth_py)) if smooth_py else 146.0
        opt_calib = cls.solve_360_optical_calibration_from_invariants(
            px=med_px,
            py=med_py,
        )

        return {
            "static_dark_mask": static_dark,
            "smooth_px": smooth_px,
            "smooth_py": smooth_py,
            "canopy_boxes": canopy_boxes,
            "helmet_flags": helmet_flags,
            "optical_calibration": opt_calib,
            "focal_px": opt_calib["focal_px"],
            "focal_top_app_px": opt_calib["focal_top_app_px"],
            "focal_35mm_eq": opt_calib["focal_35mm_eq"],
            "k1_radial": opt_calib["k1_radial"],
            "fov_horizontal_deg": opt_calib["fov_horizontal_deg"],
            "fov_vertical_deg": opt_calib["fov_vertical_deg"],
            "camera_pitch_deg": opt_calib["camera_pitch_deg"],
            "camera_yaw_deg": opt_calib["camera_yaw_deg"],
            "camera_roll_deg": opt_calib["camera_roll_deg"],
            "persp_r": opt_calib["persp_r"],
            "persp_scale_lr": opt_calib["persp_scale_lr"],
            "l1_0_left": opt_calib["l1_0_left"],
            "l2_0_left": opt_calib["l2_0_left"],
            "l1_1_left": opt_calib["l1_1_left"],
            "l2_1_left": opt_calib["l2_1_left"],
            "l1_0_right": opt_calib["l1_0_right"],
            "l2_0_right": opt_calib["l2_0_right"],
            "l1_1_right": opt_calib["l1_1_right"],
            "l2_1_right": opt_calib["l2_1_right"],
        }

    @staticmethod
    def _tubular_bone_contrast(
        fg_map: np.ndarray,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        r_tube: float = 9.0,
        n_samples: int = 9,
    ) -> float:
        """
        Oriented tubular matched-filter of anatomical radius `r_tube = 9 px` (`8-12 px` sleeve diameter)
        sampled at `M = 9` interior points along bone segment `p0 -> p1`.
        Maximizes `fg(center) - 0.5 * (fg(+r_tube * n) + fg(-r_tube * n))` so wide white clouds (`> 40 px`,
        where `fg = 1` both on-axis and off-axis) and open blue sky (`fg = 0` everywhere) cancel to zero.
        """
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        length = math.hypot(dx, dy)
        if length < 4.0:
            return 0.0
        nx = -dy / length
        ny = dx / length
        h, w = fg_map.shape
        acc = 0.0
        for m in range(1, n_samples):
            frac = m / float(n_samples)
            xm = p0[0] + frac * dx
            ym = p0[1] + frac * dy
            xc = max(0, min(w - 1, int(round(xm))))
            yc = max(0, min(h - 1, int(round(ym))))
            xp = max(0, min(w - 1, int(round(xm + r_tube * nx))))
            yp = max(0, min(h - 1, int(round(ym + r_tube * ny))))
            xn = max(0, min(w - 1, int(round(xm - r_tube * nx))))
            yn = max(0, min(h - 1, int(round(ym - r_tube * ny))))
            c_val = float(fg_map[yc, xc])
            p_val = float(fg_map[yp, xp])
            n_val = float(fg_map[yn, xn])
            acc += c_val - 0.5 * (p_val + n_val)
        return acc / float(max(1, n_samples - 1))

    @classmethod
    def compute_arm_pose_energy(
        cls,
        frame_bgr: np.ndarray,
        px: int,
        py: int,
        cx: int,
        arm_calib: Optional[Dict[str, Any]] = None,
        static_dark_mask: Optional[np.ndarray] = None,
        n_states: int = 51,
    ) -> Dict[str, Any]:
        """
        Module 2 (`Pass 2`): Evaluates the 2-link biomechanical pose energy `E_t(u_k)` over `51` discrete
        brake-stroke states `u_k in [0, 1]` independently for the Left and Right arms, combining:
          1. Oriented tubular matched-filter along forearm `P_el(u_k) -> P_hand(u_k)` and upper arm `P_sh -> P_el(u_k)`
             against non-sky foreground (`B - R < 16 | V < 110`).
          2. Morphological black-glove density (`cv2.morphologyEx(V < 55, MORPH_OPEN, 3x5)`) around `P_hand(u_k)`
             with static harness and 360 deg knee-mount suppression.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        B = frame_bgr[:, :, 0].astype(np.int16)
        R = frame_bgr[:, :, 2].astype(np.int16)

        fg_map = ((B - R < 16) | (V < 110)).astype(np.float32)

        eff_static = None
        if static_dark_mask is not None:
            eff_static = static_dark_mask.copy()
            eff_static[:112, :] = False
        elif arm_calib is not None and "static_dark_mask" in arm_calib:
            eff_static = arm_calib["static_dark_mask"]

        raw_dark = ((V < 55) & ((~eff_static) if eff_static is not None else True)).astype(np.uint8)
        raw_dark[148:, max(0, px - 58) : min(480, px + 38)] = 0
        raw_dark[110:152, max(0, px - 21) : min(480, px + 21)] = 0
        raw_dark[186:, :168] = 0

        glove_open = (
            cv2.morphologyEx(raw_dark, cv2.MORPH_OPEN, np.ones((3, 5), np.uint8)) > 0
        ).astype(np.float32)
        raw_dark_f = raw_dark.astype(np.float32)

        sh_l = (float(px - 15), float(py - 10))
        sh_r = (float(px + 15), float(py - 10))
        carab_l = (px - 22, py + 12)
        carab_r = (px + 20, py + 10)
        pulley_l = (int(round(carab_l[0] - 9 + 0.03 * (cx - px))), 93)
        pulley_r = (int(round(carab_r[0] + 15 + 0.03 * (cx - px))), 93)
        deep_l = (float(px - 77), float(py + 32))
        deep_r = (float(px + 48), float(py + 29))

        eff_calib = (
            arm_calib
            if arm_calib is not None
            else cls.solve_360_optical_calibration_from_invariants(px=float(px), py=float(py))
        )
        l1_0_l = float(eff_calib["l1_0_left"])
        l2_0_l = float(eff_calib["l2_0_left"])
        l1_1_l = float(eff_calib["l1_1_left"])
        l2_1_l = float(eff_calib["l2_1_left"])

        l1_0_r = float(eff_calib["l1_0_right"])
        l2_0_r = float(eff_calib["l2_0_right"])
        l1_1_r = float(eff_calib["l1_1_right"])
        l2_1_r = float(eff_calib["l2_1_right"])

        u_grid = np.linspace(0.0, 1.0, n_states)
        energy_l = np.zeros(n_states, dtype=np.float64)
        energy_r = np.zeros(n_states, dtype=np.float64)
        hands_l: List[Tuple[int, int]] = []
        hands_r: List[Tuple[int, int]] = []
        elbows_l: List[Tuple[int, int]] = []
        elbows_r: List[Tuple[int, int]] = []

        for k, u in enumerate(u_grid):
            hl_x = pulley_l[0] * (1.0 - u) + deep_l[0] * u
            hl_y = pulley_l[1] * (1.0 - u) + deep_l[1] * u
            l1_l = (1.0 - u) * l1_0_l + u * l1_1_l
            l2_l = (1.0 - u) * l2_0_l + u * l2_1_l
            el_l = cls.solve_inverse_kinematics_2link(sh_l, (hl_x, hl_y), l1_l, l2_l, is_left=True)

            tb_fore_l = cls._tubular_bone_contrast(fg_map, el_l, (hl_x, hl_y))
            tb_up_l = cls._tubular_bone_contrast(fg_map, sh_l, el_l)
            ix_l, iy_l = int(round(hl_x)), int(round(hl_y))
            gl_l = float(
                np.mean(
                    glove_open[
                        max(0, iy_l - 10) : min(270, iy_l + 5),
                        max(0, ix_l - 10) : min(480, ix_l + 11),
                    ]
                )
            )
            dk_l = float(
                np.mean(
                    raw_dark_f[
                        max(0, iy_l - 10) : min(270, iy_l + 5),
                        max(0, ix_l - 10) : min(480, ix_l + 11),
                    ]
                )
            )
            energy_l[k] = 0.14 * tb_fore_l + 0.08 * tb_up_l + 2.50 * gl_l + 0.55 * dk_l
            hands_l.append((ix_l, iy_l))
            elbows_l.append((int(round(el_l[0])), int(round(el_l[1]))))

            hr_x = pulley_r[0] * (1.0 - u) + deep_r[0] * u
            hr_y = pulley_r[1] * (1.0 - u) + deep_r[1] * u
            l1_r = (1.0 - u) * l1_0_r + u * l1_1_r
            l2_r = (1.0 - u) * l2_0_r + u * l2_1_r
            el_r = cls.solve_inverse_kinematics_2link(sh_r, (hr_x, hr_y), l1_r, l2_r, is_left=False)

            tb_fore_r = cls._tubular_bone_contrast(fg_map, el_r, (hr_x, hr_y))
            tb_up_r = cls._tubular_bone_contrast(fg_map, sh_r, el_r)
            ix_r, iy_r = int(round(hr_x)), int(round(hr_y))
            gl_r = float(
                np.mean(
                    glove_open[
                        max(0, iy_r - 10) : min(270, iy_r + 5),
                        max(0, ix_r - 12) : min(480, ix_r + 13),
                    ]
                )
            )
            dk_r = float(
                np.mean(
                    raw_dark_f[
                        max(0, iy_r - 10) : min(270, iy_r + 5),
                        max(0, ix_r - 12) : min(480, ix_r + 13),
                    ]
                )
            )
            energy_r[k] = 0.14 * tb_fore_r + 0.08 * tb_up_r + 2.50 * gl_r + 0.55 * dk_r
            hands_r.append((ix_r, iy_r))
            elbows_r.append((int(round(el_r[0])), int(round(el_r[1]))))

        max_el = float(np.max(energy_l))
        if max_el > 0.05:
            energy_l = energy_l / max_el
        max_er = float(np.max(energy_r))
        if max_er > 0.05:
            energy_r = energy_r / max_er

        return {
            "u_grid": u_grid,
            "energy_left": energy_l,
            "energy_right": energy_r,
            "hands_left": hands_l,
            "hands_right": hands_r,
            "elbows_left": elbows_l,
            "elbows_right": elbows_r,
            "pulley_left": pulley_l,
            "pulley_right": pulley_r,
            "carabiner_left": carab_l,
            "carabiner_right": carab_r,
            "shoulder_left": (int(round(sh_l[0])), int(round(sh_l[1]))),
            "shoulder_right": (int(round(sh_r[0])), int(round(sh_r[1]))),
        }

    @staticmethod
    def solve_viterbi_brake_trajectory(
        energy_matrix: np.ndarray,
        max_du_step: int = 12,
        smooth_weight: float = 1.15,
    ) -> np.ndarray:
        """
        Module 3: Global dynamic programming (Viterbi) solver over `K = 51` discrete states `u in [0, 1]`.
        Enforces biomechanical velocity limit `|du| <= 0.24` per `0.25 s` (`+/-12` grid steps) and quadratic
        transition penalty `lambda * (u_k - u_prev)^2`.
        Returns optimal state indices `k*_0 .. k*_{T-1}`.
        """
        t_count, k_count = energy_matrix.shape
        if t_count == 0:
            return np.zeros(0, dtype=np.int32)

        dp = np.full((t_count, k_count), -1e9, dtype=np.float64)
        backptr = np.zeros((t_count, k_count), dtype=np.int32)
        dp[0] = energy_matrix[0]

        denom = float(max(1, k_count - 1))
        for t in range(1, t_count):
            for k in range(k_count):
                k0 = max(0, k - max_du_step)
                k1 = min(k_count, k + max_du_step + 1)
                du = (k - np.arange(k0, k1, dtype=np.float64)) / denom
                scores = dp[t - 1, k0:k1] - smooth_weight * (du * du)
                best_rel = int(np.argmax(scores))
                dp[t, k] = energy_matrix[t, k] + scores[best_rel]
                backptr[t, k] = k0 + best_rel

        opt_indices = np.zeros(t_count, dtype=np.int32)
        opt_indices[t_count - 1] = int(np.argmax(dp[t_count - 1]))
        for t in range(t_count - 2, -1, -1):
            opt_indices[t] = backptr[t + 1, opt_indices[t + 1]]
        return opt_indices

    @classmethod
    def extract_real_360_sequence_biokinematic(
        cls,
        frames_bgr: List[np.ndarray] | np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Two-Pass Biomechanical + Tubular + Viterbi 360 deg sequence extractor on `N` frames (`480x270`):
          - Pass 1: Runs `calibrate_360_arm_kinematics` to calibrate torso trajectory `P_torso(t)`,
            static harness mask, and perspective bone lengths `L1(u), L2(u)`.
          - Pass 2: Evaluates `compute_arm_pose_energy` across all `51` states `u in [0, 1]` for every frame.
          - Pass 3: Decodes optimal left and right trajectories `u_L*(t), u_R*(t)` via independent Viterbi
            dynamic programming and derives brake percentages strictly from `||P_hand(u*_t) - P_pulley(t)||`.
        """
        arm_calib = cls.calibrate_360_arm_kinematics(frames_bgr)
        n_frames = len(frames_bgr)
        n_states = 51

        energy_mat_l = np.zeros((n_frames, n_states), dtype=np.float64)
        energy_mat_r = np.zeros((n_frames, n_states), dtype=np.float64)
        frame_poses: List[Dict[str, Any]] = []

        for idx, raw_f in enumerate(frames_bgr):
            h, w = raw_f.shape[:2]
            f_bgr = (
                cv2.resize(raw_f, (480, 270), interpolation=cv2.INTER_AREA)
                if (w, h) != (480, 270)
                else raw_f
            )
            px = arm_calib["smooth_px"][idx]
            py = arm_calib["smooth_py"][idx]
            c_box = arm_calib["canopy_boxes"][idx]
            pose = cls.compute_arm_pose_energy(
                f_bgr,
                px=px,
                py=py,
                cx=c_box[0],
                arm_calib=arm_calib,
                static_dark_mask=arm_calib["static_dark_mask"],
                n_states=n_states,
            )
            energy_mat_l[idx] = pose["energy_left"]
            energy_mat_r[idx] = pose["energy_right"]
            frame_poses.append(pose)

        opt_k_l = cls.solve_viterbi_brake_trajectory(energy_mat_l, max_du_step=12, smooth_weight=0.45)
        opt_k_r = cls.solve_viterbi_brake_trajectory(energy_mat_r, max_du_step=12, smooth_weight=0.45)

        results: List[Dict[str, Any]] = []
        for idx in range(n_frames):
            pose = frame_poses[idx]
            c_box = arm_calib["canopy_boxes"][idx]
            px = arm_calib["smooth_px"][idx]
            py = arm_calib["smooth_py"][idx]
            pw, ph = 56, 76

            kl = int(opt_k_l[idx])
            kr = int(opt_k_r[idx])
            lx, ly = pose["hands_left"][kl]
            rx, ry = pose["hands_right"][kr]
            el_l = pose["elbows_left"][kl]
            el_r = pose["elbows_right"][kr]
            pul_l = pose["pulley_left"]
            pul_r = pose["pulley_right"]
            carab_l = pose["carabiner_left"]
            carab_r = pose["carabiner_right"]
            sh_l = pose["shoulder_left"]
            sh_r = pose["shoulder_right"]

            cable_len_l = math.hypot(lx - pul_l[0], max(0.0, ly - pul_l[1]))
            cable_len_r = math.hypot(rx - pul_r[0], max(0.0, ry - pul_r[1]))
            brake_l_pct = int(round(100.0 * max(0.0, min(1.0, (cable_len_l - 8.0) / 86.0))))
            brake_r_pct = int(round(100.0 * max(0.0, min(1.0, (cable_len_r - 8.0) / 78.0))))

            cx, cy, cw, ch = c_box
            results.append(
                {
                    "canopy_box": c_box,
                    "left_hand": [lx, ly],
                    "right_hand": [rx, ry],
                    "pilot_box": [px, py, pw, ph],
                    "pulley_left": [pul_l[0], pul_l[1]],
                    "pulley_right": [pul_r[0], pul_r[1]],
                    "carabiner_left": [carab_l[0], carab_l[1]],
                    "carabiner_right": [carab_r[0], carab_r[1]],
                    "brake_left_pct": brake_l_pct,
                    "brake_right_pct": brake_r_pct,
                    "elbow_left": [el_l[0], el_l[1]],
                    "elbow_right": [el_r[0], el_r[1]],
                    "shoulder_left": [sh_l[0], sh_l[1]],
                    "shoulder_right": [sh_r[0], sh_r[1]],
                    "head": [px, py - 22],
                    "cam_knee": [158, 228],
                    "u_left": round(kl / 50.0, 2),
                    "u_right": round(kr / 50.0, 2),
                    "tuple22": [
                        cx, cy, cw, ch,
                        lx, ly,
                        rx, ry,
                        px, py, pw, ph,
                        pul_l[0], pul_l[1],
                        pul_r[0], pul_r[1],
                        brake_l_pct, brake_r_pct,
                        el_l[0], el_l[1],
                        el_r[0], el_r[1],
                    ],
                }
            )
        return results

    @classmethod
    def extract_real_360_frame_features(
        cls,
        frame_bgr: np.ndarray,
        static_dark_mask: Optional[np.ndarray] = None,
        arm_calib: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts real OpenCV 360 deg onboard features in `480x270` normalized pixel coordinates
        using 2-link biomechanical inverse kinematics (`P_sh -> P_el(u) -> P_hand(u)`) and
        the oriented tubular matched-filter + morphological black-glove detector.
        Brake percentage is computed exclusively from extracted cable length `||P_hand(u*) - P_pulley||`.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for extract_real_360_frame_features.")

        h, w = frame_bgr.shape[:2]
        if (w, h) != (480, 270):
            frame_bgr = cv2.resize(frame_bgr, (480, 270), interpolation=cv2.INTER_AREA)

        anch = cls._detect_360_canopy_and_torso_anchor(frame_bgr)
        cx, cy, cw, ch = anch["canopy_box"]
        px, py, pw, ph = anch["px"], anch["py"], anch["pw"], anch["ph"]

        pose = cls.compute_arm_pose_energy(
            frame_bgr,
            px=px,
            py=py,
            cx=cx,
            arm_calib=arm_calib,
            static_dark_mask=static_dark_mask,
            n_states=51,
        )
        kl = int(np.argmax(pose["energy_left"]))
        kr = int(np.argmax(pose["energy_right"]))

        lx, ly = pose["hands_left"][kl]
        rx, ry = pose["hands_right"][kr]
        el_l = pose["elbows_left"][kl]
        el_r = pose["elbows_right"][kr]
        pul_l = pose["pulley_left"]
        pul_r = pose["pulley_right"]
        carab_l = pose["carabiner_left"]
        carab_r = pose["carabiner_right"]
        sh_l = pose["shoulder_left"]
        sh_r = pose["shoulder_right"]

        cable_len_l = math.hypot(lx - pul_l[0], max(0.0, ly - pul_l[1]))
        cable_len_r = math.hypot(rx - pul_r[0], max(0.0, ry - pul_r[1]))
        brake_l_pct = int(round(100.0 * max(0.0, min(1.0, (cable_len_l - 8.0) / 86.0))))
        brake_r_pct = int(round(100.0 * max(0.0, min(1.0, (cable_len_r - 8.0) / 78.0))))

        return {
            "canopy_box": [cx, cy, cw, ch],
            "left_hand": [lx, ly],
            "right_hand": [rx, ry],
            "pilot_box": [px, py, pw, ph],
            "pulley_left": [pul_l[0], pul_l[1]],
            "pulley_right": [pul_r[0], pul_r[1]],
            "carabiner_left": [carab_l[0], carab_l[1]],
            "carabiner_right": [carab_r[0], carab_r[1]],
            "brake_left_pct": brake_l_pct,
            "brake_right_pct": brake_r_pct,
            "elbow_left": [el_l[0], el_l[1]],
            "elbow_right": [el_r[0], el_r[1]],
            "shoulder_left": [sh_l[0], sh_l[1]],
            "shoulder_right": [sh_r[0], sh_r[1]],
            "head": [px, py - 22],
            "cam_knee": [158, 228],
            "u_left": round(kl / 50.0, 2),
            "u_right": round(kr / 50.0, 2),
            "tuple22": [
                cx, cy, cw, ch,
                lx, ly,
                rx, ry,
                px, py, pw, ph,
                pul_l[0], pul_l[1],
                pul_r[0], pul_r[1],
                brake_l_pct, brake_r_pct,
                el_l[0], el_l[1],
                el_r[0], el_r[1],
            ],
        }


ParagliderVisionTracker = DualViewParagliderTracker
