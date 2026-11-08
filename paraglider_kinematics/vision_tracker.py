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
    with External Chase/Side View + Onboard Pilot POV Split-Screen/PIP.

    Extracts:
      1. Horizon line tilt angle via vertical Sobel gradient ridge regression.
      2. Paraglider canopy centroid, oriented ellipse axes, and bank/pitch attitude via 2nd-order moments.
      3. Pilot/harness centroid suspended below the canopy and true pendulum angle wrt gravity.
      4. Left/Right pilot brake hand positions wrt pulley (0%) and carabiner (100%) in the Onboard POV ROI.
      5. Command application timing events and canopy reaction phase lags.
    """

    def __init__(
        self,
        wing: Optional[WingGeometrySpec] = None,
        chase_roi_norm: Tuple[float, float, float, float] = (0.0, 0.0, 0.68, 1.0),
        onboard_roi_norm: Tuple[float, float, float, float] = (0.68, 0.0, 1.0, 1.0),
        pulley_y_norm: float = 0.20,
        carabiner_y_norm: float = 0.78,
    ) -> None:
        self.wing = wing or WingGeometrySpec()
        self.chase_roi_norm = chase_roi_norm
        self.onboard_roi_norm = onboard_roi_norm
        self.pulley_y_norm = pulley_y_norm
        self.carabiner_y_norm = carabiner_y_norm

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
    def _moment_centroid_and_orientation(mask: np.ndarray) -> Tuple[float, float, float, float]:
        """Computes centroid (cx, cy), orientation angle (deg), and area from a binary mask."""
        ys, xs = np.nonzero(mask)
        area = float(len(xs))
        if area < 8.0:
            return 0.0, 0.0, 0.0, 0.0
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        dx = xs.astype(np.float64) - cx
        dy = ys.astype(np.float64) - cy
        mu20 = float(np.mean(dx * dx))
        mu02 = float(np.mean(dy * dy))
        mu11 = float(np.mean(dx * dy))
        angle_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
        return cx, cy, math.degrees(angle_rad), area

    def detect_canopy_and_pilot(
        self, chase_rgb: np.ndarray, horizon_tilt_deg: float, horizon_mid_y: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float, float, float]:
        """
        Detects the ram-air canopy and suspended pilot in the External Chase ROI.
        Returns:
          canopy_xy, pilot_xy, pitch_deg, roll_deg, confidence
        """
        r = chase_rgb[:, :, 0].astype(np.float32)
        g = chase_rgb[:, :, 1].astype(np.float32)
        b = chase_rgb[:, :, 2].astype(np.float32)

        # Canopy segmentation: high warm chromatic saturation (orange/gold/red wing panels)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        chroma = max_rgb - min_rgb
        canopy_mask = (chroma > 45.0) & (r > b + 35.0) & (max_rgb > 110.0)

        cx, cy, canopy_axis_deg, c_area = self._moment_centroid_and_orientation(canopy_mask)

        # Pilot harness segmentation: dark compact body (luma < 32) above the ground terrain
        h, w, _ = chase_rgb.shape
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        y_grid, x_grid = np.ogrid[:h, :w]
        pilot_search_mask = (luma < 32.0) & (chroma < 28.0) & (y_grid < horizon_mid_y - 6.0)
        if c_area >= 20.0:
            pilot_search_mask &= (y_grid > cy + 25.0) & (np.abs(x_grid - cx) < 0.45 * w)

        px, py, _, p_area = self._moment_centroid_and_orientation(pilot_search_mask)

        if c_area < 20.0 or p_area < 8.0:
            return (0.5 * w, 0.25 * h), (0.5 * w, 0.58 * h), 0.0, 0.0, 0.15

        # Suspension vector from pilot (px, py) to canopy (cx, cy) in image coordinates
        dx_forward = cx - px
        dy_up = py - cy
        raw_pendulum_angle_deg = math.degrees(math.atan2(-dx_forward, max(12.0, dy_up)))
        pitch_deg = raw_pendulum_angle_deg - horizon_tilt_deg
        roll_deg = 0.0
        conf = min(1.0, (c_area / 250.0) * (p_area / 35.0))
        return (cx, cy), (px, py), float(pitch_deg), float(roll_deg), float(conf)

    def detect_onboard_controls(
        self, onboard_rgb: np.ndarray
    ) -> Tuple[float, float, float, Tuple[float, float], Tuple[float, float]]:
        """
        Detects Left & Right brake toggle hands and pilot torso lean in the Onboard POV ROI.
        """
        h, w, _ = onboard_rgb.shape
        r = onboard_rgb[:, :, 0].astype(np.float32)
        g = onboard_rgb[:, :, 1].astype(np.float32)
        b = onboard_rgb[:, :, 2].astype(np.float32)

        toggle_mask = (r > 180.0) & (g > 165.0) & (b < 110.0)
        left_half = toggle_mask.copy()
        left_half[:, w // 2 :] = False
        right_half = toggle_mask.copy()
        right_half[:, : w // 2] = False

        lx, ly, _, l_area = self._moment_centroid_and_orientation(left_half)
        rx, ry, _, r_area = self._moment_centroid_and_orientation(right_half)

        y_pulley = self.pulley_y_norm * h
        y_carabiner = self.carabiner_y_norm * h
        travel_px = max(10.0, y_carabiner - y_pulley)

        if l_area < 5.0:
            lx, ly = 0.28 * w, y_pulley
        if r_area < 5.0:
            rx, ry = 0.72 * w, y_pulley

        brake_left = float(np.clip((ly - y_pulley) / travel_px, 0.0, 1.05))
        brake_right = float(np.clip((ry - y_pulley) / travel_px, 0.0, 1.05))

        torso_mask = (b > 170.0) & (g > 150.0) & (r < 95.0)
        tx, ty, _, t_area = self._moment_centroid_and_orientation(torso_mask)
        if t_area >= 6.0:
            weight_shift = float(np.clip((tx - 0.5 * w) / (0.22 * w), -1.0, 1.0))
        else:
            weight_shift = 0.0

        return brake_left, brake_right, weight_shift, (lx, ly), (rx, ry)

    def analyze_frames(
        self,
        frames_rgb: List[np.ndarray],
        fps: float,
        source_video_url: str,
        source_video_title: str,
        maneuver_id: str,
        maneuver_name: str,
        video_timestamp_range: str,
    ) -> ManeuverKinematicsTrace:
        """Runs the full frame-by-frame CV tracker on a list of RGB frames and extracts events & phase lag metrics."""
        observations: List[FrameObservation] = []
        dt = 1.0 / max(1.0, fps)

        for idx, frame in enumerate(frames_rgb):
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
            canopy_xy, pilot_xy, pitch_deg, roll_deg, conf = self.detect_canopy_and_pilot(
                chase_roi, horizon_deg, horizon_y
            )
            brake_l, brake_r, weight_shift, l_hand, r_hand = self.detect_onboard_controls(
                onboard_roi
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
                    big_ears_norm=0.0,
                    horizon_tilt_deg=round(horizon_deg, 3),
                    pitch_deg=round(pitch_deg, 3),
                    pitch_rate_deg_sec=round(pitch_rate, 3),
                    roll_deg=round(roll_deg, 3),
                    roll_rate_deg_sec=round(roll_rate, 3),
                    canopy_forward_offset_m=round(fwd_offset_m, 3),
                    canopy_lateral_offset_m=round(lat_offset_m, 3),
                    canopy_centroid_px=[round(canopy_xy[0] + cx0, 1), round(canopy_xy[1] + cy0, 1)],
                    pilot_centroid_px=[round(pilot_xy[0] + cx0, 1), round(pilot_xy[1] + cy0, 1)],
                    left_hand_px=[round(l_hand[0] + ox0, 1), round(l_hand[1] + oy0, 1)],
                    right_hand_px=[round(r_hand[0] + ox0, 1), round(r_hand[1] + oy0, 1)],
                    confidence=round(conf, 3),
                )
            )

        if len(observations) >= 3:
            for i in range(1, len(observations) - 1):
                rate = (observations[i + 1].pitch_deg - observations[i - 1].pitch_deg) / (2.0 * dt)
                observations[i].pitch_rate_deg_sec = round(rate, 3)

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
        )

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

        # 1. First pull onset
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

        # 2. Aft pitch apex (canopy furthest behind pilot before the main forward dive)
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

        # 3. Release onset near aft_pitch_apex
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

        # 4. Forward dive apex (minimum negative pitch_deg within 2.5s after aft_idx)
        dive_search_end = min(len(frames), aft_idx + int(2.6 * 30))
        dive_idx = aft_idx + int(np.argmin(pitches[aft_idx:dive_search_end]))
        f_dive = frames[dive_idx]

        # 5. Overhead zero-crossing between aft_idx and dive_idx
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

        # 6. Optional Active Catch detection between zero_idx and dive_idx + 8 frames
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
