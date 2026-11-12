from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class WingGeometrySpec:
    """Certified geometric & physical reference of the wing filmed in the video."""

    wing_id: str = "nova_mentor7_light_s"
    display_name: str = "NOVA Mentor 7 Light S (EN-B, 70-95 kg)"
    flat_span_m: float = 11.95
    projected_span_m: float = 9.67
    flat_area_m2: float = 24.4
    projected_area_m2: float = 20.65
    line_length_m: float = 7.10
    max_brake_travel_m: float = 0.68
    trim_speed_mps: float = 9.9
    total_takeoff_mass_kg: float = 88.5


@dataclass
class ShotCalibrationProfile:
    """Per-shot appearance, camera-mount, and perspective calibration profile learned from N sampled frames."""

    shot_id: int = 0
    start_frame: int = 0
    end_frame: int = 0
    shot_mode: str = "DUAL_SPLIT"
    anchor_joint: str = "l_knee"
    anchor_px: List[float] = field(default_factory=lambda: [0.0, 0.0])
    persp_scale_lr: float = 1.0
    pulley_y_left_px: float = 0.0
    pulley_y_right_px: float = 0.0
    travel_left_px: float = 100.0
    travel_right_px: float = 100.0
    neutral_torso_lean_deg: float = 0.0
    nominal_span_px: float = 176.0
    nominal_line_len_px: float = 183.6
    camera_elevation_deg: float = 0.0
    focal_px: float = 165.0
    focal_top_app_px: float = 138.07
    focal_35mm_eq: float = 12.38
    k1_radial: float = -0.2505
    fov_horizontal_deg: float = 142.06
    fov_vertical_deg: float = 92.24
    projection_model: str = "DIVISION_FISHEYE"
    camera_pitch_deg: float = 45.0
    camera_yaw_deg: float = 14.2
    camera_roll_deg: float = -2.2
    persp_r: float = 0.89
    canopy_rgb_center: List[float] = field(default_factory=lambda: [232.0, 95.0, 35.0])
    jacket_rgb_center: List[float] = field(default_factory=lambda: [35.0, 210.0, 250.0])
    hip_rgb_center: List[float] = field(default_factory=lambda: [240.0, 50.0, 210.0])
    glove_rgb_center: List[float] = field(default_factory=lambda: [245.0, 200.0, 45.0])
    elbow_rgb_center: List[float] = field(default_factory=lambda: [65.0, 220.0, 95.0])
    head_rgb_center: List[float] = field(default_factory=lambda: [240.0, 244.0, 250.0])
    static_mask_ratio: float = 0.0


@dataclass
class FrameObservation:
    """Per-frame kinematic state extracted by Computer Vision from Dual-ROI video."""

    frame_index: int
    time_sec: float
    # Pilot control inputs extracted from Onboard POV ROI
    brake_left_norm: float
    brake_right_norm: float
    brake_rate_norm_per_sec: float
    weight_shift_norm: float
    big_ears_norm: float
    # Canopy & pendulum attitude extracted from External Chase ROI (referenced to horizon)
    horizon_tilt_deg: float
    pitch_deg: float
    pitch_rate_deg_sec: float
    roll_deg: float
    roll_rate_deg_sec: float
    yaw_rate_deg_sec: float = 0.0
    heading_deg: float = 0.0
    canopy_forward_offset_m: float = 0.0
    canopy_lateral_offset_m: float = 0.0
    # Pixel-space bounding & confidence diagnostics
    canopy_centroid_px: List[float] = field(default_factory=lambda: [0.0, 0.0])
    pilot_centroid_px: List[float] = field(default_factory=lambda: [0.0, 0.0])
    left_hand_px: List[float] = field(default_factory=lambda: [0.0, 0.0])
    right_hand_px: List[float] = field(default_factory=lambda: [0.0, 0.0])
    canopy_span_px: float = 0.0
    confidence: float = 1.0
    # One-time extracted full pilot skeleton & biomechanical angles (PiP ignored/masked)
    pilot_skeleton_px: Dict[str, List[float]] = field(default_factory=dict)
    pose_mode: str = "FULL_SKELETON"
    shoulder_roll_deg: float = 0.0
    torso_lean_deg: float = 0.0
    pip_rect_px: Optional[List[int]] = None
    # Per-shot auto-calibration metadata
    shot_id: int = 0
    shot_mode: str = "DUAL_SPLIT"
    persp_scale_lr: float = 1.0
    anchor_joint: str = "none"
    # Optional YouTube 480x270 real OpenCV anatomical & canopy bounding coordinates
    yt_canopy_box: Optional[List[float]] = None
    yt_left_hand: Optional[List[float]] = None
    yt_right_hand: Optional[List[float]] = None
    yt_pilot_box: Optional[List[float]] = None
    yt_pulley_left: Optional[List[float]] = None
    yt_pulley_right: Optional[List[float]] = None
    yt_carabiner_left: Optional[List[float]] = None
    yt_carabiner_right: Optional[List[float]] = None
    yt_elbow_left: Optional[List[float]] = None
    yt_elbow_right: Optional[List[float]] = None
    yt_shoulder_left: Optional[List[float]] = None
    yt_shoulder_right: Optional[List[float]] = None
    yt_head: Optional[List[float]] = None
    yt_cam_knee: Optional[List[float]] = None


@dataclass
class CommandTimingEvent:
    """Detected pilot command transition or canopy kinematic extremum on the timeline."""

    event_type: str
    time_sec: float
    frame_index: int
    brake_sym_norm: float
    pitch_deg: float
    pitch_rate_deg_sec: float
    description: str
    roll_deg: float = 0.0
    roll_rate_deg_sec: float = 0.0
    weight_shift_norm: float = 0.0
    brake_asym_norm: float = 0.0


@dataclass
class PhaseLagMetrics:
    """Objective timing, phase delay, and amplitude ratios extracted from a maneuver."""

    pull_onset_sec: float
    pull_duration_sec: float
    peak_brake_norm: float
    aft_pitch_apex_deg: float
    aft_pitch_apex_time_sec: float
    pull_to_aft_apex_lag_sec: float
    release_onset_sec: float
    zero_pitch_crossing_sec: float
    release_to_overhead_lag_sec: float
    dive_pitch_min_deg: float
    dive_pitch_min_time_sec: float
    max_forward_surge_m: float
    min_pitch_rate_deg_sec: float
    pendulum_half_period_sec: float
    catch_onset_sec: Optional[float] = None
    catch_peak_brake_norm: Optional[float] = None
    catch_lead_before_dive_apex_sec: Optional[float] = None
    # Lateral-directional metrics for Dynamic Turn & Roll exercises
    weight_shift_onset_sec: Optional[float] = None
    brake_to_ws_lag_sec: Optional[float] = None
    peak_roll_deg: Optional[float] = None
    peak_roll_time_sec: Optional[float] = None
    peak_roll_rate_deg_sec: Optional[float] = None
    roll_half_period_sec: Optional[float] = None
    swing_roll_peaks_deg: List[float] = field(default_factory=list)
    total_heading_carve_deg: Optional[float] = None


@dataclass
class ManeuverKinematicsTrace:
    """Standardized JSON trace exported by ParagliderKinematicsLab for ParagliderSim."""

    schema_version: str
    source_video_url: str
    source_video_title: str
    maneuver_id: str
    maneuver_name: str
    video_timestamp_range: str
    fps: float
    wing: WingGeometrySpec
    phase_metrics: PhaseLagMetrics
    events: List[CommandTimingEvent]
    frames: List[FrameObservation]
    camera_mode: str = "side_chase"
    shot_profiles: List[ShotCalibrationProfile] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, d: Dict) -> ManeuverKinematicsTrace:
        def _filter_kwargs(dc_type, raw: Dict) -> Dict:
            allowed = {fld.name for fld in fields(dc_type)}
            return {k: v for k, v in raw.items() if k in allowed}

        wing = WingGeometrySpec(**_filter_kwargs(WingGeometrySpec, d.get("wing", {})))
        pm = PhaseLagMetrics(**_filter_kwargs(PhaseLagMetrics, d.get("phase_metrics", {})))
        events = [CommandTimingEvent(**_filter_kwargs(CommandTimingEvent, e)) for e in d.get("events", [])]
        frames = [FrameObservation(**_filter_kwargs(FrameObservation, f)) for f in d.get("frames", [])]
        shot_profiles = [
            ShotCalibrationProfile(**_filter_kwargs(ShotCalibrationProfile, sp))
            for sp in d.get("shot_profiles", [])
        ]
        return cls(
            schema_version=d.get("schema_version", "1.0.0"),
            source_video_url=d.get("source_video_url", ""),
            source_video_title=d.get("source_video_title", ""),
            maneuver_id=d.get("maneuver_id", ""),
            maneuver_name=d.get("maneuver_name", ""),
            video_timestamp_range=d.get("video_timestamp_range", ""),
            fps=float(d.get("fps", 30.0)),
            wing=wing,
            phase_metrics=pm,
            events=events,
            frames=frames,
            camera_mode=d.get("camera_mode", "side_chase"),
            shot_profiles=shot_profiles,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> ManeuverKinematicsTrace:
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(content))
