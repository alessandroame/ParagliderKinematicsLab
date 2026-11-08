from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out
