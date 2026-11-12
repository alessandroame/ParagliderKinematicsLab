from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import ManeuverKinematicsTrace


@dataclass
class SimStepPoint:
    time_sec: float
    brake_sym_norm: float
    pitch_deg: float
    pitch_rate_deg_sec: float
    airspeed_mps: float
    vertical_speed_mps: float
    local_aoa_deg: float
    attached_flow_x: float
    cell_pressure_ratio: float
    forward_surge_m: float
    roll_deg: float = 0.0
    roll_rate_deg_sec: float = 0.0
    yaw_rate_deg_sec: float = 0.0
    heading_deg: float = 0.0
    weight_shift_norm: float = 0.0
    brake_asym_norm: float = 0.0
    big_ears_norm: float = 0.0
    lateral_offset_m: float = 0.0


@dataclass
class ModelResidualDiagnostic:
    model_name: str
    pitch_rmse_deg: float
    pitch_rate_rmse_deg_sec: float
    aft_apex_error_deg: float
    aft_apex_time_error_sec: float
    dive_apex_error_deg: float
    dive_apex_time_error_sec: float
    half_period_error_sec: float
    overhead_crossing_lag_error_sec: float
    energy_retention_ratio_video: float
    energy_retention_ratio_sim: float
    roll_rmse_deg: float = 0.0
    roll_rate_rmse_deg_sec: float = 0.0
    peak_roll_error_deg: float = 0.0
    roll_half_period_error_sec: float = 0.0
    entry_pitch_dive_error_deg: float = 0.0
    swing_growth_ratio_video: float = 0.0
    swing_growth_ratio_sim: float = 0.0


class ParagliderPhysicsComparator:
    """
    Replays a video-extracted pilot input trace u_video(t) = [delta_L(t), delta_R(t), w(t), big_ears(t)]
    at 500 Hz (dt = 0.002s) exclusively via the C# Single Source of Truth engine
    (`ParagliderSim.Core.MultiStripFlightEngine` invoked via `ParagliderSim.Cli --cv-sim-replay`),
    and quantifies physical residuals against the OpenCV video kinematics y_video(t).
    """

    DT = 0.002  # 500 Hz

    @staticmethod
    def resolve_csharp_cli_dll(cli_dll_path: Optional[str | Path] = None) -> Path:
        if cli_dll_path is not None:
            candidate = Path(cli_dll_path).resolve()
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"Explicit ParagliderSim.Cli.dll not found: {candidate}")

        lab_root = Path(__file__).resolve().parents[1]
        sim_root = lab_root.parent / "ParagliderSim"
        candidates = [
            sim_root / "out" / "validate-cli" / "ParagliderSim.Cli.dll",
            sim_root / "src" / "ParagliderSim.Cli" / "bin" / "Debug" / "net9.0" / "ParagliderSim.Cli.dll",
            sim_root / "src" / "ParagliderSim.Cli" / "bin" / "Release" / "net9.0" / "ParagliderSim.Cli.dll",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        raise FileNotFoundError(
            f"ParagliderSim.Cli.dll not found under {sim_root}. Build ParagliderSim.sln before running 500 Hz C# SSOT replay."
        )

    @staticmethod
    def _interpolate_inputs(
        trace: ManeuverKinematicsTrace, t: float
    ) -> Tuple[float, float, float]:
        frames = trace.frames
        if not frames:
            return 0.0, 0.0, 0.0
        if t <= frames[0].time_sec:
            sym = 0.5 * (frames[0].brake_left_norm + frames[0].brake_right_norm)
            return sym, frames[0].brake_rate_norm_per_sec, frames[0].weight_shift_norm
        if t >= frames[-1].time_sec:
            sym = 0.5 * (frames[-1].brake_left_norm + frames[-1].brake_right_norm)
            return sym, 0.0, frames[-1].weight_shift_norm

        fps = trace.fps
        exact_idx = t * fps
        i0 = max(0, min(len(frames) - 2, int(math.floor(exact_idx))))
        i1 = i0 + 1
        f0 = frames[i0]
        f1 = frames[i1]
        span = max(1e-6, f1.time_sec - f0.time_sec)
        u = max(0.0, min(1.0, (t - f0.time_sec) / span))
        b0 = 0.5 * (f0.brake_left_norm + f0.brake_right_norm)
        b1 = 0.5 * (f1.brake_left_norm + f1.brake_right_norm)
        sym = b0 + (b1 - b0) * u
        rate = (b1 - b0) / span
        ws = f0.weight_shift_norm + (f1.weight_shift_norm - f0.weight_shift_norm) * u
        return sym, rate, ws

    @classmethod
    def run_csharp_ssot_500hz(
        cls,
        trace: ManeuverKinematicsTrace,
        wing_profile: Optional[Dict] = None,
        cli_dll_path: Optional[str | Path] = None,
    ) -> List[SimStepPoint]:
        """
        Executes a strict causal open-loop 500 Hz simulation on `ParagliderSim.Core.MultiStripFlightEngine`
        using ONLY the pilot control inputs extracted from `trace.frames` (zero attitude leakage).
        """
        dll_path = cls.resolve_csharp_cli_dll(cli_dll_path)
        sim_root = dll_path.parents[2] if dll_path.parent.name == "validate-cli" else dll_path.parents[4]

        control_frames = [
            {
                "frame_index": int(f.frame_index),
                "time_sec": float(f.time_sec),
                "brake_left_norm": float(f.brake_left_norm),
                "brake_right_norm": float(f.brake_right_norm),
                "brake_rate_norm_per_sec": float(f.brake_rate_norm_per_sec),
                "weight_shift_norm": float(f.weight_shift_norm),
                "big_ears_norm": float(f.big_ears_norm),
            }
            for f in trace.frames
        ]

        payload: Dict = {
            "maneuverId": trace.maneuver_id,
            "takeoffMassKg": float(trace.wing.total_takeoff_mass_kg or 88.5),
            "wingProfile": wing_profile if wing_profile is not None else {"wingId": "mentor7_light_s"},
            "frames": control_frames,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="pgsim_py_replay_", encoding="utf-8", delete=False
        ) as tmp_f:
            json.dump(payload, tmp_f)
            tmp_req_path = Path(tmp_f.name)

        try:
            proc = subprocess.run(
                ["dotnet", str(dll_path), "--cv-sim-replay", str(tmp_req_path)],
                cwd=str(sim_root),
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        finally:
            try:
                tmp_req_path.unlink(missing_ok=True)
            except OSError:
                pass

        parsed = json.loads(proc.stdout)
        samples = parsed.get("samples", [])
        line_len = float(trace.wing.line_length_m or 7.10)

        out: List[SimStepPoint] = []
        for s in samples:
            t_sec = float(s.get("timeSec", 0.0))
            bl = float(s.get("leftBrakeNorm", 0.0))
            br = float(s.get("rightBrakeNorm", 0.0))
            sym_b = 0.5 * (bl + br)
            asym_b = br - bl
            pitch_deg = float(s.get("pitchDeg", 0.0))
            pitch_rate = float(s.get("pitchRateDegSec", 0.0))
            roll_deg = float(s.get("rollDeg", 0.0))
            roll_rate = float(s.get("rollRateDegSec", 0.0))
            yaw_rate = float(s.get("yawRateDegSec", 0.0))
            heading_deg = float(s.get("headingDeg", 0.0))
            tas_mps = float(s.get("tasKmh", 35.6)) / 3.6
            vario_mps = float(s.get("varioMs", -1.25))
            stall_l = float(s.get("leftStallRatio", 0.0))
            stall_r = float(s.get("rightStallRatio", 0.0))
            attached_x = max(0.0, min(1.0, 1.0 - 0.5 * (stall_l + stall_r)))
            cell_p = float(s.get("meanCellPressureRatio", 0.93))
            ws = float(s.get("weightShift", 0.0))
            be = float(s.get("bigEarsNorm", 0.0))
            approx_aoa_deg = 8.5 + 0.16 * pitch_deg + 11.5 * sym_b

            out.append(
                SimStepPoint(
                    time_sec=round(t_sec, 4),
                    brake_sym_norm=round(sym_b, 4),
                    pitch_deg=round(pitch_deg, 3),
                    pitch_rate_deg_sec=round(pitch_rate, 3),
                    airspeed_mps=round(tas_mps, 3),
                    vertical_speed_mps=round(vario_mps, 3),
                    local_aoa_deg=round(approx_aoa_deg, 3),
                    attached_flow_x=round(attached_x, 4),
                    cell_pressure_ratio=round(cell_p, 3),
                    forward_surge_m=round(-line_len * math.sin(math.radians(pitch_deg)), 3),
                    roll_deg=round(roll_deg, 3),
                    roll_rate_deg_sec=round(roll_rate, 3),
                    yaw_rate_deg_sec=round(yaw_rate, 2),
                    heading_deg=round(heading_deg, 2),
                    weight_shift_norm=round(ws, 4),
                    brake_asym_norm=round(asym_b, 4),
                    big_ears_norm=round(be, 3),
                    lateral_offset_m=round(line_len * math.sin(math.radians(roll_deg)), 3),
                )
            )
        return out

    @classmethod
    def evaluate_residuals(
        cls,
        trace: ManeuverKinematicsTrace,
        sim_points: List[SimStepPoint],
        model_name: str,
    ) -> ModelResidualDiagnostic:
        """Computes quantitative phase, period, and amplitude errors between video and C# 500 Hz simulation."""
        if trace.camera_mode == "rear_chase":
            return cls.evaluate_lateral_residuals(trace, sim_points, model_name)

        sim_times = np.array([p.time_sec for p in sim_points])
        sim_pitch = np.array([p.pitch_deg for p in sim_points])
        sim_rate = np.array([p.pitch_rate_deg_sec for p in sim_points])

        vid_times = np.array([f.time_sec for f in trace.frames])
        vid_pitch = np.array([f.pitch_deg for f in trace.frames])
        vid_rate = np.array([f.pitch_rate_deg_sec for f in trace.frames])

        interp_sim_pitch = np.interp(vid_times, sim_times, sim_pitch)
        interp_sim_rate = np.interp(vid_times, sim_times, sim_rate)

        pitch_rmse = float(np.sqrt(np.mean((interp_sim_pitch - vid_pitch) ** 2)))
        rate_rmse = float(np.sqrt(np.mean((interp_sim_rate - vid_rate) ** 2)))

        pm = trace.phase_metrics
        aft_lo = max(0.0, pm.aft_pitch_apex_time_sec - 3.5)
        aft_hi = pm.aft_pitch_apex_time_sec + 1.5
        aft_win_mask = (sim_times >= aft_lo) & (sim_times <= aft_hi)
        if not np.any(aft_win_mask):
            aft_win_mask = np.ones_like(sim_times, dtype=bool)
        aft_sub_pitch = sim_pitch[aft_win_mask]
        aft_sub_time = sim_times[aft_win_mask]
        sim_aft_idx = int(np.argmax(aft_sub_pitch))
        sim_aft_deg = float(aft_sub_pitch[sim_aft_idx])
        sim_aft_time = float(aft_sub_time[sim_aft_idx])

        dive_win_mask = (sim_times >= sim_aft_time) & (sim_times <= sim_aft_time + 3.2)
        if not np.any(dive_win_mask):
            dive_win_mask = sim_times >= sim_aft_time
        dive_sub_pitch = sim_pitch[dive_win_mask]
        dive_sub_time = sim_times[dive_win_mask]
        sim_dive_idx = int(np.argmin(dive_sub_pitch))
        sim_dive_deg = float(dive_sub_pitch[sim_dive_idx])
        sim_dive_time = float(dive_sub_time[sim_dive_idx])

        zero_cross_time = sim_aft_time
        for t_val, p_val in zip(dive_sub_time, dive_sub_pitch):
            if p_val <= 0.0:
                zero_cross_time = float(t_val)
                break

        sim_half_period = max(0.2, sim_dive_time - sim_aft_time)
        sim_rel_to_zero = zero_cross_time - pm.release_onset_sec

        vid_ratio = abs(pm.dive_pitch_min_deg) / max(1.0, abs(pm.aft_pitch_apex_deg))
        sim_ratio = abs(sim_dive_deg) / max(1.0, abs(sim_aft_deg))

        return ModelResidualDiagnostic(
            model_name=model_name,
            pitch_rmse_deg=round(pitch_rmse, 2),
            pitch_rate_rmse_deg_sec=round(rate_rmse, 2),
            aft_apex_error_deg=round(sim_aft_deg - pm.aft_pitch_apex_deg, 2),
            aft_apex_time_error_sec=round(sim_aft_time - pm.aft_pitch_apex_time_sec, 3),
            dive_apex_error_deg=round(sim_dive_deg - pm.dive_pitch_min_deg, 2),
            dive_apex_time_error_sec=round(sim_dive_time - pm.dive_pitch_min_time_sec, 3),
            half_period_error_sec=round(sim_half_period - pm.pendulum_half_period_sec, 3),
            overhead_crossing_lag_error_sec=round(
                sim_rel_to_zero - pm.release_to_overhead_lag_sec, 3
            ),
            energy_retention_ratio_video=round(vid_ratio, 3),
            energy_retention_ratio_sim=round(sim_ratio, 3),
        )

    @classmethod
    def evaluate_lateral_residuals(
        cls,
        trace: ManeuverKinematicsTrace,
        sim_points: List[SimStepPoint],
        model_name: str,
    ) -> ModelResidualDiagnostic:
        """Computes lateral-directional roll, roll-rate, swing resonance, and turn-entry pitch dive residuals."""
        sim_times = np.array([p.time_sec for p in sim_points])
        sim_roll = np.array([p.roll_deg for p in sim_points])
        sim_roll_rate = np.array([p.roll_rate_deg_sec for p in sim_points])
        sim_pitch = np.array([p.pitch_deg for p in sim_points])
        sim_pitch_rate = np.array([p.pitch_rate_deg_sec for p in sim_points])

        vid_times = np.array([f.time_sec for f in trace.frames])
        vid_roll = np.array([f.roll_deg for f in trace.frames])
        vid_roll_rate = np.array([f.roll_rate_deg_sec for f in trace.frames])
        vid_pitch = np.array([f.pitch_deg for f in trace.frames])
        vid_pitch_rate = np.array([f.pitch_rate_deg_sec for f in trace.frames])

        interp_roll = np.interp(vid_times, sim_times, sim_roll)
        interp_roll_rate = np.interp(vid_times, sim_times, sim_roll_rate)
        interp_pitch = np.interp(vid_times, sim_times, sim_pitch)
        interp_pitch_rate = np.interp(vid_times, sim_times, sim_pitch_rate)

        roll_rmse = float(np.sqrt(np.mean((interp_roll - vid_roll) ** 2)))
        roll_rate_rmse = float(np.sqrt(np.mean((interp_roll_rate - vid_roll_rate) ** 2)))
        pitch_rmse = float(np.sqrt(np.mean((interp_pitch - vid_pitch) ** 2)))
        pitch_rate_rmse = float(np.sqrt(np.mean((interp_pitch_rate - vid_pitch_rate) ** 2)))

        pm = trace.phase_metrics
        vid_peak_abs_roll = float(np.max(np.abs(vid_roll)))
        sim_peak_abs_roll = float(np.max(np.abs(sim_roll)))
        peak_roll_err = sim_peak_abs_roll - vid_peak_abs_roll

        swing_peaks: List[float] = []
        swing_times: List[float] = []
        cur_sgn = 0
        best_i = 0
        best_abs = 0.0
        for i, r_val in enumerate(sim_roll):
            if abs(r_val) > 4.0:
                sgn = 1 if r_val > 0 else -1
                if cur_sgn == 0:
                    cur_sgn = sgn
                    best_i = i
                    best_abs = abs(r_val)
                elif sgn == cur_sgn:
                    if abs(r_val) > best_abs:
                        best_abs = abs(r_val)
                        best_i = i
                else:
                    swing_peaks.append(float(sim_roll[best_i]))
                    swing_times.append(float(sim_times[best_i]))
                    cur_sgn = sgn
                    best_i = i
                    best_abs = abs(r_val)
        if cur_sgn != 0 and best_abs > 4.0:
            swing_peaks.append(float(sim_roll[best_i]))
            swing_times.append(float(sim_times[best_i]))

        if len(swing_times) >= 2:
            sim_roll_half_period = float(
                np.mean([swing_times[k + 1] - swing_times[k] for k in range(len(swing_times) - 1)])
            )
        else:
            sim_roll_half_period = 0.96

        vid_swings = pm.swing_roll_peaks_deg
        vid_growth = (
            abs(vid_swings[-1]) / max(1.0, abs(vid_swings[0]))
            if len(vid_swings) >= 2
            else 1.0
        )
        sim_growth = (
            abs(swing_peaks[-1]) / max(1.0, abs(swing_peaks[0]))
            if len(swing_peaks) >= 2
            else 1.0
        )

        sim_min_pitch = float(np.min(sim_pitch))
        sim_max_pitch = float(np.max(sim_pitch))
        dive_err = sim_min_pitch - pm.dive_pitch_min_deg
        aft_err = sim_max_pitch - pm.aft_pitch_apex_deg
        half_period_err = sim_roll_half_period - (pm.roll_half_period_sec or pm.pendulum_half_period_sec)

        return ModelResidualDiagnostic(
            model_name=model_name,
            pitch_rmse_deg=round(pitch_rmse, 2),
            pitch_rate_rmse_deg_sec=round(pitch_rate_rmse, 2),
            aft_apex_error_deg=round(aft_err, 2),
            aft_apex_time_error_sec=0.0,
            dive_apex_error_deg=round(dive_err, 2),
            dive_apex_time_error_sec=0.0,
            half_period_error_sec=round(half_period_err, 3),
            overhead_crossing_lag_error_sec=0.0,
            energy_retention_ratio_video=round(vid_growth, 3),
            energy_retention_ratio_sim=round(sim_growth, 3),
            roll_rmse_deg=round(roll_rmse, 2),
            roll_rate_rmse_deg_sec=round(roll_rate_rmse, 2),
            peak_roll_error_deg=round(peak_roll_err, 2),
            roll_half_period_error_sec=round(half_period_err, 3),
            entry_pitch_dive_error_deg=round(dive_err, 2),
            swing_growth_ratio_video=round(vid_growth, 3),
            swing_growth_ratio_sim=round(sim_growth, 3),
        )
