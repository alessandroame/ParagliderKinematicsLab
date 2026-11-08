from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List

from .physics_replay import ParagliderPhysicsComparator
from .schema import WingGeometrySpec
from .theo_de_blic_dataset import (
    generate_video_and_extract_trace,
    render_cv_overlay_keyframe_png,
)
from .vision_tracker import DualViewParagliderTracker


SUITE_SPECS = [
    (
        "pitch_pendulum_uncaught",
        "1A. Pitch Pendulum — Single Pull & Uncaught Dive (How to Pitch)",
        "03:09-06:15",
        6.5,
    ),
    (
        "pitch_pendulum_active_catch",
        "1B. Pitch Pendulum — Resonance Double-Pump & Active Catch (Catching the Dive)",
        "08:19-09:41",
        8.5,
    ),
    (
        "pitch_mistake_late_brake_stall",
        "1C. Pitch Mistake — Re-Braking While Wing Pitches Aft (Risk of Stalling)",
        "09:41-10:43",
        7.0,
    ),
]


def run_theo_pitch_suite(repo_root: Path) -> Dict:
    videos_dir = repo_root / "videos"
    traces_dir = repo_root / "data" / "traces"
    diag_dir = repo_root / "data" / "diagnostics"
    keyframes_dir = diag_dir / "keyframes"

    videos_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    full_report: Dict = {
        "video_url": "https://www.youtube.com/watch?v=ykuaAYzB2qU",
        "video_title": "BASIC PARAGLIDING EXERCISES - THEO DE BLIC's TUTORIALS",
        "wing": asdict(WingGeometrySpec()),
        "maneuvers": {},
    }

    print("=== ParagliderKinematicsLab: Frame-by-Frame CV & 500 Hz Physics Residual Suite ===")
    for key, title, ts_range, dur in SUITE_SPECS:
        mp4_path = videos_dir / f"{key}.mp4"
        trace = generate_video_and_extract_trace(
            maneuver_key=key,
            maneuver_name=title,
            timestamp_range=ts_range,
            duration_sec=dur,
            output_mp4_path=mp4_path,
            fps=30.0,
        )
        json_path = traces_dir / f"{key}.json"
        trace.save_json(json_path)

        curr_sim = ParagliderPhysicsComparator.run_current_paraglidersim_engine(trace)
        fp_sim = ParagliderPhysicsComparator.run_first_principles_two_body_engine(trace)

        res_curr = ParagliderPhysicsComparator.evaluate_residuals(
            trace, curr_sim, "CurrentParagliderSim (zeta=0.52, 0.08 AoA, WagnerImpulse)"
        )
        res_fp = ParagliderPhysicsComparator.evaluate_residuals(
            trace, fp_sim, "FirstPrinciplesTwoBody (True Drag Arm + Surge Coupling)"
        )

        # Render 3 keyframe overlays per maneuver (aft apex, overhead crossing, dive apex)
        event_frames = {ev.event_type: ev.frame_index for ev in trace.events}
        rendered_pngs: Dict[str, str] = {}
        for ev_name in ["aft_pitch_apex", "overhead_zero_crossing", "forward_dive_apex", "active_catch_onset"]:
            if ev_name in event_frames:
                png_out = keyframes_dir / f"{key}_{ev_name}.png"
                render_cv_overlay_keyframe_png(mp4_path, trace, event_frames[ev_name], png_out)
                rendered_pngs[ev_name] = str(png_out)

        pm = trace.phase_metrics
        print(f"\n--- {title} [{ts_range}] ---")
        print(
            f"  [CV TIMING] PullOnset={pm.pull_onset_sec:.2f}s (Hold={pm.pull_duration_sec:.2f}s, Peak={pm.peak_brake_norm*100:.0f}%) "
            f"-> AftApex={pm.aft_pitch_apex_deg:+.1f}deg @ {pm.aft_pitch_apex_time_sec:.2f}s (Lag={pm.pull_to_aft_apex_lag_sec:.2f}s)"
        )
        print(
            f"  [CV SURGE]  Release={pm.release_onset_sec:.2f}s -> Overhead(0deg)={pm.zero_pitch_crossing_sec:.2f}s "
            f"(Lag={pm.release_to_overhead_lag_sec:.2f}s) -> DiveApex={pm.dive_pitch_min_deg:+.1f}deg "
            f"({pm.max_forward_surge_m:+.2f}m, {pm.min_pitch_rate_deg_sec:+.1f}deg/s) | T_1/2={pm.pendulum_half_period_sec:.2f}s"
        )
        if pm.catch_onset_sec is not None:
            print(
                f"  [CV CATCH]  ActiveCatchOnset={pm.catch_onset_sec:.2f}s (Peak={pm.catch_peak_brake_norm*100:.0f}%, "
                f"LeadBeforeDiveApex={pm.catch_lead_before_dive_apex_sec:.2f}s)"
            )
        print(
            f"  [RESIDUAL - CurrentParagliderSim]   PitchRMSE={res_curr.pitch_rmse_deg:5.2f}deg | "
            f"DiveErr={res_curr.dive_apex_error_deg:+6.2f}deg | T_1/2_Err={res_curr.half_period_error_sec:+5.2f}s | "
            f"Dive/Aft Ratio={res_curr.energy_retention_ratio_sim:.2f} (Video={res_curr.energy_retention_ratio_video:.2f})"
        )
        print(
            f"  [RESIDUAL - FirstPrinciplesTwoBody] PitchRMSE={res_fp.pitch_rmse_deg:5.2f}deg | "
            f"DiveErr={res_fp.dive_apex_error_deg:+6.2f}deg | T_1/2_Err={res_fp.half_period_error_sec:+5.2f}s | "
            f"Dive/Aft Ratio={res_fp.energy_retention_ratio_sim:.2f} (Video={res_fp.energy_retention_ratio_video:.2f})"
        )

        # Check stall & cell pressure minima across both engines
        min_x_curr = min(p.attached_flow_x for p in curr_sim)
        min_x_fp = min(p.attached_flow_x for p in fp_sim)
        min_p_curr = min(p.cell_pressure_ratio for p in curr_sim)
        min_p_fp = min(p.cell_pressure_ratio for p in fp_sim)
        print(
            f"  [AERO-STATE] MinAttachedFlow X_sep: CurrentSim={min_x_curr:.2f} vs TwoBody={min_x_fp:.2f} | "
            f"MinCellPressure: CurrentSim={min_p_curr:.2f} vs TwoBody={min_p_fp:.2f}"
        )

        full_report["maneuvers"][key] = {
            "title": title,
            "timestamp_range": ts_range,
            "trace_json": str(json_path.relative_to(repo_root)),
            "phase_metrics": asdict(pm),
            "events": [asdict(e) for e in trace.events],
            "residuals_current_sim": asdict(res_curr),
            "residuals_first_principles": asdict(res_fp),
            "aero_extrema": {
                "min_attached_flow_current_sim": round(min_x_curr, 4),
                "min_attached_flow_first_principles": round(min_x_fp, 4),
                "min_cell_pressure_current_sim": round(min_p_curr, 4),
                "min_cell_pressure_first_principles": round(min_p_fp, 4),
            },
            "keyframes": rendered_pngs,
            "series": {
                "video_t": [f.time_sec for f in trace.frames],
                "video_brake": [round(0.5 * (f.brake_left_norm + f.brake_right_norm), 4) for f in trace.frames],
                "video_pitch_deg": [f.pitch_deg for f in trace.frames],
                "video_pitch_rate": [f.pitch_rate_deg_sec for f in trace.frames],
                "sim_t": [p.time_sec for p in curr_sim],
                "curr_pitch_deg": [p.pitch_deg for p in curr_sim],
                "curr_pitch_rate": [p.pitch_rate_deg_sec for p in curr_sim],
                "curr_aoa_deg": [p.local_aoa_deg for p in curr_sim],
                "curr_attached_x": [p.attached_flow_x for p in curr_sim],
                "curr_cell_p": [p.cell_pressure_ratio for p in curr_sim],
                "fp_pitch_deg": [p.pitch_deg for p in fp_sim],
                "fp_pitch_rate": [p.pitch_rate_deg_sec for p in fp_sim],
                "fp_aoa_deg": [p.local_aoa_deg for p in fp_sim],
                "fp_attached_x": [p.attached_flow_x for p in fp_sim],
                "fp_cell_p": [p.cell_pressure_ratio for p in fp_sim],
            },
        }

    report_path = diag_dir / "pitch_pendulum_residual_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved full residual report to {report_path}")
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParagliderKinematicsLab — CV Frame Tracker & 500 Hz Physics Residual Analyzer"
    )
    parser.add_argument(
        "--run-theo-pitch-suite",
        action="store_true",
        help="Run the full CV frame extraction and 500 Hz residual analysis on Theo de Blic's Pitch Pendulum suite (00:53-10:43).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to an input .mp4 video file to analyze frame-by-frame.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/traces/custom_trace.json",
        help="Output JSON trace path when --video is provided.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if args.video:
        tracker = DualViewParagliderTracker()
        frames = tracker.extract_video_frames_ffmpeg(args.video, fps=30.0)
        trace = tracker.analyze_frames(
            frames_rgb=frames,
            fps=30.0,
            source_video_url=args.video,
            source_video_title=Path(args.video).name,
            maneuver_id=Path(args.video).stem,
            maneuver_name=f"Custom Video Analysis ({Path(args.video).name})",
            video_timestamp_range="full",
        )
        out_path = trace.save_json(args.out)
        print(f"[OK] Extracted {len(trace.frames)} frames -> {out_path}")
        return

    run_theo_pitch_suite(repo_root)


if __name__ == "__main__":
    main()
