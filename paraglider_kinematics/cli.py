from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict

from .physics_replay import ParagliderPhysicsComparator
from .schema import ManeuverKinematicsTrace, WingGeometrySpec
from .theo_de_blic_dataset import (
    decode_real_video_frames_rgb,
    extract_trace_from_real_video,
    render_cv_overlay_keyframe_png,
)
from .vision_tracker import DualViewParagliderTracker


SUITE_SPECS = [
    (
        "pitch_pendulum_uncaught",
        "1A. Pitch Pendulum — Single Pull & Uncaught Dive (How to Pitch)",
        "07:42-08:19",
        37.25,
        "side_chase",
    ),
    (
        "pitch_pendulum_active_catch",
        "1B. Pitch Pendulum — Resonance Double-Pump & Active Catch (Catching the Dive)",
        "08:19-09:07",
        47.75,
        "side_chase",
    ),
    (
        "pitch_mistake_late_brake_stall",
        "1C. Pitch Mistake — Re-Braking While Wing Pitches Aft (Risk of Stalling)",
        "10:26-11:13",
        46.75,
        "side_chase",
    ),
    (
        "dynamic_turn_coordinated",
        "2A. Dynamic Turn — Weight-Shift Preload + Inside Brake Carving Dive (How to Dynamic Turn)",
        "04:50-05:37",
        47.00,
        "rear_chase",
    ),
    (
        "dynamic_turn_mistake_no_ws",
        "2B. Dynamic Turn Mistake — Abrupt Brake Pull Without Weight-Shift (Common Mistake)",
        "06:05-06:48",
        43.00,
        "rear_chase",
    ),
    (
        "rolls_weightshift_and_brake",
        "3A. Roll Exercises — Resonant Weight-Shift & Inside Brake Pumping (How to Induce Roll)",
        "00:48-01:28",
        39.50,
        "rear_chase",
    ),
    (
        "rolls_big_ears_pure_ws",
        "3B. Roll Exercises — Big Ears + Pure Harness Weight-Shift Resonance (Roll with Big Ears)",
        "03:05-03:51",
        45.75,
        "rear_chase",
    ),
]


def run_theo_pitch_suite(repo_root: Path, allow_cached_traces: bool = True) -> Dict:
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

    print("=== ParagliderKinematicsLab: Real OpenCV & 500 Hz C# SSOT Residual Suite (All 7 Maneuvers) ===")
    for key, title, ts_range, _dur, cam_mode in SUITE_SPECS:
        mp4_path = videos_dir / f"{key}.mp4"
        json_path = traces_dir / f"{key}.json"
        if mp4_path.is_file():
            trace = extract_trace_from_real_video(
                video_path=mp4_path,
                maneuver_key=key,
                maneuver_name=title,
                timestamp_range=ts_range,
                fps=30.0,
                camera_mode=cam_mode,
            )
            trace.save_json(json_path)
        elif allow_cached_traces and json_path.is_file():
            trace = ManeuverKinematicsTrace.load_json(json_path)
        else:
            raise FileNotFoundError(
                f"Real video clip '{mp4_path}' not found and no pre-extracted OpenCV trace at '{json_path}'."
            )

        curr_sim = ParagliderPhysicsComparator.run_csharp_ssot_500hz(
            trace, wing_profile={"wingId": "school_ena_m"}
        )
        fp_sim = ParagliderPhysicsComparator.run_csharp_ssot_500hz(
            trace, wing_profile={"wingId": "mentor7_light_s"}
        )
        if cam_mode == "rear_chase":
            res_curr = ParagliderPhysicsComparator.evaluate_lateral_residuals(
                trace, curr_sim, "CSharp500Hz_SchoolEnaM"
            )
            res_fp = ParagliderPhysicsComparator.evaluate_lateral_residuals(
                trace, fp_sim, "CSharp500Hz_Mentor7LightS"
            )
        else:
            res_curr = ParagliderPhysicsComparator.evaluate_residuals(
                trace, curr_sim, "CSharp500Hz_SchoolEnaM"
            )
            res_fp = ParagliderPhysicsComparator.evaluate_residuals(
                trace, fp_sim, "CSharp500Hz_Mentor7LightS"
            )

        event_frames = {ev.event_type: ev.frame_index for ev in trace.events}
        rendered_pngs: Dict[str, str] = {}
        candidate_events = [
            "aft_pitch_apex",
            "overhead_zero_crossing",
            "forward_dive_apex",
            "active_catch_onset",
            "weight_shift_onset",
            "asym_brake_onset",
            "roll_swing_peak_1",
            "roll_swing_peak_2",
            "roll_swing_peak_3",
            "roll_swing_peak_4",
        ]
        if mp4_path.is_file():
            for ev_name in candidate_events:
                if ev_name in event_frames:
                    png_out = keyframes_dir / f"{key}_{ev_name}.png"
                    render_cv_overlay_keyframe_png(mp4_path, trace, event_frames[ev_name], png_out)
                    rendered_pngs[ev_name] = str(png_out)

        pm = trace.phase_metrics
        print(f"\n--- {title} [{ts_range}] ({cam_mode}) ---")
        if cam_mode == "rear_chase":
            print(
                f"  [CV LATERAL] WS_Onset={pm.weight_shift_onset_sec}s (LeadBeforeBrake={pm.brake_to_ws_lag_sec:+.2f}s) "
                f"| PeakRoll={pm.peak_roll_deg:+.1f}deg @ {pm.peak_roll_time_sec:.2f}s (Rate={pm.peak_roll_rate_deg_sec:.1f}deg/s)"
            )
            print(
                f"  [CV SWINGS]  SwingPeaks={pm.swing_roll_peaks_deg} | Roll_T_1/2={pm.roll_half_period_sec:.2f}s "
                f"| EntryPitchExtrema: Aft={pm.aft_pitch_apex_deg:+.1f}deg, Dive={pm.dive_pitch_min_deg:+.1f}deg"
            )
            print(
                f"  [RESIDUAL - CurrentParagliderSim]   RollRMSE={res_curr.roll_rmse_deg:5.2f}deg | "
                f"PeakRollErr={res_curr.peak_roll_error_deg:+6.2f}deg | PitchRMSE={res_curr.pitch_rmse_deg:5.2f}deg | "
                f"EntryDiveErr={res_curr.entry_pitch_dive_error_deg:+6.2f}deg | Growth={res_curr.swing_growth_ratio_sim:.2f} (Vid={res_curr.swing_growth_ratio_video:.2f})"
            )
            print(
                f"  [RESIDUAL - FirstPrinciplesLateral] RollRMSE={res_fp.roll_rmse_deg:5.2f}deg | "
                f"PeakRollErr={res_fp.peak_roll_error_deg:+6.2f}deg | PitchRMSE={res_fp.pitch_rmse_deg:5.2f}deg | "
                f"EntryDiveErr={res_fp.entry_pitch_dive_error_deg:+6.2f}deg | Growth={res_fp.swing_growth_ratio_sim:.2f} (Vid={res_fp.swing_growth_ratio_video:.2f})"
            )
        else:
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

        min_x_curr = min(p.attached_flow_x for p in curr_sim)
        min_x_fp = min(p.attached_flow_x for p in fp_sim)
        min_p_curr = min(p.cell_pressure_ratio for p in curr_sim)
        min_p_fp = min(p.cell_pressure_ratio for p in fp_sim)

        full_report["maneuvers"][key] = {
            "title": title,
            "timestamp_range": ts_range,
            "camera_mode": cam_mode,
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
                "video_roll_deg": [f.roll_deg for f in trace.frames],
                "video_roll_rate": [f.roll_rate_deg_sec for f in trace.frames],
                "sim_t": [p.time_sec for p in curr_sim],
                "curr_pitch_deg": [p.pitch_deg for p in curr_sim],
                "curr_pitch_rate": [p.pitch_rate_deg_sec for p in curr_sim],
                "curr_roll_deg": [p.roll_deg for p in curr_sim],
                "curr_roll_rate": [p.roll_rate_deg_sec for p in curr_sim],
                "curr_aoa_deg": [p.local_aoa_deg for p in curr_sim],
                "curr_attached_x": [p.attached_flow_x for p in curr_sim],
                "curr_cell_p": [p.cell_pressure_ratio for p in curr_sim],
                "fp_pitch_deg": [p.pitch_deg for p in fp_sim],
                "fp_pitch_rate": [p.pitch_rate_deg_sec for p in fp_sim],
                "fp_roll_deg": [p.roll_deg for p in fp_sim],
                "fp_roll_rate": [p.roll_rate_deg_sec for p in fp_sim],
                "fp_aoa_deg": [p.local_aoa_deg for p in fp_sim],
                "fp_attached_x": [p.attached_flow_x for p in fp_sim],
                "fp_cell_p": [p.cell_pressure_ratio for p in fp_sim],
            },
        }

    report_path = diag_dir / "pitch_pendulum_residual_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    full_path = diag_dir / "full_tutorial_residual_report.json"
    full_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    from .inspector import build_and_export_inspector
    html_path = build_and_export_inspector(repo_root)
    print(f"\n[OK] Saved full 7-maneuver residual report to {full_path}")
    print(f"[OK] Generated Interactive Flight Inspector with Synchronized Video: {html_path}")
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParagliderKinematicsLab — CV Frame Tracker & 500 Hz Physics Residual Analyzer"
    )
    parser.add_argument(
        "--run-theo-pitch-suite",
        action="store_true",
        help="Run the full CV frame extraction and 500 Hz residual analysis on all 7 maneuvers in Theo de Blic's Tutorial (00:53-20:40).",
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
    parser.add_argument(
        "--export-video",
        type=str,
        default=None,
        help="Export annotated MP4 video with burnt-in CV detection geometry and HUD (pass maneuver key e.g. pitch_pendulum_active_catch or 'all').",
    )
    parser.add_argument(
        "--inspector",
        action="store_true",
        help="Build/refresh the standalone HTML interactive flight telemetry inspector dashboard.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if args.inspector:
        from .inspector import build_and_export_inspector
        html_path = build_and_export_inspector(repo_root)
        print(f"[OK] Interactive Flight Inspector ready at: {html_path}")
        return

    if args.export_video:
        from .video_annotator import export_annotated_mp4
        traces_dir = repo_root / "data" / "traces"
        videos_dir = repo_root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        keys = [s[0] for s in SUITE_SPECS] if args.export_video == "all" else [args.export_video]
        for k in keys:
            t_path = traces_dir / f"{k}.json"
            if not t_path.exists():
                print(f"[ERROR] Trace not found: {t_path}")
                continue
            trace = ManeuverKinematicsTrace.load_json(t_path)
            mp4_in = videos_dir / f"{k}.mp4"
            mp4_out = videos_dir / f"{k}_annotated.mp4"
            print(f"Exporting annotated video for '{k}' ({len(trace.frames)} frames)...")
            export_annotated_mp4(trace, mp4_out, video_path=mp4_in)
            print(f"[OK] Exported: {mp4_out} ({mp4_out.stat().st_size / 1024:.1f} KB)")
        return

    if args.video:
        tracker = DualViewParagliderTracker()
        frames = decode_real_video_frames_rgb(args.video, fps=30.0)
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
        from .inspector import build_and_export_inspector
        html_path = build_and_export_inspector(repo_root, custom_trace=trace, custom_video_path=args.video)
        print(f"[OK] Generated Interactive Flight Inspector for custom video: {html_path}")
        return

    run_theo_pitch_suite(repo_root)


if __name__ == "__main__":
    main()
