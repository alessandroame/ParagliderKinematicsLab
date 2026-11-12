from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paraglider_kinematics.physics_replay import ParagliderPhysicsComparator
from paraglider_kinematics.schema import ManeuverKinematicsTrace
from paraglider_kinematics.theo_de_blic_dataset import (
    decode_real_video_frames_rgb,
    extract_trace_from_real_video,
    render_cv_overlay_keyframe_png,
)
from paraglider_kinematics.video_annotator import export_annotated_mp4
from paraglider_kinematics.vision_tracker import DualViewParagliderTracker


class TestKinematicsPipeline(unittest.TestCase):
    def test_missing_real_mp4_raises_file_not_found_without_synthetic_fallback(self) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        trace_path = root_dir / "data" / "traces" / "pitch_pendulum_active_catch.json"
        trace = ManeuverKinematicsTrace.load_json(trace_path)

        with tempfile.TemporaryDirectory() as tmp:
            missing_mp4 = Path(tmp) / "non_existent_clip.mp4"
            with self.assertRaises(FileNotFoundError):
                decode_real_video_frames_rgb(missing_mp4)

            with self.assertRaises(FileNotFoundError):
                extract_trace_from_real_video(
                    video_path=missing_mp4,
                    maneuver_key="pitch_pendulum_active_catch",
                    maneuver_name="Pitch Pendulum Active Catch",
                    timestamp_range="08:19-09:07",
                )

            with self.assertRaises(FileNotFoundError):
                render_cv_overlay_keyframe_png(
                    video_path=missing_mp4,
                    trace=trace,
                    frame_index=330,
                    out_png_path=Path(tmp) / "overlay.png",
                )

            with self.assertRaises(FileNotFoundError):
                export_annotated_mp4(
                    trace=trace,
                    output_path=Path(tmp) / "annotated.mp4",
                    video_path=missing_mp4,
                )

            real_frame_png = root_dir / "videos" / "frame_08_30.png"
            if real_frame_png.is_file():
                out_png = render_cv_overlay_keyframe_png(
                    video_path=real_frame_png,
                    trace=trace,
                    frame_index=330,
                    out_png_path=Path(tmp) / "real_08_30_overlay.png",
                )
                self.assertTrue(out_png.is_file())
                self.assertGreater(out_png.stat().st_size, 10_000)

    def test_real_opencv_traces_and_csharp_ssot_500hz_longitudinal_and_lateral_replay(self) -> None:
        root_dir = Path(__file__).resolve().parents[1]
        traces_dir = root_dir / "data" / "traces"
        expected_ids = [
            "pitch_pendulum_uncaught",
            "pitch_pendulum_active_catch",
            "pitch_mistake_late_brake_stall",
            "dynamic_turn_coordinated",
            "dynamic_turn_mistake_no_ws",
            "rolls_weightshift_and_brake",
            "rolls_big_ears_pure_ws",
        ]

        for m_id in expected_ids:
            t_file = traces_dir / f"{m_id}.json"
            self.assertTrue(t_file.is_file(), f"Missing real OpenCV trace: {t_file}")
            tr = ManeuverKinematicsTrace.load_json(t_file)
            self.assertEqual(tr.maneuver_id, m_id)
            self.assertGreaterEqual(tr.frames[-1].time_sec, 35.0)
            self.assertGreaterEqual(len(tr.frames), 1050)
            max_abs_rate = max(abs(f.brake_rate_norm_per_sec) for f in tr.frames)
            if m_id != "rolls_big_ears_pure_ws":
                self.assertGreater(max_abs_rate, 0.20)

        pitch_trace = ManeuverKinematicsTrace.load_json(traces_dir / "pitch_pendulum_active_catch.json")
        f_08_22 = min(pitch_trace.frames, key=lambda f: abs(f.time_sec - 3.0))
        f_08_30_peak = min(pitch_trace.frames, key=lambda f: abs(f.time_sec - 12.5))
        self.assertLessEqual(f_08_22.brake_left_norm, 0.08)
        self.assertLessEqual(f_08_22.brake_right_norm, 0.08)
        self.assertGreaterEqual(f_08_30_peak.brake_left_norm, 0.85)
        self.assertGreaterEqual(f_08_30_peak.brake_right_norm, 0.85)
        self.assertGreater(pitch_trace.phase_metrics.aft_pitch_apex_deg, 20.0)
        self.assertLess(pitch_trace.phase_metrics.dive_pitch_min_deg, -15.0)

        sim_low_damp = ParagliderPhysicsComparator.run_csharp_ssot_500hz(
            pitch_trace, wing_profile={"wingId": "mentor7_light_s", "PitchAeroDampingScale": 0.55}
        )
        sim_high_damp = ParagliderPhysicsComparator.run_csharp_ssot_500hz(
            pitch_trace, wing_profile={"wingId": "mentor7_light_s", "PitchAeroDampingScale": 1.85}
        )
        self.assertEqual(len(sim_low_damp), len(pitch_trace.frames))
        self.assertEqual(len(sim_high_damp), len(pitch_trace.frames))

        peak_pitch_low = max(p.pitch_deg for p in sim_low_damp)
        peak_pitch_high = max(p.pitch_deg for p in sim_high_damp)
        self.assertGreater(peak_pitch_low, peak_pitch_high + 1.0)

        res_pitch = ParagliderPhysicsComparator.evaluate_residuals(
            pitch_trace, sim_low_damp, "CSharp500Hz_LowPitchDamp"
        )
        self.assertTrue(math.isfinite(res_pitch.pitch_rmse_deg))
        self.assertGreater(res_pitch.pitch_rmse_deg, 0.0)

        roll_trace = ManeuverKinematicsTrace.load_json(traces_dir / "rolls_weightshift_and_brake.json")
        sim_roll = ParagliderPhysicsComparator.run_csharp_ssot_500hz(
            roll_trace, wing_profile={"wingId": "mentor7_light_s"}
        )
        self.assertEqual(len(sim_roll), len(roll_trace.frames))
        res_roll = ParagliderPhysicsComparator.evaluate_lateral_residuals(
            roll_trace, sim_roll, "CSharp500Hz_Mentor7LightS"
        )
        self.assertTrue(math.isfinite(res_roll.roll_rmse_deg))
        self.assertGreater(max(abs(p.roll_deg) for p in sim_roll), 5.0)

    def test_real_360_riser_cable_and_skeleton_extraction_on_08_22_and_08_30(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV (cv2) not installed in current Python interpreter.")
        root_dir = Path(__file__).resolve().parents[1]
        f_08_22_path = root_dir / "videos" / "frame_08_22.png"
        f_08_30_path = root_dir / "videos" / "frame_08_30.png"
        if not (f_08_22_path.exists() and f_08_30_path.exists()):
            self.skipTest("Control frames frame_08_22.png and frame_08_30.png not present locally.")

        f22 = cv2.imread(str(f_08_22_path))
        f30 = cv2.imread(str(f_08_30_path))
        v22 = cv2.cvtColor(f22, cv2.COLOR_BGR2HSV)[:, :, 2]
        v30 = cv2.cvtColor(f30, cv2.COLOR_BGR2HSV)[:, :, 2]
        static_dark = np.maximum(v22, v30) < 55

        feat_22 = DualViewParagliderTracker.extract_real_360_frame_features(f22, static_dark)
        feat_30 = DualViewParagliderTracker.extract_real_360_frame_features(f30, static_dark)

        # At 08:22 (trim hands-up): hands at pulleys (y <= 105), brake <= 6%
        self.assertLessEqual(feat_22["left_hand"][1], 105)
        self.assertLessEqual(feat_22["right_hand"][1], 105)
        self.assertLessEqual(feat_22["brake_left_pct"], 6)
        self.assertLessEqual(feat_22["brake_right_pct"], 6)

        # At 08:30 (deep brake pull): gloved hands low along outer riser corridor, brake >= 85%
        self.assertLessEqual(feat_30["left_hand"][0], 192)
        self.assertGreaterEqual(feat_30["left_hand"][1], 158)
        self.assertGreaterEqual(feat_30["right_hand"][0], 272)
        self.assertGreaterEqual(feat_30["right_hand"][1], 158)
        self.assertGreaterEqual(feat_30["brake_left_pct"], 85)
        self.assertGreaterEqual(feat_30["brake_right_pct"], 85)

        # Verify 2-link analytical inverse kinematics preserves exact bone lengths L1, L2
        el_l = DualViewParagliderTracker.solve_inverse_kinematics_2link(
            (216.0, 136.0), (200.0, 93.0), 34.0, 39.0, is_left=True
        )
        self.assertAlmostEqual(math.hypot(el_l[0] - 216.0, el_l[1] - 136.0), 34.0, delta=0.05)
        self.assertAlmostEqual(math.hypot(200.0 - el_l[0], 93.0 - el_l[1]), 39.0, delta=0.05)

        # Verify tubular matched filter cancels wide uniform clouds (= 0.0) and highlights 10px sleeve (> 0.8)
        cloud_map = np.ones((270, 480), dtype=np.float32)
        self.assertAlmostEqual(
            DualViewParagliderTracker._tubular_bone_contrast(cloud_map, (180.0, 130.0), (155.0, 180.0)),
            0.0,
            delta=1e-5,
        )
        sleeve_map = np.zeros((270, 480), dtype=np.float32)
        cv2.line(sleeve_map, (180, 130), (155, 180), 1.0, thickness=8)
        self.assertGreater(
            DualViewParagliderTracker._tubular_bone_contrast(sleeve_map, (180.0, 130.0), (155.0, 180.0)),
            0.85,
        )

        # Verify 2-pass calibration + Viterbi sequence extraction on real control frames
        seq = DualViewParagliderTracker.extract_real_360_sequence_biokinematic([f22] * 4 + [f30] * 6)
        self.assertEqual(len(seq), 10)
        self.assertLessEqual(seq[0]["brake_left_pct"], 6)
        self.assertGreaterEqual(seq[-1]["brake_left_pct"], 85)
        self.assertGreaterEqual(seq[-1]["brake_right_pct"], 85)

        # Verify real MP4 video decoding path encoded from the real 08:22 / 08:30 video frames
        with tempfile.TemporaryDirectory() as tmp:
            real_clip_mp4 = Path(tmp) / "real_08_22_to_08_30.mp4"
            h_f, w_f = f22.shape[:2]
            writer = cv2.VideoWriter(
                str(real_clip_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w_f, h_f)
            )
            try:
                for bgr_frame in [f22] * 5 + [f30] * 5:
                    writer.write(bgr_frame)
            finally:
                writer.release()

            if real_clip_mp4.is_file() and real_clip_mp4.stat().st_size > 0:
                decoded = decode_real_video_frames_rgb(real_clip_mp4, fps=10.0, width=w_f, height=h_f)
                self.assertGreaterEqual(len(decoded), 8)
                self.assertEqual(decoded[0].shape, (h_f, w_f, 3))

    def test_analytical_360_optical_projection_and_three_geometric_invariants(self) -> None:
        opt = DualViewParagliderTracker.solve_360_optical_calibration_from_invariants()

        # Invariant 1: Canopy cone b_proj=8.80m at Z_canopy=7.50m vs upper-frame span -> f_top=138.07px, f_px=165px, k1=-0.2505
        self.assertAlmostEqual(opt["focal_px"], 165.0, delta=0.1)
        self.assertAlmostEqual(opt["focal_top_app_px"], 138.07, delta=0.1)
        self.assertAlmostEqual(opt["focal_35mm_eq"], 12.38, delta=0.05)
        self.assertAlmostEqual(opt["k1_radial"], -0.2505, delta=0.005)
        self.assertAlmostEqual(opt["fov_horizontal_deg"], 142.06, delta=0.5)
        self.assertAlmostEqual(opt["fov_vertical_deg"], 92.24, delta=0.5)
        self.assertAlmostEqual(opt["fov_horizontal_pinhole_deg"], 111.0, delta=0.5)
        self.assertAlmostEqual(opt["fov_vertical_pinhole_deg"], 78.58, delta=0.5)

        # Invariant 2: Rigid harness/riser trapezium (W_chest=0.44m, L_riser=0.48m) exact 3D & 0.00px reprojection
        self.assertAlmostEqual(opt["w_chest_3d_m"], 0.44, delta=1e-4)
        self.assertAlmostEqual(opt["l_riser_left_3d_m"], 0.48, delta=1e-4)
        self.assertAlmostEqual(opt["l_riser_right_3d_m"], 0.48, delta=1e-4)
        self.assertLess(opt["reproj_rmse_px"], 1e-5)
        self.assertLess(opt["metric_residual_m"], 1e-5)
        self.assertAlmostEqual(opt["camera_pitch_deg"], 45.0, delta=1.5)
        self.assertAlmostEqual(opt["persp_r"], 0.89, delta=0.01)
        self.assertAlmostEqual(opt["persp_scale_lr"], 1.44, delta=0.05)

        # Invariant 3: Constant 3D arm bones (0.263m, 0.295m) project via Pi to L1(0)=34->38px, L2(0)=39->48px
        self.assertAlmostEqual(opt["l1_metric_3d_m"], 0.263, delta=1e-3)
        self.assertAlmostEqual(opt["l2_metric_3d_m"], 0.295, delta=1e-3)
        self.assertAlmostEqual(opt["l1_0_left"], 34.0, delta=0.25)
        self.assertAlmostEqual(opt["l2_0_left"], 39.0, delta=0.25)
        self.assertAlmostEqual(opt["l1_1_left"], 38.0, delta=0.25)
        self.assertAlmostEqual(opt["l2_1_left"], 48.0, delta=0.25)
        self.assertAlmostEqual(opt["l1_0_right"] / opt["l1_0_left"], opt["persp_r"], delta=0.01)
        self.assertAlmostEqual(opt["l2_1_right"] / opt["l2_1_left"], opt["persp_r"], delta=0.01)

        # Mirror-symmetry invariant when camera is centered (mirror_symmetric=True)
        opt_sym = DualViewParagliderTracker.solve_360_optical_calibration_from_invariants(
            mirror_symmetric=True
        )
        self.assertAlmostEqual(opt_sym["persp_r"], 1.0, delta=1e-6)
        self.assertAlmostEqual(opt_sym["l1_0_left"], opt_sym["l1_0_right"], delta=1e-6)
        self.assertAlmostEqual(opt_sym["l2_1_left"], opt_sym["l2_1_right"], delta=1e-6)

        # Orthonormality of R_cam and exact round-trip of Pi <-> pixel_to_360_optical_ray
        r_mat = DualViewParagliderTracker.build_360_camera_rotation_matrix(45.0, 14.2, -2.2)
        self.assertAlmostEqual(float(np.linalg.det(r_mat)), 1.0, delta=1e-9)
        ray = DualViewParagliderTracker.pixel_to_360_optical_ray(207.0, 95.0, 165.0, -0.2505)
        u_back, v_back = DualViewParagliderTracker.project_360_point_to_pixel(
            1.075 * ray, f_px=165.0, k1=-0.2505
        )
        self.assertAlmostEqual(u_back, 207.0, delta=1e-6)
        self.assertAlmostEqual(v_back, 95.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
