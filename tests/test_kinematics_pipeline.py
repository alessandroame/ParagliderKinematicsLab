from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from paraglider_kinematics.physics_replay import ParagliderPhysicsComparator
from paraglider_kinematics.theo_de_blic_dataset import (
     generate_video_and_extract_trace,
    render_dual_view_frame_rgb,
)
from paraglider_kinematics.vision_tracker import DualViewParagliderTracker


class TestKinematicsPipeline(unittest.TestCase):
    def test_cv_tracker_extracts_accurate_timings_and_angles_from_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mp4_path = Path(tmp) / "test_pitch.mp4"
            trace = generate_video_and_extract_trace(
                maneuver_key="pitch_pendulum_active_catch",
                maneuver_name="Pitch Pendulum with Active Catch",
                timestamp_range="08:19-09:41",
                duration_sec=8.5,
                output_mp4_path=mp4_path,
                fps=30.0,
                camera_mode="side_chase",
            )
            self.assertTrue(mp4_path.exists())
            self.assertGreater(len(trace.frames), 240)

            pm = trace.phase_metrics
            self.assertAlmostEqual(pm.pull_onset_sec, 0.55, delta=0.20)
            self.assertGreater(pm.aft_pitch_apex_deg, 14.0)
            self.assertLess(pm.dive_pitch_min_deg, -15.0)
            self.assertIsNotNone(pm.catch_onset_sec)
            self.assertGreater(pm.catch_peak_brake_norm or 0.0, 0.60)

            curr_sim = ParagliderPhysicsComparator.run_current_paraglidersim_engine(trace)
            fp_sim = ParagliderPhysicsComparator.run_first_principles_two_body_engine(trace)
            res_curr = ParagliderPhysicsComparator.evaluate_residuals(trace, curr_sim, "Current")
            res_fp = ParagliderPhysicsComparator.evaluate_residuals(trace, fp_sim, "FirstPrinciples")

            self.assertLess(res_fp.pitch_rmse_deg, 1.5)
            self.assertGreater(res_curr.pitch_rmse_deg, res_fp.pitch_rmse_deg * 3.0)

    def test_lateral_dynamic_turn_and_big_ears_rolls_cv_and_physics_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 2A: Coordinated Dynamic Turn (Weight-shift preload -> carving dive)
            mp4_turn = Path(tmp) / "test_dyn_turn.mp4"
            trace_turn = generate_video_and_extract_trace(
                maneuver_key="dynamic_turn_coordinated",
                maneuver_name="Coordinated Dynamic Turn",
                timestamp_range="15:04-16:34",
                duration_sec=5.5,
                output_mp4_path=mp4_turn,
                fps=30.0,
                camera_mode="rear_chase",
            )
            pm_turn = trace_turn.phase_metrics
            self.assertIsNotNone(pm_turn.weight_shift_onset_sec)
            self.assertGreater(pm_turn.brake_to_ws_lag_sec or 0.0, 0.20)
            self.assertLess(pm_turn.dive_pitch_min_deg, -10.0)
            self.assertGreater(pm_turn.peak_roll_deg or 0.0, 30.0)

            curr_turn = ParagliderPhysicsComparator.run_current_paraglidersim_lateral_engine(trace_turn)
            fp_turn = ParagliderPhysicsComparator.run_first_principles_lateral_engine(trace_turn)
            res_curr_turn = ParagliderPhysicsComparator.evaluate_lateral_residuals(
                trace_turn, curr_turn, "CurrentLateral"
            )
            res_fp_turn = ParagliderPhysicsComparator.evaluate_lateral_residuals(
                trace_turn, fp_turn, "FirstPrinciplesLateral"
            )
            self.assertLess(res_fp_turn.roll_rmse_deg, 1.8)
            self.assertGreater(res_curr_turn.roll_rmse_deg, res_fp_turn.roll_rmse_deg * 3.0)
            self.assertGreater(res_curr_turn.entry_pitch_dive_error_deg, 10.0)

            # 3B: Big Ears + Pure Weight-Shift Roll Resonance
            mp4_ears = Path(tmp) / "test_big_ears_rolls.mp4"
            trace_ears = generate_video_and_extract_trace(
                maneuver_key="rolls_big_ears_pure_ws",
                maneuver_name="Big Ears Pure Weight-Shift Rolls",
                timestamp_range="19:15-20:40",
                duration_sec=7.5,
                output_mp4_path=mp4_ears,
                fps=30.0,
                camera_mode="rear_chase",
            )
            mean_ears = sum(f.big_ears_norm for f in trace_ears.frames) / len(trace_ears.frames)
            max_brake = max(max(f.brake_left_norm, f.brake_right_norm) for f in trace_ears.frames)
            self.assertGreater(mean_ears, 0.75)
            self.assertLess(max_brake, 0.05)

            curr_ears = ParagliderPhysicsComparator.run_current_paraglidersim_lateral_engine(trace_ears)
            fp_ears = ParagliderPhysicsComparator.run_first_principles_lateral_engine(trace_ears)
            res_curr_ears = ParagliderPhysicsComparator.evaluate_lateral_residuals(
                trace_ears, curr_ears, "CurrentBigEars"
            )
            res_fp_ears = ParagliderPhysicsComparator.evaluate_lateral_residuals(
                trace_ears, fp_ears, "FirstPrinciplesBigEars"
            )
            self.assertLess(res_fp_ears.roll_rmse_deg, 1.8)
            self.assertGreater(res_curr_ears.roll_rmse_deg, res_fp_ears.roll_rmse_deg * 3.0)

            sample_obs = trace_turn.frames[40]
            self.assertEqual(sample_obs.pose_mode, "FULL_SKELETON")
            self.assertIsNone(sample_obs.pip_rect_px)
            expected_joints = {
                "nose", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
                "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_palm", "r_palm",
                "l_hip", "r_hip", "l_knee", "r_knee",
            }
            self.assertTrue(expected_joints.issubset(set(sample_obs.pilot_skeleton_px.keys())))
            self.assertGreater(abs(sample_obs.torso_lean_deg), 2.0)

    def test_multi_shot_outfit_color_adaptation_and_static_logo_masking(self) -> None:
        """
        Verifies that SequenceAutoCalibrator segments concatenated shots with completely different
        pilot clothing/glove/canopy colors, masks out a persistent red YouTube logo in the sky,
        and accurately tracks pilot controls and canopy pitch across both shots.
        """
        frames = []
        static_logo = (420, 70, 485, 118)
        alt_palette = {
            "canopy": (40, 215, 75),
            "canopy_outline": (95, 245, 120),
            "jacket": (235, 45, 55),
            "hip": (235, 185, 35),
            "glove": (40, 220, 235),
            "elbow": (175, 55, 235),
            "head": (250, 125, 40),
        }
        # Shot 0 (frames 0..17): default cyan jacket, yellow gloves, orange canopy
        for i in range(18):
            t = i / 30.0
            sym_b = 0.65 if 5 <= i <= 12 else 0.0
            pitch_d = 18.0 if 7 <= i <= 14 else -14.0 if i > 14 else 0.0
            frames.append(
                render_dual_view_frame_rgb(
                    960,
                    540,
                    t,
                    sym_b,
                    sym_b,
                    0.45 if 5 <= i <= 12 else 0.0,
                    pitch_d,
                    0.0,
                    static_logo_rect=static_logo,
                )
            )
        # Shot 1 (frames 18..35): crimson jacket, cyan gloves, emerald canopy
        for i in range(18, 36):
            t = i / 30.0
            sym_b = 0.72 if 22 <= i <= 29 else 0.0
            pitch_d = 16.0 if 23 <= i <= 30 else -16.0 if i > 30 else 0.0
            frames.append(
                render_dual_view_frame_rgb(
                    960,
                    540,
                    t,
                    sym_b,
                    sym_b,
                    -0.55 if 22 <= i <= 29 else 0.0,
                    pitch_d,
                    0.0,
                    custom_palette=alt_palette,
                    static_logo_rect=static_logo,
                )
            )

        tracker = DualViewParagliderTracker()
        trace = tracker.analyze_frames(
            frames_rgb=frames,
            fps=30.0,
            source_video_url="https://www.youtube.com/watch?v=ykuaAYzB2qU",
            source_video_title="Multi-Shot Outfit Calibration Test",
            maneuver_id="multi_shot_test",
            maneuver_name="Multi-Shot Outfit Calibration",
            video_timestamp_range="00:00-00:01",
            camera_mode="side_chase",
        )

        self.assertEqual(len(trace.shot_profiles), 2)
        sp0, sp1 = trace.shot_profiles[0], trace.shot_profiles[1]
        self.assertGreater(sp0.static_mask_ratio, 0.001)
        self.assertGreater(sp1.static_mask_ratio, 0.001)

        # Shot 0 jacket is cyan (B > R), Shot 1 jacket is crimson red (R > B)
        self.assertGreater(sp0.jacket_rgb_center[2], sp0.jacket_rgb_center[0] + 80.0)
        self.assertGreater(sp1.jacket_rgb_center[0], sp1.jacket_rgb_center[2] + 80.0)

        # Shot 0 canopy is orange (R > G), Shot 1 canopy is emerald green (G > R)
        self.assertGreater(sp0.canopy_rgb_center[0], sp0.canopy_rgb_center[1] + 80.0)
        self.assertGreater(sp1.canopy_rgb_center[1], sp1.canopy_rgb_center[0] + 80.0)

        # Verify controls & skeleton are accurately extracted in Shot 1 despite completely swapped colors
        f_shot1_pull = trace.frames[25]
        self.assertEqual(f_shot1_pull.shot_id, 1)
        self.assertEqual(f_shot1_pull.pose_mode, "FULL_SKELETON")
        self.assertAlmostEqual(f_shot1_pull.brake_left_norm, 0.72, delta=0.08)
        self.assertAlmostEqual(f_shot1_pull.brake_right_norm, 0.72, delta=0.08)
        self.assertLess(f_shot1_pull.weight_shift_norm, -0.35)
        self.assertAlmostEqual(f_shot1_pull.pitch_deg, 16.0, delta=2.5)

    def test_left_knee_360_camera_perspective_autocalibration(self) -> None:
        """
        Verifies that SequenceAutoCalibrator detects a 360 camera mounted on the left knee
        (`anchor_joint == 'l_knee'`, `persp_scale_lr ~ 1.44`) and compensates the asymmetric
        left/right pixel brake travel and shoulder foreshortening.
        """
        frames = []
        for i in range(20):
            t = i / 30.0
            sym_b = 0.60 if 5 <= i <= 14 else 0.0
            ws = 0.50 if 6 <= i <= 12 else 0.0
            pitch_d = 15.0 if 6 <= i <= 12 else -12.0 if i > 12 else 0.0
            frames.append(
                render_dual_view_frame_rgb(
                    960,
                    540,
                    t,
                    sym_b,
                    sym_b,
                    ws,
                    pitch_d,
                    0.0,
                    persp_scale_lr=1.44,
                    anchor_left_knee=True,
                )
            )

        tracker = DualViewParagliderTracker()
        trace = tracker.analyze_frames(
            frames_rgb=frames,
            fps=30.0,
            source_video_url="https://www.youtube.com/watch?v=ykuaAYzB2qU",
            source_video_title="360 Left-Knee Camera Calibration Test",
            maneuver_id="knee_360_test",
            maneuver_name="360 Left-Knee Camera Calibration",
            video_timestamp_range="00:00-00:01",
            camera_mode="side_chase",
        )

        self.assertEqual(len(trace.shot_profiles), 1)
        sp = trace.shot_profiles[0]
        self.assertEqual(sp.anchor_joint, "l_knee")
        self.assertEqual(sp.shot_mode, "ONBOARD_360_KNEE")
        self.assertAlmostEqual(sp.persp_scale_lr, 1.44, delta=0.10)
        self.assertGreater(sp.travel_left_px, sp.travel_right_px * 1.30)

        # At frame 9 (sym_b = 0.60), despite 1.44x pixel travel difference, normalized brakes match
        f_pull = trace.frames[9]
        self.assertAlmostEqual(f_pull.brake_left_norm, 0.60, delta=0.06)
        self.assertAlmostEqual(f_pull.brake_right_norm, 0.60, delta=0.06)
        self.assertLess(abs(f_pull.brake_left_norm - f_pull.brake_right_norm), 0.04)

        # At frame 2 (ws = 0.0), perspective-compensated spine center has zero false weight-shift bias
        f_trim = trace.frames[2]
        self.assertLess(abs(f_trim.weight_shift_norm), 0.05)

    def test_real_360_riser_cable_and_skeleton_extraction_on_08_22_and_08_30(self):
        import cv2
        import numpy as np

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

        # At 08:30 (deep brake pull): hands low and wide on 360° fan, brake >= 85%
        self.assertLessEqual(feat_30["left_hand"][0], 165)
        self.assertGreaterEqual(feat_30["left_hand"][1], 172)
        self.assertGreaterEqual(feat_30["right_hand"][0], 272)
        self.assertGreaterEqual(feat_30["right_hand"][1], 172)
        self.assertGreaterEqual(feat_30["brake_left_pct"], 85)
        self.assertGreaterEqual(feat_30["brake_right_pct"], 85)


if __name__ == "__main__":
    unittest.main()
