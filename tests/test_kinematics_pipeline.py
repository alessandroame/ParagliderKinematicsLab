from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from paraglider_kinematics.physics_replay import ParagliderPhysicsComparator
from paraglider_kinematics.theo_de_blic_dataset import generate_video_and_extract_trace


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
            # Weight-shift must lead the inside brake pull by ~0.25-0.35s
            self.assertGreater(pm_turn.brake_to_ws_lag_sec or 0.0, 0.20)
            # Coordinated turn must carve nose-down (dive_pitch_min_deg < -10 deg) and bank > +30 deg
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
            # Current simulator pitches UP instead of diving nose-down on turn entry (entry_pitch_dive_error_deg > +10 deg)
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
            # Verify CV detected folded outer wingtips (big_ears_norm > 0.75) and zero brake input
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


if __name__ == "__main__":
    unittest.main()
