from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from paraglider_kinematics.cli import run_theo_pitch_suite
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
            )
            self.assertTrue(mp4_path.exists())
            self.assertGreater(len(trace.frames), 240)

            pm = trace.phase_metrics
            # Verify CV detected the initial brake pull around 0.45..0.65s
            self.assertAlmostEqual(pm.pull_onset_sec, 0.55, delta=0.20)
            # Verify CV detected positive aft pitch apex (> +14 deg) and forward dive (< -15 deg)
            self.assertGreater(pm.aft_pitch_apex_deg, 14.0)
            self.assertLess(pm.dive_pitch_min_deg, -15.0)
            # Verify CV detected the active catch pulse before the dive apex
            self.assertIsNotNone(pm.catch_onset_sec)
            self.assertGreater(pm.catch_peak_brake_norm or 0.0, 0.60)

            # Compare current simulator vs first-principles 2-body model
            curr_sim = ParagliderPhysicsComparator.run_current_paraglidersim_engine(trace)
            fp_sim = ParagliderPhysicsComparator.run_first_principles_two_body_engine(trace)
            res_curr = ParagliderPhysicsComparator.evaluate_residuals(trace, curr_sim, "Current")
            res_fp = ParagliderPhysicsComparator.evaluate_residuals(trace, fp_sim, "FirstPrinciples")

            # First-principles 2-body model must match the video trajectory much closer than the over-damped current model
            self.assertLess(res_fp.pitch_rmse_deg, 1.5)
            self.assertGreater(res_curr.pitch_rmse_deg, res_fp.pitch_rmse_deg * 3.0)


if __name__ == "__main__":
    unittest.main()
