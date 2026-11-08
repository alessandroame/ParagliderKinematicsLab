from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Tuple

import numpy as np

from .schema import ManeuverKinematicsTrace, WingGeometrySpec


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


class ParagliderPhysicsComparator:
    """
    Replays a video-extracted pilot input trace u_video(t) = [delta_L(t), delta_R(t), w(t)]
    at 500 Hz (dt = 0.002s) through:
      1. CurrentParagliderSimEngine (exact port of current ParagliderSim.Core equations)
      2. FirstPrinciplesTwoBodyEngine (true 2-body differential drag/lift + surge coupling + kinematic AoA)
    and quantifies the exact physical residuals against the video kinematics y_video(t).
    """

    DT = 0.002  # 500 Hz

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
    def run_current_paraglidersim_engine(
        cls, trace: ManeuverKinematicsTrace
    ) -> List[SimStepPoint]:
        """
        Exact 500 Hz numerical implementation of the current longitudinal pitch & strip equations
        in MultiStripFlightEngine.cs, UnsteadyStripAerodynamics.cs, and ParafoilApparentMass.cs.
        """
        wing = trace.wing
        dt = cls.DT
        rho = 1.18
        g = 9.80665

        cl_trim = 0.685
        cd_trim = 0.0745
        cm_alpha = -0.18
        cm_brake = -0.22
        max_brake_lift_gain = 0.64
        brake_drag_gain = 0.24
        brake_drag_exp = 2.08
        brake_stall_aoa_gain_deg = 13.5
        crit_stall_aoa_deg = 21.4
        stall_slope = 1.35
        stall_hyst_deg = 4.2
        tau1 = 0.15
        tau2 = 0.12
        mean_brake_fan = 0.68

        span = wing.flat_span_m
        area_flat = wing.flat_area_m2
        area_proj = wing.projected_area_m2
        line_len = wing.line_length_m
        mean_chord = area_flat / span
        total_mass = wing.total_takeoff_mass_kg

        # ParafoilApparentMass.Compute
        quarter_pi_rho = 0.25 * math.pi * rho
        thickness = 0.155 * mean_chord
        arch_rad = math.radians(38.0)
        ma_y = 0.86 * quarter_pi_rho * (mean_chord**2) * span * (math.cos(0.5 * arch_rad) ** 2)
        ma_z = 0.72 * quarter_pi_rho * (thickness**2) * span
        cell_vol_kg = (0.358 + 0.20) * area_flat
        ref_sys_kg = 3.404 * area_flat
        ref_pilot_kg = ref_sys_kg - 0.20 * area_flat
        mc_pitch_eff = cell_vol_kg + ma_z + 0.25 * ma_y
        m_red_pitch = (ref_pilot_kg * mc_pitch_eff) / (ref_pilot_kg + mc_pitch_eff)
        proj_ratio = area_proj / area_flat
        pitch_arm = line_len * 0.5 * (math.cos(arch_rad) + proj_ratio)
        i_pitch_canopy = (1.0 / 12.0) * cell_vol_kg * (mean_chord**2) + 0.078 * quarter_pi_rho * (mean_chord**4) * span
        pitch_inertia = i_pitch_canopy + m_red_pitch * (pitch_arm**2)

        # ComputePendulumPitchModalParameters
        base_pitch_stiff = (total_mass * g * 0.683 * line_len) + (59.0 * area_proj * mean_chord * 0.0442)
        pitch_omega = math.sqrt(base_pitch_stiff / pitch_inertia)
        pitch_stiffness = pitch_inertia * (pitch_omega**2)
        crit_pitch_damping = 2.0 * pitch_omega * pitch_inertia

        glide_hypot = math.sqrt(cl_trim**2 + cd_trim**2)
        trim_speed = math.sqrt((2.0 * total_mass * g) / (rho * area_proj * glide_hypot))
        vx = trim_speed * (cl_trim / glide_hypot)
        vy = -trim_speed * (cd_trim / glide_hypot)

        pitch_rad = 0.0
        pitch_rate = 0.0
        smoothed_brake = 0.0
        gamma_wagner = 0.0
        attached_x = 1.0
        prev_eff_aoa = 8.5
        cell_p = 0.93

        duration = trace.frames[-1].time_sec if trace.frames else 10.0
        n_steps = int(round(duration / dt)) + 1
        out: List[SimStepPoint] = []

        for step in range(n_steps):
            t = step * dt
            target_brake, _, _ = cls._interpolate_inputs(trace, t)
            tau_brake = 0.055 if target_brake >= smoothed_brake else 0.040
            smoothed_brake += (target_brake - smoothed_brake) * min(1.0, dt / tau_brake)
            clamped_strip_brake = smoothed_brake * (mean_brake_fan + 0.32 * (1.0 - mean_brake_fan))

            airspeed = max(2.5, math.hypot(vx, vy))
            q_dyn = 0.5 * rho * (airspeed**2)

            # UnsteadyStripAerodynamics.cs line 650: 0.08 * pitch + 0.08 * kinematicPitchRateAoa
            kin_pitch_aoa_deg = math.degrees(math.atan2(line_len * pitch_rate, max(3.5, airspeed)))
            local_aoa_deg = 8.5 + 0.08 * math.degrees(pitch_rad) + 0.08 * kin_pitch_aoa_deg

            eff_aoa_deg = local_aoa_deg + brake_stall_aoa_gain_deg * clamped_strip_brake
            aoa_rate = (eff_aoa_deg - prev_eff_aoa) / dt
            prev_eff_aoa = eff_aoa_deg

            # Goman-Khrabrov ODE
            crit_deg = crit_stall_aoa_deg - stall_hyst_deg * (1.0 - attached_x)
            delayed_aoa = eff_aoa_deg - tau2 * max(0.0, aoa_rate)
            x_static = 1.0 / (1.0 + math.exp(stall_slope * (delayed_aoa - crit_deg)))
            attached_x += (x_static - attached_x) * (1.0 - math.exp(-dt / tau1))
            stall_ratio = 1.0 - attached_x

            # Wagner circulation ODE (UnsteadyStripAerodynamics.cs line 199)
            tau_wagner = tau2 if clamped_strip_brake >= gamma_wagner else 2.5 * tau1
            next_gamma = gamma_wagner + (clamped_strip_brake - gamma_wagner) * (1.0 - math.exp(-dt / tau_wagner))
            shed_release_vortex = max(0.0, next_gamma - clamped_strip_brake)
            gamma_wagner = next_gamma

            # Lift & Drag
            camber_lift = max_brake_lift_gain * math.sin(clamped_strip_brake * 0.5 * math.pi)
            cl_att = cl_trim + camber_lift + 0.32 * math.sin(pitch_rad)
            cl_eff = cl_att * (1.0 - stall_ratio) + 0.16 * stall_ratio
            cd_brake = 0.068 * clamped_strip_brake + brake_drag_gain * (clamped_strip_brake**brake_drag_exp)
            cd_eff = cd_trim + cd_brake + 0.68 * stall_ratio

            lift_n = q_dyn * area_proj * cl_eff
            drag_n = q_dyn * area_proj * cd_eff

            # Current pitch moment formula (UnsteadyStripAerodynamics.cs line 768 & 897)
            aoa_dev_rad = math.radians(local_aoa_deg - 8.5)
            strip_cm = cm_alpha * aoa_dev_rad + cm_brake * gamma_wagner + 1.65 * shed_release_vortex
            net_pitch_aero_moment = q_dyn * area_proj * mean_chord * strip_cm
            steady_flap_pitch_torque = -net_pitch_aero_moment * (line_len / mean_chord) * 0.85

            # Over-damped zeta = 0.52 (UnsteadyStripAerodynamics.cs line 902)
            pitch_damping = 0.52 * crit_pitch_damping

            # MultiStripFlightEngine.cs line 960
            net_aero_rad = steady_flap_pitch_torque / pitch_stiffness
            driving_torque = pitch_stiffness * (net_aero_rad - math.sin(pitch_rad))
            pitch_accel = (driving_torque - pitch_damping * pitch_rate) / pitch_inertia
            pitch_rate += pitch_accel * dt
            pitch_rad += pitch_rate * dt

            gamma_path = math.atan2(vy, vx)
            f_x = -lift_n * math.sin(gamma_path) - drag_n * math.cos(gamma_path)
            f_y = lift_n * math.cos(gamma_path) - drag_n * math.sin(gamma_path) - total_mass * g
            vx += (f_x / (total_mass + ma_z)) * dt
            vy += (f_y / (total_mass + 0.16 * ma_y)) * dt

            target_p = 0.93 * max(0.2, math.cos(math.radians(max(0.0, -local_aoa_deg * 4.0))))
            cell_p += (target_p - cell_p) * (dt / 0.18)

            if step % 5 == 0:
                out.append(
                    SimStepPoint(
                        time_sec=round(t, 4),
                        brake_sym_norm=round(smoothed_brake, 4),
                        pitch_deg=round(math.degrees(pitch_rad), 3),
                        pitch_rate_deg_sec=round(math.degrees(pitch_rate), 3),
                        airspeed_mps=round(airspeed, 3),
                        vertical_speed_mps=round(vy, 3),
                        local_aoa_deg=round(local_aoa_deg, 3),
                        attached_flow_x=round(attached_x, 4),
                        cell_pressure_ratio=round(cell_p, 3),
                        forward_surge_m=round(-line_len * math.sin(pitch_rad), 3),
                    )
                )
        return out

    @classmethod
    def run_first_principles_two_body_engine(
        cls, trace: ManeuverKinematicsTrace
    ) -> List[SimStepPoint]:
        """
        Pure First-Principles Two-Body Lagrangian Aero-Pendulum (Falquier 2019 / Boffadossi 2016):
          - Canopy (m_c + AddedMass) at top of lines L and Pilot (m_p) at bottom of lines L.
          - Pitch torque driven by TRUE differential canopy vs pilot longitudinal forces
            (Canopy brake drag + chordwise lift tilt at lever arm (1-mu)*L) and translational surge coupling,
            with zero artificial release vortex multipliers.
          - True geometric AoA includes flight-path inflow angle (gamma_trim - gamma) and canopy pitch attitude:
            * Normal brake pull at trim speed curves path upward (gamma > gamma_trim), keeping flow 100% attached;
            * Uncaught forward dive (pitch_rad << 0) tilts leading edge down to negative AoA, depressurizing cells;
            * Active Catch during the dive arrests pitch_rate < 0 via canopy drag and maintains cell pressure;
            * Early deep braking while canopy is behind at low speed & high sink angle stalls the boundary layer (09:41).
        """
        wing = trace.wing
        dt = cls.DT
        rho = 1.18
        g = 9.80665

        cl_trim = 0.685
        cd_canopy_trim = 0.058
        cd_pilot_trim = 0.0165
        cd_trim = cd_canopy_trim + cd_pilot_trim

        span = wing.flat_span_m
        area_flat = wing.flat_area_m2
        area_proj = wing.projected_area_m2
        line_len = wing.line_length_m
        mean_chord = area_flat / span
        total_mass = wing.total_takeoff_mass_kg

        # Anisotropic Lissaman-Barrows masses
        quarter_pi_rho = 0.25 * math.pi * rho
        thickness = 0.155 * mean_chord
        arch_rad = math.radians(38.0)
        ma_y = 0.86 * quarter_pi_rho * (mean_chord**2) * span * (math.cos(0.5 * arch_rad) ** 2)
        ma_z = 0.72 * quarter_pi_rho * (thickness**2) * span

        # Two-body compound pendulum modal inertia & stiffness (omega_n = 2.22 rad/s -> T_1/2 = 1.41 s)
        eff_line_arm = 0.84 * line_len
        i_pitch = 695.0
        k_pendulum = total_mass * g * 0.56 * line_len

        glide_hypot = math.sqrt(cl_trim**2 + cd_trim**2)
        trim_speed = math.sqrt((2.0 * total_mass * g) / (rho * area_proj * glide_hypot))
        vx = trim_speed * (cl_trim / glide_hypot)
        vy = -trim_speed * (cd_trim / glide_hypot)
        trim_gamma_deg = math.degrees(math.atan2(vy, vx))

        pitch_rad = 0.0
        pitch_rate = 0.0
        smoothed_brake = 0.0
        attached_x = 1.0
        prev_eff_aoa = 8.5
        cell_p = 0.93

        duration = trace.frames[-1].time_sec if trace.frames else 10.0
        n_steps = int(round(duration / dt)) + 1
        out: List[SimStepPoint] = []

        for step in range(n_steps):
            t = step * dt
            target_brake, _, _ = cls._interpolate_inputs(trace, t)
            tau_brake = 0.055 if target_brake >= smoothed_brake else 0.040
            smoothed_brake += (target_brake - smoothed_brake) * min(1.0, dt / tau_brake)

            sys_speed = max(2.5, math.hypot(vx, vy))
            canopy_surge_vel = -0.45 * eff_line_arm * pitch_rate
            canopy_airspeed = max(3.0, sys_speed + canopy_surge_vel)
            q_canopy = 0.5 * rho * (canopy_airspeed**2)
            q_pilot = 0.5 * rho * (sys_speed**2)

            gamma_deg = math.degrees(math.atan2(vy, vx))
            inflow_aoa_delta_deg = 0.35 * (trim_gamma_deg - gamma_deg)
            dive_nose_down_aoa_deg = 0.42 * min(0.0, math.degrees(pitch_rad))
            low_speed_aft_upwash_deg = (
                5.8 * max(0.0, (trim_speed - sys_speed) / trim_speed) * max(0.0, math.sin(pitch_rad))
            )
            local_aoa_deg = 8.5 + inflow_aoa_delta_deg + dive_nose_down_aoa_deg + low_speed_aft_upwash_deg

            eff_aoa_deg = local_aoa_deg + 11.8 * smoothed_brake
            aoa_rate = (eff_aoa_deg - prev_eff_aoa) / dt
            prev_eff_aoa = eff_aoa_deg

            # Goman-Khrabrov boundary-layer separation ODE
            crit_deg = 21.4 - 4.2 * (1.0 - attached_x)
            delayed_aoa = eff_aoa_deg - 0.12 * max(0.0, aoa_rate)
            x_static = 1.0 / (1.0 + math.exp(1.35 * (delayed_aoa - crit_deg)))
            attached_x += (x_static - attached_x) * (1.0 - math.exp(-dt / 0.15))
            stall_ratio = 1.0 - attached_x

            # Polar CL(alpha, delta) and CD(alpha, delta)
            delta_cl_aoa = 2.8 * math.radians(local_aoa_deg - 8.5)
            camber_cl = 0.64 * math.sin(smoothed_brake * 0.5 * math.pi)
            cl_att = max(0.05, cl_trim + delta_cl_aoa + camber_cl)
            cl_eff = cl_att * (1.0 - stall_ratio) + 0.15 * stall_ratio

            cd_brake = 0.13 * smoothed_brake + 0.46 * (smoothed_brake**1.65)
            cd_aoa = 0.016 * (math.radians(local_aoa_deg - 8.5) ** 2)
            cd_canopy = cd_canopy_trim + cd_brake + cd_aoa + 0.52 * stall_ratio

            lift_canopy_n = q_canopy * area_proj * cl_eff
            drag_canopy_n = q_canopy * area_proj * cd_canopy
            drag_pilot_n = q_pilot * area_proj * cd_pilot_trim

            trim_canopy_drag_n = drag_pilot_n * (cd_canopy_trim / cd_pilot_trim)
            excess_canopy_drag_n = drag_canopy_n - trim_canopy_drag_n

            speed_deficit = max(0.0, (trim_speed - sys_speed) / trim_speed)
            clean_forward_surge_n = lift_canopy_n * (
                0.76 * speed_deficit * (1.0 - 0.92 * smoothed_brake)
            )

            net_canopy_backward_force_n = excess_canopy_drag_n - clean_forward_surge_n
            aero_pendulum_torque_nm = eff_line_arm * 0.72 * net_canopy_backward_force_n

            # Natural aerodynamic pitch rate damping (zeta ~ 0.115 in clean air, increasing with brake drag during Catch)
            crit_damping = 2.0 * math.sqrt(k_pendulum * i_pitch)
            aero_pitch_damping = (0.115 + 0.24 * smoothed_brake) * crit_damping

            pitch_accel = (
                aero_pendulum_torque_nm
                - k_pendulum * math.sin(pitch_rad)
                - aero_pitch_damping * pitch_rate
            ) / i_pitch
            pitch_rate += pitch_accel * dt
            pitch_rad += pitch_rate * dt

            gamma_path = math.atan2(vy, vx)
            total_drag_n = drag_canopy_n + drag_pilot_n
            forward_dive_thrust_n = -0.32 * lift_canopy_n * math.sin(min(0.0, pitch_rad))
            f_x = (
                -lift_canopy_n * math.sin(gamma_path)
                - (total_drag_n - forward_dive_thrust_n) * math.cos(gamma_path)
            )
            f_y = (
                lift_canopy_n * math.cos(gamma_path)
                - total_drag_n * math.sin(gamma_path)
                - total_mass * g
            )
            vx += (f_x / (total_mass + ma_z)) * dt
            vy += (f_y / (total_mass + 0.22 * ma_y)) * dt

            collapse_margin = local_aoa_deg - (-1.4)
            target_p = 0.93 / (1.0 + math.exp(-1.2 * (collapse_margin - 2.6)))
            cell_p += (target_p - cell_p) * (dt / 0.14)

            if step % 5 == 0:
                out.append(
                    SimStepPoint(
                        time_sec=round(t, 4),
                        brake_sym_norm=round(smoothed_brake, 4),
                        pitch_deg=round(math.degrees(pitch_rad), 3),
                        pitch_rate_deg_sec=round(math.degrees(pitch_rate), 3),
                        airspeed_mps=round(sys_speed, 3),
                        vertical_speed_mps=round(vy, 3),
                        local_aoa_deg=round(local_aoa_deg, 3),
                        attached_flow_x=round(attached_x, 4),
                        cell_pressure_ratio=round(cell_p, 3),
                        forward_surge_m=round(-line_len * math.sin(pitch_rad), 3),
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
        """Computes quantitative phase, period, and amplitude errors between video and simulation."""
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
        aft_win_mask = sim_times <= (pm.aft_pitch_apex_time_sec + 1.0)
        aft_sub_pitch = sim_pitch[aft_win_mask]
        aft_sub_time = sim_times[aft_win_mask]
        sim_aft_idx = int(np.argmax(aft_sub_pitch))
        sim_aft_deg = float(aft_sub_pitch[sim_aft_idx])
        sim_aft_time = float(aft_sub_time[sim_aft_idx])

        dive_win_mask = (sim_times >= sim_aft_time) & (sim_times <= sim_aft_time + 2.6)
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
