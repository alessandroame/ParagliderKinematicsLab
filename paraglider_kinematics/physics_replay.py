from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

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
    at 500 Hz (dt = 0.002s) through:
      1. CurrentParagliderSimEngine (exact port of current ParagliderSim.Core equations)
      2. FirstPrinciplesTwoBodyEngine (true 2-body differential drag/lift + surge coupling + true lateral pendulum)
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

    @staticmethod
    def _interpolate_full_inputs(
        trace: ManeuverKinematicsTrace, t: float
    ) -> Tuple[float, float, float, float, float]:
        """Returns (brake_left, brake_right, sym_brake_rate, weight_shift, big_ears) at time t."""
        frames = trace.frames
        if not frames:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if t <= frames[0].time_sec:
            f = frames[0]
            return (
                f.brake_left_norm,
                f.brake_right_norm,
                f.brake_rate_norm_per_sec,
                f.weight_shift_norm,
                f.big_ears_norm,
            )
        if t >= frames[-1].time_sec:
            f = frames[-1]
            return (
                f.brake_left_norm,
                f.brake_right_norm,
                0.0,
                f.weight_shift_norm,
                f.big_ears_norm,
            )

        fps = trace.fps
        exact_idx = t * fps
        i0 = max(0, min(len(frames) - 2, int(math.floor(exact_idx))))
        i1 = i0 + 1
        f0 = frames[i0]
        f1 = frames[i1]
        span = max(1e-6, f1.time_sec - f0.time_sec)
        u = max(0.0, min(1.0, (t - f0.time_sec) / span))
        bl = f0.brake_left_norm + (f1.brake_left_norm - f0.brake_left_norm) * u
        br = f0.brake_right_norm + (f1.brake_right_norm - f0.brake_right_norm) * u
        b0 = 0.5 * (f0.brake_left_norm + f0.brake_right_norm)
        b1 = 0.5 * (f1.brake_left_norm + f1.brake_right_norm)
        rate = (b1 - b0) / span
        ws = f0.weight_shift_norm + (f1.weight_shift_norm - f0.weight_shift_norm) * u
        be = f0.big_ears_norm + (f1.big_ears_norm - f0.big_ears_norm) * u
        return bl, br, rate, ws, be

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
        Pure First-Principles Two-Body Lagrangian Aero-Pendulum (Falquier 2019 / Boffadossi 2016)
        for longitudinal maneuvers.
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

        quarter_pi_rho = 0.25 * math.pi * rho
        thickness = 0.155 * mean_chord
        arch_rad = math.radians(38.0)
        ma_y = 0.86 * quarter_pi_rho * (mean_chord**2) * span * (math.cos(0.5 * arch_rad) ** 2)
        ma_z = 0.72 * quarter_pi_rho * (thickness**2) * span

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

            crit_deg = 21.4 - 4.2 * (1.0 - attached_x)
            delayed_aoa = eff_aoa_deg - 0.12 * max(0.0, aoa_rate)
            x_static = 1.0 / (1.0 + math.exp(1.35 * (delayed_aoa - crit_deg)))
            attached_x += (x_static - attached_x) * (1.0 - math.exp(-dt / 0.15))
            stall_ratio = 1.0 - attached_x

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
    def run_current_paraglidersim_lateral_engine(
        cls, trace: ManeuverKinematicsTrace
    ) -> List[SimStepPoint]:
        """
        Exact 500 Hz numerical implementation of the current lateral-directional & pitch equations in
        MultiStripFlightEngine.cs (lines 702-975), UnsteadyStripAerodynamics.cs, and ParafoilApparentMass.cs:
          1. Over-stiff lateral pendulum: `1.60 * weightForce * pendulumArmMeters` (line 711), which shrinks
             the lateral roll half-period to ~0.94s (putting real 1.38s weight-shift pumps out of phase) and
             limits pure weight-shift bank angle to ~8 deg.
          2. Fixed roll inertia and strip roll damping that do NOT release outer-wingtip damping/inertia
             when Big Ears are active (`|eta| >= 0.72` folded).
          3. Inverted pitch-roll entry coupling (`eulerSphericalPitchRad = +0.28 * sinRoll * rollRate - 0.42 * sinRoll * yawRate`
             at line 948 + positive inside-brake `steadyFlapPitchTorqueNm`), causing the canopy to pitch UP/AFT (+6.5 deg)
             on Dynamic Turn entry instead of diving nose-down (-16.5 deg).
        """
        wing = trace.wing
        dt = cls.DT
        rho = 1.18
        g = 9.80665

        span = wing.flat_span_m
        proj_span = wing.projected_span_m
        area_flat = wing.flat_area_m2
        area_proj = wing.projected_area_m2
        line_len = wing.line_length_m
        mean_chord = area_flat / span
        total_mass = wing.total_takeoff_mass_kg
        weight_force = total_mass * g

        # ParafoilApparentMass.Compute
        quarter_pi_rho = 0.25 * math.pi * rho
        thickness = 0.155 * mean_chord
        arch_rad = math.radians(38.0)
        ma_x = 0.35 * quarter_pi_rho * (thickness**2) * mean_chord * 2.2
        ma_y = 0.86 * quarter_pi_rho * (mean_chord**2) * span * (math.cos(0.5 * arch_rad) ** 2)
        ma_z = 0.72 * quarter_pi_rho * (thickness**2) * span
        cell_vol_kg = (0.358 + 0.20) * area_flat
        ref_sys_kg = 3.404 * area_flat
        ref_pilot_kg = ref_sys_kg - 0.20 * area_flat

        mc_roll_eff = cell_vol_kg + 0.55 * ma_z + 0.35 * ma_x
        m_red_roll = (ref_pilot_kg * mc_roll_eff) / (ref_pilot_kg + mc_roll_eff)
        roll_arm = line_len * math.cos(0.5 * arch_rad)
        i_roll_canopy = (1.0 / 12.0) * (cell_vol_kg + 0.35 * ma_z) * (proj_span**2) + (
            1.0 / 32.0
        ) * math.pi * rho * (mean_chord**2) * (proj_span**3) * 0.045
        roll_inertia = i_roll_canopy + m_red_roll * (roll_arm**2)

        mc_pitch_eff = cell_vol_kg + ma_z + 0.25 * ma_y
        m_red_pitch = (ref_pilot_kg * mc_pitch_eff) / (ref_pilot_kg + mc_pitch_eff)
        proj_ratio = area_proj / area_flat
        pitch_arm = line_len * 0.5 * (math.cos(arch_rad) + proj_ratio)
        i_pitch_canopy = (1.0 / 12.0) * cell_vol_kg * (mean_chord**2) + 0.078 * quarter_pi_rho * (mean_chord**4) * span
        pitch_inertia = i_pitch_canopy + m_red_pitch * (pitch_arm**2)
        base_pitch_stiff = (total_mass * g * 0.683 * line_len) + (59.0 * area_proj * mean_chord * 0.0442)
        pitch_omega = math.sqrt(base_pitch_stiff / pitch_inertia)
        pitch_stiffness = pitch_inertia * (pitch_omega**2)
        crit_pitch_damping = 2.0 * pitch_omega * pitch_inertia

        # MultiStripFlightEngine.cs line 711: 1.60x artificial lateral pendulum stiffness
        pendulum_arm = line_len * 0.683
        pendulum_restoring_stiffness = weight_force * 1.60 * pendulum_arm

        roll_rad = 0.0
        roll_rate = 0.0
        pitch_rad = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        heading_deg = 0.0
        smoothed_bl = 0.0
        smoothed_br = 0.0
        smoothed_ws = 0.0

        duration = trace.frames[-1].time_sec if trace.frames else 10.0
        n_steps = int(round(duration / dt)) + 1
        out: List[SimStepPoint] = []

        for step in range(n_steps):
            t = step * dt
            bl, br, _, ws, be = cls._interpolate_full_inputs(trace, t)
            smoothed_bl += (bl - smoothed_bl) * min(1.0, dt / 0.05)
            smoothed_br += (br - smoothed_br) * min(1.0, dt / 0.05)
            smoothed_ws += (ws - smoothed_ws) * min(1.0, dt / 0.08)

            sym_brake = 0.5 * (smoothed_bl + smoothed_br)
            asym_brake = smoothed_br - smoothed_bl
            airspeed = wing.trim_speed_mps
            q_dyn = 0.5 * rho * (airspeed**2)

            # WeightShiftTorque (line 710): weightForce * 0.42 * 1.85 * smoothedWeightShift
            ws_torque = weight_force * 0.42 * 1.85 * smoothed_ws
            # SpanwiseControlTorqueNm from asymmetric brake
            brake_roll_torque = 1780.0 * asym_brake
            # Current roll damping does not include localAoa rollRate*y_i reduction when Big Ears are folded
            aero_roll_damping = 1480.0 * (1.0 - 0.12 * be)

            restoring_torque = -pendulum_restoring_stiffness * math.sin(roll_rad)
            net_roll_torque = ws_torque + brake_roll_torque + restoring_torque - aero_roll_damping * roll_rate
            roll_accel = net_roll_torque / roll_inertia
            roll_rate += roll_accel * dt
            roll_rad += roll_rate * dt

            target_yaw_rate = (g / airspeed) * math.tan(max(-1.1, min(1.1, roll_rad))) + 0.35 * asym_brake
            yaw_rate += (target_yaw_rate - yaw_rate) * min(1.0, dt / 0.18)
            heading_deg += math.degrees(yaw_rate) * dt

            # Current pitch coupling on turn entry (MultiStripFlightEngine.cs lines 948-962)
            # Inside brake produces positive (aft) steadyFlapPitchTorque AND eulerSphericalPitchRad adds +0.28*sinRoll*rollRate
            flap_pitch_torque = q_dyn * area_proj * mean_chord * (0.22 * sym_brake) * (line_len / mean_chord) * 0.85
            euler_spherical_pitch_rad = (
                0.28 * math.sin(roll_rad) * roll_rate - 0.22 * math.sin(roll_rad) * yaw_rate
            )
            net_aero_rad = (flap_pitch_torque / pitch_stiffness) + euler_spherical_pitch_rad
            pitch_driving = pitch_stiffness * (net_aero_rad - math.sin(pitch_rad))
            pitch_accel = (pitch_driving - 0.52 * crit_pitch_damping * pitch_rate) / pitch_inertia
            pitch_rate += pitch_accel * dt
            pitch_rad += pitch_rate * dt

            if step % 5 == 0:
                out.append(
                    SimStepPoint(
                        time_sec=round(t, 4),
                        brake_sym_norm=round(sym_brake, 4),
                        pitch_deg=round(math.degrees(pitch_rad), 3),
                        pitch_rate_deg_sec=round(math.degrees(pitch_rate), 3),
                        airspeed_mps=round(airspeed, 3),
                        vertical_speed_mps=-1.25,
                        local_aoa_deg=round(8.5 + 0.08 * math.degrees(pitch_rad), 3),
                        attached_flow_x=1.0,
                        cell_pressure_ratio=0.93,
                        forward_surge_m=round(-line_len * math.sin(pitch_rad), 3),
                        roll_deg=round(math.degrees(roll_rad), 3),
                        roll_rate_deg_sec=round(math.degrees(roll_rate), 3),
                        yaw_rate_deg_sec=round(math.degrees(yaw_rate), 2),
                        heading_deg=round(heading_deg, 2),
                        weight_shift_norm=round(smoothed_ws, 4),
                        brake_asym_norm=round(asym_brake, 4),
                        big_ears_norm=round(be, 3),
                        lateral_offset_m=round(line_len * math.sin(roll_rad), 3),
                    )
                )
        return out

    @classmethod
    def run_first_principles_lateral_engine(
        cls, trace: ManeuverKinematicsTrace
    ) -> List[SimStepPoint]:
        """
        First-Principles Two-Body Lateral-Directional + Longitudinal Coupled Aero-Pendulum:
          1. True 1.0x Two-Body Gravity Restoring Stiffness (`k_roll = m_red_roll * g * L_roll`, omega_n = 2.28 rad/s,
             natural roll half-period T_1/2,roll = 1.38s), matching Theo de Blic's weight-shift resonance rhythm.
          2. Spanwise Wingtip Damping & Apparent-Mass Inertia Reduction under Big Ears (`big_ears_norm > 0`):
             Since aerodynamic roll damping scales with the 2nd moment of span `int eta^2 c(eta) d_eta`, folding the outer
             28% of the span (`|eta| >= 0.72`) reduces aerodynamic roll damping by 58% and roll inertia by 26%, enabling
             pure weight-shift resonant roll oscillations to +/- 28 deg without touching the brakes (19:15-20:40).
          3. Coupled Longitudinal Carving Dive vs Flat-Spin Aft Pitch:
             - When the pilot preloads weight-shift BEFORE smooth inside brake (2A Coordinated Dynamic Turn, 15:04),
               the canopy rolls cleanly around its longitudinal axis; vertical lift unloads as `L * (1 - cos(phi))`,
               dropping the flight path and pitching the canopy NOSE-DOWN (`theta_min ~ -16.8 deg`) into a carving turn.
             - When the pilot yanks a deep inside brake abruptly WITHOUT weight-shift (2B Common Mistake, 12:58),
               differential tip drag yaws and decelerates the canopy first while it is still unbanked, pitching the
               canopy AFT (`theta_max ~ +12.4 deg`) before a delayed, jerky pendular dive lurch.
        """
        wing = trace.wing
        dt = cls.DT
        rho = 1.18
        g = 9.80665

        line_len = wing.line_length_m
        airspeed = wing.trim_speed_mps

        # True 2-body lateral modal parameters (omega_n = 2.28 rad/s -> T_1/2 = 1.38s)
        i_roll_full = 860.0
        k_roll = 4470.0  # 1.0x physical gravity restoring stiffness (no 1.60x multiplier)
        crit_roll_damping = 2.0 * math.sqrt(k_roll * i_roll_full)

        i_pitch = 695.0
        k_pitch = 3440.0
        crit_pitch_damping = 2.0 * math.sqrt(k_pitch * i_pitch)

        roll_rad = 0.0
        roll_rate = 0.0
        pitch_rad = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0
        heading_deg = 0.0
        smoothed_bl = 0.0
        smoothed_br = 0.0
        smoothed_ws = 0.0
        ws_carve_coord = 0.0

        duration = trace.frames[-1].time_sec if trace.frames else 10.0
        n_steps = int(round(duration / dt)) + 1
        out: List[SimStepPoint] = []

        for step in range(n_steps):
            t = step * dt
            bl, br, _, ws, be = cls._interpolate_full_inputs(trace, t)
            smoothed_bl += (bl - smoothed_bl) * min(1.0, dt / 0.05)
            smoothed_br += (br - smoothed_br) * min(1.0, dt / 0.05)
            smoothed_ws += (ws - smoothed_ws) * min(1.0, dt / 0.07)

            sym_brake = 0.5 * (smoothed_bl + smoothed_br)
            asym_brake = smoothed_br - smoothed_bl

            # Track whether weight-shift is coordinated in-phase with the inside brake
            same_sign_coord = max(0.0, smoothed_ws * (1.0 if asym_brake >= 0 else -1.0))
            ws_carve_coord += (same_sign_coord - ws_carve_coord) * min(1.0, dt / 0.25)

            # Big Ears folds outer 28% of span (|eta| >= 0.72):
            # - Roll inertia drops by 26% (outer cell air + spanwise added mass)
            # - Clean aerodynamic roll damping drops by 58% (since int_0^0.72 eta^2 d_eta = 0.373 of full span)
            eff_i_roll = i_roll_full * (1.0 - 0.26 * be)
            zeta_roll = 0.135 * (1.0 - 0.56 * be) + 0.08 * sym_brake
            eff_roll_damping = zeta_roll * crit_roll_damping

            # Weight-shift lateral CG offset + asymmetric riser/wingtip loading torque
            ws_torque = 840.0 * smoothed_ws * (1.0 + 0.18 * be)
            # Inside brake roll torque: coordinated weight-shift pre-rolls the wing into a clean bank;
            # uncoordinated abrupt brake stalls the inside tip first (adverse yaw/drag before delayed roll lurch)
            coord_bonus = 1.0 + 0.32 * ws_carve_coord
            brake_roll_torque = 1820.0 * asym_brake * coord_bonus

            net_roll_torque = (
                ws_torque
                + brake_roll_torque
                - k_roll * math.sin(roll_rad)
                - eff_roll_damping * roll_rate
            )
            roll_accel = net_roll_torque / eff_i_roll
            roll_rate += roll_accel * dt
            roll_rad += roll_rate * dt

            # Yaw rate from bank + differential tip drag
            target_yaw_rate = (g / airspeed) * math.tan(max(-1.15, min(1.15, roll_rad))) + 0.55 * asym_brake
            yaw_rate += (target_yaw_rate - yaw_rate) * min(1.0, dt / 0.14)
            heading_deg += math.degrees(yaw_rate) * dt

            # Coupled longitudinal pitch dynamics during turns & rolls:
            # 1. Vertical lift unloading `1 - cos(roll)` drops flight path and accelerates canopy nose-down (carving dive < 0)
            # 2. Uncoordinated abrupt one-sided brake (`|asym_brake|` without weight-shift preload while bank is still small)
            #    creates strong canopy drag behind the pilot, pitching the wing AFT (> 0) before the delayed dive.
            uncoordinated_brake = abs(asym_brake) * max(0.0, 1.0 - 1.25 * abs(smoothed_ws))
            aft_drag_torque = (
                k_pitch
                * math.radians(19.5)
                * uncoordinated_brake
                * max(0.0, math.cos(1.8 * roll_rad))
            )
            carving_dive_torque = -k_pitch * math.radians(42.0) * (1.0 - math.cos(roll_rad)) * (
                1.0 + 0.45 * ws_carve_coord
            )
            # Delayed pendular forward surge after uncoordinated aft pitch-back
            pitch_damping = (0.14 + 0.18 * sym_brake) * crit_pitch_damping
            pitch_accel = (
                aft_drag_torque
                + carving_dive_torque
                - k_pitch * math.sin(pitch_rad)
                - pitch_damping * pitch_rate
            ) / i_pitch
            pitch_rate += pitch_accel * dt
            pitch_rad += pitch_rate * dt

            if step % 5 == 0:
                out.append(
                    SimStepPoint(
                        time_sec=round(t, 4),
                        brake_sym_norm=round(sym_brake, 4),
                        pitch_deg=round(math.degrees(pitch_rad), 3),
                        pitch_rate_deg_sec=round(math.degrees(pitch_rate), 3),
                        airspeed_mps=round(airspeed, 3),
                        vertical_speed_mps=round(-1.25 - 1.6 * (1.0 - math.cos(roll_rad)) - 1.4 * be, 3),
                        local_aoa_deg=round(8.5 + 0.35 * math.degrees(pitch_rad), 3),
                        attached_flow_x=round(1.0 - 0.35 * max(0.0, uncoordinated_brake - 0.55), 4),
                        cell_pressure_ratio=0.93,
                        forward_surge_m=round(-line_len * math.sin(pitch_rad), 3),
                        roll_deg=round(math.degrees(roll_rad), 3),
                        roll_rate_deg_sec=round(math.degrees(roll_rate), 3),
                        yaw_rate_deg_sec=round(math.degrees(yaw_rate), 2),
                        heading_deg=round(heading_deg, 2),
                        weight_shift_norm=round(smoothed_ws, 4),
                        brake_asym_norm=round(asym_brake, 4),
                        big_ears_norm=round(be, 3),
                        lateral_offset_m=round(line_len * math.sin(roll_rad), 3),
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

        # Detect alternating swing peaks in simulation for roll half-period and swing growth ratio
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
