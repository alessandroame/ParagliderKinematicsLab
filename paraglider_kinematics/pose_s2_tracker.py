from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover
    mp = None
    mp_python = None
    mp_vision = None

import numpy as np

_POSE_LANDMARKER: Any = None
_POSE_CHECKED: bool = False
_CLAHE: Any = None


def get_cached_pose_landmarker() -> Any:
    global _POSE_LANDMARKER, _POSE_CHECKED
    if _POSE_CHECKED:
        return _POSE_LANDMARKER
    _POSE_CHECKED = True
    if mp is None or mp_python is None or mp_vision is None:
        return None
    model_path = Path(__file__).resolve().parents[1] / "out" / "models" / "pose_landmarker_heavy.task"
    if not model_path.is_file():
        return None
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
    )
    _POSE_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(opts)
    return _POSE_LANDMARKER


def enhance_glare_bgr(bgr_crop: np.ndarray) -> np.ndarray:
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def pixel_to_equidistant_ray_480(
    u: float,
    v: float,
    f_eq: float = 90.0,
    cx: float = 240.0,
    cy: float = 135.0,
) -> np.ndarray:
    """
    Maps a normalized 480x270 pixel (u, v) to a unit 3D ray on S^2 under the equidistant
    fisheye projection theta = hypot(u - cx, v - cy) / f_eq (with horizontal pivot theta = pi/2
    at |u - 240| ~ 141 px for f_eq = 90.0 px).
    """
    dx = float(u - cx)
    dy = float(v - cy)
    r_px = math.hypot(dx, dy)
    if r_px < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    theta = r_px / f_eq
    sin_t = math.sin(theta)
    return np.array(
        [sin_t * (dx / r_px), sin_t * (dy / r_px), math.cos(theta)],
        dtype=np.float64,
    )


def detect_horizon_s2_480(
    bgr_480: np.ndarray,
    canopy_box: List[float] | List[int],
    head_pt: List[float] | Tuple[float, float],
    f_eq: float = 90.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Fits the Earth horizon great circle n_sky . r = 0 on S^2 via 4-step Huber IRLS SVD,
    suppressing canopy arch, riser bands, pilot helmet, and bottom-right PiP border.
    Returns (n_sky_unit, y_horizon_left_corridor, y_horizon_right_corridor).
    """
    hsv = cv2.cvtColor(bgr_480, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    B = bgr_480[:, :, 0].astype(np.float32)
    G = bgr_480[:, :, 1].astype(np.float32)
    R = bgr_480[:, :, 2].astype(np.float32)

    is_sky = ((B > 132) & (B >= R + 6) & (B >= G - 10) & (V > 122)) | (
        (V > 192) & (S < 62) & (B >= R - 8)
    )
    cx, cy, cw, ch = canopy_box
    c_x0 = max(0, int(cx - 0.54 * cw))
    c_x1 = min(480, int(cx + 0.54 * cw))
    c_y1 = min(270, int(cy + 0.56 * ch))
    pip_border = float(np.mean(V[162:168, 326:448] > 220)) > 0.65
    hy_head = float(head_pt[1])

    rays: List[np.ndarray] = []
    for x in range(8, 472, 4):
        if 176 <= x <= 214 or 258 <= x <= 298:
            continue
        y_start = c_y1 + 5 if (c_x0 <= x <= c_x1) else (30 if x > 372 else 6)
        if 214 < x < 258:
            y_end = max(y_start, int(hy_head - 14.0))
        elif pip_border and x >= 318:
            y_end = 155
        else:
            y_end = 256
        if y_end <= y_start + 10:
            continue
        col_sky = is_sky[y_start:y_end, x]
        for y_rel in range(4, len(col_sky) - 6):
            if (
                float(np.mean(col_sky[max(0, y_rel - 4) : y_rel])) >= 0.75
                and float(np.mean(col_sky[y_rel : y_rel + 6])) <= 0.20
            ):
                fy = float(y_start + y_rel)
                rays.append(pixel_to_equidistant_ray_480(float(x), fy, f_eq=f_eq))
                break

    if len(rays) < 8:
        return np.array([0.0, -0.98, 0.18], dtype=np.float64), 120.0, 120.0

    r_mat = np.stack(rays, axis=0)
    _, _, vt = np.linalg.svd(r_mat)
    n = vt[-1]
    if n[1] > 0:
        n = -n
    for _ in range(4):
        res = r_mat @ n
        mad = max(0.015, float(np.median(np.abs(res))))
        w_irls = 1.0 / np.maximum(1.0, np.abs(res) / (1.4 * mad))
        _, _, vt = np.linalg.svd(r_mat * w_irls[:, None])
        n = vt[-1]
        if n[1] > 0:
            n = -n
    n = n / max(1e-9, float(np.linalg.norm(n)))

    def hor_y_at(xq: float) -> float:
        best_gy, best_val = 125.0, 999.0
        for gy in range(60, 220, 2):
            val = abs(float(np.dot(n, pixel_to_equidistant_ray_480(xq, float(gy), f_eq))))
            if val < best_val:
                best_val, best_gy = val, float(gy)
        return best_gy

    return n, hor_y_at(185.0), hor_y_at(282.0)


def detect_below_horizon_glove_480(
    bgr_480: np.ndarray,
    side: str,
    y_hor_corridor: float,
    sh_pt: List[float] | Tuple[float, float],
) -> Tuple[Optional[List[float]], float]:
    """
    Recovers an occluded glove strictly below the fitted mountain horizon (where zero clouds exist)
    using the bimodal black-leather (V < 55) + white-print/cuff (V > 158, S < 60) signature.
    """
    hsv = cv2.cvtColor(bgr_480, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    if side == "L":
        x0, x1 = 173, max(186, min(195, int(sh_pt[0] - 16.0)))
    else:
        x0, x1 = min(280, max(271, int(sh_pt[0] + 16.0))), 292
    y0 = max(116, int(round(y_hor_corridor + 6.0)))
    y1 = 174
    if y1 <= y0 + 8 or x1 <= x0 + 5:
        return None, 0.0

    sub_v = V[y0:y1, x0:x1]
    sub_s = S[y0:y1, x0:x1]
    white_mark = ((sub_v > 158) & (sub_s < 60)).astype(np.float32)
    black_glove = (sub_v < 55).astype(np.float32)

    w_blur = cv2.GaussianBlur(white_mark, (7, 7), 1.8)
    b_blur = cv2.GaussianBlur(black_glove, (9, 9), 2.2)
    score = w_blur * b_blur
    peak = float(np.max(score))
    if peak < 0.012:
        return None, peak
    best_idx = int(np.argmax(score))
    by, bx = divmod(best_idx, score.shape[1])
    return [float(x0 + bx), float(y0 + by)], peak


def compute_brake_pct_from_pulley(
    hand_xy: Tuple[float, float] | List[float],
    pulley_xy: Tuple[float, float] | List[float],
) -> int:
    ext = math.hypot(
        0.5 * float(hand_xy[0] - pulley_xy[0]),
        max(0.0, float(hand_xy[1] - pulley_xy[1])),
    )
    return int(round(100.0 * max(0.0, min(1.0, (ext - 6.0) / 60.0))))


def extract_pose_and_s2_single_frame(
    raw_bgr: np.ndarray,
    detect_anchor_fn: Callable[[np.ndarray], Dict[str, Any]],
    ik_2link_fn: Callable[[Tuple[float, float], Tuple[float, float], float, float, bool], Tuple[float, float]],
    fallback_energy_fn: Callable[..., Dict[str, Any]],
    force_full_ensemble: bool = False,
) -> Dict[str, Any]:
    h, w = raw_bgr.shape[:2]
    sx, sy = 480.0 / float(w), 270.0 / float(h)
    bgr_480 = (
        cv2.resize(raw_bgr, (480, 270), interpolation=cv2.INTER_AREA)
        if (w, h) != (480, 270)
        else raw_bgr
    )

    anch = detect_anchor_fn(raw_bgr)
    cbox = anch["canopy_box"]
    cx, cy, cw, ch = [float(v) for v in cbox]

    head_pt = [float(anch["px"]), float(anch["py"] - 22)]
    sh_l = [float(anch["px"] - 19), float(anch["py"] + 2)]
    sh_r = [float(anch["px"] + 21), float(anch["py"] + 2)]
    el_l = [float(anch["px"] - 39), float(anch["py"] - 10)]
    el_r = [float(anch["px"] + 41), float(anch["py"] - 10)]

    l_vis, r_vis = 0.0, 0.0
    l_hand: Optional[List[float]] = None
    r_hand: Optional[List[float]] = None

    pose_lm = get_cached_pose_landmarker()
    if pose_lm is not None:
        x0 = max(0, int(round(127.5 / sx)))
        y0 = max(0, int(round(41.25 / sy)))
        side = min(w - x0, h - y0, int(round(225.0 / sx)))
        raw_crop = raw_bgr[y0 : y0 + side, x0 : x0 + side]
        infer_crop = (
            cv2.resize(raw_crop, (512, 512), interpolation=cv2.INTER_CUBIC)
            if side < 480
            else raw_crop
        )
        clahe_crop = enhance_glare_bgr(infer_crop)

        raw_l: List[Tuple[float, List[float], List[float]]] = []
        raw_r: List[Tuple[float, List[float], List[float]]] = []
        torso_anchored = False

        def eval_variant(tag: str, img_var: np.ndarray, is_flipped: bool) -> None:
            nonlocal head_pt, sh_l, sh_r, torso_anchored
            rgb = cv2.cvtColor(img_var, cv2.COLOR_BGR2RGB)
            res = pose_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.pose_landmarks:
                return
            lm = res.pose_landmarks[0]

            def pt480(idx: int) -> List[float]:
                p = lm[idx]
                nx = (1.0 - p.x) if is_flipped else p.x
                return [(x0 + nx * side) * sx, (y0 + p.y * side) * sy, float(p.visibility)]

            if not is_flipped and (tag == "raw" or not torso_anchored):
                nose = pt480(0)
                head_pt = [nose[0], nose[1] - 4.0]
                sh_l = pt480(12)[:2]
                sh_r = pt480(11)[:2]
                torso_anchored = True

            idx_el_l, idx_wr_l, idx_in_l = (13, 15, 19) if is_flipped else (14, 16, 20)
            idx_el_r, idx_wr_r, idx_in_r = (14, 16, 20) if is_flipped else (13, 15, 19)
            e_l, w_l, i_l = pt480(idx_el_l), pt480(idx_wr_l), pt480(idx_in_l)
            e_r, w_r, i_r = pt480(idx_el_r), pt480(idx_wr_r), pt480(idx_in_r)

            lh_cand = [0.25 * w_l[0] + 0.75 * i_l[0], 0.25 * w_l[1] + 0.75 * i_l[1]]
            rh_cand = [0.25 * w_r[0] + 0.75 * i_r[0], 0.25 * w_r[1] + 0.75 * i_r[1]]
            lv_cand = 0.6 * w_l[2] + 0.4 * i_l[2]
            rv_cand = 0.6 * w_r[2] + 0.4 * i_r[2]

            if 72.0 <= lh_cand[1] <= 176.0 and 148.0 <= lh_cand[0] <= head_pt[0] + 4.0:
                raw_l.append((lv_cand, lh_cand, e_l[:2]))
            if 72.0 <= rh_cand[1] <= 176.0 and head_pt[0] - 4.0 <= rh_cand[0] <= 318.0:
                raw_r.append((rv_cand, rh_cand, e_r[:2]))

        eval_variant("raw", infer_crop, False)
        max_raw_l = max((t[0] for t in raw_l), default=0.0)
        max_raw_r = max((t[0] for t in raw_r), default=0.0)
        if force_full_ensemble or max_raw_l < 0.72 or max_raw_r < 0.72:
            eval_variant("clahe", clahe_crop, False)
            eval_variant("flip_clahe", cv2.flip(clahe_crop, 1), True)

    n_sky, y_hor_l, y_hor_r = detect_horizon_s2_480(bgr_480, cbox, head_pt, f_eq=90.0)

    carab_l = [sh_l[0] - 12.0, sh_l[1] + 6.0]
    carab_r = [sh_r[0] + 12.0, sh_r[1] + 6.0]
    riser_top_l = [cx - 0.33 * cw, cy + 0.50 * ch]
    riser_top_r = [cx + 0.25 * cw, cy + 0.50 * ch]
    pul_l = [
        0.26 * carab_l[0] + 0.74 * riser_top_l[0],
        0.26 * carab_l[1] + 0.74 * riser_top_l[1],
    ]
    pul_r = [
        0.26 * carab_r[0] + 0.74 * riser_top_r[0],
        0.26 * carab_r[1] + 0.74 * riser_top_r[1],
    ]

    hands_off_l = False
    hands_off_r = False
    if pose_lm is not None:
        if raw_l:
            bv_l, bh_l, bel_l = max(raw_l, key=lambda t: t[0])
            if bv_l >= 0.65 and (
                bh_l[0] > head_pt[0] - 10.0
                or (118.0 < bh_l[1] <= 168.0 and bh_l[0] > pul_l[0] + 9.5)
            ):
                hands_off_l = True
                l_hand = [pul_l[0], pul_l[1] + 4.0]
                l_vis = 0.85
                el_l = [0.5 * (sh_l[0] + l_hand[0]) - 8.0, max(sh_l[1], l_hand[1]) + 5.0]
            elif bh_l[0] <= head_pt[0] - 10.0:
                l_vis, l_hand, el_l = bv_l, bh_l, bel_l

        if raw_r:
            bv_r, bh_r, bel_r = max(raw_r, key=lambda t: t[0])
            if bv_r >= 0.65 and (
                bh_r[0] < head_pt[0] + 10.0
                or (118.0 < bh_r[1] <= 168.0 and bh_r[0] < pul_r[0] - 7.5)
            ):
                hands_off_r = True
                r_hand = [pul_r[0], pul_r[1] + 4.0]
                r_vis = 0.85
                el_r = [0.5 * (sh_r[0] + r_hand[0]) + 8.0, max(sh_r[1], r_hand[1]) + 5.0]
            elif bh_r[0] >= head_pt[0] + 10.0:
                r_vis, r_hand, el_r = bv_r, bh_r, bel_r

    if l_vis < 0.45 and not hands_off_l:
        gl_pt, _ = detect_below_horizon_glove_480(bgr_480, "L", y_hor_l, sh_l)
        if gl_pt is not None:
            l_hand = gl_pt
            l_vis = 0.65
            el_l = [0.5 * (sh_l[0] + l_hand[0]) - 8.0, max(sh_l[1], l_hand[1]) + 5.0]
    if r_vis < 0.45 and not hands_off_r:
        gr_pt, _ = detect_below_horizon_glove_480(bgr_480, "R", y_hor_r, sh_r)
        if gr_pt is not None:
            r_hand = gr_pt
            r_vis = 0.65
            el_r = [0.5 * (sh_r[0] + r_hand[0]) + 8.0, max(sh_r[1], r_hand[1]) + 5.0]

    if pose_lm is None and (l_hand is None or r_hand is None):
        pose_fb = fallback_energy_fn(
            bgr_480, px=anch["px"], py=anch["py"], cx=int(round(cx)), n_states=51
        )
        if l_hand is None:
            kl = int(np.argmax(pose_fb["energy_left"]))
            l_hand = [float(v) for v in pose_fb["hands_left"][kl]]
            el_l = [float(v) for v in pose_fb["elbows_left"][kl]]
            l_vis = 0.5
        if r_hand is None:
            kr = int(np.argmax(pose_fb["energy_right"]))
            r_hand = [float(v) for v in pose_fb["hands_right"][kr]]
            el_r = [float(v) for v in pose_fb["elbows_right"][kr]]
            r_vis = 0.5

    e_up = n_sky
    x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e_screen_right = x_cam - float(np.dot(x_cam, e_up)) * e_up
    e_screen_right = e_screen_right / max(1e-9, float(np.linalg.norm(e_screen_right)))
    e_aft = np.cross(e_up, e_screen_right)
    e_aft = e_aft / max(1e-9, float(np.linalg.norm(e_aft)))

    r_can = pixel_to_equidistant_ray_480(cx, cy, f_eq=90.0)
    pitch_raw = math.degrees(
        math.atan2(float(np.dot(r_can, e_aft)), float(np.dot(r_can, e_up)))
    )
    roll_raw = math.degrees(
        math.atan2(float(np.dot(r_can, e_screen_right)), float(np.dot(r_can, e_up)))
    )

    torso_cx = 0.55 * (0.5 * (sh_l[0] + sh_r[0])) + 0.45 * head_pt[0]
    torso_cy = 0.5 * (sh_l[1] + sh_r[1]) + 4.0

    return {
        "cbox": [int(round(v)) for v in cbox],
        "px": int(round(torso_cx)),
        "py": int(round(torso_cy)),
        "pw": 54,
        "ph": 74,
        "head": head_pt,
        "sh_l": sh_l,
        "sh_r": sh_r,
        "el_l": el_l,
        "el_r": el_r,
        "carab_l": carab_l,
        "carab_r": carab_r,
        "pul_l": pul_l,
        "pul_r": pul_r,
        "l_hand": l_hand,
        "r_hand": r_hand,
        "l_vis": l_vis,
        "r_vis": r_vis,
        "pitch_raw": pitch_raw,
        "roll_raw": roll_raw,
    }


def extract_pose_s2_sequence(
    frames_bgr: List[np.ndarray] | np.ndarray,
    detect_anchor_fn: Callable[[np.ndarray], Dict[str, Any]],
    ik_2link_fn: Callable[[Tuple[float, float], Tuple[float, float], float, float, bool], Tuple[float, float]],
    fallback_energy_fn: Callable[..., Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for extract_pose_s2_sequence.")
    n_frames = len(frames_bgr)
    if n_frames == 0:
        return []

    raw_seq = [
        extract_pose_and_s2_single_frame(
            f, detect_anchor_fn, ik_2link_fn, fallback_energy_fn, force_full_ensemble=False
        )
        for f in frames_bgr
    ]

    def fill_hand_track(
        side_key: str, vis_key: str, pul_key: str
    ) -> Tuple[List[float], List[float], List[bool]]:
        valid_idx = [
            i for i, d in enumerate(raw_seq) if d[side_key] is not None and d[vis_key] >= 0.25
        ]
        xs = [0.0] * n_frames
        ys = [0.0] * n_frames
        was_interp = [False] * n_frames
        if not valid_idx:
            for i, d in enumerate(raw_seq):
                xs[i] = float(d[pul_key][0])
                ys[i] = float(d[pul_key][1] + 4.0)
                was_interp[i] = True
            return xs, ys, was_interp

        for i in range(n_frames):
            d = raw_seq[i]
            if d[side_key] is not None and d[vis_key] >= 0.25:
                xs[i] = float(d[side_key][0])
                ys[i] = float(d[side_key][1])
            else:
                was_interp[i] = True
                prev_candidates = [j for j in valid_idx if j < i]
                next_candidates = [j for j in valid_idx if j > i]
                j0 = prev_candidates[-1] if prev_candidates else None
                j1 = next_candidates[0] if next_candidates else None
                if j0 is not None and j1 is not None and (j1 - j0) <= 8:
                    alpha = float(i - j0) / float(j1 - j0)
                    xs[i] = (1.0 - alpha) * raw_seq[j0][side_key][0] + alpha * raw_seq[j1][side_key][0]
                    ys[i] = (1.0 - alpha) * raw_seq[j0][side_key][1] + alpha * raw_seq[j1][side_key][1]
                elif j0 is not None and (i - j0) <= 4:
                    xs[i] = float(raw_seq[j0][side_key][0])
                    ys[i] = float(raw_seq[j0][side_key][1])
                elif j1 is not None and (j1 - i) <= 4:
                    xs[i] = float(raw_seq[j1][side_key][0])
                    ys[i] = float(raw_seq[j1][side_key][1])
                else:
                    xs[i] = float(d[pul_key][0])
                    ys[i] = float(d[pul_key][1] + 4.0)
        return xs, ys, was_interp

    lx_raw, ly_raw, l_interp = fill_hand_track("l_hand", "l_vis", "pul_l")
    rx_raw, ry_raw, r_interp = fill_hand_track("r_hand", "r_vis", "pul_r")

    def hampel3(arr: List[float], thresh: float = 9.0) -> List[float]:
        if len(arr) < 3:
            return list(arr)
        out = list(arr)
        for i in range(1, len(arr) - 1):
            m = float(np.median(arr[i - 1 : i + 2]))
            if abs(arr[i] - m) > thresh:
                out[i] = m
        return out

    def med3(arr: List[float]) -> List[float]:
        if len(arr) < 3:
            return list(arr)
        out = list(arr)
        for i in range(1, len(arr) - 1):
            out[i] = float(np.median(arr[i - 1 : i + 2]))
        return out

    lx_sm = hampel3(lx_raw)
    ly_sm = hampel3(ly_raw)
    rx_sm = hampel3(rx_raw)
    ry_sm = hampel3(ry_raw)
    cx_sm = med3([float(d["cbox"][0]) for d in raw_seq])
    cy_sm = med3([float(d["cbox"][1]) for d in raw_seq])
    cw_sm = med3([float(d["cbox"][2]) for d in raw_seq])
    ch_sm = med3([float(d["cbox"][3]) for d in raw_seq])
    px_sm = med3([float(d["px"]) for d in raw_seq])
    py_sm = med3([float(d["py"]) for d in raw_seq])
    pitch_sm = med3([float(d["pitch_raw"]) for d in raw_seq])
    roll_sm = med3([float(d["roll_raw"]) for d in raw_seq])

    brk_l_all: List[int] = []
    brk_r_all: List[int] = []
    for i in range(n_frames):
        brk_l_all.append(compute_brake_pct_from_pulley((lx_sm[i], ly_sm[i]), raw_seq[i]["pul_l"]))
        brk_r_all.append(compute_brake_pct_from_pulley((rx_sm[i], ry_sm[i]), raw_seq[i]["pul_r"]))

    trim_pitch = float(np.median(pitch_sm))
    trim_roll = float(np.median(roll_sm))

    results: List[Dict[str, Any]] = []
    for idx in range(n_frames):
        d = raw_seq[idx]
        cx = int(round(cx_sm[idx]))
        cy = int(round(cy_sm[idx]))
        cw = int(round(cw_sm[idx]))
        ch = int(round(ch_sm[idx]))
        px = int(round(px_sm[idx]))
        py = int(round(py_sm[idx]))
        pw, ph = d["pw"], d["ph"]

        lx = int(round(lx_sm[idx]))
        ly = int(round(ly_sm[idx]))
        rx = int(round(rx_sm[idx]))
        ry = int(round(ry_sm[idx]))

        sh_l = [int(round(d["sh_l"][0])), int(round(d["sh_l"][1]))]
        sh_r = [int(round(d["sh_r"][0])), int(round(d["sh_r"][1]))]
        pul_l = [int(round(d["pul_l"][0])), int(round(d["pul_l"][1]))]
        pul_r = [int(round(d["pul_r"][0])), int(round(d["pul_r"][1]))]
        carab_l = [int(round(d["carab_l"][0])), int(round(d["carab_l"][1]))]
        carab_r = [int(round(d["carab_r"][0])), int(round(d["carab_r"][1]))]

        if l_interp[idx] or abs(lx - lx_raw[idx]) > 4.0 or abs(ly - ly_raw[idx]) > 4.0:
            el_l = [int(round(0.5 * (sh_l[0] + lx) - 8.0)), int(round(max(sh_l[1], ly) + 5.0))]
        else:
            el_l = [int(round(d["el_l"][0])), int(round(d["el_l"][1]))]

        if r_interp[idx] or abs(rx - rx_raw[idx]) > 4.0 or abs(ry - ry_raw[idx]) > 4.0:
            el_r = [int(round(0.5 * (sh_r[0] + rx) + 8.0)), int(round(max(sh_r[1], ry) + 5.0))]
        else:
            el_r = [int(round(d["el_r"][0])), int(round(d["el_r"][1]))]

        brake_l_pct = brk_l_all[idx]
        brake_r_pct = brk_r_all[idx]
        pitch_deg = round(pitch_sm[idx] - trim_pitch, 2)
        roll_deg = round(roll_sm[idx] - trim_roll, 2)
        pitch_decideg = int(round(pitch_deg * 10.0))
        roll_decideg = int(round(roll_deg * 10.0))

        t22 = [
            cx, cy, cw, ch,
            lx, ly,
            rx, ry,
            px, py, pw, ph,
            pul_l[0], pul_l[1],
            pul_r[0], pul_r[1],
            brake_l_pct, brake_r_pct,
            el_l[0], el_l[1],
            el_r[0], el_r[1],
        ]
        results.append(
            {
                "canopy_box": [cx, cy, cw, ch],
                "left_hand": [lx, ly],
                "right_hand": [rx, ry],
                "pilot_box": [px, py, pw, ph],
                "pulley_left": pul_l,
                "pulley_right": pul_r,
                "carabiner_left": carab_l,
                "carabiner_right": carab_r,
                "brake_left_pct": brake_l_pct,
                "brake_right_pct": brake_r_pct,
                "elbow_left": el_l,
                "elbow_right": el_r,
                "shoulder_left": sh_l,
                "shoulder_right": sh_r,
                "head": [int(round(d["head"][0])), int(round(d["head"][1]))],
                "cam_knee": [158, 228],
                "u_left": round(brake_l_pct / 100.0, 2),
                "u_right": round(brake_r_pct / 100.0, 2),
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "tuple22": t22,
                "tuple24": t22 + [pitch_decideg, roll_decideg],
            }
        )
    return results
