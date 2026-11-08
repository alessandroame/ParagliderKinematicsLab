# WORKLOG.md — ParagliderKinematicsLab

## [2026-11-08] - Estrazione Scheletro Pilota "Una Tantum", Esclusione Video PiP e Rendering Overlay (`Step 1-4`)
- **Autore**: Alessandro Ame + Antigravity Agent
- **Obiettivo**: Sostituire il rilevamento solo-mani con l'intero scheletro articolato del pilota a 15 nodi (`pilot_skeleton_px`) salvato una tantum nei tracciati JSON e disegnato sull'overlay di `inspector.py` e `video_annotator.py`, escludendo il video PiP poiché non sempre sincronizzato con il video principale.
- **Attività svolte**:
  1. Esteso `FrameObservation` (`schema.py`) con `pilot_skeleton_px`, `pose_mode`, `shoulder_roll_deg`, `torso_lean_deg` e `pip_rect_px`.
  2. Implementato `detect_pilot_skeleton` e `_smooth_and_interpolate_skeleton_offline` (`vision_tracker.py`) con esclusione/mascheramento PiP (`ignore_pip=True`) e normalizzazione freni sulla larghezza bi-scapolare `dist(l_shoulder, r_shoulder)`.
  3. Aggiornati `video_annotator.py` (`draw_pilot_skeleton_overlay`) e `inspector.py` (`getInterpolatedSkeleton` + rendering completo scheletro su `canvas#flightCanvas`) e rigenerate le 7 tracce JSON e il bundle diagnostico.

## [2026-11-08] - Rimozione Forzanti Euristiche (`R6`) in `physics_replay.py` e Rigenerazione Tracce ODE
- **Autore**: Alessandro Ame + Antigravity Agent
- **Obiettivo**: Eliminare tutte le forzanti ad-hoc (`dive_nose_down_aoa_deg`, `low_speed_aft_upwash_deg`, `clean_forward_surge_n`, `forward_dive_thrust_n`, `ws_carve_coord`, `coord_bonus`, `uncoordinated_brake`) sostituendole con l'integrazione ODE stato-spazio a 2 corpi guidata dai soli input pilota `u(t)`.
- **Attività svolte**:
  1. Rimosso ogni termine euristico `R6` da `paraglider_kinematics/physics_replay.py`.
  2. Rigenerate le 7 tracce `data/traces/*.json` e verificata la suite `python3 -m unittest discover -s tests -v` (`2/2` test `OK`).

## [2026-11-08] - Web Flight Inspector, Playback Sincronizzato Video Originale e Automazione Pipeline
- **Autore**: Alessandro Ame + Antigravity Agent
- **Obiettivo**: Fornire output interattivo per valutare la qualità del lavoro (eventi e dati fisici in playback) con sincronizzazione del video originale e generazione automatica in ogni analisi.
- **Attività svolte**:
  1. Creato modulo `paraglider_kinematics.video_annotator` per esportare video MP4 con HUD telemetrico, centroidi, asse pendolare e mani pilota.
  2. Implementato `paraglider_kinematics.inspector` e dashboard SPA `paraglider_flight_inspector.html` con player 30 FPS, supporto Side-by-Side (video originale H.264 + canvas CV), cruscotto strumenti, timeline eventi e scorecard residui.
  3. Integrata l'esecuzione automatica di `build_and_export_inspector` in ogni analisi (`--run-theo-pitch-suite`, `--video`).
  4. Generati video raw H.264 per tutte le 7 manovre del tutorial di Theo de Blic in `videos/`.
  5. Aggiunti metodi `load_json` e `from_dict` in `schema.py`.
  6. Eseguiti e validati test unitari (`test_kinematics_pipeline.py`).
