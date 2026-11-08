# MEMORY.md — ParagliderKinematicsLab

## Progetto & Architettura
- **Scopo**: Laboratorio di estrazione cinematica frame-by-frame tramite Computer Vision (Dual-ROI Side/Rear Chase + Onboard POV) e calcolo residui fisici a 500 Hz rispetto a `ParagliderSim`.
- **Riferimento Volo**: Tutorial Acro/SIV di Theo de Blic su parapendio NOVA Mentor 7 Light S (70-95 kg, fascio 7.10 m).
- **Core Modules**:
  - `paraglider_kinematics.vision_tracker`: tracciatore CV per centroide calotta/pilota, asse pendolare gravitazionale (Sobel) ed estrazione una tantum dell'intero scheletro articolato del pilota (`detect_pilot_skeleton` a 15 nodi con catena `Spalla -> Gomito -> Polso -> Mano`, normalizzazione freni su larghezza bi-scapolare `dist(l_shoulder, r_shoulder)`, smoothing zero-phase ed esclusione totale del video PiP `ignore_pip=True`).
  - `paraglider_kinematics.physics_replay`: comparatore residui First-Principles mediante integrazione ODE stato-spazio a due corpi a 500 Hz guidata dai soli comandi `u(t)` (zero forzanti euristiche di picchiata o coordinazione).
  - `paraglider_kinematics.inspector`: costruttore automatico bundle diagnostico (`inspector_bundle.json`) e app SPA interattiva (`paraglider_flight_inspector.html`) con rendering scheletro completo interpolato su `time_sec`.
  - `paraglider_kinematics.video_annotator`: esportatore video MP4 e keyframe PNG con grafo scheletrico articolato (`draw_pilot_skeleton_overlay`), telemetria e overlay geometrico.
  - `paraglider_kinematics.cli`: interfaccia CLI unificata per suite tutorial, analisi video esterni (`--video`), esportazione video (`--export-video`) e inspector (`--inspector`).

## Convenzioni Tecniche & Linee Guida
- Formato tracciati: `data/traces/<manovra>.json` conforme a `ManeuverKinematicsTrace` (include `pilot_skeleton_px`, `pose_mode`, `shoulder_roll_deg`, `torso_lean_deg`).
- Esclusione PiP: eventuali riquadri Picture-in-Picture non sono mai usati per la telemetria (non garantiti sincroni col video principale) e vengono mascherati (`mask_pip_region`).
- Visualizzazione: Web Inspector con playback video sorgente H.264 sincronizzato e modalità Side-by-Side (Video Originale + CV Skeleton & Tracking Overlay).
