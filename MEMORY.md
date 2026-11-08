# MEMORY.md — ParagliderKinematicsLab

## Progetto & Architettura
- **Scopo**: Laboratorio di estrazione cinematica frame-by-frame tramite Computer Vision (Dual-ROI Side/Rear Chase + Onboard POV) e calcolo residui fisici a 500 Hz rispetto a `ParagliderSim`.
- **Riferimento Volo**: Tutorial Acro/SIV di Theo de Blic su parapendio NOVA Mentor 7 Light S (70-95 kg, fascio 7.10 m).
- **Core Modules**:
  - `paraglider_kinematics.vision_tracker`: tracciatore CV robusto per centroide calotta/pilota, asse pendolare gravitazionale (Sobel), mani pilota su carrucole.
  - `paraglider_kinematics.physics_replay`: comparatore residui First-Principles a due corpi (braccio aerodinamico 0.25c, accoppiamento surge-incidenza, diedro alare reale).
  - `paraglider_kinematics.inspector`: costruttore automatico bundle diagnostico (`inspector_bundle.json`) e app SPA interattiva (`paraglider_flight_inspector.html`).
  - `paraglider_kinematics.video_annotator`: esportatore video MP4 con telemetria e overlay geometrico impressi a fuoco.
  - `paraglider_kinematics.cli`: interfaccia CLI unificata per suite tutorial, analisi video esterni (`--video`), esportazione video (`--export-video`) e inspector (`--inspector`).

## Convenzioni Tecniche & Linee Guida
- Formato tracciati: `data/traces/<manovra>.json` conforme a schema Pydantic/dataclass `ManeuverKinematicsTrace`.
- Diagnostica: `data/diagnostics/keyframes/*.png` per fotogrammi chiave CV con overlay.
- Visualizzazione: Web Inspector con playback video sorgente H.264 sincronizzato e modalità Side-by-Side (Video Originale + CV Tracking Overlay).
