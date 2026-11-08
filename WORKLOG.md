# WORKLOG.md — ParagliderKinematicsLab

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
