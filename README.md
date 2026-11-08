# ParagliderKinematicsLab

**Frame-by-Frame Computer Vision Flight Kinematics Extractor & 500 Hz First-Principles Residual Laboratory for [`ParagliderSim`](https://github.com/alessandroame/ParagliderSim).**

Questo progetto analizza fotogramma per fotogramma i video di volo e tutorial SIV/Acro (a partire da [*BASIC PARAGLIDING EXERCISES - THEO DE BLIC's TUTORIALS*](https://www.youtube.com/watch?v=ykuaAYzB2qU), volato su **NOVA Mentor 7 Light S**) riconoscendo:
1. **Orizzonte e Riferimento Gravitazionale** ($\phi_{\text{horizon}}(t)$) tramite gradiente verticale di Sobel e regressione lineare robusta nella vista esterna.
2. **Calotta e Pilota Sospeso** ($(x_c, y_c), (x_p, y_p)$) tramite momenti centrali del 2° ordine ($\mu_{20}, \mu_{02}, \mu_{11}$), ricavando l'angolo di pendolo longitudinale $\theta(t)$, la velocità angolare $\dot{\theta}(t)$ e l'avanzamento metrico $\Delta x_{\text{surge}}(t) = -L \sin\theta(t)$.
3. **Posizione delle Mani sui Freni e Spostamento Peso** ($\delta_L(t), \delta_R(t), \dot{\delta}(t), w(t)$) nella vista Onboard POV rispetto ai riferimenti strutturali fissi (carrucola $= 0\%$, moschettone $= 100\%$).
4. **Confronto Chiuso a 500 Hz (System Identification senza "Ricettoni")**: inietta la curva di comando reale $\mathbf{u}_{\text{video}}(t)$ estratta dai fotogrammi nel modello fisico a 500 Hz e misura i residui $\mathbf{e}(t) = \mathbf{y}_{\text{sim}}(t) - \mathbf{y}_{\text{video}}(t)$ per individuare quali termini fisici delle equazioni differenziali in `ParagliderSim.Core` necessitano correzione.

---

## Struttura del Repository

```text
ParagliderKinematicsLab/
├── paraglider_kinematics/
│   ├── schema.py                 # Contratto JSON (ManeuverKinematicsTrace, FrameObservation, PhaseLagMetrics)
│   ├── vision_tracker.py         # Riconoscitore CV Dual-ROI (Orizzonte, Calotta, Pilota, Mani/Freni, Eventi)
│   ├── physics_replay.py         # Simulatore 500 Hz (CurrentParagliderSim vs FirstPrinciplesTwoBody)
│   ├── theo_de_blic_dataset.py   # Generatore/Estrattore suite Pitch Pendulum & Active Catch (00:53-10:43)
│   └── cli.py                    # Entrypoint CLI (pg-kinematics)
├── data/
│   ├── traces/                   # Tracciati cinematici JSON standardizzati a 30 Hz per ParagliderSim
│   └── diagnostics/              # Report dei residui fisici e keyframe annotati con HUD Computer Vision
└── tests/
    └── test_kinematics_pipeline.py
```

---

## Esecuzione Rapida

```bash
# Esegue l'estrazione CV fotogramma per fotogramma e il confronto a 500 Hz sulla suite di Théo de Blic
python3 -m paraglider_kinematics.cli --run-theo-pitch-suite

# Analizza un file video .mp4 personalizzato ed esporta il tracciato JSON per ParagliderSim
python3 -m paraglider_kinematics.cli --video path/to/flight_video.mp4 --out data/traces/custom_trace.json

# Esegue la suite di test automatici
python3 -m unittest discover -s tests -v
```
