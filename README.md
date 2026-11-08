# ParagliderKinematicsLab

**Frame-by-Frame Computer Vision Flight Kinematics Extractor & 500 Hz First-Principles Residual Laboratory for [`ParagliderSim`](https://github.com/alessandroame/ParagliderSim).**

Questo progetto analizza fotogramma per fotogramma i video di volo e tutorial SIV/Acro (a partire da [*BASIC PARAGLIDING EXERCISES - THEO DE BLIC's TUTORIALS*](https://www.youtube.com/watch?v=ykuaAYzB2qU), volato su **NOVA Mentor 7 Light S**) riconoscendo in doppia modalità (**Side-Chase** per le manovre longitudinali e **Rear-Chase** per le manovre latero-direzionali e chiusure d'estremità):
1. **Orizzonte e Riferimento Gravitazionale** ($\phi_{\text{horizon}}(t)$) tramite gradiente verticale di Sobel e regressione lineare robusta nella vista esterna.
2. **Calotta, Apertura Alare Utile (`Big Ears`) e Pilota Sospeso** ($(x_c, y_c), (x_p, y_p), b_{\text{open}}(t)$) tramite momenti centrali del 2° ordine ($\mu_{20}, \mu_{02}, \mu_{11}$), ricavando l'angolo di pendolo longitudinale $\theta(t)$, l'angolo di rollio $\phi(t)$, le velocità angolari $\dot{\theta}(t), \dot{\phi}(t), \dot{\psi}(t)$ e il grado di chiusura delle orecchie (`big_ears_norm`).
3. **Posizione delle Mani sui Freni e Spostamento Peso in Selletta** ($\delta_L(t), \delta_R(t), \dot{\delta}(t), w(t)$) nella vista Onboard POV rispetto ai riferimenti strutturali fissi (carrucola $= 0\%$, moschettone $= 100\%$).
4. **Confronto Chiuso a 500 Hz (System Identification senza "Ricettoni")**: inietta la curva di comando reale $\mathbf{u}_{\text{video}}(t)$ estratta dai fotogrammi nel modello fisico a 500 Hz e misura i residui $\mathbf{e}(t) = \mathbf{y}_{\text{sim}}(t) - \mathbf{y}_{\text{video}}(t)$ su tutte le **7 manovre del video** (`1A–1C` Pitch Pendulum, `2A–2B` Dynamic Turn, `3A–3B` Roll con Spostamento Peso e Big Ears).

---

## Manovre Analizzate (`ykuaAYzB2qU`)

| ID | Manovra | Timestamp | Camera Mode | File Tracciato JSON |
| :--- | :--- | :--- | :--- | :--- |
| **`1A`** | Pitch Pendulum — Single Pull & Uncaught Dive | `03:09–06:15` | `side_chase` | `data/traces/pitch_pendulum_uncaught.json` |
| **`1B`** | Pitch Pendulum — Double-Pump & Active Catch | `08:19–09:41` | `side_chase` | `data/traces/pitch_pendulum_active_catch.json` |
| **`1C`** | Pitch Mistake — Early Re-Braking Stall | `09:41–10:43` | `side_chase` | `data/traces/pitch_mistake_late_brake_stall.json` |
| **`2A`** | Dynamic Turn — Weight-Shift Preload + Carving Dive | `15:04–16:34` | `rear_chase` | `data/traces/dynamic_turn_coordinated.json` |
| **`2B`** | Dynamic Turn Mistake — Abrupt Brake Without Weight-Shift | `12:58–14:40` | `rear_chase` | `data/traces/dynamic_turn_mistake_no_ws.json` |
| **`3A`** | Roll Exercises — Resonant Weight-Shift & Inside Brake | `17:43–19:15` | `rear_chase` | `data/traces/rolls_weightshift_and_brake.json` |
| **`3B`** | Roll Exercises — Big Ears + Pure Harness Weight-Shift | `19:15–20:40` | `rear_chase` | `data/traces/rolls_big_ears_pure_ws.json` |

---

## Esecuzione Rapida

```bash
# Esegue l'estrazione CV fotogramma per fotogramma e il confronto a 500 Hz su tutte le 7 manovre
python3 -m paraglider_kinematics.cli --run-theo-pitch-suite

# Analizza un file video .mp4 personalizzato ed esporta il tracciato JSON per ParagliderSim
python3 -m paraglider_kinematics.cli --video path/to/flight_video.mp4 --out data/traces/custom_trace.json

# Esegue la suite di test automatici
python3 -m unittest discover -s tests -v
```
