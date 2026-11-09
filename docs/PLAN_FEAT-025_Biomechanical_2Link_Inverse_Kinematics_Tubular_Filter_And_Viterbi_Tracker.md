# Piano Operativo FEAT-025: Riconoscimento Biomeccanico a 2 Segmenti Articolati (`L1, L2`), Filtro Tubolare Anti-Nuvole e Decodifica Globale di Viterbi (`360° Onboard CV`)

## 1. Obiettivo e Contesto
Superare i limiti del tracciamento locale per singolo blob (soggetto a falsi positivi su nuvole bianche, ombre sull'imbrago e bretelle verticali) nel riconoscimento dei comandi pilota dal video reale `videos/ykuaAYzB2qU.mp4` (`480x270`), introducendo in [`vision_tracker.py`](file:///d:/Github/ParagliderKinematicsLab/paraglider_kinematics/vision_tracker.py) un **estrattore biomeccanico a 2 passaggi** basato su:
1. **Calibrazione antropometrica prospettica su $N$ fotogrammi (`Pass 1`)** delle lunghezze ossee apparenti di braccio $L_1(u) = \lVert \mathbf{P}_{\text{el}} - \mathbf{P}_{\text{sh}} \rVert$ e avambraccio $L_2(u) = \lVert \mathbf{P}_{\text{hand}} - \mathbf{P}_{\text{el}} \rVert$ nel riferimento solidale al busto/casco $\mathbf{P}_{\text{torso}}(t)$.
2. **Cinematica Inversa a 2 Cerchi (`Spalla -> Gomito -> Mano`)** che riduce i 4 gradi di libertà rumorosi $(x_{\text{el}}, y_{\text{el}}, x_{\text{hand}}, y_{\text{hand}})$ a un **unico parametro biomeccanico di corsa freno $u \in [0, 1]$** per ciascun braccio ($u_L(t), u_R(t)$ indipendenti).
3. **Filtro Tubolare Orientato (`8–12 px`) + Rilevamento Guanto/Cavo (`Pass 2`)** lungo l'intera catena cinematica $\overline{\mathbf{P}_{\text{sh}}\,\mathbf{P}_{\text{el}}(u)} + \overline{\mathbf{P}_{\text{el}}(u)\,\mathbf{P}_{\text{hand}}(u)}$, immune sia alle nuvole bianche larghe (`> 40 px`) sia all'ombra grigia sul sotto-braccio.
4. **Programmazione Dinamica di Viterbi nel Tempo** per trovare la traiettoria ottima globale $\{u_L^*(t), u_R^*(t)\}_{t=0}^{T-1}$ rispettando i limiti fisici di velocità e accelerazione del braccio umano sotto carico (`10–45 N`).

---

## 2. Risultanze dell'Audit Empirico sui Frame Reali (`ykuaAYzB2qU.mp4`) e Requisiti Invarianti

| Insidia Rilevata sull'Audit Reale | Evidenza Misurata su `ykuaAYzB2qU.mp4` (`480x270`) | Vincolo Progettuale Obbligatorio (Zero Ricettoni) |
| :--- | :--- | :--- |
| **1. Nuvole bianche vs manica in ombra** | Il filtro acromatico `V > 165, S < 45` conta `1717 px` a `08:22` e `1858 px` a `08:30` su `y = [95..183]` perché cattura i banchi di nuvole bianche dietro al pilota; a `08:30` i 2/3 inferiori della manica sono in ombra (`V = 85..145`). | Vietato usare la sola luminosità assoluta del bianco: usare un **matched-filter tubolare orientato** di larghezza `8–12 px` con contrasto rispetto a due fasce esterne a $\pm 9\text{ px}$ ortogonali all'osso + esclusione del cielo azzurro (`B - R >= 16`). |
| **2. Sfondo cielo/montagne in movimento** | Durante le pendolate di beccheggio (`±35°`, es. `08:31`, `t = 12.0s`) l'orizzonte e le nuvole scorrono di `> 80 px`, mandando in saturazione qualsiasi sottrazione di sfondo statico `I_t - I_bg`. | Vietata la sottrazione di sfondo fisso sul cielo: tutte le coordinate $(\mathbf{P}_{\text{sh}}, \mathbf{P}_{\text{pul}}, \mathbf{P}_{\text{el}}, \mathbf{P}_{\text{hand}})$ sono ancorate frame-per-frame al centro dinamico del casco/busto $\mathbf{P}_{\text{torso}}(t)$. |
| **3. Scorcio prospettico 360° variabile con $u$** | Con camera 360° sul ginocchio sinistro (`ONBOARD_360_KNEE`), la mano che scende verso l'anca (`08:30`) si avvicina all'obiettivo: sul lato SX $L_2(u)$ cresce da `~38 px` (`u = 0`) a `~48 px` (`u = 1`). | Calibrare su $N$ fotogrammi i raggi prospettici agli estremi $(L_{1,0}, L_{2,0})$ a freno zero e $(L_{1,1}, L_{2,1})$ a fondo-corsa, interpolando $L_1(u) = (1-u)L_{1,0} + u L_{1,1}$ e $L_2(u) = (1-u)L_{2,0} + u L_{2,1}$. |
| **4. Asimmetria SX/DX e tagli di montaggio** | Nelle manovre `2A`, `2B`, `3A` un braccio è a `0%` e l'altro a `75%`; inoltre alcuni segmenti staccano su riprese esterne. | Eseguire due ottimizzazioni di Viterbi **indipendenti** per $u_L(t)$ e $u_R(t)$ su griglia di `51` stati (`u = 0.00 .. 1.00`, passo `0.02`) con gate automatico di presenza casco/imbrago 360°. |

---

## 3. Architettura Matematica dei 3 Moduli da Implementare

### Modulo 1 — Calibrazione Antropometrica Multi-Frame (`calibrate_360_arm_kinematics`)
Su una sequenza di $N$ fotogrammi `frames_bgr` (`480x270`):
1. **Tracciamento Busto/Casco e Carrucole ($\mathbf{P}_{\text{torso}}(t), \mathbf{P}_{\text{sh},L/R}(t), \mathbf{P}_{\text{pul},L/R}(t)$)**:
   - Individua il casco rosso nella fascia centrale `y ∈ [100, 140], x ∈ [208, 252]`, applica uno smoothing mediano temporale a 5 punti per ottenere il centro busto $\mathbf{P}_{\text{torso}}(t) = (p_x(t), p_y(t))$, le spalle $\mathbf{P}_{\text{sh},L}(t) = (p_x - 15, p_y - 10)$, $\mathbf{P}_{\text{sh},R}(t) = (p_x + 15, p_y - 10)$ e le carrucole $\mathbf{P}_{\text{pul},L}(t), \mathbf{P}_{\text{pul},R}(t)$.
2. **Stima Multi-Frame delle Lunghezze Prospettiche $(L_{1,0}, L_{2,0})$ e $(L_{1,1}, L_{2,1})$**:
   - Seleziona dal campione di $N$ fotogrammi il sottoinsieme di frame a **braccio raccolto** ($u \approx 0$, massima densità del guanto nero vicino alla carrucola `y ∈ [86, 114]`) e misura la distanza mediana spalla-gomito $L_{1,0}$ e gomito-mano $L_{2,0}$:
     - Lato SX: $L_{1,0}^L \approx 34\text{ px}, L_{2,0}^L \approx 39\text{ px}$; a fondo-corsa ($u = 1$): $L_{1,1}^L \approx 38\text{ px}, L_{2,1}^L \approx 48\text{ px}$.
     - Lato DX (scorciato prospetticamente dalla camera sul ginocchio SX): $L_{1,0}^R \approx 31\text{ px}, L_{2,0}^R \approx 36\text{ px}$; a fondo-corsa ($u = 1$): $L_{1,1}^R \approx 34\text{ px}, L_{2,1}^R \approx 42\text{ px}$.
3. **Legge Cinematica Inversa a 2 Cerchi ($\mathbf{P}_{\text{hand}}(u) \to \mathbf{P}_{\text{el}}(u)$)**:
   - Data la curva parametrica della mano lungo il ventaglio radiale 360° $\mathbf{P}_{\text{hand}}(u)$ (da $\mathbf{P}_{\text{pul}}$ per $u=0$ fino alla tasca bassa esterna per $u=1$):
     - Calcola $L_1(u) = (1-u)L_{1,0} + u L_{1,1}$ e $L_2(u) = (1-u)L_{2,0} + u L_{2,1}$.
     - Sia $\mathbf{d} = \mathbf{P}_{\text{hand}}(u) - \mathbf{P}_{\text{sh}}$ e $D = \lVert \mathbf{d} \rVert$ (limitato a $\max(|L_1 - L_2| + 1, \min(L_1 + L_2 - 1, D))$).
     - La proiezione lungo $\mathbf{d}$ è $a = \frac{L_1(u)^2 - L_2(u)^2 + D^2}{2 D}$ e l'altezza ortogonale (apertura del gomito verso il basso/esterno) è $h = \sqrt{\max(0, L_1(u)^2 - a^2)}$.
     - Il gomito $\mathbf{P}_{\text{el}}(u)$ è dato analiticamente da:
       $$\mathbf{P}_{\text{el}}(u) = \mathbf{P}_{\text{sh}} + \frac{a}{D}\mathbf{d} + \frac{h}{D}\mathbf{n}_{\text{out}}$$
       dove $\mathbf{n}_{\text{out}}$ è il versore normale orientato verso l'esterno/basso del pilota.

### Modulo 2 — Funzione di Energia Tubolare su Catena Articolata (`compute_arm_pose_energy`)
Per ogni fotogramma $t$ e per ogni stato discreto $u_k \in \{0.00, 0.02, \dots, 1.00\}$ (`51` stati):
1. **Risposta Tubolare dell'Omero $\overline{\mathbf{P}_{\text{sh}}\,\mathbf{P}_{\text{el}}(u_k)}$ e dell'Avambraccio $\overline{\mathbf{P}_{\text{el}}(u_k)\,\mathbf{P}_{\text{hand}}(u_k)}$**:
   - Campiona $M = 9$ punti lungo ciascuno dei due segmenti ossei. In ogni punto $\mathbf{x}_m$ con normale unitaria $\mathbf{n}_m$ al segmento:
     - Valuta la presenza di **foreground non-cielo** (`fg = (B - R < 16) | (V < 110)`) sull'asse centrale $\mathbf{x}_m$ rispetto ai due punti di controllo laterali $\mathbf{x}_m \pm r_{\text{tube}}\mathbf{n}_m$ ($r_{\text{tube}} = 9\text{ px}$).
     - Un segmento osseo reale (`8–12 px` di diametro contro il cielo) massimizza `fg(centro) - 0.5 * (fg(+9px) + fg(-9px))`, mentre una nuvola estesa (`fg = 1` ovunque) o il cielo aperto (`fg = 0` ovunque) danno contributo nullo o negativo.
2. **Risposta del Guanto Nero $\mathbf{P}_{\text{hand}}(u_k)$ e Continuità del Cavo Freno**:
   - Valuta la densità della maschera morfologica del guanto nero (`cv2.morphologyEx(V < 55, MORPH_OPEN, 3x5)`) nell'intorno $9 \times 9\text{ px}$ di $\mathbf{P}_{\text{hand}}(u_k)$, penalizzando i pixel statici dell'imbrago.
3. **Somma Pesata Senza Soglie Ad-Hoc**:
   $$\mathcal{E}_t(u_k) = w_{\text{forearm}}\,S_{\text{tube}}\!\left(\overline{\mathbf{P}_{\text{el}}(u_k)\,\mathbf{P}_{\text{hand}}(u_k)}\right) + w_{\text{upper}}\,S_{\text{tube}}\!\left(\overline{\mathbf{P}_{\text{sh}}\,\mathbf{P}_{\text{el}}(u_k)}\right) + w_{\text{glove}}\,S_{\text{glove}}\!\left(\mathbf{P}_{\text{hand}}(u_k)\right)$$

### Modulo 3 — Decodifica Globale di Viterbi (`solve_viterbi_brake_trajectory`)
Data la matrice delle energie $\mathcal{E}[t, k]$ di dimensione $T \times 51$ (a `4 Hz`, $\Delta t = 0.25\text{ s}$):
1. **Transizione Biomeccanica Ammissibile**:
   - In $\Delta t = 0.25\text{ s}$, la variazione massima fisicamente possibile della corsa freno è $|\Delta u| \le 0.24$ (`±12` livelli su `51`).
   - Costo di transizione quadratico: $C(k_{\text{prev}} \to k) = \lambda (u_k - u_{k_{\text{prev}}})^2$ per $|k - k_{\text{prev}}| \le 12$, altrimenti $+\infty$.
2. **Backtracking Ottimo Globale**:
   - Risolve con programmazione dinamica forward-backward la sequenza ottima $k^*_0, \dots, k^*_{T-1}$ indipendentemente per il braccio sinistro e per il braccio destro.
   - Da $u^*_t = k^*_t / 50.0$, ricava le coordinate esatte $(\mathbf{P}_{\text{hand}}(u^*_t), \mathbf{P}_{\text{el}}(u^*_t))$ e calcola la percentuale di freno **esclusivamente** dalla lunghezza del cavo estratto $\lVert \mathbf{P}_{\text{hand}}(u^*_t) - \mathbf{P}_{\text{pul}}(t) \rVert$.

---

## 4. File Coinvolti e Passi di Esecuzione
1. [`d:\Github\ParagliderKinematicsLab\paraglider_kinematics\vision_tracker.py`](file:///d:/Github/ParagliderKinematicsLab/paraglider_kinematics/vision_tracker.py):
   - Implementare `solve_inverse_kinematics_2link`, `compute_arm_pose_energy`, `solve_viterbi_brake_trajectory` e `extract_real_360_sequence_biokinematic` in `DualViewParagliderTracker`.
   - Aggiornare `extract_real_360_frame_features` affinché usi la cinematica inversa a 2 cerchi + filtro tubolare anche su singolo frame.
2. [`d:\Github\ParagliderKinematicsLab\scripts\extract_real_youtube_tracks.py`](file:///d:/Github/ParagliderKinematicsLab/scripts/extract_real_youtube_tracks.py):
   - Eseguire l'estrattore biomeccanico a 2 passaggi + Viterbi su tutte le 7 manovre di `videos/ykuaAYzB2qU.mp4` e rigenerare [`d:\Github\ParagliderSim\core\real-youtube-cv-tracks.mjs`](file:///d:/Github/ParagliderSim/core/real-youtube-cv-tracks.mjs).
3. **Verifiche e Collaudo**:
   - Verificare sui frame di controllo `08:22` (mani alte alle carrucole, `~0%` freno, gomito basso piegato a V) e `08:30` (frenata profonda simmetrica, `~95%` freno, braccio esteso lungo il ventaglio 360°) e sui frame intermedi `08:24–08:29` l'assenza di falsi positivi sulle nuvole bianche.
   - Eseguire `C:\Python310\python.exe -m unittest tests/test_kinematics_pipeline.py -v`, `node --test tests/web/cv-lab-sync.test.mjs` e `ui-smoke-runner.mjs` su `http://127.0.0.1:15480/cv-lab?mock=1`.
