# Experiment registry

This lists every experiment, the thesis claim it serves, and where its evidence lives. I added experiments here,
against a claim.  The claims are the proposal's predictions
P1–P5 and its Parts 1–3 (`docs/proposal/`); M stands for measurement validity. Each result file starts with its
experiment ID (`results/NAMING.md`). In the Analysis column, nbNN means `notebooks/analysis/NN_<IDs>_….ipynb`, and
each notebook also lists the IDs it covers.

| ID | Question | Serves | Status | Results | Analysis |
|---|---|---|---|---|---|
| **Part 1 · Measure and predict** | | | | | |
| D1 | How refusal decays over 2M tokens of code, and where the decay law bends | P1 | done | `erosion/D1_*` | nb07 |
| D2 | 0.8B, gentle regime, code vs chat-like data. Compares knowledge with refusal, and text with image inputs | P1 P2 P5 | done | `erosion/D2_*` | nb01 |
| D3 | Same design at 2B, gentle regime (matched fp16) | P2 (size) P5 | done | `erosion/D3_*` | nb01 nb02 |
| D4 | 0.8B, gentle regime, all four skills, with checkpoints saved | P1 | done | `erosion/D4_*` | nb02 |
| D5 | A second seed of D2 (chat-like and code) | P2 P5 (robustness) | done. Chat-like replicates; code does not | `erosion/D5_*` | nb01 |
| R1 | Do skills erode in the realistic regime (AdamW, fp32 master weights, lr 2e-5), code vs chat-like? | P1 (milestone) | done (partial) | `erosion/R1_*` | nb03 |
| R2 | Dose–response at lr 2e-6 and 5e-6: which behaviour breaks first? | P1 (ordering) P5 | done. All four dose predictions pass, and refusal erodes before skills | `erosion/R2_*` | nb03 |
| R3 | Does a first-order overlap score predict legal and news erosion before training? | P5 | done. Not supported (1 of 4) | `forecast/R3_*` | nb04 |
| M1 | Does the first-token refusal score match generated refusals? | M | done. Valid for text, not for images | `measurement/M1_*` | nb04 |
| **Multimodal (spans Parts 1–3)** | | | | | |
| E3/E4 | How text and image refusal compare: geometry, cross-modal ablation, held-out data, 2 sizes, 2 seeds | P2 P3 (static) | done | `multimodal/E3-E4_*`, `multimodal/E4_*` | nb05 |
| E5 | Values: splitting the image direction into a shared part and a picture-only part | P3 P4 | done | `multimodal/E5_*` | nb05 |
| E6 | Does the text direction reach natural images (VLSBench, real and generated)? | P3 (external validity) | done. 66% survives removal of the text direction | `multimodal/E6_*` | nb05 |
| **Part 2 · Aim the repair** | | | | | |
| W1 | Narrow replay of the model's own chat answers, fixed vs triggered, on code at lr 2e-5 | Part 2, P3 (dynamic) | done. Restores refusal but breaks tools and charts | `repair/W1_*` | nb03 |
| W2 | Replay on demand (calibrated sensors that route broad replay), compared with fixed broad replay, narrow replay (coverage), reference answers (consistency) and a merge sweep | Part 2 (core) | done 2026-09-27. R2, R3 and R5 pass; R1 and R6 fail (charts); R4 fails (tools 77%). Pre-registered (§7) | `repair/W2_*` | nb06 |
| **Part 3 · Harden and standardise (background for now)** | | | | | |
| A | Rank cost: how many directions carry refusal, and spectral vs functional rank | P4, Part 3 cost card | done | `background/A_*` | nb07 |
| C | Are the refusal and value directions real and distinct? (3 models, 2 seeds) | M, P4 | done | `background/C_*` | nb07 |
| **Motivation** | | | | | |
| B | Can a threshold alarm on raw refusal work? (No: the baseline drifts 4 SD.) | motivates Part 2 forecasting | done | `background/B_*` | nb07 |

superseded or mistaken work in `archive/`, with the reason for each.pre-registered predictions and their
outcomes are in `docs/preregistration.md`, and what each result means for the thesis is in `THESIS_LOG.md`.
