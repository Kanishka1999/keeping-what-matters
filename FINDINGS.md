# Findings: what the preliminary experiments show

*Keeping What Matters: predicting which post-trained behaviours continual training erodes, across text and image
inputs, and aiming the repair.* Every number here comes from `results/`, and each section names the notebook that
recomputes it (in `notebooks/analysis/`; the files are named by experiment ID, see `results/NAMING.md`). I recorded
what each result changed in `THESIS_LOG.md` (entries 1–28). `docs/preregistration.md` holds the predictions I fixed in
advance and their outcomes, and `EXPERIMENTS.md` maps every experiment to the claim it serves.

---

## In plain English

1. What post-training taught erodes first, and knowledge erodes last. On code, science-question accuracy stayed within
   8 of 300 items over 4M tokens while refusal wore away, and under realistic training the skills wore away too. On
   chat-like data at realistic rates knowledge does fall, but only after refusal and the skills. (I corrected this on
   27 Sep; the earlier version said knowledge barely moves in every run.)
2. There is an order: safety first, then skills, then knowledge. At the gentlest realistic rate, refusal had eroded
   before any skill moved, on both data types, and knowledge on code never moved. The order held at every rate I
   tested, and all four pre-registered checks passed.
3. What you train on matters more than how different it is. Chat-like text erodes refusal 2.5–6× faster than code, and
   legal and news text sit in between (about 6% per million tokens, 2.4× code). Legal text leaves image refusal
   untouched.
4. I could not yet predict the rate in advance. A score I computed before training ranked the known cases almost
   perfectly but got legal and news wrong (1 of 4 pre-registered tests passed). It tells dangerous data from mild data,
   but it cannot rank the mild corpora.
5. Skills erode in two ways. Chart reading and tool calling lose their output format (short answers, valid calls) and
   drop to zero fast, while maths keeps its format and loses accuracy.
6. Harmful photos are refused mostly through a route of their own. When I remove the internal refusal direction found
   from text, two thirds of photo refusal stays intact (a third for rendered text), and this holds for real photos.
7. So far, image refusal erodes as an inclination and not yet as behaviour. For text, my fast refusal score matches what
   the model writes. For images at 0.8B, the score falls during training but the model's written refusals don't.
8. Replay is not neutral. Replaying the model's own general-chat answers restored refusal (it even overshot) but wiped
   out tool calling and chart reading, which is worse than no replay; a replay burst of just 8 steps did it. At a
   gentler rate, chat-only replay cost maths (0.12 vs 0.42 with no replay). Broad replay protects tools, but on a fixed
   schedule it costs maths and still loses chart reading. Replaying reference answers instead of the model's own moves
   every skill further (pre-registered, passed).
9. When and how much you repair matters more than how much you replay. Replay fired on demand used a fifth of the
   replay of a fixed schedule and ended 19.5 points ahead across all skills, keeping all the domain learning
   (pre-registered, passed; one run). It fired once on an early refusal warning and once on drift sensors, and the
   warning fired a checkpoint before any skill fell.

---

## Where each prediction stands

| Claim | Status | Key evidence | Notebook |
|---|---|---|---|
| P1 Knowledge erodes last; post-trained behaviour first | Supported (corrected 27 Sep) | On code, ARC net loss is at most 8/300 over 4M tokens while refusal and skills fall (lr 2e-5: instruction −21 pts, maths −28, charts down to 0). On chat-like data at realistic rates, ARC falls (11–185/300 at 2M), after refusal and skills wherever the order is resolvable | 01 02 03 06 |
| Order: safety, then skills, then knowledge | Supported (pre-registered D1–D4, all pass) | At lr 2e-6, refusal −5.6% (code) and −17% (chat-like) by 262k tokens, before any skill; damage rises monotonically with rate | 03 |
| P2 Image-input behaviour erodes faster | Qualified | At 0.8B on chat-like data, 2× faster and replicated across seeds, but on the first-token measure only. Not at 2B, not replicated on code, and generated image refusals didn't fall | 01 04 |
| P3 Text-to-image transfer through a shared route | Static version supported, including real photos. Dynamic version refuted | Removing the text direction leaves 32–38% (rendered) and 66% (photos) of image refusal; text-only replay moved image refusal unpredictably (+26%, −28% of the loss) | 03 05 |
| P4 Constraint count sets spread | Suggestive only | Refusal is roughly one direction; values about 60% per direction; spectral rank is not functional rank | 05 07 |
| P5 Data overlap sets the rate, predictably | Not supported as a first-order forecast | Retrodiction 0.94; legal and news predictions 1 of 4; order wrong | 04 |
| Milestone: skills erode under realistic training | Reached | AdamW, full-precision master weights, lr 2e-6 to 2e-5 | 03 |
| Part 2: narrow replay | Harms skills at code lr 2e-5. At chat-like 5e-6 it costs maths, and on tools it is far below broad replay | On code, tools fell from 0.78 to 0.00–0.08. On chat-like data at 2M: maths 0.12 vs 0.42 with no replay; tools 0.34 (no replay 0.11, broad 0.75) | 03 06 |
| R1 Fixed broad replay keeps tools and charts | Fail (charts) | Tools 0.85 / 0.75 at 1M / 2M; charts 0.00 / 0.11 | 06 |
| R2 On-demand matches fixed with at most half the replay | Pass, and superior (one run) | 19% of the replay; +19.5 pts on 476 items (one-sided lower bound +15.5) | 06 |
| R3 Refusal canary warns before skills fall | Pass | Crossed at 131k tokens; first skill drop at 262k | 06 |
| R4 Merge at α = 0.5 restores tools and charts to at least 80% | Fail (tools 77%) | Charts 99%; 16% of the domain gain given back | 06 |
| R5 Own answers preserve, reference answers shift | Pass | Reference answers move skills 17.3 pts more (paired 95% +13.4 to +21.0) | 06 |
| R6 Broad beats chat-only replay on tools and charts | Fail (charts) | Tools +34 / +41 pts at 1M / 2M; charts lost in both arms | 06 |

---

## The findings in detail

### Knowledge vs alignment, gentle regime (notebook 01)
I trained the 0.8B and 2B models with SGD and pure 16-bit weights for 1M tokens. At 0.8B, chat-like data eroded
wrapped-text refusal by 14% (seed 1: 14.5%) and image refusal by 30% (29%). Code eroded them by 2.5% / 3.5% (seed 1:
5.7% / 4.7%, so code's rate depends on the sample). At 2B, with matched precision, chat-like data gave 7.6% / 2.4% and
code 2.3% / 2.7%. My laptop and the Kaggle T4 agree to within 1%.

### Realistic regime and the order of erosion (notebook 03)
Here I used AdamW with full-precision master weights and 16k-token batches.

| at 2M tokens | lr 2e-6 | lr 5e-6 | lr 2e-5 |
|---|---|---|---|
| code: wrapped refusal lost | 6.8% | 9.2% | 18.5% |
| chat-like: wrapped refusal lost | 18.3% | 40.5% | 94.1% |

The order shows at 2e-6. Refusal crossed 5% by 262k tokens. On code, no skill had fallen by 2M; on chat-like data, the
skills first fell at 1M. Knowledge on code stayed within 1 point at 2e-6 and 5e-6.

Skill loss comes in two kinds. Chart accuracy tracks chart format (short answers went from 0.98 to 0.04–0.07), and tool
accuracy tracks call validity (from 0.86 to 0.16–0.20). Maths is different: its format rises (from 0.41 to 0.6–0.8)
while its accuracy falls.

With Adam, the strength of training goes with learning rate times number of steps (lr × steps), not with tokens. Small
batches take far more steps per token.

### Prediction test (notebook 04)
I computed a first-order overlap score (the gradient alignment between a corpus and each refusal framing) on the
untrained model

The score ranks the six known rates (Spearman 0.94).

### Measurement validity (notebook 04)
For text, the first-token refusal score predicts written refusals with AUROC 0.97–1.00, and it falls with them during
training.

For images the AUROC is 0.80 at baseline, and the two decouple during training: the gap falls by up to 32% while
written refusals of harmful images stay at about 38%.

### Cross-modal routes (notebook 05)
With rendered text (2B, 2 seeds, held-out; also 0.8B), removing the text-derived direction leaves 32–38% of image
refusal.

With harmful photos (2B), the photo direction sits 78° from the text direction. Removing the text direction leaves 66%
of photo refusal (real 53%, generated 85%). Removing the photo direction removes 89% of it, but only a third of text
refusal. Random directions are inert, and ability is intact.

### Replay pilot (notebook 03)
This was on code at lr 2e-5. The replay data were the model's own answers to 192 general chat prompts.

| at 1M tokens | no replay | fixed (31% of batches) | triggered (one 8-step burst) |
|---|---|---|---|
| tool calling | 0.78 | 0.00 | 0.08 |
| instruction following | 0.36 | 0.22 | 0.20 |
| maths | 0.10 | 0.35 | 0.07 |
| text / image refusal kept | 80% / 69% | 102% / 76% | 107% / 115% |

The triggered arm matched the no-replay run exactly until its single burst, so I can attribute the tool collapse to
replay. The generic sensor was blind: the loss on the model's own chat answers fell while refusal and skills eroded.

### Replay on demand, W2 (notebook 06)
I trained on chat-like data with AdamW at lr 5e-6 for 2M tokens, with the same data order in every arm. The replay pool
covers five behaviour families (chat, instruction following, tools, maths, charts) and uses the model's own answers,
except in A3, which uses reference answers.

| at 2M tokens (baseline) | A0 none | A1 fixed broad | A2 on demand | A3 fixed, reference | A4 fixed, chat only | merge α = 0.5 |
|---|---|---|---|---|---|---|
| replay tokens (share of A1) | 0 | 100% | 19% | 87% | 94% | 0 |
| instruction (0.58) | 0.24 | 0.51 | 0.57 | 0.28 | 0.31 | 0.48 |
| maths (0.38) | 0.42 | 0.21 | 0.40 | 0.05 | 0.12 | 0.21 |
| tools (0.69) | 0.11 | 0.75 | 0.83 | 0.95 | 0.34 | 0.53 |
| charts (0.70) | 0.03 | 0.11 | 0.63 | 0.00 | 0.00 | 0.69 |
| text refusal kept | 0.60 | 1.43 | 1.46 | 0.56 | 2.23 | 0.88 |
| ARC (0.86) | 0.73 | 0.88 | 0.87 | 0.89 | 0.88 | 0.86 |
| domain learning kept | 100% | 91% | 100% | 92% | 96% | 84% |

The causal case is clean here too: A2 equals A0 in every logged value until its first trigger at 131k tokens.

Two bursts were enough: the canary burst (131k, all families) and a chat + instruction burst (459k). There was no
replay after about 0.5M tokens. In A0 the loss on the model's own answers rises most between 131k and 507k; in A2 it
stays flat for the remaining 1.5M tokens.

Fixed replay has side effects. Maths is 21 points below no replay, and chart answers stay short but wrong.

There are caveats. This is one run of one 0.8B model. Tool scores above baseline (A2, A3) are partly template
learning. I report image refusal but do not interpret it.

### Background experiments: A, B, C and D1 (notebook 07)
This notebook covers the rank cost (spectral rank is not functional rank), the drift-alarm negative (a threshold on raw
refusal cannot be calibrated), value geometry, and the first scaling run.

---

## How far from "validated"?

A PhD proposal needs a real, measurable problem, predictions that can fail, and a credible path to the method. Here is
where I stand on each.

| | Status |
|---|---|
| The problem (post-training erodes, in an order, before knowledge) | Backed: 2 sizes, 4 data types, 4 regimes, 2 seeds; the pre-registered order checks pass |
| Multimodal angle | Backed statically, including real photos. Dynamic image erosion is backed only as first-token inclination |
| Measurement you can trust | Backed for text (generation check, held-out re-fits, gates, cross-platform agreement). Not yet for images |
| Predictions that can fail | Shown, repeatedly. P2 failed at 2B, P5's forecast failed, P3's dynamic version failed, and I revised the proposal each time |
| Repair (Part 2) | First pre-registered positive result: on-demand replay beat fixed replay with a fifth of the replay (one run). Fixed replay, reference answers, narrow replay and merging each have measured side effects. Not yet replicated on a second data order |

Overall, the measurement half is well backed, with a pre-registered order result and a real-photo multimodal result.
The prediction half is negative for a first-order forecast, which is itself a research question, but an early warning
did fire before the skills went. The repair half has a pre-registered positive result from one run, plus the measured
costs of the alternatives. The open gaps I would name are:
- an image refusal judge that reads generated text;
- 4B/9B scale;
- W2 on a second data order, and a burst-timing test (early vs late repair).

---

## Validity checks used throughout
- Gates (legibility, reproducibility, separability, behaviour present, random direction inert, capability intact after
  every ablation) and held-out re-fits.
- A generation check of the refusal measure.
- Pre-registration with timestamps.
- Reproducibility: identical runs reproduce exactly, the laptop and Kaggle agree within 1%, and I ran second seeds.
- Numerics: I verified the fast kernels and chose fp16 by capability; the emulated-bf16 run is flagged and excluded.
