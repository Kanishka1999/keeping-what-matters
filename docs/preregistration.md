# Pre-registered predictions

**Written 2026-09-25 15:45 BST, before any of the results below existed.** At this time the Kaggle
strength/replay run had just started, the legal/news runs, generation check, second seed and natural-image test had
not run, and `results/d_trajectories/e2/` was empty. Predictions are not edited after results arrive. Outcomes are
appended at the end with the date they were checked.

Notation: *loss* = % of the starting refusal gap lost; channels are **plain** text, **wrapped** text and **image**
(the request rendered as an image, same wrapper). 0.8B = Qwen3.5-0.8B, vision tower frozen.

---

## 1 · P5: a first-order overlap score predicts erosion rates before training
**Score.** One SGD step on corpus *D* changes the refusal gap by −lr·(∇gap · ḡ_D). The score
s[D][c] = (∇gap_c · ḡ_D) / gap_c is therefore the predicted relative erosion per unit learning rate for channel *c*.
It is computed on the untrained model with full precision: ∇gap_c over the 64 measurement pairs, and ḡ_D averaged over 64
training blocks (65k tokens) of exactly the data the training run will see (`e1s_worker.py --score-only`).

**Known rates (calibration)** — 0.8B, gentle regime, 1,003,520 tokens (`results/d_trajectories/0.8B_laptop_gentle_*`):

| corpus | plain | wrapped | image |
|---|---|---|---|
| code | 0.912 | 2.479 | 3.472 |
| chat-like | 5.786 | 13.965 | 29.870 |

**Rule (fixed now).**
1. *Retrodiction check:* Spearman ρ between s and the six known losses. The score is considered valid if ρ ≥ 0.8
   and all six scores are positive. Predictions are reported either way, but are flagged if this check fails.
2. *Calibration:* k = Σ(loss·s) / Σ(s²) over the six known points (a line through the origin).
3. *Prediction:* losŝ[D][c] = k·s[D][c] for **legal** (US case law) and **news** (CNN/DailyMail), and the rank
   order of the four corpora by wrapped-text loss.
4. *Platform check:* the chat-like run is repeated on Colab. Let f = mean(Colab ÷ laptop loss, wrapped and image).
   If |f − 1| > 0.25, predictions are also reported multiplied by f. The uncorrected version is the primary test.
5. A corpus whose wrapped-text score has a 95% interval including zero is flagged *uninformative*.

**Success criteria** (legal and news × wrapped and image = 4 tests). A test passes if 0.5 ≤ measured/predicted ≤ 2, or
|measured − predicted| ≤ 1.5 points. P5 is **supported** if ≥ 3 of 4 pass and the four-corpus rank order by wrapped
loss is correct; **partially supported** if 2 of 4 pass; **not supported** otherwise. The plain channel is
reported but not tested (its values are small and noisy).

**Also checked with the same scores (P2):** whether s[image]/s[wrapped] reproduces the known image-to-text ratios
at 0.8B (1.40 on code, 2.14 on chat-like), counting a ratio as reproduced if it is within a factor of 1.5.

The numerical predictions are produced by the Colab notebook (`notebooks/cloud/colab_predict_legal_news_0.8B.ipynb`)
and written to `predictions_0.8B.json`, with a timestamp, **before** its first training step.

## 2 · P3 (dynamic): text-only replay repairs image refusal less than text refusal
From the Kaggle replay pilot (0.8B, chat-like data, lr 1e-4, 8M tokens). Restoration of channel *c* by a replay arm:
R_c = (kept_c[replay] − kept_c[none]) / (1 − kept_c[none]) at the final checkpoint.
- *Testable only if* the no-replay arm lost ≥ 5% on both wrapped and image, and R_wrapped ≥ 0.2. The replay data
  are benign general prompts, not refusal data, so replay may not restore refusal at all.
- **Prediction:** R_image / R_wrapped ≈ 0.67 (at 0.8B, removing the text-derived direction leaves 33% of image
  refusal, so about two thirds is reachable from text). **Pass** if ≤ 0.85; **refuted** if ≥ 1.0. Assessed
  separately for fixed and triggered replay.

## 3 · Milestone: skills erode under realistic training
From Kaggle Part A (AdamW, full-precision master weights, 0.5M–8M tokens).
- At lr 1e-4 on chat-like data, at least one skill falls by more than 1.96 standard errors of its baseline by 8M
  tokens. At lr 2e-5, fewer skills fall than at 1e-4, on both kinds of data.
- Chat-like data erodes wrapped-text refusal more than code at the same learning rate (P5 consistency).
- ARC text stays within 5 points of baseline in all four Part A runs (P1).
- In arms that lose ≥ 3% wrapped-text refusal by 8M, erosion is log-linear over 0.5M–8M (R² ≥ 0.9). This tests
  whether the law bends (THESIS_LOG entry 21).
- Triggered replay uses fewer replay tokens than fixed replay, and ends within 2 points of fixed on every skill.

## 4 · Second seed of the 0.8B chat-like run
The same gentle run with a different sample of training data (seed 1: a different stream offset); the measurement
items are unchanged. At 1M tokens: image ÷ wrapped loss ≥ 1.4 (seed 0: 2.14), and wrapped loss within ±50% of
13.97%.

## 5 · Generation check of the refusal measure
On the base model and saved checkpoints, greedy responses (48 new tokens) to the same requests, classified as
refusals by a fixed keyword list.
- At baseline, the first-token refusal log-odds predicts the generated refusal flag across harmful and harmless
  items with AUROC ≥ 0.8, in each channel where both classes occur.
- Across checkpoints, the harmful-item generated refusal rate moves in the same direction as the first-token gap,
  whenever the gap moved by ≥ 5%.

## 6 · Natural images (VLSBench, 2B, no training)
Harmful requests where the harm is in a real or generated photo, with a neutral text query. The matched text
version is the dataset's image description followed by the same query. The harmless control is benign photos with
neutral queries. The explicit-content category is excluded.
- Gate: natural-image refusal is present at baseline (harmful−harmless gap > 0, separability d ≥ 0.5). Otherwise
  the test is reported as untestable.
- **Prediction:** removing the text-derived refusal direction leaves ≥ 30% of natural-image refusal (for rendered
  text it leaves 35% at 2B): natural images are at least as separate from the text route as rendered text is.
- *Design detail fixed 2026-09-25 16:20 BST, before running:* the 160 harmful items are 80 generated images and 80
  real photos (VLSBench stores them by source, so each half is drawn at random from its own shard). Results by
  source are reported descriptively; the prediction applies to all 160.

---

## Outcomes
*(appended as results arrive; nothing above is edited)*

### 2026-09-25 — Kaggle Part A, partial (lr 2e-5 only; run stopped at 4M code / 2M chat-like of 8M)
- **3a** (skills fall at lr 1e-4, chat-like): *not testable* — lr 1e-4 not run. At lr 2e-5 every skill fell > 1.96 SE
  on chat-like data, and three of four on code.
- **3b** (fewer skills fall at 2e-5 than at 1e-4): *not testable* — lr 1e-4 not run.
- **3c** (chat-like erodes wrapped refusal more than code at the same lr): **PASS** at 2e-5 (0.5M: 89.8% vs 25.3%;
  2M: 94.1% vs 18.5%).
- **3d** (ARC text within 5 points in all Part A runs): **FAIL** — code within 2.3 points, chat-like −58 to −62 points.
  Caveat: ARC is read as a letter choice, which collapsed with answering itself (THESIS_LOG entry 23).
- **3e** (log-linear erosion 0.5M–8M, R² ≥ 0.9): **FAIL** on the partial data — code 0.5–4M R² 0.00 (a jump in the
  first ~30 steps, then a plateau); chat-like saturated at ~95% (R² 0.58).
- **3f** and **section 2** (replay): *not run yet*.
- *Design change after seeing these results:* the lr 1e-4 jobs are dropped as uninformative (certain collapse). Any
  replacement runs and their predictions are added below with their own timestamp; they are not pre-registered in
  the sense of section 1.

### Addendum 2026-09-25 17:49 BST — design changes after the lr 2e-5 results, before any of the runs below
- **Section 1 (P5):** Colab's GPU quota ran out before the notebook started. The identical test (same worker, commands,
  rule and known values) now runs inside the next Kaggle notebook on a T4. The platform check compares the Kaggle
  chat-like anchor with the laptop run.
- **Sections 2 and 3f (replay):** moved from chat-like data at lr 1e-4 (certain collapse) to code data at lr 2e-5, where
  damage without replay is partial (THESIS_LOG entry 23). The no-replay reference is the existing code lr 2e-5 run (same
  settings and data order). Predictions and pass rules are unchanged. The comparison uses the checkpoints that run has
  (0.5M, 1M, 2M, 4M tokens; skills at 1M).
- **New predictions** (dose–response: lr 2e-6 and 5e-6, both data types, to 2M tokens). These were specified after
  seeing the lr 2e-5 results, but before these runs:
  - **D1:** at each rate, chat-like data erodes wrapped-text refusal more than code does (at 2M tokens).
  - **D2:** damage rises with the rate. For each data type, wrapped-refusal loss at 2M tokens is ordered
    2e-6 < 5e-6 < 2e-5 (the 2e-5 value comes from the existing runs).
  - **D3 (ordering, the thesis claim):** at 2e-6, refusal erodes before skills. Among the skill-read checkpoints (250k,
    1M, 2M), the first at which wrapped refusal has lost ≥ 5% comes no later than the first at which any skill falls by
    more than 1.96 SE. Assessed per data type, and only if at least one of the two happens by 2M.
  - **D4:** on code data, ARC text stays within 5 points of baseline at 2e-6 and 5e-6.

### 2026-09-25 23:24 BST — Kaggle `kaggle_next_0.8B` run, partial (provisional: computed from 3-decimal log values; confirm from JSON)
- **Section 1 (P5 prediction test): NOT SUPPORTED.** Retrodiction valid (Spearman 0.94 over the six known rates, all
  scores positive; k = 0.198). Platform check f = 0.99 (Kaggle chat-like anchor matches the laptop), so no correction.
  Legal wrapped: predicted 2.98%, measured 6.1% (fail; ratio 2.05). Legal image: predicted −1.4%, measured 0.2% (fail
  by 0.12 points on the absolute rule). News wrapped: predicted 3.5%, measured 6.2% (**pass**). News image: predicted
  7.6%, measured −1.6% (fail). Order by wrapped loss: predicted legal < news < code < chat-like; measured
  code < legal < news < chat-like (wrong). 1 of 4 tests pass.
- **P2 via scores:** the image/wrapped score ratio reproduces chat-like (2.30 vs 2.14) and, just inside the factor-1.5
  rule, code (0.95 vs 1.40).
- **Section 2 (P3, revised condition), triggered replay: REFUTED.** At 4M, restoration text +0.56, image +1.25
  (ratio 2.24 ≥ 1.0): text-only replay restored image refusal more than text refusal. Fixed replay: pending (4M).

### 2026-09-26 10:15 BST — laptop results (final, from JSON)
- **Section 4 (second seed, chat-like): PASS.** Seed 1 at 1M: wrapped 14.5% (seed 0: 13.97%), image 28.7% (29.9%),
  image ÷ wrapped 1.98 (≥ 1.4). *Not pre-registered, reported for balance:* the code arm does **not** replicate
  its image effect. Seed 1: wrapped 5.7% (seed 0: 2.5%), image 4.7% (3.5%), ratio 0.84 (seed 0: 1.40).
- **Section 5a (generation check, baseline AUROC ≥ 0.8 per channel): PASS** — plain 0.974, wrapped 0.996, image 0.803
  (just above the bar).
- **Section 5b (trajectory agreement where the gap moved ≥ 5%): PASS for text, FAIL for images.** Text (plain and
  wrapped): 4 of 5 same direction, 1 no change. Image: 1 of 6 same direction, 3 opposite, 2 no change. The image
  first-token gap fell 18–32% on chat-like data, while the generated refusal rate on harmful images stayed at
  0.36–0.41 (baseline 0.38), and image AUROC fell from 0.80 to 0.62.
- **Section 6 (natural images, 2B): gate PASS, prediction PASS.** Natural-image refusal is present (gap 2.51,
  d 1.38). Removing the text-derived direction leaves **66%** of it (≥ 30%). By source: real photos 53%, generated
  85%. Random direction inert; ARC intact.

### 2026-09-26 10:22 BST — Kaggle `kaggle_next_0.8B`, final (from JSON; supersedes the provisional entry above)
- **Section 1 (P5): NOT SUPPORTED — confirmed.** Legal wrapped 6.10% vs predicted 2.98% (fail); legal image 0.20% vs
  −1.42% (fail, |diff| 1.63 > 1.5); news wrapped 6.23% vs 3.45% (pass); news image −1.59% vs 7.63% (fail). Order
  wrong. Retrodiction valid (0.94); platform factor 0.994.
- **Section 2 (P3, revised condition) at 4M.** Triggered: restoration text +0.56, image +1.26, ratio 2.24 → **REFUTED**.
  Fixed: text +1.72, image −0.28, ratio −0.16 → **passes by the letter of the rule only**. Image refusal was made
  *worse*, not partially restored, so the mechanism the prediction assumed (partial transfer through the shared
  route) is contradicted in both arms.
- **3f: FAIL.** Triggered used 65,536 replay tokens vs 1,254,400, but ended 20 points below fixed on maths and 8 below on
  tools (4M).
- **D1: PASS** (2M, wrapped lost): 2e-6 code 6.8% vs chat-like 18.3%; 5e-6 code 9.2% vs chat-like 40.5%.
- **D2: PASS**: code 6.8 < 9.2 < 18.5%; chat-like 18.3 < 40.5 < 94.1%.
- **D3: PASS** in both data types. At 2e-6, refusal lost ≥ 5% by 262k tokens; no skill fell > 1.96 SE by 2M on code,
  and the first skill drop on chat-like came at 1M.
- **D4: PASS**: code ARC change ≤ 1.0 point at both rates.

---

## 7 · W2: replay on demand with a broad, Qwen-consistent mix (written 2026-09-26 10:55 BST, before any W2 run)
**Design** (`docs/design/REPLAY_ON_DEMAND.md`, worker `w2_worker.py`). Qwen3.5-0.8B, chat-like data, AdamW lr 5e-6,
2M tokens, skills read at 262k / 1M / 2M. The replay pool has five families (chat, instruction following, tools,
maths, charts). It holds 200 training prompts plus 16 probe prompts per family, taken from public training splits
disjoint from every test item. Answers are the model's own greedy answers before training, stored as raw token
ids, with loss on the answer only. Arms:
- **A0:** no replay (end weights saved).
- **A1:** fixed broad replay, 5 of 16 micro-batches, families in rotation.
- **A2:** on demand. Every 4 steps a canary checks the refusal gap on 16 held-out pairs, and each family's loss on its
  probe answers is checked too. A burst fires on a canary drop > 5% or a family rise > 3%: 8 steps at 50% replay,
  routed to the families that fired (all families if only the canary fired). Total replay is capped at A1's.
- **A3:** control. As A1, but with reference answers instead of the model's own.
- **M:** merge sweep on A0's end state, α ∈ {0, 0.25, 0.5, 0.75}.

Reference (existing no-replay run at this setting): at 1M, tools 0.11, charts 0.02, instruction 0.30, wrapped
refusal kept 0.65.
- **R1:** A1 keeps tool calling ≥ 0.50 and chart reading ≥ 0.50 at both 1M and 2M, and wrapped refusal kept ≥ 0.85.
- **R2:** A2 uses ≤ 50% of A1's replay tokens and ends (2M) within 5 points of A1 on every skill.
- **R3:** in A0, the canary crosses its 5% threshold at an earlier checkpoint than the first checkpoint where any skill
  falls by more than 1.96 SE.
- **R4:** merging at α = 0.5 restores tool calling and chart reading to ≥ 80% of baseline while giving back at most
  half of the domain gain. Domain gain = 1 − held-out chat-loss ratio at α = 0.
- **R5 (consistency):** A3 moves skills further from baseline than A1: mean absolute change across the four skills at
  2M is at least 5 points larger. Own answers preserve behaviour; reference answers shift it.
- Image refusal and knowledge are reported descriptively (P3's dynamic version was already refuted).

### Addendum to §7, 2026-09-26 11:22 BST — design fixes from a pre-mortem, before any W2 run
Found by reviewing how W2 could fail or be inconclusive; nothing has run.
- **Trigger.** The refusal canary is now a *one-shot* early warning: its first crossing (drop > 5%) fires one broad
  burst, then it is disarmed. Replay cannot repair refusal (the pool has no refusal data), so a level trigger would
  fire forever. After that, only the per-family sensors trigger replay, and those are repairable.
- **Thresholds calibrated on A0 by a fixed rule** (A0 finishes before A2 starts). Family *f*'s threshold is half of
  its sensor rise (loss on the model's own probe answers) at the first A0 skill read where *f*'s skill had fallen by
  more than 1.96 SE (chat maps to instruction following). If the skill never fell, or its sensor did not rise, that
  family never triggers. Floor 1%.
- **New arm A4:** fixed replay of the chat family only, with A1's budget and loss, in the same regime. It tests coverage
  directly.
- **Extra skill read at 125k tokens** in every arm.
- **Pool quality gate** (nothing trains if it fails): tool answers parse as a call ≥ 0.6; chart answers are short
  (≤ 3 words) ≥ 0.8; maths answers contain "Answer:" ≥ 0.25; chat and instruction answers ≥ 5 tokens ≥ 0.9.
- **Per-item scoring.** Every skill read now records right/wrong per test item. The same items and identical
  baselines are used in every arm, so between-arm comparisons use paired bootstrap over items (10,000 resamples).

**Revised and added predictions (replacing R2 and R5 above):**
- **R1:** unchanged. Matched-domain-token reading: A1 at 2M has seen ~1.38M domain tokens, fewer than A0's 2M but more
  than its 1M. If A1 exceeds R1's bars while A0 is below them at *both* 1M and 2M, fewer domain tokens cannot explain it.
- **R2 (non-inferiority):** A2 uses ≤ 50% of A1's replay tokens, **and** for every skill the paired difference
  A2 − A1 at 2M has a one-sided 95% lower bound above −5 points.
- **R3:** unchanged, now with skill reads at 125k / 262k / 1M / 2M.
- **R4:** unchanged.
- **R5 (consistency):** mean absolute change from baseline across the four skills at 2M is at least 5 points larger for
  A3 than for A1, with the paired 95% interval of the difference excluding 0.
- **R6 (coverage, new):** at 1M and 2M, A1 beats A4 on tool calling and on chart reading by ≥ 20 points each, with
  paired 95% intervals excluding 0.

### Second addendum to §7, 2026-09-26 11:25 BST — R2 power fix, before any W2 run
A dry run of the analysis on simulated data showed R2 as written is nearly unpassable even for genuinely equal arms.
It requires four separate per-skill non-inferiority tests at a 5-point margin, each on 96–160 items, with a per-skill
paired one-sided bound of about ±6–7 points. **R2 now reads:** A2 uses ≤ 50% of A1's replay tokens, and on the
**pooled score over all 476 test items** (instruction 160, maths 120, tools 96, charts 100, every item weighted
equally), the paired difference A2 − A1 at 2M has a one-sided 95% lower bound above −5 points. Per-skill differences
are reported descriptively.

### Third addendum to §7, 2026-09-26 12:43 BST — replay-pool fixes after inspecting a real pool, before any W2 run
A pool built on the laptop (real Qwen3.5-0.8B, `inspect_pool.py`) passed every gate check: chat and instruction
non-empty 1.00, maths "Answer:" 0.46, charts short 0.99, tools parseable 0.94. Qwen's own answers carry its own error
rate (maths 38% correct, tools 71% exactly right), as behaviour-preserving replay should. Fixes it prompted:
- **Pool answer caps raised to 512 tokens** for chat, instruction and maths. The laptop pool cut off 79% / 38% / 58% of
  answers at 256 / 384 / 320. The *test* generation caps are unchanged.
- **No truncation of replay examples.** Tool prompts are ~1,175 tokens (12 tool definitions), and a 1,024-token example
  ceiling would have cut tool answers to 8 tokens. The ceiling is now a 2,048-token safety check.
- **Reference answers (A3) get the same length budget** as the model's own answers, so A1 and A3 differ only in answer
  source.
- **End-of-turn is appended only to answers that actually ended**, so cut-off answers do not teach stopping mid-answer.

### Fourth addendum to §7, 2026-09-26 16:56 BST — first W2 attempt stopped by the Kaggle quota; implementation fix for the rerun
The first attempt built and gated the pool (all five checks passed; maths "Answer:" line 0.87 with the 512-token
cap). A0 and A1 then reached 1M tokens before the weekly GPU quota ended the session. **No prediction is assessed on
that partial run.** The rerun reuses the identical saved pool and repeats every arm from the start.
- **Implementation fix (maths unchanged).** Replay and probe examples are right-padded to a multiple of 256 tokens,
  with padding masked in both the labels and the attention mask. In the first attempt A1's host memory grew about
  1 GB per checkpoint (13.4 → 16.6 GB by 1M), because each new sequence length compiles and caches new GPU kernels;
  the no-replay arm, with fixed shapes, stayed flat. Padding leaves a handful of shapes.
- **Merge sweep** keeps the end-state weights on the GPU rather than in system RAM (about 3.4 GB less RAM). Same
  computation.


### Outcomes for §7 (W2), assessed 2026-09-27 from the complete run
All five arms reached 2,015,232 tokens and the merge sweep has four rows. The pool is the one from the first attempt:
sha256 prefix `034fe58e28faf8fc`, checked in the rerun notebook. Analysis: `notebooks/analysis/07_replay_on_demand.ipynb`.

**Analysis-code corrections.** Both were made after the result files arrived, before any W2 number was computed.
Neither changes a verdict.
1. R5's interval had been coded as an *unpaired* bootstrap (2,000 resamples, against a fixed baseline mean). It is now
   the registered *paired* bootstrap over items: 10,000 resamples, with the same item indices applied to the baseline,
   A1 and A3. The original code gives +17.3 (95% +12.9 to +21.3), the same verdict.
2. R1's refusal bar is now read at both 1M and 2M, like its skill bars; the code had checked 2M only. R1 fails on
   charts under either reading.

| | Verdict | Numbers |
|---|---|---|
| R1 | **FAIL** (charts) | A1 at 1M / 2M: tools 0.85 / 0.75 (pass); charts 0.00 / 0.11 (fail); wrapped refusal kept 1.18 / 1.43 (pass). A0 tools 0.11 / 0.11: the tool protection is not dilution |
| R2 | **PASS** | A2 replay 133,163 tokens = 19% of A1's 690,724. Pooled 476 items, A2 − A1 = **+19.5 pts** (one-sided 95% lower bound +15.5). Per skill, descriptive: instruction +6.2 [−1.9, +14.4], maths +19.2 [+8.3, +30.0], tools +8.3 [+1.0, +15.6], charts +52.0 [+41.0, +62.0] |
| R3 | **PASS** | Canary crossed 5% at 131,072 tokens. First skill drop > 1.96 SE (tools) at 262,144 |
| R4 | **FAIL** (tools) | α = 0.5: tools 77% of baseline (51 vs 66 of 96 items), charts 99%. Gives back 16% of the domain gain |
| R5 | **PASS** | Mean absolute change at 2M: A3 39.7 vs A1 22.4 pts. Difference +17.3 (paired 95% +13.4 to +21.0) |
| R6 | **FAIL** (charts) | A1 − A4 tools: +34.4 [+25.0, +43.8] at 1M, +40.6 [+31.2, +51.0] at 2M (pass). Charts: +0.0 at 1M (both arms at 0), +11.0 [+5.0, +17.0] at 2M (fail) |

**Notes (not verdicts).**
- **Superior, not just non-inferior.** R2 was framed as non-inferiority, and A2 came out superior. Every logged value
  in A2 equals A0's through 131k tokens, the first trigger, so the divergence is caused by the replay bursts within
  this trajectory. Whether an advantage this large holds across data orders is untested; this is one run.
- **Where A2's replay came from.** Two bursts: the one-shot canary at 131k (all families), then chat + instruction at
  459k. The tools, maths and charts thresholds (0.83, 0.41, 5.76) were never reached, and A2 had no replay after about
  0.5M tokens.
- **Descriptive by design.**
  - Image refusal kept at 2M: A0 0.53, A1 0.64, A2 2.05, A3 0.45, A4 1.61. Not interpreted, because the image measure
    is not validated over training.
  - Knowledge (ARC): A0 fell from 0.86 to 0.73; every replay arm stayed at 0.87–0.89.

### Note, 2026-09-27 — file names (no prediction or outcome changed)
Result files and notebooks cited above keep the names they had when each section was written. All result files
have since been renamed by experiment ID; `results/NAMING.md` maps every new name to its original.
