# Thesis log: what I found, and what it changed

> **File names (27 Sep 2026).** The entries cite result files and notebooks by the names they had when I wrote them.
> I have since renamed every result file by experiment ID (`results/NAMING.md` maps each new name to its original),
> and I renumbered the analysis notebooks in thesis order (each name lists the experiment IDs it covers). The
> proposal's work packages WP1–WP3 are now called Parts 1–3 (measure and predict; aim the repair; harden and
> standardise).

> The early entries use names from before the first reorganisation. Where those files live now:
>
> - `scaling_ood.json` is `results/erosion/D1_decay-law_Qwen3.5-0.8B_code_gentle_2M-tokens.json`
> - `e1_local_0.8B_{ood,benign}.json` are `results/erosion/D2_erosion_Qwen3.5-0.8B_{code,chat}_gentle_seed0.json`
> - `e1_2B_kaggle_all.json` is `results/erosion/flagged/D3_erosion_Qwen3.5-2B_code_gentle_BF16-SUSPECT.json`
> - `e1s_2B_kaggle_fp16_{ood,benign}.json` are `results/erosion/D3_erosion_Qwen3.5-2B_{code,chat}_gentle.json`
> - `A_rank_cost*`, `B_sensor*` and `C_geometry*` are in `results/background/`, and `E_*` in `results/multimodal/`
> - `MULTIMODAL_LIT.md` is `docs/literature.md`, and the design notes are in `docs/design/`
>
> On arm names: "ood" is the code arm, and "benign" is the chat-like arm (ultrachat user turns).

I add one entry per result: what I found, how I know it is valid, and the change it forced in the thesis. Numbers
come from `results/` unless I note otherwise. The newest entry is last.

**1. Capability holds; alignment erodes.** (`scaling_ood.json`)

I continually trained Qwen3.5-0.8B on 1M tokens of code at LR 1e-4. ARC stayed flat at 0.832–0.848 (net ±4 of 500),
while refusal fell along a log law: refusal drop% = 0.728·log₂(tokens) − 10.93, R² = 0.96. As for validity, this is
one arm; the benign control is E1.

So the thesis centres on post-trained behaviour rather than capability.

**2. A drift alarm on raw alignment levels cannot work.** (`B_sensor`)

The baseline drifted 4.05 SD over 84 reads, and zero thresholds both catch damage and stay quiet on the control.

This changed replay on demand. It becomes forecast, then correct on residuals, in place of threshold detection.

**3. The implicit-bias theory does not hold here.** (`ABLATION_RESISTANCE.md` §2.1)

Its theorems need linear models, plain gradient descent, no weight decay and infinite training time.

So I moved the theory spine from the optimiser to the objective: the coverage hypothesis.

**4. Spectral rank is not functional rank.** (`A_rank_cost`)

Qwen3.5-2B has stable rank 8.4, yet one ablation removes 68.5% of its refusal. Qwen3-1.7B has stable rank 1.2, and
one ablation removes 71%.

That means P4 has to be stated in functional terms, and I now state it that way.

**5. Thomson measured no vision.** (report text; Hugging Face configs)

Both base models are vision–language models (27-layer vision tower). Thomson ran no vision evaluation, and uploads go
through OCR. Nowhere is it stated that the vision tower was removed.

So multimodal retention becomes the thesis's distinctive chapter.

**6. The literature disagrees on text-to-vision transfer.** (`MULTIMODAL_LIT.md`)

References [4, 11, 12] in that file report transfer, and [3, 5] there report divergence.

That is why I introduced P3.

**7. E3: image refusal is weaker and partly separate.** (`E_crossmodal__Qwen3.5-2B__seed0`)

I rendered 160 matched harmful/harmless pairs as images. On validity, the images are legible (ARC 0.84 image vs 0.92
text) and all directions are reproducible (split-half 9–10°, floor 89.9°). The image refusal gap is 5.74, against 8.09
for matched text (71%). The text and image directions are 39.9° apart, yet each separates the other modality's inputs
about equally well (d 3.64 vs 3.50).

This confirmed P2's starting condition. It also means P3 must be defined functionally, because the angle overstates
the gap.

**8. E4: cross-modal ablation is asymmetric.** (same file)

When I remove the text-derived direction, 5% of text refusal is kept and 35% of image refusal. When I remove the
image-derived direction, text keeps 0% and image 6%. A random direction keeps 99% / 102%. ARC is unchanged (text
0.91–0.93, image 0.85–0.87), and all 7 gates pass.

This quantifies P3: ~65% of image refusal runs on the shared route, so text-only replay should protect at most ~65% of
it. That is a testable WP2 prediction.

WP3 also gains a preliminary result. Text-derived ablation (the basis of abliteration, and of text-driven realignment)
is incomplete on image inputs, while the image-derived direction covers both.

It also gave me a method rule for the whole thesis: judge directions by what removing them does, not by geometry
(angle, spectral rank).

The limits are one model, one seed and one behaviour, and the images are rendered text rather than natural images.

Next: E1/E2 on Colab (P1 control; P2 erosion rate). After that, I will run E3/E4 on natural-image requests, a second
seed, and Qwen3.5-0.8B / 4B.

**9. Replication across size and seed.** (`E_crossmodal__Qwen3.5-2B__seed1`, `__Qwen3.5-0.8B__seed0`)

All gates pass in both runs. Some results hold at both sizes and both seeds. Text-derived ablation leaves 33–35% of
image refusal, the random direction is inert, ARC is unaffected, and each direction separates the other modality's
inputs nearly as well as its own (2B: d 3.64 vs 3.50; 0.8B: 2.07 vs 2.51).

Other results depend on scale. Image refusal is 71% of text at 2B but 34% at 0.8B, where harmful image requests lean
toward compliance on average (margin −1.08). The image-derived direction removes all text refusal at 2B but only about
two thirds at 0.8B. The directions are 40° apart at 2B and 51° at 0.8B.

One validity caveat: the two seeds draw 160 of the same 200 items, so the seed replication is only partial.

So I now scope the "asymmetry" to 2B. The claim that holds across both sizes and seeds is the one-third image-specific
residual.

This raised a new question. Image-input alignment strengthens and converges with text alignment as models grow, yet
the residual stays near a third. Two sizes are not a trend; 4B is the third point.

It also changed the E1/E2 design. At 0.8B image refusal has little headroom. The notebook still passes its gate, but
P2 is better tested at 2B.

I also made a consistency fix. The E1/E2 notebook now scores refusal with the same token lists as every local pilot
(it had drifted by two words).

**10. E4 on contested values: the image-specific gap is refusal-specific.** (`E_crossmodal_value__Qwen3.5-2B__seed0`, 340 items)

The behaviour here is adopting the side a prompt frames as consensus, read as an A/B logit margin. Legibility is fine,
and the directions are reproducible (text 13–14°, image 30°). Image value behaviour is 49% of text. Removing the
text-derived direction keeps 34% (text) and 36% (image). A random direction keeps 93% / 92%, so ~8% is this
behaviour's perturbation noise floor.

One row is invalid. Removing the image-derived direction cut ARC by 15 points (0.92 to 0.77 text, 0.84 to 0.69
image), so its effect is general damage, not a targeted one. The capability gate had only checked the text-direction
ablation; I changed it to check every ablation in both modalities. When I re-checked, all refusal results still pass
and only this row fails.

That means the one-third image-specific residual holds for refusal, not for contested values. For values the text
direction is equally (in)complete in both modalities.

Value behaviour is also less single-directional. One direction removes ~two thirds even in text (for refusal it is
95%). This is consistent with P4 (17 issues ≈ many constraints; refusal ≈ one), but it is observational only, since
the two behaviours differ in more than constraint count.

I revised the realignment point. For values the incompleteness is not image-specific: a single direction leaves about
a third in either modality. Thomson may ablate several directions and follows with DPO, so this bears on
single-direction ablation only.

It also changed my method. Capability must be checked for every intervention, not just the one I expect to be safe.

**11. The discarded row is itself a finding.** (same file; reinterpretation, no new data)

Removing the image-derived value direction cost 15 ARC points in both modalities. The text-derived value direction
cost 2–3, and the image-derived refusal direction cost nothing. So the value signal extracted from image inputs is
entangled with general ability, and only for values. It is still not evidence about targeted value removal (a damaged
model narrows any gap), but it does change the thesis.

(a) Attack cost depends on which modality the direction is extracted from, so the WP3 cost card must report cost per
extraction modality. (b) It is a naturally occurring case of alignment entangled with capability, which is the
property tamper-resistance work tries to engineer on purpose. (c) It is a direct test bed for Fisher-routed ablation,
whose whole purpose is to cut this cost.

The caveats: this is one run, and the image value direction was the noisiest I measured (split-half 30°).

The follow-ups, all on the laptop, are to replicate on seed 1, to split the image direction into its text-shared and
image-specific parts and ablate each, and to build a Fisher-routed version using `fisher.py` / `edits.py`.

**12. Split test: the image-specific value route is clean; the damage lives in the combination.** (`E_split_value__Qwen3.5-2B__seed0`)

The checks pass. The random direction is inert, the layer matches (16), and the full image-derived direction
reproduces the prior run exactly (value kept 31%/20%, ARC 0.77/0.69). The image direction is 75% along the text
direction and 66% orthogonal to it. The orthogonal ("picture-only") part has ~0 cosine with the image-minus-text
modality offset, so it is not just "this is an image", and it still separates image inputs (d 2.13).

Removing the picture-only part keeps value at 89% text / 65% image, with ARC at 0.92/0.82, so it does no damage.
Removing both parts together keeps 26% / 20%, with ARC at 0.76/0.71. That is the 15-point damage.

This corrects entry 10. Equal leftover fractions after text-direction removal did not mean "no image-specific route",
and I had that wrong. Values have a clean image-specific route too (a third of image-input behaviour). Both behaviours
I have tested now show one, which is a more general claim than before.

It also corrects entry 11. The capability entanglement is in the combination of the shared and picture-only parts,
not in the image-specific part.

For WP3, stripping values from image inputs has a price curve. The first third is free, and the rest costs ~15 ARC
points. This is the shape the cost card should report.

The limits are one model, one seed and one value behaviour, and I found and tested the directions on the same items
(a held-out re-run is pending).

**13. E1/E2 at 0.8B (laptop, both arms): the data matters more than the domain; image refusal erodes ~2× faster.** (`e1_local_0.8B_*.json`)

I trained 1M tokens per arm with SGD+momentum at LR 1e-4, in fp16 as on the T4 runs. All gates passed (on this run I
set the gates to record rather than halt, and none fired), and one fp16 step was skipped. At baseline, ARC is 0.860
text / 0.780 image, and the refusal gap is 5.79 text and 1.79 image (31%).

In the code arm, refusal kept 97.5% (text) / 96.5% (image) at 1M and ARC was unchanged. The decay slopes were 0.39 vs
0.65 %/doubling (R² 0.91 / 0.78).

In the "benign" chat arm (ultrachat user turns as plain text), refusal kept 86% / 70%. ARC moved −8 of 300 items
(≈2.7 pts), and ARC-image went from 0.78 to 0.75. The slopes were 2.19 vs 4.29 %/doubling (R² 0.86 / 0.92).

The image/text relative loss ratio was 1.4–2.8 at every checkpoint past 32k, in both arms.

So the control is not a control. I had planned the chat arm as one, but chat-like text erodes refusal ~5–6× faster
than code. What the model is trained on matters far more than distance from its original domain. This is consistent
with benign fine-tuning degrading safety (Qi et al. 2024). For alignment, "benign" training does not exist; there is
only more or less damaging data.

P1 holds in both arms. Competence moves ≤3 pts while refusal loses up to 14% (text) / 30% (image).

P2 is supported in relative terms: image-input refusal loses about twice the share of its margin. In absolute logits
it loses less, because it starts smaller at 0.8B. The 2B run (where image refusal starts at 71% of text) separates
the two readings.

The log law holds in both arms, but its slope depends on the data. That means forecasting for replay on demand must be
done per data type.

For Thomson, this means code was the mild case. Legal and news prose is natural language, likely closer to chat, so
WP1 needs a legal/news arm. Code and chat bracket the likely range.

Some caveats. This is one model and one seed. I read refusal from the first response token; the harmful-minus-harmless
gap cancels generic first-token shifts, but generation-based checks should confirm it. Binding (120 items, baseline
14-item difference) is too noisy at 0.8B to draw a conclusion, though it falls in both arms, more on chat. The
notebook's printed "excess erosion" and P2 line assumed a clean control and are misleading here; the per-arm
comparison above is the valid reading.

**14. Held-out re-runs: the ablation findings survive.** (`*_heldout.json`)

I fitted the directions on half the pairs and measured all effects on the other half. For refusal at 2B,
text-direction ablation leaves 38% of image refusal (in-sample 35%), the image direction still removes both, and all
gates pass. For refusal at 0.8B the figure is 32% (in-sample 33%), and all gates pass. For values at 2B, the text
direction removes ~60% in both modalities (in-sample ~65%). The image-derived value direction does not reproduce at
170 fitting items (split-half 45.8°), and it again damages capability.

So in-sample inflation was at most ~4 points, and entries 7–9 stand. Value directions need >170 items.

**15. E1/E2 at 2B (Kaggle), code arm: P2 holds in absolute terms too.** (`results/e1_2B_kaggle_all.json`)
I completed the code arm to 1M tokens. The chat arm was cut off at 16k by the session limit (nothing had moved yet).
Baseline image refusal was 69% of text. From 131k tokens on, image-input refusal lost 1.8–1.9× the share of text
(4.4% vs 2.3% at 1M). Unlike at 0.8B, it also lost more in absolute logits (0.24 vs 0.18), because it starts large
enough. Decay slopes were 0.49 vs 0.25 %/doubling (R² 0.80 / 0.85). ARC was unchanged (net +2 items), ARC-image went
from 0.82 to 0.85, and binding stayed flat within noise.

So P2 is supported at 2B in relative and absolute terms for the code arm. The 0.8B caveat (small starting margin)
does not explain the effect.

Scale did not protect either: 2B lost the same text share as 0.8B (2.3% vs 2.5%) and a larger image share (4.4%
vs 3.5%). That is two sizes, one arm each. It means the most informative run left is the 2B chat arm (5× the erosion at 0.8B).

The caveats are that this is one seed, refusal is measured with a first-token proxy, and each 2B evaluation took ~41 min
on Kaggle (CPU-bound).

**16. Novelty check (literature, no new data): the claim moves to dynamics × modality × explanation.**
I read these in full. Xu et al. ICLR 2025 show that text safety fails to transfer to images when a VLM is *built*:
image hidden states shift from their text equivalents at the safety layers, and their fix is text-guided alignment.
Fu et al. CVPR 2026 (CARE) use causal localisation and separate visual and textual safety subspaces, and argue that
single-modality steering is insufficient. He, Xia & Henderson 2024 look at which benign SFT data breaks safety: data
near harmful examples in representation/gradient space, where 100 selected examples take GPT-3.5 ASR from <20% to
>70%. I also verified Kalajdzievski 2024 (forgetting follows a shifted power law during fine-tuning).

So several things become background and I do not claim them: forgetting laws, weaker image safety at a fixed point,
"the data matters", dual-modal subspaces for defence, and CapTrack-style measurement.

The contributions I claim are (a) the trajectory of image-input behaviour under benign text-only continual training;
(b) a predictive account of which post-trained behaviours erode and how fast, across modalities (coverage + overlap);
and (c) forecast-driven, behaviour-targeted repair.

I repositioned P5 (overlap sets the rate) as extending He et al. from subset selection in SFT to rate forecasting for
whole corpora in continual pre-training, across skills and modalities. I positioned the WP3 cost card as the
attack-side complement to CARE: its job is quantification rather than discovery.

This changed the proposal: I rewrote it (v2, 2,190 words, 20 references, all cited) and kept v1 for reference.

**17. Method caveat: numeric precision of training.**
The Kaggle notebooks chose dtype with `torch.cuda.is_bf16_supported()`, which returns True when bf16 is only
*emulated* (T4, capability 7.5). The parallel 2B run therefore started in emulated bf16, which is slow and 8× coarser
than fp16. The first 2B Kaggle run (entry 15) used the same logic and so probably ran in bf16 too. I cannot confirm
this, because its dtype line was never captured. I fixed the logic: native bf16 only from capability 8.0, otherwise
fp16 with the existing 1024× loss scale. I re-verified the kernel check in the same dtype.

There is a broader caveat for every run so far. Training updates pure fp16/bf16 weights with SGD, without fp32 master
weights. Updates smaller than half a unit in the last place round away, so the effective learning rate is lower than
nominal and uneven across parameters. Within-run comparisons (code vs chat, image vs text, skills vs refusal) are
unaffected. Absolute erosion rates, and comparisons across runs of different precision, are not.

So I need to decide before thesis-grade runs: adopt fp32 master weights, and quantify the difference once on the
laptop (0.8B, code arm, fp16 vs fp32-master).

**18. 2B chat arm, fp16 + fast kernels (parallel run): P2 fails at 2B on chat-like text; scale protects; skills hold.** (`results/e1s_2B_kaggle_fp16_benign.json`)
This run was ~10× faster than the first 2B run (≈50 min for 1M tokens + 4 evaluations + 2 skill evaluations). It used
the same number format (fp16) as the 0.8B laptop runs, so it is the first fair 0.8B-vs-2B comparison. At 1M tokens,
text refusal (wrapped) lost 7.6%, plain text lost 0% and image refusal lost 2.4% (image/text 0.23–0.57 across
checkpoints). ARC moved by +3 items, ARC-image went from 0.83 to 0.86, and held-out chat perplexity changed by −11%.
On skills (n 80/60/48/60), instruction following went from 0.688 to 0.650, maths from 0.333 to 0.400 and charts from
0.683 to 0.700, all within ±1 SE. Tool calling was at ceiling (1.00) at 2B.

So P2 does not hold for 2B on chat-like text. At 0.8B (same format), image refusal lost 30% vs text 14%.

This corrects entry 15, where I had it wrong. "Scale did not protect" came from comparing a probably-bf16 2B run with
fp16 0.8B runs. With matching precision, 2B eroded ~half as much (text) and >10× less (image) on chat-like text.

Skills hold at 2B while text refusal erodes, which is the "safety is uniquely thin" branch of the framing. Power only
rules out drops > ~10 points, and the tool test is too easy for 2B and needs harder tasks.

Erosion at 2B is framing-specific. Requests relayed through a wrapper lost refusal; direct requests did not. This may
be relevant to indirect-instruction safety, but I have not tested it.

The decisive next step is the same run's 2B code arm (same format and kernels).

**19. Matched 2B run, both arms (fp16 + fast kernels): P2 does not hold at 2B.** (`results/e1s_2B_kaggle_fp16_{ood,benign}.json`)
At 1M tokens in the code arm, plain text lost 3.0%, wrapped 2.3% and image 2.7% (image/text 0.9–1.2). In the
chat-like arm, plain lost 0.0%, wrapped 7.6% and image 2.4%. At 0.8B in the same format, image refusal eroded ~2×
faster in both arms.

So P2 is size-dependent: it holds at 0.8B and not at 2B. The headline "image-input safety erodes twice as fast" is a
0.8B result and I must state it as such.

The earlier 2B code-arm support (entry 15, 1.9×) was most likely a precision artefact. Its text erosion matches this
run almost exactly (2.29% vs 2.28%), but its image erosion was higher (4.36% vs 2.70%) under probable emulated bf16.

I have a candidate explanation, consistent with the coverage hypothesis but untested. At 0.8B image refusal is weak
(31% of text) and loosely integrated; at 2B it is strong (70%) and integrated (entries 7–9). Testing it needs 4B/9B,
which do not train on T4s.

One caveat now matters. Erosion at 2B is small everywhere, and pure-fp16 SGD drops sub-ULP updates (entry 17), so "2B is
robust" may partly be "2B trained little". I need full-precision master weights.

Skills hold at 2B in both arms (instruction −1/−4 pts, charts −2/+2, maths +10/+7, the same direction twice, each
within ~1.6 SE; tools at ceiling). Knowledge holds or improves.

P5 at 2B is framing-specific. Chat-like text erodes wrapped-request refusal 3.3× more than code does (5.6× at 0.8B),
but not plain requests (0.0% vs 3.0%) or images (2.4% vs 2.7%).

On the 0.8B skills (laptop, code arm, 131k tokens), instruction following went from 0.57 to 0.56, maths from 0.35 to
0.31, tools from 0.84 to 0.81 and charts from 0.70 to 0.70. That is small drift; later checkpoints are pending.

**20. Reinterpretation: "skills hold" means "skills had not degraded yet, in very gentle training".** (no new data)
Thomson's report states that continual pre-training degrades post-trained capabilities and that merging was necessary
to restore them. That is the reason replay and merges exist. My pilots use 1M tokens (200,000× fewer than Thomson's
200B), a deliberately gentle SGD setting I chose early on for baseline stationarity, pure 16-bit weights that drop
small updates (entry 17), and skill tests that only detect drops > ~10 points (tools at ceiling at 2B). Chat
perplexity moved only 3–11%.

The valid reading is that in early, gentle training, safety erodes before skills. That is an ordering result, and it
suggests replay priorities should shift over training (safety early, skills later). The invalid reading is that
skills are robust to realistic continual pre-training.

So before I make thesis-grade claims about skills or repair, I need a realistic-strength regime: a standard optimiser
with full-precision master weights, a sweep over training strength (learning rate × tokens, Thomson's own measure)
with tens of millions of tokens, and harder, larger skill tests. Milestone 1 is to reproduce skill degradation. Only
then will I test replay and merging on top.

**21. The first scaling run's 2M-token checkpoint: the log law may bend.** (`0.8B_colab_gentle_code.json`, 10 checkpoints)
Refusal is flat to ~32k tokens, then follows 0.728 %/doubling from 66k to 1M (R² 0.96, entry 1). The 2M checkpoint
added only 0.11 points where that law predicted 0.73. Including it, the fit is 0.680 %/doubling (R² 0.97). ARC stays
flat throughout (0.832–0.848).

This is one point in the gentle regime (pure 16-bit SGD, entry 17), so it may be noise or rounding-limited updates.

Forecast-then-correct replay (entry 2) assumes the law extrapolates, so I must check it over longer horizons before I
trust it. The realistic-strength run (8M tokens, AdamW with full-precision master weights) gives five checkpoints to
test this.

That means the proposal wording becomes "erosion is log-linear over 66k–1M tokens", not "log-linear" without a range.

The same day I also dealt with infrastructure (no finding). The first Kaggle strength/replay attempt exceeded the ~29 GB
system-RAM limit (two workers, each with ~6 GB of AdamW state in pinned memory rounded up to powers of two, plus an
fp32 model copy while loading). I fixed it: weights load straight onto the GPU, optimiser state sits in exact-size
ordinary memory (same maths), the second GPU's queue starts only when measured memory allows, and completed jobs are
skipped on re-run.

**22. Predictions are now pre-registered; the thesis's "predicting" claim gets its first real test.** (`docs/preregistration.md`, 2026-09-25 15:45 BST, no new data)
Until now I measured every rate after the fact. The title promises prediction, so P5 now has a first-order score. One
SGD step on corpus D moves the refusal gap by −lr·(∇gap·ḡ_D), and s = (∇gap·ḡ_D)/gap predicts relative erosion per
framing before training. I calibrate it on the two measured corpora (code, chat-like), then use it to predict legal
(US case law, Thomson's domain) and news (CNN/DailyMail), with pass/fail rules fixed in advance. At the same time I
pre-registered P3's dynamic version (text-only replay should restore at most ~2/3 as much image refusal as text
refusal), the skills milestone, the second seed, a generation check of the first-token measure, and a natural-image
version of the cross-modal ablation (VLSBench).

The score is standard continual-learning gradient interference (the GEM / A-GEM line) applied per behaviour and per
input modality. That puts P5 in the lab's own lineage rather than beside it.

A miss is informative either way. If the score retrodicts code vs chat but misses legal/news, the first-order view is
incomplete, and the PhD must model the trajectory, not just the starting direction.

**23. Realistic regime, first two jobs (lr 2e-5, partial): skills collapse; the milestone is passed, and overshot.** (`results/d_trajectories/e2/e2_0.8B_lr2e-05_none_{ood,benign}.json`; the run stopped ~2 h in: code at 4M, chat-like at 2M of 8M tokens; the four lr 1e-4 jobs never started.)
I used AdamW, full-precision master weights and 16k-token batches.

On code, ARC was unchanged (0.84–0.86). Refusal lost 25% (text) / 35% (image) within 0.5M tokens (31 optimiser
steps), 10× the gentle regime's 1M-token loss, and then stayed flat (18–26%). At 1M, instruction following was
−21 pts (−5.4 SE), maths −28 (−6.4 SE), chart reading went from 70% to 0%, and tool calling was +9. The model barely
learned the code (held-out code loss 0.96 at 4M, and worse, 1.18, at 0.5M), so the damage came before the learning.

On chat-like text the model had collapsed completely by 0.5M: refusal −90 to −96%, every skill ≈ 0, ARC at chance
(0.26), while held-out chat loss fell 30%. The model learned to write user questions and stopped answering at all.

So P1 now holds for skills too, in the realistic regime. On code, knowledge is intact while refusal, instruction
following, maths and chart reading erode. I have reproduced the replay/merge premise at small scale.

One caveat matters here. ARC is read as a letter choice, which is itself a post-trained behaviour (small base models
are near chance at it). In the chat arm, ARC at chance means answering collapsed, not that knowledge was lost. Chart
reading at exactly 0 with image-ARC intact is most likely the same thing: the answers are no longer in the parseable
short form. Skills here measure behaviour-plus-format, and I need a format-validity rate to separate the two.

Erosion is not log-linear here. There is a jump in the first ~30 steps (during warm-up), then a plateau (code 0.5–4M:
R² 0.00). Forecast-based replay on demand would miss it, so protection has to start at step one. This revises
entry 21.

LR × tokens is not a usable strength measure across batch sizes with Adam. Each step moves weights ~lr whatever the
batch, so my 16k-token batches take ~250× more steps per token than a ~4M-token production batch. I should report
strength as lr × steps (and batch size).

That makes lr 1e-4 uninformative now (certain collapse). Next I will run lower rates for a dose–response (which
behaviour breaks first as strength rises), and test replay where no-replay damage is partial (code, 2e-5).

**24. Prediction test and first replay results (Kaggle, provisional from the run log).**

First, the prediction test (P5). I computed the first-order score on the untrained model and wrote the predictions down before training. The score retrodicts the six known rates (Spearman 0.94) and correctly separates dangerous from mild data: chat-like scores ~3.4× code. But it fails the pre-registered test (1 of 4 pass). Legal and news each erode wrapped-text refusal by ~6% per 1M tokens (2.4× code, under half of chat-like), about double what I predicted. Neither touches image refusal (0.2%, −1.6%). The score got the sign of legal's image channel right, but it predicted 7.6% for news.

So, as pre-registered in entry 22, the starting gradient direction separates dangerous from mild corpora but cannot rank within the mild group. To forecast, I need the trajectory as well as the first step. P5 moves from preliminary result to open question.

There is also a domain finding for Thomson. Legal text (US case law) erodes text refusal faster than code and leaves image refusal intact, and news does the same. The Kaggle chat-like anchor reproduces the laptop numbers within 1% (f = 0.99), so gentle-regime numbers hold across hardware.

Then the first replay runs, on code at lr 2e-5. Here replay means the model's own answers to 192 general chat prompts. Both forms of replay restore refusal, and even over-restore it, while destroying tool calling and chart reading, which ends up worse than no replay at all. Triggered replay had one trigger at 655k, with 65k replay tokens (1.6% of the data). At 1M, tools were at 0.08 vs 0.78 without replay and instruction following at 0.20 vs 0.36, while refusal was at 107%/115% of baseline. Fixed replay (31% of every batch) had tools at 0.17 and charts at 0 by 262k, held-out chat loss at 2.0× by 2M, ARC-image down from 0.76 to 0.58 at 0.5M, and image refusal down to 51% at 2M.

That means replay is not neutral. A narrow replay set restores the behaviours it resembles (refusal, maths format) and overwrites the ones it lacks (tool-call and short-answer formats). This is direct evidence for "aiming the repair": replay has to be composed per behaviour. There is a confound I need to remove. The pool is tiny (192 answers, cycled ~15× by 2M), so overfitting explains part of the effect. Next I plan a diverse pool covering every skill family, with prompts disjoint from the test items.

P3 was pre-registered, and it is refuted: triggered replay restored image refusal more than text refusal (ratio 2.2). Fixed replay damaged image refusal at 2M. So text-only repair moves image behaviour a lot, though not predictably by the "two-thirds shared" rule.

The dose–response run on chat-like data at lr 5e-6 (to 2M) shows a graded order. Refusal goes first (−12% at 131k). Tool calling and instruction following come next (−14 and −10 points, about −3 SE, at 262k). Knowledge goes last (ARC −31 items, and only from 1M). This is the order the thesis claims (safety, then skills, then knowledge), seen at a moderate rate.

As a reproducibility check, before its first trigger the triggered arm matched the earlier no-replay run exactly (0.747 / 0.648 at 0.5M).

**25. Overnight laptop results: text claims hold; image erosion is not yet visible in generated text; harmful photos run mostly off the text route.**

The generation check (M1) shows that the first-token refusal score predicts generated refusals very well for text (AUROC 0.97–1.00) and tracks them over training: the text refusal rate falls 6–9 points when the gap falls 6–17%. For image inputs the baseline agreement is weaker (AUROC 0.80), and it breaks down over training. The image gap fell by up to 32% on chat-like data, while harmful-image refusals in generated text stayed at ~0.38 (the 0.8B model already complies with most harmful image requests at baseline).

So text-refusal erosion is validated in generated behaviour, but **P2's image claim is qualified**. At 0.8B the *inclination* to refuse images erodes faster (first-token), but that does not yet show up as more compliance within 48 generated tokens with a keyword judge. I need longer generations and a stronger judge for image responses, and a 2B check, where image refusal starts stronger. The proposal must say "first-token" for images.

With a second seed (D5), chat-like replicates tightly (wrapped 14.5 vs 14.0%, image 28.7 vs 29.9%, ratio 1.98 vs 2.14). Code does not replicate its image effect (ratio 0.84 vs 1.40), and with a different code sample its wrapped loss doubled (5.7 vs 2.5%).

That means "chat-like erodes far more than code" holds across seeds (2.5–5.6×), but code's rate depends on *which* code: the data sample matters within a data type. At 0.8B, P2 stands on chat-like data only.

For skills in the gentle regime (D4), only tool calling moved, in both arms (−9 to −11 points, ~2.1–2.4 SE). Instruction following, maths and charts stayed flat to 1M. That is consistent with the post-trained format layer going first, but the effect is small.

With natural images (E6, 2B), harmful photos with neutral queries elicit refusal (gap 2.51, a third of text strength), and more refusal than their own text description does (1.60). The natural-image refusal direction sits 78° from the text direction (for rendered text it is 42°). Removing the text direction leaves 66% of natural-image refusal (real photos 53%, generated 85%). Removing the natural-image direction removes 89% of it, but only a third of text refusal.

This is my strongest multimodal result so far, and it holds outside rendered text. Text-derived safety interventions (and text-only repair) reach about a third of photo refusal, and the rest runs on a largely separate route. That strengthens WP3's cost card and the case for a behaviour-by-modality view of repair.

**26. Kaggle run, final: the ordering claim holds; P5 and P3 fail; skills erode in two different ways.** (`results/d_trajectories/e2/e2_0.8B_dr_*`, `results/prediction/`; notebooks 05, 06)

All four pre-registered dose–response checks pass. Damage rises with the learning rate for both data types, and chat-like erodes more than code at every rate. At the lowest rate (2e-6), refusal erodes (≥ 5% by 262k tokens) before any skill falls. On code no skill fell by 2M; on chat-like data skills first fell at 1M. Knowledge on code stayed within 1 point.

So the order (safety, then skills, then knowledge) is now shown at two data types and three rates. It is the best-supported of my results on which behaviours erode, and it is what keeps "predicting" in the title defensible (see entry 27).

The new format rates let me separate format from ability. Chart reading collapses with its format (short answers fall from 0.98 to 0.04–0.07). Tool calling collapses with call validity (from 0.86 to 0.16–0.20). Maths goes the other way: format rises (from 0.41 to 0.6–0.8) while accuracy falls. So charts and tools lose the post-trained *output format*, while maths loses *ability*.

That means "skills" is two layers with different fragility. Format is the thinnest layer and goes first. The thesis grid should separate the two, and repair can target them separately.

The prediction (P5), now confirmed from the JSON, is not supported (1 of 4), as in entry 24.

In the final replay results, both narrow-replay arms end with tools at ~0 and charts at 0. Fixed replay also ends with held-out chat loss 2.6× worse, text refusal at 118% (over-restored) and image refusal at 68% (below no replay's 75%). The dynamic version of P3 fails in both arms, in opposite directions: text-only replay moved image refusal a lot, but not in the "about two thirds" way I predicted. 3f fails too: triggered replay used 5% of the replay tokens, but ended far below fixed replay on maths.

The sensor data, also from the JSON, shows the triggered arm's drift loss (on the model's own chat answers) falling during early code training (from 1.62 to 1.60, baseline 1.64), while refusal lost 11% and skills fell. So the generic sensor is blind, which is worse than slow.

For the replay-on-demand regime, erosion is gradual at 2e-6 and 5e-6 on chat-like data (refusal first at 131–262k, skills from 262k–1M). This changed the W2 setup: it moves from code at 2e-5 (a jump in ~30 steps) to chat-like data at 5e-6. There the canary has a lead of ~130k tokens over the skills.

**27. W2: on-demand replay beats fixed replay with a fifth of the replay; timing and dose did the work rather than routing.** (`results/d_trajectories/w2/`; notebook 07; pre-registration §7 outcomes)

The verdicts: R2, R3 and R5 pass. R1, R4 and R6 fail, R1 and R6 on chart reading and R4 on tool calling.

I compared on-demand replay (A2) with fixed broad replay (A1). A2 used 19% of A1's replay tokens and ended +19.5 points ahead on all 476 test items (95% +14.5 to +24.4). At 2M it sits at baseline on instruction following and maths, above baseline on tools and 7 points below on charts, and it keeps all of A0's domain learning (A1 keeps 91%). A2 matches A0 exactly until its first trigger, so the gap is caused by the replay.

The replay came in two bursts. The refusal canary fired the first at 131k (all families), and the chat and instruction sensors fired the second at 459k. Nothing fired after about 0.5M tokens, and the tools, maths and charts sensors never fired.

So "aim the repair" has its first pre-registered positive result. But the aiming that worked was **when and how much**: an early broad burst plus one top-up. Routing by behaviour was barely exercised, and the proposal must say this.

Erosion is front-loaded. In A0, the loss on the model's own answers rose mostly between 131k and 507k tokens (tools from ×1.2 to ×8.1). That is around the end of warm-up, where the learning rate is highest. A2 was restored inside that window, and then did not erode over the next 1.5M domain tokens.

This gives me a hypothesis, still untested: damage is set in an early, high-rate window, and repair inside that window lasts. I can test it by moving the burst (early vs late) and repeating on a second data order. If it holds, the question "when to repair" becomes a prediction problem, which links WP1 and WP2.

Continuous replay has its own costs. A1 cost 21 maths points relative to no replay, and it still lost chart reading. Chart answers were replayed, and the answers stayed short but wrong. Reference answers (A3) moved skills 17 points further than the model's own answers did (R5): maths fell to 0.05 and charts to 0, while tools rose above baseline.

So replay is itself a training signal with side effects, and it has to be dosed. Repair should be measured with the same instrument as erosion. This vindicates the Qwen-consistency design choice.

On coverage (R6), tools need tool replay: A1 beat chat-only A4 by +34 to +41 points. Charts can't be judged here, because fixed replay protected them in neither arm. Chat-only replay (A4) over-restored text refusal (2.2× baseline). Against no replay it cost maths (0.12 vs 0.42) but held tools better (0.34 vs 0.11). So entry 26's "narrow replay is worse than none" holds here for maths only.

On merging (R4), at α = 0.5 the merge brings chart reading back almost fully (99%) but tools only to 77%, with maths 17.5 points down, and it gives back 16% of the domain gain. A2 beats that merge on instruction following, maths and tools, and is 6 points behind on charts (95% −12 to 0).

That means format comes back by interpolation more easily than ability does, which is consistent with the format/ability split in entry 26.

On early warning (R3), the refusal canary crossed at 131k, one checkpoint (about 130k tokens) before the first skill drop.

Together with the order, this is what keeps "predicting" in the title defensible. The P5 rate forecast failed, but two things hold: *which behaviour goes first*, and *a signal that fires before the skills go*.

The caveats are these. This is one run of one 0.8B model. Tools above baseline in A2 and A3 is partly template learning, because the pool's tool prompts share the test templates, so I report tools as "kept" rather than "improved". I do not interpret image refusal (A2 2.05× baseline).

**28. Correction: knowledge is not immune. It erodes last.**

In FINDINGS (26 Sep) I wrote that science-question accuracy "barely moves" in every run (ARC ≤ 8/300 net lost). That was true on code and in the gentle regime, and I generalised it too far. On chat-like data in the realistic regime, ARC net loss at 2M is 11/300 at lr 2e-6, 40 at 5e-6 and 185 at 2e-5.

The order still holds where it can be resolved. At 2e-6 and 5e-6, refusal went first (262k / 131k), then skills (1M / 262k), then knowledge (2M / 1M). At 2e-5 everything was already down at the first checkpoint (507k). Replay protected knowledge in every W2 arm.

So P1 now reads: "knowledge erodes last; on code it stayed within 8/300 over 4M tokens". The order claim is unchanged. I have corrected FINDINGS and the README.
