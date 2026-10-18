# Keeping What Matters

This repository holds the preliminary experiments for my PhD proposal to the Thomson Reuters–Imperial Frontier AI
Research Lab. The proposal is about predicting which post-trained behaviours continual training erodes, across text
and image inputs, and aiming the repair at them.

When a post-trained model keeps training on new domain text, its knowledge mostly survives. What it was *taught
to do* (refuse harmful requests, follow instructions, call tools) wears away. Replay and model merging repair
this, but they cost compute. I set out to answer three questions:
- Which behaviours erode, and how fast?
- Does behaviour on image inputs erode differently from text?
- Can erosion be forecast, so that replay runs only when and where it is needed?

I suggest starting with [`FINDINGS.md`](FINDINGS.md). It gives the results in plain English, the status of each
prediction, and an account of what is still missing. [`EXPERIMENTS.md`](EXPERIMENTS.md) maps every experiment
to the thesis claim it serves. [`docs/ROADMAP.md`](docs/ROADMAP.md) lists the steps that remain, ending with
finalising the repo. [`THESIS_LOG.md`](THESIS_LOG.md) records every result in order, with the change it
forced in the thesis.

## Status (27 Sep 2026)
| | |
|---|---|
| Post-training erodes in an order: safety first, then skills, then knowledge (knowledge goes last, though it is not immune) | done: the pre-registered order checks pass (notebooks 01, 03); I corrected the knowledge claim on 27 Sep |
| Data type sets the rate (code, chat-like, legal, news) | done; my first-order forecast of the rate failed its pre-registered test (04) |
| Image vs text | first-token erosion is faster for images only at 0.8B on chat-like data; harmful photos use a largely separate route (01, 04, 05) |
| Skills under realistic training | they erode in two ways: output format (charts, tools) and ability (maths) (03) |
| Replay | narrow replay restores refusal but destroys tool calling and chart reading (03) |
| Replay on demand (W2) | beat fixed broad replay by 19.5 pts while using 19% of its replay, and kept all domain learning; 3 of 6 pre-registered predictions pass; this is one run (06) |
| Next | the proposal (with the W2 replication written in as a first-year experiment), then finalising the repo (`docs/ROADMAP.md`) |



## Repository map
```
FINDINGS.md            results in plain English, prediction status, validation gap
THESIS_LOG.md          one entry per result: finding, validity, what it changed (entries 1-28)
EXPERIMENTS.md         registry: every experiment ID, the claim it serves, its results and notebook
src/                   shared library: models, probes, measurement, geometry, edits, gates, vision, skills
experiments/
  continual_training/  every continual-training run: gentle-regime trajectories (D1-D5), the forecast test (R3),
                       the realistic regime (R1-R2), replay (W1, W2), the generation check (M1); laptop scripts
                       and the Kaggle per-GPU workers
  multimodal/          E3-E6: text vs image refusal and value geometry, ablation, split test, natural photos
  background/          A rank cost, B raw refusal alarm, C refusal vs value directions; run_abc_batch.py runs all three
notebooks/
  analysis/            recompute every table and figure from results/, in thesis order; each name lists its
                       experiment IDs (01_D2-D3-D5_… to 07_A-B-C-D1_…). build_notebooks.py regenerates them,
                       kwm.py holds shared helpers
  cloud/               the Colab/Kaggle notebooks that produced the GPU results, named by experiment ID
results/               every kept result, named <experiment ID>_<what>_<model>_<data>_<regime> (NAMING.md
                       maps each file to its experiment, notebook and original name)
  erosion/         D1-D5 gentle-regime trajectories; R1-R2 realistic regime (flagged/: an excluded 2B run)
  forecast/        R3: the pre-registered forecast of erosion rates and its test
  measurement/         M1: first-token refusal score vs generated refusals
  multimodal/          E3-E6: text vs image refusal and value routes (ablations, natural photos)
  repair/          W1 narrow replay pilot; W2 replay on demand, fixed, reference-answer, chat-only, merge
  background/          A rank cost, B raw refusal alarm, C refusal vs value directions
tools/rename_results.py  maps worker output names to the names above (safe to re-run)
docs/
  proposal/            the proposal (.docx, in research-paper format, built by build_paper.js from paper.json) and its figure (make_figures.py)
  preregistration.md   predictions and pass/fail rules, fixed before the results (2026-09-25 15:45 BST)
  literature.md        verified references, novelty map, full-read notes
  design/              experiment specification, its review, and two design notes (ablation resistance,
                       constitution throughout training)
archive/               superseded, unfinished or mistaken material, each with the reason (see archive/README.md)
```

## Running things

You need Python 3.11+; install the dependencies with `pip install -r requirements.txt`. On NVIDIA GPUs,
`flash-linear-attention` speeds up Qwen3.5's linear-attention layers. The notebooks install it and keep it only if
it reproduces the reference logits (`results/infrastructure/fast-kernel-reference-logits_Qwen3.5-0.8B.json`).

The analysis notebooks run on a CPU in seconds. Open `notebooks/analysis/0*.ipynb`, or re-execute them all:
```bash
cd notebooks/analysis && python build_notebooks.py && for n in 0*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$n"; done
```

I have kept the experiment scripts exactly as they ran. Run them from the repository root. They write to
`results/` at the top level under their original names, and `python tools/rename_results.py` then gives them their
experiment names and folders.
```bash
PYTHONPATH=src:experiments/background python3 experiments/background/run_abc_batch.py --seeds 0 1
```
```bash
PYTHONPATH=src python3 experiments/multimodal/exp_e_crossmodal.py --model Qwen/Qwen3.5-2B --seed 0 --heldout
```
```bash
PYTHONPATH=src:experiments/multimodal python3 experiments/multimodal/exp_e_split.py --model Qwen/Qwen3.5-2B --seed 0 --behaviour value
```
```bash
PYTHONPATH=src python3 experiments/continual_training/local_e1s.py
```
Every script accepts `--dry` or `--smoke`, so you can exercise all code paths at trivial scale first.
`gen_check.py` and `local_e1_e2.py --seed N` run from `experiments/continual_training/`, because they read
`local_e1s.py` and `checkpoints/` from there. `exp_e_natural.py` needs `PYTHONPATH=src:experiments/multimodal`.

To rerun a GPU trajectory, upload a notebook from `notebooks/cloud/` to Kaggle (GPU T4 ×2, internet on) and use
*Save & Run All*. Each notebook writes its own worker and data files, so nothing else needs uploading. Copy the
resulting JSON files into the `results/` folder named in `results/NAMING.md`'s "Original name" column, run
`python tools/rename_results.py`, and re-run the analysis notebooks.

## Conventions and caveats
- File names call the data arms `code` and `chat` (chat-like text: ultrachat user turns). Inside the JSON files and
  the worker code, the same arms are called `ood` and `benign`. `legal` is US case law (Caselaw Access Project) and
  `news` is CNN/DailyMail.
- I read refusal from the first response token, as the harmful-minus-harmless log-odds gap of starting a refusal.
  *Kept* = gap now ÷ gap at start. *Wrapped* text uses the same wrapper as the image prompts, so the image-vs-text
  comparisons are matched.
- The gentle regime (D1-D5, R3) uses SGD + momentum, pure 16-bit weights and ≤ 2M tokens. It is far weaker than real
  continual pre-training, and small updates round away. The realistic regime (R1, R2, W1, W2) uses AdamW with
  full-precision master weights, 16k-token batches and lr 2e-6 to 2e-5.
- Most runs use one seed (D5 repeats D2 on a second seed), and W2 is one run. Images are rendered text except in E6
  (natural photos).
- The W2 replay pool (`results/repair/W2_replay-pool_model-own-answers_Qwen3.5-0.8B.json.gz`, 7.9 MB) holds the
  model's own answers that W2 replays. I leave it out of the zipped copy to keep the zip small, so it may be missing
  from yours. The sha256 prefix of the uncompressed file is `034fe58e28faf8fc`, and I can send it on request. You
  need it only to rerun W2; every analysis notebook runs without it.

