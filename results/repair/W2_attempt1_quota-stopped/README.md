# W2 first attempt: stopped by the Kaggle GPU quota (2026-09-26)

The replay pool built and passed its quality gate. A0 (no replay) and A1 (fixed broad replay) reached 1,015,808 of
2,015,232 tokens before my weekly GPU quota ran out, and A2, A3, A4 and the merge sweep never started. I did not save
weights or optimiser state mid-run, so these files cannot be resumed. The rerun starts every arm from zero on the
same pool.

These partial files are not assessed. Under pre-registration addendum 4 (16:56 BST), no W2 prediction is judged on
them.

They do give a reproducibility check. Partial A0 reproduces the earlier chat-like lr 5e-6 run
(`../../erosion/R2_dose-response_Qwen3.5-0.8B_chat_adamw-lr5e-6_no-replay.json`) to within 0.001 of refusal kept at
every checkpoint. The rerun's first 1M tokens should match these files to the same tolerance, which checks
run-to-run determinism.

The pool is `../W2_replay-pool_model-own-answers_Qwen3.5-0.8B.json.gz`, with sha256 prefix `034fe58e28faf8fc` (of
the uncompressed file). The rerun notebook checks this fingerprint.
