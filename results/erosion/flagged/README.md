# Flagged: probable emulated-bf16 run

The files `D3_erosion_Qwen3.5-2B_*_BF16-SUSPECT*` come from my first 2B Kaggle run. Its notebook chose bf16 via
`torch.cuda.is_bf16_supported()`, which returns True on a T4 even though bf16 is only emulated there (slow, and 8×
coarser than fp16). I never captured the run's dtype line, so this is probable but not confirmed.

Its text refusal erosion matches the matched fp16 run almost exactly (2.29% vs 2.28% at 1M tokens), but its image
erosion was higher (4.36% vs 2.70%). **Do not use it for the image-vs-text comparison.** Use
`../D3_erosion_Qwen3.5-2B_*_gentle.json` instead. See THESIS_LOG entries 15, 17 and 19.
