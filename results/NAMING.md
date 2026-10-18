# Result file names

| New name (under `results/`) | Experiment | Notebook | Original name |
|---|---|---|---|
| `erosion/D1_decay-law_Qwen3.5-0.8B_code_gentle_2M-tokens.json` | D1 | 07 | `d_trajectories/0.8B_colab_gentle_code.json` |
| `erosion/D1_regime-sweep_lr-x-steps_Qwen3.5-0.8B_code-vs-chat.json` | D1 | 07 | `d_trajectories/0.8B_colab_window_sweep.json` |
| `erosion/D2_erosion_Qwen3.5-0.8B_code_gentle_seed0.json` | D2 | 01 | `d_trajectories/0.8B_laptop_gentle_code.json` |
| `erosion/D2_erosion_Qwen3.5-0.8B_chat_gentle_seed0.json` | D2 | 01 | `d_trajectories/0.8B_laptop_gentle_chat.json` |
| `erosion/D5_erosion_Qwen3.5-0.8B_code_gentle_seed1.json` | D5 | 01 | `d_trajectories/0.8B_laptop_gentle_code_seed1.json` |
| `erosion/D5_erosion_Qwen3.5-0.8B_chat_gentle_seed1.json` | D5 | 01 | `d_trajectories/0.8B_laptop_gentle_chat_seed1.json` |
| `erosion/D3_erosion_Qwen3.5-2B_code_gentle.json` | D3 | 01, 02 | `d_trajectories/2B_kaggle_fp16_code.json` |
| `erosion/D3_erosion_Qwen3.5-2B_chat_gentle.json` | D3 | 01, 02 | `d_trajectories/2B_kaggle_fp16_chat.json` |
| `erosion/flagged/D3_erosion_Qwen3.5-2B_code_gentle_BF16-SUSPECT.json` | D3 (excluded) | 01 | `d_trajectories/flagged/2B_kaggle_BF16SUSPECT_code.json` |
| `erosion/flagged/D3_erosion_Qwen3.5-2B_chat_gentle_BF16-SUSPECT_partial.json` | D3 (excluded) | 01 | `d_trajectories/flagged/2B_kaggle_BF16SUSPECT_chat_partial.json` |
| `erosion/flagged/README.md` | D3 (excluded) |  | `d_trajectories/flagged/README.md` |
| `erosion/D4_skills_Qwen3.5-0.8B_code_gentle.json` | D4 | 02 | `d_trajectories/0.8B_laptop_gentle_skills_code.json` |
| `erosion/D4_skills_Qwen3.5-0.8B_chat_gentle.json` | D4 | 02 | `d_trajectories/0.8B_laptop_gentle_skills_chat.json` |
| `erosion/R1_realistic_Qwen3.5-0.8B_code_adamw-lr2e-5_no-replay.json` | R1 | 03 | `d_trajectories/e2/e2_0.8B_lr2e-05_none_ood.json` |
| `erosion/R1_realistic_Qwen3.5-0.8B_chat_adamw-lr2e-5_no-replay.json` | R1 | 03 | `d_trajectories/e2/e2_0.8B_lr2e-05_none_benign.json` |
| `erosion/R2_dose-response_Qwen3.5-0.8B_code_adamw-lr2e-6_no-replay.json` | R2 | 03 | `d_trajectories/e2/e2_0.8B_dr_lr2e-06_none_ood.json` |
| `erosion/R2_dose-response_Qwen3.5-0.8B_chat_adamw-lr2e-6_no-replay.json` | R2 | 03 | `d_trajectories/e2/e2_0.8B_dr_lr2e-06_none_benign.json` |
| `erosion/R2_dose-response_Qwen3.5-0.8B_code_adamw-lr5e-6_no-replay.json` | R2 | 03 | `d_trajectories/e2/e2_0.8B_dr_lr5e-06_none_ood.json` |
| `erosion/R2_dose-response_Qwen3.5-0.8B_chat_adamw-lr5e-6_no-replay.json` | R2 | 03 | `d_trajectories/e2/e2_0.8B_dr_lr5e-06_none_benign.json` |
| `erosion/logs/D2_erosion_laptop.log` | D2 |  | `d_trajectories/logs/local_e1_e2_full.log` |
| `erosion/logs/D5_erosion_code_seed1_laptop.log` | D5 |  | `d_trajectories/logs/local_e1_seed1_ood.log` |
| `erosion/logs/D5_erosion_chat_seed1_laptop.log` | D5 |  | `d_trajectories/logs/local_e1_seed1_benign.log` |
| `erosion/logs/D3_erosion_2B_code_kaggle.log` | D3 |  | `d_trajectories/logs/kaggle_2B_fp16_code.log` |
| `erosion/logs/D3_erosion_2B_chat_kaggle.log` | D3 |  | `d_trajectories/logs/kaggle_2B_fp16_chat.log` |
| `erosion/logs/D4_skills_laptop.log` | D4 |  | `d_trajectories/logs/local_e1s.log` |
| `erosion/logs/M1-D5-E6_overnight_chain_laptop.log` | M1, D5, E6 |  | `d_trajectories/logs/overnight_chain.log` |
| `forecast/R3_overlap-scores_before-training_Qwen3.5-0.8B.json` | R3 | 04 | `prediction/e1s_0.8B_overlap.json` |
| `forecast/R3_forecasts_preregistered_Qwen3.5-0.8B.json` | R3 | 04 | `prediction/predictions_0.8B.json` |
| `forecast/R3_test-run_Qwen3.5-0.8B_legal_gentle.json` | R3 | 04 | `prediction/e1s_0.8B_legal.json` |
| `forecast/R3_test-run_Qwen3.5-0.8B_news_gentle.json` | R3 | 04 | `prediction/e1s_0.8B_news.json` |
| `forecast/R3_anchor-run_Qwen3.5-0.8B_chat_gentle.json` | R3 | 04 | `prediction/e1s_0.8B_benign.json` |
| `forecast/R3_verdict_Qwen3.5-0.8B.json` | R3 | 04 | `prediction/check_0.8B.json` |
| `measurement/M1_first-token-vs-generated-refusal_Qwen3.5-0.8B.json` | M1 | 04 | `measurement/gen_check_0.8B.json` |
| `measurement/logs/M1_first-token-vs-generated-refusal.log` | M1 |  | `measurement/logs/gen_check.log` |
| `multimodal/E3-E4_refusal_crossmodal-ablation_Qwen3.5-0.8B_seed0.json` | E3/E4 | 05 | `e_crossmodal/E_crossmodal__Qwen3.5-0.8B__seed0.json` |
| `multimodal/E3-E4_refusal_crossmodal-ablation_Qwen3.5-0.8B_seed0_heldout.json` | E3/E4 | 05 | `e_crossmodal/E_crossmodal__Qwen3.5-0.8B__seed0_heldout.json` |
| `multimodal/E3-E4_refusal_crossmodal-ablation_Qwen3.5-2B_seed0.json` | E3/E4 | 05 | `e_crossmodal/E_crossmodal__Qwen3.5-2B__seed0.json` |
| `multimodal/E3-E4_refusal_crossmodal-ablation_Qwen3.5-2B_seed0_heldout.json` | E3/E4 | 05 | `e_crossmodal/E_crossmodal__Qwen3.5-2B__seed0_heldout.json` |
| `multimodal/E3-E4_refusal_crossmodal-ablation_Qwen3.5-2B_seed1.json` | E3/E4 | 05 | `e_crossmodal/E_crossmodal__Qwen3.5-2B__seed1.json` |
| `multimodal/E4_values_crossmodal-ablation_Qwen3.5-2B_seed0.json` | E4 | 05 | `e_crossmodal/E_crossmodal_value__Qwen3.5-2B__seed0.json` |
| `multimodal/E4_values_crossmodal-ablation_Qwen3.5-2B_seed0_heldout.json` | E4 | 05 | `e_crossmodal/E_crossmodal_value__Qwen3.5-2B__seed0_heldout.json` |
| `multimodal/E5_values_picture-only-route-split_Qwen3.5-2B_seed0.json` | E5 | 05 | `e_crossmodal/E_split_value__Qwen3.5-2B__seed0.json` |
| `multimodal/E6_refusal_natural-photos-ablation_Qwen3.5-2B_seed0.json` | E6 | 05 | `e_crossmodal/E_natural__Qwen3.5-2B__seed0.json` |
| `multimodal/logs/E3-E4_refusal_crossmodal-ablation_2B_seed0.log` | E3/E4 |  | `e_crossmodal/logs/E_crossmodal_full.log` |
| `multimodal/logs/E3-E4_refusal_crossmodal-ablation_2B_seed1.log` | E3/E4 |  | `e_crossmodal/logs/E_crossmodal_2B_s1.log` |
| `multimodal/logs/E3-E4_refusal_crossmodal-ablation_0.8B_seed0.log` | E3/E4 |  | `e_crossmodal/logs/E_crossmodal_0.8B_s0.log` |
| `multimodal/logs/E3-E4_refusal_crossmodal-ablation_2B_heldout.log` | E3/E4 |  | `e_crossmodal/logs/E_heldout_refusal_2B.log` |
| `multimodal/logs/E3-E4_refusal_crossmodal-ablation_0.8B_heldout.log` | E3/E4 |  | `e_crossmodal/logs/E_heldout_refusal_0.8B.log` |
| `multimodal/logs/E4_values_crossmodal-ablation_2B_seed0.log` | E4 |  | `e_crossmodal/logs/E_crossmodal_value_2B_s0.log` |
| `multimodal/logs/E4_values_crossmodal-ablation_2B_heldout.log` | E4 |  | `e_crossmodal/logs/E_heldout_value_2B.log` |
| `multimodal/logs/E5_values_picture-only-route-split_2B.log` | E5 |  | `e_crossmodal/logs/E_split_value_2B_s0.log` |
| `multimodal/logs/E6_refusal_natural-photos-ablation_2B.log` | E6 |  | `e_crossmodal/logs/E_natural_2B.log` |
| `repair/W1_narrow-replay-fixed_Qwen3.5-0.8B_code_adamw-lr2e-5.json` | W1 | 03 | `d_trajectories/e2/e2_0.8B_dr_lr2e-05_fixed_ood.json` |
| `repair/W1_narrow-replay-triggered_Qwen3.5-0.8B_code_adamw-lr2e-5.json` | W1 | 03 | `d_trajectories/e2/e2_0.8B_dr_lr2e-05_triggered_ood.json` |
| `repair/W2_A0_no-replay_Qwen3.5-0.8B_chat_adamw-lr5e-6.json` | W2 | 06 | `d_trajectories/w2/w2_0.8B_A0_none_benign.json` |
| `repair/W2_A1_fixed-broad-replay_Qwen3.5-0.8B_chat_adamw-lr5e-6.json` | W2 | 06 | `d_trajectories/w2/w2_0.8B_A1_fixed_benign.json` |
| `repair/W2_A2_on-demand-replay_Qwen3.5-0.8B_chat_adamw-lr5e-6.json` | W2 | 06 | `d_trajectories/w2/w2_0.8B_A2_demand_benign.json` |
| `repair/W2_A3_fixed-replay-reference-answers_Qwen3.5-0.8B_chat_adamw-lr5e-6.json` | W2 | 06 | `d_trajectories/w2/w2_0.8B_A3_fixedref_benign.json` |
| `repair/W2_A4_fixed-replay-chat-only_Qwen3.5-0.8B_chat_adamw-lr5e-6.json` | W2 | 06 | `d_trajectories/w2/w2_0.8B_A4_narrow_benign.json` |
| `repair/W2_M_merge-sweep_base-with-A0_Qwen3.5-0.8B.json` | W2 | 06 | `d_trajectories/w2/w2_0.8B_M_merge_benign.json` |
| `repair/W2_replay-pool_model-own-answers_Qwen3.5-0.8B.json.gz` | W2 |  | `d_trajectories/w2/w2_pool.json.gz` |
| `repair/W2_attempt1_quota-stopped/README.md` | W2 (first attempt) |  | `d_trajectories/w2/attempt1_quota_stopped/README.md` |
| `repair/W2_attempt1_quota-stopped/W2_A0_no-replay_Qwen3.5-0.8B_chat_adamw-lr5e-6_partial-1M.json` | W2 (first attempt) |  | `d_trajectories/w2/attempt1_quota_stopped/w2_0.8B_A0_none_benign.json` |
| `repair/W2_attempt1_quota-stopped/W2_A1_fixed-broad-replay_Qwen3.5-0.8B_chat_adamw-lr5e-6_partial-1M.json` | W2 (first attempt) |  | `d_trajectories/w2/attempt1_quota_stopped/w2_0.8B_A1_fixed_benign.json` |
| `background/A_rank-cost_refusal-directions_Qwen3.5-2B_seed0.json` | A | 07 | `a_rank_cost/A_rank_cost__Qwen3.5-2B__seed0.json` |
| `background/A_rank-cost_refusal-directions_Qwen3.5-2B_seed1.json` | A | 07 | `a_rank_cost/A_rank_cost__Qwen3.5-2B__seed1.json` |
| `background/A_rank-cost_refusal-directions_Qwen3-1.7B_seed0.json` | A | 07 | `a_rank_cost/A_rank_cost__Qwen3-1.7B__seed0.json` |
| `background/A_rank-cost_refusal-directions_Qwen3-1.7B_seed1.json` | A | 07 | `a_rank_cost/A_rank_cost__Qwen3-1.7B__seed1.json` |
| `background/precision_check/A_rank-cost_refusal-directions_Qwen3.5-2B_seed1_fp32.json` | A |  | `a_rank_cost/precision_check/A_rank_cost__Qwen3.5-2B__seed1.fp32.json` |
| `background/precision_check/A_rank-cost_refusal-directions_Qwen3-1.7B_seed1_fp32.json` | A |  | `a_rank_cost/precision_check/A_rank_cost__Qwen3-1.7B__seed1.fp32.json` |
| `background/logs/A_rank-cost_refusal-directions_bf16.log` | A |  | `a_rank_cost/logs/fullA_bf16.log` |
| `background/B_raw-refusal-alarm_Qwen3.5-0.8B_seed0.json` | B | 07 | `b_sensor/B_sensor__Qwen3.5-0.8B__seed0.json` |
| `background/B_raw-refusal-alarm_Qwen3.5-0.8B_seed1.json` | B | 07 | `b_sensor/B_sensor__Qwen3.5-0.8B__seed1.json` |
| `background/logs/B_raw-refusal-alarm.log` | B |  | `b_sensor/logs/rerunB.log` |
| `background/C_refusal-vs-value-directions_Qwen3-1.7B_seed0.json` | C | 07 | `c_value_geometry/C_geometry__Qwen3-1.7B__seed0.json` |
| `background/C_refusal-vs-value-directions_Qwen3-1.7B_seed1.json` | C | 07 | `c_value_geometry/C_geometry__Qwen3-1.7B__seed1.json` |
| `background/C_refusal-vs-value-directions_Qwen3.5-0.8B_seed0.json` | C | 07 | `c_value_geometry/C_geometry__Qwen3.5-0.8B__seed0.json` |
| `background/C_refusal-vs-value-directions_Qwen3.5-0.8B_seed1.json` | C | 07 | `c_value_geometry/C_geometry__Qwen3.5-0.8B__seed1.json` |
| `background/C_refusal-vs-value-directions_Qwen3.5-2B_seed0.json` | C | 07 | `c_value_geometry/C_geometry__Qwen3.5-2B__seed0.json` |
| `background/C_refusal-vs-value-directions_Qwen3.5-2B_seed1.json` | C | 07 | `c_value_geometry/C_geometry__Qwen3.5-2B__seed1.json` |
| `infrastructure/fast-kernel-reference-logits_Qwen3.5-0.8B.json` | infrastructure |  | `infrastructure/kernel_reference_0.8B.json` |
