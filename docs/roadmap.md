# Implementation Roadmap (Next Stage)

This roadmap is constrained to your dissertation scope and ordered for practical execution.

## Stage 0 (done in this commit): Foundation
- [x] Define repository layout.
- [x] Define typed experiment config schema.
- [x] Add config validation CLI.
- [x] Add baseline SAC+RandomShift config template.

## Stage 1: Pixel SAC Training Pipeline Hardening
- [ ] Implement `scripts/train_sac_pixel.py` using config schema.
- [ ] Standardize replay buffer, encoder, actor/critic update frequencies.
- [ ] Add deterministic seeding and device logging.
- [ ] Save checkpoints at interval + best model by eval reward.

## Stage 2: Modular Augmentation Interface
- [ ] Implement `augmentations/base.py` interface (`forward(obs)` + config).
- [ ] Implement `augmentations/random_shift.py`.
- [ ] Wire augmentation selection from config (none / random_shift).
- [ ] Ensure augmentation applied only where intended (typically critic input path).

## Stage 3: Long-Horizon Training (200k–500k)
- [ ] Add schedule and logging for long runs.
- [ ] Add periodic evaluation and early diagnostics.
- [ ] Add auto-resume from latest checkpoint.

## Stage 4: Robustness Environment Suite
- [ ] Spatial corruption wrappers (noise/color/texture).
- [ ] Temporal interruption wrappers (freeze/blackout/delay).
- [ ] Physics shift loaders (friction/load/motion constraints).
- [ ] Unified scenario registry with severity levels.

## Stage 5: Evaluation Metrics and Robustness Stats
- [ ] Episode reward.
- [ ] Episode length.
- [ ] Survival ratio (SR).
- [ ] Relative degradation vs in-domain baseline.
- [ ] Aggregate mean/std + confidence interval across seeds.

## Stage 6: Multi-seed Experiment Management
- [ ] Implement launcher for N seeds per config.
- [ ] Structured output folders: `runs/{exp_name}/seed_{k}`.
- [ ] Seed aggregation utility.

## Stage 7: Logging, Plotting, and Analysis
- [ ] Unified JSONL/CSV logging.
- [ ] Plot scripts for learning curves + robustness degradation bars.
- [ ] Comparison scripts for augmentation methods vs baselines.

## Stage 8: Reproducibility Surface
- [ ] One-command train/eval examples.
- [ ] Version capture (git SHA, package versions, config snapshot).
- [ ] README experiment matrix for dissertation tables/figures.

