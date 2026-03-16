# Repository Architecture Blueprint

This blueprint is tailored for the dissertation scope:
- pixel-based SAC training
- modular data augmentation
- corruption/domain-shift robustness evaluation
- multiple environments
- multi-seed experiment management
- logging/checkpoint/plotting
- future algorithm comparison (SAC, DrQ-v2, TD-MPC2)

## 1) Folder structure

```text
.
├── README.md
├── pyproject.toml
├── configs/
│   ├── defaults/
│   │   ├── train.json
│   │   ├── eval.json
│   │   └── logger.json
│   ├── algorithms/
│   │   ├── sac_pixel.json
│   │   ├── drqv2_pixel.json
│   │   └── tdmpc2_pixel.json
│   ├── augmentations/
│   │   ├── none.json
│   │   ├── random_shift.json
│   │   ├── color_jitter.json
│   │   └── gaussian_noise.json
│   ├── envs/
│   │   ├── dmc_cartpole_swingup.json
│   │   ├── dmc_cheetah_run.json
│   │   └── atari_pong.json
│   ├── shifts/
│   │   ├── spatial/
│   │   │   ├── gaussian_noise_levels.json
│   │   │   ├── color_distortion_levels.json
│   │   │   └── texture_variation_levels.json
│   │   ├── temporal/
│   │   │   ├── frame_freeze_levels.json
│   │   │   ├── blackout_levels.json
│   │   │   └── frame_delay_levels.json
│   │   └── physics/
│   │       ├── friction_shift_levels.json
│   │       ├── load_shift_levels.json
│   │       └── motion_constraint_levels.json
│   └── experiments/
│       ├── sac_rs_cartpole_200k.json
│       ├── sac_none_cartpole_200k.json
│       └── drqv2_rs_cartpole_200k.json
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── launch_multiseed.py
│   ├── aggregate_results.py
│   ├── plot_learning_curves.py
│   └── plot_robustness_bars.py
├── src/vrl_dissertation/
│   ├── config/
│   │   ├── schema.py
│   │   ├── loader.py
│   │   └── validate.py
│   ├── agents/
│   │   ├── common/
│   │   │   ├── replay_buffer.py
│   │   │   ├── encoder.py
│   │   │   ├── networks.py
│   │   │   └── utils.py
│   │   ├── sac/
│   │   │   ├── agent.py
│   │   │   └── update.py
│   │   ├── drqv2/
│   │   │   └── agent.py
│   │   └── tdmpc2/
│   │       └── agent.py
│   ├── augmentations/
│   │   ├── base.py
│   │   ├── random_shift.py
│   │   ├── color_jitter.py
│   │   ├── gaussian_noise.py
│   │   └── registry.py
│   ├── envs/
│   │   ├── builders/
│   │   │   ├── dmc_builder.py
│   │   │   ├── atari_builder.py
│   │   │   └── registry.py
│   │   ├── wrappers/
│   │   │   ├── frame_stack.py
│   │   │   ├── action_repeat.py
│   │   │   ├── pixel_observation.py
│   │   │   └── time_limit.py
│   │   └── shifts/
│   │       ├── spatial.py
│   │       ├── temporal.py
│   │       ├── physics.py
│   │       └── scenario_registry.py
│   ├── evaluation/
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   ├── robustness_stats.py
│   │   └── reporting.py
│   ├── experiments/
│   │   ├── launcher.py
│   │   ├── seed_manager.py
│   │   └── run_naming.py
│   ├── logging/
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   ├── csv_writer.py
│   │   └── jsonl_writer.py
│   └── utils/
│       ├── seed.py
│       ├── device.py
│       └── timing.py
├── tests/
│   ├── test_config_validation.py
│   ├── test_augmentation_registry.py
│   ├── test_metrics.py
│   └── test_seed_reproducibility.py
└── runs/
    └── (generated artifacts, ignored by git)
```

## 2) Purpose of each key file

### Root files
- `README.md`: quick-start, command examples, experiment matrix.
- `pyproject.toml`: package/dependency/tooling entry.

### `configs/`
- `defaults/train.json`: shared train defaults (steps, warmup, update freq).
- `defaults/eval.json`: eval defaults (episodes, interval, scenario set).
- `defaults/logger.json`: logging/checkpoint intervals.
- `algorithms/*.json`: algorithm-specific hyperparameters.
- `augmentations/*.json`: augmentation hyperparameters.
- `envs/*.json`: per-environment settings.
- `shifts/**`: corruption/shift severity definitions.
- `experiments/*.json`: resolved experiment recipes for exact reproducibility.

### `scripts/`
- `train.py`: single-run training entrypoint.
- `evaluate.py`: evaluate saved policy in clean + shifted environments.
- `launch_multiseed.py`: run same experiment across N seeds.
- `aggregate_results.py`: aggregate multi-seed metrics and CI.
- `plot_learning_curves.py`: training/eval learning curves.
- `plot_robustness_bars.py`: robustness degradation bar charts.

### `src/vrl_dissertation/config/`
- `schema.py`: typed dataclasses/pydantic-style schema.
- `loader.py`: config composition/merging.
- `validate.py`: constraints and sanity checks.

### `src/vrl_dissertation/agents/`
- `common/replay_buffer.py`: shared replay for SAC/DrQ-v2/TD-MPC2.
- `common/encoder.py`: pixel encoder backbone.
- `common/networks.py`: shared actor/critic modules.
- `sac/agent.py`: SAC orchestration logic.
- `sac/update.py`: SAC update equations (critic/actor/alpha).
- `drqv2/agent.py`: future DrQ-v2 implementation (compatible API).
- `tdmpc2/agent.py`: future TD-MPC2 implementation (compatible API).

### `src/vrl_dissertation/augmentations/`
- `base.py`: augmentation interface.
- `random_shift.py`: random shift implementation.
- `registry.py`: name->augmentation factory.

### `src/vrl_dissertation/envs/`
- `builders/*`: instantiate env families.
- `wrappers/*`: frame-stack/action-repeat/pixel preprocessing.
- `shifts/spatial.py`: noise/color/texture perturbations.
- `shifts/temporal.py`: freeze/blackout/delay injection.
- `shifts/physics.py`: friction/load/constraint changes.
- `shifts/scenario_registry.py`: scenario set definitions.

### `src/vrl_dissertation/evaluation/`
- `runner.py`: rollout in clean/shifted envs.
- `metrics.py`: reward, episode length, SR.
- `robustness_stats.py`: degradation and aggregate stats.
- `reporting.py`: export summary tables.

### `src/vrl_dissertation/experiments/`
- `launcher.py`: single/multi job orchestration.
- `seed_manager.py`: deterministic seed assignment.
- `run_naming.py`: canonical run folder naming.

### `src/vrl_dissertation/logging/`
- `logger.py`: unified scalar logger.
- `checkpoint.py`: save/load periodic + best checkpoint.
- `csv_writer.py` / `jsonl_writer.py`: durable machine-readable logs.

### `tests/`
- small but critical tests for reproducibility and correctness.

### `runs/`
- training outputs and evaluation artifacts.

## 3) Implementation order (practical)

1. **Config core first**
   - `src/vrl_dissertation/config/schema.py`
   - `src/vrl_dissertation/config/loader.py`
   - `src/vrl_dissertation/config/validate.py`
   - `configs/experiments/sac_none_cartpole_50k.json`

2. **Minimal env and wrappers**
   - `envs/builders/dmc_builder.py`
   - `envs/wrappers/frame_stack.py`, `action_repeat.py`, `pixel_observation.py`

3. **Minimal SAC trainable baseline**
   - `agents/common/replay_buffer.py`
   - `agents/common/encoder.py`
   - `agents/sac/agent.py`, `agents/sac/update.py`
   - `scripts/train.py`

4. **Augmentation module (Random Shift first)**
   - `augmentations/base.py`
   - `augmentations/random_shift.py`
   - `augmentations/registry.py`
   - connect to SAC update path

5. **Evaluation + metrics**
   - `evaluation/runner.py`, `metrics.py`
   - `scripts/evaluate.py`

6. **Robustness shifts**
   - `envs/shifts/spatial.py`
   - `envs/shifts/temporal.py`
   - `envs/shifts/physics.py`
   - `scenario_registry.py`

7. **Multi-seed + aggregation + plotting**
   - `scripts/launch_multiseed.py`
   - `scripts/aggregate_results.py`
   - `scripts/plot_learning_curves.py`, `plot_robustness_bars.py`

8. **Future algorithm plug-ins**
   - `agents/drqv2/agent.py`, `agents/tdmpc2/agent.py`

## 4) Which file to create first

Create **`src/vrl_dissertation/config/schema.py` first**.

Why this file first:
- all training/evaluation scripts depend on a stable config contract,
- it prevents ad-hoc arguments from diverging across SAC/DrQ-v2/TD-MPC2,
- it is the foundation for reproducibility, multi-seed orchestration, and logging conventions.
