# Robustness Evaluation Design (Controlled Domain Shift)

This design keeps **training and evaluation fully separated**:
- Training env: unchanged (clean source domain).
- Evaluation env: wrapped by shift wrappers per scenario.

## 1) Wrapper-based implementation

Yes — wrapper-based is the best fit because it keeps shift logic isolated and composable.

### Files
- `src/vrl_dissertation/envs/shifts/base.py`
  - defines `ShiftSpec`.
- `src/vrl_dissertation/envs/shifts/spatial.py`
  - pixel corruption functions.
- `src/vrl_dissertation/envs/shifts/temporal.py`
  - stateful temporal interruption module.
- `src/vrl_dissertation/envs/shifts/physics.py`
  - physics-shift applier (backend adapter point).
- `src/vrl_dissertation/envs/shifts/wrappers.py`
  - `ShiftedEnvWrapper` to inject one scenario into an eval env.
- `src/vrl_dissertation/envs/shifts/scenario_registry.py`
  - standardized scenario list for reproducible runs.
- `src/vrl_dissertation/evaluation/protocol.py`
  - evaluation protocol object and seed mapping.

## 2) Parameterization design (per corruption)

### Spatial corruption
1. `gaussian_noise`
   - parameter: `sigma` (std in pixel space 0-255).
2. `color_distortion`
   - parameters: `brightness` additive offset, `contrast` multiplicative factor.
3. `texture_variation`
   - parameters: `alpha` overlay strength, `block_size` texture granularity.

### Temporal interruption
1. `frame_freeze`
   - parameter: `freeze_prob`.
2. `blackout`
   - parameter: `blackout_prob`.
3. `frame_delay`
   - parameter: `delay_k` (frame lag steps).

### Physics parameter shift
1. `friction_scale`
   - parameter: `scale`.
2. `mass_scale`
   - parameter: `scale`.
3. `action_limit_scale`
   - parameter: `scale`.

> Physics application is env-family-specific. Keep current `apply_physics_shift` as generic hook and implement exact API calls in your DMC adapter next.

## 3) Separation of training vs evaluation protocols

- `train_sac_pixel.py` should continue to train on clean env only.
- Evaluation should instantiate:
  1) clean eval env,
  2) shifted eval env = `ShiftedEnvWrapper(clean_env, spec, seed)`.
- Use `EvalProtocol` from `evaluation/protocol.py` for fixed episode counts and deterministic scenario seeds.

## 4) Reproducibility and multi-seed comparability

- Every scenario has a stable `scenario_id` in `ShiftSpec`.
- Scenario seed is deterministic via `protocol.scenario_seed(idx)`.
- Run naming convention: `scenario_id/seed_k` from `scenario_to_run_name`.
- Keep the same trained checkpoint for all scenario evaluations in a seed.
- Aggregate by `(scenario_id, seed)` and then compute mean/std across seeds.

## 5) Minimal corruption settings to test first

Use these first (already encoded in `scenario_registry.py` and `configs/shifts/minimal_robustness_v1.json`):

### Spatial
- Gaussian noise: `sigma = {8, 16}`
- Color distortion: `(brightness, contrast) = {(8, 0.9), (16, 0.8)}`

### Temporal
- Frame freeze: `freeze_prob = {0.10, 0.20}`
- Frame delay: `delay_k = {1, 2}`

### Physics
- Friction scale: `{1.2, 1.5}`
- Mass scale: `{1.2, 1.5}`

This gives a compact yet representative first robustness suite.

## Implementation steps (next PRs)

1. Add `scripts/evaluate_robustness.py`:
   - load checkpoint, load scenario registry, run clean + shifted eval.
2. Write per-scenario outputs:
   - `reward_mean`, `reward_std`, `episode_len_mean`, `survival_ratio`.
3. Add `scripts/aggregate_robustness.py`:
   - aggregate across seeds and compute degradation vs clean baseline.
4. Add plotting:
   - robustness degradation bars per category/severity.
