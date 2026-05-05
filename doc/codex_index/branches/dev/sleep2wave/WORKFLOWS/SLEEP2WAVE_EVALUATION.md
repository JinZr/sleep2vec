# sleep2wave Evaluation Workflow

## Purpose

Evaluate generated sleep2wave artifact directories across waveform, feature, event, efficiency, and downstream metric families.

## Canonical Path

1. Load `stage: evaluation` config with `load_sleep2wave_config`.
2. Resolve optional CLI overrides.
3. Validate generated artifact directory:
   - `manifest.json`
   - `generated.npz`
   - `uncertainty.npz`
   - `masks.npz`
   - `metadata.jsonl`
4. Load optional reference, baseline, event, and downstream metrics files.
5. Compute requested metric families, applying `evaluation.corruption_mask_policy` to waveform/feature epoch masks.
6. Write `metrics.json` and `metrics.csv`.

## Command

```bash
python -m sleep2wave.evaluate_generation \
  --config configs/sleep2wave/sleep2wave_eval_tiny.yaml \
  --generated-dir outputs/sleep2wave_generate_run \
  --output-dir outputs/sleep2wave_eval_run
```

## Metric Families

- `waveform`: waveform RMSE, MAE, correlation, spectral distance, optional shifted metrics and SNR improvement
- `feature`: modality-specific feature metrics
- `event`: interval overlap/event metrics from events JSON
- `efficiency`: artifact/sample-count summary
- `downstream`: optional downstream metrics JSON passthrough

## Corruption Mask Policy

- `exclude`: default translation-style behavior; corrupted epochs are not scored.
- `include`: corruption masks do not affect metric masks.
- `only_corrupted`: only corrupted epochs are scored; missing corruption masks produce no scored epochs.

## Edit Hotspots

- Evaluation orchestration and output schema: `sleep2wave/evaluate_generation.py`
- Waveform metrics: `sleep2wave/evaluation/waveform_metrics.py`
- Feature metrics: `sleep2wave/evaluation/feature_metrics.py`
- Event metrics: `sleep2wave/evaluation/event_metrics.py`
- Efficiency metrics: `sleep2wave/evaluation/efficiency.py`
- Downstream hooks: `sleep2wave/evaluation/downstream_hooks.py`

## Tests

```bash
python3.10 -m pytest -q \
  tests/test_sleep2wave_evaluate_cli.py \
  tests/test_sleep2wave_waveform_metrics.py \
  tests/test_sleep2wave_feature_metrics.py \
  tests/test_sleep2wave_event_metrics.py
```
