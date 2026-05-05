# sleep2wave Evaluation Functions

## `sleep2wave.evaluate_generation.run_evaluation`

- File: `sleep2wave/evaluate_generation.py`
- Signature: `run_evaluation(args: argparse.Namespace) -> Path`
- Purpose and contract: evaluate generated sleep2wave artifact directories and write metrics output.
- Important inputs/outputs: evaluation-stage config and optional CLI overrides in, including `evaluation.corruption_mask_policy`; output directory path out.
- Side effects: reads config, artifact NPZ/JSON files, optional reference/baseline/events/downstream files, and writes `metrics.json` and `metrics.csv`.
- Key callers/callees: `main`; callees include `_require_artifact_dir`, `_load_generated_mean`, `_load_metric_epoch_mask`, waveform/feature/event/efficiency/downstream metric helpers, and `_write_metrics`.
- Reuse guidance: use this as the evaluation orchestration path.
- Duplication-risk notes: generated artifact schema assumptions live here and must stay aligned with `generate.py` and `export/artifacts.py`.

## `sleep2wave.evaluate_generation._load_metric_epoch_mask`

- File: `sleep2wave/evaluate_generation.py`
- Signature: `_load_metric_epoch_mask(masks_npz: np.lib.npyio.NpzFile, modality: str, epoch_count: int, *, corruption_mask_policy: str = "exclude") -> np.ndarray`
- Purpose and contract: derive the epoch mask used for metric computation from target, availability, quality, and corruption masks. `exclude` keeps translation-style behavior, `include` ignores corruption masks, and `only_corrupted` scores only corrupted epochs.
- Important inputs/outputs: masks NPZ and modality in; bool epoch mask out.
- Side effects: none.
- Reuse guidance: keep metric masking policy here.

## `sleep2wave.evaluate_generation._write_metrics`

- File: `sleep2wave/evaluate_generation.py`
- Signature: `_write_metrics(output_dir: Path, payload: dict[str, Any]) -> None`
- Purpose and contract: write nested metrics to JSON plus flattened family/modality/metric CSV.
- Important inputs/outputs: output directory and payload in; files on disk out.
- Side effects: creates directory and writes files.
- Reuse guidance: use for evaluation output schema changes.

## `sleep2wave.evaluation.waveform_metrics.compute_waveform_metrics`

- File: `sleep2wave/evaluation/waveform_metrics.py`
- Signature: `compute_waveform_metrics(reference: Any, generated: Any, *, baseline: Any | None = None, max_shift_frames: int = 0) -> dict[str, float]`
- Purpose and contract: compute waveform-level RMSE, MAE, correlation, spectral distance, optional shifted metrics, and optional SNR improvement.
- Important inputs/outputs: reference/generated arrays and optional baseline in; metric dict out.
- Side effects: none.
- Key callers/callees: `evaluate_generation.run_evaluation`.
- Reuse guidance: add waveform metrics here, not in the orchestration file.

## `sleep2wave.evaluation.feature_metrics.compute_feature_metrics`

- File: `sleep2wave/evaluation/feature_metrics.py`
- Signature: `compute_feature_metrics(modality: str, reference: Any, generated: Any, *, sample_rate_hz: int) -> dict[str, float]`
- Purpose and contract: compute modality-specific feature metrics such as EEG bandpower error, EMG tone error, IBI MAE, SpO2 nadir metrics, respiratory amplitude error, and ECG peak metrics when applicable.
- Important inputs/outputs: modality and arrays in; metric dict out.
- Side effects: none.
- Key callers/callees: `evaluate_generation.run_evaluation`.
- Reuse guidance: keep feature extraction here.

## `sleep2wave.evaluation.event_metrics.compute_event_metric_groups`

- File: `sleep2wave/evaluation/event_metrics.py`
- Signature: `compute_event_metric_groups(groups: Mapping[str, Any], *, iou_threshold: float = 0.5) -> dict[str, Any]`
- Purpose and contract: compute grouped event metrics from event-interval payloads.
- Important inputs/outputs: event groups in; metric dict out.
- Side effects: none.
- Key callers/callees: `evaluate_generation.run_evaluation`.
- Reuse guidance: use for event-family evaluation.

## `sleep2wave.evaluation.event_metrics.compute_generated_signal_event_groups`

- File: `sleep2wave/evaluation/event_metrics.py`
- Signature: `compute_generated_signal_event_groups(reference_by_modality, generated_by_modality, *, sample_rates, iou_threshold) -> dict[str, Any]`
- Purpose and contract: derive simple event intervals from generated/reference SpO2 and respiratory waveforms when no external events JSON is provided.
- Important inputs/outputs: modality arrays in; desaturation and low-amplitude respiratory event metrics out.
- Side effects: none.
- Reuse guidance: use as the generated-signal adapter before adding heavier clinical event detectors.

## `sleep2wave.evaluation.efficiency.summarize_generation_efficiency`

- File: `sleep2wave/evaluation/efficiency.py`
- Signature: `summarize_generation_efficiency(manifest: dict[str, Any], generated_npz: np.lib.npyio.NpzFile, uncertainty_npz: np.lib.npyio.NpzFile) -> dict[str, Any]`
- Purpose and contract: summarize generated modality counts, sample counts, and artifact-level efficiency metadata.
- Important inputs/outputs: manifest and artifact NPZs in; summary dict out.
- Side effects: none.
- Key callers/callees: `evaluate_generation.run_evaluation`.
- Reuse guidance: use for the `efficiency` metric family.

## `sleep2wave.evaluation.downstream_hooks.load_downstream_metrics`

- File: `sleep2wave/evaluation/downstream_hooks.py`
- Signature: `load_downstream_metrics(path: str | Path | None) -> dict[str, Any]`
- Purpose and contract: load optional downstream metrics JSON for evaluation aggregation.
- Important inputs/outputs: optional path in; metrics dict out.
- Side effects: reads JSON when provided.
- Key callers/callees: `evaluate_generation.run_evaluation`.
- Reuse guidance: keep downstream metric import behavior here.

## Tests

- `tests/test_sleep2wave_evaluate_cli.py`
- `tests/test_sleep2wave_waveform_metrics.py`
- `tests/test_sleep2wave_feature_metrics.py`
- `tests/test_sleep2wave_event_metrics.py`
