# Sleep2Wave Generative Data Functions

## `sleep2wave.data.modalities`

- File: `sleep2wave/data/modalities.py`
- Symbols:
  - `CANONICAL_MODALITIES`
  - `MODALITY_SPECS`
  - `MODALITY_ALIASES`
  - `normalize_modality_name(name: str) -> str`
  - `get_modality_spec(name: str) -> ModalitySpec`
  - `validate_modality_sequence(modalities, *, allow_aliases: bool = False) -> list[str]`
- Purpose and contract: define canonical Sleep2Wave modalities, sample rates, frames per epoch, and accepted aliases.
- Important inputs/outputs: modality names in; canonical names/specs out.
- Side effects: none.
- Key callers/callees: config parser, dataset, model, task sampler, generation CLI, evaluation metrics.
- Reuse guidance: use these constants and validators for all modality logic.
- Duplication-risk notes: do not maintain separate modality order or sample-rate tables in configs or tests.

## `sleep2wave.data.generative_dataset.build_sample_indices_from_frame`

- File: `sleep2wave/data/generative_dataset.py`
- Signature: `build_sample_indices_from_frame(df: pd.DataFrame, *, index_source: str, split: str | Sequence[str] | None = None, context_epochs: int, stride_epochs: int | None = None, columns: IndexColumnConfig = IndexColumnConfig(), require_all_masks: bool = True) -> list[SampleIndex]`
- Purpose and contract: convert an index DataFrame into schema-versioned fixed-context Sleep2Wave `SampleIndex` windows.
- Important inputs/outputs: index rows with `path`, `duration`, `split`, `subject_id`, `night_id`, and modality mask columns in; `SampleIndex` list out.
- Side effects: none.
- Key callers/callees: `build_sample_indices_from_index`, `build_sleep2wave_presets`, `Sleep2WaveGenerativeDataset`; callees include `validate_subject_split_boundaries`, `resolve_modality_mask_columns`, and `normalize_mask_frame`.
- Reuse guidance: use this for every index-to-preset path.
- Duplication-risk notes: this is the canonical place for Sleep2Wave preset payload schema.

## `sleep2wave.data.generative_dataset.build_sample_indices_from_index`

- File: `sleep2wave/data/generative_dataset.py`
- Signature: `build_sample_indices_from_index(index_path: str | Path, *, split: str | Sequence[str] | None = None, context_epochs: int, stride_epochs: int | None = None, columns: IndexColumnConfig = IndexColumnConfig(), require_all_masks: bool = True) -> list[SampleIndex]`
- Purpose and contract: read a CSV index and delegate window construction to `build_sample_indices_from_frame`.
- Important inputs/outputs: CSV path in; `SampleIndex` list out.
- Side effects: reads CSV.
- Reuse guidance: use when callers have an index CSV instead of an existing preset.

## `sleep2wave.data.generative_dataset.Sleep2WaveGenerativeDataset`

- File: `sleep2wave/data/generative_dataset.py`
- Signature: `Sleep2WaveGenerativeDataset(*, preset_path=None, index=None, split=None, context_epochs=15, stride_epochs=None, condition_modalities=None, target_modalities=None, task_type="translation", corruption_name=None, corruption_kwargs=None, seed=0)`
- Purpose and contract: materialize Sleep2Wave waveform windows and masks from either a preset or index.
- Important inputs/outputs: exactly one data source in; batch samples with `clean_signals`, `observed_signals`, `availability_mask`, `quality_mask`, `corruption_mask`, `epoch_index`, `night_position`, `metadata`, and task modality fields out.
- Side effects: reads preset pickle, CSV, and NPZ files.
- Key callers/callees: training entrypoints and generation; callees include `load_npz`, `resolve_availability_mask`, `resolve_quality_mask`, and `apply_corruption`.
- Reuse guidance: use this dataset for autoencoder, diffusion, and generation data loading.
- Duplication-risk notes: do not add NPZ slicing logic in entrypoints.

## `sleep2wave.data.generative_batch.collate_sleep2wave_generative`

- File: `sleep2wave/data/generative_batch.py`
- Signature: `collate_sleep2wave_generative(samples: list[dict[str, Any]]) -> dict[str, Any]`
- Purpose and contract: stack signal and mask dictionaries from dataset samples, padding channel dimensions where needed.
- Important inputs/outputs: per-sample dicts in; batched dict out.
- Side effects: none.
- Key callers/callees: used by `Sleep2WaveGenerativeDataset.dataloader`.
- Reuse guidance: pass this as the collate function for every Sleep2Wave generative DataLoader.

## `sleep2wave.data.corruptions.apply_corruption`

- File: `sleep2wave/data/corruptions.py`
- Signature: `apply_corruption(name: str, signal: torch.Tensor, *, seed: int | None = None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]`
- Purpose and contract: apply named synthetic corruption and return the observed signal plus corruption mask.
- Important inputs/outputs: signal tensor in; corrupted tensor and bool mask out.
- Side effects: random generation controlled by seed.
- Key callers/callees: `Sleep2WaveGenerativeDataset._maybe_corrupt`.
- Reuse guidance: add new corruption functions here and route through `apply_corruption`.

## `sleep2wave.data.quality.resolve_availability_mask` and `resolve_quality_mask`

- File: `sleep2wave/data/quality.py`
- Signatures:
  - `resolve_availability_mask(npz, key: str | None, start: int, end: int, *, available: bool) -> torch.Tensor`
  - `resolve_quality_mask(npz, key: str | None, start: int, end: int, *, available: bool) -> torch.Tensor`
- Purpose and contract: produce per-epoch availability and quality masks for each modality.
- Important inputs/outputs: NPZ source and optional mask key in; `[context_epochs]` tensor out.
- Side effects: reads arrays from NPZ.
- Reuse guidance: keep mask fallback behavior here rather than duplicating it in datasets.

## `sleep2wave.data.derivations.validate_subject_split_boundaries`

- File: `sleep2wave/data/derivations.py`
- Signature: `validate_subject_split_boundaries(df: pd.DataFrame, *, subject_id_col: str = "subject_id", split_col: str = "split") -> None`
- Purpose and contract: reject missing subject/split values and subjects appearing in multiple splits.
- Important inputs/outputs: index DataFrame in; raises on invalid split contract.
- Side effects: none.
- Key callers/callees: generative index builder and derivation planning.
- Reuse guidance: call before any Sleep2Wave index operation that might split by subject.

## `sleep2wave.preprocess.build_sleep2wave_presets.build_sleep2wave_presets`

- File: `sleep2wave/preprocess/build_sleep2wave_presets.py`
- Signature: `build_sleep2wave_presets(*, index_path: Path, output_path: Path, split: list[str] | None, context_epochs: int, stride_epochs: int | None, columns, dry_run: bool = False) -> list`
- Purpose and contract: build and optionally write schema-versioned Sleep2Wave generative preset pickles.
- Important inputs/outputs: CSV index path and output path in; generated samples out.
- Side effects: reads CSV, writes pickle unless `dry_run=True`.
- Key callers/callees: CLI `main`; callee is `build_sample_indices_from_frame`.
- Reuse guidance: use this for reproducible preset generation.

## `sleep2wave.preprocess.validate_sleep2wave_index.validate_sleep2wave_index`

- File: `sleep2wave/preprocess/validate_sleep2wave_index.py`
- Signature: `validate_sleep2wave_index(index_path: Path) -> None`
- Purpose and contract: validate Sleep2Wave index column and modality-mask readiness.
- Important inputs/outputs: CSV path in; raises on invalid index.
- Side effects: reads CSV.
- Reuse guidance: run before large preset jobs.

## Tests

- `tests/test_sleep2wave_modalities.py`
- `tests/test_sleep2wave_generative_dataset.py`
- `tests/test_sleep2wave_corruptions.py`
- `tests/test_sleep2wave_preprocess_contract.py`
