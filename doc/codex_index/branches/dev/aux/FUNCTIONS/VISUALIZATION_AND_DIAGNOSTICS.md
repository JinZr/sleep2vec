# Visualization And Diagnostics

## `sleep2vec.diagnostics.attach_diagnostics`

- File: `sleep2vec/diagnostics.py`
- Signature: `attach_diagnostics(model, opts) -> ModelDiagnostic`
- Purpose and contract: install forward and backward hooks that accumulate tensor and gradient diagnostics for modules and parameters.
- Important inputs/outputs: model and diagnostic options in; `ModelDiagnostic` out.
- Side effects: registers hooks on the model.
- Key callers/callees: callers are `Sleep2vecPretraining.__init__` and `Sleep2vecFinetuning.__init__`.
- Reuse guidance: use when adding short diagnostic runs or tensor-stat reporting.
- Duplication risk notes: diagnostic hooks should remain centralized in this module.

## `PairAccLoggerCallback.on_validation_epoch_end`

- File: `sleep2vec/callbacks/pair_acc_logger.py`
- Signature: `PairAccLoggerCallback.on_validation_epoch_end(self, trainer, pl_module) -> None`
- Purpose and contract: aggregate pair-specific validation accuracies across ranks, log per-pair metrics, and optionally emit a pair-accuracy heatmap to W&B.
- Important inputs/outputs: cached per-loader sums and counts in; no return value.
- Side effects: Lightning logging and optional W&B image logging.
- Key callers/callees: called by Lightning; callees include `_build_matrix` and `render_pair_acc_heatmap`.
- Reuse guidance: keep pair-wise validation reporting in this callback.
- Duplication risk notes: do not reimplement per-pair metric aggregation in the Lightning module.

## `PairAccLoggerCallback.on_train_epoch_end`

- File: `sleep2vec/callbacks/pair_acc_logger.py`
- Signature: `PairAccLoggerCallback.on_train_epoch_end(self, trainer, pl_module) -> None`
- Purpose and contract: inspect the training batch sampler for pair-distribution statistics, log skew and optional unique-coverage metrics, and warn when distribution drift exceeds thresholds.
- Important inputs/outputs: trainer state in; no return value.
- Side effects: logging, W&B/Lightning metric emission.
- Key callers/callees: called by Lightning; callee `_resolve_train_pair_sampler`.
- Reuse guidance: use this callback path to observe pair-first sampling behavior.
- Duplication risk notes: depends on sampler-side auxiliary APIs; keep those interfaces aligned.

## `sleep2vec.visualization.layer_mix.render_layer_mix_heatmap`

- File: `sleep2vec/visualization/layer_mix.py`
- Signature: `render_layer_mix_heatmap(matrix, modality_names, layer_ids, *, title="Layer-Mix Weights") -> plt.Figure`
- Purpose and contract: render a modality-by-layer heatmap and validate matrix shape against modality and layer labels.
- Important inputs/outputs: normalized weight matrix in; matplotlib figure out.
- Side effects: none beyond figure creation.
- Key callers/callees: caller is `Sleep2vecFinetuning._log_layer_mix_weights`.
- Reuse guidance: use for layer-mix visualization rather than open-coding seaborn plots.
- Duplication risk notes: shape validation is part of the tested contract.

## `sleep2vec.visualization.layer_mix.build_layer_mix_rows`

- File: `sleep2vec/visualization/layer_mix.py`
- Signature: `build_layer_mix_rows(stage, epoch, shared, layer_ids, effective_by_modality) -> list[dict[str, Any]]`
- Purpose and contract: flatten layer-mix weights into table rows suitable for W&B or CSV-like logging.
- Important inputs/outputs: snapshot metadata in; list of normalized rows out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._log_layer_mix_weights`.
- Reuse guidance: use when serializing layer-mix snapshots to tabular artifacts.
- Duplication risk notes: keep the row schema stable with the visualization logic.

## `sleep2vec.visualization.downstream_eval.DownstreamEvalVisualizer.log_ahi_summary_scatter`

- File: `sleep2vec/visualization/downstream_eval.py`
- Signature: `log_ahi_summary_scatter(self, *, stage, preds, targets, label_name, current_epoch) -> None`
- Purpose and contract: emit the built-in AHI scalar-summary scatter plot (`true_ahi` vs `pred_ahi`) for val/test when `eval_visualizations.regression_scatter` is enabled.
- Important inputs/outputs: aggregated scalar AHI arrays in; no return value.
- Side effects: may log a W&B image under `<stage>_eval/regression_scatter`.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callee is `render_regression_scatter_plot`.
- Reuse guidance: use this AHI-specific path instead of trying to force built-in `ahi` through the generic non-sequence regression visualizer.
- Duplication risk notes: AHI final evaluation still bypasses the generic confusion-matrix / ROC visualizer path.

## `sleep2vec.visualization.pair_acc.render_pair_acc_heatmap`

- File: `sleep2vec/visualization/pair_acc.py`
- Signature: `render_pair_acc_heatmap(matrix, modality_names, *, title="Alignment Accuracy (Top-1)") -> plt.Figure`
- Purpose and contract: render a symmetric modality-pair accuracy heatmap, forcing diagonal values to `1.0`.
- Important inputs/outputs: square accuracy matrix in; figure out.
- Side effects: none beyond figure creation.
- Key callers/callees: caller is `PairAccLoggerCallback.on_validation_epoch_end`.
- Reuse guidance: use for pair-alignment visualization.
- Duplication risk notes: matrix-shape validation belongs here, not in callers.

## `sleep2vec.downstreams.temporal_aggregation.build_temporal_aggregator`

- File: `sleep2vec/downstreams/temporal_aggregation/__init__.py`
- Signature: `build_temporal_aggregator(name: str | None, hidden_size: int, **kwargs: Any) -> TemporalAggregator`
- Purpose and contract: resolve the named temporal aggregator, defaulting to mean pooling and only using `hidden_size` for attention pooling.
- Important inputs/outputs: aggregator name plus hidden size in; aggregator module out.
- Side effects: module instantiation only.
- Key callers/callees: caller is `Sleep2vecDownstreamModel.__init__`.
- Reuse guidance: all temporal pooling construction should flow through this helper.
- Duplication risk notes: it is the canonical name-to-module map for temporal aggregation.

## `sleep2vec.downstreams.channel_aggregation.build_channel_aggregator`

- File: `sleep2vec/downstreams/channel_aggregation/__init__.py`
- Signature: `build_channel_aggregator(name: str | None, feature_dim: int, n_mods: int, **kwargs: Any) -> ChannelAggregator`
- Purpose and contract: resolve the named cross-modality aggregator, defaulting to mean aggregation.
- Important inputs/outputs: aggregator name, feature dimension, and modality count in; aggregator module out.
- Side effects: module instantiation only.
- Key callers/callees: used by downstream head/fusion modules.
- Reuse guidance: all channel-fusion construction should go through this helper.
- Duplication risk notes: keep the supported aggregator names centralized here.
