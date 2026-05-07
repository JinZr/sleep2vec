# Visualization And Diagnostics

## `sleep2expert.routing_analysis.build_routing_rows`

- File: `sleep2expert/routing_analysis.py`
- Signature: `build_routing_rows(moe_aux, batch, args, expert_groups) -> list[dict[str, Any]]`
- Purpose and contract: convert `Sleep2vecPretrainModel.last_moe_aux` records into sample/modality/layer/expert CSV rows, aggregating top-k expert usage and router diagnostics while respecting padding, optional CLS, and downstream label grouping.
- Important inputs/outputs: MoE aux records, current batch, normalized task args, and expert-id-to-group lookup in; rows with the fixed routing CSV columns out. `usage_count` comes from top-k selections, while `mean_router_prob` comes from original router probabilities rather than renormalized top-k weights.
- Side effects: none.
- Key callers/callees: caller is `run_routing_analysis`; callees include local mask, label, token-axis alignment, entropy, and sample-context helpers.
- Reuse guidance: reuse this helper for route-collapse, expert-usage, modality, label, and site/source diagnostic tables.
- Duplication risk notes: keep routing analysis on the existing summary-row contract; optional heatmaps should be derived from these rows rather than adding a parallel token-dump path.

## `sleep2expert.routing_analysis.build_routing_usage_matrices`

- File: `sleep2expert/routing_analysis.py`
- Signature: `build_routing_usage_matrices(rows, moe_cfg) -> dict[str, np.ndarray]`
- Purpose and contract: aggregate routing CSV-style rows into one expert-usage-share matrix per modality, with rows ordered by configured MoE layer and columns ordered by expert id.
- Important inputs/outputs: routing rows and MoE config in; modality-to-matrix mapping out. Values are row-normalized usage shares over allowed experts, allowed-but-unused experts are zero, and unavailable experts are `NaN` for masked rendering.
- Side effects: none.
- Key callers/callees: caller is `write_routing_heatmaps`; callee is `_allowed_experts_for_modality`.
- Reuse guidance: use this when deriving MoE routing usage heatmaps from exported rows.
- Duplication risk notes: do not recompute routing usage from raw `last_moe_aux` in plotting code.

## `sleep2expert.routing_analysis.write_routing_heatmaps`

- File: `sleep2expert/routing_analysis.py`
- Signature: `write_routing_heatmaps(rows, moe_cfg, output_dir, *, split, analysis_tag, log_to_wandb=False) -> dict[str, Path]`
- Purpose and contract: render and save per-modality MoE routing usage heatmap PNGs, optionally logging the same images to W&B.
- Important inputs/outputs: routing rows, MoE config, output directory, split/tag metadata in; written PNG paths keyed by modality out.
- Side effects: creates the output directory, writes PNG files, and may log W&B images.
- Key callers/callees: caller is `run_routing_analysis`; callees include `build_routing_usage_matrices` and `sleep2expert.visualization.routing_heatmap.render_routing_usage_heatmap`.
- Reuse guidance: use this as the only routing heatmap output path.
- Duplication risk notes: keep W&B keys and file naming centralized here.

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
- Important inputs/outputs: cached pair sums and counts in; no return value.
- Side effects: Lightning logging and optional W&B image logging.
- Key callers/callees: called by Lightning; callees include `_build_matrix` and `render_pair_acc_heatmap`.
- Reuse guidance: keep pair-wise validation reporting in this callback.
- Duplication risk notes: do not reimplement per-pair metric aggregation in the Lightning module.

## `PairAccLoggerCallback.on_train_epoch_end`

- File: `sleep2vec/callbacks/pair_acc_logger.py`
- Signature: `PairAccLoggerCallback.on_train_epoch_end(self, trainer, pl_module) -> None`
- Purpose and contract: inspect the training batch sampler for pair-distribution statistics, log skew and optional unique-coverage metrics, and warn when distribution drift exceeds thresholds.
- Important inputs/outputs: trainer state in; no return value.
- Side effects: logging and W&B/Lightning metric emission.
- Key callers/callees: called by Lightning; callee is `_resolve_train_pair_sampler`.
- Reuse guidance: use this callback path to observe pair-first sampling behavior.
- Duplication risk notes: depends on sampler-side auxiliary APIs; keep those interfaces aligned.

## `sleep2vec.callbacks.progress_bar.build_distributed_ahi_progress_bar`

- File: `sleep2vec/callbacks/progress_bar.py`
- Signature: `build_distributed_ahi_progress_bar()`
- Purpose and contract: choose the Lightning progress-bar implementation used for multi-GPU AHI finetune runs.
- Important inputs/outputs: no inputs; returns a progress-bar callback instance.
- Side effects: callback instantiation only.
- Key callers/callees: caller is `sleep2vec.finetune.supervised`.
- Reuse guidance: use this helper rather than selecting AHI progress-bar classes directly in entrypoints.
- Duplication risk notes: AHI-specific progress-bar choice should remain centralized.

## `sleep2vec.visualization.downstream_eval.DownstreamEvalVisualizer.log`

- File: `sleep2vec/visualization/downstream_eval.py`
- Signature: `log(*, stage: str, preds: np.ndarray, targets: np.ndarray, is_classification: bool, output_dim: int | None, label_name: str, current_epoch: int, class_labels: Sequence[str] | None = None) -> None`
- Purpose and contract: log confusion matrices, ROC curves, or regression scatter plots for one downstream evaluation stage, gated by config and active W&B state.
- Important inputs/outputs: predictions, targets, and task metadata in; no return value.
- Side effects: emits W&B images and closes matplotlib figures.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callees include `render_confusion_matrix_heatmap`, `render_binary_roc_curve_plot`, and `render_regression_scatter_plot`.
- Reuse guidance: keep downstream evaluation plotting in this helper.
- Duplication risk notes: task-specific plot gating belongs here, not in trainer step code.

## `sleep2vec.visualization.downstream_eval.DownstreamEvalVisualizer.log_ahi_summary_scatter`

- File: `sleep2vec/visualization/downstream_eval.py`
- Signature: `log_ahi_summary_scatter(*, stage: str, preds: np.ndarray, targets: np.ndarray, label_name: str, current_epoch: int) -> None`
- Purpose and contract: log the AHI summary scatter plot used after event-based aggregation has produced one scalar prediction per recording.
- Important inputs/outputs: aggregated AHI arrays in; no return value.
- Side effects: emits a W&B image and closes the figure.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callee is `render_regression_scatter_plot`.
- Reuse guidance: use this rather than logging AHI summary plots manually in trainer code.
- Duplication risk notes: AHI summary plotting is a post-aggregation concern and should stay separate from generic classification plotting.

## `sleep2vec.visualization.downstream_eval_plots.render_confusion_matrix_heatmap`

- File: `sleep2vec/visualization/downstream_eval_plots.py`
- Signature: `render_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray, class_labels: Sequence[str], *, title: str = "Confusion Matrix", normalize_rows: bool = False, show_raw_counts: bool = False)`
- Purpose and contract: render a confusion-matrix heatmap, optionally row-normalized with percentage annotations and raw-count overlays.
- Important inputs/outputs: targets, predicted labels, and class-label names in; matplotlib figure out.
- Side effects: none beyond figure creation.
- Key callers/callees: caller is `DownstreamEvalVisualizer.log`; callee is `render_matrix_heatmap`.
- Reuse guidance: use this for downstream confusion matrices instead of open-coding `sklearn.metrics.confusion_matrix` plots.
- Duplication risk notes: row normalization and annotation formatting belong here.

## `sleep2vec.visualization.downstream_eval_plots.render_regression_scatter_plot`

- File: `sleep2vec/visualization/downstream_eval_plots.py`
- Signature: `render_regression_scatter_plot(targets: np.ndarray, preds: np.ndarray, *, title: str = "Prediction vs Target")`
- Purpose and contract: render the standardized regression scatter plot used for downstream scalar targets and AHI summary plots.
- Important inputs/outputs: targets and predictions in; matplotlib figure out.
- Side effects: none beyond figure creation.
- Key callers/callees: callers are `DownstreamEvalVisualizer.log` and `log_ahi_summary_scatter`; callee is `render_prediction_scatter`.
- Reuse guidance: use this for regression-style downstream plots.
- Duplication risk notes: preserve shared styling by reusing this helper.

## `sleep2vec.visualization.downstream_eval_plots.render_binary_roc_curve_plot`

- File: `sleep2vec/visualization/downstream_eval_plots.py`
- Signature: `render_binary_roc_curve_plot(targets: np.ndarray, preds: np.ndarray, *, title: str = "ROC Curve")`
- Purpose and contract: render a binary ROC curve from two-logit classification outputs, returning `None` when the target set lacks both classes.
- Important inputs/outputs: binary targets plus logits/probabilities in; matplotlib figure or `None` out.
- Side effects: none beyond figure creation.
- Key callers/callees: caller is `DownstreamEvalVisualizer.log`; callees include `binary_positive_scores_from_two_logits`, `roc_curve`, and `render_binary_roc_curve`.
- Reuse guidance: use this helper for binary downstream ROC plots.
- Duplication risk notes: positive-class score extraction belongs here.

## `sleep2vec.visualization.layer_mix.render_layer_mix_heatmap`

- File: `sleep2vec/visualization/layer_mix.py`
- Signature: `render_layer_mix_heatmap(matrix, modality_names, layer_ids, *, title="Layer-Mix Weights") -> plt.Figure`
- Purpose and contract: render a modality-by-layer heatmap and validate matrix shape against modality and layer labels.
- Important inputs/outputs: normalized weight matrix in; matplotlib figure out.
- Side effects: none beyond figure creation.
- Key callers/callees: caller is `Sleep2vecFinetuning._log_layer_mix_weights`.
- Reuse guidance: use for layer-mix visualization rather than open-coding heatmaps.
- Duplication risk notes: shape validation is part of the tested contract.

## `sleep2vec.visualization.routing_heatmap.render_routing_usage_heatmap`

- File: `sleep2vec/visualization/routing_heatmap.py` with package-local mirrors in `sleep2vec2/` and `sleep2expert/`
- Signature: `render_routing_usage_heatmap(matrix, layer_labels, expert_labels, *, title, colorbar_label="Expert usage share") -> plt.Figure`
- Purpose and contract: render white-background OpenAI-style MoE routing usage heatmaps with fixed `0..1` color range and gray masked cells for unavailable experts.
- Important inputs/outputs: matrix plus row/column labels in; matplotlib figure out. Matrix shape must match the labels, and `NaN` cells are rendered as masked unavailable expert cells.
- Side effects: none beyond figure creation.
- Key callers/callees: caller is `sleep2expert.routing_analysis.write_routing_heatmaps`; callee uses package-local visualization theme title/text styling, colors, colormap, and font selection.
- Reuse guidance: use this renderer for MoE routing heatmaps in each namespace instead of adapting confusion-matrix or layer-mix heatmaps.
- Duplication risk notes: preserve package-local copies for standalone variant boundaries.

## `sleep2vec.visualization.layer_mix.build_layer_mix_rows`

- File: `sleep2vec/visualization/layer_mix.py`
- Signature: `build_layer_mix_rows(stage, epoch, shared, layer_ids, effective_by_modality) -> list[dict[str, Any]]`
- Purpose and contract: flatten layer-mix weights into table rows suitable for W&B or CSV-like logging.
- Important inputs/outputs: snapshot metadata in; list of normalized rows out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._log_layer_mix_weights`.
- Reuse guidance: use when serializing layer-mix snapshots to tabular artifacts.
- Duplication risk notes: keep the row schema stable with the visualization logic.

## `sleep2vec.visualization.pair_acc.render_pair_acc_heatmap`

- File: `sleep2vec/visualization/pair_acc.py`
- Signature: `render_pair_acc_heatmap(matrix, modality_names, *, title="Alignment Accuracy (Top-1)") -> plt.Figure`
- Purpose and contract: render a symmetric modality-pair accuracy heatmap, forcing diagonal values to `1.0`.
- Important inputs/outputs: square accuracy matrix in; figure out.
- Side effects: none beyond figure creation.
- Key callers/callees: caller is `PairAccLoggerCallback.on_validation_epoch_end`.
- Reuse guidance: use for pair-alignment visualization.
- Duplication risk notes: matrix-shape validation belongs here, not in callers.

## `sleep2vec.visualization.theme.use_openai_like_theme`

- File: `sleep2vec/visualization/theme.py`
- Signature: `use_openai_like_theme() -> None`
- Purpose and contract: register local fonts and apply the shared plot style used by downstream evaluation figures.
- Important inputs/outputs: no inputs or return value.
- Side effects: mutates matplotlib global style state.
- Key callers/callees: used indirectly by the plot-rendering helpers under `sleep2vec/visualization/`.
- Reuse guidance: keep plot styling centralized here rather than restyling each figure separately.
- Duplication risk notes: plot-theme configuration should not be redefined in each render helper.
