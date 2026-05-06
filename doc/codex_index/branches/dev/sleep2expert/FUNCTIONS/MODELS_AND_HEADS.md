# Models And Heads

## `Sleep2vecPretrainModel.__init__`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel.__init__(..., model_config: ModelConfig | None = None, projection_config: ProjectionConfig | None = None)`
- Purpose and contract: build the shared backbone module, tokenizer mapping, CLS strategy, mask embeddings, hidden projection layer, and optional projection head.
- Important inputs/outputs: either a full `ModelConfig` or legacy manual dimensions/channel names; produces a ready-to-run module.
- Side effects: instantiates many submodules and logs parameter counts.
- Key callers/callees: callers are `Sleep2vecPretraining`, `Sleep2vecFinetuning`, and adaptation-related code; callees include `build_tokenizers_and_dim`, `build_encoder_factory`, `build_cls_embedding`, and `build_projection`.
- Reuse guidance: this is the canonical backbone construction path.
- Duplication risk notes: when `model_config is None`, a legacy hardcoded tokenizer map is used; do not extend that path for new work.

## `Sleep2vecPretrainModel._token_embeddings_to_hidden`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel._token_embeddings_to_hidden(token_embeddings, batch, *, return_hidden_states: bool = False) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]`
- Purpose and contract: project token features to transformer hidden size, prepend CLS if configured, build attention mask, and run the encoder.
- Important inputs/outputs: token embeddings plus batch lengths in; returns last hidden states, attention mask, and optionally all hidden states.
- Side effects: none beyond compute.
- Key callers/callees: callers are `Sleep2vecPretrainModel.forward`, `Sleep2vecPretrainModel.encode`, and `Sleep2vecDownstreamModel.forward`; callees are `apply_padding_mask` and `_run_encoder`.
- Reuse guidance: reuse this helper whenever downstream code needs encoded token states from already-tokenized features.
- Duplication risk notes: CLS and attention-mask handling must remain centralized here.

## `Sleep2vecPretrainModel.set_tokenizers_trainable`

- File: `sleep2vec/pretrain_model.py`
- Signature: `set_tokenizers_trainable(self, trainable: bool) -> None`
- Purpose and contract: freeze or unfreeze tokenizer parameters without changing module train/eval mode.
- Important inputs/outputs: boolean flag in; no return value.
- Side effects: mutates `requires_grad` flags on tokenizer parameters.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`.
- Reuse guidance: use this helper instead of mutating tokenizer parameters ad hoc.
- Duplication risk notes: tokenizer freeze policy should stay centralized.

## `Sleep2vecPretrainModel.get_adaptation_param_groups` and `apply_adaptation_freeze_policy`

- File: `sleep2vec/pretrain_model.py`
- Signatures:
  - `get_adaptation_param_groups(self, new_channels: Sequence[str]) -> dict[str, list[tuple[str, nn.Parameter]]]`
  - `apply_adaptation_freeze_policy(self, *, phase: str, new_channels: Sequence[str], train_shared_projection: bool = False) -> None`
- Purpose and contract: partition backbone parameters into encoder/CLS, shared projection, legacy modalities, and new modalities, then apply stage-specific trainability and module-mode policy.
- Important inputs/outputs: new-channel set and stage metadata in; no return value.
- Side effects: mutates `requires_grad`, records forced train/eval module lists, and logs trainable-parameter counts.
- Key callers/callees: callers are `Sleep2vecAdaptation.__init__` and `Sleep2vecAdaptation.train`; callees include `_resolve_adaptation_channels`, `_set_adaptation_group_trainable`, `_tokenizer_modules`, and `apply_forced_module_modes`.
- Reuse guidance: use this model-side policy for adaptation rather than creating a second freeze system in the trainer.
- Duplication risk notes: stage1 vs stage2 trainability boundaries belong here.

## `Sleep2vecPretrainModel.forward`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel.forward(batch, apply_mask)`
- Purpose and contract: choose two channels, optionally apply modality masking, encode each channel independently, and optionally project the hidden states for contrastive loss.
- Important inputs/outputs: batch with `tokens`, `mlm_mask`, and `length`; returns `(first_hidden, second_hidden)`.
- Side effects: random channel selection unless `specified_two_mods` is fixed.
- Key callers/callees: caller is `Sleep2vecPretraining._contrastive_step`; callees include `_tokenize_two_random_channels`, `_mask_modalities`, and `_token_embeddings_to_hidden`.
- Reuse guidance: this is the canonical contrastive pretrain forward path.
- Duplication risk notes: do not recreate channel-selection and masking flow in the Lightning module.

## `Sleep2vecDownstreamModel.__init__`

- File: `sleep2vec/downstream_model.py`
- Signature: `Sleep2vecDownstreamModel.__init__(target, backbone, channel_names, output_dim, is_classification, is_seq, ..., model_config, layer_mix_cfg, head_config)`
- Purpose and contract: wrap the shared backbone with temporal aggregation, downstream head creation, optional layer-mix support, and CLS / adapter compatibility checks.
- Important inputs/outputs: configured backbone plus task/head metadata in; downstream model out.
- Side effects: builds head modules and may raise when CLS semantics are inconsistent.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`; callees are `create_head`, `build_temporal_aggregator`, and `LayerMix`.
- Reuse guidance: this is the canonical downstream assembly surface.
- Duplication risk notes: head configuration should enter through `head_config`, not through one-off constructor branches.

## `Sleep2vecDownstreamModel.forward`

- File: `sleep2vec/downstream_model.py`
- Signature: `Sleep2vecDownstreamModel.forward(batch)`
- Purpose and contract: encode each modality through the shared backbone, then apply either sequence or pooled downstream logic, optional layer mixing, optional separate-adapter switching, and head prediction.
- Important inputs/outputs: batch with `tokens` and `length`; returns logits or regression outputs.
- Side effects: may switch active LoRA adapters when separate adapters are enabled; eval forwards collect `backbone.last_moe_aux` for MoE backbones, and train forwards collect it only when `collect_train_moe_aux` is enabled.
- Key callers/callees: caller is `Sleep2vecFinetuning._shared_step`; callees include `Sleep2vecPretrainModel._tokenize_all`, `_token_embeddings_to_hidden`, `_forward_seq`, `_forward_nonseq`, and `_call_head`.
- Reuse guidance: all downstream predictions should flow through this method.
- Duplication risk notes: sequence/non-sequence branching and CLS behavior are easy to get wrong if reimplemented elsewhere.

## `Sleep2vecDownstreamModel.load_pretrained_backbone`

- File: `sleep2vec/downstream_model.py`
- Signature: `Sleep2vecDownstreamModel.load_pretrained_backbone(ckpt_path, use_ema: bool | str | None = True) -> None`
- Purpose and contract: load pretraining weights into the backbone, optionally preferring an averaged-model prefix such as `ema_model.`.
- Important inputs/outputs: checkpoint path and averaging preference in; no return value.
- Side effects: loads checkpoint from disk, mutates backbone weights, emits warnings for CLS mismatches and dropped CLS weights.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`; callees include `load_checkpoint`, `_warn_on_cls_mismatch`, and `_warn_on_dropped_cls_weights`.
- Reuse guidance: use for any downstream weight initialization from pretrain checkpoints.
- Duplication risk notes: prefix selection and fallback behavior are non-trivial; do not slice checkpoint keys manually in callers.

## `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora`

- File: `sleep2vec/downstream_model.py`
- Signature: `freeze_backbone_and_insert_lora(insert_lora: bool = True, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, target_modules=("query", "key", "value"), separate_adapters: bool = False) -> None`
- Purpose and contract: freeze backbone parameters, optionally inject LoRA adapters into the encoder, and optionally create separate adapters per channel.
- Important inputs/outputs: LoRA config in; no return value.
- Side effects: mutates backbone module structure and parameter trainability.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`; callee is `get_peft_model`.
- Reuse guidance: this is the canonical LoRA insertion path for downstream training.
- Duplication risk notes: adapter naming and trainable-parameter enabling should stay centralized here.

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
- Key callers/callees: used by downstream head and fusion modules.
- Reuse guidance: all channel-fusion construction should go through this helper.
- Duplication risk notes: keep the supported aggregator names centralized here.

## `Sleep2vecPretraining._contrastive_step`

- File: `sleep2vec/sleep2vec_modelling.py`
- Signature: `Sleep2vecPretraining._contrastive_step(batch, log_prefix=None, model=None)`
- Purpose and contract: run the contrastive forward pass, apply the configured loss, add any variant-local MoE regularization, log step/epoch metrics, and cache validation metrics for epoch-end aggregation.
- Important inputs/outputs: batch in; returns `(total_loss, contrastive_acc)`.
- Side effects: logs Lightning metrics and mutates validation caches.
- Key callers/callees: callers are `training_step` and `validation_step`; callees are `self.model(...)`, `self.loss_fn(...)`, and for `sleep2expert` MoE configs `compute_moe_regularization(...)`.
- Reuse guidance: reuse this helper for any pretrain stage that wants identical loss/logging semantics.
- Duplication risk notes: pretrain metric naming and aggregation live here, not in the callback; keep auxiliary MoE losses outside the contrastive loss registry.

## `sleep2expert.losses.moe_regularization.compute_moe_regularization`

- File: `sleep2expert/losses/moe_regularization.py`
- Signature: `compute_moe_regularization(moe_aux, moe_cfg, batch, *, prefix=None) -> LossOutput`
- Purpose and contract: convert `Sleep2vecPretrainModel.last_moe_aux` routing records into MoE regularization losses and routing diagnostics without changing the contrastive loss API; enabled MoE must provide non-empty aux records.
- Important inputs/outputs: MoE aux records plus `MoeConfig` and batch in; returns `LossOutput` with scalar auxiliary loss and `moe_*` metrics. Modality balance is computed as per-modality load balance over each modality's accessible experts, not by matching distributions across modalities.
- Side effects: none.
- Key callers/callees: caller is `sleep2expert.sleep2vec_modelling.Sleep2vecPretraining._contrastive_step`; callees are tensor reductions over router load, importance, entropy, z-loss, and route probabilities.
- Reuse guidance: use this direct helper for `sleep2expert` MoE auxiliary losses; do not register it via `create_loss`.
- Duplication risk notes: route consistency depends on the two-view `last_moe_aux` contract, requires configured consistency layers when its coefficient is positive, and excludes CLS when the aux sequence includes a prepended CLS token.

## `sleep2expert.losses.moe_regularization.compute_downstream_moe_regularization` and `compute_downstream_moe_metrics`

- File: `sleep2expert/losses/moe_regularization.py`
- Signature: `compute_downstream_moe_regularization(moe_aux, reg_cfg, batch, *, prefix=None) -> LossOutput`
- Signature: `compute_downstream_moe_metrics(moe_aux, batch, *, prefix=None) -> dict[str, Tensor]`
- Purpose and contract: convert downstream routing aux records into opt-in router z-loss plus routing diagnostics for supervised finetuning, or metrics-only eval diagnostics when no loss should be added.
- Important inputs/outputs: downstream `backbone.last_moe_aux`, `finetune.moe_tuning.moe_regularization`, and batch in; returns `LossOutput` with scalar z-loss contribution and `downstream_moe_*` metrics.
- Important inputs/outputs: metrics-only eval path returns an empty dict when no aux is available, and otherwise returns scalar `downstream_moe_*` tensors without collecting routing rows.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._shared_step`; callees reuse the local MoE routing-stat helpers.
- Reuse guidance: use this helper for supervised downstream MoE diagnostics instead of the pretraining helper.
- Duplication risk notes: downstream route consistency, load balancing, modality balancing, and entropy losses are intentionally rejected for now because the pretraining route-consistency contract assumes two routing records.

## `Sleep2vecPretraining.configure_optimizers`

- File: `sleep2vec/sleep2vec_modelling.py`
- Signature: `Sleep2vecPretraining.configure_optimizers(self)`
- Purpose and contract: create AdamW parameter groups and a warmup-plus-cosine scheduler for pretraining.
- Important inputs/outputs: uses `self.args` and `self.trainer`; returns Lightning optimizer/scheduler structure.
- Side effects: none beyond object creation.
- Key callers/callees: called by Lightning.
- Reuse guidance: use as the reference optimizer schedule when aligning pretrain and finetune behavior.
- Duplication risk notes: this policy overlaps with finetune and adaptation schedules.

## `Sleep2vecAdaptation.configure_optimizers`

- File: `sleep2vec/sleep2vec_adaptation.py`
- Signature: `Sleep2vecAdaptation.configure_optimizers(self)`
- Purpose and contract: create stage-specific AdamW parameter groups for adaptation, with learning-rate scaling across encoder/CLS, shared legacy projection, and new modalities.
- Important inputs/outputs: uses `self.args`, `self.adapt_config`, and trainer step count; returns Lightning optimizer/scheduler structure.
- Side effects: none beyond object creation.
- Key callers/callees: called by Lightning; callees include `self.model.get_adaptation_param_groups`.
- Reuse guidance: use this as the reference optimizer path for adaptation-specific schedules.
- Duplication risk notes: stage1 vs stage2 group composition belongs here.

## `Sleep2vecFinetuning._shared_step`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._shared_step(batch, stage: str, model=None)`
- Purpose and contract: run forward, compute loss if valid labels exist, optionally add enabled downstream MoE router z-loss during train, log train/val/test scalar MoE diagnostics when aux is available, log stage loss or accumulate eval loss, and cache either generic predictions or AHI event records for epoch-end metrics.
- Important inputs/outputs: batch and stage name in; returns training loss for `train`, otherwise `None`.
- Side effects: logs Lightning metrics and mutates `_stage_outputs`.
- Key callers/callees: callers are `training_step`, `validation_step`, and `test_step`; callees are `self.model(...)`, `_compute_loss`, optional `compute_downstream_moe_regularization`, `_extract_valid_predictions`, and `_extract_ahi_event_records`.
- Reuse guidance: all downstream stage logic should continue to funnel through this helper.
- Duplication risk notes: stage branching should not leak into entrypoints.

## `Sleep2vecFinetuning._compute_loss`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._compute_loss(logits, batch)`
- Purpose and contract: compute classification, regression, or multilabel loss, ignoring invalid labels and returning `None` when the batch contains no valid targets.
- Important inputs/outputs: logits and batch in; returns `(loss, valid_count)` or `None`.
- Side effects: none.
- Key callers/callees: caller is `_shared_step`.
- Reuse guidance: this is the canonical downstream loss application logic.
- Duplication risk notes: ignore-index and valid-label filtering must stay aligned with `_extract_valid_predictions`.

## `Sleep2vecFinetuning._extract_ahi_event_records`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `_extract_ahi_event_records(self, batch, logits) -> list[dict[str, np.ndarray]]`
- Purpose and contract: convert one evaluation batch into per-sample AHI event records containing truth, score, path, token offset, true summary AHI, TST, and stage5 context.
- Important inputs/outputs: batch and logits in; list of per-sample evaluation records out.
- Side effects: none.
- Key callers/callees: caller is `_shared_step`; downstream consumers are `_compute_ahi_metrics_for_stage` and `extract_ahi_summary_scatter_arrays`.
- Reuse guidance: use this helper rather than rebuilding AHI records in metrics or entrypoints.
- Duplication risk notes: token-start and stage5 alignment must stay centralized.

## `Sleep2vecFinetuning._compute_ahi_metrics_for_stage` and `_compute_or_broadcast_ahi_metrics`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signatures:
  - `_compute_ahi_metrics_for_stage(self, stage: str, records: list[dict[str, np.ndarray]]) -> tuple[dict[str, float], float, tuple[np.ndarray, np.ndarray] | None]`
  - `_compute_or_broadcast_ahi_metrics(self, stage: str, records: list[dict[str, np.ndarray]]) -> tuple[dict[str, float], float, tuple[np.ndarray, np.ndarray] | None]`
- Purpose and contract: compute AHI event metrics and validation-fitted threshold locally or on rank zero and broadcast the result to other ranks.
- Important inputs/outputs: prepared record list in; metrics dict, threshold, and optional scatter arrays out.
- Side effects: stores `self._ahi_eval_threshold` on validation.
- Key callers/callees: caller is `_finalize_epoch`; callees include `_prepare_ahi_records`, `_compute_ahi_event_metrics_from_prepared`, `_aggregate_prepared_ahi_records`, and `extract_ahi_summary_scatter_arrays`.
- Reuse guidance: keep AHI threshold search and rank-aware result fan-out in these helpers.
- Duplication risk notes: do not compute AHI validation thresholds in entrypoints.

## `Sleep2vecFinetuning._finalize_epoch`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._finalize_epoch(stage: str)`
- Purpose and contract: reduce cached outputs for one stage, log generic or AHI-specific metrics, emit eval visualizations, and clear the stage cache.
- Important inputs/outputs: stage name in; returns predictions/targets or AHI records for downstream consumers, or `None` when nothing was cached.
- Side effects: logs metrics, writes W&B visualizations, updates the fitted AHI threshold, and clears cached outputs.
- Key callers/callees: callers are `on_train_epoch_end`, `on_validation_epoch_end`, and `on_test_epoch_end`; callees include `_compute_reduced_ahi_train_pointwise_metrics`, `_compute_or_broadcast_ahi_metrics`, `compute_downstream_metrics`, and `DownstreamEvalVisualizer`.
- Reuse guidance: use this pattern for any new stage-level downstream metrics or plots.
- Duplication risk notes: do not introduce separate metric reducers per stage.

## `Sleep2vecFinetuning.configure_optimizers`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning.configure_optimizers(self)`
- Purpose and contract: create AdamW parameter groups and a warmup-plus-cosine scheduler for finetuning. In `sleep2expert`, optional `finetune.moe_tuning` switches optimizer grouping to semantic downstream MoE groups while preserving the same scheduler.
- Important inputs/outputs: uses `self.args` and `self.trainer`; returns Lightning optimizer/scheduler structure.
- Side effects: none beyond object creation.
- Key callers/callees: called by Lightning.
- Reuse guidance: use as the finetune optimizer reference path; downstream MoE freeze/LR policy stays class-local in `Sleep2vecFinetuning`.
- Duplication risk notes: this policy overlaps with pretrain and adaptation schedules; avoid creating additional variants casually. Keep `moe_tuning=None` on the legacy two-group path.

## Loss factory family

- Files: `sleep2vec/losses/base.py`, `sleep2vec/losses/info_nce.py`, `sleep2vec/losses/weighted_info_nce.py`
- Functions and methods:
  - `create_loss(name: str, **kwargs) -> ContrastiveLoss`
  - `InfoNCELoss.forward(first_hidden, second_hidden, batch) -> LossOutput`
  - `WeightedInfoNCELoss.forward(first_hidden, second_hidden, batch) -> LossOutput`
- Purpose and contract: resolve contrastive objectives by name and compute symmetric token-level InfoNCE, optionally with weight and hardness matrices.
- Important inputs/outputs: two encoded views plus optional `w/h` batch tensors in; `LossOutput` out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecPretraining._build_loss` and then `_contrastive_step`.
- Reuse guidance: new contrastive objectives should register here.
- Duplication risk notes: `_contrastive_accuracy` is duplicated between the two shipped loss modules.

## Model-averaging family

- Files: `sleep2vec/averagings/base.py`, `sleep2vec/averagings/ema.py`, `sleep2vec/averagings/running_mean.py`
- Functions and methods:
  - `build_model_averager(cfg, student) -> BaseModelAverager | None`
  - `EmaModelAverager.on_train_batch_end(...)`
  - `RunningAverageModelAverager.on_train_batch_end(...)`
- Purpose and contract: optionally maintain an evaluation copy of the student model, updated either by EMA or arithmetic running average.
- Important inputs/outputs: averaging config plus student model in; averager object or `None` out.
- Side effects: clone the student model, update averaged weights during training, and may inject missing averaged weights during checkpoint resume.
- Key callers/callees: callers are `Sleep2vecPretraining` and `Sleep2vecFinetuning`.
- Reuse guidance: reuse this infrastructure for any averaged-eval model behavior.
- Duplication risk notes: averaging behavior should remain inside the averager classes, not in trainers or entrypoints.
