# Models And Heads

## `Sleep2vecPretrainModel.__init__`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel.__init__(model_config: ModelConfig, *, device: str = "cuda", specified_two_mods: list[str] | None = None)`
- Purpose and contract: build the shared backbone module, tokenizer mapping, CLS strategy, mask embeddings, hidden projection layer, and optional projection head.
- Important inputs/outputs: a full `ModelConfig` in; produces a ready-to-run module. `specified_two_mods` only fixes pretraining channel selection for tests or controlled runs.
- Side effects: instantiates many submodules and logs parameter counts.
- Key callers/callees: callers are `Sleep2vecPretraining`, `Sleep2vecFinetuning`, embedding extraction, and tests; callees include `build_tokenizers_and_dim`, `build_encoder_factory`, `build_cls_embedding`, and `build_projection`.
- Reuse guidance: this is the canonical backbone construction path.
- Duplication risk notes: construction is config-only; do not reintroduce manual channel, hidden-size, projection, or encoder-factory constructor branches.

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

## `sleep2vec2` / `sleep2expert` `RoFormerEncoderModel`

- File: `sleep2vec2/backbones/roformer/configuration.py`, `sleep2vec2/backbones/roformer/model.py`, `sleep2vec2/backbones/roformer/modeling_roformer.py`, `sleep2expert/backbones/roformer/configuration.py`, `sleep2expert/backbones/roformer/model.py`, and `sleep2expert/backbones/roformer/modeling_roformer.py`
- Signature: `RoFormerEncoderModel.forward(...)` accepts embeddings-first runtime calls plus optional `input_ids` / `token_type_ids` kwargs that PEFT may pass through; `sleep2expert` also accepts `modality_name` and `collect_moe_aux`.
- Purpose and contract: package-local standalone RoFormer encoder used by `sleep2vec2` and `sleep2expert` instead of the root/Hugging Face RoFormer module; `sleep2expert` layers may swap dense FFN blocks for MoE experts. The lightweight config exposes dict-like `get` for PEFT compatibility while keeping attribute-based runtime access.
- Important inputs/outputs: token ids or embeddings plus attention controls in; RoFormer-style output with last hidden state, optional hidden states, and optional attentions out. `model.backbone.attention_backend` selects eager attention or SDPA, while `output_attentions=True` keeps eager attention for visualization.
- Side effects: none beyond compute.
- Key callers/callees: built by package-local `backbones.encoder_factory.build_roformer`; exercised by package-local `pretrain_model.Sleep2vecPretrainModel` and downstream wrappers.
- Reuse guidance: keep standalone backbone changes inside the variant namespace and preserve RoFormer parity tests.
- Duplication risk notes: standalone variants reject legacy sleep2vec/HF RoFormer checkpoint key layouts; do not add silent compatibility shims.

## `sleep2expert.backbones.roformer.moe.TopKRouter.forward`

- File: `sleep2expert/backbones/roformer/moe.py`
- Signature: `TopKRouter.forward(hidden_states, modality_name=None, token_mask=None) -> MoERoutingOutput`
- Purpose and contract: compute per-token top-k expert choices for learned, random, hard-modality, or hard-group routing, applying modality group masks when enabled.
- Important inputs/outputs: hidden states, modality name, and optional token mask in; `MoERoutingOutput` with logits, probabilities, top-k indices/probabilities, load, importance, z-loss, entropy, modality name, and layer index out.
- Side effects: uses random logits for `router_type="random"` and training-time router noise when configured.
- Key callers/callees: caller is `SparseMoEFFN.forward`; callees include `_allowed_experts`, `_hard_route`, `_mask_logits`, and `_build_output`.
- Reuse guidance: add router behavior here, not in trainer or loss code.
- Duplication risk notes: modality-to-expert group validation is paired with `sleep2expert.config._validate_moe_config`.

## `sleep2expert.backbones.roformer.moe.SparseMoEFFN.forward`

- File: `sleep2expert/backbones/roformer/moe.py`
- Signature: `SparseMoEFFN.forward(hidden_states, input_tensor, *, modality_name=None, attention_mask=None, collect_aux=False) -> tuple[torch.Tensor, MoERoutingOutput | None]`
- Purpose and contract: replace a dense RoFormer FFN with sparse expert execution, weighted top-k expert aggregation, residual dropout, layer norm, and optional routing aux return.
- Important inputs/outputs: hidden states, residual tensor, modality name, attention mask, and aux flag in; transformed hidden states and optional routing aux out.
- Side effects: none beyond compute.
- Key callers/callees: caller is the standalone sleep2expert RoFormer layer; callee is `TopKRouter.forward`.
- Reuse guidance: keep sparse FFN execution here; losses and export should consume aux records rather than recomputing routes.
- Duplication risk notes: hard-routing and group-mask semantics must not be duplicated in downstream heads.

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
- Side effects: may switch active LoRA adapters when separate adapters are enabled.
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

## `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora` and package-local variant mirrors

- File: `sleep2vec/downstream_model.py`, `sleep2vec2/downstream_model.py`, `sleep2expert/downstream_model.py`
- Signature: `freeze_backbone_and_insert_lora(insert_lora: bool = True, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, target_modules=("query", "key", "value"), use_dora: bool = False, separate_adapters: bool = False) -> None`
- Purpose and contract: freeze backbone parameters, optionally inject LoRA/DoRA adapters into the encoder with configurable rank/alpha/dropout/target modules, and optionally create separate adapters per channel. `sleep2expert` can opt in expert targets by naming `dense_in` or `dense_out`; router targets are rejected by config parsing.
- Important inputs/outputs: adapter enable flag, rank, alpha, dropout, target-module names, DoRA flag, and separate-adapter flag in; no return value.
- Side effects: mutates backbone module structure and parameter trainability; with separate adapters, the default LoRA adapter is frozen and only `ch_<channel>` adapter weights are trainable.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`; callee is `get_peft_model`.
- Reuse guidance: this is the canonical LoRA insertion path for downstream training; standalone variants keep package-local copies aligned with this contract.
- Duplication risk notes: adapter naming and trainable-parameter enabling should stay centralized here.

## `Sleep2vecDownstreamModel._enable_all_adapters_trainable` and package-local variant mirrors

- File: `sleep2vec/downstream_model.py`, `sleep2vec2/downstream_model.py`, `sleep2expert/downstream_model.py`
- Signature: `_enable_all_adapters_trainable(self) -> None`
- Purpose and contract: after separate channel adapters are added, freeze all LoRA parameters and re-enable only the configured channel adapter weights.
- Important inputs/outputs: uses `self.channel_adapters`; mutates encoder parameter `requires_grad` flags.
- Side effects: changes adapter trainability.
- Key callers/callees: caller is `freeze_backbone_and_insert_lora`.
- Reuse guidance: keep separate-adapter trainability changes here instead of editing PEFT parameters from trainer code.
- Duplication risk notes: default adapter parameters must not be accidentally left trainable when channel-specific adapters are requested.

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
- Purpose and contract: run the contrastive forward pass, apply the configured loss, add sleep2expert MoE pretrain regularization when enabled in that namespace, log step/epoch metrics, and cache validation metrics for epoch-end aggregation.
- Important inputs/outputs: batch in; returns `(total_loss, contrastive_acc)`.
- Side effects: logs Lightning metrics and mutates validation caches.
- Key callers/callees: callers are `training_step` and `validation_step`; callees are `self.model(...)` and `self.loss_fn(...)`.
- Reuse guidance: reuse this helper for any pretrain stage that wants identical loss/logging semantics.
- Duplication risk notes: pretrain metric naming and aggregation live here, not in the callback.

## `Sleep2vecPretraining.configure_optimizers`

- File: `sleep2vec/sleep2vec_modelling.py`
- Signature: `Sleep2vecPretraining.configure_optimizers(self)`
- Purpose and contract: create AdamW parameter groups and a warmup-plus-cosine scheduler for pretraining.
- Important inputs/outputs: uses `self.args` and `self.trainer`; returns Lightning optimizer/scheduler structure.
- Side effects: none beyond object creation.
- Key callers/callees: called by Lightning; callees include `build_warmup_cosine_scheduler`.
- Reuse guidance: use as the reference optimizer schedule when aligning pretrain and finetune behavior.
- Duplication risk notes: optimizer grouping remains local to this method; warmup-plus-cosine scheduling is shared through package-local `schedulers.py`.

## `build_warmup_cosine_scheduler`

- File: `sleep2vec/schedulers.py`
- Signature: `build_warmup_cosine_scheduler(optimizer, *, total_steps: int, warmup_steps: int | None) -> LambdaLR`
- Purpose and contract: build the shared step-wise LR schedule used by pretrain, finetune, and adaptation.
- Important inputs/outputs: optimizer and trainer step count in; `LambdaLR` out.
- Side effects: none beyond scheduler construction.
- Key callers/callees: callers are `Sleep2vecPretraining.configure_optimizers`, `Sleep2vecFinetuning.configure_optimizers`, and `Sleep2vecAdaptation.configure_optimizers`.
- Reuse guidance: use this helper for package-local warmup-plus-cosine LR schedules instead of inlining `lr_lambda`.
- Duplication risk notes: variants keep package-local copies to preserve namespace isolation.

## `Sleep2vecAdaptation.configure_optimizers`

- File: `sleep2vec/sleep2vec_adaptation.py`
- Signature: `Sleep2vecAdaptation.configure_optimizers(self)`
- Purpose and contract: create stage-specific AdamW parameter groups for adaptation, with learning-rate scaling across encoder/CLS, shared legacy projection, and new modalities.
- Important inputs/outputs: uses `self.args`, `self.adapt_config`, and trainer step count; returns Lightning optimizer/scheduler structure.
- Side effects: none beyond object creation.
- Key callers/callees: called by Lightning; callees include `self.model.get_adaptation_param_groups` and `build_warmup_cosine_scheduler`.
- Reuse guidance: use this as the reference optimizer path for adaptation-specific schedules.
- Duplication risk notes: stage1 vs stage2 group composition belongs here; scheduler construction should stay in package-local `schedulers.py`.

## `Sleep2vecFinetuning._shared_step`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._shared_step(batch, stage: str, model=None)`
- Purpose and contract: run forward, compute loss if valid labels exist, add sleep2expert downstream MoE regularization/metrics when configured in that namespace, log stage loss or accumulate eval loss, and cache either generic predictions or AHI event records for epoch-end metrics.
- Important inputs/outputs: batch and stage name in; returns training loss for `train`, otherwise `None`.
- Side effects: logs Lightning metrics and mutates `_stage_outputs`.
- Key callers/callees: callers are `training_step`, `validation_step`, and `test_step`; callees are `self.model(...)`, `_compute_loss`, `_extract_valid_predictions`, and `_extract_ahi_event_records`.
- Reuse guidance: all downstream stage logic should continue to funnel through this helper.
- Duplication risk notes: stage branching should not leak into entrypoints.

## `Sleep2vecFinetuning._compute_loss`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._compute_loss(logits, batch)`
- Purpose and contract: compute classification, regression, or multilabel loss, ignoring invalid labels and returning `None` when the batch contains no valid targets; configured `class_weights` affect CrossEntropyLoss and configured `pos_weight` affects BCEWithLogitsLoss for multilabel/AHI.
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
- Purpose and contract: create AdamW parameter groups and a warmup-plus-cosine scheduler for finetuning.
- Important inputs/outputs: uses `self.args` and `self.trainer`; returns Lightning optimizer/scheduler structure.
- Side effects: none beyond object creation.
- Key callers/callees: called by Lightning; callees include `build_warmup_cosine_scheduler`.
- Reuse guidance: use as the finetune optimizer reference path.
- Duplication risk notes: optimizer grouping remains local to this method; warmup-plus-cosine scheduling is shared through package-local `schedulers.py`.

## Loss factory family

- Files: `sleep2vec/losses/base.py`, `sleep2vec/losses/info_nce.py`, `sleep2vec/losses/weighted_info_nce.py`, `sleep2vec/losses/utils.py`
- Functions and methods:
  - `create_loss(name: str, **kwargs) -> ContrastiveLoss`
  - `contrastive_accuracy(logits_12, logits_21, labels) -> torch.Tensor`
  - `InfoNCELoss.forward(first_hidden, second_hidden, batch) -> LossOutput`
  - `WeightedInfoNCELoss.forward(first_hidden, second_hidden, batch) -> LossOutput`
- Purpose and contract: resolve contrastive objectives by name and compute symmetric token-level InfoNCE, optionally with weight and hardness matrices.
- Important inputs/outputs: two encoded views plus optional `w/h` batch tensors in; `LossOutput` out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecPretraining._build_loss` and then `_contrastive_step`.
- Reuse guidance: new contrastive objectives should register here.
- Duplication risk notes: contrastive accuracy is centralized in package-local `losses/utils.py`; variants should keep their own local helper rather than cross-namespace imports.

## `sleep2expert.losses.moe_regularization.compute_moe_regularization`

- File: `sleep2expert/losses/moe_regularization.py`
- Signature: `compute_moe_regularization(moe_aux, moe_cfg, batch, *, prefix: str | None = None) -> LossOutput`
- Purpose and contract: compute pretrain-time MoE auxiliary loss and metrics from `model.last_moe_aux`.
- Important inputs/outputs: routing aux records, `MoeConfig`, and batch in; `LossOutput` with total auxiliary loss and metrics out.
- Side effects: none.
- Key callers/callees: caller is `sleep2expert.sleep2vec_modelling.Sleep2vecPretraining._contrastive_step`; callees include `_load_balance_loss`, `_modality_balance_loss`, `_route_consistency_loss`, `_routing_stats`, and `_expert_usage_entropy`.
- Reuse guidance: route all pretrain MoE regularization through this function.
- Duplication risk notes: route consistency strips CLS tokens and compares only common expert support; keep that logic centralized.

## `sleep2expert.losses.moe_regularization.compute_downstream_moe_regularization`

- File: `sleep2expert/losses/moe_regularization.py`
- Signature: `compute_downstream_moe_regularization(moe_aux, reg_cfg, batch, *, prefix: str | None = None) -> LossOutput`
- Purpose and contract: compute the supported downstream MoE auxiliary loss subset from `model.backbone.last_moe_aux`.
- Important inputs/outputs: routing aux records, finetune MoE regularization config, and batch in; `LossOutput` with router z-loss contribution and metrics out.
- Side effects: none.
- Key callers/callees: caller is `sleep2expert.sleep2vec_finetuning.Sleep2vecFinetuning._compute_loss`; callee is `_downstream_moe_metric_values`.
- Reuse guidance: use this for sleep2expert downstream MoE finetuning regularization.
- Duplication risk notes: downstream route consistency, load balance, modality balance, and entropy regularization are intentionally unsupported and should not be implemented piecemeal elsewhere.

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
