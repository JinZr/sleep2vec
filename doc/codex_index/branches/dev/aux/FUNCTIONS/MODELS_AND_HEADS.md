# Models And Heads

## `Sleep2vecPretrainModel.__init__`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel.__init__(..., model_config: ModelConfig | None = None, projection_config: ProjectionConfig | None = None)`
- Purpose and contract: build the shared backbone module, tokenizer mapping, CLS strategy, mask embeddings, hidden projection layer, and optional projection head.
- Important inputs/outputs: either a full `ModelConfig` or legacy manual dimensions/channel names; produces a ready-to-run module.
- Side effects: instantiates many submodules and logs parameter counts.
- Key callers/callees: callers are `Sleep2vecPretraining` and `Sleep2vecFinetuning`; callees include `build_tokenizers_and_dim`, `build_encoder_factory`, `build_cls_embedding`, and `build_projection`.
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

## `Sleep2vecPretrainModel.forward`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel.forward(batch, apply_mask)`
- Purpose and contract: choose two channels, optionally apply modality masking, encode each channel independently, and optionally project the hidden states for contrastive loss.
- Important inputs/outputs: batch with `tokens`, `mlm_mask`, and `length`; returns `(first_hidden, second_hidden)`.
- Side effects: random channel selection unless `specified_two_mods` is fixed.
- Key callers/callees: caller is `Sleep2vecPretraining._contrastive_step`; callees include `_tokenize_two_random_channels`, `_mask_modalities`, and `_token_embeddings_to_hidden`.
- Reuse guidance: this is the canonical contrastive pretrain forward path.
- Duplication risk notes: do not recreate channel-selection and masking flow in the Lightning module.

## `Sleep2vecPretrainModel.encode`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel.encode(batch, channel_name)`
- Purpose and contract: encode one named channel through tokenization, CLS handling, encoder, and optional projection head.
- Important inputs/outputs: batch plus channel name in; encoded token sequence out.
- Side effects: none.
- Key callers/callees: not heavily used in the reviewed runtime, but it reuses `_token_embeddings_to_hidden`.
- Reuse guidance: use when a caller needs one channel's encoded representation without contrastive pair sampling.
- Duplication risk notes: this is the correct single-channel path; do not inline tokenizer-plus-encoder steps elsewhere.

## `Sleep2vecDownstreamModel.__init__`

- File: `sleep2vec/downstream_model.py`
- Signature: `Sleep2vecDownstreamModel.__init__(target, backbone, channel_names, output_dim, is_classification, is_seq, ..., model_config, layer_mix_cfg, head_config, auxiliary_task_cfg)`
- Purpose and contract: wrap the shared backbone with temporal aggregation, channel aggregation, downstream head creation, optional metadata auxiliary prediction, and optional layer-mix support.
- Important inputs/outputs: configured backbone plus task/head metadata in; downstream model out.
- Side effects: builds fusion/head modules and may raise when CLS semantics are inconsistent.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`; callees are `create_head`, `build_temporal_aggregator`, and `LayerMix`.
- Reuse guidance: this is the canonical downstream assembly surface.
- Duplication risk notes: main and metadata-auxiliary prediction should share this assembly surface; do not open a second downstream-model builder for auxiliary tasks.

## `Sleep2vecDownstreamModel.forward`

- File: `sleep2vec/downstream_model.py`
- Signature: `Sleep2vecDownstreamModel.forward(batch)`
- Purpose and contract: encode each modality through the shared backbone, then apply either sequence or pooled downstream logic, optional layer mixing, channel fusion, main-head prediction, and optional metadata auxiliary prediction from the same shared features.
- Important inputs/outputs: batch with `tokens` and `length`; returns main logits or a small `{main_logits, aux_logits}` container when an auxiliary task is enabled.
- Side effects: may switch active LoRA adapters when separate adapters are enabled.
- Key callers/callees: caller is `Sleep2vecFinetuning._shared_step`; callees include `Sleep2vecPretrainModel._tokenize_all`, `_token_embeddings_to_hidden`, `_forward_seq`, `_forward_nonseq`, and `_call_head`.
- Reuse guidance: all downstream predictions should flow through this method.
- Duplication risk notes: sequence/non-sequence branching, CLS behavior, and metadata auxiliary pooling all belong here; do not recompute pooled auxiliary features inside the trainer.

## `sleep2vec.downstreams.heads.temporal_unet.TemporalUNetHead`

- File: `sleep2vec/downstreams/heads/temporal_unet.py`
- Signature: `TemporalUNetHead(feature_dim, n_mods, out_dim, *, agg='gated_scalar', hidden_dim=None, dropout=0.1, act=nn.GELU, num_levels=4, blocks_per_level=2, kernel_size=5, downsample_stride=2)`
- Purpose and contract: reuse `FeatureFusion` and `TemporalConvBlock` to build a large-context sequence head with a 1D U-Net-style encoder/decoder over token features, preserving padded-token masking.
- Important inputs/outputs: list of per-modality `[B, L, D]` tensors plus optional `token_mask`; returns `[B, L, C]` logits.
- Side effects: none beyond compute.
- Key callers/callees: resolved through the existing head registry; reuses `FeatureFusion` and `TemporalConvBlock`.
- Reuse guidance: use this head when the task needs larger temporal context than the plain MLP `classification` head provides, but keep metadata auxiliary prediction on the existing pooled-head path.
- Duplication risk notes: do not clone the temporal-conv residual block family inside another sequence head unless the contract really diverges.

## `Sleep2vecDownstreamModel.load_pretrained_backbone`

- File: `sleep2vec/downstream_model.py`
- Signature: `Sleep2vecDownstreamModel.load_pretrained_backbone(ckpt_path, use_ema: bool | str | None = True) -> None`
- Purpose and contract: load pretraining weights into the backbone, optionally preferring an averaged model prefix such as `ema_model.`.
- Important inputs/outputs: checkpoint path and averaging preference in; no return value.
- Side effects: loads checkpoint from disk, mutates backbone weights, emits warnings for CLS mismatches and dropped CLS weights.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`; callees include `_warn_on_cls_mismatch`, `_warn_on_dropped_cls_weights`.
- Reuse guidance: use for any downstream weight initialization from pretrain checkpoints.
- Duplication risk notes: prefix selection and fallback behavior are non-trivial; do not slice checkpoint keys manually in callers.

## `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora`

- File: `sleep2vec/downstream_model.py`
- Signature: `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora(insert_lora: bool = True, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05, target_modules=("query", "key", "value"), separate_adapters: bool = False) -> None`
- Purpose and contract: freeze backbone parameters, optionally inject LoRA adapters into the encoder, and optionally create separate adapters per channel.
- Important inputs/outputs: LoRA config in; no return value.
- Side effects: mutates backbone module structure and parameter trainability.
- Key callers/callees: caller is `Sleep2vecFinetuning.__init__`; callee is `get_peft_model`.
- Reuse guidance: this is the canonical LoRA insertion path for downstream training.
- Duplication risk notes: adapter naming and trainable-parameter enabling should stay centralized here.

## `Sleep2vecPretraining._contrastive_step`

- File: `sleep2vec/sleep2vec_modelling.py`
- Signature: `Sleep2vecPretraining._contrastive_step(batch, log_prefix=None, model=None)`
- Purpose and contract: run the contrastive forward pass, apply the configured loss, log step/epoch metrics, and cache validation metrics for epoch-end aggregation.
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
- Key callers/callees: called by Lightning.
- Reuse guidance: use as the reference optimizer schedule when aligning pretrain and finetune behavior.
- Duplication risk notes: this logic is nearly duplicated in `Sleep2vecFinetuning.configure_optimizers`.

## `Sleep2vecFinetuning._shared_step`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._shared_step(batch, stage: str, model=None)`
- Purpose and contract: run forward, compute loss if valid labels exist, log stage loss, and cache valid predictions for epoch-end metrics.
- Important inputs/outputs: batch and stage name in; returns training loss for `train`, otherwise `None`.
- Side effects: logs Lightning metrics and mutates `_stage_outputs`.
- Key callers/callees: callers are `training_step`, `validation_step`, and `test_step`; callees are `self.model(...)`, `_compute_loss`, `_extract_valid_predictions`.
- Reuse guidance: all downstream stage logic should continue to funnel through this helper.
- Duplication risk notes: stage branching should not leak into entrypoints.

## `Sleep2vecFinetuning._compute_loss`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._compute_loss(logits, batch)`
- Purpose and contract: compute classification or regression loss, ignoring invalid labels (`-1` or `-1.0`) and returning `None` when no valid labels exist.
- Important inputs/outputs: logits and batch in; returns `(loss, valid_count)` or `None`.
- Side effects: none.
- Key callers/callees: caller is `_shared_step`.
- Reuse guidance: this is the canonical downstream loss application logic.
- Duplication risk notes: ignore-index and valid-label filtering must stay aligned with `_extract_valid_predictions`.

## `Sleep2vecFinetuning._finalize_epoch`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning._finalize_epoch(stage: str)`
- Purpose and contract: concatenate cached predictions/targets for one stage, reduce them through `compute_downstream_metrics`, log results, and clear the stage cache.
- Important inputs/outputs: stage name in; returns `(preds, gts)` for test/plotting or `None` when nothing was cached.
- Side effects: logs metrics and clears cached outputs.
- Key callers/callees: callers are `on_train_epoch_end`, `on_validation_epoch_end`, and `on_test_epoch_end`; callee is `compute_downstream_metrics`.
- Reuse guidance: use this pattern for any new stage-level downstream metrics.
- Duplication risk notes: do not introduce separate metric reducers per stage.

## `Sleep2vecFinetuning.configure_optimizers`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `Sleep2vecFinetuning.configure_optimizers(self)`
- Purpose and contract: create AdamW parameter groups and a warmup-plus-cosine scheduler for finetuning.
- Important inputs/outputs: uses `self.args` and `self.trainer`; returns Lightning optimizer/scheduler structure.
- Side effects: none beyond object creation.
- Key callers/callees: called by Lightning.
- Reuse guidance: use as the finetune optimizer reference path.
- Duplication risk notes: nearly duplicates the pretrain scheduler policy.

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
- Side effects: clone student model, update averaged weights during training, may inject missing averaged weights during checkpoint resume.
- Key callers/callees: callers are both Lightning modules.
- Reuse guidance: reuse this infrastructure for any averaged-eval model behavior.
- Duplication risk notes: averaging behavior should remain inside the averager classes, not in trainers or entrypoints.
