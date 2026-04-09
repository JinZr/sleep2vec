# Functions: Models And Adaptation

## `sleep2vec.pretrain_model.Sleep2vecPretrainModel.__init__`

- File: `sleep2vec/pretrain_model.py`
- Signature: `Sleep2vecPretrainModel(..., model_config: ModelConfig | None = None, projection_config: ProjectionConfig | None = None)`
- Purpose and contract: constructs the shared tokenizer mapping, encoder factory, optional CLS strategy, mask embeddings, embedding projection, and optional projection head used by both contrastive and downstream paths.
- Important inputs: either full `ModelConfig` or the legacy explicit constructor arguments.
- Important outputs: initialized model module.
- Side effects: builds `nn.Module` subtrees and logs parameter counts.
- Notable callers/callees: used by `Sleep2vecPretraining` and `Sleep2vecFinetuning`; depends on `build_tokenizers_and_dim`, `build_encoder_factory`, `build_projection`, and `build_cls_embedding`.
- Reuse guidance: canonical shared backbone wrapper.
- Duplication-risk notes: very high.

## `sleep2vec.pretrain_model.Sleep2vecPretrainModel._token_embeddings_to_hidden`

- File: `sleep2vec/pretrain_model.py`
- Signature: `_token_embeddings_to_hidden(self, token_embeddings, batch, *, return_hidden_states=False) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]`
- Purpose and contract: projects token embeddings into encoder hidden size, applies CLS strategy and padding mask, runs the encoder, and optionally returns hidden states for layer mixing.
- Important inputs: token embeddings from per-modality tokenizers, batch lengths, hidden-state flag.
- Important outputs: last hidden state, attention mask, optional hidden states.
- Side effects: none beyond tensor compute.
- Notable callers/callees: used by `Sleep2vecDownstreamModel.forward`.
- Reuse guidance: canonical token-to-encoder bridge.
- Duplication-risk notes: high.

## `sleep2vec.sleep2vec_modelling.Sleep2vecPretraining._contrastive_step`

- File: `sleep2vec/sleep2vec_modelling.py`
- Signature: `_contrastive_step(self, batch, log_prefix=None, model=None)`
- Purpose and contract: runs the shared pretrain model, computes the configured contrastive loss, logs step and epoch metrics, and caches validation metrics for epoch-end reduction.
- Important inputs: collated batch, optional log prefix, optional eval model.
- Important outputs: `(total_loss, contrastive_acc)`.
- Side effects: Lightning metric logging and in-memory validation cache mutation.
- Notable callers/callees: used by `training_step` and `validation_step`; delegates to `self.loss_fn`.
- Reuse guidance: canonical contrastive objective wrapper.
- Duplication-risk notes: medium-high.

## `sleep2vec.downstream_model.Sleep2vecDownstreamModel.forward`

- File: `sleep2vec/downstream_model.py`
- Signature: `forward(self, batch)`
- Purpose and contract: tokenizes each requested modality, optionally mixes hidden states across layers, applies sequence or non-sequence aggregation logic, merges token masks when needed, and calls the configured downstream head.
- Important inputs: collated batch with `tokens`, `length`, and possibly sequence labels.
- Important outputs: downstream logits or predictions.
- Side effects: may log a one-time warning when `is_seq=True` conflicts with CLS-only downstream mode.
- Notable callers/callees: used by `Sleep2vecFinetuning._shared_step`; depends on `Sleep2vecPretrainModel._token_embeddings_to_hidden`, temporal aggregation, head registry, and optional `LayerMix`.
- Reuse guidance: canonical downstream forward path.
- Duplication-risk notes: very high.

## `sleep2vec.downstream_model.Sleep2vecDownstreamModel.load_pretrained_backbone`

- File: `sleep2vec/downstream_model.py`
- Signature: `load_pretrained_backbone(self, ckpt_path, use_ema: bool | str | None = True)`
- Purpose and contract: loads checkpoint-init weights into the backbone, preferring averaged-model prefixes when requested, and warns about CLS mismatches or dropped CLS weights.
- Important inputs: checkpoint path and prefix preference.
- Important outputs: none.
- Side effects: checkpoint reads; model parameter mutation; logging warnings.
- Notable callers/callees: called by `Sleep2vecFinetuning.__init__`; uses `load_checkpoint` and `load_pretrain_init_weights`.
- Reuse guidance: canonical downstream checkpoint-init path.
- Duplication-risk notes: high.

## `sleep2vec.downstream_model.Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora`

- File: `sleep2vec/downstream_model.py`
- Signature: `freeze_backbone_and_insert_lora(self, insert_lora=True, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=("query", "key", "value"), separate_adapters=False)`
- Purpose and contract: freezes the shared backbone, optionally wraps the encoder with PEFT LoRA modules, and can add separate adapters per channel.
- Important inputs: LoRA hyperparameters and separate-adapter toggle.
- Important outputs: none.
- Side effects: mutates trainable parameters and encoder module structure; logs parameter counts.
- Notable callers/callees: called by `Sleep2vecFinetuning.__init__`.
- Reuse guidance: canonical LoRA insertion path.
- Duplication-risk notes: medium-high.

## `sleep2vec.sleep2vec_adaptation.build_new_modality_pair_probs`

- File: `sleep2vec/sleep2vec_adaptation.py`
- Signature: `build_new_modality_pair_probs(pairs, *, new_channels, new_pair_ratio) -> dict[Pair, float]`
- Purpose and contract: partitions all channel pairs into new-modality and legacy groups, then allocates probability mass according to `new_pair_ratio`.
- Important inputs: full pair list, `adapt.new_channels`, desired ratio.
- Important outputs: normalized pair-probability dictionary.
- Side effects: none.
- Notable callers/callees: used by `initial_pair_probs_for_phase` and `AdaptPairScheduleCallback`.
- Reuse guidance: canonical pair-distribution builder for adaptation.
- Duplication-risk notes: high.

## `sleep2vec.sleep2vec_adaptation.AdaptPairScheduleCallback.on_train_epoch_start`

- File: `sleep2vec/sleep2vec_adaptation.py`
- Signature: `on_train_epoch_start(self, trainer, pl_module) -> None`
- Purpose and contract: computes training progress from epoch position, resolves the scheduled `new_pair_ratio`, and pushes the corresponding pair distribution into the train batch sampler when the sampler supports it.
- Important inputs: Lightning trainer and module.
- Important outputs: none.
- Side effects: mutates sampler pair probabilities; logging.
- Notable callers/callees: active only in adapt stage 2.
- Reuse guidance: canonical schedule bridge from config to sampler behavior.
- Duplication-risk notes: medium-high.

## `sleep2vec.sleep2vec_adaptation.Sleep2vecAdaptation.configure_optimizers`

- File: `sleep2vec/sleep2vec_adaptation.py`
- Signature: `configure_optimizers(self)`
- Purpose and contract: creates phase-aware optimizer groups, scaling learning rates differently for encoder/CLS, shared legacy weights, and new modalities, while keeping the pretrain-style warmup+cosine schedule.
- Important inputs: adaptation phase, `AdaptConfig`, and grouped parameters from the model.
- Important outputs: Lightning optimizer/scheduler structure.
- Side effects: none beyond optimizer construction.
- Notable callers/callees: called by Lightning; depends on `self.model.get_adaptation_param_groups(...)`.
- Reuse guidance: canonical adaptation optimizer policy.
- Duplication-risk notes: medium-high.
