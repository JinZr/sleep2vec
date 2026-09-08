# Hospital whole-night initialization

This is a concept-only reconstruction from an actual initialization session.
Propose a bounded technical search domain and at most two complete opening
configurations, with reasons for the boundaries and fixed settings. Explain what
the opening comparison could establish and what remains unknown. Use the supplied
evidence only. Do not publish a recipe, run consultation, contact a server,
launch work or evaluate a model. Numerical choices have no reference answer.

## Evidence boundary

The boundary is the first search-domain construction in the recorded authoring
command on 2026-09-07 at 17:16:24 UTC. The provisional base below was constructed
immediately before the search in that same command; it was not an independently
validated or previously trained configuration. The original search domain,
opening pair and all subsequent outcomes are withheld. No comparable downstream
training result or duration measurement appears in the supplied pre-search
material. That does not establish that none existed elsewhere.

## User request and fixed scientific scope

The user requested whole-night heartbeat and breath disease tuning, maximizing
test macro-AUROC, with twelve total runs, at most two concurrent runs and two GPUs
per run. They supplied a pretrained epoch-89 checkpoint, confirmed that the 0/1
labels are valid negatives/positives, and required same-night, same-channel median
imputation while preserving the whole-night timeline. Masking entirely missing
internal tokens was proposed by the agent, but mask propagation was still an
unverified preparation requirement at this boundary; the user explicitly
confirmed median filling. Labels, patient-exclusive chronological split, checkpoint,
modalities, preprocessing and execution host are fixed for this exercise.

The task has 19 multilabel outputs. Each night has 1,440 ordered 30-second tokens.
The test set is explicitly allowed to participate in selection; findings must
remain described as test-tuned. No new independent final-test access is granted.
The execution target is a Slurm GPU host; no wall-clock or GPU-hour limit was
specified by the user. An epoch throughput estimate is not available here.

## Provisional base before search construction

These technical settings were agent-authored defaults, not user-selected values
or measurements of what trains well:

```yaml
runtime:
  precision: bf16-mixed
  epochs: 8
  batch_size: 4
  num_workers: 2
  lr: 0.00001
  weight_decay: 0.0001
  warmup_steps: 200
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  patience: 8
  check_val_every_n_epoch: 1
  ckpt_every_n_epochs: 1
model_head:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.1
  channel_aggregation: gated_scalar
  temporal_aggregation: mean
adaptation:
  preset: full
  tokenizers_train: false
layer_mix:
  enabled: false
loss:
  pos_weight: null
sampler:
  weighted_random: false
```

The recorded pre-search inspection printed the checkpoint config and pretraining
CLI. The relevant inherited model settings are:

```yaml
backbone:
  name: roformer
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 16
  attention_backend: sdpa
selected_channel_tokenizers:
  name: sundial2
  input_dim: 120
  out_dim: 768
  norm_layer: true
  pre_norm: false
  residual_scale: 0.1
  ff_scale: 1.0
  clamp_value: null
  modality_scale: 1.0
  num_mlp_layers: 3
projection:
  name: simclr
  enabled: true
  hidden_dim: 768
  out_dim: 256
cls:
  embedding_type: bert
  downstream: tokens
model_averaging:
  name: ema
  params:
    enabled: true
    base_momentum: 0.996
    final_momentum: 1.0
    use_for_eval: true
```

The provisional construction retains these settings, selects heartbeat/breath and
removes their old aliases. Other pretraining channels are deliberately omitted
from this case. The pretraining CLI used 180 tokens, LR 5e-5, weight decay 0.01,
100 epochs, cosine decay with floor 0.1 and `warmup_steps: null`; effective
pretraining warmup is not established here. These describe pretraining, not the
effective downstream scheduler. No downstream performance follows from
them. The draft recipe named `sleep2vec`; executable checkpoint/variant
compatibility and mask propagation still require verification. Preserve the
checkpoint rather than changing identity to make the draft appear ready.

No explicit scheduler family, shape or floor was set in the captured draft.
Their effective runtime defaults and optimizer-update count are not supplied.
Patience counts validation checks. Saving/testing each epoch adds work beyond
fit time, and changing the configured horizon can also affect scheduling.

The familiar scalar knobs include learning rate, epochs, weight decay, warmup,
clipping, patience and head dropout. You may propose other supported technical
axes if you can justify their cost and coupling; unknown config semantics must
remain unknown. Keep the user-fixed scientific scope intact. Give a concept plan
and identify any information still required before publication.
