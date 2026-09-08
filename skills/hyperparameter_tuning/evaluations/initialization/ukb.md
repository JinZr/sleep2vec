# Cox initialization after a completed-results review

This is a concept-only reconstruction from an actual initialization session.
Propose a bounded technical search domain and at most two complete opening
configurations, with reasons for the boundaries and fixed settings. Explain what
the opening comparison could establish and what remains unknown. Use the supplied
evidence only. Do not publish a recipe, run consultation, contact a server,
launch work or evaluate a model. Numerical choices have no reference answer.

## Evidence boundary

Use information available by 2026-09-07 16:31:00 UTC, before the new adaptive
recipe was authored at 16:31:59 UTC. The old experiment is evidence, not a plan
to rewrite or resume. The new domain, opening pair and later outcomes are hidden.

## User request and fixed scientific scope

After inspecting the old candidate results, the user requested adaptive tuning
based on the observed trends, at most two concurrent runs and two GPUs per run,
with restrained resource use. They explicitly reset the budget to twelve new
fits; prior completed/interrupted runs do not consume it.

Keep the existing sleep2vec2 actigraphy input, 624 Cox endpoints, participant
splits, absence of age/sex covariates, epoch-89 pretrained checkpoint and Slurm
host/runtime identity. Maximize checkpoint-level test C-index, with test feedback
explicitly unlocked. These are descriptive test-tuned results. No independent
final-test access, new data split or covariate change is authorized. The prior
source records that full-cohort endpoint selection included test diagnoses and
complete EHR coverage at the administrative cutoff remained unverified. Preserve
these limitations when interpreting scores. The user did not give a GPU-hour or wall-clock budget.

## Prior configuration and results

The prior source used the following joint baseline. All unlisted model/data
semantics remain fixed; their full source bytes are not supplied in this case.

```yaml
runtime:
  precision: bf16-mixed
  epochs: 18
  batch_size: 48
  accumulate_grad_batches: 1
  lr: 0.0001
  weight_decay: 0.0001
  warmup_steps: 1000
  lr_scheduler: decay
  lr_decay_shape: cosine
  lr_decay_floor: 0.01
  gradient_clip_val: 1.0
  patience: 100
  check_val_every_n_epoch: 1
  ckpt_every_n_epochs: 1
model_head:
  hidden_dim: 256
  num_layers: 3
  dropout: 0.1
lora:
  r: 8
  alpha: 16
  dropout: 0.05
layer_mix:
  layer_indices: [9, 10, 11, 12]
```

The old recipe requested four GPUs per run, eight data-loader workers, ten CPUs
and 512 GB RAM. The new user limit is two GPUs per run. Old requests are not
measured minima. The recorded prior research log describes batch 48 per GPU as
user-authorized; retain that batch size for this exercise. The original user
message establishing it is not included. Accumulation for the new two-GPU setup
is still a technical choice. No directly comparable two-GPU throughput
measurement is supplied.

The following five runs completed their required checkpoint tests. Parameter
changes are relative to the joint baseline above; scores are each run's best
checkpoint test C-index. Checkpoint epochs are zero-based.

| Run | Changes from baseline | Best test C-index | Best checkpoint epoch |
|---|---|---:|---:|
| old-000 | None | 0.627977 | 12 |
| old-001 | LR 3e-5; epochs 12 | 0.617135 | 11 |
| old-002 | LR 3e-4; epochs 8; warmup 500 | 0.630411 | 7 |
| old-003 | Weight decay 1e-3; dropout 0.3; epochs 12 | 0.626569 | 11 |
| old-004 | Weight decay 1e-5; dropout 0; epochs 12 | 0.627080 | 11 |

The complete checkpoint trajectories and training/validation loss curves are not
included. The historical baseline before this search scored 0.627974. An earlier
model with age/sex covariates was reported in a prior-session summary as scoring
0.670920; its result artifact was not freshly read in this capture. It is a different scientific
configuration, not a candidate for this search or an isolated feature effect.
There is no repeat/seed-variance estimate in the supplied evidence.

Old head-capacity variants were interrupted following the stop instruction;
they have no completed test objective and must not be assigned low scores.
Untested old candidates are not observations. The largest LR in these five
completed observations is 3e-4; no authorized range for the new workflow has yet
been chosen. Patience counts validation checks. The decay schedule and its
warmup are coupled to training length and optimizer updates. A two-GPU run may
need a different accumulation setting to retain the old effective batch.
