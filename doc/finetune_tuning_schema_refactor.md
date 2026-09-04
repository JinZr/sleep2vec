# Finetune Trainability Schema (`finetune.tuning`)

Reference for the `finetune.tuning` block of `sleep2vec/`, `sleep2vec2/` and
`sleep2expert/`. It is the single source of truth for which parameters train and at
what learning-rate scale.

**Converting a config that still uses the old keys?** Skip to
[Converting a legacy config](#converting-a-legacy-config). The parsers reject
`finetune.lora.freeze_backbone_and_insert_lora`, `finetune.lora.insert_lora`,
`finetune.freeze_tokenizer` and `finetune.moe_tuning`, and their error messages point
here for that table.

Two things this document does not cover. `sex_age_baseline` trains its own model rather
than adapting a pretrained backbone; its config module never reads `finetune.tuning`, so
a block written there is silently ignored. And `adapt.stage2.lr_scales` is a separate
trainability system with its own vocabulary (`encoder`, `shared_legacy`,
`new_modalities`, built from `get_adaptation_param_groups`) that selects parameters by
*phase* and skips `requires_grad=False` tensors, so it never overloads `lr_scale` as a
freeze switch. Note the tension it leaves: `finetune.tuning.groups.encoder.lr_scale` and
`adapt.stage2.lr_scales.encoder` are two spellings of one concept, which AGENTS.md
("one canonical spelling and location for each concept") would eventually want
reconciled.

## Why one block

Trainability used to be decided by four independent YAML entry points that the runtime
applied in sequence, last writer wins.

| Order | Key | Effect |
| --- | --- | --- |
| 1 | `finetune.lora.freeze_backbone_and_insert_lora` | Freezes every backbone parameter |
| 2 | `finetune.lora.insert_lora` | Inserts LoRA; **only read when 1 is true** |
| 3 | `finetune.freeze_tokenizer` | Re-writes tokenizer trainability after 1/2 |
| 4 | `finetune.moe_tuning.mode`, `lr_scales[g] == 0`, `freeze_router`, `freeze_experts` | Overwrites every group (`sleep2expert` only) |

Nothing derived the final state from one place, so a policy at step 4 could silently
invalidate an assumption captured at step 1 — and the failure was always quiet. Two
further hazards came with it: `lr_scales` carried two meanings at once (a learning-rate
multiplier, and a freeze switch at `0.0`), and `insert_lora: true` under
`freeze_backbone_and_insert_lora: false` was inert, so a config could advertise a LoRA
recipe that never ran.

The block below replaces all four. Every parameter is classified into exactly one group,
the group table is materialized once at parse time, and the trainer applies it once.

## The schema

```yaml
finetune:
  tuning:
    preset: head_only          # full | head_only | lora | moe_conservative
                               # moe_conservative_routers | moe_top_experts | custom
    groups:                    # a partition of the parameters, not a hierarchy;
                               # every parameter lands in exactly one group
      head:       {train: true,  lr_scale: 1.0}
      encoder:    {train: false}   # backbone blocks only, excluding the groups below
      tokenizers: {train: false}
      experts:    {train: false}
      routers:    {train: false}
      projection: {train: false}
      lora:       {train: false}
    lora:                      # adapter shape; read only when groups.lora.train
      r: 8
      alpha: 16
      dropout: 0.05
      target_modules: [query, key, value]
      use_dora: false
      separate_adapters: false
    moe:
      layer_indices: null      # required by preset moe_top_experts
  moe_regularization:          # auxiliary loss terms; NOT a trainability knob
    enabled: false
```

`experts`, `routers` and `moe` exist only on `sleep2expert`, and only when
`model.backbone.moe.enabled`. `sleep2vec` and `sleep2vec2` have the five-group
vocabulary and no `moe` block.

`moe_regularization` is a sibling of `tuning`, not part of it: which parameters receive
gradient and which auxiliary loss terms are added to the objective are unrelated
concerns, and only the first belongs in `tuning`.

### Invariants

1. `train` is the only freeze switch. `lr_scale` only scales the learning rate and
   must be `> 0`; `lr_scale: 0` is a parse error, not a silent freeze.
2. LoRA insertion is `groups.lora.train`. There is no second gate, so
   "configured but inert" is unrepresentable. `train` sets `requires_grad`; it never
   sets train/eval mode, which stays derived (see "Frozen encoder + LoRA").
3. `preset` is sugar: the parser materializes it into a complete group table and
   discards the preset afterwards. The trainer reads that one table and applies it
   **once**.
4. `moe.layer_indices` is only accepted for `moe_top_experts`; `groups.experts` and
   `groups.routers` are only accepted when `model.backbone.moe.enabled`.
5. The whole `finetune` block rejects unknown keys.
6. The groups are **disjoint and exhaustive**. `encoder` is the residual group
   (`backbone.*` minus tokenizers/experts/routers/projection/lora), so
   `{encoder, tokenizers}` x `{train, freeze}` spells all four combinations
   directly and none of them depends on key ordering.
7. Each variant keeps its **own** schema module. The variants are enforced forks (see
   "Why each variant implements the schema itself"); a cross-variant conformance test,
   not a shared import, is what keeps them from drifting.

The group is named `encoder`, not `backbone`, because `self.backbone` the module
*contains* the tokenizers — reusing that name for the group that *excludes* them is
exactly the ambiguity this schema exists to remove.

### Preset table

Materialized group tables (`t` = trains, `-` = frozen).

| preset | head | encoder | experts | routers | projection | lora | tokenizers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | t | t | t | t | t | - | t |
| `head_only` | t | - | - | - | - | - | - |
| `lora` | t | - | - | - | - | t | - |
| `moe_conservative` | t | t | t | - | - | - | - |
| `moe_conservative_routers` | t | t | t | t | - | - | - |
| `moe_top_experts` | t | - | selected only | - | - | - | - |
| `custom` | explicit `groups` required | | | | | | |

`full` means **full**: every group trains, `tokenizers` included. A config that wants
the tokenizers frozen — which in practice is nearly all of them — writes
`tokenizers: {train: false}` explicitly, so the freeze is visible in the file that
performs it rather than hidden in a preset named "full". The other presets keep
`tokenizers` frozen because that is what those named policies actually do, not because
of an inherited default.

`head_only` means head only. Under the legacy schema `mode: head_only` still trained any
LoRA parameter, because `lr_scales.lora` defaulted to `1.0`; here `lora` is a separate
preset.

`moe_conservative_routers` scales the routers by `0.01` rather than `1.0`, matching the
learning-rate defaults its legacy mode carried.

### Tokenizers vs encoder

These are separate axes and the schema keeps them separate. Under the legacy keys the
four combinations were reachable only as a side effect of evaluation order: the trainer
first applied `freeze_backbone_and_insert_lora` (which froze the *whole* backbone,
tokenizers included) and then let `freeze_tokenizer: false` unfreeze the tokenizers back
out of it.

| Intent | Legacy | `finetune.tuning` |
| --- | --- | --- |
| neither | `freeze_backbone_and_insert_lora: true`, `freeze_tokenizer: true` | `encoder: {train: false}`, `tokenizers: {train: false}` |
| tokenizers only | `freeze_backbone_and_insert_lora: true`, `insert_lora: false`, `freeze_tokenizer: false` | `encoder: {train: false}`, `tokenizers: {train: true}` |
| encoder only | `freeze_backbone_and_insert_lora: false`, `freeze_tokenizer: true` | `encoder: {train: true}`, `tokenizers: {train: false}` |
| both | `freeze_backbone_and_insert_lora: false`, `freeze_tokenizer: false` | `encoder: {train: true}`, `tokenizers: {train: true}` |

The legacy column's meaning depended on `freeze_tokenizer` being applied *after* the
LoRA helper — reorder the two blocks and two of the four rows change behavior silently.
Independent `train` flags on disjoint groups remove that: the table is the schema, not
an emergent property of it.

### Frozen encoder + LoRA

This is `preset: lora`, i.e. `encoder: {train: false}` + `lora: {train: true}`. It is
unambiguous even though the adapters physically live *inside* the encoder, because
`_semantic_group_for_param` matches `lora_` **first**: adapter weights are in the
`lora` group, never in `encoder`. The group table is a partition of parameters, so
"encoder frozen" and "adapters training" are not in conflict — they describe disjoint
parameter sets.

Two consequences worth stating, because they are the questions this configuration
actually raises:

**`train: false` on a group means `requires_grad = False`, not `.eval()`.** The
train/eval mode contract is *derived* from trainability, not configured (see
`_apply_backbone_mode_policy`). With adapters inserted, the encoder submodule contains
trainable parameters, so it stays in **train** mode — dropout active — which is what
LoRA training requires. `tokenizer_mapping` has no trainable parameter, so it is forced
to eval. That asymmetry is deliberate and should not be spelled in YAML; adding a
`mode:` key per group would let a config express "trainable but in eval mode", which is
never correct.

**`encoder: {train: true}` + `lora: {train: true}` is rejected.** Adapter insertion
freezes the backbone first, so the combination asks for something the insertion path
cannot build. Supporting it would mean reordering that freeze, which is a behavior
change; revisit only if a recipe actually wants adapters on a trainable encoder.

`separate_adapters` and the rest of `tuning.lora` are adapter *shape*, read only when
`groups.lora.train`.

## Converting a legacy config

The mapping is not a textual rename. Converting a config means **evaluating** the old
keys into a group table and then writing the new block. This is the table the loaders'
rejection messages point at:

| Old state | New |
| --- | --- |
| `freeze: false` (no `moe_tuning`) | `preset: full` + explicit `groups.tokenizers.train: false` |
| `freeze: true, insert: false` | `preset: head_only` |
| `freeze: true, insert: true` | `preset: lora` + `tuning.lora` shape |
| `moe_tuning.mode: head_only` | `preset: head_only` |
| `moe_tuning.mode: conservative_full_router_frozen` | `preset: moe_conservative` |
| `moe_tuning.mode: conservative_full_router_trainable` | `preset: moe_conservative_routers` |
| `moe_tuning.mode: top_moe_layer_expert_only` | `preset: moe_top_experts` + `moe.layer_indices` |
| `moe_tuning.mode: custom` | `preset: custom` + explicit `groups` |
| `freeze_tokenizer: true/false` | `groups.tokenizers.train` |
| any non-default `lr_scales[g]` | `groups[g].lr_scale` (`lr_scales.backbone` -> `groups.encoder`) |

Rule: build the group table from what the legacy keys *did* at runtime, compare it
against the preset table above, use the matching preset name, and fall back to
`preset: custom` with an explicit `groups` map when nothing matches exactly. Never
translate a key by name alone.

Four defaults make this more than a rename, and all four are easy to miss by reading a
config's text:

- `moe_tuning.lr_scales` defaulted **per mode**, and a `0.0` scale was itself a freeze.
- `train_moe_layer_indices` defaulted to the deepest MoE layer.
- `insert_lora` defaulted to `True` on `sleep2vec` and `False` on `sleep2vec2` and
  `sleep2expert`, so one file with `freeze_backbone_and_insert_lora: true` and no
  `insert_lora` describes two different runs depending on which variant loaded it.
- An `insert_lora: true` with `freeze_backbone_and_insert_lora: false` inserted nothing.
  Such a config converts to `groups.lora.train: false` — its real behavior — not to
  `true`. Translating the literal value would enable LoRA in a recipe that never used
  it and break reproducibility of results produced from it.

All four are transcribed executably in `tests/config/legacy_finetune_semantics.py`,
which is the authoritative statement of what the old keys did.

One legacy combination has no representation in the new schema. `sleep2expert` read
`freeze_backbone_and_insert_lora` and `moe_tuning` from the same config, so setting both
evaluates to a table that trains the encoder *and* inserts LoRA — which the new schema
rejects by design. Such a config has to be resolved by deciding which of the two
policies it meant, not converted.

**Old finetune checkpoints do not resume.** The optimizer now carries one parameter
group per `(semantic group, decay)` pair, so its state does not load. Restart through
`--pretrained-backbone-path`, as the README says. Inference on an existing checkpoint is
unaffected: `extract_embeddings` reads the config from the YAML file, never from the
checkpoint, so a converted YAML reconstructs identical adapter geometry.

## Why each variant implements the schema itself

**The isolation is deliberate and enforced.** `sleep2vec2/` and `sleep2expert/` are full
forks of the runtime, not layers over `sleep2vec/`. There are zero cross-variant imports,
and two tests actively forbid adding one:
`tests/variants/test_sleep2expert_namespace.py::test_sleep2expert_copied_runtime_uses_local_namespace`
rejects any `sleep2expert/` file matching `(from|import) (sleep2vec2|sleep2vec|data|preprocess)`,
and `tests/variants/test_sleep2vec2_namespace.py` does the same for `sleep2vec2/`. A
shared schema module placed in either variant would fail those tests; placed in a new
top-level package it would still cut against the fork policy AGENTS.md assigns to the
`variant-maintainer` role.

**The vocabularies genuinely differ.** `sleep2expert` has seven groups; `sleep2vec` and
`sleep2vec2` have five and no MoE block at all. A shared module would have to carry
`experts`/`routers`/`moe.layer_indices` as permanently-inert keys in two of the three
variants, which reintroduces exactly the "configured but does nothing" failure mode this
schema exists to remove.

**The thing worth sharing is small and is better tested than imported.** The trainability
schema is a compact block per variant, and the only real risk of duplication is default
drift — which is what the legacy `insert_lora` split was. Two things prevent it:
`preset` is a **required** key with no default, and a required key cannot drift; and
`tests/variants/test_finetune_tuning_conformance.py` pins what the three schemas must
agree on — the shared group names and their relative order, the shared preset names,
identical meanings for `full`/`head_only`/`lora` over the shared groups, every preset
covering every group, every preset training the head, no preset pairing a trainable
encoder with LoRA, and no preset pairing a frozen group with a non-neutral `lr_scale`
(replayed through the parser too, since an override could otherwise defeat it).

That gives the same protection against drift as a shared import, costs one test file,
and keeps the fork boundary intact.
