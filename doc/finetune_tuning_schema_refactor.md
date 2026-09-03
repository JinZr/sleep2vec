# Finetune Trainability Schema Refactor

Design note. No code changes yet; this document is the proposal to review before
implementation.

## Scope

In scope: the `finetune` block of `sleep2vec/`, `sleep2vec2/`, `sleep2expert/`.

Explicitly out of scope, and why:

- **`adapt.stage2.lr_scales`** is a parallel trainability system with a different
  vocabulary (`encoder`, `shared_legacy`, `new_modalities`, built from
  `get_adaptation_param_groups`) and a different mechanism: it selects parameters by
  *phase* and skips `requires_grad=False` tensors, so it never overloads `lr_scale`
  as a freeze switch. It does not have the bug this refactor fixes. Note the tension
  it leaves behind: after this change, `finetune.tuning.groups.encoder.lr_scale` and
  `adapt.stage2.lr_scales.encoder` are two spellings of one concept, which AGENTS.md
  ("one canonical spelling and location for each concept") would eventually want
  reconciled. Follow-up, not this change.
- **`sleep2vec_moe/`** ships `pretrain` only — no finetune surface.
- **`sex_age_baseline/config.py`** has its own unrelated finetune config and none of
  these four keys.

## Problem

Four independent YAML entry points decide which parameters train, and the runtime
applies them in sequence, last writer wins.

| Order | Key | Effect | Owner |
| --- | --- | --- | --- |
| 1 | `finetune.lora.freeze_backbone_and_insert_lora` | Freezes every backbone parameter | `sleep2vec_finetuning.py` |
| 2 | `finetune.lora.insert_lora` | Inserts LoRA; **only read when 1 is true** | `downstream_model.freeze_backbone_and_insert_lora` |
| 3 | `finetune.freeze_tokenizer` | Re-writes tokenizer trainability after 1/2 | `backbone.set_tokenizers_trainable` |
| 4 | `finetune.moe_tuning.mode`, `lr_scales[g] == 0`, `freeze_router`, `freeze_experts` | Overwrites every group (sleep2expert only) | `_apply_moe_tuning_policy` |

Nothing derives the final state from one place, so a policy at step 4 can silently
invalidate an assumption captured at step 1. PR #230 is exactly that failure: the
"keep a frozen backbone in eval mode" contract was attached to step 1, while the
head-only recipe freezes the backbone at step 4.

### Evidence

Measured over the 69 finetune configs under `configs/` that carry a `finetune.lora` block:

| Observation | Count |
| --- | --- |
| `insert_lora: true` **and** `freeze_backbone_and_insert_lora: true` (LoRA actually inserted) | 3 |
| `insert_lora: true` but `freeze_backbone_and_insert_lora: false` (**inert; LoRA never inserted**) | 30 |
| `freeze_backbone_and_insert_lora: true` with `insert_lora: false` (the path PR #230's flag was written for) | 0 |
| `freeze_tokenizer: true` | 69 (never varied) |
| `moe_tuning` present | 9 |

- 30 recipes advertise LoRA that never runs. `agent_tools/domain/finetune_hparam_profile.py:257`
  already carries a comment working around "an inert `insert_lora=true`", so the
  schema is misleading its automated consumers too.
- The only frozen-backbone recipe in the tree freezes through `moe_tuning`, not through
  the LoRA helper — i.e. the flag PR #230 keyed on is dead in every checked-in config.
- `lr_scales` carries two meanings: a learning-rate multiplier and a freeze switch
  (`== 0.0` freezes, in `_set_param_trainability_from_policy`).
- Parsing strictness is inconsistent: `lora` and `data` are built with
  `LoraConfig(**block)` (unknown key raises), while the `finetune` block is read key by
  key with `.get()` (`sleep2expert/config.py:1401`), so a typo there is silently ignored.
- The schema is triplicated across `sleep2vec/config.py`, `sleep2vec2/config.py`,
  `sleep2expert/config.py` (954 / 989 / 1453 lines) and the copies have already drifted:
  `LoraConfig.insert_lora` defaults to `True` in sleep2vec and `False` in sleep2expert.

## Target schema

One block owns trainability. LoRA describes adapter shape only.

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

`moe_regularization` moves out of `moe_tuning` and becomes a sibling of `tuning`.
The old block conflated two unrelated concerns: which parameters receive gradient,
and which auxiliary loss terms are added to the objective. Only the first belongs
in `tuning`. One config (`configs/sleep2expert/moe/router_trainable.yaml`) sets it
to something other than `{enabled: false}`, so the migration must carry it across
rather than drop it.

### Invariants

1. `train` is the only freeze switch. `lr_scale` only scales the learning rate and
   must be `> 0`; `lr_scale: 0` is a parse error, not a silent freeze. Migration
   rewrites every existing `lr_scales[g]: 0.0` into `train: false`.
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
7. Each variant keeps its **own** schema module. The variants are enforced forks
   (see "Why not one shared module"); a cross-variant conformance test, not a
   shared import, is what keeps them from drifting.

### Preset table

Materialized group tables (`t` = trains, `-` = frozen).

`full` means **full**: every group trains, `tokenizers` included. That is deliberately
*not* what today's configs do — all 69 freeze the tokenizers — so the 57 configs that
map to `full` migrate with an explicit `tokenizers: {train: false}` override rather than
inheriting a frozen default from a preset named "full". The freeze becomes visible in
every file that performs it, and the preset name stops lying. Cost: 57 extra lines in
the migration diff.

| preset | head | encoder | experts | routers | projection | lora | tokenizers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | t | t | t | t | t | - | t |
| `head_only` | t | - | - | - | - | - | - |
| `lora` | t | - | - | - | - | t | - |
| `moe_conservative` | t | t | t | - | - | - | - |
| `moe_conservative_routers` | t | t | t | t | - | - | - |
| `moe_top_experts` | t | - | selected only | - | - | - | - |
| `custom` | explicit `groups` required | | | | | | |

Only `full` changed meaning here. The other presets keep `tokenizers` frozen because
that is what those named policies actually do — `moe_conservative` and friends set the
tokenizer scale to `0.0` today — not because of an inherited default.

Semantic tightening to note: today `mode: head_only` still trains any LoRA parameter
(`lr_scales.lora` defaults to `1.0`), so "head only" is really "head plus LoRA". Under
the new table `head_only` means head only and `lora` is a separate preset. No current
config changes behavior, because no `head_only` config inserts LoRA.

### Tokenizers vs encoder

These are separate axes and the schema must keep them separate. Today the four
combinations are reachable only as a side effect of evaluation order: the trainer
first applies `freeze_backbone_and_insert_lora` (which freezes the *whole* backbone,
tokenizers included) and then lets `freeze_tokenizer: false` unfreeze the tokenizers
back out of it.

| Intent | Today | New |
| --- | --- | --- |
| neither | `freeze_backbone_and_insert_lora: true`, `freeze_tokenizer: true` | `encoder: {train: false}`, `tokenizers: {train: false}` |
| tokenizers only | `freeze_backbone_and_insert_lora: true`, `insert_lora: false`, `freeze_tokenizer: false` | `encoder: {train: false}`, `tokenizers: {train: true}` |
| encoder only | `freeze_backbone_and_insert_lora: false`, `freeze_tokenizer: true` | `encoder: {train: true}`, `tokenizers: {train: false}` |
| both | `freeze_backbone_and_insert_lora: false`, `freeze_tokenizer: false` | `encoder: {train: true}`, `tokenizers: {train: true}` |

Two observations. The "tokenizers only" row is the dead combination found in the
audit: **0 of 69** configs use it, and it is the only way to reach that state, which
is a fair sign nobody realised it was reachable. And the row's meaning depends on
`freeze_tokenizer` being applied *after* the LoRA helper — reorder the two blocks and
two of the four rows change behavior silently. Independent `train` flags on disjoint
groups remove both problems: the table above is the schema, not an emergent property
of it.

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

**`encoder: {train: true}` + `lora: {train: true}` is rejected.** Today it is
unreachable: `insert_lora` is only read when `freeze_backbone_and_insert_lora` is true,
and that helper's first act is an unconditional freeze of every backbone parameter. The
new schema makes the combination *spellable*, so the parser must reject it explicitly,
otherwise a config could ask for something the insertion path cannot build. Supporting
it would mean reordering the freeze inside `freeze_backbone_and_insert_lora`, which is
a behavior change and out of scope here.

`separate_adapters` and the rest of `tuning.lora` are adapter *shape*, read only when
`groups.lora.train` — matching today's rule that they are meaningless without
insertion. On the inference side this is what collapses
`_finetune_adapters_enabled` (currently the literal AND of the two booleans) into a
single flag.

## Migration

The mapping is not a textual rename. The migration tool must **evaluate** the old
config into a group table and then emit the new block:

| Old state | New |
| --- | --- |
| `freeze: false` (no `moe_tuning`) — 57 configs | `preset: full` + explicit `groups.tokenizers.train: false` |
| `freeze: true, insert: false` | `preset: head_only` |
| `freeze: true, insert: true` | `preset: lora` + `tuning.lora` shape |
| `moe_tuning.mode: head_only` | `preset: head_only` |
| `moe_tuning.mode: conservative_full_router_frozen` | `preset: moe_conservative` |
| `moe_tuning.mode: conservative_full_router_trainable` | `preset: moe_conservative_routers` |
| `moe_tuning.mode: top_moe_layer_expert_only` | `preset: moe_top_experts` + `moe.layer_indices` |
| `moe_tuning.mode: custom` | `preset: custom` + explicit `groups` |
| `freeze_tokenizer: true/false` | `groups.tokenizers.train` |
| any non-default `lr_scales[g]` | `groups[g].lr_scale` (`lr_scales.backbone` -> `groups.encoder`) |

Rule for the tool: build the group table from the *current* runtime semantics, compare
it against each preset, emit the matching preset name, and fall back to
`preset: custom` with an explicit `groups` map when nothing matches exactly. Never
translate a key by name alone.

### The one dangerous case

The 30 configs with an inert `insert_lora: true` must migrate to
`groups.lora.train: false` — their real behavior today. Translating the literal `true`
would silently enable LoRA in 30 recipes and break reproducibility of every result
produced from them. The tool should list these files for explicit sign-off, since some
of them were probably *intended* to use LoRA:

```
configs/cls_emb/sleep2vec_dense_finetune_cls.yaml
configs/cls_emb/sleep2vec_dense_finetune_reg.yaml
configs/examples/age/FINETUNE_EXAMPLE.yaml
configs/examples/ahi/FINETUNE_EXAMPLE.yaml
configs/examples/arousal/FINETUNE_EXAMPLE.yaml
configs/examples/stage3/FINETUNE_EXAMPLE.yaml
configs/heartbeat_breath_age_finetune_large.yaml
configs/heartbeat_breath_ahi_finetune_large.yaml
configs/heartbeat_breath_sex_finetune_large.yaml
configs/ppg_age_finetune_large.yaml
configs/ppg_ahi_finetune.yaml
configs/ppg_ahi_finetune_large.yaml
configs/ppg_ahi_finetune_large_temporal_conv.yaml
configs/ppg_cox_finetune_large.yaml
configs/ppg_sex_finetune_large.yaml
configs/ppg_stage3_finetune.yaml
configs/ppg_stage3_finetune_large.yaml
configs/ppg_stage4_finetune.yaml
configs/ppg_stage4_finetune_large.yaml
configs/ppg_stage5_finetune.yaml
configs/ppg_stage5_finetune_large.yaml
configs/sleep2vec_dense_finetune_cls.yaml
configs/sleep2vec_dense_finetune_custom_cls.yaml
configs/sleep2vec_dense_finetune_custom_reg.yaml
configs/sleep2vec_dense_finetune_reg.yaml
configs/stage5_all_9_channels_finetune_kaldi.yaml
configs/token_emb/sleep2vec_dense_finetune_cls_attn.yaml
configs/token_emb/sleep2vec_dense_finetune_cls_mean.yaml
configs/token_emb/sleep2vec_dense_finetune_reg_attn.yaml
configs/token_emb/sleep2vec_dense_finetune_reg_mean.yaml
```

The three configs that really use LoRA — `configs/examples/{sex,stage4,stage5}/FINETUNE_EXAMPLE.yaml`
— migrate to `preset: lora`.

### Hard cut

No compatibility window. `load_finetune_config` rejects every old key
(`finetune.lora.freeze_backbone_and_insert_lora`, `finetune.lora.insert_lora`,
`finetune.freeze_tokenizer`, `finetune.moe_tuning`) with an error naming the replacement
key and the migration command. All 69 in-tree configs are rewritten in the same commit;
the equivalence gate below is what makes that safe.

Consequence to plan for: a config outside this repo, or an hparam plan already frozen
against the old keys, fails immediately instead of drifting. The error message is the
mitigation, so it must carry the offending key, the file, and the exact replacement
block.

### Equivalence gate

For all 69 checked-in configs: parse through the compat shim and through the new block,
materialize both group tables, and assert `(train, lr_scale)` are identical per group.
This runs without torch and is the gate that makes the migration reviewable.

## Persisted artifacts

The keys do not only live in YAML. Three on-disk surfaces derive from them.

**The run status file, renamed `moe_finetune_status.json` -> `finetune_status.json`.**
Written per run by `sleep2expert/finetune.py:222`, and on the `allowed_files` whitelist
at `sleep2expert/finetune.py:127`. Its schema is shaped by the old keys:
`moe_tuning_present`, `moe_tuning_mode`, a `lr_scales` map, and a `param_groups` map
that exists only when `moe_tuning` is present — otherwise it falls back to a single
synthetic `"legacy"` group. After the refactor every config has a group table, so
`moe_tuning_present` is always true, `moe_tuning_mode` becomes `preset`, and the
`"legacy"` branch disappears. The file is then no longer MoE-specific and all three
variants emit it, which is why it takes the honest name. The old name is **retired, not
dual-written** — consistent with the hard-cut decision. Concretely: update the
`allowed_files` whitelist, and keep a `schema_version` field inside the file so a future
change does not have to be inferred from field names.

**Logged metrics.** `_flatten_moe_status` turns that dict into logger keys such as
`moe_finetune/lr_scales/backbone` and `moe_finetune/moe_tuning_mode`. The group rename
changes those keys. Confirmed with the repo owner that nothing outside this repo reads
them, so this needs no announcement and no separate commit — it rides along with the
hard cut. Historical runs keep their old metric names; a dashboard spanning the cut will
show two series.

**Checkpoints.** `on_save_checkpoint` stores `checkpoint["finetune_config"]` and
`finetune_config_yaml`. Confirmed **write-only**: nothing in the repo reads either field
back. Existing checkpoints therefore keep the old shape forever with no runtime effect.

Related reassurance, worth stating because it is the first question a reviewer will ask:
**old checkpoints stay usable.** `extract_embeddings._load_config_bundle` reads the
config from the YAML file (`load_finetune_config(args.config)`), never from the
checkpoint, so inference on an existing checkpoint works as long as its YAML is migrated
with the rest. The equivalence gate is what guarantees the migrated YAML reconstructs
identical adapter geometry.

## Rollback

The hard cut is one commit touching 69 configs plus three parsers, so rollback is
`git revert` of that commit — but only until a new run has written a
`finetune_status.json` in the new format, or a new checkpoint has embedded a new
`finetune_config`. Neither is read back, so even then the revert is safe; the cost is
that any config edited *after* the cut has to be hand-translated back. Practical rule:
land the cut early in a week, and do not batch unrelated config edits into it.

## Why not one shared module

An earlier draft of this document proposed extracting one finetune schema module for
all three variants. That is the wrong call here, for three reasons.

**The isolation is deliberate and enforced.** `sleep2vec2/` and `sleep2expert/` are
full forks of the runtime, not layers over `sleep2vec/`. There are zero cross-variant
imports today, and two tests actively forbid adding one:
`tests/variants/test_sleep2expert_namespace.py::test_sleep2expert_copied_runtime_uses_local_namespace`
rejects any `sleep2expert/` file matching `(from|import) (sleep2vec2|sleep2vec|data|preprocess)`,
and `tests/variants/test_sleep2vec2_namespace.py` does the same for `sleep2vec2/`. A
shared schema module placed in either variant would fail those tests; placed in a new
top-level package it would still cut against the fork policy AGENTS.md assigns to the
`variant-maintainer` role. Reversing that policy is a much larger decision than this
refactor, and this refactor does not need it.

**The vocabularies genuinely differ.** `FinetuneLrScalesConfig` exists only in
`sleep2expert/config.py`. The MoE variant has seven groups; `sleep2vec` and
`sleep2vec2` have four (`head`, `encoder`, `tokenizers`, `lora`) and no MoE block at
all. A shared module would have to carry `experts`/`routers`/`moe.layer_indices` as
permanently-inert keys in two of the three variants, which reintroduces exactly the
"configured but does nothing" failure mode this refactor exists to remove.

**The thing worth sharing is small and is better tested than imported.** The
trainability schema is roughly 67 lines per variant, and `diff sleep2vec sleep2vec2`
over that region shows only the `insert_lora` default (`True` vs `False`) and two
`covariate_fusion` fields. The only real risk of duplication is that kind of default
drift, and the fix for it is:

- Make `preset` a **required** key with no default. A required key cannot drift.
- Add `tests/variants/test_finetune_schema_conformance.py`: load one shared YAML
  fixture through all three `load_finetune_config` functions and assert the
  materialized group tables are identical for every preset the variant supports.
  Group names outside a variant's vocabulary must raise, and the test asserts which
  names each variant rejects.

That gives the same protection against drift as a shared import, costs one test file,
and keeps the fork boundary intact.

## Blast radius

| Area | Files |
| --- | --- |
| Schema and parsing | `sleep2vec/config.py`, `sleep2vec2/config.py`, `sleep2expert/config.py` (three parallel implementations, one conformance test) |
| Namespace flattening | `{sleep2vec,sleep2vec2,sleep2expert}/common.py:402` |
| Policy application | `{sleep2vec,sleep2vec2,sleep2expert}/sleep2vec_finetuning.py`; `_apply_moe_tuning_policy` and `_set_param_trainability_from_policy` become the single apply site for all three variants |
| Adapter reconstruction | `{sleep2vec,sleep2vec2,sleep2expert}/extract_embeddings.py:492` (`_finetune_adapters_enabled` collapses to one flag) |
| Agent tooling | `agent_tools/domain/finetune_summary.py:159`, `agent_tools/domain/finetune_hparam_profile.py:247` — the `adaptation.strategy` axis becomes three presets on `yaml:/finetune/tuning/preset` instead of a two-flag product on `yaml:/finetune/lora` |
| Config validation | `utils/check_configs.py:110` keys off `"moe_tuning" in finetune_block` |
| Run artifacts | `sleep2expert/finetune.py:127,222` (rename to `finetune_status.json`, update whitelist); `_build_moe_finetune_status` / `_flatten_moe_status` in `sleep2expert/sleep2vec_finetuning.py:331,389`; the same emitter added to `sleep2vec/` and `sleep2vec2/` |
| Optimizer groups | `sleep2expert/sleep2vec_finetuning.py:2036` `configure_optimizers` reads `_finetune_lr_scales[group]` — the rename to `encoder` lands here too |
| Documentation | `README.md:426,440,457-468` documents `freeze_tokenizer` / `freeze_backbone_and_insert_lora` / `insert_lora` as the public interface |
| Configs | 69 finetune YAMLs |
| Tests | Ranked by number of references to the old keys: `tests/variants/test_sleep2expert_moe_config.py` (41), `tests/variants/test_sleep2expert_finetune_moe_tuning.py` (27), `tests/agent_tools/test_agent_tools_hparam_profiles.py` (14), `tests/models/test_downstream_separate_adapters.py` (8), `tests/config/test_common_finetune_apply.py` (8), `tests/config/test_config_loading.py` (7), `tests/agent_tools/test_agent_tools_config_summary.py` (5), `tests/config/test_check_configs.py` (3), `tests/variants/test_sleep2{vec2,expert}_kaldi_backend.py` (3 each), `tests/variants/test_sleep2expert_moe_forward.py` (2), `tests/variants/test_sleep2expert_subnetwork_export.py` (1) |

Stored hparam recipes and run manifests that recorded the JSON pointer
`yaml:/finetune/lora` will not resolve after the rename. With the hard cut, those plans
have to be re-frozen, and the pointer resolver must fail loudly on the old form rather
than silently yielding an empty axis.

## Sequencing

1. Land the PR #230 eval-mode fix. It is a correctness bug and is independent of the schema.
2. Implement the schema per variant, plus the cross-variant conformance test (no
   behavior change, no YAML change).
3. Add the new `finetune.tuning` block and the equivalence gate.
4. Write the migration tool as `utils/migrate_finetune_tuning.py` (there is no existing
   migration-script precedent in `utils/`, so this sets one: it must be idempotent,
   support `--check` for CI, and refuse to run on a dirty tree).
5. Run the migration tool; commit the rewritten YAMLs, the new block, and the deletion
   of the old keys as one hard cut. Update `utils/check_configs.py` and `README.md` in
   the same commit — a config validator that still looks for `moe_tuning` would pass
   every migrated file vacuously.
6. Rename the status file to `finetune_status.json` (add `schema_version`, drop the
   `"legacy"` branch, emit it from all three variants, update the `allowed_files`
   whitelist and retire the old name).
7. Update agent tooling axes and pointers; re-freeze any plan pinned to
   `yaml:/finetune/lora`.

## Decisions

Settled 2026-09-04.

- **`lr_scale: 0` is rejected.** `train: false` is the only way to freeze a group. The
  migration tool converts the affected `moe_tuning` configs.
- **`tokenizers` stays a group.** It already exists in the sleep2expert group split and
  in `lr_scales`; dropping it would cost code changes for no gain.
- **No deprecation window.** Old keys raise; all 69 configs migrate in one commit.
- **The 30 inert-LoRA configs migrate to `lora.train: false`** — behavior unchanged,
  historical results reproducible. The migration tool still emits that list separately so
  individual files can be moved to `preset: lora` later, deliberately.
- **No shared schema module.** Each variant implements the schema itself; a
  cross-variant conformance test replaces the shared import. See
  "Why not one shared module".
- **`encoder: train: true` together with `lora: train: true` is a parse error**,
  preserving today's reachable set. Revisit only if a recipe actually wants adapters on
  a trainable encoder.
- **`adapt.stage2.lr_scales` is out of scope** and keeps its own vocabulary. See "Scope".
- **The group is named `encoder`, not `backbone`.** `self.backbone` the module
  *contains* the tokenizers, so reusing the name for the group that *excludes* them is
  the ambiguity this refactor should kill. The hard cut makes the rename free; only
  `lr_scales.backbone` is affected, in sleep2expert configs. Confirmed that nothing
  outside this repo consumes the `moe_finetune/*` logged metrics, so the metric rename
  needs no announcement and no separate commit.
- **`preset: full` trains tokenizers.** The 57 configs that map to it carry an explicit
  `tokenizers: {train: false}` override, so the freeze is visible per file instead of
  hidden in a preset default.
- **The status file is renamed `finetune_status.json`**; `moe_finetune_status.json` is
  retired, not dual-written.

### Settled during implementation

- **`finetune.tuning.lora` is allowed under a preset that does not train LoRA.** The
  block holds hyperparameters (`r`, `alpha`, `dropout`, `target_modules`, `use_dora`,
  `separate_adapters`), not a switch, so an unused one is not the `insert_lora` defect
  coming back. Rejecting it would also make preset sweeps impossible: the automatic
  hparam profile flips `finetune.tuning.preset` alone across `full`/`head_only`/`lora`,
  and a config that carries LoRA hyperparameters would fail validation on two of the
  three arms. `moe.layer_indices` keeps its strict check, because it selects *which
  parameters train* and a stale value there is a trainability bug, not dead
  hyperparameters.
- **`lr_scale` must be finite.** `nan <= 0.0` is false, so the `> 0` check alone let a
  NaN scale through into the optimizer's learning rate.
- **The migration tool scans `recipes/` as well as `configs/`.** Finetune configs also
  ship as recipe fixtures (`recipes/examples/fixtures/tiny_finetune_config.yaml`), and
  they need the same rewrite.
- **A config with no legacy keys still gets a `tuning` block.** `finetune.tuning` is
  required now, so "no legacy keys" means "relied on the legacy defaults", which trained
  everything except the tokenizers. Only the `sex_age_baseline` configs are skipped:
  they are a different runtime with no trainability groups at all.
- **The migration manifest merges instead of replacing.** An entry can only be derived
  from a config's *legacy* text, which is gone once that config is migrated, so a second
  run of the tool must not drop the entries the first run recorded.
- **Two groups have sub-group granularity, and the policy pass has to honor both.**
  `moe_top_experts` trains only the experts of the selected MoE layers, and
  `separate_adapters` trains only the `ch_<channel>` adapters — never the `default`
  adapter PEFT creates alongside them. A pass that writes `requires_grad` from the group
  table alone un-freezes `default`, because adapter insertion runs first and the policy
  pass runs last. The adapter question is answered in one place,
  `Sleep2vecDownstreamModel.lora_param_is_trainable`, which both the insertion helper
  and the policy pass call.

## What was verified

`tests/config/test_finetune_tuning_equivalence.py` replays the manifest's 70 recorded
legacy tables against the new parser: for every checked-in config, the group table the
new block produces equals the table the old keys produced. That is the substantive
proof that the hard cut preserved behavior.

`tests/variants/test_finetune_tuning_conformance.py` pins what the three forked schemas
must agree on: the shared group names and their relative order, the shared preset names,
identical meanings for `full`/`head_only`/`lora` over the shared groups, every preset
covering every group, every preset training the head, no preset pairing a trainable
encoder with LoRA, and no preset pairing a frozen group with a non-neutral `lr_scale`.

Not verified here: anything that imports torch. `torch`, `pytorch_lightning` and `peft`
are absent from the development environment, so the runtime apply sites
(`_apply_finetune_tuning_policy`, `_assert_tuning_invariants`, `configure_optimizers`)
are covered by review and by the config-layer equivalence gate, not by an executed test.
`tests/models/test_finetune_separate_adapters.py` builds a full finetuning module under
`preset: lora` with `separate_adapters: true` and asserts the `default` adapter reaches
neither `requires_grad` nor an optimizer group; it skips here for the same reason.
The pre-existing failures in `tests/variants` are identical before and after this change.
