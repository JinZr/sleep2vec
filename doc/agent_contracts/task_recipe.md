# Task Recipe Contract

Task recipes under `recipes/` bind one task to an experiment and step. The accepted fields and finite allowlists are defined in the [task recipe schema](../../recipes/schemas/task_recipe.schema.md).

## Authored-input closure

Recipe shape is validated before config inspection and before any workspace, script, manifest, or event is created. Validation is task-aware and owner-based: `experiment_workspace` owns `experiment` and `step`; task decision owners own `inputs`, `evaluation_policy`, `execution`, `search`, and `adaptive`; renderer mappings own runtime and preset CLI fields; and `plans` owns top-level routing and artifacts. This does not add a second schema registry or general recipe facade.

Unknown and task-inapplicable fields fail with their original field path and source layer. For hparam recipes, the base finetune source and local tuning overlay are validated independently before merged semantics run. Raw authored `_...` fields are reserved and rejected.

## Effective recipe

Planning produces one effective recipe:

```text
recipe fields + recipe decisions + explicit user decisions
  -> materialized recipe
  -> config summary and consultation
  -> frozen plan and resolved recipe
```

Recipe decisions with a task-owned canonical field are written into that field first, then explicit user decisions may override both the canonical field and effective decision mapping before config inspection and consultation are rerun. A layered hparam recipe takes its task only from the local overlay or explicit user decision; the base finetune task cannot become the effective tuning task. Policy-only decisions remain under `decisions` rather than creating inert recipe sections. For finetune and hparam tasks, an explicit `required_channels` decision must match `preset_build.required_channels` in the selected config. For hparam, `inputs.ckpt_path` is reserved for the selected final-evaluation checkpoint and is not rendered into tuning finetune commands. Empty or null rendered decisions remain unresolved instead of falling back to older canonical values; explicit `pretrained_backbone_path: null` retains its established train-without-pretraining meaning. `plan.json` and `recipe.resolved.yaml` must contain the same complete effective recipe. Retained base/local recipe copies are source audit only; launch, selection, adaptive, and postprocess consumers read the effective recipe. Frozen recipes containing trusted `_base_recipe` or `_local_recipe` metadata are consumed through `run_artifacts.read_hparam_plan`; they are not re-entered through the authored-recipe loader.

Decision-file behavior and precedence belong to [user_decisions.md](user_decisions.md).

## Experiment binding

Every runnable recipe declares complete `experiment` metadata (`id`, `title`, `objective`, `root`, and `baseline`) and a `step` (`id`, `phase`, and `purpose`). A hparam recipe declares its own binding rather than inheriting it from its base finetune recipe.

The plan directory must be inside `experiment.root`. Workspace layout, path canonicalization, step registration, and lifecycle ownership belong to [experiment_workspace.md](experiment_workspace.md).

## Task and variant routing

| task | accepted variant | generated runtime |
| --- | --- | --- |
| `sleep2stat` | omitted or `null` | `python -m sleep2stat` |
| `preset_prepare` | `sleep2vec`, `sleep2vec2`, `sleep2expert` | package-local `preprocess/save_dataset_presets.py` |
| `finetune`, `hparam_tune` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.finetune` |
| `infer`, `evaluate` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.infer` |

Preset preparation routes each variant to its package-local script. Variant scripts reject root-only `manifest_output` and `write_sidecar_manifest` fields instead of falling back to the root runtime. `sex_age_baseline` does not own preset generation.

`pretrain` and `adapt` have direct runtime skills and CLIs but are not runnable task-recipe values because agent tools have no renderer for them. Missing or unsupported routing blocks command generation.

Runnable non-hparam scripts enter `REPO_ROOT` for cwd and PYTHONPATH. User-authored semantic dataset and checkpoint paths retain their supplied meaning.

## Hparam workflow

Search keys are explicit `runtime.<name>` fields or `yaml:/json/pointer/path` config overrides. `search.max_runs` is a required positive budget. Removed bare or `param.*` forms are rejected rather than translated.

The optional `execution` block configures the managed launcher. `hparam-launch` is the supported execution entry; generated leaf scripts are frozen snapshots. Frozen execution identity and its canonical owner are defined in [run_manifest.md](run_manifest.md).

The optional `adaptive` block defines append-only rounds bounded by `adaptive.max_runs_total`. Test/external objectives require explicit test feedback authorization. The current source recipe and each generated suggestion must pass read-only preflight before digest, suggestion, or event artifacts are written. Earlier round plans, configs, logs, and checkpoints are not rewritten.

`reports/ranking.csv` is shared across plans in the same step. Runnable hparam plans in that step must use the same selection metric and mode; selection replaces current-plan keys and reranks the complete step. Candidate ownership, frozen-field validation, and checkpoint evidence are defined in [run_manifest.md](run_manifest.md). Final external-test generation follows [external_test_locking.md](external_test_locking.md).

Managed run identity, status, atomic commit, projections, and evidence belong exclusively to [run_manifest.md](run_manifest.md).
