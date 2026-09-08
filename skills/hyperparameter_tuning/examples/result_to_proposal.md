# Worked result-to-proposal example

This is invented teaching evidence, not a runnable recipe or an experiment result.
The scientific task, validation objective, data, checkpoint source and execution
identity are already authorized. The initial domain includes learning rate
`[3e-5, 6e-4]`, epochs `[8, 16]` and head dropout `[0.1, 0.3]`; each round permits
two runs and eight total runs remain. Other settings are fixed with a recorded
reason. These values illustrate the reasoning, not recommended defaults.

## Design the opening batch

Keep the domain separate from the first two points. This source fragment spends
two runs on an LR comparison while leaving training length and dropout available
for later proposals:

```yaml
search:
  method: grid
  max_runs: 2
  parameters:
    runtime.lr: [0.00003, 0.0006]
    runtime.epochs: [8, 16]
    yaml:/model/head/dropout: [0.1, 0.3]
  configurations:
    - runtime.lr: 0.0001
      runtime.epochs: 8
      yaml:/model/head/dropout: 0.1
    - runtime.lr: 0.0003
      runtime.epochs: 8
      yaml:/model/head/dropout: 0.1
```

It requires the enabled `agent_proposal` workflow described in the skill.
The numeric intervals come from the domain endpoints here; explicit bounds can
instead define them. The two points need not use those endpoints. The same
`search.configurations` field is not valid alongside parameters in a static grid.

## Read the evidence

| Completed run | Learning rate | Epochs | Head dropout | Validation AUROC | Best checkpoint epoch |
| --- | --- | --- | --- | --- | --- |
| A | 1e-4 | 8 | 0.1 | 0.730 | 7 |
| B | 3e-4 | 8 | 0.1 | 0.738 | 7 |

The frozen configs differ only in learning rate. Both successful runs have complete
required results; B is the workflow incumbent. Epochs are zero-indexed, so both
best checkpoints are at the final epoch. There are no repeated runs or available
learning curves in this example. Read earlier round results and rationale before
assuming this pair contains the best historical evidence.

A weak rationale is: "B is best, so increase learning rate and epochs."
It neither separates the two possible explanations nor acknowledges uncertainty.

## Choose two informative points

| Candidate | Learning rate | Epochs | Head dropout | Purpose |
| --- | --- | --- | --- | --- |
| C | 3e-4 | 12 | 0.1 | Test a longer training horizon at the incumbent learning rate. |
| D | 4e-4 | 8 | 0.1 | Test a higher learning rate at the existing training horizon. |

Use two complete `configurations` points; independent lists for learning rate and
epochs would accidentally request four combinations. Copy the issued request and
evidence identities using the current [proposal handshake](../../../doc/agent_contracts/task_recipe.md#proposal-handshake),
not the illustrative A/B labels above. For example, the existing `rationale`
string could contain:

> B exceeds A by 0.008 AUROC while differing only in learning rate. With no repeats,
> this is a promising direction rather than a known improvement above noise. Both
> best checkpoints occur at the last epoch, which leaves training length unresolved;
> missing curves prevent an underfitting claim. C extends training at B's learning
> rate. D tests a higher learning rate while keeping the original horizon. Dropout
> stays fixed so these two runs address those alternatives within the authorized
> bounds and remaining budget. If C improves and D does not, longer training gains
> support; the reverse favors this local LR direction. If neither improves, retain
> B and reconsider the alternatives rather than automatically narrowing around C
> or D. If both improve, a later joint point may be worthwhile, without attributing
> its outcome to one parameter alone.

This is one defensible allocation, not a mandatory strategy. Repeating B could be
more useful when existing evidence suggests noise dominates the observed gap and
the contract permits the intended repeat. Missing curves should lead to a stated
limitation or targeted reading of existing artifacts, not invented evidence.

## Learn from the next results

Suppose C returns 0.741 and D returns 0.731. Record the observation and update the
interpretation through `experiment-note`: the longer horizon is promising at this
LR; the small advantage still lacks repeat evidence; the higher-LR point did not
improve on B. Keep the actual incumbent across all completed rounds, including B
if later candidates are worse. Check C's best epoch and available trajectory before
choosing another longer horizon. Do not enlarge a frozen boundary or spend beyond
the authorized budget merely because the new best point is on a boundary.

## Same best score, different trajectories

These are two alternative synthetic histories for C, both with a best validation
AUROC of 0.741. Each reports validation points at epochs 3, 7 and 11; the training
loss values come from the corresponding logged epochs. Suppose separate completion
evidence confirms that both reached the configured 12-epoch horizon.

| History | Validation AUROC at epochs 3, 7, 11 | Training loss at those epochs | A defensible next comparison |
| --- | --- | --- | --- |
| Still improving | 0.730, 0.737, 0.741 | 0.49, 0.43, 0.40 | Extend C to 16 epochs with LR and dropout fixed; compare with a different LR at 12 epochs if both fit the remaining budget. |
| Earlier peak | 0.741, 0.734, 0.729 | 0.49, 0.43, 0.40 | Compare a lower LR with stronger dropout at the same horizon, changing one axis per point. |

The first history makes a longer horizon worth testing, without proving that it
will help. No improvement beyond C's best checkpoint weakens that explanation;
an improved LR point would instead support exploring that local LR direction.
The second history makes generalization loss after the early peak plausible.
If stronger dropout lowers training fit without improving validation, that weakens
this proposed remedy; if lower LR preserves or improves validation later in
training, optimizer behavior deserves further attention. A shorter horizon can
also test whether comparable quality costs fewer epochs, but cannot recover an
accuracy gain beyond a checkpoint already included in selection merely by ending
earlier. These are alternatives to justify, not a required pair of candidates.

Logged curves may be sparse or sampled. Three points do not establish what
happened between them, and the final logged epoch alone does not establish the
actual training end. If the completion evidence above is absent, leave that
uncertain. A short log is not an early-stopping diagnosis; use an explicit recorded
cause when available. When a joint LR/dropout/adaptation change helps, attribute
the observation to that joint strategy until a comparison separates its effects.

For adaptation specifically, suppose matched full-adaptation runs repeatedly show
the earlier-peak pattern while a supported head-only strategy is already in the
frozen domain. Testing head-only with other settings fixed is a defensible
alternative to another scalar adjustment. If it worsens both training fit and
validation, that weakens restricting adaptation as the remedy. This comparison
requires the complete block choices described below; the trajectory itself does
not authorize adding a new strategy to the domain.

## A worse batch and an uncertain gain

Suppose the next batch returns 0.735 and 0.733 while historical C remains at 0.741.
Keep C as the incumbent. The new points weaken the explanations they tested; they
do not justify recentering the search on 0.735. One possible next batch explores
an untested authorized direction near C, while another tests a distinct strategy.
If both again regress, reconsider those explanations rather than treating each
latest batch as progress.

For a separate synthetic case, suppose a candidate exceeds C by only 0.001, and
available matched repeats vary by about 0.006. That variation makes confirmation
more valuable than the same gap would be with consistent repeat evidence. A
repeat comparison is useful only if the frozen contract already represents the
intended repeat. Duplicate points within one submitted batch are rejected; an
earlier run can be repeated in a later batch for a stated question. Reusing its
fixed seed does not measure independent-seed variability, and changing an
unrelated setting does not create a repeat. If the intended repeats cannot be expressed, state that limitation
and use an informative allowed comparison without claiming to measure noise.
If later matched results consistently favor the candidate, exploiting its
neighborhood gains support; if the ordering reverses, confidence in the small
gain falls. With no repeat evidence at all, do not invent the 0.006 noise scale:
continued exploration can still be reasonable, but the gain remains uncertain.

## Choose complete configuration blocks

For a separate example, suppose the authorized task has a pretrained backbone,
and its variant config supports these adaptation and LayerMix choices. Declare
whole values in `search.parameters` before initialization:

```yaml
yaml:/finetune/tuning:
  - {preset: full}
  - {preset: head_only}
yaml:/finetune/layer_mix:
  - {enabled: false, layer_indices: null, shared_across_modalities: false}
  - {enabled: true, layer_indices: [1, 2], shared_across_modalities: false}
```

The initial points can select the full-adaptation and disabled-LayerMix choices,
leaving other blocks available later. A later complete point can select the
enabled LayerMix block without also changing adaptation, making that comparison
easier to interpret. Replacing `finetune.tuning` replaces source `groups` overrides
as well; changing only its `preset` would retain those overrides. Every point must
still include all searched keys, including any runtime axes.

Do not add a second axis inside a searched block, such as
`yaml:/finetune/layer_mix/enabled`: overlapping paths are rejected because they
would modify the whole frozen choice. A new layer list or adaptation block that
was not declared is a domain expansion, even if its resulting config is valid.
