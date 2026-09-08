# Worked result-to-proposal example

This is invented teaching evidence, not a runnable recipe or an experiment result.
The scientific task, validation objective, data, checkpoint source and execution
identity are already authorized. The initial domain includes learning rate
`[3e-5, 6e-4]`, epochs `[8, 16]` and head dropout `[0.1, 0.3]`; each round permits
two runs and eight total runs remain. Other settings are fixed with a recorded
reason. These values illustrate the reasoning, not recommended defaults.

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
