# Choosing an initial search domain

This invented concept-only example is separate from the historical UKB review
case. Its scores and curves are synthetic, not experiment evidence or reusable
parameter recommendations. Existing frozen searches stay unchanged.

## A strong interior point and a weaker boundary point

Suppose the user authorizes eight new fits, two concurrent runs, and selection
on validation AUROC. A prior comparison used the same model, data, effective
batch, schedule family, 24-epoch cap and 1,200 warmup updates throughout:

| Prior point | LR | Best validation AUROC | Best checkpoint epoch, zero-based | Supplied curve evidence |
| --- | ---: | ---: | ---: | --- |
| A | 2e-5 | 0.731 | 23 | Training loss falls; validation AUROC improves over the last four saved checkpoints. |
| B | 8e-5 | 0.744 | 15 | Training loss continues falling while validation AUROC declines after epoch 16. |
| C | 2e-4 | 0.724 | 6 | Validation AUROC declines after its early peak. |

B is the strongest observed point and is interior to the observed LR range.
A supplies a duration hypothesis, but its final-checkpoint placement does not
make it the incumbent. C supplies no direct reason to extend the high-LR edge.
There are no repeats or wall-clock measurements, so neither noise magnitude nor
a compute-optimal horizon is established.

## Allocate the domain to competing explanations

One defensible allocation preserves an LR comparison near B while leaving a
bounded longer-horizon option for A. Another uses the opening pair to compare
regularization around B because its supplied training/validation divergence
makes that question useful. Explain which question gets the first two fits and
why the other deserves later room, or is deliberately excluded from eight fits.
The small difference between A and B does not itself establish significance.

A wider LR range would need a hypothesis that addresses C's observed behavior;
a last-checkpoint winner in a weaker run is not sufficient justification. Equally,
fixing every horizon at 24 would exclude testing A's continued improvement. Both
choices have opportunity costs. State a practical upper horizon as a provisional
compute judgment when throughput is unavailable, and include checkpoint evaluation
cost. A fit-count or concurrency cap is not a measured time budget.

Changing the horizon may also change a horizon-dependent decay schedule. A new
regularization axis needs complete valid values and an explicit reason to hold
head capacity and adaptation settings fixed. Do not add many unrelated axes just
because the controller can represent them, or invent a seed axis after freezing.

The useful explanation connects each opening point to evidence, gives reasons
for both searched bounds and fixed settings, and describes which later results
would change the allocation. It does not need a new report schema or approval for
individual technical values. Publication still resolves effective configuration
and consultation; launch still needs its existing authority. Judge the domain
using the evidence available when it was chosen, not later outcomes.
