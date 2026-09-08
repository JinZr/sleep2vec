# Choosing an initial search domain

Use this example before authoring a new adaptive domain. It illustrates reasons
to preserve useful choices within a small budget; the numbers are observations,
not reusable parameter recommendations. Existing frozen searches stay unchanged.

## An incumbent at the observed LR and horizon edges

Suppose a completed LR 3e-4, eight-epoch, warmup-500 run wins at epoch seven.
A slower LR 3e-5, twelve-epoch run and two regularization variants win at epoch
eleven. The larger LR run is the strongest observed joint configuration. Neither
its LR effect nor a continuing upward trajectory has been isolated.

Fixing eight epochs and ending the LR domain at 3e-4 leaves no room to investigate
two visible boundary hypotheses. That can still be a deliberate budget choice,
but it needs a reason. One defensible allocation keeps more technical axes fixed
and gives the domain a bounded horizon alternative or an LR value beyond the
observed range. Another spends the opening pair separating horizon at a common
LR, or LR at a common horizon. Explain which uncertainty is being prioritized;
changing LR, horizon and warmup together supports a joint comparison.

The frozen domain is permission for later choices, not a promise to survey every
axis or Cartesian combination. With twelve fits, including head width, depth,
dropout and three LoRA knobs carries an opportunity cost even if each choice is
valid. Fix settings that lack a useful hypothesis for this budget, with reasons;
include a coupled block only when its complete choices are understood. A seed or
repeat axis cannot be invented later to explain a small score difference.

Reducing four GPUs to two while retaining a user-authorized per-device batch can
change effective batch unless accumulation changes too. Preserve the intended
comparison and account for optimizer-update-dependent warmup and horizon-dependent
scheduling. Old RAM requests and GPU count alone do not measure epoch cost.

## State the decision in ordinary prose

A useful initialization explanation identifies the evidence behind the opening
points, why the bounds were chosen, why other axes are fixed, and what the next
decision would be if the first comparison improves at a boundary or deteriorates.
An intentionally excluded possibility is an acknowledged limitation, not something
to recover through an unauthorized domain change. No new report schema or approval
for individual technical values is needed.

These are concept choices. Publication still resolves effective configuration
and consultation; launch still needs its existing authority. Judge initialization
by the evidence and options available then, not by whether later results happened
to favor a parameter value outside the original range.
