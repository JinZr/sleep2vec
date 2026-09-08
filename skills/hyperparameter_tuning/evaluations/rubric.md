# Reviewing proposal reasoning

Use `cases.json` when changing the result-to-proposal guidance or the evidence
presented to an agent. These are invented concept-planning cases. They do not
authorize training, model evaluation, recipe publication or a W&B refresh.

Give an independent agent the cases and the version of the tuning guidance being
assessed. Ask for up to two complete configurations per case, evidence references,
an explanation and an outcome that would change the interpretation. Do not give
it this rubric or a preferred answer. For a before/after comparison, keep the
cases, model/settings, available evidence and task wording the same; record the
guidance revision and retain both outputs. An agent may read the supplied files;
no executable test runner or experiment is needed.

Assess each dimension as **supported**, **mixed** or **unsupported**, citing the
actual output. There is no reference parameter vector and no string-matching
score. A different defensible allocation can be equally good.

| Dimension | What to inspect |
| --- | --- |
| Evidence sensitivity | The rising and declining cases have the same best score, best checkpoint and configured horizon. Does the explanation respond to the different logged trajectories? Longer training is an unresolved possibility in the first; the second supplies evidence of deterioration after the best observed point. Candidates may overlap if the tradeoff is explained. |
| Honest uncertainty | Does missing history remain missing? Does the agent avoid turning `_step=900` into trainer steps, a final logged point into the actual training endpoint, or an empty stop reason into early stopping? LR extrema do not establish schedule order. |
| Learning across rounds | Does the agent retain `run-000` in the historical-incumbent case and revise the prior higher-LR/longer-horizon explanations? A latest-round winner is not the workflow winner. |
| Joint effects and boundaries | In the small-gain case, does the agent keep the LR/dropout explanation joint, avoid claiming significance without repeats, and stay within the frozen bounds? It may propose a separating comparison or another justified allocation. |
| Useful next decision | Are complete candidate points within the domain and remaining budget, linked to cited evidence and a meaningful possible revision? It must not invent an unfrozen seed axis. Repeats require a stated question and a permitted form; repeating the same seed alone does not estimate independent-seed variability. |

Report disagreements and unchanged behavior as well as improvements. A handful
of synthetic cases can expose a reasoning failure or show that a workflow is
usable; it cannot establish better real-world tuning performance, statistical
significance, or general superiority of one skill revision. If both versions
already reason well, say so. Code CI checks evidence transport and invariants;
these case reviews assess the resulting explanations and decisions separately.

For the separate decision of choosing a new domain before initialization, use
the [historical initialization cases and review](initialization/review.md). Their
evidence cuts withhold the original search choices and all subsequent outcomes.
