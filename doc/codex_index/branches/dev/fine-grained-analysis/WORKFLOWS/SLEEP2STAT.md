# sleep2stat Workflow

## Editing Output Metrics

1. Identify whether the field is analyzer-produced or reducer-produced.
2. Reuse `StageSourceResolver` for stage denominators instead of creating local masks.
3. Encode units and denominators in public night-stat field names.
4. Keep deprecated aliases only for high-risk existing fields and only when their semantics remain clear.
5. Update `sleep2stat/plot.py` to prefer new fields and fallback to old result bundles.
6. Add or update focused tests in `tests/test_sleep2stat_analyzers.py`, `tests/test_sleep2stat_reducers.py`,
   or `tests/test_sleep2stat_cli.py`.

## Verification

Use the `exp` environment on this machine:

```bash
conda run -n exp python -m pytest -q tests/test_sleep2stat_reducers.py tests/test_sleep2stat_analyzers.py tests/test_sleep2stat_cli.py
conda run -n exp python utils/check_configs.py configs/sleep2stat/*.yaml
git diff --check
```
