# Delta From Main

This branch contains a `sleep2stat` analysis surface that is not covered by the current `main` branch
index. Main-branch guidance still applies to unchanged model, data, preprocessing, variant, and runtime
modules.

## Branch-Specific Additions

- `sleep2stat/`: sidecar analyzer, reducer, writer, config, registry, and plotting package.
- `configs/sleep2stat/`: example recipes for model-first, YASA, SpO2, respiratory, and microstructure
  analysis.
- `tests/test_sleep2stat_*.py`: contract tests for analyzers, reducers, config validation, writers, and
  plotting.

## Branch-Specific Risks

- Night-level result rows are merged by record id; duplicate field names from different reducers can
  silently overwrite earlier values.
- Clinical-facing output fields need explicit units and denominators because result tables are consumed
  directly by reports and plots.
