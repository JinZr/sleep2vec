# System Overview

This branch adds the `sleep2stat` sidecar analysis toolkit around existing sleep model outputs and
PSG-like signal arrays. It does not replace pretrain, finetune, infer, dataset, or variant runtime
contracts from the main index.

## sleep2stat Flow

1. `sleep2stat` loads `SleepRecord` rows from NPZ or Kaldi manifests.
2. Analyzers emit zero or more alignment tables: epoch, second, event, and night-level rows.
3. Reducers consume analyzer results and add derived night-level statistics.
4. Writers merge per-record and global result tables without mutating source NPZ/index data.
5. Plotting reads result tables and chooses recognized field names with legacy fallback where needed.

## Contract Boundaries

- Stage labels use the integer convention `0=W`, `1=N1`, `2=N2`, `3=N3`, `4=REM`; unscored/artifact
  values are outside the scored set.
- Clinical-facing field names must encode units or denominator when ambiguity is likely.
- Model-derived event rates must not be silently labeled as clinical AHI unless a sleep-time denominator
  is available.
- YASA event density follows YASA's source contract: stage-grouped density is event count divided by
  minutes in that stage.
