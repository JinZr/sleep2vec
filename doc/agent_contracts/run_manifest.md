# Run Manifest Contract

Training and preset sidecar manifests are JSON files with `schema_version: 1`, `kind`, timestamps, input paths, output paths, and relevant metrics or counts.

Finetune run manifests live at `log-finetune/<version>/run_manifest.json`.
Preset sidecar manifests live next to generated preset pickle files as `<preset>.manifest.json`.
