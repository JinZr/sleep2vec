#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE="${S2V_EGS_REMOTE:-https://huggingface.co/datasets/zrjin/s2v_egs}"
HUB_DIR="${S2V_EGS_HUB_DIR:-${REPO_ROOT}/egs/data_hub}"
DEST_DIR="${S2V_EGS_DEST_DIR:-${REPO_ROOT}/egs/data/npz}"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required." >&2
  exit 1
fi

if [ -d "${HUB_DIR}/.git" ]; then
  echo "Updating ${HUB_DIR}"
  git -C "${HUB_DIR}" pull --ff-only
else
  echo "Cloning ${REMOTE} -> ${HUB_DIR}"
  git clone "${REMOTE}" "${HUB_DIR}"
fi

mkdir -p "${DEST_DIR}"

found=0
while IFS= read -r -d '' src; do
  found=1
  name="$(basename "${src}")"
  ln -sfn "${src}" "${DEST_DIR}/${name}"
  echo "Linked ${DEST_DIR}/${name} -> ${src}"
done < <(find "${HUB_DIR}" -type f -name '*.npz' -print0 | sort -z)

if [ "${found}" -eq 0 ]; then
  echo "ERROR: no .npz files found under ${HUB_DIR}" >&2
  exit 1
fi

echo "Done. Example data links are ready under ${DEST_DIR}"
