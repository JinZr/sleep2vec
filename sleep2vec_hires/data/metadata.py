import math
import typing as t

import numpy as np
import torch

from sleep2vec_hires.data.samplers import WeightedRandomDistributedSampler


def _equal_matrix_from_ids(vals: t.Sequence[str]) -> torch.Tensor:
    """
    Map string/any-hashable values to integer IDs and return pairwise equality matrix.
    Shape [N, N], float32 in {0,1}.
    """
    mapping = {}
    ids = []
    for v in vals:
        v = str(v)
        if v not in mapping:
            mapping[v] = len(mapping)
        ids.append(mapping[v])
    x = torch.tensor(ids, dtype=torch.long)
    return (x[:, None] == x[None, :]).to(torch.float32)


def build_w_h_age_sex_center(
    age: torch.Tensor,
    sex: torch.Tensor,
    center: t.Sequence[str],
    path: t.Sequence[str],
    *,
    sigma_age: float = 20.0,
    alpha_sex: float = 0.8,
    gamma_same: float = 1.3,
    gamma_diff: float = 0.8,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build two [N,N] matrices:
      - w: negative sample weights (row-normalized on valid negatives)
      - h: same-path off-diagonal mask (1 for same path, else 0)
    """
    N = age.shape[0]
    age = age.detach().cpu().float()
    sex = sex.detach().cpu().long()

    ai, aj = age[:, None], age[None, :]
    valid_age = (ai >= 0) & (aj >= 0)
    d = (ai - aj).abs()
    age_sim = torch.zeros(N, N, dtype=torch.float32)
    if valid_age.any():
        age_sim = torch.exp(-d / float(sigma_age)) * valid_age.float()

    si, sj = sex[:, None], sex[None, :]
    valid_sex = (si >= 0) & (sj >= 0)
    same_sex = (si == sj) & valid_sex
    diff_sex = (si != sj) & valid_sex
    sex_coef = same_sex.float() + float(alpha_sex) * diff_sex.float()

    k = (age_sim * sex_coef).clamp(0.0, 1.0)

    same_center = _equal_matrix_from_ids(center)
    w_gate = same_center * float(gamma_same) + (1.0 - same_center) * float(gamma_diff)

    same_path = _equal_matrix_from_ids(path)
    offdiag = torch.ones(N, N, dtype=torch.float32) - torch.eye(N, dtype=torch.float32)
    same_path_off = same_path * offdiag

    w_raw = k * w_gate + eps
    w_raw.fill_diagonal_(1.0)

    w = w_raw.clone()
    w[same_path_off.bool()] = eps

    valid_neg_mask = (offdiag - same_path_off).clamp_min(0.0)
    with torch.no_grad():
        num_valid = valid_neg_mask.sum(1, keepdim=True)
        need_fallback = num_valid.squeeze(1) == 0
        if need_fallback.any():
            rows = need_fallback.nonzero(as_tuple=True)[0]
            w[rows] = w_raw[rows]
            valid_neg_mask[rows] = offdiag[rows]
            num_valid[rows] = offdiag[rows].sum(1, keepdim=True)

        denom = (w * valid_neg_mask).sum(1, keepdim=True) / (num_valid + eps)
        w = w / denom.clamp_min(eps)

    h = same_path_off.clone().to(torch.float32)
    h.fill_diagonal_(0.0)
    return w, h


def safe_cast(v, default=-1):
    try:
        if isinstance(v, str) and v.lower() == "nan":
            return default
        if v is None:
            return default
        if isinstance(v, float) and math.isnan(v):
            return default
        return int(float(v))
    except Exception:
        return default


def safe_cast_float(v, default=-1.0):
    try:
        if isinstance(v, str) and v.lower() == "nan":
            return default
        if v is None:
            return default
        if isinstance(v, float) and math.isnan(v):
            return default
        f = float(v)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default


def _encode_binary_label(v):
    """Normalize various representations of binary labels to {0,1,-1}."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"male", "1", "1.0", "x"}:
            return 1
        if s in {"female", "0", "0.0"}:
            return 0
        return -1

    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return v if v in (0, 1) else -1

    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return -1
        return int(v) if v in (0.0, 1.0) else -1

    return -1


def process_metadata(samples, disease_names, regression_names: t.Sequence[str] | None = None):
    regression_names = set(regression_names or [])
    batch_metadata = {
        "age": [],
        "sex": [],
        "source": [],
        "path": [],
    }
    for disease_name in disease_names:
        batch_metadata[disease_name] = []

    for s in samples:
        meta = s.metadata
        batch_metadata["age"].append(meta.get("age", "nan"))
        batch_metadata["sex"].append(meta.get("sex", "nan"))
        batch_metadata["source"].append(meta.get("source", "nan"))
        batch_metadata["path"].append(meta.get("path", "nan"))
        for disease_name in disease_names:
            batch_metadata[disease_name].append(meta.get(disease_name, "nan"))

    processed = {}
    processed["age"] = torch.tensor([safe_cast(v, -1) for v in batch_metadata["age"]], dtype=torch.float)
    processed["sex"] = torch.tensor([_encode_binary_label(v) for v in batch_metadata["sex"]], dtype=torch.long)
    for disease_name in disease_names:
        values = batch_metadata[disease_name]
        if disease_name in regression_names:
            processed[disease_name] = torch.tensor(
                [safe_cast_float(v, -1.0) for v in values],
                dtype=torch.float,
            )
        else:
            processed[disease_name] = torch.tensor(
                [_encode_binary_label(v) for v in values],
                dtype=torch.long,
            )

    processed["source"] = [v for v in batch_metadata["source"]]
    processed["path"] = [v for v in batch_metadata["path"]]
    return processed


def extract_binary_labels(dataset, target_name: str):
    labels = np.fromiter(
        (
            (
                _encode_binary_label(s.metadata[target_name])
                if (hasattr(s, "metadata") and (target_name in s.metadata))
                else -1
            )
            for s in dataset.data
        ),
        dtype=np.int64,
    )
    return labels


def make_weighted_sampler_from_labels(labels: np.ndarray, epoch_size: int | None = None, *, seed: int = 0):
    valid = labels != -1
    if not valid.any():
        return None

    uniq, counts = np.unique(labels[valid], return_counts=True)
    class_weight = {int(c): 1.0 / float(n) for c, n in zip(uniq.tolist(), counts.tolist())}

    w = np.zeros_like(labels, dtype=np.float32)
    for y in (0, 1):
        if y in class_weight:
            w[labels == y] = class_weight[y]

    num_samples = int(valid.sum()) if epoch_size is None else int(epoch_size)
    sampler = WeightedRandomDistributedSampler(
        weights=torch.as_tensor(w, dtype=torch.float32),
        num_samples=num_samples,
        seed=seed,
    )
    return sampler
