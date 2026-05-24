import numpy as np
import torch

from sleep2vec.metrics import _evaluate_single_ahi_record, _merge_ahi_window_records


def prediction_export_enabled(args) -> bool:
    return getattr(args, "inference_prediction_csv_path", None) not in (None, "")


def extract_prediction_records(args, batch, logits, targets) -> list[dict[str, object]]:
    labels = targets.detach().cpu()
    logits = logits.detach().cpu()
    paths = list(batch["metadata"]["path"])
    token_starts = batch.get("token_start")
    if token_starts is None:
        starts = [0 for _ in paths]
    else:
        starts = [int(value) for value in token_starts.detach().cpu().tolist()]

    if getattr(args, "is_multilabel", False):
        return _extract_multilabel_prediction_records(paths, starts, labels, logits)
    if args.is_classification:
        return _extract_classification_prediction_records(paths, starts, labels, logits)
    return _extract_regression_prediction_records(args, paths, starts, labels, logits)


def _extract_multilabel_prediction_records(paths, starts, labels, logits) -> list[dict[str, object]]:
    probs = torch.sigmoid(logits).to(torch.float32)
    records: list[dict[str, object]] = []
    for idx, path in enumerate(paths):
        sample_labels = labels[idx]
        sample_probs = probs[idx]
        mask = sample_labels != -1.0
        if not mask.any():
            continue
        valid_probs = sample_probs[mask].numpy()
        valid_labels = sample_labels[mask].to(torch.int64).numpy()
        records.append(
            {
                "path": str(path),
                "token_start": starts[idx],
                "kind": "multilabel",
                "groundtruth": valid_labels.tolist(),
                "prob": valid_probs.tolist(),
                "prediction": (valid_probs >= 0.5).astype(np.int64).tolist(),
                "is_sequence": True,
            }
        )
    return records


def _extract_classification_prediction_records(paths, starts, labels, logits) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if logits.dim() == 3:
        probs = torch.softmax(logits, dim=-1).to(torch.float32)
        for idx, path in enumerate(paths):
            sample_labels = labels[idx].reshape(-1)
            sample_probs = probs[idx].reshape(-1, probs.size(-1))
            mask = sample_labels != -1
            if not mask.any():
                continue
            valid_probs = sample_probs[mask].numpy()
            valid_labels = sample_labels[mask].to(torch.int64).numpy()
            records.append(
                {
                    "path": str(path),
                    "token_start": starts[idx],
                    "kind": "classification",
                    "groundtruth": valid_labels.tolist(),
                    "probabilities": valid_probs.tolist(),
                    "prediction": valid_probs.argmax(axis=-1).astype(np.int64).tolist(),
                    "is_sequence": True,
                }
            )
        return records

    probs = torch.softmax(logits, dim=-1).to(torch.float32)
    flat_labels = labels.reshape(-1)
    for idx, path in enumerate(paths):
        if idx >= probs.size(0) or flat_labels[idx].item() == -1:
            continue
        prob = probs[idx].numpy()
        records.append(
            {
                "path": str(path),
                "token_start": starts[idx],
                "kind": "classification",
                "groundtruth": int(flat_labels[idx].item()),
                "probabilities": prob.tolist(),
                "prediction": int(prob.argmax()),
                "is_sequence": False,
            }
        )
    return records


def _extract_regression_prediction_records(args, paths, starts, labels, logits) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if getattr(args, "is_seq", False):
        preds = logits.to(torch.float32)
        if preds.dim() == labels.dim() + 1 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)
        for idx, path in enumerate(paths):
            sample_labels = labels[idx].reshape(-1).float()
            sample_preds = preds[idx].reshape(-1)
            mask = sample_labels != -1.0
            if not mask.any():
                continue
            valid_preds = sample_preds[mask].numpy()
            valid_labels = sample_labels[mask].numpy()
            records.append(
                {
                    "path": str(path),
                    "token_start": starts[idx],
                    "kind": "regression",
                    "groundtruth": valid_labels.tolist(),
                    "prediction": valid_preds.tolist(),
                    "is_sequence": True,
                }
            )
        return records

    preds = logits.reshape(-1).to(torch.float32)
    flat_labels = labels.reshape(-1).float()
    for idx, path in enumerate(paths):
        if idx >= preds.numel() or flat_labels[idx].item() == -1.0:
            continue
        records.append(
            {
                "path": str(path),
                "token_start": starts[idx],
                "kind": "regression",
                "groundtruth": float(flat_labels[idx].item()),
                "prediction": float(preds[idx].item()),
                "is_sequence": False,
            }
        )
    return records


def build_prediction_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = _group_prediction_records(records)
    rows: list[dict[str, object]] = []
    for path, items in grouped.items():
        if not items:
            continue
        kind = items[0].get("kind")
        if kind == "classification":
            rows.append(_build_classification_prediction_row(path, items))
        elif kind == "regression":
            rows.append(_build_regression_prediction_row(path, items))
        elif kind == "multilabel":
            rows.append(_build_multilabel_prediction_row(path, items))
    return rows


def _group_prediction_records(records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["path"]), []).append(record)
    for items in grouped.values():
        items.sort(key=lambda item: int(item.get("token_start", 0)))
    return grouped


def _token_starts_for_prediction_items(items: list[dict[str, object]]) -> list[int]:
    return [int(item.get("token_start", 0)) for item in items]


def _build_classification_prediction_row(path: str, items: list[dict[str, object]]) -> dict[str, object]:
    token_starts = _token_starts_for_prediction_items(items)
    if any(bool(item.get("is_sequence")) for item in items):
        probs = np.concatenate([np.asarray(item["probabilities"], dtype=np.float32) for item in items], axis=0)
        groundtruth = np.concatenate([np.asarray(item["groundtruth"], dtype=np.int64).reshape(-1) for item in items])
        prediction = probs.argmax(axis=-1).astype(np.int64)
        row: dict[str, object] = {
            "path": path,
            "groundtruth": groundtruth.tolist(),
            "prediction": prediction.tolist(),
            "n_predictions": int(groundtruth.size),
            "n_windows": len(items),
            "token_starts": token_starts,
        }
        for class_idx in range(probs.shape[1]):
            row[f"prob_{class_idx}"] = probs[:, class_idx].tolist()
        return row

    probs = np.asarray([item["probabilities"] for item in items], dtype=np.float32).mean(axis=0)
    row = {
        "path": path,
        "groundtruth": int(items[0]["groundtruth"]),
        "prediction": int(probs.argmax()),
        "n_predictions": len(items),
        "n_windows": len(items),
        "token_starts": token_starts,
    }
    for class_idx, value in enumerate(probs.tolist()):
        row[f"prob_{class_idx}"] = float(value)
    return row


def _build_regression_prediction_row(path: str, items: list[dict[str, object]]) -> dict[str, object]:
    token_starts = _token_starts_for_prediction_items(items)
    if any(bool(item.get("is_sequence")) for item in items):
        groundtruth = np.concatenate([np.asarray(item["groundtruth"], dtype=np.float32).reshape(-1) for item in items])
        prediction = np.concatenate([np.asarray(item["prediction"], dtype=np.float32).reshape(-1) for item in items])
        return {
            "path": path,
            "groundtruth": groundtruth.tolist(),
            "prediction": prediction.tolist(),
            "n_predictions": int(groundtruth.size),
            "n_windows": len(items),
            "token_starts": token_starts,
        }

    groundtruth = np.asarray([item["groundtruth"] for item in items], dtype=np.float32)
    prediction = np.asarray([item["prediction"] for item in items], dtype=np.float32)
    return {
        "path": path,
        "groundtruth": float(groundtruth.mean()),
        "prediction": float(prediction.mean()),
        "n_predictions": len(items),
        "n_windows": len(items),
        "token_starts": token_starts,
    }


def _build_multilabel_prediction_row(path: str, items: list[dict[str, object]]) -> dict[str, object]:
    token_starts = _token_starts_for_prediction_items(items)
    groundtruth = np.concatenate([np.asarray(item["groundtruth"], dtype=np.int64).reshape(-1) for item in items])
    prob = np.concatenate([np.asarray(item["prob"], dtype=np.float32).reshape(-1) for item in items])
    prediction = (prob >= 0.5).astype(np.int64)
    return {
        "path": path,
        "groundtruth": groundtruth.tolist(),
        "prediction": prediction.tolist(),
        "prob": prob.tolist(),
        "n_predictions": int(groundtruth.size),
        "n_windows": len(items),
        "token_starts": token_starts,
    }


def build_ahi_prediction_rows(records: list[dict[str, np.ndarray]], threshold: float) -> list[dict[str, object]]:
    token_starts_by_path: dict[str, list[int]] = {}
    for record in records:
        token_starts_by_path.setdefault(str(record["path"]), []).append(int(record.get("token_start", 0)))

    rows: list[dict[str, object]] = []
    for record in _merge_ahi_window_records(records):
        path = str(record["path"])
        truth = np.asarray(record["truth"], dtype=np.int64).reshape(-1)
        score = np.asarray(record["score"], dtype=np.float32).reshape(-1)
        prediction = (score > float(threshold)).astype(np.int64)
        _, pred_ahi, _ = _evaluate_single_ahi_record(record, threshold=float(threshold))
        token_starts = sorted(token_starts_by_path.get(path, []))
        rows.append(
            {
                "path": path,
                "groundtruth": truth.tolist(),
                "prediction": prediction.tolist(),
                "prob": score.tolist(),
                "n_predictions": int(truth.size),
                "n_windows": len(token_starts),
                "token_starts": token_starts,
                "ahi_threshold": float(threshold),
                "true_ahi": float(record["true_ahi"]),
                "pred_ahi": float(pred_ahi) if pred_ahi is not None else None,
                "tst_hours": float(record["tst_hours"]),
            }
        )
    return rows
