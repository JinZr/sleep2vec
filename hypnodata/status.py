from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from agent_tools.models import json_ready
from agent_tools.progress import progress_path


def write_hypnodata_progress(
    run_dir: str | Path,
    *,
    status: str,
    total_records: int,
    processed_records: int,
    succeeded_records: int,
    failed_records: int,
    skipped_records: int,
    started_at: float,
    current_record_id: str | None = None,
    message: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "status": status,
        "task": "hypnodata",
        "total_records": int(total_records),
        "processed_records": int(processed_records),
        "succeeded_records": int(succeeded_records),
        "failed_records": int(failed_records),
        "skipped_records": int(skipped_records),
        "current_record_id": current_record_id,
        "started_at": _format_time(started_at),
        "updated_at": _format_time(time.time()),
        "message": message,
        "processed": int(processed_records + skipped_records),
        "total": int(total_records),
        "success": int(succeeded_records + skipped_records),
        "failed": int(failed_records),
    }
    path = progress_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def _format_time(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))
