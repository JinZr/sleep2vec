from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AnalyzerResult:
    name: str
    record_id: str
    epoch: pd.DataFrame | None = None
    second: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    night: dict[str, Any] | None = None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FailureRecord:
    record_id: str
    analyzer: str
    error_type: str
    message: str
