from __future__ import annotations

from dataclasses import dataclass

from sleep2stat.config import Sleep2statConfig


@dataclass
class Sleep2statContext:
    config: Sleep2statConfig
    device: str
    num_workers: int
    batch_size: int | None = None
