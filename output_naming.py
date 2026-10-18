from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def _sanitize_component(value) -> str:
    text = str(value)
    text = text.strip()
    text = text.replace(" ", "-")
    text = text.replace(",", "-")
    text = text.replace("/", "-")
    text = text.replace("=", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_.")
    return text or "na"


def build_output_path(output_csv: str | None, default_stem: str, tags: dict | None = None, output_dir: str = "outputs") -> Path:
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    parts = [default_stem]
    if tags:
        for key, value in tags.items():
            if value is None or value == "":
                continue
            parts.append(f"{_sanitize_component(key)}-{_sanitize_component(value)}")
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    filename = "__".join(parts) + ".csv"
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_related_output_path(base_path: Path, stem_suffix: str) -> Path:
    suffix = base_path.suffix or ".csv"
    related = base_path.with_name(f"{base_path.stem}__{_sanitize_component(stem_suffix)}{suffix}")
    related.parent.mkdir(parents=True, exist_ok=True)
    return related
