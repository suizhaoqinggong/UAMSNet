from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class OutputSettings:
    root_dir: Path
    backup_code: bool


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    experiment_config_path: Path
    model_config_path: Path
    output: OutputSettings


@dataclass(frozen=True, slots=True)
class RunLayout:
    root_dir: Path
    experiment_id: str
    experiment_hash: str
    experiment_dir: Path
    model_id: str
    model_hash: str
    model_dir: Path
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path
    plots_dir: Path
    confusion_matrix_dir: Path
    logs_dir: Path


@dataclass(frozen=True, slots=True)
class CatalogRecord:
    item_id: str
    item_hash: str
    source_path: str
    source_stem: str
    created_at_utc: str


def resolve_run_layout(config: LayoutConfig) -> RunLayout:
    root_dir = config.output.root_dir
    root_dir.mkdir(parents=True, exist_ok=True)

    experiment_path = config.experiment_config_path.resolve()
    model_path = config.model_config_path.resolve()
    experiment_mapping = load_toml_mapping(experiment_path)
    model_mapping = load_toml_mapping(model_path)
    experiment_name = required_nested_string(
        mapping=experiment_mapping,
        table="experiment",
        key="name",
        path=experiment_path,
    )
    model_name = required_nested_string(
        mapping=model_mapping,
        table="model",
        key="name",
        path=model_path,
    )

    experiment_hash = semantic_hash(experiment_mapping)
    model_hash = semantic_hash(model_mapping)

    experiments_catalog_path = root_dir / "experiments.tsv"
    experiment_records = read_catalog(
        path=experiments_catalog_path,
        id_column="experiment_id",
        hash_column="experiment_hash",
        path_column="config_path",
        stem_column="config_stem",
    )
    experiment_record = find_record(
        records=experiment_records,
        source_path=str(experiment_path),
        item_hash=experiment_hash,
    )
    if experiment_record is None:
        experiment_id = build_unique_item_id(
            stem=experiment_name,
            item_hash=experiment_hash,
            existing_ids={record.item_id for record in experiment_records},
        )
        experiment_record = CatalogRecord(
            item_id=experiment_id,
            item_hash=experiment_hash,
            source_path=str(experiment_path),
            source_stem=experiment_path.stem,
            created_at_utc=utc_now(),
        )
        append_catalog_row(
            path=experiments_catalog_path,
            record=experiment_record,
            id_column="experiment_id",
            hash_column="experiment_hash",
            path_column="config_path",
            stem_column="config_stem",
        )
    experiment_dir = root_dir / experiment_record.item_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    write_text_if_missing(
        experiment_dir / "experiment.snapshot.toml",
        read_text(experiment_path),
    )
    write_json_if_missing(
        experiment_dir / "experiment.meta.json",
        {
            "experiment_id": experiment_record.item_id,
            "experiment_hash": experiment_record.item_hash,
            "experiment_name": experiment_name,
            "config_path": experiment_record.source_path,
            "config_stem": experiment_record.source_stem,
            "created_at_utc": experiment_record.created_at_utc,
        },
    )

    if config.output.backup_code:
        backup_directory_once(
            source=project_root() / "src" / "framework",
            target=experiment_dir / "code" / "framework",
        )
        backup_directory_once(
            source=project_root() / "src" / "data_adapters",
            target=experiment_dir / "code" / "data_adapters",
        )
        backup_directory_once(
            source=project_root() / "src" / "tasks",
            target=experiment_dir / "code" / "tasks",
        )

    models_catalog_path = experiment_dir / "models.tsv"
    model_records = read_catalog(
        path=models_catalog_path,
        id_column="model_id",
        hash_column="model_hash",
        path_column="model_path",
        stem_column="model_stem",
    )
    model_record = find_record(
        records=model_records,
        source_path=str(model_path),
        item_hash=model_hash,
    )
    if model_record is None:
        model_id = build_unique_item_id(
            stem=model_name,
            item_hash=model_hash,
            existing_ids={record.item_id for record in model_records},
        )
        model_record = CatalogRecord(
            item_id=model_id,
            item_hash=model_hash,
            source_path=str(model_path),
            source_stem=model_path.stem,
            created_at_utc=utc_now(),
        )
        append_catalog_row(
            path=models_catalog_path,
            record=model_record,
            id_column="model_id",
            hash_column="model_hash",
            path_column="model_path",
            stem_column="model_stem",
        )
    model_dir = experiment_dir / model_record.item_id
    model_dir.mkdir(parents=True, exist_ok=True)

    write_text_if_missing(
        model_dir / "model.snapshot.toml",
        read_text(model_path),
    )
    write_json_if_missing(
        model_dir / "model.meta.json",
        {
            "model_id": model_record.item_id,
            "model_hash": model_record.item_hash,
            "model_name": model_name,
            "model_path": model_record.source_path,
            "model_stem": model_record.source_stem,
            "created_at_utc": model_record.created_at_utc,
        },
    )

    if config.output.backup_code:
        backup_directory_once(
            source=project_root() / "src" / "models",
            target=model_dir / "code" / "models",
        )

    run_id = build_run_id()
    run_dir = model_dir / "runs" / run_id
    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    plots_dir = run_dir / "plots"
    confusion_matrix_dir = plots_dir / "confusion_matrix"
    logs_dir = run_dir / "logs"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return RunLayout(
        root_dir=root_dir,
        experiment_id=experiment_record.item_id,
        experiment_hash=experiment_record.item_hash,
        experiment_dir=experiment_dir,
        model_id=model_record.item_id,
        model_hash=model_record.item_hash,
        model_dir=model_dir,
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        tensorboard_dir=tensorboard_dir,
        plots_dir=plots_dir,
        confusion_matrix_dir=confusion_matrix_dir,
        logs_dir=logs_dir,
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_toml_mapping(path: Path) -> dict[str, object]:
    with path.open("rb") as file_obj:
        loaded = tomllib.load(file_obj)
    if not isinstance(loaded, dict):
        raise ValueError(f"TOML document is not a table: {path}")
    return cast(dict[str, object], loaded)


def semantic_hash(mapping: Mapping[str, object]) -> str:
    normalized = normalize_for_hash(mapping)
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def required_nested_string(
    *,
    mapping: Mapping[str, object],
    table: str,
    key: str,
    path: Path,
) -> str:
    raw_table = mapping.get(table)
    if not isinstance(raw_table, Mapping):
        raise ValueError(f"Missing or invalid [{table}] table in {path}")
    value = raw_table.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Missing or invalid {table}.{key} in {path}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{table}.{key} cannot be empty in {path}")
    return stripped


def normalize_for_hash(value: object) -> object:
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("Configuration mappings must use string keys.")
            normalized_mapping[key] = normalize_for_hash(value[key])
        return normalized_mapping

    if isinstance(value, list):
        return [normalize_for_hash(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_hash(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value

    raise ValueError(f"Unsupported TOML value type for hashing: {type(value)!r}")


def read_catalog(
    *,
    path: Path,
    id_column: str,
    hash_column: str,
    path_column: str,
    stem_column: str,
) -> list[CatalogRecord]:
    fieldnames = [id_column, hash_column, path_column, stem_column, "created_at_utc"]
    ensure_catalog(path=path, fieldnames=fieldnames)

    records: list[CatalogRecord] = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        if reader.fieldnames != fieldnames:
            raise ValueError(
                f"Catalog header mismatch at {path}: expected {fieldnames}, got {reader.fieldnames}"
            )
        for row in reader:
            records.append(
                CatalogRecord(
                    item_id=required_cell(row, id_column, path),
                    item_hash=required_cell(row, hash_column, path),
                    source_path=required_cell(row, path_column, path),
                    source_stem=required_cell(row, stem_column, path),
                    created_at_utc=required_cell(row, "created_at_utc", path),
                )
            )
    return records


def append_catalog_row(
    *,
    path: Path,
    record: CatalogRecord,
    id_column: str,
    hash_column: str,
    path_column: str,
    stem_column: str,
) -> None:
    fieldnames = [id_column, hash_column, path_column, stem_column, "created_at_utc"]
    ensure_catalog(path=path, fieldnames=fieldnames)
    with path.open("a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter="\t")
        writer.writerow(
            {
                id_column: record.item_id,
                hash_column: record.item_hash,
                path_column: record.source_path,
                stem_column: record.source_stem,
                "created_at_utc": record.created_at_utc,
            }
        )


def ensure_catalog(*, path: Path, fieldnames: Sequence[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def required_cell(row: Mapping[str, str], key: str, path: Path) -> str:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing column '{key}' in catalog row: {path}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"Empty column '{key}' in catalog row: {path}")
    return stripped


def find_record(
    *,
    records: Sequence[CatalogRecord],
    source_path: str,
    item_hash: str,
) -> CatalogRecord | None:
    for record in records:
        if record.source_path == source_path and record.item_hash == item_hash:
            return record
    return None


def build_unique_item_id(*, stem: str, item_hash: str, existing_ids: set[str]) -> str:
    normalized_stem = normalize_stem(stem)
    prefix_length = 6
    while prefix_length <= len(item_hash):
        candidate = f"{normalized_stem}-{item_hash[:prefix_length]}"
        if candidate not in existing_ids:
            return candidate
        prefix_length += 1
    raise ValueError("Unable to produce unique id from hash.")


def normalize_stem(stem: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-"
        for character in stem.strip()
    )
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "config"


def build_run_id() -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:6]
    return f"{timestamp}-{suffix}"


def backup_directory_once(*, source: Path, target: Path) -> None:
    if target.exists():
        return
    if not source.exists():
        raise FileNotFoundError(f"Backup source not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def write_text_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


def write_json_if_missing(path: Path, content: Mapping[str, Any]) -> None:
    if path.exists():
        return
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()
