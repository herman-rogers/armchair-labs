"""Small declarative registry; recipes are trusted repository code."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_REGISTRY = Path(__file__).with_name("registry.yaml")


def table_name(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError(f"Invalid table name: {value!r}")
    return value


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    grain: str
    primary_key: list[str] = Field(min_length=1)
    kind: Literal["python", "sql", "parquet", "source"]
    builder: str | None = None
    sql_file: str | None = None
    source: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    # None means explicit/manual refresh only.
    refresh_hours: int | None = Field(default=None, gt=0)
    parameters: dict = Field(default_factory=dict)
    # Data files, relative to data_dir, included in the input fingerprint.
    inputs: list[str] = Field(default_factory=list)
    # Python module names whose implementations affect the builder's results.
    code_modules: list[str] = Field(default_factory=list)
    min_rows: int = Field(default=1, ge=0)
    required_columns: dict[str, str] = Field(default_factory=dict)
    observation_cutoff: dict = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_recipe(self):
        fields = {"python": "builder", "sql": "sql_file", "parquet": "source", "source": "source"}
        selected = fields[self.kind]
        if not getattr(self, selected) or any(
            getattr(self, f) is not None for f in fields.values() if f != selected
        ):
            raise ValueError(f"{self.kind} requires exactly {selected}")
        if len(set(self.primary_key)) != len(self.primary_key):
            raise ValueError("Duplicate primary-key columns")
        for name in self.dependencies:
            table_name(name)
        return self


def read_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, TableSpec]:
    raw = yaml.safe_load(path.read_text())
    if raw.get("schema_version") != 1 or set(raw) != {"schema_version", "tables"}:
        raise ValueError("Unsupported table registry")
    result = {table_name(k): TableSpec.model_validate(v) for k, v in raw["tables"].items()}
    order(result, list(result))  # reject missing dependencies and cycles before building
    return result


def order(registry: dict[str, TableSpec], names: list[str]) -> list[str]:
    done: list[str] = []
    visiting: set[str] = set()

    def visit(name):
        if name not in registry:
            raise ValueError(f"Unknown table: {name}")
        if name in visiting:
            raise ValueError(f"Cyclic table dependency: {name}")
        if name in done:
            return
        visiting.add(name)
        for dep in registry[name].dependencies:
            visit(dep)
        visiting.remove(name)
        done.append(name)

    for name in names:
        visit(name)
    return done
