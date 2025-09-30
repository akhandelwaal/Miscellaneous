#!/usr/bin/env python3
"""
dbt_scaffold.py — Generate a first-cut dbt project from the IR produced by inf2ir.py.

Inputs (IR dir):
  - mappings/<folder>/<mapping>.json (required)
  - lineage.csv, inventory.csv, expressions.csv (optional)

Outputs (dbt project dir):
  - dbt_project.yml
  - packages.yml
  - README.md
  - models/<folder>/{staging,intermediate,marts}/... .sql
  - models/<folder>/schema.yml (sources + basic tests)
  - macros/surrogate_key.sql (starter)

Heuristics ("first cut"):
  - Create one "staging" model per unique source definition (select * from source).
  - Create one "mart" model per target table in a mapping.
  - If Update Strategy is present, configure incremental + unique_key from target primary keys.
  - Generate sources.yml entries per schema/owner; add freshness placeholder.
  - Add basic tests (unique + not_null) on primary keys when available.

Extend later:
  - Column-level transformations using expressions + lineage propagation.
  - Lookup-to-join materialization in intermediate models.
  - Session overrides & parameters to vars/env mapping.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

# ---------------------------- Data Types ----------------------------

@dataclass
class IRPort:
    name: str
    datatype: str | None = None
    precision: str | None = None
    scale: str | None = None
    expression: str | None = None
    default: str | None = None

@dataclass
class IRSourceTarget:
    name: str
    type: str
    schema: str | None
    connection: str | None
    table: str | None
    query: str | None
    cols: List[Dict[str, Any]]
    keys: Dict[str, List[str]]

@dataclass
class IRTransformation:
    name: str
    type: str
    ports: List[IRPort]
    attributes: Dict[str, Any]
    lookup_source: str | None
    lookup_condition: List[Dict[str, str]]
    groups: List[Dict[str, Any]]

@dataclass
class IRMapping:
    folder: str
    mapping_name: str
    sources: List[IRSourceTarget]
    targets: List[IRSourceTarget]
    transformations: List[IRTransformation]
    connectors: List[Dict[str, str]]
    update_strategy: Dict[str, Any]
    incremental_hints: Dict[str, Any]
    parameters: Dict[str, Any]
    lineage: List[Dict[str, str]]
    complexity_flags: List[str]
    notes: List[str]

# ----------------------------- Utils --------------------------------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (name or "").strip())

# -------------------------- Generators ------------------------------

def generate_project_files(out_dir: Path, project_name: str) -> None:
    ensure_dir(out_dir)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "macros").mkdir(exist_ok=True)

    dbt_project = f"""
name: {project_name}
version: 1.0.0
profile: {project_name}
config-version: 2
models:
  {project_name}:
    +materialized: view
    staging:
      +schema: stg
    intermediate:
      +schema: int
    marts:
      +schema: mart
""".lstrip()

    profiles_stub = f"""
# This is a stub profile. Move to ~/.dbt/profiles.yml or dbt Cloud environment.
{project_name}:
  target: dev
  outputs:
    dev:
      type: oracle
      host: <host>
      port: 1521
      service: <service_name>
      user: <user>
      password: <password>
      protocol: tcp
      threads: 4
      # Refer to dbt-oracle docs for full config options.
""".lstrip()

    packages_yml = """
packages:
  - package: dbt-labs/dbt_utils
    version: ">=1.1.1"
""".lstrip()

    macro_surrogate = """
{% macro surrogate_key(cols) %}
    md5(concat_ws('||', {% for c in cols %} coalesce({{ c }}, '__NULL__') {% if not loop.last %}, {% endif %}{% endfor %}))
{% endmacro %}
"""

    readme = f"""
# {project_name}

This project was generated from an Informatica → IR → dbt scaffold. Treat the generated SQL as a starting point — review joins, filters, and expressions.
"""

    (out_dir / "dbt_project.yml").write_text(dbt_project, encoding="utf-8")
    (out_dir / "packages.yml").write_text(packages_yml, encoding="utf-8")
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "profiles.stub.yml").write_text(profiles_stub, encoding="utf-8")
    (out_dir / "macros" / "surrogate_key.sql").write_text(macro_surrogate, encoding="utf-8")


def collect_ir_mappings(ir_dir: Path) -> List[IRMapping]:
    mappings_dir = ir_dir / "mappings"
    if not mappings_dir.exists():
        raise FileNotFoundError(f"IR dir missing 'mappings': {mappings_dir}")
    mappings: List[IRMapping] = []
    for folder_dir in sorted(mappings_dir.glob("*")):
        for f in sorted(folder_dir.glob("*.json")):
            raw = read_json(f)
            # Coerce into dataclasses (shallow)
            srcs = [IRSourceTarget(**s) for s in raw.get("sources", [])]
            tgts = [IRSourceTarget(**t) for t in raw.get("targets", [])]
            tfs: List[IRTransformation] = []
            for t in raw.get("transformations", []):
                ports = [IRPort(**p) for p in t.get("ports", [])]
                tfs.append(IRTransformation(
                    name=t.get("name"),
                    type=t.get("type"),
                    ports=ports,
                    attributes=t.get("attributes", {}),
                    lookup_source=t.get("lookup_source"),
                    lookup_condition=t.get("lookup_condition", []),
                    groups=t.get("groups", []),
                ))
            mp = IRMapping(
                folder=raw.get("folder"),
                mapping_name=raw.get("mapping_name"),
                sources=srcs,
                targets=tgts,
                transformations=tfs,
                connectors=raw.get("connectors", []),
                update_strategy=raw.get("update_strategy", {}),
                incremental_hints=raw.get("incremental_hints", {}),
                parameters=raw.get("parameters", {}),
                lineage=raw.get("lineage", []),
                complexity_flags=raw.get("complexity_flags", []),
                notes=raw.get("notes", []),
            )
            mappings.append(mp)
    return mappings


def group_sources_by_schema(mappings: List[IRMapping]) -> Dict[Tuple[str, str], List[IRSourceTarget]]:
    """Group (schema, connection) → [sources]."""
    groups: Dict[Tuple[str, str], List[IRSourceTarget]] = {}
    seen: Set[str] = set()
    for m in mappings:
        for s in m.sources:
            key = (s.schema or "UNKNOWN", s.connection or "")
            sig = f"{key}|{s.table}|{s.name}"
            if sig in seen:
                continue
            seen.add(sig)
            groups.setdefault(key, []).append(s)
    # Sort consistently
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: (x.schema or "", x.table or "", x.name))
    return groups


def write_sources_yml(project_dir: Path, project_name: str, mappings: List[IRMapping]) -> None:
    groups = group_sources_by_schema(mappings)
    lines: List[str] = ["version: 2", "sources:"]
    for (schema, _cnx), srcs in sorted(groups.items(), key=lambda kv: kv[0]):
        src_name = sanitize(schema)
        lines.append(f"  - name: {src_name}")
        lines.append(f"    schema: {schema}")
        lines.append("    tables:")
        for s in srcs:
            tname = s.table or s.name
            lines.append(f"      - name: {tname}")
            if s.cols:
                lines.append("        columns:")
                for c in s.cols:
                    cname = c.get("name")
                    if not cname:
                        continue
                    lines.append(f"          - name: {cname}")
                    # placeholder tests may be added later
        lines.append("    freshness:")
        lines.append("      warn_after: {count: 48, period: hour}")
        lines.append("      error_after: {count: 7, period: day}
")
    models_dir = project_dir / "models"
    ensure_dir(models_dir)
    (models_dir / "sources.yml").write_text("\n".join(lines), encoding="utf-8")


def write_folder_schema_yml(folder_dir: Path, folder: str, targets: List[IRSourceTarget]) -> None:
    lines: List[str] = ["version: 2", "models:"]
    for t in targets:
        model_name = f"{sanitize(folder)}__{sanitize(t.name)}"
        lines.append(f"  - name: {model_name}")
        if t.keys.get("primary"):
            lines.append("    tests:")
            # Use dbt_utils.unique + not_null on concatenated keys (emit both)
            for k in t.keys["primary"]:
                lines.append("      - not_null:")
                lines.append(f"          column_name: {k}")
            for k in t.keys["primary"]:
                lines.append("      - unique:")
                lines.append(f"          column_name: {k}")
    (folder_dir / "schema.yml").write_text("\n".join(lines), encoding="utf-8")


def write_staging_models(project_dir: Path, project_name: str, folder: str, sources: List[IRSourceTarget]) -> None:
    stg_dir = project_dir / "models" / sanitize(folder) / "staging"
    ensure_dir(stg_dir)
    for s in sources:
        src_schema_name = sanitize(s.schema or "UNKNOWN")
        table_name = s.table or s.name
        model_name = f"src__{sanitize(table_name)}"
        sql = [
            "{{ config(materialized='view') }}",
            "",
            "-- First-cut staging model generated from IR",
            f"select * from {{% raw %}}{{{{ source('{src_schema_name}', '{table_name}') }}}}{{% endraw %}}",
            "",
        ]
        (stg_dir / f"{model_name}.sql").write_text("\n".join(sql), encoding="utf-8")


def _unique_key_for_target(t: IRSourceTarget) -> str | None:
    keys = t.keys.get("primary") or []
    return keys[0] if keys else None


def write_mart_models(project_dir: Path, folder: str, mapping: IRMapping) -> None:
    mart_dir = project_dir / "models" / sanitize(folder) / "marts"
    ensure_dir(mart_dir)

    # Build a naive FROM clause: choose the first source in this mapping
    base_src = mapping.sources[0] if mapping.sources else None
    src_schema_name = sanitize(base_src.schema) if base_src and base_src.schema else "UNKNOWN"
    base_table = base_src.table or base_src.name if base_src else "<REPLACE_ME>"

    for tgt in mapping.targets:
        model_name = f"{sanitize(folder)}__{sanitize(tgt.name)}"
        unique_key = _unique_key_for_target(tgt)
        cfg_bits = ["materialized='table'"]
        if mapping.update_strategy:
            cfg_bits.append("materialized='incremental'")
            if unique_key:
                cfg_bits.append(f"unique_key='{unique_key}'")
            cfg_bits.append("incremental_strategy='merge'")
        cfg = ", ".join(sorted(set(cfg_bits)))

        select_cols = []
        for c in tgt.cols:
            cname = c.get("name")
            if not cname:
                continue
            select_cols.append(f"    {cname}")
        if not select_cols:
            select_cols = ["    * -- TODO: enumerate columns"]

        sql_lines = [
            f"{{{{ config({cfg}) }}}}",
            "",
            f"-- Mapping: {mapping.mapping_name}",
            f"-- Complexity flags: {', '.join(mapping.complexity_flags) if mapping.complexity_flags else 'none'}",
            "-- NOTE: This is a naive first-cut. Review joins/filters/expressions.",
            "",
            "with base as (",
            f"  select * from {{% raw %}}{{{{ source('{src_schema_name}', '{base_table}') }}}}{{% endraw %}}",
            ")",
            "select",
            ",\n".join(select_cols),
            "from base",
        ]

        # Incremental guard band hint if we have incremental_hints
        hints = mapping.incremental_hints.get("watermark_candidates") if mapping.incremental_hints else None
        if mapping.update_strategy and hints:
            wm = hints[0]
            sql_lines.extend([
                "",
                "{% if is_incremental() %}",
                f"where {wm} >= {{ dbt.dateadd('day', -3, 'current_date') }}",
                "{% endif %}",
            ])

        (mart_dir / f"{model_name}.sql").write_text("\n".join(sql_lines), encoding="utf-8")


def scaffold_project(ir_dir: Path, out_dir: Path, project_name: str) -> None:
    mappings = collect_ir_mappings(ir_dir)
    if not mappings:
        raise RuntimeError("No mappings found in IR dir")

    generate_project_files(out_dir, project_name)

    # Write global sources.yml once under models/
    write_sources_yml(out_dir, project_name, mappings)

    # For each folder, write staging models for its sources and schema.yml for targets
    by_folder: Dict[str, Dict[str, Any]] = {}
    for m in mappings:
        slot = by_folder.setdefault(m.folder, {"sources": {}, "targets": []})
        for s in m.sources:
            key = (s.schema, s.table or s.name)
            slot["sources"][key] = s
        slot["targets"].extend(m.targets)

    for folder, bundle in by_folder.items():
        folder_dir = out_dir / "models" / sanitize(folder)
        ensure_dir(folder_dir / "staging")
        ensure_dir(folder_dir / "intermediate")
        ensure_dir(folder_dir / "marts")
        write_staging_models(out_dir, project_name, folder, list(bundle["sources"].values()))
        write_folder_schema_yml(folder_dir, folder, bundle["targets"])

    # Mart models per mapping/target
    for m in mappings:
        write_mart_models(out_dir, m.folder, m)

    print(f"\n✅ dbt scaffold generated at: {out_dir}")

# ------------------------------ CLI ---------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a dbt project from IR")
    ap.add_argument("--ir", required=True, help="Path to IR directory (from inf2ir.py)")
    ap.add_argument("--out", required=True, help="Output directory for dbt project")
    ap.add_argument("--name", default="inf_dbt_project", help="dbt project name")
    args = ap.parse_args()

    ir_dir = Path(args.ir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaffold_project(ir_dir, out_dir, args.name)


if __name__ == "__main__":
    main()
