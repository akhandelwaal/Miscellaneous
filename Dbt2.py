#!/usr/bin/env python3
"""
dbt_scaffold_expr.py — Enhanced scaffolder (IR → dbt project) with basic
Informatica expression translation.

Adds to the first‑cut:
- Translates common Informatica expressions to Oracle/ANSI SQL in SELECT lists
  (IIF → CASE, DECODE → CASE, NVL/ISNULL → COALESCE, LTRIM/RTRIM → TRIM, UCASE/LCASE, REG_REPLACE → REGEXP_REPLACE).
- Uses one‑hop lineage (from IR) to map each target column back to an upstream
  port; if that port has an expression, the translated SQL is used.
- Leaves TODO notes for ambiguous patterns.

Run:
  python dbt_scaffold_expr.py --ir ./out_ir --out ./dbt_oracle --name my_oracle_project
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
import re

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

# ---------------------- Expression Translator -----------------------

class ExprTranslator:
    """Conservative rule‑based translator for common Informatica expressions → SQL."""

    def __init__(self) -> None:
        self.re_space = re.compile(r"\s+")

    def normalize(self, s: str) -> str:
        return self.re_space.sub(" ", (s or "").strip())

    def translate(self, expr: str) -> tuple[str, str]:
        if not expr:
            return "", ""
        sql = self.normalize(expr)

        # $$PARAM → {{ var('PARAM') }}  (dbt vars)
        # Use a callback to avoid backref surprises
        sql = re.sub(r"\$\$([A-Za-z0-9_]+)", lambda m: "{{ var('" + m.group(1) + "') }}", sql)

        # NVL/ISNULL → COALESCE
        sql = re.sub(r"\bNVL\s*\(", "COALESCE(", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bISNULL\s*\(", "COALESCE(", sql, flags=re.IGNORECASE)

        # LTRIM(RTRIM(x)) → TRIM(x)
        sql = re.sub(r"LTRIM\s*\(\s*RTRIM\s*\(([^)]+)\)\s*\)", r"TRIM(\1)", sql, flags=re.IGNORECASE)

        # UCASE/LCASE → UPPER/LOWER
        sql = re.sub(r"\bUCASE\s*\(", "UPPER(", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bLCASE\s*\(", "LOWER(", sql, flags=re.IGNORECASE)

        # REG_REPLACE → REGEXP_REPLACE (Oracle)
        sql = re.sub(r"\bREG_REPLACE\s*\(", "REGEXP_REPLACE(", sql, flags=re.IGNORECASE)

        # DECODE(x, k1, v1, k2, v2, d) → CASE WHEN ... END
        def translate_decode(match: re.Match) -> str:
            inner = match.group(1)
            args = self._split_args(inner)
            if len(args) < 3:
                return f"/* TODO: review DECODE */ DECODE({inner})"
            x = args[0]
            pairs = args[1:]
            parts = ["CASE"]
            for i in range(0, len(pairs) - 1, 2):
                k = pairs[i]
                v = pairs[i + 1]
                parts.append(f" WHEN {x} = {k} THEN {v}")
            if len(pairs) % 2 == 1:
                parts.append(f" ELSE {pairs[-1]}")
            parts.append(" END")
            return "".join(parts)

        # Apply DECODE repeatedly in case of nesting
        for _ in range(4):
            before = sql
            sql = re.sub(r"\bDECODE\s*\(([^()]|\([^()]*\))*\)",
                         lambda m: translate_decode(re.match(r"DECODE\s*\((.*)\)", m.group(0), re.IGNORECASE) or m),
                         sql,
                         flags=re.IGNORECASE)
            if sql == before:
                break

        # IIF(cond, a, b) → CASE WHEN cond THEN a ELSE b END
        def translate_iif(match: re.Match) -> str:
            inner = match.group(1)
            args = self._split_args(inner)
            if len(args) != 3:
                return f"/* TODO: review IIF */ IIF({inner})"
            cond, a, b = args
            return f"CASE WHEN {cond} THEN {a} ELSE {b} END"

        for _ in range(6):
            before = sql
            sql = re.sub(r"\bIIF\s*\(([^()]|\([^()]*\))*\)",
                         lambda m: translate_iif(re.match(r"IIF\s*\((.*)\)", m.group(0), re.IGNORECASE) or m),
                         sql,
                         flags=re.IGNORECASE)
            if sql == before:
                break

        note = ""
        if "TODO:" in sql:
            note = "Check complex DECODE/IIF translation."
        return sql, note

    def _split_args(self, s: str) -> List[str]:
        args: List[str] = []
        buf: List[str] = []
        depth = 0
        quote: Optional[str] = None
        i = 0
        while i < len(s):
            ch = s[i]
            if quote:
                buf.append(ch)
                if ch == quote and (i == 0 or s[i-1] != '\\'):
                    quote = None
            else:
                if ch in ("'", '"'):
                    quote = ch
                    buf.append(ch)
                elif ch == '(':
                    depth += 1
                    buf.append(ch)
                elif ch == ')':
                    depth -= 1
                    buf.append(ch)
                elif ch == ',' and depth == 0:
                    args.append("".join(buf).strip())
                    buf = []
                else:
                    buf.append(ch)
            i += 1
        if buf:
            args.append("".join(buf).strip())
        return args

# ----------------------------- Utils --------------------------------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (name or "").strip())

# ----------------------- Project Generators -------------------------

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
# Stub profile. Move to ~/.dbt/profiles.yml or configure in dbt Cloud.
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
Generated from Informatica IR. Expressions translated where possible; review
TODOs and adjust joins/filters before production.
"""

    (out_dir / "dbt_project.yml").write_text(dbt_project, encoding="utf-8")
    (out_dir / "packages.yml").write_text(packages_yml, encoding="utf-8")
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "profiles.stub.yml").write_text(profiles_stub, encoding="utf-8")
    (out_dir / "macros" / "surrogate_key.sql").write_text(macro_surrogate, encoding="utf-8")

# -------------------------- IR Loading --------------------------------

def collect_ir_mappings(ir_dir: Path) -> List[IRMapping]:
    mappings_dir = ir_dir / "mappings"
    if not mappings_dir.exists():
        raise FileNotFoundError(f"IR dir missing 'mappings': {mappings_dir}")
    mappings: List[IRMapping] = []
    for folder_dir in sorted(mappings_dir.glob("*")):
        for f in sorted(folder_dir.glob("*.json")):
            raw = read_json(f)
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

# ----------------------- Sources & Schemas ---------------------------

def group_sources_by_schema(mappings: List[IRMapping]) -> Dict[Tuple[str, str], List[IRSourceTarget]]:
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
        lines.append("    freshness:")
        lines.append("      warn_after: {count: 48, period: hour}")
        lines.append("      error_after: {count: 7, period: day}\n")
    (project_dir / "models" / "sources.yml").write_text("\n".join(lines), encoding="utf-8")

# ---------------------- Staging & Mart Models ------------------------

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
            "-- Staging model generated from IR",
            f"select * from {{% raw %}}{{{{ source('{src_schema_name}', '{table_name}') }}}}{{% endraw %}}",
            "",
        ]
        (stg_dir / f"{model_name}.sql").write_text("\n".join(sql), encoding="utf-8")


def _unique_key_for_target(t: IRSourceTarget) -> str | None:
    keys = t.keys.get("primary") or []
    return keys[0] if keys else None


def _build_port_index(mapping: IRMapping) -> Dict[Tuple[str, str], IRPort]:
    idx: Dict[Tuple[str, str], IRPort] = {}
    for t in mapping.transformations:
        for p in t.ports:
            idx[(t.name, p.name)] = p
    for s in mapping.sources:
        for c in s.cols:
            fake = IRPort(name=c.get("name"), datatype=c.get("datatype"))
            idx[(s.name, c.get("name"))] = fake
    return idx


def _target_inbound_edges(mapping: IRMapping, target_name: str, col_name: str) -> List[Dict[str, str]]:
    return [e for e in mapping.connectors if e.get("to_instance") == target_name and e.get("to_port") == col_name]


def _render_expr_or_col(et: ExprTranslator, port: IRPort) -> tuple[str, Optional[str]]:
    if port and port.expression:
        sql, note = et.translate(port.expression)
        return sql, (note or None)
    return port.name, None


def write_mart_models(project_dir: Path, folder: str, mapping: IRMapping, et: ExprTranslator) -> None:
    mart_dir = project_dir / "models" / sanitize(folder) / "marts"
    ensure_dir(mart_dir)

    base_src = mapping.sources[0] if mapping.sources else None
    src_schema_name = sanitize(base_src.schema) if base_src and base_src.schema else "UNKNOWN"
    base_table = base_src.table or base_src.name if base_src else "<REPLACE_ME>"

    port_index = _build_port_index(mapping)

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

        select_lines: List[str] = []
        todo_notes: List[str] = []
        for c in tgt.cols:
            cname = c.get("name")
            if not cname:
                continue
            edges = _target_inbound_edges(mapping, tgt.name, cname)
            if edges:
                e = edges[0]
                upstream = port_index.get((e.get("from_instance"), e.get("from_port")))
                if upstream:
                    expr_sql, note = _render_expr_or_col(et, upstream)
                    alias = f" as {cname}" if expr_sql.strip().upper() != cname.upper() else ""
                    select_lines.append(f"    {expr_sql}{alias}")
                    if note:
                        todo_notes.append(f"{cname}: {note}")
                    continue
            select_lines.append(f"    {cname}")

        if not select_lines:
            select_lines = ["    * -- TODO: enumerate columns"]

        lines = [
            f"{{{{ config({cfg}) }}}}",
            "",
            f"-- Mapping: {mapping.mapping_name}",
            f"-- Complexity flags: {', '.join(mapping.complexity_flags) if mapping.complexity_flags else 'none'}",
        ]
        if todo_notes:
            lines.append("-- TODOs:")
            for tnote in todo_notes:
                lines.append(f"--   - {tnote}")
        lines.extend([
            "",
            "with base as (",
            f"  select * from {{% raw %}}{{{{ source('{src_schema_name}', '{base_table}') }}}}{{% endraw %}}",
            ")",
            "select",
            ",\n".join(select_lines),
            "from base",
        ])

        hints = mapping.incremental_hints.get("watermark_candidates") if mapping.incremental_hints else None
        if mapping.update_strategy and hints:
            wm = hints[0]
            lines.extend([
                "",
                "{% if is_incremental() %}",
                f"where {wm} >= {{ dbt.dateadd('day', -3, 'current_date') }}",
                "{% endif %}",
            ])

        (mart_dir / f"{model_name}.sql").write_text("\n".join(lines), encoding="utf-8")

# ----------------------- Schema.yml per folder -----------------------

def write_folder_schema_yml(folder_dir: Path, folder: str, targets: List[IRSourceTarget]) -> None:
    lines: List[str] = ["version: 2", "models:"]
    for t in targets:
        model_name = f"{sanitize(folder)}__{sanitize(t.name)}"
        lines.append(f"  - name: {model_name}")
        if t.keys.get("primary"):
            lines.append("    tests:")
            for k in t.keys["primary"]:
                lines.append("      - not_null:")
                lines.append(f"          column_name: {k}")
            for k in t.keys["primary"]:
                lines.append("      - unique:")
                lines.append(f"          column_name: {k}")
    (folder_dir / "schema.yml").write_text("\n".join(lines), encoding="utf-8")

# ------------------------------ Orchestration ------------------------

def scaffold_project(ir_dir: Path, out_dir: Path, project_name: str) -> None:
    et = ExprTranslator()
    mappings = collect_ir_mappings(ir_dir)
    if not mappings:
        raise RuntimeError("No mappings found in IR dir")

    generate_project_files(out_dir, project_name)
    write_sources_yml(out_dir, project_name, mappings)

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

    for m in mappings:
        write_mart_models(out_dir, m.folder, m, et)

    print(f"\n✅ dbt scaffold (with expression translation) generated at: {out_dir}")

# ------------------------------ CLI ---------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a dbt project from IR (with expression translation)")
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
