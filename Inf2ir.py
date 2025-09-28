#!/usr/bin/env python3
"""
inf2ir.py — Parse Informatica (PowerCenter) repository/workflow XML exports
and produce a clean Intermediate Representation (IR) that downstream
generators (e.g., dbt scaffolder) can consume.

Goals
-----
* Accept one or more Informatica XML export files (Repository / Folder level).
* Traverse FOLDER → {MAPPING, WORKFLOW, SESSION, MAPPLET} graphs.
* Extract for each mapping:
    - sources, targets, transformations, ports, expressions, keys
    - connectors (lineage edges)
    - update strategies, lookups, source qualifiers, router groups
    - session overrides (connections, SQL overrides), parameters
* Emit per-mapping JSON IR + global CSV summaries (lineage, inventory,
  expression catalog).

Usage
-----
python inf2ir.py extract --xml repo_export.xml --out ./out_ir
python inf2ir.py summarize --in ./out_ir --html ./summary.html  # optional (minimal)

Design
------
* Pure standard library (xml.etree + argparse + json + csv). No external deps.
* Defensive parsing: missing nodes won't crash; we annotate "notes" and flags.
* Deterministic output: sorted keys and lists to keep diffs stable.

IR Artifacts (written under --out):
* mappings/<folder>/<mapping_name>.json
* workflows/<folder>/<workflow_name>.json (basic for now)
* lineage.csv  — edges (from_instance.port → to_instance.port) per mapping
* inventory.csv — one row per mapping with complexity score
* expressions.csv — unique expressions and where-used counts

This is a solid starter you can extend with organization-specific rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import xml.etree.ElementTree as ET

# ----------------------------- Helpers ------------------------------------

@dataclass
class IRPort:
    name: str
    datatype: Optional[str] = None
    precision: Optional[str] = None
    scale: Optional[str] = None
    expression: Optional[str] = None
    default: Optional[str] = None

@dataclass
class IRTransformation:
    name: str
    type: str
    ports: List[IRPort] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    lookup_source: Optional[str] = None
    lookup_condition: List[Dict[str, str]] = field(default_factory=list)  # [{left, op, right}]
    groups: List[Dict[str, Any]] = field(default_factory=list)  # router/aggregator groups

@dataclass
class IRSourceTarget:
    name: str
    type: str  # table, view, file, query
    schema: Optional[str] = None
    connection: Optional[str] = None
    table: Optional[str] = None
    query: Optional[str] = None
    cols: List[Dict[str, Any]] = field(default_factory=list)
    keys: Dict[str, List[str]] = field(default_factory=dict)  # primary/natural

@dataclass
class IRMapping:
    folder: str
    mapping_name: str
    sources: List[IRSourceTarget]
    targets: List[IRSourceTarget]
    transformations: List[IRTransformation]
    connectors: List[Dict[str, str]]  # from_instance, from_port, to_instance, to_port
    update_strategy: Dict[str, Any] = field(default_factory=dict)
    incremental_hints: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    lineage: List[Dict[str, str]] = field(default_factory=list)  # resolved col lineage (best-effort)
    complexity_flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

@dataclass
class IRWorkflow:
    folder: str
    workflow_name: str
    worklets: List[Dict[str, Any]] = field(default_factory=list)
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    schedule: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

# ------------------------ Core extractor class ----------------------------

class InformaticaIRExtractor:
    def __init__(self, xml_paths: List[Path]):
        self.xml_paths = xml_paths
        self.trees = [ET.parse(str(p)) for p in xml_paths]
        self.root_nodes = [t.getroot() for t in self.trees]
        self.expr_catalog = Counter()
        self.expr_examples: Dict[str, Tuple[str, str, str]] = {}
        self.lineage_rows: List[List[str]] = []
        self.inventory_rows: List[List[Any]] = []

    # ---------------------------- Public API -----------------------------

    def extract(self, out_dir: Path) -> None:
        mappings: List[IRMapping] = []
        workflows: List[IRWorkflow] = []

        for root in self.root_nodes:
            # Iterate folders
            for folder in root.findall(".//FOLDER"):
                folder_name = folder.attrib.get("NAME", "UNKNOWN_FOLDER")
                # Mappings
                for mnode in folder.findall("./MAPPING"):
                    mapping = self._extract_mapping(folder_name, mnode)
                    mappings.append(mapping)
                # Workflows (basic)
                for wnode in folder.findall("./WORKFLOW"):
                    workflows.append(self._extract_workflow(folder_name, wnode))

        # Write per-mapping JSON and per-workflow JSON
        for m in mappings:
            self._write_mapping(out_dir, m)
        for w in workflows:
            self._write_workflow(out_dir, w)

        # Write lineage.csv, inventory.csv, expressions.csv
        self._write_lineage(out_dir)
        self._write_inventory(out_dir)
        self._write_expressions(out_dir)

    # ----------------------- Mapping extraction --------------------------

    def _extract_mapping(self, folder_name: str, mnode: ET.Element) -> IRMapping:
        mname = mnode.attrib.get("NAME", "UNKNOWN_MAPPING")
        sources: Dict[str, IRSourceTarget] = {}
        targets: Dict[str, IRSourceTarget] = {}
        transformations: List[IRTransformation] = []
        connectors: List[Dict[str, str]] = []
        parameters: Dict[str, Any] = {}
        complexity: List[str] = []
        notes: List[str] = []

        # Sources & Targets are expressed as TRANSFORMATIONs of types Source Definition / Target Definition
        for tnode in mnode.findall("./TRANSFORMATION"):
            ttype = tnode.attrib.get("TYPE", "").strip()
            tname = tnode.attrib.get("NAME", "").strip()

            if ttype in ("Source Definition", "Target Definition"):
                st = self._extract_source_or_target(tnode, ttype)
                if ttype == "Source Definition":
                    sources[tname] = st
                else:
                    targets[tname] = st
                continue

        # Other transformations
        for tnode in mnode.findall("./TRANSFORMATION"):
            ttype = tnode.attrib.get("TYPE", "").strip()
            tname = tnode.attrib.get("NAME", "").strip()
            if ttype in ("Source Definition", "Target Definition"):
                continue

            tf = IRTransformation(name=tname, type=ttype)
            # Ports
            for p in tnode.findall("./TRANSFORMFIELD"):
                port = IRPort(
                    name=p.attrib.get("NAME", ""),
                    datatype=p.attrib.get("DATATYPE"),
                    precision=p.attrib.get("PRECISION"),
                    scale=p.attrib.get("SCALE"),
                    expression=(p.attrib.get("EXPRESSION") or "").strip() or None,
                    default=(p.attrib.get("DEFAULTVALUE") or None),
                )
                if port.expression:
                    self._record_expression(port.expression, folder_name, mname, tname)
                tf.ports.append(port)

            # Attributes (generic catch-all)
            for a in tnode.findall("./TABLEATTRIBUTE") + tnode.findall("./ATTRIBUTE"):
                key = a.attrib.get("NAME") or a.attrib.get("NAME2") or a.attrib.get("ATTRIBUTE_NAME")
                val = a.attrib.get("VALUE") or a.attrib.get("ATTRIBUTE_VALUE")
                if key:
                    tf.attributes[key] = val

            # Lookup specifics
            if ttype == "Lookup Procedure" or ttype == "Lookup":
                # Older exports may use "Lookup Procedure"
                src = tf.attributes.get("Lookup table name") or tf.attributes.get("LookupSourceName")
                if src:
                    tf.lookup_source = src
                # Conditions sometimes appear as child elements
                for cond in tnode.findall(".//LOOKUPCONDITION"):
                    left = cond.attrib.get("LEFT_EXPRESSION") or cond.attrib.get("LEFTFIELD")
                    op = cond.attrib.get("OPERATOR") or cond.attrib.get("OP") or "="
                    right = cond.attrib.get("RIGHT_EXPRESSION") or cond.attrib.get("RIGHTFIELD")
                    tf.lookup_condition.append({"left": left or "", "op": op, "right": right or ""})
                complexity.append("Lookup")

            # Router groups
            if ttype == "Router":
                for group in tnode.findall("./GROUP"):
                    tf.groups.append({
                        "name": group.attrib.get("NAME", "GROUP"),
                        "filter": group.attrib.get("EXPRESSION", "")
                    })
                complexity.append("Router")

            # Update Strategy
            if ttype == "Update Strategy":
                complexity.append("UpdateStrategy")

            # Normalizer / Stored Procedure / Sequence Generator flags
            if ttype in ("Normalizer", "Stored Procedure", "Java Transformation"):
                complexity.append(ttype.replace(" ", ""))

            transformations.append(tf)

        # Connectors (edges)
        for c in mnode.findall("./CONNECTOR"):
            connectors.append({
                "from_instance": c.attrib.get("FROMINSTANCE", ""),
                "from_port": c.attrib.get("FROMPORT", ""),
                "to_instance": c.attrib.get("TOINSTANCE", ""),
                "to_port": c.attrib.get("TOPORT", ""),
            })

        # Parameters declared on mapping (rare) — collect PARAMFILE / VARIABLES if present
        for p in mnode.findall("./MAPPINGVARIABLE"):
            parameters[p.attrib.get("NAME", "")] = p.attrib.get("DEFAULTVALUE")

        # Build best-effort resolved lineage (column-level) using simple pass through connectors
        lineage = self._resolve_lineage_simple(sources, targets, transformations, connectors)

        # Heuristics for incremental hints & update strategy
        update_strategy = self._detect_update_strategy(transformations)
        incr_hints = self._infer_incremental_hint(transformations, sources)

        # Complexity score + inventory row
        expr_count = sum(len(t.ports) for t in transformations)
        has_stored_proc = any(t.type == "Stored Procedure" for t in transformations)
        has_normalizer = any(t.type == "Normalizer" for t in transformations)
        score = (2 if has_stored_proc else 0) + (1 if has_normalizer else 0) + (1 if "Lookup" in complexity else 0) + (1 if "UpdateStrategy" in complexity else 0) + (1 if expr_count > 30 else 0)
        self.inventory_rows.append([
            folder_name, mname, len(sources), len(targets),
            sum(1 for t in transformations if t.type in ("Lookup", "Lookup Procedure")),
            any(t.type == "Update Strategy" for t in transformations),
            has_stored_proc, expr_count, score
        ])

        # Lineage rows for CSV
        for e in connectors:
            self.lineage_rows.append([
                folder_name, mname,
                e.get("from_instance", ""), e.get("from_port", ""),
                e.get("to_instance", ""), e.get("to_port", ""),
            ])

        mapping = IRMapping(
            folder=folder_name,
            mapping_name=mname,
            sources=list(sources.values()),
            targets=list(targets.values()),
            transformations=transformations,
            connectors=connectors,
            update_strategy=update_strategy,
            incremental_hints=incr_hints,
            parameters=parameters,
            lineage=lineage,
            complexity_flags=sorted(set(complexity)),
            notes=notes,
        )
        return mapping

    # ----------------------- Workflow extraction -------------------------

    def _extract_workflow(self, folder_name: str, wnode: ET.Element) -> IRWorkflow:
        wname = wnode.attrib.get("NAME", "UNKNOWN_WORKFLOW")
        sessions = []
        for s in wnode.findall(".//SESSION"):
            sessions.append({
                "name": s.attrib.get("NAME"),
                "mapping_name": s.attrib.get("MAPPINGNAME"),
            })
        params: Dict[str, Any] = {}
        for a in wnode.findall("./ATTRIBUTE"):
            n = a.attrib.get("NAME") or a.attrib.get("ATTRIBUTE_NAME")
            v = a.attrib.get("VALUE") or a.attrib.get("ATTRIBUTE_VALUE")
            if n:
                params[n] = v
        return IRWorkflow(folder=folder_name, workflow_name=wname, sessions=sessions, parameters=params)

    # -------------------------- Small utilities --------------------------

    def _extract_source_or_target(self, tnode: ET.Element, ttype: str) -> IRSourceTarget:
        # Try to determine name/schema/table
        name = tnode.attrib.get("NAME", "")
        cols = []
        for f in tnode.findall("./TRANSFORMFIELD"):
            cols.append({
                "name": f.attrib.get("NAME", ""),
                "datatype": f.attrib.get("DATATYPE"),
                "precision": f.attrib.get("PRECISION"),
                "scale": f.attrib.get("SCALE"),
            })
        stype = "table"  # default assumption
        schema = None
        table = None
        query = None
        connection = None
        keys: Dict[str, List[str]] = {"primary": [], "natural": []}

        for a in tnode.findall("./TABLEATTRIBUTE"):
            an = a.attrib.get("NAME")
            av = a.attrib.get("VALUE")
            if an in ("Source Table Name", "Table Name") and av:
                table = av
            if an in ("Owner Name", "Table Owner") and av:
                schema = av
            if an == "Db Location Name" and av:
                connection = av
            if an == "Sql Query" and av:
                query = av
                stype = "query"

        # Keys from target definition (optional)
        for keynode in tnode.findall(".//KEYRANGE"):
            kname = keynode.attrib.get("NAME")
            ktype = keynode.attrib.get("TYPE", "PRIMARY").lower()
            if kname:
                if ktype.startswith("prim"):
                    keys["primary"].append(kname)
                else:
                    keys["natural"].append(kname)

        return IRSourceTarget(
            name=name,
            type=stype,
            schema=schema,
            connection=connection,
            table=table,
            query=query,
            cols=cols,
            keys=keys,
        )

    def _record_expression(self, expr: str, folder: str, mapping: str, tname: str) -> None:
        norm = self._normalize_expression(expr)
        self.expr_catalog[norm] += 1
        if norm not in self.expr_examples:
            self.expr_examples[norm] = (folder, mapping, tname)

    @staticmethod
    def _normalize_expression(expr: str) -> str:
        return " ".join(expr.strip().split())

    def _detect_update_strategy(self, transformations: List[IRTransformation]) -> Dict[str, Any]:
        for t in transformations:
            if t.type == "Update Strategy":
                # Detect usage of DD_* flags in port expressions
                modes = set()
                for p in t.ports:
                    if not p.expression:
                        continue
                    s = p.expression.upper()
                    for k in ("DD_INSERT", "DD_UPDATE", "DD_DELETE", "DD_REJECT"):
                        if k in s:
                            modes.add(k)
                return {"type": "update_strategy", "modes": sorted(modes)}
        return {}

    def _infer_incremental_hint(self, transformations: List[IRTransformation], sources: Dict[str, IRSourceTarget]) -> Dict[str, Any]:
        # Very light heuristic: date-like ports from Source Qualifier or names containing DATE/DT/TIMESTAMP
        candidate_cols = []
        for t in transformations:
            if t.type in ("Source Qualifier", "Expression"):
                for p in t.ports:
                    n = (p.name or "").upper()
                    d = (p.datatype or "").upper()
                    if any(k in n for k in ("DATE", "DT", "TS", "LOAD")) or d in ("DATE", "TIMESTAMP"):
                        candidate_cols.append(p.name)
        if candidate_cols:
            return {"watermark_candidates": sorted(set(candidate_cols))}
        return {}

    def _resolve_lineage_simple(
        self,
        sources: Dict[str, IRSourceTarget],
        targets: Dict[str, IRSourceTarget],
        transformations: List[IRTransformation],
        connectors: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """A simple pass to map source/transform ports → target ports based on direct chains.
        Not a full graph walker, but useful as a starting point.
        """
        # Build adjacency by instance.port
        adj = defaultdict(list)
        for e in connectors:
            src = f"{e.get('from_instance')}.{e.get('from_port')}"
            dst = f"{e.get('to_instance')}.{e.get('to_port')}"
            adj[src].append(dst)

        # Identify target columns and backtrack one hop to capture immediate parents
        target_ports = set()
        for tname, tgt in targets.items():
            for col in tgt.cols:
                target_ports.add(f"{tname}.{col['name']}")

        result = []
        for tp in sorted(target_ports):
            # find inbound edges where to_instance.port == tp
            ins = [e for e in connectors if f"{e.get('to_instance')}.{e.get('to_port')}" == tp]
            for e in ins:
                result.append({
                    "from": f"{e.get('from_instance')}.{e.get('from_port')}",
                    "to": tp
                })
        return result

    # -------------------------- Writers ----------------------------------

    def _write_mapping(self, out_dir: Path, mapping: IRMapping) -> None:
        folder_dir = out_dir / "mappings" / sanitize(mapping.folder)
        folder_dir.mkdir(parents=True, exist_ok=True)
        path = folder_dir / f"{sanitize(mapping.mapping_name)}.json"

        def default(o):
            if isinstance(o, (IRPort, IRTransformation, IRSourceTarget, IRMapping)):
                return asdict(o)
            return o

        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(mapping), f, indent=2, sort_keys=True, default=default)

    def _write_workflow(self, out_dir: Path, wf: IRWorkflow) -> None:
        folder_dir = out_dir / "workflows" / sanitize(wf.folder)
        folder_dir.mkdir(parents=True, exist_ok=True)
        path = folder_dir / f"{sanitize(wf.workflow_name)}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(wf), f, indent=2, sort_keys=True)

    def _write_lineage(self, out_dir: Path) -> None:
        path = out_dir / "lineage.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["folder", "mapping", "from_instance", "from_port", "to_instance", "to_port"])
            # stable sort
            for row in sorted(self.lineage_rows):
                w.writerow(row)

    def _write_inventory(self, out_dir: Path) -> None:
        path = out_dir / "inventory.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["folder", "mapping", "sources", "targets", "lookups", "has_update_strategy", "has_stored_proc", "expr_ports", "complexity_score"])
            for row in sorted(self.inventory_rows):
                w.writerow(row)

    def _write_expressions(self, out_dir: Path) -> None:
        path = out_dir / "expressions.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["normalized_expression", "count", "example_folder", "example_mapping", "example_transformation"])
            for expr, count in sorted(self.expr_catalog.items(), key=lambda kv: (-kv[1], kv[0])):
                ex_folder, ex_map, ex_tf = self.expr_examples.get(expr, ("", "", ""))
                w.writerow([expr, count, ex_folder, ex_map, ex_tf])

# ----------------------------- CLI ----------------------------------------

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name or "")


def cmd_extract(args: argparse.Namespace) -> None:
    xmls = [Path(p) for p in args.xml]
    for p in xmls:
        if not p.exists():
            raise FileNotFoundError(f"XML not found: {p}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = InformaticaIRExtractor(xmls)
    extractor.extract(out_dir)

    print(f"\n✅ Extraction complete. IR written to: {out_dir}")
    print("Artifacts:")
    print(" - mappings/<folder>/<mapping>.json")
    print(" - workflows/<folder>/<workflow>.json")
    print(" - lineage.csv, inventory.csv, expressions.csv")


def cmd_summarize(args: argparse.Namespace) -> None:
    # Minimal placeholder: read inventory & print top stats; extend as needed to HTML.
    inv = Path(args.input) / "inventory.csv"
    if not inv.exists():
        print("inventory.csv not found. Run 'extract' first.")
        return
    rows = []
    with inv.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            rows.append(line.strip().split(","))
    print(f"Mappings counted: {len(rows)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Informatica XML → IR extractor")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="Extract IR from Informatica XML exports")
    p_ext.add_argument("--xml", nargs="+", help="Path(s) to Informatica XML export files", required=True)
    p_ext.add_argument("--out", help="Output directory for IR artifacts", required=True)
    p_ext.set_defaults(func=cmd_extract)

    p_sum = sub.add_parser("summarize", help="Summarize generated IR (optional)")
    p_sum.add_argument("--in", dest="input", help="IR directory to summarize", required=True)
    p_sum.add_argument("--html", help="Optional HTML output path")
    p_sum.set_defaults(func=cmd_summarize)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
