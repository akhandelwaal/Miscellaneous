
#!/usr/bin/env python3
import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# --------------------------
# Heuristics & helpers
# --------------------------

ORACLE_FUNC_PATTERNS = [
    r"\bDECODE\s*\(", r"\bNVL2\s*\(", r"\bLISTAGG\s*\(", r"\bREGEXP_[A-Z_]+\s*\(",
    r"\bCONNECT_BY_[A-Z_]+\b", r"\bSYS_CONNECT_BY_PATH\s*\(",
    r"\bMODEL\b", r"\bPIVOT\b", r"\bUNPIVOT\b",
    r"\bRATIO_TO_REPORT\s*\(", r"\bLAG\s*\(", r"\bLEAD\s*\(", r"\bDENSE_RANK\s*\(",
    r"\bROW_NUMBER\s*\(", r"\bOVER\s*\("
]
ORACLE_FUNC_RE = re.compile("|".join(ORACLE_FUNC_PATTERNS), re.IGNORECASE)
IIF_RE = re.compile(r"\bIIF\s*\(", re.IGNORECASE)

def read_xml(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as e:
        sys.stderr.write(f"Failed to parse XML: {e}\n")
        sys.exit(2)

def attr(elem, name, default=None):
    return elem.attrib.get(name, default)

def yesno(flag):
    return "Yes" if flag else "No"

def normalize_pushdown(val):
    if not val:
        return "None"
    v = val.strip().lower()
    if "full" in v:
        return "Full"
    if "source" in v or "partial" in v:
        return "Partial"
    return "None"

def density_bucket(value):
    if value <= 0:
        return "None"
    if value <= 3:
        return "Some"
    return "Heavy"

# --------------------------
# Core extraction
# --------------------------

def extract_mappings(root):
    results = {}

    for mapping in root.findall(".//MAPPING"):
        mname = attr(mapping, "NAME")
        d = defaultdict(int)

        # Instances: sources/targets
        for inst in mapping.findall("./INSTANCE"):
            itype = (attr(inst, "TYPE", "") or "").upper()
            if itype == "SOURCE":
                d["Source_Count"] += 1
            elif itype == "TARGET":
                d["Target_Count"] += 1

        # Transformations & heuristics
        for tr in mapping.findall("./TRANSFORMATION"):
            d["Transformations_Count"] += 1
            ttype = (attr(tr, "TYPE", "") or "").lower()

            if ttype == "joiner":
                d["Join_Count"] += 1
            elif ttype in ("lookup", "lookup procedure"):
                d["Lookup_Count"] += 1
                # dynamic lookup flags
                for a in tr.findall("./TABLEATTRIBUTE"):
                    if "dynamic" in (attr(a, "NAME", "") or "").lower() and (attr(a, "VALUE", "") or "").lower() in ("yes","true"):
                        d["Dynamic_Lookup_Flag"] = 1
                for a in tr.findall("./ATTRIBUTE"):
                    if "dynamic" in (attr(a, "NAME", "") or "").lower() and (attr(a, "VALUE", "") or "").lower() in ("yes","true"):
                        d["Dynamic_Lookup_Flag"] = 1
            elif ttype == "aggregator":
                d["Aggregation_Count"] += 1
            elif ttype in ("union transformation","union"):
                d["Union_Branches"] += 1

            # expressions for function/IIF heuristics
            for exp in tr.findall(".//EXPRTBL/EXPR"):
                txt = (attr(exp, "VALUE") or "") + " " + (exp.text or "")
                if txt.strip():
                    d["Oracle_Func_Hits"] += len(ORACLE_FUNC_RE.findall(txt))
                    d["IIF_Hits"] += len(IIF_RE.findall(txt))
            for exp in tr.findall(".//EXPRESSION"):
                txt = (attr(exp, "VALUE") or "") + " " + (exp.text or "")
                if txt.strip():
                    d["Oracle_Func_Hits"] += len(ORACLE_FUNC_RE.findall(txt))
                    d["IIF_Hits"] += len(IIF_RE.findall(txt))

        # SQL Overrides in Source Qualifier
        for tr in mapping.findall("./TRANSFORMATION"):
            if (attr(tr, "TYPE", "") or "").lower() in ("source qualifier", "source qualifier transformation"):
                for a in tr.findall("./TABLEATTRIBUTE"):
                    an = (attr(a, "NAME", "") or "").lower()
                    av = (attr(a, "VALUE", "") or "").strip()
                    if an in ("sql query","user defined join","source filter") and av:
                        d["SQL_Override_Count"] += 1
                for a in tr.findall("./ATTRIBUTE"):
                    an = (attr(a, "NAME", "") or "").lower()
                    av = (attr(a, "VALUE", "") or "").strip()
                    if an in ("sql query","user defined join","source filter") and av:
                        d["SQL_Override_Count"] += 1

        d["Dynamic_Lookup"] = yesno(d.get("Dynamic_Lookup_Flag", 0) > 0)
        d["Multi_Target"] = yesno(d.get("Target_Count", 0) > 1)
        d["Oracle_Specific_Functions"] = density_bucket(d.get("Oracle_Func_Hits", 0))
        d["Macro_UDFs"] = density_bucket(d.get("IIF_Hits", 0))

        results[mname] = dict(d)

    return results

def extract_workflow_sessions(root):
    info = defaultdict(lambda: {"Workflow": "", "Session_Count": 0, "Pushdown_Optimization": "None", "Merge_Upsert": "No"})
    for wf in root.findall(".//WORKFLOW"):
        wf_name = attr(wf, "NAME", "")
        for sess in wf.findall(".//SESSION") + wf.findall(".//SESSIONEXT"):
            map_name = attr(sess, "MAPPINGNAME") or attr(sess, "MAPPINGNAMEVAR") or ""
            if not map_name:
                for a in sess.findall("./ATTRIBUTE"):
                    if (attr(a, "NAME", "") or "").lower() in ("mapping name","mapping"):
                        map_name = attr(a, "VALUE", "") or ""
                        break
            if not map_name:
                continue

            rec = info[map_name]
            rec["Workflow"] = wf_name or rec["Workflow"]
            rec["Session_Count"] += 1

            # Pushdown
            pdo = "None"
            for a in sess.findall("./ATTRIBUTE"):
                if "pushdown" in (attr(a, "NAME", "") or "").lower():
                    pdo = attr(a, "VALUE", "") or pdo
            for a in sess.findall("./CONFIGREFERENCE/ATTRIBUTE"):
                if "pushdown" in (attr(a, "NAME", "") or "").lower():
                    pdo = attr(a, "VALUE", "") or pdo
            rec["Pushdown_Optimization"] = normalize_pushdown(pdo) or rec["Pushdown_Optimization"]

            # Merge/Upsert heuristic
            merge_flag = False
            for a in sess.findall("./ATTRIBUTE"):
                nm = (attr(a, "NAME", "") or "").lower()
                val = (attr(a, "VALUE", "") or "").lower()
                if "treat source rows" in nm and ("update" in val or "insert" in val):
                    merge_flag = True
                if "update strategy" in nm and ("dd_update" in val or "dd_insert" in val):
                    merge_flag = True
                if "target load type" in nm and "merge" in val:
                    merge_flag = True
            rec["Merge_Upsert"] = "Yes" if merge_flag else rec["Merge_Upsert"]

    return info

CSV_COLUMNS = [
    "Mapping_ID","Workflow","Session_Count",
    "Source_Count","Target_Count","Transformations_Count",
    "Join_Count","Lookup_Count","Aggregation_Count","Union_Branches",
    "Custom_SQL","SQL_Override_Count","Dynamic_Lookup",
    "SCD_Type","Incremental_Strategy",
    "Data_Volume_GB","Refresh_Frequency","Dependencies_Count",
    "Reusable_Objects","Pre_Post_SQL","Error_Handling",
    "Pushdown_Optimization","Oracle_Specific_Functions",
    "Multi_Target","Merge_Upsert",
    "Parameterization_Level","Macro_UDFs","Orchestration_Complexity",
    "Tests_Required","Notes"
]

def assemble_rows(mappings, wfinfo):
    rows = []
    for mname, d in mappings.items():
        wf = wfinfo.get(mname, {})
        row = {
            "Mapping_ID": mname,
            "Workflow": wf.get("Workflow", ""),
            "Session_Count": wf.get("Session_Count", 0),
            "Source_Count": d.get("Source_Count", 0),
            "Target_Count": d.get("Target_Count", 0),
            "Transformations_Count": d.get("Transformations_Count", 0),
            "Join_Count": d.get("Join_Count", 0),
            "Lookup_Count": d.get("Lookup_Count", 0),
            "Aggregation_Count": d.get("Aggregation_Count", 0),
            "Union_Branches": d.get("Union_Branches", 0),
            "Custom_SQL": "",
            "SQL_Override_Count": d.get("SQL_Override_Count", 0),
            "Dynamic_Lookup": d.get("Dynamic_Lookup", ""),
            "SCD_Type": "",
            "Incremental_Strategy": "",
            "Data_Volume_GB": "",
            "Refresh_Frequency": "",
            "Dependencies_Count": "",
            "Reusable_Objects": "",
            "Pre_Post_SQL": "",
            "Error_Handling": "",
            "Pushdown_Optimization": wf.get("Pushdown_Optimization", "None"),
            "Oracle_Specific_Functions": d.get("Oracle_Specific_Functions", ""),
            "Multi_Target": d.get("Multi_Target", ""),
            "Merge_Upsert": wf.get("Merge_Upsert", "No"),
            "Parameterization_Level": "",
            "Macro_UDFs": d.get("Macro_UDFs", ""),
            "Orchestration_Complexity": "",
            "Tests_Required": "",
            "Notes": ""
        }
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser(description="Generate dbt migration complexity CSV from Informatica XML.")
    parser.add_argument("--xml", required=True, help="Path to Informatica export XML")
    parser.add_argument("--out", required=True, help="Path to write CSV")
    args = parser.parse_args()

    root = read_xml(args.xml)
    mappings = extract_mappings(root)
    wfinfo = extract_workflow_sessions(root)
    rows = assemble_rows(mappings, wfinfo)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
