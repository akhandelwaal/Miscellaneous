import os
import time
import requests
from datetime import datetime, timezone
from typing import Dict, Generator, Iterable, List, Optional

SN_INSTANCE = os.getenv("SN_INSTANCE", "dev12345")
SN_USER = os.getenv("SN_USER", "api.user")
SN_PASSWORD = os.getenv("SN_PASSWORD", "supersecret")
BASE_URL = f"https://{SN_INSTANCE}.service-now.com/api/now/table/incident"

def _do_request(params: Dict[str, str], session: Optional[requests.Session] = None) -> Dict:
    sess = session or requests.Session()
    backoff = 1.2
    for attempt in range(6):
        r = sess.get(
            BASE_URL,
            params=params,
            auth=(SN_USER, SN_PASSWORD),
            headers={"Accept": "application/json"},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = max(backoff**attempt, float(r.headers.get("Retry-After", 0) or 0))
            time.sleep(wait)
            continue
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"ServiceNow error {r.status_code}: {detail}")
    raise RuntimeError("Exceeded retry limit contacting ServiceNow.")

# --- key difference lives here ---
def build_encoded_query_with_gs(since_dt: datetime, extra_query: Optional[str] = None) -> str:
    """
    Build sysparm_query using javascript:gs.dateGenerate('<YYYY-MM-DD>', '<HH:MM:SS>').

    Notes:
      - gs.dateGenerate() interprets the strings in the **instance's time zone**.
        If your instance is GMT/UTC (common), pass UTC values.
      - Below we convert the input to UTC first, then format the parts.
    """
    if since_dt.tzinfo is None:
        since_dt = since_dt.astimezone()  # assume local, make aware
    dt_utc = since_dt.astimezone(timezone.utc)
    d = dt_utc.strftime("%Y-%m-%d")
    t = dt_utc.strftime("%H:%M:%S")

    clause = f"sys_updated_on>javascript:gs.dateGenerate('{d}','{t}')"
    parts = [clause, "ORDERBYsys_updated_on"]
    if extra_query:
        parts.insert(0, extra_query.strip("^ "))
    return "^".join(parts)

def fetch_incidents_since(
    since_dt: datetime,
    *,
    extra_query: Optional[str] = None,
    fields: Optional[Iterable[str]] = None,
    page_size: int = 200,
    max_records: Optional[int] = None,
) -> Generator[Dict, None, None]:
    sysparm_query = build_encoded_query_with_gs(since_dt, extra_query)
    sysparm_fields = ",".join(fields) if fields else None

    offset = 0
    yielded = 0
    session = requests.Session()

    while True:
        params = {
            "sysparm_query": sysparm_query,
            "sysparm_limit": str(page_size),
            "sysparm_offset": str(offset),
            "sysparm_display_value": "false",
            "sysparm_exclude_reference_link": "true",
        }
        if sysparm_fields:
            params["sysparm_fields"] = sysparm_fields

        data = _do_request(params, session=session)
        batch: List[Dict] = data.get("result", []) or []
        if not batch:
            break

        for rec in batch:
            yield rec
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return

        offset += len(batch)
        if len(batch) < page_size:
            break

# --- example usage ---
if __name__ == "__main__":
    # Example: fetch incidents updated on/after Sept 1, 2025 00:00 *local* time.
    # We'll convert to UTC before passing to gs.dateGenerate (instance usually runs in UTC).
    since_local = datetime(2025, 9, 1, 0, 0, 0)  # naive -> treated as local in code above
    extra = "active=true^priorityIN1,2"  # optional

    fields = [
        "number", "sys_id", "short_description", "state", "priority",
        "assigned_to", "sys_updated_on", "opened_at", "caller_id",
    ]

    for inc in fetch_incidents_since(
        since_local,
        extra_query=extra,
        fields=fields,
        page_size=200,
    ):
        print(f"{inc.get('number')} | {inc.get('priority')} | {inc.get('sys_updated_on')} | {inc.get('short_description')}")
