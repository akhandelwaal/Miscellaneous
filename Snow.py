import os
import time
import requests
from datetime import datetime, timezone
from typing import Dict, Generator, Iterable, List, Optional

# -----------------------------
# Configuration (set these)
# -----------------------------
# You can export these in your shell instead of hardcoding:
#   export SN_INSTANCE="dev12345"            # just the subdomain (no .service-now.com)
#   export SN_USER="api.user"
#   export SN_PASSWORD="supersecret"
SN_INSTANCE = os.getenv("SN_INSTANCE", "dev12345")
SN_USER = os.getenv("SN_USER", "api.user")
SN_PASSWORD = os.getenv("SN_PASSWORD", "supersecret")

# API base URL
BASE_URL = f"https://{SN_INSTANCE}.service-now.com/api/now/table/incident"

# -----------------------------
# Helpers
# -----------------------------
def to_servicenow_ts(dt: datetime) -> str:
    """
    ServiceNow accepts 'YYYY-MM-DD HH:MM:SS' (UTC).
    If dt is naive, assume it's in local time and convert to UTC.
    """
    if dt.tzinfo is None:
        dt = dt.astimezone()  # assume local
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def build_encoded_query(since_dt: datetime, extra_query: Optional[str] = None) -> str:
    """
    Build a sysparm_query that returns incidents updated since 'since_dt' (UTC),
    ordered by sys_updated_on to keep pagination stable.
    You can append your own query parts (caret ^ combines clauses).
    """
    since_str = to_servicenow_ts(since_dt)
    parts = [f"sys_updated_on>={since_str}", "ORDERBYsys_updated_on"]
    if extra_query:
        parts.insert(0, extra_query.strip("^ "))  # prepend extra filters
    return "^".join(parts)

def _do_request(
    params: Dict[str, str],
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
    backoff_base: float = 1.2,
) -> Dict:
    """
    GET request with exponential backoff on 429/5xx.
    """
    sess = session or requests.Session()
    for attempt in range(max_retries):
        resp = sess.get(
            BASE_URL,
            params=params,
            auth=(SN_USER, SN_PASSWORD),
            headers={"Accept": "application/json"},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            # Rate-limited or transient: backoff
            sleep_s = (backoff_base ** attempt)
            # Respect Retry-After if provided
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_s = max(sleep_s, float(retry_after))
            time.sleep(sleep_s)
            continue
        # Hard error: raise with details
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"ServiceNow error {resp.status_code}: {detail}")
    raise RuntimeError("Exceeded retry limit contacting ServiceNow.")

# -----------------------------
# Main fetcher
# -----------------------------
def fetch_incidents_since(
    since_dt: datetime,
    *,
    extra_query: Optional[str] = None,
    fields: Optional[Iterable[str]] = None,
    page_size: int = 100,           # 1..10000 depending on instance settings
    max_records: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Yields incident records updated on/after 'since_dt'.

    Arguments:
      - since_dt: datetime (aware or naive). Converted to UTC for the query.
      - extra_query: Optional encoded filter (e.g., "state!=7^priority=1").
      - fields: Optional list of fields to return (reduces payload).
      - page_size: Page size for pagination (respect your instance limits).
      - max_records: If set, stop after yielding this many records.

    Notes:
      - Uses sysparm_offset pagination with ORDERBYsys_updated_on for stable paging.
      - You can also filter on created time by changing to 'sys_created_on>='.
    """
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    sysparm_query = build_encoded_query(since_dt, extra_query)
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

        # Advance to next page
        offset += len(batch)
        # Guard: if fewer than page_size were returned, we’re done
        if len(batch) < page_size:
            break

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: fetch incidents updated on/after Sept 1, 2025 00:00 IST
    # Convert your local time to a timezone-aware datetime first if needed.
    # Here we build a naive datetime (assumed local) for simplicity:
    since_local = datetime(2025, 9, 1, 0, 0, 0)  # your local time
    # Optional: narrow down further, e.g. only active P1/P2
    extra = "(active=true^priorityIN1,2)"

    wanted_fields = [
        "number",
        "sys_id",
        "short_description",
        "state",
        "priority",
        "assigned_to",
        "sys_updated_on",
        "opened_at",
        "caller_id",
    ]

    count = 0
    for incident in fetch_incidents_since(
        since_local,
        extra_query=extra,
        fields=wanted_fields,
        page_size=200,
        max_records=None,  # or set a cap like 5000
    ):
        print(f"{incident.get('number')} | {incident.get('priority')} | {incident.get('sys_updated_on')} | {incident.get('short_description')}")
        count += 1

    print(f"\nTotal incidents fetched: {count}")
