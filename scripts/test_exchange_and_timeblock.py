import asyncio
import time
from typing import List, Dict, Any

import sys
sys.path.append('.')

from backend.api.routes import ask_question_enhanced  # type: ignore
from backend.core.types import QueryRequest  # type: ignore


class TestCase:
    def __init__(self, case_id: str, query: str, required_sql_parts: List[str], min_rows: int = 0, description: str = ""):
        self.case_id = case_id
        self.query = query
        self.required_sql_parts = [p.lower() for p in (required_sql_parts or [])]
        self.min_rows = min_rows
        self.description = description


TESTS: List[TestCase] = [
    # Country exchange monthly with country filtering
    TestCase(
        case_id="FCDE_monthly_exchange_bangladesh",
        query="What is the monthly power exchange with Bangladesh in 2024?",
        required_sql_parts=[
            "from factcountrydailyexchange",
            "join dimcountries",
            "join dimdates",
            "strftime('%y-%m', d.actualdate)".replace("%y", "%Y"),
        ],
        min_rows=0,
        description="Country exchange monthly grouping",
    ),
    # International link flow with intrinsic time column (no DimDates join)
    TestCase(
        case_id="FITLF_total_flow_2024",
        query="What is the line-wise total energy exchanged in 2024",
        required_sql_parts=[
            "from factinternationaltransmissionlinkflow",
        ],
        min_rows=0,
        description="International link flow using intrinsic time column",
    ),
    # Transmission link flow monthly
    TestCase(
        case_id="FTLF_monthly_flow_2025",
        query="What is the monthly transmission link flow in 2025",
        required_sql_parts=[
            "from facttransmissionlinkflow",
        ],
        min_rows=0,
        description="Transmission link flow monthly",
    ),
    # Time-block power hourly/day series
    TestCase(
        case_id="FTBPD_hourly_2025_01_01",
        query="What is the hourly power on 2025-01-01 from time block data",
        required_sql_parts=[
            "from facttimeblockpowerdata",
        ],
        min_rows=0,
        description="Time-block power hourly",
    ),
    TestCase(
        case_id="FTBG_hourly_2025_01_01",
        query="Show hourly generation on 2025-01-01 from time block generation",
        required_sql_parts=[
            "from facttimeblockgeneration",
        ],
        min_rows=0,
        description="Time-block generation hourly",
    ),
]


async def run_test_case(tc: TestCase) -> Dict[str, Any]:
    t0 = time.perf_counter()
    req = QueryRequest(question=tc.query, user_id='tester', session_id='sess-tests')
    res = await ask_question_enhanced(req)
    t1 = time.perf_counter()

    sql = (res.get('sql_query') or res.get('sql') or '')
    data = res.get('data') or []
    success = bool(res.get('success')) and bool(sql)
    error = res.get('error')

    sql_lower = sql.lower()
    required_ok = all(part in sql_lower for part in tc.required_sql_parts)
    row_ok = (len(data) >= tc.min_rows)

    return {
        "id": tc.case_id,
        "passed": success and required_ok and row_ok,
        "success": success,
        "required_ok": required_ok,
        "row_ok": row_ok,
        "rows": len(data),
        "duration_ms": int((t1 - t0) * 1000),
        "sql": sql,
        "error": error,
        "candidate": res.get("selected_candidate_source"),
    }


async def main():
    results: List[Dict[str, Any]] = []
    for tc in TESTS:
        r = await run_test_case(tc)
        results.append(r)
        status = "PASS" if r['passed'] else "FAIL"
        print(f"[{status}] {r['id']} ({r['duration_ms']} ms) rows={r['rows']} candidate={r.get('candidate')}")
        if r.get('sql'):
            print(f"SQL ({r['id']}): {r['sql']}")

    print("\nSummary:")
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    print(f"- Total: {total}\n- Passed: {passed}\n- Failed: {total - passed}")

    failures = [r for r in results if not r['passed']]
    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"- {r['id']} success={r['success']} req_ok={r['required_ok']} rows={r['rows']} error={r['error']}")
            if r.get('sql'):
                print("  SQL:", r['sql'])


if __name__ == '__main__':
    asyncio.run(main())


