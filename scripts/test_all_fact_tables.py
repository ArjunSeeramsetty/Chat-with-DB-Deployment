import asyncio
import time
from typing import List, Dict, Any

import sys
sys.path.append('.')

from backend.api.routes import ask_question_enhanced  # type: ignore
from backend.core.types import QueryRequest  # type: ignore


class TestCase:
    def __init__(self, case_id: str, query: str, required_sql_parts: List[str], optional_sql_parts: List[str] = None,
                 forbidden_sql_parts: List[str] = None, min_rows: int = 0, description: str = "",
                 category: str = "semantic"):
        self.case_id = case_id
        self.query = query
        self.required_sql_parts = [p.lower() for p in (required_sql_parts or [])]
        self.optional_sql_parts = [p.lower() for p in (optional_sql_parts or [])]
        self.forbidden_sql_parts = [p.lower() for p in (forbidden_sql_parts or [])]
        self.min_rows = min_rows
        self.description = description
        self.category = category


def build_test_cases() -> List[TestCase]:
    tests: List[TestCase] = []

    # FactAllIndiaDailySummary - region monthly shortage
    tests.append(TestCase(
        case_id="FAIDS_monthly_shortage_region",
        query="what is the monthly energy shortage of Northern Region in 2025?",
        required_sql_parts=[
            "from factallindiadailysummary",
            "join dimregions",
            "join dimdates",
            "strftime('%y-%m', d.actualdate)".replace("%y", "%Y"),
            "energyshortage",
            "r.regionname = 'northern region'",
        ],
        min_rows=1,
        description="Region-level monthly shortage with correct joins",
    ))

    # FactAllIndiaDailySummary - outage monthly
    tests.append(TestCase(
        case_id="FAIDS_monthly_outage_region",
        query="what is the monthly total outage of Northern Region in 2025?",
        required_sql_parts=[
            "from factallindiadailysummary",
            "join dimregions",
            "join dimdates",
            "strftime('%y-%m', d.actualdate)".replace("%y", "%Y"),
            "outage",
            "r.regionname = 'northern region'",
        ],
        min_rows=1,
        description="Outage columns detection and monthly grouping",
    ))

    # Outage monthly - all regions central sector only
    tests.append(TestCase(
        case_id="FAIDS_all_regions_monthly_outage_central",
        query="What is the monthly central sector outage of all regions in 2024?",
        required_sql_parts=[
            "from factallindiadailysummary",
            "centralsectoroutage",
            "join dimregions",
            "join dimdates",
            "strftime('%y-%m', d.actualdate)".replace("%y", "%Y"),
            "group by r.regionname",
        ],
        min_rows=0,
        description="All regions central outage monthly",
    ))

    # FactAllIndiaDailySummary - all regions monthly shortage (group by RegionName)
    tests.append(TestCase(
        case_id="FAIDS_all_regions_monthly_shortage",
        query="what is the monthly energy shortage of all regions in 2025?",
        required_sql_parts=[
            "from factallindiadailysummary",
            "join dimregions",
            "join dimdates",
            "group by r.regionname",
        ],
        min_rows=1,
        description="All regions grouped with monthly",
    ))

    # FactStateDailyEnergy - state monthly consumption
    tests.append(TestCase(
        case_id="FSDE_monthly_consumption_state",
        query="what is the monthly energy consumption of Andhra Pradesh in 2025?",
        required_sql_parts=[
            "from factstatedailyenergy",
            "join dimstates",
            "join dimdates",
            "s.statename = 'andhra pradesh'",
            "energymet",
        ],
        min_rows=1,
        description="State table selection and correct joins",
    ))

    # FactStateDailyEnergy - all states monthly shortage (group by StateName)
    tests.append(TestCase(
        case_id="FSDE_all_states_monthly_shortage",
        query="what is the monthly energy shortage of all states in 2025?",
        required_sql_parts=[
            "from factstatedailyenergy",
            "join dimstates",
            "join dimdates",
            "group by s.statename",
        ],
        min_rows=1,
        description="All states grouped with monthly",
    ))

    # FactStateDailyEnergy + DimRegions - states in a region monthly
    tests.append(TestCase(
        case_id="FSDE_states_in_region_monthly",
        query="what is the monthly energy met of all states in southern region in 2025?",
        required_sql_parts=[
            "from factstatedailyenergy",
            "join dimstates",
            "join dimregions",
            "join dimdates",
            "r.regionname = 'southern region'",
            "strftime('%y-%m', d.actualdate)".replace("%y", "%Y"),
            "group by s.statename",
        ],
        min_rows=0,
        description="States filtered by region and grouped monthly",
    ))

    # FactDailyGenerationBreakdown - monthly generation by source
    tests.append(TestCase(
        case_id="FDGB_monthly_generation_by_source",
        query="what is the monthly generation by source in 2025?",
        required_sql_parts=[
            "from factdailygenerationbreakdown",
            "join dimdates",
            "strftime('%y-%m', d.actualdate)".replace("%y", "%Y"),
            "group by dgs.sourcename",
        ],
        min_rows=0,
        description="Generation by source monthly grouping",
    ))

    # FactDailyGenerationBreakdown - generation by source
    tests.append(TestCase(
        case_id="FDGB_generation_by_source",
        query="What is the renewable generation by source in 2025?",
        required_sql_parts=[
            "from factdailygenerationbreakdown",
            "generationamount",
        ],
        min_rows=0,  # allow 0 if table not populated
        description="Generation breakdown table coverage",
    ))

    # FactCountryDailyExchange - international exchange for Bangladesh
    tests.append(TestCase(
        case_id="FCDE_exchange_bangladesh",
        query="Show international power exchange with Bangladesh in 2024",
        required_sql_parts=[
            "from factcountrydailyexchange",
            "join dimcountries",
            "join dimdates",
            "bangladesh",
        ],
        min_rows=0,
        description="Country exchange detection and joins",
    ))

    # FactInternationalTransmissionLinkFlow - explicit table coverage
    tests.append(TestCase(
        case_id="FITLF_explicit_table",
        query="Show total flow from FactInternationalTransmissionLinkFlow in 2024",
        required_sql_parts=[
            "from factinternationaltransmissionlinkflow",
        ],
        min_rows=0,
        description="Explicit table coverage: international transmission link flow",
        category="explicit_table",
    ))

    # FactTransmissionLinkFlow - explicit table coverage
    tests.append(TestCase(
        case_id="FTLF_explicit_table",
        query="Show total flow from FactTransmissionLinkFlow in 2024",
        required_sql_parts=[
            "from facttransmissionlinkflow",
        ],
        min_rows=0,
        description="Explicit table coverage: transmission link flow",
        category="explicit_table",
    ))

    # FactTimeBlockPowerData - explicit table coverage
    tests.append(TestCase(
        case_id="FTBPD_explicit_table",
        query="Show hourly data from FactTimeBlockPowerData for 2025-01-01",
        required_sql_parts=[
            "from facttimeblockpowerdata",
        ],
        min_rows=0,
        description="Explicit table coverage: time-block power data",
        category="explicit_table",
    ))

    # FactTimeBlockGeneration - explicit table coverage
    tests.append(TestCase(
        case_id="FTBG_explicit_table",
        query="Show hourly generation from FactTimeBlockGeneration for 2025-01-01",
        required_sql_parts=[
            "from facttimeblockgeneration",
        ],
        min_rows=0,
        description="Explicit table coverage: time-block generation",
        category="explicit_table",
    ))

    return tests


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
    # Verify required parts
    required_ok = all(part in sql_lower for part in tc.required_sql_parts)
    forbidden_ok = all(part not in sql_lower for part in tc.forbidden_sql_parts) if tc.forbidden_sql_parts else True
    row_ok = (len(data) >= tc.min_rows)

    passed = success and required_ok and forbidden_ok and row_ok

    return {
        "id": tc.case_id,
        "category": tc.category,
        "description": tc.description,
        "passed": passed,
        "success": success,
        "required_ok": required_ok,
        "forbidden_ok": forbidden_ok,
        "row_ok": row_ok,
        "rows": len(data),
        "duration_ms": int((t1 - t0) * 1000),
        "sql": sql,
        "error": error,
        "candidate": res.get("selected_candidate_source"),
    }


async def main():
    tests = build_test_cases()
    results: List[Dict[str, Any]] = []

    # Run sequentially to avoid shared singletons re-init overhead
    for tc in tests:
        res = await run_test_case(tc)
        results.append(res)
        # Print brief line per test
        status = "PASS" if res["passed"] else "FAIL"
        cand = res.get("candidate") or "?"
        print(f"[{status}] {res['id']} ({res['duration_ms']} ms) rows={res['rows']} success={res['success']} candidate={cand}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    avg_ms = int(sum(r['duration_ms'] for r in results) / total) if total else 0
    slowest = max(results, key=lambda r: r['duration_ms']) if results else None

    print("\nSummary:")
    print(f"- Total: {total}")
    print(f"- Passed: {passed}")
    print(f"- Failed: {total - passed}")
    print(f"- Avg Duration: {avg_ms} ms")
    if slowest:
        print(f"- Slowest: {slowest['id']} at {slowest['duration_ms']} ms")

    # Detailed failures
    failures = [r for r in results if not r['passed']]
    if failures:
        print("\nFailures (details):")
        for r in failures:
            print(f"- {r['id']}: success={r['success']} required_ok={r['required_ok']} forbidden_ok={r['forbidden_ok']} row_ok={r['row_ok']} err={r['error']}")
            # Print first 200 chars of SQL
            sql = (r.get('sql') or '')
            if sql:
                print("  SQL:", sql[:200].replace('\n', ' '))


if __name__ == '__main__':
    asyncio.run(main())


