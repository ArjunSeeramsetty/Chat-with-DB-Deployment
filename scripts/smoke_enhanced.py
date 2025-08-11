import asyncio
import sys

sys.path.append('.')

from backend.api.routes import ask_question_enhanced  # type: ignore
from backend.core.types import QueryRequest  # type: ignore


async def run_query(q: str) -> None:
    req = QueryRequest(question=q, user_id='smoke', session_id='sess-smoke')
    res = await ask_question_enhanced(req)
    sql = res.get('sql_query') or res.get('sql') or ''
    print('Q:', q)
    print('success:', res.get('success'))
    print('error:', res.get('error'))
    print('sql:', (sql or '')[:600])
    data = res.get('data') or []
    print('rows:', len(data))
    print('-' * 80)


async def main() -> None:
    tests = [
        'what is the monthly energy shortage Southern Region in 2025?',
        'what is the monthly energy shortage of all regions in 2025?',
        'what is the monthly energy shortage of all states in 2025?',
        'what is the monthly average energy demand of Northern Region in 2025?',
        'what is the monthly total outage of Northern Region in 2025?',
    ]
    for q in tests:
        await run_query(q)


if __name__ == '__main__':
    asyncio.run(main())


