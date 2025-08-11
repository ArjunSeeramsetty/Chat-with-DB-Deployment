import asyncio
import sys

sys.path.append('.')

from backend.api.routes import ask_question_enhanced  # type: ignore
from backend.core.types import QueryRequest  # type: ignore


async def main():
    q = sys.argv[1] if len(sys.argv) > 1 else 'what is the monthly total outage of Northern Region in 2025?'
    req = QueryRequest(question=q, user_id='diag', session_id='sess-diag')
    res = await ask_question_enhanced(req)
    sql = res.get('sql_query') or res.get('sql') or ''
    print('Q:', q)
    print('success:', res.get('success'))
    print('error:', res.get('error'))
    print('sql:', sql)
    data = res.get('data') or []
    print('rows:', len(data))
    if len(data) > 0:
        print('first_row:', data[0])

if __name__ == '__main__':
    asyncio.run(main())


