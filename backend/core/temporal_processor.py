from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import re


@dataclass
class TemporalConstraints:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    relative_period: Optional[str] = None

    def has_any(self) -> bool:
        return any([self.year, self.month, self.day, self.relative_period])

    def to_where_clauses(self, date_alias: str = "dt") -> List[str]:
        clauses: List[str] = []
        if self.year is not None:
            clauses.append(f"{date_alias}.Year = {int(self.year)}")
        if self.month is not None:
            clauses.append(f"{date_alias}.Month = {int(self.month)}")
        if self.day is not None:
            clauses.append(f"{date_alias}.DayOfMonth = {int(self.day)}")
        return clauses


class TemporalConstraintProcessor:
    def __init__(self, mdl_temporal_config: dict | None = None):
        self.mdl_temporal_config = mdl_temporal_config or {}
        self.month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

    # -------------------- Extraction --------------------
    def extract_temporal_constraints(self, query: str) -> TemporalConstraints:
        ql = (query or "").lower()
        cons = TemporalConstraints()

        # YYYY
        year_match = re.search(r"\b(19|20)\d{2}\b", ql)
        if year_match:
            cons.year = int(year_match.group(0))

        # Month name + optional year
        for m_name, m_num in self.month_names.items():
            if re.search(rf"\b{m_name}\b", ql):
                cons.month = m_num
                # Try to fetch year after month
                my = re.search(rf"\b{m_name}\b\s+(\d{{4}})", ql)
                if my and not cons.year:
                    cons.year = int(my.group(1))
                break

        # Day formats: "15th June 2025" | "June 15, 2025" | "15 June 2025"
        # 1) 15th June 2025
        dmy = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", ql)
        if dmy:
            cons.day = int(dmy.group(1))
            cons.month = self.month_names.get(dmy.group(3), cons.month)
            cons.year = int(dmy.group(4))

        # 2) June 15, 2024
        mdy = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(st|nd|rd|th)?[,\s]+(\d{4})\b", ql)
        if mdy:
            cons.month = self.month_names.get(mdy.group(1), cons.month)
            cons.day = int(mdy.group(2))
            cons.year = int(mdy.group(4))

        return cons

    # -------------------- Application --------------------
    def _find_date_alias(self, sql: str) -> Optional[str]:
        # Prefer explicit DimDates alias
        m = re.search(r"JOIN\s+DimDates\s+(\w+)\s+ON", sql, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        # Fall back: if we see d.ActualDate later, use 'd'
        if 'd.ActualDate' in sql:
            return 'd'
        if 'dt.ActualDate' in sql:
            return 'dt'
        return None

    def _find_time_column(self, sql: str) -> Optional[str]:
        # Try common patterns
        for col in ['d.ActualDate', 'dt.ActualDate', 'fs.Timestamp', 'fs.TimeBlock']:
            if col in sql:
                return col
        # Try to detect any alias with ActualDate
        m = re.search(r"(\w+)\.ActualDate", sql)
        if m:
            return f"{m.group(1)}.ActualDate"
        return None

    def _has_month_filter(self, sql: str) -> bool:
        return bool(re.search(r"\b(dt|d)\.Month\s*=\s*\d+", sql, flags=re.IGNORECASE) or
                    re.search(r"strftime\('\%m',\s*[^)]+\)\s*=\s*'\d{2}'", sql, flags=re.IGNORECASE))

    def _has_year_filter(self, sql: str) -> bool:
        return bool(re.search(r"\b(dt|d)\.Year\s*=\s*\d{4}", sql, flags=re.IGNORECASE) or
                    re.search(r"strftime\('\%Y',\s*[^)]+\)\s*=\s*'\d{4}'", sql, flags=re.IGNORECASE))

    def _has_day_filter(self, sql: str) -> bool:
        return bool(re.search(r"\b(dt|d)\.DayOfMonth\s*=\s*\d{1,2}", sql, flags=re.IGNORECASE) or
                    re.search(r"date\(\s*[^)]+\)\s*=\s*'\d{4}-\d{2}-\d{2}'", sql, flags=re.IGNORECASE))

    def _append_where(self, sql: str, clauses: List[str]) -> str:
        if not clauses:
            return sql
        if re.search(r"\bWHERE\b", sql, flags=re.IGNORECASE):
            return re.sub(r"\bWHERE\b", "WHERE " + " AND ".join(clauses) + " AND", sql, count=1, flags=re.IGNORECASE)
        # If no WHERE, append one
        return sql + " WHERE " + " AND ".join(clauses)

    def apply_temporal_constraints(self, sql: str, constraints: TemporalConstraints) -> str:
        if not constraints or not constraints.has_any():
            return sql

        date_alias = self._find_date_alias(sql)
        time_col = self._find_time_column(sql)

        # Prefer DimDates alias style when available
        if date_alias:
            clauses = []
            if constraints.year is not None and not self._has_year_filter(sql):
                clauses.append(f"{date_alias}.Year = {int(constraints.year)}")
            if constraints.month is not None and not self._has_month_filter(sql):
                clauses.append(f"{date_alias}.Month = {int(constraints.month)}")
            if constraints.day is not None and not self._has_day_filter(sql):
                clauses.append(f"{date_alias}.DayOfMonth = {int(constraints.day)}")
            return self._append_where(sql, clauses)

        # Otherwise use strftime/date on a detected time column
        if time_col:
            clauses = []
            if constraints.year is not None and not self._has_year_filter(sql):
                clauses.append(f"strftime('%Y', {time_col}) = '{int(constraints.year)}'")
            if constraints.month is not None and not self._has_month_filter(sql):
                clauses.append(f"strftime('%m', {time_col}) = '{int(constraints.month):02d}'")
            if constraints.day is not None and not self._has_day_filter(sql) and constraints.month is not None and constraints.year is not None:
                clauses.append(f"date({time_col}) = '{int(constraints.year)}-{int(constraints.month):02d}-{int(constraints.day):02d}'")
            return self._append_where(sql, clauses)

        # If we cannot find time alias/column, return as-is
        return sql
