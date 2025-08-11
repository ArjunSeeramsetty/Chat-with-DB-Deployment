from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class DataAnalysis:
    def __init__(self,
                 is_time_series: bool,
                 is_multi_dimensional: bool,
                 time_field: Optional[str],
                 category_field: Optional[str],
                 numeric_fields: List[str]):
        self.is_time_series = is_time_series
        self.is_multi_dimensional = is_multi_dimensional
        self.time_field = time_field
        self.category_field = category_field
        self.numeric_fields = numeric_fields


class TemporalContext:
    def __init__(self, granularity: Optional[str], time_column: Optional[str]):
        self.granularity = granularity
        self.time_column = time_column


class WrenAIVisualizationService:
    """Lightweight chart intelligence aligned with Wren AI/Vega-Lite approach.

    Generates a vega-lite spec when possible and also includes a simple
    fallback (chartType/options) for current UI compatibility.
    """

    def __init__(self, mdl_context: Optional[Dict[str, Any]] = None):
        self.mdl_context = mdl_context or {}

    # -------------------- Public API --------------------
    def generate_visualization(self, data: List[Dict[str, Any]], query: str, sql: str) -> Optional[Dict[str, Any]]:
        if not data:
            return None

        analysis = self._analyze_data_structure(data)
        temporal = self._detect_temporal_patterns(data, sql, analysis)

        # Example: multi-line temporal for "monthly energy met of all states"
        if analysis.is_time_series and analysis.is_multi_dimensional:
            if ("state" in (query or "").lower()) and (temporal.granularity in {"monthly", "month", "m"}):
                spec = self._vega_multi_line_temporal(data, analysis, temporal)
                return {
                    "chartType": "vega_lite",
                    "vegaSpec": spec,
                    # Fallback for current UI
                    "options": {
                        "xField": analysis.time_field or temporal.time_column or "Month",
                        "yField": (analysis.numeric_fields[0] if analysis.numeric_fields else "Value"),
                        "seriesField": analysis.category_field or "StateName",
                    },
                }

        # Generic recommendation: if time series single-metric → line; else bar
        if analysis.is_time_series:
            spec = self._vega_single_line(data, analysis, temporal)
            return {
                "chartType": "vega_lite",
                "vegaSpec": spec,
                "options": {
                    "xField": analysis.time_field or temporal.time_column or "Month",
                    "yField": (analysis.numeric_fields[0] if analysis.numeric_fields else "Value"),
                },
            }

        # Categorical bar fallback
        spec = self._vega_simple_bar(data, analysis)
        return {
            "chartType": "vega_lite",
            "vegaSpec": spec,
            "options": {
                "xField": analysis.category_field or "Category",
                "yField": (analysis.numeric_fields[0] if analysis.numeric_fields else "Value"),
            },
        }

    # -------------------- Heuristics --------------------
    def _analyze_data_structure(self, rows: List[Dict[str, Any]]) -> DataAnalysis:
        if not rows:
            return DataAnalysis(False, False, None, None, [])

        sample = rows[0]
        fields = list(sample.keys())

        # Detect likely time and category fields
        time_candidates = [
            f for f in fields
            if f.lower() in {"month", "date", "actualdate", "day", "year"}
            or "date" in f.lower()
        ]
        time_field = time_candidates[0] if time_candidates else None

        category_candidates = [
            f for f in fields
            if any(k in f.lower() for k in ["state", "region", "source", "name", "category"]) and f != time_field
        ]
        category_field = category_candidates[0] if category_candidates else None

        numeric_fields = [
            f for f in fields
            if isinstance(sample.get(f), (int, float))
        ]

        # Multi-dimensional if both category and time are present
        is_time_series = time_field is not None
        is_multi_dimensional = is_time_series and (category_field is not None)

        return DataAnalysis(is_time_series, is_multi_dimensional, time_field, category_field, numeric_fields)

    def _detect_temporal_patterns(self, rows: List[Dict[str, Any]], sql: str, analysis: DataAnalysis) -> TemporalContext:
        # Granularity inference by fields / SQL content
        gran = None
        if analysis.time_field:
            lf = analysis.time_field.lower()
            if "month" in lf:
                gran = "monthly"
            elif "day" in lf:
                gran = "daily"
            elif "year" in lf:
                gran = "yearly"

        s = (sql or "").lower()
        if not gran:
            if "strftime('%y'" in s or ".year" in s:
                gran = "yearly"
            elif "strftime('%m'" in s or ".month" in s:
                gran = "monthly"
            elif "dayofmonth" in s or ".dayofmonth" in s:
                gran = "daily"

        return TemporalContext(granularity=gran, time_column=analysis.time_field)

    # -------------------- Vega-Lite generators --------------------
    def _vega_multi_line_temporal(self, rows: List[Dict[str, Any]], analysis: DataAnalysis, temporal: TemporalContext) -> Dict[str, Any]:
        x_field = analysis.time_field or temporal.time_column or "Month"
        y_field = (analysis.numeric_fields[0] if analysis.numeric_fields else "Value")
        color_field = analysis.category_field or "StateName"

        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": rows},
            "mark": {"type": "line", "point": True, "strokeWidth": 2},
            "encoding": {
                "x": {"field": x_field, "type": "temporal" if "date" in x_field.lower() or x_field.lower() in {"month", "day", "year"} else "ordinal"},
                "y": {"field": y_field, "type": "quantitative"},
                "color": {"field": color_field, "type": "nominal"},
                "tooltip": [color_field, x_field, y_field],
            },
        }

    def _vega_single_line(self, rows: List[Dict[str, Any]], analysis: DataAnalysis, temporal: TemporalContext) -> Dict[str, Any]:
        x_field = analysis.time_field or temporal.time_column or "Month"
        y_field = (analysis.numeric_fields[0] if analysis.numeric_fields else "Value")
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": rows},
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {"field": x_field, "type": "temporal" if "date" in x_field.lower() or x_field.lower() in {"month", "day", "year"} else "ordinal"},
                "y": {"field": y_field, "type": "quantitative"},
                "tooltip": [x_field, y_field],
            },
        }

    def _vega_simple_bar(self, rows: List[Dict[str, Any]], analysis: DataAnalysis) -> Dict[str, Any]:
        x_field = analysis.category_field or "Category"
        y_field = (analysis.numeric_fields[0] if analysis.numeric_fields else "Value")
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": rows},
            "mark": "bar",
            "encoding": {
                "x": {"field": x_field, "type": "nominal"},
                "y": {"field": y_field, "type": "quantitative"},
                "tooltip": [x_field, y_field],
            },
        }


