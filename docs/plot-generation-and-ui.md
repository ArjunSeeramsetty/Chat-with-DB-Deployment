## Plot Generation and Rendering – Current Application

This document explains how the application generates chart/plot recommendations in the backend and how the React UI renders them, across each workflow path.

### 1) End-to-end data flow
- User query → FastAPI endpoint (`/api/v1/ask` / `/api/v1/ask-enhanced` / `/api/v1/ask-fixed` / `/api/v1/ask-agentic`).
- Backend processes query (Enhanced unified pipeline, Semantic-first/Hybrid/Traditional fallback, Agentic workflow).
- Backend executes the selected SQL and returns `{ sql, data, plot, ... }`.
- Frontend consumes `plot` and renders a chart accordingly.

### 2) Backend plot generation

#### 2.1 Enhanced unified pipeline (primary)
- File: `backend/services/enhanced_rag_service.py`
- Method: `_run_unified_enhanced_pipeline`
- Step C (Visualization) after a candidate SQL is chosen and successfully executed:
  - Instantiates `VisualizationAgent` from `backend/core/agentic_framework.py`.
  - Calls `viz_agent._generate_visualization(executed.data, query)`.
  - Normalizes output into a `plot` object:
    - `plot = { "chartType": viz_dict.get("chart_type", "bar"), "options": viz_dict.get("config", {}) }`.
- The `plot` object is attached to the pipeline result and returned to the API layer.

Behavior:
- If execution returns rows, the visualization agent proposes a chart type based on detected shape of the result (e.g., time-series → line, categories → bar).
- If data is empty or something fails, `plot` is omitted or set to `None`.

#### 2.2 Semantic-first / Hybrid / Traditional fallbacks
- File: `backend/services/enhanced_rag_service.py`
- When the unified pipeline fails and the service falls back to other processing modes, those paths can still return a `plot` in two ways:
  - If they route through `_run_unified_enhanced_pipeline` (preferred), you get the same visualization logic as above.
  - If they return a direct result (e.g., `_process_traditional`), no visualization is automatically created there; the API layer in `routes.py` does not synthesize a plot itself. As a result, `plot` may be `None` for pure traditional paths.

Notes:
- The codebase aims to rely on the unified pipeline first, so plots are commonly available when the enhanced approach succeeds.

#### 2.3 Agentic workflow
- Endpoint: `/api/v1/ask-agentic` (see `backend/api/routes.py`).
- Agent stack defined in `backend/core/agentic_framework.py` includes a `VisualizationAgent` used to produce chart recommendations similarly to the enhanced pipeline.
- The endpoint normalizes response to include `query_response.sql_query` and attaches `plot` if present in the agentic result.

### 3) API response shape (frontend contract)
- File: `backend/api/routes.py`, function `ask_question_enhanced`.
- Normalized payload includes:
  - `success: boolean`
  - `error?: string`
  - `sql: string` (also mirrored as `sql_query` for compatibility)
  - `data: Array<Record<string, any>>`
  - `plot?: { chartType: string; options: Record<string, any> }`
  - `processing_mode, selected_candidate_source, confidence, ...`

Example (trimmed):
```json
{
  "success": true,
  "sql": "SELECT ...",
  "data": [ {"Month":"2025-06","Value":123.4}, ... ],
  "plot": {
    "chartType": "line",
    "options": { "xField": "Month", "yField": "Value", "seriesField": null }
  }
}
```

### 4) Frontend rendering

#### 4.1 API access
- File: `agent-ui/src/hooks/useQueryService.js`
- Uses global `state.selectedEndpoint` to determine which backend endpoint to call.
- Returns the parsed response including `plot`.

#### 4.2 UI components
- File: `agent-ui/src/App.js`
- After a successful query, the UI reads `response.plot`.
- If `plot` exists, it renders a chart component using `plot.chartType` and `plot.options`.
- If no `plot` exists, the UI just renders the table (derived from `data`) or the SQL preview.

Notes:
- The chart renderer expects a generic schema: `chartType` (bar|line|area|dualAxis|pie...) and an `options` object with fields like `xField`, `yField`, `seriesField`, `data`, axes, and color/theme overrides.
- The `options` structure is intentionally backend-driven, so the UI only needs to map chart type to a lightweight chart wrapper.

### 5) Workflow-by-workflow summary

- **Enhanced unified (preferred)**
  - Visualization: YES (VisualizationAgent)
  - Plot returned: Usually YES
  - Where: `enhanced_rag_service._run_unified_enhanced_pipeline` Step C

- **Semantic-first**
  - Visualization: If routed through unified scoring or if explicit call to VisualizationAgent is made.
  - Plot returned: Often YES, else NULL

- **Hybrid**
  - Visualization: Same as semantic-first; depends on whether result goes through unified scoring
  - Plot returned: Often YES, else NULL

- **Traditional**
  - Visualization: Not automatically created in `_process_traditional`
  - Plot returned: Often NULL

- **Agentic workflow**
  - Visualization: YES (VisualizationAgent inside agent pipeline)
  - Plot returned: Usually YES
  - Where: `agentic_framework.VisualizationAgent` used by `/api/v1/ask-agentic`

### 6) Practical guidelines
- Prefer Enhanced endpoint: more reliable charts due to unified pipeline.
- Ensure `VisualizationAgent` produces a compact `options` payload; the UI is a thin renderer.
- If a path must use traditional assembler, consider adding a small visualization call in that path before returning.
- Keep the `plot` schema stable across endpoints to simplify the UI.

### 7) Troubleshooting
- No chart shown: check backend payload includes `plot` and `data`.
- Wrong chart type: review `VisualizationAgent` heuristics in `backend/core/agentic_framework.py`.
- Mismatched fields: ensure `options` fields (xField/yField/seriesField/data) match the actual `data` row keys returned by SQL.
