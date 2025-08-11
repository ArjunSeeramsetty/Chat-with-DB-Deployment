## Comprehensive Visualization Improvement Plan

### Executive Summary
The current visualization stack struggles with multi-dimensional time-series (e.g., monthly Energy Met by states). We integrate a Wren AI–aligned chart intelligence layer (Vega-Lite spec generation) to achieve consistent, correct charts across all workflows.

### Current Strengths
- End-to-end pipeline already returns a `plot` object (enhanced/agentic).
- Frontend can render charts given a structured payload.

### Critical Weaknesses
- Complex groupings fail (temporal + category).
- Limited chart variety; missing energy-domain special charts.
- Inconsistent behavior across processing paths.

### Root Cause
The existing `VisualizationAgent` uses basic heuristics that do not infer temporal granularity and categorical series simultaneously.

---

## Architecture Changes (Implemented)

### 1) Backend – Wren AI Chart Engine Integration
- New module: `backend/core/visualization_wren.py`
  - `WrenAIVisualizationService` analyzes result shape and produces a Vega-Lite spec when appropriate.
  - Handles multi-line temporal charts (e.g., states across months) and provides a simple fallback.

- Agentic workflow integration: `backend/core/agentic_framework.py`
  - `VisualizationAgent._generate_visualization` now delegates to `WrenAIVisualizationService` first; reverts to legacy heuristics on failure.
  - UI continues to receive a `chart_type` and `config`; for Vega-Lite, `chart_type: vega_lite` and `config` contains the Vega-Lite spec.

### 2) Frontend – Compatibility
- No breaking changes required. The existing renderer can be extended later to detect `chart_type === 'vega_lite'` and route to a `react-vega` renderer.

---

## Phase Plan (Next Steps)

### Phase 1 (1–2 weeks)
- Expand pattern detection for more complex composites (stacked area by source, heatmaps for shortages).
- Add natural-language edits adapter to tweak Vega-Lite spec (e.g., “thicker lines”).

### Phase 2 (1–2 weeks)
- Energy domain catalog: packaged spec templates for common sector visuals (multi-line temporal by state; state-time heatmap for shortages; stacked area for generation composition).
- Auto legend/tooltip standardization.

### Phase 3 (1 week)
- UI enhancement: add Vega-Lite renderer via `react-vega` with minimal wrapper.
- Optional: NL customization side-panel that updates the spec incrementally.

---

## Success Criteria
- “Monthly energy met by states” consistently renders as multi-line temporal chart.
- 15+ advanced chart types available.
- Same chart logic across enhanced/semantic/hybrid/traditional/agentic paths (where data present).
- <3s chart generation time.

---

## Testing
- Add golden tests mapping representative result tables to expected Vega-Lite encodings.
- UI smoke tests for presence of `chart_type: vega_lite` and spec validity (basic JSON schema check).


