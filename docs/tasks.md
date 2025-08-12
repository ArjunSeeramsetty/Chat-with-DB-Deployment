# Wren AI + MDL Integration – Final Tasks

## MDL schema completion
- Extract full SQLite schema (19 tables) and generate a complete MDL file
  - Implement/extend schema introspection to enumerate all tables, columns, PKs, FKs
  - Output: enrich `mdl/energy_domain.yaml` with:
    - models: names, descriptions, preferred_time_column, preferred_value_columns
    - columns: name, type, semantic_role (metric/dimension), units
    - relationships: all FKs with cardinality and join hints
- Align column names and units with DB
  - Verify all metric names (EnergyMet, EnergyShortage, MaximumDemand, outages, generation fields, exchange columns) match SQLite schema
- Add temporal granularity hints
  - For each model, set `has_date_fk`, `preferred_time_column`, and valid granularities (year, month, day, time-block)
- Add business semantics per model
  - Tag columns as dimensions vs metrics; annotate canonical names and synonyms

## MDL business rules and metrics
- Encode domain calculations
  - Define metric expressions (SUM/AVG/MAX/MIN/COUNT) and aggregation defaults
  - Outage rules: Central/State/Private/Total with COALESCE and safe sums
  - Generation breakdown mapping to sources (coal, hydro, gas, nuclear, solar, wind, RES)
- Temporal requirements
  - Annotate which metrics require DimDates join, and which time granularities apply (e.g., FAIDS monthly/daily; time-block for FTBPD/FTBG)

## Wren MDL-aware SQL generation
- Prompt refinement
  - Update `_build_mdl_aware_prompt` to include “MODEL HINTS”, NEVER rules, and temporal granularities per model
  - Include the complete model list and relationship constraints in the prompt context
- Postprocessing with MDL hints
  - Expand `_postprocess_sql_with_mdl_hints` to:
    - Replace invalid metric names by preferred_value_columns
    - Enforce joins based on FK relationships
    - Select correct time column and granularities from MDL hints

## Vector store coverage
- Index all models/relationships/columns
  - Ensure `schema_semantics`, `mdl_models`, `mdl_relationships`, `mdl_columns` collections include all 19-table MDL objects
  - Add synonyms/aliases and business keywords to embeddings (fallback hash for offline)

## Validator and auto-repair (schema-driven)
- FK-driven joins
  - Given selected model(s), auto-inject mandatory joins via FK rules from MDL (DimDates/DimRegions/DimStates/others)
- Alias/column correction
  - Use MDL-provided canonical aliases and columns to fix invalid references
- Temporal enforcement
  - Enforce required DimDates usage for models with `has_date_fk`
  - Ensure aggregation and grouping align with requested granularity

## Explicit-table candidate
- Safe select builder
  - Add a candidate path that detects explicit table mention and builds a SELECT with discovered columns (limit to safe columns, add WHERE scaffolding) using MDL schema

## Visualization alignment
- Chart metadata from MDL
  - For each metric, add recommended visualization (e.g., multi-line temporal by state/region; stacked area for source composition)
  - Include MDL-driven field mappings in plot options (xAxis/yAxis/groupBy defaults)

## Testing and verification
- Comprehensive smoke suite for all fact models
  - Expand `scripts/test_all_fact_tables.py` to cover all 19 tables:
    - semantic and explicit table queries
    - expected SQL substrings and row counts
    - candidate source tracking
- Temporal tests
  - Add unit tests for “monthly”, “daily”, “yearly”, “time-block” behaviors per model
- Outage and energy domain tests
  - Verify all outage permutations and generation-by-source queries

## Performance and initialization
- One-time MDL + vector initialization
  - Ensure enhanced service loads MDL and vector store once per process; cache embeddings
- Logging/observability
  - Add structured logs for selected model(s), detected relationships, chosen time column, and final plot config

## Deliverables
- Updated MDL: `mdl/power-sector-mdl.json` or enriched `energy_domain.yaml`
- Code updates: `wren_ai_integration.py`, `enhanced_rag_service.py`, validator, vector indexing
- Tests: expanded `scripts/test_all_fact_tables.py` + unit tests
- Docs: short “MDL coverage and usage” README (how models, relationships, and metrics are used by Wren)

## Acceptance criteria
- MDL includes all 19 tables, columns, and FK relationships
- Wren MDL candidate is primary for all supported queries
- Temporal granularity correctly enforced (monthly/daily/yearly/time-block)
- Outage, generation breakdown, and exchange queries pass tests
- Plot objects consistently render grouped series (states/regions/sources) without manual overrides


