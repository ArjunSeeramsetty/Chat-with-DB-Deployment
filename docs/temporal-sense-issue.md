# Temporal-Sense Propagation Issue

## 1  Problem Statement
User questions frequently contain an *implicit* or *explicit* time constraint –­­ e.g. “in **June 2025**” or “on **15 July 2024**”.

Although the pipeline correctly **detects** these tokens (month/day/year) during intent-analysis, the information does **not always reach the final SQL** that is returned to the front-end.
The result is valid SQL syntactically, but **semantically wrong**, because it
a. returns an annual result when the user asked for a single month or day, or
b. groups incorrectly (monthly average vs. daily value).

## 2  Where It Breaks
```
ask_question_enhanced  ─►  process_query_enhanced
                           ├─ unified enhanced pipeline  (ideal path)
                           │      • _validate_and_correct_sql  ✓ adds month/day filters
                           │
                           ├─ adaptive fall-backs
                           │      • _process_semantic_first
                           │      • _process_hybrid
                           │      • _process_agentic
                           │      • _process_traditional   ← *problematic path*
                           │           └─ SQLAssembler.generate_sql
                           │                └─ _generate_with_templates
                           │                      (has no intrinsic month filter)
                           └─ post-processing & response
```
1. **Enhanced path OK** – `_validate_and_correct_sql` contains the “temporal-refinement” block that injects `strftime('%m', …)` / `dt.Month` / `dt.DayOfMonth` clauses.
2. **Traditional path FAILS** – when the unified pipeline errors → falls back to `_process_traditional` → assembler template → post-processing. The template itself only receives a *year* filter (detected elsewhere) and drops the month/day.
3. **Alias Re-write side-effect** – `_post_process_sql` normalises aliases (`ds`→`s`, `dt`→`d`). If month/day filters were added *before* this step using `dt.Month`, the alias swap may strip them out, leaving only the year clause.

## 3  Symptoms
* SQL shows `WHERE strftime('%Y', d.ActualDate) = '2025'` but **no month/day restriction**.
* Front-end results therefore aggregate the whole year; monthly charts are wrong.
* Logs show:
  - `TEMPLATE GENERATION SUCCEEDED` (state_query)
  - **No** `Temporal refinement: Added month filter …` lines (because validator wasn’t executed).

## 4  Root Causes (checked-in before fix)
1. In `_generate_candidate_sqls` the **explicit-table shortcut** skipped `_validate_and_correct_sql` for candidates already mentioning the right fact table.
2. `_process_traditional` returned the assembler SQL *before* sending it through the validator.
3. Assembler templates themselves did not append month/day filters – they relied on the validator.
4. Temporal-refinement block only recognised `d.ActualDate`, `fs.Timestamp`, `fs.TimeBlock`; if the template used `dt.Month` the regex failed.
5. Missing DB file during unit-tests caused the enhanced pipeline to throw – silently switching to the traditional path.

## 5  Fixes Implemented
| # | Component | Change |
|---|-----------|--------|
|1|`backend/core/assembler.py`| Month/day/year parsing added; filters appended directly to `state_query` / `region_query` templates, **before** alias rewrite.|
|2|`backend/services/enhanced_rag_service.py` (validator)| Removed explicit-table shortcut; validation now *always* runs so temporal refinement executes.|
|3|`backend/services/enhanced_rag_service.py` (`_process_traditional`)| Now pipes assembler SQL through `_validate_and_correct_sql` before execution.|
|4|Validator temporal-refinement| Recognises `dt.ActualDate` and infers from `DateID` joins; extensive logging added.|
|5|Service init guard| DB-dependent components now stubbed when `db_path` missing – prevents silent fallback in tests.|

## 6  Current Behaviour (after patch)
Example – “**maximum Energy Met of all states in June 2025**”
```sql
… WHERE dt.Year = 2025
      AND dt.Month = 6
GROUP BY ds.StateName;
```
*Always* produced – regardless of which pipeline branch wins.

## 7  Follow-ups / Hardening
1. Mirror the new filter-injection in any future templates (`transmission_query`, etc.).
2. Refactor temporal-refinement into a reusable helper to avoid duplication.
3. Add unit test `test_temporal_refinement.py` to CI – fails if month/day not present.
4. Consider moving time-sense parsing earlier (IntentAnalyzer) and passing a structured `TimeFilter` object through the pipeline.
