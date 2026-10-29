# Bug Fix Verification: Hare School Cut-off Mark (evidence/fix2_before_after.md)

## Bug Description
- **Reported Issue**: In the live UI, the question *"What is the cut-off mark for Hare School?"* returned **"80"** with confidence `0.9678` and a "Grounded Answer" label.
- **Root Cause Analysis**:
  1. **Tabular Row Cross-Contamination**: In passage `CIR-006-table`, the markdown table lists multiple schools:
     ```markdown
     | School | Seats | Cut-off marks |
     |---|---|---|
     | Hare School | 120 | 78 |
     | Baranagar Ramakrishna Mission Ashrama High School | 80 | 80 |
     ```
     `80` is the cut-off of Baranagar Ramakrishna Mission, while Hare School's cut-off is **78**.
  2. **Passage Ordering & Confusion**: The cross-encoder reranker ranked `CIR-006-note` (score 6.81) above `CIR-006-table` (5.09). `CIR-006-note` contains *"Baranagar Ramakrishna Mission Ashrama High School has the highest cut-off (80 marks) and Hare School has the most seats (120 seats)"*. The 3B model confused the cut-off mark of Baranagar with Hare School.

## General Mechanisms Implemented (No Hardcoded Overrides)
1. **Entity-Scoped Context (`filter_entity_scoped_tables`)**:
   Before constructing the prompt, for each passage containing a markdown table, the system keeps the table header, separator, and **only** the rows whose first cell matches a detected query entity (normalized string similarity $\ge 0.85$). Unrelated rows from other schools are filtered out.
2. **Post-Generation Answer Verification (`verify_answer_facts`)**:
   Substantive numbers/dates in the answer are checked against cited chunks: each number must co-occur in a table row or sentence containing the entity's name. If verification fails, the pipeline retries once with strictly entity-scoped rows; if it fails again, confidence is lowered below threshold ($0.45 < 0.85$) and the system safely abstains with *"I cannot answer this from the available sources."*
3. **Readable Sentence Formatting**:
   Prompt instructs the model to return a concise, readable short sentence.
4. **Verification Status in UI & CLI**:
   Adds `"verification": "Verified"` to output JSON schema, displaying `Verified answer` in green card only upon verified factual consistency.

## Before vs After Comparison

| Attribute | Before Fix | After Fix |
|---|---|---|
| **Query** | `What is the cut-off mark for Hare School?` | `What is the cut-off mark for Hare School?` |
| **Answer** | `80` (Wrong - Baranagar RK Mission cut-off) | `Hare School's cut-off is 78 marks.` |
| **Target Fact** | `78` | `78` (MATCHED) |
| **Confidence** | `0.9678` | `0.9678` |
| **Verification Field** | None | `Verified` |
| **UI Label** | `Grounded Answer` (Unverified) | `Verified answer` |
| **Citations** | `['CIR-006-table']` | `['CIR-006-table']` |
| **Citation Correctness** | Valid | Valid |
| **Status** | **BUG (Hallucinated 80)** | **RESOLVED (Correct 78 Verified)** |

## Verification Check
- Confirmed Hare School cut-off returned: **78 marks**
- Verification passed: **True**
- Citations valid: **True**
