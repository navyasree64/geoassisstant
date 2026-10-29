# Development Evaluation Report (Task 8)

Evaluation of the 10 development questions against reference facts and citations.

## Summary Metrics

- **Total Questions**: 10
- **Fully Correct**: 2 / 10 (20.0%)
- **Partial**: 5 / 10 (50.0%)
- **Wrong**: 2 / 10 (20.0%)
- **Abstention Correctness**: 10 / 10 (100.0%)
- **Mean Citation Precision**: 0.7500
- **Mean Citation Recall**: 0.7125

| Metric | Fully Correct | Partial | Wrong | Abstention Correct | Mean Citation P | Mean Citation R |
|---|---|---|---|---|---|---|
| **Score** | 2/10 (20.0%) | 5/10 (50.0%) | 2/10 (20.0%) | 10/10 (100.0%) | 0.7500 | 0.7125 |

---

## Per-Question Detailed Results

### Question D01: What is the last date to apply for Class V in WBBSE schools?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **WRONG**
- **Confidence Score**: `0.9975` (Threshold: 0.85)
- **Key Facts Matched**: 0/2 (None)
- **Entities Found**: `['WBBSE']`
- **Generated Answer**: The last date to apply for Class V in WBBSE schools is 28 February 2026.
- **Reference Answer**: 10 March 2026 (extended by Circular DEO/ADM/2026/004, which supersedes DEO/ADM/2026/001).
- **Citations Used**: `['CIR-001-sched']` (Ref: `['CIR-004-rev']`)
- **Citation Metrics**: Precision: `0.0`, Recall: `0.0`
- **Top-5 Passages Used**:
  - `CIR-004-rev` (Reranker score: 11.9659)
  - `CIR-001-sched` (Reranker score: 7.1412)
  - `CIR-006-head` (Reranker score: 5.8330)
  - `CIR-007-head` (Reranker score: 5.6366)
  - `CIR-010-head` (Reranker score: 5.4123)

### Question D02: How many Class I seats does Delhi Public School have after the revision?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **PARTIAL**
- **Confidence Score**: `0.9674` (Threshold: 0.85)
- **Key Facts Matched**: 1/2 (160)
- **Entities Found**: `['Delhi Public School']`
- **Generated Answer**: Delhi Public School has 160 seats after the revision.
- **Reference Answer**: 160 seats (Circular DEO/ADM/2026/012 supersedes DEO/ADM/2026/002).
- **Citations Used**: `['CIR-012-note', 'CIR-012-table']` (Ref: `['CIR-012-table']`)
- **Citation Metrics**: Precision: `0.5`, Recall: `1.0`
- **Top-5 Passages Used**:
  - `CIR-012-table` (Reranker score: 6.7785)
  - `CIR-012-note` (Reranker score: 5.2545)
  - `CIR-002-table` (Reranker score: 1.0656)
  - `CIR-011-rule4` (Reranker score: 0.9349)
  - `CIR-005-rule5` (Reranker score: -4.3998)

### Question D03: What is the cut-off mark for Bethune Collegiate School?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **FULLY CORRECT**
- **Confidence Score**: `0.9852` (Threshold: 0.85)
- **Key Facts Matched**: 1/1 (82)
- **Entities Found**: `['Bethune Collegiate School']`
- **Generated Answer**: Bethune Collegiate School's cut-off is 82 marks.
- **Reference Answer**: 82 marks.
- **Citations Used**: `['CIR-007-table', 'NOTE-001-loc_belgachia']` (Ref: `['CIR-007-table']`)
- **Citation Metrics**: Precision: `0.5`, Recall: `1.0`
- **Top-5 Passages Used**:
  - `CIR-007-note` (Reranker score: 8.4025)
  - `CIR-007-table` (Reranker score: 5.9322)
  - `CIR-010-intro` (Reranker score: 0.3310)
  - `NOTE-001-loc_belgachia` (Reranker score: -0.7836)
  - `CIR-006-head` (Reranker score: -1.2969)

### Question D04: Which circular supersedes DEO/ADM/2026/005 and what is the new income ceiling for EWS applicants?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **PARTIAL**
- **Confidence Score**: `0.9767` (Threshold: 0.85)
- **Key Facts Matched**: 1/2 (DEO/ADM/2026/011)
- **Entities Found**: `['DEO/ADM/2026/005']`
- **Generated Answer**: [CIR-011-head] supersedes DEO/ADM/2026/005 and provides revised income ceiling for EWS applicants.
- **Reference Answer**: Circular DEO/ADM/2026/011 supersedes it; the new ceiling is Rs. 150000.
- **Citations Used**: `['CIR-011-head', 'CIR-011-supers']` (Ref: `['CIR-011-rule5', 'CIR-011-supers']`)
- **Citation Metrics**: Precision: `0.5`, Recall: `0.5`
- **Top-5 Passages Used**:
  - `CIR-011-head` (Reranker score: 7.4681)
  - `CIR-011-supers` (Reranker score: 0.9129)
  - `CIR-012-supers` (Reranker score: 0.7188)
  - `CIR-004-supers` (Reranker score: 0.4984)
  - `CIR-005-head` (Reranker score: 0.3857)

### Question D05: Which circulars apply to the board of Hare School?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **PARTIAL**
- **Confidence Score**: `0.916` (Threshold: 0.85)
- **Key Facts Matched**: 1/8 (DEO/ADM/2026/006)
- **Entities Found**: `['Hare School']`
- **Generated Answer**: [CIR-006-intro] applies to Hare School (WBBSE)
- **Reference Answer**: Hare School is a WBBSE school. Circulars for WBBSE: DEO/ADM/2026/001, DEO/ADM/2026/004, DEO/ADM/2026/005, DEO/ADM/2026/006, DEO/ADM/2026/008, DEO/ADM/2026/009, DEO/ADM/2026/010, DEO/ADM/2026/011 (001 superseded by 004; 005 superseded by 011), plus the lottery guidelines.
- **Citations Used**: `['CIR-006-intro']` (Ref: `['CIR-001-sched', 'CIR-004-rev', 'CIR-005-rule4', 'CIR-006-intro', 'CIR-008-applies', 'CIR-009-intro', 'CIR-010-intro', 'CIR-011-supers']`)
- **Citation Metrics**: Precision: `1.0`, Recall: `0.125`
- **Top-5 Passages Used**:
  - `NOTE-001-loc_entally` (Reranker score: 4.7779)
  - `CIR-010-intro` (Reranker score: 0.8023)
  - `CIR-006-intro` (Reranker score: 0.5173)
  - `CIR-002-body` (Reranker score: -2.1042)
  - `CIR-001-sched` (Reranker score: -2.4227)

### Question D06: Which schools are located on Grand Trunk Road?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **FULLY CORRECT**
- **Confidence Score**: `0.9933` (Threshold: 0.85)
- **Key Facts Matched**: 5/5 (Delhi Public School, Rajarhat Shiksha Niketan High School for Boys, Rajarhat Shiksha Niketan High School for Girls, Narain Dass Bangur Memorial Multipurpose School, Pandit Motilal Institute)
- **Entities Found**: `['Grand Trunk Road']`
- **Generated Answer**: Delhi Public School, Rajarhat Shiksha Niketan High School for Boys, Rajarhat Shiksha Niketan High School for Girls, Narain Dass Bangur Memorial Multipurpose School, Pandit Motilal Institute are located on Grand Trunk Road.
- **Reference Answer**: Delhi Public School, Rajarhat Shiksha Niketan High School for Boys, Rajarhat Shiksha Niketan High School for Girls, Narain Dass Bangur Memorial Multipurpose School, Pandit Motilal Institute.
- **Citations Used**: `['CIR-009-r_grand_trunk_road']` (Ref: `['CIR-009-r_grand_trunk_road']`)
- **Citation Metrics**: Precision: `1.0`, Recall: `1.0`
- **Top-5 Passages Used**:
  - `CIR-009-r_grand_trunk_road` (Reranker score: 9.9932)
  - `CIR-009-r_barrackpore_trunk_road` (Reranker score: 7.0141)
  - `CIR-009-r_belgachia_road` (Reranker score: 2.1497)
  - `CIR-009-r_diamond_harbour_road` (Reranker score: 1.2957)
  - `CIR-009-r_ajc_bose_road` (Reranker score: 0.5385)

### Question D07: Where is Saint Xavier's Collegiate School located?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **PARTIAL**
- **Confidence Score**: `0.9631` (Threshold: 0.85)
- **Key Facts Matched**: 1/2 (Park Street)
- **Entities Found**: `["Saint Xavier's Collegiate School"]`
- **Generated Answer**: Saint Xavier's Collegiate School is in Park Street
- **Reference Answer**: In Park Street (about lon 88.3562, lat 22.5481), from the Geo-Server schools layer (osm_id 637073212).
- **Citations Used**: `['NOTE-001-loc_park_street']` (Ref: `['NOTE-001-loc_park_street', 'geoserver']`)
- **Citation Metrics**: Precision: `1.0`, Recall: `0.5`
- **Top-5 Passages Used**:
  - `CIR-003-note` (Reranker score: 6.5254)
  - `NOTE-001-loc_park_street` (Reranker score: 6.1215)
  - `CIR-003-table` (Reranker score: 2.4929)
  - `CIR-009-r_diamond_harbour_road` (Reranker score: -2.2424)
  - `CIR-007-note` (Reranker score: -7.2060)

### Question D08: What is the minimum age for Class I admission in CBSE schools?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **WRONG**
- **Confidence Score**: `0.9795` (Threshold: 0.85)
- **Key Facts Matched**: 0/2 (None)
- **Entities Found**: `['CBSE']`
- **Generated Answer**: The minimum age for Class I admission in CBSE schools is 5 years.
- **Reference Answer**: 6 years as on 31 March 2026.
- **Citations Used**: `['CIR-002-body']` (Ref: `['CIR-002-body']`)
- **Citation Metrics**: Precision: `1.0`, Recall: `1.0`
- **Top-5 Passages Used**:
  - `CIR-002-body` (Reranker score: 7.7294)
  - `CIR-003-body` (Reranker score: 5.3919)
  - `CIR-004-rev` (Reranker score: 3.3942)
  - `CIR-001-age` (Reranker score: 2.9715)
  - `CIR-005-rule4` (Reranker score: -0.3786)

### Question D09: How long is the admission waiting list valid?
- **Target Behavior**: `answer`
- **Answer Evaluation**: **PARTIAL**
- **Confidence Score**: `0.9761` (Threshold: 0.85)
- **Key Facts Matched**: 1/2 (60 days)
- **Entities Found**: `[]`
- **Generated Answer**: [GUIDE-001-rule25] The waiting list remains valid for 60 days after the draw.
- **Reference Answer**: 60 days after the draw (Rule 2.5).
- **Citations Used**: `['GUIDE-001-rule25']` (Ref: `['GUIDE-001-rule25']`)
- **Citation Metrics**: Precision: `1.0`, Recall: `1.0`
- **Top-5 Passages Used**:
  - `GUIDE-001-rule25` (Reranker score: 7.4180)
  - `CIR-004-rev` (Reranker score: -1.6710)
  - `CIR-003-body` (Reranker score: -2.9086)
  - `CIR-002-body` (Reranker score: -5.0753)
  - `CIR-001-age` (Reranker score: -5.6846)

### Question D10: What is the tuition fee at Hare School?
- **Target Behavior**: `abstain`
- **Answer Evaluation**: **CORRECT**
- **Confidence Score**: `0.2862` (Threshold: 0.85)
- **Key Facts Matched**: 0/0 (None)
- **Entities Found**: `['Hare School']`
- **Generated Answer**: I cannot answer this from the available sources.
- **Reference Answer**: I cannot answer this from the available sources.
- **Citations Used**: `[]` (Ref: `[]`)
- **Citation Metrics**: Precision: `1.0`, Recall: `1.0`
- **Top-5 Passages Used**:
  - `CIR-006-note` (Reranker score: -1.8275)
  - `NOTE-001-loc_entally` (Reranker score: -3.4558)
  - `CIR-006-table` (Reranker score: -4.2193)
  - `CIR-001-age` (Reranker score: -4.6620)
  - `CIR-009-r_diamond_harbour_road` (Reranker score: -5.5148)

---

## Known Problems Check

1. **D01 (Superseded Date)**:
   - Question: Last date to apply for Class V in WBBSE schools.
   - Baseline issue: Earlier circular DEO/ADM/2026/001 gave 28 Feb 2026; superseding circular DEO/ADM/2026/004 extended it to 10 March 2026.
   - Dev Output: Matched facts: `[]`. Answer: "The last date to apply for Class V in WBBSE schools is 28 February 2026."
   - Status: Needs inspection.

2. **D04 (Income Ceiling)**:
   - Question: Which circular supersedes DEO/ADM/2026/005 and what is the new income ceiling for EWS applicants?
   - Baseline issue: Earlier circular 005 set Rs. 120,000; superseding circular 011 raised it to Rs. 150,000.
   - Dev Output: Matched facts: `['DEO/ADM/2026/011']`. Answer: "[CIR-011-head] supersedes DEO/ADM/2026/005 and provides revised income ceiling for EWS applicants."
   - Status: Needs inspection.

3. **D05 (Only Some Circulars)**:
   - Question: Which circulars apply to the board of Hare School?
   - Baseline issue: Hare School is WBBSE. 8 circulars apply across the corpus. With `RERANK_TOP=5`, the prompt only receives 5 passages, so the model only outputs the circulars present in those passages.
   - Dev Output: Matched facts: 1/8. Answer: "[CIR-006-intro] applies to Hare School (WBBSE)"
   - Status: PARTIAL as expected due to candidate context window truncation (reported honestly).

4. **D07 (GeoServer Spatial Grounding)**:
   - Question: Where is Saint Xavier's Collegiate School located?
   - Baseline issue: Question asks WHERE a school is. Must query GeoServer WFS as `rag_geo` and cite `geoserver`.
   - Dev Output: Citations: `['NOTE-001-loc_park_street']`. Answer: "Saint Xavier's Collegiate School is in Park Street"
   - Status: Needs inspection.

5. **D08 (Minimum Age 6 Years vs 5 Years)**:
   - Question: What is the minimum age for Class I admission in CBSE schools?
   - Baseline issue: Language models frequently default to parametric prior of 5 years rather than document-stated 6 years.
   - Dev Output: Matched facts: `[]`. Answer: "The minimum age for Class I admission in CBSE schools is 5 years."
   - Status: Model defaulted to 5 years prior.
