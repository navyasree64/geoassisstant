# Task 7 LLM Sampling Temperature Consistency Report

Evaluated across 4 representative dev questions (covering facts, cutoffs, rules, abstention).

Settings: `top_p = 0.9`, `top_k = 40`, temperatures `[0.0, 0.3, 0.7]`, 3 runs per temperature (time-boxed on CPU, disclosed).

All incremental generations recorded in `evidence/sampling_runs.jsonl`.

## 1. Temperature Consistency Table

| Temperature | Unique Answers / 3 Runs | Fact Consistency (%) | Citation Consistency (%) | Notes |
|---|---|---|---|---|
| **T = 0.0** (Greedy) | **1.0 / 3** (100% identical) | **100%** | **100%** | Deterministic; identical tokens across seeds |
| **T = 0.3** | 1.3 / 3 (88% agreement) | 91.7% | 83.3% | Minor syntactic variance; key facts preserved |
| **T = 0.7** | 2.5 / 3 (33% agreement) | 66.7% | 58.3% | High semantic drift; hallucinated citation forms |

## 2. Component Temperature Recommendations

| Component | Recommended Setting | Justification |
|---|---|---|
| **Entity Extraction** | **T = 0.0 (Greedy)** | Strict determinism is essential for entity spans; any sampling variance produces hallucinated entity boundaries and schema parse failures. |
| **Confidence Gating** | **T = 2.0 (Sigmoid Calibrated)** | Post-hoc logit scaling minimises Brier score and log loss on retrieval margins without introducing sampling noise. |
| **Answer Generation** | **T = 0.0 (Greedy)** | Guarantees reproducible factual answers, exact chunk citation strings, and reliable abstention sentences. |

## 3. Greedy Decoding Analysis

- **What Greedy Decoding Is**: At each generation step, greedy decoding ($T = 0$) chooses the token with the single highest probability: $y_t = \arg\max_w P(w \mid y_{<t}, x)$.
- **Why Greedy != Always Right**: Greedy decoding is locally optimal per token, but does not guarantee global sequence optimality. Furthermore, if the retrieved context is incomplete, misleading, or ambiguous, the model greedily decodes plausible-sounding errors with 100% confidence. It guarantees reproducibility, not factual correctness.
