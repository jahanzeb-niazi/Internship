# Evaluation Results & Prompt Engineering Comparison

## Overview
This report documents the baseline accuracy of the screening agent and the results of a targeted prompt engineering experiment aimed at reducing LLM overconfidence on borderline candidates.

---

## 1. Baseline Evaluation (Before Prompt Change)
The initial evaluation run was performed on a subset of 8 test cases before hitting free-tier API rate limits. 

**Metrics:**
- **Total Test Cases Run:** 8
- **Profile Extraction Accuracy:** 100.0%
- **Decision Accuracy:** 75.0%
- **Average Confidence:** 0.906

**Failure Analysis:**
The baseline prompt failed on two specific test cases (`TC006` and `TC008`).
- **TC006 (Borderline candidate):** Expected `needs_human_review`, but the LLM output `not_qualified` with a high confidence of **0.80**.
- **TC008 (Vague application):** Expected `needs_human_review`, but the LLM output `not_qualified` with an extremely high confidence of **0.90**.

**Conclusion from Baseline:** The LLM was overconfident when dealing with ambiguous or missing data, opting to decisively reject candidates rather than safely flagging them for human review.

---

## 2. Prompt Engineering Intervention
To fix this overconfidence, the system prompt in `qualifier.py` was updated. 

**Added Instructions:**
```markdown
3. Make a decision: "qualified", "not_qualified", or "needs_human_review".
   - CRITICAL: If the candidate is borderline, has ambiguous experience, or the application is vague/missing details, you MUST choose "needs_human_review".
4. Provide a confidence score between 0.0 and 1.0 based on how clear the evidence is.
   - CRITICAL: If the application is vague, borderline, or missing details, your confidence MUST be BELOW 0.70. Do NOT be overconfident on vague applications.
```

---

## 3. Post-Intervention Evaluation (After Prompt Change)
A second evaluation was run on a targeted subset of 4 test cases (2 controls that passed previously, and the 2 that failed).

**Metrics:**
- **Total Test Cases Run:** 4
- **Profile Extraction Accuracy:** 100.0%
- **Decision Accuracy:** 50.0% (Regression)
- **Average Confidence:** 0.863

**Failure Analysis:**
- **TC006 & TC008 (Ambiguous):** Still failed. The LLM completely ignored the "CRITICAL" instructions to lower its confidence. For TC008, it actually *increased* its confidence to **0.95** while outputting `not_qualified`.
- **TC001 (Clear Pass Control):** Failed (Regression). The clearly qualified candidate that easily passed in the baseline was now flagged as `needs_human_review` because the LLM's confidence artificially dropped to **0.60** (below our 0.70 threshold). 

---

## 4. Final Takeaways
The zero-shot prompt engineering attempt **failed and caused a regression**. 
Instructing the LLM to "not be overconfident" via text rules caused it to second-guess perfectly clear candidates (dropping TC001 confidence to 0.60), while simultaneously having zero effect on its hallucinated certainty for vague applications (TC008 confidence remained 0.95). 

**Next Steps for Accuracy:** 
Instead of adding more rules to the prompt, the `qualifier.py` module must be updated to use **Few-Shot Prompting**. We need to provide the LLM with 2-3 explicit JSON examples of vague resumes paired with the expected `needs_human_review` output and a `0.50` confidence score.
