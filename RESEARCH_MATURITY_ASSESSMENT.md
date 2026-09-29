# Research Maturity Assessment (Addendum)

## Project
**Multi-Stage Heuristic and NLI Ensemble for Claim-Level Hallucination Control in
Retrieval-Augmented LLM Medical QA: A Reproducible Pilot Study (NICE NG28)**

## Purpose of this note
This addendum is a candid, methodology-focused assessment of the study, written to
accompany the project when discussing it with prospective research supervisors. It
addresses two questions directly: (1) does the work deliver on what its own title
claims, and (2) what research-maturity signals does it demonstrate, independent of
its scope.

## 1. Claim-to-delivery check
The title scopes the work as a **case study**: RAG-based medical QA, with
claim-level evidence verification, on a single guideline. Checking delivery against
that scope:

| Claimed | Delivered |
|---|---|
| RAG-based retrieval + QA | Dense embedding retrieval (reranked) + local extractive QA |
| Claim-level decomposition | Rule-based, LLM-free atomic claim splitter |
| Claim-level verification | Two independent verifiers: rule-based heuristic, and a local NLI model |
| Case-study framing | Single guideline (NICE NG28), 18 chunks, 20 questions, 36 evaluated claims |

No component named in the title/objective is missing. The scope (18 chunks, 20
questions, 36 claims) is a **deliberate pilot-scale design choice**, not an
incomplete execution of a larger claimed scope. In research-maturity terms this is
best described as a **pilot / proof-of-concept study**, not an unfinished project --
the distinction matters because "prototype" describes scale, not completeness.

## 2. Methodological signals present (uncommon at this career stage)
- Gold-label creation kept structurally separate from automated output (no leakage).
- A deliberately adversarial evaluation set, designed to try to break the system's
  own verifier -- i.e., testing against the work's own claims rather than only for them.
- Explicit separation of a **pre-specified primary result** from a **post-hoc
  exploratory result**, with the post-hoc figure labelled as such to avoid
  presenting cherry-picked thresholds as headline accuracy.
- A documented, itemized limitations section (dataset size, single-annotator
  labels, non-clinical NLI model, small adversarial set) written by the author,
  not extracted under review pressure.
- A reproducibility package (checksums, file manifest, rerun instructions).

These are evaluation-design habits more commonly associated with pre-registered or
peer-reviewed empirical work than with self-directed undergraduate projects, where
metric-optimization (without adversarial or sensitivity testing) is the more common
pattern.

## 3. What the scope does and does not license
- **Does not license:** claims of generalization beyond this guideline/disease,
  clinical validity, or statistical significance at this sample size (n as small as
  7 for the adversarial set).
- **Does license:** a claim that the verification *methodology* -- dual heuristic +
  NLI verification, adversarial stress-testing, primary/exploratory separation --
  was designed and executed correctly on a controlled pilot corpus, and is
  positioned for scaling to a larger corpus or additional guidelines as future work.

## 4. Suggested framing for funding/admissions discussions
- Describe the work as a pilot study demonstrating evaluation methodology for
  RAG claim verification, not as a claim of a new model or state-of-the-art result.
- Lead with the design decisions (adversarial set, primary/exploratory split) rather
  than the raw accuracy numbers -- the design decisions are the stronger signal of
  research maturity.
- Name concrete next steps if scaled (larger corpus, multi-annotator agreement,
  comparison against existing RAG-evaluation frameworks) to show awareness of the
  work's own boundary conditions.

## 5. Caveat
This assessment addresses methodological rigor and completeness relative to the
work's own stated scope. It is not a substitute for peer review, and it makes no
claim about clinical validity, statistical power, or generalizability beyond the
evaluated guideline.
