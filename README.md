# Multi-Stage Heuristic and NLI Ensemble for Claim-Level Hallucination Control in Retrieval-Augmented LLM Medical QA
### A Reproducible Pilot Study (NICE NG28)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/Models-Local%20%2B%20Free-brightgreen)
![Status](https://img.shields.io/badge/Status-Pilot%20Study-yellow)

**Live demo:** https://researchwork1-app.streamlit.app/

---

## What this is

A **16-stage, fully reproducible research pipeline** for grounding medical
question-answering in retrieved evidence, verified **at the level of individual
atomic claims** — not just at the level of the answer as a whole. Every claim
produced by the system is checked against retrieved guideline text by **two
independent, local, free verifiers**:

1. A **lexical/numeric heuristic** verifier (similarity + term overlap + numeric consistency)
2. A **natural-language-inference (NLI) model** (`cross-encoder/nli-MiniLM2-L6-H768`) that actually models entailment and contradiction

The two verifiers are evaluated **against each other** and against **29
human-annotated gold claims** and **7 deliberately adversarial claims** designed
to try to break the system. Results are split into a **pre-specified primary
result** and a **post-hoc exploratory result**, so no threshold is silently
cherry-picked.

> **Case-study scope, by design.** This is a pilot study on an 18-chunk excerpt
> of one clinical guideline (NICE NG28), not a production system or a
> clinically validated tool. See [Limitations](#limitations) below — nothing
> here is hidden, only right-sized to what a controlled pilot can support.

## Why claim-level verification matters

Most RAG evaluation stops at "did the answer look plausible." This project
asks a stricter question: **for every individual factual claim inside a
generated answer, can we point to the exact guideline sentence that supports
or contradicts it** — and does a second, independently-reasoning model agree?

## Pipeline overview

| Stage | Purpose |
|---|---|
| 1–3 | Configuration, corpus loading, corpus validation |
| 4–7 | Dense embedding index, lexical/metadata rerank, top-k retrieval |
| 8 | Retrieval evaluation (Hit@3 on a 20-question set) |
| 9 | Evidence display + human gold/reference claim annotation |
| 10 | Local extractive QA (`distilbert-base-cased-distilled-squad`) |
| 11 | Rule-based, **LLM-free** atomic claim decomposition |
| 12 | Heuristic claim verifier + final report |
| 13 | Reference-claim verification (human gold vs. automated, P/R/F1) |
| 14 | Adversarial claim evaluation (7 deliberately contradictory claims) |
| 15 | **NLI-based** claim verification (independent of Stage 12's heuristic) |
| 16 | Threshold sensitivity analysis + heuristic-OR-NLI ensemble |

Full detail, code, and self-tests for every stage: [`notebook/`](./notebook).

## Primary result (decision threshold 0.70, pre-specified)

| Metric | Result |
|---|---|
| Stage 12 heuristic — adversarial accuracy | 2/7 = 28.57% |
| Stage 15 NLI — adversarial accuracy | 4/7 = 57.14% |
| NLI — human-gold "Supported" agreement (n=24) | 24/24 = 100% |
| NLI — human-gold "Insufficient Evidence" agreement (n=5) | 5/5 = 100% |

## Exploratory result (post-hoc, clearly labelled — not the headline number)

| Setting | Adversarial accuracy |
|---|---|
| NLI threshold 0.50 | 5/7 = 71.43% |
| NLI threshold 0.45 | 5/7 = 71.43% |
| Heuristic-OR-NLI ensemble | 5/7 = 71.43% |

These exploratory thresholds were selected **after** seeing the primary
result and are not calibrated on a held-out set — reported transparently as
exploratory, not as the system's validated accuracy.

## Live demo

[`app/app.py`](./app/app.py) ports the core pipeline (retrieval → extractive QA
→ claim decomposition → dual verification) into an interactive Streamlit app,
so a reviewer can type a question and watch every stage execute against the
live evidence, without opening a notebook.

### Run locally
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

### Deploy for free (recommended: Hugging Face Spaces)
Hugging Face Spaces is the better free-tier fit here specifically because
this app's own models (`sentence-transformers/all-MiniLM-L6-v2`,
`distilbert-base-cased-distilled-squad`, `cross-encoder/nli-MiniLM2-L6-H768`)
are hosted on the Hugging Face Hub, so the Space downloads and caches them on
the same platform with no extra setup, and the free CPU tier has enough
memory for these three small models (all locate under CPU RAM comfortably).

1. Create a new Space at huggingface.co/new-space → SDK: **Streamlit**.
2. Push (or upload) the contents of [`app/`](./app) to the Space repo root
   (`app.py`, `requirements.txt`, `corpus.json`).
3. The Space builds automatically and gives you a public URL — paste it into
   the "Live demo" line at the top of this README.

**Alternative:** Streamlit Community Cloud (share.streamlit.io) works the
same way, pointed at this GitHub repo with `app/app.py` as the entry point —
slightly simpler if you already have a Streamlit Cloud account, though its
free-tier memory ceiling is a little tighter for three loaded models at once.

## Repository structure
```
.
├── README.md
├── LICENSE
├── notebook/
│   └── multi_stage_heuristic_nli_ensemble_claim_verification_ng28.ipynb   # full 16-stage pipeline, all self-tests
├── app/
│   ├── app.py            # interactive Streamlit demo (retrieval -> QA -> claims -> dual verify)
│   ├── requirements.txt
│   └── corpus.json       # the 18-chunk NICE NG28 excerpt used throughout
└── docs/
    └── RESEARCH_MATURITY_ASSESSMENT.md   # candid scope/maturity note (see below)
```

## Limitations

This is a **pilot-scale case study**, and its scope is a deliberate design
choice, not an oversight:

- **Corpus:** an 18-chunk excerpt of NICE NG28, not the full guideline.
- **Evaluation set:** 20 questions, 36 evaluated claims (29 human gold + 7
  adversarial). Small enough that single-digit differences move the
  percentages meaningfully — no statistical significance claim is made.
- **Human labels are single-annotator**, not independently double-checked.
- **The NLI model is general-language**, not clinically fine-tuned or
  clinically validated.
- **"Supported" means textual/lexical consistency with retrieved evidence**,
  not proof of clinical correctness, guideline compliance, or safety.
- Findings are **specific to this guideline and this pilot corpus** and are
  not claimed to generalize to other conditions or larger corpora without
  further evaluation.

Full itemized limitations are documented inside the notebook itself (Stage 12
and Stage 15 limitation blocks) and are intentionally not summarized away —
see [`docs/RESEARCH_MATURITY_ASSESSMENT.md`](./docs/RESEARCH_MATURITY_ASSESSMENT.md)
for a candid discussion of what this scope does and does not license as a
claim.

## How to (re)run the full pipeline

1. Open [`notebook/multi_stage_heuristic_nli_ensemble_claim_verification_ng28.ipynb`](./notebook) in
   Google Colab (colab.research.google.com → File → Upload notebook).
2. Runtime → Run all. Every stage installs its own dependencies and uses only
   local model inference — no API key or paid account required.
3. First run downloads three small models from the Hugging Face Hub
   (`all-MiniLM-L6-v2`, `distilbert-base-cased-distilled-squad`,
   `cross-encoder/nli-MiniLM2-L6-H768`) — an internet connection is needed for
   that download only.

## License

MIT — see [`LICENSE`](./LICENSE).

## Citation

If you reference this pilot study, please cite it as a case study on
claim-level RAG verification methodology (dual heuristic + NLI verification,
adversarial stress-testing, primary/exploratory threshold separation) rather
than as a validated clinical system.
