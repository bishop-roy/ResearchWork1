"""
Claim-Level Evidence Verification for RAG-Based Medical QA — Live Demo
A Type 2 Diabetes Case Study (NICE NG28)

Interactive Streamlit port of the 16-stage research notebook's core pipeline:
retrieval -> extractive QA -> rule-based claim decomposition -> dual
verification (heuristic + local NLI model). All models are free, local,
and run on CPU. No API key, no paid service.

This app is a DEMO of the pipeline's mechanics on a fixed 18-chunk excerpt of
NICE NG28. It is a research/educational artifact, not a clinical tool.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# --------------------------------------------------------------------------
# Stage 1 — configuration (unchanged from the research notebook)
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"

TOP_K = 3
STEM_OVERLAP_WEIGHT = 0.22
DISTINCTIVE_STEM_WEIGHT = 0.10
DISTINCTIVE_STEM_CAP = 0.20
NUMBER_OVERLAP_WEIGHT = 0.18
META_OVERLAP_WEIGHT = 0.08
DISTINCTIVE_STEM_MIN_LEN = 6

QA_MAX_ANSWER_LEN = 40
QA_TOP_K_LOGITS = 20
QA_MAX_SEQ_LEN = 384

SIMILARITY_THRESHOLD = 0.35
TERM_OVERLAP_THRESHOLD = 0.50
CLAIM_MIN_CONTENT_STEMS = 4

NLI_ENTAILMENT_THRESHOLD = 0.70
NLI_CONTRADICTION_THRESHOLD = 0.70

NO_EVIDENCE_TEXT = "The retrieved evidence is insufficient to answer this question."

# --------------------------------------------------------------------------
# Stage 5 — query preprocessing utilities
# --------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with",
    "is", "are", "was", "be", "been", "being", "should", "would", "could",
    "when", "what", "how", "who", "which", "that", "this", "those", "these",
    "if", "at", "by", "from", "as", "it", "their", "them", "they", "do",
    "does", "did", "not", "no", "yes", "than", "then", "into", "about",
    "adults", "adult", "people", "person", "type", "diabetes", "nice",
}

STEM_ALIASES = {
    "intensified": "intensif", "intensify": "intensif", "intensifying": "intensif",
    "intensification": "intensif", "rise": "rise", "rises": "rise",
    "rising": "rise", "risen": "rise", "rose": "rise", "medicine": "medicin",
    "medicines": "medicin", "medication": "medicin", "medications": "medicin",
    "target": "target", "targets": "target", "offered": "offer", "offer": "offer",
    "offering": "offer", "measure": "measur", "measured": "measur",
    "measurement": "measur", "monitoring": "monitor", "monitor": "monitor",
    "contraindicated": "contraindic", "contraindication": "contraindic",
    "tolerated": "tolerat", "tolerance": "tolerat",
}


def light_stem(token: str) -> str:
    t = token.lower()
    if t in STEM_ALIASES:
        return STEM_ALIASES[t]
    for suf in ("ation", "ations", "ing", "ed", "es", "s", "ly"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            break
    return STEM_ALIASES.get(t, t)


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("%", " percent ")
    text = text.replace("mmol/mol", " mmol mol ")
    text = text.replace("hba1c", " hba1c ")
    text = text.replace("-", " ")
    return re.findall(r"[a-z]+\d*[a-z]*|\d+(?:\.\d+)?", text)


def content_stems(text: str) -> set[str]:
    return {light_stem(tok) for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1}


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text.lower()))


# --------------------------------------------------------------------------
# Stage 6/7 — lexical rerank + retrieval
# --------------------------------------------------------------------------
def lexical_bonus(question: str, row: dict[str, Any]) -> dict[str, float]:
    q_stems = content_stems(question)
    d_stems = content_stems(row["text"])
    meta_stems = content_stems(f"{row['section']} {row['recommendation_number']}")

    stem_overlap = (len(q_stems & d_stems) / len(q_stems)) if q_stems else 0.0
    distinctive = {s for s in (q_stems & d_stems) if len(s) >= DISTINCTIVE_STEM_MIN_LEN}
    distinctive_bonus = min(DISTINCTIVE_STEM_CAP, DISTINCTIVE_STEM_WEIGHT * len(distinctive))

    q_nums = extract_numbers(question)
    d_nums = extract_numbers(row["text"])
    number_overlap = (len(q_nums & d_nums) / len(q_nums)) if q_nums else 0.0

    meta_overlap = (len(q_stems & meta_stems) / len(q_stems)) if (q_stems and meta_stems) else 0.0

    bonus = (
        STEM_OVERLAP_WEIGHT * stem_overlap
        + distinctive_bonus
        + NUMBER_OVERLAP_WEIGHT * number_overlap
        + META_OVERLAP_WEIGHT * meta_overlap
    )
    return {"stem_overlap": stem_overlap, "number_overlap": number_overlap,
            "meta_overlap": meta_overlap, "lexical_bonus": bonus}


def retrieve_top_k(question: str, records, embeddings, model, k: int = TOP_K) -> list[dict[str, Any]]:
    query_vec = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    cosine = embeddings @ query_vec

    scored = []
    for i, row in enumerate(records):
        lex = lexical_bonus(question, row)
        scored.append({"index": i, "cosine": float(cosine[i]),
                        "final_score": float(cosine[i]) + lex["lexical_bonus"], **lex})
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    hits = []
    for rank, item in enumerate(scored[:k], start=1):
        row = records[item["index"]]
        hits.append({
            "rank": rank, "chunk_id": row["chunk_id"], "section": row["section"],
            "recommendation_number": row["recommendation_number"], "page": row["page"],
            "cosine_similarity": item["cosine"], "lexical_bonus": item["lexical_bonus"],
            "similarity": item["final_score"], "text": row["text"], "source_url": row["source_url"],
        })
    return hits


# --------------------------------------------------------------------------
# Stage 10 — extractive QA
# --------------------------------------------------------------------------
def extract_answer_span(tokenizer, model, question: str, context: str,
                         max_answer_len: int = QA_MAX_ANSWER_LEN,
                         top_k: int = QA_TOP_K_LOGITS) -> tuple[str, float]:
    inputs = tokenizer(question, context, return_tensors="pt",
                        truncation="only_second", max_length=QA_MAX_SEQ_LEN)
    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    sequence_ids = inputs.sequence_ids(0)
    context_token_indices = [i for i, sid in enumerate(sequence_ids) if sid == 1]
    if not context_token_indices:
        return "", 0.0

    ctx_start, ctx_end = context_token_indices[0], context_token_indices[-1]
    k = min(top_k, ctx_end - ctx_start + 1)
    start_candidates = (torch.topk(start_logits[ctx_start:ctx_end + 1], k=k).indices + ctx_start).tolist()
    end_candidates = (torch.topk(end_logits[ctx_start:ctx_end + 1], k=k).indices + ctx_start).tolist()

    best_score = float("-inf")
    best_start, best_end = None, None
    for s in start_candidates:
        for e in end_candidates:
            if e < s or (e - s + 1) > max_answer_len:
                continue
            score = (start_logits[s] + end_logits[e]).item()
            if score > best_score:
                best_score, best_start, best_end = score, s, e

    if best_start is None or best_start == 0:
        return "", 0.0

    input_ids = inputs["input_ids"][0]
    answer_text = tokenizer.decode(input_ids[best_start:best_end + 1], skip_special_tokens=True).strip()
    if not answer_text:
        return "", 0.0

    start_probs = torch.softmax(start_logits, dim=0)
    end_probs = torch.softmax(end_logits, dim=0)
    confidence = round(float((start_probs[best_start] * end_probs[best_end]).item()), 4)
    return answer_text, confidence


# --------------------------------------------------------------------------
# Stage 11 — rule-based, LLM-free claim decomposition
# --------------------------------------------------------------------------
NUMERIC_PAIR_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*mmol/mol\s*\(\d+(?:\.\d+)?%\)", re.IGNORECASE)


def split_into_atomic_claims(span: str) -> list[str]:
    span = span.strip()
    if not span:
        return []
    segments = [s.strip() for s in span.split(";") if s.strip()]
    final_segments = []
    for seg in segments:
        numeric_matches = list(NUMERIC_PAIR_PATTERN.finditer(seg))
        if len(numeric_matches) >= 2:
            split_point = numeric_matches[1].start()
            left = seg[:split_point].rstrip(" ,").rstrip()
            if left.lower().endswith(" and"):
                left = left[: -len(" and")].rstrip()
            right = seg[split_point:].strip()
            if len(left.split()) >= 3 and len(right.split()) >= 3:
                final_segments.extend([left, right])
                continue
        and_split = re.split(r"\s+and\s+", seg)
        if len(and_split) == 2 and all(len(p.split()) >= 3 for p in and_split):
            final_segments.extend(p.strip() for p in and_split)
        else:
            final_segments.append(seg)
    return [f for f in final_segments if f]


# --------------------------------------------------------------------------
# Stage 12a — claim quality filter
# --------------------------------------------------------------------------
GENERIC_FRAGMENT_STOPLIST = {"diet", "renal", "pregnancy", "recommendations", "recommendation", "nutritional"}
DANGLING_STARTERS = {"if", "when", "unless", "although", "while", "since", "because"}
DATE_VERSION_KEYWORDS = {"amended", "revised", "updated", "version", "edition", "reviewed", "published"}
SPECIFIC_CONTENT_PATTERNS = [
    r"\d+(?:\.\d+)?\s*mmol/mol", r"\d+(?:\.\d+)?\s*%", r"\d+\s*times?\s*a\s*day",
    r"\bat least\b", r"\bmonotherapy\b", r"\bmodified-release\b", r"\bmultiple daily\b",
    r"\binsulin\b", r"\bmetformin\b", r"\bsglt-?2\b", r"\bdpp-?4\b", r"\bglp-?1\b",
    r"\bsulfonylurea\b", r"\bpioglitazone\b", r"\binjections?\b", r"\binhibitor\b",
]


def is_pure_date_or_version(claim: str) -> bool:
    raw_tokens = [t for t in claim.lower().replace(",", " ").split() if t]
    if not raw_tokens:
        return False
    for tok in raw_tokens:
        if tok in DATE_VERSION_KEYWORDS or re.fullmatch(r"\d{4}", tok) or re.fullmatch(r"v?\d+(?:\.\d+)*", tok):
            continue
        return False
    return True


def matches_specific_content(claim_lower: str) -> bool:
    return any(re.search(p, claim_lower) for p in SPECIFIC_CONTENT_PATTERNS)


def classify_claim_quality(claim: str) -> tuple[str, str]:
    stripped = claim.strip()
    tokens = stripped.split()
    lower = stripped.lower()

    if len(tokens) <= 1:
        return "Fragment", "One-word claim -- not a checkable proposition."
    if is_pure_date_or_version(stripped):
        return "Fragment", "Pure date/version string with no substantive content."
    if lower.rstrip(".") in GENERIC_FRAGMENT_STOPLIST:
        return "Fragment", "Generic single-concept fragment with no complete proposition."
    if matches_specific_content(lower):
        return "Valid", "Specific treatment/threshold/eligibility content detected."

    stems = content_stems(stripped)
    if len(stems) < CLAIM_MIN_CONTENT_STEMS:
        return "Fragment", f"Only {len(stems)} content token(s) -- below completeness threshold."

    first_word = tokens[0].lower().strip(",.")
    has_attached_outcome = ("," in stripped) or (" then " in lower)
    if first_word in DANGLING_STARTERS and not has_attached_outcome:
        return "Fragment", "Dangling conditional clause with no attached outcome."

    return "Valid", "Passes completeness checks."


# --------------------------------------------------------------------------
# Stage 12b — heuristic verifier (similarity + lexical overlap + numeric only)
# --------------------------------------------------------------------------
def verify_claim_against_evidence(claim_text: str, top_hit: dict) -> tuple[str, str]:
    claim_nums = extract_numbers(claim_text)
    evidence_nums = extract_numbers(top_hit["text"])
    claim_stems = content_stems(claim_text)
    evidence_stems = content_stems(top_hit["text"])

    term_overlap = (len(claim_stems & evidence_stems) / len(claim_stems)) if claim_stems else 0.0
    similarity = top_hit["similarity"]
    chunk_id = top_hit["chunk_id"]

    numeric_conflict = bool(claim_nums) and bool(evidence_nums) and not (claim_nums & evidence_nums)
    if numeric_conflict:
        return "Unsupported", (
            f"Numeric mismatch: claim number(s) {sorted(claim_nums)} do not appear among "
            f"top-evidence number(s) {sorted(evidence_nums)} (chunk {chunk_id})."
        )
    if similarity < SIMILARITY_THRESHOLD or term_overlap < TERM_OVERLAP_THRESHOLD:
        return "Insufficient Evidence", (
            f"similarity={similarity:.3f} (threshold {SIMILARITY_THRESHOLD}), "
            f"term_overlap={term_overlap:.2f} (threshold {TERM_OVERLAP_THRESHOLD})."
        )
    return "Supported", (
        f"similarity={similarity:.3f}, term_overlap={term_overlap:.2f}, no numeric conflict, "
        f"against chunk {chunk_id}. Lexical/numeric consistency only -- not proof of entailment."
    )


# --------------------------------------------------------------------------
# Stage 15 — local NLI verifier
# --------------------------------------------------------------------------
def nli_probs(tokenizer, model, id2label, premise: str, hypothesis: str) -> dict:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()
    return {id2label[i]: probs[i] for i in range(len(probs))}


def classify_claim_nli(tokenizer, model, id2label, claim_text: str, hits: list[dict]) -> dict:
    best = None
    for hit in hits:
        probs = nli_probs(tokenizer, model, id2label, hit["text"], claim_text)
        strongest_label = max(probs, key=probs.get)
        relevant_score = max(probs["entailment"], probs["contradiction"])
        candidate = {
            "chunk_id": hit["chunk_id"], "entailment": probs["entailment"],
            "contradiction": probs["contradiction"], "neutral": probs["neutral"],
            "strongest_label": strongest_label, "relevant_score": relevant_score,
        }
        if best is None or candidate["relevant_score"] > best["relevant_score"]:
            best = candidate

    entail, contra, neutral = best["entailment"], best["contradiction"], best["neutral"]
    if entail >= NLI_ENTAILMENT_THRESHOLD and best["strongest_label"] == "entailment":
        status = "Supported"
    elif contra >= NLI_CONTRADICTION_THRESHOLD and best["strongest_label"] == "contradiction":
        status = "Unsupported"
    else:
        status = "Insufficient Evidence"

    return {"nli_status": status, "entailment_score": round(entail, 4),
            "contradiction_score": round(contra, 4), "neutral_score": round(neutral, 4),
            "selected_chunk_id": best["chunk_id"]}


# --------------------------------------------------------------------------
# Cached model / corpus loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading corpus and embedding index...")
def load_corpus_and_index():
    corpus_path = Path(__file__).parent / "corpus.json"
    records = json.loads(corpus_path.read_text())
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [r["text"] for r in records]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return records, embed_model, embeddings


@st.cache_resource(show_spinner="Loading local QA model (distilbert)...")
def load_qa_model():
    tok = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    model.eval()
    return tok, model


@st.cache_resource(show_spinner="Loading local NLI model (cross-encoder)...")
def load_nli_model():
    tok = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.eval()
    id2label = {int(k): str(v).strip().lower() for k, v in model.config.id2label.items()}
    return tok, model, id2label


STATUS_COLOR = {"Supported": "🟢", "Unsupported": "🔴", "Insufficient Evidence": "🟡"}

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Claim-Level Evidence Verification — NG28 Demo", page_icon="🩺", layout="wide")

st.title("🩺 Claim-Level Evidence Verification for RAG-Based Medical QA")
st.caption("A Type 2 Diabetes Case Study — NICE NG28 · pilot-scale research demo, not a clinical tool")

with st.expander("ℹ️ What this demo is, and what it is NOT", expanded=False):
    st.markdown(
        "- **Scope:** an 18-chunk excerpt of NICE NG28 (type 2 diabetes management). "
        "Not the full guideline, not a general medical QA system.\n"
        "- **Pipeline:** dense retrieval (MiniLM) + lexical rerank → extractive QA "
        "(DistilBERT) → rule-based claim splitting → two *independent* verifiers: "
        "a lexical/numeric heuristic, and a local NLI model (cross-encoder/nli-MiniLM2-L6-H768).\n"
        "- **Not clinically validated.** A 'Supported' label means the retrieved guideline "
        "text is lexically/numerically consistent (heuristic) or textually entails "
        "(NLI) the claim — it is not proof of clinical correctness or safety.\n"
        "- All models are free, local, and run on CPU. No API key, no paid service."
    )

records, embed_model, embeddings = load_corpus_and_index()

example_questions = [
    "What is the HbA1c target for type 2 diabetes?",
    "What first-line medicines should be offered if there is no relevant comorbidity?",
    "When should treatment be intensified?",
    "Should aspirin be offered if there is no cardiovascular disease?",
    "What healthy eating advice should adults with type 2 diabetes follow?",
]

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Ask a question about NICE NG28 (type 2 diabetes):",
                              value=example_questions[0])
with col2:
    st.write("")
    st.write("")
    run = st.button("Run pipeline", type="primary", use_container_width=True)

st.caption("Try: " + " · ".join(f"'{q}'" for q in example_questions[1:]))

if run and question.strip():
    hits = retrieve_top_k(question, records, embeddings, embed_model, k=TOP_K)

    st.subheader("Stage 1 — Retrieved evidence (top-3)")
    for h in hits:
        st.markdown(
            f"**Rank {h['rank']} · {h['chunk_id']}** (§{h['section']}, rec. "
            f"{h['recommendation_number']}, p.{h['page']}) — score {h['similarity']:.3f}"
        )
        st.markdown(f"> {h['text']}")
        st.caption(h["source_url"])

    qa_tok, qa_model = load_qa_model()
    top_hit = hits[0]
    span, qa_conf = extract_answer_span(qa_tok, qa_model, question, top_hit["text"])

    st.subheader("Stage 2 — Extractive answer span")
    if span:
        st.success(f"**{span}**  (confidence {qa_conf:.3f}, from {top_hit['chunk_id']})")
    else:
        span = NO_EVIDENCE_TEXT
        st.warning(span)

    st.subheader("Stage 3 — Rule-based claim decomposition (no LLM)")
    claims = split_into_atomic_claims(span) if span != NO_EVIDENCE_TEXT else []
    if not claims:
        st.info("No atomic claims to verify (insufficient evidence or single-clause span).")
    else:
        for c in claims:
            st.markdown(f"- {c}")

    if claims:
        st.subheader("Stage 4 — Dual claim verification")
        nli_tok, nli_model, id2label = load_nli_model()

        for claim in claims:
            quality, quality_reason = classify_claim_quality(claim)
            claim_hits = retrieve_top_k(claim, records, embeddings, embed_model, k=TOP_K)

            st.markdown(f"#### Claim: *\"{claim}\"*")
            if quality == "Fragment":
                st.warning(f"Quality filter: **Fragment** — {quality_reason} "
                            f"(skipped by verifiers; treated as Insufficient Evidence)")
                continue

            heur_status, heur_reason = verify_claim_against_evidence(claim, claim_hits[0])
            nli_result = classify_claim_nli(nli_tok, nli_model, id2label, claim, claim_hits)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Heuristic verifier:** {STATUS_COLOR[heur_status]} {heur_status}")
                st.caption(heur_reason)
            with c2:
                st.markdown(f"**NLI verifier:** {STATUS_COLOR[nli_result['nli_status']]} {nli_result['nli_status']}")
                st.caption(
                    f"entailment={nli_result['entailment_score']}, "
                    f"contradiction={nli_result['contradiction_score']}, "
                    f"neutral={nli_result['neutral_score']} "
                    f"(chunk {nli_result['selected_chunk_id']})"
                )

st.divider()
st.caption(
    "Research demo only · pilot-scale corpus (18 chunks) · single-annotator design · "
    "general-purpose NLI model, not clinically validated · "
    "see the project README for full limitations."
)
