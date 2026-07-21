# Shared helpers for the active experiment batteries (bateria7.py / E10,
# bateria8.py / E11, e12.py / E12). These were byte-identical copies in the
# three scripts and are unified here without any behavior change.
#
# IMPORTANT: this module imports llamalib, which lives in src/kmd. Callers
# must run their sys.path bootstrap (inserting src/kmd) BEFORE importing
# this module.

import glob
import os
import unicodedata

import llamalib as L


def norm(s: str) -> str:
    """Lower-case and strip combining accents (NFD) from a string.

    Scoring is accent-insensitive on purpose: small models often answer
    "Dario" for "Darío" or drop tildes under greedy decoding, and those are
    still correct recalls of the injected fact. Comparing on the normalized
    form keeps the metric about *memory content*, not orthography.
    """
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def battery(ctx, vocab, n_vocab, mem_h, base: int, questions) -> "tuple[int, list]":
    """Run a question battery against a prepared KV cache and score recall.

    ``base`` is the number of tokens already resident in the cache (prefix +
    memory). For each ``(question, expected_substrings)`` pair we:

    1. Append the question with the Spanish prompt scaffolding. The exact
       string is an experimental constant shared by every condition and every
       run — do not translate or reword it, or results stop being comparable.
    2. Greedy-decode up to 32 tokens (temperature 0, deterministic).
    3. Count a hit only if ALL expected substrings appear (accent-insensitive)
       in the answer. Questions target only the synthetic facts injected by
       gen_corpus.py, so the model cannot answer from parametric (pretrained)
       knowledge — a hit demonstrates recall from the KV memory itself.
    4. Roll the cache back to ``base`` with llama_memory_seq_rm. This erases
       the question and its answer, so every question is asked against the
       identical memory state and cannot leak hints into later questions.

    Returns ``(hits, detail)`` where ``detail`` is one dict per question.
    """
    hits, detail = 0, []
    for q, expected in questions:
        toks = L.tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
        L.decode(ctx, toks, base, 0)
        ans = L.greedy(ctx, vocab, n_vocab, base + len(toks), 0, 32)
        ok = all(norm(e) in norm(ans) for e in expected)
        hits += ok
        detail.append({"q": q, "answer": ans, "ok": ok})
        assert L.lib.llama_memory_seq_rm(mem_h, 0, base, -1)
    return hits, detail


def latest_kmd(kmd_dir: str, slug: str) -> str:
    """Return the most recent .kmd module for ``slug`` under ``<repo>/<kmd_dir>``.

    Module files are named ``<slug>.<suffix>.kmd``; lexicographic sort picks
    the latest build, matching the original inline glob in each script.
    """
    return sorted(glob.glob(os.path.join(L.ROOT, kmd_dir, f"{slug}.*.kmd")))[-1]
