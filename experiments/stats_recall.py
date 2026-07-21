# -*- coding: utf-8 -*-
"""Paired significance analysis for the recall claims (offline, no GPU).

Re-analyses the versioned per-question pass/fail vectors in ../results/*.json.
For each (reference, other) comparison over an aligned question set it reports:
  - McNemar exact (binomial) two-sided p-value  [paired test of "any difference"]
  - accuracies and delta = acc_other - acc_ref
  - Newcombe (1998, method 10) 95% CI for the paired difference p_ref - p_other
Non-inferiority is assessed against a pre-declared margin (MARGIN, 10 pp).

Reproduces the pooled numbers cited in the paper:
  single-module (N=420): p=0.69, CI on the deficit [-1.7, +3.1] pp   (Sec 5.2)
  multi-module   (N=140): p<0.001, CI [+7.4, +20.5] pp               (Sec 5.3)
  workspace      (N=120): p=1.0,  CI [-7.9, +6.2] pp                 (Sec 5.7)

Usage: python experiments/stats_recall.py
"""
import json
import os
import sys
from math import comb, sqrt

sys.stdout.reconfigure(encoding="utf-8")  # accented chars / arrows

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MARGIN = 0.10  # non-inferiority margin on recall (10 pp), declared post hoc


def load(f):
    return json.load(open(os.path.join(RESULTS, f), encoding="utf-8"))


def okvec(cond):
    return [1 if e.get("ok") else 0 for e in cond["detail"]]


def qvec(cond):
    return [e.get("q") for e in cond["detail"]]


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def newcombe10(a, b, c, d, z=1.959963984540054):
    """95% CI for theta = p1 - p2, with p1=(a+b)/n (ref), p2=(a+c)/n (other).
    a=both correct, b=ref-only, c=other-only, d=both wrong."""
    n = a + b + c + d
    if n == 0:
        return (0.0, 0.0)
    p1, p2 = (a + b) / n, (a + c) / n
    diff = p1 - p2
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    A = (a + b) * (c + d) * (a + c) * (b + d)
    phi = 0.0 if A == 0 else max(min((a * d - b * c) / sqrt(A), 1.0), -1.0)
    lo = diff - sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = diff + sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (lo, hi)


def cells_counts(ref, other):
    """Contingency counts for one aligned (ref, other) pair."""
    r, o = okvec(ref), okvec(other)
    assert len(r) == len(o), "length mismatch"
    qr, qo = qvec(ref), qvec(other)
    aligned = all(x == y for x, y in zip(qr, qo) if x is not None and y is not None)
    a = sum(1 for x, y in zip(r, o) if x == 1 and y == 1)
    b = sum(1 for x, y in zip(r, o) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(r, o) if x == 0 and y == 1)
    d = sum(1 for x, y in zip(r, o) if x == 0 and y == 0)
    return a, b, c, d, aligned


def report(title, pairs):
    """pairs: list of (file, ref_cond, other_cond). Prints per-cell + pooled."""
    A = B = C = D = 0
    print(f"\n#### {title}   (ref = joint prefill; other = restored/composed)")
    for f, ref, oth in pairs:
        d = load(f)
        if ref not in d or oth not in d:
            continue
        a, b, c, dd, aligned = cells_counts(d[ref], d[oth])
        n = a + b + c + dd
        acc_r, acc_o = (a + b) / n, (a + c) / n
        p = mcnemar_exact_p(b, c)
        lo, hi = newcombe10(a, b, c, dd)
        flag = "" if aligned else "  [!! questions not aligned]"
        label = f"{f.replace('resultados-', '').replace('.json', '')}:{ref}/{oth}"
        print(f"  {label:44s} n={n:3d} ref={acc_r*100:5.1f}% oth={acc_o*100:5.1f}% "
              f"d={ (acc_o-acc_r)*100:+5.1f}pp b={b} c={c} p={p:.3f} "
              f"CI(ref-oth)=[{lo*100:+5.1f},{hi*100:+5.1f}]pp{flag}")
        A += a; B += b; C += c; D += dd
    n = A + B + C + D
    if n == 0:
        return
    acc_r, acc_o = (A + B) / n, (A + C) / n
    p = mcnemar_exact_p(B, C)
    lo, hi = newcombe10(A, B, C, D)
    print(f"  {'POOLED':44s} n={n:3d} ref={acc_r*100:5.1f}% oth={acc_o*100:5.1f}% "
          f"d={(acc_o-acc_r)*100:+5.1f}pp b={B} c={C} McNemar p={p:.3f} "
          f"CI(ref-oth)=[{lo*100:+.1f},{hi*100:+.1f}]pp")
    verdict = "PASS" if hi <= MARGIN else "FAIL"
    print(f"  -> non-inferiority @ {MARGIN*100:.0f}pp: upper CI on deficit = {hi*100:+.1f}pp  [{verdict}]")


# ---- single-module insertion: linked (naive) vs joint prefill ----
single = []
for f in ["resultados-bateria2-qwen.json", "resultados-bateria2-llama.json",
          "resultados-bateria2-coder-E1.json", "resultados-bateria2-coder-E2.json",
          "resultados-bateria2-gemma3-4b-srv.json"]:
    for t in ("E1_corto", "E2_largo"):
        single.append((f, f"{t}_joint", f"{t}_naive"))
for f in ["resultados-bateria6-qwen.json", "resultados-bateria6-qwen-srv.json",
          "resultados-bateria6-qwen14b-srv.json", "resultados-bateria6-gemma3-4b-srv.json"]:
    single.append((f, "joint", "naive"))
report("Single-module (core battery)", single)

extended = single + [(f, "joint", "naive") for f in
                     ["resultados-e14-qwen-srv.json", "resultados-e17-qwen-srv.json",
                      "resultados-e17-coder-srv.json", "resultados-e17-qwen14b-srv.json"]]
report("Single-module (core + long-context/two-hop)", extended)

# ---- multi-module composition deficit: composed vs joint ----
multi = []
for f in ["resultados-bateria2-qwen.json", "resultados-bateria2-llama.json",
          "resultados-bateria2-gemma3-4b-srv.json", "resultados-bateria2-coder-E3.json"]:
    multi.append((f, "E3_joint2", "E3_composed2"))
for f in ["resultados-bateria4-qwen.json", "resultados-bateria4-llama.json",
          "resultados-bateria4-coder.json"]:
    multi.append((f, "joint2", "composed2"))
report("Multi-module composition (raw)", multi)

# ---- splice-k repair (best sep+splice config) vs joint ----
repaired = [(f, "joint2", "sep_splice32") for f in
            ["resultados-bateria4-qwen.json", "resultados-bateria4-llama.json",
             "resultados-bateria4-coder.json"]]
report("Multi-module composition (splice-k repaired)", repaired)

# ---- three-module workspace vs joint ----
report("Workspace (three modules)", [("resultados-bateria8-qwen-srv.json", "joint", "workspace")])
