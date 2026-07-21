# Phase 1b — where do naive and affine separate BEHAVIORALLY in the hybrid linker?
#
# In naive (recurrent state := S_M) the ONLY thing lost is whatever prefix
# information lives in the GDN state; the prefix's attention KV stays intact in
# both conditions. Hypothesis: the model leans on the recurrent state mostly for
# RECENT information (just before the link point), since the constant-size folded
# state decays older content under the contractive gating. Design:
#
#   prefix = long conversation (~1.1k tok) + 8 "session facts" placed at the very
#   end -> link the module -> immediately ask about those facts (SES_Q).
#   Controls: module questions (ANC_Q) and facts from the START of the prefix
#   (INI_Q). Decay contrast: short module (294 tok) vs long (1282 tok) — with a
#   long module the contractive gating should equalize naive ~= affine, because
#   T_M·S_P ~ 0 after 1.3k update steps.
#
# Usage: python hibrido3.py <model_path.gguf> <tag>

import json
import os
import sys
import time

import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

sys.argv = sys.argv[:3]
import hibrido2 as H  # reuse the harness: probes, software rebase, battery, ChatML

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]

SESION = ("\n\nDatos operativos de esta sesión (memorízalos, son de hoy):\n"
          "- Código de verificación: 8842-ZK.\n"
          "- Sala de guerra reservada: Bravo-3.\n"
          "- Alias del despliegue en curso: 'lince'.\n"
          "- Canal temporal de la incidencia: #inc-4471.\n"
          "- Contraseña del panel de emergencia: trebol-92.\n"
          "- Runbook activo: RB-207.\n"
          "- Última build verde: 2026.07.19-b4.\n"
          "- Responsable de comunicación: Irene.\n\n")

SES_Q = [
    ("¿Cuál es el código de verificación de esta sesión?", ["8842-zk"]),
    ("¿Qué sala de guerra está reservada?", ["bravo-3"]),
    ("¿Cuál es el alias del despliegue en curso?", ["lince"]),
    ("¿Cuál es el canal temporal de la incidencia?", ["#inc-4471"]),
    ("¿Cuál es la contraseña del panel de emergencia?", ["trebol-92"]),
    ("¿Qué runbook está activo?", ["rb-207"]),
    ("¿Cuál es la última build verde?", ["2026.07.19-b4"]),
    ("¿Quién es la responsable de comunicación?", ["irene"]),
]
INI_Q = [  # facts from the START of the prefix (far from the link point)
    ("¿Quién lleva la guardia principal esta semana?", ["marcos"]),
    ("¿Qué día se reinicia automáticamente el clúster de pruebas de rendimiento?",
     ["miercoles"]),
    ("¿Qué proveedor de correo transaccional es el principal?", ["sendgrid"]),
]
ANC_Q = H.ANC_Q[:6]  # control: facts from the short module
MEM_Q = H.MEM_Q[:6]  # control: facts from the long module


def main():
    L.quiet()
    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    arch = H.meta_str(model, "general.architecture")
    sv = int(H.meta_str(model, f"{arch}.ssm.state_size"))
    hv = int(H.meta_str(model, f"{arch}.ssm.time_step_rank"))
    head_dim = int(H.meta_str(model, f"{arch}.attention.key_length", "128"))
    n_rot = int(H.meta_str(model, f"{arch}.rope.dimension_count", str(head_dim)))
    base = float(H.meta_str(model, f"{arch}.rope.freq_base", "10000"))
    scale = 1.0 / float(H.meta_str(model, f"{arch}.rope.scaling.factor", "1"))
    rope = (head_dim, n_rot, base, scale)

    pre_text = (open(os.path.join(L.DATA, "prefijo-largo.md"), encoding="utf-8").read()
                + SESION)
    prefix = L.tokenize(vocab, "<|im_start|>system\n" + pre_text)
    mods_text = {
        "corto": open(os.path.join(L.DATA, "memoria-ancla.md"), encoding="utf-8").read(),
        "largo": open(os.path.join(L.DATA, "memoria-agente.md"), encoding="utf-8").read(),
    }
    ctl_q = {"corto": ANC_Q, "largo": MEM_Q}

    out_file = os.path.join(L.RESULTS, f"resultados-hibrido3-{TAG}.json")
    results = {"model": os.path.basename(MODEL_PATH), "P": len(prefix)}

    for mname, mtext in mods_text.items():
        mem_toks = L.tokenize(vocab, mtext)
        P, M = len(prefix), len(mem_toks)
        L.log(f"===== module {mname} ({M} tok; prefix {P}) =====")
        mod = H.compile_module(model, mem_toks, hv, sv)
        r = {"M": M, "val_extraccion": mod["val"]}
        qsets = [("ses", SES_Q), ("ini", INI_Q), ("mod", ctl_q[mname])]

        # joint + diagnostics
        ctx = L.new_ctx(model)
        mem_h = L.lib.llama_get_memory(ctx)
        L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
        S_J = H.s_arrays(H.get_recr(ctx, 0), mod["info"], hv, sv)
        r["joint"] = H.battery("joint", ctx, vocab, n_vocab, mem_h, P + M, qsets)
        L.lib.llama_free(ctx)

        for cond, affine in (("naive", False), ("affine", True)):
            ctx = L.new_ctx(model)
            mem_h = L.lib.llama_get_memory(ctx)
            L.decode(ctx, prefix, 0, 0, logits_last=False)
            if cond == "naive":
                S_P = H.s_arrays(H.get_recr(ctx, 0), mod["info"], hv, sv)
                S_L = [np.einsum("hjk,hki->hji", sp, t) + sm
                       for sp, t, sm in zip(S_P, mod["T"], mod["S_M"])]
                r["diag"] = {"rel_naive_vs_joint": H.rel_err(mod["S_M"], S_J),
                             "rel_affine_vs_joint": H.rel_err(S_L, S_J)}
                L.log(f"   diag: naive {r['diag']['rel_naive_vs_joint']:.3f} | "
                      f"affine {r['diag']['rel_affine_vs_joint']:.3f}")
            dt = H.link_hybrid(ctx, mem_h, mod, P, M, hv, sv, rope, affine)
            r[cond] = H.battery(cond, ctx, vocab, n_vocab, mem_h, P + M, qsets)
            r[cond]["t_link_s"] = round(dt, 3)
            L.lib.llama_free(ctx)

        results[mname] = r
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    L.lib.llama_model_free(model)
    L.log(f"results -> {out_file}")
    for mname in mods_text:
        r = results[mname]
        L.log(f"  module {mname}: " + " | ".join(
            f"{c}: ses {r[c]['ses']['score']}/8 ini {r[c]['ini']['score']}/3 "
            f"mod {r[c]['mod']['score']}/6" for c in ("joint", "naive", "affine")))


if __name__ == "__main__":
    main()
