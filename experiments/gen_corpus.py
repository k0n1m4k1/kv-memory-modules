# E10 corpus generator — large linked memories (phase 4: the 8k/10k/15k-token
# point).
#
# Builds three MDs linked with [[refs]]: narrative bulk from long Spanish
# Wikipedia articles (CC BY-SA, attributed inside each MD) + unique synthetic
# facts injected between sections, with a FIXED seed (20260719) so the corpus
# is reproducible. Questions score ONLY the synthetic facts: the Wikipedia
# content is in the model's weights, so asking about it would not discriminate
# memory recall from parametric knowledge — a fake expedient number, owner,
# budget and status cannot be answered without the memory in context.
#
# The corpus text, note templates and questions are experimental constants and
# stay in Spanish; token counts are measured with the target model's tokenizer
# so each MD lands on its size budget.
#
# Usage (on the server): venv/bin/python gen_corpus.py
# Output: memoria-hist.md (~15k tok), memoria-tec.md (~10k tok),
#         memoria-ops.md (~8k tok), preguntas-e10.json

import json
import random
import re
import urllib.parse
import urllib.request

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

MODEL = os.path.join(L.MODELS, "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

# (slug, target token count, source articles, [[links]] to sibling memories)
MDS = [
    ("memoria-hist", 15000, ["Segunda Guerra Mundial"],
     ["memoria-tec", "memoria-ops"]),
    ("memoria-tec", 10000, ["Inteligencia artificial", "Computadora"],
     ["memoria-ops"]),
    ("memoria-ops", 8000, ["Internet", "Criptografía"],
     ["memoria-hist"]),
]

PERSONAS = ["Aitana", "Bruno", "Carla", "Darío", "Elvira", "Fermín", "Gadea",
            "Héctor", "Inés", "Jorge", "Katia", "Lorenzo", "Maider", "Néstor",
            "Olalla", "Pau", "Quima", "Ramiro", "Sole", "Telmo"]
ESTADOS = ["aprobado", "en revisión", "bloqueado", "archivado", "en curso"]


def wiki_text(titulo: str) -> str:
    """Fetch the plain-text extract of a Spanish Wikipedia article."""
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "redirects": 1, "titles": titulo})
    req = urllib.request.Request(
        f"https://es.wikipedia.org/w/api.php?{q}",
        headers={"User-Agent": "vm-llm-mem-poc/0.1 (experimento academico)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    page = next(iter(data["query"]["pages"].values()))
    txt = page["extract"]
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    # Drop trailing boilerplate sections (See also / References / ...).
    txt = re.split(r"\n== (?:Véase también|Referencias|Bibliografía|Enlaces externos) ==", txt)[0]
    return txt


def hechos(rng: random.Random, slug: str, n: int) -> list:
    """Generate n synthetic 'internal note' facts plus their question set.

    Each note fabricates an expedient id, assignee, budget, status and
    deadline, all drawn from the seeded RNG — deterministic across runs and
    guaranteed absent from any pretraining corpus. Three questions per note
    probe the three recallable fields (assignee, budget, status).
    Returns a list of (note_markdown, [(question, expected_substrings)]).
    """
    out = []
    for i in range(n):
        exp = f"EXP-{slug[-4:].upper()}-{rng.randint(1000, 9999)}"
        persona = rng.choice(PERSONAS)
        eur = rng.randint(10, 980) * 100
        estado = rng.choice(ESTADOS)
        dia = rng.randint(1, 28)
        mes = rng.choice(["enero", "febrero", "marzo", "abril", "mayo", "junio"])
        texto = (f"\n> **Nota interna {i + 1} ({exp})**: expediente asignado a "
                 f"{persona}, presupuesto {eur} euros, estado \"{estado}\", "
                 f"fecha límite el {dia} de {mes} de 2027.\n")
        qs = [
            (f"¿A quién está asignado el expediente {exp}?", [persona.lower()]),
            (f"¿Qué presupuesto en euros tiene el expediente {exp}?", [str(eur)]),
            (f"¿En qué estado está el expediente {exp}?", [estado]),
        ]
        out.append((texto, qs))
    return out


def main() -> None:
    rng = random.Random(20260719)  # fixed seed: corpus must be reproducible
    L.quiet()
    # Load the target model tokenizer-only (ngl=0) to measure token budgets
    # with the same vocabulary the experiments will use.
    model = L.load_model(MODEL, ngl=0)
    vocab = L.lib.llama_model_get_vocab(model)

    def ntok(s: str) -> int:
        return len(L.tokenize(vocab, s))

    todas = {}
    for slug, objetivo, articulos, enlaces in MDS:
        bulto = "\n\n".join(wiki_text(a) for a in articulos)
        parrafos = [p for p in bulto.split("\n\n") if p.strip()]
        n_hechos = max(12, objetivo // 900)
        notas = hechos(rng, slug, n_hechos)

        enl = " ".join(f"[[{e}]]" for e in enlaces)
        cab = (f"# Memoria del agente: {slug}\n\n"
               f"Dossier de contexto del dominio. Memorias relacionadas: {enl}.\n"
               f"El material narrativo procede de Wikipedia en español (CC BY-SA 4.0: "
               f"{', '.join(articulos)}); las notas internas son la fuente de verdad "
               f"del equipo.\n\n")

        # Interleave: append Wikipedia paragraphs and drop the i-th note as
        # soon as the running token count crosses its evenly-spaced threshold,
        # so facts are spread across the whole document (not clustered where
        # attention might favor them).
        cuerpo, qs_md, tok = [cab], [], ntok(cab)
        umbral = objetivo / (n_hechos + 1)
        i_nota = 0
        for p in parrafos:
            cuerpo.append(p + "\n\n")
            tok += ntok(p)
            while i_nota < n_hechos and tok >= umbral * (i_nota + 1):
                texto, qs = notas[i_nota]
                cuerpo.append(texto)
                tok += ntok(texto)
                qs_md.extend(qs)
                i_nota += 1
            if tok >= objetivo and i_nota >= n_hechos:
                break

        md = "".join(cuerpo)
        path = os.path.join(L.DATA, f"{slug}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        todas[slug] = {"tokens": ntok(md), "preguntas": qs_md}
        print(f"{slug}: {todas[slug]['tokens']} tok, {len(qs_md)} preguntas, "
              f"{i_nota} notas -> {path}")

    with open(os.path.join(L.DATA, "preguntas-e10.json"), "w", encoding="utf-8") as f:
        json.dump(todas, f, ensure_ascii=False, indent=2)
    L.lib.llama_model_free(model)


if __name__ == "__main__":
    main()
