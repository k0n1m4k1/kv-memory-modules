# Mapa de evidencia

> Una fila por afirmación: **afirmación → sección del paper → experimento → script → JSON crudo → restricciones**.
> Es la espina de trazabilidad citable. Para la historia cronológica completa de la
> investigación (incluidas hipótesis revisadas y resultados negativos) ver
> [`NOTEBOOK.es.md`](NOTEBOOK.es.md) — pero cita el paper y este fichero, no el cuaderno.
>
> English version: [EVIDENCE.md](EVIDENCE.md).

Todos los experimentos corren sobre llama.cpp release **b10068** (máquina A: binarios
oficiales win-vulkan-x64; máquina B: compilado desde fuente con CUDA), salvo E9/E19
(vLLM 0.22.1). Scripts en `experiments/`; salidas crudas en `results/` (integridad:
`results/SHA256SUMS.txt`). El recall se puntúa por coincidencia de subcadena insensible
a acentos sobre hechos sintéticos inyectados, verificado con controles sin memoria.

## Afirmaciones centrales

| Afirmación | § | Exp | Script | JSON crudo | Restricciones |
|---|---|---|---|---|---|
| Restore en frío bate al prefill (TTFT ×4,3 GPU / ×8,4 CPU), bytes agnósticos al backend | §5.1 | Fase A | `scripts/run-poc*.ps1` | `resultados.json`, `resultados-cpu.json`, `resultados-prefijo.json` | solo restore prefijo; es la base del linker |
| El prefix caching es frágil: un prefijo variable de 47 tokens reprocesa todo | §5.1 | Fase A | `scripts/run-poc*.ps1` | `resultados-prefijo.json` | motiva el linker |
| Inserción no-prefijo de un módulo = prefill conjunto (sin diferencia detectable) | §5.2 | E1, E2 | `bateria2.py` | `resultados-bateria2-*.json` | N=20/25 por celda; prefijo adversario ~1k; ver base estadística abajo |
| Se mantiene a 5,1k tokens / N=60 y a 14B | §5.2 | E8 | `bateria6.py` | `resultados-bateria6-*.json` | tarea no saturada (85 % joint); 14B en 57/60 paridad |
| La composición de dos módulos tiene un déficit de atribución del 10–20 % | §5.3 | E3 | `bateria2.py`, `bateria4.py` | `resultados-bateria2/4-*.json` | **estadísticamente robusto** (McNemar agregado *p*<0,001, N=140) |
| Splice-k (~⅓ recomputado) reduce el déficit | §5.3 | E5 | `bateria4.py` | `resultados-bateria4-*.json` | N=10–20/celda: sin potencia individual; **no** cierra del todo a escala micro (*p*=0,039); paridad limpia solo a escala workspace (§5.7) |
| Carga perezosa en mitad de sesión funciona (load-then-requestion) | §5.4 | E4a, E4b | `bateria3.py`, `bateria3b.py` | `resultados-bateria3*-*.json` | N=10; E4a (pregunta antes que evidencia) es el control negativo; E4b es el arreglo |
| El linker se extiende a híbridos de atención lineal como par afín de tamaño constante | §5.5 | E7 | `hibrido2.py`–`hibrido4.py` | `resultados-hibrido*-*.json` | paridad total 2B/4B/9B; naive ≡ afín conductualmente (afín aún no demostrado *necesario*) |
| El `.kmd` restaura en un segundo runtime (vLLM), token-idéntico, ×2,4 TTFT | §5.6/§6.4 | E9 | `fase3_vllm.py` | `fase3/resultados-fase3-*.json` | **solo restore prefijo**; no es un linker no-prefijo en vLLM |
| linked ≥ joint a 8–15k tokens desde disco | §5.6 | E10 | `bateria7.py` | `resultados-bateria7-*.json` | linked>joint tratado como observación de un modelo, no efecto general |
| La ventaja de arranque es función del cómputo (×7,0 GPU → ×27,6 CPU) | §5.6 | E12 | `e12.py` | `resultados-e12-coder-eb-*.json` | medianas de N=5 ejecuciones; restore en frío-NVMe (page cache vaciada en cada pasada), dispersión <2 %; prefill <3 %; recall 6/6 en todas las celdas |
| Paridad a 51,8k tokens (f16 y q8_0), sin acantilado de cuantización | §5.6 | E14 | `e14.py` | `resultados-e14-*.json` | el recall absoluto también cae en joint (capacidad, no linker); paridad = "sin coste añadido", no "recall alto" |
| Workspace de 3 módulos y 33,4k tokens = prefill conjunto | §5.7 | E11 | `bateria8.py` | `resultados-bateria8-*.json` | **paridad con potencia** (McNemar *p*=1,0, N=120) |
| La serialización del KV del cabezal MTP preserva la especulación | §5.8 | E13 | `e13v2.py` | `resultados-e13v2-mtp*.json` | **requiere el parche de librería `patches/`**; un modelo/cuant; restaura en posiciones de compilación (sin rebase del draft-KV) |
| La ventana deslizante enlaza sin cambios; el módulo hereda la visibilidad de la ventana | §5.9 | E20 | `bateria2.py`, `bateria6.py` (Gemma) | `resultados-bateria2/6-gemma3-4b-srv.json` | paridad exacta ≲ ventana; colapso simétrico ≫ ventana; splice-k sin probar en SWA |
| Dtype KV q8_0 = f16 gratis; q4/q5 colapsan en silencio pasado un acantilado por modelo | §6.2 | E6 | `bateria5.py` | `resultados-bateria5-*.json` | el acantilado es propiedad de modelo×dtype×longitud, prefill conjunto incluido |
| La especulación MTP en vLLM funciona out-of-the-box (65,7 vs 43,1 t/s de media, +52 %, n=3); conector+híbrido bloqueado | §6.4 | E19 | `e19.py` | `resultados-e19-*.json` | N pequeño (3 preguntas); lo relevante es el **resultado negativo**: conector + híbrido falla con "failed to convert KV cache specs to one unified type" — espejo del hueco de celdas compartidas de llama.cpp |
| La evicción + compactación en conversación es de ms y conductualmente neutra | §6.6 | E15, E15b | `e15.py`, `e15b.py` | `resultados-e15*-*.json` | full-attention 0,5–1 ms; híbridos ~5 ms checkpoint-and-replay; un modelo cada uno |
| Una conversación mayor que la ventana sobrevive sellando segmentos | §6.7 | E16 | `e16.py` | `resultados-e16-*.json` | un modelo; page-in 142 ms; recall del segmento sellado ≥ residente |
| La lectura paginada bajo presupuesto de 4k supera al contexto completo en 15 puntos | §6.7 | E18 | `e18.py` | `resultados-e18-*.json` | **tabla de páginas oráculo determinista** (selección no evaluada e2e); replica en 3 modelos (49/46/60 de 60) |
| El recall two-hop cuesta igual a joint y a linked | §7 | E17 | `e17.py` | `resultados-e17-*.json` | N=40/celda; varianza por modelo en ambos sentidos; multi-hop *entre* páginas sin probar |

## Base estadística

Las afirmaciones de recall se sostienen con tests pareados **McNemar exacto** e **IC 95 %
de Newcombe**, recalculados offline desde los vectores `detail` por pregunta con
`experiments/stats_recall.py` (sin modelo ni GPU):

| Comparación | N (pareado) | Resultado agregado | No-inferioridad @ 10 pp |
|---|---|---|---|
| Single-módulo (nuclear) linked vs joint | 420 | Δ −0,7 pp, McNemar *p*=0,69, IC [−1,7, +3,1] pp | PASA |
| Single-módulo + contexto largo/two-hop | 600 | Δ +0,2 pp, *p*=1,0, IC [−2,6, +2,2] pp | PASA |
| Multi-módulo composed vs joint | 140 | Δ −13,6 pp, *p*<0,001, IC [+7,4, +20,5] pp | FALLA (déficit real) |
| Splice-k reparado vs joint (micro) | 60 | Δ −13,3 pp, *p*=0,039 | FALLA (no cerrado a escala micro) |
| Workspace 3 módulos vs joint | 120 | Δ +0,8 pp, *p*=1,0, IC [−7,9, +6,2] pp | PASA |

Los N por celda (10–60) no tienen potencia individual — ninguna celda es significativa en
ningún sentido — así que los veredictos valen a nivel *agregado*. El margen de
no-inferioridad de 10 pp se fijó *post hoc*, no pre-registrado.

## Lo que *no* se afirma

- No hay linker no-prefijo funcional en vLLM (E9 es restore prefijo; §6.4 es una propuesta).
- No hay KV draft MTP reubicable (E13 restaura en posiciones de compilación).
- No hay número end-to-end de lectura paginada (E18 usa selector oráculo).
- No hay síntesis multi-hop entre páginas, ni caches MLA/multimodales.
- No hay contabilidad de coste total / break-even frente a recuperación textual selectiva.
