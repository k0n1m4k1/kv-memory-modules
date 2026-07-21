# Cuaderno de hallazgos — kv-memory-modules

> English version: [NOTEBOOK.md](NOTEBOOK.md)

> **Este es el registro cronológico de laboratorio, no la ruta de lectura.** Recoge la
> investigación tal como ocurrió —incluidas hipótesis luego revisadas, callejones sin
> salida y resultados negativos— porque esa historia es en sí evidencia de cómo se
> sometieron a prueba las afirmaciones. Es deliberadamente desordenado y solo se añade.
>
> **Para conclusiones citables, usa el paper (`paper/PAPER.md`) y el mapa de evidencia
> ([`EVIDENCE.es.md`](EVIDENCE.es.md)).** Léelos primero; ven aquí solo para ver *por qué*
> una afirmación tiene la forma que tiene. Los hallazgos van numerados H1–H42; cada uno
> mapea a un experimento y a un JSON versionado en `results/`.

Proyecto: gestión de memorias de agentes LLM como módulos KV "precompilados" inyectables en el runtime de inferencia, sin re-prefill. Analogía guía: bytecode Java (`.md` → fuente, tensores KV → `.class`, prefill → compilación, inyección → classloader, re-base de posiciones + fixups → linker).

Entorno de pruebas inicial: Windows 11, Intel Arc 140V (16 GB), llama.cpp release **b10068** (binarios oficiales win-vulkan-x64), modelo **Qwen3-4B-Instruct-2507 Q4_K_M**; ampliado después con un servidor Ubuntu 24.04 / RTX 4070 Ti SUPER 16 GB (CUDA) y los modelos del Apéndice A del paper. Repos clonados para estudio de código: `llama.cpp/` y `vllm/`. Artefactos de las PoC en `experiments/` y `src/kmd/`.

---

## H1. Estado del arte en código

- **llama.cpp** serializa estado KV por secuencia: `llama_state_seq_{save,load}_file` (`include/llama.h:845-913`). Formato: magic + versión + tokens del prompt + por celda (posición absoluta, seq_ids) + por capa tensores K/V crudos (`src/llama-kv-cache.cpp:1957-2200`). Restauración: celdas físicas se reubican libremente, posiciones lógicas se restauran tal cual (`:2202`). Validación débil (solo arquitectura + tipos por capa; TODO explícito en `src/llama-context.cpp:3134`).
- **Primitiva de relocation existente pero no cableada a la carga**: `llama_memory_seq_add` (`src/llama-kv-cache.cpp:566`) + grafo K-shift (`:1909`) re-rota K con RoPE en device, incluso KV cuantizada (dequant → Hadamard → RoPE → requant). V no se toca (sin información posicional en RoPE).
- **vLLM**: prefix caching por bloques direccionados por contenido con hash encadenado `hash(padre, tokens, extra_keys)` (`vllm/v1/core/kv_cache_utils.py:596`) → la restricción solo-prefijo es estructural. Punto de plugin: `KVConnectorBase_V1` (scheduler: `get_num_new_matched_tokens` — contrato con forma de prefijo; worker: `start_load_kv`/`save_kv_layer` async por capa). Sin equivalente de K-shift expuesto.
- **Conclusión**: el "bytecode + classloader" existe en ambos; el **linker** (inserción no-prefijo + composición de módulos) no existe en ninguno.

## H2. PoC Fase A — persistencia y restauración de prefijo (ÉXITO)

Memoria MD de ~1380 tokens compilada a módulo de **194 MB** (~147 KB/token con KV f16; factor ~36.000× sobre el MD). Scripts: `scripts/run-poc.ps1`, `scripts/run-poc-cpu.ps1`.

| Métrica | GPU Vulkan | CPU (mismo módulo) |
|---|---|---|
| Restaurar en frío | 288 ms | 358 ms |
| Consulta tras restaurar | 46 tok / 218 ms | 46 tok / 671 ms |
| Línea base sin módulo | 1425 tok / 2196 ms | 1425 tok / 8668 ms |
| Mejora TTFT | ×4,3 | ×8,4 |

Respuestas idénticas (greedy) entre restaurado y línea base; recall correcto. **Portabilidad entre backends demostrada**: módulo compilado en Vulkan restaurado sin cambios en CPU (la serialización copia bytes host vía `ggml_backend_tensor_get/set` → formato agnóstico del backend).

## H3. Invalidación por prefijo (experimento natural + dirigido)

- Editar una línea del MD invalidó el módulo desde el punto de divergencia: la consulta reutilizó ~20 tokens de 1380 (experimento natural: el MD cambió entre compilación y uso).
- Prefijo variable delante (`system + fecha/hora`): el módulo restaurado quedó **completamente inutilizado** — la consulta reprocesó 1464/1464 tokens (`scripts/run-poc-prefijo.ps1`). La semántica solo-prefijo hace inservible el módulo en el escenario real de agentes (contexto variable antes de la memoria).

## H4. PoC Fase B — linker con inserción no-prefijo (ÉXITO MECÁNICO, DÉFICIT DE CALIDAD PARCIAL)

Arnés `poc/tool/linker.py` (Python + ctypes sobre `llama.dll` oficial): carga el módulo en seq auxiliar, re-basa posiciones (`llama_memory_seq_add`, +47), fusiona en la conversación (`seq_cp`/`seq_rm`) y decodifica la pregunta. El K-shift RoPE se ejecuta en el siguiente decode. Condiciones comparadas con listas de tokens idénticas:

| Condición | Pregunta integración (fecha+memoria) | Pregunta recall (URL staging) |
|---|---|---|
| JOINT (prefill conjunto, referencia) | "14 días, a las 03:00 CET" | URL correcta |
| COMPOSED (módulo insertado, linker) | "14 días, a las 03:00 CET" ✔ igual que JOINT | **falló** (repitió respuesta anterior) |
| NOMEM (control) | "1 día y 12 horas" (inventado) | URL inventada |

- La mecánica completa funciona: `seq_pos_max` verificado, generación coherente, y el contenido del módulo **es legible tras la inserción** (el "03:00 CET" solo existe en el módulo).
- El "déficit" observado en la 2ª pregunta de esta prueba resultó ser un **artefacto del encadenamiento de preguntas** (la respuesta anterior contaminaba el contexto), no un fallo del linker — desmentido con metodología limpia en **H9**.
- Coste del linker: cargar+re-basar+fusionar = 588 ms (módulo de 203 MB).

## H5. Compatibilidad por familias de modelo

- **MTP en llama.cpp**: NO compatible con save/restore — la cache del contexto MTP comparte celdas con la del objetivo (puntero `other`, `src/llama-kv-cache.h:272`) y todas las operaciones de estado hacen **no-op silencioso** (`[TAG_KV_CACHE_SHARE_CELLS]`). Especulación clásica de dos modelos SÍ compatible (el server guarda el estado del draft aparte, `tools/server/server-context.cpp:236-238`).
- **vLLM**: prefix caching convive con especulación (grupos KV propios para drafts EAGLE/MTP); en conectores el soporte es por conector (LMCache solo `deepseek_mtp` + EAGLE, TODO para el resto).
- Vetados para módulos en este punto de la investigación: MTP, M-RoPE (modelos VL; `seq_add` hace assert con `n_pos_per_embd > 1`), SWA (Gemma-3: enmascarado dependiente de posición), híbridos recurrentes (Mamba: estado solo parcial). *(Revisión posterior: todos los vetos cayeron — M-RoPE/híbridos en H17, MTP en H29, SWA en H41.)*

## H6. Ejes de ABI del módulo (qué rompe la compatibilidad)

1. **Modelo exacto** (pesos): el KV es función de los pesos; cada actualización invalida los módulos. → clave de versionado = hash del GGUF/checkpoint.
2. **Tokenizer**: los tokens guardados deben re-tokenizar igual. → hash del tokenizer.
3. **Tipo de KV cache** (`-ctk/-ctv`): independiente de la cuantización de los pesos. f16 por defecto. q8_0 reduce el módulo a la mitad con coste de calidad pequeño, PERO cuantizar V exige flash-attn activado en llama.cpp, lo que fuerza el eje 4.
4. **`v_trans` = `!flash_attn`**: un módulo guardado con FA off no carga con FA on ("incompatible V transposition", `src/llama-kv-cache.cpp:2340`). Como `-fa auto` se resuelve por GPU, el mismo comando puede producir módulos incompatibles en máquinas distintas. → fijar FA explícitamente y registrarlo en la cabecera.
5. **Backend (Vulkan/CUDA/SYCL/CPU): NO es eje de ABI** — demostrado empíricamente (H2). Matiz: diferencias numéricas de fp16 entre backends existen pero no afectan al formato ni, en lo observado, al resultado.

## H7. ¿Formato binario por MD + hash, agnóstico del runtime? (llama.cpp / vLLM / otros)

Factible como **formato de intercambio canónico** con cargadores por runtime (analogía: ELF con loaders distintos, u ONNX):

- El contenido matemático (tensores K/V por capa y token + posiciones) es función de (modelo, tokenizer, texto, política RoPE) — **no del runtime**. llama.cpp guarda exactamente eso; LMCache/CacheGen definen serializaciones equivalentes del lado vLLM.
- Cabecera propuesta: `{hash_modelo, hash_tokenizer, hash_texto_md, n_tokens, tokens[], dtype_kv, layout (v_trans/paged), flags_compilación (FA, tipos), versión_formato}`. La identidad del módulo = hash de la tupla completa → direccionable por contenido, cacheable, distribuible.
- Los layouts difieren (llama.cpp: filas contiguas por celda, V opcionalmente transpuesta; vLLM: bloques paginados de 16 tokens, normalmente bf16): el formato canónico debe ser layout-neutral (p. ej. K/V por token, f16/bf16 declarado) y el loader de cada runtime hace el scatter a su layout — coste O(bytes), igual que hoy hace `state_read_data` con su "slow path" no contiguo.
- Reserva: mezclar dtype de compilación y de ejecución (módulo f16 → runtime bf16) requiere conversión y tiene coste de calidad no medido aún.

## H8. Viabilidad de validar en vLLM en esta máquina

Difícil hoy: vLLM no soporta Windows nativo; hay WSL2 (Ubuntu) pero sin GPU NVIDIA (Arc 140V requeriría el stack XPU/oneAPI en WSL, frágil) y el backend CPU de vLLM exige compilación desde fuente. Lo validado en llama.cpp (H2) tiene equivalente funcional en vLLM vía LMCache/conectores (misma semántica solo-prefijo); lo de la Fase B (H4) **no tiene equivalente en vLLM hoy** — exigiría un conector nuevo + cooperación del scheduler (su contrato es de prefijo). Plan sugerido: validación vLLM en una máquina Linux/cloud con LMCache como paso separado.

## H9. Batería de recall — el linker naïve iguala al prefill conjunto (RESULTADO CENTRAL)

`poc/experiments/bateria.py`: 20 preguntas de recall con corrección objetiva por subcadenas (respuestas presentes solo en la memoria MD), cada una desde el mismo estado base con rollback (`seq_rm`) para aislar preguntas — la lección metodológica de H4. Escenario: prefijo variable (system + fecha, 47 tok) + módulo de 1379 tokens. Seis condiciones:

| Condición | Aciertos | Setup (ms) |
|---|---|---|
| joint (prefill conjunto, referencia) | 17/20 | 1910 |
| **naive (linker: rebase + fusión)** | **18/20** | **610** |
| drop1 (linker + descartar celda-sink) | 18/20 | 598 |
| drop4 | 18/20 | 597 |
| splice64 (warm-splice 64 tok) | 18/20 | 731 |
| nomem (control) | 5/20 | 16 |

- **Los fallos son idénticos entre condiciones y compartidos con la referencia**: "11 de 2026" en vez de "noviembre" (semánticamente correcto, corrección estricta) y la confusión con el distractor "14:30" del prefijo (también falla en joint). **No hay déficit de atención cruzada medible** en este régimen (prefijo corto, N=20, un modelo): diferencia joint/naive = ±1, ruido.
- Los fixups (drop-sink, warm-splice) no aportan nada porque no hay déficit que corregir en este régimen.
- Coste: el linker monta el contexto en 610 ms vs 1910 ms del prefill conjunto (×3,1 con una memoria de solo 1,4k tokens; la brecha crece linealmente con el tamaño del módulo, y el prefill con el cuadrático de la atención).
- **Límites de validez declarados**: prefijo corto (47 tok) — con prefijos largos e informativos el déficit podría aparecer (el módulo nunca atiende al prefijo); N=20; un modelo (Qwen3-4B); una memoria. Escalar es el siguiente experimento.

## H10. Escalado y replicación con segundo modelo (E1-E3, `poc/experiments/bateria2.py`)

Dos modelos (Qwen3-4B-Instruct-2507 y Llama-3.2-3B-Instruct, arquitecturas distintas), módulos compilados por modelo. Aciertos (joint = referencia):

| Experimento | Qwen3-4B joint/naive | Llama-3.2-3B joint/naive | Coder-7B joint/naive | nomem Q/L/C |
|---|---|---|---|---|
| E1 prefijo corto (47 tok), 20 preguntas | 17 / **18** | 20 / **19** | 20 / **20** | 5 / 1 / 4 |
| E2 prefijo largo adversario (~1k tok), 25 preguntas | 23 / **23** | 23 / **24** | 24 / **24** | 4 / 8 / 9 |
| E3 DOS módulos compuestos (1,4k + 0,3k tok), 20 preguntas | 18 / **16** | 20 / **16** | 20 / **18** | — |

(Tercer modelo añadido el mismo día: Qwen2.5-Coder-7B-Instruct Q4_K_M — punto de escala 7B, especialización coder, tercera generación arquitectónica. E1/E2: paridad perfecta. E3: el déficit de atribución persiste pero se reduce con la capacidad del modelo: -2/20 en 4B, -4/20 en 3B, -2/20 en 7B con base 20/20. Nota de coste: en E2-coder el setup naive (5,1 s) superó al joint (3,9 s) — única inversión observada; el ahorro de setup no es universal, depende de la relación I/O-módulo vs prefill-batch del hardware.)

- **La inserción de UN módulo es indistinguible del prefill conjunto en ambos modelos, incluso con prefijo largo y distractores adversarios** (±1 = ruido). El déficit teórico de atención cruzada no aparece en este régimen.
- **La composición de DOS módulos degrada ~10-20 % en ambos modelos** y el modo de fallo es *confusión de atribución entre módulos* (p. ej., a "¿qué base de datos usa Ancla?" responde "PostgreSQL 16", que es la del módulo A). Los módulos nunca se atendieron entre sí: éste es el déficit real y replicado. Candidatos a fixup: recomputación parcial de fronteras (CacheBlend) y cabeceras de ámbito más fuertes.
- Coste de setup consistentemente menor en naive (Qwen E2: 2163 vs 3376 ms; Llama E2: 1267 vs 1939 ms).

## H11. Carga perezosa de módulos enlazados (E4, `poc/experiments/bateria3.py` y `bateria3b.py`)

Escenario "classloader" completo: system de agente grande (~1k tok) + módulo de memoria general (~2,1k tok, compilado, con referencia `[[memoria-ancla]]`) enlazado tras el system; la pregunta sobre un detalle de Ancla dispara la carga del módulo `memoria-ancla` precompilado (0,3k tok) en mitad de la conversación, sin releer el MD. 10 preguntas de Ancla:

| Condición | Qwen3-4B | Llama-3.2-3B | Coder-7B |
|---|---|---|---|
| joint (todo prefilleado, referencia) | 10/10 | 10/10 | 10/10 |
| lazy naïve (módulo insertado DESPUÉS de la pregunta) | 3/10 | 7/10 | 5/10 |
| **lazy load-then-requestion** (rollback de la pregunta ~20 tok, insertar módulo, re-decodificar pregunta) | **8/10** | 6/10 | **9/10** |
| noload (control) | 1/10 | 3/10 | 3/10 |
| memoria general sobre la base lazy | 8/10 | 9/10 | 9/10 |

- La memoria general enlazada tras un system grande funciona: preguntas generales 8/10 (Qwen) y 9/10 (Llama) sobre esa base.
- **El orden importa**: con el módulo después de la pregunta, Qwen responde "no se menciona en la memoria" (ancla la respuesta en la declaración de la memoria general de que los detalles no están ahí). El fixup *load-then-requestion* — trivial con nuestro rollback, coste ~590 ms — recupera Qwen de 3 a 8. El hueco restante hasta 10/10 coincide con el déficit de atribución multi-módulo de H10/E3.
- Coste de la carga perezosa: ~420-590 ms por enlace (módulo de 31-41 MB) incluyendo re-decode de la pregunta.

## H12. Staleness y procedencia: requisito de primera clase

El linker actual confía en el archivo del módulo: si el MD fuente cambió y el módulo no se recompiló, se inserta **memoria caducada silenciosamente** (sin error; observado en vivo en H3). Cruzar módulos entre modelos falla en alto (mismatch de capas/tipos), pero dentro del mismo modelo no hay ninguna atadura módulo↔fuente↔pesos. El formato v0 (H7) debe tratar esto como el *classpath hell* de Java: cabecera con `hash(pesos) + hash(tokenizer) + hash(texto MD) + flags de compilación`, verificación en carga, y recompilación (o rechazo) si el hash del MD vigente no coincide. En los experimentos la contaminación es imposible por construcción: contexto nuevo por condición, módulos recompilados por ejecución y etiquetados por modelo, rollback con aserciones.

## H13. La frontera híbrida queda fuera del conjunto compatible (Qwen3.5 / "qwopus")

Qwen3.5 (incluidos sus destilados tipo `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`) es un híbrido de atención lineal: 3 de cada 4 capas usan Gated DeltaNet (recurrente) y opcionalmente lleva capas MTP/NextN (`llama.cpp/src/models/qwen35.cpp:8-27`). Doble incompatibilidad con el linker *tal como está implementado*: (a) el estado recurrente no tiene entradas por token que re-ubicar; (b) MTP no se serializa (H5). Como tercer modelo se eligió en su lugar Qwen2.5-Coder-7B-Instruct (compatible; ver H10-H11). **Revisión posterior: sí existe hack matemático para (a) — ver H15**; la incompatibilidad es de implementación, no de fondo. **Cierre definitivo: el linker híbrido se implementó y validó en H17** — Qwen3.5/GDN está dentro del conjunto compatible.

## H14. Fixups del déficit multi-módulo: la recomputación de frontera lo cierra (E5, `poc/experiments/bateria4.py`)

Seis condiciones × 20 preguntas (10 por módulo) × 3 modelos. El déficit vive por completo en el módulo B (el segundo enlazado); el módulo A no se degrada nunca:

| Condición | Qwen3-4B (A/B) | Llama-3.2-3B (A/B) | Coder-7B (A/B) |
|---|---|---|---|
| joint2 (referencia) | 18 (8/10) | 19 (9/10) | 20 (10/10) |
| composed2 (naïve) | 16 (8/8) | 16 (10/6) | 18 (10/8) |
| sep (separadores frescos) | 15 (8/7) | 17 (10/7) | 18 (10/8) |
| splice32 (recomputar 32 tok de B, ~11%) | 15 (8/7) | 18 (10/8) | 17 (10/7) |
| **splice96 (~33% de B)** | 15 (8/7) | **19 (10/9)** | **19 (10/9)** |
| sep+splice32 | 15 (9/6) | 17 (10/7) | 17 (10/7) |

- **Veredicto: recomputar ~⅓ del módulo insertado (splice96) recupera el nivel de la referencia** en los dos modelos con déficit claro (Llama 16→19 con joint 19; Coder 18→19 con joint 20). En Llama el efecto escala monótonamente con k (B: 6→8→9). Es la confirmación casera de la tesis de CacheBlend, implementada solo con rollback + link con `drop_k`.
- Coste del fixup: prefill de 96 tokens (decenas de ms) — despreciable frente al módulo completo.
- Los separadores de ámbito solos aportan poco (+0/+1) y no suman al splice.
- En Qwen3-4B el déficit compuesto ya estaba dentro del ruido y ningún fixup mueve la aguja.
- **Receta de producción**: módulo único → link naïve (sin coste); composición multi-módulo → splice-k con k≈33% del módulo insertado.

## H15. Sí hay hack matemático: composición afín de estados recurrentes (revisión de H13) y aclaración de FA/MTP

Pregunta que motivó la revisión: ¿por qué sería inviable con Qwen3.5 o con flash attention? ¿No hay hack matemático? Tras el análisis, son tres casos distintos:

**1. Flash attention: nunca fue inviable — es solo ABI de almacenamiento.** FA calcula exactamente la misma atención (es un algoritmo exacto, no una aproximación); lo único que cambia es el layout de V serializada (`v_trans = !flash_attn`). Un conversor offline trivial (transponer V) hace interconvertibles los módulos FA-on ↔ FA-off. Cero matemáticas; una utilidad de ~50 líneas.

**2. MTP: ingeniería pendiente, no matemáticas.** El KV de la capa MTP es atención normal; simplemente no se serializa por el refactor pendiente `[TAG_KV_CACHE_SHARE_CELLS]`. Además es degradable con gracia: una cache MTP fría solo baja la tasa de aceptación especulativa (el modelo objetivo verifica siempre) — la corrección no se ve afectada, solo la velocidad, y se re-calienta sola al decodificar.

**3. Qwen3.5/GDN: EXISTE el hack, y es elegante — superposición.** La recurrencia de Gated DeltaNet (y de Mamba/GLA en general) es **lineal en el estado**: `S_t = A_t·S_{t-1} + B_t`, con `A_t = α_t(I − β_t k_t k_tᵀ)` y `B_t` dependientes solo de la entrada del token t. Linealidad ⇒ el efecto de un módulo M entero es un **operador afín**: `S(P;M) = T_M · S(P) + S_M`, donde `T_M = Π A_t` (producto de transiciones del módulo) y `S_M` = estado final del módulo compilado desde cero. Es decir: **un módulo recurrente precompilado = el par (T_M, S_M) por capa/cabeza, de TAMAÑO CONSTANTE** (~d×d, p. ej. 128×128 f16 ≈ 32 KB/cabeza) independiente de la longitud del módulo — ¡más compacto que el KV por token! El "link" = un matmul por capa. Esta asociatividad es exactamente la que usan los algoritmos de *chunked/parallel scan* con los que estos modelos se entrenan — la matemática está probada; lo que nadie ha hecho es **externalizarla como artefacto persistente enlazable**.

Matices honestos: (a) la misma aproximación "compilado sin ver el prefijo" que ya validamos empíricamente para atención (las A_t/B_t del módulo en conjunto diferirían algo vía las capas inferiores) — exacta condicionada a las entradas del módulo, aproximada globalmente; medible con la misma metodología E1-E5; (b) el producto de transiciones es contractivo (autovalores ≤1): estable, pero decae información — es el olvido inherente de la arquitectura, presente también en operación normal; (c) llama.cpp no expone la extracción de `T_M` — requiere un paso de grafo propio (contribución de investigación real: *recurrent-state linking*); (d) en el híbrido, las capas de atención completa (1 de cada 4) se enlazan exactamente como hoy (rebase RoPE), y las GDN ni siquiera necesitan rebase: el estado recurrente es invariante a la posición absoluta por construcción — la posición ayuda, no estorba.

**Conclusión revisada**: ningún caso es matemáticamente inviable. FA = conversor de layout; MTP = serialización pendiente + degradación con gracia; híbridos lineales = composición afín (T_M, S_M), tamaño constante, con la misma clase de aproximación ya validada para atención. El "linker híbrido" es la extensión natural del paper y probablemente su contribución futura más valiosa.

## H16. `mdc` y formato de módulo .kmd v0 — implementados (`poc/tool/mdc.py`)

Formato `.kmd`: `magic "KMD0" | uint32 | cabecera JSON | blob de estado KV (llama_state_seq_get_data)`. La cabecera ata el módulo a su procedencia: `module_id = sha256(versión|hash_gguf|hash_md|dtype_kv|fa)`, más tokens, rutas, `links` (`[[...]]` extraídos del MD) y tamaño del blob. Identidad direccionable por contenido — resuelve H12 (staleness y procedencia) por construcción.

CLI con 5 verbos, todos probados:
- `compile`: semántica *make* (no recompila si `module_id` coincide); `--kv q8_0` disponible (marca FA-on en cabecera).
- `index`: compila el MD índice + todos los `[[enlazados]]` recursivamente; avisa de referencias rotas (probado con `[[memoria-runbook-pagos]]` inexistente).
- `verify` (sin cargar el modelo, hash con sidecar cacheado): detecta MD caducado y módulo de otro modelo; códigos de salida para CI.
- `info`: cabecera sin tokens.
- `link`: demo con receta H14 integrada (1º módulo naïve, siguientes con splice-k 33%) y respuesta a una pregunta. Probado: 2 módulos (2111+294 tok), link 648 ms, dato del módulo enlazado recuperado.

Mejora técnica: estado por bytes en memoria (`llama_state_seq_get/set_data`) en vez de archivos de llama.cpp — el `.kmd` es autocontenido; `llama_log_set` silenciado para salida limpia de CLI.

**Soporte completo de tipos de KV** (los 9 que admite llama.cpp, `common/arg.cpp:301`): f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl — los cuantizados compilan con FA on (la V cuantizada lo exige) y el ABI queda en cabecera; `mdc link` rechaza mezclar ABIs. Verificado en Vulkan/Arc 140V: módulo de 294 tok = 41,4 MB (f16) → 22,0 (q8_0, ×0,53) → 11,6 (q4_0 / iq4_nl, ×0,28), y **el link con rebase RoPE sobre K cuantizada funciona** (ruta dequant→Hadamard→RoPE→requant): q8_0 y q4_0 respondieron exactos ("7070, SQLite en modo WAL"), link en 254 ms.

**E6 — coste de calidad por dtype (`bateria5.py`, Qwen3-4B, batería E1, módulo 1379 tok):**

| ABI | joint | naive (módulo enlazado) | Tamaño módulo |
|---|---|---|---|
| f16 (FA off) | 17/20 | 18/20 | 203 MB |
| f16 + FA | 18/20 | 18/20 | 203 MB |
| q8_0 | 18/20 | 18/20 | 108 MB |
| q5_1 | 17/20 | 17/20 | 76 MB |
| q4_0 | 17/20 | 18/20 | 57 MB |
| iq4_nl | 18/20 | 18/20 | 57 MB |

Veredicto: **en esta batería de recall, la cuantización de la KV no cuesta nada medible ni siquiera a 4 bits** (todo en la banda 17-18, la misma del ruido entre condiciones), y —lo importante para el linker— **naive nunca cae por debajo de joint en ningún dtype**: la cuantización no interactúa mal con el rebase. Módulos 3,6× más pequeños gratis. Cautela declarada: N=20, preguntas de recall, un modelo; razonamiento multi-hop de contexto largo podría discriminar donde el recall no lo hace.

**Tamaño de módulo: NO es estable entre modelos.** `bytes/token = n_capas × (dim_K+dim_V por GQA) × bytes(dtype_KV)`; y el nº de tokens del mismo MD varía con el tokenizer. Medido: Qwen3-4B = 147,5 KB/tok (36 capas × 8 cabezas KV × 128); Llama-3.2-3B ≈ 109; **Qwen2.5-Coder-7B ≈ 57** (28 capas × 4 cabezas KV) — el 7B produce módulos ~2,6× más pequeños que el 4B (GQA manda, no el tamaño del modelo). La cuantización de PESOS (Q4/Q8 del GGUF) no afecta en nada al tamaño del módulo; solo el dtype de la KV (f16→q8_0 ≈ ×0,53). MLA (DeepSeek) lo comprimiría un orden de magnitud más.

## H17. Linker híbrido VALIDADO — Qwen3.5/GDN sin toolchain C++ (E7, `poc/experiments/hibrido2.py`)

La hipótesis H15 se implementó y validó **íntegramente en Python** sobre los binarios oficiales b10068, con Qwen3.5-2B (Q4_K_M, `unsloth/Qwen3.5-2B-GGUF`): 24 capas, 18 recurrentes (GDN, estado S = 16 cabezas × 128×128 f32 por capa, ~20 MB fijos por secuencia independientes de la longitud) + 6 de atención completa con M-RoPE.

**Hallazgos de mecánica previos (`hibrido0.py`):**
- **Fase A funciona en híbridos tal cual**: `llama_state_seq_get/set_data` serializa el estado completo (atención + recurrente) y la restauración en contexto nuevo da continuación idéntica. Los módulos *prefijo* para Qwen3.5 funcionan HOY sin nada nuevo.
- La memoria recurrente exige **continuidad estricta de posiciones** (`find_slot`, llama-memory-recurrent.cpp:638) y no admite `seq_rm` parcial → el patrón de rollback de las baterías se sustituye por **checkpoint con `seq_cp` a una secuencia auxiliar** (la memoria recurrente hace copy-on-write de celdas).
- El flag `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` permite leer/escribir **solo la parte recurrente** de una memoria híbrida — la puerta de entrada del linker.
- Formato del blob recurrente totalmente parcheable: `magic|ver | cell_count | pos,n_seq | s_trans,n_layer | capas R (conv) | capas S (DeltaNet)`, todo F32.

**Dos bloqueos y sus soluciones (ambas en espacio de usuario):**
1. `llama_memory_seq_add` está **vetado para M-RoPE** (`n_pos_per_embd()==4`, assert en llama-kv-cache.cpp:573) → el rebase RoPE en dispositivo no existe para Qwen3.5. Solución: **rebase software** — rotación NEOX de las filas K del blob en numpy (`θ_i = Δ·base^(-2i/d)`; con posiciones textuales iguales en las 4 secciones, M-RoPE ≡ NEOX, el mismo workaround que usa el K-shift de llama.cpp, PR 13870) + parcheo de las posiciones de celda en el meta (ojo: las celdas M-RoPE llevan 8 bytes extra de `llama_kv_cell_ext`). Verificado contra compilación directa en la posición destino: err máx 7,7e-2 en K f16 (doble redondeo, no error de álgebra). El linker ya **no necesita `seq_add` en absoluto**.
2. Qwen3.5-2B no responde en modo completado (EOG inmediato) → arnés ChatML con bloque `<think>` descartado.

**Extracción del par afín (T_M, S_M) por sondeo — sin tocar C++:** `S_M` = estado tras compilar el módulo desde cero. `T_M` = una sonda **por capa** recurrente: estado inicial identidad en la capa ℓ y ceros en el resto (las capas inferiores quedan idénticas a la compilación ⇒ entradas de ℓ exactas; el mapa por capa es exactamente afín dadas las entradas ⇒ vale ε=1): `T_M[ℓ] = sonda[ℓ] − S_M[ℓ]`. Con el layout de llama.cpp (fila S = [k contiguo, v, cabeza] ⇒ numpy X[h,j,i]), el link es `X_link[h] = X_P[h] @ T[h] + S_M[h]` sin transposiciones. Validación con estado aleatorio en capas 9 y 17: **err rel 4-7e-3** (un error de orientación daría O(1)). Coste de compilación ×19 (1 base + 18 sondas): 1282 tok → 2 s + 21 s de sondas. El artefacto por capa es de **tamaño constante**: T (16×128×128 f32 = 1 MB) + S_M (1 MB) por capa — 38 MB en total para el 2B, independiente de la longitud del módulo.

**E7 — batería E1 (ChatML, 4 condiciones), dos escenarios:**

| Escenario | joint | naive (S:=S_M) | affine (T·S_P+S_M) | nomem |
|---|---|---|---|---|
| 1: prefijo 45 + módulo 1282 tok (20q mem + 2q pre) | 20/20 + 2/2 | **20/20 + 2/2** | **20/20 + 2/2** | 4/20 + 1/2 |
| 2: prefijo ~1,1k + módulo 294 tok (10q mem + 6q pre) | 10/10 + 5/6 | **10/10 + 6/6** | **10/10 + 6/6** | 1/10 + 6/6 |

Distancia relativa del estado enlazado al estado joint (‖·‖₂, 18 capas): esc1 naive 0,197 / affine 0,183; esc2 naive **0,379** / affine **0,266** — el término afín recupera parte real del estado del prefijo, y más cuanto más corto el módulo (con módulo largo, el gating contractivo de GDN hace decaer `T_M·S_P` → naive ≈ affine). Link naive 110-258 ms, affine 602-648 ms, vs prefill joint 600 ms (a esta escala pequeña el ahorro es modesto; crece con el tamaño).

**Veredicto:** (a) la inserción no-prefijo en híbridos GDN **funciona con paridad total de recall** (ambas variantes, 2 escenarios); (b) la composición afín está **validada numéricamente** (extracción 5e-3, estados más cercanos al joint) aunque esta batería no la discrimina conductualmente de naive — el déficit no aflora a estas escalas (N=1 modelo, memorias ≤1,3k tok); (c) la frontera híbrida de H13 **queda cerrada**: el conjunto compatible ahora incluye Qwen3.5/GDN. Cautela: falta ver dónde naive se rompe conductualmente (prefijos con dependencias fuertes hacia el futuro del enlace, módulos muy cortos, multi-hop).

## H18. Conversor FA↔noFA implementado (`mdc.py convert`) — fase 2 del roadmap

La predicción de H15.1 ("solo ABI de layout, ~50 líneas") se confirma. `mdc.py convert <kmd>`: transpone la sección V del blob por capa entre los dos layouts (`v_trans=1`: `[gqa][celdas]` con cabecera `tipo|el|gqa`; `v_trans=0`: `[celdas][gqa]` con `tipo|fila`), voltea `flash_attn` en la cabecera, recomputa `module_id` y registra `converted_from`. Restricción de fondo: solo dtypes por elemento (f32/f16/bf16) — la V cuantizada exige FA on, no hay ABI destino.

Validación con el módulo memoria-ancla f16/Qwen3-4B: roundtrip fa0→fa1→fa0 **byte-idéntico** (transposición = permutación, sin pérdida) con `module_id` restaurado; funcional: el módulo convertido enlazado en contexto FA-on responde idéntico al original en FA-off ("7070, Nuria.", link ~250 ms). Los módulos f16 son ahora interoperables entre los dos ABIs de FA.

## H19. Fase 1b: réplica 4B, búsqueda de separación naive/afín (negativa) y una trampa metodológica con modelos *thinking*

**Réplica E7 en Qwen3.5-4B (32 capas, 24 recurrentes; `resultados-hibrido-qwen35-4b.json`): paridad total**, igual que el 2B — esc1 20/20+2/2 y esc2 10/10+6/6 en joint, naive y afín. La atribución con compilación directa en posición destino (`hibrido4.py`, sin rebase software) da también 20/20+2/2 en ambas políticas → **el rebase software es conductualmente inocuo** (su err numérico f16, 1,9e-1 máx en el 4B, no toca el recall) y el término afín tampoco daña.

**Trampa metodológica descubierta (y documentada para el paper): el presupuesto de generación con modelos *thinking*.** La primera pasada del 4B mostró una degradación aparente (naive 16/20, afín 11/20, con afín < naive) que resultó ser 100% artefacto: todas las respuestas falladas estaban **vacías** — el 4B entraba en `<think>` largo, agotaba los 64 tokens de generación y el filtro dejaba cadena vacía. Con `<think>\n\n</think>\n\n` pre-rellenado en el turno del asistente (modo no-think) la paridad es total. Moraleja: en baterías con modelos híbridos de razonamiento, fijar el modo de thinking o presupuestar su longitud; una condición "más rara" para el modelo puede pensar más y morir por el cap de tokens, simulando un déficit de calidad que no existe. (La pasada con artefacto queda en `resultados-hibrido-qwen35-4b-think-artefacto.json`.)

**Búsqueda de separación conductual naive vs afín (`hibrido3.py`, 2B): resultado negativo limpio.** Diseño: 8 "hechos de sesión" colocados justo antes del punto de enlace (máxima recencia → máxima dependencia esperada del estado recurrente), módulo corto (290) y largo (1282), preguntas inmediatas. Resultado: **8/8 en ses, 3/3 en inicio y 6/6-5/6 en módulo para joint, naive y afín por igual** — en recall factual, las capas de atención (1 de cada 4) cubren íntegramente lo que naive pierde del estado recurrente del prefijo. El único fallo (5/6, módulo largo) es compartido por ambas políticas y es un **déficit de atribución prefijo↔módulo** (el modelo responde con el hecho análogo del prefijo — "viernes 22:00" del entorno de demos — en vez del hecho del módulo — "lunes 03:00" de staging): la misma clase que E3/H14, candidato a splice-k, independiente de la política de estado.

**Conclusión operativa**: para GDN híbridos, **naive (S:=S_M) es la política de producción** — más barata (95-680 ms vs 650-2000 ms), sin matrices T en el módulo (mitad de tamaño), y conductualmente indistinguible del afín y del joint en todo lo medido. El afín queda validado como mecanismo (estados más cercanos al joint: 0,289 vs 0,410 con módulo corto) y como reserva para los casos donde no hay atención que compense: **modelos puramente recurrentes (Mamba/RWKV sin capas de atención)** serían el banco de pruebas donde el afín es la única forma de conservar el prefijo — anotado como extensión natural.

**`mdc`/`.kmd` con soporte híbrido (maquinaria compartida extraída a `poc/tool/hyblib.py`)**: `compile` detecta la arquitectura (claves ssm del GGUF), ejecuta las sondas y adjunta las matrices T al blob (cabecera `hybrid` con forma del estado, `rope` para el rebase software — ojo: en el 2B head_dim=256 con n_rot=64, hay que leerlo del GGUF — y validación de extracción); `link` re-basa por software y aplica `--recr naive|affine` con arnés ChatML no-think; `convert` rechaza híbridos explícitamente. Módulo ancla-2B: 42,6 MB = 23,8 estado + 18,9 T (las T se omitirían en un módulo "naive-only": política por defecto razonable).

## H20. E8 — escala ×3,7 y N=60: la paridad aguanta y el ahorro de setup se ensancha a ×7 (`bateria6.py`)

Primer paso de la fase 4 (endurecer evidencia). Memoria sintética determinista (seed fija, `data/memoria-grande.md`, generada por `bateria6.py`): 40 microservicios con 10 atributos únicos cada uno + 20 incidencias, **5114 tokens** (~×3,7 la memoria de E1-E6); prefijo adversario de 1046 tok; **N=60 preguntas** (~×3 el N anterior); Qwen3-4B, f16, arnés crudo con rollback.

| Condición | Recall | Setup |
|---|---|---|
| joint (prefill completo, 6160 tok) | 51/60 | 12,13 s |
| **naive (prefill 1k + link del módulo)** | **50/60** | **1,7 s (×7,1)** |
| nomem (control) | 0/60 | — |

- **La paridad del linker aguanta la escala**: 50 vs 51 (±1 = ruido); 8 de los ~10 fallos son idénticos entre condiciones, y los 3 no compartidos son el mismo tipo de pregunta (día de despliegue, el atributo más confundible entre 40 servicios) en servicios distintos — ruido de muestreo, no firma de inserción.
- **La tarea ya no está saturada** (joint 85%, antes ~90-100%): hay margen para que un déficit aflorara, y no aflora.
- **La ventaja de setup crece con la escala como predice la teoría** (O(bytes) vs O(compute)): ×3,1 con memoria de 1,4k tok (H9) → **×7,1 con 5,1k tok** (12,13 s → 1,7 s). El control nomem 0/60 confirma que ningún hecho es adivinable.
- Módulo de 754 MB (5,1k tok × 147,5 KB/tok ✓), compilado en 8 s. A esta escala el peso del módulo empieza a importar operacionalmente (compresión/q8 de la fase 5 pasan de "nice" a necesarios: en q8_0 serían ~400 MB, en q4 ~210 MB, gratis en recall según E6).
- Pendiente de la fase 4: punto a 10-50k tok (ampliar generador), modelo 13B+, workload multi-hop, N≥100.

## H21. E9 — fase 3 primer punto: restore-vs-prefill replicado en vLLM (conector KV nativo)

Servidor Ubuntu 24.04 (RTX 4070 Ti SUPER 16 GB, 20 núcleos, 62 GB RAM), venv Python 3.14.6, vLLM 0.22.1 + torch 2.11.0+cu130. LMCache descartado en este venv (declara `requires-python <3.14`); se usó el conector nativo de disco `ExampleConnector` (el antiguo `SharedStorageConnector`), que es además mejor testigo del contrato scheduler-connector v1. Script `fase3_vllm.py` (copia en el servidor), resultados en `results/fase3/`.

- Diseño: memoria-grande.md (E8, 5,1k tok) + pregunta, Qwen3-4B-Instruct-2507 bf16, `enforce_eager`, 3 procesos independientes (store / restore / baseline), 5 preguntas, greedy.
- **TTFT medio: baseline (prefill completo) 0,142 s → restore (KV desde disco) 0,060 s = ×2,4**; store (prefill + volcado) 0,195 s (+37% sobre baseline, coste de compilación).
- **Recall 5/5 en las tres condiciones y respuestas de restore == baseline token a token** (greedy): la restauración es conductualmente sin pérdida, replicando la Fase A de llama.cpp en el segundo runtime.
- La ganancia es menor que en el portátil (×4,3-8,4) porque esta GPU prefillea 5,1k tok en 0,14 s: coherente con la ley O(bytes) vs O(compute) de E8 — la ventaja crecerá con memorias 10-50k y modelos mayores. Caveat: el restore leyó ~1,4 GB de safetensors probablemente desde page cache (tier RAM caliente de ARCHITECTURE.md); falta el punto NVMe frío (`drop_caches`).
- Hallazgo para el RFC: `ExampleConnector` indexa por **hash del prompt completo** (alineado a bloque) — no hay reutilización entre preguntas distintas sobre la misma memoria, ni por prefijo. El hueco no-prefijo que ataca nuestro linker ni siquiera tiene soporte de prefijo genérico en el conector nativo; LMCache lo palía con hash por bloques, pero sigue siendo solo-prefijo (H8).

## H22. E10 — corpus 8k/10k/15k con módulos q4 desde disco: el linked iguala o SUPERA al joint

Corpus generado (`gen_corpus.py`, seed 20260719, en el servidor): 3 MD enlazados con `[[refs]]` — memoria-hist 15.171 tok, memoria-tec 10.219 tok, memoria-ops 8.035 tok. Bulto narrativo de Wikipedia ES (CC BY-SA, atribuido) + notas internas sintéticas inyectadas cada ~900 tok; las 120 preguntas puntúan SOLO sobre las notas (el contenido Wikipedia está en los pesos; control nomem = 0/120 confirma que los hechos fake no se contaminan). Compilación `mdc index` siguiendo enlaces: q4_0 = 1,39 GB vs f16 = 4,93 GB (×3,55). Batería `bateria7.py` (Qwen3-4B, cache q4+FA en las tres condiciones, prefijo adversario 1,1k, lectura del `.kmd` desde disco DENTRO del cronómetro de setup):

| MD | joint | linked | nomem | setup joint→linked |
|---|---|---|---|---|
| hist 15k | 42/48 | **46/48** | 0/48 | 2,47 s → 0,83 s (×3,0) |
| tec 10k | 33/36 | **35/36** | 0/36 | 1,53 s → 0,68 s (×2,2) |
| ops 8k | 35/36 | 35/36 | 0/36 | 1,12 s → 0,55 s (×2,0) |

- Total: joint 110/120, **linked 116/120**. El linked supera al joint en los dos MD grandes, direccionalmente consistente. Hipótesis: en el joint los tokens de la memoria atienden también al prefijo adversario durante el prefill (dilución); el módulo se compila aislado (solo se atiende a sí mismo) y se rebasa después — a 15k la "compilación aislada" parece proteger el recall. Necesita réplica antes de afirmarlo (¿artefacto de esta batería?), pero abre una tesis fuerte: el módulo no es solo más barato, puede ser MEJOR memoria.
- q4 en cache y módulo a 15k tokens: sin coste medible (consistente con E6 a 1,4k).
- La ventaja de setup crece con el tamaño del MD incluso en GPU rápida (×2,0 → ×3,0), pese a incluir la lectura de 333-629 MB de disco.

## H23. Transporte entre máquinas del módulo híbrido GDN: funciona

`memoria-ancla.22cffd7ae6f6.kmd` (Qwen3.5-2B, compilado en el portátil Windows con sondas por capa) copiado al servidor Linux/CUDA y enlazado allí con `mdc link --recr naive` sobre el GGUF byte-idéntico: link 58,7 ms, respuesta exacta multi-hecho ("puerto 7070... SQLite... modo WAL"). Con esto el mapa de portabilidad queda completo: el artefacto `.kmd` (atención y GDN) es portable entre SO/backend/GPU; la equivalencia es conductual (paridad de recall exacta, E8), NO bit a bit (los kernels Vulkan/CUDA no producen floats idénticos — no afirmar "byte stable" numérico). Entre runtimes (vLLM) se transporta el contrato, no el formato (E9/H21).

## H24. E11 — workspace multi-módulo a 33k tok y lazy a escala

`bateria8.py` (Qwen3-4B, cache q4+FA, ctx 40960, módulos q4 de disco, receta H14: hist link directo + tec/ops con splice-k 33 %):

- **workspace 106/120 vs joint 105/120** — la composición de 3 módulos (15k+10k+8k) es indistinguible del prefill conjunto de 33,4k tok. El déficit de atribución de E3 sigue resuelto a ×20 la escala original.
- **Lazy (load-then-requestion) a escala: pre 0/6 → post 6/6**, carga del módulo tec (10,2k tok, 424 MB) en 1,13 s. El pre=0 confirma aislamiento entre memorias (sin fuga).
- Setup: workspace 8,18 s vs joint 9,25 s (×1,13): en GPU rápida el splice-k (6,1k tok recomputados) + lectura de 1,39 GB + pegado en VRAM se comen casi toda la ventaja. El multi-módulo compone por CALIDAD y por lazy/granularidad, no por setup, en hardware potente; en doméstico la aritmética cambia (§5.6).
- El efecto linked>joint de E10/H22 no se replica claramente en el workspace combinado (106≈105) — la réplica per-MD con Coder-7B está en H26.

## H25. La cache KV q4 NO es gratis universalmente: colapso de Qwen2.5-Coder-7B a 8k+

Al replicar E10 con Coder-7B sobre módulos q4_0, TODAS las condiciones (incluido joint, que no usa módulos) dieron 0: generación degenerada ("de de de..."). Diagnóstico controlado (joint puro, memoria-ops 8k, mismas 8 preguntas): **f16 7/8 vs q4_0+FA 0/8**. E6 midió "q4 gratis" a 1,4k tok en Qwen3-4B; E10 lo confirmó a 15k en Qwen3-4B; pero Coder-7B colapsa con cache q4 ya a ~9k. Conclusión: la tolerancia a KV cuantizada es **dependiente de modelo Y de escala** — el eje kv_dtype del ABI necesita validación por modelo (un `mdc verify --recall`-style check en producción). La réplica Coder se relanzó con f16 (`replica-coder-f16.log`). Nota: el fallo del primer intento de réplica fue doble — módulos compilados para otro modelo (abortó bien por hash) y, tras recompilar, este colapso q4.

Barrido de dtype en el mismo banco (Coder, 9k, joint): **f16 7/8 = q8_0 7/8; q5_1 0/8 (números plausibles pero erróneos — fallo silencioso); q4_0 0/8 (degenerado)**. Política de producción: **q8_0 por defecto** (mitad de tamaño que f16, sin pérdida medida); q4 solo validado por modelo×longitud. q5_1 es el caso peligroso: falla sin síntomas visibles.

## H26. Réplica del corpus con Coder-7B (f16): paridad; el "linked>joint" de H22 no generaliza

`bateria7` con Qwen2.5-Coder-7B y módulos f16 (tras H25): hist joint 48/48 vs linked 48/48; tec 36 vs 34; ops 34 vs 35 — **total 118 vs 117, paridad**. El efecto linked>joint de H22 no aparece (Coder va a techo y no deja margen): se queda como observación específica de Qwen3-4B bajo prefijo adversario, no como efecto general — así se refleja en el paper. Consolidado: linked ≥ joint en 2 modelos × 3 tamaños de MD; **la ventaja de setup crece con el tamaño del modelo incluso en GPU rápida** (hist 15k: ×5,1 en el 7B vs ×3,0 en el 4B) — coherente con §5.6 (más cómputo por token = más ventaja O(bytes)). nomem 3/48 y 2/36: algún acierto por adivinación en preguntas de estado (1 de 5 valores posibles), sin impacto.

## H27. E12 — el ratio prefill/restore según régimen de cómputo (la tesis §5.6, medida)

`e12.py` (Coder-7B, módulo f16 de 870 MB del MD hist 15,2k, lectura de disco dentro del cronómetro, mismas 6 preguntas fake): barrido de `ngl` en la misma máquina (RTX 4070 Ti S + 20 núcleos). Los valores son **medianas de una re-medición N=5** con restore en **frío** (`posix_fadvise(DONTNEED)` antes de cada lectura); superan la lectura única inicial (~0,9 s de restore, ×21–×6,4):

| Régimen | Prefill | Restore (frío NVMe) | Ratio |
|---|---|---|---|
| ngl=0 (solo CPU) | 18,9 s (804 t/s) | 0,69 s | **×27,6** |
| ngl=12/28 (offload) | 13,7 s (1110 t/s) | 0,72 s | ×19,0 |
| ngl=99 (GPU plena) | 5,5 s (2781 t/s) | 0,78 s | ×7,0 |

Recall 6/6 en TODAS las celdas y en las 5 pasadas. Dispersión del prefill <3 %, del restore frío <2 %. El restore es plano (copia de bytes limitada por relocación, sube ligeramente hacia GPU plena al subir más estado al dispositivo); el prefill escala con el cómputo → el ratio crece justo donde el cómputo escasea. **Corrección importante**: la extrapolación "~18 min de prefill" derivada de los 13,9 t/s del smoke del 27B era errónea — aquel número salió del modo interactivo del nuevo llama-cli, no de un prefill por lotes; un CPU de 20 núcleos moderno prefillea un 7B a ~800 t/s. Los "minutos vs milisegundos" requieren CPU débil (portátil), modelos mayores o memorias más largas — la dirección de escala es inequívoca y multiplicativa en los tres factores, pero las cifras dramáticas hay que medirlas, no extrapolarlas. El paper §5.6 queda corregido con estos números. (La batería del 27B se descartó: los puntos ya medidos cubren el argumento.)

## H28. Compresión genérica de módulos: NO compensa (banco sobre blobs reales)

Hipótesis evaluada: ¿compensa gzip en disco + descompresión al cargar? Medido sobre módulos reales (portátil): módulo f16 311 MB (Qwen3-4B) y módulo híbrido f16+f32 42 MB (Qwen3.5-2B).

| Método | f16 311 MB | híbrido 42 MB | comp / descomp |
|---|---|---|---|
| zlib-1 | 105 % (¡expande!) | 101 % | 105 / 346 MB/s |
| zlib-6 (gzip) | 92,8 % | 86,9 % | 47 / ~300 MB/s |
| shuffle+zlib-1 | 105 % | 100 % | — |
| lzma-1 (cota lossless) | 91,2 % | 85,8 % | 9 / 27 MB/s |

Los tensores KV f16 son ~ruido de alta entropía: lossless genérico rasca 7-13 % y la descompresión (~300 MB/s) multiplicaría ×3-5 el tiempo de restore frente a NVMe (1-3 GB/s). En q8/q4 sería aún peor (más entropía). **Veredicto: la palanca de almacenamiento es el eje dtype (q8_0 = 53 %, q4_0 = 28 %), no la compresión**; si hiciera falta más, el camino es lossy à la CacheGen (deltas entre tokens + codificación aritmética), y zstd solo para transporte por red lenta. Script: `poc/experiments/comp_bench.py`.

## H29. Soporte MTP: investigación completa + diseño del parche (E13 en curso)

Decisión de alcance: implementar el soporte MTP (es un problema de ingeniería, no de investigación). Investigación de código (b10068) + experimento de caracterización con `Qwopus3.5-4B-Coder-MTP` (GGUF local, arch `qwen35`, 33 bloques = 32 + capa MTP embebida, HÍBRIDO GDN):

**Mapa del código (el hueco exacto):**
- La cache del contexto MTP comparte celdas con la del target: **mismo objeto** (`v_cells_impl(other ? other->v_cells_impl : ...)`, `llama-kv-cache.cpp:84`). Algunas capas pueden compartir tensores con la cache madre (`share && other`, `:174`) — solo las capas propias (la cabeza MTP) necesitan serialización.
- `state_write`/`state_read` hacen **no-op incondicional** si `other` (`:1959`, `:2029`). Todas las seq-ops también (`seq_rm` devuelve `true` fingiendo éxito, `:381`). El K-shift on-device está vetado (`GGML_ASSERT(!other)`, `:1911`) → para relocation de la capa MTP vale nuestro **rebase software** (ya implementado).
- El server: el prompt-cache RAM ya intenta guardar el estado del draft (`server-context.cpp:223-238`) pero para MTP recibe un blob vacío (no-op silencioso); el save/restore de slots a disco solo serializa `ctx_tgt` (`:2531/2569`). El contexto MTP lo crea el server con `cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP` + `ctx_other` (`:1093`) — campos que nuestra `ContextParams` de llamalib ya tiene.
- Activación: `--spec-type draft-mtp` (+ `--spec-draft-n-max 2`); timings del server exponen `draft_n`/`draft_n_accepted`.

**E13 (caracterización, binarios stock, Arc/Vulkan): INVALIDADO CON HALLAZGO.** Baseline y restored dieron aceptación IDÉNTICA (78/122 = 0,639) porque `prompt_n=1327` en ambas: el server **descartó el estado restaurado y re-prefilleó**. Control sin especulación: igual (prompt_n=1308). Causa: el modelo es híbrido — el server no puede extender un estado recurrente restaurado (sin rollback parcial ni checkpoints del fichero) y degrada en silencio a re-prefill. **Hallazgo colateral importante: el slot-restore del server está roto de facto para híbridos** (funciona, pero no sirve de nada); nuestro harness ctypes sí mantiene la continuidad (H-hibrido1, continuación token-idéntica). +1 argumento para el linker en híbridos. Script: `scripts/e13-mtp.ps1`; resultados `results/resultados-e13-mtp.json`.

**Parche ESCRITO: `patches/llama.cpp-b10068-mtp-kv-state-shared-cells.patch`** (+119/−18 líneas sobre `llama-kv-cache.{h,cpp}`; ver `patches/README.md`). Implementa 1-2 del diseño de abajo; verificado con `git apply --check` sobre b10068 prístino y dejado aplicado en `third_party/llama.cpp`. Decisiones: flag `kv_layer::shared` marcado en el constructor; write reutiliza el escaneo de celdas tal cual (las celdas compartidas son válidas) y salta capas prestadas; read usa `state_read_meta_shared` nuevo (localiza por (pos,seq) en single-seq, por orden del blob en whole-cache; sin alocar ni `clear()` — el camino de error tampoco limpia celdas ajenas). El punto 3 (slots de disco del server) queda FUERA del parche: el RAM prompt-cache ya llama al draft y para híbridos el restore del server es inútil de todos modos → E13v2 irá por arnés ctypes. Caminos no compartidos byte-idénticos al original. Pendiente: compilar y validar (diseño original a continuación):
1. `llama_kv_cache::state_write` con `other`: construir `cell_ranges` desde las celdas compartidas (misma lógica que el camino normal — las celdas son válidas), y escribir meta + datos SOLO de las capas propias (no compartidas con `other`).
2. `state_read` con `other`: NO alocar (las celdas ya las restauró la cache madre — orden: target primero); localizar cada celda por (pos, seq) en las celdas compartidas y hacer scatter de las filas K/V de las capas propias sobre esos índices. Error explícito si las posiciones no existen (= target no restaurado antes).
3. Server: el camino de disco (`handle_slots_save/restore`) debe llamar también a `ctx_dft` como ya hace el prompt-cache RAM (`:236-238`); orden target→draft en restore.
4. `mdc`: sección opcional `mtp` en la cabecera `.kmd` (blob del contexto MTP); compile debe asegurar que la capa MTP procesó los tokens del módulo (verificar cuándo puebla su KV el `impl_draft_mtp` — probable prefill propio del draft); link = scatter + rebase software de su K.
5. E13v2 tras el parche: mismo guion, esperando aceptación restaurada ≈ baseline; y E13-ctypes para aislar del server (usar checkpoints `seq_cp` como en híbridos).

Convergencia bonita: el modelo de prueba es híbrido+MTP → valida a la vez nuestro linker GDN y el parche MTP. Nota: leer `AGENTS.md` de llama.cpp antes de escribir el parche (norma del repo) — objetivo final: PR upstream (el TODO `[TAG_KV_CACHE_SHARE_CELLS]` es de los propios mantenedores; vLLM/LMCache ya serializan KV de capas MTP, precedente directo).

**E13v2 (servidor CUDA, parche compilado): VALIDADO.** Script `poc/experiments/e13v2.py` (4 fases, cada una con server frío). Dos correcciones metodológicas sobre E13: guardar en la frontera EXACTA (`n_predict=0`, no 1) y prompts como **arrays de tokens** cortados por el prefijo común exacto (con strings, el `\n` final de la memoria se fusiona con el `\n\n` de la pregunta y el prefijo restaurado deja de casar → re-prefill silencioso; el split real cayó en 1284 de 1285). Resultados (Qwopus3.5-4B-Coder-MTP Q6_K, memoria 1284 tok, pregunta 43 tok, 120 gen, temp 0):

| Fase | prompt_n | aceptación MTP | gen t/s |
|---|---|---|---|
| A baseline (prefill completo) | 1327 | 0,690 (69/100) | 203,7 |
| C restore target+draft (parcheado) | **43** | **0,722 (70/97)** | 207,2 |
| D restore sin `.draft` (control causal) | 43 | **0,587 (64/109)** | 185,2 |

Lecturas: (1) la aceptación sobre KV restaurado ≈ baseline (0,72 vs 0,69) → **el parche restaura el estado MTP correctamente**; (2) sin el blob draft la aceptación cae a 0,587 y el throughput ×0,89 → causalidad demostrada (y el server avisa: "no draft state ... will degrade", con respuesta correcta igualmente — degradación elegante, el target verifica); (3) blob draft 5,0 MB vs 90,4 MB target (1 capa de 33 + proporción por GQA/dims de la cabeza MTP). Respuesta correcta (staging + refresco + MTX-4907) en todas las fases. Réplica (segunda pasada completa): resultados idénticos dígito a dígito (temp 0, greedy).

**REVISIÓN del hallazgo colateral de E13**: el slot-restore del server para híbridos NO está roto de fondo — con frontera exacta a nivel de token, el server **extiende el estado recurrente restaurado sin re-prefill** (prompt_n=43). Lo que sí es real: es FRÁGIL — cualquier desajuste del prefijo (un token generado de más al guardar, una fusión BPE en la frontera) fuerza re-prefill completo y SILENCIOSO, porque los slot files no serializan checkpoints y el camino "hybrid/recurrent sin checkpoint" resetea (`server-context.cpp:3332`). Regla práctica para módulos/slots con híbridos: guardar con `n_predict=0` y continuar con prompts tokenizados sobre el prefijo exacto (o verificar el prefijo vía `/tokenize` antes de confiar en el restore).

Parche experimental del server (slots de disco con `.draft`): `patches/llama.cpp-b10068-server-slots-draft-state.patch` — save escribe `<slot>.bin.draft` con `llama_state_seq_save_file(ctx_dft,...)`, restore lo carga tras el target (orden obligatorio) y degrada con WARN si falta.

## H30. MTP por backend: vLLM lo resuelve por diseño; el grosor del "driver" depende de la API, no del formato

Extensión natural: ¿y vLLM y otros backends? Verificado en el código de `third_party/vllm` (v1):

**vLLM: el KV del cabezal MTP es ciudadano de primera clase.** Las capas draft MTP/EAGLE se registran como grupos normales del gestor unificado de KV cache — `KVCacheGroupSpec.is_eagle_group` (`vllm/v1/kv_cache_interface.py:956`) — y el coordinador de prefix-cache las trata junto al resto (`vllm/v1/core/kv_cache_coordinator.py:508-515`, `SpecGroup`). Consecuencias:
1. El KV MTP vive en el mismo pool paginado, direccionable por `layer_name` → la **KVConnector API** (nuestra puerta de la Fase 3) lo ve sin parchear el núcleo. La sección `mtp` del `.kmd` se mapea a esas capas extra desde el conector.
2. Convención a respetar por el linker: el KV de EAGLE/MTP está **desplazado una posición** respecto al target — por eso vLLM descarta el último bloque de los grupos draft en aciertos de prefix-cache ("EAGLE last-block drop", documentado en ese mismo código). La sección `mtp` del formato debe registrar esta convención de posiciones.

**Contraste arquitectónico (refuerza el paper):** mismo problema, dos filosofías. vLLM modela el KV draft como estado gestionado (grupos + paginación + prefix-cache) → entrada por API, driver fino. llama.cpp lo modela como segundo `llama_context` soldado al target (celdas compartidas, serialización no-op, H29) → 30-60 líneas de C++. El `.kmd` es neutro; **el grosor del driver por backend lo determina cuán abierta es la API del backend, no nuestro diseño**.

**Otros backends (nivel arquitectura, sin verificar en código salvo vLLM):**
- ollama / LM Studio / llamafile: envuelven llama.cpp → heredan el parche al subir upstream.
- SGLang: prefix-cache RadixAttention + soporte EAGLE/MTP; estructura análoga a vLLM → previsiblemente adaptador, sin parche de núcleo (hipótesis).
- TensorRT-LLM: soporta MTP pero el KV vive en un engine compilado (cerrado) → el caso difícil; fuera de alcance de la PoC.

**Propiedad de diseño a explicitar en el formato:** la sección `mtp` es carga útil **opcional y neutra para la corrección**. El MTP es solo cabezal borrador y el target verifica cada token especulado: un backend que no sepa inyectarla carga el módulo y responde correcto, solo pierde aceptación especulativa sobre posiciones restauradas (exactamente lo medido en E13). Degradación elegante, no incompatibilidad.

La validación empírica llegó en E19 (H39): la especulación MTP funciona out-of-the-box; el camino conector para híbridos está bloqueado por un gap del motor.

## H31. Punto 14B: paridad exacta y la ventaja de setup crece con el tamaño del modelo (E8 sobre Qwen3-14B)

Punto de escala de modelo. Batería E8 estándar (memoria
sintética 5.114 tok, prefijo adversario 1.046 tok, N=60) sobre **Qwen3-14B Q4_K_M**
(`unsloth/Qwen3-14B-GGUF`, 8,6 GB), servidor CUDA, GPU completa (`VMLLM_NGL=99`):

| Condición | Recall | Setup |
|---|---|---|
| joint (prefill conjunto) | **57/60** | 2,87 s |
| naive (módulo enlazado) | **57/60** | 0,82 s (×3,5) |
| no-mem (control) | 0/60 | — |

Lecturas:
1. **Paridad exacta** joint = linked (57/57), mejor puntuación absoluta que el 4B
   (51/50): el déficit de inserción sigue sin aparecer al subir de escala de modelo —
   consistente con E3 (el déficit multi-módulo *se reduce* con la capacidad).
2. **La ventaja de setup crece con el tamaño del modelo a igual hardware**: ×3,5 en
   el 14B vs ×1,7 en el 4B sobre la misma RTX 4070 Ti S (el prefill es O(cómputo) y
   crece con la profundidad/anchura; el restore sigue siendo O(bytes)). Tercera
   dimensión del argumento §5.6 (junto a régimen de cómputo y longitud de memoria).
3. Módulo de 837,9 MB f16 (~168 KB/token: 40 capas GQA 8) compilado en 2,1 s.

Resultado en `results/resultados-bateria6-qwen14b-srv.json`; modelo en
`scripts/models.txt`. Integrado en el paper: Apéndice A y §5.2.

## H32. Punto 50k (E14): linked ≥ joint también a 50k, sin acantilado de dtype, y el ABI FA-off deja de ser viable

Punto de longitud de memoria (`poc/experiments/e14.py`).
Generador E8 escalado ×10: 440 microservicios (40 nombres × 11 regiones, puertos
únicos globales) + 120 incidencias = **51.790 tokens**, seed 20260721, prefijo
adversario 1k, N=60, Qwen3-4B, GPU completa, módulo leído de disco dentro del
cronómetro (protocolo E10). Dos brazos de dtype (mismo dtype en TODAS las
condiciones de cada brazo):

| Brazo | joint | naive | nomem | Setup joint | Setup naive | Módulo |
|---|---|---|---|---|---|---|
| q8_0 + FA | 31/60 | **34/60** | 0/60 | 15,89 s | 2,62 s (**×6,1**) | 4,06 GB |
| f16 + FA | 29/60 | **32/60** | 0/60 | 13,85 s | 4,79 s (×2,9) | 7,64 GB |

Lecturas:
1. **El linker no cuesta nada tampoco a 50k**: naive ≥ joint en ambos brazos
   (fallos compartidos 23/26 de ~28 — la firma E8: los fallos son propiedad de la
   tarea, no de la inserción). Cuarto y quinto punto de "linked ≥ joint" en
   Qwen3-4B (E10 ×2, E14 ×2); sigue siendo observación de un solo modelo.
2. **Sin acantilado q8 a 50k**: f16 (29/32) == q8 (31/34) dentro del ruido →
   la caída absoluta (~52-57 % vs 85 % del E8 a 5k) es interferencia/capacidad
   del modelo con 440 servicios, y golpea igual al prefill conjunto. Extiende el
   mapa de tolerancia de H25 (Qwen3-4B aguanta q8 hasta 50k). El modo de fallo
   es el de H25 (valores plausibles equivocados, concentrados en el atributo más
   confundible: día de ventana), por eso la ablación f16 era obligada antes de
   atribuir la causa.
3. **A 50k el ABI práctico es FA-on (16 GB)**: FA-off + f16 hace OOM (KQ
   materializado, medido en la 1ª pasada) y FA-off + V cuantizada está vetado
   por llama.cpp (assert en la creación del contexto). Registrado en el propio
   script; el JSON lleva `flash_attn: 1`.
4. **El dtype compra velocidad de link**: q8 enlaza en 2,62 s vs 4,79 s de f16
   (mitad de bytes) — a módulos multi-GB el I/O domina, como predice O(bytes).
5. La ventaja de setup crece con la longitud en la misma GPU: ×1,7 (5k, E8) →
   ×6,1 (50k, q8) — segunda dimensión del argumento §5.6 junto a la de H31.

Resultados: `results/resultados-e14-qwen-srv.json` (q8) y
`resultados-e14-qwen-f16-srv.json` (f16). Integrado en §5.6 y §6.1 del paper.

## H33. Punto NVMe frío (E12 ampliado): el restore cuesta lo mismo con el módulo en page cache que leído en frío del NVMe

Punto NVMe frío. Brazo nuevo en `e12.py`: antes del
restore cronometrado se expulsa el `.kmd` de la page cache con
`posix_fadvise(POSIX_FADV_DONTNEED)` (sin root, solo ese fichero). Módulo E12
original (memoria-hist 15,2k tok, f16, 870 MB, Coder-7B, GPU completa):

| Medida | Tiempo (mediana, N=5) |
|---|---|
| prefill | 5,5 s (2781 t/s) |
| restore caliente (page cache) | 0,65 s |
| **restore frío (NVMe real)** | **0,78 s** |

Recall 6/6 en todas y en las 5 pasadas. La eviction está verificada aparte (la
lectura del fichero baja a velocidad NVMe tras fadvise): el disco frío añade solo
~0,13 s de I/O a un restore de ~0,7 s — **la subida/scatter del estado al
dispositivo domina el coste, no el almacenamiento**. En hardware de clase NVMe,
"frío" y "caliente" difieren solo en ~0,1 s; el módulo se puede servir
directamente de disco sin warm-up. (El punto frío *refuerza* la ley
O(bytes); en HDD/red la historia sería otra — límite honesto a declarar.)

Resultados: la re-medición N=5 está en `results/resultados-e12-coder-eb-1..5.json`
(claves `t_restore_cold_s`, `restore_cold`); la lectura única original es
`results/resultados-e12-coder-cold-srv.json`. Integrado en §5.6 del paper
(barras de error de la Fig. 2 = min–max sobre las 5 pasadas).

## H34. E15: desfragmentación en vivo del contexto (bank switching agéntico) — mecánicamente sub-milisegundo y conductualmente neutra

Hipótesis: en una conversación, cargar un documento pesado, responder,
**liberar sus celdas y compactar el hueco** (seq_rm + seq_add negativo
sobre la cola = el mismo K-shift perezoso del rebase), cargar el siguiente, etc.
La compactación en VRAM debería ser muy rápida; la incógnita es cómo afecta al
LLM en ejecución.
`poc/experiments/e15.py` (5 iteraciones de arnés, todas documentadas aquí):
bucle agéntico donde el modelo pide documentos con `CARGAR(<doc>)`, 7 turnos
sobre los 3 módulos E10 intercalados (6 conmutaciones + 1 page-hit), condición
control sin evicción. Qwen3-4B, GPU, servidor.

**Mecánica (respuesta a "¿es rápido?"): sí, sub-milisegundo.**
- Evict+compact: **0,5–1,0 ms** (solo metadatos; el K-shift aterriza en el
  siguiente decode, que cuesta ~0,42 s — *menos* que el control 0,83 s porque
  el contexto residente es menor). Page-hit: 7–21 ms. Link: 0,3–4,2 s según
  tamaño (de disco). Ningún error del runtime en 6 ciclos × 5 versiones.
- **Working set acotado**: pico 16.124 celdas (defrag) vs 34.361 (control) —
  el bank switching real que motivó ARCHITECTURE.md §8.

**Conducta (respuesta a "¿cómo afecta al LLM?"): neutra.** Con el instrumento
validado (batería aislada E10, 6 preguntas con rollback tras cada conmutación):
**defrag 22/42 == control 21/42**. La conversación además sobrevive: la sonda
final ("¿cuál fue la primera pregunta?") se responde LITERALMENTE tras 6
compactaciones — las filas KV de las respuestas generadas mientras el documento
estaba cargado conservan su contenido contextualizado aunque el documento ya no
esté (la predicción teórica se confirma: lo dicho se recuerda; lo no preguntado
se pierde con el módulo).

**Trampas de arnés agéntico (3 modos de fallo, TODOS presentes también en el
control — ninguno es efecto de la paginación):**
1. *Pregunta-antes-de-evidencia a distancia de módulo* (v1): enlazar el doc
   después de la pregunta mete 8-15k tokens entre pregunta y generación → el
   modelo resume el documento en vez de responder. Arreglo: re-pregunta tras la
   carga (protocolo E4b §5.4). El patrón tool estándar de §6.5 asume resultados
   de tool cortos; con módulos multi-k la re-pregunta es necesaria.
2. *Auto-imitación conversacional* (v2-v3): turnos adyacentes con plantilla
   igual ("¿presupuesto de X?") hacen que el modelo copie el VALOR de su
   respuesta anterior en vez de leer el documento recién cargado (la confusión
   de atribución E3 en versión conversacional); en el transcript compactado el
   patrón "Usuario→CARGAR" domina y captura también la re-pregunta.
3. *Contaminación few-shot* (v4): el modelo confunde los ejemplos del system
   prompt con la conversación real (la sonda citó la pregunta del ejemplo).
El QA conversacional del 4B sobre documentos recién enlazados queda en ~50 %
con el mismo instrumento que da ~97 % en contexto limpio (E10) — coste del
*entorno conversacional*, idéntico con y sin defrag. Tool-calling: 7/7 estable
(v2+) con etiqueta `[doc]` en la pregunta y regla dura en el system.

**Híbridos GDN: matiz importante**. NO es que quede un hueco irrecuperable — es lo contrario: las capas
recurrentes no tienen celdas por token (estado de tamaño FIJO, ~20 MB constantes
en el 2B), así que ahí no hay fragmentación posible ni memoria que recuperar; y
las capas de atención del híbrido (1 de 4) se compactan con el mismo primitivo
E15 sin cambios. El límite es *semántico*: el estado recurrente es un acumulador
con pérdida (`S_t = A_t·S_{t-1} + B_t`, un reduce) — la contribución del
documento queda multiplicativamente enredada con todo lo posterior y `T_doc` es
contractivo (inversa mal condicionada), así que no se puede "des-reducir".
Esquema práctico (fase 7, ingeniería no matemática): **checkpoint + replay** —
snapshot de S (20 MB, vía `PARTIAL_ONLY` como en E7) antes de enlazar el doc;
al evictar, compactar atención como en E15, restaurar el snapshot y re-decodificar
solo los turnos posteriores al documento (decenas de tokens, nunca el doc).

Resultados: `results/resultados-e15-qwen-{defrag,control}-srv.json`.
Integrado: ARCHITECTURE.md §5.4 (primitivo evict+compact validado) y §6.6 del
paper (datos de evicción).

**E15b: la laguna híbrida queda CERRADA.** `poc/experiments/e15b.py`: mismo diseño (7 visitas,
batería aislada tras cada conmutación, control sin evicción) sobre
**Qwen3.5-4B** (GDN), cargas scriptadas, arnés ChatML con think fijado.
Evicción = **checkpoint + replay**: `seq_cp(0→2)` antes de enlazar (snapshot
completo COW: celdas de atención por pertenencia + copia del estado recurrente);
al evictar, wipe de seq 0 + `seq_cp(2→0)` + re-decode de la cola (solo los
turnos post-doc, nunca el documento). Detalle empírico que costó una pasada:
el `seq_rm` parcial de cola está VETADO en memoria recurrente (H17 — no hay
historia por token que truncar), por eso el checkpoint es de secuencia
completa y no un blob `PARTIAL_ONLY` a secas.

| Métrica | defrag híbrido | control |
|---|---|---|
| Batería aislada | **41/42** | 39/42 |
| Chat recall | **7/7** | 6/7 |
| Evict+replay | **4,7–4,9 ms** (~50 tok replay) | — |
| Pico de celdas | **13.657** | 29.619 |
| Sonda coherencia | cita literal la 1ª pregunta | cita una posterior |

Coste de evicción O(cola)≈5 ms, conductualmente neutro (marginalmente superior,
como en E10/E14: el documento recién enlazado "limpio" rinde ≥ que el residente
diluido). Nota de arnés: el ChatML con think fijado da casi techo (41/42) donde
el arnés de texto crudo del 4B full-attention daba 22/42 — el instrumento
importa más que la paginación. Módulos naive de una pasada (sin sondas T):
compile 1,2-2,3 s por doc de 7-13k tok. Resultados:
`results/resultados-e15b-qwen35-{defrag,control}-srv.json`.

## H35. K-shift + MTP: el mismo hueco de celdas compartidas que H29, ahora en la relocación (verificado en código)

Pregunta de seguimiento: ¿afecta la relocación (defrag/rebase) a los modelos
MTP? Verificado en `llama-kv-cache.cpp` (b10068): **sí, hay una laguna gemela de H29**.
`llama_kv_cache::update()` aplica el grafo de K-shift **solo a las capas del
cache que ejecuta el update** y a continuación hace `cells.reset_shift()`
(línea ~888) sobre las `v_cells` — que en MTP están COMPARTIDAS con el contexto
draft. Consecuencia: el primer contexto que decodifica tras un `seq_add`
(siempre el target) consume los deltas de desplazamiento y re-rota solo SUS K;
cuando el contexto draft decodifica ya no queda shift pendiente → **las K del
cabezal MTP quedan sin re-rotar permanentemente** tras cualquier relocación:
nuestro link no-prefijo, nuestra compactación E15, o el propio context-shift
nativo de llama.cpp (esto último afecta a llama.cpp vanilla con modelos MTP,
sin intervención nuestra — argumento directo para el PR/issue upstream).

Impacto y degradación (misma física que E13): **la corrección nunca se rompe**
(el target verifica todo token especulado); el coste es aceptación/throughput
sobre las posiciones desplazadas (medido en E13 como −9 % t/s con draft-KV
inválido). Mitigaciones:
1. **Para nuestro tooling, sin parche**: el rebase software (hyblib) puede
   rotar TAMBIÉN las filas K del blob draft de la sección `mtp` del `.kmd`
   antes de restaurar — la relocación ocurre host-side y el hueco del runtime
   ni se pisa. (El restore de E13v2 no lo sufrió porque restaura en las
   posiciones originales, sin `seq_add`.)
2. **Upstream (candidato a 3er parche / issue)**: el `reset_shift()` de celdas
   compartidas debería diferirse hasta que todos los caches que las comparten
   hayan aplicado su shift (o el update del target debería cubrir las capas
   del cache `other`). Pendiente de diseño fino; reportable como issue aunque
   no llevemos parche.
3. Operativo mientras tanto: en modelos MTP, evitar `seq_add` con especulación
   activa o aceptar la degradación elegante (solo velocidad).

## H36. E17: multi-hop sobre módulo enlazado — paridad agregada, con varianza por modelo en ambos sentidos

Cierre del último «untested» de §7 del paper (`poc/experiments/e17.py`). Misma memoria
determinista de E8 (mismo generador y seed → artefacto bit-idéntico) pero
preguntas de **2 saltos** sobre claves únicas por construcción (puerto →
servicio → atributo; incidencia → servicio → atributo), N=40, tres condiciones
E8, tres modelos:

| Modelo | joint | naive | nomem | Fallos compartidos / solo-joint / solo-naive |
|---|---|---|---|---|
| Qwen3-4B | 22/40 | 22/40 | 0/40 | 15 / 3 / 3 |
| Coder-7B | 22/40 | 17/40 | 1/40 | 17 / 1 / 6 |
| Qwen3-14B | 29/40 | **35/40** | 0/40 | 5 / 6 / 0 |
| **Agregado** | **73/120** | **74/120** | 1/120 | — |

Lecturas:
1. **El multi-hop es más duro para ambos por igual**: 55-73 % vs 85-95 % del
   single-hop — el coste del salto compuesto lo paga también el prefill
   conjunto (mayoría de fallos compartidos).
2. **Paridad agregada** (74 vs 73/120) con varianza por modelo en AMBOS
   sentidos: Coder −5 (único déficit naive observado en módulo único de toda
   la investigación; 6 fallos solo-naive vs 1 solo-joint — señal débil con
   N=40, reportar sin inflar) y 14B **+6** (naive no pierde NINGUNA que joint
   acierte; joint pierde 6 que naive acierta). Sin patrón sistemático contra
   el módulo enlazado.
3. Integrado en §7 (Limitations) del paper: paridad agregada con la varianza
   por modelo declarada en ambos sentidos.

Resultados: `results/resultados-e17-{qwen,coder,qwen14b}-srv.json`.

## H37. E16: memoria virtual conversacional — una conversación de 5,5k tokens vive en una ventana de 4k

Validación del §8.4b de ARCHITECTURE.md (`poc/experiments/e16.py`). Primitivo nuevo sobre los existentes: **sellar un
RANGO de la secuencia viva como blob reubicable** — `seq_cp(0→1, p0, p1)` →
`state_seq_get(1)` → evict+compact E15; el page-in es el linker con delta
`destino − p0`. Qwen3-4B, n_ctx **4.096** deliberadamente pequeño, watermark
3.000, 14 informes scriptados con 3 hechos únicos cada uno:

| Métrica | Valor |
|---|---|
| Conversación total | **5.528 tokens** (> n_ctx: sin sellado abortaría) |
| Residente final | 2.781 celdas (7 segmentos sellados a RAM, 58 MB/u) |
| Sellado | 244 ms de media (dominado por serializar 58 MB) |
| Page-in | **142 ms** de media |
| Recall residentes | 14/21 |
| Recall archivados SIN page-in | **0/21** (aislamiento perfecto, sin fugas) |
| Recall archivados CON page-in | **15/21** (≥ residentes) |

Lecturas:
1. **La ventana deja de acotar la conversación**: lo direccionable lo acota el
   almacenamiento; lo residente, el watermark. La conversación completa siguió
   siendo consultable (0/21 → 15/21 con page-in de 142 ms por consulta).
2. **Sexta aparición del patrón "recién enlazado ≥ residente"**: el segmento
   paginado de vuelta (adyacente a la pregunta) rinde igual o mejor que los
   que nunca se fueron (diluidos en mitad de la conversación).
3. El coste de sellado (244 ms) es serialización de bytes, solapable con la
   inactividad entre turnos (diseño §5.4 de ARCHITECTURE.md); el page-in (142 ms)
   es I/O de RAM + K-shift — imperceptible en interactivo.
4. v1 del arnés (10 segmentos): la conversación quedó JUSTO bajo n_ctx —
   corregido a 14 segmentos; los números v1 (aislamiento 0/9, page-in 9/9)
   son consistentes con v2.

Resultados: `results/resultados-e16-qwen-srv.json`. Integrado en §6.7 del
paper (con E15/E15b y E18 forma la historia completa: evict, defrag, archivo y
page-in = memoria virtual del contexto sobre primitivas stock).

## H38. E18: lectura paginada del documento de 51,8k — la paginación SUPERA al contexto completo (+15 pts) con ventana 14× menor

El banco de §8.5 de ARCHITECTURE.md (`poc/experiments/e18.py`). Mismo generador, seed
y muestra de 60 preguntas que E14 (comparabilidad directa). El documento nunca
existe como un contexto:

- **Compilación paginada (§8.4 validada)**: 28 chunks de ~2k (cortes en
  fronteras semánticas `###`/`##`, con subdivisión por líneas para secciones
  sin subcabeceras — la lista de 120 incidencias, 5,4k tok, causó el único
  fallo de arnés). **51.790 tokens compilados en 8,82 s sin que n_ctx supere
  4.096** — el OOM que costó 3 iteraciones en E14 es imposible por
  construcción. Almacén: 7,6 GB f16 (28 blobs).
- **Lectura con presupuesto de 4.096** (vs 57.344 de E14): tabla de páginas
  determinista (clave svc-/INC- de la pregunta → chunk que la contiene;
  selector model-free para aislar la calidad de la PAGINACIÓN de la del
  selector — el híbrido RAG+tool de §8.2 es la versión de producción).
  Page-in medio **109 ms**; 0 fallos de página.

| Condición | Recall | n_ctx |
|---|---|---|
| E14 joint (prefill 51,8k) | 31/60 | 57.344 |
| E14 naive (link 51,8k) | 34/60 | 57.344 |
| **E18 paginado (1 chunk/pregunta)** | **49/60** | **4.096** |

**La paginación no es (solo) un ahorro: a esta escala es MEJOR** — +15-18
puntos sobre el contexto completo, porque el modelo lee una página de ~2k con
los hechos del servicio en vez de pelear contra la interferencia de 440
servicios (H32 diagnosticó esa interferencia como el límite; E18 la elimina).
Séptima aparición, y la más fuerte, del patrón "contexto pequeño y fresco >
residente grande". Los 11 fallos restantes son TODOS del atributo confundible
de siempre ("día de ventana" respondido como fecha — formato, no recuerdo).
La tesis del bank switching (ARCHITECTURE.md §8) queda validada en su forma
fuerte: working
set acotado + documento infinito direccionable + calidad superior.

Resultados: `results/resultados-e18-qwen-srv.json`. Integrado en §6.7 del
paper ("Context virtual memory").

**Réplica multi-modelo**: Coder-7B **46/60** (page-in 44 ms, store 3,0 GB — su GQA
de 4 cabezas hace las páginas 2,6× más ligeras) y Qwen3-14B **60/60 PERFECTO**
(page-in 121 ms, store 8,5 GB). El patrón replica en tres modelos (49/46/60 de
60); matiz honesto: la referencia full-context de 50k (31-34/60) solo existe
para el 4B — para el 14B, su propio E8 single-hop a 5k fue 57/60, así que 60/60
a 51,8k paginado es techo absoluto. `resultados-e18-{coder,qwen14b}-srv.json`.

## H39. E19: MTP funciona en vLLM out-of-the-box (+52 % t/s), pero el camino conector rechaza modelos híbridos

Validación empírica parcial de H30 (`poc/experiments/e19.py`, vLLM 0.22.1, Qwopus3.5-4B-Coder safetensors con `mtp_num_hidden_layers: 1`):

1. **La especulación MTP funciona en vLLM sin tocar nada**:
   `speculative_config={"method": "mtp", "num_speculative_tokens": 2}` sobre
   el checkpoint tal cual — **65,7 t/s vs 43,1 t/s sin especulación, media de 3
   preguntas (+52 %)** (el par 67,5/48,4 citado antes era una sola pregunta),
   respuestas correctas (con el pin `<think>\n\n</think>` — la trampa H19
   también aplica en vLLM), TTFT ~0,57 s.
2. **HALLAZGO NEGATIVO (relevante para un RFC upstream): conector KV + modelo híbrido =
   incompatible hoy** — con `kv_transfer_config` activo, vLLM desactiva su
   gestor híbrido de KV y aborta:
   `ValueError: Hybrid KV cache manager is disabled but failed to convert the
   KV cache specs to one unified type.`
   Es el espejo vLLM de los huecos de celdas compartidas de llama.cpp
   (H29/H35): cada motor tiene SU hueco para el mismo objetivo; la sección
   híbrida/mtp del `.kmd` no tiene camino de ingestión en vLLM hasta que el
   conector soporte specs heterogéneos. → El H30 arquitectónico (capas draft
   como grupos de primera clase) sigue verificado en código; su validación
   end-to-end por conector queda BLOQUEADA por este gap para híbridos (un
   modelo MTP no-híbrido pequeño la desbloquearía; no hay a mano).
3. Trampas de arnés vLLM documentadas en el script: guard `__main__`
   obligatorio (spawn), `ninja` para JITs, `VLLM_USE_FLASHINFER_SAMPLER=0` si
   el nvcc del sistema no compila flashinfer, `json default=str` para métricas,
   y `get_metrics()` requiere stat logging (deshabilitado en offline por
   defecto — el proxy de t/s basta).

Resultados: `results/resultados-e19-{baseline,nospec}.json`; el fallo del
conector queda reproducible en los logs y comentado en el script.

## H40. Sección `mtp` del `.kmd` (formato v1) — implementada y validada por round-trip byte a byte

El `.kmd` gana una sección
opcional `mtp` para transportar el estado del cabezal draft (el payload
neutro-para-corrección de H30/E13):

- **Formato v1** (retrocompatible): el blob principal deja de leerse hasta EOF
  y pasa a leerse por `blob_bytes` exactos; el blob draft va anexado detrás.
  Los módulos clásicos siguen siendo v0 byte-idénticos; un lector v0 ante un
  fichero v1 falla limpio (assert de versión) en vez de leer mal.
- **`mdc mtp-pack <target> <draft> --model M [--md src]`**: empaqueta el par de
  slot files del server parcheado (contenedores `llama_state_seq_save_file`,
  guardados verbatim y marcados `container: seq_file`) con identidad
  direccionada por contenido; los tokens se extraen del propio contenedor de
  sesión (parser best-effort del formato: magic+versión+count+ids). Solo
  hashea los pesos, no carga el modelo.
- **`mdc mtp-unpack <mod.kmd>`**: reproduce los dos ficheros para SLOT_RESTORE
  (orden target→draft registrado en la cabecera, junto a `pos_offset: 1` —
  la convención EAGLE/MTP de H30 — y el parche requerido).
- **Validado en servidor**: pack de los blobs exactos de E13v2 (94,8 MB target
  + 5,3 MB draft, 1.284 tokens detectados) → unpack → `cmp` **byte-idéntico**
  en ambos ficheros. Esos bytes son los que E13v2 validó conductualmente
  (aceptación 0,722), así que el ciclo compile-server → .kmd → restore-server
  queda cerrado end-to-end.
- Límite declarado en el propio comando: la reubicación del KV draft (rebase
  host-side, mitigación H35) no está implementada — estos módulos restauran en
  sus posiciones de compilación, exactamente el protocolo E13v2.

## H41. E20: SWA entra en el conjunto compatible — el linker no añade déficit y la ventana deslizante limita a joint y linked por igual

Primera caracterización SWA (fila que faltaba en la tabla de compatibilidad).
Modelo: Gemma 3 4B it Q4_K_M (atención intercalada 5 locales : 1 global,
ventana 1024), llama.cpp b10068 en el servidor, **sin tocar nada**: mismas
baterías estándar (`bateria2.py`, `bateria6.py`), mismo scoring, iSWA activa
(verificado aritméticamente: 47 KB/token medidos vs ~49 predichos para ~6
capas globales completas + 28 capas SWA guardando solo 1024/4593 de sus
posiciones; full-attention serían ~136).

- **Mecánica: todo funciona a la primera.** `state_seq_get/set` sobre la caché
  iSWA, link tras prefijo adversarial de ~1k y rebase (`seq_add`) — cero
  errores, cero parches. SWA pasa de "pendiente de caracterizar" a validado.
- **Recall, módulo ≲ ventana (bateria2, módulo ~1,4k):** paridad EXACTA.
  E1 joint 16/20 = naive 16/20; E2 (prefijo adversarial + rebase) joint
  15/25 = naive 15/25; nomem 0/20 y 5/25. El linker no cuesta nada.
- **Recall, módulo ≫ ventana (bateria6, memoria 4,6k):** colapso SIMÉTRICO —
  joint 16/60 vs naive 14/60 (nomem 0/60). Las respuestas tienen formato
  correcto pero confunden atributos entre servicios: solo ~1/6 de las capas
  (las globales) ve la memoria completa. Es el techo de la propia
  arquitectura, no un déficit del linker: joint sufre exactamente igual.
- **Composición (E3):** joint2 18/20 vs composed2 14/20 — el mismo déficit de
  atribución multi-módulo conocido de full-attention (H9); la reparación
  splice-k (H14) queda sin probar en SWA.
- **Economía de almacenamiento invertida a favor:** las capas SWA solo
  serializan su ventana → 215,8 MB para 4.593 tokens (47 KB/token, ~3× menos
  que un 4B full-attention). El coste del módulo NO crece linealmente con el
  documento en las capas locales — solo las globales pagan tamaño completo.
- Implicación de diseño: en SWA el módulo precompilado hereda la semántica de
  visibilidad del modelo (lo que la ventana no vería en joint prefill tampoco
  se ve linked). La regla operativa para el manager: módulos ≤ ventana SWA
  van "gratis"; módulos mayores rinden lo que el modelo mismo rinde a esa
  distancia. Resultados: `resultados-bateria2-gemma3-4b-srv.json`,
  `resultados-bateria6-gemma3-4b-srv.json`.

## H42. Re-análisis estadístico pareado (McNemar exacto + IC Newcombe) sobre los crudos ya versionados — sin servidor

Motivado por revisión externa: "no measurable recall loss" / "lossless" son
afirmaciones de *ausencia* que N=10–25 por celda no puede sostener; se pide
test pareado e intervalos de confianza. Los `detail` por pregunta de cada
condición (`{q, answer, ok}`) están alineados entre condiciones → pareable, y
se reanaliza sin repetir ninguna ejecución (`experiments/stats_recall.py`).

- **Single-módulo (pool nuclear, N=420):** linked 78,6 % vs joint 79,3 %,
  **McNemar exacto p=0,69**, IC 95 % Newcombe del déficit **[−1,7, +3,1] pp**.
  Ampliado a contexto largo + two-hop (E14/E17, N=600): Δ +0,2 pp, p=1,0. Sin
  diferencia detectable; margen de no-inferioridad 10 pp (declarado *post hoc*).
- **Composición multi-módulo (N=140):** joint 95,0 % vs composed 81,4 %,
  **McNemar p<0,001**, IC [+7,4, +20,5] pp — el déficit es estadísticamente
  robusto, a diferencia del caso single-módulo.
- **Reparación splice-k a escala micro (N=60):** aún −13,3 pp, p=0,039 —
  splice-k *reduce pero no cierra* el déficit de 2 módulos a este N; ninguna
  config individual alcanza paridad con significancia a N=20.
- **Workspace 3 módulos (N=120):** joint 87,5 % vs workspace 88,3 %,
  **McNemar p=1,0**, IC [−7,9, +6,2] pp — paridad con potencia. Este, y no el
  micro-benchmark de dos módulos, es el resultado limpio de la receta.

Punto metodológico transversal: a N=20/celda McNemar no tiene potencia —
ninguna celda individual es significativa; la señal solo emerge en el pool. Es
la crítica de N pequeño, ahora cuantificada. El texto del paper se actualizó en
consecuencia (lossless → sin diferencia estadísticamente detectable; repaired →
reducido, con paridad a escala de workspace). Análisis:
`experiments/stats_recall.py` (offline, sin modelo ni GPU).

## Estado final

La investigación E1–E20 / H1–H42 está cerrada y consolidada en el paper
(`paper/PAPER.md`, `paper/latex/main.tex`); cada afirmación del paper apunta a
los scripts de `poc/experiments/` (aquí, `experiments/`) y a los JSON de
`results/`. Extensiones abiertas:

- Modelos puramente recurrentes (Mamba/RWKV): el banco donde la política afín
  (H15/H17) es imprescindible, al no haber capas de atención que compensen.
- MLA (DeepSeek) y cachés multimodales: fuera del conjunto compatible
  caracterizado.
- Reubicación host-side del blob draft MTP (mitigación de H35) y compilación
  de la sección `mtp` in-process.
- Upstream: PR del parche de serialización MTP (H29), issue del K-shift sobre
  celdas compartidas (H35), gap conector↔híbridos de vLLM (H39).
