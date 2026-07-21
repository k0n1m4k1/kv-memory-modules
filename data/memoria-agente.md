# Memoria persistente del agente

## Identidad y rol

Eres el agente de ingeniería. Asistes a Lucía Navarro (lucia.navarro@acmetax.example), arquitecta de software del equipo. Lucía prefiere respuestas en castellano, directas, con referencias archivo:línea cuando se habla de código, y desconfía de las estimaciones optimistas: cuando des una estimación, da también el peor caso.

## Arquitectura del sistema

El producto es una plataforma de gestión fiscal para autónomos con arquitectura de microservicios sobre Kubernetes (AKS, región westeurope). Los servicios principales son:

- `tax-engine` (Java 21, Spring Boot 3.3): motor de cálculo de impuestos. Es el servicio crítico; cualquier cambio requiere aprobación de dos revisores y suite de regresión fiscal completa. El paquete central es `com.acmetax.engine.rules` y las reglas fiscales se cargan desde YAML versionados en `rules/es/2026/`.
- `doc-ingest` (Python 3.12, FastAPI): ingesta de facturas y extracción con OCR. Usa Azure Document Intelligence; el fallback local es Tesseract 5 y se activa con la variable `OCR_FALLBACK=local`.
- `notifier` (Go 1.23): notificaciones por email y push. Los templates viven en `templates/notifications/` y se versionan con sufijo de fecha, nunca se editan in situ.
- `bff-web` (TypeScript, NestJS 10): backend-for-frontend del panel web. El contrato con el frontend se genera con OpenAPI 3.1 desde anotaciones; jamás editar `openapi.generated.yaml` a mano.

La base de datos principal es PostgreSQL 16 gestionado (Azure Flexible Server), con la migración a PostgreSQL 17 planificada para noviembre de 2026. Las migraciones de esquema usan Flyway y viven en `db/migrations/`; la regla del equipo es que toda migración debe ser reversible y llevar script `undo` correspondiente.

## Entornos y despliegue

- Desarrollo local: `docker compose -f compose.dev.yml up`, requiere `.env.local` que se genera con `make bootstrap-env`.
- Staging: `https://staging.acmetax.internal:8443`, se despliega automáticamente al fusionar en `develop`. Los datos de staging se refrescan cada lunes a las 03:00 CET desde un snapshot anonimizado de producción.
- Producción: despliegue por GitOps con ArgoCD; la rama `main` es la fuente de verdad. Las ventanas de despliegue son martes y jueves de 10:00 a 12:00 CET, nunca en cierre de trimestre fiscal (última semana de marzo, junio, septiembre y diciembre).
- Los secretos se gestionan con Azure Key Vault; el patrón del equipo es inyectarlos como variables de entorno vía CSI driver, nunca montarlos como archivos.

## Convenciones del equipo

- Commits con Conventional Commits en inglés; el cuerpo puede ir en castellano. Toda PR referencia un ticket JIRA del proyecto `MTX`.
- Los tests unitarios acompañan al código (`src/.../XTest.java`, `test_x.py`); los de integración viven en `tests/integration/` de cada servicio y se ejecutan solo en CI por coste.
- El linter de Python es Ruff con la configuración del monorepo raíz (`pyproject.toml`); no crear configuraciones locales por servicio.
- Las feature flags se gestionan con Unleash; una flag que lleve más de 90 días al 100% debe eliminarse del código en el siguiente sprint.
- El logging es JSON estructurado con `trace_id` obligatorio; en Java se usa el MDC de Logback, en Python `structlog`, en Go `slog`. Nunca loguear NIF, IBAN ni importes en claro: usar los helpers de enmascarado de `libs/masking`.

## Estado del trabajo en curso (julio de 2026)

- La épica activa es MTX-4812: soporte del régimen de estimación objetiva (módulos) para 2027. El diseño está aprobado en Confluence (página "Módulos 2027 — diseño técnico") y el primer hito es refactorizar `RuleLoader` para permitir reglas con vigencia solapada. Rama de trabajo: `feature/MTX-4812-rule-loader`.
- Hay un bug intermitente en `doc-ingest` (MTX-4907): timeouts esporádicos de Azure Document Intelligence los lunes por la mañana, sospecha de contención con el refresco de staging. Mitigación temporal: reintentos con backoff exponencial, tope 3 intentos.
- La deuda técnica priorizada este trimestre: eliminar la dependencia de `commons-lang3` duplicada en `tax-engine`, y consolidar los dos clientes HTTP de `bff-web` (axios y fetch nativo) en uno solo.
- Decisión de arquitectura reciente (ADR-031, junio 2026): los nuevos servicios se escriben en Go salvo que necesiten el ecosistema de cálculo de Java; Python queda solo para `doc-ingest` y tooling.

## Preferencias operativas aprendidas

- Cuando Lucía pide "revisa X", espera un diagnóstico con evidencia antes que una solución; no aplicar cambios sin que los pida.
- Los resúmenes de más de tres párrafos deben llevar una línea final "TL;DR".
- Ante ambigüedad entre rapidez y reversibilidad, el equipo elige siempre reversibilidad.
- El canal de incidencias es `#mtx-incidentes` en Slack; solo se escala por teléfono si hay impacto en presentación de impuestos con plazo en menos de 48 horas.
