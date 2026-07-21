# Memoria del proyecto Ancla

Proyecto secundario del equipo: **Ancla**, un servicio interno de firma y sellado de documentos.

## Datos técnicos

- Repositorio: `git.internal/plataforma/ancla`.
- Lenguaje: Rust 1.79, con el framework Axum.
- El servicio escucha en el puerto 7070 (HTTP interno, sin TLS; el TLS lo termina el gateway).
- Base de datos propia: SQLite en modo WAL, archivo `ancla.db`; no usa la PostgreSQL corporativa.
- Los artefactos de release se publican en el bucket `s3://ancla-artifacts`.
- La versión actual en producción es la 0.9.3; la 1.0.0 está prevista para septiembre de 2026.

## Organización

- El equipo responsable es el equipo Delta, con Nuria como tech lead.
- Los tickets usan el prefijo `ANC` en JIRA.
- El despliegue de Ancla es manual y solo los viernes por la mañana, tras el corte de las 10:00.
- La observabilidad va por Grafana, con el dashboard "Ancla — firma y sellado".
- La clave de firma se custodia en un HSM; la rotación de la clave es anual, cada enero.
