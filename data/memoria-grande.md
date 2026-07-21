# Memoria del dominio de plataforma (generada, snapshot 2026-07)
Inventario canónico de microservicios, incidencias y acuerdos del último trimestre. Esta memoria es la fuente de verdad del dominio.
## Microservicios
### svc-albatros
Servicio del equipo Estano, escrito en Kotlin con Ktor. Escucha en el puerto 7979 y persiste en SQLite. La versión desplegada en producción es la 4.0.14 con 6 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Telmo.
### svc-boreal
Servicio del equipo Estano, escrito en Go con Gin. Escucha en el puerto 7305 y persiste en Redis. La versión desplegada en producción es la 4.8.16 con 6 réplicas. Su ventana de despliegue es el martes. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Ramiro.
### svc-cierzo
Servicio del equipo Estano, escrito en Java con Quarkus. Escucha en el puerto 7566 y persiste en DynamoDB. La versión desplegada en producción es la 1.1.1 con 6 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Lorenzo.
### svc-dolmen
Servicio del equipo Cobre, escrito en Elixir con Phoenix. Escucha en el puerto 7625 y persiste en Cassandra. La versión desplegada en producción es la 3.9.4 con 5 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Fermin.
### svc-esparto
Servicio del equipo Vanadio, escrito en Python con FastAPI. Escucha en el puerto 7975 y persiste en DynamoDB. La versión desplegada en producción es la 4.5.15 con 9 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Dario.
### svc-faro
Servicio del equipo Titanio, escrito en Kotlin con Ktor. Escucha en el puerto 7322 y persiste en MySQL. La versión desplegada en producción es la 2.2.6 con 9 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Lorenzo.
### svc-granito
Servicio del equipo Titanio, escrito en Kotlin con Ktor. Escucha en el puerto 7173 y persiste en MongoDB. La versión desplegada en producción es la 4.9.14 con 9 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Nestor.
### svc-helice
Servicio del equipo Plata, escrito en C# con ASP.NET. Escucha en el puerto 7953 y persiste en SQLite. La versión desplegada en producción es la 4.0.8 con 2 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Ramiro.
### svc-islote
Servicio del equipo Vanadio, escrito en Java con Quarkus. Escucha en el puerto 7751 y persiste en DynamoDB. La versión desplegada en producción es la 2.0.4 con 2 réplicas. Su ventana de despliegue es el martes. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Ines.
### svc-jara
Servicio del equipo Wolframio, escrito en Go con Gin. Escucha en el puerto 7393 y persiste en Redis. La versión desplegada en producción es la 1.6.4 con 3 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Hector.
### svc-kraken
Servicio del equipo Wolframio, escrito en Python con FastAPI. Escucha en el puerto 7372 y persiste en Cassandra. La versión desplegada en producción es la 3.2.8 con 7 réplicas. Su ventana de despliegue es el martes. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Elvira.
### svc-lince
Servicio del equipo Vanadio, escrito en Rust con Axum. Escucha en el puerto 7144 y persiste en DynamoDB. La versión desplegada en producción es la 1.5.10 con 8 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Katia.
### svc-mistral
Servicio del equipo Cobre, escrito en Kotlin con Ktor. Escucha en el puerto 7429 y persiste en MySQL. La versión desplegada en producción es la 1.8.11 con 9 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Olalla.
### svc-nogal
Servicio del equipo Hierro, escrito en Rust con Axum. Escucha en el puerto 7484 y persiste en MongoDB. La versión desplegada en producción es la 1.3.8 con 5 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Ines.
### svc-ocaso
Servicio del equipo Wolframio, escrito en Go con Gin. Escucha en el puerto 7997 y persiste en DynamoDB. La versión desplegada en producción es la 0.3.3 con 5 réplicas. Su ventana de despliegue es el martes. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Carla.
### svc-pinar
Servicio del equipo Vanadio, escrito en Java con Quarkus. Escucha en el puerto 7330 y persiste en SQLite. La versión desplegada en producción es la 0.9.8 con 3 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Telmo.
### svc-quasar
Servicio del equipo Cobre, escrito en TypeScript con NestJS. Escucha en el puerto 7149 y persiste en Redis. La versión desplegada en producción es la 2.6.3 con 7 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Pau.
### svc-roble
Servicio del equipo Plata, escrito en Kotlin con Ktor. Escucha en el puerto 7580 y persiste en PostgreSQL. La versión desplegada en producción es la 1.9.0 con 7 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Gadea.
### svc-sargazo
Servicio del equipo Wolframio, escrito en Kotlin con Ktor. Escucha en el puerto 7297 y persiste en Cassandra. La versión desplegada en producción es la 4.9.9 con 4 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Pau.
### svc-tejo
Servicio del equipo Estano, escrito en Rust con Axum. Escucha en el puerto 7512 y persiste en Redis. La versión desplegada en producción es la 4.8.5 con 11 réplicas. Su ventana de despliegue es el viernes. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Olalla.
### svc-umbral
Servicio del equipo Estano, escrito en TypeScript con NestJS. Escucha en el puerto 7152 y persiste en Redis. La versión desplegada en producción es la 2.7.9 con 12 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Lorenzo.
### svc-vereda
Servicio del equipo Hierro, escrito en Java con Quarkus. Escucha en el puerto 7355 y persiste en MySQL. La versión desplegada en producción es la 3.1.3 con 10 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Quima.
### svc-wolframio
Servicio del equipo Plata, escrito en Python con FastAPI. Escucha en el puerto 7207 y persiste en SQLite. La versión desplegada en producción es la 3.3.6 con 10 réplicas. Su ventana de despliegue es el martes. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Ines.
### svc-xenon
Servicio del equipo Cobre, escrito en Go con Gin. Escucha en el puerto 7467 y persiste en MongoDB. La versión desplegada en producción es la 0.6.3 con 8 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Pau.
### svc-yunque
Servicio del equipo Cobre, escrito en TypeScript con NestJS. Escucha en el puerto 7186 y persiste en Cassandra. La versión desplegada en producción es la 0.0.18 con 6 réplicas. Su ventana de despliegue es el viernes. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Jorge.
### svc-zocalo
Servicio del equipo Cobre, escrito en Elixir con Phoenix. Escucha en el puerto 7682 y persiste en PostgreSQL. La versión desplegada en producción es la 0.3.18 con 7 réplicas. Su ventana de despliegue es el viernes. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Aitana.
### svc-abedul
Servicio del equipo Wolframio, escrito en Python con FastAPI. Escucha en el puerto 7045 y persiste en MongoDB. La versión desplegada en producción es la 0.8.1 con 10 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Carla.
### svc-brezo
Servicio del equipo Cobre, escrito en Elixir con Phoenix. Escucha en el puerto 7842 y persiste en Redis. La versión desplegada en producción es la 2.2.13 con 4 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Olalla.
### svc-canela
Servicio del equipo Plata, escrito en Kotlin con Ktor. Escucha en el puerto 7940 y persiste en Cassandra. La versión desplegada en producción es la 3.9.12 con 5 réplicas. Su ventana de despliegue es el martes. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Katia.
### svc-dedalo
Servicio del equipo Estano, escrito en TypeScript con NestJS. Escucha en el puerto 7044 y persiste en Cassandra. La versión desplegada en producción es la 1.9.10 con 3 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Hector.
### svc-enebro
Servicio del equipo Hierro, escrito en Rust con Axum. Escucha en el puerto 7273 y persiste en SQLite. La versión desplegada en producción es la 1.0.19 con 6 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.99%. La guardia principal la lleva Carla.
### svc-fresno
Servicio del equipo Plata, escrito en TypeScript con NestJS. Escucha en el puerto 7365 y persiste en PostgreSQL. La versión desplegada en producción es la 2.0.6 con 7 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Bruno.
### svc-grulla
Servicio del equipo Plata, escrito en Python con FastAPI. Escucha en el puerto 7268 y persiste en DynamoDB. La versión desplegada en producción es la 0.9.8 con 9 réplicas. Su ventana de despliegue es el viernes. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Gadea.
### svc-hinojo
Servicio del equipo Titanio, escrito en Go con Gin. Escucha en el puerto 7086 y persiste en Cassandra. La versión desplegada en producción es la 3.3.12 con 8 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Lorenzo.
### svc-iris
Servicio del equipo Titanio, escrito en Rust con Axum. Escucha en el puerto 7360 y persiste en Cassandra. La versión desplegada en producción es la 1.2.4 con 11 réplicas. Su ventana de despliegue es el jueves. El SLO de disponibilidad acordado es 99.5%. La guardia principal la lleva Ramiro.
### svc-junco
Servicio del equipo Hierro, escrito en Kotlin con Ktor. Escucha en el puerto 7901 y persiste en PostgreSQL. La versión desplegada en producción es la 3.5.5 con 5 réplicas. Su ventana de despliegue es el viernes. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Quima.
### svc-kiosco
Servicio del equipo Estano, escrito en Kotlin con Ktor. Escucha en el puerto 7803 y persiste en DynamoDB. La versión desplegada en producción es la 1.1.20 con 9 réplicas. Su ventana de despliegue es el lunes. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Jorge.
### svc-laurel
Servicio del equipo Wolframio, escrito en Java con Quarkus. Escucha en el puerto 7712 y persiste en PostgreSQL. La versión desplegada en producción es la 1.7.11 con 12 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Dario.
### svc-madrono
Servicio del equipo Vanadio, escrito en Python con FastAPI. Escucha en el puerto 7490 y persiste en MySQL. La versión desplegada en producción es la 1.6.11 con 6 réplicas. Su ventana de despliegue es el viernes. El SLO de disponibilidad acordado es 99.95%. La guardia principal la lleva Lorenzo.
### svc-nacar
Servicio del equipo Estano, escrito en Kotlin con Ktor. Escucha en el puerto 7717 y persiste en MongoDB. La versión desplegada en producción es la 4.9.16 con 11 réplicas. Su ventana de despliegue es el miércoles. El SLO de disponibilidad acordado es 99.9%. La guardia principal la lleva Pau.
## Incidencias del trimestre
- **INC-2400** (25 de abril): caída de svc-hinojo durante 35 minutos, causada por un agotamiento del pool de conexiones. Postmortem publicado en la wiki.
- **INC-2407** (2 de abril): caída de svc-ocaso durante 19 minutos, causada por un desbordamiento de la cola de eventos. Postmortem publicado en la wiki.
- **INC-2414** (26 de abril): caída de svc-albatros durante 81 minutos, causada por un desbordamiento de la cola de eventos. Postmortem publicado en la wiki.
- **INC-2421** (18 de mayo): caída de svc-lince durante 131 minutos, causada por un desbordamiento de la cola de eventos. Postmortem publicado en la wiki.
- **INC-2428** (22 de junio): caída de svc-quasar durante 104 minutos, causada por una regresión en la serialización. Postmortem publicado en la wiki.
- **INC-2435** (20 de abril): caída de svc-brezo durante 83 minutos, causada por una migración de esquema bloqueante. Postmortem publicado en la wiki.
- **INC-2442** (16 de junio): caída de svc-zocalo durante 208 minutos, causada por un límite de rate en la API externa. Postmortem publicado en la wiki.
- **INC-2449** (24 de abril): caída de svc-tejo durante 61 minutos, causada por un límite de rate en la API externa. Postmortem publicado en la wiki.
- **INC-2456** (4 de junio): caída de svc-lince durante 149 minutos, causada por un límite de rate en la API externa. Postmortem publicado en la wiki.
- **INC-2463** (19 de junio): caída de svc-xenon durante 59 minutos, causada por un desbordamiento de la cola de eventos. Postmortem publicado en la wiki.
- **INC-2470** (16 de mayo): caída de svc-junco durante 190 minutos, causada por un certificado TLS caducado. Postmortem publicado en la wiki.
- **INC-2477** (27 de abril): caída de svc-dedalo durante 137 minutos, causada por un límite de rate en la API externa. Postmortem publicado en la wiki.
- **INC-2484** (6 de junio): caída de svc-tejo durante 238 minutos, causada por un certificado TLS caducado. Postmortem publicado en la wiki.
- **INC-2491** (22 de abril): caída de svc-esparto durante 44 minutos, causada por un desbordamiento de la cola de eventos. Postmortem publicado en la wiki.
- **INC-2498** (12 de abril): caída de svc-enebro durante 14 minutos, causada por un límite de rate en la API externa. Postmortem publicado en la wiki.
- **INC-2505** (24 de abril): caída de svc-grulla durante 123 minutos, causada por un certificado TLS caducado. Postmortem publicado en la wiki.
- **INC-2512** (10 de mayo): caída de svc-helice durante 21 minutos, causada por una regresión en la serialización. Postmortem publicado en la wiki.
- **INC-2519** (28 de abril): caída de svc-laurel durante 135 minutos, causada por un certificado TLS caducado. Postmortem publicado en la wiki.
- **INC-2526** (28 de junio): caída de svc-madrono durante 115 minutos, causada por un límite de rate en la API externa. Postmortem publicado en la wiki.
- **INC-2533** (24 de abril): caída de svc-faro durante 125 minutos, causada por un certificado TLS caducado. Postmortem publicado en la wiki.
## Acuerdos y convenciones
- Los despliegues fuera de ventana requieren aprobación del comité de cambios y un ticket CHG.
- Toda incidencia de más de 60 minutos exige postmortem en 72 horas.
- Los SLO se revisan trimestralmente con los equipos propietarios.
