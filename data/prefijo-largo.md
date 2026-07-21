Eres un asistente de ingeniería. Responde de forma breve y precisa.
Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.

Transcripción de la conversación de hoy hasta este momento:

Usuario: Buenos días. Antes de nada, ¿me recuerdas cómo quedó ayer el tema del clúster de pruebas de rendimiento?

Asistente: Claro. El clúster de pruebas de rendimiento quedó configurado para reiniciarse automáticamente los miércoles a las 05:00 CET, fuera del horario de las pruebas de carga. El cambio lo aplicó el equipo de plataforma con el ticket PERF-2214 y quedó documentado en la wiki interna.

Usuario: Perfecto. ¿Y el entorno de demos para el cliente nuevo?

Asistente: El entorno de demos quedó desplegado en https://demo.acmetax.internal:9443 con datos sintéticos. Se reconstruye automáticamente los viernes a las 22:00 CET para que el lunes esté limpio. El certificado TLS caduca el 3 de octubre de 2026; hay un recordatorio creado con el ticket OPS-1187.

Usuario: Vale. Ayer también hablamos de la base de datos del sistema legado de contabilidad, ¿no?

Asistente: Sí. El sistema legado sigue en MySQL 8.0 sobre una máquina virtual en la región northeurope, y se acordó no migrarlo este año. Solo se le aplican parches de seguridad el primer domingo de cada mes a las 02:00 CET. El responsable es el equipo de sistemas, con el ticket LEG-0450 como paraguas de mantenimiento.

Usuario: ¿Cómo quedó la rotación de guardias de esta semana?

Asistente: La guardia principal de esta semana la lleva Marcos, del equipo de infraestructura, y la secundaria Elena, de plataforma. El teléfono de guardia se rota los lunes a las 09:00. Las alertas críticas llegan por PagerDuty y las no críticas al canal #ops-avisos de Slack.

Usuario: ¿Y el simulacro de recuperación ante desastres?

Asistente: Está programado para el jueves 24 de septiembre de 2026 a las 10:00 CET. Se restaurará el backup completo en la región de contingencia, que es francecentral, y se medirá el RTO objetivo de 4 horas. La coordinación es del ticket DR-0033.

Usuario: Recuérdame también lo del proveedor de correo transaccional.

Asistente: Se decidió mantener SendGrid como proveedor principal y añadir Mailgun como respaldo. El cambio de failover automático se probará el martes 29 de julio a las 16:00 CET. El presupuesto aprobado es de 850 euros mensuales entre ambos, revisable en enero de 2027.

Usuario: ¿Algo pendiente del inventario de certificados?

Asistente: Sí, quedan tres certificados por renovar antes de fin de año: el del API gateway público (caduca el 12 de noviembre de 2026), el del entorno de demos que ya mencioné, y el interno del bus de mensajería, que caduca el 30 de diciembre de 2026. El seguimiento está en la hoja compartida de operaciones y en el ticket OPS-1201.

Usuario: ¿Y cómo va la actualización de Kubernetes que se pospuso?

Asistente: La actualización del clúster de desarrollo a la versión 1.33 se pospuso al 8 de octubre de 2026 por incompatibilidad con el operador de la cola de mensajes. El clúster de producción no se tocará hasta validar el de desarrollo durante al menos dos semanas. El plan está en el ticket K8S-0771.

Usuario: Una última cosa de contexto: ¿el presupuesto de la formación del equipo?

Asistente: Aprobado: 6.000 euros para el segundo semestre, priorizando observabilidad y seguridad. Cada persona puede gastar hasta 750 euros sin aprobación adicional. Las solicitudes se hacen por el formulario interno antes del día 20 de cada mes.

Usuario: Gracias. Ahora carga mi memoria de proyecto y sigue con lo que te pregunte.

Asistente: Entendido, memoria cargada. Adelante con tus preguntas.
