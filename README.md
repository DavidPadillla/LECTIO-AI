# LECTIO-AI

API local para analizar y pronosticar demanda bibliotecaria. El pipeline usa
SQLAlchemy con PostgreSQL en solo lectura y entrena un MLPRegressor global.
El pipeline procesa todas las reservas válidas de libros físicos existentes
(activas y devueltas), excluye las canceladas y no depende de etiquetas demo.

## Preparar el entorno

En PowerShell, desde la raíz del proyecto:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configura `DB_URL` en `.env` con una URL válida de PostgreSQL. No guardes `.env`
en Git ni compartas su contenido.

El proyecto fija Python en `.python-version`. Las dependencias incluyen FastAPI,
Uvicorn, SQLAlchemy, `psycopg2-binary`, `python-dotenv`, APScheduler y las
bibliotecas utilizadas para cargar y ejecutar el modelo.

## Generar análisis y modelo

```powershell
python -m app.services.pipeline
```

El pipeline lee únicamente los datos necesarios, incluye reservas activas y
devueltas asociadas a libros físicos existentes, excluye reservas canceladas,
inválidas o duplicadas, y escribe resultados agregados sin datos personales en
`data/processed/`, `data/models/` y `data/reports/`. Los datos procesados e
informes se excluyen de Git; solo `data/models/demand_model.joblib` se incluye
como modelo inicial de despliegue. Si la historia no alcanza los mínimos de
entrenamiento, el pipeline informa el problema y no crea un modelo.

## Iniciar la API

```powershell
uvicorn main:app --reload
```

Swagger está disponible en `/docs`. La API expone el diagnóstico de solo lectura
en `/database/status`, el estado del modelo en `/model/status` y, una vez
generados los artefactos, predicciones, recomendaciones, métricas y análisis.

## Desplegar en Railway

Railpack detecta el `Procfile`, que inicia `main:app` con Uvicorn en
`0.0.0.0:$PORT` (usa `8000` como valor local predeterminado). `.python-version`
fija Python 3.14.2. No se debe añadir `--reload` ni iniciar múltiples workers.
Configura `/health` como health check en el servicio Railway.

En **Variables** del servicio, configura `DB_URL` como variable secreta con la
URL de Neon. La conexión de la aplicación fuerza PostgreSQL/psycopg2 y
transacciones de solo lectura; no subas `.env` ni copies sus valores a archivos
del repositorio. Railway proporciona `PORT`.

En **Volumes**, crea un volumen persistente con punto de montaje `/data` y
configura `LECTIO_DATA_DIR=/data`. La aplicación copia el modelo inicial
incluido en `data/models/demand_model.joblib` al volumen mediante reemplazo
atómico si todavía no existe; no sobrescribe un modelo persistido. El código
local continúa usando `data/` por defecto. El volumen conserva el modelo
actualizado, los informes y el historial de ejecuciones entre reinicios.

El servicio debe ejecutarse con **una sola réplica y un solo proceso Uvicorn**:
el programador y sus bloqueos son locales al volumen. Las actualizaciones
semanales permanecen activas por defecto los lunes a las 02:00
(`America/Bogota`). Variables opcionales:

- `LECTIO_UPDATE_ENABLED` (`true`/`false`).
- `LECTIO_UPDATE_DAY` (predeterminado `monday`).
- `LECTIO_UPDATE_HOUR` y `LECTIO_UPDATE_MINUTE` (predeterminados `2` y `0`).
- `LECTIO_UPDATE_TIMEZONE` (predeterminado `America/Bogota`).
- `LECTIO_UPDATE_MIN_BASELINE_IMPROVEMENT` y
  `LECTIO_UPDATE_MAX_INCUMBENT_REGRESSION` para los criterios de aceptación.
- `LECTIO_SAFETY_MARGIN` para el margen de seguridad de recomendaciones.

El modelo distribuido se carga correctamente en el entorno local actual:
Python 3.14.2, scikit-learn 1.9.1, NumPy 2.5.4 y joblib 1.6.0; esas versiones
están fijadas para cargar el artefacto de forma reproducible. El archivo
serializado no contiene un manifiesto de versiones que permita verificar de
forma independiente las versiones exactas usadas durante su entrenamiento.
El modelo local actualizado se entrenó con reservas válidas de Neon y conserva
IDs de libros, categorías y métricas de inventario junto con el historial
agregado. Confirma la autorización para compartir ese artefacto antes de
incluirlo en un repositorio público.

Railway `railway.json`/`railway.toml` Config as Code está en proceso de
descontinuación para servicios nuevos. El `Procfile` y la configuración del
servicio en Railway se usan aquí para evitar depender de ese formato legado.

El pronóstico es semanal y cubre los próximos 30 días. Las recomendaciones son
orientativas; no actualizan inventario ni ejecutan compras. El margen de
seguridad predeterminado es 20 % y se puede ajustar mediante
`LECTIO_SAFETY_MARGIN` (rango permitido: 0 a 2).

## Actualización semanal

Al iniciar FastAPI se carga el último modelo local válido y se intenta iniciar
un único programador APScheduler. El horario predeterminado es lunes a las
02:00, zona `America/Bogota`. Se puede ajustar con:

- `LECTIO_UPDATE_ENABLED` (`true`/`false`).
- `LECTIO_UPDATE_DAY` (día en inglés o español; predeterminado `monday`).
- `LECTIO_UPDATE_HOUR` y `LECTIO_UPDATE_MINUTE`.
- `LECTIO_UPDATE_TIMEZONE` (predeterminado `America/Bogota`).
- `LECTIO_UPDATE_MIN_BASELINE_IMPROVEMENT` (mejora mínima, proporción 0–<1).
- `LECTIO_UPDATE_MAX_INCUMBENT_REGRESSION` (regresión máxima permitida contra
  el modelo actual, proporción 0–1; predeterminado 0).

El candidato debe superar MAE y RMSE del baseline y cumplir el criterio contra
el modelo actual en las mismas fechas de validación. Los artefactos se preparan
en staging y el modelo se publica con reemplazo atómico solo después de pasar
los criterios. Si el trabajo falla o se rechaza, continúa activo el modelo
anterior. Bloqueos locales coordinan el programador y la tarea entre procesos;
para producción se recomienda un solo worker y almacenamiento persistente
compartido local entre los procesos de la instancia.

La tarea puede invocarse manualmente sin esperar al próximo horario:

```powershell
python -m app.services.weekly_update --run-now
```

Esta ejecución sí consulta Neon en modo de solo lectura y puede publicar un
nuevo modelo si supera los criterios. El resultado y contadores quedan en
`data/reports/model_update_history.json`; el estado se consulta en
`GET /model/update-status`. Mantén `data/` en almacenamiento persistente y
excluido de Git.
