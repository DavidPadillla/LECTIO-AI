# Informe técnico final — LECTIO-AI, etapas 3 a 8

**Fecha del informe:** 2026-11-09  
**Alcance:** revisión del código y de los artefactos generados existentes. No se volvió a entrenar el modelo ni se escribió o consultó Neon para elaborar este informe.

## Resumen ejecutivo

Las etapas 3–8 cuentan con una implementación local y con ejecución registrada del pipeline y de la API. Los resultados disponibles corresponden a **datos sintéticos de demostración**, no a transacciones reales. El modelo global `MLPRegressor` se evaluó cronológicamente y superó al baseline de promedio móvil en MAE y RMSE en esa partición; esto no demuestra generalización a datos reales.

## 1. Preparación de datos

| Indicador | Resultado guardado |
|---|---:|
| Reservas consultadas | 981 |
| Reservas sintéticas válidas usadas | 980 |
| Libros en el catálogo | 101 |
| Libros de demostración analizados | 100 |
| Libros demostrativos con historial suficiente para entrenar | 45 |
| Libros demostrativos con historial insuficiente | 55 |
| Periodo analizado | 2025-12-01 a 2026-11-02 |
| Periodos | 49 semanas |
| Frecuencia | Semanal |
| Solicitudes demostrativas promedio | 20 por semana |

Se eligió una frecuencia **semanal** para reducir la dispersión de solicitudes diarias con el volumen disponible. El horizonte de 30 días se representa mediante cuatro semanas completas y dos días de una quinta semana.

Las reservas demostrativas se identifican por la relación `reservas.usuario_id` → `usuarios.id` con `username` que empieza por `demo_user_`; el correo `@demo.lectio.local` se usa como criterio alternativo. Los libros demo se identifican por `[DEMO-LECTIO]` en `libros_fisicos.descripcion`. Todos los estados (`ACTIVA`, `DEVUELTA`, `CANCELADA`) cuentan como una sola solicitud histórica por reserva; no se vuelve a contar una reserva al cambiar de estado.

| Motivo de exclusión | Cantidad |
|---|---:|
| Reservas no sintéticas excluidas | 1 |
| Reservas duplicadas eliminadas | 0 |
| Identificadores inválidos | 0 |
| Fechas inválidas | 0 |
| Estados inválidos | 0 |
| Referencias a libros desconocidos | 0 |
| Reservas sintéticas asociadas a libros no demo | 0 |
| Libros no demo excluidos del entrenamiento | 1 |

El panel procesado contiene 100 libros demo × 49 semanas (4.900 libro-semanas): 856 con demanda y 4.044 con demanda cero. Los predictores temporales se calculan con valores desplazados para evitar usar demanda futura. Se persiste demanda agregada y variables no personales; no se guardan nombres, correos ni contraseñas en los artefactos.

## 2. Análisis exploratorio

- Demanda demostrativa total: **980 solicitudes**.
- Libros analizados: **100**; periodos semanales: **49**.
- Media: **20 solicitudes por semana** en el conjunto.
- Tendencia descriptiva: la media semanal de la segunda mitad fue **52,97 %** mayor que la primera mitad. Es una comparación de mitades cronológicas, no una prueba estadística ni una afirmación de tendencia futura.
- Estacionalidad anual: **no evaluable**; el historial guardado es menor de dos años.
- Se generaron estadísticas de demanda por libro, categoría, mes y día de semana. La demanda por día de semana usa las fechas reales `fecha_creacion`, no el día artificial de inicio de la semana agregada.

**Libros demo con mayor demanda histórica** (identificados solo mediante ID):

| ID de libro | Solicitudes |
|---|---:|
| `d22d2ed0-cd37-42c1-84d0-24b47a4aa202` | 30 |
| `884b79a8-502f-4ff5-ad22-c33e2cfb37ee` | 30 |
| `a0c52ec4-5954-44fc-a927-6334d47f9f2d` | 30 |

**Categorías con mayor demanda:** Novelas 247; Historia 162; Fantasía 154; Filosofía 150; Ciencia 134; Arte 133.

**Gráficas guardadas en `data/reports/`:**

- `demand_over_time.png` — evolución mensual.
- `top_books.png` — libros con más demanda.
- `demand_by_category.png` — demanda por categoría.
- `demand_distribution.png` — distribución de solicitudes por libro y semana.
- `period_comparison.png` — comparación de demanda media entre mitades del periodo.

Las gráficas y reportes están generados en el workspace. No se revisó visualmente cada imagen; su existencia se verificó en el inventario de artefactos.

## 3. Modelo predictivo

- **Implementado y entrenado:** Pipeline Scikit-learn con codificación one-hot de categorías y escalado estándar de predictores numéricos, seguido por un `MLPRegressor` global para los libros con historia suficiente.
- **Arquitectura:** una capa oculta de 16 neuronas; `relu`; solver `adam`; `alpha=1.0`; `learning_rate_init=0.001`; `max_iter=1000`; `tol=0.001`; `random_state=42`; `early_stopping=False`.
- **Estrategia:** un modelo global aprende entre libros; no se ajustan modelos independientes por libro.
- **Predictores:** ID de libro, categoría, seno/coseno del mes, seno/coseno de la semana del año, lags de 1, 2 y 4 semanas, medias móviles anteriores de 4 y 8 semanas y diferencia de tendencia entre esas medias.
- **Muestras:** 1.440 en entrenamiento y 405 en prueba; prueba con las mismas filas/fechas para modelo y baseline.
- **Separación:** entrenamiento hasta 2026-08-31; prueba del 2026-09-07 al 2026-11-02. La división es cronológica.
- **Artefacto:** [demand_model.joblib](C:/Users/HP/OneDrive/Escritorio/LECTIO-AI/data/models/demand_model.joblib). Fecha registrada en el artefacto: 2026-11-09T05:31:57.525510+00:00.
- **Dataset procesado:** [weekly_demand.csv](C:/Users/HP/OneDrive/Escritorio/LECTIO-AI/data/processed/weekly_demand.csv).

## 4. Evaluación

Métricas guardadas en [evaluation.json](C:/Users/HP/OneDrive/Escritorio/LECTIO-AI/data/reports/evaluation.json):

| Modelo | MAE | RMSE |
|---|---:|---:|
| MLPRegressor | 0,5548 | 0,7496 |
| Baseline: media móvil anterior de 4 semanas | 0,5895 | 0,7994 |

En esta partición temporal el MLP obtuvo menor MAE y RMSE. Hubo **7 predicciones negativas sin corregir en la evaluación**; se conservaron al calcular métricas para no ocultar el error. La API recorta a cero las predicciones negativas al generar pronósticos.

En el subconjunto de baja demanda (16 libros; 144 muestras), el MLP obtuvo MAE **0,4912** y RMSE **0,6569**, frente al baseline con MAE **0,5208** y RMSE **0,6997**. Estos resultados describen únicamente la cohorte sintética y una partición temporal; no son evidencia de rendimiento en reservas reales.

## 5. Predicciones y recomendaciones

- Horizonte: próximos **30 días**, a partir del último periodo guardado. Para el artefacto actual, el rango es 2026-11-09 a 2026-12-09 (fin exclusivo).
- Las predicciones se generan iterativamente semana a semana, alimentando los lags con predicciones anteriores; el resultado diario se aproxima ponderando la quinta semana parcial por 2/7.
- Predicciones negativas se recortan a cero en el endpoint; las métricas de evaluación conservan las predicciones crudas.
- Libros con historial insuficiente reciben estado `DATOS_INSUFICIENTES`, demanda pronosticada `null` y reposición recomendada cero. En esta corrida hubo 45 libros con predicción y 56 con datos insuficientes (55 libros demo más un libro no demo).

**Ejemplo de predicción guardada/calculada con el modelo existente:**

| Libro | Categoría | Demanda pronosticada a 30 días | Estado |
|---|---|---:|---|
| `e47fee18-12c5-4583-943e-610a67e8fbfc` | Novelas | 2,53 | `PREDICCION_DISPONIBLE` |
| `974943b2-5f10-41d3-9736-cbf2adba3e43` | Historia | No disponible | `DATOS_INSUFICIENTES` |

**Lógica implementada de riesgo:**

- `ALTO` si la demanda pronosticada supera el stock disponible.
- `MEDIO` si no lo supera, pero el stock disponible no cubre la demanda más el margen de seguridad.
- `BAJO` en otro caso.
- `DATOS_INSUFICIENTES` si no existe predicción fiable.

El margen predeterminado es 20 % (`LECTIO_SAFETY_MARGIN`, rango permitido 0–2). La reposición se calcula con:

`max(0, ceil(demanda_pronosticada_30_dias × (1 + margen) − stock_disponible))`

El stock disponible no incluye las unidades ya reservadas. `total_inventory` se informa aparte como `stock + reservado`. La recomendación es orientativa y no modifica inventario ni realiza compras.

**Ejemplo de recomendación:** libro `e47fee18-12c5-4583-943e-610a67e8fbfc`, stock 1, reservado 2, inventario total 3, pronóstico 2,53, riesgo `ALTO`, reposición recomendada 3. En los resultados actuales: 45 `ALTO`, 0 `MEDIO`, 0 `BAJO` y 56 `DATOS_INSUFICIENTES`.

## 6. API FastAPI

Las rutas se implementan en `app/api/routes.py`, con modelos Pydantic en `app/api/schemas.py`.

| Endpoint | Función | Verificación registrada |
|---|---|---|
| `GET /` | Información general de LECTIO-AI | HTTP 200 |
| `GET /health` | Estado de la API | HTTP 200 |
| `GET /database/status` | Conexión, conteos, esquema y estados; sin credenciales ni datos personales | HTTP 200 |
| `GET /model/status` | Estado del modelo, fecha, tipo de datos y métricas | HTTP 200 |
| `GET /predictions` | Pronósticos a 30 días; `limit` validado | HTTP 200 |
| `GET /predictions/{book_id}` | Pronóstico de un libro | HTTP 200 en libro existente |
| `GET /recommendations` | Riesgo y reposición orientativa; `limit` validado | HTTP 200 |
| `GET /recommendations/{book_id}` | Recomendación de un libro | HTTP 200 en libro existente |
| `GET /metrics` | Métricas contra baseline | HTTP 200 |
| `GET /analytics` | Resumen exploratorio guardado | HTTP 200 |

Se registró también HTTP 200 para `/docs` y `/openapi.json`. Dirección local de Swagger al iniciar el servidor: **http://127.0.0.1:8000/docs**.

## 7. Pruebas y validación

El registro de ejecución de esta sesión muestra **6 pruebas unitarias aprobadas**:

1. `test_model_not_trained_status_and_prediction_error` — pasa; verifica el estado sin modelo y el error HTTP 503 para predicciones sin artefacto.
2. `test_all_statuses_count_once_and_zero_weeks_are_filled` — pasa; verifica conteo una sola vez, semanas en cero y análisis por día.
3. `test_duplicates_invalid_rows_and_non_demo_are_excluded` — pasa; verifica duplicados, filas inválidas y exclusión no demo.
4. `test_features_use_only_previous_periods_and_contain_no_personal_fields` — pasa; verifica lags y ausencia de campos personales.
5. `test_recommendation_risk_and_replenishment_thresholds` — pasa; verifica niveles y cantidades recomendadas.
6. `test_train_evaluate_forecast_and_recommend_without_negative_demand` — pasa; entrena un artefacto temporal de prueba, comprueba división cronológica, métricas, predicciones no negativas y reposiciones no negativas.

Además, se registró una prueba HTTP de humo con HTTP 200 para todos los endpoints listados, para `/docs` y OpenAPI; `/database/status` devolvió conteos y estado conectado. Una verificación adicional mostró `transaction_read_only = on` en PostgreSQL. `pip check` indicó que no hay dependencias rotas; se comprobó que `.env` y los artefactos generados están excluidos por `.gitignore`.

No hay un archivo de log de pruebas persistido en `tests/`; los resultados anteriores corresponden a las salidas de ejecución disponibles en la sesión. No se reportaron errores pendientes en esas ejecuciones.

## 8. Archivos

**Creados para las etapas 3–8:**

- `app/services/data_preparation.py`
- `app/services/exploratory_analysis.py`
- `app/services/training.py`
- `app/services/prediction.py`
- `app/services/recommendation.py`
- `app/services/pipeline.py`
- `app/api/routes.py`
- `app/api/schemas.py`
- `tests/test_ml_pipeline.py`
- `README.md`

**Modificados:**

- `main.py` — registra el router y el ciclo de vida de la aplicación.
- `app/database/diagnostics.py` — inspección de columnas requeridas e identificación de reservas y libros demo según las reglas vigentes.
- `.gitignore` — excluye reportes, datasets procesados y modelos generados.
- `requirements.txt` — agrega Matplotlib y registra el conjunto congelado de dependencias instaladas.

**Artefactos generados por el pipeline:**

- `data/processed/weekly_demand.csv`
- `data/models/demand_model.joblib`
- `data/reports/preparation.json`
- `data/reports/analytics.json`
- `data/reports/evaluation.json`
- `data/reports/demand_over_time.png`
- `data/reports/top_books.png`
- `data/reports/demand_by_category.png`
- `data/reports/demand_distribution.png`
- `data/reports/period_comparison.png`

`data/models/`, `data/processed/` y `data/reports/` están excluidos de Git; el modelo y los resultados son locales a este workspace.

## 9. Estado del proyecto

**Implementado y ejecutado:** preparación semanal de la cohorte sintética; análisis exploratorio y gráficas; entrenamiento y evaluación cronológica de MLP global; artefactos locales; pronóstico y recomendaciones de 30 días; endpoints documentados por OpenAPI.

**Probado:** seis pruebas unitarias aprobadas; smoke test HTTP de todos los endpoints; lectura de Neon con transacciones en modo solo lectura; Swagger accesible durante la validación.

**Limitaciones y pendiente:**

- El dataset disponible es sintético; las métricas no representan rendimiento en datos reales.
- Solo hay 49 semanas, insuficientes para evaluar estacionalidad anual.
- 55 de 100 libros demo no alcanzaron el umbral mínimo de historia; se etiquetan sin pronóstico. El único libro no demo también se excluye del modelo.
- La evaluación es una única partición cronológica; convendría validar con más periodos y con datos representativos antes de decisiones operativas.
- La base y tablas no fueron modificadas. El acceso de lectura depende de que las credenciales de Neon también tengan permisos de solo lectura; el cliente configura transacciones read-only, pero no sustituye un rol de base de datos restringido.
- No se desplegó la aplicación ni se integró con Spring Boot. Para ello aún se requieren decisiones de despliegue, gestión segura de secretos, permisos/rol de lectura en Neon, observabilidad, revisión funcional y pruebas de integración con el sistema Java.

## Comandos locales

Desde la raíz del proyecto y con `venv`:

```powershell
python -m app.services.pipeline
uvicorn main:app --reload
python -m unittest discover -s tests -v
```

El pipeline vuelve a consultar Neon y reentrena; no es necesario ejecutarlo para iniciar la API si ya existen los artefactos locales.
