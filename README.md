# Predicción de Tg en TeO₂–ZnO con composición y descriptores atomísticos

## Estado del nuevo trabajo

El código prepara datos experimentales y define el prototipo para comparar
composición frente a composición más estructura DFT/AIMD. Las simulaciones,
la extracción de descriptores y la evaluación híbrida definitiva siguen
pendientes. El manuscrito en desarrollo se conserva localmente y **no forma
parte de este repositorio público**.

## Trabajar desde Debian/Linux con Codex

Este repositorio contiene código, configuración y pruebas, pero **no** el
manuscrito, los datos SciGlass derivados, los PDF de artículos ni los resultados
generados.
El CSV procesado se regenera localmente desde el archivo de GlassPy en Zenodo
(DOI y MD5 indicados en «Fuente»); no se atribuye a ese CSV una licencia
propia que no haya sido verificada. La descarga inicial necesita conexión a
Internet y el archivo original ocupa aproximadamente 127 MB.

```bash
git clone https://github.com/aqppaulet/Tg_atomista.git
cd Tg_atomista
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests
python scripts/atomistic_extension.py --estado
PYTHONPATH=src python run_pipeline.py
python scripts/atomistic_extension.py
codex
```

`run_pipeline.py` descarga, verifica y prepara los datos, además de ejecutar
la evaluación composicional histórica; puede tardar varios minutos. El último
comando Python crea el inventario binario y una plantilla **sin descriptores**.
Para ejecutar simulaciones AIMD todavía se requieren un motor externo y un
protocolo físico validado: este repositorio no produce un nuevo \(R^2\) híbrido.

## Estructura inicial de código

El objetivo es comparar composición frente a composición más estructura
obtenida por DFT/AIMD, usando Tg experimental como variable objetivo.

| Etapa | Ubicación | Estado |
|---|---|---|
| Plan y decisiones pendientes | `configs/atomistic_plan.json` | Borrador |
| Inventario experimental binario | `scripts/atomistic_extension.py` | Preparación disponible |
| Preparación y ejecución AIMD | `src/tellurite_tg/atomistic/simulation.py` | Interfaces pendientes |
| Extracción estructural | `src/tellurite_tg/atomistic/descriptors.py` | Contrato de datos e interfaz pendientes |
| Comparación predictiva | `scripts/atomistic_extension.py --descriptors ...` | Prototipo pendiente de validar con datos AIMD |

Para consultar el estado sin entrenar ni simular:

```powershell
python scripts/atomistic_extension.py --estado
```

Para preparar el inventario y la plantilla de descriptores:

```powershell
python scripts/atomistic_extension.py
```

Las funciones de simulación y extracción pendientes generan
`NotImplementedError` con una explicación. El programa de dinámica molecular,
el funcional, las condiciones de simulación y los recursos de cómputo siguen
por definir. Python organizará el flujo; el motor DFT/AIMD será una dependencia
externa. Entrenar un potencial interatómico propio queda como extensión futura.

El umbral de 15 composiciones en el prototipo de evaluación es una restricción
provisional de ejecución, **no un tamaño de muestra estadísticamente validado**.
El diseño definitivo de evaluación, la selección de modelos y las métricas
se completarán en la siguiente etapa. No se han calculado resultados híbridos.

## Implementación composicional anterior (referencia histórica)

Los datos, modelos y resultados siguientes pertenecen al estudio anterior.
Se conservan como contexto y para auditoría; no son resultados del nuevo
planteamiento. Los manuscritos e informes históricos tampoco se publican aquí.

Implementación reproducible en Python para predecir la temperatura de
transición vítrea (Tg) a partir de la composición química. Se utilizan todos
los vidrios que cumplen:

- al menos 52 % en masa de TeO₂;
- suma composicional original entre 99 y 101 mol%;
- Tg entre 100 y 600 °C como control amplio de plausibilidad.

No se imponen límites de ZnO, TeO₂ molar, La₂O₃ ni otros modificadores. Todas
las mediciones repetidas se conservan.

## Inventario exploratorio previo

El trabajo actual mide la predicción de Tg a partir de **composición**. El
análisis de las composiciones estrictamente binarias TeO₂--ZnO de SciGlass
encuentra **117 mediciones, 25 composiciones y 42 publicaciones**. Para
investigar si la estructura aporta información adicional, se preparó un
protocolo prospectivo de comparación composición sola frente a composición
más descriptores DFT/AIMD. **No se han calculado ni suministrado simulaciones
DFT/AIMD; no existe un nuevo R² híbrido.** Este inventario es exploratorio;
el tamaño definitivo del nuevo estudio sigue por establecerse.

Preparar el inventario y la plantilla, sin fabricar descriptores:

```powershell
python scripts/atomistic_extension.py
```

Esto crea `data/atomistic/binary_teo2_zno_candidates.csv` y
`data/atomistic/descriptor_template.csv`. Después de ejecutar simulaciones
independientes de Tg y registrar la procedencia de cada una, se podrá usar:

```powershell
python scripts/atomistic_extension.py --descriptors RUTA\descriptores_aimd.csv
```

El prototipo de evaluador exige al menos 15 composiciones simuladas; compara los dos
modelos sobre las mismas mediciones con cinco pliegues por publicación y una
segunda prueba que excluye del entrenamiento las composiciones de prueba.
No debe rellenarse la plantilla con valores calculados de Tg, estimaciones
composicionales o números hipotéticos y llamarlos resultados AIMD.

## Fuente

- SciGlass convertido por GlassPy 0.5.3.
- DOI: <https://doi.org/10.5281/zenodo.13625981>
- archivo: `GlassPy_files.zip`
- MD5: `b5821e817bf1d15adc7fef6002b6f9ec`

## Cohorte

- 4,020 mediciones después de los filtros composicionales.
- 3 valores físicamente anómalos excluidos.
- 4,017 mediciones finales.
- 3,113 composiciones exactas.
- 3,954 combinaciones publicación–composición.
- 747 publicaciones.
- 358 composiciones repetidas entre publicaciones.
- 1,238 registros pertenecientes a composiciones repetidas.
- 95 variables composicionales.
- Tg entre 104 y 562 °C.

## Validaciones

1. `GroupKFold by publication`: excluye publicaciones completas, pero conserva
   la posibilidad de que una receta haya sido medida por otra publicación.
2. `Purged GroupKFold by publication`: también elimina del entrenamiento las
   composiciones exactas presentes en prueba.
3. `Random 80/20 reference`: referencia de interpolación; mezcla publicaciones
   y no constituye el resultado principal.

## Resultados

XGBoost fue el mejor modelo en validación por publicación:

- MAE: 23.24 ± 2.29 °C.
- RMSE: 33.93 ± 3.75 °C.
- R²: 0.650 ± 0.093.

En la validación purgada:

- MAE: 24.63 ± 2.79 °C.
- RMSE: 34.99 ± 3.98 °C.
- R²: 0.629 ± 0.092.

El R² se mantiene al purgar recetas repetidas, por lo que la mejora no se debe
solamente a memorizar composiciones. En la división aleatoria, el bosque
aleatorio obtuvo RMSE 25.05 ± 1.08 °C y R² 0.811 ± 0.024, pero compartió en
promedio 402.4 publicaciones entre entrenamiento y prueba.

Para 1,135 registros cuya composición exacta tenía un antecedente en otra
publicación, XGBoost obtuvo RMSE 22.71 °C y R² 0.830. Para 2,882 composiciones
no observadas, obtuvo RMSE 37.64 °C y R² 0.594.

## Reproducción

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python run_pipeline.py
python scripts/compare_riley_protocol.py
python scripts/evaluate_riley_comparable.py
python scripts/evaluate_family_experts.py
```

No se generan documentos Word ni PDF.

## Archivos principales generados localmente

- `data/processed/tellurite_tg_dataset.csv`
- `data/processed/cross_publication_replicates.csv`
- `data/processed/cross_publication_replicate_summary.csv`
- `data/processed/dataset_audit.json`
- `artifacts/results/metrics_summary.csv`
- `artifacts/results/split_metrics.csv`
- `artifacts/results/predictions.csv`
- `artifacts/results/repetition_stratified_metrics.csv`
- `artifacts/results/riley_comparable_metrics.csv`
- `artifacts/results/riley_comparable_predictions.csv`
- `artifacts/results/riley_comparable_split_metrics.csv`
- `artifacts/results/riley_protocol_comparison.csv`
- `artifacts/results/family_expert_summary.csv`
- `artifacts/results/family_expert_split_metrics.csv`
- `artifacts/results/family_expert_predictions.csv`
- `artifacts/results/family_expert_membership.csv`
- `artifacts/results/shap_importance.csv`
- `artifacts/figures/figure_*.png`
- `artifacts/models/*.joblib`

Los datos y resultados de la lista anterior se crean al ejecutar el pipeline;
no se suben a GitHub.

## Predicción externa

```powershell
python run_pipeline.py --external data/external/lab_validation.csv
```

La entrada debe contener TeO₂ y una suma positiva. Sin `Tg_C` se generan
predicciones; con valores experimentales completos se calculan métricas
externas.
