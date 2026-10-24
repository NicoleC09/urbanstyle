# Auditoría del proyecto UrbanStyle

Fecha: 2026-10-23. Base: checkout local, configuración y artefactos existentes. Validación ejecutada en Windows con el entorno `.venv`.

## Estado de ejecución

- `python -m pytest -q`: **50 passed**, 14 avisos de deprecación de Pyparsing/Matplotlib; ningún fallo.
- `python scripts/evaluate_all.py`: terminó correctamente y regeneró evaluación de Task B (CSV de métricas y predicciones de test) y Task C (CSV, JSON y gráfico de pricing).
- Entrenamiento de referencia vigente: LSTM seed 42, 26 épocas por early stopping; PPO 12 000 pasos seed 42; DQN 12 000 pasos seed 43; cGAN seed 42, 80 épocas, CPU, 60 000 imágenes; clasificador 5 épocas.
- Los checkpoints esperados están presentes: `models/lstm.pt`, `models/ppo.zip`, `models/dqn.zip`, `models/cgan_generator.pt`, `models/fmnist_classifier.pt` y `models/dcgan_generator.pt`, además de checkpoints cGAN por épocas.

## Inventario funcional

| Área | Estado | Evidencia y cifra principal | Riesgo de interpretación/demo |
| --- | --- | --- | --- |
| A · Pathfinding | Implementada | BFS y A* sobre grilla 12×8, tres layouts, 20 corridas × 24 clientes. En las tres configuraciones evaluadas A* expandió menos nodos que BFS; el camino y permanencia agregados son iguales por layout. | Menos nodos no implica necesariamente menor tiempo medible en todo mapa; el tiempo es milisegundos en simulación local. Congestión depende de seed y desempates. |
| B · Forecast | Implementada | CSV fuente 1 000 transacciones; 52 semanas completas, seis series y 8 semanas test. Contexto 8: LSTM MAE/RMSE 3.352/4.061; media móvil 3.604/4.285; ingenuo 4.562/5.320. LSTM vence a media móvil en 4/6 y al ingenuo en 5/6. | Muy pocas semanas/test; Clothing·Male no es la mejor con LSTM. No afirmar que el modelo gana en todas las series. |
| C · Pricing RL | Implementada y reentrenada | Reward de margen, coste 0.55×, precio 0.6×–4×, paso 0.05× y horizonte 64. 100 episodios pareados: PPO/DQN 31 877.25 ± 3 874.71 de margen; óptimo teórico 31 920.33 ± 3 958.47. | Elasticidad de las tres categorías = fallback -1.2; no es causal ni calibrada. Los agentes son demostrativos, no afirmar convergencia/uso comercial. |
| D · Generación | Implementada | Checkpoint cGAN de 80 épocas/60 000 imágenes; clasificador 91.11% test accuracy; mejor EMA 98.75% accuracy condicional en época 23 y 91.20% confianza media sobre 80 muestras fijas. Póster 1080×1350. | Accuracy condicional no es FID ni evaluación humana; las imágenes 28×28 son proxies, no productos reales. DCGAN presente es corrida de 5 épocas/6 000 imágenes. |
| Pygame | Implementada | Inicio y pestañas A–D; 60 FPS configurados; bounds responsive y contraste WCAG AA; capturas headless en 3 resoluciones. El p95 render máximo medido fue 7.73/7.72/9.37 ms en 1280×720/1366×768/1920×1080. | Medición headless de render local, no garantía de frecuencia física en hardware externo. |
| Integración | Implementada | Pipeline B→C→A→D; worker de escenario y worker de generación/poster en D; `scenario_summary.json` guardado. Resumen actual seed 42: Clothing, 18.53 unidades pronosticadas, acción Bajar, precio 47.50, 18 unidades, ingreso 855, congestión 24, clase proxy Vestido. | Simulación y proxy visual. La exportación del escenario registra ingreso aunque C aprende por margen; explicar ambos nombres sin confundir objetivo y métrica. |

## Datos y calibración

El archivo fuente `data/raw/retail_sales_dataset.csv` contiene 1 000 filas entre 2023-01-01 y 2024-01-01, tres categorías y dos valores de género. `prepare_data.py` excluye los periodos de borde W-SUN incompletos: los que terminan 2023-01-01 y 2024-01-07. `weekly_sales.csv` contiene 312 observaciones: 52 semanas × 6 series, desde 2023-01-08 hasta 2023-12-31. El escalador del LSTM se ajusta solo con las primeras 36 semanas por serie; 8 se usan para validación y 8 para test.

En `calibration.json`, Beauty usa fallback -1.2 porque la pendiente estimada es positiva (p=0.00069, R²=0.0555); Clothing lo usa por no significancia (p=0.4975); Electronics por pendiente positiva (p=0.0275, R²=0.0222). Estos fallos justifican tratar el precio óptimo 3.30× como resultado matemático del supuesto isoelástico, no como hallazgo empírico.

## Inventario de modelos

| Ruta | Uso | Estado |
| --- | --- | --- |
| `models/lstm.pt` | Pronóstico semanal global | Presente |
| `models/ppo.zip`, `models/dqn.zip` | Política de pricing | Presentes, reentrenados 12 000 pasos con reward de margen |
| `models/cgan_generator.pt` | Generador condicional principal | Presente; selección EMA, 80 épocas |
| `models/cgan_generator_epoch_010.pt` … `_080.pt` | Checkpoints de evolución | Presentes cada 10 épocas |
| `models/fmnist_classifier.pt` | Puntúa clase solicitada | Presente, 91.11% test accuracy |
| `models/dcgan_generator.pt` | Comparación no condicional | Presente, corrida histórica 5 épocas/6 000 imágenes |

## Artefactos relevantes

- Forecast: `outputs/forecast_metrics.csv`, `forecast_metrics.json`, `forecast_test_predictions.csv`, `predictions.csv`, `forecast_training_curve.png`.
- RL: `outputs/pricing_evaluation.csv`, `pricing_evaluation.json`, `pricing_evaluation.png`, `ppo_training.json`, `dqn_training.json` y curvas por agente.
- Layout: `outputs/layout_experiment.csv` y `.png`.
- GAN: `outputs/gan/classifier_metrics.json`, `metrics.json`, `losses.json`, `samples_epoch_*.png`, `samples_best_ema.png`, y carpetas `cgan/` y `dcgan/` con curvas/checkpoints de entrenamiento.
- Integración: `outputs/integration/scenario_summary.json`, póster y galería.
- QA visual: 15 capturas en `outputs/screens/<resolución>/` y `outputs/screens/performance.csv`; imágenes 1280×720 en `docs/captura_*.png`.

## Riesgos para la demo

1. Antes de salir, ejecutar `python scripts/prepare_demo.py`; los modelos/pesos y Fashion-MNIST deben estar accesibles. La ruta normal de recuperación si faltan es ejecutar `python scripts/prepare_data.py`, `python scripts/train_lstm.py`, `python scripts/train_rl.py --algorithm both --timesteps 12000`, `python scripts/train_classifier.py` y `python scripts/train_gan.py --epochs 80` (el último comando puede tardar aproximadamente 87.5 minutos en la máquina validada).
2. El escenario integrado puede tardar por inferencia cGAN/póster, aunque se ejecuta fuera del hilo de UI. En demo real conviene generar el resumen/poster antes y tener captura de respaldo.
3. Las semillas fijan la simulación; el entorno puede variar por hardware y versiones de dependencias. No prometer identidad bit-a-bit fuera del equipo.
4. No presentar `31 920` como beneficio real: es margen simulado con ε=-1.2 y coste del 55 %.
5. La auditoría no midió audio/proyector ni probó el portátil del aula; esos pasos quedan manuales.

## Conclusión

Las seis tareas solicitadas tienen implementación y los gates automatizados pasan. Los principales faltantes no son de código básico, sino de validez externa: calibración causal de demanda, más series/seeds y evaluación perceptual de imágenes. Los documentos antiguos deben actualizarse para no repetir la recompensa de ingreso, los resultados previos de forecast/pricing o el escenario histórico.
