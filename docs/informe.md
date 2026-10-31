# Informe técnico UrbanStyle

## Problema

UrbanStyle explora cómo conectar cuatro herramientas de IA en una tienda ficticia: pronóstico de unidades semanales, ajuste de precios, análisis de tráfico en la tienda y generación de material promocional. Es un prototipo educativo reproducible, no un sistema listo para decisiones comerciales reales.

## Datos y preparación

Se utilizó Retail Sales Dataset de Kaggle, de Mohammad Talib, bajo licencia CC0 declarada por Kaggle. El archivo disponible contiene 1 000 transacciones, tres categorías (Beauty, Clothing, Electronics) y dos valores de género. La preparación valida esquema, fechas, tipos, nulos y duplicados; descarta los periodos W-SUN incompletos de borde (terminan el 2023-01-01 y 2024-01-07) y genera 52 semanas completas por cada categoría×género, del 2023-01-08 al 2023-12-31. El criterio usa cobertura calendario, no actividad diaria. También calcula estadísticas de precio y proporciones de venta.

La regresión log-log disponible no produjo una elasticidad negativa válida para ninguna categoría: Beauty y Electronics tienen pendiente positiva; Clothing no es significativa (p=0.4975). Task C usa el fallback configurable de -1.2 en las tres categorías. Este supuesto limita las conclusiones económicas.

## Métodos

### A. Rutas y layout

BFS y A* se implementan desde cero sobre una grilla ortogonal. A* usa distancia Manhattan. Tres layouts (Clásico, Abierto y Serpentina) definen pasillos, estanterías y secciones. La simulación pondera las visitas por ventas observadas y acumula tráfico por celda.

### B. Pronóstico

Un LSTM global pequeño con embedding de serie pronostica unidades por semana para las seis series. El split es cronológico 70/15/15 y el escalador se ajusta solo con train. Early stopping utiliza validación. Las comparaciones incluyen último valor (ingenuo) y media móvil.

### C. Precios

El entorno Gymnasium presenta precio relativo, segmento de sensibilidad, pronóstico y ventas recientes. Las acciones discretas bajan, mantienen o suben el precio en pasos de 0.05×, acotado entre 0.6× y 4.0× del precio de referencia, durante 64 pasos. La demanda esperada es isoelástica con estacionalidad y ruido. La recompensa es margen bruto `(precio - coste unitario) × unidades`; el coste unitario es 0.55× del precio de referencia y el ingreso queda como métrica secundaria. Para ε=-1.2, el óptimo continuo del modelo es 3.30×. PPO y DQN se entrenaron durante 12 000 pasos cada uno y se compararon con cuatro políticas estáticas.

### D. Imágenes

DCGAN y cGAN se entrenan con Fashion-MNIST. El cGAN nuevo es convolucional, usa embedding de etiqueta, EMA, label smoothing, ruido de instancia decreciente y truncation de z. Una CNN separada puntúa accuracy/confianza condicional. Las clases Fashion-MNIST son proxies genéricos, no fotografías del catálogo.

## Resultados observados

### Forecast test, contexto de 8 semanas

| Modelo | MAE medio (unidades) | RMSE medio (unidades) |
| --- | ---: | ---: |
| LSTM | 3.352 | 4.061 |
| Media móvil | 3.604 | 4.285 |
| Ingenuo | 4.562 | 5.320 |

La evaluación actual cubre 8 objetivos semanales por serie. El LSTM supera a la media móvil en 4 de 6 series y al ingenuo en 5 de 6; no gana en todas. En Clothing · Male, MAE LSTM = 3.425 frente a 3.281 de la media móvil y 2.625 del baseline ingenuo.

El entrenamiento se detuvo tras 26 épocas por early stopping y el mejor checkpoint es el de la época 1 (MSE de validación normalizado 1.038). La validación no mejoró después, de modo que el modelo aporta poca señal más allá del nivel medio de cada serie. Esto explica por qué su pronóstico se ve casi plano y por qué las ventanas de 4, 8 y 12 semanas dan casi el mismo error.

| Serie | MAE LSTM | RMSE LSTM | MAE media móvil | MAE ingenuo |
| --- | ---: | ---: | ---: | ---: |
| Beauty · Female | 2.858 | 3.312 | 3.344 | 4.875 |
| Beauty · Male | 2.624 | 3.031 | 3.281 | 4.125 |
| Clothing · Female | 3.414 | 4.245 | 4.125 | 6.250 |
| Clothing · Male | 3.425 | 4.459 | 3.281 | 2.625 |
| Electronics · Female | 4.225 | 5.579 | 4.281 | 5.875 |
| Electronics · Male | 3.565 | 3.738 | 3.313 | 3.625 |

Figura: [curva de entrenamiento](forecast_training_curve.png).

### Pricing, 100 episodios pareados

| Política | Margen medio ± desviación | Ingreso medio ± desviación |
| --- | ---: | ---: |
| Precio bajo | 5 969.88 ± 718.56 | 56 005.30 ± 6 825.87 |
| Precio medio | 22 891.50 ± 2 789.40 | 50 870.00 ± 6 198.66 |
| Precio alto | 31 877.25 ± 3 874.71 | 42 656.98 ± 5 165.88 |
| Óptimo teórico (3.30×) | 31 920.33 ± 3 958.47 | 42 957.45 ± 5 296.83 |
| PPO (12 000 pasos) | 31 877.25 ± 3 874.71 | 42 656.98 ± 5 165.88 |
| DQN (12 000 pasos) | 31 877.25 ± 3 874.71 | 42 656.98 ± 5 165.88 |

Los seis métodos se compararon en 100 episodios con condiciones pareadas. PPO y DQN coinciden aquí con Precio alto y su margen medio queda aproximadamente 0.13 % por debajo del óptimo teórico. Precio bajo maximiza ingreso bruto, pero produce mucho menos margen; por ello el objetivo de aprendizaje es margen, no ingreso. La dispersión entre episodios sigue siendo apreciable y las elasticidades son supuestos fallback: no es una estimación causal ni una prueba de superioridad general.

Figura: [margen RL y políticas estáticas](pricing_evaluation.png).

### Layouts

La corrida actual incluye las 20 repeticiones configuradas y 24 clientes por repetición.

| Layout | Congestión BFS | Congestión A* | Longitud media | Permanencia media |
| --- | ---: | ---: | ---: | ---: |
| Clásico | 35.65 | 28.85 | 7.08 | 28.37 |
| Abierto | 35.60 | 35.60 | 5.88 | 23.47 |
| Serpentina | 26.85 | 32.05 | 6.68 | 26.41 |

El camino mínimo es común a BFS/A*; la congestión cambia por el desempate de rutas y las seeds.

### GAN

La CNN clasificadora alcanzó 91.30% de test accuracy con 60 000 imágenes en 5 épocas; Camisa fue la clase más difícil con 67.3%. El nuevo cGAN completó 80 épocas con 60 000 imágenes usando aceleración MPS. El mejor EMA quedó en época 38 con 97.50% accuracy condicional y 90.16% de confianza media sobre 80 muestras fijas (8 por clase); la época 80 midió 96.25%. Es un proxy pequeño de selección basado en el clasificador, no FID ni evaluación humana. El entrenamiento tardó unos 55 minutos en un Mac M4.

Las figuras DCGAN corresponden a una corrida histórica de 5 épocas y 6 000 imágenes, cuyo checkpoint no se regeneró; no representan el entrenamiento cGAN principal. Pérdidas finales guardadas de cada corrida:

| Modelo | Generador | Discriminador |
| --- | ---: | ---: |
| DCGAN | 1.3425 | 0.8104 |
| cGAN | 1.0378 | 1.2816 |

La accuracy condicional sirve como proxy de control de clase, no como métrica perceptual tipo FID. Figuras: [pérdidas DCGAN](dcgan_losses.png), [muestras DCGAN](dcgan_epoch_005.png), [pérdidas cGAN](cgan_losses.png), [evolución cGAN época 5](cgan_epoch_005.png), [época 60](cgan_epoch_060.png), [época 80](cgan_epoch_080.png) y [mejor EMA](cgan_best_ema.png). Los checkpoints cGAN se guardan cada diez épocas en `models/`.

### Escenario integrado

El resumen integrado guardado actual (seed 42, PPO) eligió Clothing con 18.15 unidades pronosticadas; subió el precio a 52.50, simuló 16 unidades e ingreso 840, y encontró 24 visitas en la celda más congestionada. D generó un póster con proxy Fashion-MNIST “Vestido”, sin descuento (0%) y con el texto “Edición premium”. Es una demostración funcional, no una recomendación comercial.

## Limitaciones y siguientes pasos

- Dataset sintético, pequeño y con solo tres categorías y un año de transacciones.
- Pocas semanas para entrenamiento y test de la serie temporal; algunas series contienen ceros frecuentes. El mejor LSTM es el de la época 1, señal de que aprendió poco con estos datos.
- Elasticidad reemplazada por un fallback configurable en las tres categorías.
- Segmentos RL son multiplicadores de sensibilidad, no cohortes estimadas del CSV.
- El entorno no modela inventario, faltantes, costes variables ni competencia.
- No hay FID/IS ni evaluación humana de imágenes; la accuracy condicional solo mide reconocimiento de clase.
- Fashion-MNIST no corresponde al surtido de una tienda UrbanStyle real.
- Evaluar agentes con más pasos y múltiples semillas, calibrar demanda con datos experimentales y conectar un catálogo visual propio serían los siguientes pasos.

## Interfaz y QA

La interfaz usa en `config.yaml` una paleta azul/negro/blanco: fondo `#060A14`, panel `#0C1424`, elevado `#121D33`, primario `#2F6BFF`, cielo `#7CC4FF` y texto `#F4F7FC`. El chequeo de Fase 8 genera capturas headless de Inicio y A–D a 1280×720, 1366×768 y 1920×1080. Bounds y contraste WCAG AA pasan para el texto principal y atenuado sobre fondo, panel y panel elevado.

El perfil de render de 30 frames por vista y resolución registró p95 máximo de 3.23, 3.32 y 4.29 ms, respectivamente, bajo el presupuesto de 16.67 ms/frame. Es una medición headless local, no una garantía de refresco físico. La aplicación limita el loop a 60 FPS y las tareas pesadas de escenario/campaña se ejecutan en workers.

## Reproducibilidad

Python 3.10/3.11, dependencias fijadas en `requirements.txt`, semillas configurables en `config.yaml`; NumPy, PyTorch, Python, Gymnasium y los entornos usan semillas. El pipeline escribe resultados bajo `outputs/` y los pesos bajo `models/`; ambos directorios de resultados pesados están ignorados por Git.

## Créditos y licencias

- Retail Sales Dataset, Mohammad Talib, Kaggle: licencia CC0 indicada en la ficha del dataset.
- Fashion-MNIST, Zalando Research: licencia MIT; artículo de Han Xiao, Kashif Rasul y Roland Vollgraf, arXiv:1708.07747.
- PyTorch, Torchvision, Gymnasium, Stable-Baselines3, Pygame, pandas, NumPy, SciPy y Matplotlib: proyectos de código abierto; consultar las licencias de sus distribuciones.
