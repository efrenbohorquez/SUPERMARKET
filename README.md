# SUPERMARKET
El proyecto "Modelos de Redes Neuronales para la Optimización de Supermercados Minoristas" consiste en el desarrollo de tres modelos de inteligencia artificial, .  

## Estructura del Repositorio

```
supermarket_nn_models/
├── data/                     # Datos de entrada (ej. supermarket_sales.xlsx, si se incluye)
├── docs/
│   └── modelos_conceptuales.md # Definición detallada de los modelos
├── models/                   # Modelos entrenados guardados (.keras, .npy para encoders)
│   ├── model_1_rating_prediction.keras
│   ├── model_2_autoencoder.keras
│   ├── model_2_encoder.keras
│   ├── model_3_product_line_prediction.keras
│   └── model_3_label_encoder.npy
├── results/
│   ├── figures/              # Gráficos generados (ej. clusters, matriz de confusión)
│   │   ├── model_2_customer_segments.png
│   │   └── model_3_confusion_matrix.png
│   ├── model_1_evaluation_results.txt
│   ├── model_2_segmentation_results.txt
│   └── model_3_classification_results.txt
├── src/
│   ├── model_1_rating_prediction.py
│   ├── model_2_customer_segmentation.py
│   └── model_3_product_line_prediction.py
├── web/                      # Archivos para la publicación web (HTML, CSS, JS)
│   ├── index.html
│   └── style.css
├── supermarket_data_analysis.txt # Análisis exploratorio inicial de los datos
├── requirements.txt          # Dependencias de Python
└── README.md                 # Este archivo
```

## Modelos Desarrollados

1.  **Predicción de la Calificación del Cliente (Regresión):**
    *   **Objetivo:** Predecir la calificación que un cliente podría dar a su experiencia de compra.
    *   **Utilidad para Minoristas:** Ayuda a identificar factores clave que influyen en la satisfacción del cliente, permitiendo tomar acciones proactivas para mejorar la experiencia y fidelizar clientes.
    *   **Script:** `src/model_1_rating_prediction.py`

2.  **Segmentación de Clientes (Autoencoder + Clustering):**
    *   **Objetivo:** Agrupar clientes en segmentos con comportamientos y características similares.
    *   **Utilidad para Minoristas:** Permite personalizar estrategias de marketing, promociones y surtido de productos para cada segmento, aumentando la efectividad de las campañas y la satisfacción del cliente.
    *   **Script:** `src/model_2_customer_segmentation.py`

3.  **Predicción de la Línea de Producto Comprada (Clasificación Multiclase):**
    *   **Objetivo:** Predecir qué categoría de producto es más probable que un cliente compre.
    *   **Utilidad para Minoristas:** Facilita la optimización del inventario, la planificación de la disposición de productos en tienda (planogramas) y la creación de ofertas y promociones dirigidas.
    *   **Script:** `src/model_3_product_line_prediction.py`

## Requisitos Previos

*   Python 3.9+
*   pip (manejador de paquetes de Python)

## Instalación de Dependencias

1.  Clona este repositorio (o descarga los archivos).
2.  Navega al directorio raíz del proyecto `supermarket_nn_models`.
3.  Crea un entorno virtual (recomendado):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # En Linux/macOS
    # venv\Scripts\activate    # En Windows
    ```
4.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
5.  Asegúrate de tener el archivo `supermarket_sales.xlsx` en la ruta esperada por los scripts (actualmente configurado como `/home/ubuntu/upload/supermarket_sales.xlsx`). Puedes moverlo a la carpeta `data/` y ajustar las rutas en los scripts si es necesario.

## Ejecución de los Modelos

Cada modelo se puede ejecutar individualmente desde la carpeta `src/`:

```bash
python3 src/model_1_rating_prediction.py
python3 src/model_2_customer_segmentation.py
python3 src/model_3_product_line_prediction.py
```

Los modelos entrenados se guardarán en la carpeta `models/` y los resultados de la evaluación y gráficos en la carpeta `results/`.

## Publicación Web

Se ha generado una página web simple en la carpeta `web/` que explica estos modelos y su utilidad para los dueños de supermercados. Puedes abrir el archivo `web/index.html` en un navegador para visualizarla.

## Alineación con la Rúbrica Académica

Este proyecto busca cumplir con los criterios de evaluación académica, incluyendo:

*   **R1 (Propuesta):** Identificación clara del problema, justificación de la necesidad, objetivos definidos y metodología propuesta (ver `docs/modelos_conceptuales.md`).
*   **R2 (Medio Término):** Diseño metodológico (preprocesamiento, arquitecturas de red), planificación (scripts modulares).
*   **R3 (Final):** Demostración del proyecto (scripts funcionales, resultados), presentación (este README, publicación web).
*   **R4 (Informe):** Descripción de conceptos, detalles técnicos (código comentado), conclusiones (resultados en `results/`), discusión (utilidad para minoristas).
*   **R5 (Evaluación Blanda):** Conocimiento técnico aplicado.

## Contribuciones

Este proyecto fue desarrollado por Manus, un agente de IA.

## Licencia

(Especificar licencia si es necesario, por ejemplo, MIT License)
