# Definición Conceptual de Tres Modelos de Redes Neuronales para Supermercados

Basado en el análisis del dataset "supermarket_sales.xlsx", la rúbrica de evaluación y las recomendaciones previas, se proponen los siguientes tres modelos de redes neuronales. Cada modelo aborda un problema relevante para los dueños de supermercados minoristas y se alinea con los criterios de la rúbrica.

## Modelo 1: Predicción de la Calificación del Cliente (Regresión)

*   **Justificación (Alineación con Rúbrica R1-a, R1-b):**
    *   El archivo `pasted_content.txt` y la presentación analizada previamente ("Prediccion-de-Satisfaccion-del-Cliente-con-Redes-Neuronales.pdf") destacan la importancia de predecir la satisfacción del cliente. El dataset `supermarket_sales.xlsx` contiene la columna "Rating", que es una medida directa de la satisfacción.
    *   **Problema:** Los supermercados necesitan entender qué factores impactan la satisfacción del cliente para poder mejorarla, fidelizar clientes y aumentar la rentabilidad.
    *   **Necesidad:** Un modelo predictivo permite identificar estos factores y anticipar posibles calificaciones bajas, permitiendo acciones correctivas.
    *   **Sistemas Existentes:** Aunque se pueden usar análisis estadísticos tradicionales, las redes neuronales pueden capturar relaciones no lineales complejas entre múltiples variables (demográficas, de producto, transaccionales, temporales) que los métodos más simples podrían pasar por alto.

*   **Objetivos (Alineación con Rúbrica R1-c):
    *   **General:** Desarrollar un modelo de red neuronal capaz de predecir la calificación ("Rating") que un cliente otorgará a su experiencia de compra, basándose en las características de la transacción y del cliente.
    *   **Específicos:**
        1.  Identificar las variables más influyentes en la calificación del cliente (ej. tipo de cliente, línea de producto, hora del día, método de pago, sucursal, ciudad, género, total de la compra).
        2.  Proporcionar una herramienta que permita a los gerentes de supermercado simular escenarios y entender cómo cambios en ciertas variables podrían afectar la satisfacción.
        3.  Establecer una base para futuras estrategias de mejora de la experiencia del cliente.

*   **Metodología Propuesta (Alineación con Rúbrica R1-c, R2-a):
    *   **Preprocesamiento de Datos:**
        *   Manejo de variables categóricas (Branch, City, Customer type, Gender, Product line, Payment): Aplicar One-Hot Encoding o Embedding Layers.
        *   Manejo de variables temporales (Date, Time): Extraer características relevantes como día de la semana, mes, hora del día (mañana, tarde, noche).
        *   Normalización/Escalado de variables numéricas (Unit price, Quantity, Tax 5%, Total, cogs, gross income): Usar StandardScaler o MinMaxScaler.
        *   Selección de características: Analizar la relevancia de cada variable para la predicción del "Rating".
    *   **Arquitectura de la Red Neuronal:**
        *   Tipo: Red Neuronal Densa (Feedforward Neural Network).
        *   Capas de Entrada: Acorde al número de características seleccionadas tras el preprocesamiento.
        *   Capas Ocultas: Múltiples capas ocultas (ej. 2-3 capas) con un número decreciente de neuronas (ej. 128, 64, 32). Funciones de activación ReLU.
        *   Capa de Salida: Una única neurona con activación lineal (para regresión directa del rating) o sigmoidea si el rating se normaliza entre 0 y 1 y luego se re-escala.
    *   **Entrenamiento y Validación:**
        *   Función de Pérdida: Mean Squared Error (MSE) o Mean Absolute Error (MAE).
        *   Optimizador: Adam o RMSprop.
        *   Métricas de Evaluación: MSE, MAE, R-cuadrado (R²).
        *   División de Datos: Conjuntos de entrenamiento, validación y prueba.
        *   Técnicas para evitar overfitting: Dropout, Regularización L1/L2.

## Modelo 2: Segmentación de Clientes mediante Autoencoders y Clustering

*   **Justificación (Alineación con Rúbrica R1-a, R1-b):**
    *   `pasted_content.txt` sugiere la "Segmentación Dinámica de Clientes mediante Autoencoders y Clustering Neuronal".
    *   **Problema:** Los supermercados atienden a una clientela diversa con diferentes necesidades y comportamientos de compra. Una estrategia única para todos es ineficiente.
    *   **Necesidad:** Identificar grupos homogéneos de clientes (segmentos) permite personalizar estrategias de marketing, promociones, surtido de productos y servicios, mejorando la relevancia y efectividad de las acciones comerciales.
    *   **Sistemas Existentes:** Métodos de clustering tradicionales como K-Means pueden ser efectivos, pero los autoencoders pueden ayudar a aprender representaciones de datos más ricas y no lineales en espacios de menor dimensionalidad, lo que puede mejorar la calidad de la segmentación, especialmente con datos complejos.

*   **Objetivos (Alineación con Rúbrica R1-c):
    *   **General:** Desarrollar un modelo de aprendizaje no supervisado para segmentar a los clientes del supermercado en grupos distintos y significativos, basados en su comportamiento de compra y características demográficas.
    *   **Específicos:**
        1.  Utilizar un autoencoder para aprender representaciones latentes (embeddings) de los perfiles de los clientes.
        2.  Aplicar un algoritmo de clustering sobre estas representaciones latentes para identificar los segmentos.
        3.  Caracterizar cada segmento identificado analizando sus variables predominantes (ej. productos más comprados, gasto promedio, tipo de cliente, frecuencia, etc.).
        4.  Proporcionar información accionable para que los minoristas puedan diseñar estrategias específicas para cada segmento.

*   **Metodología Propuesta (Alineación con Rúbrica R1-c, R2-a):
    *   **Preprocesamiento de Datos:**
        *   Selección de características relevantes para la segmentación: Customer type, Gender, Product line, Payment, cogs, Quantity, Unit price, Total. Se podrían derivar variables como gasto total por cliente (si hubiera ID de cliente único y múltiples transacciones, lo cual no es directo aquí, pero se puede trabajar a nivel de transacción representativa del tipo de cliente), frecuencia de compra (difícil sin historial longitudinal), diversidad de productos.
        *   Codificación de categóricas y normalización/escalado de numéricas, similar al Modelo 1.
    *   **Arquitectura del Modelo:**
        *   **Autoencoder:**
            *   Encoder: Varias capas densas que reducen progresivamente la dimensionalidad de la entrada hasta una capa cuello de botella (espacio latente).
            *   Decoder: Varias capas densas que reconstruyen la entrada original a partir de la representación del espacio latente.
            *   Funciones de activación: ReLU en capas ocultas, sigmoide o lineal en la capa de salida del decoder dependiendo de la normalización de los datos de entrada.
        *   **Clustering:**
            *   Opción 1: Aplicar K-Means o DBSCAN sobre los embeddings generados por el encoder.
            *   Opción 2 (Más integrada): Implementar Deep Embedding Clustering (DEC), que optimiza simultáneamente los parámetros del autoencoder y los centroides de los clusters.
    *   **Entrenamiento y Validación:**
        *   Autoencoder: Pérdida de reconstrucción (ej. MSE entre la entrada y la salida del decoder).
        *   DEC: Combinación de pérdida de reconstrucción y una pérdida de clustering (ej. KL divergence entre la distribución de asignación de puntos a clusters y una distribución objetivo).
        *   Métricas de Evaluación de Clustering: Coeficiente de Silueta, Índice de Davies-Bouldin. Análisis cualitativo de la interpretabilidad y separabilidad de los segmentos.
        *   Determinación del número de clusters (k): Método del codo, análisis de silueta, o conocimiento del negocio.

## Modelo 3: Predicción de la Siguiente Línea de Producto (Clasificación Multiclase)

*   **Justificación (Alineación con Rúbrica R1-a, R1-b):**
    *   Relacionado con la idea de "Sistema de Recomendación Personalizado" de `pasted_content.txt`. Aunque el dataset no permite un historial de cliente robusto para recomendaciones personalizadas complejas, podemos enfocarlo en predecir la categoría de producto de interés.
    *   **Problema:** Los supermercados buscan optimizar el inventario, mejorar la disposición de los productos en tienda (planogramas) y realizar promociones efectivas.
    *   **Necesidad:** Predecir qué tipo de producto es probable que un cliente compre a continuación (o en una transacción dada) puede ayudar en la gestión de stock, marketing dirigido (ej. cupones para categorías de interés) y potencialmente en la venta cruzada.
    *   **Sistemas Existentes:** Reglas de asociación (Apriori) pueden encontrar productos comprados juntos, pero las redes neuronales pueden modelar secuencias o probabilidades condicionadas a un conjunto más amplio de características del cliente y la transacción.

*   **Objetivos (Alineación con Rúbrica R1-c):
    *   **General:** Desarrollar un modelo de red neuronal para predecir la línea de producto ("Product line") que un cliente comprará, dadas otras características de la transacción y del cliente.
    *   **Específicos:**
        1.  Identificar los factores (ej. tipo de cliente, género, hora del día, sucursal, compras previas en la misma transacción si se modela secuencialmente) que influyen en la elección de una línea de producto.
        2.  Proporcionar probabilidades de compra para cada línea de producto, permitiendo enfocar esfuerzos promocionales.
        3.  Servir como un componente básico para un sistema de recomendación más simple o para la optimización del surtido.

*   **Metodología Propuesta (Alineación con Rúbrica R1-c, R2-a):
    *   **Preprocesamiento de Datos:**
        *   Variable Objetivo: "Product line" (categórica).
        *   Características de Entrada: Branch, City, Customer type, Gender, Unit price, Quantity, Tax 5%, Total, Date (procesada), Time (procesada), Payment, cogs, gross income, Rating. Se podría experimentar excluyendo variables directamente calculadas a partir de la línea de producto si causan fuga de datos (ej. cogs, total si están muy correlacionados con una línea específica antes de la predicción).
        *   Codificación de categóricas y normalización/escalado de numéricas, similar al Modelo 1.
    *   **Arquitectura de la Red Neuronal:**
        *   Tipo: Red Neuronal Densa (Feedforward Neural Network).
        *   Capas de Entrada: Acorde al número de características.
        *   Capas Ocultas: Múltiples capas ocultas (ej. 2-3 capas) con funciones de activación ReLU.
        *   Capa de Salida: N neuronas, donde N es el número de categorías únicas en "Product line" (6 en este dataset). Función de activación Softmax para obtener una distribución de probabilidad sobre las clases.
    *   **Entrenamiento y Validación:**
        *   Función de Pérdida: Entropía Cruzada Categórica (Categorical Crossentropy).
        *   Optimizador: Adam o RMSprop.
        *   Métricas de Evaluación: Precisión (Accuracy), F1-Score (ponderado o por clase), Matriz de Confusión, Curva ROC AUC (para cada clase en enfoque One-vs-Rest).
        *   División de Datos: Conjuntos de entrenamiento, validación y prueba.
        *   Manejo de Desbalance de Clases (si existe para "Product line"): Técnicas como ponderación de clases (class weights), oversampling (SMOTE) o undersampling.

Estos tres modelos cubren diferentes tipos de tareas de aprendizaje automático (regresión, clustering/reducción de dimensionalidad y clasificación) y ofrecen valor tangible a los supermercados minoristas, alineándose con los criterios de innovación, aplicabilidad y rigor metodológico esperados por la rúbrica.
