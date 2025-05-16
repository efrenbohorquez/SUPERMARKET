# Modelo 2: Segmentación de Clientes (Autoencoder + Clustering)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # For visualization

# Crear directorios para guardar modelos y resultados si no existen
MODEL_DIR = "/home/ubuntu/supermarket_nn_models/models"
RESULTS_DIR = "/home/ubuntu/supermarket_nn_models/results"
FIGURES_DIR = "/home/ubuntu/supermarket_nn_models/results/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_2_AUTOENCODER_PATH = os.path.join(MODEL_DIR, "model_2_autoencoder.keras")
MODEL_2_ENCODER_PATH = os.path.join(MODEL_DIR, "model_2_encoder.keras")
RESULTS_2_FILE = os.path.join(RESULTS_DIR, "model_2_segmentation_results.txt")
CLUSTER_PLOT_PATH = os.path.join(FIGURES_DIR, "model_2_customer_segments.png")

def load_and_preprocess_data_segmentation(file_path="/home/ubuntu/upload/supermarket_sales.xlsx"):
    """Carga y preprocesa los datos para segmentación."""
    df = pd.read_excel(file_path)
    
    # Para segmentación, podríamos querer agregar más información del cliente si estuviera disponible.
    # Por ahora, usaremos características de la transacción como proxy del comportamiento.
    # Convertir Date y Time a características más útiles
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Hour"] = df["Time"].apply(lambda x: x.hour)

    # Seleccionar características para segmentación
    # Incluimos variables que describen el comportamiento de compra y tipo de cliente
    features_segmentation = [
        "Branch", "City", "Customer type", "Gender", "Product line",
        "Unit price", "Quantity", "Total", "Payment", "cogs", "gross income", "Rating",
        "DayOfWeek", "Hour"
    ]
    X_seg = df[features_segmentation].copy()

    numeric_features = ["Unit price", "Quantity", "Total", "cogs", "gross income", "Rating", "DayOfWeek", "Hour"]
    categorical_features = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]

    # Imputación y escalado para numéricas
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())])

    # Imputación y OneHotEncoding para categóricas
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # sparse_output=False para autoencoder
    ])

    preprocessor_seg = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ], remainder="passthrough" # En caso de que alguna columna no se liste explícitamente
    )
    
    X_processed_seg = preprocessor_seg.fit_transform(X_seg)
    return X_processed_seg, preprocessor_seg, X_seg # Devolvemos X_seg para análisis posterior de clusters

def build_autoencoder(input_dim, encoding_dim=32):
    """Construye el modelo autoencoder."""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation="relu")(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dropout(0.2)(encoded)
    encoder_output = Dense(encoding_dim, activation="relu")(encoded) # Capa cuello de botella
    
    encoder_model = Model(input_layer, encoder_output, name="encoder")

    # Decoder
    decoded = Dense(64, activation="relu")(encoder_output)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dropout(0.2)(decoded)
    decoder_output = Dense(input_dim, activation="sigmoid")(decoded) # Sigmoid si los datos están escalados a [0,1] o normalizados
                                                                # Si se usa StandardScaler, la activación lineal podría ser mejor, o re-escalar datos a [0,1]
                                                                # Para este ejemplo, asumimos que StandardScaler es aceptable con sigmoid y que la red aprenderá.

    autoencoder_model = Model(input_layer, decoder_output, name="autoencoder")
    autoencoder_model.compile(optimizer="adam", loss="mean_squared_error")
    return autoencoder_model, encoder_model

def main():
    print("\nIniciando Modelo 2: Segmentación de Clientes")
    X_processed_seg, preprocessor_seg, X_original_seg = load_and_preprocess_data_segmentation()
    
    input_dim = X_processed_seg.shape[1]
    encoding_dim = 32 # Dimensión del espacio latente, se puede ajustar

    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.summary()
    encoder.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Entrenar el autoencoder
    # Para autoencoders, la entrada y la salida son las mismas (X_processed_seg)
    history_ae = autoencoder.fit(
        X_processed_seg, X_processed_seg, 
        epochs=100, 
        batch_size=32, 
        shuffle=True, 
        validation_split=0.2, # Usar parte del training set para validación interna
        callbacks=[early_stopping],
        verbose=1
    )

    # Guardar el autoencoder y el encoder
    autoencoder.save(MODEL_2_AUTOENCODER_PATH)
    encoder.save(MODEL_2_ENCODER_PATH)
    print(f"Modelo Autoencoder guardado en: {MODEL_2_AUTOENCODER_PATH}")
    print(f"Modelo Encoder guardado en: {MODEL_2_ENCODER_PATH}")

    # Obtener las representaciones latentes (embeddings)
    X_encoded = encoder.predict(X_processed_seg)
    print(f"Forma de los datos codificados (embeddings): {X_encoded.shape}")

    # Aplicar K-Means sobre los embeddings
    # Determinar el número óptimo de clusters (ej. método del codo o silueta)
    # Aquí usaremos un número fijo para el ejemplo, por ejemplo 4 clusters
    n_clusters = 4 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_encoded)

    # Añadir las etiquetas de cluster al dataframe original para análisis
    X_original_seg["Cluster"] = cluster_labels

    # Evaluar la calidad del clustering
    silhouette_avg = silhouette_score(X_encoded, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(X_encoded, cluster_labels)
    print(f"\nResultados de la Segmentación (K-Means sobre embeddings del Autoencoder):")
    print(f"Número de Clusters: {n_clusters}")
    print(f"Coeficiente de Silueta: {silhouette_avg:.4f}")
    print(f"Índice de Davies-Bouldin: {davies_bouldin_avg:.4f}")

    # Guardar resultados
    with open(RESULTS_2_FILE, "w") as f:
        f.write("Resultados de la Evaluación del Modelo 2: Segmentación de Clientes\n")
        f.write("---------------------------------------------------------------------\n")
        f.write(f"Dimensión del espacio latente (encoding_dim): {encoding_dim}\n")
        f.write(f"Número de Clusters (K-Means): {n_clusters}\n")
        f.write(f"Coeficiente de Silueta: {silhouette_avg:.4f}\n")
        f.write(f"Índice de Davies-Bouldin: {davies_bouldin_avg:.4f}\n\n")
        f.write("Autoencoder Arquitectura:\n")
        autoencoder.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("\nEncoder Arquitectura:\n")
        encoder.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("\nPrimeras 5 filas con etiquetas de cluster:\n")
        f.write(X_original_seg.head().to_string() + "\n")
    print(f"Resultados de la segmentación guardados en: {RESULTS_2_FILE}")

    # Visualización de los clusters (usando PCA para reducir a 2D los embeddings)
    if X_encoded.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_encoded_pca = pca.fit_transform(X_encoded)
    else:
        X_encoded_pca = X_encoded # Si ya tiene 2 dimensiones o menos

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_encoded_pca[:, 0], X_encoded_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
    plt.title(f"Segmentos de Clientes (PCA de Embeddings, k={n_clusters})")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=set(cluster_labels), title="Clusters")
    plt.grid(True)
    plt.savefig(CLUSTER_PLOT_PATH)
    print(f"Gráfico de clusters guardado en: {CLUSTER_PLOT_PATH}")
    # plt.show() # Descomentar si se ejecuta en un entorno con GUI

if __name__ == "__main__":
    main()

