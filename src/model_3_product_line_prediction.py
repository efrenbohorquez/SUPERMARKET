# Modelo 3: Predicción de la Línea de Producto (Clasificación Multiclase)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical # For target variable
import os
import matplotlib.pyplot as plt
import seaborn as sns # For confusion matrix heatmap

# Crear directorios para guardar modelos y resultados si no existen
MODEL_DIR = "/home/ubuntu/supermarket_nn_models/models"
RESULTS_DIR = "/home/ubuntu/supermarket_nn_models/results"
FIGURES_DIR = "/home/ubuntu/supermarket_nn_models/results/figures"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_3_PATH = os.path.join(MODEL_DIR, "model_3_product_line_prediction.keras")
RESULTS_3_FILE = os.path.join(RESULTS_DIR, "model_3_classification_results.txt")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(FIGURES_DIR, "model_3_confusion_matrix.png")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "model_3_label_encoder.npy")

def load_and_preprocess_data_classification(file_path="/home/ubuntu/upload/supermarket_sales.xlsx"):
    """Carga y preprocesa los datos para clasificación de línea de producto."""
    df = pd.read_excel(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Hour"] = df["Time"].apply(lambda x: x.hour)

    # Características de entrada y variable objetivo
    features = [
        "Branch", "City", "Customer type", "Gender", 
        "Unit price", "Quantity", "Total", "Payment", "cogs", "gross income", "Rating",
        "DayOfWeek", "Hour"
    ]
    target = "Product line"

    X = df[features]
    y_raw = df[target]

    # Codificar la variable objetivo
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    # Guardar el encoder para decodificar predicciones después
    np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
    print(f"Clases de la variable objetivo: {label_encoder.classes_}")
    num_classes = len(label_encoder.classes_)

    numeric_features = ["Unit price", "Quantity", "Total", "cogs", "gross income", "Rating", "DayOfWeek", "Hour"]
    categorical_features = ["Branch", "City", "Customer type", "Gender", "Payment"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)])

    return X, y, preprocessor, num_classes, label_encoder

def build_classification_model(input_shape, num_classes):
    """Construye el modelo de red neuronal para clasificación multiclase."""
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")  # Softmax para clasificación multiclase
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    print("\nIniciando Modelo 3: Predicción de Línea de Producto")
    X, y_encoded, preprocessor, num_classes, label_encoder = load_and_preprocess_data_classification()

    # Convertir y_encoded a formato categórico para Keras (one-hot encoding)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    X_train, X_test, y_train_cat, y_test_cat = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded # stratify por y_encoded (antes de to_categorical)
    )
    # También necesitamos y_test_encoded para classification_report y confusion_matrix
    _, _, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    print(f"Forma de X_train_processed: {X_train_processed.shape}")

    model = build_classification_model(X_train_processed.shape[1], num_classes)
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    history = model.fit(
        X_train_processed,
        y_train_cat,
        epochs=150, # Puede necesitar más épocas para clasificación
        validation_split=0.2,
        callbacks=[early_stopping],
        batch_size=32,
        verbose=1
    )

    # Evaluación del modelo
    loss, accuracy = model.evaluate(X_test_processed, y_test_cat, verbose=0)
    print(f"\nResultados de la Evaluación del Modelo 3:")
    print(f"Pérdida (Categorical Crossentropy) en el conjunto de prueba: {loss:.4f}")
    print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy:.4f}")

    # Predicciones
    y_pred_proba = model.predict(X_test_processed)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)

    # Métricas de clasificación detalladas
    report = classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_, zero_division=0)
    print("\nInforme de Clasificación:")
    print(report)

    # Matriz de confusión
    cm = confusion_matrix(y_test_encoded, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Matriz de Confusión - Modelo 3")
    plt.ylabel("Clase Verdadera")
    plt.xlabel("Clase Predicha")
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    print(f"Gráfico de Matriz de Confusión guardado en: {CONFUSION_MATRIX_PLOT_PATH}")

    # Guardar el modelo
    model.save(MODEL_3_PATH)
    print(f"Modelo 3 guardado en: {MODEL_3_PATH}")

    # Guardar resultados de evaluación
    with open(RESULTS_3_FILE, "w") as f:
        f.write("Resultados de la Evaluación del Modelo 3: Predicción de Línea de Producto\n")
        f.write("--------------------------------------------------------------------------\n")
        f.write(f"Pérdida (Categorical Crossentropy) en el conjunto de prueba: {loss:.4f}\n")
        f.write(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy:.4f}\n\n")
        f.write("Informe de Clasificación:\n")
        f.write(report + "\n\n")
        f.write("Arquitectura del Modelo:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Resultados de la evaluación guardados en: {RESULTS_3_FILE}")

if __name__ == "__main__":
    main()


