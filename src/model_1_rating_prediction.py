# Modelo 1: Predicción de la Calificación del Cliente (Regresión)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import os

# Crear directorios para guardar modelos y resultados si no existen
MODEL_DIR = "/home/ubuntu/supermarket_nn_models/models"
RESULTS_DIR = "/home/ubuntu/supermarket_nn_models/results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_1_PATH = os.path.join(MODEL_DIR, "model_1_rating_prediction.keras")
RESULTS_1_FILE = os.path.join(RESULTS_DIR, "model_1_evaluation_results.txt")

def load_and_preprocess_data(file_path="/home/ubuntu/upload/supermarket_sales.xlsx"):
    """Carga y preprocesa los datos del supermercado."""
    df = pd.read_excel(file_path)

    # Convertir Date y Time a características más útiles
    df["Date"] = pd.to_datetime(df["Date"])
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Hour"] = df["Time"].apply(lambda x: x.hour)

    # Seleccionar características y variable objetivo
    # Excluimos Invoice ID, Date, Time (ya procesadas), y gross margin percentage (constante)
    # cogs, Tax 5%, Total, gross income son derivados del precio y cantidad, pueden causar multicolinealidad o data leakage si se usan descuidadamente.
    # Para predecir Rating, usaremos variables que el supermercado puede conocer ANTES o durante la transacción para influir.
    
    features = [
        "Branch", "City", "Customer type", "Gender", "Product line", 
        "Unit price", "Quantity", "Payment",
        "DayOfWeek", "Month", "Hour"
    ]
    target = "Rating"

    X = df[features]
    y = df[target]

    # Definir transformadores para columnas numéricas y categóricas
    numeric_features = ["Unit price", "Quantity", "DayOfWeek", "Month", "Hour"]
    categorical_features = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]

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

    return X, y, preprocessor

def build_nn_model(input_shape):
    """Construye el modelo de red neuronal para regresión."""
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)  # Capa de salida lineal para regresión
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model

def main():
    print("Iniciando Modelo 1: Predicción de Calificación del Cliente")
    X, y, preprocessor = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar preprocesamiento
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convertir a densa si es sparse matrix (OneHotEncoder puede devolver sparse)
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    print(f"Forma de X_train_processed: {X_train_processed.shape}")

    model = build_nn_model(X_train_processed.shape[1])
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_processed,
        y_train,
        epochs=100,
        validation_split=0.2, # Usar parte del training set para validación interna
        callbacks=[early_stopping],
        batch_size=32,
        verbose=1
    )

    # Evaluación del modelo
    loss, mae = model.evaluate(X_test_processed, y_test, verbose=0)
    print(f"\nResultados de la Evaluación del Modelo 1:")
    print(f"Pérdida (MSE) en el conjunto de prueba: {loss:.4f}")
    print(f"Error Absoluto Medio (MAE) en el conjunto de prueba: {mae:.4f}")

    # Guardar el modelo
    model.save(MODEL_1_PATH)
    print(f"Modelo 1 guardado en: {MODEL_1_PATH}")

    # Guardar resultados de evaluación
    with open(RESULTS_1_FILE, "w") as f:
        f.write("Resultados de la Evaluación del Modelo 1: Predicción de Calificación\n")
        f.write("---------------------------------------------------------------------\n")
        f.write(f"Pérdida (MSE) en el conjunto de prueba: {loss:.4f}\n")
        f.write(f"Error Absoluto Medio (MAE) en el conjunto de prueba: {mae:.4f}\n")
        f.write("\nArquitectura del Modelo:\n")
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Resultados de la evaluación guardados en: {RESULTS_1_FILE}")

if __name__ == "__main__":
    main()


