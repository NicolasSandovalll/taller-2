# ============================================================
# REGRESIÓN LINEAL MÚLTIPLE - PREDICCIÓN DE PRECIO DE AUTOS
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Carpeta donde se guardarán los gráficos generados
DIR_RESULTADOS = "resultados"
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# ============================================================
# SECCIÓN 1: CARGA Y EXPLORACIÓN DE DATOS
# ============================================================

df = pd.read_csv("automovil_dataset.csv")
print("=" * 60)
print("SECCIÓN 1: EXPLORACIÓN INICIAL DE DATOS")
print("=" * 60)

print("\n>>> Primeras 5 filas del dataset:")
print(df.head())

print("\n>>> Shape del dataset:", df.shape)

print("\n>>> describe():")
print(df.describe())

print("\n>>> Valores nulos por columna:")
print(df.isnull().sum())

print("\n>>> Filas duplicadas:", df.duplicated().sum())

# ============================================================
# SECCIÓN 2: ANÁLISIS DE CORRELACIÓN Y GRÁFICOS
# ============================================================

print("\n" + "=" * 60)
print("SECCIÓN 2: CORRELACIÓN CON PRICE Y GRÁFICOS")
print("=" * 60)

# Correlación de cada variable con price (ordenada descendente)
variables_indep = ["horsepower", "age", "mileage", "engine_size"]
correlaciones = df[variables_indep + ["price"]].corr()["price"].drop("price").sort_values(ascending=False)
print("\n>>> Correlación de cada variable con price (ordenada):")
for var, corr in correlaciones.items():
    print(f"    {var}: {corr:.6f}")

# Heatmap de correlación
plt.figure(figsize=(8, 6))
matriz_corr = df[variables_indep + ["price"]].corr()
sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".4f",
            linewidths=0.5, square=True,
            annot_kws={"size": 10})
plt.title("Heatmap de Correlación - Precio de Autos", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"\n>>> Guardado: {DIR_RESULTADOS}/correlation_heatmap.png")

# Scatter plot de cada variable independiente vs price
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, var in enumerate(variables_indep):
    axes[i].scatter(df[var], df["price"], alpha=0.6, edgecolors="k", linewidth=0.3)
    axes[i].set_xlabel(var.capitalize(), fontsize=11)
    axes[i].set_ylabel("Precio (price)", fontsize=11)
    axes[i].set_title(f"Scatter: {var} vs price", fontsize=12)
    # Agregar línea de tendencia
    m, b = np.polyfit(df[var], df["price"], 1)
    axes[i].plot(df[var], m * df[var] + b, color="red", linewidth=1.5)
plt.suptitle("Scatter Plots: Variables Independientes vs Price", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "scatter_price.png"), dpi=150)
plt.close()
print(f">>> Guardado: {DIR_RESULTADOS}/scatter_price.png")

# ============================================================
# SECCIÓN 3: REGRESIÓN LINEAL MÚLTIPLE
# ============================================================

print("\n" + "=" * 60)
print("SECCIÓN 3: ENTRENAMIENTO - REGRESIÓN LINEAL MÚLTIPLE")
print("=" * 60)

# Separar variables independientes (X) y dependiente (y)
X = df[variables_indep]
y = df["price"]

# División train/test 80/20 con random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n>>> Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f">>> Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

# Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones en train y test
y_train_pred = modelo.predict(X_train)
y_test_pred = modelo.predict(X_test)

# ============================================================
# SECCIÓN 4: COEFICIENTES Y MÉTRICAS
# ============================================================

print("\n" + "=" * 60)
print("SECCIÓN 4: INTERCEPTO, COEFICIENTES Y MÉTRICAS")
print("=" * 60)

print(f"\n>>> Intercepto (β₀): {modelo.intercept_:.4f}")

print("\n>>> Coeficientes (βᵢ):")
for var, coef in zip(variables_indep, modelo.coef_):
    print(f"    {var}: {coef:.4f}")

print("\n>>> MÉTRICAS:")
# R²
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"    R² (Train): {r2_train:.6f}")
print(f"    R² (Test):  {r2_test:.6f}")

# MAE
mae = mean_absolute_error(y_test, y_test_pred)
print(f"    MAE:        {mae:.4f}")

# MSE
mse = mean_squared_error(y_test, y_test_pred)
print(f"    MSE:        {mse:.4f}")

# RMSE
rmse = np.sqrt(mse)
print(f"    RMSE:       {rmse:.4f}")

# ============================================================
# SECCIÓN 5: GRÁFICOS DE EVALUACIÓN (TEST)
# ============================================================

print("\n" + "=" * 60)
print("SECCIÓN 5: GRÁFICOS DE EVALUACIÓN")
print("=" * 60)

# Gráfico de valores reales vs predichos (conjunto de test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k", linewidth=0.3)
# Línea y = x (predicción perfecta)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Predicción perfecta")
plt.xlabel("Valores Reales", fontsize=12)
plt.ylabel("Valores Predichos", fontsize=12)
plt.title(f"Valores Reales vs Predichos (Test) - R²={r2_test:.4f}", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "real_vs_predicho.png"), dpi=150)
plt.close()
print(f">>> Guardado: {DIR_RESULTADOS}/real_vs_predicho.png")

# Gráfico de residuos (test)
residuos = y_test - y_test_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuos, alpha=0.6, edgecolors="k", linewidth=0.3)
plt.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="Residuo = 0")
plt.xlabel("Valores Predichos", fontsize=12)
plt.ylabel("Residuos", fontsize=12)
plt.title("Gráfico de Residuos (Test)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_RESULTADOS, "residuos.png"), dpi=150)
plt.close()
print(f">>> Guardado: {DIR_RESULTADOS}/residuos.png")

# ============================================================
# SECCIÓN 6: PREDICCIÓN DE UN NUEVO AUTO
# ============================================================

print("\n" + "=" * 60)
print("SECCIÓN 6: PREDICCIÓN DE UN NUEVO AUTO")
print("=" * 60)

# Valores del auto a predecir
nuevo_auto = {
    "horsepower": 165,
    "age": 4,
    "mileage": 58000,
    "engine_size": 2.0
}

print("\n>>> Características del nuevo auto:")
for k, v in nuevo_auto.items():
    print(f"    {k}: {v}")

# Predicción con el modelo
X_nuevo = pd.DataFrame([nuevo_auto])
prediccion = modelo.predict(X_nuevo)[0]

# Cálculo manual: precio = intercepto + Σ(coeficiente_i * valor_i)
calculo_manual = modelo.intercept_
for i, var in enumerate(variables_indep):
    termino = modelo.coef_[i] * nuevo_auto[var]
    calculo_manual += termino
    print(f"    {var}: {modelo.coef_[i]:.4f} * {nuevo_auto[var]} = {termino:.4f}")

print(f"\n>>> Intercepto:                              {modelo.intercept_:.4f}")
print(f">>> Precio predicho (modelo):              ${prediccion:,.2f}")
print(f">>> Precio predicho (cálculo manual):      ${calculo_manual:,.2f}")
print(f">>> Diferencia modelo vs manual:           ${abs(prediccion - calculo_manual):.10f}")
print(f"\n>>> CONCLUSIÓN: El precio estimado del auto es ${prediccion:,.2f}")
