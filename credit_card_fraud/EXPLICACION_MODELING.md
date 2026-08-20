# Explicacion de la carpeta `modeling`

Este documento explica como funcionan los codigos de la carpeta:

```text
house_price/modeling/
|-- __init__.py
|-- train.py
`-- predict.py
```

Aunque el proyecto se llama `house_price`, el codigo actual trabaja con el dataset `creditcard.csv`, que corresponde a transacciones con tarjeta de credito. El objetivo real del modelado es detectar fraude, usando la columna `Class` como variable objetivo:

- `Class = 0`: transaccion normal.
- `Class = 1`: transaccion fraudulenta.

## 1. Flujo general del proyecto

El archivo que une todo el proceso es `run_pipeline.py`. Su flujo es:

1. Carga los datos desde `data/raw/creditcard.csv` usando `CreditCardDataset`.
2. Prepara las variables con `FeatureEngineer`.
3. Divide los datos en entrenamiento y prueba.
4. Escala las columnas `Time` y `Amount`.
5. Entrena dos modelos de deep learning:
   - un modelo MLP supervisado;
   - un autoencoder para deteccion de anomalias.
6. Genera predicciones con ambos modelos.
7. Evalua los resultados con metricas de clasificacion.
8. Guarda una tabla comparativa en `reports/model_results.csv`.
9. Guarda una grafica Precision-Recall en `reports/figures/pr_curves.png`.

El comando principal para ejecutar todo es:

```bash
python run_pipeline.py
```

## 2. Archivo `train.py`

El archivo `train.py` contiene la logica para construir, entrenar, guardar y usar los modelos.

### 2.1 Clase `BaseModel`

`BaseModel` es una clase base abstracta. Sirve como plantilla comun para los modelos.

Define atributos y metodos compartidos:

- `input_dim`: cantidad de columnas o caracteristicas que recibe el modelo.
- `model`: guarda el modelo de Keras.
- `history`: guarda el historial del entrenamiento.
- `build()`: metodo obligatorio para construir el modelo.
- `train()`: metodo obligatorio para entrenarlo.
- `save()`: guarda el modelo entrenado en disco.
- `predict()`: devuelve las predicciones del modelo.

La idea de esta clase es que todos los modelos tengan una estructura parecida. Asi el codigo queda mas ordenado y es mas facil agregar otro modelo en el futuro.

### 2.2 Clase `MLPModel`

`MLPModel` es un modelo supervisado de tipo red neuronal multicapa, tambien llamado MLP o Multi-Layer Perceptron.

Es supervisado porque aprende usando:

- `X_train`: las variables de entrada;
- `y_train`: la etiqueta real, es decir, si la transaccion fue normal o fraudulenta.

La arquitectura del modelo es:

```text
Entrada
Dense(64, relu)
BatchNormalization
Dropout
Dense(32, relu)
BatchNormalization
Dropout
Dense(1, sigmoid)
```

Cada parte cumple una funcion:

- `Dense(64, relu)` y `Dense(32, relu)`: capas neuronales que aprenden patrones en los datos.
- `BatchNormalization`: estabiliza el entrenamiento y ayuda a que la red aprenda mejor.
- `Dropout`: apaga aleatoriamente algunas neuronas durante el entrenamiento para reducir sobreajuste.
- `Dense(1, sigmoid)`: devuelve un valor entre 0 y 1, interpretado como probabilidad de fraude.

El modelo se compila con:

```python
loss="binary_crossentropy"
metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"), "accuracy"]
```

Se usa `binary_crossentropy` porque el problema tiene dos clases: fraude o no fraude. Tambien se usa `PR-AUC` porque el dataset esta muy desbalanceado: hay muchas mas transacciones normales que fraudulentas. En estos casos, Precision-Recall suele ser mas informativo que solo accuracy.

Durante el entrenamiento se usan dos callbacks:

- `EarlyStopping`: detiene el entrenamiento si el modelo deja de mejorar.
- `ModelCheckpoint`: guarda automaticamente la mejor version del modelo.

El mejor modelo MLP se guarda en:

```text
models/model_mlp.keras
```

### 2.3 Clase `AutoencoderModel`

`AutoencoderModel` es una red neuronal no supervisada orientada a detectar anomalias.

La idea del autoencoder es aprender a reconstruir transacciones normales. Si despues recibe una transaccion fraudulenta, normalmente la reconstruye peor. Esa diferencia se llama error de reconstruccion.

La arquitectura es:

```text
Entrada
Dense(20, tanh)
Dense(encoding_dim, relu)
Dense(20, tanh)
Dense(input_dim, linear)
```

El modelo tiene dos partes conceptuales:

- Codificador: comprime la informacion de entrada en una representacion mas pequena.
- Decodificador: intenta reconstruir la entrada original desde esa representacion comprimida.

En `run_pipeline.py`, el autoencoder se entrena solo con transacciones normales:

```python
X_train_normal = engineer.X_train[engineer.y_train == 0].to_numpy()
autoencoder.train(X_train_normal, epochs=EPOCHS, batch_size=BATCH_SIZE)
```

Esto es importante porque el modelo aprende el patron de lo normal. Luego, cuando una transaccion se aleja mucho de ese patron, su error de reconstruccion sube y puede considerarse sospechosa.

El autoencoder usa como perdida:

```python
loss="mean_squared_error"
```

Esto mide que tan diferente es la entrada original respecto a la reconstruida.

El mejor modelo autoencoder se guarda en:

```text
models/model_autoencoder.keras
```

### 2.4 Metodo `reconstruction_error`

Este metodo calcula el puntaje de anomalia del autoencoder.

Primero reconstruye los datos:

```python
reconstructed = self.predict(X)
```

Luego calcula el error cuadratico medio por fila:

```python
mse = np.mean(np.power(X - reconstructed, 2), axis=1)
```

Despues normaliza los errores entre 0 y 1:

```python
(mse - mse_min) / (mse_max - mse_min)
```

Mientras mas alto sea el valor, mas anomala parece la transaccion.

### 2.5 Metodo `anomaly_threshold`

Este metodo permite calcular un umbral para decidir desde que error una transaccion se considera anomala.

Usa un percentil, por defecto el 95:

```python
np.percentile(errors, percentile)
```

Eso significa que se puede tomar como sospechoso todo lo que supere el error habitual de la mayoria de transacciones normales.

## 3. Archivo `predict.py`

El archivo `predict.py` contiene la logica para cargar modelos ya entrenados, generar predicciones y evaluar resultados.

### 3.1 Constante `THRESHOLD`

```python
THRESHOLD = 0.5
```

Este valor se usa como umbral por defecto. Si la probabilidad o puntaje de un modelo es mayor o igual a 0.5, se clasifica como fraude.

Sin embargo, en `run_pipeline.py` se busca un umbral optimo para cada modelo usando el metodo `optimal_threshold()`.

### 3.2 Clase `ModelEvaluator`

`ModelEvaluator` recibe las etiquetas reales de prueba:

```python
def __init__(self, y_test: np.ndarray) -> None:
    self.y_test = y_test
```

Su funcion es comparar las predicciones de los modelos contra los valores reales.

### 3.3 Metodo `evaluate`

Este metodo recibe un diccionario de predicciones. Por ejemplo:

```python
predictions = {
    "DL 1: MLP Supervisado": ...,
    "DL 2: Autoencoder": ...,
}
```

Para cada modelo:

1. Toma sus probabilidades o puntajes.
2. Aplica un umbral.
3. Convierte los puntajes en clases:

```python
preds = (probs >= threshold).astype(int)
```

4. Calcula metricas:

- `Accuracy`: porcentaje total de aciertos.
- `Precision`: de las transacciones marcadas como fraude, cuantas realmente eran fraude.
- `Recall`: de todos los fraudes reales, cuantos encontro el modelo.
- `F1-Score`: equilibrio entre precision y recall.
- `ROC-AUC`: capacidad general de separar clases.
- `PR-AUC`: area bajo la curva Precision-Recall, util en datasets desbalanceados.

Finalmente devuelve una tabla `DataFrame` con los resultados.

### 3.4 Metodo `optimal_threshold`

Este metodo busca el mejor umbral de decision para maximizar el `F1-Score`.

En vez de usar siempre 0.5, prueba distintos umbrales generados por la curva Precision-Recall y selecciona el que obtiene mejor equilibrio entre precision y recall.

Esto es especialmente importante porque en fraude el dataset esta muy desbalanceado. Un umbral fijo puede no ser el mejor.

### 3.5 Metodo `plot_precision_recall`

Este metodo dibuja la curva Precision-Recall de cada modelo.

La curva muestra la relacion entre:

- `Precision`: que tan confiables son las alertas de fraude.
- `Recall`: que tanta cantidad de fraudes reales se logran detectar.

La grafica se guarda en:

```text
reports/figures/pr_curves.png
```

### 3.6 Funcion `run_inference`

Esta funcion carga los modelos guardados:

```python
mlp_model = keras.models.load_model(mlp_path)
autoencoder_model = keras.models.load_model(autoencoder_path)
```

Luego los coloca dentro de sus clases correspondientes:

```python
mlp_wrapper = MLPModel(X_test.shape[1])
autoencoder_wrapper = AutoencoderModel(X_test.shape[1])
```

Finalmente devuelve las predicciones:

- El MLP devuelve probabilidades de fraude.
- El autoencoder devuelve errores de reconstruccion normalizados.

## 4. Relacion con `features.py`

Aunque `features.py` no esta dentro de `modeling`, es clave para entender como llegan los datos a los modelos.

La clase `FeatureEngineer` hace tres cosas importantes:

1. Separa variables predictoras y etiqueta:

```python
features = frame.drop(target, axis=1)
labels = frame[target]
```

2. Divide en entrenamiento y prueba:

```python
train_test_split(..., stratify=labels)
```

El parametro `stratify=labels` mantiene una proporcion similar de fraudes y no fraudes en entrenamiento y prueba.

3. Escala `Amount` y `Time` con `RobustScaler`.

Se usa `RobustScaler` porque es menos sensible a valores extremos. Esto es util en transacciones, ya que los montos pueden variar mucho.

Ademas, los escaladores se ajustan solo con entrenamiento:

```python
self._amount_scaler.fit(self.X_train["Amount"].values.reshape(-1, 1))
self._time_scaler.fit(self.X_train["Time"].values.reshape(-1, 1))
```

Eso evita fuga de datos, porque el modelo no aprende informacion del conjunto de prueba.

## 5. Resultados obtenidos

El archivo `reports/model_results.csv` guarda una comparacion entre modelos. Los resultados actuales son:

| Modelo | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| DL 1: MLP Supervisado | 0.9998 | 0.9994 | 0.8283 | 0.8367 | 0.8325 | 0.9720 | 0.7095 |
| DL 2: Autoencoder | 0.0097 | 0.9971 | 0.3282 | 0.6531 | 0.4369 | 0.9697 | 0.2925 |

Interpretacion:

- El MLP obtuvo mejor `Precision`, `Recall`, `F1-Score` y `PR-AUC`.
- El autoencoder tambien separa bastante bien las clases segun `ROC-AUC`, pero tiene menor precision.
- Para este caso, el MLP supervisado es el modelo mas fuerte porque tiene etiquetas reales durante el entrenamiento.
- El autoencoder es util como enfoque alternativo cuando se quiere detectar comportamientos raros o cuando hay pocas etiquetas de fraude.

## 6. Como explicarlo oralmente al docente

Una forma sencilla de explicarlo seria:

> Mi proyecto carga un dataset de transacciones con tarjeta de credito y busca detectar fraude. Primero preparo los datos separando entrenamiento y prueba, y escalo las columnas `Time` y `Amount` para que los modelos trabajen con valores comparables. Despues entreno dos modelos de deep learning. El primero es un MLP supervisado, que aprende directamente con las etiquetas `Class`, donde 0 significa normal y 1 fraude. El segundo es un autoencoder, que aprende solamente con transacciones normales y despues detecta posibles fraudes cuando no logra reconstruir bien una transaccion. Finalmente comparo ambos modelos usando accuracy, precision, recall, F1, ROC-AUC y PR-AUC, y guardo los resultados en un CSV y una grafica Precision-Recall.

## 7. Diferencia principal entre los dos modelos

| Modelo | Tipo | Que aprende | Salida |
|---|---|---|---|
| MLP | Supervisado | Relacion entre variables y etiqueta `Class` | Probabilidad de fraude |
| Autoencoder | No supervisado / anomalias | Patron de transacciones normales | Error de reconstruccion |

## 8. Idea central del proyecto

La idea central es comparar dos estrategias de deep learning para detectar fraude:

1. Una estrategia supervisada, donde el modelo aprende con ejemplos ya etiquetados.
2. Una estrategia de anomalias, donde el modelo aprende lo normal y marca como sospechoso lo que se aleja de ese comportamiento.

Segun los resultados guardados, el MLP supervisado tiene mejor rendimiento general para este dataset.
