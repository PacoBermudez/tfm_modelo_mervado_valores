import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Cargar los datos de Numerai Signals
data = pd.read_csv('numerai_signals_data.csv')  # Asegúrate de ajustar la ruta

# Ordenar los datos por fecha y ticker
data = data.sort_values(['ticker', 'date'])

# Escalar los features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[['feature1', 'feature2', ...]])  # Ajustar según los nombres de las columnas

# Añadir las features escaladas al DataFrame
data[['feature1', 'feature2', ...]] = scaled_features

# Crear un diccionario para almacenar los datos de cada empresa
companies = data['ticker'].unique()
company_data = {company: data[data['ticker'] == company] for company in companies}

# Función para crear secuencias de datos para LSTM y mantener un seguimiento de los tickers
def create_sequences(data, ticker, time_steps=10):
    X, y, tickers = [], [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :-1])  # Usar todas las columnas excepto la última (target)
        y.append(data[i + time_steps, -1])       # Usar solo la última columna (target)
        tickers.append(ticker)                   # Mantener el ticker correspondiente a la secuencia
    return np.array(X), np.array(y), tickers

# Crear las secuencias de entrenamiento para cada empresa
X_train, y_train, tickers_train = [], [], []
for company, company_df in company_data.items():
    company_values = company_df[['feature1', 'feature2', ..., 'target']].values  # Ajustar según los nombres de las columnas
    X, y, tickers = create_sequences(company_values, company)
    X_train.append(X)
    y_train.append(y)
    tickers_train.extend(tickers)

# Concatenar los datos de todas las empresas
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
tickers_train = np.array(tickers_train)  # Convertir la lista de tickers en un array

# Dividir los datos en entrenamiento y validación manteniendo los tickers
X_train, X_val, y_train, y_val, tickers_train, tickers_val = train_test_split(
    X_train, y_train, tickers_train, test_size=0.2, random_state=42
)

# Verificar las formas de los datos
print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de y_train: {y_train.shape}")
print(f"Tickers de entrenamiento: {len(tickers_train)}")


# Definir el modelo LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))  # Salida única por empresa

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluar el modelo en los datos de validación
loss = model.evaluate(X_val, y_val)
print(f"Loss en validación: {loss}")


# Predicciones en los datos de validación
predictions = model.predict(X_val)

# Crear un DataFrame para visualizar las predicciones con sus tickers correspondientes
results = pd.DataFrame({
    'Ticker': tickers_val,
    'Real': y_val,
    'Predicción': predictions.flatten()
})

# Mostrar algunas filas de resultados
print(results.head())

# Graficar resultados de una empresa específica
import matplotlib.pyplot as plt

# Seleccionar un ticker para graficar
selected_ticker = 'AAPL'  # Cambia según el ticker de interés
ticker_results = results[results['Ticker'] == selected_ticker]

plt.plot(ticker_results['Real'], label='Real')
plt.plot(ticker_results['Predicción'], label='Predicción')
plt.title(f'Resultados para {selected_ticker}')
plt.legend()
plt.show()