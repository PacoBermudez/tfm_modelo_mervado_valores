import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

def normalizar_datos(train, feature_set):
    scaler = StandardScaler()

    features =  train[feature_set]
    targets = train["target"]

    scaled_features = scaler.fit_transform(features)

    # Configurar la ventana de tiempo para la RNN
    look_back = 10  # Número de pasos atrás en la secuencia que la RNN considerará
    X, y = [], []

    for i in range(look_back, len(scaled_features)):
        X.append(scaled_features[i-look_back:i])
        y.append(targets[i])

    X, y = np.array(X), np.array(y)

    # Reshape para que sea compatible con la entrada de la RNN
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    return X,y

def model_rnn_estructura(X,y):
    model = Sequential()

    # Capa SimpleRNN
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))

    # Segunda capa SimpleRNN
    model.add(SimpleRNN(units=50))
    model.add(Dropout(0.2))

    # Capa de salida
    model.add(Dense(units=1, activation='linear'))  # 'sigmoid' para predicción binaria

    # Compilar el modelo
    model.compile(optimizer='adam', lloss='mean_squared_error', metrics=['mean_absolute_error'])

    model.fit(X, y, epochs=10, batch_size=32)

    return model


def model_rnn(train, feature_set):
    X,y = normalizar_datos(train, feature_set)
    model = model_rnn_estructura(X,y)

    return model