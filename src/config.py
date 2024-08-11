import json
import pandas as pd

# elegir tipo de variables a entrenar ["small", "medium", "all"]

def load_features(tipo_dataset):
    feature_metadata = json.load(open(f"v4.3/features.json"))
    feature_sets = feature_metadata["feature_sets"]
    feature_set = feature_sets[tipo_dataset]
    return feature_set


def load_balanced_data(feature_set):

    df = pd.read_parquet(
        f"datos_balanceados/df.parquet",
        columns=["era", "target"] + feature_set
            )
    
    return df


def load_validation_data(feature_set, train):
    validation = pd.read_parquet(
      f"v4.3/validation_int8.parquet",
      columns=["era", "data_type", "target"] + feature_set
  )
    validation = validation[validation["data_type"] == "validation"]
    del validation["data_type"]

        # Downsample to every 4th era to reduce memory usage and speedup evaluation (suggested for Colab free tier)
        # Comment out the line below to use all the data (slower and higher memory usage, but more accurate evaluation)
    validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

        # Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
        # so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
    last_train_era = int(train["era"].unique()[-1])
    eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
    validation = validation[~validation["era"].isin(eras_to_embargo)]

    return validation





def save_model_params(model, nombre_modelo):
    import os

    nombre_carpeta = 'pesos_modelos'

    # Verifica si la carpeta ya existe
    if not os.path.exists(nombre_carpeta):
        # Crea la carpeta si no existe
        os.makedirs(nombre_carpeta)
        print(f'Carpeta "{nombre_carpeta}" creada.')
    else:
        print(f'La carpeta "{nombre_carpeta}" ya existe.')
    
    # Obtener los parámetros del modelo
    import pickle

    # Guardar el modelo en un archivo
    with open(f'{nombre_carpeta}/{nombre_modelo}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Modelo de regresión lineal guardado en 'linear_regression_model.pkl'.")

def load_model(nombre_modelo):
    import pickle

    # Cargar el modelo desde el archivo
    with open(f'pesos_modelos/{nombre_modelo}_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    print(f"Modelo de {nombre_modelo} cargado desde '{nombre_modelo}.pkl'.")

    return loaded_model