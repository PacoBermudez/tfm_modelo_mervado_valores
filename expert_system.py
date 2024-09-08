# -*- coding: utf-8 -*-
# %%
import pandas as pd
import json
import metric_visualization as metric
from numerai_tools.scoring import numerai_corr, correlation_contribution


def load_pred():
    df_pred_xgboost = pd.read_parquet(
        f"resultados_modelos/XGBoost/predicciones_XGBoost.parquet",
        
            )
    df_pred_TFT = pd.read_parquet(
        f"resultados_modelos/TFT/prediccions_TFT.parquet",
        
            )
    df_pred_RandomForest = pd.read_parquet(
        f"resultados_modelos/Random_forest/predicciones_Random_forest.parquet",
        
            )

    df_pred_RandomForest = df_pred_RandomForest.drop("Target", axis=1)

    df_pred_LIGHTGBM = pd.read_parquet(
        f"resultados_modelos/LightGBM/predicciones_LightGBM.parquet",
        
            )

    df_pred_LIGHTGBM = df_pred_LIGHTGBM.drop("Target", axis=1)

    
    df_pred_Regresion = pd.read_parquet(
        f"resultados_modelos/regresion_lineal/predicciones_regresion_lineal.parquet",
        
            )
        
    df_pred_Regresion = df_pred_Regresion.drop("Target", axis=1)


    feature_metadata = json.load(open(f"v4.3/features.json"))
    feature_sets = feature_metadata["feature_sets"]
    feature_set = feature_sets["medium"]

    train = pd.read_parquet(
        f"v4.3/train_int8.parquet",
        columns=["era", "target"] + feature_set
            )

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


    df_final = pd.concat([df_pred_xgboost, df_pred_RandomForest, df_pred_LIGHTGBM,df_pred_Regresion,df_pred_TFT, validation[["era"]]], axis=1)







    return df_final

# %%
df_analisis = load_pred()

# %%
df_analisis.columns

# %%
df_analisis = df_analisis.reset_index()


# %%
def calculo_eras(df, output_folder, nombre_modelo):

    import os
    from matplotlib.ticker import MultipleLocator


    df_eras = pd.DataFrame()

    for columna in df.columns:

        print(columna)
        
        if columna not in ["Target", "era", "id"]:
            per_era_corr = df.groupby("era").apply(
            lambda x: numerai_corr(x[[columna]].dropna(), x["Target"].dropna())
        )
            
            df_eras[columna] = per_era_corr.squeeze()

    #df_eras = df_eras.drop("Target", axis=1)
    
    color_dict = {
        'XGBoost': 'tab:blue',
        'Random_forest': 'tab:green',
        'LGBM': 'tab:orange',
        'regresion_lineal': 'tab:red',
        'TFT': 'tab:purple',
        'Sistema_experto': 'tab:cyan'
    }

    # Crear una figura más grande para la visualización
    plt.figure(figsize=(40, 10))

    # Lista para almacenar los handles y labels únicos para la leyenda
    handles_labels = {}

    for columna in df_eras.columns:
        # Determinar el grupo del modelo
        if 'XGBoost' in columna:
            group = 'XGBoost'
        elif 'Random_forest' in columna:
            group = 'Random_forest'
        elif 'LGBM' in columna:
            group = 'LGBM'
        elif 'TFT' in columna:
            group = 'TFT'
        elif 'Sistema_experto' in columna:
            group = 'Sistema_experto'
        else:
            group = 'regresion_lineal'
        
        color = color_dict[group]
        
        # Graficar la línea con el color especificado
        line, = plt.plot(df_eras[columna].cumsum(), label=columna, color=color)
        
        # Solo añadir un handle y label por grupo a la leyenda
        if group not in handles_labels:
            handles_labels[group] = line

    # Crear la leyenda solo con los nombres de los grupos
    plt.legend(handles_labels.values(), handles_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    # Añadir título y etiquetas de los ejes
    plt.title("Comparacion modelos Propios")
    plt.xlabel("era")
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.ylabel("Cumulative Sum")

    # Ajustar el layout
    plt.subplots_adjust(right=0.75)

    # Guardar el gráfico como una imagen
    plt.savefig(os.path.join(output_folder, f"Cumulative_Validation_MMC_{nombre_modelo}.png"))
    plt.show()

    return df_eras

# %%
df_eras = calculo_eras(df_analisis, "resultados_modelos", "Comparacion_modelos")

# %%
descripcion = df_eras.describe()

descripcion.to_excel("Datos_Modelos.xlsx")


suma_columnas = df_eras.sum()
suma_df = pd.DataFrame(suma_columnas).transpose()
suma_df.index = ['Suma']

result_df = pd.concat([descripcion, suma_df])
result_df

# %%
descripcion = result_df.T.sort_values("mean", ascending=False).reset_index().rename(columns = {"index":"Modelo"})


descripcion.to_excel("Datos_Modelos.xlsx")

sistema_experto = descripcion[["Modelo","mean"]].head(5)
sistema_experto = sistema_experto[["Modelo","mean"]].tail(4)

sistema_experto["Porcentaje"] = sistema_experto["mean"]/sistema_experto["mean"].sum()
sistema_experto


# %%
df_eras

# %%
import os
from matplotlib.ticker import MultipleLocator




#df_eras = df_eras.drop("Target", axis=1)

color_dict = {
    'XGBoost': 'tab:blue',
    'Random_forest': 'tab:green',
    'LGBM': 'tab:orange',
    'regresion_lineal': 'tab:red',
    'TFT': 'tab:purple',
    'Sistema_experto': 'tab:cyan'
}

# Crear una figura más grande para la visualización
plt.figure(figsize=(40, 10))

# Lista para almacenar los handles y labels únicos para la leyenda
handles_labels = {}

for columna in df_eras.columns:
    # Determinar el grupo del modelo
    if 'XGBoost' in columna:
        group = 'XGBoost'
    elif 'Random_forest' in columna:
        group = 'Random_forest'
    elif 'LGBM' in columna:
        group = 'LGBM'
    elif 'TFT' in columna:
        group = 'TFT'
    elif 'Sistema_experto' in columna:
        group = 'Sistema_experto'
    else:
        group = 'regresion_lineal'
    
    color = color_dict[group]
    
    # Graficar la línea con el color especificado
    line, = plt.plot(df_eras[columna].cumsum(), label=columna, color=color)
    
    # Solo añadir un handle y label por grupo a la leyenda
    if group not in handles_labels:
        handles_labels[group] = line

# Crear la leyenda solo con los nombres de los grupos
plt.legend(handles_labels.values(), handles_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

# Añadir título y etiquetas de los ejes
plt.title("Comparacion modelos Propios", fontsize=25)
plt.xlabel("era")
plt.gca().xaxis.set_major_locator(MultipleLocator(10))
plt.ylabel("Cumulative Sum")

# Ajustar el layout
plt.subplots_adjust(right=0.75)

# Guardar el gráfico como una imagen
plt.savefig(os.path.join("resultados_modelos", f"Cumulative_Validation_MMC_comparacion_modelos_propios.png"))
plt.show()

# %%

# %%
sistema_experto.set_index("Modelo", inplace=True)

# %%
sistema_experto.loc["LGBM_10","Porcentaje"]

# %%
df_analisis["Sistema_experto"]=((df_analisis["LGBM_10"]*sistema_experto.loc["LGBM_10","Porcentaje"])+
                                (df_analisis["LGBM_11"]*sistema_experto.loc["LGBM_11","Porcentaje"])+
                                (df_analisis["LGBM_9"]*sistema_experto.loc["LGBM_9","Porcentaje"])+
                                (df_analisis["LGBM_13"]*sistema_experto.loc["LGBM_13","Porcentaje"])
                                )

# %%
df_analisis

# %%
df_eras = calculo_eras(df_analisis, "resultados_modelos", "Comparacion_modelos")

# %%
descripcion = df_eras.describe()
descripcion

# %%
df_analisis.to_parquet("Prediccions_modelos_propios.parquet")

# %%
