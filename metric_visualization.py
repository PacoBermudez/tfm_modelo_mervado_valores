from numerai_tools.scoring import numerai_corr, correlation_contribution
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.config import load_model

def extract_metrics(validation,df, feature_set,model, nombre_modelo, directorio, parametros, df_estadisticas):

    output_folder = directorio
    os.makedirs(output_folder, exist_ok=True)


    validation["prediction"] = model.predict(validation[feature_set])
    validation[["era", "prediction", "target"]]

    y_test = validation["target"]
    y_pred = validation["prediction"]

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(nombre_modelo)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')

    validation["meta_model"] = pd.read_parquet(
    f"v4.3/meta_model.parquet"
)["numerai_meta_model"]


    per_era_corr = validation.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
    )

    # Compute the per-era mmc between our predictions, the meta model, and the target values
    per_era_mmc = validation.dropna().groupby("era").apply(
        lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
    )


    # Plot the per-era correlation
    plt.figure(figsize=(8, 4))
    per_era_corr.plot(
        title=f"Validation CORR {nombre_modelo}",
        kind="bar",
        xticks=[],
        legend=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Validation_CORR_{nombre_modelo}.png"))

    # Plot the per-era MMC
    plt.figure(figsize=(8, 4))
    per_era_mmc.plot(
        title=f"Validation MMC {nombre_modelo}",
        kind="bar",
        xticks=[],
        legend=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Validation_MMC_{nombre_modelo}.png"))

    # Plot the cumulative validation CORR
    plt.figure(figsize=(8, 4))
    per_era_corr.cumsum().plot(
        title=f"Cumulative Validation CORR {nombre_modelo}",
        kind="line",
        legend=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Cumulative_Validation_CORR_{nombre_modelo}.png"))

    # Plot the cumulative validation MMC
    plt.figure(figsize=(8, 4))
    per_era_mmc.cumsum().plot(
        title=f"Cumulative Validation MMC {nombre_modelo}",
        kind="line",
        legend=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Cumulative_Validation_MMC_{nombre_modelo}.png"))

    # Mostrar los gráficos
    plt.show()
    mean_corr_numerai = per_era_corr.mean()

    mean_mmc_numerai = per_era_mmc.mean()
    

    print(f'Mean Corr Numerai: {mean_corr_numerai}')

    df["Target"] = y_test

    df[nombre_modelo] = y_pred


    df_estadisticas = df_estadisticas.append(
    {'Modelo' : nombre_modelo , 'Parámetros' : parametros,'MAE': mae, 'MSE':mse, "Numerai Coor":mean_corr_numerai[0], 'MMC':mean_mmc_numerai[0]},
    ignore_index=True
    )

    return df, df_estadisticas
    # y_pred.plot(
    # title=f"Cumulative Validation MMC {nombre_modelo}",
    # kind="line",
    # figsize=(8, 4),
    # legend=False
    # )

def extract_metrics_models(validation,df, feature_set, nombre_modelo, directorio, parametros, df_estadisticas):
    output_folder = directorio
    os.makedirs(output_folder, exist_ok=True)

    model =load_model(nombre_modelo)

    validation["prediction"] = model.predict(validation[feature_set])
    validation[["era", "prediction", "target"]]

    y_test = validation["target"]
    y_pred = validation["prediction"]

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(nombre_modelo)

    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')

    validation["meta_model"] = pd.read_parquet(
    f"v4.3/meta_model.parquet"
)["numerai_meta_model"]


    per_era_corr = validation.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
    )

    # Compute the per-era mmc between our predictions, the meta model, and the target values
    per_era_mmc = validation.dropna().groupby("era").apply(
        lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
    )

    mean_corr_numerai = per_era_corr.mean()

    mean_mmc_numerai = per_era_mmc.mean()

    print(f'Mean Corr Numerai: {mean_corr_numerai}')

    df["Target"] = y_test

    df[nombre_modelo] = y_pred


    df_estadisticas = df_estadisticas.append(
    {'Modelo' : nombre_modelo , 'Parámetros' : parametros,'MAE': mae, 'MSE':mse, "Numerai Coor":mean_corr_numerai[0], 'MMC':mean_mmc_numerai[0]},
    ignore_index=True
    )


    return df, df_estadisticas


def visualizar_modelo(validation, df):

    df["era"] = validation["era"]

    df_eras = pd.DataFrame()

    for columna in df.columns:
        per_era_corr = df.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]].dropna(), x["Target"].dropna())
    )
        
    df_eras[columna] = per_era_corr[columna]





def calculo_eras(validation, df, output_folder, nombre_modelo):

    df["era"] = validation["era"]
    df_eras = pd.DataFrame()

    for columna in df.columns:
        per_era_corr = df.groupby("era").apply(
        lambda x: numerai_corr(x[[columna]].dropna(), x["Target"].dropna())
    )
        
        df_eras[columna] = per_era_corr[columna]

    df_eras = df_eras.drop("Target", axis=1)

    df_eras.cumsum().plot(
            title=f"Cumulative Validation CORR Numerai",
            kind="line",
            legend=True
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"Cumulative_Validation_MMC_{nombre_modelo}.png"))

    return df_eras






