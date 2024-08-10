from numerai_tools.scoring import numerai_corr, correlation_contribution
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def extract_metrics(validation, model, feature_set, nombre_modelo):
    validation["prediction"] = model.predict(validation[feature_set])
    validation[["era", "prediction", "target"]]

    y_test = validation["target"]
    y_pred = validation["prediction"]

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

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
    per_era_corr.plot(
    title=f"Validation CORR {nombre_modelo}",
    kind="bar",
    figsize=(8, 4),
    xticks=[],
    legend=False,
    snap=False
    )
    per_era_mmc.plot(
    title=f"Validation MMC {nombre_modelo}",
    kind="bar",
    figsize=(8, 4),
    xticks=[],
    legend=False,
    snap=False
    )
    per_era_corr.cumsum().plot(
    title=f"Cumulative Validation CORR {nombre_modelo}",
    kind="line",
    figsize=(8, 4),
    legend=False
    )
    per_era_mmc.cumsum().plot(
    title=f"Cumulative Validation MMC {nombre_modelo}",
    kind="line",
    figsize=(8, 4),
    legend=False
    )

