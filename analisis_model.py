# -*- coding: utf-8 -*-
# %%
from src.config import load_features, load_balanced_data, load_validation_data, load_train_data, load_validation_benchmarck, load_validation_examples, load_predicciones
import metric_visualization as metric
import pandas as pd
import matplotlib.pyplot as plt
import time
from numerai_tools.scoring import numerai_corr, correlation_contribution

# %%
val_benchmark = load_validation_benchmarck()

val_examples =load_validation_examples()

pred_lightgbm = load_predicciones("predicciones_lightgmb.parquet")



pred_lightgbm["meta_model"] = pd.read_parquet(
    f"v4.3/meta_model.parquet"
)["numerai_meta_model"]






# %%
pred_lightgbm = pred_lightgbm.reset_index()

val_benchmark = val_benchmark.reset_index()

# %%
pred_lightgbm.columns

# %%
pred_lightgbm = pred_lightgbm.merge(val_benchmark, on="id")

# %%
pred_lightgbm =pred_lightgbm[['Target','LGBM_11', 'era_x', 'LGBM_10','meta_model',
    'v43_lgbm_teager60', 'v43_lgbm_teager20', 'v43_lgbm_cyrus60',
       'v43_lgbm_cyrus20', 'v42_example_preds', 'v42_lgbm_teager60',
       'v42_lgbm_teager20', 'v42_lgbm_agnes20', 'v42_lgbm_claudia20',
       'v42_lgbm_rowan20', 'v41_lgbm_xerxes60', 'v41_lgbm_xerxes20',
       'v41_lgbm_sam60', 'v41_lgbm_sam20', 'v41_lgbm_cyrus60',
       'v41_lgbm_cyrus20', 'v41_lgbm_caroline60', 'v41_lgbm_caroline20',
       'v4_lgbm_waldo60', 'v4_lgbm_waldo20', 'v4_lgbm_victor60',
       'v4_lgbm_victor20', 'v4_lgbm_tyler60', 'v4_lgbm_tyler20',
       'v4_lgbm_ralph60', 'v4_lgbm_ralph20', 'v4_lgbm_nomi60',
       'v4_lgbm_nomi20', 'v4_lgbm_jerome60', 'v4_lgbm_jerome20',
       'v3_example_preds', 'v2_example_preds', 'v41_example_preds',
       'v42_rain_ensemble', 'v42_rain_ensemble2', 'v42_teager_plus_cyrus',
       'v42_teager_ensemble', 'v42_lgbm_ct_blend', 'v43_lgbm_ct_blend']]

# %%
import os
df_eras = pd.DataFrame()

for columna in pred_lightgbm.columns:
    per_era_corr = pred_lightgbm.groupby("era_x").apply(
    lambda x: numerai_corr(x[[columna]].dropna(), x["Target"].dropna())
)
    
    df_eras[columna] = per_era_corr[columna]

df_eras = df_eras.drop("Target", axis=1)

df_eras.cumsum().plot(
        title=f"Cumulative Validation CORR Numerai",
        kind="line",
        legend=True
    )

output_folder = "resultados_modelos"
nombre_modelo = "Light"
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"Cumulative_Validation_MMC_{nombre_modelo}.png"))

# %%
import plotly.express as px
import plotly.graph_objects as go
import os


# %%
fig = px.line(df_eras.cumsum(), 
              title="Cumulative Validation CORR Numerai")

# Añadir etiquetas y leyenda
fig.update_layout(
    xaxis_title="Eras",
    yaxis_title="Cumulative CORR",
    legend_title="Features"
)

# Mostrar el gráfico
fig.show()

# Guardar la figura como imagen
output_folder = "resultados_modelos"
nombre_modelo = "Light"

# Asegúrate de que la carpeta de salida existe
os.makedirs(output_folder, exist_ok=True)

# Guardar la figura
fig.to_html(os.path.join(output_folder, f"Cumulative_Validation_MMC_{nombre_modelo}.html"))

# %%
pip install -U kaleido

# %%
