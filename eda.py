# -*- coding: utf-8 -*-
# %%
from src.config import PAISES
import pandas as pd
import numpy as np
import statsmodels
from math import sqrt
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics  import mean_squared_error
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# %%
df_train = pd.read_parquet(
    f"signals/v1.0/train.parquet"
)

df_validation = pd.read_parquet(
    f"signals/v1.0/validation.parquet"
)

# %%
df_signal=pd.concat([df_train,df_validation])
df_signal['Country'] = df_signal['feature_country'].map(PAISES)
df_signal.columns

# %%
df_signal.head(5)

# %%
df_signal.info()

# %%
df_signal.describe()

# %%
columns_analisis=['date', 'Country','data_type','numerai_ticker','feature_impact_cost_factor', 'feature_beta_factor',
       'feature_volatility_factor', 'feature_momentum_12w_factor',
       'feature_momentum_26w_factor', 'feature_momentum_52w_factor',
       'feature_momentum_52w_less_4w_factor',
       'feature_trix_130d_country_ranknorm',
       'feature_trix_60d_country_ranknorm',
       'feature_rsi_130d_country_ranknorm', 'feature_rsi_60d_country_ranknorm',
       'feature_rsi_90d_country_ranknorm',
       'feature_ppo_60d_90d_country_ranknorm',
       'feature_ppo_60d_130d_country_ranknorm', 'feature_country',
       'feature_adv_20d_factor', 'feature_market_cap_factor',
       'feature_price_factor', 'feature_earnings_yield_factor',
       'feature_dividend_yield_factor', 'feature_book_to_price_factor',
       'feature_value_factor', 'feature_growth_factor', 'target_factor_neutral_20',]

# %%
df_analisis = df_signal[columns_analisis].copy()



# %%
df_analisis.groupby("data_type")["date"].min()

# %%
df_analisis.groupby("data_type")["date"].max()

# %%

country_counts = df_analisis['data_type'].value_counts()

plt.figure(figsize=(12, 8))
bars = plt.bar(country_counts.index, country_counts.values, color='skyblue')

# Función para formatear números con separadores de miles
def format_number(x, pos):
    return f'{int(x):,}'

formatter = FuncFormatter(format_number)
plt.gca().yaxis.set_major_formatter(formatter)

# Añadir etiquetas a cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, format_number(yval, None), ha='center', va='bottom', fontsize=20)

# Personalizar el gráfico
plt.title('Tamaños de datasets', fontsize=40)
plt.xlabel('Sets', fontsize=40)
plt.ylabel('Frecuencia', fontsize=40)
plt.xticks(rotation=90, fontsize=25)
plt.yticks(fontsize=25)
plt.grid(axis='y', linestyle='--', alpha=0.7)

output_path = 'graficos_eda/distribucion_datasets.jpg'

# Mostrar el gráfico
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

# %%
df_analisis['target_str'] = df_analisis['target_factor_neutral_20'].astype(str)
country_counts = df_analisis['target_str'].value_counts()

plt.figure(figsize=(20, 20))
bars = plt.bar(country_counts.index, country_counts.values, color='skyblue')

# Función para formatear números con separadores de miles
formatter = FuncFormatter(format_number)
plt.gca().yaxis.set_major_formatter(formatter)

# Añadir etiquetas a cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, format_number(yval, None), ha='center', va='bottom', rotation=0, fontsize=20)

# Personalizar el gráfico
plt.title('Distribucion Target', fontsize=80)
plt.xlabel('Target', fontsize=60)
plt.ylabel('Frecuencia', fontsize=60)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

output_path = 'graficos_eda/distribucion_target.jpg'

# Mostrar el gráfico
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

# %%
paises = df_analisis.groupby("Country")["numerai_ticker"].count().reset_index()
paises["% pais"]=paises["numerai_ticker"]/paises["numerai_ticker"].sum()*100
paises = paises.sort_values("numerai_ticker", ascending=False)
paises["% Acumulado"]=paises["% pais"].cumsum()

# %%
paises.head(9)

# %%
balanceo_datos = df_analisis.groupby(["data_type","target_factor_neutral_20"])["numerai_ticker"].count().reset_index()
balanceo_datos

# %%

country_counts = df_analisis['Country'].value_counts()


plt.figure(figsize=(12, 8))
bars = plt.bar(country_counts.index, country_counts.values, color='skyblue')

formatter = FuncFormatter(format_number)
plt.gca().yaxis.set_major_formatter(formatter)

# Añadir etiquetas a cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width(), yval, format_number(yval, None), ha='center', va='bottom',rotation=80, fontsize=14 )

# Personalizar el gráfico
plt.title('Frecuencia Paises', fontsize=40)
plt.xlabel('País', fontsize=30)
plt.ylabel('Frecuencia', fontsize=30)
plt.xticks(rotation=90, fontsize=17)
plt.yticks( fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)

output_path = 'graficos_eda/frecuencia_paises.jpg'

# Mostrar el gráfico
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

# %%

corr_matrix = df_analisis.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', linewidths=0.5, annot_kws={"size": 6} )
plt.title('Mapa de Calor de Correlaciones')
output_path = 'graficos_eda/correlaciones.jpg'

# Mostrar el gráfico
plt.tight_layout()
plt.savefig(output_path)
plt.show()

# %%

# %%
empresa="AAPL"


# %%
def adfuller_test(riders):
    result = adfuller(riders)
    labels = ['ADF Test Statistic', 'p-value','#Lags used', 'Number of Observations used']
    for value,label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print('strong evidence against the null hypothesis(Ho), reject the null hypothesis, data is stationary')
    else:
        print('weak evidence against the null hypothesis(Ho), data is not stationary')


# %%
def generar_eda_empresa(df_analisis, empresa):
    df_filtrado = df_analisis[df_analisis["numerai_ticker"].str.contains(empresa, na=False)]
    fecha_inicio = df_filtrado["date"].min()  # Fecha de inicio
    fecha_fin = df_filtrado["date"].max()     # Fecha de fin
    frecuencia = 'W-FRI' 

    fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=frecuencia)

    # Crear un DataFrame con la columna de fechas
    df = pd.DataFrame(fechas, columns=['date'])

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')


    #df['date'] = pd.to_datetime(df["date"])
    df_filtrado['date'] = pd.to_datetime(df_filtrado["date"])
    # df_filtrado['date'] = df_filtrado['date'].dt.strftime('%Y-%m-%d')

    # df_filtrado

    # df = df.merge(df_filtrado, on="date", how="left")
    # df['target_factor_neutral_20'].interpolate(method='linear')
    df_filtrado.set_index('date', inplace=True)

    #df_filtrado_describe =df_filtrado.describe()
    #df_filtrado_describe.to_excel(f"Descriptivo_{empresa}.xlsx")

    df_filtrado["target_movil"] = df_filtrado["target_factor_neutral_20"].rolling(5).mean()
    df_filtrado = df_filtrado.dropna(subset="target_movil")
    df_filtrado=df_filtrado[["target_factor_neutral_20","target_movil"]]
    fig, ax = plt.subplots(figsize=(12, 6))  # Crea la figura y el eje

    # Graficar los datos
    df_filtrado.plot(ax=ax)

    # Añadir un título al gráfico
    fig.suptitle(f'Serie Temporal de {empresa}', fontsize=16)

    # Ajustar el tamaño del gráfico si es necesario (esto se hace en la creación del gráfico)
    # fig.set_size_inches(10, 8)  # Opcional: solo si necesitas cambiar el tamaño después de la creación

    # Guardar el gráfico como imagen
    fig.savefig(f'graficos_eda/serie_temporal_{empresa}.png')  # Cambia el nombre del archivo y formato si es necesario

    # Mostrar el gráfico
    plt.show()

    prueba = df_filtrado['target_movil']

    results = seasonal_decompose(prueba, model='additive', extrapolate_trend='freq', period=64)
    fig = results.plot()

    fig.suptitle(f'Descomposición Estacional de la Serie Temporal de {empresa}', fontsize=16)

    # Ajustar el tamaño del gráfico si es necesario
    fig.set_size_inches(10, 8)  # Puedes ajustar el tamaño según tus necesidades

    # Guardar el gráfico como imagen
    fig.savefig(f'graficos_eda/descomposicion_estacional_{empresa}.png')  # Cambia el nombre del archivo y formato si es necesario

    # Mostrar el gráfico
    plt.show()

    test_adfuller = adfuller_test(prueba)

    print("----")
    print(test_adfuller)
    print("-----")

    # Crear una figura con dos subgráficos
    fig = plt.figure(figsize=(12, 8))

    # Primer subgráfico: ACF
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(prueba, lags=40, ax=ax1)
    ax1.set_title(f'Función de Autocorrelación (ACF) {empresa}', fontsize=14)

    # Segundo subgráfico: PACF
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(prueba, lags=40, ax=ax2)
    ax2.set_title(f'Función de Autocorrelación Parcial (PACF) {empresa}', fontsize=14)

    # Ajustar el espacio entre subgráficos
    plt.tight_layout()

    # Guardar la figura como imagen
    fig.savefig(f'graficos_eda/acf_pacf_{empresa}.png')  # Cambia el nombre del archivo y formato si es necesario

    # # Mostrar el gráfico
    # plt.show()
    # df_filtrado.plot.bar(
    # title="Análisis Variables en el tiempo",
    # figsize=(16, 10),
    # layout=(7,5),
    # xticks=[],
    # subplots=True,
    # sharex=False,
    # legend=False,
    # snap=False
    # )
    # for ax in plt.gcf().axes:
    #     ax.set_xlabel("")
    #     ax.title.set_fontsize(11)
    # plt.tight_layout(pad=1.5)
    # plt.gcf().suptitle("Análisis Variables en el tiempo", fontsize=25)
    # plt.savefig(f'graficos_eda/descriptivo_variables_{empresa}.png')


# %%
df_filtrado = df_analisis[df_analisis["numerai_ticker"].str.contains(empresa, na=False)]
df_filtrado.items()

# %%
df_filtrado.reset_index(inplace=True)

# %%
df_filtrado

# %%
lista_top_five_sp500 = ["GOOGL"]
["NVDA", "AAPL","MSFT","GOOGL","AMZN","TSLA"]


for empresa in lista_top_five_sp500:
    generar_eda_empresa(df_analisis, empresa)


 # %%
 adfuller_test(prueba)

# %%
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(prueba, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(prueba, lags=40, ax=ax2)

# %%
stepwise_fit = auto_arima(prueba, trace=True)

# %%
stepwise_fit.summary()

# %%
