# %%
import numerapi
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance


import simplejson

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests as re
import datetime

load_dotenv() 

NAPI_PUBLIC_KEY = os.environ["PUBLIC_ID"]
NAPI_PRIVATE_KEY = os.environ["SECRET_ID"]
napi = numerapi.SignalsAPI(NAPI_PUBLIC_KEY, NAPI_PRIVATE_KEY)

# %%
df_train = pd.read_parquet(
    f"signals/v1.0/train.parquet"
)

df_validation = pd.read_parquet(
    f"signals/v1.0/validation.parquet"
)

ticket = list(set(df_train["numerai_ticker"]))

# %%
df_final=pd.concat([df_train,df_validation])
df_final


# %%
def plot_ticker_counts_per_date(df, title, color_column):
    df['date'] = pd.to_datetime(df['date'])
    
    # Count unique 'numerai_ticker' per 'date'
    nticker_count_per_date = df.groupby('date')['numerai_ticker'].nunique().reset_index(name='numerai_ticker_count')

    # Check if color_column exists in the DataFrame
    if color_column not in df.columns:
        raise ValueError(f"Column '{color_column}' not found in the DataFrame")

    # Map categorical values to colors
    unique_categories = df[color_column].unique()
    color_map = {cat: plt.cm.tab10(i / len(unique_categories)) for i, cat in enumerate(unique_categories)}
    
    # Add colors to the DataFrame
    df['color'] = df[color_column].map(color_map)
    
    # Merge colors into the counts DataFrame
    color_map_per_date = df.drop_duplicates(subset=['date', color_column]).set_index('date')['color'].to_dict()
    nticker_count_per_date['color'] = nticker_count_per_date['date'].map(color_map_per_date)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(nticker_count_per_date['date'], nticker_count_per_date['numerai_ticker_count'], 
                c=nticker_count_per_date['color'], edgecolor='k', marker='o')

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat, 
                          markersize=10, markerfacecolor=color_map[cat]) 
               for cat in unique_categories]
    plt.legend(handles=handles, title=color_column)
    
    plt.tight_layout()
    plt.show()




# %%
plot_ticker_counts_per_date(df_final, 'Ticker Counts per Date', 'data_type')

# %%
#eligible_tickers = pd.Series(napi.ticker_universe(), name='ticker') 

eligible_tickers = pd.Series(list(set(df_final["numerai_ticker"])), name='ticker') 
print(f"Number of eligible tickers: {len(eligible_tickers)}")



# %%
list_us=[x for x in eligible_tickers if "GOOGL" in x]
list_us




# %%
def extraer_ticket_yahoo(x):
    x=x.split(" ")[0]
    return x


# %%
df_final["prueba"]=df_final["numerai_ticker"].apply(extraer_ticket_yahoo)

# %%
eligible_tickers = pd.Series(list(set(df_final["prueba"])), name='ticker') 
print(f"Number of eligible tickers: {len(eligible_tickers)}")

# %%
n = 600  #chunk row size
chunk_df = [yfinance_tickers.iloc[i:i+n] for i in range(0, len(yfinance_tickers), n)]

# %%
lista_top_five_sp500 = ["AAPL","MSFT","GOOGL","AMZN","TSLA"]

# %%
full_data = pd.concat(concat_dfs)
full_data

# %%
full_data.to_parquet('yahoo/datos_yahoo.parquet')  

# %%
full_data.columns = ['date', 'ticker', 'price']
full_data.set_index('date', inplace=True)

# convert yahoo finance tickers back to numerai tickers
full_data['ticker'] = full_data.ticker.map(dict(zip(yfinance_tickers, numerai_tickers)))

print(f"Number of tickers with data: {len(full_data.ticker.unique())}")
full_data.head()


# %%
def RSI(prices, interval=14):
  '''Computes Relative Strength Index given a price series and lookback interval
  Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
  See more here https://www.investopedia.com/terms/r/rsi.asp'''
  delta = prices.diff()

  dUp, dDown = delta.copy(), delta.copy()
  dUp[dUp < 0] = 0
  dDown[dDown > 0] = 0

  RolUp = dUp.rolling(interval).mean()
  RolDown = dDown.rolling(interval).mean().abs()

  RS = RolUp / RolDown
  RSI = 100.0 - (100.0 / (1.0 + RS))
  return RSI


# %%
ticker_groups = full_data.groupby('ticker')
full_data['RSI'] = ticker_groups['price'].transform(lambda x: RSI(x))

# %%
date_groups = full_data.groupby(full_data.index)
date_groups['RSI']

# %%
for date, group in date_groups:
    print(f"Date: {date}")
    print(group['RSI'])
    print("------")

# %%
date_groups = full_data.groupby(full_data.index)
full_data['RSI_quintile'] = date_groups['RSI'].transform(lambda group: pd.qcut(group, 5, labels=False, duplicates='drop'))
full_data.dropna(inplace=True)

full_data.head()

# %%
ticker_groups = full_data.groupby('ticker')

#create lagged features, lag 0 is that day's value, lag 1 is yesterday's value, etc
num_days = 5
for day in range(num_days+1):
    full_data[f'RSI_quintile_lag_{day}'] = ticker_groups['RSI_quintile'].transform(lambda group: group.shift(day))

# %%
full_data.tail()

# %%
for day in range(num_days):
    full_data[f'RSI_diff_{day}'] = full_data[f'RSI_quintile_lag_{day}'] - full_data[f'RSI_quintile_lag_{day + 1}']
    full_data[f'RSI_abs_diff_{day}'] = np.abs(full_data[f'RSI_quintile_lag_{day}'] - full_data[f'RSI_quintile_lag_{day + 1}'])

# %%
full_data.tail()

# %%
feature_names = [f'RSI_quintile_lag_{num}' for num in range(num_days)] + [f'RSI_diff_{num}' for num in range(num_days)] + [f'RSI_abs_diff_{num}' for num in range(num_days)]
print(f'Features for training:\n {feature_names}')

# %%
df_train = df_train.rename(columns={'numerai_ticker': 'ticker'})

df_train["date"] = pd.to_datetime(df_train["date"])

# %%
ML_data = pd.merge(df_train, full_data.reset_index(), on=['date','ticker'], how="left").set_index('date')

# %%
filtrado=ML_data[(ML_data["price"].isna()) & (ML_data["ticker"]=="MSFT US")]

# %%
from scipy.stats.mstats import winsorize


# %%
def winsorized_mean(series, limits=(0.05, 0.05)):
    if series.dropna().empty:
        return float('nan')  # O un valor predeterminado apropiado
    winsorized_series = winsorize(series.dropna(), limits=limits)
    return pd.Series(winsorized_series).mean()

# Aplicar winsorization para cada ticker
def calculate_winsorized_means(group):
    return winsorized_mean(group['price'], limits=(0.05, 0.05))


# %%
winsorized_means = ML_data.groupby('ticker').apply(calculate_winsorized_means)

# %%
winsorized_means.reset_index()

# %%
