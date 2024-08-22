# %%
from numerapi import NumerAPI
napi = NumerAPI()

[f for f in napi.list_datasets() if f.startswith("signals/v1.0")] 

signals = ['signals/v1.0/live.parquet',
 'signals/v1.0/live_example_preds.csv',
 'signals/v1.0/live_example_preds.parquet',
 'signals/v1.0/train.parquet',
 'signals/v1.0/validation.parquet',
 'signals/v1.0/validation_example_preds.csv',
 'signals/v1.0/validation_example_preds.parquet']
 
# Download the training data 

for signal in signals:
    napi.download_dataset(signal)

# %%
import pandas as pd 
df = pd.read_parquet(
    f"signals/v1.0/train.parquet"
)

# %%
df.groupby("numerai_ticker")["target_irina_60"].count()

# %%
for column in df.columns:
    print(column)

# %%
