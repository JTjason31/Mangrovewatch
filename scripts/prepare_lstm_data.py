import pandas as pd
import numpy as np

df = pd.read_csv('data/matang_ndvi_hybrid_timeseries.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Encode month cyclically (so December and January are numerically "close")
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(f"Total months: {len(df)}")

WINDOW_SIZE = 6
FEATURES = ['ndvi_value', 'month_sin', 'month_cos']

data = df[FEATURES].values  # shape (n_months, 3)

X, y = [], []
for i in range(len(data) - WINDOW_SIZE):
    X.append(data[i:i+WINDOW_SIZE])       # past 6 months, all 3 features
    y.append(data[i+WINDOW_SIZE][0])      # predict just the NDVI value

X = np.array(X)
y = np.array(y)

print(f"\nCreated {len(X)} sequences of window size {WINDOW_SIZE}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

np.save('data/lstm_X.npy', X)
np.save('data/lstm_y.npy', y)
print("\nSaved X and y arrays to data/lstm_X.npy and data/lstm_y.npy")