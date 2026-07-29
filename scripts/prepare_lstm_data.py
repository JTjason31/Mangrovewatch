import pandas as pd
import numpy as np

df = pd.read_csv('data/matang_ndvi_timeseries.csv')

# Drop months with no data (image_count == 0)
df = df.dropna(subset=['mean_ndvi']).reset_index(drop=True)

# Sort chronologically just to be safe
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
df = df.sort_values('date').reset_index(drop=True)

print(f"Total usable months: {len(df)}")
print(df[['date', 'mean_ndvi']].head(10))

ndvi_values = df['mean_ndvi'].values

# Create sequences: use past 6 months to predict the next month
WINDOW_SIZE = 6

X, y = [], []
for i in range(len(ndvi_values) - WINDOW_SIZE):
    X.append(ndvi_values[i:i+WINDOW_SIZE])
    y.append(ndvi_values[i+WINDOW_SIZE])

X = np.array(X)
y = np.array(y)

print(f"\nCreated {len(X)} sequences of window size {WINDOW_SIZE}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

np.save('data/lstm_X.npy', X)
np.save('data/lstm_y.npy', y)
print("\nSaved X and y arrays to data/lstm_X.npy and data/lstm_y.npy")