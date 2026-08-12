import numpy as np
import pandas as pd
from tensorflow import keras

model = keras.models.load_model('models/lstm_ndvi_forecaster.keras')

df = pd.read_csv('data/matang_ndvi_hybrid_timeseries.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

WINDOW_SIZE = 6
FORECAST_HORIZON = 24

last_window = df[['ndvi_value', 'month_sin', 'month_cos']].values[-WINDOW_SIZE:].tolist()

last_date = df['date'].iloc[-1]
forecast_dates = pd.date_range(
    last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS'
)

forecasts = []
current_window = last_window.copy()

for step in range(FORECAST_HORIZON):
    input_seq = np.array(current_window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 3)
    next_ndvi = float(np.clip(model.predict(input_seq, verbose=0)[0][0], 0, 1))
    forecasts.append(next_ndvi)

    next_month = forecast_dates[step].month
    next_sin = np.sin(2 * np.pi * next_month / 12)
    next_cos = np.cos(2 * np.pi * next_month / 12)
    current_window.append([next_ndvi, next_sin, next_cos])

forecast_df = pd.DataFrame({
    'date': forecast_dates,
    'forecasted_ndvi': forecasts
})

forecast_df.to_csv('data/ndvi_24month_forecast.csv', index=False)

print("24-month NDVI forecast:")
print(forecast_df.to_string(index=False))
print("\nSaved to data/ndvi_24month_forecast.csv")