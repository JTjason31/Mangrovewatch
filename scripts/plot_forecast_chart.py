import pandas as pd
import matplotlib.pyplot as plt

# Load historical (hybrid) series for context
hist_df = pd.read_csv('data/matang_ndvi_hybrid_timeseries.csv')
hist_df['date'] = pd.to_datetime(hist_df['date'])
hist_df = hist_df.sort_values('date')

# Load the 24-month forecast
forecast_df = pd.read_csv('data/ndvi_24month_forecast.csv')
forecast_df['date'] = pd.to_datetime(forecast_df['date'])
forecast_df = forecast_df.sort_values('date')

# Only show the last 24 months of history for context, so the chart isn't too cluttered
hist_recent = hist_df.tail(24)

plt.figure(figsize=(12, 5))
plt.plot(hist_recent['date'], hist_recent['ndvi_value'], marker='o', markersize=3,
         linewidth=1.5, color='tab:blue', label='Historical NDVI (2023-2024)')
plt.plot(forecast_df['date'], forecast_df['forecasted_ndvi'], marker='o', markersize=3,
         linewidth=1.5, color='tab:orange', linestyle='--', label='LSTM 24-Month Forecast (2025-2026)')

plt.axvline(x=hist_recent['date'].iloc[-1], color='gray', linestyle=':', linewidth=1)

plt.title('24-Month NDVI Forecast - Matang Mangrove Forest Reserve')
plt.xlabel('Date')
plt.ylabel('NDVI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/ndvi_24month_forecast_chart.png', dpi=200)
print("Chart saved to data/ndvi_24month_forecast_chart.png")