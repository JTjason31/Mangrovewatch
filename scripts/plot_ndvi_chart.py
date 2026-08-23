import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/matang_ndvi_timeseries.csv')
df = df.dropna(subset=['mean_ndvi'])
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
df = df.sort_values('date')

plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['mean_ndvi'], marker='o', markersize=3, linewidth=1)
plt.title('Monthly Mean NDVI - Matang Mangrove Forest Reserve (2016-2024)')
plt.xlabel('Date')
plt.ylabel('Mean NDVI')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/ndvi_timeseries_chart.png', dpi=200)
print("Chart saved to data/ndvi_timeseries_chart.png")