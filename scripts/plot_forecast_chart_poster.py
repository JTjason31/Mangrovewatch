import pandas as pd
import matplotlib.pyplot as plt

OCHRE = "#B98B4E"
LINE = "#E1D9C6"

df = pd.read_csv('data/ndvi_24month_forecast.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=200)
ax.plot(range(len(df)), df['forecasted_ndvi'], color=OCHRE, linewidth=2.5,
        marker='o', markersize=4, markerfacecolor=OCHRE)
ax.fill_between(range(len(df)), df['forecasted_ndvi'], 0.25, color=OCHRE, alpha=0.10)
ax.set_ylim(0.25, 0.55)
ax.set_ylabel('Forecasted NDVI', fontsize=11)
ax.set_title('24-Month LSTM NDVI Forecast (2025\u20132026)', fontsize=12, fontweight='bold', color=OCHRE, pad=10)

tick_idx = list(range(0, len(df), 3))
ax.set_xticks(tick_idx)
ax.set_xticklabels([df['date'].iloc[i].strftime('%Y-%m') for i in tick_idx], fontsize=8.5)
ax.grid(axis='y', color=LINE, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('data/lstm_forecast_chart.png', dpi=200, transparent=True)
print("Saved to data/lstm_forecast_chart.png")