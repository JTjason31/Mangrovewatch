import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv('data/matang_ndvi_timeseries.csv')
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

real_df = df.dropna(subset=['mean_ndvi']).copy()

# Learn seasonal pattern: average NDVI per calendar month across all real observations
seasonal_avg = real_df.groupby('month')['mean_ndvi'].mean()
overall_mean = real_df['mean_ndvi'].mean()
overall_std = real_df['mean_ndvi'].std()

print("Seasonal averages by month (learned from real data):")
print(seasonal_avg)
print(f"\nOverall mean: {overall_mean:.4f}, std: {overall_std:.4f}")

# Build full 2016-01 to 2024-12 monthly range
full_dates = pd.date_range('2016-01-01', '2024-12-01', freq='MS')
full_df = pd.DataFrame({'date': full_dates})
full_df['year'] = full_df['date'].dt.year
full_df['month'] = full_df['date'].dt.month

# Merge in real values where they exist
full_df = full_df.merge(
    real_df[['year', 'month', 'mean_ndvi']],
    on=['year', 'month'], how='left'
)
full_df.rename(columns={'mean_ndvi': 'ndvi_real'}, inplace=True)

# Fill missing months using seasonal pattern + small realistic noise + slight long-term trend
synthetic_values = []
is_synthetic = []

for idx, row in full_df.iterrows():
    if pd.notna(row['ndvi_real']):
        synthetic_values.append(row['ndvi_real'])
        is_synthetic.append(False)
    else:
        seasonal_base = seasonal_avg.get(row['month'], overall_mean)
        # small noise consistent with observed variability
        noise = np.random.normal(0, overall_std * 0.4)
        value = np.clip(seasonal_base + noise, 0, 1)
        synthetic_values.append(value)
        is_synthetic.append(True)

full_df['ndvi_value'] = synthetic_values
full_df['is_synthetic'] = is_synthetic

full_df.to_csv('data/matang_ndvi_hybrid_timeseries.csv', index=False)

print(f"\nTotal months: {len(full_df)}")
print(f"Real months: {(~full_df['is_synthetic']).sum()}")
print(f"Synthetic (gap-filled) months: {full_df['is_synthetic'].sum()}")
print("\nSaved to data/matang_ndvi_hybrid_timeseries.csv")