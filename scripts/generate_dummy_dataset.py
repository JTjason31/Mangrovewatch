import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 500

LON_MIN, LON_MAX = 100.52, 100.68
LAT_MIN, LAT_MAX = 4.70, 4.90

species_list = [
    'Rhizophora apiculata',
    'Rhizophora mucronata',
    'Avicennia marina',
    'Sonneratia alba',
    'Bruguiera gymnorhiza'
]

species_profiles = {
    'Rhizophora apiculata':   {'sal_opt': 15, 'sal_tol': 8,  'inund_opt': 55, 'inund_tol': 20},
    'Rhizophora mucronata':   {'sal_opt': 18, 'sal_tol': 8,  'inund_opt': 60, 'inund_tol': 20},
    'Avicennia marina':       {'sal_opt': 25, 'sal_tol': 10, 'inund_opt': 35, 'inund_tol': 20},
    'Sonneratia alba':        {'sal_opt': 20, 'sal_tol': 9,  'inund_opt': 65, 'inund_tol': 18},
    'Bruguiera gymnorhiza':   {'sal_opt': 12, 'sal_tol': 7,  'inund_opt': 45, 'inund_tol': 18},
}

# Each plot is assigned a site-quality tier FIRST (guarantees balanced, separable classes)
tiers = ['High', 'Moderate', 'Low']
tier_counts = [167, 167, 166]

# How far off-optimal (in units of species tolerance) each tier's conditions are
tier_mismatch = {
    'High':     {'sal_dev_factor': 0.2, 'inund_dev_factor': 0.2, 'wave_p': [0.80, 0.17, 0.03], 'sed_dev': 4,  'ndvi_mean': 0.48},
    'Moderate': {'sal_dev_factor': 1.1, 'inund_dev_factor': 1.1, 'wave_p': [0.35, 0.50, 0.15], 'sed_dev': 14, 'ndvi_mean': 0.30},
    'Low':      {'sal_dev_factor': 2.4, 'inund_dev_factor': 2.4, 'wave_p': [0.10, 0.30, 0.60], 'sed_dev': 25, 'ndvi_mean': 0.12},
}

wave_energy_categories = ['Low', 'Moderate', 'High']
wave_energy_penalty = {'Low': 0, 'Moderate': -8, 'High': -20}

rows = []
plot_num = 1

for tier, count in zip(tiers, tier_counts):
    tm = tier_mismatch[tier]
    for _ in range(count):
        plot_id = f"P{plot_num:04d}"
        plot_num += 1

        lat = np.random.uniform(LAT_MIN, LAT_MAX)
        lon = np.random.uniform(LON_MIN, LON_MAX)

        species = np.random.choice(species_list)
        profile = species_profiles[species]

        # Deviation scaled by tier (small deviation for High tier, large for Low tier)
        sal_dev = np.random.normal(0, profile['sal_tol'] * tm['sal_dev_factor'] * 0.6)
        inund_dev = np.random.normal(0, profile['inund_tol'] * tm['inund_dev_factor'] * 0.6)

        salinity = np.clip(profile['sal_opt'] + sal_dev, 5, 35)
        inundation = np.clip(profile['inund_opt'] + inund_dev, 10, 90)
        wave_energy = np.random.choice(wave_energy_categories, p=tm['wave_p'])
        sedimentation = np.clip(np.random.normal(20 + (tm['sed_dev'] if np.random.rand() > 0.5 else -tm['sed_dev']*0.3), 5), 5, 50)
        ndvi_baseline = np.clip(np.random.normal(tm['ndvi_mean'], 0.06), -0.1, 0.6)

        planting_date = datetime(2016, 1, 1) + timedelta(days=int(np.random.uniform(0, 2900)))

        sal_penalty = -((salinity - profile['sal_opt']) ** 2) / (2 * profile['sal_tol'] ** 2) * 40
        inund_penalty = -((inundation - profile['inund_opt']) ** 2) / (2 * profile['inund_tol'] ** 2) * 35
        wave_penalty = wave_energy_penalty[wave_energy]
        sediment_penalty = -abs(sedimentation - 20) * 0.4
        ndvi_bonus = ndvi_baseline * 25

        survival_rate = np.clip(
            80 + sal_penalty + inund_penalty + wave_penalty + sediment_penalty + ndvi_bonus
            + np.random.normal(0, 1),
            0, 100
        )

        rows.append({
            'plot_id': plot_id,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'species': species,
            'planting_date': planting_date.strftime('%Y-%m-%d'),
            'soil_salinity_ppt': round(salinity, 2),
            'tidal_inundation_freq_pct': round(inundation, 2),
            'wave_energy_exposure': wave_energy,
            'sedimentation_rate_mm_yr': round(sedimentation, 2),
            'ndvi_at_planting': round(ndvi_baseline, 3),
            'survival_rate_pct': round(survival_rate, 2),
            'survival_outcome': tier
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/matang_dummy_restoration_dataset.csv', index=False)

print(f"Generated {len(df)} synthetic restoration records.")
print(df['survival_outcome'].value_counts())
print("\nSaved to data/matang_dummy_restoration_dataset.csv")