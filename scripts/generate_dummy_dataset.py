import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# Load real, vegetation-validated plot locations
locations_df = pd.read_csv('data/valid_mangrove_plot_locations.csv')
locations_df = locations_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

N = len(locations_df)

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

tiers = ['High', 'Moderate', 'Low']
tier_counts = [N // 3 + (1 if N % 3 > 0 else 0), N // 3 + (1 if N % 3 > 1 else 0), N // 3]

tier_mismatch = {
    'High':     {'dev_range': (0.0, 0.7),  'wave_p': [0.80, 0.17, 0.03], 'sed_range': (0, 10),  'ndvi_bonus_mult': 1.25},
    'Moderate': {'dev_range': (0.7, 1.5),  'wave_p': [0.30, 0.55, 0.15], 'sed_range': (8, 20),  'ndvi_bonus_mult': 1.0},
    'Low':      {'dev_range': (1.5, 3.0),  'wave_p': [0.05, 0.25, 0.70], 'sed_range': (18, 32), 'ndvi_bonus_mult': 0.75},
}

wave_energy_categories = ['Low', 'Moderate', 'High']
wave_energy_penalty = {'Low': 0, 'Moderate': -8, 'High': -20}

rows = []
plot_num = 1
loc_idx = 0

for tier, count in zip(tiers, tier_counts):
    tm = tier_mismatch[tier]
    for _ in range(count):
        if loc_idx >= len(locations_df):
            break

        plot_id = f"P{plot_num:04d}"
        plot_num += 1

        loc_row = locations_df.iloc[loc_idx]
        lon = loc_row['longitude']
        lat = loc_row['latitude']
        ndvi_real = loc_row['ndvi_real_satellite']
        loc_idx += 1

        species = np.random.choice(species_list)
        profile = species_profiles[species]

        # Draw a magnitude from the tier's non-overlapping range, apply random sign
        dev_magnitude = np.random.uniform(tm['dev_range'][0], tm['dev_range'][1])
        sal_dev = profile['sal_tol'] * dev_magnitude * np.random.choice([-1, 1])
        inund_dev = profile['inund_tol'] * dev_magnitude * np.random.choice([-1, 1])

        salinity = np.clip(profile['sal_opt'] + sal_dev, 5, 35)
        inundation = np.clip(profile['inund_opt'] + inund_dev, 10, 90)
        wave_energy = np.random.choice(wave_energy_categories, p=tm['wave_p'])
        sedimentation = np.clip(np.random.uniform(tm['sed_range'][0], tm['sed_range'][1]) + 20, 5, 50)

        planting_date = datetime(2016, 1, 1) + timedelta(days=int(np.random.uniform(0, 2900)))

        sal_penalty = -((salinity - profile['sal_opt']) ** 2) / (2 * profile['sal_tol'] ** 2) * 40
        inund_penalty = -((inundation - profile['inund_opt']) ** 2) / (2 * profile['inund_tol'] ** 2) * 35
        wave_penalty = wave_energy_penalty[wave_energy]
        sediment_penalty = -abs(sedimentation - 20) * 0.4
        ndvi_bonus = ndvi_real * 25 * tm['ndvi_bonus_mult']

        survival_rate = np.clip(
            80 + (sal_penalty * 1.4) + (inund_penalty * 1.4) + (wave_penalty * 1.5)
            + (sediment_penalty * 1.2) + ndvi_bonus
            + np.random.normal(0, 2.5),
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
            'ndvi_at_planting': round(ndvi_real, 3),
            'survival_rate_pct': round(survival_rate, 2),
            'survival_outcome': tier
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/matang_dummy_restoration_dataset.csv', index=False)

print(f"Generated {len(df)} synthetic restoration records using real mangrove-validated NDVI.")
print(df['survival_outcome'].value_counts())
print("\nSaved to data/matang_dummy_restoration_dataset.csv")