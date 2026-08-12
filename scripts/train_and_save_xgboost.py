import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

df = pd.read_csv('data/matang_dummy_restoration_dataset.csv')

le_species = LabelEncoder()
df['species_encoded'] = le_species.fit_transform(df['species'])

le_wave = LabelEncoder()
df['wave_energy_encoded'] = le_wave.fit_transform(df['wave_energy_exposure'])

le_outcome = LabelEncoder()
df['outcome_encoded'] = le_outcome.fit_transform(df['survival_outcome'])

feature_cols = [
    'species_encoded', 'soil_salinity_ppt', 'tidal_inundation_freq_pct',
    'wave_energy_encoded', 'sedimentation_rate_mm_yr', 'ndvi_at_planting'
]

X = df[feature_cols]
y = df['outcome_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# Save the model and the label encoders (needed to decode predictions later)
joblib.dump(model, 'models/xgboost_classifier.joblib')
joblib.dump(le_species, 'models/le_species.joblib')
joblib.dump(le_wave, 'models/le_wave.joblib')
joblib.dump(le_outcome, 'models/le_outcome.joblib')

print("XGBoost model and encoders saved to models/")
print("Species classes:", list(le_species.classes_))
print("Wave energy classes:", list(le_wave.classes_))
print("Outcome classes:", list(le_outcome.classes_))