import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import xgboost as xgb

df = pd.read_csv('data/matang_dummy_restoration_dataset.csv')

# Encode categorical features
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

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=le_outcome.classes_))