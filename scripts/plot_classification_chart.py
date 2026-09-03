import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb

TEAL = "#0E4F52"
MANGROVE = "#47795A"
OCHRE = "#B98B4E"
LINE = "#E1D9C6"

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
    n_estimators=400, max_depth=6, learning_rate=0.05,
    random_state=42, eval_metric='mlogloss'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, target_names=le_outcome.classes_, output_dict=True)

classes = list(le_outcome.classes_)
precision = [report[c]['precision'] for c in classes]
recall = [report[c]['recall'] for c in classes]
f1 = [report[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25
fig, ax = plt.subplots(figsize=(6.2, 3.7), dpi=200)
ax.bar(x - width, precision, width, label='Precision', color=TEAL)
ax.bar(x, recall, width, label='Recall', color=MANGROVE)
ax.bar(x + width, f1, width, label='F1-score', color=OCHRE)
ax.set_ylim(0, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_title('XGBoost Classifier Performance by Class', fontsize=12, fontweight='bold', color=TEAL, pad=42)
ax.legend(loc='lower center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.02))
ax.grid(axis='y', color=LINE, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)

for i, v in enumerate(precision):
    ax.text(i - width, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
for i, v in enumerate(recall):
    ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
for i, v in enumerate(f1):
    ax.text(i + width, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('data/xgboost_classification_chart.png', dpi=200, transparent=True)
print("Saved to data/xgboost_classification_chart.png")
print(f"Accuracy: {(y_pred == y_test).mean():.4f}")