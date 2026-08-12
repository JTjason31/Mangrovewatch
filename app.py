from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

app = Flask(__name__)

# Load models and encoders once at startup
xgb_model = joblib.load('models/xgboost_classifier.joblib')
le_species = joblib.load('models/le_species.joblib')
le_wave = joblib.load('models/le_wave.joblib')
le_outcome = joblib.load('models/le_outcome.joblib')

lstm_model = keras.models.load_model('models/lstm_ndvi_forecaster.keras')
hybrid_df = pd.read_csv('data/matang_ndvi_hybrid_timeseries.csv')
hybrid_df['date'] = pd.to_datetime(hybrid_df['date'])
hybrid_df = hybrid_df.sort_values('date').reset_index(drop=True)

WINDOW_SIZE = 6
FORECAST_HORIZON = 24


@app.route('/')
def home():
    return jsonify({
        'message': 'MangroveWatch API is running',
        'endpoints': ['/predict/suitability (POST)', '/predict/forecast (GET)']
    })


@app.route('/predict/suitability', methods=['POST'])
def predict_suitability():
    data = request.get_json()

    required_fields = [
        'species', 'soil_salinity_ppt', 'tidal_inundation_freq_pct',
        'wave_energy_exposure', 'sedimentation_rate_mm_yr', 'ndvi_at_planting'
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        species_encoded = le_species.transform([data['species']])[0]
        wave_encoded = le_wave.transform([data['wave_energy_exposure']])[0]
    except ValueError as e:
        return jsonify({'error': f'Invalid category value: {str(e)}'}), 400

    features = np.array([[
        species_encoded,
        data['soil_salinity_ppt'],
        data['tidal_inundation_freq_pct'],
        wave_encoded,
        data['sedimentation_rate_mm_yr'],
        data['ndvi_at_planting']
    ]])

    pred_encoded = xgb_model.predict(features)[0]
    pred_proba = xgb_model.predict_proba(features)[0]
    outcome = le_outcome.inverse_transform([pred_encoded])[0]

    proba_dict = {
        cls: round(float(prob), 4)
        for cls, prob in zip(le_outcome.classes_, pred_proba)
    }

    return jsonify({
        'predicted_outcome': outcome,
        'probabilities': proba_dict
    })


@app.route('/predict/forecast', methods=['GET'])
def predict_forecast():
    df = hybrid_df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    last_window = df[['ndvi_value', 'month_sin', 'month_cos']].values[-WINDOW_SIZE:].tolist()
    last_date = df['date'].iloc[-1]
    forecast_dates = pd.date_range(
        last_date + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq='MS'
    )

    forecasts = []
    current_window = last_window.copy()

    for step in range(FORECAST_HORIZON):
        input_seq = np.array(current_window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 3)
        next_ndvi = float(np.clip(lstm_model.predict(input_seq, verbose=0)[0][0], 0, 1))
        forecasts.append(next_ndvi)

        next_month = forecast_dates[step].month
        next_sin = np.sin(2 * np.pi * next_month / 12)
        next_cos = np.cos(2 * np.pi * next_month / 12)
        current_window.append([next_ndvi, next_sin, next_cos])

    result = [
        {'date': str(d.date()), 'forecasted_ndvi': round(v, 4)}
        for d, v in zip(forecast_dates, forecasts)
    ]

    return jsonify({'forecast_horizon_months': FORECAST_HORIZON, 'forecast': result})


if __name__ == '__main__':
    app.run(debug=True, port=5000)