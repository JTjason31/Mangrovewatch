from flask import Flask, request, jsonify, render_template, send_file, make_response
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from datetime import datetime, timezone
import io
import csv
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

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
    return render_template('index.html')


@app.route('/api/info')
def api_info():
    return jsonify({
        'message': 'MangroveWatch API is running',
        'endpoints': [
            '/predict/suitability (POST)',
            '/predict/forecast (GET)',
            '/predict/species_recommendation (POST)',
            '/export/csv (POST)',
            '/export/geojson (POST)',
            '/export/pdf (POST)'
        ]
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

@app.route('/predict/species_recommendation', methods=['POST'])
def predict_species_recommendation():
    data = request.get_json()

    required_fields = [
        'soil_salinity_ppt', 'tidal_inundation_freq_pct',
        'wave_energy_exposure', 'sedimentation_rate_mm_yr', 'ndvi_at_planting'
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        wave_encoded = le_wave.transform([data['wave_energy_exposure']])[0]
    except ValueError as e:
        return jsonify({'error': f'Invalid category value: {str(e)}'}), 400

    results = []
    for species_name in le_species.classes_:
        species_encoded = le_species.transform([species_name])[0]

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

        # Probability of "High" outcome specifically, used for ranking
        high_idx = list(le_outcome.classes_).index('High')
        high_prob = float(pred_proba[high_idx])

        results.append({
            'species': species_name,
            'predicted_outcome': outcome,
            'probability_high_success': round(high_prob, 4)
        })

    # Rank species by likelihood of High success, best first
    results.sort(key=lambda x: x['probability_high_success'], reverse=True)

    return jsonify({
        'site_conditions': data,
        'species_ranked_by_suitability': results
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


def _extract_export_payload(data):
    """Normalises the JSON body shared by all three export endpoints.
    Expects: { site: {...}, suitability: {...}, species_recommendation: {...}, forecast: {...} (optional) }
    """
    site = data.get('site', {})
    suitability = data.get('suitability', {})
    species_rec = data.get('species_recommendation', {})
    forecast = data.get('forecast', {})
    missing = [k for k in ('site', 'suitability', 'species_recommendation') if k not in data]
    if missing:
        raise ValueError(f'Missing required fields in export payload: {missing}')
    return site, suitability, species_rec, forecast


@app.route('/export/csv', methods=['POST'])
def export_csv():
    """FR10: export current analysis as a CSV prediction table."""
    data = request.get_json() or {}
    try:
        site, suitability, species_rec, forecast = _extract_export_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(['MangroveWatch Restoration-Suitability Analysis'])
    writer.writerow(['Generated', datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')])
    writer.writerow([])

    writer.writerow(['Site Conditions'])
    for key in ('species', 'latitude', 'longitude', 'soil_salinity_ppt',
                'tidal_inundation_freq_pct', 'wave_energy_exposure',
                'sedimentation_rate_mm_yr', 'ndvi_at_planting'):
        if key in site:
            writer.writerow([key, site[key]])
    writer.writerow([])

    writer.writerow(['Predicted Outcome', suitability.get('predicted_outcome', '')])
    writer.writerow(['Class', 'Probability'])
    for cls, prob in suitability.get('probabilities', {}).items():
        writer.writerow([cls, prob])
    writer.writerow([])

    writer.writerow(['Species Recommendation (ranked)'])
    writer.writerow(['Rank', 'Species', 'Predicted Outcome', 'Probability of High Success'])
    for i, s in enumerate(species_rec.get('species_ranked_by_suitability', []), start=1):
        writer.writerow([i, s.get('species'), s.get('predicted_outcome'), s.get('probability_high_success')])

    if forecast.get('forecast'):
        writer.writerow([])
        writer.writerow(['24-Month NDVI Forecast'])
        writer.writerow(['Date', 'Forecasted NDVI'])
        for row in forecast['forecast']:
            writer.writerow([row.get('date'), row.get('forecasted_ndvi')])

    output = make_response(buf.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=mangrovewatch_analysis.csv'
    output.headers['Content-Type'] = 'text/csv'
    return output


@app.route('/export/geojson', methods=['POST'])
def export_geojson():
    """FR10: export current analysis as a GeoJSON map layer (single-site point feature)."""
    data = request.get_json() or {}
    try:
        site, suitability, species_rec, forecast = _extract_export_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    lat = site.get('latitude')
    lon = site.get('longitude')
    if lat is None or lon is None:
        return jsonify({'error': 'site.latitude and site.longitude are required for GeoJSON export'}), 400

    top_species = None
    ranked = species_rec.get('species_ranked_by_suitability', [])
    if ranked:
        top_species = ranked[0].get('species')

    feature = {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [float(lon), float(lat)]
        },
        'properties': {
            'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            'species_selected': site.get('species'),
            'soil_salinity_ppt': site.get('soil_salinity_ppt'),
            'tidal_inundation_freq_pct': site.get('tidal_inundation_freq_pct'),
            'wave_energy_exposure': site.get('wave_energy_exposure'),
            'sedimentation_rate_mm_yr': site.get('sedimentation_rate_mm_yr'),
            'ndvi_at_planting': site.get('ndvi_at_planting'),
            'predicted_outcome': suitability.get('predicted_outcome'),
            'probability_high': suitability.get('probabilities', {}).get('High'),
            'probability_moderate': suitability.get('probabilities', {}).get('Moderate'),
            'probability_low': suitability.get('probabilities', {}).get('Low'),
            'top_recommended_species': top_species,
        }
    }
    geojson = {
        'type': 'FeatureCollection',
        'name': 'mangrovewatch_analysis',
        'features': [feature]
    }

    output = make_response(json.dumps(geojson, indent=2))
    output.headers['Content-Disposition'] = 'attachment; filename=mangrovewatch_analysis.geojson'
    output.headers['Content-Type'] = 'application/geo+json'
    return output


@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    """FR10: export current analysis as a PDF report."""
    data = request.get_json() or {}
    try:
        site, suitability, species_rec, forecast = _extract_export_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=18, spaceAfter=4)
    meta_style = ParagraphStyle('MetaStyle', parent=styles['Normal'], fontSize=9, textColor=colors.grey)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceBefore=14, spaceAfter=6)

    story = [
        Paragraph('MangroveWatch — Restoration-Suitability Report', title_style),
        Paragraph('Matang Mangrove Forest Reserve, Perak', meta_style),
        Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", meta_style),
        Spacer(1, 10),
    ]

    # Site conditions table
    story.append(Paragraph('Site Conditions', h2))
    site_rows = [['Parameter', 'Value']]
    labels = {
        'species': 'Species', 'latitude': 'Latitude', 'longitude': 'Longitude',
        'soil_salinity_ppt': 'Soil salinity (ppt)',
        'tidal_inundation_freq_pct': 'Tidal inundation frequency (%)',
        'wave_energy_exposure': 'Wave energy exposure',
        'sedimentation_rate_mm_yr': 'Sedimentation rate (mm/yr)',
        'ndvi_at_planting': 'NDVI at planting',
    }
    for key, label in labels.items():
        if key in site:
            site_rows.append([label, str(site[key])])
    site_table = Table(site_rows, colWidths=[8 * cm, 8 * cm])
    site_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0E4F52')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F4F2')]),
    ]))
    story.append(site_table)

    # Suitability outcome
    story.append(Paragraph('Predicted Restoration Outcome', h2))
    story.append(Paragraph(f"<b>{suitability.get('predicted_outcome', 'N/A')}</b>", styles['Normal']))
    prob_rows = [['Class', 'Probability']]
    for cls, prob in suitability.get('probabilities', {}).items():
        prob_rows.append([cls, f"{prob:.2%}" if isinstance(prob, (int, float)) else str(prob)])
    prob_table = Table(prob_rows, colWidths=[8 * cm, 8 * cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0E4F52')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
    ]))
    story.append(Spacer(1, 6))
    story.append(prob_table)

    # Species recommendation
    story.append(Paragraph('Species Recommendation (ranked)', h2))
    sp_rows = [['Rank', 'Species', 'Predicted Outcome', 'P(High Success)']]
    for i, s in enumerate(species_rec.get('species_ranked_by_suitability', []), start=1):
        prob = s.get('probability_high_success')
        sp_rows.append([
            str(i), s.get('species', ''), s.get('predicted_outcome', ''),
            f"{prob:.2%}" if isinstance(prob, (int, float)) else str(prob)
        ])
    sp_table = Table(sp_rows, colWidths=[1.5 * cm, 6 * cm, 4.5 * cm, 4 * cm])
    sp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0E4F52')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F4F2')]),
    ]))
    story.append(sp_table)

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        'MangroveWatch is a Final Year Project prototype. Predictions are generated from an XGBoost '
        'classifier and LSTM forecaster trained on a structured, supervisor-approved synthetic restoration '
        'dataset combined with real Sentinel-2 satellite imagery. Results are illustrative of the platform\u2019s '
        'predictive capability and should not be used for actual restoration decision-making without further '
        'validation against field data.',
        meta_style
    ))

    doc.build(story)
    buf.seek(0)
    return send_file(
        buf, mimetype='application/pdf',
        as_attachment=True, download_name='mangrovewatch_report.pdf'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)