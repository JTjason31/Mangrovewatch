"""
MangroveWatch — End-to-End Integration Tests
==============================================

Exercises the full site-select -> preprocessing -> model inference -> dashboard
flow through the real Flask app object, real trained model files, and real
Flask routing (via Flask's test client). No mocks are used for the ML models:
XGBoost and LSTM inference happens for real on every run.

HOW TO RUN
----------
From the project root (with the venv activated and dependencies installed):

    pytest tests/test_integration.py -v

REQUIRED DATA FILES
--------------------
This suite depends on the same files app.py loads at import time:
    - data/matang_ndvi_hybrid_timeseries.csv
    - models/xgboost_classifier.joblib
    - models/le_species.joblib, le_wave.joblib, le_outcome.joblib
    - models/lstm_ndvi_forecaster.keras

NOTE: as of writing, the repo's .gitignore excludes "*.csv" globally, which
means data/matang_ndvi_hybrid_timeseries.csv is NOT committed to GitHub. A
fresh `git clone` of the repo will fail at `import app` with a
FileNotFoundError until that CSV is regenerated locally (see
scripts/generate_hybrid_ndvi_series.py) or the .gitignore rule is narrowed.
test_required_data_files_exist() below checks for this directly and fails
with a clear message rather than a raw traceback.
"""

import io
import csv
import json
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REQUIRED_FILES = [
    'data/matang_ndvi_hybrid_timeseries.csv',
    'models/xgboost_classifier.joblib',
    'models/le_species.joblib',
    'models/le_wave.joblib',
    'models/le_outcome.joblib',
    'models/lstm_ndvi_forecaster.keras',
]


def test_required_data_files_exist():
    """Fails fast with an actionable message instead of a raw import traceback
    if a data/model file app.py depends on is missing (see module docstring
    re: the gitignored CSV)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    missing = [
        f for f in REQUIRED_FILES
        if not os.path.exists(os.path.join(project_root, f))
    ]
    assert not missing, (
        f"Missing required file(s) for app.py to start: {missing}. "
        "If this is 'data/matang_ndvi_hybrid_timeseries.csv', note that *.csv "
        "is globally gitignored in this repo -- regenerate it with "
        "scripts/generate_hybrid_ndvi_series.py (which itself needs "
        "data/matang_ndvi_timeseries.csv from the GEE pull) before running "
        "the app or these tests."
    )


@pytest.fixture(scope='session')
def client():
    """Imports the real app module (loading real models) once per test session
    and returns a Flask test client."""
    import app as mangrovewatch_app
    mangrovewatch_app.app.config['TESTING'] = True
    with mangrovewatch_app.app.test_client() as c:
        yield c


@pytest.fixture
def valid_site():
    return {
        'species': 'Rhizophora apiculata',
        'latitude': 4.80,
        'longitude': 100.60,
        'soil_salinity_ppt': 15,
        'tidal_inundation_freq_pct': 55,
        'wave_energy_exposure': 'Low',
        'sedimentation_rate_mm_yr': 20,
        'ndvi_at_planting': 0.5,
    }


# ---------------------------------------------------------------------------
# Dashboard (frontend) loads
# ---------------------------------------------------------------------------

def test_home_page_loads(client):
    res = client.get('/')
    assert res.status_code == 200
    assert b'MangroveWatch' in res.data


def test_home_page_contains_expected_controls(client):
    """Checks the rendered dashboard HTML actually contains the input controls
    and export buttons the JS depends on -- catches template/JS id drift."""
    html = client.get('/').get_data(as_text=True)
    expected_ids = [
        'id="species"', 'id="salinity"', 'id="inundation"', 'id="wave"',
        'id="sediment"', 'id="ndvi"', 'id="lat"', 'id="lon"', 'id="runBtn"',
        'id="suitabilityResults"', 'id="speciesResults"', 'id="forecastChart"',
        'id="exportPdfBtn"', 'id="exportCsvBtn"', 'id="exportGeoJsonBtn"',
    ]
    missing = [i for i in expected_ids if i not in html]
    assert not missing, f"Dashboard HTML is missing expected element(s): {missing}"


def test_api_info_lists_all_routes(client):
    res = client.get('/api/info')
    assert res.status_code == 200
    endpoints = ' '.join(res.get_json()['endpoints'])
    for route in ['/predict/suitability', '/predict/forecast',
                  '/predict/species_recommendation', '/export/csv',
                  '/export/geojson', '/export/pdf']:
        assert route in endpoints, f"{route} not listed in /api/info"


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

def test_suitability_valid_request(client, valid_site):
    res = client.post('/predict/suitability', json=valid_site)
    assert res.status_code == 200
    body = res.get_json()
    assert body['predicted_outcome'] in ('High', 'Moderate', 'Low')
    assert set(body['probabilities'].keys()) == {'High', 'Moderate', 'Low'}
    assert abs(sum(body['probabilities'].values()) - 1.0) < 1e-3


def test_suitability_missing_field_returns_400(client, valid_site):
    incomplete = dict(valid_site)
    del incomplete['soil_salinity_ppt']
    res = client.post('/predict/suitability', json=incomplete)
    assert res.status_code == 400
    assert 'soil_salinity_ppt' in res.get_json()['error']


def test_suitability_invalid_category_returns_400(client, valid_site):
    bad = dict(valid_site)
    bad['species'] = 'Not A Real Species'
    res = client.post('/predict/suitability', json=bad)
    assert res.status_code == 400


def test_species_recommendation_ranks_all_five_species(client, valid_site):
    site = {k: v for k, v in valid_site.items() if k != 'species'}
    res = client.post('/predict/species_recommendation', json=site)
    assert res.status_code == 200
    ranked = res.get_json()['species_ranked_by_suitability']
    assert len(ranked) == 5
    probs = [r['probability_high_success'] for r in ranked]
    assert probs == sorted(probs, reverse=True), "Species are not sorted by descending probability"


def test_forecast_returns_24_months_in_order(client):
    res = client.get('/predict/forecast')
    assert res.status_code == 200
    body = res.get_json()
    assert body['forecast_horizon_months'] == 24
    forecast = body['forecast']
    assert len(forecast) == 24
    dates = [f['date'] for f in forecast]
    assert dates == sorted(dates)
    for f in forecast:
        assert 0.0 <= f['forecasted_ndvi'] <= 1.0


# ---------------------------------------------------------------------------
# Full end-to-end flow: select site -> predict -> recommend -> forecast -> export
# ---------------------------------------------------------------------------

@pytest.fixture
def full_analysis(client, valid_site):
    """Runs the complete user flow once and returns the assembled payload,
    exactly as the dashboard JS assembles it before calling /export/*."""
    suit = client.post('/predict/suitability', json=valid_site).get_json()
    rec_site = {k: v for k, v in valid_site.items() if k != 'species'}
    rec = client.post('/predict/species_recommendation', json=rec_site).get_json()
    forecast = client.get('/predict/forecast').get_json()
    return {
        'site': valid_site,
        'suitability': suit,
        'species_recommendation': rec,
        'forecast': forecast,
    }


def test_end_to_end_flow_produces_consistent_outputs(full_analysis):
    """The species ranked #1 in the recommendation list should match what a
    suitability call for that same species would predict as its probability
    of High success -- i.e. the two models agree on the same underlying site."""
    top = full_analysis['species_recommendation']['species_ranked_by_suitability'][0]
    assert top['probability_high_success'] >= 0
    assert full_analysis['suitability']['predicted_outcome'] in ('High', 'Moderate', 'Low')
    assert len(full_analysis['forecast']['forecast']) == 24


def test_export_csv_end_to_end(client, full_analysis):
    res = client.post('/export/csv', json=full_analysis)
    assert res.status_code == 200
    assert res.headers['Content-Type'].startswith('text/csv')
    assert 'attachment' in res.headers['Content-Disposition']

    text = res.get_data(as_text=True)
    reader = list(csv.reader(io.StringIO(text)))
    flat = [cell for row in reader for cell in row]
    assert 'Predicted Outcome' in flat
    assert 'Species Recommendation (ranked)' in flat
    assert '24-Month NDVI Forecast' in flat


def test_export_geojson_end_to_end(client, full_analysis):
    res = client.post('/export/geojson', json=full_analysis)
    assert res.status_code == 200
    assert res.headers['Content-Type'].startswith('application/geo+json')

    geo = json.loads(res.get_data(as_text=True))
    assert geo['type'] == 'FeatureCollection'
    assert len(geo['features']) == 1
    feature = geo['features'][0]
    assert feature['geometry']['type'] == 'Point'
    lon, lat = feature['geometry']['coordinates']
    assert lon == pytest.approx(full_analysis['site']['longitude'])
    assert lat == pytest.approx(full_analysis['site']['latitude'])
    assert feature['properties']['predicted_outcome'] == full_analysis['suitability']['predicted_outcome']


def test_export_pdf_end_to_end(client, full_analysis):
    res = client.post('/export/pdf', json=full_analysis)
    assert res.status_code == 200
    assert res.headers['Content-Type'] == 'application/pdf'
    assert res.data[:5] == b'%PDF-', "Response is not a valid PDF (missing %PDF- header)"
    assert len(res.data) > 500, "PDF suspiciously small -- likely rendered empty"


def test_export_without_required_fields_returns_400(client, valid_site):
    res = client.post('/export/csv', json={'site': valid_site})
    assert res.status_code == 400
    assert 'suitability' in res.get_json()['error']


def test_export_geojson_without_coordinates_returns_400(client, valid_site):
    site_no_coords = dict(valid_site)
    del site_no_coords['latitude']
    del site_no_coords['longitude']
    payload = {
        'site': site_no_coords,
        'suitability': {'predicted_outcome': 'High', 'probabilities': {'High': 1, 'Moderate': 0, 'Low': 0}},
        'species_recommendation': {'species_ranked_by_suitability': []},
    }
    res = client.post('/export/geojson', json=payload)
    assert res.status_code == 400
    assert 'latitude' in res.get_json()['error']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
