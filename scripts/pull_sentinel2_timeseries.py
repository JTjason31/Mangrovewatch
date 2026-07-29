import ee
import pandas as pd

ee.Initialize(project='mangrovewatch-fyp')

matang_boundary = ee.Geometry.Rectangle([
    100.52, 4.70,
    100.68, 4.90
])

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)

def get_monthly_ndvi(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(matang_boundary)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        .map(mask_s2_clouds)
    )

    count = collection.size().getInfo()
    if count == 0:
        return {'year': year, 'month': month, 'image_count': 0, 'mean_ndvi': None}

    composite = collection.median().clip(matang_boundary)
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

    mean_ndvi = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=matang_boundary,
        scale=30,
        maxPixels=1e9
    ).get('NDVI').getInfo()

    return {'year': year, 'month': month, 'image_count': count, 'mean_ndvi': mean_ndvi}

if __name__ == "__main__":
    results = []
    # Sentinel-2 data starts mid-2015, so we'll cover 2016-2024 for full-year coverage
    for year in range(2016, 2025):
        for month in range(1, 13):
            print(f"Processing {year}-{month:02d}...")
            row = get_monthly_ndvi(year, month)
            results.append(row)
            print(row)

    df = pd.DataFrame(results)
    df.to_csv('data/matang_ndvi_timeseries.csv', index=False)
    print("Saved to data/matang_ndvi_timeseries.csv")