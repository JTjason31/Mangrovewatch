import ee
import pandas as pd
from datetime import datetime

ee.Initialize(project='mangrovewatch-fyp')

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)

def get_ndvi_at_point(lon, lat, date_str, buffer_days=45):
    point = ee.Geometry.Point([lon, lat])
    target_date = ee.Date(date_str)
    start = target_date.advance(-buffer_days, 'day')
    end = target_date.advance(buffer_days, 'day')

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
        .map(mask_s2_clouds)
    )

    count = collection.size().getInfo()
    if count == 0:
        return None

    composite = collection.median()
    ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

    value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(30),
        scale=10,
        maxPixels=1e9
    ).get('NDVI').getInfo()

    return value

if __name__ == "__main__":
    df = pd.read_csv('data/matang_dummy_restoration_dataset.csv')

    real_ndvi_values = []
    for idx, row in df.iterrows():
        print(f"Processing {row['plot_id']} ({idx+1}/{len(df)})...")
        val = get_ndvi_at_point(row['longitude'], row['latitude'], row['planting_date'])
        real_ndvi_values.append(val)
        print(f"  -> NDVI: {val}")

    df['ndvi_real_satellite'] = real_ndvi_values
    df['ndvi_real_satellite'] = df['ndvi_real_satellite'].fillna(df['ndvi_at_planting'])

    df.to_csv('data/matang_dummy_restoration_dataset_with_real_ndvi.csv', index=False)
    print("\nSaved to data/matang_dummy_restoration_dataset_with_real_ndvi.csv")
    print(f"\nMatched real NDVI for {df['ndvi_real_satellite'].notna().sum()} / {len(df)} plots")