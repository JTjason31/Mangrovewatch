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

# Build a stable, cloud-free NDVI composite using a wider date range
# so we get good coverage across the whole study area
collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(matang_boundary)
    .filterDate('2023-01-01', '2024-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
)

print("Number of images in composite:", collection.size().getInfo())

composite = collection.median().clip(matang_boundary)
ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Mask to plausible mangrove-vegetation NDVI range only
mangrove_mask = ndvi.gte(0.25).And(ndvi.lte(0.75))
ndvi_masked = ndvi.updateMask(mangrove_mask)

# Randomly sample 600 points from valid mangrove pixels (buffer above 500
# in case some points fail later steps)
samples = ndvi_masked.sample(
    region=matang_boundary,
    scale=10,
    numPixels=15000,
    seed=42,
    geometries=True,
    tileScale=4
)

sample_list = samples.getInfo()['features']
print(f"Sampled {len(sample_list)} valid mangrove-vegetation points")

rows = []
for feature in sample_list:
    coords = feature['geometry']['coordinates']
    ndvi_val = feature['properties']['NDVI']
    rows.append({
        'longitude': coords[0],
        'latitude': coords[1],
        'ndvi_real_satellite': ndvi_val
    })

df = pd.DataFrame(rows)

# Trim to exactly 500 if we got more
df = df.head(500).reset_index(drop=True)

df.to_csv('data/valid_mangrove_plot_locations.csv', index=False)
print(f"\nSaved {len(df)} valid plot locations to data/valid_mangrove_plot_locations.csv")
print(df['ndvi_real_satellite'].describe())