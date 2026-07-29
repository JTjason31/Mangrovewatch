import ee
import geemap

ee.Initialize(project='mangrovewatch-fyp')

matang_boundary = ee.Geometry.Rectangle([
    100.52, 4.70,
    100.68, 4.90
])

# Cloud masking function for Sentinel-2 using the QA60 band
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)

# Pull one month of Sentinel-2 surface reflectance data
collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(matang_boundary)
    .filterDate('2024-01-01', '2024-02-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(mask_s2_clouds)
)

# Create a median composite (reduces noise from multiple images)
composite = collection.median().clip(matang_boundary)

# Compute NDVI (Normalized Difference Vegetation Index)
ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')

def get_map():
    Map = geemap.Map()
    Map.add_basemap('SATELLITE')

    vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}
    Map.addLayer(composite, vis_params, 'Sentinel-2 True Color')

    ndvi_vis = {'min': -1, 'max': 1, 'palette': ['red', 'yellow', 'green']}
    Map.addLayer(ndvi, ndvi_vis, 'NDVI')

    Map.centerObject(matang_boundary, zoom=11)
    return Map