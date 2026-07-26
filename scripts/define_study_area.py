import ee
import geemap

ee.Initialize(project='mangrovewatch-fyp')

matang_boundary = ee.Geometry.Rectangle([
    100.52, 4.70,   # min longitude, min latitude
    100.68, 4.90    # max longitude, max latitude
])

def get_map():
    Map = geemap.Map()
    Map.add_basemap('SATELLITE')

    outline = ee.Image().byte().paint(
        featureCollection=ee.FeatureCollection([ee.Feature(matang_boundary)]),
        color=1,
        width=8
    )
    Map.addLayer(outline, {'palette': 'FF0000'}, 'Matang Study Area')
    Map.centerObject(matang_boundary, zoom=10)
    return Map