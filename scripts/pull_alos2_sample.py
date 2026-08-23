import ee
import geemap

ee.Initialize(project='mangrovewatch-fyp')

matang_boundary = ee.Geometry.Rectangle([
    100.52, 4.70,
    100.68, 4.90
])

# ALOS-2 PALSAR-2 yearly mosaic (global, HH/HV polarization backscatter)
collection = (
    ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
    .filterBounds(matang_boundary)
    .filterDate('2015-01-01', '2022-01-01')
    .select('HH')
)

print("Number of ALOS-2 images found:", collection.size().getInfo())

composite = collection.mosaic().clip(matang_boundary)

# Convert digital number (DN) to backscatter coefficient (dB), standard ALOS-2 calibration
def dn_to_db(image):
    hh = image.select('HH')
    hh_db = hh.pow(2).log10().multiply(10).subtract(83.0).rename('HH_dB')
    return image.addBands(hh_db)

composite = dn_to_db(composite)

def get_map():
    Map = geemap.Map()
    Map.add_basemap('SATELLITE')

    vis_params = {'bands': ['HH_dB'], 'min': -35, 'max': 13, 'palette': ['black', 'gray', 'white']}
    Map.addLayer(composite, vis_params, 'ALOS-2 HH Backscatter (dB)')

    Map.centerObject(matang_boundary, zoom=11)
    return Map