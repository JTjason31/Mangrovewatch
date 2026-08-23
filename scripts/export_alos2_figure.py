import ee
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO

ee.Initialize(project='mangrovewatch-fyp')

matang_boundary = ee.Geometry.Rectangle([
    100.52, 4.70,
    100.68, 4.90
])

collection = (
    ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
    .filterBounds(matang_boundary)
    .filterDate('2015-01-01', '2022-01-01')
    .select('HH')
)

composite = collection.mosaic().clip(matang_boundary)
hh_db = composite.pow(2).log10().multiply(10).subtract(83.0).rename('HH_dB')

vis_params = {'bands': ['HH_dB'], 'min': -35, 'max': 13, 'palette': ['000000', '808080', 'FFFFFF']}

url = hh_db.getThumbURL({
    **vis_params,
    'region': matang_boundary,
    'dimensions': 800,
    'format': 'png'
})

response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.save('data/alos2_hh_backscatter.png')

print(f"Number of ALOS-2 images used in mosaic: {collection.size().getInfo()}")
print("Saved to data/alos2_hh_backscatter.png")