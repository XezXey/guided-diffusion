from envmap import EnvironmentMap, rotation_matrix
from PIL import Image
import numpy as np

# ee = EnvironmentMap('./envmap/zwartkops_start_morning_8k.exr', 'latlong')
ee = EnvironmentMap('./envmap/zwartkops_start_morning_8k.hdr', 'latlong')
dcm = rotation_matrix(azimuth=0,elevation=0,roll=0)    

crop = ee.project(vfov=60., rotation_matrix=dcm, ar=1.0, resolution=(512, 512),projection="perspective", mode="normal")

print(crop.max(), crop.min(), crop.shape)
img = Image.fromarray((crop*255).astype(np.uint8))
# img = Image.fromarray((crop*255).clip(0, 255).astype(np.uint8))
img.save("crop.png")
