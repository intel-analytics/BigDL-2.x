import os
import numpy as np
import matplotlib.pyplot as plt

from pyspark import SparkContext

from bigdl.util.common import init_engine
from bigdl.util.common import create_spark_conf
from bigdl.util.common import JavaCreator
from bigdl.util.common import Sample
from vision.image3d.transformation import *
import h5py
from math import pi

sample = h5py.File('./a.mat')['meniscus_im']
sample = np.array(sample)
sample = Sample.from_ndarray(features=sample, label=np.array(-1))
# sample = np.expand_dims(sample,0)

print(sample.features[0].shape)

sc = SparkContext(appName="test", conf=create_spark_conf())
JavaCreator.set_creator_class("com.intel.analytics.zoo.transform.vision.image3d.python.api.VisionPythonBigDL")
init_engine()

data_rdd = sc.parallelize([sample])

start_loc = [13, 80, 125]
patch = [5, 40, 40]
# end_loc = [17,119,164]
crop = Crop(start=start_loc, patch_size=patch)
crop_rdd = crop(data_rdd)
crop_data = crop_rdd.first()
print(crop_data.features[0].shape)

yaw = 0.0
pitch = 0.0
roll = pi/6
rotate_30 = Rotate([yaw, pitch, roll])
rotate_30_rdd = rotate_30(crop_rdd)
rotate_30_data = rotate_30_rdd.first()
print(rotate_30_data.features[0].shape)


yaw = 0.0
pitch = 0.0
roll = pi/2
rotate_90 = Rotate([yaw, pitch, roll])
rotate_90_rdd = rotate_90(crop_rdd)
rotate_90_data = rotate_90_rdd.first()
print(rotate_90_data.features[0].shape)

affine = AffineTransform(JTensor.from_ndarray(np.random.rand(3,3)))
affine_rdd = affine(crop_rdd)
affine_data = affine_rdd.first()
print(affine_data.features[0].shape)

pipe = Pipeline([crop, rotate_90])
out_rdd = pipe(data_rdd)
out_data = out_rdd.first()

cropped_sample = crop.transform(sample)

fig = plt.figure(figsize=[10, 10])
y = fig.add_subplot(3,3,1)
y.add_patch(plt.Rectangle((start_loc[2]-1,start_loc[1]-1),
                          patch[1],
                          patch[1], fill=False,
                          edgecolor='red', linewidth=1.5)
            )
y.text(start_loc[2]-45, start_loc[1]-15,
       'Cropped Region',
       bbox=dict(facecolor='green', alpha=0.5),
       color='white')
sample_np = sample.features[0].to_ndarray()
y.imshow(sample_np[15, :, :],cmap='gray')
y.set_title('Original Image')

y = fig.add_subplot(3, 3, 2)
crop_data_np = crop_data.features[0].to_ndarray()
y.imshow(crop_data_np[2, :, :],cmap='gray')
y.set_title('Cropped Image')

y = fig.add_subplot(3, 3, 3)
rotate_30_data_np = rotate_30_data.features[0].to_ndarray()
y.imshow(rotate_30_data_np[2, :, :],cmap='gray')
y.set_title('Rotate 30 Deg')

y = fig.add_subplot(3, 3, 4)
rotate_90_data_np = rotate_90_data.features[0].to_ndarray()
y.imshow(rotate_90_data_np[2, :, :],cmap='gray')
y.set_title('Rotate 90 Deg')

y = fig.add_subplot(3, 3, 5)
affine_data_np = affine_data.features[0].to_ndarray()
y.imshow(affine_data_np[2, :, :],cmap='gray')
y.set_title('Random Affine Transformation')

y = fig.add_subplot(3, 3, 6)
out_data_np = out_data.features[0].to_ndarray()
y.imshow(out_data_np[2, :, :],cmap='gray')
y.set_title('Pipeline Transformation')

y = fig.add_subplot(3, 3, 7)
cropped_sample_np = cropped_sample.features[0].to_ndarray()
y.imshow(cropped_sample_np[2, :, :],cmap='gray')
y.set_title('Cropped Sample')

fig.show()
print("finish")

