# Running Orca TF2 YoloV3 example


## Environment

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

```bash
conda create -n analytics-zoo python==3.7
conda activate analytics-zoo
pip install tensorflow
pip install pandas
```

Then download and install latest nightly-build Analytics Zoo 

```bash
pip install --pre --upgrade analytics-zoo[ray]
```

## Training Data

Download VOC2009 dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar) 


## Pretrained Weights

Download pretrained weights [here](https://pjreddie.com/media/files/yolov3.weights)

## Running example

### Train
Example command:

```bash
python yoloV3.py --data_dir ${data_dir} --weights ${weights} --class_num ${class_num} --names ${names}
```
Result:
```bash
  1/217 [..............................]
(pid=8091)  - ETA: 1:06:59 - loss: 9804.1631 - yolo_output_0_loss: 457.6100 - yolo_output_1_loss: 1600.1824 - yolo_output_2_loss: 7735.6562
(pid=8091) 2021-06-18 08:01:40.067494: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
...
218/218 [==============================] - 205s 939ms/step - loss: 29.3999 - yolo_output_0_loss: 9.6335 - yolo_output_1_loss: 5.0190 - yolo_output_2_loss: 12.8991
```

### Predict
Example command:

```bash
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf2/yolov3/predict.py --checkpoint ${checkpoint} --names ${names} --class_num ${class_num} --image ${image}
--output ${output}
```
Result:
```bash
detections:
cup, 0.9980731010437012, [0.14266217 0.52777606 0.27184254 0.65748256]
```
You can also find new generated images stored in output_path, and the objects in the images are with a box around them [labeled "name"]
