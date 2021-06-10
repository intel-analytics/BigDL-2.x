# Running Orca TF2 YoloV3 example


## Environment

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

```bash
conda create -n analytics-zoo python==3.7.
conda activate analytics-zoo
pip install tensorflow
pip install pandas
```

Then download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip).

E.g.
```bash
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl[ray]
```

## Training Data

Download VOC2009 [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar) 


## Pretrained Weights

Download pretrained weights [here](https://pjreddie.com/media/files/yolov3.weights)

## Running example

Example command:

```
python yoloV3.py --data ${data_dir} --names ${voc_names} --weights ${weights} --class_num ${num}
```

