# Object Detection
This is a simple example of object detection using Analytics Zoo Object Detection API. We use SSD-MobileNet to predict instances of target classes in the given video, which can be regarded as a sequence of images. The video is a scene of training a dog from ([YouTube-8M dataset](https://research.google.com/youtube8m/)) and the people and the dog are among target classes. Proposed areas are labeled with boxes and class scores.

## Environment
* Python 2.7
* Apache Spark 1.6.0
* Analytics Zoo 0.1.0

## Run with Jupyter
* Download Analytics Zoo and build it.
* Run `export ZOO_HOME=the root directory of the Analytics Zoo project`
* Prepare the video to detect. ([YouTube-8M](https://research.google.com/youtube8m/))
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie MASTER = local\[physcial_core_number\]
```bash
MASTER=local[*]
${ZOO_HOME}/scripts/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 8g  \
    --total-executor-cores 2  \
    --executor-cores 2  \
    --executor-memory 8g \
```