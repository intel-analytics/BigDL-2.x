# Inception Model on Imagenet
This example demonstrates how to use Analytics-zoo to train [Inception v1](https://arxiv
.org/abs/1409.4842) architecture on the [ImageNet](http://image-net.org/index) data.
## Get the JAR
You can build one by refer to the
[Build Page](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/#download-analytics-zoo-source) from the source code. We
will release a pre-build package soon.

## Prepare the data
You can download imagenet-2012 data from <http://image-net.org/download-images>.
 
After you download the files(**ILSVRC2012_img_train.tar** and **ILSVRC2012_img_val.tar**), 
run the following commands to prepare the data.

```bash
mkdir train
mv ILSVRC2012_img_train.tar train/
cd train
tar -xvf ILSVRC2012_img_train.tar
rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read CLASS_NAME ; do mkdir -p "${CLASS_NAME%.tar}"; tar -xvf "${CLASS_NAME}" -C "${CLASS_NAME%.tar}"; done
rm *.tar
cd ../
mkdir val
mv ILSVRC2012_img_val.tar val/
cd val
tar -xvf ILSVRC2012_img_val.tar
cat classes.lst | while read CLASS_NAME; do mkdir -p ${CLASS_NAME}; done
cat img_class.lst | while read PARAM; do mv ${PARAM/ n[0-9]*/} ${PARAM/ILSVRC*JPEG /}; done
rm ILSVRC2012_img_val.tar
```

Now all the images belonging to the same category are moved to the same folder.

This command will transform the images into hadoop sequence files, which are 
more suitable for a distributed training.

```bash
spark-submit --class com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator bigdl-VERSION-jar-with-dependencies.jar -f imagenet_folder -o output_folder -p cores_number
```

It will generate the hadoop sequence files in the output folder.




## Train the Model
* Spark standalone, example command
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
--master spark://xxx.xxx.xxx.xxx:xxxx \  
--executor-cores 32 \  
--num-executors 16 \  
--executor-memory 150G \ 
--driver-memory 20G \ 
--conf spark.network.timeout=10000000  pyzoo/zoo/examples/inception/inception.py \ 
--batchSize 1024 \
--learningRate 0.065 \
--weightDecay 0.0002 \
--checkpointIteration 1000 \ 
-f hdfs://... \
--checkpoint /models/inception \ 
--maxIteration 90000
```

* Spark yarn client mode, example command
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
--master yarn \
--deploy-mode client \
--executor-cores 32 \
--num-executors 16 \ 
--executor-memory 150G \ 
--driver-memory 20G \ 
--conf spark.network.timeout=10000000  pyzoo/zoo/examples/inception/inception.py \ 
--batchSize 1024 \
--learningRate 0.065 \
--weightDecay 0.0002 \
--checkpointIteration 1000 \ 
-f hdfs://... \
--checkpoint /models/incepton \ 
--maxIteration 90000
```

In the above commands
* -f: where you put your ImageNet data, it should be a hdfs folder
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as optimMethod.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* --batchSize: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number *
core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
* --learningRate: inital learning rate. Note in this example, we use a Poly learning rate decay
policy.
* --weightDecay: weight decay.
* --checkpointIteration: the checkpoint interval in iteration.
* --maxLr: optional. Max learning rate after warm up. It has to be set together with warmupEpoch.
* --warmupEpoch: optional. Epoch numbers need to take to increase learning rate from learningRate to maxLR.
* --gradientL2NormThreshold: optional. Gradient L2-Norm threshold used for norm2 gradient clipping.
* --gradientMin: optional. Max gradient clipping by value, used in constant gradient clipping.
* --gradientMax: optional. Min gradient clipping by value, used in constant gradient clipping.
* --maxIteration: max iteration
