#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_HOME
export ANALYTICS_ZOO_HOME_DIST=$ANALYTICS_ZOO_HOME/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME_DIST}/conf/spark-bigdl.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

echo "#3 start app test for using_variational_autoencoder_and_deep_feature_loss_to_generate_faces"
chmod +x ./apps/ipynb2py.sh
./apps/ipynb2py.sh ./apps/variational_autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces
sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g" ./apps/variational_autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py

FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/bigdl_vgg-16_imagenet_0.4.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG model"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational_autoencoder/bigdl_vgg-16_imagenet_0.4.0.model --no-host-directories 
   echo "Finished"
fi
        
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py, ${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/utils.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py        
        
                                  

echo "#4 start app test for using_variational_autoencoder_to_generate_faces"
chmod +x ./apps/ipynb2py.sh
./apps/ipynb2py.sh ./apps/variational_autoencoder/using_variational_autoencoder_to_generate_faces
sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g" ./apps/variational_autoencoder/using_variational_autoencoder_to_generate_faces.py


FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational_autoencoder/img_align_celeba.zip --no-host-directories 
   unzip ${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/img_align_celeba.zip
   echo "Finished"
fi

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/using_variational_autoencoder_to_generate_faces.py, ${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/utils.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/using_variational_autoencoder_to_generate_faces.py
        
echo "#5 start app test for using_variational_autoencoder_to_generate_digital_numbers"        
chmod +x ./apps/ipynb2py.sh
./apps/ipynb2py.sh ./apps/variational_autoencoder/using_variational_autoencoder_to_generate_digital_numbers

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py
        

