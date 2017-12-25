mvn clean package
BigDL_PYTHON=/home/jxy/code/Bigdl/spark-dl/spark/dist/target/bigdl-0.4.0-SNAPSHOT-spark-2.0.0-scala-2.11.8-all-dist/lib/bigdl-0.4.0-SNAPSHOT-python-api.zip
cp ${BigDL_PYTHON} .
rm -r python
unzip ${BigDL_PYTHON} -d python
unzip -n target/models-0.1-SNAPSHOT-python-api.zip -d python
cd python
zip -r ../target/bigdl-models-0.1-SNAPSHOT-python-api.zip *


