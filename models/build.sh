argLen=$#
if [ $argLen -eq 0 ]; then
    mvn clean package
else
    profile=$1
    if [ $profile == "all-in-one" ]; then
    mvn clean package -P $profile
    else
    mvn clean package
    fi
fi
# Donwload BigDL python api
BigDL=dist-spark-2.1.1-scala-2.11.8-all-0.4.0-dist
if [ ! -d $BigDL ]; then
    wget https://s3-ap-southeast-1.amazonaws.com/bigdl-download/$BigDL.zip
    unzip $BigDL.zip -d $BigDL
fi
# Merge BigDL python API and models python API
BigDL_PYTHON=${BigDL}/lib/bigdl-0.4.0-python-api.zip
cp ${BigDL_PYTHON} .
if [ -d python ]; then
    rm -r python
fi
unzip ${BigDL_PYTHON} -d python
unzip -n target/models-0.1-SNAPSHOT-python-api.zip -d python
cd python
zip -r ../target/bigdl-models-0.1-SNAPSHOT-python-api.zip *
rm -r ../python


