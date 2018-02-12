argLen=$#
if [ $argLen -eq 0 ]; then
    mvn clean package -DskipTests
else
    profile=$1
    if [ $profile == "all-in-one" ]; then
    mvn clean package -P $profile
    else
    mvn clean package -DskipTests
    fi
fi
# Donwload BigDL python api
BigDL=dist-spark-2.1.1-scala-2.11.8-all-0.5.0-20180202.211829-27-dist
if [ ! -d $BigDL ]; then
    wget https://oss.sonatype.org/content/groups/public/com/intel/analytics/bigdl/dist-spark-2.1.1-scala-2.11.8-all/0.5.0-SNAPSHOT/$BigDL.zip
    unzip $BigDL.zip -d $BigDL
fi
# Merge BigDL python API and models python API
BigDL_PYTHON=${BigDL}/lib/bigdl-0.5.0-SNAPSHOT-python-api.zip
cp ${BigDL_PYTHON} .
if [ -d python ]; then
    rm -r python
fi
unzip ${BigDL_PYTHON} -d python
unzip -n target/segmentation-0.1-SNAPSHOT-python-api.zip -d python
cd python
zip -r ../target/bigdl-segmentation-0.1-SNAPSHOT-python-api.zip *
rm -r ../python