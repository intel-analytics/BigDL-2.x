#!/bin/bash

# mvn install opencv jar
if [ ! -f "opencv-analytics-3.2.0.jar" ]
then
    wget https://github.com/SeaOfOcean/OpencvWrapper/raw/master/opencv-analytics-3.2.0.jar
    mvn install:install-file -Dfile=opencv-analytics-3.2.0.jar -DgroupId=com.intel.analytics -DartifactId=opencv-analytics -Dversion=3.2.0 -Dpackaging=jar -DgeneratePom=true
fi

# package
mvn clean package -DskipTests -P mac