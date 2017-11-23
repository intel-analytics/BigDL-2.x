#!/bin/bash

cd ../../transform/vision
mvn clean install
cd ../../pipeline/objectDetection
mvn clean package -DskipTests