#!/bin/bash

# Assume currently at the root of analytics-zoo
# mvn install image transformer lib
cd transform/vision
mvn clean install

# package ssd
cd ../../pipeline/ssd
mvn clean package -DskipTests
