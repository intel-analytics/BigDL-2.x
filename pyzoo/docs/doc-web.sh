#!/bin/bash

## Usage ###################
# Run ./doc-web.sh [port] to launch a http server to view python doc after files have been generated
# Example
# ./doc-web.sh 8080
############################

python doc-web.py $1
