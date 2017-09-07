#!/usr/bin/env bash

SCRIPT_PATH=`dirname $0`
CLASS_PATH=/home/vorobyev/ProgramFiles/projects/java/searcher/target/miner-jar-with-dependencies.jar
INPUT_PATH=/home/vorobyev/ProgramFiles/projects/data-sets
OUTPUT_PATH=${SCRIPT_PATH}/resources/data-sets/data-set.json
#INPUT_PATH=/home/vorobyev/ProgramFiles/projects/data-sets/java-files/joda-time/src/main/java
#OUTPUT_PATH=${SCRIPT_PATH}/resources/data-sets/joda-time-v2.json

java -jar ${CLASS_PATH} --input=${INPUT_PATH} --output=${OUTPUT_PATH}
