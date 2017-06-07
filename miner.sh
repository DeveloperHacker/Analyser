#!/usr/bin/env bash

SCRIPT_PATH=`dirname $0`
CLASS_PATH=/home/vorobyev/ProgramFiles/projects/java/searcher/target/miner-jar-with-dependencies.jar
#DATA_SETS_PATH=/home/vorobyev/ProgramFiles/projects/data-sets/java-files/joda-time/src/main/java
#DATA_SETS_PATH=/home/vorobyev/ProgramFiles/projects/data-sets/java-sources
DATA_SETS_PATH=/home/vorobyev/ProgramFiles/projects/data-sets
XML=${SCRIPT_PATH}/resources/data-sets/methods.xml

java -cp .:${CLASS_PATH} Main --dir=${DATA_SETS_PATH} --xml=${XML}
