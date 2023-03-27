#!/bin/bash
JOBDIR="./runs/base_trains"
CONFIGS_DIR="./configs/base_trains"

for i in $(ls $CONFIGS_DIR | grep .json); do
    CONFIG_FILE="$CONFIGS_DIR/$i"
    ID="${i%.*}"
    echo "Running $CONFIG_FILE"
    python allrank/main.py --config-file-name $CONFIG_FILE --run-id $ID --job-dir $JOBDIR
done