#!/usr/bin/env bash

CONFIG=roland_gru_ucimsg
GRID=example_grid
REPEAT=3
MAX_JOBS=10
SLEEP=1

python configs_gen.py --config configs/ROLAND/${CONFIG}.yaml \
  --grid grids/ROLAND/${GRID}.txt \
  --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP

# aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID}
