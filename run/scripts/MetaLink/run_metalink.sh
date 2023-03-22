#!/usr/bin/env bash

cd ../..

DIR=MetaLink
CONFIG=mol_classification
GRID=basic
REPEAT=10
MAX_JOBS=8

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS

# aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID}
