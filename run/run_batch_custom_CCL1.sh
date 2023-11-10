#!/usr/bin/env bash

CONFIG=example_custom_CCL1
GRID=example_custom_CCL1
REPEAT=1
MAX_JOBS=20

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/${CONFIG}.yaml \
 --config_budget configs/${CONFIG}.yaml \
 --grid grids/${GRID}.txt \
 --out_dir configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS
# rerun missed / stopped experiments
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS

# aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID}