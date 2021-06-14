# ROLAND: Graph Neural Networks for Dynamic Graphs
This repository contains code associated with the ROLAND project and more.
You can firstly walk through the *how-to* sections to run experiments on existing
public datasets.
After understanding how to run and analyze experiments, you can read through the *development topics* to run our 


## TODO: add figures to illustrate the ROLAND framework.

## How to Download Datasets
Most of datasets are used in our paper can be found at `https://snap.stanford.edu/data/index.html`.

```bash
# Or Use your own dataset directory.
mkdir ./all_datasets/
cd ./all_datasets
wget 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'
wget 'https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz'
wget 'https://snap.stanford.edu/data/as-733.tar.gz'
wget 'https://snap.stanford.edu/data/CollegeMsg.txt.gz'
wget 'https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv'
wget 'https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv'
wget 'http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv'

# Unzip files
gunzip CollegeMsg.txt.gz
gunzip soc-sign-bitcoinalpha.csv.gz
gunzip soc-sign-bitcoinotc.csv.gz
tar xf ./as-733.tar.gz

# Rename files, this step is required by our loader.
# You can leave the web-redditEmbeddings-subreddits.csv file unchanged.
mv ./soc-sign-bitcoinotc.csv ./bitcoinotc.csv
mv ./soc-sign-bitcoinalpha.csv ./bitcoinalpha.csv

mv ./soc-redditHyperlinks-body.tsv ./reddit-body.tsv
mv ./soc-redditHyperlinks-title.tsv ./reddit-title.tsv
```
You should expect 740 files, including the zipped `as-733.tar.gz`, by checking `ls | wc -l`.
The total disk space required is approximately 950MiB.
## How to Run Single Experiments from Our Paper
**WARNING**: for each `yaml` file in `./run/configs/ROLAND`, you need to update the `dataset.dir` field to the correct path of datasets downloaded above.

The ROLAND project focuses on link-predictions for homogenous dynamic graphs.
Here we demonstrate example runs using 

To run link-prediction task on `CollegeMsg.txt` dataset with default settings:
```bash
cd ./run
python3 main_dynamic.py --cfg configs/ROLAND/roland_gru_ucimsg.yaml --repeat 1
```
For other datasets:
```bash
python3 main_dynamic.py --cfg configs/ROLAND/roland_gru_btcalpha.yaml --repeat 1

python3 main_dynamic.py --cfg configs/ROLAND/roland_gru_btcotc.yaml --repeat 1

python3 main_dynamic.py --cfg configs/ROLAND/roland_gru_ucimsg.yaml --repeat 1

python3 main_dynamic.py --cfg configs/ROLAND/roland_gru_reddittitle.yaml --repeat 1

python3 main_dynamic.py --cfg configs/ROLAND/roland_gru_redditbody.yaml --repeat 1
```
The `--repeat` argument controls for number of random seeds used for each experiment. For example, setting `--repeat 3` runs each single experiments for three times with three different random seeds.

To explore training result:
```bash
cd ./run
tensorboard --logdir=./runs_live_update --port=6006
```
**WARNING** The x-axis of plots in tensorboard is **not** epochs, they are snapshot IDs (e.g., the $i^{th}$ day or the $i^{th}$ week) instead.

## Examples on Heterogenous Graph Snapshots
```bash
Under development.
```

## How to Run Grid Search / Batch Experiments
To run grid search / batch experiments, one needs a `main.py` file, a `base_config.yaml`, and a `grid.txt` file. The main and config files are the same as in the single experiment setup above.
If one wants to do link-prediction on `CollegeMsg.txt` dataset with configurations from  `configs/ROLAND/roland_gru_ucimsg.yaml`, in addition, she wants to try out (1) *different numbers of GNN message passing layers* and (2) *different learning rates*.
In this case, one can use the following grid file:
```text
# grid.txt, lines starting with # are comments.
gnn.layers_mp mp [2,3,4,5]
optim.base_lr lr [0.003,0.01,0.03]
```
**WARNING**: the format of each line is crucial: `NAME_IN_YAML<space>SHORT_ALIAS<space>LIST_OF_VALUES`, and there should **not** be any space in the list of values.

The `grid.txt` above will generate $4\times 3=12$ different configurations by modifying `gnn.layers_mp` and `gnn.layers_mp` to the respective levels in base config file `roland_gru_ucimsg.yaml`.

Please see `./run/grids/ROLAND/example_grid.txt` for a complete example of grid search text file.

To run the experiment using `example_grid.txt`:
```bash
bash ./run_roland_batch.sh
```
## How to Export Tensorboard Results to CSV
We provide a simple script to aggregate results from a batch of tensorboard files, please feel free to look into `tabulate_events.py` and modify it.
```bash
# Usage: python3 ./tabulate_events.py <tensorboard_logdir> <output_file_name>
python3 ./tabulate_events.py ./live_update ./out.csv
```

## Development Topic: Use Your Own Dataset
We provided two examples of constructing your own datasets, please refer to
(1) `./graphgym/contrib/loader/roland_template.py` and (2) `./graphgym/contrib/loader/roland_template_hetero.py` for examples of building loaders.
