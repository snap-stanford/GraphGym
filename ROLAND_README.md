# Use case: ROLAND: Graph Neural Networks for Dynamic Graphs
Code associated with the ROLAND project.


## TODO: add figures to illustrate the ROLAND framework.

## Datasets
Most of datasets are used in our paper can be found at `https://snap.stanford.edu/data/index.html`.

```bash
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

# Rename files.
mv ./soc-sign-bitcoinotc.csv ./bitcoinotc.csv
mv ./soc-sign-bitcoinalpha.csv ./bitcoinalpha.csv

mv ./soc-redditHyperlinks-body.tsv ./reddit-body.tsv
mv ./soc-redditHyperlinks-title.tsv ./reddit-title.tsv
```
## Examples of ROLAND Use Cases
The ROLAND project focuses on link-predictions for homogenous dynamic graphs.
To run link-prediction task on `CollegeMsg.txt` dataset:
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

To explore training result:
```bash
cd ./run
tensorboard --logdir=./runs_live_update --port=6006
```

## Examples on Heterogenous Graph Snapshots
`Under development`

## How to Load Your Own Dataset
Please refer to `./graphgym/contrib/loader/roland_template.py` and `./graphgym/contrib/loader/roland_template_hetero.py` for examples of building loaders.
