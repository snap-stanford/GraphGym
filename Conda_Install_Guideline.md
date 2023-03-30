# GraphGym - Simple Stable Install for MacOS and M1 laptops
GraphGym is a platform for designing and evaluating Graph Neural Networks (GNN).
GraphGym is proposed in *[Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)*, 
Jiaxuan You, Rex Ying, Jure Leskovec, **NeurIPS 2020 Spotlight**.

Please also refer to [PyG](https://www.pyg.org) for a tightly integrated version of GraphGym and PyG.

The following is a step-by-step guideline for setting up a failsafe stable version of GraphGym on any laptop operating system. We provide the instructions for MacOS 12.4 with the Apple Silicon M1 chip as a demo. If intending to run ROLAND as a use case, please complete these steps first to create the Conda Environment, and proceed with installing roland_environment_M1.yml. Happy coding! 


```bash
conda update conda

# creates new Conda environment named GraphGymM1, but feel free to name the environment anything you'd like
conda create -n GraphGymM1  python=3.9

conda activate GrapahGymM1

# This is critical for all Apple Silicon installations
conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64

# please change 12.4 to your target operating system version. This torch version has been verified to work 
MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m pip --no-cache-dir   install torch==1.11.0 torchvision torchaudio

# Check the torch version is correct, this is particularly important if running ROLAND use case
python -c "import torch; print(torch.__version__)"  #---> (Confirm the version is 1.12.1 or 1.11 for ROLAND)

# must install the following packages specifically, paying attention to version numbers. The following versions are needed to run ROLAND. 
MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11+${cpu}.html

MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.12.1+${cpu}.html

MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.11+${cpu}.html

MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-geometric

MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m pip --no-cache-dir  install matplotlib

MACOSX_DEPLOYMENT_TARGET=12.4 CC=clang CXX=clang++ python -m pip --no-cache-dir  install networkx
```

**Running ROLAND Use Case**

If running ROLAND use case, please install Conda via the roland_environment_M1.yml.

```bash
conda env create --name {your name} -f roland_environment_M1.yml
./install.sh

python setup.py install 

# paying attention to use the right pip
python -m pip {your install}
```
