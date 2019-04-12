# P2C
Code for paper on "Unpaired Photo-to-Caricature Translation on Faces in the Wild" (<a href="https://arxiv.org/abs/1711.10735">arXiv:1711.10735</a>)
## Installation
1. We use [Miniconda3](https://conda.io/miniconda.html) for the basic environment. If you installed the Miniconda3 in path `Conda_Path`, please install `tensorflow-gpu` using the command `Conda_Path/bin/conda install -c anaconda tensorflow-gpu==1.8`.
2. Install dependencies by `Conda_Path/bin/pip install -r requirements.txt` (if necessary). The `requirements.txt` file is provided in this package.

## Data preparation
```
├── datasets
   └── demo
       ├── trainA
           ├── 000001.jpg (The traint image that you want, name does not matter)
           ├── 000002.jpg
           └── ...
       ├── trainB
           ├── 000001.jpg (The traint image that you want, name does not matter)
           ├── 000002.jpg
           └── ...
       ├── testA
           ├── a.jpg (The test image that you want)
           ├── b.png
           └── ...
       ├── testB
           ├── a.jpg (The test image that you want)
           ├── b.png
           └── ...
```
## Codes
```
- `base.py`: train and test model of `P2C`.
- `utils.py`: basic utils of `P2C`.
```
## train
Conda_Path/bin/python base.py --phase train --dataset_dir demo --checkpoint ./checkpoint ./checkpoints/demo --sample_dir ./checkpoints/demo/sample --epoch 120 --gpu 0

## test
Conda_Path/bin/python base.py --phase test --dataset_dir demo --checkpoint ./checkpoint ./checkpoints/demo --test_dir ./checkpoints/demo/test --gpu 0

