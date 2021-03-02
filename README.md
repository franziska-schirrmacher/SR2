# SR²

This repository contains the implementation of the paper

F. Schirrmacher, B. Lorch, B. Stimpel, T. Köhler and, C. Riess, "SR²: Super-Resolution With Structure-Aware Reconstruction," in IEEE International Conference on Image Processing (ICIP), pp. 533-537, 2020, doi: 10.1109/ICIP40778.2020.9191253. [IEEE](https://ieeexplore.ieee.org/abstract/document/9191253) 

If you use this code in your work, please cite:

        @INPROCEEDINGS{9191253,
            author={F. {Schirrmacher} and B. {Lorch} and B. {Stimpel} and T. {Köhler} and C. {Riess}},
            booktitle={2020 IEEE International Conference on Image Processing (ICIP)}, 
            title={SR2: Super-Resolution With Structure-Aware Reconstruction},
            year={2020},
            pages={533--537},
            doi={10.1109/ICIP40778.2020.9191253}}

## Getting started

To download the code, fork the repository or clone it using the following command:

```
  git clone https://github.com/franziska-schirrmacher/sr2.git
```

### Requirements

- python 3.7
- keras 2.3.1
- tensorflow 1.14
- h5py 2.10.0


### Code structure

- **checkpoints**: This folder contains the stored weights (needs to be created)

- **data**: This folder contains all datasets for each of the experiments (needs to be created)

- **src**: This folder contains the source code of the experiments and the proposed architectur

    - **architecture**: This folder contains the model and the network components
    - **utils**: Here are all the helper functions for training and testing


### Datasets

In order to reproduce the results, you need to download the [MNIST](http://yann.lecun.com/exdb/mnist/) and the [SVHN](http://ufldl.stanford.edu/housenumbers/) datset.
In the Train.py file uncomment line #22 to store the results the first time you use the dataset

## Experiments

The bash scripts to train all models described in the experiments are provided. In general, you can use the Train.py file to train a network. The utils/parse.py file contains all the possible arguments you can use.
Additionally, the models can either be trained with a generator (line #40 and #65) or by loading all training images at once (line #39 and #64)

### Pretrain super-resolution networks 

```
  ./pretrainSR.sh
```

### Comparison of Architectures

```
  ./runExp1.sh
```

### Influence of the Classification-driven Regularization

```
  ./runExp2.sh
```

###  Influence of the Number of Shared Layers

```
  ./runExp3s.sh
```

### Testing the models
In the Test.py file you can specifiy the model or a folder containing multiple models that you want to evaluate. The arguments that can be passed to the test environment are stated in the utils.config.py file in the get_dicts_test() function in line 5. With noiseLow and noiseHigh you can specify a range of noise levels. For each sample, a value is randomly drawn from a uniform distribution between those two values. The results will be stored in a json file with multiple dictionaries. The name of the experiment contains the noise type and the lower noise level (changes can be done in Test.py line #41).

Example:

```
  ./runEval.sh
```

