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



### Datasets

In order to reproduce the results, you need to download the [MNIST](http://yann.lecun.com/exdb/mnist/) and the [SVHN](http://ufldl.stanford.edu/housenumbers/) datset.
In the **Train.py** file uncomment line #22 to store the results the first time you use the dataset
