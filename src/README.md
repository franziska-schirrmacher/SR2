### Code structure

- **architecture**: This folder contains the model and the network components
- **utils**: Here are all the helper functions for training and testing
- **Train.py**: Script to train a model
- **Test.py**: Script to test one or multiple models


### Datasets

In order to reproduce the results, you need to download the [MNIST](http://yann.lecun.com/exdb/mnist/) and the [SVHN](http://ufldl.stanford.edu/housenumbers/) datset.
In the **Train.py** file uncomment line #22 to store the results the first time you use the dataset

## Experiments

The bash scripts to train all models described in the experiments are provided. In general, you can use the **Train.py** file to train a network. Additionally, the models can either be trained with a generator (line #40 and #65) or by loading all training images at once (line #39 and #64). The **utils/parse.py** file contains all the possible arguments you can use.

### REMARK

The level of noise described in the paper is not the standard deviation but the variance. The code in this repository gets the standard deviation as argument. The updated version of the paper can be found [here](https://faui1-files.cs.fau.de/public/publications/mmsec/2020-Schirrmacher-SR2.pdf)

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
In the **Test.py** file you can specifiy the model or a folder containing multiple models that you want to evaluate. The arguments that can be passed to the test environment are stated in the **utils.config.py** file in the get_dicts_test() function in line 5. With noiseLow and noiseHigh you can specify a range of noise levels. For each sample, a value is randomly drawn from a uniform distribution between those two values. The results will be stored in a json file with multiple dictionaries. The name of the experiment contains the noise type and the lower noise level (changes can be done in Test.py line #41).

Example:

```
  ./runEval.sh
```

