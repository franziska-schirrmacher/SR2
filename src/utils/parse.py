"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/utils
@author: Franziska Schirrmacher
"""

import argparse

parser = argparse.ArgumentParser(description='SR2: Super-Resolution With Structure-Aware Reconstruction')

# Add Arguments specific to models, training, testing and so on
## Architecture
parser.add_argument(
    '--model',
    default='sr2',
    choices=['cl', 'sr', 'seq', 'sr2'],
    help='Choose whether you want to perform classification only (cl), super-resolution only (sr), '
         'sequential approach (seq), or the proposed sr2',
    type=str)
parser.add_argument(
    '--sr-net',
    help='network to choose for the super-resolution part',
    choices = ['wdsr','fsrcnn'],
    default='wdsr',
    type=str
)
parser.add_argument(
    '--cl-net',
    help='network to choose for the classification part',
    choices = ['res'],
    default='res',
    type=str
)
parser.add_argument(
    '--split',
    help='number of res blocks after which we split into SR and Cl task',
    default=4,
    type=int
)
parser.add_argument(
    '--num-filters',
    type=int,
    default=32,
    help='number of filters in the conv layers of the residual blocks')
parser.add_argument(
    '--num-res-blocks',
    type=int,
    default=16,
    help='number of residual blocks')
parser.add_argument(
    '--reg-strength',
    type=float,
    default= 0.01,
    help='L2 regularization of kernel weights'
)
parser.add_argument(
    '--common-block',
    type=str,
    default='res',
    choices = ['res', 'conv'],
    help='which basic block should be used in the common layers of the network'
)


## Training
parser.add_argument(
    '--batch-size',
    help='Batch size for training steps',
    type=int,
    default=100)
parser.add_argument(
    '--epochs',
    type=int,
    default=120,
    help='number of epochs to train')
parser.add_argument(
    '--learning-rate',
    type=float,
    default=0.001,
    help='learning rate')

# for the learning rate callback
parser.add_argument(
    '--learning-rate-step-size',
    type=int,
    default=30,
    help='learning rate step size in epochs -> learning rate decay at the end')
parser.add_argument(
    '--learning-rate-decay',
    type=float,
    default=0.2,
    help='learning rate decay for reduceLROnPlateu')
parser.add_argument(
    '--patience',
    type=int,
    default=5,
    help='when validation loss increases, after patience epochs the learning rate is divided by 10')
parser.add_argument(
    '--min-lr',
    type=float,
    default=0.0000001,
    help='lr does not go beyond min-lr')
parser.add_argument(
    '--pretrained-model',
    type=str,
    help='path to pre-trained model',
    default = "")


## Arguments for the datasets
parser.add_argument(
    '--dataset',
    help='decide on which dataset you want to train/test you network',
    type=str,
    choices = ['mnist','svhn'],
    default='czech')
parser.add_argument(
    '--dataDir',
    help='where is your data stored',
    type=str,
    default='../data')
parser.add_argument(
    '--num-images',
    help = 'number of images we use for training',
    type=int,
    default=604388)
parser.add_argument(
    '--scale',
    help='Magnification factor for image super-resolution',
    default=8,
    type=int)
parser.add_argument(
    '--noise',
    help='Add noise to training/validation data',
    action='store_true')
parser.add_argument(
    '--noiseType',
    help = 'type of noise',
    type=str,
    choices=['gaussian', 'sp', 'speckle'],
    default="gaussian")
parser.add_argument(
    '--noiseLow',
    help='Lower bound for the noise',
    type=float,
    default=0.0001)
parser.add_argument(
    '--noiseHigh',
    help='Upper bound for the noise',
    type=float,
    default=0.1)



## Multi-task stuff
parser.add_argument(
    '--weight-sr',
    help='weight of the sr_loss in the total loss',
    default=0.1,
    type=float
)
parser.add_argument(
    '--weight-cl',
    help='weight of the classification_loss in the total loss',
    default=0.9,
    type=float
)

## Storing
parser.add_argument(
    '--period',
    help='the model will be stored after every period-th epoch',
    default= 5,
    type=int
)
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    default= '../checkpoints',
    required=False)
parser.add_argument(
    '--name',
    default='test_stuff',
    help='name of the folder where data is stored')

parser.add_argument(
    '--save-best-models-only',
    action='store_true',
    help='save only models with improved validation psnr (overridden by --benchmark)'
)
parser.add_argument(
    '--save-models-after-epoch',
    type=int,
    default=0,
    help='start saving models only after given epoch')
parser.add_argument(
    '--initial-epoch',
    type=int,
    default=0,
    help='resumes training of provided model if greater than 0')


args = parser.parse_args()

assert (args.split >= 0), "Splits must be a positive integer (including 0)"
assert (args.initial_epoch >= 0 ), "Initial epoch needs to be at least 0"
assert (args.num_filters > 0), "Network needs more than one filter"
assert (args.num_res_blocks > 0), "Network needs more than one res block"
assert (args.reg_strength > 0), "Regularization strength needs to be larger than 0"
assert (args.batch_size > 0), "Batch size needs to be larger than 0"
assert (args.initial_epoch < args.epochs), "Initial epoch is higher than maximum number of epochs"
assert (args.period < args.epochs), "Period to store the checkpoints is higher " \
                                    "than maximum number of epochs"
assert (args.scale % 2 == 0 and args.scale <=8 and args.scale > 0), "Even number for scale"
assert (args.noiseLow < args.noiseHigh), "Lower bound of the noise is higher than the higher bound"
