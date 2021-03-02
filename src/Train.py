"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src
@author: Franziska Schirrmacher
"""

from utils.datasets import Dataset
from utils.config import get_dicts_train
from architecture.model import BuildModel
from utils.data_generator import DataGenerator
from utils.trainer import Trainer
from keras.losses import mean_absolute_error, categorical_crossentropy
from utils.metric import ssim
from keras.metrics import categorical_accuracy
from keras import backend as K

# get all the parsed arguments in dictionaries
data,parameter,design,store,training = get_dicts_train()

#define loss functions and metrics for the classification task depending on the dataset that is used
loss_cl = {"svhn":categorical_crossentropy,'mnist':categorical_crossentropy}
metric_cl = {"svhn":categorical_accuracy,'mnist':categorical_accuracy}

#create a dataset using the parsed arguments related to data
dataset = Dataset(data=data)

# store the dataset if necessary
# dataset.store_data()

# preload the hdf5 files for faster access
dataset.load_files()

# load the data
X_lr_val,X_hr_val,l_val = dataset.load_data(mode='val',num=-1)
# you can either load the data before training or with a generator
# In line 54/55 its specified whether a generator or a dataset is used

# X_lr_train, X_hr_train,l_train = dataset.load_data(mode='train',num=-1)
generator = DataGenerator(dataset=dataset,training=training,model_type=design["model_type"])

# Build the model
builder = BuildModel(design=design,data=data,parameter=parameter)
model = builder.setup_model()

# setup training
trainer = Trainer(training=training,model=model,store=store)

# choose loss functions
loss_weights = [ K.variable(value=parameter["w_sr"], name='weight_sr'),
                 K.variable(value=parameter["w_cl"], name='weight_cl')]

loss = {'sr': mean_absolute_error, 'cl': loss_cl[data["dataset"]]}
metric = {'sr': ssim, 'cl': metric_cl[data["dataset"]]}

# set callbacks:
trainer.set_callbacks(["store","tb"])

# compile the model
trainer.compile_model(loss=loss,loss_weights=loss_weights,metric=metric)

# train the model

# trainer.train(X_lr_train, X_hr_train,l_train,X_lr_val, X_hr_val,l_val)
trainer.train_generator(generator=generator,X_lr_val=X_lr_val, X_hr_val=X_hr_val,l_val=l_val)
