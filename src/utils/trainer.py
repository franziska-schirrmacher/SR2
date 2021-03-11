"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/utils
@author: Franziska Schirrmacher
"""

from utils.callback import ModelCheckpointAfter
from keras.callbacks import  TensorBoard,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam
import os


class Trainer:
    def __init__(self,training,model,store):
        # Dictionaries with all the parameter regarding training and model setup
        self.training = training
        self.model = model
        # List of callbacks that are used
        self.callbacks = []
        # Temporary dict that stores all possible callbacks
        self.callbacks_all = {}
        # Dictionary with all the parameter regarding the storage of the model
        self.store = store
        # Create the workspace
        self.create_train_workspace()
        # Set the list of callbacks
        self.def_callbacks()


    # Function to create the workspace (where the model and the json files are stored)
    def create_train_workspace(self):
        # Create the directory of the experiment
        os.makedirs(self.store["dir"], exist_ok=True)
        # Stores the models
        models_dir = os.path.join(self.store["dir"], 'models')
        # For the tensorboard file
        log_dir = os.path.join(self.store["dir"], 'log')

        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.models_dir = models_dir
        self.log_dir = log_dir

    # Compile the model
    def compile_model(self,loss,loss_weights,metric):
        if self.model.name == "cl":
            self.model.compile(optimizer=Adam(lr=self.training["lr"]),
                               loss = loss['cl'],
                               metrics = [metric['cl']])
        elif self.model.name == "sr":
            self.model.compile(optimizer=Adam(lr=self.training["lr"]),
                               loss = loss['sr'],
                               metrics = [metric['sr']])
        else:
            self.model.compile(optimizer=Adam(lr=self.training["lr"]),
                               loss = loss,
                               loss_weights = loss_weights,
                               metrics = metric)

    def set_callbacks(self,list):
        for i in range(len(list)):
            self.callbacks.append(self.callbacks_all[list[i]])
        del self.callbacks_all

    def def_callbacks(self):
        self.callbacks_all["tb"] = TensorBoard(log_dir=self.log_dir,
                                               write_graph=False, write_grads=False )
        self.callbacks_all["reduceLR"] = ReduceLROnPlateau(monitor=self.training["monitor"],
                                                           patience=self.training["patience"],
                                                           factor=self.training["lr_decay"],
                                                           min_lr=self.training["lr_min"])
        self.callbacks_all["earlyStop"] = EarlyStopping(monitor=self.training["monitor"],
                                                        patience= self.training["patience"],
                                                        mode='min')
        p = self.store["period"]
        if self.store["best"]:
            p = 1


        self.callbacks_all["store"] = ModelCheckpointAfter(0, filepath=os.path.join(self.models_dir, 'epoch-best.h5'),
                                                           monitor=self.training["monitor"],
                                                           save_best_only=self.store["best"],
                                                           mode='min', period=p,
                                                           save_weights_only=True)


    def train_generator(self,generator, X_lr_val, X_hr_val,l_val):
        if self.model.name == "cl":
            self.model.fit_generator(generator=generator,
                           epochs=self.training["epochs"],
                           initial_epoch=self.training["epoch_init"],
                           validation_data=(X_lr_val, l_val),
                           verbose=1,
                           shuffle=True,
                           callbacks=self.callbacks)
        elif self.model.name == "sr":
            self.model.fit_generator(generator=generator,
                           epochs=self.training["epochs"],
                           initial_epoch=self.training["epoch_init"],
                           validation_data=(X_lr_val, X_hr_val),
                           verbose=1,
                           shuffle=True,
                           callbacks=self.callbacks)
        else:

            self.model.fit_generator(generator=generator,
                            epochs=self.training["epochs"],
                            initial_epoch=self.training["epoch_init"],
                            validation_data=(X_lr_val, [X_hr_val, l_val]),
                            verbose=1,
                            shuffle=True,
                            callbacks=self.callbacks)

    def train(self,X_lr_train, X_hr_train,l_train,X_lr_val, X_hr_val,l_val):
        if self.model.name == "cl":
            self.model.fit(x = X_lr_train, y = l_train,
                        epochs=self.training["epochs"],
                        initial_epoch=self.training["epoch_init"] ,
                        batch_size= self.training["n_batch"],
                        validation_data= (X_lr_val,l_val),
                        verbose = 2,
                        shuffle=True,
                        callbacks=self.callbacks)
        elif self.model.name == "sr":
            self.model.fit(x = X_lr_train, y = X_hr_train,
                        epochs=self.training["epochs"],
                        initial_epoch=self.training["epoch_init"] ,
                        batch_size= self.training["n_batch"],
                        validation_data= (X_lr_val,X_hr_val),
                        verbose = 2,
                        shuffle=True,
                        callbacks=self.callbacks)
        else:
            self.model.fit(x = X_lr_train, y = [X_hr_train,l_train],
                        epochs=self.training["epochs"],
                        initial_epoch=self.training["epoch_init"] ,
                        batch_size= self.training["n_batch"],
                        validation_data= (X_lr_val,[X_hr_val,l_val]),
                        verbose = 2,
                        shuffle=True,
                        callbacks=self.callbacks)


