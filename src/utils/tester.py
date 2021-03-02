"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/utils
@author: Franziska Schirrmacher
"""

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, multilabel_confusion_matrix
import numpy as np
from skimage.measure import compare_ssim as ssim
from statistics import mean, stdev
import json
from architecture.model import BuildModel
import datetime
import os
import matplotlib.pyplot as plt
import csv

# Class which has all functions related to testing the trained models
class Tester:
    def __init__(self,folder,epoch,data):
        # folder: specifies the location of the trained model
        # epoch: specifies the epoch of the trained model that is loaded
        self.folder = folder
        self.epoch = epoch
        self.data = data
        self.key = f"{data['noiseType']}_{data['noise_low']}_{data['noise_high']}"
        self.dict = {}
        self.gen_logs = {}
        self.gen_logs["last_modified"] = self.timestamp()
        self.gen_logs[self.key] = {}
        self.gen_logs_file = os.path.join(self.folder, "results.json")
        with open(gen_logs_file, "w") as file:
            json.dump(self.gen_logs, file)


    # Function to load the trained model
    def load_pretrained(self):
        # load the information regarding the network from the json file
        total = json.load(open( os.path.join(self.folder,"dict.json"), 'r' ))
        self.store = total["store"]
        self.cl_net = total["design"]["cl_net"]
        # get the name of the pretrained model
        model_path = os.path.join(self.folder,"models")
        if self.epoch > 0:
            model_idx = 'epoch-{0:03d}.h5'.format(self.epoch)
        else:
            # if no epoch is give, get the last one
            model_idx = os.listdir(model_path)[-1]
        # build the model
        builder = BuildModel(data=total["data"], parameter=total["parameter"], design=total["design"])
        self.model = builder.setup_model()
        # load the pretrained weights
        self.model.load_weights(os.path.join(model_path, model_idx), by_name=True)

    # Function to compute SSIM of a batch of images
    def batch_ssim(self,X_hr_test, X_hr_pred):
        # _test: ground truth images
        # _pred: prediction of the trained network
        assert (X_hr_pred.shape == X_hr_test.shape)
        # store results
        ssim_all = []
        for idx in range(X_hr_pred.shape[0]):
            if X_hr_pred.shape[-1] == 3:
                s = ssim(X_hr_pred[idx].astype(np.float32), X_hr_test[idx], multichannel=True)
                ssim_all.append(s)

            else:
                s = ssim(X_hr_pred[idx,:,:,0].astype(np.float32), X_hr_test[idx,:,:,0])
                ssim_all.append(s)
        # Compute mean and std
        ssim_all_m = mean(ssim_all)
        ssim_all_s = stdev(ssim_all)
        print("SSIM {0}".format(ssim_all_m))

        # Store the results in member variables
        self.ssim_all_m = ssim_all_m
        self.ssim_all_s = ssim_all_s

    def acc_from_conf(self,matrix):
        return (matrix[0, 0] + matrix[1, 1]) / (matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1])

    # Evaluate the classification task
    def evaluate_cl(self,l_test,l_pred):
        # _test: ground truth label
        # _pred: prediction of the network

        # Compute accuracy
        acc = accuracy_score(l_test, l_pred)
        print("accuracy {0}".format(acc))

        # Compute F1-score
        f1= f1_score(l_test, l_pred, average='weighted')
        print("f1 {0}".format(f1))

        # Store the results in member variables
        self.acc_w = acc
        self.f1_w = f1

    # main function which is called from the Test.py file
    def test(self,X_lr_test, X_hr_test,l_test):

        # Only evaluate the classification task
        if self.model.name == "cl":
            l_pred = self.model.predict(x = X_lr_test)

            self.evaluate_cl(l_test=l_test.argmax(axis=1), l_pred=l_pred.argmax(axis=1))
            self.gen_logs[self.key]["accuracy"] = self.acc
            self.gen_logs[self.key]["f1"] = self.f1

        # Only evaluate the classification task
        elif self.model.name == "sr":

            X_hr_pred = self.model.predict(x = X_lr_test)
            self.batch_ssim(X_hr_test=X_hr_test,X_hr_pred=X_hr_pred)
            self.gen_logs[self.key]["SSIM"] = self.ssim_all_m

        else:

            X_hr_pred, l_pred = self.model.predict(x = X_lr_test)
            self.batch_ssim(X_hr_test=X_hr_test,X_hr_pred=X_hr_pred)

            self.evaluate_cl(l_test=l_test.argmax(axis=1), l_pred=l_pred.argmax(axis=1))

            self.gen_logs[self.key]["accuracy"] = self.acc
            self.gen_logs[self.key]["f1"] = self.f1
            self.gen_logs[self.key]["SSIM"] = self.ssim_all_m

        with open(self.gen_logs_file, "w") as file:
            json.dump(self.gen_logs, file)


    def show_results(self,X_lr_test, X_hr_test,l_test,name):

        if self.model.name == "sr":

            X_hr_pred = self.model.predict(x = X_lr_test)
            for i in range(X_hr_pred.shape[0]):
                plt.subplot(1,2,1)
                plt.imshow(X_hr_test[i,:,:,0],cmap="gray")
                plt.title("Ground truth")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(X_hr_pred[i, :, :, 0], cmap="gray")
                plt.title("Prediction")
                plt.axis("off")
                plt.savefig(os.path.join(self.folder, f"{name}_{i}.png"))
                plt.show(block=False)
                plt.pause(3)
                plt.close()

        elif self.model.name == "sr2":

            X_hr_pred, l_pred = self.model.predict(x = X_lr_test)

            for i in range(X_hr_pred.shape[0]):
                plt.subplot(1,2,1)
                plt.imshow(X_hr_test[i,:,:,0],cmap="gray")
                plt.title("Ground truth")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(X_hr_pred[i, :, :, 0], cmap="gray")
                plt.title("Prediction {0}".format(l_pred[i,]))
                plt.axis("off")
                plt.savefig(os.path.join(self.folder, f"{name}_{i}.png"))
                plt.show(block=False)
                plt.pause(3)
                plt.close()
    # Obtaining the current timestamp in an human-readable way
    def timestamp(self):
        timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')

        return timestamp