"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src
@author: Franziska Schirrmacher
"""

from datasets import Dataset
from utils.config import get_dicts_test
from utils.tester import Tester
from keras import backend as K
import os

#create a dataset using the parsed arguments related to data
data = get_dicts_test()
dataset = Dataset(data=data)

# preload the hdf5 files for faster access
dataset.load_files()

# load the data
X_lr_test, X_hr_test, l_test = dataset.load_data(mode='val', num=10)


job_dir = os.path.abspath("..//checkpoints")

# Test specific folder
dir = f"{data['dataset']}"
dirs = ["s2_wdsr_split4"]
# Test all folders in the subfolder
# dir = f"{data['dataset']}\\dropout"
# dirs = os.listdir(os.path.join(job_dir,dir))

for i in range(len(dirs)):
    print(dirs[i])
    dir_curr = os.path.join(job_dir, dir, dirs[i])
    tester = Tester(folder=dir_curr, epoch=-1, data=data)
    tester.load_pretrained()
    tester.test(X_lr_test=X_lr_test, X_hr_test=X_hr_test, l_test=l_test)
    tester.show_results(X_lr_test=X_lr_test[0:2, ], X_hr_test=X_hr_test[0:2, ], l_test=l_test[0:2, ],
                        name=f"prediction_{data['noiseType']}_{data['noise_low']}")
    del tester
    K.clear_session()
