"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/utils
@author: Franziska Schirrmacher
"""

from utils.parse import args
import os
import json

def get_dicts_test():
    data = {}
    data["dataset"] = args.dataset
    data["scale"] = args.scale
    data["noise"] = args.noise
    data["noiseType"] = args.noiseType
    data["noise_low"] = args.noiseLow
    data["noise_high"] = args.noiseHigh
    data["local_dir"] = args.dataDir
    data["n_images"] = args.num_images

    # dataset specific information
    if data["dataset"] == "mnist":
        data["in_shape"] = (int(28 / args.scale), int(28 / args.scale))
        data["out_shape"] = (28, 28)
        data["n_channels"] = 1
        data["n_classes"] = 10
    elif data["dataset"] == "svhn":
        data["in_shape"] = (int(32 / args.scale), int(32 / args.scale))
        data["out_shape"] = (32, 32)
        data["n_channels"] = 3
        data["n_classes"] = 10
    elif data["dataset"] == "czech":
        data["in_shape"] = (int(120 / args.scale), int(520 / args.scale))
        data["out_shape"] = (120, 520)
        data["n_channels"] = 1
        data["n_classes"] = 7 * 37

    return data

def get_dicts_train():
    # create dictionary for all information regarding the data!
    data = get_dicts_test()

    # create dictionary for all information regarding the hyperparameter of the network
    parameter = {}
    parameter["reg_strength"] = args.reg_strength
    parameter["n_filter"] = args.num_filters
    parameter["kernel_size"] = 3
    parameter["w_sr"] = args.weight_sr
    parameter["w_cl"] = args.weight_cl


    # create dictionary for the information regarding the design of the network
    design = {}
    design["model_type"] = args.model
    design["cl_net"] = args.cl_net
    design["sr_net"] = args.sr_net
    design["common"] = args.common_block
    design["split"] = args.split
    design["n_res_blocks"] = args.num_res_blocks

    # create dictionary for the information regarding the training setup
    training = {}
    training["n_batch"] = args.batch_size
    training["lr"] = args.learning_rate
    training["epochs"] = args.epochs
    training["epoch_init"] = args.initial_epoch
    training["lr_steps"] = args.learning_rate_step_size
    training["lr_decay"] = args.learning_rate_decay
    training["monitor"] = 'val_loss'
    training["patience"] = args.patience
    training["lr_min"] = args.min_lr
    training["shuffle"] = True

    # create dictionary for all storing stuff
    store = {}
    store["pretrained_dir"] = os.path.join(args.job_dir,args.dataset,args.pretrained_model)
    store["period"] = args.period
    store["job_dir"] = args.job_dir
    store["dir"] = os.path.join(args.job_dir,args.dataset,args.name)
    store["best"] = args.save_best_models_only


    total = {}
    total["data"] = data
    total["parameter"] = parameter
    total["design"] = design
    total["store"] = store
    total["training"] = training

    # Store the parameter
    if not os.path.isdir(store["dir"]):
        os.makedirs(store["dir"], exist_ok=True)
    json.dump( total, open( "{0}/dict.json".format(store["dir"]), 'w' ))

    return data,parameter,design,store,training

