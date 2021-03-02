"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/architecture
@author: Franziska Schirrmacher
"""

from keras import Input
from architecture.networks import common_block_res,common_block_conv,common_fsrcnn,\
    fsrcnn_red,wdsr,fsrcnn,resnet_mine_red
from keras.models import Model

# Class to build the model
class BuildModel:
    def __init__(self,data,parameter,design):
        self.in_shape = data["in_shape"]
        self.n_channels = data["n_channels"]
        self.n_classes = data["n_classes"]
        self.scale = data["scale"]
        self.reg_strength = parameter["reg_strength"]
        self.n_filter = parameter["n_filter"]
        self.kernel_size = parameter["kernel_size"]
        self.cl_net = design["cl_net"]
        self.sr_net = design["sr_net"]
        self.model_type =  design["model_type"]
        self.common = design["common"]
        self.split = design["split"]
        self.n_res_blocks = design["n_res_blocks"]

    # build the model (sr, cl, or sr2)
    def setup_model(self):
        # First the common layer
        x_in = Input(shape=(self.in_shape[0],self.in_shape[1],self.n_channels))


        if self.model_type == "cl":
            x = self.setup_common(x_in)
            x_cl = self.setup_cl(x)
            return Model(inputs=x_in, outputs=x_cl, name="cl")
        elif self.model_type == "sr":
            x = self.setup_common(x_in)
            x_sr = self.setup_sr(x)
            return Model(inputs=x_in, outputs=x_sr, name="sr")
        elif self.model_type == 'seq':
            x = self.setup_common(x_in)
            x_sr = self.setup_sr(x)
            x_cl = self.setup_cl(x_sr)
            return Model(inputs=x_in, outputs=x_cl, name="seq")

        x = self.setup_common(x_in)
        x_sr = self.setup_sr(x)
        x_cl = self.setup_cl(x)
        return Model(inputs = x_in, outputs = [x_sr, x_cl], name="sr2")

    # build the shared layer, which can either be conv layers or residual blocks
    def setup_common(self,x):
        if self.common == "res":
            return common_block_res(x,split=self.split,n_filter=self.n_filter,
                                    kernel_size=self.kernel_size)
        elif self.common == "conv":
            if self.sr_net == "fsrcnn":
                return common_fsrcnn(x,reg_strength=self.reg_strength)
            else:
                return common_block_conv(x,split=self.split,n_filter=self.n_filter,
                                     kernel_size=self.kernel_size,reg_strength=self.reg_strength)
        return x

    # build the super-resolution part, which can either be WDSR or FSRCNN
    #You can call your own super-resolution network here
    # To do this, additionally update the parse.py file in line 24 to allow more choices for the sr-net argument
    def setup_sr(self,x):
        if self.sr_net == "fsrcnn":
            if self.common == "conv":
                return fsrcnn_red(x,scale=self.scale,n_channels=self.n_channels,reg_strength=self.reg_strength)
            else:
                return fsrcnn(x,scale=self.scale,n_channels=self.n_channels, reg_strength=self.reg_strength)
        elif self.sr_net == "wdsr":
            return wdsr(x,scale=self.scale,n_res_blocks=self.n_res_blocks,n_channels=self.n_channels,
                        n_filter=self.n_filter,reg_strength=self.reg_strength)
        return x

    # Currently, only a reduced Version of Res-Net is implemented. You can call your own classification network here
    # To do this, additionally update the parse.py file in line 31 to allow more choices for the cl-net argument
    def setup_cl(self,x):
        if self.cl_net == "res":
            return resnet_mine_red(x,n_classes=self.n_classes,reg_strength=self.reg_strength)
        return x


