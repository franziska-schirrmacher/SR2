"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/architecture
@author: Franziska Schirrmacher
"""

import tensorflow as tf
from keras.initializers import he_normal
from keras import regularizers
from keras.layers import Dense, Conv2D, BatchNormalization, Lambda, PReLU, GlobalAveragePooling2D, \
    Add, MaxPooling2D

## Code based on https://github.com/krasserm/super-resolution
## Here you can also add your own architecture which needs to be called in model.py

###

# This file contains all the network architecturs which are called in the architecture/model.py file

###

def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.compat.v1.depth_to_space(x, scale), **kwargs)

## Basic residual blocks from https://arxiv.org/pdf/1603.05027
def block_res(inputs,num_identity, n_filter=16, kernel_size=3,reg_strength=0.001):
    x = inputs
    for i in range(num_identity):
        x = res_identity(x,n_filter=n_filter,kernel_size=kernel_size,reg_strength=reg_strength,name='sr_i{0}'.format(i))
    return x

def res_identity_last(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test'):
    x = inputs
    x = BatchNormalization(name='{0}_bn1'.format(name))(x)
    x = PReLU(alpha_initializer=he_normal(),name='{0}_prelu1'.format(name))(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='{0}_bn2'.format(name))(x)
    x = PReLU(alpha_initializer=he_normal(),
              name='{0}_prelu2'.format(name))(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength),
               name='last_common')(x)


    return Add(name='{0}_add'.format(name))([x, inputs])



def res_identity(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test'):
    x = inputs
    x = BatchNormalization(name='{0}_bn1'.format(name))(x)
    x = PReLU(alpha_initializer=he_normal(),name='{0}_p1'.format(name))(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='{0}_bn2'.format(name))(x)
    x = PReLU(alpha_initializer=he_normal(),name='{0}_p2'.format(name))(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_conv2'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    return Add(name='{0}_add'.format(name))([x, inputs])


def res_down(inputs, n_filter=16, kernel_size=3,reg_strength=0.001,name='test'):

    x = inputs
    x = BatchNormalization(name='{0}_large_bn1'.format(name))(x)
    x = PReLU(alpha_initializer=he_normal(),name='{0}_large_p1'.format(name))(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=2,
                  padding='same', name='{0}_large_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='{0}_large_bn2'.format(name))(x)
    x = PReLU(alpha_initializer=he_normal(),name='{0}_large_p2'.format(name))(x)
    x = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=1,
                  padding='same',name='{0}_large_conv2'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)


    y = inputs
    y = BatchNormalization(name='{0}_small_bn1'.format(name))(y)
    y = PReLU(alpha_initializer=he_normal(),name='{0}_small_p1'.format(name))(y)
    y = Conv2D(n_filter,
                  kernel_size=kernel_size,strides=2,
                  padding='same',name='{0}_small_conv1'.format(name),
                  kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(y)
    return Add(name='{0}_add'.format(name))([x, y])

## All possible common layers

def common_fsrcnn(inputs,reg_strength=0.001):
    x = inputs
    x = Conv2D(56,kernel_size=5,strides=1,padding='same',name='last_common',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='common_bn')(x)
    x = PReLU(alpha_initializer=he_normal(), name='common_p')(x)

    return x

def common_block_res(inputs,split, n_filter=16, kernel_size=3,reg_strength=0.001):

    x = inputs
    if split > 0:
        x = Conv2D(n_filter, 3, padding='same', kernel_initializer=he_normal())(x)
    for i in range(split-1):
        x = res_identity(x,n_filter=n_filter,
                             kernel_size=kernel_size,reg_strength=reg_strength,
                             name='common_res_i{0}'.format(i))
    if split > 0:
        x = res_identity_last(x,n_filter=n_filter,
                              kernel_size=kernel_size,reg_strength=reg_strength,
                              name='common_res_last_i{0}'.format(split-1))


    return x

def common_block_conv(inputs,split=4,n_filter=16,kernel_size=3,reg_strength=0.001):
    x = inputs
    for i in range(split-1):
        x = Conv2D(n_filter, kernel_size=kernel_size,padding='same',
                      kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength),name='common_conv_conv{0}'.format(i))(x)

        x = BatchNormalization(name='common_conv_bn{0}'.format(i))(x)
        x = PReLU(alpha_initializer=he_normal(),name='common_conv_p{0}'.format(i))(x)
    if split > 0:
        x = Conv2D(n_filter, kernel_size=kernel_size,padding='same',
                      kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength),name='common_last')(x)
    return x


## Super-Resolution Networks

def fsrcnn_red(inputs, scale=4, n_channels=3,reg_strength=0.001):
    x = inputs

    ## Conv. Layer 2: Shrinking
    x = Conv2D(12, kernel_size=1, padding='same',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_2_conv')(x )
    x = BatchNormalization(name='sr_2_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_2_p')(x)

    ## Conv. Layers 3–6: Mapping
    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_3_conv')(x )
    x = BatchNormalization(name='sr_3_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_3_p')(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_4_conv')(x )
    x = BatchNormalization(name='sr_4_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_4_p')(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_5_conv')(x )
    x = BatchNormalization(name='sr_5_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_5_p')(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_6_conv')(x )
    x = BatchNormalization(name='sr_6_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_6_p')(x)

    ##Conv.Layer  7: Expanding
    x = Conv2D(56, kernel_size=1, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_7_conv')(x )
    x = BatchNormalization(name='sr_7_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_7_p')(x)

    ##DeConv Layer 8: Deconvolution
    x = Conv2D(3 * scale ** 2, 9, padding='same', name='sr_8_conv_up{0}'.format(scale),
               kernel_initializer=he_normal())(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2D(1, 1, padding='same', name='sr', kernel_initializer=he_normal())(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1




def fsrcnn(inputs, scale=4, n_channels=3,reg_strength=0.001):
    x = inputs
    ## Conv. Layer 1: feature extraction layer 1
    x = Conv2D(56, kernel_size=5,padding='same',
                  kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_1_conv')(x)
    x = BatchNormalization(name='sr_1_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_1_p')(x)

    ## Conv. Layer 2: Shrinking
    x = Conv2D(12, kernel_size=1, padding='same',
                kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_2_conv')(x )
    x = BatchNormalization(name='sr_2_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_2_p')(x)

    ## Conv. Layers 3–6: Mapping
    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_3_conv')(x )
    x = BatchNormalization(name='sr_3_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_3_p')(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_4_conv')(x )
    x = BatchNormalization(name='sr_4_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_4_p')(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_5_conv')(x )
    x = BatchNormalization(name='sr_5_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_5_p')(x)

    x = Conv2D(12, kernel_size=3, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_6_conv')(x )
    x = BatchNormalization(name='sr_6_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_6_p')(x)

    ##Conv.Layer  7: Expanding
    x = Conv2D(56, kernel_size=1, padding='same',
           kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength),
               name='sr_7_conv')(x )
    x = BatchNormalization(name='sr_7_bn')(x)
    x = PReLU(alpha_initializer=he_normal(),name='sr_7_p')(x)

    ##DeConv Layer 8: Deconvolution

    x = Conv2D(3 * scale ** 2, 9, padding='same', name='sr_8_conv_up{0}'.format(scale),
               kernel_initializer=he_normal())(x)


    if n_channels == 1:
        x = SubpixelConv2D(scale)(x)
        x1 = Conv2D(1, 1, padding='same', name='sr', kernel_initializer=he_normal())(x)
    else:
        x1 = SubpixelConv2D(scale, name='sr')(x)

    return x1


def wdsr(inputs, scale=4,n_res_blocks=16, n_channels=3,n_filter=32,reg_strength=0.001):
    x = inputs
    x = Conv2D(n_filter, 3, padding='same', name='sr_1_conv', kernel_initializer=he_normal())(x)
    ## res blocks main branch
    m = block_res(x, num_identity=n_res_blocks, n_filter=n_filter, kernel_size=3, reg_strength=reg_strength)
    ## Deconvolution
    m = Conv2D(3 * scale ** 2, 3, padding='same', name='sr_main_last_conv',kernel_initializer=he_normal())(m)
    m = BatchNormalization(name='sr_main_last_bn')(m)
    m = PReLU(alpha_initializer=he_normal(),name='sr_main_last_p')(m)
    m = SubpixelConv2D(scale)(m)
    if n_channels == 1:
        m1 = Conv2D(1, 1, padding='same', name='sr_main_1dconv',kernel_initializer=he_normal())(m)
    else:
        m1 = m

    # skip branch
    s = Conv2D(3 * scale ** 2, 5, padding='same', name='sr_skip_last_conv',
               kernel_initializer=he_normal())(x)

    s = BatchNormalization(name='sr_skip_last_bn')(s)
    s = PReLU(alpha_initializer=he_normal(), name='sr_skip_last_p')(s)
    s = SubpixelConv2D(scale)(s)
    if n_channels == 1:
        s1 = Conv2D(1, 1, padding='same', name='sr_skip_1dconv',kernel_initializer=he_normal())(s)
    else:
        s1 = s

    return Add(name='sr')([m1, s1])

## Classification network
def resnet_mine_red(inputs, n_classes=10, reg_strength=0.001):
    x = inputs
    x = BatchNormalization(name='cl_first_bn_layer')(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding="same", name='cl_max_pooling')(x)
    x = PReLU(alpha_initializer=he_normal(), name='cl_first_prelu_layer')(x)
    x = Conv2D(64,kernel_size=3,strides=1,padding='same',name='inbetween',
               kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(reg_strength))(x)
    x = BatchNormalization(name='cl_s_bn_layer')(x)
    x = PReLU(alpha_initializer=he_normal(), name='cl_s_prelu_layer')(x)
    x = res_identity(x, 64, 3, reg_strength, name='cl_identity_64_1')
    x = res_identity(x, 64, 3, reg_strength, name='cl_identity_64_2')
    x = res_down(x, 128, 3, reg_strength, name='cl_first_down_128')
    x = res_identity(x, 128, 3, reg_strength, name='cl_identity_128_2')
    x = res_down(x, 256, 3, reg_strength, name='cl_second_down_256')
    x = res_identity(x, 256, 3, reg_strength, name='cl_identity_256_2')
    x = GlobalAveragePooling2D(name='cl_gap')(x)
    x = Dense(1000, kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(reg_strength), name='cl_dense')(
        x)
    x = BatchNormalization(name='cl_last_bn')(x)
    y = PReLU(alpha_initializer=he_normal(), name='cl_last_prelu')(x)
    outputs = Dense(n_classes,
                    activation='softmax',
                    kernel_initializer=he_normal(), name='cl')(y)
    return outputs
