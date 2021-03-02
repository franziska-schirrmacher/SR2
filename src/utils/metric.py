"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/utils
@author: Franziska Schirrmacher
"""

import tensorflow.image as tfimg

def psnr(hr, sr):
    return tfimg.psnr(hr, sr,1)

def ssim(hr,sr):
    return tfimg.ssim(hr,sr,1)

