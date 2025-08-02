"""
## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch

from skimage import img_as_ubyte
import cv2
from eval_tools import defocus_util as utils
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx

import lpips
alex = lpips.LPIPS(net='alex').cuda()


parser = argparse.ArgumentParser(description='LPIPS Calculation using Restormer')

#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATNoChanLocal-GoPro-width16/visualization/gopro-test/output'  # Overall: PSNR 31.555100 LPIPS 0.110610
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATNoChanGlobl-GoPro-width16/visualization/gopro-test/output'  # Overall: PSNR 32.027387 LPIPS 0.102813
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATNoChan-GoPro-width16/visualization/gopro-test/output'  # Overall: PSNR 31.874558 LPIPS 0.106823
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATChanLocal-GoPro-width16/visualization/gopro-test/output'  # Overall: PSNR 31.968329 LPIPS 0.104809
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATChanGlobl-GoPro-width16/visualization/gopro-test/output'  # Overall:  PSNR 32.059710 LPIPS 0.102536
fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/DeblurDiNATChan-GoPro-width16/visualization/gopro-test/output'  # Overall: PSNR 32.062954 LPIPS 0.103455
#fdir1 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/Baseline-GoPro-width16/visualization/gopro-test/output'  # PSNR 30.318796 LPIPS 0.137132

#fdir1 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GoPro_/test/testB'
#fdir1 = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/HIDE/test/testA'

fdir2 = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/images/Baseline-GoPro-width16/visualization/gopro-test/gt' 

args = parser.parse_args()


filesC = natsorted(glob(os.path.join(fdir2, '*.png')))  # Clean images
filesR = natsorted(glob(os.path.join(fdir1, '*.png')))  # Restored images


psnr, pips = [], []
with torch.no_grad():
    for fileC, fileR in tqdm(zip(filesC, filesR), total=len(filesC)):

        imgC = np.float32(utils.load_img(fileC))/255.
        imgR = np.float32(utils.load_img(fileR))/255.
        
        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0,3,1,2).cuda()
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0,3,1,2).cuda()

        pips.append(alex(patchC, patchR, normalize=True).item())

        psnr.append(utils.PSNR(imgC, imgR))


print("Overall: PSNR {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(pips)))