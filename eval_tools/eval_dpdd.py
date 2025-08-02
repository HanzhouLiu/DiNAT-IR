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


parser = argparse.ArgumentParser(description='Dual Pixel Defocus Deblurring using Restormer')

# DPDD
parser.add_argument('--clean_dir', default='/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/DPDD/RestoreDiNAT-DPDD-width48-test/visualization/dpdd16-test', type=str, help='Directory for results')
parser.add_argument('--restored_dir', default='/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/DPDD/RestoreDiNAT-DPDD-width48-test/visualization/dpdd16-test', type=str, help='Directory of validation images')
#parser.add_argument('--restored_dir', default='/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/DPDD/Dual_Pixel_Defocus_Deblurring', type=str, help='Directory of validation images')
args = parser.parse_args()


filesC = natsorted(glob(os.path.join(args.clean_dir, 'gt', '*.png')))  # Clean images
filesR = natsorted(glob(os.path.join(args.restored_dir, '*.png')))
#filesR = natsorted(glob(os.path.join(args.restored_dir, 'output', '*.png')))  # Restored images

indoor_labels  = np.load('./datasets/DPDD/test/indoor_labels.npy')
outdoor_labels = np.load('./datasets/DPDD/test/outdoor_labels.npy')

psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileC, fileR in tqdm(zip(filesC, filesR), total=len(filesC)):

        imgC = np.float32(utils.load_img16(fileC))/65535.
        imgR = np.float32(utils.load_img16(fileR))/65535.
        
        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0,3,1,2).cuda()
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0,3,1,2).cuda()

        pips.append(alex(patchC, patchR, normalize=True).item())

        psnr.append(utils.PSNR(imgC, imgR))
        mae.append(utils.MAE(imgC, imgR))
        ssim.append(utils.SSIM(imgC, imgR))

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels-1], mae[indoor_labels-1], ssim[indoor_labels-1], pips[indoor_labels-1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels-1], mae[outdoor_labels-1], ssim[outdoor_labels-1], pips[outdoor_labels-1]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae), np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor), np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor), np.mean(mae_outdoor), np.mean(pips_outdoor)))