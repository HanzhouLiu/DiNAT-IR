import torch
import numpy as np
import cv2 as cv
from torch import nn
from tqdm import tqdm
import kornia
import cv2
from collections import OrderedDict
# from torchvision.models import resnet18
from basicsr.models.archs.DeblurDiNAT_arch import DeblurDiNATLocal as Net
from natsort import natsorted
from glob import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

erf1 = np.load('/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/erf/erf_avg_DeblurDiNATChanLocal-GoPro-width16.npy')
erf2 = np.load('/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/erf/erf_avg_DeblurDiNATNoChanLocal-GoPro-width16.npy')

diff = erf1 - erf2
diff = np.clip(diff, -0.1, 0.1)  # Limit for visualization contrast

plt.figure(figsize=(6, 6))
sns.heatmap(diff, cmap='coolwarm', center=0,
            xticklabels=False, yticklabels=False, cbar=True)
plt.title('ERF Difference (Chan - NoChan)')
plt.savefig('erf_diff_Chan_vs_NoChan.png', dpi=700, bbox_inches='tight')