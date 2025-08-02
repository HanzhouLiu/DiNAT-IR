import torch
import numpy as np
import cv2 as cv
from torch import nn
from tqdm import tqdm
import kornia
from collections import OrderedDict
from basicsr.models.archs.DeblurDiNAT_arch import DeblurDiNATLocal as Net
from natsort import natsorted
from glob import glob
import os
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def load_img(filepath):
    return cv.cvtColor(cv.imread(filepath), cv.COLOR_BGR2RGB)

if __name__ == '__main__':

    # Model settings
    inp_channels=3
    out_channels=3
    dim = 32
    num_blocks = [2,3,3,4]
    num_refinement_blocks = 2
    heads = [1,2,4,8]
    dilations = [36, 18, 9, 4]
    ffn_expansion_factor = 2.66
    bias = False
    LayerNorm_type = 'WithBias'
    chan_adapt=True
    mode='globl'
    dual_pixel_task = False

    #model_name = 'DeblurDiNATNoChanLocal-GoPro-width16'  # 384: 27.6986 dB 704: 27.9197 dB
    #test_layer = Net(dim=16, chan_adapt=False, mode='local').cuda()  # Ablation study settings
    
    #model_name = 'DeblurDiNATChanLocal-GoPro-width16'  # 384: 29.2173 dB 704: 29.5059 dB
    #test_layer = Net(dim=16, chan_adapt=True, mode='local').cuda()  # Ablation study settings

    #model_name = 'DeblurDiNATNoChanGlobl-GoPro-width16'  # 384: 30.9516 dB
    #test_layer = Net(dim=16, chan_adapt=False, mode='globl').cuda()

    model_name = 'DeblurDiNATChanGlobl-GoPro-width16'  # 384: 30.1074 dB
    test_layer = Net(dim=16, chan_adapt=True, mode='globl').cuda()

    # Load weights
    local_path = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/models'
    remot_path = 'scratch/group/3dsr/lowlevel/codes/RestoreNAT/experiments'
    path_w = 'models/net_g_latest.pth'
    weights = os.path.join(local_path, remot_path, model_name, path_w)
    print(f"===> Using weights: {weights}")
    checkpoint = torch.load(weights)

    try:
        test_layer.load_state_dict(checkpoint['params'])
    except:
        try:
            test_layer.load_state_dict(checkpoint['state_dict'])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            test_layer.load_state_dict(new_state_dict)

    test_layer.eval()
    with torch.no_grad():
        test_layer(torch.rand(1, 3, 256, 256).cuda())  # TCL warm-up

    # Paths
    img_end = 1111
    crop_size = 384
    eval_crop = 384
    blur_dir = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GOPRO/test/testA'
    gt_dir = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GOPRO/test/testB'

    blur_files = natsorted(glob(os.path.join(blur_dir, '*.png')) + glob(os.path.join(blur_dir, '*.jpg')))
    gt_files = [f.replace('testA', 'testB') for f in blur_files]

    psnr_total = 0
    for blur_path, gt_path in tqdm(zip(blur_files[:img_end], gt_files[:img_end]), total=img_end):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # Load and normalize
        blur_img = np.float32(load_img(blur_path)) / 255.
        gt_img = np.float32(load_img(gt_path)) / 255.

        blur_tensor = torch.from_numpy(blur_img).permute(2, 0, 1).unsqueeze(0).cuda()
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).cuda()

        # Crop to center 448x448 as input
        input_tensor = kornia.geometry.center_crop(blur_tensor, [crop_size, crop_size])
        gt_tensor_crop = kornia.geometry.center_crop(gt_tensor, [eval_crop, eval_crop])

        with torch.no_grad():
            restored = test_layer(input_tensor)
            restored_crop = kornia.geometry.center_crop(restored, [eval_crop, eval_crop])

        # Convert to numpy
        restored_np = restored_crop.squeeze().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        gt_np = gt_tensor_crop.squeeze().clamp(0, 1).cpu().permute(1, 2, 0).numpy()

        # Compute PSNR
        psnr = compare_psnr(gt_np, restored_np, data_range=1.0)
        psnr_total += psnr

    avg_psnr = psnr_total / img_end
    print(f"[{model_name}] Center 384x384 PSNR over {img_end} images: {avg_psnr:.4f} dB")
