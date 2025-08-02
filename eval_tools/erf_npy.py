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

"""
Additional Packages: kornia, seaborn
Codes modified from:
https://github.com/INVOKERer/LoFormer/blob/main/Motion_Deblurring/erf.py
@inproceedings{xintm2024LoFormer, 
    title = {LoFormer: Local Frequency Transformer for Image Deblurring},
    author = {Xintian Mao, Jiansheng Wang, Xingran Xie, Qingli Li and Yan Wang}, 
    booktitle = {Proc. ACM MM}, 
    year = {2024}
    }
"""

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def get_effective_receptive_field(input, output, log=True, grad=512):
    """
    draw effective receptive field
    :param input: image (1, C, H, W)
    :param output: feature e.g. (1, C', H', W')
    :return: numpy array (0~255)
    """

    init_grad = torch.zeros_like(output)

    # according to the feature shape
    init_grad[:, :, init_grad.shape[2] // 2, init_grad.shape[3] // 2] += grad

    grad = torch.autograd.grad(output, input, init_grad)

    rf = torch.transpose(torch.squeeze(grad[0]), 0, 2).numpy()
    # rf[rf < 0] = 0
    rf = np.abs(rf)
    if log:
        rf = np.log(rf+1)
    rf /= np.ptp(rf)

    return rf


if __name__ == '__main__':

    # Provide network args
    # The default settings.
    inp_channels=3, 
    out_channels=3, 
    dim = 32,
    num_blocks = [2,3,3,4], 
    num_refinement_blocks = 2,
    heads = [1,2,4,8],
    dilations = [36, 18, 9, 4],
    ffn_expansion_factor = 2.66,
    bias = False,
    LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    chan_adapt=False, 
    mode='hybrid',
    dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    
    #model_name = 'DeblurDiNATNoChan-GoPro-width16'
    #test_layer = Net(dim=16, chan_adapt=False, mode='hybrid').cuda()  # Ablation study settings
    
    #model_name = 'DeblurDiNATChan-GoPro-width16'
    #test_layer = Net(dim=16, chan_adapt=True, mode='hybrid').cuda()  # Ablation study settings

    #model_name = 'DeblurDiNATNoChanLocal-GoPro-width16'
    #test_layer = Net(dim=16, chan_adapt=False, mode='local').cuda()  # Ablation study settings
    
    #model_name = 'DeblurDiNATChanLocal-GoPro-width16'
    #test_layer = Net(dim=16, chan_adapt=True, mode='local').cuda()  # Ablation study settings

    #model_name = 'DeblurDiNATNoChanGlobl-GoPro-width16'
    #test_layer = Net(dim=16, chan_adapt=False, mode='globl').cuda()  # Ablation study settings
    
    model_name = 'DeblurDiNATChanGlobl-GoPro-width16'
    test_layer = Net(dim=16, chan_adapt=True, mode='globl').cuda()  # Ablation study settings
    #"""
    # test_layer = Net().cuda() # nn.Sequential(*list(resnet.children())[:-3])
    
    local_path = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/models'
    remot_path = 'scratch/group/3dsr/lowlevel/codes/RestoreNAT/experiments'
    path_w = 'models/net_g_latest.pth'
    weights = os.path.join(local_path, remot_path, model_name, path_w)
    print(weights)
    img_end = 100  # take the first 100 images as the inputs
    crop_size = 448
    checkpoint = torch.load(weights)
    # Activate TCL Layers
    init_feat = torch.rand([1, 3, 256, 256]).cuda()
    with torch.no_grad():
        test_layer.forward(init_feat)

    inp_dir = '/mnt/g/RESEARCH/PHD/Motion_Deblurred/datasets/GOPRO/test/testA'  # PATH to the dataset
    out_dir = '/mnt/d/Data/Research/low_level/RestoreNAT/results/remote/GoPro-abl/erf'
    os.makedirs(out_dir, exist_ok=True)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    # model_restoration.load_state_dict(checkpoint['params'])
    try:
        test_layer.load_state_dict(checkpoint['params'])
        state_dict = checkpoint["params"]
        # for k, v in state_dict.items():
        #     print(k)
    except:
        try:
            test_layer.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # print(k)
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            test_layer.load_state_dict(new_state_dict)
    print("===>Testing using weights: ", weights)
    erf = np.zeros((crop_size, crop_size))
    # test_layer.train()
    # with torch.no_grad():
    for file_ in tqdm(files[:img_end]):  # [:1]
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(load_img(file_)) / 255.

        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()
        # print(input_.shape, input_.dtype)
        input_tensor = kornia.geometry.center_crop(input_, [crop_size, crop_size])
        img_gray = cv.cvtColor(kornia.tensor_to_image(input_tensor.cpu()), cv.COLOR_BGR2GRAY)
        # print(input_tensor.shape)
        input_tensor.requires_grad = True
        # inputx = input.clone().detach().requires_grad_(True) # torch.tensor(input, requires_grad=True)
        feature = test_layer(input_tensor)
        feature = feature - input_tensor
        # 特征图的channel维度算均值且去掉batch维度，得到二维张量
        feature_map = feature.mean(dim=[0,1], keepdim=False).squeeze()
        # 对二维张量中心点（标量）进行backward
        # init_grad = feature_map[(feature_map.shape[0] // 2 - 1)][(feature_map.shape[1] // 2 - 1)]
        # init_grad = torch.zeros_like(feature)
        #
        # # accrding to the feature shape
        # init_grad[:, :, init_grad.shape[2] // 2, init_grad.shape[3] // 2] += 512
        # init_grad.backward(retain_graph=True)
        feature_map[(feature_map.shape[0] // 2 - 1)][(feature_map.shape[1] // 2 - 1)].backward(retain_graph=True)
        # 对输入层的梯度求绝对值
        grad = torch.abs(input_tensor.grad)
        # 梯度的channel维度算均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
        grad = grad.mean(dim=1, keepdim=False).squeeze()

        # 累加所有图像的梯度，由于后面要进行归一化，这里可以不算均值
        rf = grad.cpu().numpy()
        # rf[(rf.shape[0] // 2 - 1), (rf.shape[1] // 2 - 1)] = np.log(rf[(rf.shape[0] // 2 - 1), (rf.shape[1] // 2 - 1)] + 1)
        # rf = np.log(rf + 1)
        rf = np.clip(rf, 0., 0.25)
        rf /= np.ptp(rf)
        # rf = (img_gray + rf) / 2.
        # print(rf.max())
        # rf = get_effective_receptive_field(input_tensor.cpu(), feature.cpu(), log=True, grad=512)
        erf += rf # np.mean(rf, axis=-1)
        # print(erf.max())
    erf /= img_end
    print(erf.max())
    np.save(os.path.join(out_dir, f'erf_avg_{model_name}.npy'), erf)
    # erf[(erf.shape[0] // 2 - 1), (erf.shape[1] // 2 - 1)] -= 0.7
    # cv.imshow('receptive field', erf)
    # cv.waitKey(0)
    # cv.imwrite(os.path.join(out_dir, 'test.png'), erf * 255)
    """
    sns_plot = plt.figure()
    sns.heatmap(erf, cmap='Blues_r', linewidths=0.0, vmin=0, vmax=0.25,
                xticklabels=False, yticklabels=False, cbar=True) # RdBu_r Reds_r .invert_yaxis()
    # out_way = os.path.join(out_root, 'attn_matrix_cosine_tar-center_DCT_LN_local' + '.png')
    out_way = os.path.join(out_dir, attn_type + '_heatmap_br_nolog' + '.png')
    sns_plot.savefig(out_way, dpi=700)
    """