# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numbers
#from models.torch_wavelets import DWT_2D, IDWT_2D
from natten import NeighborhoodAttention1D, NeighborhoodAttention2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransBlock(nn.Module):
    def __init__(self, dim, num_heads, dilation, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt):
        super(TransBlock, self).__init__()
        kernel = 7
        # print(dilation)
        self.heads = num_heads
        self.kernel = kernel
        self.na2d = NeighborhoodAttention2D(dim=dim, kernel_size=kernel, 
                                            dilation=dilation, num_heads=num_heads)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if chan_adapt:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.chan_adapt = chan_adapt
        
    def forward(self, x):
        x_norm1 = self.norm1(x)
        if self.chan_adapt:
            x = x + self.attn(x_norm1)*self.chan_mod(x_norm1)
        else: 
            x = x + self.attn(x_norm1)
        x = x + self.ffn(self.norm2(x))

        return x

    def attn(self, x):
        # b, c, h, w -> b, h, w, c
        x = x.permute(0, 2, 3, 1)
        x = self.na2d(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
    def chan_mod(self, x):
        score = self.pool(x)
        B, C, H, W = score.shape
        #score = rearrange(score, 'b c h w -> (b h w) 1 c')
        #score = self.conv(score)
        #score = rearrange(score, '(b h w) 1 c -> b c h w', b = B, h = H, w = W)
        if W != 1: 
            score = score.permute(0, 2, 3, 1).reshape(B * H * W, C, 1, 1)
        score = self.conv(score.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        if W != 1: 
            score = score.reshape(B, H, W, C).permute(0, 3, 1, 2)
        score = self.sigmoid(score)
        return score.expand_as(x)


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    

class DualTransBlock(nn.Module):
    def __init__(self, dim, num_heads, dilation, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt, mode):
        super(DualTransBlock, self).__init__()
        if mode == "local_only":
            self.local = TransBlock(dim, num_heads, 1, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt)
            self.globl = TransBlock(dim, num_heads, 1, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt)
        elif mode == 'globl_only':
            self.local = TransBlock(dim, num_heads, dilation, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt)
            self.globl = TransBlock(dim, num_heads, dilation, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt)
        else: 
            self.local = TransBlock(dim, num_heads, 1, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt)
            self.globl = TransBlock(dim, num_heads, dilation, ffn_expansion_factor, bias, LayerNorm_type, chan_adapt)
    def forward(self, x):
        x = self.local(x)
        x = self.globl(x)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class RestoreDiNAT(nn.Module):
    def __init__(self, 
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
    ):

        super(RestoreDiNAT, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[DualTransBlock(dim=dim, num_heads=heads[0], dilation=dilations[0], ffn_expansion_factor=ffn_expansion_factor, 
                                                             bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[DualTransBlock(dim=int(dim*2**1), num_heads=heads[1], dilation=dilations[1], ffn_expansion_factor=ffn_expansion_factor, 
                                                             bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[DualTransBlock(dim=int(dim*2**2), num_heads=heads[2], dilation=dilations[2], ffn_expansion_factor=ffn_expansion_factor, 
                                                             bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[DualTransBlock(dim=int(dim*2**3), num_heads=heads[3], dilation=dilations[3], ffn_expansion_factor=ffn_expansion_factor, 
                                                     bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[DualTransBlock(dim=int(dim*2**2), num_heads=heads[2], dilation=dilations[2], ffn_expansion_factor=ffn_expansion_factor, 
                                                             bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[DualTransBlock(dim=int(dim*2**1), num_heads=heads[1], dilation=dilations[1], ffn_expansion_factor=ffn_expansion_factor, 
                                                             bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[DualTransBlock(dim=int(dim*2**1), num_heads=heads[0], dilation=dilations[0], ffn_expansion_factor=ffn_expansion_factor, 
                                                             bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[DualTransBlock(dim=int(dim*2**1), num_heads=heads[0], dilation=dilations[0], ffn_expansion_factor=ffn_expansion_factor, 
                                                         bias=bias, LayerNorm_type=LayerNorm_type, chan_adapt=chan_adapt, mode=mode) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

class RestoreDiNATLocal(Local_Base, RestoreDiNAT):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        RestoreDiNAT.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]  # 28 for nafnet
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = RestoreDiNAT(dim=48, chan_adapt=True).cuda()


    inp_shape = (3, 128, 128)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

    torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracker

    # Before operation
    mem_before = torch.cuda.memory_allocated() / 1e9  # Convert to GB

    # Perform some computation
    x = torch.randn(1, 3, 256, 256, device="cuda")
    # with torch.no_grad():
    x = net(x)

    # After operation
    mem_after = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB

    print(f"Before: {mem_before:.4f} GB, After: {mem_after:.4f} GB, Peak: {peak_mem:.4f} GB")

    # Debug TCL
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = RestoreDiNATLocal().to(device)
    # net = HINetLocal().to(device)
    #for size in [256,512]:
    #    img = torch.randn(1, 3, size, size).to(device)
    #    outputs = net(img)
        # print(*[x.shape for x in outputs])