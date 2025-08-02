[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dinat-ir-exploring-dilated-neighborhood/image-deblurring-on-gopro)](https://paperswithcode.com/sota/image-deblurring-on-gopro?p=dinat-ir-exploring-dilated-neighborhood)

# DiNAT-IR: Exploring Dilated Neighborhood Attention for High-Quality Image Restoration

Official PyTorch implementation of the paper:  
**[DiNAT-IR: Exploring Dilated Neighborhood Attention for High-Quality Image Restoration](https://arxiv.org/abs/2507.17892)**

**Hanzhou Liu, Binghan Li, Chengkai Liu, Mi Lu**

> Transformers, with their self-attention mechanisms for modeling long-range dependencies, have become a dominant paradigm in image restoration tasks. However, the high computational cost of self-attention limits scalability to high-resolution images, making efficiency-quality trade-offs a key research focus. To address this, Restormer employs channel-wise self-attention, which computes attention across channels instead of spatial dimensions. While effective, this approach may overlook localized artifacts that are crucial for high-quality image restoration. To bridge this gap, we explore Dilated Neighborhood Attention (DiNA) as a promising alternative, inspired by its success in high-level vision tasks. DiNA balances global context and local precision by integrating sliding-window attention with mixed dilation factors, effectively expanding the receptive field without excessive overhead. However, our preliminary experiments indicate that directly applying this global-local design to the classic deblurring task hinders accurate visual restoration, primarily due to the constrained global context understanding within local attention. To address this, we introduce a channel-aware module that complements local attention, effectively integrating global context without sacrificing pixel-level precision. The proposed DiNAT-IR, a Transformer-based architecture specifically designed for image restoration, achieves competitive results across multiple benchmarks, offering a high-quality solution for diverse low-level computer vision problems.

---

## 🔥 News
- **2025.08.02**: Code and pretrained models are released!
- **2025.07.23**: Paper is available on arXiv! [📄](https://arxiv.org/abs/2507.17892)

---

## 📦 Supported Tasks

We evaluate DiNAT-IR on multiple image restoration tasks:
- ✅ Motion Deblurring (e.g., GoPro)
- ✅ Dual-Pixel Defocus Deblurring (e.g., DPDD)
- ✅ Single Image Defocus Deblurring
- ✅ Image Denoising (e.g., SIDD)
- ✅ Image Deraining (e.g., Rain100H, Rain1400)

---

## 🧱 Installation

This implementation is based on [BasicSR](https://github.com/xinntao/BasicSR).

```bash
git clone https://github.com/HanzhouLiu/DiNAT-IR.git
cd DiNAT-IR
conda create -n DiNAT-IR python=3.8
conda activate DiNAT-IR
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install natten==0.14.6+torch200cu118 -f https://shi-labs.com/natten/wheels
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
