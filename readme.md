[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dinat-ir-exploring-dilated-neighborhood/image-deblurring-on-gopro)](https://paperswithcode.com/sota/image-deblurring-on-gopro?p=dinat-ir-exploring-dilated-neighborhood)

# DiNAT-IR: Exploring Dilated Neighborhood Attention for High-Quality Image Restoration

Official PyTorch implementation of the paper:  
**[DiNAT-IR: Exploring Dilated Neighborhood Attention for High-Quality Image Restoration](https://arxiv.org/abs/2507.17892)**

**Hanzhou Liu, Binghan Li, Chengkai Liu, Mi Lu**

> We propose DiNAT-IR, a novel image restoration Transformer that builds upon dilated neighborhood attention. It balances local and global context via multi-dilated windowed attention, and introduces a channel-aware module to overcome the locality bottleneck. DiNAT-IR achieves competitive performance across diverse low-level tasks, including motion deblurring, defocus deblurring, denoising, and deraining.

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
git clone https://github.com/xxx/DiNAT-IR.git
cd DiNAT-IR
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
