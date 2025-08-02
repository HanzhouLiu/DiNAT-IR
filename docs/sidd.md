## 🚀 Reproducing Results on SIDD Dataset

We provide instructions to reproduce **DiNAT-IR** results on the SIDD denoising dataset.

---

### 1️⃣ Data Preparation

#### 🔹 Training Data and Evaluation Data

- Download the **SIDD dataset** from [NAFNet]https://github.com/megvii-research/NAFNet/blob/main/docs/SIDD.md)
- Organize the data into the following directory structure  (or, you can use soft link):

```
./datasets/SIDD/train/input_crops.lmdb/
./datasets/SIDD/train/gt_crops.lmdb/
```
```
./datasets/SIDD/val/input_crops.lmdb/
./datasets/SIDD/val/gt_crops.lmdb/
```

---

### 2️⃣ Training DiNAT-IR

Train the model using 8 GPUs (40GB each, default). You may change `--nproc_per_node` to match your GPU count.

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((12000 + RANDOM % 10000)) basicsr/train.py -opt options/train/SIDD/RestoreDiNAT-width48.yml --launcher pytorch
```