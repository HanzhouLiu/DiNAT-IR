## 🚀 Reproducing Results on GoPro Dataset

We provide instructions to reproduce **DiNAT-IR** results on the GoPro motion deblurring dataset.

---

### 1️⃣ Data Preparation

#### 🔹 Training Data

- Download the **GoPro training set** from [NAFNet](https://github.com/megvii-research/NAFNet/blob/main/docs/GoPro.md)
- Organize the data into the following directory structure  (or, you can use soft link):

```
./datasets/GoPro/train/input/
./datasets/GoPro/train/target/
```

- Crop image pairs into 512×512 patches and convert to **LMDB** format:

```bash
python scripts/data_preparation/gopro.py
```

#### 🔹 Evaluation Data

- Download the **GoPro test set** (already in LMDB format) from [NAFNet](https://github.com/megvii-research/NAFNet/blob/main/docs/GoPro.md)
- Place the test files as follows (or, you can use soft link):

```
./datasets/GoPro/test/input.lmdb
./datasets/GoPro/test/target.lmdb
```

---

### 2️⃣ Training DiNAT-IR

Train the model using 8 GPUs (40GB each, default). You may change `--nproc_per_node` to match your GPU count.

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((12000 + RANDOM % 10000)) basicsr/train.py -opt options/train/GoPro/RestoreDiNAT-width48.yml --launcher pytorch
```
After the first stage training, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((12000 + RANDOM % 10000)) basicsr/train.py -opt options/train/GoPro/RestoreDiNATFineTune-width48.yml --launcher pytorch
```
---

### 3️⃣ Evaluation on GoPro

#### 🔹 Pretrained Model

- Download the pretrained weights from this repository:  
  📁 `./experiments/RestoreDiNATFineTune-GoPro-width48/models/net_g_latest.pth`

#### 🔹 Run Evaluation (Single GPU)

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) basicsr/test.py -opt options/test/GoPro/RestoreDiNATFineTune-width48.yml --launcher pytorch
```

---

### ✅ Output

After testing, the PSNR/SSIM results and predicted images will be saved to the corresponding directory.

---
