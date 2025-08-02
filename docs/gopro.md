## 🚀 Reproducing Results on GoPro Dataset

We provide instructions to reproduce **DiNAT-IR** results on the GoPro motion deblurring dataset.

---

### 1️⃣ Data Preparation

#### 🔹 Training Data

- Download the **GoPro training set** from [Google Drive](#) or [百度网盘](#)
- Organize the data into the following directory structure:

```
./datasets/GoPro/train/input/
./datasets/GoPro/train/target/
```

- Crop image pairs into 512×512 patches and convert to **LMDB** format:

```bash
python scripts/data_preparation/gopro.py
```

#### 🔹 Evaluation Data

- Download the **GoPro test set** (already in LMDB format) from [Google Drive](#) or [百度网盘](#)
- Place the test files as follows:

```
./datasets/GoPro/test/input.lmdb
./datasets/GoPro/test/target.lmdb
```

---

### 2️⃣ Training DiNAT-IR

Train the model using 8 GPUs (default). You may change `--nproc_per_node` to match your GPU count.

```bash
python -m torch.distributed.launch \
  --nproc_per_node=8 --master_port=4321 \
  basicsr/train.py -opt options/train/GoPro/DiNAT-IR.yml \
  --launcher pytorch
```

---

### 3️⃣ Evaluation on GoPro

#### 🔹 Pretrained Model

- Download the pretrained weights from this repository:  
  📁 `./experiments/pretrained_models/DiNAT-IR-GoPro.pth`

#### 🔹 Run Evaluation (Single GPU)

```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 --master_port=4321 \
  basicsr/test.py -opt options/test/GoPro/DiNAT-IR.yml \
  --launcher pytorch
```

> 💡 For multi-GPU evaluation, set `--nproc_per_node` to the number of GPUs.

---

### ✅ Output

After testing, the PSNR/SSIM results and predicted images will be saved to the directory specified in `options/test/GoPro/DiNAT-IR.yml`.

---
