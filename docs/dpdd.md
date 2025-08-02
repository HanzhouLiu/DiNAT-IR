## 🚀 Reproducing Results on GoPro Dataset

We provide instructions to reproduce **DiNAT-IR** results on the GoPro motion deblurring dataset.

---

### 1️⃣ Data Preparation

#### 🔹 Training Data

- Download the **DPDD training set** from [Restormer](https://github.com/swz30/Restormer/tree/main/Defocus_Deblurring), or follow the link in [dpdd_download](/scripts/data_preparation/dpdd_download.py) to download it manually.
- Organize the data into the following directory structure  (or, you can use soft link):

```
./datasets/DPDD/train/inputC/
./datasets/DPDD/train/inputL/
./datasets/DPDD/train/inputR/
./datasets/DPDD/train/target/
```

- Crop image pairs into 512×512 patches:

```bash
python scripts/data_preparation/dpdd_crop.py
```

- You might want to change this function [create_lmdb_for_dpdd](/basicsr/utils/create_lmdb.py) so that it can handle the correct data folders. After that, convert patches into lmdb files:
```bash
python scripts/data_preparation/dpdd_lmdb.py
```

#### 🔹 Evaluation Data

- Download the **DPDD test set** (already in LMDB format) from [Restormer](https://github.com/swz30/Restormer/tree/main/Defocus_Deblurring), or follow the link in ./scripts/data_preparation/dpdd_download.py to download it manually.
- Place the test files as follows (you can generate these lmdb files for inference following our provided codes [create_lmdb_for_dpdd](/basicsr/utils/create_lmdb.py)):

```
./datasets/DPDD/test/inputC.lmdb
./datasets/DPDD/test/inputL.lmdb
./datasets/DPDD/test/inputR.lmdb
./datasets/DPDD/test/target.lmdb
```

---

### 2️⃣ Training DiNAT-IR

Train the model using 8 GPUs (40GB each, default). You may change `--nproc_per_node` to match your GPU count.

Dual-Pixel
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((12000 + RANDOM % 10000)) basicsr/train.py -opt options/train/DPDD/RestoreDiNAT-width48.yml --launcher pytorch
```
Single-Image
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((12000 + RANDOM % 10000)) basicsr/train.py -opt options/train/SPDD_8bit/RestoreDiNAT-width48.yml --launcher pytorch
```
---

### 3️⃣ Evaluation on DPDD

#### 🔹 Pretrained Model

- Download the pretrained weights from this repository:  
  📁 `./experiments/RestoreDiNAT-DPDD-width48/models`
and
  📁 `./experiments/RestoreDiNAT-SPDD-width48/models`

#### 🔹 Run Evaluation (Single GPU)

Dual-Pixel
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) basicsr/test.py -opt options/test/DPDD/RestoreDiNAT-width48.yml --launcher pytorch
```
Single-Image
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) basicsr/test.py -opt options/test/SPDD_8bit/RestoreDiNAT-width48.yml --launcher pytorch
```
---

### ✅ Output

Run [eval_dpdd.py](eval_tools/eval_dpdd.py) to calculate metrics for dual-pixel defocus deblurring.
Run [eval_spdd.py](eval_tools/eval_spdd.py) to calculate metrics for single-image defocus deblurring.
---
