## 🚀 Reproducing Results on Rain13K Dataset

We provide instructions to reproduce **DiNAT-IR** results on the Rain13K deraining dataset.

---

### 1️⃣ Data Preparation

#### 🔹 Training Data

- Download the **Rain13K training set** from [Restormer](https://github.com/swz30/Restormer/blob/main/Deraining/download_data.py).
- Organize the data into the following directory structure  (or, you can use soft link):

```
./datasets/Rain13K/train/Rain13K/input/
./datasets/Rain13K/train/Rain13K/target/
```

- You might want to change this function [create_lmdb_for_rain13k](/basicsr/utils/create_lmdb.py) so that it can handle the correct data folders. After that, convert patches into lmdb files:
```bash
python scripts/data_preparation/rain13k_lmdb.py
```

#### 🔹 Evaluation Data

- Download the **Rain13K test set** from [Restormer](https://github.com/swz30/Restormer/blob/main/Deraining/download_data.py).
- Place the test files as follows (you can generate these lmdb files for inference following our provided codes [create_lmdb_for_rain13k](/basicsr/utils/create_lmdb.py) and [create_lmdb_for_rain13k](/scripts/data_preparation/rain13k_lmdb.py)):

```
./datasets/Rain13K/test/Rain100H/input.lmdb
./datasets/Rain13K/test/Rain100H/target.lmdb
./datasets/Rain13K/test/Rain100L/input.lmdb
./datasets/Rain13K/test/Rain100L/target.lmdb
./datasets/Rain13K/test/Test100/input.lmdb
./datasets/Rain13K/test/Test100/target.lmdb
./datasets/Rain13K/test/Test1200/input.lmdb
./datasets/Rain13K/test/Test1200/target.lmdb
./datasets/Rain13K/test/Test2800/input.lmdb
./datasets/Rain13K/test/Test2800/target.lmdb
```

---

### 2️⃣ Training DiNAT-IR

Train the model using 8 GPUs (40GB each, default). You may change `--nproc_per_node` to match your GPU count.

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((12000 + RANDOM % 10000)) basicsr/train.py -opt options/train/Rain13K/RestoreDiNAT-width48.yml --launcher pytorch
```
After the first stage training, run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$((12000 + RANDOM % 10000)) basicsr/train.py -opt options/train/Rain13K/RestoreDiNATFineTune-width48.yml --launcher pytorch
```
---

### 3️⃣ Evaluation on Deraining Datasets

#### 🔹 Pretrained Model

- Download the pretrained weights from this repository:  
  📁 `./experiments/RestoreDiNATFineTune-Rain13K-width48/models`

#### 🔹 Run Evaluation (Single GPU)

Let's take Rain100H for example,
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) basicsr/test.py -opt options/test/Rain13K/RestoreDiNAT-Rain100H-width48.yml --launcher pytorch
```
---
