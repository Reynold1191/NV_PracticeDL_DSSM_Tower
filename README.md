# 📌 DSSM Tower Model for Recommendation

## 🔍 Introduction
This repository implements a DSSM (Deep Structured Semantic Model) with **tower architecture** using PyTorch Lightning. The model is trained to learn semantic representations of users and items for recommendation or matching tasks.

---

## 📦 Features

- Dual-tower DSSM architecture (user tower + item tower)
- Embedding for sparse features (e.g., gender, age, occupation, genres)
- Each movie can have up to 6 genres, represented as a vector [g1, g2, g3, g4, g5, g6] — if any are missing, they will be padded with 0. Using mean pooling to combine the 6 genre embeddings into one vector.
- MLP layers with dropout for regularization
- Loss: `BCEWithLogitsLoss`
- AUC, accuracy, and loss metrics for training/validation/testing
- PyTorch Lightning for cleaner training loop and logging

---

## 📁 Dataset

This implementation works with datasets like **MovieLens 1M** or any custom user-item interaction dataset.

Each sample contains:
- User features: `gender`, `age`, `occupation`
- Item features: `genre`, `title` 
- Label: binary 

---

## 🛠️ Requirements

```bash
pip install pytorch_lightning
```

---

## ⚖️ Comparison of MLP Structures

Here are some comparision between four model structures

---

### 🧱 Structure ReLU (w mlp[300, 300, 128]) model

```python
Sequential(
  (0): Linear(in_features=256, out_features=300, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.2, inplace=False)
  (3): Linear(in_features=300, out_features=300, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.2, inplace=False)
  (6): Linear(in_features=300, out_features=128, bias=True)
  (7): ReLU()
  (8): Dropout(p=0.2, inplace=False)
)
Sequential(
  (0): Linear(in_features=128, out_features=300, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.2, inplace=False)
  (3): Linear(in_features=300, out_features=300, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.2, inplace=False)
  (6): Linear(in_features=300, out_features=128, bias=True)
  (7): ReLU()
  (8): Dropout(p=0.2, inplace=False)
)
```

### 🧱 Structure Tanh (w mlp[256, 256, 256]) model

```python
Sequential(
  (0): Linear(in_features=256, out_features=256, bias=True)
  (1): Tanh()
  (2): Dropout(p=0.2, inplace=False)
  (3): Linear(in_features=256, out_features=256, bias=True)
  (4): Tanh()
  (5): Dropout(p=0.2, inplace=False)
  (6): Linear(in_features=256, out_features=256, bias=True)
  (7): Tanh()
  (8): Dropout(p=0.2, inplace=False)
)
Sequential(
  (0): Linear(in_features=128, out_features=256, bias=True)
  (1): Tanh()
  (2): Dropout(p=0.2, inplace=False)
  (3): Linear(in_features=256, out_features=256, bias=True)
  (4): Tanh()
  (5): Dropout(p=0.2, inplace=False)
  (6): Linear(in_features=256, out_features=256, bias=True)
  (7): Tanh()
  (8): Dropout(p=0.2, inplace=False)
)
```

### 🧱 Structure Tanh (w mlp[256, 256, 256], BatchNorm1d and xavier_normal_ )  model - Dropout **After** Tanh Activation - Tanh V2 Model

```python
Sequential(
  (0): Linear(in_features=256, out_features=256, bias=True)
  (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): Tanh()
  (3): Dropout(p=0.3, inplace=False)
  (4): Linear(in_features=256, out_features=256, bias=True)
  (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): Tanh()
  (7): Dropout(p=0.3, inplace=False)
  (8): Linear(in_features=256, out_features=256, bias=True)
  (9): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (10): Tanh()
  (11): Dropout(p=0.3, inplace=False)
)
Sequential(
  (0): Linear(in_features=128, out_features=256, bias=True)
  (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): Tanh()
  (3): Dropout(p=0.3, inplace=False)
  (4): Linear(in_features=256, out_features=256, bias=True)
  (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): Tanh()
  (7): Dropout(p=0.3, inplace=False)
  (8): Linear(in_features=256, out_features=256, bias=True)
  (9): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (10): Tanh()
  (11): Dropout(p=0.3, inplace=False)
)
```

### 🧱 Structure Tanh (w mlp[256, 256, 256, BatchNorm1d and xavier_normal_  )  model - Dropout **Before** Linear Layer - Tanh V3 Model

```python
Sequential(
  (0): Dropout(p=0.3, inplace=False)
  (1): Linear(in_features=256, out_features=256, bias=True)
  (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Tanh()
  (4): Dropout(p=0.3, inplace=False)
  (5): Linear(in_features=256, out_features=256, bias=True)
  (6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (7): Tanh()
  (8): Dropout(p=0.3, inplace=False)
  (9): Linear(in_features=256, out_features=256, bias=True)
  (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (11): Tanh()
)
Sequential(
  (0): Dropout(p=0.3, inplace=False)
  (1): Linear(in_features=128, out_features=256, bias=True)
  (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): Tanh()
  (4): Dropout(p=0.3, inplace=False)
  (5): Linear(in_features=256, out_features=256, bias=True)
  (6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (7): Tanh()
  (8): Dropout(p=0.3, inplace=False)
  (9): Linear(in_features=256, out_features=256, bias=True)
  (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (11): Tanh()
)
```

---

## 🚀 Training

```bash
python main.py
```
---

## 📈 Performance (The best values are italicized)
| Model  | Training Loss  | Val Loss |
|--------|----------------|----------|
| **ReLU (w mlp[300, 300, 128])** | 0.588 | 0.593 |
| **Tanh (w mlp[256, 256, 256])** | 0.556 | 0.548 |
| **Tanh V2** | 0.526 | 0.548 |
| **Tanh V3** | _0.494_ | _0.515_ |

### ReLU (w mlp[300, 300, 128]) model
![Out1](res/res_training.png)

### Tanh (w mlp[256, 256, 256]) model
![Out2](res/res_training_tanh.png)

### Tanh V2 model
![Out3](res/res_training_v2.png)

### Tanh V3 model
![Out4](res/res_training_v3.png)

### Comparision
![Out5](res/res_compare_final.png)

---

## 🧪 Test Results (The best values are italicized)
| Model  | Test Loss | Test AUC | Test ACC |
|--------|-----------|----------|----------|
| **ReLU (w mlp[300, 300, 128])** | 0.594 | 77.84% | 71.61% |
| **Tanh (w mlp[256, 256, 256])** | 0.550 | 79.51% | 73.04% |
| **Tanh V2** | 0.552 | 81.58% | 74.42% |
| **Tanh V3** | _0.514_ | _81.74%_ | _74.56%_ |

### ReLU (w mlp[300, 300, 128]) model
![Out6](res/res_test.png)

### Tanh (w mlp[256, 256, 256]) model
![Out7](res/res_test_tanh.png)

### Tanh V2 model
![Out8](res/res_test_v2.png)

### Tanh V3 model
![Out9](res/res_test_v3.png)
