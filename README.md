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

## 🚀 Training

```bash
python main.py
```
---

## 📈 Performance
| Model  | Training Loss  | Val Loss |
|--------|----------------|----------|
| ReLU (w mlp[300, 300, 128]) | 0.588 | 0.593 |
| Tanh (w mlp[256, 256, 256]) | 0.548 | 0.556 |

### ReLU (w mlp[300, 300, 128]) model
![Out1](res/res_training.png)

### Tanh (w mlp[256, 256, 256]) model
![Out2](res/res_training_tanh.png)

## Comparision
![Out3](res/res_compare.png)

---

## 🧪 Test Results

| Model  | Test Loss | Test AUC | Test ACC |
|--------|-----------|----------|----------|
| ReLU (w mlp[300, 300, 128]) | 0.594 | 77.84% | 71.61% |
| Tanh (w mlp[256, 256, 256]) | 0.550 | 79.51% | 73.04% |

### ReLU (w mlp[300, 300, 128]) model
![Out4](res/res_test.png)

### Tanh (w mlp[256, 256, 256]) model
![Out5](res/res_test_tanh.png)
