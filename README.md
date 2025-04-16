# DSSM Tower Model for Recommendation

This repository implements a DSSM (Deep Structured Semantic Model) with **tower architecture** using PyTorch Lightning. The model is trained to learn semantic representations of users and items for recommendation or matching tasks.

## 📌 Features

- Dual-tower DSSM architecture (user tower + item tower)
- Embedding for sparse features (e.g., gender, age, occupation, genres)
- MLP layers with dropout for regularization
- Loss: `BCEWithLogitsLoss`
- AUC, accuracy, and loss metrics for training/validation/testing
- PyTorch Lightning for cleaner training loop and logging

---

## 📁 Dataset

This implementation works with datasets like **MovieLens 1M** or any custom user-item interaction dataset.

Each sample contains:
- User features: `gender`, `age`, `occupation`
- Item features: `genre`, `title` (bag-of-words or trigram embedding)
- Label: binary click/view interaction

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

## 📌 Performance
| Metric  | Value  |
|---------|--------|
| Training loss | 0.578 |
| Val loss | 0.595 |

![Out1](res/res_training.png)
![Out2](res/tensorboard.png)

---

## 🧪 Test Results

| Metric  | Value  |
|---------|--------|
| Test Loss | 0.593 |
| Test AUC | 78.15% |
| Test ACC | 71.46% |

![Out3](res/res_test.png)
