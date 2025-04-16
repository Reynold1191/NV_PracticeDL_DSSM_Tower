import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.functional import accuracy, auroc

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class MovieLens1MDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'user_id': torch.tensor(row['user_id'], dtype=torch.long),
            'gender': torch.tensor(row['gender'], dtype=torch.long),
            'age': torch.tensor(row['age'], dtype=torch.long),
            'occupation': torch.tensor(row['occupation'], dtype=torch.long),
            'movie_id': torch.tensor(row['movie_id'], dtype=torch.long),
            'genres': torch.tensor(row['genres'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float),
        }

    def __len__(self):
        return len(self.data)


class MovieLens1MDataModule:
    def __init__(self, data_dir, batch_size=128):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        users = pd.read_csv(os.path.join(self.data_dir, 'users.dat'), sep='::', engine='python',
                            names=['user_id', 'gender', 'age', 'occupation', 'zip'])
        movies = pd.read_csv(os.path.join(self.data_dir, 'movies.dat'), sep='::', engine='python',
                             names=['movie_id', 'title', 'genres'], encoding='latin-1')
        ratings = pd.read_csv(os.path.join(self.data_dir, 'ratings.dat'), sep='::', engine='python',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'])

        users['gender'] = users['gender'].map({'F': 0, 'M': 1})
        genres_split = movies['genres'].str.split('|').str[0]
        le_genres = LabelEncoder()
        movies['genres'] = le_genres.fit_transform(genres_split)

        data = ratings.merge(users, on='user_id').merge(movies, on='movie_id')
        # data['label'] = (data['rating'] >= 4).astype('float32')
        data['label'] = (data['rating'] >= 4).astype('float32')
        
        le_user = LabelEncoder()
        le_movie = LabelEncoder()
        le_occ = LabelEncoder()
        le_age = LabelEncoder()

        data['user_id'] = le_user.fit_transform(data['user_id'])
        data['movie_id'] = le_movie.fit_transform(data['movie_id'])
        data['occupation'] = le_occ.fit_transform(data['occupation'])
        data['age'] = le_age.fit_transform(data['age'])

        full_dataset = MovieLens1MDataset(data)

        total_len = len(full_dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        self.train_data, self.val_data, self.test_data = random_split(
            full_dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class DSSM_Tower(pl.LightningModule):
    def __init__(self, n_users, n_items, n_ages, n_occs, n_genres, embed_dim=64, mlp_sizes=[300, 300, 128], dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.gender_embedding = nn.Embedding(2, embed_dim)
        self.age_embedding = nn.Embedding(n_ages, embed_dim)
        self.occ_embedding = nn.Embedding(n_occs, embed_dim)

        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.genre_embedding = nn.Embedding(n_genres, embed_dim)

        user_input_dim = embed_dim * 4
        item_input_dim = embed_dim * 2

        self.user_mlp = self._mlp(user_input_dim, mlp_sizes, dropout)
        self.item_mlp = self._mlp(item_input_dim, mlp_sizes, dropout)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def _mlp(self, input_dim, layer_sizes, dropout):
        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = size
        return nn.Sequential(*layers)

    def forward(self, batch):
        u = torch.cat([
            self.user_embedding(batch['user_id']),
            self.gender_embedding(batch['gender']),
            self.age_embedding(batch['age']),
            self.occ_embedding(batch['occupation'])
        ], dim=1)

        v = torch.cat([
            self.item_embedding(batch['movie_id']),
            self.genre_embedding(batch['genres'])
        ], dim=1)

        u = self.user_mlp(u)
        v = self.item_mlp(v)

        score = F.cosine_similarity(u, v)
        return score

    def training_step(self, batch, batch_idx):
        labels = batch['label']
        output = self.forward(batch) * 10  # scale to match BCEWithLogitsLoss
        loss = self.loss_fn(output, labels)
        # x, y = batch
        # y = y.float()
        # output = self(x)
        # loss = self.loss_fn(output, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['label']
        output = self.forward(batch) * 10
        loss = self.loss_fn(output, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        labels = batch['label']
        output = self.forward(batch) * 10

        # Eval metrics including Loss, ACC, AUC
        loss = self.loss_fn(output, labels)

        acc = accuracy(output, labels.int(), task='binary')

        auc = auroc(output, labels.int(), task='binary')

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_auc', auc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)



if __name__ == '__main__':

    # Load and prepare data
    datamodule = MovieLens1MDataModule(data_dir='ml-1m', batch_size=1024)
    datamodule.prepare_data()

    n_users = datamodule.train_data.dataset.data['user_id'].nunique()
    n_items = datamodule.train_data.dataset.data['movie_id'].nunique()
    n_ages = datamodule.train_data.dataset.data['age'].nunique()
    n_occs = datamodule.train_data.dataset.data['occupation'].nunique()
    n_genres = datamodule.train_data.dataset.data['genres'].nunique()

    print("###################################################################################################")
    print(n_users, n_items, n_ages, n_occs)
    print("###################################################################################################")
    
    model = DSSM_Tower(
        n_users=n_users,
        n_items=n_items,
        n_ages=n_ages,
        n_occs=n_occs,
        n_genres=n_genres
    )

    logger = TensorBoardLogger("tb_logs", name="DSSM_Tower")

    # Checkpoint and Early Stopping
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.4f}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min"
    )

    # Trainer
    trainer = Trainer(
        max_epochs=20,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu'
    )

    # Train
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # Evaluate
    trainer.test(model, datamodule.test_dataloader())

