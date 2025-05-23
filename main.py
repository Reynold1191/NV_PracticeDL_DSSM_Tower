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
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'user_id': torch.tensor(row['user_id'], dtype=torch.long),
            'gender': torch.tensor(row['gender'], dtype=torch.long),
            'age': torch.tensor(row['age'], dtype=torch.long),
            'occupation': torch.tensor(row['occupation'], dtype=torch.long),
            'movie_id': torch.tensor(row['movie_id'], dtype=torch.long),
            'genres': torch.tensor(row['genre_ids'], dtype=torch.long),  # (6,)
            'label': torch.tensor(row['label'], dtype=torch.float)
        }


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

        all_genres = sorted(set(g for genre_str in movies['genres'] for g in genre_str.split('|')))
        genre_encoder = {genre: idx + 1 for idx, genre in enumerate(all_genres)} 

        def encode_genres(genre_str):
            genre_ids = [genre_encoder[g] for g in genre_str.split('|') if g in genre_encoder]
            return genre_ids[:6] + [0] * (6 - len(genre_ids)) 

        movies['genre_ids'] = movies['genres'].apply(encode_genres)

        data = ratings.merge(users, on='user_id').merge(movies[['movie_id', 'title', 'genre_ids']], on='movie_id')
        data['label'] = (data['rating'] >= 4).astype(int)

        for col in ['user_id', 'movie_id', 'occupation', 'age']:
            data[col] = LabelEncoder().fit_transform(data[col])

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


class DSSM_Tower(pl.LightningModule): #mlp_sizes=[256, 256, 256]
    def __init__(self, n_users, n_items, n_ages, n_occs, n_genres, embed_dim=64, mlp_sizes=[300, 300, 128], dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.gender_embedding = nn.Embedding(2, embed_dim)
        self.age_embedding = nn.Embedding(n_ages, embed_dim)
        self.occ_embedding = nn.Embedding(n_occs, embed_dim)

        self.item_embedding = nn.Embedding(n_items, embed_dim)
        self.genre_embedding = nn.Embedding(n_genres, embed_dim, padding_idx=0) 

        user_input_dim = embed_dim * 4
        item_input_dim = embed_dim * 2

        self.user_mlp = self._mlp(user_input_dim, mlp_sizes, dropout)
        self.item_mlp = self._mlp(item_input_dim, mlp_sizes, dropout)

        print(self.user_mlp)
        print(self.item_mlp)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def _mlp(self, input_dim, layer_sizes, dropout):
        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_dim, size))
            # layers.append(nn.ReLU())
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

        genre_embed = self.genre_embedding(batch['genres'])  # [B, max_genres, D]
        genre_embed = genre_embed.mean(dim=1)                # reduce to [B, D]

        v = torch.cat([
            self.item_embedding(batch['movie_id']),
            genre_embed
        ], dim=1)

        u = self.user_mlp(u)
        v = self.item_mlp(v)

        score = F.cosine_similarity(u, v)
        return score


    def training_step(self, batch, batch_idx):
        labels = batch['label']
        output = self.forward(batch) * 10  
        loss = self.loss_fn(output, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['label']
        output = self.forward(batch) * 10
        loss = self.loss_fn(output, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        labels = batch['label'].float()
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
    data_dir = 'ml-1m' 
    datamodule = MovieLens1MDataModule(data_dir, batch_size=1024)
    datamodule.prepare_data()

    # Extract unique values for the embedding sizes
    n_users = datamodule.train_data.dataset.data['user_id'].nunique()
    n_items = datamodule.train_data.dataset.data['movie_id'].nunique()
    n_ages = datamodule.train_data.dataset.data['age'].nunique()
    n_occs = datamodule.train_data.dataset.data['occupation'].nunique()
    n_genres = max(g for genre_list in datamodule.train_data.dataset.data['genre_ids'] for g in genre_list) + 1

    # Instantiate the model
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
