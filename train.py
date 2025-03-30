import argparse
import os.path

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from data.omics_data import SCOmicsData, SCOmicsDataWrapper
from data.preprocessing import DataTransform, Compose, BinningTransform, SourceNameExtractor, FeatureIdExtractor
from scomics.model.model import TransformerModel
from torchmetrics import MeanMetric, Accuracy

from utils.utils import save_ckpt, seed_everything


def training(net: torch.nn.Module, optimizer: Optimizer, dataset: SCOmicsData, source_id: int, device: torch.device):
    training_data = SCOmicsDataWrapper(dataset, SEQ_LEN, PAD_TOKEN_ID, MASK_TOKEN_ID, len(SPECIAL_TOKENS), source_id)
    data_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    net.train()
    train_loss, train_acc = MeanMetric(), MeanMetric()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean')
    acc_metric = Accuracy(task='multiclass', num_classes=N_CLASSES, ignore_index=PAD_TOKEN_ID).to(device)
    for index, batch in enumerate(data_loader):
        # Forward pass
        optimizer.zero_grad()
        output = net(batch)

        preds, actual = output.view(-1, N_CLASSES), batch['X_bin_labels'].view(-1).to(device)
        loss = loss_fn(preds, actual)
        acc = acc_metric(preds.softmax(dim=-1), actual)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item())
        train_acc.update(acc.item())

        # Print statistics
        if index > 0 and index % 50 == 0:
            print(f"Step {index + 1}/{len(data_loader)}: "
                  f"Average Loss = {train_loss.compute().item():.4f}, "
                  f"Accuracy = {train_acc.compute().item():.4f}")
            train_loss.reset()
            train_acc.reset()


def evaluation(net: torch.nn.Module, dataset: SCOmicsData, source_id: int, device: torch.device):
    validation_data = SCOmicsDataWrapper(dataset, SEQ_LEN, PAD_TOKEN_ID, MASK_TOKEN_ID, len(SPECIAL_TOKENS), source_id)
    val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

    net.eval()
    val_loss, val_acc = MeanMetric(), MeanMetric()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean')
    acc_metric = Accuracy(task='multiclass', num_classes=N_CLASSES, ignore_index=PAD_TOKEN_ID).to(device)
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            output = net(batch)

            # Compute loss
            preds, actual = output.view(-1, N_CLASSES),  batch['X_bin_labels'].view(-1).to(device)
            loss = loss_fn(preds, actual)
            acc = acc_metric(preds.softmax(dim=-1), actual)

            val_loss.update(loss.item())
            val_acc.update(acc.item())

        loss, acc = val_loss.compute().item(), val_acc.compute().item()
        # Print statistics
        print(f"Validation Loss = {loss:.4f}, Validation Accuracy = {acc:.4f}")

    return loss, acc


def fit(net: nn.Module, train_dataset: SCOmicsData, val_dataset: SCOmicsData, device: torch.device, ckpt_path: str):
    net = net.to(device)
    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")
        training(net, optimizer, train_dataset, source_id=-1, device=device)

        scheduler.step()
        if epoch % VALIDATE_EVERY != 0:
            continue

        for source_id in range(len(DATA_FILES)):
            source_name = DATA_FILES[source_id].split("_")[-1]
            loss, acc = evaluation(net, val_dataset, source_id, device)
            print(f"[{source_name}]: Validation Loss = {loss:.4f}, Validation Accuracy = {acc:.4f}")

        loss, acc = evaluation(net, val_dataset, -1, device)
        print(f"[ALL]: Validation Loss = {loss:.4f}, Validation Accuracy = {acc:.4f}")
        save_ckpt(epoch, net, optimizer, scheduler, ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='resources/dataset')
    parser.add_argument("--ckpt_folder", type=str, default='ckpt')
    parser.add_argument("--seed", type=int, default=2025, help='Random seed.')
    parser.add_argument("--n_bins", type=int, default=51, help='Number of bins for binning.')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size.')
    parser.add_argument("--seq_len", type=int, default=512, help='Sequence length.')
    parser.add_argument("--val_dataset_name", type=str, default='proteomics', help='Validation dataset name.')
    parser.add_argument("--kfold", type=int, default=10, help='K fold')
    parser.add_argument("--learning_rate", type=float, default=0.001, help='Learning rate.')
    parser.add_argument("--n_epochs", type=int, default=10, help='Number of epochs.')
    parser.add_argument("--validate_every", type=int, default=1, help='Validation every n epochs.')
    parser.add_argument("--n_workers", type=int, default=4, help='Number of workers for data loading.')

    args = parser.parse_args()

    SEED = args.seed
    DATA_PATH = args.data_path
    CKPT_FOLDER = args.ckpt_folder
    K_FOLD = args.kfold
    N_BINS = args.n_bins
    BATCH_SIZE = args.batch_size
    VAL_DATASET_NAME = args.val_dataset_name
    PAD_TOKEN_ID = 0
    MASK_TOKEN_ID = 1
    SPECIAL_TOKENS = [PAD_TOKEN_ID, MASK_TOKEN_ID]

    N_CLASSES = N_BINS + len(SPECIAL_TOKENS)
    SEQ_LEN = args.seq_len
    LEARNING_RATE = args.learning_rate
    N_EPOCHS = args.n_epochs
    VALIDATE_EVERY = args.validate_every
    N_WORKERS = args.n_workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(SEED)

    DATA_FILES = [
        '20231023_092657_imputed_drugresponse.csv',
        '20231023_092657_imputed_metabolomics.csv',
        '20231023_092657_imputed_proteomics.csv',
        '20231023_092657_imputed_methylation.csv',
        '20231023_092657_imputed_transcriptomics.csv',
    ]

    X = []
    X_masked = []
    feature_names = []
    for data_id, data_file_name in enumerate(DATA_FILES):
        data = pd.read_csv(os.path.join(DATA_PATH, data_file_name), index_col=0)
        feature_names += data.columns.tolist()  # Collect all feature names
        data.rename(columns=lambda x: f'{data_id}_{x}', inplace=True)   # Rename columns to keep track of data source
                                                                        # Check SCOmicsData class for more details
        if VAL_DATASET_NAME in data_file_name:
            X_masked.append(data)
        else:
            X.append(data)

    X_masked = X_masked[0]    # Only proteomics data is to be masked
    X = pd.concat(X, axis=1)
    feature_names = list(sorted(set(feature_names)))
    feature_map = {name: i for i, name in enumerate(feature_names)}

    transforms = Compose([
        DataTransform('X_masked_names', 'X_masked_source', SourceNameExtractor()),
        DataTransform('X_masked_names', 'X_masked_names', FeatureIdExtractor(feature_map)),
        DataTransform('X_names', 'X_source', SourceNameExtractor()),
        DataTransform('X_names', 'X_names', FeatureIdExtractor(feature_map)),
        BinningTransform('X', 'X_bin', 'X_source', N_BINS),
        BinningTransform('X_masked', 'X_masked_bin', 'X_masked_source', N_BINS),
    ])

    # K-Fold split
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
    for i, (train_indices, val_indices) in enumerate(kf.split(X_masked)):
        print(f"Fold {i}")
        print("TRAIN:", len(train_indices), "VAL:", len(val_indices))

        model = TransformerModel(
            ntoken=len(feature_names) + len(SPECIAL_TOKENS),
            n_input_bins=N_CLASSES,
            n_sources=len(DATA_FILES) + len(SPECIAL_TOKENS),
            d_model=512,
            nhead=8,
            d_hid=512,
            nlayers=4,
            dropout=0.2,
            pad_token_id=PAD_TOKEN_ID,
        )

        training_data = SCOmicsData(X_masked.iloc[train_indices], X.iloc[train_indices], transforms)
        validation_data = SCOmicsData(X_masked.iloc[val_indices], X.iloc[val_indices], transforms)

        ckpt_path = os.path.join(CKPT_FOLDER, f"model_fold_{i}")
        fit(model, training_data, validation_data, device, ckpt_path)