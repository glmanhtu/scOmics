import argparse
import os.path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from data.omics_data import SCOmicsData, SCOmicsDataWrapper
from data.preprocessing import DataTransform, Compose, BinningTransform, SourceNameExtractor, FeatureIdExtractor
from scomics.model.model import TransformerModel
from torchmetrics import MeanMetric, PearsonCorrCoef

from utils.utils import save_ckpt, seed_everything, load_ckpt


def training(net: nn.Module,
             optimizer: Optimizer,
             scaler: torch.amp.GradScaler,
             dataset: SCOmicsData,
             source_id: int,
             device: torch.device):
    train_data = SCOmicsDataWrapper(dataset, SEQ_LEN, PAD_TOKEN_ID, MASK_TOKEN_ID, len(SPECIAL_TOKENS), source_id)
    print("Effective training data size:", len(train_data))
    data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, pin_memory=True)

    net.train()
    train_loss = MeanMetric()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    for index, batch in enumerate(data_loader):
        # Forward pass
        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=AMP):
            output = net(batch)

            pad_mask = batch['X_original_labels_mask']
            preds, actual = output[pad_mask].view(-1), batch['X_original_labels'][pad_mask].view(-1).to(device)

            loss = loss_fn(preds, actual)

        # Backward pass and optimization
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        train_loss.update(loss.item())

        # Print statistics
        if index > 0 and index % 50 == 0:
            print(f"Step {index + 1}/{len(data_loader)}: Average Loss = {train_loss.compute().item():.4f}")
            train_loss.reset()


@torch.no_grad()
def evaluation(net: torch.nn.Module, dataset: SCOmicsData, source_id: int, device: torch.device):
    val_data = SCOmicsDataWrapper(dataset, SEQ_LEN, PAD_TOKEN_ID, MASK_TOKEN_ID, len(SPECIAL_TOKENS), source_id, eval_mode=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

    net.eval()
    pearson_coff_fn = PearsonCorrCoef()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    predictions, actuals = torch.tensor([]), torch.tensor([])
    for index, batch in enumerate(val_loader):
        with torch.amp.autocast(device.type, enabled=AMP):
            output = net(batch)

            pad_mask = batch['X_original_labels_mask']
            preds, actual = output[pad_mask].view(-1),  batch['X_original_labels'][pad_mask].view(-1).to(device)
            predictions = torch.cat((predictions, preds.cpu()), dim=0)
            actuals = torch.cat((actuals, actual.cpu()), dim=0)

    loss = loss_fn(predictions, actuals)
    pearson_coff = pearson_coff_fn(predictions, actuals)

    return loss.item(), pearson_coff.item()


@torch.no_grad()
def testing(net: nn.Module, dataset: SCOmicsData, source_id: int, feature_idx_map, device: torch.device):
    test_data = SCOmicsDataWrapper(dataset, SEQ_LEN, PAD_TOKEN_ID, MASK_TOKEN_ID, len(SPECIAL_TOKENS), source_id, eval_mode=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

    net.eval()
    predictions = {}
    actuals = {}
    for index, batch in enumerate(test_loader):
        with torch.amp.autocast(device.type, enabled=AMP):
            output = net(batch)
            B, _, _ = output.shape
            preds, actual = output, batch['X_original_labels'].to(device)
            label_mask = batch['X_original_labels_mask']

            # Reconstruct the predictions and actual dataframe
            for i in range(B):
                i_preds = preds[i][label_mask[i]].view(-1)
                i_actual = actual[i][label_mask[i]].view(-1)
                features = batch['X_input_names'][i][label_mask[i]] - len(SPECIAL_TOKENS)
                item_id = batch['X_id'][i].item()
                for f_id,  feature in enumerate(features):
                    feature_name = feature_idx_map[feature.item()]
                    predictions.setdefault(feature_name, np.zeros((len(dataset), )))[item_id] = i_preds[f_id].item()
                    actuals.setdefault(feature_name, np.zeros((len(dataset), )))[item_id] = i_actual[f_id].item()

    predictions = pd.DataFrame.from_dict(predictions)
    predictions.index = dataset.X_masked.index
    actuals = pd.DataFrame.from_dict(actuals)
    actuals.index = dataset.X_masked.index
    prediction_torch = torch.tensor(predictions.to_numpy().flatten())
    actuals_torch = torch.tensor(actuals.to_numpy().flatten())
    pearson_coff_fn = PearsonCorrCoef()
    pearson_coff = pearson_coff_fn(prediction_torch, actuals_torch)
    mse_loss = nn.MSELoss(reduction='mean')(prediction_torch, actuals_torch)

    print(f"Test Loss = {mse_loss.item():.4f}, Pearson Coefficient = {pearson_coff.item():.4f}")
    return predictions, actuals


def fit(net: nn.Module, train_dataset: SCOmicsData, val_dataset: SCOmicsData, device: torch.device, ckpt_path: str):
    net = net.to(device)
    optimizer = AdamW(net.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device.type, enabled=AMP)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    best_loss = float('inf')
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}/{N_EPOCHS}")
        training(net, optimizer, scaler, train_dataset, source_id=-1, device=device)

        scheduler.step()
        if epoch % VALIDATE_EVERY != 0:
            continue

        for source_id in range(len(DATA_FILES)):
            source_name = DATA_FILES[source_id].split("_")[-1]
            if VAL_DATASET_NAME in source_name:
                continue
            loss, pearson_coff = evaluation(net, val_dataset, source_id, device)
            print(f"[{source_name}]: Validation Loss = {loss:.4f}, Pearson Coefficient = {pearson_coff:.4f}")

        loss, pearson_coff = evaluation(net, val_dataset, -1, device)
        print(f"[ALL]: Validation Loss = {loss:.4f}, Pearson Coefficient = {pearson_coff:.4f}")
        save_ckpt(f'epoch-{epoch}', net, optimizer, scheduler, ckpt_path)
        if loss < best_loss:
            best_loss = loss
            save_ckpt('best', net, optimizer, scheduler, ckpt_path)
            print(f"Best model saved with loss: {best_loss:.4f} at epoch {epoch + 1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='resources/dataset')
    parser.add_argument("--ckpt_folder", type=str, default='ckpt')
    parser.add_argument("--seed", type=int, default=2025, help='Random seed.')
    parser.add_argument("--n_bins", type=int, default=21, help='Number of bins for binning.')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size.')
    parser.add_argument("--seq_len", type=int, default=512, help='Sequence length.')
    parser.add_argument("--val_dataset_name", type=str, default='proteomics', help='Validation dataset name.')
    parser.add_argument("--kfold", type=int, default=10, help='K fold')
    parser.add_argument("--learning_rate", type=float, default=0.001, help='Learning rate.')
    parser.add_argument("--n_epochs", type=int, default=10, help='Number of epochs.')
    parser.add_argument("--validate_every", type=int, default=1, help='Validation every n epochs.')
    parser.add_argument("--n_workers", type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument("--enable-amp", action="store_true", help='Enable automatic mixed precision.')

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

    SEQ_LEN = args.seq_len
    LEARNING_RATE = args.learning_rate
    N_EPOCHS = args.n_epochs
    VALIDATE_EVERY = args.validate_every
    N_WORKERS = args.n_workers
    AMP = args.enable_amp

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
    feature_idx_map = {i: name for i, name in enumerate(feature_names)}

    transforms = Compose([
        DataTransform('X_masked_names', 'X_masked_source', SourceNameExtractor()),
        DataTransform('X_masked_names', 'X_masked_names', FeatureIdExtractor(feature_map)),
        DataTransform('X_names', 'X_source', SourceNameExtractor()),
        DataTransform('X_names', 'X_names', FeatureIdExtractor(feature_map)),
        BinningTransform('X', 'X_bin', 'X_source', N_BINS),
        BinningTransform('X_masked', 'X_masked_bin', 'X_masked_source', N_BINS),
    ])

    train_indices, val_indices = train_test_split(np.arange(len(X_masked)), test_size=0.1, random_state=SEED)
    train_indices, test_indices = train_indices[5:], train_indices[:5]  # 5 samples for testing and visualization
    print("TRAIN:", len(train_indices), "VAL:", len(val_indices))
    print("TEST indices:", test_indices)

    model = TransformerModel(
        ntoken=len(feature_names) + len(SPECIAL_TOKENS),
        n_input_bins=N_BINS + len(SPECIAL_TOKENS),
        n_sources=len(DATA_FILES) + len(SPECIAL_TOKENS),
        d_model=1024,
        nhead=4,
        d_hid=1024,
        nlayers=4,
        dropout=0.2,
        pad_token_id=PAD_TOKEN_ID,
    )

    training_data = SCOmicsData(X_masked.iloc[train_indices], X.iloc[train_indices], transforms)
    validation_data = SCOmicsData(X_masked.iloc[val_indices], X.iloc[val_indices], transforms)

    fit(model, training_data, validation_data, device, CKPT_FOLDER)

    # Load the best model for evaluating on the test set
    model = load_ckpt(model, "best", CKPT_FOLDER, device)
    test_data = SCOmicsData(X_masked.iloc[test_indices], X.iloc[test_indices], transforms)

    df_preds, df_gt = testing(model, test_data, -1, feature_idx_map, device)
    df_preds.to_csv(os.path.join(CKPT_FOLDER, 'test_predictions.csv'))
    df_gt.to_csv(os.path.join(CKPT_FOLDER, 'test_ground_truth.csv'))

    # Compute MAE dataframe from predictions and ground truth
    mae_df = pd.DataFrame(index=df_preds.index, columns=df_preds.columns)
    for col in df_preds.columns:
        mae_df[col] = np.abs(df_preds[col] - df_gt[col])

    mae_df.to_csv(os.path.join(CKPT_FOLDER, 'test_mae.csv'))