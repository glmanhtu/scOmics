import argparse
import os.path

import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data.omics_data import SCOmicsData
from data.preprocessing import DataTransform, Sequential, BinningTransform

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='resources/dataset')
parser.add_argument("--seed", type=int, default=2025, help='Random seed.')
parser.add_argument("--n_bins", type=int, default=51, help='Number of bins for binning.')
parser.add_argument("--batch_size", type=int, default=128, help='Batch size.')
parser.add_argument("--seq_len", type=int, default=512, help='Sequence length.')
parser.add_argument("--kfold", type=int, default=10, help='K fold')

args = parser.parse_args()

SEED = args.seed
DATA_PATH = args.data_path
K_FOLD = args.kfold
N_BINS = args.n_bins
BATCH_SIZE = args.batch_size

data_files = [
    '20231023_092657_imputed_drugresponse.csv',
    '20231023_092657_imputed_metabolomics.csv',
    '20231023_092657_imputed_proteomics.csv',
    '20231023_092657_imputed_methylation.csv',
    '20231023_092657_imputed_transcriptomics.csv',
]

X = []
X_masked = []
feature_names = []
for data_id, data_file_name in enumerate(data_files):
    data = pd.read_csv(os.path.join(DATA_PATH, data_file_name), index_col=0)
    feature_names += data.columns.tolist()  # Collect all feature names
    data.rename(columns=lambda x: f'{data_id}_{x}', inplace=True)   # Rename columns to keep track of data source
                                                                    # Check SCOmicsData class for more details
    if 'proteomics' in data_file_name:
        X_masked.append(data)
    else:
        X.append(data)

X_masked = X_masked[0]    # Only proteomics data is to be masked
X = pd.concat(X, axis=1)
feature_names = list(sorted(set(feature_names)))
feature_map = {name: i for i, name in enumerate(feature_names)}

transforms = Sequential(
    DataTransform('X_masked_names', 'X_masked_source', lambda x: int(x.split('_', 1)[0])),
    DataTransform('X_masked_names', 'X_masked_names', lambda x: feature_map[x.split('_', 1)[1]]),
    DataTransform('X_names', 'X_source', lambda x: int(x.split('_')[0])),
    DataTransform('X_names', 'X_names', lambda x: feature_map[x.split('_', 1)[1]]),
    BinningTransform('X', 'X_bin', 'X_source', N_BINS),
    BinningTransform('X_masked', 'X_masked_bin', 'X_masked_source', N_BINS),
)

# K-Fold split
kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
for i, (train_indices, val_indices) in enumerate(kf.split(X_masked)):
    print(f"Fold {i}")
    print("TRAIN:", train_indices, "VAL:", val_indices)
    print("TRAIN:", len(train_indices), "VAL:", len(val_indices))


    training_data = SCOmicsData(X_masked.iloc[train_indices], X.iloc[train_indices], transforms)
    data_loader = DataLoader(training_data, batch_size=BATCH_SIZE)


    validation_data = SCOmicsData(X_masked.iloc[val_indices], X.iloc[val_indices], transforms)