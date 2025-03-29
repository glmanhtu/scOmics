import argparse
import os.path

import pandas as pd
from sklearn.model_selection import KFold

from data.omics_data import SCOmicsData
from data.preprocessing import DataTransform, Sequential, BinningTransform

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='resources/dataset')
parser.add_argument("--seed", type=int, default=2025, help='Random seed.')
parser.add_argument("--kfold", type=int, default=10, help='K fold')

args = parser.parse_args()

SEED = args.seed
DATA_PATH = args.data_path
K_FOLD = args.kfold

data_files = [
    '20231023_092657_imputed_drugresponse.csv',
    '20231023_092657_imputed_metabolomics.csv',
    '20231023_092657_imputed_proteomics.csv',
    '20231023_092657_imputed_methylation.csv',
    '20231023_092657_imputed_transcriptomics.csv',
]

not_tobe_masked = []
tobe_masked = []
feature_names = []
for data_id, data_file_name in enumerate(data_files):
    data = pd.read_csv(os.path.join(DATA_PATH, data_file_name), index_col=0)
    feature_names += data.columns.tolist()  # Collect all feature names
    data.rename(columns=lambda x: f'{data_id}_{x}', inplace=True)   # Rename columns to keep track of data source
                                                                    # Check SCOmicsData class for more details
    if 'proteomics' in data_file_name:
        tobe_masked.append(data)
    else:
        not_tobe_masked.append(data)

tobe_masked = tobe_masked[0]    # Only proteomics data is to be masked
not_tobe_masked = pd.concat(not_tobe_masked, axis=1)
feature_names = list(sorted(set(feature_names)))
feature_map = {name: i for i, name in enumerate(feature_names)}

transforms = Sequential(
    DataTransform('tobe_masked_names', 'tobe_masked_source', lambda x: int(x.split('_', 1)[0])),
    DataTransform('tobe_masked_names', 'tobe_masked_names', lambda x: feature_map[x.split('_', 1)[1]]),
    DataTransform('not_tobe_masked_names', 'not_tobe_masked_source', lambda x: int(x.split('_')[0])),
    DataTransform('not_tobe_masked_names', 'not_tobe_masked_names', lambda x: feature_map[x.split('_', 1)[1]]),
    BinningTransform('not_tobe_masked', 'not_tobe_masked_bin', 'not_tobe_masked_source', 51),
)

# K-Fold split
kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)
for i, (train_indices, val_indices) in enumerate(kf.split(tobe_masked)):
    print(f"Fold {i}")
    print("TRAIN:", train_indices, "VAL:", val_indices)
    print("TRAIN:", len(train_indices), "VAL:", len(val_indices))


    training_data = SCOmicsData(tobe_masked.iloc[train_indices], not_tobe_masked.iloc[train_indices], transforms)
    print(training_data[0])
    validation_data = SCOmicsData(tobe_masked.iloc[val_indices], not_tobe_masked.iloc[val_indices], transforms)