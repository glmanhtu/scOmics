import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class SCOmicsData(Dataset):
    def __init__(self, X_masked, X,  transform=None):
        """
        A Simple dataset that holds the masked data and the original data.
        Each sample of the dataset is a row (or a single cell) composed of the masked data and the original data.
        When iterating over the dataset, it returns a transformed dictionary of the masked data and the original data.

        :param X_masked: DataFrame of the data to be masked (proteomics)
        :param X: DataFrame of the data to be used as input (other omics)
        :param transform: a function to transform the data, should be the composition of the transforms
        """
        self.X_masked = X_masked
        self.X = X
        self.transform = transform

        self.items = []
        for i in tqdm(range(len(self)), desc="Preparing dataset"):
            X_masked_row = self.X_masked.iloc[i]
            X_row = self.X.iloc[i]

            assert X_masked_row.name == X_row.name

            results = {
                'X_masked': X_masked_row.values.astype(np.float32),
                'X_masked_names': self.X_masked.columns.tolist(),
                'X': X_row.values.astype(np.float32),
                'X_names': self.X.columns.tolist(),
            }
            if self.transform:
                results = self.transform(results)
            self.items.append(results)

    def __len__(self) -> int:
        return len(self.X_masked)

    def __getitem__(self, idx: int):
       return self.items[idx]


class SCOmicsDataWrapper(Dataset):
    def __init__(self,
                 dataset: SCOmicsData,
                 seq_len: int,
                 pad_token_id: int,
                 mask_token_id: int,
                 token_shifting=0,
                 source_id=-1,
                 n_label_range=(5, 10),
                 eval_mode=False,
                 ):
        """
        A wrapper for the SCOmicsData class to create a dataset for the model.
        Since our dataset is quite small, we split the masked target dataset (proteomics) into multiple chunks,
        then we use the other omics data to predict these masked chunks.

        :param dataset: SCOmicsData
        :param seq_len: sequence length for each sample, will be padded if the length is less than seq_len
        :param pad_token_id: The ID of the padding token.
        :param mask_token_id: The ID of the mask token.
        :param token_shifting: Number of tokens to shift for the input data to avoid overlap with special tokens.
        :param source_id: Filter the source ID for the input data, default to -1 (no filter).
        :param eval_mode: If True, there will be less randomization in the data.
        :param n_label_range: Chunk size range for the masked target data.
        """
        self.dataset = dataset

        self.items = []
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.token_shifting = token_shifting
        self.source_id = source_id
        self.eval_mode = eval_mode
        X_masked_len = len(self.dataset.X_masked.columns.tolist())
        for i in range(len(self.dataset)):
            X_masked_indices = np.arange(X_masked_len)
            np.random.shuffle(X_masked_indices)
            while len(X_masked_indices) > 0:
                n_labels = np.random.randint(*n_label_range)
                indices, X_masked_indices = X_masked_indices[:n_labels], X_masked_indices[n_labels:]
                self.items.append({
                    'idx': i,
                    'item': self.dataset[i],
                    'X_masked_indices': indices
                })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        chunk = self.items[idx]
        item = chunk['item']

        X = item["X"]
        X_bin = item["X_bin"] + self.token_shifting
        X_source = item["X_source"] + self.token_shifting
        X_names = item["X_names"] + self.token_shifting

        if self.source_id != -1:
            mask = item["X_source"] == self.source_id
            X = X[mask]
            X_bin = X_bin[mask]
            X_source = X_source[mask]
            X_names = X_names[mask]

        X_masked = item["X_masked"]
        X_masked_bin = item["X_masked_bin"] + self.token_shifting
        X_masked_source = item["X_masked_source"] + self.token_shifting
        X_masked_names = item["X_masked_names"] + self.token_shifting

        n_labels = len(chunk['X_masked_indices'])
        X_masked_indices = chunk['X_masked_indices']
        X_masked = X_masked[X_masked_indices]
        X_masked_bin = X_masked_bin[X_masked_indices]
        X_masked_source = X_masked_source[X_masked_indices]
        X_masked_names = X_masked_names[X_masked_indices]

        if not self.eval_mode:
            n_X = np.random.randint(int(0.5 * self.seq_len), self.seq_len - n_labels)
            n_X = n_X if n_X < len(X) else len(X)   # to avoid index out of range
            X_indices = np.random.choice(len(X), n_X, replace=False)
        else:
            n_X = min(int(0.7 * self.seq_len), len(X))
            X_indices = np.linspace(0, len(X) - 1, n_X, dtype=int)

        X = X[X_indices]
        X_bin = X_bin[X_indices]
        X_source = X_source[X_indices]
        X_names = X_names[X_indices]

        X_input = np.full((self.seq_len,), self.pad_token_id, dtype=np.int64)
        X_input[:n_X] = X_bin
        X_input[n_X:n_X + n_labels] = self.mask_token_id

        X_input_source = np.full((self.seq_len,), self.pad_token_id, dtype=np.int64)
        X_input_source[:n_X] = X_source
        X_input_source[n_X:n_X + n_labels] = X_masked_source

        X_input_names = np.full((self.seq_len,), self.pad_token_id, dtype=np.int64)
        X_input_names[:n_X] = X_names
        X_input_names[n_X:n_X + n_labels] = X_masked_names

        X_labels = np.full((self.seq_len,), self.pad_token_id, dtype=np.int64)
        X_labels[n_X:n_X + n_labels] = X_masked_bin

        X_original_labels = np.full((self.seq_len,), self.pad_token_id, dtype=np.float32)
        X_original_labels[n_X:n_X + n_labels] = X_masked

        X_original_labels_mask = np.full((self.seq_len,), False, dtype=np.bool)
        X_original_labels_mask[n_X:n_X + n_labels] = True

        X_key_padding_mask = np.full((self.seq_len,), True, dtype=np.int64)
        X_key_padding_mask[:n_X + n_labels] = False

        return {
            "X_id": chunk['idx'],
            "X_bin_input": X_input,
            "X_input_source": X_input_source,
            "X_input_names": X_input_names,
            "X_bin_labels": X_labels,
            "X_original_labels": X_original_labels,
            "X_original_labels_mask": X_original_labels_mask,
            "X_key_padding_mask": X_key_padding_mask
        }
