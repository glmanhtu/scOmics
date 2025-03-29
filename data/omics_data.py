from typing import List

import numpy as np
from torch.utils.data import Dataset


class SCOmicsData(Dataset):
    def __init__(self, X_masked, X,  transform=None):
        self.X_masked = X_masked
        self.X = X
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X_masked)

    def __getitem__(self, idx: int) -> List:
        X_masked_row = self.X_masked.iloc[idx]
        X_row = self.X.iloc[idx]

        assert X_masked_row.name == X_row.name

        results = {
            'X_masked': X_masked_row.values.astype(np.float32),
            'X_masked_names': self.X_masked.columns.tolist(),
            'X': X_row.values.astype(np.float32),
            'X_names': self.X.columns.tolist(),
        }

        if self.transform:
            results = self.transform(results)
        return results

