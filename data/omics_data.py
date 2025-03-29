from typing import List

import numpy as np
from torch.utils.data import Dataset


class SCOmicsData(Dataset):
    def __init__(self, tobe_masked, not_tobe_masked,  transform=None):
        self.tobe_masked = tobe_masked
        self.not_tobe_masked = not_tobe_masked
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tobe_masked)

    def __getitem__(self, idx: int) -> List:
        tobe_masked_row = self.tobe_masked.iloc[idx]
        not_tobe_masked_row = self.not_tobe_masked.iloc[idx]

        assert tobe_masked_row.name == not_tobe_masked_row.name

        results = {
            'tobe_masked': tobe_masked_row.values.astype(np.float32),
            'tobe_masked_names': self.tobe_masked.columns.tolist(),
            'not_tobe_masked': not_tobe_masked_row.values.astype(np.float32),
            'not_tobe_masked_names': self.not_tobe_masked.columns.tolist(),
        }

        if self.transform:
            results = self.transform(results)
        return results

