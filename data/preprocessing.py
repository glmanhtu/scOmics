from typing import Dict, Any

import numpy as np
from torch import nn


def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


class Sequential(nn.Sequential):
    r"""A container to host a sequence of text transforms."""

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input


class DataTransform(nn.Module):
    def __init__(self, from_key: str, to_key: str, transform_fn):
        """
        Extract the information from the input data, transform it, and save it to the output data.

        :param from_key: the key to extract the information from the input data
        :param to_key: the key to store the transformed information
        """
        super().__init__()
        self.from_key = from_key
        self.to_key = to_key
        self.transform_fn = transform_fn

    def forward(self, item):
        data_source = [self.transform_fn(x) for x in item[self.from_key]]
        item[self.to_key] = np.array(data_source)
        return item


class BinningTransform(nn.Module):
    def __init__(self, from_key: str, to_key: str, group_key: str, n_bins: int):
        """
        Binning transform.

        :param from_key: the key to extract the information from the input data
        :param to_key: the key to store the transformed information
        :param group_key: the key to extract the group information (i.e. omics modality) from the input data
        :param n_bins: number of bins
        """
        super().__init__()
        self.from_key = from_key
        self.to_key = to_key
        self.group_key = group_key
        self.n_bins = n_bins

    def forward(self, item):
        data_source = item[self.from_key]
        group_source = item[self.group_key]
        assert data_source.shape[0] == group_source.shape[0]

        # Split the data into groups
        unique_groups = np.unique(group_source)
        binned_data = np.zeros_like(data_source, dtype=np.int64)
        for group in unique_groups:
            group_indices = np.where(group_source == group)[0]
            group_data = data_source[group_indices]

            non_zero_ids = group_data.nonzero()
            non_zero_row = group_data[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))

            non_zero_digits = _digitize(non_zero_row, bins)
            assert non_zero_digits.min() >= 1
            assert non_zero_digits.max() <= self.n_bins - 1
            binned_row = np.zeros_like(group_data, dtype=np.int64)
            binned_row[non_zero_ids] = non_zero_digits
            binned_data[group_indices] = binned_row

        item[self.to_key] = binned_data
        return item
