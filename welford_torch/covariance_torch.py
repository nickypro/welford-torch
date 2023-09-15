# Source: https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html

import numpy as np
import torch

class OnlineCovariance:
    """
    A class to calculate the mean and the covariance matrix
    of the incrementally added, n-dimensional data.
    """
    def __init__(self, order, dtype=torch.float32, device=None):
        """
        Parameters
        ----------
        order: int, The order (=="number of features") of the incrementally added
        dataset and of the resulting covariance matrix.
        dtype: torch.dtype, data type to use for calculations. Default is torch.float32.
        device: str, device to use for calculations. Default is None.
        """
        self._order = order
        self._shape = (order, order)
        self._identity = torch.eye(order, dtype=dtype, device=device)
        self._count = 0
        self._mean = torch.zeros(order, dtype=dtype, device=device)
        self._cov = torch.zeros(self._shape, dtype=dtype, device=device)

    @property
    def count(self):
        """
        int, The number of observations that has been added
        to this instance of OnlineCovariance.
        """
        return self._count

    @property
    def mean(self):
        """
        double, The mean of the added data.
        """
        return self._mean

    @property
    def cov(self):
        """
        tensor, The covariance matrix of the added data.
        """
        return self._cov

    @property
    def corrcoef(self):
        """
        tensor, The normalized covariance matrix of the added data.
        Consists of the Pearson Correlation Coefficients of the data's features.
        """
        if self._count < 1:
            return None
        variances = torch.diagonal(self._cov)
        denominator = torch.sqrt(variances[None, :] * variances[:, None])
        return self._cov / denominator

    def add(self, observation):
        """
        Add the given observation to this object.

        Parameters
        ----------
        observation: tensor, The observation to add.
        """
        if self._order != len(observation):
            raise ValueError(f'Observation to add must be of size {self._order}')

        self._count += 1
        delta_at_nMin1 = observation - self._mean
        self._mean += delta_at_nMin1 / self._count
        weighted_delta_at_n = (observation - self._mean) / self._count

        D_at_n = weighted_delta_at_n.expand(self._shape).T
        D = (delta_at_nMin1 * self._identity).matmul(D_at_n.T)
        self._cov = self._cov * (self._count - 1) / self._count + D

    def merge(self, other):
        """
        Merges the current object and the given other object into a new OnlineCovariance object.

        Parameters
        ----------
        other: OnlineCovariance, The other OnlineCovariance to merge this object with.

        Returns
        -------
        OnlineCovariance
        """
        if other._order != self._order:
            raise ValueError(
                   f'''
                   Cannot merge two OnlineCovariances with different orders.
                   ({self._order} != {other._order})
                   ''')

        merged_cov = OnlineCovariance(self._order)
        merged_cov._count = self.count + other.count
        count_corr = (other.count * self.count) / merged_cov._count
        merged_cov._mean = (self.mean/other.count + other.mean/self.count) * count_corr
        flat_mean_diff = self._mean - other._mean
        mean_diffs = flat_mean_diff.unsqueeze(1).repeat(1, self._order).t()
        merged_cov._cov = (self._cov * self.count \
                           + other._cov * other.count \
                           + mean_diffs * mean_diffs.T * count_corr) \
                          / merged_cov.count
        return merged_cov
