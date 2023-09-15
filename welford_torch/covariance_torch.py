# Source: https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html

import torch

class OnlineCovariance:
    """
    A class to calculate the mean and the covariance matrix
    of the incrementally added, n-dimensional data.
    """
    def __init__(self, elements=None, dtype=torch.float32, device=None):
        """
        Parameters
        ----------
        elements (array(S, D)): data samples.
        dtype (torch.dtype): data type to use for calculations.
            default: torch.float32.
        device: str, device to use for calculations. Default is None.
        """

        self.__dtype = dtype
        self.__detached = False

        # Additional parameters for eigenvalue decomposition & whitening.
        self.__eig_last_count = 0
        self.__eig_vectors = None
        self.__eig_values = None

        if elements is None:
            self.__device = device
            self.__order = None
            self.__shape = None
            self.__count = 0
            self.__mean = None
            self.__cov = None
            self.__identity = None

        else:
            self.init_zeros(elements[0], device)

            for elem in elements:
                self.add(elem)

    def init_zeros(self, element, device=None):
        self.__device = element.device if (device is None) else device
        self.__order = element.shape
        self.__shape = torch.Size((*self.__order, self.__order[-1]))
        self.__count = 0
        self.__mean = torch.zeros(self.__order, dtype=self.__dtype, device=self.__device)
        self.__cov = torch.zeros((self.__shape), dtype=self.__dtype, device=self.__device)
        self.__identity = \
            torch.eye(
                self.__shape[-1], dtype=self.__dtype, device=self.__device
            ).repeat((*self.__shape[:-2], 1, 1))

        if self.__detached:
            self.detach()

    @property
    def count(self):
        """
        int, The number of observations that has been added
        to this instance of OnlineCovariance.
        """
        return self.__count

    @property
    def mean(self):
        """
        double, The mean of the added data.
        """
        return self.__mean

    @property
    def cov(self):
        """
        tensor, The covariance matrix of the added data.
        """
        return self.__cov

    @property
    def corrcoef(self):
        """
        tensor, The normalized covariance matrix of the added data.
        Consists of the Pearson Correlation Coefficients of the data's features.
        """
        if self.__count < 1:
            return None
        variances = torch.diagonal(self.__cov, dim1=-2, dim2=-1)
        denominator = torch.sqrt(variances[..., None, :] * variances[..., :, None])
        return self.__cov / denominator

    @property
    def var_p(self):
        """
        tensor, The (population) variance of the added data.
        """
        return self.__getvars(ddof=0)

    @property
    def var_s(self):
        """
        tensor, The (sample) variance of the added data.
        """
        return self.__getvars(ddof=1)

    @property
    def eig_val(self):
        """ tensor, Return the eigenvalues of the covariance matrix. """
        if self.__count < 1:
            return None
        self.__compute_eig()
        return self.__eig_values

    @property
    def eig_vec(self):
        """ tensor, Return the eigenvectors of the covariance matrix. """
        if self.__count < 1:
            return None
        self.__compute_eig()
        return self.__eig_vectors

    @property
    def whit(self):
        """ tensor, The whitening matrix of the added data. """
        if self.__count < 1:
            return None
        return self.__compute_whitening()

    @property
    def whit_inv(self):
        """ tensor, The inverse whitening matrix of the added data. """
        if self.__count < 1:
            return None
        return self.__compute_whitening_inverse()

    @property
    def identity(self):
        """ tensor, The identity in the shape of the added data. """
        return self.__identity

    def add(self, observation):
        """
        Add the given observation to this object.

        Parameters
        ----------
        observation: tensor, The observation to add.
        """
        if self.__shape is None:
            self.init_zeros(observation, device=self.__device)

        assert observation.shape == self.__order
        observation = observation.to(self.__dtype).to(self.__device)

        self.__count += 1
        delta_at_nMin1 = observation - self.__mean
        self.__mean += delta_at_nMin1 / self.__count
        weighted_delta_at_n = (observation - self.__mean) / self.__count

        D_at_n = self.__expand_last_dim(weighted_delta_at_n).transpose(-2, -1)
        D = (self.__expand_last_dim(delta_at_nMin1) * self.__identity) @ D_at_n
        self.__cov = self.__cov * (self.__count - 1) / self.__count + D

    def add_all(self, elements):
        """ add_all

        add multiple data samples.

        Args:
            elements (array(S, D)): data samples.
            backup_flg (boolean): if True, backup previous state for rollbacking.

        """
        # backup for rollbacking
        elements = elements.to(dtype=self.__dtype, device=self.__device)
        for elem in elements:
            self.add(elem)

    def merge(self, other):
        """
        Merges the current object and the given other object into the current object.

        Parameters
        ----------
        other: OnlineCovariance, The other OnlineCovariance to merge this object with.

        Returns
        -------
        self
        """
        if other.__order != self.__order:
            raise ValueError(
                   f'''
                   Cannot merge two OnlineCovariances with different orders.
                   ({self.__order} != {other.__order})
                   ''')

        assert other.__shape == self.__shape
        assert other.__dtype == self.__dtype

        # Compute the merged covariance matrix.
        __merged_count = self.count + other.count

        count_corr = (other.count * self.count) / __merged_count
        __merged_mean = (self.mean/other.count + other.mean/self.count) * count_corr

        flat_mean_diff = self.__mean - other.__mean
        mean_diffs = self.__expand_last_dim(flat_mean_diff)
        __merged_cov = (self.__cov * self.count \
                           + other.__cov * other.count \
                           + mean_diffs * mean_diffs.transpose(-2, -1) * count_corr) \
                          / __merged_count

        # Update the current object.
        self.__count = __merged_count
        self.__mean = __merged_mean
        self.__cov = __merged_cov

        return self

    def __getvars(self, ddof=0):
        """
        tensor, The variance of the added data, optionally including degrees of freedom.
        """
        if self.__count <= 0:
            return None
        min_count = ddof
        if self.__count <= min_count:
            return torch.full(self.__order, torch.nan, dtype=self.__dtype).to(self.__device)
        else:
            variances = torch.diagonal(self.__cov, dim1=-2, dim2=-1)
            if ddof == 0:
                return variances
            return variances *  ( self.__count/(self.__count-ddof) )

    def __expand_last_dim(self, tensor):
        """
        Expand the last dimension of the given tensor.

        Example: tensor [2, 3, 4] (__order) -> expanded_tensor [2, 3, 4, 4] (__shape)
        """
        shape = tensor.shape # __order
        repeat_shape = ( *( [1]*len(shape) ), shape[-1] ) # 1, ..., 1, n
        return tensor.unsqueeze(-1).repeat(repeat_shape)

    def __compute_eig(self):
        """
        Compute the eigenvalues and eigenvectors of the covariance matrix.

        Returns:
            eigenvalues: tensor, The eigenvalues of the covariance matrix.
            eigenvectors: tensor, The eigenvectors of the covariance matrix.
        """
        # Check if we have already computed them
        if self.__eig_last_count == self.__count:
            return self.__eig_values, self.__eig_vectors
        self.__eig_last_count = self.__count

        # Initialize the eigenvectors tensor
        flat_shape = (-1, *self.__shape[-2:])
        self.__eig_vectors = self.__cov.clone().reshape(flat_shape)
        self.__eig_values = []

        for i in range(self.__eig_vectors.shape[0]):
            sigma = self.__eig_vectors[i]
            eigenvalues, eigenvectors = torch.linalg.eigh(sigma, UPLO='U')
            self.__eig_vectors[i] = eigenvectors
            self.__eig_values.append(eigenvalues)

        self.__eig_vectors = self.__eig_vectors.view(self.__shape)
        self.__eig_values  = torch.stack(self.__eig_values).view(self.__order)

        return self.__eig_values, self.__eig_vectors

    def __compute_whitening(self):
        """ Compute the whitening matrix """
        val, vec = self.__compute_eig()
        D_inv_sqrt = self.__expand_last_dim(1.0 / torch.sqrt(val)) * self.__identity
        return (vec @ D_inv_sqrt @ vec.transpose(-2, -1))

    def __compute_whitening_inverse(self):
        """ Compute inverse of the whitening matrix """
        val, vec = self.__compute_eig()
        D_sqrt = self.__expand_last_dim(torch.sqrt(val)) * self.__identity
        return (vec @ D_sqrt @ vec.transpose(-2, -1))

    def detach(self):
        self.__detached = True
        if not (self.__shape is None):
            self.__mean = self.__mean.detach()
            self.__cov  = self.__cov.detach()
            self.__identity = self.__identity.detach()
        return self
