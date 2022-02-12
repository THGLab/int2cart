import torch

class OnehotDigitizer:
    def __init__(self, smearing, binary=True) -> None:
        """
        This helper class turns arrays of float values to the one hot vectors based on the
        specified number of bins in the given range of values.

        Parameters
        ----------
        features: ndarray
            The input array of (n_batch, n_values) shape.

        smearing: tuple
            (start, stop, n_bins)

        binary: bool, optional (default: False)
            if True, returns binary digits, otherwise returns index of class.

        Returns
        -------
        ndarray: The 3D numpy array of (n_batch, n_values, n_bins/1) shape.

        """
        start, stop, n_bins = smearing
        self.nbins = n_bins
        self.offset = torch.linspace(start, stop, n_bins)
        self.binary = binary

    def digitize(self, features):
        original_shape = features.shape
        features = features.flatten()
        diff = torch.abs(features[:,None] - self.offset[None, :])

        digit = torch.argmin(diff, axis=-1)  # n_batch, n_values
        digit = digit.reshape(original_shape)

        if not self.binary:
            return digit
        else:
            digit_size = digit.nelement()
            ohe = torch.zeros((digit_size, self.nbins))
            ohe[torch.arange(digit_size), digit.flatten()] = 1
            ohe = ohe.reshape(tuple(digit.shape) + (self.nbins,))

            return ohe

    def get_reference(self):
        return self.offset.numpy()

    def __call__(self, features):
        return self.digitize(features)
