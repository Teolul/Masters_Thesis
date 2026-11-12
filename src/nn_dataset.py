import torch
from torch.utils.data import Dataset


# ----------------------------
# Neural Networks dataset definition
# ----------------------------


class MyDataset(Dataset):
    def __init__(self, X, Y):
        """Store paired input and target arrays as float tensors.

        Args:
            X: Sample-major input array.
            Y: Sample-major target array.

        Returns:
            None. Initializes the dataset tensors.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        """Return the number of samples in the dataset.

        Args:
            None.

        Returns:
            Number of stored input samples.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """Return one input and target sample.

        Args:
            idx: Sample index.

        Returns:
            Input tensor and corresponding target tensor.
        """
        return self.X[idx], self.Y[idx]


class MyDataset2(Dataset):
    def __init__(self, X, Y):
        """Store per-function input arrays and targets as float tensors.

        Args:
            X: Sequence of sample-major input arrays, one per function.
            Y: Sample-major target array.

        Returns:
            None. Initializes the dataset tensors.
        """
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        """Return the number of target samples in the dataset.

        Args:
            None.

        Returns:
            Number of stored target samples.
        """
        return len(self.Y)

    def __getitem__(self, idx):
        """Return one sample from each input array and its target.

        Args:
            idx: Sample index.

        Returns:
            List of per-function input tensors and the corresponding target tensor.
        """
        x = [x_func[idx] for x_func in self.X]
        return x, self.Y[idx]
