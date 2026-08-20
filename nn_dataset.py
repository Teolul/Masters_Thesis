import torch
from torch.utils.data import Dataset


# ----------------------------
# Neural Networks dataset definition
# ----------------------------


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MyDataset2(Dataset):
    def __init__(self, X, Y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = [x_func[idx] for x_func in self.X]
        return x, self.Y[idx]