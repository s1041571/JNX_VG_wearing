import torch
from torch.utils.data import Dataset

dtype = torch.float32

class WearingDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        # self.X = pre_processing(X)
        self.X = X
        self.y = torch.tensor(y)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image =  self.X[idx]
        image = image[:,:,::-1].transpose(2,0,1)
        image = torch.tensor(image.copy(), dtype=dtype)
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

