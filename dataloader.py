import torch
from torch.utils.data import Dataset, DataLoader

class ClassifyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)
        self.x_data = x
        self.y_data = y
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    