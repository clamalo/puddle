import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from constants import BATCH_SIZE

class PuddleDataset(Dataset):
    def __init__(self, coarse, fine, times, tiles):
        self.coarse = coarse
        self.fine = fine
        self.times = times
        self.tiles = tiles

    def __len__(self):
        return self.coarse.shape[0]

    def __getitem__(self, idx):
        coarse_sample = self.coarse[idx]
        fine_sample = self.fine[idx]
        time_sample = self.times[idx]
        tile_sample = self.tiles[idx]

        # Convert arrays to tensors
        coarse_tensor = torch.tensor(coarse_sample, dtype=torch.float32)
        fine_tensor = torch.tensor(fine_sample, dtype=torch.float32)
        time_tensor = torch.tensor(time_sample, dtype=torch.float32)
        tile_tensor = torch.tensor(tile_sample, dtype=torch.int64)

        return coarse_tensor, fine_tensor, time_tensor, tile_tensor

def create_dataloaders(npz_path):
    # Load the npz file
    data = np.load(npz_path)

    train_coarse = data['train_coarse']  # shape: (N_train, Xc, Yc)
    train_fine = data['train_fine']      # shape: (N_train, Xf, Yf)
    train_times = data['train_times']    # shape: (N_train,)
    train_tiles = data['train_tiles']    # shape: (N_train,)

    test_coarse = data['test_coarse']    # shape: (N_test, Xc, Yc)
    test_fine = data['test_fine']        # shape: (N_test, Xf, Yf)
    test_times = data['test_times']      # shape: (N_test,)
    test_tiles = data['test_tiles']      # shape: (N_test,)

    # Create dataset objects
    train_dataset = PuddleDataset(train_coarse, train_fine, train_times, train_tiles)
    test_dataset = PuddleDataset(test_coarse, test_fine, test_times, test_tiles)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return train_loader, test_loader