import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from src.tile_utils import tile_coordinates
from src.constants import BATCH_SIZE, RANDOM_SEED

def process_data(training_tiles, start_month, end_month, data_dir=Path('/Volumes/T9/monthly'),
                 elevation_file='elevation.nc',
                 output_path=Path('/Users/clamalo/documents/puddle/data.npz')):
    """
    Process monthly data for specified tiles and date range, and save results to an NPZ file.
    This function also reads the elevation data, interpolates it for each tile at the fine resolution,
    and stores the elevation arrays in the NPZ file as well.
    
    Parameters
    ----------
    training_tiles : list of int
        List of tile indices to process.
    start_month : tuple of (int, int)
        (year, month) of the start month, inclusive.
    end_month : tuple of (int, int)
        (year, month) of the end month, inclusive.
    data_dir : pathlib.Path, optional
        Directory where monthly NetCDF files are stored. Files must be named as 'YYYY-MM.nc'.
    elevation_file : str, optional
        Path to the elevation file (NetCDF).
    output_path : pathlib.Path, optional
        Output path for the resulting NPZ file.
    """
    start_year, start_m = start_month
    end_year, end_m = end_month

    # Validate month inputs
    if not (1 <= start_m <= 12):
        raise ValueError("Start month must be between 1 and 12.")
    if not (1 <= end_m <= 12):
        raise ValueError("End month must be between 1 and 12.")

    def month_less_equal(y1, m1, y2, m2):
        return (y1 < y2) or (y1 == y2 and m1 <= m2)
    
    def month_range(sy, sm, ey, em):
        months = []
        y, m = sy, sm
        while month_less_equal(y, m, ey, em):
            months.append((y, m))
            m += 1
            if m > 12:
                m = 1
                y += 1
        return months

    months_list = month_range(start_year, start_m, end_year, end_m)
    total_steps = len(months_list) * len(training_tiles)
    pbar = tqdm(total=total_steps, desc="Processing data")

    # Load elevation data once
    elevation_ds = xr.open_dataset(elevation_file)

    # Storage lists
    coarse_all = []
    fine_all = []
    times_all = []

    # We'll also store elevation arrays for each tile once.
    # The order of these arrays will match the order of training_tiles.
    tile_elevations = []
    for tile_id in training_tiles:
        _, _, fine_latitudes, fine_longitudes = tile_coordinates(tile_id)
        tile_elev = elevation_ds.interp(Y=fine_latitudes, X=fine_longitudes).fillna(0)
        elev_array = tile_elev.to_array().values.squeeze()
        tile_elevations.append(elev_array)

    # Convert to arrays
    tile_elevations = np.array(tile_elevations)  # shape: (num_tiles, lat, lon)

    # Iterate over all months
    for (year, month) in months_list:
        file_path = data_dir / f"{year}-{month:02d}.nc"
        if not file_path.is_file():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        ds = xr.open_dataset(file_path)

        monthly_coarse = []
        monthly_fine = []
        monthly_times = []

        # Process each tile
        for tile_id in training_tiles:
            coarse_lats, coarse_lons, fine_lats, fine_lons = tile_coordinates(tile_id)

            # Interpolate the dataset to coarse and fine grids for this tile
            coarse_ds = ds.interp(lat=coarse_lats, lon=coarse_lons)
            fine_ds = ds.interp(lat=fine_lats, lon=fine_lons)

            # Convert datasets to arrays
            coarse_array = coarse_ds.to_array().values.transpose(1, 0, 2, 3)  # (time, var, lat, lon)
            fine_array = fine_ds.to_array().values.transpose(1, 0, 2, 3)     # (time, var, lat, lon)

            # Add tile ID as an extra "variable" dimension for coarse
            tile_id_arr = np.full((coarse_array.shape[0], 1, coarse_array.shape[2], coarse_array.shape[3]), tile_id)
            coarse_array = np.concatenate((coarse_array, tile_id_arr), axis=1)

            monthly_coarse.append(coarse_array)
            monthly_fine.append(fine_array)
            monthly_times.append(coarse_ds.time.values)

            pbar.update(1)  # Update progress after each tile

        # Stack results for all tiles in this month
        monthly_coarse = np.stack(monthly_coarse, axis=0)  # (tile, time, var, lat, lon)
        monthly_fine = np.stack(monthly_fine, axis=0)      # (tile, time, var, lat, lon)
        monthly_times = np.stack(monthly_times, axis=0)    # (tile, time)

        coarse_all.append(monthly_coarse)
        fine_all.append(monthly_fine)
        times_all.append(monthly_times)

    pbar.close()

    # Concatenate all months
    coarse = np.concatenate(coarse_all, axis=1)  # (tile, time, var, lat, lon)
    fine = np.concatenate(fine_all, axis=1)      # (tile, time, var, lat, lon)
    times = np.concatenate(times_all, axis=1)    # (tile, time)

    # Use provided random seed for reproducible shuffling
    np.random.seed(RANDOM_SEED)
    num_time = coarse.shape[1]
    indices = np.random.permutation(num_time)
    split_idx = int(0.8 * num_time)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    train_coarse = coarse[:, train_idx]
    test_coarse = coarse[:, test_idx]
    train_fine = fine[:, train_idx]
    test_fine = fine[:, test_idx]
    train_times = times[:, train_idx]
    test_times = times[:, test_idx]

    # Extract and remove tile dimension (stored in last var dimension of coarse)
    train_tiles = train_coarse[:, :, -1, 0, 0]
    test_tiles = test_coarse[:, :, -1, 0, 0]

    train_coarse = train_coarse[:, :, :-1]  # remove tile var
    test_coarse = test_coarse[:, :, :-1]

    # Reshape: (tile, time, var, lat, lon) -> (tile*time, var, lat, lon)
    def reshape_data(data):
        return data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])

    train_coarse = reshape_data(train_coarse)
    test_coarse = reshape_data(test_coarse)
    train_fine = reshape_data(train_fine)
    test_fine = reshape_data(test_fine)
    train_times = train_times.reshape(-1)
    test_times = test_times.reshape(-1)
    train_tiles = train_tiles.reshape(-1)
    test_tiles = test_tiles.reshape(-1)

    # Save all processed data including elevation
    # We'll save:
    #  - "tiles_unique": the list of unique tiles
    #  - "tile_elevations": array of shape (num_tiles, lat, lon) for each tile in tiles_unique
    tiles_unique = np.array(training_tiles)
    np.savez(
        output_path,
        train_coarse=train_coarse,
        test_coarse=test_coarse,
        train_fine=train_fine,
        test_fine=test_fine,
        train_times=train_times,
        test_times=test_times,
        train_tiles=train_tiles,
        test_tiles=test_tiles,
        tiles_unique=tiles_unique,
        tile_elevations=tile_elevations
    )


class WeatherDataset(Dataset):
    """
    A PyTorch Dataset for coarse/fine weather data and associated metadata.
    """
    def __init__(self, coarse, fine, times, tiles, tiles_unique, tile_elevations):
        """
        Parameters
        ----------
        coarse : np.ndarray
            Coarse-resolution input data, shape: (samples, vars, lat, lon)
        fine : np.ndarray
            Fine-resolution target data, shape: (samples, vars, lat, lon)
        times : np.ndarray
            1D array of time values (np.datetime64).
        tiles : np.ndarray
            1D array of tile IDs associated with each sample.
        tiles_unique : np.ndarray
            Array of unique tile IDs that matches the indexing of tile_elevations.
        tile_elevations : np.ndarray
            Elevation arrays for each tile_id in tiles_unique. Shape: (num_tiles, lat, lon).
        """
        self.coarse = coarse
        self.fine = fine
        self.times = times
        self.tiles = tiles
        self.tiles_unique = tiles_unique
        self.tile_elevations = tile_elevations

        # Create a mapping from tile_id to index in tiles_unique
        self.tile_to_idx = {t: i for i, t in enumerate(self.tiles_unique)}

    def __len__(self):
        return self.coarse.shape[0]

    def __getitem__(self, idx):
        coarse_sample = torch.from_numpy(self.coarse[idx]).float()
        fine_sample = torch.from_numpy(self.fine[idx]).float()

        # Convert datetime64 to seconds since epoch
        epoch = np.datetime64('1970-01-01T00:00:00Z')
        time_delta = self.times[idx] - epoch
        time_seconds = time_delta / np.timedelta64(1, 's')
        time_sample = torch.tensor(time_seconds, dtype=torch.float32)

        tile_id = int(self.tiles[idx])
        tile_sample = torch.tensor(tile_id, dtype=torch.long)

        # Get the elevation array for this tile
        tile_idx = self.tile_to_idx[tile_id]
        elevation_array = self.tile_elevations[tile_idx]
        elevation_sample = torch.from_numpy(elevation_array).float()

        return coarse_sample, fine_sample, time_sample, tile_sample, elevation_sample


def load_dataloaders(file_path):
    """
    Load data from an NPZ file (including elevation arrays) and return train/test PyTorch DataLoaders.
    
    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the NPZ file containing train/test split data and elevation info.

    Returns
    -------
    tuple
        (train_loader, test_loader) as PyTorch DataLoaders.
    """
    data = np.load(file_path)
    tiles_unique = data['tiles_unique']
    tile_elevations = data['tile_elevations']

    train_dataset = WeatherDataset(
        coarse=data['train_coarse'],
        fine=data['train_fine'],
        times=data['train_times'],
        tiles=data['train_tiles'],
        tiles_unique=tiles_unique,
        tile_elevations=tile_elevations
    )

    test_dataset = WeatherDataset(
        coarse=data['test_coarse'],
        fine=data['test_fine'],
        times=data['test_times'],
        tiles=data['test_tiles'],
        tiles_unique=tiles_unique,
        tile_elevations=tile_elevations
    )

    # Set torch random seed for reproducibility in dataloader sampling
    torch.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train_loader, test_loader


training_tiles = [7, 9, 18]
start_month = (1979, 10)
end_month = (1980, 2)
process_data(training_tiles, start_month, end_month)
train_loader, test_loader = load_dataloaders('/Users/clamalo/documents/puddle/data.npz')




from model import UNetWithAttention
model = UNetWithAttention(in_channels=1, out_channels=1)

# Move model to device
device = torch.device('mps')

for batch in train_loader:
    coarse, fine, time, tile, elevation = batch


    sample_to_plot = 0
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature

    tile = tile[sample_to_plot].item()
    coarse_lat, coarse_lon, fine_lat, fine_lon = tile_coordinates(tile)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    cf = ax.pcolormesh(fine_lon, fine_lat, fine[sample_to_plot, 0], transform=ccrs.PlateCarree(), cmap='viridis')
    ax.coastlines()
    ax.add_feature(cartopy.feature.STATES)
    ax.set_title(f"Coarse-resolution data for tile {tile}")
    plt.colorbar(cf, ax=ax)
    plt.show()