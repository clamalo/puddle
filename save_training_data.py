import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm

from constants import RAW_DIRECTORY
from tile_coordinates import tile_coordinates

def save_training_data(tiles, start, end, save_path='data.npz'):
    """
    Saves training data from monthly NetCDF files into a .npz file.
    
    Steps:
    1. Iterate over each month between start and end (inclusive).
    2. For each month, load the "tp" variable from the corresponding NetCDF file.
    3. For each tile, interpolate coarse and fine grids using tile_coordinates().
    4. Accumulate all months of data into coarse/fine arrays and a times array.
    5. Shuffle all samples along the time dimension with a fixed seed, ensuring that
       coarse, fine, and times arrays are shuffled in the exact same way.
    6. Split into train and test sets (0.8/0.2 split) along the time dimension.
    7. Merge tiles and times dimensions and create train_tiles/test_tiles arrays.
    8. Shuffle all merged samples again along their sample dimension in unison.
    9. Save all arrays to a .npz file.

    Inputs:
    - tiles: list of tile indices (e.g. [0,1,2])
    - start: tuple (year, month) inclusive, e.g. (1979, 10)
    - end: tuple (year, month) inclusive, e.g. (1980, 2)
    - save_path: path to the .npz file

    Output:
    - Saves a .npz file containing:
      train_coarse, test_coarse, train_fine, test_fine,
      train_times, test_times, train_tiles, test_tiles
    """
    
    # Helper to iterate over months
    def month_range(start_ym, end_ym):
        start_y, start_m = start_ym
        end_y, end_m = end_ym
        current_y, current_m = start_y, start_m
        while (current_y < end_y) or (current_y == end_y and current_m <= end_m):
            yield current_y, current_m
            current_m += 1
            if current_m > 12:
                current_m = 1
                current_y += 1

    all_months = list(month_range(start, end))
    total_months = len(all_months)
    total_steps = total_months * len(tiles)

    # Determine shapes from a sample tile
    sample_tile = tiles[0]
    c_lat, c_lon, f_lat, f_lon = tile_coordinates(sample_tile)
    coarse_shape = (len(c_lat), len(c_lon))
    fine_shape = (len(f_lat), len(f_lon))

    all_times = []
    coarse_data_list = []
    fine_data_list = []

    # Load data month by month and interpolate
    with tqdm(total=total_steps, desc="Processing tiles over months") as pbar:
        for (year, month) in all_months:
            fn = os.path.join(RAW_DIRECTORY, f"{year}-{month:02d}.nc")
            if not os.path.exists(fn):
                # If file doesn't exist, just update progress for each tile and continue
                for _ in tiles:
                    pbar.update(1)
                continue

            ds = xr.open_dataset(fn)
            tp = ds['tp']
            times = ds['time'].values  # numpy array of times

            month_coarse = np.empty((len(tiles), len(times), coarse_shape[0], coarse_shape[1]), dtype=np.float32)
            month_fine = np.empty((len(tiles), len(times), fine_shape[0], fine_shape[1]), dtype=np.float32)

            for i, tile_id in enumerate(tiles):
                c_lat, c_lon, f_lat, f_lon = tile_coordinates(tile_id)

                coarse_tp = tp.interp(lat=c_lat, lon=c_lon)
                month_coarse[i] = coarse_tp.values

                fine_tp = tp.interp(lat=f_lat, lon=f_lon)
                month_fine[i] = fine_tp.values

                pbar.update(1)

            ds.close()

            all_times.append(times)
            coarse_data_list.append(month_coarse)
            fine_data_list.append(month_fine)

    if len(coarse_data_list) == 0:
        raise ValueError("No data found for the given date range.")

    # Concatenate all months along the time dimension
    # shape: (n_tiles, total_samples, Xc, Yc)
    coarse_all = np.concatenate(coarse_data_list, axis=1)
    fine_all = np.concatenate(fine_data_list, axis=1)
    all_times = np.concatenate(all_times)  # shape: (total_samples,)

    n_tiles = len(tiles)
    n_samples = coarse_all.shape[1]

    # Shuffle along the time dimension (same shuffle for coarse, fine, and times)
    np.random.seed(0)
    perm = np.random.permutation(n_samples)
    coarse_all = coarse_all[:, perm, :, :]
    fine_all = fine_all[:, perm, :, :]
    all_times = all_times[perm]

    # Train/test split
    split_idx = int(0.8 * n_samples)
    train_coarse = coarse_all[:, :split_idx, :, :]
    test_coarse = coarse_all[:, split_idx:, :, :]
    train_fine = fine_all[:, :split_idx, :, :]
    test_fine = fine_all[:, split_idx:, :, :]
    train_times = all_times[:split_idx]
    test_times = all_times[split_idx:]

    n_train = train_coarse.shape[1]
    n_test = test_coarse.shape[1]

    # Merge tile and time dimensions
    # After merging: shape: (n_tiles*n_train, Xc, Yc) for train_coarse, etc.
    train_coarse_merged = train_coarse.reshape(n_tiles * n_train, *train_coarse.shape[2:])
    test_coarse_merged = test_coarse.reshape(n_tiles * n_test, *test_coarse.shape[2:])
    train_fine_merged = train_fine.reshape(n_tiles * n_train, *train_fine.shape[2:])
    test_fine_merged = test_fine.reshape(n_tiles * n_test, *test_fine.shape[2:])

    # Expand times and tiles arrays
    train_times_merged = np.tile(train_times, n_tiles)
    test_times_merged = np.tile(test_times, n_tiles)
    train_tiles_merged = np.repeat(tiles, n_train)
    test_tiles_merged = np.repeat(tiles, n_test)

    # Shuffle again after merging (same shuffle for all train arrays and separately for all test arrays)
    np.random.seed(0)
    perm_train = np.random.permutation(n_tiles * n_train)
    train_coarse_merged = train_coarse_merged[perm_train]
    train_fine_merged = train_fine_merged[perm_train]
    train_times_merged = train_times_merged[perm_train]
    train_tiles_merged = train_tiles_merged[perm_train]

    perm_test = np.random.permutation(n_tiles * n_test)
    test_coarse_merged = test_coarse_merged[perm_test]
    test_fine_merged = test_fine_merged[perm_test]
    test_times_merged = test_times_merged[perm_test]
    test_tiles_merged = test_tiles_merged[perm_test]

    # Save final arrays
    np.savez(
        save_path,
        train_coarse=train_coarse_merged,
        test_coarse=test_coarse_merged,
        train_fine=train_fine_merged,
        test_fine=test_fine_merged,
        train_times=train_times_merged,
        test_times=test_times_merged,
        train_tiles=train_tiles_merged,
        test_tiles=test_tiles_merged
    )