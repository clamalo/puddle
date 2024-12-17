import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from tqdm import tqdm

from constants import RAW_DIRECTORY
from tile_coordinates import tile_coordinates

def save_training_data(tiles, start, end, save_path='data.npz'):
    """
    Data flow (as requested):
    1. Load monthly data and interpolate to each tile, accumulating (n_tiles, n_samples, X, Y).
    2. Shuffle all samples in the same way across tiles with respect to the time dimension (so that the same times are rearranged consistently for all tiles).
    3. Train/test split while still having separate tile and time dimensions (i.e., 4D arrays).
    4. Merge tile and sample dimensions after splitting.
    5. Shuffle again after merging tile and sample dimension (train and test sets are shuffled separately to ensure full mixing).

    Inputs:
    - tiles: list of tile indices (e.g. [0,1,2])
    - start: tuple (year, month) inclusive
    - end: tuple (year, month) inclusive

    Output:
    - Saves a npz file with:
      train_coarse, test_coarse, train_fine, test_fine,
      train_times, test_times, train_tiles, test_tiles
    """

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

    # Load and interpolate data month by month
    with tqdm(total=total_steps, desc="Processing tiles over months") as pbar:
        for (year, month) in all_months:
            fn = os.path.join(RAW_DIRECTORY, f"{year}-{month:02d}.nc")
            if not os.path.exists(fn):
                # File not found, still update progress for each tile
                for _ in tiles:
                    pbar.update(1)
                continue

            ds = xr.open_dataset(fn)
            tp = ds['tp']
            times = ds['time'].values

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

    # Concatenate along time dimension
    # shape: (n_tiles, total_samples, X, Y)
    coarse_all = np.concatenate(coarse_data_list, axis=1)
    fine_all = np.concatenate(fine_data_list, axis=1)
    all_times = np.concatenate(all_times)  # (total_samples,)

    n_tiles = len(tiles)
    n_samples = coarse_all.shape[1]

    # 1) Shuffle all samples in the same way across tiles with respect to time dimension
    np.random.seed(0)
    perm = np.random.permutation(n_samples)
    coarse_all = coarse_all[:, perm, :, :]
    fine_all = fine_all[:, perm, :, :]
    all_times = all_times[perm]

    # 2) Train-test split (before merging tile and sample dimensions)
    split_idx = int(0.8 * n_samples)
    train_coarse = coarse_all[:, :split_idx, :, :]
    test_coarse = coarse_all[:, split_idx:, :, :]

    train_fine = fine_all[:, :split_idx, :, :]
    test_fine = fine_all[:, split_idx:, :, :]

    train_times = all_times[:split_idx]
    test_times = all_times[split_idx:]

    # At this point:
    # train_coarse: (n_tiles, n_train, Xc, Yc)
    # test_coarse:  (n_tiles, n_test, Xc, Yc)
    # train_fine:   (n_tiles, n_train, Xf, Yf)
    # test_fine:    (n_tiles, n_test, Xf, Yf)
    # train_times:  (n_train,)
    # test_times:   (n_test,)

    n_train = train_coarse.shape[1]
    n_test = test_coarse.shape[1]

    # 3) Merge tile dimension with sample dimension AFTER splitting
    train_coarse_merged = train_coarse.reshape(n_tiles * n_train, *train_coarse.shape[2:])
    test_coarse_merged = test_coarse.reshape(n_tiles * n_test, *test_coarse.shape[2:])

    train_fine_merged = train_fine.reshape(n_tiles * n_train, *train_fine.shape[2:])
    test_fine_merged = test_fine.reshape(n_tiles * n_test, *test_fine.shape[2:])

    # Expand times and tiles for merged arrays
    train_times_merged = np.repeat(train_times, n_tiles)
    test_times_merged = np.repeat(test_times, n_tiles)

    train_tiles_merged = np.tile(tiles, n_train)
    test_tiles_merged = np.tile(tiles, n_test)

    # 4) Shuffle again after merging
    # Shuffle train set
    perm_train = np.random.permutation(n_tiles * n_train)
    train_coarse_merged = train_coarse_merged[perm_train]
    train_fine_merged = train_fine_merged[perm_train]
    train_times_merged = train_times_merged[perm_train]
    train_tiles_merged = train_tiles_merged[perm_train]

    # Shuffle test set
    perm_test = np.random.permutation(n_tiles * n_test)
    test_coarse_merged = test_coarse_merged[perm_test]
    test_fine_merged = test_fine_merged[perm_test]
    test_times_merged = test_times_merged[perm_test]
    test_tiles_merged = test_tiles_merged[perm_test]

    # Save final data
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

if __name__ == "__main__":
    tiles = list(range(0, 3))
    start_month = (2020, 10)
    end_month = (2021, 9)
    save_training_data(tiles, start_month, end_month)

    import numpy as np
    file = "data.npz"
    data = np.load(file)
    print(data['train_coarse'].shape)
    print(data['test_coarse'].shape)
    print(data['train_fine'].shape)
    print(data['test_fine'].shape)
    print(data['train_times'].shape, data['train_times'][0:10])
    print(data['test_times'].shape, data['test_times'][0:10])
    print(data['train_tiles'].shape, data['train_tiles'][0:10])
    print(data['test_tiles'].shape, data['test_tiles'][0:10])

    # print the number of unique times in train_times
    print(len(np.unique(data['train_times'])))
    # print the number of unique times in test_times
    print(len(np.unique(data['test_times'])))