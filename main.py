# from plot_tiles import plot_tiles
# plot_tiles(tiles_dict)


from save_training_data import save_training_data
tiles = list(range(0, 3))
start_month = (2017, 1)
end_month = (2017, 3)
save_training_data(tiles, start_month, end_month)


from create_dataloaders import create_dataloaders
train_loader, test_loader = create_dataloaders('data.npz')
for i, (coarse, fine, times, tiles) in enumerate(train_loader):
    print(coarse.shape, fine.shape, times.shape, tiles.shape)
    if i == 0:
        break