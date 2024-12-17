from tile_bounds import tile_bounds
tiles_dict = tile_bounds()


from tile_coordinates import tile_coordinates
tile = 0
coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)


from plot_tiles import plot_tiles
plot_tiles(tiles_dict)