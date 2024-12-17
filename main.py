from tile_bounds import tile_bounds
tiles = tile_bounds()


from tile_coordinates import tile_coordinates
tile = 0
coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes = tile_coordinates(tile)
print(f"Coarse latitudes: {coarse_latitudes}")
print(f"Coarse longitudes: {coarse_longitudes}")
print(f"Fine latitudes: {fine_latitudes}")
print(f"Fine longitudes: {fine_longitudes}")


from plot_tiles import plot_tiles
plot_tiles(tiles)