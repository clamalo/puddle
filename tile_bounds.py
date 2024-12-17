import json

from constants import MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, TILE_SIZE, FINE_RESOLUTION


def tile_bounds():
    # Determine how many degrees each tile covers
    degree_step = TILE_SIZE * FINE_RESOLUTION
    
    tiles_dict = {}
    tile_counter = 0
    
    current_lat = MIN_LAT
    while current_lat + degree_step <= MAX_LAT:
        lat_upper = current_lat + degree_step
        
        current_lon = MIN_LON
        while current_lon + degree_step <= MAX_LON:
            lon_upper = current_lon + degree_step
            
            # Store the full tile
            tiles_dict[tile_counter] = [[current_lat, lat_upper],
                                   [current_lon, lon_upper]]
            tile_counter += 1
            
            current_lon += degree_step
            
        current_lat += degree_step
    
    return tiles_dict