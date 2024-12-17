import numpy as np

from constants import COARSE_RESOLUTION, FINE_RESOLUTION, PADDING
from tile_bounds import tile_bounds


def tile_coordinates(tile):

    bounds = tile_bounds()[tile]

    min_lat, max_lat = bounds[0]
    min_lon, max_lon = bounds[1]

    # Generate fine-resolution arrays without including the max endpoints (no padding)
    fine_latitudes = np.arange(min_lat, max_lat, FINE_RESOLUTION)
    fine_longitudes = np.arange(min_lon, max_lon, FINE_RESOLUTION)

    # Apply padding only to coarse arrays if pad is a float
    if isinstance(PADDING, (float, int)):
        coarse_min_lat = min_lat - PADDING
        coarse_max_lat = max_lat + PADDING
        coarse_min_lon = min_lon - PADDING
        coarse_max_lon = max_lon + PADDING
    else:
        # No padding
        coarse_min_lat = min_lat
        coarse_max_lat = max_lat
        coarse_min_lon = min_lon
        coarse_max_lon = max_lon

    # Generate coarse-resolution arrays without including the max endpoints
    coarse_latitudes = np.arange(coarse_min_lat, coarse_max_lat, COARSE_RESOLUTION)
    coarse_longitudes = np.arange(coarse_min_lon, coarse_max_lon, COARSE_RESOLUTION)

    return coarse_latitudes, coarse_longitudes, fine_latitudes, fine_longitudes