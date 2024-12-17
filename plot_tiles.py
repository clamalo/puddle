import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import os

from constants import FIGURES_DIRECTORY


def plot_tiles(tiles_dict, buffer_deg=2.0, filename="tiles.png"):
    # Determine overall bounds
    lats = []
    lons = []
    for _, ((lat_min, lat_max), (lon_min, lon_max)) in tiles_dict.items():
        lats.extend([lat_min, lat_max])
        lons.extend([lon_min, lon_max])
    
    lat_min_global = min(lats) - buffer_deg
    lat_max_global = max(lats) + buffer_deg
    lon_min_global = min(lons) - buffer_deg
    lon_max_global = max(lons) + buffer_deg

    # Create figure and axis with a PlateCarree projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set the dynamic extent with buffer
    ax.set_extent([lon_min_global, lon_max_global, lat_min_global, lat_max_global], ccrs.PlateCarree())
    
    # Add some geographical features for context
    ax.coastlines(linewidth=2, zorder=1)
    ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=2, zorder=1)

    # Plot each tile above coastlines and states
    for tile_id, bounds in tiles_dict.items():
        lat_range, lon_range = bounds
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range
        
        # Compute the width and height of the tile
        width = lon_max - lon_min
        height = lat_max - lat_min
        
        # Create a red rectangle to represent the tile
        rect = Rectangle(
            (lon_min, lat_min), width, height,
            facecolor='none', edgecolor='red', linewidth=2,
            transform=ccrs.PlateCarree(), zorder=10
        )
        ax.add_patch(rect)
        
        # Add the tile ID at the center of the tile
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2
        ax.text(
            center_lon, center_lat, str(tile_id),
            transform=ccrs.PlateCarree(),
            ha='center', va='center', color='red', fontsize=16, fontweight='bold',
            zorder=10
        )
    
    # Tighten layout to reduce whitespace
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{os.path.join(FIGURES_DIRECTORY, filename)}', dpi=300)