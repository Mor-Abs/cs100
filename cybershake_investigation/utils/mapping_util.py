


import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import Stamen
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar



# All imported sizes are in mm
def mm_to_inches(mm):
    """Convert millimeters to inches."""
    return mm / 25.4

def add_scale_bar(ax, out_epsg,
                  location='lower right',
                  length_fraction = 0.15,
                  font_size = 6):
    """
    Add a scale bar to the map using GeoPandas and matplotlib_scalebar.

    Parameters:
    - ax: Matplotlib axis object where the scale bar will be added.
    - out_epsg: EPSG code for the projection to calculate accurate distances.
    - location: Location of the scale bar on the map (default: 'lower right').
    """
    crs = "EPSG:4326" #input wgs1984
    crs = "EPSG:2193" #input New Zealand Transverse Mercator
    # Define two points in geographic coordinates
    x1, x2, y1, y2 = ax.axis()
    mid_lat = (y1 + y2) / 2
    p1, p2 = Point(x1, mid_lat), Point(x1 + 1, mid_lat)
    
    # Create a GeoDataFrame and reproject to out_epsg
    datatemp = gpd.GeoDataFrame({
        'geometry': [p1, p2]
    }, crs=crs)
    datatemp = datatemp.to_crs(epsg=out_epsg)
    
    # Calculate the distance between the two points
    distance_meters = datatemp.geometry.iloc[0].distance(datatemp.geometry.iloc[1])
    
    # Add the scale bar
    scalebar = ScaleBar(
        distance_meters,
        units="m",
        location=location,
        length_fraction=length_fraction,  # Scale bar length relative to the axis
        scale_loc="bottom",
        box_color="white",
        box_alpha=0.8,
        font_properties={'size': font_size}
    )
    ax.add_artist(scalebar)
    
def add_lat_lon_markers(ax, interval=5):
    """
    Add latitude and longitude markers to the map.
    - ax: The axis to add the markers.
    - interval: Interval for ticks in degrees.
    """
    gl = ax.gridlines(
        draw_labels=True, linestyle='--', color='gray', alpha=0.5, linewidth=0.5
    )
    gl.xlocator = plt.MultipleLocator(interval)
    gl.ylocator = plt.MultipleLocator(interval)
    gl.xlabel_style = {'size': 8, 'color': 'black'}
    gl.ylabel_style = {'size': 8, 'color': 'black'}


def add_north_arrow(ax, location=(0.95, 0.95), size=10):
    """
    Add a north arrow to the map.
    - ax: The axis to add the north arrow.
    - location: Tuple of (x, y) coordinates in axis fraction (0 to 1).
    - size: Font size of the north arrow label.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    arrow_x = x0 + (x1 - x0) * location[0]
    arrow_y = y0 + (y1 - y0) * location[1]

    ax.annotate(
        'N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - (y1 - y0) * 0.1),
        arrowprops=dict(facecolor='black', arrowstyle='wedge,tail_width=0.5'),
        ha='center', fontsize=size, color='black', transform=ax.transData
    )



# Function 1: Initialize a Base Map
def create_base_map(projection=ccrs.epsg(2193),
                    figsize=(89, 105),
                    majorfont=8,
                    minorfont=6,
                    majormarker_size=6,
                    minormarker_size=3,
                    extent=None,
                    watrebodies=False,
                    majorcities=False,
                    ):
    """
    Create a base map with Cartopy focused on the NZ region

    Parameters:
    - projection: Cartopy CRS for the map projection (default: ccrs.epsg(2193),  
                  New Zealand Transverse Mercator 2000 Projection).
                  Defines the coordinate reference system to use for the map.
    - figsize: Tuple specifying the figure size (width, height) in millimeters.
               This is converted to inches internally (1 inch = 25.4 mm).
    - majorfont: Font size for major city labels (default: 8).
                 Determines the text size for annotations of major cities.
    - minorfont: Font size for minor city labels or additional text (default: 6).
                 Used for smaller labels on the map.
    - majormarker_size: Marker size for major features (default: 6).
                        Determines the size of markers for highlighting important points.
    - minormarker_size: Marker size for minor features (default: 3).
                        Smaller markers for less prominent features.
    - extent: Map extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to New Zealand:
              [165.0, 180.0, -48.0, -33.0]. Specifies the geographic area to display.
    - watrebodies: Boolean (default: False). If True, adds lakes and rivers to the map.
                   Displays prominent water features such as rivers and lakes.
    - majorcities: Boolean (default: False). If True, adds major city markers and labels
                   for Auckland, Christchurch, and Wellington.

    Returns:
    - fig: Matplotlib figure object. Represents the entire figure containing the map.
    - ax: Matplotlib axis object with the specified map projection.
          Contains the plotted map and its features.

    Example Usage:
    ---------------
    # Create a map with major cities and water bodies
    fig, ax = create_base_map(
        figsize=(150, 100),  # Figure size in mm
        majorfont=10,
        minorfont=8,
        extent=[165.0, 180.0, -48.0, -33.0],
        watrebodies=True,
        majorcities=True
    )
    plt.show()
    """
    
    figsize = (mm_to_inches(figsize[0]), mm_to_inches(figsize[1]))
    # Default extent for New Zealand if not provided
    if extent is None:
        # extent = [166.0, 179.0, -48.0, -33.0]  # New Zealand bounding box for "EPSG:4326" usage
        extent = [1000000, 2100000, 4700000, 6500000]  # New Zealand bounding box for "EPSG:2193" usage

    # Calculate the aspect ratio of the map extent
    lon_diff = extent[1] - extent[0]
    lat_diff = extent[3] - extent[2]
    map_aspect_ratio = lat_diff / lon_diff  # Height / Width

    # Figure size (width, height) provided by the user
    fig_width, fig_height = figsize
    fig_aspect_ratio = fig_height / fig_width  # Figure aspect ratio (height / width)

    # Determine the axis size within the figure to preserve aspect ratio
    if fig_aspect_ratio > map_aspect_ratio:  # Taller figure
        height = map_aspect_ratio / fig_aspect_ratio
        ax_position = [0.0, (1.0 - height) / 2.0, 1.0, height]
    else:  # Wider figure
        width = fig_aspect_ratio / map_aspect_ratio
        ax_position = [(1.0 - width) / 2.0, 0.0, width, 1.0]

    # Create the figure
    fig = plt.figure(figsize=figsize)

    # Add a dummy background axis to preserve white space
    bg_ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    bg_ax.set_xticks([])
    bg_ax.set_yticks([])
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)

    # Add the actual map axis
    ax = fig.add_axes(ax_position, projection=projection)

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAND, color='lightgray')
    if watrebodies:
        ax.add_feature(cfeature.LAKES, linewidth=0.5, edgecolor='blue')
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue')
    # Add state boundaries
    ax.add_feature(cfeature.STATES, linestyle='--', edgecolor='gray')
    # Add urban areas
    urban_areas = cfeature.NaturalEarthFeature(
        category='cultural',
        name='urban_areas',
        scale='10m',
        facecolor='tan'
        )
    ax.add_feature(urban_areas, alpha=0.5)
 

    # Add Major Cities
    if majorcities:
        # Define Stations DataFrame with Auckland, Christchurch, Wellington
        stations_df = pd.DataFrame({
            'lon': [174.7633, 172.6362, 174.7762],  # Longitudes in WGS84
            'lat': [-36.8485, -43.5321, -41.2865],  # Latitudes in WGS84
            'name': ['Auckland', 'Christchurch', 'Wellington']  # Station Names
        })

        # Define offsets in WGS84 (degrees) for each city
        city_offsets = {
            'Auckland': {'x_offset': -2.3, 'y_offset': -0.2},  # Degrees
            'Christchurch': {'x_offset': 0.4, 'y_offset': 0.0},  # Degrees
            'Wellington': {'x_offset': 0.4, 'y_offset': -0.7}   # Degrees
        }

        if isinstance(projection, ccrs.PlateCarree):  # If WGS84 is used
            for _, row in stations_df.iterrows():
                ax.plot(
                    row['lon'], row['lat'],
                    marker='o', color='red', markersize=minormarker_size,
                    transform=ccrs.PlateCarree()
                )
                offsets = city_offsets[row['name']]
                ax.text(
                    row['lon'] + offsets['x_offset'], row['lat'] + offsets['y_offset'],
                    row['name'], fontsize=minorfont, transform=ccrs.PlateCarree()
                )
        else:  # If a projected CRS like EPSG:2193 is used
            # Set up a transformer to convert WGS84 to the projection
            # Extract EPSG code or fallback to default
            if hasattr(projection, 'proj4_init'):  # For projections like PlateCarree
                target_crs = projection.proj4_init
            elif hasattr(projection, 'to_epsg'):  # For EPSG-based projections
                target_crs = projection.to_epsg()
            else:
                raise ValueError("Unsupported projection format.")

            transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

            # Transform coordinates for all cities
            stations_df[['x', 'y']] = stations_df.apply(
                lambda row: pd.Series(transformer.transform(row['lon'], row['lat'])), axis=1
            )

            # Transform offsets for each city
            for city, offsets in city_offsets.items():
                transformed_offsets = transformer.transform(
                    stations_df.loc[stations_df['name'] == city, 'lon'].iloc[0] + offsets['x_offset'],
                    stations_df.loc[stations_df['name'] == city, 'lat'].iloc[0] + offsets['y_offset']
                )
                city_offsets[city]['x_offset_transformed'] = transformed_offsets[0] - stations_df.loc[stations_df['name'] == city, 'x'].iloc[0]
                city_offsets[city]['y_offset_transformed'] = transformed_offsets[1] - stations_df.loc[stations_df['name'] == city, 'y'].iloc[0]

            # Plot cities with transformed coordinates and offsets
            for _, row in stations_df.iterrows():
                ax.plot(
                    row['x'], row['y'],
                    marker='o', color='red', markersize=minormarker_size,
                    transform=projection
                )
                offsets = city_offsets[row['name']]
                ax.text(
                    row['x'] + offsets['x_offset_transformed'], row['y'] + offsets['y_offset_transformed'],
                    row['name'], fontsize=minorfont, transform=projection
                )
                

    # Set the extent of the map
    ax.set_extent(extent, crs=projection)
    
    # Add the scale bar
    add_scale_bar(ax, out_epsg=2193, location='lower right', font_size=minorfont)

    # Add the north arrow
    add_north_arrow(ax, location=(0.9, 0.9), size=12)

    # Add latitude and longitude markers
    add_lat_lon_markers(ax, interval=5)

    return fig, ax


# Function 2: Plot Point Data
def add_points(ax, df, lon_col, lat_col, color='red', size=20, label=None):
    """
    Add point data to the map.

    Parameters:
    - ax: Matplotlib axis (map).
    - df: DataFrame containing point data.
    - lon_col: Column name for longitude.
    - lat_col: Column name for latitude.
    - color: Point color (default: red).
    - size: Point size (default: 20).
    - label: Label for the points (default: None).

    Returns:
    - None
    """
    ax.scatter(
        df[lon_col], df[lat_col],
        color=color, s=size, label=label,
        transform=ccrs.PlateCarree()
    )

# Function 3: Plot Raster Data
def add_raster(ax, raster_data, extent, cmap='viridis', alpha=0.6):
    """
    Add raster data (e.g., DEM) to the map.

    Parameters:
    - ax: Matplotlib axis (map).
    - raster_data: 2D NumPy array of raster values.
    - extent: [lon_min, lon_max, lat_min, lat_max] for the raster bounds.
    - cmap: Colormap for the raster (default: viridis).
    - alpha: Opacity of the raster (default: 0.6).

    Returns:
    - None
    """
    ax.imshow(
        raster_data, extent=extent, cmap=cmap, alpha=alpha,
        transform=ccrs.PlateCarree(), origin='upper'
    )

# Function 4: Plot Polygon Data
def add_polygons(ax, gdf, edgecolor='black', facecolor='none', linewidth=0.5):
    """
    Add polygon (boundary) data to the map.

    Parameters:
    - ax: Matplotlib axis (map).
    - gdf: GeoDataFrame containing polygon data.
    - edgecolor: Color for polygon edges (default: black).
    - facecolor: Fill color for polygons (default: none).
    - linewidth: Edge line width (default: 0.5).

    Returns:
    - None
    """
    gdf.plot(
        ax=ax, edgecolor=edgecolor, facecolor=facecolor,
        linewidth=linewidth, transform=ccrs.PlateCarree()
    )
    
import cartopy.io.shapereader as shpreader

# # Load a shapefile
# shapefile = shpreader.Reader('path_to_shapefile.shp')
# for record, geometry in zip(shapefile.records(), shapefile.geometries()):
#     ax.add_geometries([geometry], ccrs.PlateCarree(), edgecolor='black', facecolor='none')


# Function 5: Save Map
def save_map(fig, filename, dpi=300):
    """
    Save the map as a high-DPI image.

    Parameters:
    - fig: Matplotlib figure object.
    - filename: Output file path (e.g., 'map.png').
    - dpi: Resolution of the saved image (default: 300).

    Returns:
    - None
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')

