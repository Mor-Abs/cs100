import pygmt

def add_scale_bar(fig, length="100k", position="jBR+w100k+o0.5c/0.5c"):
    """
    Add a scale bar to the PyGMT map.

    Parameters:
    - fig: The PyGMT figure object.
    - length: Scale bar length (default: "100k" for 100 kilometers).
    - position: Position of the scale bar (default: bottom right corner).
    """
    fig.basemap(map_scale=position)



def add_lat_lon_markers(fig, interval="5"):
    """
    Add latitude and longitude markers to the map.

    Parameters:
    - fig: The PyGMT figure object.
    - interval: Gridline interval in degrees (default: 5).
    """
    fig.basemap(frame=["af", f"x{interval}g{interval}", f"y{interval}g{interval}"])


def add_north_arrow(fig, position="jTR+o0.3c/0.3c", size="1c"):
    """
    Add a north arrow to the map.

    Parameters:
    - fig: The PyGMT figure object.
    - position: Position of the north arrow (default: top-right corner).
    - size: Size of the north arrow (default: "1c").
    """
    fig.basemap(rose=position)

import pygmt

import pygmt

def create_base_map(region, projection="M6i", figsize="6i", majorcities=False):
    """
    Create a base map with PyGMT focused on the specified region.

    Parameters:
    - region: Map extent as [lon_min, lon_max, lat_min, lat_max].
    - projection: PyGMT projection string (default: "M6i" for Mercator).
    - figsize: Figure size (default: "6i").
    - majorcities: Boolean (default: False). If True, adds major city markers.

    Returns:
    - fig: PyGMT figure object with the base map.
    """
    fig = pygmt.Figure()

    # Draw the base map
    fig.basemap(region=region, projection=projection, frame=True)

    # Add coastline and land features
    fig.coast(
        region=region,
        projection=projection,
        land="lightgray",
        water="skyblue",
        borders=["1/0.5p,black"],
        shorelines="1/0.8p,black",
    )

    # Add major cities (if enabled)
    if majorcities:
        cities = {
            "Auckland": {"lon": 174.7633, "lat": -36.8485},
            "Christchurch": {"lon": 172.6362, "lat": -43.5321},
            "Wellington": {"lon": 174.7762, "lat": -41.2865},
        }
        for city, coord in cities.items():
            fig.plot(
                x=coord["lon"],
                y=coord["lat"],
                style="c0.3c",  # Circle symbol with size 0.3 cm
                fill="red",  # Fill color (used instead of 'color')
                pen="black"  # Outline pen: black
            )
            fig.text(
                x=coord["lon"] + 0.5,
                y=coord["lat"] - 0.5,
                text=city,
                font="8p,black"
            )

    return fig
