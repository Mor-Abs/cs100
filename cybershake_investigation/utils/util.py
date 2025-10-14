import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xarray as xr
from tqdm import tqdm
import h5py
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS

# Utilities for file handling and plotting
# Need to update GMT on hypocenter by Jake! Once it is updated, uncomment these lines.
# from pygmt_helper import plots
from pygmt_helper import plotting


# Load Stations Data
def get_station_data(
    stations_ll_path: Path,
    stations_vs30_path: Path,
    stations_z_path: Path,
):
    """
    function for loading the station data including station name,
    long, lat, vs30, Z_1.0(km), Z_2.5(km), sigma
    -------------
    Inputs:
    stations_ll_path: Path,
            Full file path to stations ll file, as configured for empirical_support.py module.
    stations_vs30_path: Path,
            Full file path to stations vs30 file, , as configured for empirical_support.py module.
    stations_z_path: Path,
            Full file path to stations z file, , as configured for empirical_support.py module.

    -------------
    Outputs:
            A dataframe of all combined data.

    """

    with open(stations_ll_path, "r") as temp_file:
        temp_data = [line.strip().split() for line in temp_file]

    stations_df = pd.DataFrame(temp_data, columns=["long", "lat", "station_name"])
    stations_df = stations_df[["station_name", "long", "lat"]]
    del temp_data

    with open(stations_vs30_path, "r") as temp_file:
        temp_data = [line.strip().split() for line in temp_file]

    temp_df = pd.DataFrame(temp_data, columns=["station_name", "vs30"])
    stations_df = stations_df.merge(temp_df, how="inner", on="station_name")

    del temp_data, temp_df

    with open(stations_z_path, "r") as temp_file:
        next(temp_file)
        temp_data = [line.strip().split(",") for line in temp_file]

    temp_df = pd.DataFrame(
        temp_data, columns=["station_name", "Z_1.0(km)", "Z_2.5(km)", "sigma"]
    )
    stations_df = stations_df.merge(temp_df, how="inner", on="station_name")

    return stations_df


def resize_image_to_width(
    image_path, output_path=None, desired_width_cm=8.5, dpi=1200, replace=False
):
    """
    Resize an image to a specific width in centimeters while maintaining the aspect ratio.

    Parameters:
    - image_path (str): Path to the input image file.
    - output_path (str, optional): Path to save the resized image. Ignored if replace=True.
    - desired_width_cm (float): Desired width of the image in centimeters.
    - dpi (int, optional): DPI (dots per inch) to maintain high resolution. Default is 1200.
    - replace (bool, optional): If True, replaces the original image. Default is False.

    Returns:
    - None
    """
    # Convert width from cm to pixels
    desired_width_px = int((desired_width_cm / 2.54) * dpi)

    # Open the image
    img = Image.open(image_path)

    # Calculate new height to maintain aspect ratio
    aspect_ratio = img.height / img.width
    desired_height_px = int(desired_width_px * aspect_ratio)

    # Resize the image
    resized_img = img.resize((desired_width_px, desired_height_px))

    # Determine the output path
    if replace:
        output_path = image_path
    elif not output_path:
        raise ValueError("output_path must be specified if replace is False")

    # Save the resized image
    resized_img.save(output_path, dpi=(dpi, dpi))

    if replace:
        print(f"Original image replaced with resized image at: {output_path}")
    else:
        print(f"Resized image saved at: {output_path}")


def plot_specific_stations(
    st_list: list,
    desired_width_cm: float,
    intended_dpi: float = 900,
    file_name: str = None,
    plots_saving_dir: Path = None,
    plot_using_mapdata: bool = False,
    base_dir: Path = None,
    map_data_ffp: Path = None,
    point_style: str = "a0.2c",
    point_fill: str = "red",
    point_label: bool = True,
    point_pen: str = "0.05p,red",
    save_fig: bool = True,
):
    """
    Function to plot specific stations on NZ map.

    -------------
    Inputs:
    - st_list: list
        A list of station names to be plotted.
    - desired_width_cm: float
        Desired width of the output map image in centimeters.
    - intended_dpi: float, optional (default: 900)
        Desired DPI for the output image.
    - file_name: str
        Name of the file where the figure will be saved. Must be a valid string.
    - plots_saving_dir: Path
        Path to the directory where the plot will be saved. Must be a valid Path object.
    - plot_using_mapdata: bool, optional (default: False)
        If True, loads additional map data (e.g., roads, topography) to overlay on the map.
    - base_dir: Path, optional
        Path to the base directory containing station files (e.g., `.ll`, `.vs30`, `.z`).
        If not provided, defaults to the `base_data` directory under the project root.
    - map_data_ffp: Path, optional
        Full file path to load specific map data. If not provided, defaults to a preset directory.

    -------------
    Outputs:
    - fig: pygmt.Figure
        The generated map figure, with stations plotted and saved to the specified location.

    -------------
    Example Usage:
    ```python
    import os
    from pathlib import Path
    import importlib

    file_dir = os.getcwd()
    root_dir = os.path.dirname(file_dir)
    plots_saving_dir = file_dir + "/plots"
    os.makedirs(plots_saving_dir, exist_ok=True)
    os.chdir(root_dir)

    import utils.util as util

    importlib.reload(util)
    from utils.util import plot_specific_stations

    # List of stations to plot
    st_list = ["HNPS", "TRMS"]

    # Call the function to generate a map
    fig, st_df = plot_specific_stations(
        st_list=st_list,
        desired_width_cm=8.5,
        intended_dpi=900,
        file_name="sstations_map_selected.png",
        plots_saving_dir=Path(plots_saving_dir),
        plot_using_mapdata=False,
    )
    """

    if not isinstance(file_name, str):
        raise ValueError("The 'file_name' must be a string")

    if not isinstance(plots_saving_dir, Path):
        raise ValueError("The 'plots_saving_dir' must be a Path")

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "base_data"

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    station_files = {
        "stations_ll": "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll",
        "stations_vs30": "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30",
        "stations_z": "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.z",
    }

    if not isinstance(st_list, list):
        raise ValueError("The 'st_list' parpameter must be a list")

    # check if the files exisits
    for _, filename in station_files.items():
        file_path = base_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"File [{filename}] not found in base directory [{base_dir}]"
            )

    # get all stations data
    stations_df = get_station_data(
        os.path.join(base_dir, station_files["stations_ll"]),
        os.path.join(base_dir, station_files["stations_vs30"]),
        os.path.join(base_dir, station_files["stations_z"]),
    )

    # extracting the listed stations
    st_df = stations_df[stations_df["station_name"].isin(st_list)]

    if plot_using_mapdata == True:
        # loading the mapdata
        map_data_ffp = Path(__file__).resolve().parent.parent / "qcore/data"
        map_data = (
            plotting.NZMapData.load(map_data_ffp) if map_data_ffp is not None else None
        )
    else:
        map_data = None

    fig = plotting.gen_region_fig(
        region=(165.7, 178.7, -47.5, -34),
        projection="M8.5c",
        map_data=map_data,
        plot_roads=False,
        plot_highways=False,
        plot_topo=True,
        config_options=dict(
            MAP_FRAME_TYPE="graph",
            FORMAT_GEO_MAP="ddd.xx",
            MAP_GRID_PEN="0.5p,gray",
            MAP_TICK_PEN_PRIMARY="1p,black",
            MAP_FRAME_PEN="thinner,black",
            MAP_FRAME_AXES="WSEN",
            FONT_ANNOT_PRIMARY="7p,Helvetica,black",
            FONT_LABEL="7p",  # Font size for axis labels
            FONT_TITLE="9p",  # Font size for the title
        ),
    )
    
    fig.plot(
        x=st_df.long.values.astype(float),
        y=st_df.lat.values.astype(float),
        style=point_style,
        fill=point_fill,
        pen=point_pen,
    )
    
    if point_label:
        fig.text(
        x=st_df.long.values.astype(float),
        y=st_df.lat.values.astype(float),
        text=st_df.station_name.values,  # Station names as labels
        font="6p,Helvetica-Oblique,red",
        justify="LM",  # Align text to the left-middle of the point
        offset="0.2c/0.1c",  # Slight offset to avoid overlapping with stars
        )

    fig.basemap(map_scale="jBR+w200k+o1c/1c", rose="jTL+w1.5c+o0.5c/0.5c")
    
    if save_fig:
        saving_path = plots_saving_dir / file_name
        if saving_path.exists():
            saving_path.unlink()

        fig.savefig(saving_path, dpi=1200)

        resize_image_to_width(
            image_path=saving_path,
            desired_width_cm=desired_width_cm,
            dpi=intended_dpi,
            replace=True,
        )

    fig.show()

    return fig, st_df


def load_sim_data(
    sim_dir: str, faults: list = [], component: str = "geom", proc_flag: bool = False
):
    """
    Function to load simulation data from a specified directory, process it by fault, and organize it into a dictionary.
    Each fault is associated with an xarray.DataArray containing the simulation data.

    -------------
    Inputs:
    sim_dir: str
        Path to the simulation directory. This directory should contain subdirectories for each fault,
        and within those, CSV files with simulation data.
    faults: list, optional
        List of fault names to process. If None or empty, all faults found in the simulation directory will be processed.
    component: str, default="geom"
        The specific component of interest to extract from the data.
        Must be one of ["000", "090", "geom", "rotd100_50", "rotd50", "ver"]
    proc_flag: bool, default=False, optional
        If True, prints the processing fault name.

    -------------
    Outputs:
    data_dic: dict
        A dictionary where keys are fault names, and values are xarray.DataArray objects.
        Each DataArray contains the processed simulation data for the corresponding fault with 'sation', 'IM', and
        'realization's as dimensions.

    -------------
    Description:
    - The function iterates through each fault and processes all CSV files matching "*REL*.csv" under the fault's directory.
    - It validates that the specified component exists in the data and checks station and IM consistency across files.
    - The data for each fault is concatenated into a single xarray.DataArray with dimensions ["station", "IM", "realization"].
    - The resulting dictionary enables quick access to simulation data for each fault.

    -------------
    Example Usage:

    # Path to simulation directory
    sim_dir = "/path/to/simulation/directory"

    # Load data for all faults in the directory
    temp_dic1 = load_sim_data(
        sim_dir=sim_dir,
        component="geom",
        proc_flag=False,
    )

    # Load data for a specific fault
    temp_dic2 = load_sim_data(
        sim_dir=sim_dir,
        faults=["FiordSZ03"],
        component="geom",
        proc_flag=True,
    )

    # Output example
    # temp_dic2["FiordSZ03"] will return an xarray.DataArray with dimensions: 'station', 'IM', 'realization'

    """

    counter = 0

    if not Path(sim_dir).exists():
        raise FileNotFoundError(f"Simulation directory {sim_dir} does not exist.")

    if not faults:
        all_faults = sorted(
            [cur_dir.stem for cur_dir in Path(sim_dir).iterdir() if cur_dir.is_dir()]
        )
    else:
        all_faults = faults

    data_dic = {fault: None for fault in all_faults}

    valid_component_list = ["000", "090", "geom", "rotd100_50", "rotd50", "ver"]

    if component not in valid_component_list:
        raise ValueError(f"component must be one of the {valid_component_list}")

    for cur_fault in tqdm(all_faults):

        if proc_flag == True:
            print(f"Processing fault: {cur_fault}")

        cur_im_files = sorted(
            list((Path(sim_dir) / cur_fault / "IM").rglob("*REL*.csv"))
        )

        da_list = []
        rel_list = []
        counter = 0

        for filename in cur_im_files:
            counter += 1
            temp_df = pd.read_csv(filename)

            if counter == 1:
                stations = temp_df["station"].unique().tolist()
                ims = [
                    col
                    for col in temp_df.columns
                    if col not in ["station", "component"]
                ]

            # Check if the component of interest is in the file
            if temp_df.empty:
                raise ValueError(
                    f"The components in {filename} do not match the component of interest: {component}."
                )

            # Read the component of interest from multi-component files
            temp_df = temp_df[temp_df["component"] == component]

            # Check if stations are consistent
            if not set(temp_df["station"]).issubset(stations):
                raise ValueError(
                    f"The stations in {filename} do not match with the expected list of stations."
                )

            # Check if the IMs are consistent
            if not set(temp_df.columns) - {"component"} - {"station"} <= set(ims):
                raise ValueError(
                    f"The IMs in {filename} do not match with the expected list of IMs."
                )

            temp_df = temp_df.set_index("station").drop("component", axis=1)

            temp_da = xr.DataArray(
                temp_df.values,
                dims=["station", "IM"],
                coords={
                    "station": temp_df.index,
                    "IM": temp_df.columns,
                },
            )

            da_list.append(temp_da)
            rel_list.append(filename.stem)

        combined_da = xr.concat(da_list, dim="realization")
        combined_da = combined_da.assign_coords(realization=rel_list)

        data_dic[cur_fault] = combined_da

    return data_dic


def load_emp_data(
    emp_dir: str, faults: list = [], component: str = "geom", proc_flag: bool = False
):
    """
    Function to load empirical data from a specified directory, process it by fault, and organize it into a dictionary.
    Each fault is associated with an xarray.DataArray containing the processed empirical data.

    -------------
    Inputs:
    emp_dir: str
        Path to the empirical directory. This directory should contain subdirectories for each fault,
        and within those, CSV files with empirical data.
    faults: list, optional
        List of fault names to process. If None, all faults found in the empirical directory will be processed.
    component: str, default="geom"
        The specific component of interest to extract from the data.
        Must be one of ["000", "090", "geom", "rotd100_50", "rotd50", "ver"]
    proc_flag: bool, default=False, optional
        If True, prints the processing fault name.

    -------------
    Outputs:
    data_dic: dict
        A dictionary where keys are fault names, and values are xarray.DataArray objects.
        Each DataArray contains the processed empirical data for the corresponding fault with dimensions 'station', 'IM',
        and 'statistics' (e.g., mean, std_total, std_inter, std_intra).

    -------------
    Description:
    - The function iterates through each fault and processes its CSV file, which contains the empirical data.
    - It validates that the specified component exists in the data and checks station and IM consistency.
    - The function extracts and processes four types of statistics: mean, total standard deviation, inter-event standard
      deviation, and intra-event standard deviation.
    - The data for each fault is combined into a single xarray.DataArray with dimensions ["station", "IM", "statistics"].

    -------------
    Example Usage:

    # Path to simulation directory
    sim_dir = "/path/to/empirical/directory"

    # Load data for all faults in the directory
    temp_dic1 = load_emp_data(
        sim_dir=sim_dir,
        component="geom",
        proc_flag=False,
    )

    # Load data for a specific fault
    temp_dic2 = load_emp_data(
        sim_dir=sim_dir,
        faults=["FiordSZ03"],
        component="geom",
        proc_flag=True,
    )

    # Output example
    # temp_dic2["FiordSZ03"] will return an xarray.DataArray with dimensions: 'station', 'IM', 'statistics'

    """

    im_true_order = [
        "PGA",
        "PGV",
        "CAV",
        "AI",
        "Ds575",
        "Ds595",
        "MMI",
        "pSA_0.01",
        "pSA_0.02",
        "pSA_0.03",
        "pSA_0.04",
        "pSA_0.05",
        "pSA_0.075",
        "pSA_0.1",
        "pSA_0.12",
        "pSA_0.15",
        "pSA_0.17",
        "pSA_0.2",
        "pSA_0.25",
        "pSA_0.3",
        "pSA_0.4",
        "pSA_0.5",
        "pSA_0.6",
        "pSA_0.7",
        "pSA_0.75",
        "pSA_0.8",
        "pSA_0.9",
        "pSA_1.0",
        "pSA_1.25",
        "pSA_1.5",
        "pSA_2.0",
        "pSA_2.5",
        "pSA_3.0",
        "pSA_4.0",
        "pSA_5.0",
        "pSA_6.0",
        "pSA_7.5",
        "pSA_10.0",
    ]

    if not Path(emp_dir).exists():
        raise FileNotFoundError(f"Empirical directory {emp_dir} does not exist.")

    if not faults:
        all_faults = sorted(
            [cur_dir.stem for cur_dir in Path(emp_dir).iterdir() if cur_dir.is_dir()]
        )
    else:
        all_faults = faults

    data_dic = {fault: None for fault in all_faults}

    valid_component_list = ["000", "090", "geom", "rotd100_50", "rotd50", "ver"]

    if component not in valid_component_list:
        raise ValueError(f"component must be one of the {valid_component_list}")

    for cur_fault in tqdm(all_faults):

        if proc_flag == True:
            print(f"Processing fault: {cur_fault}")

        filename = Path(emp_dir) / str(cur_fault) / (str(cur_fault) + ".csv")
        temp_df = pd.read_csv(filename)
        stations = temp_df["station"].unique().tolist()
        ims = [col.replace("_mean", "") for col in temp_df.columns if "_mean" in col]
        ims = sorted(ims, key=im_true_order.index)
        temp_df = temp_df[temp_df["component"] == component].set_index("station")
        temp_df = temp_df[
            [
                col
                for col in temp_df.columns
                if any(base in col for base in im_true_order)
            ]
        ]
        # reorder columns

        temp_df_mean = temp_df[
            [col for col in temp_df.columns if "_mean" in col]
        ].rename(columns=lambda x: x.replace("_mean", ""))
        temp_da_mean = xr.DataArray(
            temp_df_mean.values,
            dims=["station", "IM"],
            coords={
                "station": temp_df_mean.index,
                "IM": temp_df_mean.columns,
            },
        )

        temp_df_std_total = temp_df[
            [col for col in temp_df.columns if "_std_Total" in col]
        ].rename(columns=lambda x: x.replace("_std_Total", ""))
        temp_da_std_total = xr.DataArray(
            temp_df_std_total.values,
            dims=["station", "IM"],
            coords={
                "station": temp_df_std_total.index,
                "IM": temp_df_std_total.columns,
            },
        )

        temp_df_std_inter = temp_df[
            [col for col in temp_df.columns if "_std_Inter" in col]
        ].rename(columns=lambda x: x.replace("_std_Inter", ""))
        temp_da_std_inter = xr.DataArray(
            temp_df_std_inter.values,
            dims=["station", "IM"],
            coords={
                "station": temp_df_std_inter.index,
                "IM": temp_df_std_inter.columns,
            },
        )

        temp_df_std_intra = temp_df[
            [col for col in temp_df.columns if "_std_Intra" in col]
        ].rename(columns=lambda x: x.replace("_std_Intra", ""))
        temp_da_std_intra = xr.DataArray(
            temp_df_std_intra.values,
            dims=["station", "IM"],
            coords={
                "station": temp_df_std_intra.index,
                "IM": temp_df_std_intra.columns,
            },
        )

        combined_da = xr.concat(
            [temp_da_mean, temp_da_std_total, temp_da_std_inter, temp_da_std_intra],
            dim="statistics",
        )
        combined_da = combined_da.assign_coords(
            statistics=["mean", "std_total", "std_inter", "std_intra"]
        )

        data_dic[cur_fault] = combined_da

    return data_dic


def calc_sim_statistics(
    sim_dir: str, faults: list = [], component: str = "geom", proc_flag: bool = False
):
    """
    Function to calculate simulation data statistics.

    -------------
    Inputs:
    sim_dir: str
        Path to the simulation directory. This directory should contain subdirectories for each fault,
        and within those, simulation data files.
    faults: list, optional
        List of fault names to process. If None, all faults found in the simulation directory will be processed.
    component: str, default="geom"
        The specific component of interest to extract from the data.
        Must be one of ["000", "090", "geom", "rotd100_50", "rotd50", "ver"].
    proc_flag: bool, default=False
        If True, prints the processing fault name. This flag is passed to `util.load_sim_data`.

    -------------
    Outputs:
    data_stat_dic: dict
        A dictionary where keys are fault names, and values are `xarray.DataArray` objects.
        Each DataArray contains the mean of the log-transformed simulation data for the corresponding fault.
        The dimensions of the DataArray are 'station', 'IM', and 'statistics'.

    -------------
    Description:
    - The function validates the existence of the input directory and checks for valid components.
    - Calls `load_sim_data` to load simulation data into a dictionary (`data_dic`), where keys are fault names.
    - Applies a natural logarithm transformation to the data, ignoring non-positive values by replacing them with `NaN`.
    - Computes the mean of the log-transformed data along the "realization" dimension for each fault.
    - Stores the resulting statistics in an `xarray.DataArray`, which is organized by dimensions: 'station', 'IM', and 'statistics'.
    - Returns a dictionary that maps each fault to its corresponding statistics DataArray.

    -------------
    Example Usage:

    # Path to simulation directory
    sim_dir = "/path/to/simulation/directory"

    # Calculate statistics for all faults in the directory
    temp_dic1 = calc_sim_statistics(
        sim_dir=sim_dir,
        component="geom",
        proc_flag=False,
    )

    # Calculate statistics for a specific fault
    temp_dic2 = calc_sim_statistics(
        sim_dir=sim_dir,
        faults=["FiordSZ03"],
        component="geom",
        proc_flag=True,
    )

    # Output example
    # temp_dic2["FiordSZ03"] will return an xarray.DataArray with dimensions: 'station', 'IM', 'statistics'

    """

    if not Path(sim_dir).exists():
        raise FileNotFoundError(f"Simulation directory {sim_dir} does not exist.")

    if not faults:
        all_faults = sorted(
            [cur_dir.stem for cur_dir in Path(sim_dir).iterdir() if cur_dir.is_dir()]
        )
    else:
        all_faults = faults

    valid_component_list = ["000", "090", "geom", "rotd100_50", "rotd50", "ver"]

    if component not in valid_component_list:
        raise ValueError(f"component must be one of the {valid_component_list}")

    data_stat_dic = {fault: None for fault in all_faults}

    data_dic = load_sim_data(
        sim_dir=sim_dir, faults=faults, component=component, proc_flag=proc_flag
    )

    data_dic_ln = {
        key: np.log(temp_da.where(temp_da > 0)) for key, temp_da in data_dic.items()
    }  # returns NA for zero (and negative) values

    for cur_fault in data_dic_ln.keys():
        temp_da_mean = data_dic_ln[cur_fault].mean(dim="realization", skipna=True)

        combined_da = xr.concat([temp_da_mean], dim="statistics")
        combined_da = combined_da.assign_coords(statistics=["mean"])

        data_stat_dic[cur_fault] = combined_da

    return data_stat_dic


def calc_mean_residual(
    sim_dic: dict,
    emp_dic: dict,
    faults: list = [],
    IMs: list = [],
    normalized: bool = False,
):
    """
    Calculates mean residuals between simulation (sim_dic) and empirical (emp_dic) data
    for specified faults and intensity measures (IMs). Supports normalization of residuals.

    Inputs:
    -------
    sim_dic: dict
        A dictionary containing simulation data for faults as xarray.DataArray.
    emp_dic: dict
        A dictionary containing empirical data for faults as xarray.DataArray.
    faults: list, optional
        List of faults to include. Defaults to all common faults between sim_dic and emp_dic.
    IMs: list, optional
        List of intensity measures (IMs) to calculate residuals for. Defaults to all
        common IMs between sim_dic and emp_dic for each fault.
    normalized: bool, default=False
        Whether to normalize the residuals by the empirical standard deviation (std_total).

    Outputs:
    --------
    residual_dic: dict
        A dictionary containing residuals for each fault as pandas.DataFrame.
        Rows represent common stations, columns represent common IMs.

    -------------
    Description:
    1. Fault Handling:
        - If `faults` is not specified, the function uses common faults between `sim_dic` and `emp_dic`.
    2. IM Handling:
        - If `IMs` is not specified, common IMs for each fault are used.
        - Raises an error if any specified IM does not exist in the data.
    3. Residual Calculation:
        - Residuals are computed as the difference between simulation and empirical means.
        - If `normalized=True`, residuals are divided by the empirical standard deviation.
    4. Automatic Alignment:
        - The function aligns simulation and empirical data by their common indices (stations)
        and columns (IMs) before performing calculations.

    Key Notes:
    ----------
    - Ensure that the input dictionaries (sim_dic and emp_dic) contain xarray.DataArray objects
    with properly aligned dimensions (station, IM, statistics).
    - Use normalized=True for normalized residuals, where empirical standard deviation
    (std_total) is used as a normalization factor.

    -------------
    Example Usage:
    residuals = calc_mean_residual(
        sim_dic=sim_dic,
        emp_dic=emp_dic,
        faults=["FiordSZ03"],
        IMs = ["pSA_0.01", "pSA_5.0"],
        normalized=True,
        )

    # Output:residual_dic: A dictionary where keys are faults, and values are pandas.DataFrame objects
    #          containing the residuals.

    """

    im_true_order = [
        "PGA",
        "PGV",
        "CAV",
        "AI",
        "Ds575",
        "Ds595",
        "MMI",
        "pSA_0.01",
        "pSA_0.02",
        "pSA_0.03",
        "pSA_0.04",
        "pSA_0.05",
        "pSA_0.075",
        "pSA_0.1",
        "pSA_0.12",
        "pSA_0.15",
        "pSA_0.17",
        "pSA_0.2",
        "pSA_0.25",
        "pSA_0.3",
        "pSA_0.4",
        "pSA_0.5",
        "pSA_0.6",
        "pSA_0.7",
        "pSA_0.75",
        "pSA_0.8",
        "pSA_0.9",
        "pSA_1.0",
        "pSA_1.25",
        "pSA_1.5",
        "pSA_2.0",
        "pSA_2.5",
        "pSA_3.0",
        "pSA_4.0",
        "pSA_5.0",
        "pSA_6.0",
        "pSA_7.5",
        "pSA_10.0",
    ]

    if not faults:
        print(
            "No fault specified. Proceeding with common faults between sim_dic and emp_dic."
        )
        all_faults = sorted(list(set(sim_dic.keys()).intersection(set(emp_dic.keys()))))
        print(
            f"Processing {len(all_faults)} common faults found between sim_dic and emp_dic."
        )
    else:
        all_faults = sorted(faults)

    fault_IM_dic = {
        fault: sorted(
            list(
                set(sim_dic[fault].coords["IM"].values)
                & set(emp_dic[fault].coords["IM"].values)
            ),
            key=im_true_order.index,
        )
        for fault in all_faults
    }

    if not IMs:
        print(
            "No IM specified. Proceeding with common IMs between sim_dic and emp_dic for each fault."
        )
        all_IMs = fault_IM_dic
    else:
        IM_mask = {IM: None for IM in IMs}
        checker = False
        for cur_IM in IMs:
            temp_mask = []
            for cur_fault in all_faults:
                temp_mask.append(cur_IM in fault_IM_dic[cur_fault])
            IM_mask[cur_IM] = temp_mask
            if False in temp_mask:
                missing_faults = [
                    fault for fault, exists in zip(all_faults, temp_mask) if not exists
                ]
                print(
                    f"{cur_IM} does not exist in the database for the following faults: \n {missing_faults}"
                )
                checker = True
        if checker:
            raise ValueError(
                "One or more IMs are not exist in the database. Review the list of IMs."
            )

        all_IMs = {fault: sorted(IMs, key=im_true_order.index) for fault in all_faults}

    residual_dic = {fault: None for fault in all_faults}

    for cur_fault in all_faults:
        temp_sim_df = sim_dic[cur_fault].sel(statistics="mean").to_pandas()
        temp_emp_df = emp_dic[cur_fault].sel(statistics="mean").to_pandas()
        temp_emp_sdt_df = emp_dic[cur_fault].sel(statistics="std_total").to_pandas()

        common_indices = temp_sim_df.index.intersection(temp_emp_df.index)
        common_columns = temp_sim_df.columns.intersection(
            temp_emp_df.columns.intersection(all_IMs[cur_fault])
        )

        temp_sim_df_common = temp_sim_df.loc[common_indices, common_columns]
        temp_emp_df_common = temp_emp_df.loc[common_indices, common_columns]
        temp_emp_sdt_df_common = temp_emp_sdt_df.loc[common_indices, common_columns]

        temp_res_df = temp_sim_df_common - temp_emp_df_common

        if normalized:
            N_factor = temp_emp_sdt_df_common
        else:
            N_factor = 1

        residual_dic[cur_fault] = temp_res_df / N_factor

    return residual_dic


def fault_mapping_from_flt_site_source_db(hdf5_path: str) -> dict:
    """
    Function to precompute the mapping of fault names to fault IDs from the `flt_site_source.db` HDF5 file.

    -------------
    Inputs:
    hdf5_path: str
        Path to the HDF5 file containing fault information in the "faults" key.

    -------------
    Outputs:
    fault_mapping: dict
        A dictionary where keys are fault names (str), and values are their corresponding fault IDs (int).

    -------------
    Description:
    - The function reads the "faults" table from the specified HDF5 file using the `pandas.read_hdf` method.
    - The table should contain a "fault_name" column and an index representing fault IDs.
    - It creates a mapping of fault names to fault IDs by zipping the "fault_name" column with the index of the DataFrame.
    - The resulting dictionary can be used to quickly retrieve fault IDs for a given fault name.

    -------------
    Example Usage:

    # Path to the HDF5 file
    hdf5_path = "/path/to/flt_site_source.db"

    # Precompute the fault mapping
    fault_mapping = fault_mapping_from_flt_site_source_db(hdf5_path)

    # Access fault ID for a specific fault name
    fault_id = fault_mapping.get("HikHBaymax")
    print(f"Fault ID for 'HikHBaymax': {fault_id}")

    # Output example
    # {'HikHBaymax': 1, 'FiordSZ03': 2, 'FiordSZ09': 3, ...}

    """
    faults_df = pd.read_hdf(hdf5_path, key="faults")
    return dict(zip(faults_df["fault_name"], faults_df.index))


def get_distance_from_db(
    station: str, fault_name: str, file_handle: h5py.File, fault_mapping: dict
) -> pd.DataFrame:
    """
    Function to retrieve distances for a given station and fault from the `flt_site_source.db` HDF5 file.

    -------------
    Inputs:
    station: str
        The name of the station (e.g., 'station_0200000').
    fault_name: str
        The name of the fault to retrieve distances for.
    file_handle: h5py.File
        An open HDF5 file handle for the database.
    fault_mapping: dict
        A dictionary mapping fault names to fault IDs (e.g., the output of the
        'fault_mapping_from_flt_site_source_db' function).

    -------------
    Outputs:
    pd.DataFrame
        A DataFrame containing the distances with columns: ['rjb', 'rrup', 'rx', 'ry', 'rtvz'].
        Returns an empty DataFrame if no matching rows are found.

    -------------
    Description:
    - Maps the fault name to its fault ID using the provided `fault_mapping`.
    - Checks if the station exists in the HDF5 file.
    - Retrieves and filters distance data where the fault ID matches the station's data.
    - Converts the matching data to a pandas DataFrame for further processing.

    -------------
    Example Usage:

    # Open the HDF5 file
    import h5py

    file_path = "/path/to/flt_site_source.db"
    with h5py.File(file_path, "r") as file_handle:
        # Precompute fault mapping
        fault_mapping = fault_mapping_from_flt_site_source_db(file_path)

        # Retrieve distances
        station = "HNPS"
        fault_name = "HikHBaymax"
        distances = get_distance_from_db(station, fault_name, file_handle, fault_mapping)
        print(distances)

    # Output example
    #        rjb       rrup         rx          ry          rtvz
    # 0  3.195569  20.23909  103.464213  11.429241          NaN

    """

    station_key = f"distances/station_{station}"
    fault_id = fault_mapping[fault_name]

    # Check if the station exists in the HDF5 file
    if station_key not in file_handle:
        raise ValueError(f"Station '{station}' not found in the HDF5 file.")

    # Access the station group
    station_group = file_handle[station_key]

    # Extract block0_values and block1_values
    block0_values = station_group["block0_values"][:].flatten()
    block1_values = station_group["block1_values"][:]

    # Filter rows where block0_values matches the fault_id
    matching_rows = block1_values[block0_values == fault_id]

    # If no matching rows, return an empty DataFrame
    if matching_rows.size == 0:
        return pd.DataFrame(columns=["rjb", "rrup", "rx", "ry", "rtvz"])

    # Convert to a DataFrame with column names
    columns = ["rjb", "rrup", "rx", "ry", "rtvz"]
    return pd.DataFrame(matching_rows, columns=columns)


def single_station_get_distance_from_db(
    station, fault_name, file_handle, fault_mapping
):
    """
    Function to process 'get_distance_from_db' function for a single station.

    -------------
    Inputs:
    station: str
        The name of the station to process (e.g., 'station_0200000').
    fault_name: str
        The name of the fault to retrieve distance data for.
    file_handle: h5py.File
        An open HDF5 file handle for the database.
    fault_mapping: dict
        A dictionary mapping fault names to fault IDs.

    -------------
    Outputs:
    pd.DataFrame
        A DataFrame containing the distances with columns: ['rjb', 'rrup', 'rx', 'ry', 'rtvz'].
        Returns an empty DataFrame if the station or fault is not found.

    -------------
    Description:
    - Calls `get_distance_from_db` to retrieve distance data for the given station and fault.
    - Handles missing stations or faults gracefully by returning an empty DataFrame.

    -------------
    Example Usage:

    distances = process_fault(station, fault_name, file_handle, fault_mapping)
    print(distances)

    # Output example
    #        rjb       rrup         rx          ry          rtvz
    # 0  3.195569  20.23909  103.464213  11.429241          NaN

    """

    try:
        return get_distance_from_db(
            station=station,
            fault_name=fault_name,
            file_handle=file_handle,
            fault_mapping=fault_mapping,
        )
    except ValueError:
        return pd.DataFrame(columns=["rjb", "rrup", "rx", "ry", "rtvz"])


def parallel_multi_station_get_distance_from_db(
    stations: list, fault_name: str, file_handle: h5py.File, fault_mapping: dict
):
    """
    Function to retrieve distances for multiple stations for a single fault in parallel.

    -------------
    Inputs:
    fault_name: str
        Name of the fault to process.
    stations: list
        List of station names to process (e.g., ['station_0200000', 'station_0200001']).
    file_handle: h5py.File
        An open HDF5 file handle for the database.
    fault_mapping: dict
        A dictionary mapping fault names to fault IDs.

    -------------
    Outputs:
    pd.DataFrame
        A DataFrame containing station-distance information with the following columns:
        ['station', 'rjb', 'rrup', 'rx', 'ry', 'rtvz'].
        Returns an empty DataFrame for stations with no matching data.

    -------------
    Description:
    - Checks if the fault exists in the provided `fault_mapping`. Raises an error if not found.
    - Retrieves distance data for all specified stations in parallel using `single_station_get_distance_from_db`.
    - Combines the results into a single DataFrame, with station names as an index.

    -------------
    Example Usage:

    # Open the HDF5 file
    import h5py

    file_path = "/path/to/flt_site_source.db"
    with h5py.File(file_path, "r") as file_handle:
        # Precompute fault mapping
        fault_mapping = fault_mapping_from_flt_site_source_db(file_path)

        # Process multiple stations for a fault
        fault_name = "HikHBaymax"
        stations = ["HNPS", "WHNS"]
        distances_df = parallel_multi_station_get_distance_from_db(
            fault_name=fault_name,
            stations=stations,
            file_handle=file_handle,
            fault_mapping=fault_mapping,
        )

        print(distances_df)

    # Output example
    #      station       rjb       rrup         rx          ry          rtvz
    # 0      HNPS  3.195569  20.23909  103.464213  11.429241          NaN
    # 1      WHNS  2.948273  19.87542   95.134213   9.329241          NaN

    """

    print(f"Processing distances for Fault: {fault_name}")

    if fault_name not in fault_mapping:
        raise ValueError(f"Fault name '{fault_name}' not found in fault mapping.")

    # Retrieve distances for all stations in parallel
    with ThreadPoolExecutor() as executor:
        station_rrups = list(
            executor.map(
                lambda station: single_station_get_distance_from_db(
                    station=station,
                    fault_name=fault_name,
                    file_handle=file_handle,
                    fault_mapping=fault_mapping,
                ),
                stations,
            )
        )

    # Combine distances into a DataFrame
    station_rrups_df = (
        pd.concat(station_rrups, keys=stations, names=["station"])
        .reset_index(level=1, drop=True)
        .reset_index()
    )

    return station_rrups_df


def filter_stations_within_polygon(stations_df, boundary_coords):
    """
    Filters stations within a polygon defined by boundary_coords.
    
    Parameters:
    - stations_df: DataFrame with 'lat' and 'lon' columns.
    - boundary_coords: List of four [lon, lat] coordinate pairs (in order forming a closed polygon).
    
    Returns:
    - Filtered GeoDataFrame of stations within the polygon.
    """
    # Ensure the polygon is closed (repeat the first point at the end if not already)
    if boundary_coords[0] != boundary_coords[-1]:
        boundary_coords.append(boundary_coords[0])

    # Create GeoDataFrame of stations
    gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.long, stations_df.lat),
        crs="EPSG:4326"  # WGS84
    )

def filter_stations_within_a_circle(stations_df, lat0, lon0, R_km):
    """
    Filters stations within a radius from a given center point using GeoPandas and NZTM2000 projection.

    Parameters:
    -----------
    stations_df : pandas.DataFrame
        DataFrame containing station information. Must include 'lat' and 'lon' columns (WGS84 coordinates).
        
    lat0 : float
        Latitude of the center point (in decimal degrees, WGS84).
        
    lon0 : float
        Longitude of the center point (in decimal degrees, WGS84).
        
    R_km : float
        Radius around the center point within which to select stations (in kilometers).

    Returns:
    --------
    pandas.DataFrame
        The filtered original DataFrame (stations_df), containing only rows within R_km of the center.
        The returned DataFrame is in the original CRS (no projection changes).
    
    Notes:
    ------
    - Uses EPSG:2193 (NZTM2000), the standard metric CRS for New Zealand, for accurate distance calculations.
    - Input coordinates and DataFrame must be in geographic (lat/lon, EPSG:4326) format.
    """
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df.long, stations_df.lat),
        crs="EPSG:4326"
    )

    # Create center point and project both to NZTM2000
    center = gpd.GeoSeries([Point(lon0, lat0)], crs="EPSG:4326").to_crs(epsg=2193)
    gdf_proj = gdf.to_crs(epsg=2193)

    # Buffer around the center point
    buffer = center.buffer(R_km * 1000).iloc[0]

    # Filter and return only matching rows from the original DataFrame
    filtered_indices = gdf_proj.geometry.within(buffer)
    return stations_df[filtered_indices.values].reset_index(drop=True)

def find_closest_station(stations_df, target_lon, target_lat):
    """
    Find the closest station using accurate distance in projected NZ coordinates.

    Parameters:
    - stations_df: DataFrame with 'long' and 'lat', the output of util.get_station_data
    - target_lon: float, longitude of the point
    - target_lat: float, latitude of the point

    Returns:
    - pandas Series for the closest station
    """
    # 1. Convert station points to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["long"], stations_df["lat"]),
        crs="EPSG:4326"  # WGS84 (lat/lon)
    )
    
    # 2. Project to NZ Transverse Mercator (for accurate distances in meters)
    gdf = gdf.to_crs("EPSG:2193")

    # 3. Create target point and project it
    target_point = gpd.GeoSeries([Point(target_lon, target_lat)], crs="EPSG:4326").to_crs("EPSG:2193").iloc[0]

    # 4. Compute distances and find closest
    distances = gdf.geometry.distance(target_point)
    closest_idx = distances.idxmin()
    return gdf.loc[closest_idx], distances[closest_idx]

def a_to_mw_leonard(a, rake, leonard_ds = 4.00, leonard_ss = 3.99):
    # From qcore/qcore/uncertainities/mag_scaling.py
    if round(rake % 360 / 90.0) % 2:
        mw = np.log10(a) + leonard_ds
    else:
        mw = np.log10(a) + leonard_ss
    return mw

def a_to_mw_skarlatoudis(a):
    # From qcore/qcore/uncertainities/mag_scaling.py
    return np.log10(a) + 3.722


def mw_sigma_leonard(rake):
    """
    sigma values are determined from the Leonard 2014 paper, based on Table 3 - S(a) passing it through the equation
    also based on the this values can be obtained from the Table 5 of the Leonard 2010 and patthing the Sa values to 
    the M0 to Mw conversion relation (eq.2.5) of Brendon's book. See my notes on Zotero.

    rake: to determine if DS or SS
    returns sigma value for mean Mw
    """
    # From qcore/qcore/uncertainities/mag_scaling.py
    if round(rake % 360 / 90.0) % 2:
        sigma = 0.3
    else:
        sigma = 0.26
    return sigma



def mw_sigma_skarlatoudis():
    """
    # 1.498 is the sigma value from Table 3 in Skarlatoudis 2016
    """
    # From qcore/qcore/uncertainities/mag_scaling.py
    return np.log10(1.498)  


# def get_distance_from_db(station: str, fault_name: str, hdf5_path: str) -> pd.DataFrame:
#     """
#     Retrieve the distances for a given station and fault name from the flt_site_source.db.

#     Parameters:
#     - station (str): The name of the station (e.g., 'station_0200000').
#     - fault_name (str): The fault name to lookup.
#     - hdf5_path (str): Path to the HDF5 file.

#     Returns:
#     - pd.DataFrame: DataFrame containing block0_values with columns ['rjb', 'rrup', 'rx', 'ry', 'rtvz'].

#     Example usage:
#     station_name = 'HNPS' #'0200000'
#     fault_name = 'HikHBaymax'
#     file_path = os.path.join(base_dir, 'flt_site_source.db')

#     try:
#         df = get_distance_from_db(station_name, fault_name, file_path)
#         print(df)
#     except ValueError as e:
#     print(e)

#     # Sample Output
#             rjb       rrup      rx          ry          rtvz
#          0  3.195569  20.23909  103.464213  11.429241   NaN
#     """
#     # Load fault names to create a mapping from fault_name to fault_id
#     station = "station_" + station
#     faults_df = pd.read_hdf(hdf5_path, key="faults")
#     fault_name_to_id = dict(zip(faults_df["fault_name"], faults_df.index))

#     # Check if the fault name exists in the mapping
#     if fault_name not in fault_name_to_id:
#         raise ValueError(f"Fault name '{fault_name}' not found.")

#     # Get the fault_id for the given fault name
#     fault_id = fault_name_to_id[fault_name]

#     with h5py.File(hdf5_path, "r") as f:
#         # Check if the specified station exists in the file
#         if f"distances/{station}" not in f:
#             raise ValueError(f"Station '{station}' not found in the HDF5 file.")

#         # Navigate to the specific station group
#         station_group = f[f"distances/{station}"]

#         # Extract block0_values and block1_values
#         block0_values = station_group["block0_values"][:].flatten()
#         block1_values = station_group["block1_values"][:]

#         # Filter rows in block0_values where block1_values matches the fault_id
#         matching_rows = block1_values[block0_values == fault_id]

#         # Convert to a DataFrame with the specified column names
#         columns = ["rjb", "rrup", "rx", "ry", "rtvz"]
#         df = pd.DataFrame(matching_rows, columns=columns)

#         return df


# def get_batch_distances(stations, fault_name, hdf5_path):
#     """
#     Retrieve rrup values for multiple stations at once.
#     """
#     data = []
#     for station in stations:
#         df = get_distance_from_db(station, fault_name, hdf5_path)
#         if not df.empty:
#             data.append(df["rrup"].iloc[0])
#         else:
#             data.append(np.nan)
#     return data
