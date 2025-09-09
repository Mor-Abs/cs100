"""
Module for computing the seismic hazard for the
New Zealand 2010 Seismic Hazard Model using
the results from physics-based GM simulations
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from .. import hazard, utils


def load_sim_im_data(im_data_dir: Path, verbose: bool = False):
    """
    Loads the IM data for each fault

    Note: Using a dataset as the
    number of realisations and the stations
    with data varies per fault, making a 4D
    unsuitable.

    Parameters
    ----------
    im_data_dir: Path
        Directory that contains a folder for
        each fault, which contains the IM data
        csv files.

    Returns
    -------
    fault_im_dict: dict
        The IM data for each fault as a DataArray
    """
    # Available faults
    faults = [cur_dir.stem for cur_dir in im_data_dir.iterdir() if cur_dir.is_dir()]

    fault_im_dict = {}
    for cur_fault in tqdm(faults):
        cur_im_files = (im_data_dir / cur_fault / "IM").rglob("*REL*.csv")

        # Create DataArray for each fault
        cur_im_dataarrays = []
        for ix, cur_im_file in enumerate(cur_im_files):
            cur_im_df = pd.read_csv(cur_im_file, index_col=0)
            stations = cur_im_df.index.astype(str)
            ims = cur_im_df.columns[1:]

            # Crearte DataArray for current IM file
            if verbose:
                print(
                    f"Processing {cur_im_file} ({ix + 1}/{len(list(im_data_dir.glob(f'{cur_fault}/IM/*REL*.csv')))}): {len(stations)} stations, {len(ims)} IMs"
                )
            da = xr.DataArray(
                data=cur_im_df[ims].values,
                dims=("station", "IM"),
                coords={"station": stations, "IM": ims},
            )

            da = da.expand_dims(realisation=[cur_im_file.stem.rsplit("_", 1)[1]])
            cur_im_dataarrays.append(da)

        # Concatenate along "realisation", aligning by "station" and "IM"
        cur_im_array = xr.concat(cur_im_dataarrays, dim="realisation", join="inner")
        cur_im_array = cur_im_array.transpose('station', 'IM', 'realisation')

        # 1. Remove stations or IMs with any NaN
        arr = cur_im_array.values
        valid_stations_mask = ~np.any(np.isnan(arr), axis=(1, 2))  # No NaN in any IM or realization for a station
        valid_IMs_mask = ~np.any(np.isnan(arr), axis=(0, 2)) # No NaN in any station or realization for an IM
        cur_im_array = cur_im_array.isel(station=valid_stations_mask, IM=valid_IMs_mask)
        
        # 2. Remove stations or IMs with any *exact integer* value
        def has_exact_int(arr, axis):
            # True if any element along axis is an integer value (including 0/1)
            return np.any((arr % 1 == 0), axis=axis)

        arr = cur_im_array.values
        valid_stations_mask = ~np.any((arr % 1 == 0), axis=(1, 2))
        valid_IMs_mask = ~np.any((arr % 1 == 0), axis=(0, 2))
        cur_im_array = cur_im_array.isel(station=valid_stations_mask, IM=valid_IMs_mask)

        fault_im_dict[cur_fault] = cur_im_array

    return fault_im_dict


def get_sim_site_ims(fault_im_dict: dict[str, xr.DataArray], site: str):
    """
    Get the IM data for a specific site

    Parameters
    ----------
    fault_im_dict: dict
        The IM data for each fault as a DataArray
    site: str

    Returns
    -------
    pd.DataFrame
        The IM data for the specified site
        with a multi-index of fault and realisation
    """
    # Get data per fault and convert to DataFrame
    cur_results = []
    for cur_fault, cur_array in fault_im_dict.items():
        if site not in cur_array.station:
            continue

        cur_df = cur_array.sel(station=site).to_dataframe(name="value").reset_index()
        cur_df["fault"] = cur_fault
        cur_df = cur_df.pivot(
            index=["fault", "realisation"], columns="IM", values="value"
        )

        cur_results.append(cur_df)

    return pd.concat(cur_results, axis=0)


def compute_sim_hazard(
    site_im_df: pd.DataFrame,
    flt_erf_df: pd.DataFrame,
    ims: Sequence[str] = None,
    im_levels: dict[str, np.ndarray[float]] = None,
):
    """
    Computes the simulation-based seismic hazard
    for a single site.

    Parameters
    ----------
    site_im_df: pd.DataFrame
        The IM data for the site.
        Index has to be a MultIndex [fault, realisation]
    flt_erf_df: pd.DataFrame
        The 2010 NSHM fault ERF data
    ims: Sequence of str
        The IMs for which to compute the hazard
    im_levels: dict
        The IM levels for each IM

    Returns
    -------
    hazard_results: dict
        The hazard curve for each IM
    """
    rec_prob = 1 / flt_erf_df["recur_int_median"]

    ims = site_im_df.columns if ims is None else ims
    if im_levels is not None:
        if any([True for cur_im in ims if cur_im not in im_levels]):
            raise ValueError("Not all IMs found in im_levels!")

    hazard_results = {}
    for cur_im in ims:
        cur_im_levels = utils.get_im_levels(cur_im)
        if im_levels is not None:
            cur_im_levels = im_levels.get(cur_im)

        cur_gm_prob_excd = hazard.non_parametric_gm_excd_prob(
            cur_im_levels, site_im_df[cur_im]
        )
        hazard_results[cur_im] = hazard.hazard_curve(cur_gm_prob_excd, rec_prob)

    return hazard_results
