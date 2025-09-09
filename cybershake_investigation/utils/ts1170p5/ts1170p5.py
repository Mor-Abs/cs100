"""
Code Description:
Module for computing the TS1170.5 response spectra

Author: Morteza
Version History:
- Version 1.0: June 2025, Initial version

---------------------------------------------------------------------------------
Example Usage:
---------------------------------------------------------------------------------
elastic_site_spectrum(
    T=1.0,
    soil_type='II',
    RP=500,
    long = 173.1,
    lat= -34.32,
)

elastic_site_spectrum(
    T=0.00,
    soil_type='III',
    RP=500,
    name='Christchurch',
)

gridded_location_data(
    target_long=173.1,
    target_lat=-34.32,
    RP=500,
)

get_site_D(
    lat= -34.32,
    long=173.1,
)

df = load_table_3p5(500)
print(f"Soil Type I:\n{df['I-Tc'].describe()}\n\n")
print(f"Soil Type II:\n{df['II-Tc'].describe()}\n\n")
print(f"Soil Type III:\n{df['III-Tc'].describe()}\n\n")
print(f"Soil Type IV:\n{df['IV-Tc'].describe()}\n\n")
print(f"Soil Type V:\n{df['V-Tc'].describe()}\n\n")
print(f"Soil Type VI:\n{df['VI-Tc'].describe()}\n\n")

M_named_location(
    target_location='Levin',
    RP=500,
)

D_named_location(
    target_location='Levin',
)

"""

import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

@lru_cache(maxsize=1)
def load_table_3p4(
    RP: Union[float, int],
):
    """
    Load Table 3.4 data from CSV file.
    
    Parameters
    ----------
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing Table 3.4 data for the specified return period.
    """
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    return pd.read_csv(DATA_DIR / f"named_location_report_apoe({RP}).csv")


@lru_cache(maxsize=1)
def load_table_3p5(
    RP: Union[float, int],
):
    """
    Load Table 3.5 data from CSV file.
    
    Parameters
    ----------
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing Table 3.5 data for the specified return period.
    """
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    return pd.read_csv(DATA_DIR / f"gridded_location_report_apoe({RP}).csv")


def elastic_site_spectrum(
    T: Union[float, int], 
    soil_type: str,
    RP: float = 500,
    name: str = None,
    lat: float = None,
    long: float = None,
    
) -> float:
    """
    Compute the elastic site spectrum as per Sec. 3.1.3.
    Parameters
    ----------
    T: Union[float, int]
        Period in seconds.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: float, optional
        Return period in years, default is 500 years.
    name : str, optional
        Named location (e.g., "Levin"). If provided, overrides lat/long.
    lat : float, optional
        Latitude in decimal degrees.
    long : float, optional
        Longitude in decimal degrees.
    Returns
    -------
    C: float
        Elastic site spectrum value for the specified period, soil type, distance to fault,
        return period and location.
    """

    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    l_d = get_site_D(name, lat=lat, long=long)  # Get the location-specific D value
    # the shortest distance, km, from the site to the nearest fault listed in Table 3.2,
    # which shall be obtained from Table 3.4 for listed locations and from Table 3.5 for grid points.
    
    C = specteral_acceleration(T, soil_type, RP, name, lat, long) * near_fault_factor(T, l_d, RP)
    
    return C


def near_fault_factor(
    T: Union[float, int], 
    D: float,
    RP: float,
) -> float:
    """
    Compute the near fault factor as per Sec. 3.1.4.

    Parameters
    ----------
    T: Union[float, int]
        Period in seconds.
    D: float
        the shortest distance, km, from the site to the nearest fault listed in Table 3.2,
        which shall be obtained from Table 3.4 for listed locations and from Table 3.5 for
        grid points
    RP: float
        Return period in years, default is 500 years. 
        Restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.

    Returns
    -------
    N: float
        Near fault factor for specified T and D.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if np.isnan(D):
        D = 21 # A value larger than 20 km is used for na values obtained from Table 3.4 and 3.5
    
    if RP <= 250:
        N = 1
    else:
        if D <= 2:
            N = N_max(T)
        elif D <=20:
            N = 1+ (N_max(T) - 1) * (20 - D) / 18   
        else:
            N = 1
    
    return N


def N_max(
    T: Union[float, int]
) -> float:
    """
    Compute the maximum near fault factor as per Sec. 3.1.4.

    Parameters
    ----------
    T: Union[float, int]
        Period in seconds.

    Returns
    -------
    N_max: float
        Maximum near fault factor for specified T.
    """
    periods = [1.5, 2, 3, 4, 5]
    values = [1.00, 1.12, 1.36, 1.60, 1.72]
    
    if T <= periods[0]:
        return values[0]
    elif T >= periods[-1]:
        return values[-1]
    else:
        for i in range(len(periods) - 1):
            if periods[i] <= T <= periods[i + 1]:
                N_max = np.interp(T, periods[i:i + 2], values[i:i + 2])
                break
        else:
            raise ValueError("Period T is out of bounds for N_max calculation.")
    
    return N_max

def get_site_data(
    RP: Union[float, int],
    name: str = None,
    lat: float = None,
    long: float = None,
    ) -> pd.Series:
    """
    Parameters
    ----------
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
    name : str, optional
        Named location (e.g., "Levin"). If provided, overrides lat/long.
    lat : float, optional
        Latitude in decimal degrees.
    long : float, optional
        Longitude in decimal degrees.

    Returns
    -------
    pd.Series
        Site-specific data based on name or coordinates.
        Series containing the location data for the specified coordinates/location and return period.
    """
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")

    if name is not None and (lat is not None or long is not None):
        raise ValueError("Provide either 'name' OR 'lat/long', not both.")
    elif name is not None:
        return named_location_data(name)
    elif lat is not None and long is not None:
        return gridded_location_data(target_lat=lat, target_long=long)
    else:
        raise ValueError("You must provide either a location 'name' or both 'lat' and 'long'.")


def gridded_location_data(
    target_long: float,
    target_lat: float,
    RP: Union[float, int],
) -> pd.Series:
    """
    Get the gridded location data from Table 3.5 for a specified longitude and latitude.

    Parameters
    ----------
    target_long: float
        Longitude of the targeted site.
    target_lat: float
        Latitude of the targeted site.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.

    Returns
    -------
    pd.Series
        Series containing the gridded location data for the specified coordinates and return period.
    """
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")

    df = load_table_3p5(RP)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data not loaded correctly.")

    df["c_long"] = df["location"].apply(lambda x: x.split("~")[1]).astype(float)
    df["c_lat"] = df["location"].apply(lambda x: x.split("~")[0]).astype(float)

    match = df[(df["c_lat"] == round(target_lat,1)) & (df["c_long"] == round(target_long,1))]
    if not match.empty:
        closest_row = match.iloc[0]
    else:
        print(
            "Warning: Targeted location is not within predefined grid in Table 3.5, calculating closest location.\nBetter to check the targeted location."
        )
        dist = np.sqrt(
            (df["c_long"] - target_long) ** 2 + (df["c_lat"] - target_lat) ** 2
        )
        closest_row = df.loc[dist.idxmin()]

    return closest_row

def get_named_locations(
) -> list[str]:
    """
    Get the named locations data from Table 3.4.

    Parameters
    ----------
    No input parameters.

    Returns
    -------
    list[str]
        List containing the named location data from Table 3.4.
    """
    
    RP = 25  # Default return period
    df = load_table_3p4(RP)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data not loaded correctly.")

    return df["location"].tolist()


def named_location_data (
    target_laction: str,
    RP: Union[float, int],
) -> pd.Series :
    """
    Get the named location data from Table 3.4 for a specified location.

    Parameters
    ----------
    target_location: str
        Name of the targeted location.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.

    Use get_named_locations() to get the list of available locations.
    
    Returns
    -------
    pd.Series
        Series containing the named location data for the specified location and return period.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")

    df = load_table_3p4(RP)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data not loaded correctly.")

    match = df[df["location"] == target_laction]
    
    if not match.empty:
        return match.iloc[0]
    else:
        raise ValueError(f"Location '{target_laction}' not found in Table 3.4.")
    
def get_site_D(
    name: str = None,
    lat: float = None,
    long: float = None,
    ) -> float:
    """
    Parameters
    ----------
    name : str, optional
        Named location (e.g., "Levin"). If provided, overrides lat/long.
    lat : float, optional
        Latitude in decimal degrees.
    long : float, optional
        Longitude in decimal degrees.

    Returns
    -------
    float
        Site specific D value for the specified location and return period.
        If D value is definedd as nan in Table 3.4, it will return numpy.nan.
        D value is the shortest distance, km, from the site to the nearest fault listed in Table 3.2,
        which shall be obtained from Table 3.4 for listed locations and from Table 3.5 for grid points.
    """

    if name is not None and (lat is not None or long is not None):
        raise ValueError("Provide either 'name' OR 'lat/long', not both.")
    elif name is not None:
        return D_named_location(name)
    elif lat is not None and long is not None:
        return D_gridded_location(target_lat=lat, target_long=long)
    else:
        raise ValueError("You must provide either a location 'name' or both 'lat' and 'long'.")
    
    
def D_gridded_location (
    target_long: float,
    target_lat: float,
    ) -> float:
    """
    Get the gridded location D value from Table 3.5 for a specified longitude and latitude.
    Parameters
    ----------
    target_long: float
        Longitude of the targeted site.
    target_lat: float
        Latitude of the targeted site.

    Returns
    -------
    float
        Gridded location D value for the specified coordinates.
        If D value is definedd as nan in Table 3.5, it will return numpy.nan.
    """
    closest_row = gridded_location_data(target_long, target_lat, 25)
    
    return closest_row['D'].astype(float)


def D_named_location(
    target_location: str,
) -> float:
    """
    Get the named location D value from Table 3.4 for a specified location.
    
    Parameters
    ----------
    target_location: str
        Name of the targeted location.
        
    Returns
    -------
    float
        Named location D value for the specified location.
        If D value is definedd as nan in Table 3.4, it will return numpy.nan.
    """
    closest_row = named_location_data(target_location, 25)
    
    return closest_row['D'].astype(float)


def M_gridded_location (
    target_long: float,
    target_lat: float,
    RP: Union[float, int],
    ) -> float:
    """
    Get the gridded location M value from Table 3.5 for a specified longitude and latitude.
    Parameters
    ----------
    target_long: float
        Longitude of the targeted site.
    target_lat: float
        Latitude of the targeted site.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Gridded location M value for the specified coordinates and return period.
    """
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    closest_row = gridded_location_data(target_long, target_lat, RP)
    
    return closest_row['M'].astype(float)


def M_named_location(
    target_location: str,
    RP: Union[float, int],
) -> float:
    """
    Get the named location M value from Table 3.4 for a specified location.
    
    Parameters
    ----------
    target_location: str
        Name of the targeted location.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Named location M value for the specified location and return period.
        If D value is definedd as nan in Table 3.4, it will return numpy.nan.
    """
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    closest_row = named_location_data(target_location, RP)
    
    return closest_row['M'].astype(float)

def PGA_gridded_location(
    target_long: float,
    target_lat: float,
    soil_type: str,
    RP: Union[float, int],
) -> float:
    """
    Get the gridded location PGA value from Table 3.5 for a specified longitude and latitude.
    
    Parameters
    ----------
    target_long: float
        Longitude of the targeted site.
    target_lat: float
        Latitude of the targeted site.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Gridded location PGA value for the specified coordinates and return period.
        If PGA value is defined as nan in Table 3.5, it will return numpy.nan.
    """
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if soil_type not in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        raise ValueError("Soil type must be one of ['I', 'II', 'III', 'IV', 'V', 'VI']")
    
    closest_row = gridded_location_data(target_long, target_lat, RP)
    
    return closest_row[f'{soil_type}-PGA'].astype(float)

def PGA_named_location(
    target_location: str,
    soil_type: str,
    RP: Union[float, int],
) -> float:
    """
    Get the named location PGA value from Table 3.4 for a specified location.
    
    Parameters
    ----------
    target_location: str
        Name of the targeted location.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Named location PGA value for the specified location and return period.
        If PGA value is defined as nan in Table 3.4, it will return numpy.nan.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if soil_type not in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        raise ValueError("Soil type must be one of ['I', 'II', 'III', 'IV', 'V', 'VI']")
    
    closest_row = named_location_data(target_location, RP)
    
    return closest_row[f'{soil_type}-PGA'].astype(float)

def Sas_gridded_location(
    target_long: float,
    target_lat: float,
    soil_type: str,
    RP: Union[float, int],
) -> float:
    """
    Get the gridded location Sas value from Table 3.5 for a specified longitude and latitude.
    
    Parameters
    ----------
    target_long: float
        Longitude of the targeted site.
    target_lat: float
        Latitude of the targeted site.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Gridded location Sas value for the specified coordinates and return period.
        If Sas value is defined as nan in Table 3.5, it will return numpy.nan.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if soil_type not in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        raise ValueError("Soil type must be one of ['I', 'II', 'III', 'IV', 'V', 'VI']")
    
    closest_row = gridded_location_data(target_long, target_lat, RP)
    
    return closest_row[f'{soil_type}-Sas'].astype(float)


def Sas_named_location(
    target_location: str,
    soil_type: str,
    RP: Union[float, int],
) -> float:
    """
    Get the named location Sas value from Table 3.4 for a specified location.
    
    Parameters
    ----------
    target_location: str
        Name of the targeted location.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Named location Sas value for the specified location and return period.
        If Sas value is defined as nan in Table 3.4, it will return numpy.nan.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if soil_type not in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        raise ValueError("Soil type must be one of ['I', 'II', 'III', 'IV', 'V', 'VI']")
    
    closest_row = named_location_data(target_location, RP)
    
    return closest_row[f'{soil_type}-Sas'].astype(float)

def Tc_gridded_location(
    target_long: float,
    target_lat: float,
    soil_type: str,
    RP: Union[float, int],
) -> float:
    """
    Get the gridded location Tc value from Table 3.5 for a specified longitude and latitude.
    
    Parameters
    ----------
    target_long: float
        Longitude of the targeted site.
    target_lat: float
        Latitude of the targeted site.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Gridded location Tc value for the specified coordinates and return period.
        If Tc value is defined as nan in Table 3.5, it will return numpy.nan.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if soil_type not in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        raise ValueError("Soil type must be one of ['I', 'II', 'III', 'IV', 'V', 'VI']")
    
    closest_row = gridded_location_data(target_long, target_lat, RP)
    
    return closest_row[f'{soil_type}-Tc'].astype(float)


def Tc_named_location(
    target_location: str,
    soil_type: str,
    RP: Union[float, int],
) -> float:
    """
    Get the named location Tc value from Table 3.4 for a specified location.
    
    Parameters
    ----------
    target_location: str
        Name of the targeted location.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
        
    Returns
    -------
    float
        Named location Tc value for the specified location and return period.
        If Tc value is defined as nan in Table 3.4, it will return numpy.nan.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if soil_type not in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        raise ValueError("Soil type must be one of ['I', 'II', 'III', 'IV', 'V', 'VI']")
    
    closest_row = named_location_data(target_location, RP)
    
    return closest_row[f'{soil_type}-Tc'].astype(float)

def specteral_acceleration(
    T: Union[float, int],
    soil_type: str,
    RP: Union[float, int],
    name: str = None,
    lat: float = None,
    long: float = None,
    ) -> float:
    """
    Parameters
    ----------
    T: Union[float, int]
        Interested vibration period in seconds.
    soil_type: str
        Soil type, either 'I', 'II', 'III', 'IV', 'V', or 'VI'.
    RP: Union[float, int]
        Return period in years, restricted to be 25, 50, 100, 250, 500, 1000, 2500 years.
    name : str, optional
        Named location (e.g., "Levin"). If provided, overrides lat/long.
    lat : float, optional
        Latitude in decimal degrees.
    long : float, optional
        Longitude in decimal degrees.

    Returns
    -------
    float
        Spectral acceleration value for the specified return period, soil type and location.
        If the location is not found, it will raise a ValueError.
    """
    
    if RP not in [25, 50, 100, 250, 500, 1000, 2500]:
        raise ValueError("RP must be one of [25, 50, 100, 250, 500, 1000, 2500]")
    
    if soil_type not in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        raise ValueError("Soil type must be one of ['I', 'II', 'III', 'IV', 'V', 'VI']")

    if name is not None and (lat is not None or long is not None):
        raise ValueError("Provide either 'name' OR 'lat/long', not both.")
    elif name is not None:
        l_pga = PGA_named_location(name, soil_type, RP) # location specific PGA from Table 3.4
        l_sas = Sas_named_location(name, soil_type, RP) # location specific Sas from Table 3.4
        l_tc = Tc_named_location(name, soil_type, RP)   # location specific Tc from Table 3.4
        
    elif lat is not None and long is not None:
        l_pga = PGA_gridded_location(long, lat, soil_type, RP)  # gridded location PGA from Table 3.5
        l_sas = Sas_gridded_location(long, lat, soil_type, RP)  # gridded location Sas from Table 3.5
        l_tc = Tc_gridded_location(long, lat, soil_type, RP)    # gridded location Tc from Table 3.5
        
    else:
        raise ValueError("You must provide either a location 'name' or both 'lat' and 'long'.")
    
    l_td = 3.0 # The default spectral-velocity-plateau corner period as per Sec. 3.1.1
    
    if T == 0:
        Sa = l_pga
    elif (T < 0.1):
        Sa = (l_sas - l_pga) / (0.1 - 0) * T + l_pga
    elif (T < l_tc):
        Sa = l_sas
    elif (T < l_td):
        Sa = l_sas * (l_tc / T)
    else:
        Sa = l_sas * (l_tc / T) * (l_td / T) ** 0.5
    
    return Sa
