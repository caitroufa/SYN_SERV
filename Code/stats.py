# via Alex Haberlie https://github.com/ahaberlie/MCS_Climate/blob/main/examples/stats.py

import numpy as np
from scipy.stats import mannwhitneyu, levene

import geopandas as gpd
import regionmask
import numpy as np

def write_sig_summary(m_data, sim_names, reg_names, analysis_name, var_name, test='mann'):

    for sim in sim_names.keys():

        print(" ", sim)
    
        for region, region_states in reg_names.items():
    
            ast = ""
    
            hist_ = m_data['historical'][region][analysis_name]
            hist_years = hist_.mean(('west_east', 'south_north'))
            
            hist_mean = hist_.mean('time')
            hist_mean = hist_mean.mean(('west_east', 'south_north'))

            if sim != 'historical':
    
                futr_ = m_data[sim][region][analysis_name]
                futr_years = futr_.mean(('west_east', 'south_north'))
                
                futr_mean = futr_.mean('time')
                futr_mean = futr_mean.mean(('west_east', 'south_north'))

                if test == 'mann':
    
                    _, p = mannwhitneyu(hist_years[var_name].values, futr_years[var_name].values)

                elif test == 'leve':
                    
                    _, p = levene(hist_years[var_name].values, futr_years[var_name].values)

                else:
                    raise ValueError("test not implemented:", test)
        
                if p < 0.05:
                    ast = "*"
    
                dif = futr_mean[var_name].values - hist_mean[var_name].values
                print(f"   {sim} - historical", region, ":", dif, ast)
                
            else:
                print("   ", region, ":", hist_mean[var_name].values)

def get_analysis_regions():

    regions = {'SP': ['TX', 'OK', 'KS'],
               'NP': ['NE', 'SD', 'ND'],
               'MW': ['WI', 'MI', 'OH', 'IL', 'IN', 'MN', 'IA', 'MO', 'KY'],
               'SE': ['AR', 'LA', 'TN', 'MS', 'AL', 'SC', 'NC', 'FL', 'GA'],
               'NE': ['NJ', 'PA', 'NY', 'VT', 'NH', 'CT', 'VA', 'WV', 'MD', 'DC', 'DE', 'RI', 'MA', 'ME'],
               'EC': []}

    #ECONUS is comprised of states in all other regions
    [regions['EC'].extend(regions[x]) for x in regions.keys() if x!='EC'];
    
    region_names = {'EC': 'ECONUS', 
                    'NP': 'Northern Plains', 
                    'SP': 'Southern Plains', 
                    'MW': 'Midwest', 
                    'SE': 'Southeast', 
                    'NE': 'Northeast'}

    return regions, region_names

def mask_data(input_dataset, states, lon_name='lon', lat_name='lat', conus_shapefile_path="../data/geog/CONUS.shp"):
    r"""Masks an input xarray dataset using the
    two character state IDs in states based on the 
    lon_name and lat_name coordinates in input_dataset and
    if they are within the given list of states. Original dataset 
    is not modified.
    
    Parameters
    ----------
    input_dataset: xarray dataset
        Dataset that you want to mask. Must have a lon and lat coordinate.
    states: list or ndarray
        List of states used to extract data subset.      
    lon_name: str
        Name of longitude coordinate in input_dataset. Default is 'lon'.
    lat_name: str
        Name of latitude coordinate in input_dataset. Default is 'lat'.
    conus_shapefile_path: str
        Path to shapefile used to extract geography information.
        
    Returns
    -------
    output_dataset: xarray dataset
        Modified input_dataset where grids not in states are set to nan.
    """
    
    usa = gpd.read_file(conus_shapefile_path)
    usa = usa[usa.STUSPS.isin(states)]

    state_mask = regionmask.mask_geopandas(usa, input_dataset[lon_name], input_dataset[lat_name])
    
    ma = state_mask.values
    ma[~np.isnan(ma)] = 1
    output_dataset = input_dataset * ma
    
    return output_dataset

def grid_significance(ds1, ds2, expected_dims=(15, 44, 69)):
    r"""Performs a grid-to-grid significance test on ds1 and ds2.
    Returns a grid of the same size with p-values from the Mann
    Whitney U test.
    
    Parameters
    ----------
    ds1: (t, y, x) ndarray
        An ndarray in the format of (time, y, x) and shape of expected_dims. 
    ds2: (t, y, x) ndarray
        An ndarray in the format of (time, y, x) and shape of expected_dims.  
    expected_dims: tuple
        The expected shape of ds1 and ds2. Default is (15, 44, 69).
        
    Returns
    -------
    results: (y, x) ndarray
        Results of significance testing in the form of p-values.
    """    
    
    if ds1.shape == expected_dims and ds2.shape == expected_dims:
        #Assumption of non-significance with all values == 1
        results = np.ones(shape=(ds1.shape[1], ds1.shape[2]), dtype=float)

        for i in range(ds1.shape[1]):
            for j in range(ds1.shape[2]):

                ds1_dist = ds1[:, i, j]
                ds2_dist = ds2[:, i, j]

                if np.mean(ds1_dist > 0) or np.mean(ds2_dist > 0):

                    s, p = mannwhitneyu(ds1_dist, ds2_dist)
                        
                    results[i, j] = p

        return results
    
    else:
        
        raise ValueError("Dimensions are not as expected, given", ds1.shape, "expected", expected_dims)