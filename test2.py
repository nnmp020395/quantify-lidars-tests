import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pickle
import numpy.ma as ma
from sklearn.tree import DecisionTreeRegressor
import pickle

import sys
home_dir = '/homedata/nmpnguyen/comparaison/'
sys.path.append(Path(home_dir, 'Codes'))
from fonctions import dataset, conditions, get_file, check, conversion, plots

from datetime import datetime



# years = ['2018', '2019', '2020']
# maindir = '/homedata/nmpnguyen/IPRAL/NETCDF/v_simple'
# pattern = 'ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'
# all_sr355 = []
# all_sr532 = []
# for year in years:
#     for file in sorted(Path(maindir, year).glob(pattern)):
#         print(file)
#         dt = xr.open_dataset(file)
#         z = (dt['range'] < 20000)

#         # sr355 = (dt['calibrated']/dt['simulated']).where((d['flags']==0) & (d['range']<20000), drop=True).sel(wavelength=355)
#         # sr532 = (dt['calibrated']/dt['simulated']).where((d['flags']==0) & (d['range']<20000), drop=True).sel(wavelength=532)

#         # if (sr355.shape[0] != 0) | (sr532.shape[0] != 0):
#         #     sr355 = sr355.resample(time = '15min').mean('time', skipna=True)
#         #     sr532 = sr532.resample(time = '15min').mean('time', skipna=True)

#         try :
#             calibrated355 = dt['simulated'].where((dt['flags']==0) & (dt['range']<20000), drop=True).sel(wavelength=355)
#             calibrated532 = dt['simulated'].where((dt['flags']==0) & (dt['range']<20000), drop=True).sel(wavelength=532)
#             # if (calibrated355.shape[0] != 0) | (calibrated532.shape[0] != 0):  
#             calibrated355 = calibrated355.resample(time = '15min').mean('time', skipna=True)
#             calibrated532 = calibrated532.resample(time = '15min').mean('time', skipna=True)

#             all_sr355.append(calibrated355)
#             all_sr532.append(calibrated532)
#         except:
#             print('pass')
#             pass

# all_sr355 = xr.concat(all_sr355, dim='time')
# all_sr532 = xr.concat(all_sr532, dim='time')

# a0, a = np.intersect1d(all_sr355['time'], all_sr532['time'], return_indices=True)[1:]
# all_sr355 = all_sr355.isel(time=a0)
# all_sr532 = all_sr532.isel(time=a)

# all_sr355.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-355-mes_THEORICAL_CONDT-None-None-None.nc')
# all_sr532.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-532-mes_THEORICAL_CONDT-None-None-None.nc')

# all_sr355.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-355-mes_betamol.nc')
# all_sr532.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-532-mes_betamol.nc')


print(f'----------------GENERATE DATA--------------------')

condt = conditions('None', 'None', None)
condt.name = 'None'
# maindir = '/homedata/nmpnguyen/comparaison/'
# pattern = ['IPRAL2018_IPRAL2019_IPRAL2020-', f'-mes_THEORICAL_CONDT-{condt.name}-{condt.value}-{condt.close}.nc']
# dataset_name = 'IPRAL2018_IPRAL2019_IPRAL2020'
maindir = '/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/'
pattern = ['HSRL2_ER2_allsr', '_v3.nc']
dataset_name = 'HSRL2_ER2'


dataset355mes = xr.open_dataarray(Path(maindir, f'{pattern[0]}355{pattern[1]}')).values
dataset532mes = xr.open_dataarray(Path(maindir, f'{pattern[0]}532{pattern[1]}')).values
mask = np.logical_and(dataset355mes>0, ~np.isnan(dataset355mes))
dataset355mes = dataset355mes[mask]
dataset532mes = dataset532mes[mask]

print('----------------CONVERSION--------------------')
const_test = [5.3] # np.arange(4.0, 5.3, 0.1)

for const in const_test:
    const = np.round(const, decimals=1)
    method_name = 'THEORITICAL'
    method_name = f'{method_name}-{const}'
    dataset532pred = dataset355mes * const   

    print('----------------QUANTIFY--------------------')

    unit_value = np.arange(0, 1.5, 0.1)
    min_max_value = [0.0,80.0]
    pts_stats = []
    for u in unit_value:
        pts = check(min_max_value[0], min_max_value[0], min_max_value[1], min_max_value[1], u, dataset532mes, dataset532pred)
        pts_stats.append(pts.quantify())
        # print(f'Quantify predicted data within +/- {u} unit around the diagonal and between {min_max_value}: {pts.quantify()} %')

    print(f'Quantify predicted data when SR355 x {const} \n{pts_stats}')

    pts_stats = pd.DataFrame(pts_stats, index=unit_value)
    pts_stats.to_pickle(Path(home_dir, f'{dataset_name}-{method_name}-Stats-between-{min_max_value[0]}-{min_max_value[1]}-{condt.value}-{condt.close}.pkl'))
