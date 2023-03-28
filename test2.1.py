import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



years = ['2018', '2019', '2020']
maindir = '/homedata/nmpnguyen/IPRAL/NETCDF/v_simple'
pattern = 'ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'
# all_sr355 = []
# all_sr532 = []
# for year in years:
#     for file in sorted(Path(maindir, year).glob(pattern)):
#         print(file)
#         dt = xr.open_dataset(file)
#         z = (dt['range'] < 20000)

#         sr355 = (dt['calibrated']/dt['simulated']).isel(range = z, wavelength=0)
#         sr355 = sr355.where(dt['flags'].isel(wavelength=0) == 0, drop=True)   
#         sr532 = (dt['calibrated']/dt['simulated']).isel(range = z, wavelength=1)
#         sr532 = sr532.where(dt['flags'].isel(wavelength=1) == 0, drop=True)
#         if (sr355.shape[0] != 0) | (sr532.shape[0] != 0):
#             sr355 = sr355.resample(time = '15min').mean('time', skipna=True)
#             sr532 = sr532.resample(time = '15min').mean('time', skipna=True)

#         all_sr355.append(sr355)
#         all_sr532.append(sr532)

# all_sr355 = xr.concat(all_sr355, dim='time')
# all_sr532 = xr.concat(all_sr532, dim='time')

# a0, a = np.intersect1d(all_sr355['time'], all_sr532['time'], return_indices=True)[1:]
# all_sr355 = all_sr355.isel(time=a0)
# all_sr532 = all_sr532.isel(time=a)

# all_sr355.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-355-mes_THEORICAL_CONDT-None-None-None.nc')
# all_sr532.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-532-mes_THEORICAL_CONDT-None-None-None.nc')

method_name = 'THEORICAL'

print(f'----------------GENERATE DATA--------------------')
condt = conditions('None', 'None', None)

dataset355mes = xr.open_dataarray(f'/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-355-mes_THEORICAL_CONDT-None-None-None.nc').values
dataset532mes = xr.open_dataarray(f'/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-532-mes_THEORICAL_CONDT-None-None-None.nc').values
mask = np.logical_and(dataset355mes>0, ~np.isnan(dataset355mes))
dataset355mes = dataset355mes[mask]
dataset532mes = dataset532mes[mask]

print('----------------CONVERSION--------------------')
const_test = np.arange(4.8, 5.5, 0.1)

for const in const_test:
    dataset532pred = dataset355mes * const   
    method_name = f'THEORITICAL-{const}'

    print('----------------QUANTIFY--------------------')

    unit_value = np.arange(0.05, 1.5, 0.1)
    min_max_value = [0.0,80.0]
    pts_stats = []
    for u in unit_value:
        pts = check(min_max_value[0], min_max_value[1], u, dataset532mes, dataset532pred)
        pts_stats.append(pts.quantify())
        # print(f'Quantify predicted data within +/- {u} unit around the diagonal and between {min_max_value}: {pts.quantify()} %')

    print(f'Quantify predicted data when SR355 x {const} \n{pts_stats}')

    pts_stats = pd.DataFrame(pts_stats, index=unit_value)
    pts_stats.to_pickle(Path(home_dir, f'{dataset_name}-{method_name}-Stats-between-{min_max_value[0]}-{min_max_value[1]}-{condt.value}-{condt.close}.pkl'))
