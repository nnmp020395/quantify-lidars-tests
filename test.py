import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pickle
import numpy.ma as ma
import pickle

import sys
home_dir = '/homedata/nmpnguyen/comparaison/'
sys.path.append(Path(home_dir, 'Codes'))
from fonctions import dataset, conditions, get_file, check, conversion, plots

from datetime import datetime

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#---------------------

ipral_characters = dataset('calibrated', 'simulated', 'flags', 0, 20000)

#---------------------
dataset_name = 'ipral_2020'
maindir = '/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/'
pattern = 'ipral_1a_Lz1R15mF30sPbck_v01_2020*_000000_1440.nc'
#---------------------

# THEORICAL METHOD
# method_name = 'THEORICAL'

# print(f'{now} {method_name}')
# print(f'----------------GENERATE DATA--------------------')
# condt = conditions(1.0, 'right', None)
# print('----355')
# dataset355, ids = get_file.get_folder_dataset(maindir, pattern, ipral_characters, 355, grouped=True, conditionF=condt, return_index=False)
# print(ids)
# # dataset355 = dataset355[ids]
# condt2 = conditions('None', 'None', None)
# print('----532')
# dataset532mes, _ = get_file.get_folder_dataset(maindir, pattern, ipral_characters, 532, True, condt2, False)
# dataset532pred = conversion.conversion_theorical(dataset355, 5.3)
# print(dataset355.shape, dataset532mes.shape, dataset532pred.shape)

# # if condt is None : 
# #     condt = conditions('None', None, 'None')

# with open(Path(home_dir,f'{dataset_name}-355-mes_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'), 'wb') as output_dataset:
#     pickle.dump(dataset355, output_dataset)          

# with open(Path(home_dir,f'{dataset_name}-532-mes_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'), 'wb') as output_dataset:
#     pickle.dump(dataset532mes, output_dataset)

# with open(Path(home_dir,f'{dataset_name}-532-pred_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'), 'wb') as output_dataset:
#     pickle.dump(dataset532pred, output_dataset)
    
#---------------------
# READ PICKLE DATA
# THEORICAL METHOD
# print('----------------GET DATA--------------------')
# dataset532mes = pd.read_pickle(Path(home_dir,f'{dataset_name}-532-mes_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'))
# dataset532pred = pd.read_pickle(Path(home_dir,f'{dataset_name}-532-pred_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'))
# NotNans = np.logical_and(~np.isnan(dataset532mes.ravel()), ~np.isnan(dataset532pred.ravel()))

# dataset532mes = dataset532mes.ravel()[NotNans]
# dataset532pred = dataset532pred.ravel()[NotNans]

# NotNegative = np.logical_and(dataset532mes > 0, dataset532pred > 0)
# dataset532mes = dataset532mes[NotNegative]
# dataset532pred = dataset532pred[NotNegative]

# dataset355mes = xr.open_dataarray(f'/homedata/nmpnguyen/comparaison/IPRAL2020-355-mes_THEORICAL_CONDT-SR355-{condt.value}-{condt.close}.nc').values
# dataset532mes = xr.open_dataarray(f'/homedata/nmpnguyen/comparaison/IPRAL2020-532-mes_THEORICAL_CONDT-SR355-{condt.value}-{condt.close}.nc').values
# mask = np.logical_and(dataset355mes>0, ~np.isnan(dataset355mes))
# dataset355mes = dataset355mes[mask]
# dataset532mes = dataset532mes[mask]

# dataset532pred = dataset355mes*5.3
#---------------------
# READ PICKLE DATA
# LEARNING METHOD
method_name = 'LEARNING'
condt = conditions('None', 'None', None)

print('----------------GET LEARNED DATA--------------------')

home_dir = '/home/nmpnguyen/conversion_model/comparaison/'
years = ['2018', '2019', '2020']
pattern = 'ipral_'+'-'.join(years)
dataset_name = 'IPRAL'.join(years)
condt = conditions('None', 'None', None)
condt.name = 'None'

dataset355mes = pd.read_pickle(Path(home_dir, f'{pattern}_learned_train_dataset.pkl')).values
dataset532mes = pd.read_pickle(Path(home_dir, f'{pattern}_learned_traintarget_dataset.pkl')).values


# if (condt.close != 'None'):
#     datatrain_before = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_train_dataset.pkl')).values
#     _, ids = get_file.get_closed_data(datatrain_before[:, 0], condt.value, condt.close)
#     ids = ids[0]
#     datatrain = datatrain_before[ids, :]
#     targettrain = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_traintarget_dataset.pkl')).values[ids,:]


#     datatest_before = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_TEST_dataset.pkl')).values
#     _, ids = get_file.get_closed_data(datatest_before[:,0], condt.value, condt.close)
#     ids = ids[0]
#     datatest = datatest_before[ids,:]
#     dataset532mes = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_TESTtarget_dataset.pkl')).values[ids,:]
# else:
#     datatrain = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_train_dataset.pkl')).values
#     targettrain = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_traintarget_dataset.pkl')).values
#     datatest = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_TEST_dataset.pkl')).values
#     dataset532mes = pd.read_pickle(Path(home_dir,f'{dataset_name}_learned_TESTtarget_dataset.pkl')).values

# print(datatrain.shape, targettrain.shape, datatest.shape)

print('----------------LEARNING--------------------')
# model_loaded = pickle.load(open('/home/nmpnguyen/conversion_model/tree_3f.sav', 'rb'))
# dataset532pred = conversion.conversion_by_learning(datatrain, targettrain, datatest)
# pd.DataFrame(dataset532pred).to_pickle(Path(home_dir, f'{dataset_name}-testpredict_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'))

with open(Path(home_dir, 'comparaison_model.sav'), 'rb') as f:
    loaded_model = pickle.load(f)

dataset532pred = loaded_model.predict(dataset355mes)
pd.DataFrame(dataset532pred).to_pickle(Path(home_dir, f'{pattern}_learned_TESTpredict_dataset.pkl'))

#---------------------
print('----------------QUANTIFY--------------------')

if (len(dataset532mes.shape) == 2) :
    dataset532mes = dataset532mes.ravel()
if (len(dataset532pred.shape) == 2) :
    dataset532pred = dataset532pred.ravel()

# unit_value = [0.05, 0.1, 0.15]
unit_value = np.arange(0.05, 1.5, 0.1)
min_max_value = [0.0,80.0]
pts_stats = []
for u in unit_value:
    pts = check(min_max_value[0], min_max_value[1], min_max_value[0], min_max_value[1], u, dataset532mes, dataset532pred)
    pts_stats.append(pts.quantify())
    # print(f'Quantify predicted data within +/- {u} unit around the diagonal and between {min_max_value}: {pts.quantify()} %')

print(f'Quantify predicted data \n{pts_stats}')

pts_stats = pd.DataFrame(pts_stats, index=unit_value)
pts_stats.to_pickle(Path(home_dir, f'{dataset_name}-{method_name}-Stats-between-{min_max_value[0]}-{min_max_value[1]}-{condt.value}-{condt.close}.pkl'))

# print('----------------PLOT--------------------')

# captions_plots = [f'{dataset_name} all dataset flagged, {dataset532mes.shape[0]} points (Not NaN)', 
#     "_".join(['SR355', str(condt.value), condt.close])]
# plot_params = plots(method_name, dataset_name, pts_stats, unit_value, min_max_value[0], min_max_value[1], dataset532mes, dataset532pred,
#     captions = captions_plots, labels=['SR532 measured', 'SR532 predicted'], output_path=home_dir)
# plot_params.precent_hist2d()

