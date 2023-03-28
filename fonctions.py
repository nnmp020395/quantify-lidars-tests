import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pickle
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

'''
Fct de conversion théorique 
'''
class conversion:
    def conversion_theorical(input_sr355, coef=5.3):
        print('Theorical coef:', coef)
        print('Shape of sr355 input:', input_sr355.shape)
        output_array532 = coef * input_sr355
        print('Shape of sr532 output', output_array532.shape)
        return output_array532


    def conversion_by_learning(input_train, input_target, input_test):
        '''
        Read model or learn model ? 
        '''
        from sklearn.tree import DecisionTreeRegressor
        learning_model = DecisionTreeRegressor()
        learning_model.fit(input_train, input_target)
        output_data = learning_model.predict(input_test)
        return output_data


class conditions:
    def __init__(self, value: float, close: str, id_where):
        self.value = value
        self.close = close 
        self.where = id_where

        
class dataset:
    def __init__(self, calibrated_variable: str, simulated_variable: str, 
                 flag_variable: str, flag_value_valid: int, limite_altitude: float):
        self.calibrated_variable = calibrated_variable
        self.simulated_variable = simulated_variable
        self.flag_variable = flag_variable
        self.flag_value_valid = flag_value_valid
        self.limite_altitude = limite_altitude

#------------------------------------------

'''
Fct de récupération des données (1 profil, 1 dataset) et 
sélection en fonction de condition (input by user)
'''
class get_file:
    def get_closed_data(input_data, value, close):
        if close == 'left':                
            output_data = ma.masked_where((input_data > 0.0)&(input_data < value), input_data)
            output_data = output_data.filled(fill_value=np.nan)
            id_where = np.where(input_data > value)
            print(id_where)
        elif close == 'right':
            output_data = ma.masked_where(input_data > value, input_data)
            output_data = output_data.filled(fill_value=np.nan)   
            id_where = np.where(input_data < value)
            print(id_where)
        elif close == 'both':
            print()
            id_where = []
        elif (close is None):
            output_data = input_data
            id_where = []
        return output_data, id_where


    def set_dataset_with_condition(input_data, condition=None, return_index=False):   

        print('Shape of input:', input_data.shape)    
        print('Conditions for apply: ', condition.value, condition.close, condition.where)
        #------------------------
        if (condition.where is not None):
            print('1st case')
            id_where = condition.where
            data_where = input_data[id_where]
            output_data, _ = get_file.get_closed_data(data_where, condition.value, condition.close)            
        else: 
            print('2nd case')
            output_data, id_where = get_file.get_closed_data(input_data, condition.value, condition.close)
             
        #------------------------    
        print('Shape of output after apply conditions:', output_data.shape)
        if (return_index):
            return output_data, id_where
        else:
            return output_data


    def get_file_dataset(mainfile, characters, wavelength, conditionf=None, return_index = False):
        '''
        dataset should be the same format: 
            - netCDF
            - variable: calibrated, simulated, flags
        '''
        input_data = xr.open_dataset(mainfile)
        limites_range = input_data['range'][input_data['range'] < characters.limite_altitude]

        sr_data = input_data[characters.calibrated_variable].sel(wavelength=wavelength, range=limites_range)/input_data[characters.simulated_variable].sel(wavelength=wavelength, range=limites_range)
        flagged_data = sr_data.where(input_data[characters.flag_variable].sel(wavelength=wavelength)==characters.flag_value_valid, drop=False)

        output_data = flagged_data.resample(time = '15min', skipna=True).mean('time')
        # print('get_file_dataset', conditionf.where)
        final_output_data = get_file.set_dataset_with_condition(output_data, conditionf, return_index = False)          
        return final_output_data



    def get_folder_dataset(mainfolder: str, patternfile: str, characters, wavelength, 
                           grouped=False, conditionF=None, return_index=False):
        from tqdm import tqdm
        listfiles = sorted(Path(mainfolder).glob(patternfile))
        outputs_data = []
        if grouped:
            for file in tqdm(listfiles):
                print(file)          
                output_1_data = get_file.get_file_dataset(file, characters, wavelength, conditions(np.nan, None, None), False)
                outputs_data.append(output_1_data)
        
            grouped_outputs_data = np.concatenate(outputs_data, axis=0)

            # check y_shape of grouped output data
            #-----------------------------------    
            if (output_1_data.shape[1] == grouped_outputs_data.shape[1]):
                print('Shape of output data after groupping', grouped_outputs_data.shape)
                print('------------Groupping: Done-------------')
                # return grouped_outputs_data
            else:
                print('------------Groupping: Error-------------')
                print('Shape of 1 output:', output_1_data.shape)
                print('Shape of list outputs:', outputs_data.shape)
                return 0
        else:
            for file in tqdm(listfiles):
                print(file)    
                print('before', conditionF.where)        
                output_1_data = get_file.get_file_dataset(file, characters, wavelength, None, True)
                print('after', conditionF.where)
                outputs_data.append(output_1_data)

            grouped_outputs_data = outputs_data
            print('Shape of output data without groupping', grouped_outputs_data.shape)
            print('------------Groupping: Done-------------')

        final_output_data, ids_where = get_file.set_dataset_with_condition(grouped_outputs_data, conditionF, return_index=True)
        print('Shape of output data after setting conditions', final_output_data.shape)
        print('------------Setting: Done-------------')
        return final_output_data, ids_where


#------------------------------------------

'''
Fonctions for create 2d-histogram & 
get proportion following closed unit of the diagonal of 2d-histogram
'''
    
class check:
    def __init__(self, min_value_x: float, min_value_y: float, max_value_x: float, max_value_y: float, closed_unit: float,
                 x_data: float, y_data: float):
        self.x1, self.y1 = min_value_x-closed_unit, min_value_y
        self.x2, self.y2 = min_value_x+closed_unit, min_value_y
        self.x3, self.y3 = max_value_x+closed_unit, max_value_y
        self.x4, self.y4 = max_value_x-closed_unit, max_value_y
        if (len(x_data.shape) > 1) | (len(y_data.shape) > 1) :
            print('No support for Data2D')
            return None
        else:
            self.x, self.y = x_data, y_data
 
    # A function to check whether point P(x, y) lies inside the rectangle
    # formed by A(x1, y1), B(x2, y2), C(x3, y3) and D(x4, y4)
    def check_point(self):
        if self is None: 
            print('Cannot found data to quantify')
            return 0
        else: 
            area = lambda X1, Y1, X2, Y2, X3, Y3 : abs((X1 * (Y2 - Y3) + X2 * (Y3 - Y1) + X3 * (Y1 - Y2)) / 2.0)
            # Calculate area of rectangle ABCD
            A = (area(self.x1, self.y1, self.x2, self.y2, self.x3, self.y3) + 
                 area(self.x1, self.y1, self.x4, self.y4, self.x3, self.y3))

            # Calculate area of triangle PAB
            A1 = area(self.x, self.y, self.x1, self.y1, self.x2, self.y2)

            # Calculate area of triangle PBC
            A2 = area(self.x, self.y, self.x2, self.y2, self.x3, self.y3)

            # Calculate area of triangle PCD
            A3 = area(self.x, self.y, self.x3, self.y3, self.x4, self.y4)

            # Calculate area of triangle PAD
            A4 = area(self.x, self.y, self.x1, self.y1, self.x4, self.y4)

            # Check if sum of A1, A2, A3
            # and A4 is same as A
            # print('Aire PAB + PBC + PCD + PAD', np.round(A1 + A2 + A3 + A4, decimals=2))
            # print('Aire A', np.round(A, decimals=2))
            return (np.round(A, decimals=2) == np.round(A1 + A2 + A3 + A4, decimals=2))

    def quantify(self):
        if self is None:
            return 0
        else:
            points_checked = self.check_point()
            proportion_coef = 100*np.where(points_checked == True)[0].shape[0]/points_checked.shape[0]
            return proportion_coef
        

class plots:
    def __init__(self, method_name, dataset_name, stats, closed_units, min_value, max_value, mesures, predictes, 
        captions, labels, output_path):
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.stats = stats
        self.closed_units = closed_units
        bins_array = np.arange(min_value, max_value, 0.1)
        H, self.xedges, self.yedges = np.histogram2d(mesures, predictes, bins = [bins_array, bins_array])
        self.bins_array = bins_array
        Hpercents = (H/mesures.shape[0])*100
        self.Hpercents = Hpercents
        self.labels = labels
        self.captions = captions
        self.output_path = output_path

    def precent_hist2d(self):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        p = ax.pcolormesh(self.xedges, self.yedges, (self.Hpercents).T, norm=LogNorm(vmin=1e-3, vmax=1e0))
        plt.colorbar(p, ax=ax, extend='both', label='Probability, %')
        # the diagonal line
        ax.plot(self.bins_array, self.bins_array, '-k')
        ax.plot(self.bins_array + self.closed_units[-1], self.bins_array, linestyle ='--', label = f'+/- {self.closed_units[-1]}', color = 'red')
        ax.plot(self.bins_array - self.closed_units[-1], self.bins_array, linestyle ='--', color = 'red')
        # code_colors_lines = ['m', 'g', 'c']
        # for unit, c in zip(self.closed_units, code_colors_lines):
        #     ax.plot(self.bins_array + unit, self.bins_array, linestyle ='--', label = f'+/- {unit}', color = c)
        #     ax.plot(self.bins_array - unit, self.bins_array, linestyle ='--', color = c)
        ax.legend()
        # grid
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
        # title
        plt.suptitle(f'{self.method_name}-{self.dataset_name}', ha='left', fontsize=16)
        # subtitle        
        title_chains = [f'+/- {unit} : {np.round(sts, decimals=2)}% \n' for unit,sts in zip(self.closed_units, self.stats)]
        title_chains.append(self.captions[0])
        plt.title(" ".join(title_chains), loc='left', fontsize=11)
        # AXIS LABELS
        plt.ylabel(self.labels[1])
        plt.xlabel(self.labels[0])
        # CAPTION
        plt.text(-0.5, -10.5, self.captions[0], ha='left', fontsize = 11, alpha=0.9)

        plt.tight_layout()
        print(Path(self.output_path, f'{self.dataset_name}_{self.method_name}_{self.captions[1]}.png'))
        plt.savefig(Path(self.output_path, f'{self.dataset_name}_{self.method_name}_{self.captions[1]}.png'))
        plt.close()