import sys

import sys
from cells_simulation import *

import sys
sys.path.insert(0, "../")

from plotting_src.read_ggp_run import * 
import numpy as np
import subprocess

# ========== Simulation parameters ========== #
save_dataset = True    # saves data set as csv
# run_ggp = True         # runs the ggp code (located in "../bin/" relative to this nb)

dt = 1e-2
dt_measument = 3 # in minutes
n_cells = 100 
discard_cells = 5
lanes = 10

div_mode = "adder"

division_log_length = 1+np.log(2)   # for sizer: division, when log_length hits division_log_length
division_time = 3000 - 1e-10          # for timer: division, when cell cycle time hits division_time
division_addition = np.log(2)       # for adder: divsion, when division_addition in log_length was added in cell cycle



const_q =True

if const_q:
    out_dir = mk_missing_dir('../../fluctuations_paper_data/simulations/vary_cv_gamma_n{:d}_{:s}_const_q'.format(n_cells*lanes, div_mode))
else:
    out_dir = mk_missing_dir('../../fluctuations_paper_data/simulations/vary_cv_gamma_n{:d}_{:s}_noise_q'.format(n_cells*lanes, div_mode))
    
for norm_tau2 in np.arange(0.1, 0.8, 0.1):
    for cv in np.arange(0.1, 0.8, 0.1):
        mean_lambda = 0.0114535
        gamma_lambda = mean_lambda/norm_tau2
        var_lambda = 2*gamma_lambda *(cv*mean_lambda)**2
        
        print(gamma_lambda, var_lambda)
        # ========== Model parameters ========== #
        parameter_set = {   "mean_lambda": mean_lambda, 
                            "gamma_lambda": gamma_lambda,
                            "var_lambda": var_lambda,
                            "mean_q": 167.754,
                            "gamma_q": 0.0448522,
                            "beta": 0.,
                            "var_x": 0.,
                            "var_g": 0.,
                            "var_dx": 0.,
                            "var_dg": 0.}
        if const_q:
            parameter_set["var_q"] = 0.
        else:
            parameter_set["var_q"] = 414.859
        dataset = pd.DataFrame()
        for i in np.arange(lanes):
            cells_simulated = simulate_cells(dt, n_cells, parameter_set, div_mode,
                                            division_log_length, 
                                            division_time, 
                                            division_addition, 
                                            tree=False, 
                                            discard_cells=discard_cells)

            temp_dataset = build_data_set_fixed_dt(cells_simulated, parameter_set['var_x'], parameter_set['var_g'], dt_measument,atol=dt/10)
            temp_dataset['lane'] = i+1
            # add the last "lane" to the data frame
            dataset = dataset.append(temp_dataset)
        
        # ----------- SAVE ----------- #
        directory, filename = get_next_file_name(out_dir)
        
        write_param_file(os.path.join(directory, "parameters_simulation.txt"), parameter_set) 

        # csv_config_file = write_csv_config(os.path.join(directory, "csv_config.txt"), lane="lane")
        
        dataset.to_csv(filename)
        
        print("New simulation saved in", filename)

    