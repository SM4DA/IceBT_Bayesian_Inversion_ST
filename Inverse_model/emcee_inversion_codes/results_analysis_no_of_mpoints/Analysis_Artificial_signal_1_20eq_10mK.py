#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import dill
#from tqdm import tqdm
import time
from scipy.stats import qmc
from scipy.interpolate import CubicSpline
import scipy.stats as stats
from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import Matern

from ..Forward_model.generate_IC_fixed_grid import get_generate_IC_fixed_grid
from ..Forward_model.solution_using_FD import get_solution_using_FD
from ..Forward_model.linear_interpolation import interpolate_linear,interpolate_linear_fill_value
from ..Forward_model.cubic_spline_interpolation import interpolate_cubicSpline

import emcee


#-----------------------------------------------------------------combine_chains------------autocorrelation_time_caluculation------------------------------------------------- 


sampler1=None 
with open('/albedo/work/projects/p_inverT/Ice_BT/output/sampler_output/KS_final_runs/Artificial_signal_1_20eq_10mK.pkl', 'rb') as f:
    sampler1=dill.load(f)

sampler2=None
with open('/albedo/work/projects/p_inverT/Ice_BT/output/sampler_output/KS_final_runs/Artificial_signal_1_20eq_10mK_conti1.pkl', 'rb') as f2:
    sampler2=dill.load(f2)
    
chains = []

for splr in [sampler1, sampler2]:
    if splr is not None:
        chains.append(splr.get_chain())

# Concatenate along the step axis (nsteps)
combined_chain = np.concatenate(chains, axis=0)

print("Combined chain shape:", combined_chain.shape)

print("Calculates autocorrelation time: failure implies chain is not converged")

# Autocorrelation time
tau = emcee.autocorr.integrated_time(combined_chain)
print("Tau:", tau)

print("Calculated autocorrelation time--> chain converged")

tau_max = np.max(tau)
print("tau_max:", tau_max)

#-----------------------------------------------------------------forward_model_parameters---function_for_creating_time_series--------------------------------------------- 
np.random.seed(9001)


# Initialize an empty dictionary to store the data
ics={}
with open('/albedo/work/projects/p_inverT/Ice_BT/ICs/IC_ws64_wb0_tb14_H2782_a24634_ic.pkl', 'rb') as f:
    ics=dill.load(f)

EDML=pd.read_excel('/albedo/work/projects/p_inverT/Ice_BT/Forward_model/paper_site_densities.xlsx')
borehole3000_densityProf=list(EDML['den_EDML'].dropna())
borehole3000_depths= list(EDML['dep_EDML'].dropna())

def get_fwd_param():
    

    ws=64
    Tb=-1.4
    wb=0
    
    m=11
    alpha=2.4634
    beta=0
    velocity=[]
    max_depth=2782
    max_time=500-1
    dt_approx=2**-4
    dz_approx=2**2
    
    return max_depth, max_time, dz_approx, dt_approx, velocity,Tb,alpha,ws, wb,m

def select_ICs(ics,Ts_new):
    ic=ics['%.3f'%(np.round(Ts_new,3))]
       
    return ic

max_depth, max_time, dz_approx, dt_approx, velocity,Tb,alpha,ws, wb,m = get_fwd_param()
tem_ic=select_ICs(ics,-45.0)
fwd_Ts=get_solution_using_FD(max_depth, max_time, dz_approx, dt_approx, borehole3000_depths, borehole3000_densityProf,velocity,tem_ic,-45.0,Tb,alpha,ws, wb,m)
time_steps=fwd_Ts.times
tim=time_steps+1524#1024

def kernel_based_model(t, t_i, alpha_i, length_scale, nu):

    t = np.atleast_1d(t).reshape(-1, 1)
    t_i = np.array(t_i).reshape(-1, 1)
    alpha_i = np.array(alpha_i).reshape(-1)

    kernel = Matern(length_scale=length_scale, nu=nu)

    K = kernel(t, t_i)

    st = K @ alpha_i 
    
    return st

def generate_surface_temperature(alpha_list,time_list,time_steps,pom,scale_value):
    
    t_i = time_list 
    
    alpha_i = alpha_list

    t_eval =time_steps
    st_values = kernel_based_model(t_eval, t_i, alpha_i, length_scale=scale_value, nu=np.inf)+pom

    return  st_values.flatten()
    

#--------------------------------------------------------------------------creating_time_series--------------------------------------------------------------------------  
scale_value=20
burnin=20000


splrs = [sampler1, sampler2]

# 1. Combine chains
full_chain = np.concatenate(
    [splr.get_chain() for splr in splrs if splr is not None],
    axis=0
)

full_log_prob = np.concatenate(
    [splr.get_log_prob() for splr in splrs if splr is not None],
    axis=0
)

# 2. Discard global burn-in
chain_burned = full_chain[burnin:, :, :]
log_prob_burned = full_log_prob[burnin:, :]

# 3. Flatten
flat_samples = chain_burned.reshape(-1, chain_burned.shape[-1])
log_probs = log_prob_burned.reshape(-1)


max_prob_index = np.argmax(log_probs)
map_estimate = flat_samples[max_prob_index]

N_alpha = 40

alpha_map = map_estimate[:N_alpha]   # all the alpha parameters
POM_map = map_estimate[N_alpha]  

means = np.mean(flat_samples, axis=0)
stds = np.std(flat_samples, axis=0)
param_stats_array = np.column_stack((np.arange(len(means)), means, stds))

t_init = np.linspace(0, 499, 40)  

annual_time_steps=np.arange(0,500,1)

def surface_temp_updated_wrapper(args):
    sample_row, t_init, annual_time_steps, scale_value = args
    return generate_surface_temperature(sample_row[:-1], t_init, annual_time_steps, sample_row[-1], scale_value)

args_list = [
    (flat_samples[i], t_init, annual_time_steps, scale_value)
    for i in range(flat_samples.shape[0])
]
start = time.time()

from multiprocessing import Pool

with Pool(122) as pool:
    results = pool.map(surface_temp_updated_wrapper, args_list)
    

#------------------------------------------------------------------------posterior_analysis------------------------------------------------------------------------------


qmean= generate_surface_temperature(param_stats_array[:-1,1], t_init, annual_time_steps,param_stats_array[-1,1],scale_value)
qmap=generate_surface_temperature(list(alpha_map), t_init, annual_time_steps,POM_map,scale_value)  
sts=np.array(results)
s_count=flat_samples.shape[0]
POM_map=POM_map
alpha_map=alpha_map
POM_mean=param_stats_array[-1,1]
param_stats_array=param_stats_array
lower_bound = np.percentile(sts, 2.5, axis=0)
upper_bound = np.percentile(sts, 97.5, axis=0)

#----------------------------------------------------------------------------saving_results------------------------------------------------------------------------------
np.save('/albedo/work/projects/p_inverT/Ice_BT/output/output_analysed/KS_final_runs/Artificial_signal_1_20eq_10mK_POM_mean',POM_mean)
np.save('/albedo/work/projects/p_inverT/Ice_BT/output/output_analysed/KS_final_runs/Artificial_signal_1_20eq_10mK_POM_map',POM_map)
np.save('/albedo/work/projects/p_inverT/Ice_BT/output/output_analysed/KS_final_runs/Artificial_signal_1_20eq_10mK_ic',select_ICs(ics,qmean[0]))
np.save('/albedo/work/projects/p_inverT/Ice_BT/output/output_analysed/KS_final_runs/Artificial_signal_1_20eq_10mK_qmean',qmean)
np.save('/albedo/work/projects/p_inverT/Ice_BT/output/output_analysed/KS_final_runs/Artificial_signal_1_20eq_10mK_qmap',qmap)
np.save('/albedo/work/projects/p_inverT/Ice_BT/output/output_analysed/KS_final_runs/Artificial_signal_1_20eq_10mK_lower_bound',lower_bound)
np.save('/albedo/work/projects/p_inverT/Ice_BT/output/output_analysed/KS_final_runs/Artificial_signal_1_20eq_10mK_upper_bound',upper_bound)

