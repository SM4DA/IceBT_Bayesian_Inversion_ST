import numpy as np
import pandas as pd
import dill 
from tqdm import tqdm
import time
from scipy.stats import qmc
import random
import scipy.stats as stats
from scipy.stats import multivariate_normal

from ...Forward_model.generate_IC_fixed_grid import get_generate_IC_fixed_grid
from ...Forward_model.solution_using_FD import get_solution_using_FD
from ...Forward_model.linear_interpolation import interpolate_linear,interpolate_linear_fill_value
from ...Forward_model.cubic_spline_interpolation import interpolate_cubicSpline
import os
np.random.seed(9001)

ics={}
with open('/home/kshaju/IceBT_Bayesian_inversion/Forward_model/IC_ws64_wb0_tb14_H2782_a24634_ic.pkl', 'rb') as f:
    ics=dill.load(f)

EDML=pd.read_excel('/home/kshaju/IceBT_Bayesian_inversion/Forward_model/paper_site_densities.xlsx')
borehole3000_densityProf=list(EDML['den_EDML'].dropna())
borehole3000_depths= list(EDML['dep_EDML'].dropna())

def surface_temp_updated(Temp_list,time_list,time_steps):
    q=interpolate_linear(x_ref=time_list,y_ref=Temp_list,x_req=time_steps)
    return q

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
tem_ic=select_ICs(ics,-45.000)
fwd_Ts=get_solution_using_FD(max_depth, max_time, dz_approx, dt_approx, borehole3000_depths, borehole3000_densityProf,velocity,tem_ic,-45.0,Tb,alpha,ws, wb,m)
time_steps=fwd_Ts.times
tim=time_steps+1524#1024

samples=None
with open('/home/kshaju/IceBT_Bayesian_inversion/RJMCMC_reconstructions/output/sampler_output/A1_0m_signal_2_30mk_10m3.pkl', 'rb') as f:
    samples=dill.load(f)

samples_aft_burnin=samples[:500000]

sts=np.zeros((len(samples_aft_burnin),time_steps.shape[0]))
s_count=0
for x in samples_aft_burnin:
    if x != 0:
        name, tlist, Tlist, POM, selected_pos= x.name, x.t_points, x.T_points, x.POM, x.selected_pos
        Tlist=list(Tlist)
        tlist=list(tlist)

        q= surface_temp_updated(np.array(Tlist), np.array(tlist), time_steps)
        sts[s_count,:]=q
        s_count=s_count+1

np.save('/home/kshaju/IceBT_Bayesian_inversion/RJMCMC_reconstructions/output/output_analysis/sts_A1_0m_signal_2_30mk_10m3',sts)

