#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import dill
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

np.random.seed(9001)

ics={}
with open('/albedo/work/projects/p_inverT/Ice_BT/ICs/IC_ws64_wb0_tb14_H2782_a24634_ic.pkl', 'rb') as f:
    ics=dill.load(f)


measured_depths=np.linspace(0,200,40)
measured_temps=np.load('/albedo/work/projects/p_inverT/Ice_BT/data/case_study/Realistic_artificial_measurements/Realistic_artificial_measured_temps_40eq_rsigna_2_smooth_ensemble.npy')[11]
measurmt_error=0.001 # standard deviation of measurement uncertainty in K 

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

def generate_surface_temperature(alpha_list,time_list,time_steps,pom):
    
    t_i = time_list 
    
    alpha_i = alpha_list

    t_eval =time_steps
    st_values = kernel_based_model(t_eval, t_i, alpha_i, length_scale=20, nu=np.inf)+pom

    return  st_values.flatten()
    
mu_T=0
sigma_pT=1.2 # prior on 40 kernel weights


def ln_prior_alpha(p):
    k = len(p)
    T=p
    T_prior=np.zeros(k)+mu_T

    sig_T = np.zeros((k,k))
    np.fill_diagonal(sig_T, sigma_pT**2, wrap=True)
    sig_T = np.asmatrix(sig_T)

    prior=multivariate_normal.logpdf(T, T_prior, sig_T) 

    return prior

def ln_prior_pom(p):
    return -np.inf if not (-50.0 < p < -40.0) else 0


def lnlike(p, tlist, measured_depths, measured_temps, sigma_m): 

    max_depth, max_time, dz_approx, dt_approx, velocity,Tb,alpha,ws, wb,m = get_fwd_param()
    
    Alist=list(p[:-1])
    tlist=list(tlist)
    pom=p[-1]

    qnew= generate_surface_temperature(np.array(Alist), np.array(tlist), time_steps, pom)

    lp_ic = ln_prior_pom(np.round(qnew[0],3))

    if not np.isfinite(lp_ic):
        return -np.inf   

    else:
               
        _ic=select_ICs(ics,qnew[0])
 
        T_new_model=get_solution_using_FD(max_depth, max_time, dz_approx, dt_approx, borehole3000_depths, borehole3000_densityProf,velocity,_ic,qnew,Tb,alpha,ws, wb,m)
        T_new=interpolate_cubicSpline(x_ref=T_new_model.depths,y_ref=T_new_model.T[:,-1],x_req=measured_depths)
    
        lp= multivariate_normal.logpdf(T_new, measured_temps, sigma_m) 

        return lp + lp_ic
    
def lnprob(p,tlist,measured_depths,measured_temps,sigma_m):

    lp1 = ln_prior_alpha(p[:-1])
    lp2 = ln_prior_pom(p[-1])
    if not np.isfinite(lp1) or not np.isfinite(lp2):
        return -np.inf
    ll = lnlike(p,tlist,measured_depths,measured_temps,sigma_m)
    return lp1 +lp2 + ll


Nwalker=122
Ndim=41
t_init = np.linspace(0, 499, 40)  
    
#alpha_init = np.random.normal(loc=0, scale=sigma_pT, size=40)
#POM_init = -45.0

#p0 = [
#    list(alpha_init + np.random.normal(0, 0.005, Ndim-1)) + 
#    [POM_init + np.random.normal(0, 1)] 
#    for i in range(Nwalker)
#]

with open('/albedo/work/projects/p_inverT/Ice_BT/output/sampler_output/KS_final_runs/Realistic_smooth_signal_PA1_12_1mK.pkl', 'rb') as f:
    loaded_sample=dill.load(f)

last_state = loaded_sample.get_last_sample()
last_positions = last_state.coords
rstate = last_state.random_state 


Nm=len(measured_depths)
sig_m = np.zeros((Nm,Nm))
np.fill_diagonal(sig_m, measurmt_error**2, wrap=True)
sigma_m = np.asmatrix(sig_m)

import emcee
from multiprocessing import Pool

 
with Pool(122) as pool:
    sampler = emcee.EnsembleSampler(
            Nwalker, Ndim, lnprob,
            args=(list(t_init), measured_depths, measured_temps, sigma_m),
            pool=pool
        )
    pos,prob,state = sampler.run_mcmc(last_positions, 40000, rstate0=rstate, progress=True)
    #pos,prob,state = sampler.run_mcmc(p0, 80000)


with open('/albedo/work/projects/p_inverT/Ice_BT/output/sampler_output/KS_final_runs/Realistic_smooth_signal_PA1_12_1mK_conti1.pkl', 'wb') as f:
    dill.dump(sampler,f)



