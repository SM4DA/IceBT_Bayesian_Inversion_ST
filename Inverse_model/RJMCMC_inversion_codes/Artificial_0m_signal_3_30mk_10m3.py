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

measured_temps=np.load('/home/kshaju/IceBT_Bayesian_inversion/Data/Gaussian_pulse_signals/Artificial_measurements/Artificial_measurements_4_2_temps_40eq_signal_3.npy')
measured_depths=np.linspace(0,200,40)
measured_error=0.03

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


def surface_temp_updated(Temp_list,time_list,time_steps):
    q=interpolate_linear(x_ref=time_list,y_ref=Temp_list,x_req=time_steps)
    return q

# Possible model types:
# (i) PERTURB_T: Perturb one temperature value, T_i
# (ii) PERTURB_t: Perturb one time value, t_i.
# (iii) BIRTH: Create a new GST point (birth).
# (iv) DEATH: Delete one GST point (death).
# (v) PERTURB_POM: Perturb pre observational mean

# Model definition:
class MODELS:
    def __init__(self,name=None,T_points=None,t_points=None,POM=None):
        self.name=name
        self.t_points=t_points
        self.T_points=T_points
        self.POM=POM
        self.selected_pos=None
        self.acc_ratio=None
        self.fwd_sim=None
        self.log_likelihood=-np.inf
        self.log_general_prior=-np.inf

# prior_settings
#Temperature prior 
mu_T= -44
sigma_pT= 1

#Dimensions prior
kmin_val=2
kmax_val=15

#POM prior    
POM_min=-50.0
POM_max=-40.0


# Prior functions for the models
import math
from scipy.stats import poisson

def prior_T(p):
    k = len(p.T_points)
    T=p.T_points
    T_prior=np.zeros(k)+mu_T

    sig_T = np.zeros((k,k))#uncertainty in Temperature
    np.fill_diagonal(sig_T, sigma_pT**2, wrap=True)
    sig_T = np.asmatrix(sig_T)

    prior=multivariate_normal.logpdf(T, T_prior, sig_T) 
    # print('prior_T end')
    return prior

def prior_time(p):
    k=len(p.t_points)

    prior_t=math.factorial(k)/(500**k)

    # below steps are implemented in model specific priors
    # product= 1
    # if k!=1:      
    #     for i in range(k-1):
    #         product = product * (p.t_points[i+1]-p.t_points[i])    
    # prior_t=prior_t*product

    return np.log(prior_t)
    
    

def prior_dimension(p):
    k=len(p.t_points)

    return -np.inf if not (kmin_val <= k <= kmax_val) else 0


def prior_POM(p):
    pom = p.POM
    # print('prior_POM end')
    return -np.inf if not (POM_min < pom < POM_max) else 0

#prior ratios
def prior_ratio(p,p_prev): 
    n=len(p.t_points)
    k=p.selected_pos
    
    tlist=list(p.t_points)
    tlist_old=list(p_prev.t_points)

    if k == 0:
        t_minus=t_old_minus = 0      
    else:
        t_minus=t_old_minus=tlist_old[k-1] 

    if p.name=='PERTURB_t':
        t_plus= t_old_plus=tlist_old[k+1]
    
        t_new= tlist[k] 
        t_old=tlist_old[k]
    
        pdf=((t_plus-t_new)*(t_new-t_minus))/((t_old_plus-t_old)*(t_old-t_old_minus))
    
    elif p.name=='BIRTH':
        t_new= tlist[k]
        t_plus= tlist[k+1]  
        pdf=((t_plus-t_new)*(t_new-t_minus))/(t_plus-t_minus)

    elif p.name=='DEATH':
        t_old=tlist_old[k]
        t_old_plus=tlist_old[k+1]    
        pdf=((t_old_plus-t_old)*(t_old-t_old_minus))/(t_old_plus-t_old_minus)
        pdf=1/pdf

    return pdf

def jacobian(p,p_prev): 
    n=len(p.t_points)
    k=p.selected_pos
    
    tlist=list(p.t_points)
    tlist_old=list(p_prev.t_points)

    sigma_bT=10**-3
    sigma_bt=1
    beta=-sigma_bT*sigma_bt
    
    if k == 0:
        t_minus=t_old_minus = 0      
    else:
        t_minus=t_old_minus=tlist_old[k-1] 
       
    if p.name=='PERTURB_t':
        J=1

    elif p.name=='BIRTH':
        t_plus= tlist[k+1] 
        _minus= -(t_plus-t_minus)
        J=_minus*beta

    elif p.name=="DEATH":
        t_old_plus=tlist_old[k+1]
        _minus=-(t_old_plus-t_old_minus)
        J=1/(_minus*beta)

    # print('J',J)
        
    return J

def lnlike(p,measured_depths, measured_temps, sigma_m):
    # print('lnlike ')
    max_depth, max_time, dz_approx, dt_approx, velocity,Tb,alpha,ws, wb,m = get_fwd_param()
    POM,Tlist, tlist= p.POM,p.T_points,p.t_points
    Tlist=list(Tlist)
    tlist=list(tlist)

    _ic=select_ICs(ics,POM)

    qnew= surface_temp_updated(np.array(Tlist), np.array(tlist), time_steps)
    # print("here after",len(qnew))
    T_new_model=get_solution_using_FD(max_depth, max_time, dz_approx, dt_approx, borehole3000_depths, borehole3000_densityProf,velocity,_ic,qnew,Tb,alpha,ws, wb,m)
    T_new=interpolate_cubicSpline(x_ref=T_new_model.depths,y_ref=T_new_model.T[:,-1],x_req=measured_depths)

    lp= multivariate_normal.logpdf(T_new, measured_temps, sigma_m) 
    # print('lnlike end')
    return lp

def find_pos(model,points):
    if model.name == 'BIRTH' or model.name == 'PERTURB_t' or model.name == 'DEATH':
        # print('find_pos BIRTH/ PERTURB_t/ DEATH end')
        return random.randint(1,points)  
        
    else:

        return random.randint(0,points-1)    

def model_PERTURB_POM(current_model):
    # print('model_PERTURB_POM ')
    new_model=MODELS(name='PERTURB_POM',T_points=current_model.T_points,t_points=current_model.t_points,POM=current_model.POM)
    new_model.selected_pos=None
    random_uniform_POM = np.round(np.random.uniform(-50.00, -40.00),3)
    new_model.POM= random_uniform_POM
    # print('model_PERTURB_POM end',new_model.POM)
    return new_model

def model_PERTURB_T(current_model):
    # print('model_PERTURB_T ')
    new_model=MODELS(name='PERTURB_T',T_points=current_model.T_points,t_points=current_model.t_points,POM=current_model.POM)
    new_model.selected_pos=None
    pos=find_pos(new_model,len(current_model.t_points))
    random_gaussian_T = np.random.normal(0, 1.0)
    sigma_mT=0.1 #scaling factor
    i=pos #** -- +1
    k=len(current_model.T_points)
    Tlist=list(new_model.T_points)
    Tlist[pos]= current_model.T_points[pos] + random_gaussian_T*sigma_mT*np.exp((k-i)/ k)
    new_model.T_points=tuple(Tlist)
    # print('model_PERTURB_T end')
    return new_model

def model_PERTURB_t(current_model):
    # position of time point value in the timeseries to be perturbed
    # print('model_PERTURB_t')
    new_model=MODELS(name='PERTURB_t',T_points=current_model.T_points,t_points=current_model.t_points,POM=current_model.POM)
    new_model.selected_pos=None
    
    pos=find_pos(new_model,len(current_model.t_points)-2)  #**
    new_model.selected_pos=pos
    random_gaussian_t = np.random.normal(0, 1)
    sigma_mt=0.05 #scaling factor
    i=pos #** -- +1
    k=len(current_model.t_points)
    # print('pos,k',pos,k)
    if pos != 0:
        tminus=current_model.t_points[pos-1]
    else:
        tminus=0
    tplus=current_model.t_points[pos+1]
    tnew=current_model.t_points[pos] + np.abs(tminus-tplus)*random_gaussian_t*sigma_mt*np.exp((k-i)/ k)
    
    # print('tnew',tnew)
    tlist=list(new_model.t_points)
    tlist[pos]= tnew#earest_possible_t(tnew)
    new_model.t_points=tuple(tlist)
    
    # print('t_points_new',new_model.t_points)
    # print('t_points_old',current_model.t_points)
    # print('model_PERTURB_t end')
    return new_model

def model_BIRTH(current_model):
    # print('model_BIRTH')
    new_model=MODELS(name='BIRTH',T_points=current_model.T_points,t_points=current_model.t_points,POM=current_model.POM)
    new_model.selected_pos=None
    
    pos=find_pos(new_model,len(current_model.t_points)-1)
    
    #**
    t1=new_model.t_points[pos-1]
    t2=new_model.t_points[pos]
    T1=new_model.T_points[pos-1]
    T2=new_model.T_points[pos]
    
    random_uniform_t = np.random.uniform(0, 1)
    sigma_bt=1 #scaling factor
    tnew= t1 + random_uniform_t*sigma_bt*(t2-t1)

    tlist=list(new_model.t_points)
    Tlist=list(new_model.T_points)      

    sigma_bT=10**-3
    random_gaussian_T = np.random.normal(0, 1.0)
    Tnew= ((T2-T1)/(t2-t1))*(tnew-t1)+  random_gaussian_T*sigma_bT + T1

    
    tlist.insert(pos,tnew)
    new_model.selected_pos=pos
    Tlist.insert(pos,Tnew)
        
    new_model.t_points=tuple(tlist)
    new_model.T_points=tuple(Tlist)

    # print('model_BIRTH end')
    return new_model

def model_DEATH(current_model):
    # print('model_DEATH')
    new_model=MODELS(name='DEATH',T_points=current_model.T_points,t_points=current_model.t_points,POM=current_model.POM)
    new_model.selected_pos=None
    pos=find_pos(new_model,len(current_model.t_points)-2) #**
    tlist=list(new_model.t_points)
    Tlist=list(new_model.T_points)
    tlist.pop(pos)
    Tlist.pop(pos)
    new_model.t_points=tuple(tlist)
    new_model.T_points=tuple(Tlist)
    new_model.selected_pos=pos
    # print('model_DEATH end')
    return new_model



def propose_jump(current_model):
    # print('propose_jump')
    # print("len(current_model.t_points)", len(current_model.t_points))
    # Propose jump to model1, model2, model3, model4 or model5

    flag="NO_k_CHANGE_proposal"
    
    if (len(current_model.t_points) <= kmin_val): #kmin
        # print('# len(current_model.t_points) <= 3')
        scenario1= random.randint(1, 2)
        if scenario1 == 1:
            # print('# pmodel_PERTURB_T')
            return model_PERTURB_T(current_model),flag           
        elif scenario1 == 2:
            # print('# model_BIRTH')
            flag="BIRTH_k_min_proposal"
            return model_BIRTH(current_model),flag
        else:
            # print('model not selected!!!')
            return current_model,flag

    elif (len(current_model.t_points) >= kmax_val): #kmax
        flag="scenario2"
        scenario2= random.randint(1, 3)
        if scenario2 == 1:
            # print('# model_PERTURB_T')
            return model_PERTURB_T(current_model),flag
        elif scenario2 == 2:
            # print('# model_PERTURB_t')
            return model_PERTURB_t(current_model),flag
        elif scenario2 == 3:
            # print('# model_DEATH')
            flag="DEATH_k_max_proposal"
            return model_DEATH(current_model),flag
        else:
            # print('model not selected!!!')
            return current_model,flag
        
    else:
        scenario3= random.randint(1, 5)
        if scenario3 == 1:
            # print('# model_PERTURB_T')
            return model_PERTURB_T(current_model),flag
        elif scenario3 == 2:
            # print('# model_PERTURB_t')
            return model_PERTURB_t(current_model),flag
        elif scenario3 == 3:
            # print('# model_BIRTH')
            flag="BIRTH_proposal"
            return model_BIRTH(current_model),flag
        elif scenario3 == 4:
            # print('# model_DEATH')
            flag="DEATH_proposal"
            return model_DEATH(current_model),flag
        elif scenario3 == 5:
            # print('# model_PERTURB_POM')
            return model_PERTURB_POM(current_model),flag
        else:
            # print('model not selected!!!')
            return current_model,flag

def log_prior_likelihood(model, measured_depths, measured_temps, sigma_m):
    # print('log_prior_likelihood')
    # priors:
    log_general_prior=prior_T(model) + prior_time(model) + prior_dimension(model) +prior_POM(model) 
    
    log_likelihood = lnlike(model, measured_depths, measured_temps, sigma_m)

    # print('log_prior_likelihood end')
    
    return log_likelihood, log_general_prior

def log_model_specific_prior(proposed_model,current_model):

    if proposed_model.name in ['PERTURB_t','BIRTH','DEATH']:
        Prior_ratio=prior_ratio(proposed_model,current_model)
        J=jacobian(proposed_model,current_model)
        if Prior_ratio < 0 or J < 0:
            return -np.inf, 0 
        return np.log(Prior_ratio), np.log(J)
    else:
        return 0, 0

def log_calculate_proposal_ratio(new_model,current_model,proposal_ratio_flag):

    proposal_ratio=1
    dk=bk=1/5 
    L=500
    k=len(new_model.t_points)
    kmin=kmin_val
    kmax=kmax_val
    if proposal_ratio_flag == "BIRTH_proposal" :
        proposal_ratio=L/(k+1)
    elif proposal_ratio_flag == "DEATH_proposal":
        proposal_ratio=k/L
    elif proposal_ratio_flag == "BIRTH_k_min_proposal":
        proposal_ratio=(2*L)/(5*(kmin+1))
    elif proposal_ratio_flag == "DEATH_k_max_proposal":
        proposal_ratio=(3*(kmax-1))/(5*L)
    else:
        # print("NO_k_CHANGE_proposal")
        proposal_ratio=1
      
    return np.log(proposal_ratio)   
        
    

# RJ-MCMC: The main function for sampling
def rjmcmc_step(current_model,measured_depths, measured_temps, sigma_m):
    # print('Propose a new model and parameters')
    
    new_model,proposal_ratio_flag = propose_jump(current_model)
    if new_model == current_model:
        log_pl_new =-np.inf
        # print('new_model==current_model',log_pl_new)
        return current_model, False
    else:
        log_likelihood_new, log_general_prior_new =log_prior_likelihood(new_model,measured_depths, measured_temps, sigma_m)
        log_pl_new= log_likelihood_new+ log_general_prior_new 
        # print('log_pl_new',log_pl_new)
            
    log_pl_current= current_model.log_likelihood + current_model.log_general_prior
    # print('log_pl_current',log_pl_current)
        
    log_prior_ratio, log_jacobian=log_model_specific_prior(new_model,current_model)
    
    log_proposal_ratio=log_calculate_proposal_ratio(new_model,current_model,proposal_ratio_flag)
    
    log_acceptance_ratio = log_pl_new - log_pl_current +log_prior_ratio + log_jacobian + log_proposal_ratio
    
    # print('Propose a new model and parameters end',log_acceptance_ratio)

    if np.log(np.random.rand()) < log_acceptance_ratio:
        new_model.acc_ratio=log_acceptance_ratio
        new_model.log_general_prior=log_general_prior_new
        new_model.log_likelihood=log_likelihood_new
        return new_model, True
    else:
        return current_model, False

def main():
    
    # Initialize the MCMC chain
    init_T=tuple([-44,-44,-44,-44,-44,-44,-44])
    init_t=tuple([time_steps[0] , time_steps[-6400],  time_steps[-4800] , time_steps[-3200] ,time_steps[-1600] ,time_steps[-800] ,time_steps[-1]])
    init_model = MODELS('PERTURB_T',init_T, init_t, -44)
    
    # measurement error
    Nm=len(measured_depths)
    sig_m = np.zeros((Nm,Nm))#uncertainty in sensors
    np.fill_diagonal(sig_m, measured_error**2, wrap=True) 
    sigma_m = np.asmatrix(sig_m)
    
    # Perform RJ-MCMC sampling
    #num_steps = 1000
    num_samples = 500000
    samples = []
    run_time=[]
    samples.append(init_model)
    step=0
    start_time = time.time()#perf_counter()
    while  len(samples)< num_samples:
        model, accepted= rjmcmc_step(samples[-1],measured_depths, measured_temps, sigma_m)
        if accepted:
            # print (model, accepted)
            samples.append(model)          
        if len(samples)%50000 == 0:
            end_time = time.time()#perf_counter()
            total_seconds = end_time - start_time
            time_in_hours = total_seconds / 3600
            run_time.append(time_in_hours)
            with open('/home/kshaju/IceBT_Bayesian_inversion/RJMCMC_reconstructions/output/sampler_output/A1_0m_signal_3_30mk_10m3.pkl', 'wb') as f:
                dill.dump(samples,f) 
            # print("so far tested",step)
            # print("so far accepted",len(samples))
        step = step+1
    with open('/home/kshaju/IceBT_Bayesian_inversion/RJMCMC_reconstructions/output/sampler_output/A1_0m_signal_3_30mk_10m3.pkl', 'wb') as f:
        dill.dump(samples,f) 
    np.save('/home/kshaju/IceBT_Bayesian_inversion/RJMCMC_reconstructions/output/sampler_output/run_time_A1_0m_signal_3_30mk_10m3',run_time)
main()
