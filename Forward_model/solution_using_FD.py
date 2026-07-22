import numpy as np
from .linear_interpolation import interpolate_linear
from .run_PDE_solver import PDE

def get_solution_using_FD(max_depth, max_time, dz_approx, dt_approx, borehole3000_depths,borehole3000_densityProf,velocity,ic,Ts,Tb,alpha,ws, wb,m):
    time_grid_no=int(max_time/dt_approx)
    depth_grid_no=int(max_depth/dz_approx) 
    k_depths= np.linspace(0,max_depth,depth_grid_no)
    densityProfile=interpolate_linear(x_ref=borehole3000_depths,y_ref=borehole3000_densityProf,x_req=k_depths)
    
    initial_conditions=ic
  
    
    if isinstance(Ts, float):#mistake it was Tb-- corrected on 15th oct 2024
        top_boundary=np.zeros(time_grid_no)+Ts
    else:
        top_boundary=Ts 
    
    if isinstance(Tb, float):
        bottom_boundary=np.zeros(time_grid_no)+Tb
    else:
        bottom_boundary=Tb        
    
    steady_state=[]
    case_sample_data=PDE(max_depth=max_depth,max_time=max_time,depth_grid_no=depth_grid_no,time_grid_no=time_grid_no,densityProfile=densityProfile, velocity=velocity,initial_conditions=initial_conditions,top_boundary=top_boundary,bottom_boundary=bottom_boundary )
    case_sample_data.diffusion_advection_FD_full_matrix(alpha=alpha,beta=0,ws=ws,wb=wb,m=m)
    
    return case_sample_data
    