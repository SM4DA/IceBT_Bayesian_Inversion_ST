import numpy as np
from .linear_interpolation import interpolate_linear
from .run_PDE_solver import PDE

def get_generate_IC_fixed_grid(max_depth, max_time, dz_approx, dt_approx, borehole3000_depths, borehole3000_densityProf,velocity,Ts,Tb,alpha,beta,ws, wb,m):
    time_grid_no=int(max_time/dt_approx)
    depth_grid_no=int(max_depth/dz_approx) 
    k_depths= np.linspace(0,max_depth,depth_grid_no)
    densityProfile=interpolate_linear(x_ref=borehole3000_depths,y_ref=borehole3000_densityProf,x_req=k_depths)
    
    initial_conditions=np.linspace(Ts,Tb,depth_grid_no,endpoint=True)
    top_boundary=np.zeros(2)+Ts 
    bottom_boundary=np.zeros(2)+Tb
    
    steady_state=[]
    case_sample_data=PDE(max_depth=max_depth,max_time=max_time,depth_grid_no=depth_grid_no,time_grid_no=time_grid_no,densityProfile=densityProfile, velocity=[],initial_conditions=initial_conditions,top_boundary=top_boundary,bottom_boundary=bottom_boundary )
    steady_state=case_sample_data.getIC_diffusion_advection_FD(alpha=alpha,beta=beta,ws=ws,wb=wb,m=m)
    
    print(case_sample_data.top_boundary)
    print(case_sample_data.bottom_boundary)
    
    return steady_state
    