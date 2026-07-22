
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:41:53 2023

@author: ksshaj001
"""
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import time

class ThermalProperties:
    def __init__(self):
        self.heatCapacity
        self.conductivity
        self.diffusivity
        self.density
        self.prev_temp 
                
    def heat_capacity(self):
        _ci=152.5+7.122*(self.prev_temp+273.15) 
        self.heatCapacity=_ci*(self.density/917)+1005*(1-(self.density/917))
        return self.heatCapacity

    def thermal_conductivity(self,alpha, beta):
        _ki=2.22*(1-(0.0067*(self.prev_temp)))
        self.conductivity=_ki*(self.density/917)**(alpha-(beta*(self.density/917)))
        return self.conductivity  
            
    def thermal_diffusivity(self):
        self.diffusivity=(self.conductivity/(self.density*self.heatCapacity))*24*3600*365
        return self.diffusivity



class PDE(ThermalProperties):
    def __init__(self,max_depth,max_time,time_grid_no,depth_grid_no,densityProfile,velocity,initial_conditions,top_boundary,bottom_boundary):
        self.max_depth=max_depth
        self.max_time=max_time
        self.densityProfile=np.array(densityProfile)
        self.velocity=np.array(velocity)
        self.time_grid_no=time_grid_no
        self.depth_grid_no=depth_grid_no
        self.top_boundary=top_boundary##
        self.bottom_boundary=bottom_boundary
        self.depths=[]
        self.times=[]
        self.initial_conditions=initial_conditions##
        self.status='nil'
       
    #newly updated oct_5_23
    def set_vertical_velocity(self, ws,wb,m,z):
        if len(self.velocity)!=0:
            # print("using given velocity profile")
            pass
        else:
            # print("using calculated velocity profile")
            self.velocity=(917/self.densityProfile)*(ws-(ws-wb)*(((m+2)/(m+1))*(z/self.max_depth))*(1-((z/self.max_depth)**(m+1))/(m+2)))*10**-3
        # print("executed set_vertical_velocity")
    
    def set_initial_conditions(self,depths,L,category):
        if category=="d":
            self.initial_conditions=0.0  *depths -0
        elif category=="da":
            self.initial_conditions=0.0  *depths -0
        else:
            # print("\n wrong category-- 6*np.sin(np.pi*depths/L)")
            pass
            
    #newly updated oct_5_23
    def set_boundary_conditions(self):
        self.boundary_conditions=[self.top_boundary, self.bottom_boundary]
            
    #newly updated oct_6_23   for 2 column calculations  
    def assign_initial_boundary_conditions(self, Temp):
        if len(Temp[0 , : ]) !=2 :
            # print('len(Temp[0 , : ]) !=2 :')
            Temp[0 , : ]=self.boundary_conditions[0]
        if len(Temp[-1 , : ]) !=2 :
            # print('len(Temp[-1 , : ]) !=2 :')
            Temp[-1 , : ]=self.boundary_conditions[1]
        Temp[ : , 0]=self.initial_conditions
        
        # print("executed assign_initial_boundary_conditions")
        
        return Temp  
    
           
    
    def getIC_diffusion_advection_FD(self,alpha,beta,ws,wb,m): 
        
        
        vector_start_time = time.time()
        print("In getIC_diffusion_advection_FD()")
        t = np.linspace(0,self.max_time,self.time_grid_no) 
        self.no_of_dt=len(t)
        self.times=t
        if len(self.top_boundary)==0:
            print("initial top boundary is 0")
                      
        z= np.linspace(0,self.max_depth,self.depth_grid_no) 
        self.no_of_dz=len(z)
        self.depths=z
        L=z[-1]
        self.T=np.zeros((self.no_of_dz,2))
        
        
        if len(self.initial_conditions)==0:
            self.set_initial_conditions(z,L,"da")
        self.set_boundary_conditions()

        self.T=self.assign_initial_boundary_conditions(self.T)
        
        self.set_vertical_velocity(ws,wb,m,z)
         
        dt=self.max_time/self.time_grid_no
        dz=self.max_depth/self.depth_grid_no
        l=dt/(dz**2)
        la=dt/dz
        self.density=self.densityProfile
            
        for j in tqdm(range(1,self.no_of_dt)):
            self.T[0,1]  =  self.top_boundary[0]  
            self.T[-1,1]  =  self.bottom_boundary[0]
            
            self.prev_temp = self.T[:,0]
            self.heat_capacity()
            self.thermal_conductivity(alpha, beta)
            self.thermal_diffusivity()
            
            self.T[1:self.no_of_dz-1,1] = self.T[1:self.no_of_dz-1,0] + l*self.diffusivity[1:self.no_of_dz-1]*(self.T[2:self.no_of_dz,0] - 2*self.T[1:self.no_of_dz-1,0] + self.T[0:self.no_of_dz-2,0])-la*self.velocity[1:self.no_of_dz-1]*(self.T[2:self.no_of_dz,0]-self.T[1:self.no_of_dz-1,0])
            
            last_state=list(self.T[:,1])
            second_last_state=list(self.T[:,0])
            if(last_state==second_last_state):
                print('steady state reached')
                self.status='reached'
                return last_state
            
            else:
                self.status='not reached'
                self.T[:,0]= self.T[:,1]
            
         
        # print("\nend of getIC_diffusion_advection_FD(), time taken:",time.time() - vector_start_time)  
        if self.status=='not reached':
            # print('steady state not reached, returning state at yr', self.max_time)
            return list(self.T[:,1])
        
    def diffusion_advection_FD_full_matrix(self,alpha,beta,ws,wb,m): 
        
        vector_start_time = time.time()
        # print("In diffusion_advection_FD_full_matrix()")
        t = np.linspace(0,self.max_time,self.time_grid_no) 
        self.no_of_dt=len(t)
        self.times=t
        if len(self.top_boundary)==0:
            # print("initial top boundary is 0")
            pass
                      
        z= np.linspace(0,self.max_depth,self.depth_grid_no) 
        self.no_of_dz=len(z)
        self.depths=z
        L=z[-1]
        self.T=np.zeros((self.no_of_dz,self.no_of_dt))
        
        
        if len(self.initial_conditions)==0:
            self.set_initial_conditions(z,L,"da")
        self.set_boundary_conditions()

        self.T=self.assign_initial_boundary_conditions(self.T)
        
        self.set_vertical_velocity(ws,wb,m,z)
         
        dt=self.max_time/self.time_grid_no
        dz=self.max_depth/self.depth_grid_no
        l=dt/(dz**2)
        la=dt/dz
        self.density=self.densityProfile
            
        for j in range(1,self.no_of_dt):
            self.prev_temp = self.T[:,j-1]
            self.heat_capacity()
            self.thermal_conductivity(alpha, beta)
            self.thermal_diffusivity()
            
            self.T[1:self.no_of_dz-1,j] = self.T[1:self.no_of_dz-1,j-1] + l*self.diffusivity[1:self.no_of_dz-1]*(self.T[2:self.no_of_dz,j-1] - 2*self.T[1:self.no_of_dz-1,j-1] + self.T[0:self.no_of_dz-2,j-1])-la*self.velocity[1:self.no_of_dz-1]*(self.T[2:self.no_of_dz,j-1]-self.T[1:self.no_of_dz-1,j-1])
            
           
        last_time=int(self.max_time/dt)-1
        #self.plot_2D(dz,dt,last_time)
        # print("\nend of diffusion_advection_FD(), time taken:",time.time() - vector_start_time)

   
