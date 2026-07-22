#calculating required resolution by interpolation of rerence resolution

from scipy.interpolate import interp1d

def interpolate_linear(x_ref,y_ref,x_req):
    #print('in interpolate_timeseries()')
    try:
        y_f=interp1d(x_ref,y_ref,'linear')
        y_req=[]
        for d in x_req:
            y_req.append(y_f(d))         
    except ValueError: 
        print("Oops!  That was no valid number.  Try again...")
    
      
    return y_req




def interpolate_linear_fill_value(x_ref,y_ref,x_req):
    #print('in interpolate_timeseries()')
    try:
        y_f=interp1d(x_ref,y_ref,'linear',fill_value="extrapolate")
        y_req=[]
        for d in x_req:
            y_req.append(y_f(d))         
    except ValueError: 
        print("Oops!  That was no valid number.  Try again...")
    
      
    return y_req
