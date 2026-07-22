#calculating required resolution by interpolation of rerence resolution

from scipy.interpolate import CubicSpline

def interpolate_cubicSpline(x_ref,y_ref,x_req):
    
    try:
        cubicSpline_interpolant = CubicSpline(x_ref, y_ref)
        y_req= cubicSpline_interpolant(x_req)        
    except ValueError: 
        print("Oops!  That was no valid number.  Try again...")
    
      
    return y_req


