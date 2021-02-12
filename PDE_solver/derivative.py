import numpy as np
import numba


@numba.njit
def derivative(u, h, axis, scheme_order=1,boundary_order=1):
    dh=(h[1] - h[0])
    if scheme_order==1:
        if axis == 1:
            u_shift_down = np.roll(u, 1)
            u_shift_up = np.roll(u, -1)
        if axis == 0:
            u_transposed = np.transpose(u)
            
            u_shift_down = np.roll(u_transposed, 1)
            u_shift_down = np.transpose(u_shift_down)
    
            u_shift_up = np.roll(u_transposed, -1)
            u_shift_up = np.transpose(u_shift_up)
    
        du = (u_shift_up - u_shift_down) / (2 * dh)
    
    if scheme_order==2:
        if axis == 1:
            u_shift_down_1 = np.roll(u, 1)
            u_shift_down_2 = np.roll(u, 2)
            u_shift_up_1 = np.roll(u, -1)
            u_shift_up_2 = np.roll(u, -2)
        
        
        if axis == 0:
            u_transposed = np.transpose(u)
    
            u_shift_down_1 = np.roll(u_transposed, 1)
            u_shift_down_1 = np.transpose(u_shift_down_1)
    
            u_shift_down_2 = np.roll(u_transposed, 2)
            u_shift_down_2 = np.transpose(u_shift_down_2)
    
    
            u_shift_up_1 = np.roll(u_transposed, -1)
            u_shift_up_1 = np.transpose(u_shift_up_1)
    
            u_shift_up_2 = np.roll(u_transposed, -2)
            u_shift_up_2 = np.transpose(u_shift_up_2)


        a_down = -1.5 / dh
        b_down = 2. / dh
        c_down = -0.5 / dh
    
        a_up = 0.5 / dh
        b_up = -2. / dh
        c_up = 1.5 / dh
    
        du = -0.5*(b_down*u_shift_down_1+c_down*u_shift_down_2+a_up*u_shift_up_2+b_up*u_shift_up_1)
       
    if boundary_order==1:
        
        if axis == 1:
          du[:, 0] = (u[:, 1] - u[:, 0]) / dh
          du[:, -1] = (u[:, -1] - u[:, -2]) /  dh
          if scheme_order==2:
              du[:, 1] = (u[:, 2] - u[:, 1]) / dh
              du[:, -2] = (u[:, -2] - u[:, -3]) /  dh             
            
        if axis == 0:
           du[0, :] = (u[1, :] - u[0, :]) /  dh
           du[-1, :] = (u[-1, :] - u[-2, :]) /  dh
           if scheme_order==2:
               du[1, :] = (u[2, :] - u[1, :]) /  dh
               du[-2, :] = (u[-2, :] - u[-3, :]) /  dh
    

    if boundary_order==2:
        a_down = -1.5 / dh
        b_down = 2. / dh
        c_down = -0.5 / dh
    
        a_up = 0.5 / dh
        b_up = -2. / dh
        c_up = 1.5 / dh
        if axis == 0:
            du[0, :] = a_down*u[0, :]+b_down*u[1, :]+c_down*u[2, :]
            du[-1, :] = a_up*u[-3, :] + b_up*u[-2, :]+c_up*u[-1, :]
            if scheme_order==2:
                du[1, :] = a_down*u[1, :]+b_down*u[2, :]+c_down*u[3, :]
                du[-2, :] = a_up*u[-4, :] + b_up*u[-3, :]+c_up*u[-2, :]
           # du[0, :] = (u[1, :] - u[0, :]) /  (h[1] - h[0])
        if axis == 1:
            du[:, 0] = a_down*u[:, 0]+b_down*u[:, 1]+c_down*u[:, 2]
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            du[:, -1] = a_up*u[:, -3] + b_up*u[:, -2]+c_up*u[:, -1]
            if scheme_order==2:
                du[:, 1] = a_down*u[:, 1]+b_down*u[:, 2]+c_down*u[:, 3]
                du[:, -2] = a_up*u[:, -4] + b_up*u[:, -3]+c_up*u[:, -2]

    return du

            





def second_derivative(u, x, t, axis):
    if axis == 1:
        u_shift_down = np.roll(u, 1)
        u_shift_up = np.roll(u, -1)
        du = (u_shift_up - 2 * u + u_shift_down) / (x[0, 1] - x[0, 0]) ** 2
        du[:, 0] = (u[:, 2] - 2 * u[:, 1] + u[:, 0]) / (x[0, 1] - x[0, 0]) ** 2
        du[:, -1] = (u[:, -3] - 2 * u[:, -2] + u[:, -1]) / (x[0, 1] - x[0, 0]) ** 2

    if axis == 0:
        u_shift_down = np.roll(u, 1, axis=0)
        u_shift_up = np.roll(u, -1, axis=0)
        du = (u_shift_up - 2 * u + u_shift_down) / (t[1, 0] - t[0, 0]) ** 2
        du[0, :] = (u[2, :] - 2 * u[1, :] + u[0, :]) / (t[1, 0] - t[0, 0]) ** 2
        du[-1, :] = (u[-3, :] - 2 * u[-2, :] + u[-1, :]) / (t[1, 0] - t[0, 0]) ** 2

    return du


if __name__ == "__main__":
    x1 = np.linspace(0, 1, 20)
    t1 = np.linspace(0, 1, 20)

    x, t = np.meshgrid(x1, t1)
    u = x + t ** 2
    
    du = derivative(u, x1, axis=1, scheme_order=1, boundary_order=1)
    # du = derivative(du, t1, axis=0)
    print(du)
