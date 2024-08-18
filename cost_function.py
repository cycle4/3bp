import numpy as np
from scipy.integrate import solve_ivp
from loguru import logger

import sys
sys.path.append('./')
from cr3bp import cr3bp_alpha_mod
from init_values import init_constants, init_values_opt

def cost_function(X0):
    # Define constants and initial values
    _, _, _, mu, _, _, _, n, _ = init_constants()
    s0, sd, thrust_max_nd, m_dot_max_nd, N_step, time_of_flight, tspan, _ = init_values_opt()

    # Define the ODE function to be passed to solve_ivp
    def cr3bp_alpha(t, x):
        return cr3bp_alpha_mod(t, x, mu, n, thrust_max_nd, m_dot_max_nd, X0)

    # Extract the start and end times from tspan
    t0, tf = tspan[0], tspan[-1]

    # # ODE solver options equivalent to MATLAB's ode45 with custom tolerances
    # sol = solve_ivp(cr3bp_alpha, (t0, tf), s0, method='RK45', t_eval=tspan, rtol=1e-12, atol=1e-12)
    sol = solve_ivp(cr3bp_alpha, (t0, tf), s0, method='RK45', t_eval=tspan, rtol=1e-12, atol=1e-12)

    # Calculate the distance from the desired final state
    dist = np.linalg.norm(sol.y[:3, :] - sd[:3, np.newaxis], axis=0)
    
    # Find the index of minimum distance
    distmin_idx = np.argmin(dist)
    logger.success("Distance : {} at time : {}", dist[distmin_idx], sol.t[distmin_idx])

    # Time of flight to the closest point
    tof = sol.t[distmin_idx]

    # Maximum position error term
    max_position_error = np.linalg.norm(sol.y[3:6, distmin_idx]) * time_of_flight / N_step
    
    # Maximum velocity error term
    max_velocity_error = np.linalg.norm(sd[3:6]) * 0.01

    # Weight constants as per the MATLAB version
    W1 = 80
    W2 = 2
    W3 = 1e5
    W4 = 1e3

    # Final mass at the last point
    m_final_nd = sol.y[6, -1]

    # Calculate the cost function as per MATLAB's formulation
    Jfun = (-0.5 * W1 * (1 - m_final_nd) + 
            W2 * tof + 
            W3 * (abs(sd[3] - sol.y[3, distmin_idx]) - max_velocity_error) + 
            W3 * (abs(sd[4] - sol.y[4, distmin_idx]) - max_velocity_error) + 
            W4 * (abs(sd[0] - sol.y[0, distmin_idx]) - max_position_error) + 
            W4 * (abs(sd[1] - sol.y[1, distmin_idx]) - max_position_error))

    logger.success("Cost function : {}", Jfun)

    return Jfun
