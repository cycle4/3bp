from loguru import logger
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import sys 

# Add custom paths
sys.path.append('./')

# Import custom modules
from cr3bp import cr3bp_alpha_mod
from init_values import init_constants, init_values_opt, init_char_values
from cost_function import cost_function
from nonlcon import nonlcon

def main_optimization():
    logger.debug("Starting main optimization.")
    
    # Initialize constants
    G, m_earth, m_moon, mu, r_earth, r_orbit, mu_earth, n, g0 = init_constants()

    # Initialize characteristic values
    lstar, mstar, mu_star, tstar, vstar, astar = init_char_values(G, m_earth, m_moon)

    # Initialize optimization values
    s0, sd, thrust_max_nd, m_dot_max_nd, N_step, time_of_flight, tspan, m_min = init_values_opt()

    # Options for solve_ivp (integration settings)
    options = {'rtol': 1e-12, 'atol': 1e-12}

    # Define the control vector (initial guess for the control)
    uref = 1 - 58568e-10  # u0

    # Perform the integration using solve_ivp (equivalent to ode45 in MATLAB)
    sol = solve_ivp(
        lambda t, x: cr3bp_alpha_mod(t, x, mu, n, thrust_max_nd, m_dot_max_nd, uref),
        (tspan[0], tspan[-1]), s0, method='RK45', t_eval=tspan, **options
    )
    
    if not sol.success:
        logger.error("Integration failed: {}", sol.message)
        raise RuntimeError("Integration failed.")

    logger.success("First integrator performed successfully.")

    # Calculate distances to the target
    dist = np.linalg.norm(sol.y[:3, :].T - sd[:3], axis=1)
    mindist_t = np.argmin(dist)

    # Initialize arrays for thrusts and control vectors
    Thrusts = np.zeros((len(sol.t), 3))
    X0 = np.zeros((len(sol.t), 2))

    # Calculate thrust direction and control vectors
    for i in range(mindist_t + 1):
        Thrusts[i, :] = uref * sol.y[3:6, i] / np.linalg.norm(sol.y[3:6, i])
        X0[i, :] = [uref, np.arctan2(Thrusts[i, 0], Thrusts[i, 1])]

    logger.debug("Vector X0 calculated successfully.")

    # Extract time associated with the closest distance
    tf = tspan[mindist_t]

    # Create a new row with tf and a placeholder for the control vector
    new_row = np.array([[tf, np.nan]])  # Placeholder can be np.nan or 0

    # Stack the new row with X0
    X0 = np.vstack([X0, new_row])

    # Define bounds for the optimizer (equivalent to lb and ub in MATLAB)
    bounds = [(0.01, uref), (-np.pi, np.pi), (0, tf)]  # Bounds for each control variable

    # Define constraints (equivalent to nonlcon in MATLAB)
    constraints = [
        {'type': 'ineq', 'fun': lambda x: nonlcon(x)[0]},  # Inequality constraints
        {'type': 'eq', 'fun': lambda x: nonlcon(x)[1]}     # Equality constraints
    ]

    # Flatten X0 to match the optimizer's input format
    x0_flattened = X0.flatten()

    # Create bounds for each element in x0_flattened
    bounds = []
    for i in range(len(x0_flattened)):
        if i % 3 == 0:  # uref bounds
            bounds.append((0.01, uref))
        elif i % 3 == 1:  # angle bounds
            bounds.append((-np.pi, np.pi))
        else:  # time bounds
            bounds.append((0, tf))
    
    # Sanity check to ensure bounds and flattened X0 match
    if len(bounds) != len(x0_flattened):
        raise ValueError("Mismatch between bounds and x0_flattened lengths")

    # # Perform optimization using minimize
    # result = minimize(
    #     cost_function, x0_flattened, method='SLSQP', 
    #     bounds=bounds, constraints=constraints, 
    #     options={'maxiter': 1000, 'disp': True}
    # )

    # Perform optimization using differential_evolution
    result = differential_evolution(
        cost_function,           # Your cost function
        bounds,                  # Bounds on variables
        strategy='best1bin',     # Strategy for mutation and crossover
        maxiter=1000,            # Maximum number of generations
        popsize=10,              # Population size
        tol=1e-6,                # Tolerance for convergence
        # mutation=(0.5, 1),       # Mutation constant or tuple for lower and upper bounds
        recombination=0.7,       # Recombination constant
        disp=True,               # Display progress during optimization
        polish=True              # Use `minimize` for polishing the best solution
    )
    X = result.x
    fval = result.fun

    logger.success("Optimization completed successfully.")

    return X0, Thrusts, X, fval

def plot_results(X0, Thrusts):
    import matplotlib.pyplot as plt

    # Create figure and axes
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot thrust direction in the x-y plane
    ax[0].plot(Thrusts[:, 0], Thrusts[:, 1], label='Thrust direction')
    ax[0].set_xlabel('Thrust in x-direction')
    ax[0].set_ylabel('Thrust in y-direction')
    ax[0].set_title('Thrust direction in x-y plane')
    ax[0].legend()

    # Plot control vector (thrust magnitude and angle)
    ax[1].plot(X0[:, 0], X0[:, 1], label='Control vector')
    ax[1].set_xlabel('Thrust magnitude')
    ax[1].set_ylabel('Thrust angle')
    ax[1].set_title('Control vector')
    ax[1].legend()

    plt.show()

if __name__ == "__main__":
    try:
        # Run the main optimization process
        X0, Thrusts = main_optimization()
        logger.info("Optimization completed successfully.")
        
        # Display results
        print("X0:", X0)
        print("Thrusts:", Thrusts)

        # Plot the results
        plot_results(X0, Thrusts)

    except Exception as e:
        logger.exception("An error occurred during the optimization process.")
