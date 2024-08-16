from loguru import logger
import numpy as np

def init_char_values(G, m_earth, m_moon):
    lstar = 384400  # Distance from Earth to Moon (km)
    mstar = m_earth + m_moon  # Total mass of the system (kg)
    mu_star = G * mstar  # Characteristic gravitational parameter
    tstar = np.sqrt(lstar**3 / mu_star)  # Characteristic time (s)
    vstar = lstar / tstar  # Characteristic velocity (m/s)
    astar = vstar / tstar  # Characteristic acceleration (m/s^2)

    logger.success("Characteristic values initialized successfully.")
    return lstar, mstar, mu_star, tstar, vstar, astar

def init_constants():
    G = 6.67430e-20  # km^3/kg/s^2
    m_earth = 5.97219e24  # kg
    m_moon = 7.34767309e22  # kg
    mu = m_moon / (m_earth + m_moon)
    r_earth = 6378.1  # km
    r_orbit = 300.0  # km
    mu_earth = 398600.44  # km^3/s^2
    n = np.sqrt(mu_earth / (r_earth + r_orbit)**3)
    g0 = 9.80665e-3  # km/s^2

    logger.success("Constants initialized successfully.")
    return G, m_earth, m_moon, mu, r_earth, r_orbit, mu_earth, n, g0

def init_values_opt():
    # Assuming G, m_earth, and m_moon are available globally or passed as parameters
    G, m_earth, m_moon = init_constants()[:3]
    lstar, mstar, mu_star, tstar, vstar, astar = init_char_values(G, m_earth, m_moon)
    
    s0 = np.array([1.05, 0, 0, 0, -0.9, 0, 1])
    sd = np.array([0.379, 0.866, 0, 0, 0, 0, 0])
    thrust_max_nd = 0.0006
    m_dot_max_nd = 0.001

    # Calculate minimum mass (non-dimensional)
    m_min = s0[6] - m_dot_max_nd * 20 * 30 * 24 * 3600 / 2  # assuming half the time with max thrust

    s0[6] = m_min  # Set initial mass to minimum mass for optimization

    # Update sd with minimum mass
    sd = np.array([0.379, 0.866, 0, -0.069, 0.079, 0, m_min])

    # Initial guess for time of flight (non-dimensional)
    time_of_flight = 20 * 30 * 24 * 3600  # in seconds
    time_of_flight /= tstar  # make it non-dimensional

    # Number of steps
    N_step = 10000

    # Define tspan for integration
    tspan = np.linspace(0, time_of_flight, N_step)

    logger.success("Optimization values initialized successfully.")
    return s0, sd, thrust_max_nd, m_dot_max_nd, N_step, time_of_flight, tspan, m_min
