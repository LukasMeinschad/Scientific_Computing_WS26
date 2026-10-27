""" 
Solution to the Heat Transfer Equation


Numerical Method is:
T_i^{n+1} = T_i^n + Delta t / Delta x^2 * (T_{i+1}^n - 2 * T_i^n + T_{i-1}^n)
Where 
- T_i^n is the temperature at position i and time n
- Delta t is the time step
- Delta x is the spatial step

Initial Values are given by:
+ T(0,x) = 350 - 100x + 200sin(pi x)

And Dirichlet Boundary Conditions are
+ T(t,0) = 350
+ T(t,1) = 250

"""

import numpy as np
import matplotlib.pyplot as plt

T_t_0 = 350
T_t_1 = 250

def T_0_x(x):
    """ 
    Initial Value Equation
    """
    x = np.asarray(x, dtype=float)
    values = 350.0 - 100.0 * x + 200.0 * np.sin(np.pi * x)
    return values

def solve(n_points=15, delta_t=0.001, n_steps=50):
    """
    Solve the one-dimensional heat equation using
    the explicit euler method

    n_points: number of spatial points
    delta_t: time step
    n_steps: number of time steps to simulate
    """
    # Initialize spatial grid
    x = np.linspace(0.0, 1.0, n_points)
    # Calculate spatial step size
    delta_x = x[1] - x[0]
    # Initialize temperature values at initial time
    values = np.asarray(T_0_x(x), dtype=float).reshape(-1)
    values[0] = T_t_0
    values[-1] = T_t_1

    time_evolution = [values.copy()]

    # Precompute the coeficient
    coefficient = delta_t / delta_x**2
    for _ in range(n_steps):
        previous = values.copy()
        # Slice 1:-1 contains
        # all values exepct the boundary values
        values[1:-1] = previous[1:-1] + coefficient * (
            previous[2:] - 2.0 * previous[1:-1] + previous[:-2]
        )
        values[0] = T_t_0
        values[-1] = T_t_1
        time_evolution.append(values.copy())

    print("Final Length of values:", len(values))  
    print("Final Temperature Distribution:", values)
    return x, values, time_evolution

delta_t = 0.001

x, temperature, time_evolution = solve(n_points=15, delta_t=delta_t, n_steps=200)


# plot the time evolution of the temperature distribution
plt.figure(figsize=(10, 6))
cmap = plt.get_cmap('magma')
for i, temp in enumerate(time_evolution):
    if i % 10 == 0:  # Plot every 10th time step
        # compute time
        time = i * delta_t
        plt.plot(x, temp, label=f'Time step {time}', color=cmap(i / len(time_evolution)))
plt.title('Time Evolution of Temperature Distribution')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.grid()
plt.legend()
plt.savefig("heat_conduction_time_ev.png",dpi=300)
plt.close()

# For a given x spacing analyse the cfl condition
n_points = 15
delta_ts = [0.0001, 0.0005, 0.0007, 0.001, 0.002, 0.0025, 0.0026, 0.0027, 0.0028]

plt.figure(figsize=(10,6))
for delta_t in delta_ts:
    x, temperature, time_evolution = solve(n_points=n_points, delta_t=delta_t, n_steps=200)
    plt.plot(x, temperature, label=f'ratio = {delta_t / ((x[1] - x[0])**2):.4f}, delta_t = {delta_t}')
plt.title('Final Temperature Distribution for Different Time Steps')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.grid()
plt.legend()
plt.savefig("heat_conduction_final_temp.png",dpi=300)
plt.close()



