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

def solve(n_points=15, delta_t=0.001, n_steps=50, initial_condition_func=T_0_x, T_t_0=350, T_t_1=250):
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
    values = np.asarray(initial_condition_func(x), dtype=float).reshape(-1)
    values[0] = T_t_0
    values[-1] = T_t_1

    # Steady state for Dirichlet-Dirichlet is the linear profile between the boundaries
    # As stated in the VO
    steady_state = T_t_0 + (T_t_1 - T_t_0) * x

    time_evolution = [values.copy()]
    # Maximum deviation from the steady state at every time step
    amplitude = [np.max(np.abs(values - steady_state))]

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
        amplitude.append(np.max(np.abs(values - steady_state)))

    print("Final Length of values:", len(values))
    print("Final Temperature Distribution:", values)
    return x, values, time_evolution, np.array(amplitude)

delta_t = 0.001

x, temperature, time_evolution, amplitude = solve(n_points=15, delta_t=delta_t, n_steps=200)

# We now try out different initial values and how this influences the decay of the temperature distribution. We will use the same boundary conditions as before, but we will change the initial values to be a constant value of 350.
T_t_0s = [350, 300, 250, 200, 150]
T_t_1 = 100
n_steps = 500
decay_results = []
plt.figure(figsize=(10, 6))
plt.title("Time Evolution of Temperature Distribution for Different Initial Values")
for T_t_0 in T_t_0s:
    x, temperature, time_evolution, amplitude = solve(n_points=15, delta_t=delta_t, n_steps=n_steps, initial_condition_func=lambda x: np.full_like(x, T_t_0), T_t_0=T_t_0, T_t_1=T_t_1)
    plt.plot(x, temperature, label=f'T(t={n_steps * delta_t:.2f}, x) with T(0,x)={T_t_0}')
    decay_results.append((T_t_0, amplitude))

plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.legend()
plt.savefig("dirichlet_bc_time_evolution.png")
plt.close()

# Decay rate: the deviation from the steady state decays like exp(-pi^2 t)
times = np.arange(n_steps + 1) * delta_t
plt.figure(figsize=(10, 6))
plt.title("Decay of Deviation from Steady State (Dirichlet BCs)")
for T_t_0, amplitude in decay_results:
    plt.semilogy(times, amplitude, label=f'T(0,x)={T_t_0}')
plt.xlabel('Time (t)')
plt.ylabel('max |T - T_steady|')
plt.grid(which='both')
plt.legend()
plt.savefig("dirichlet_bc_decay.png")
plt.close()



# Next we implement homogenious boundary conditions on the right end of the rod

def solve(n_points=15, delta_t=0.001, n_steps=50, initial_condition_func=T_0_x,
          T_t_0=350, T_t_1=250, right_bc="dirichlet"):
    x = np.linspace(0.0, 1.0, n_points)
    delta_x = x[1] - x[0]
    values = np.asarray(initial_condition_func(x), dtype=float).reshape(-1)
    values[0] = T_t_0
    if right_bc == "dirichlet":
        values[-1] = T_t_1
    else:
        values[-1] = values[-2]  # Homogeneous Neumann BC

    # Steady state: linear profile for Dirichlet, constant T_t_0 for Neumann
    if right_bc == "dirichlet":
        steady_state = T_t_0 + (T_t_1 - T_t_0) * x
    else:
        steady_state = np.full_like(x, T_t_0)

    time_evolution = [values.copy()]
    amplitude = [np.max(np.abs(values - steady_state))]
    coefficient = delta_t / delta_x**2
    for _ in range(n_steps):
        previous = values.copy()
        values[1:-1] = previous[1:-1] + coefficient * (
            previous[2:] - 2.0 * previous[1:-1] + previous[:-2]
        )
        values[0] = T_t_0
        if right_bc == "dirichlet":
            values[-1] = T_t_1
        else:
            values[-1] = values[-2]  # Homogeneous Neumann BC
        time_evolution.append(values.copy())
        amplitude.append(np.max(np.abs(values - steady_state)))

    return x, values, time_evolution, np.array(amplitude)


T_t_0 = 350
T_t_1 = 200


# Homogeneous Neumann boundary condition on the right end of the rod
plt.figure(figsize=(10, 6))
plt.title("Time Evolution with Homogeneous Neumann BC at x=1")
x, temperature, time_evolution, amplitude = solve(n_points=15, delta_t=delta_t, n_steps=2000,
                                                  right_bc="neumann")
for n in [20, 50, 200, 500, 600, 700, 800, 900, 1000, 1500, 2000]:
    plt.plot(x, time_evolution[n], label=f't = {n * delta_t:.2f}')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (T)')
plt.legend()
plt.savefig("neumann_bc_time_evolution.png")
plt.close()

# Decay rate with Neumann BC: steady state is constant T_t_0 and the slowest
# mode is sin(pi x / 2), so the deviation decays like exp(-pi^2 / 4 t)
times = np.arange(len(time_evolution)) * delta_t
plt.figure(figsize=(10, 6))
plt.title("Decay of Deviation from Steady State (Neumann BC at x=1)")
plt.semilogy(times, amplitude)
plt.xlabel('Time (t)')
plt.ylabel('max |T - T_steady|')
plt.grid(which='both')
plt.savefig("neumann_bc_decay.png")
plt.close()

