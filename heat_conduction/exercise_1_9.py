"""
Implementation of the implicit vs explicit euler scheme

Explicit Scheme
T_i^n+1 = T_i^n + Delta_t / Delta x^2 * (T_{i+1}^n - 2 * T_i^n + T_{i-1}^n)

Implicit Scheme
T_i^n+1 = T_i^n + Delta_t / delta x^2 * (T_{i+1}^n+1 - 2 * T_i^n+1 + T_{i-1}^n+1)


where
- T_i^n is the temperature at position i and time n
- Delta t is the time step
- Delta x is the spatial step
"""
import numpy as np
import matplotlib.pyplot as plt



def T_0_x(x):
    """ 
    Initial Value Equation
    """
    x = np.asarray(x, dtype=float)
    values = 350.0 - 100.0 * x + 200.0 * np.sin(np.pi * x)
    return values


def explicit_euler(n_points=15, delta_t= 0.001, n_steps=50, T_t_0 = 350, T_t_1 = 250):
    """
    Solve the one-dimensional heat equation using
    the explicit euler method

    n_points: number of spatial points
    delta_t: time step
    n_steps: number of time steps to simulate
    T_t_0: temperature at x=0
    T_t_1: temperature at x=1
    """
    
    # Spatial Grid
    x = np.linspace(0.0, 1.0, n_points)
    dx = x[1] - x[0]
    T = np.asarray(T_0_x(x), dtype=float).reshape(-1)
    T[0] = T_t_0
    T[-1] = T_t_1
    time_evolution = [T.copy()]
    # Coeff 
    coef= delta_t / dx**2
    for _ in range(n_steps):
        T_prev = T.copy()
        T[1:-1] = T_prev[1:-1] + coef * (T_prev[2:] - 2.0 * T_prev[1:-1] + T_prev[:-2])
        T[0] = T_t_0
        T[-1] = T_t_1
        time_evolution.append(T.copy())
    print("Explicit Euler completed")
    return x, T, time_evolution

def implicit_euler(n_points=15, delta_t= 0.001, n_steps=50, T_t_0 = 350, T_t_1 = 250):
    """ 
    Solve the implicit euler scheme for the one-dimensional heat equation
    """
    x = np.linspace(0.0, 1.0, n_points)
    dx = x[1] - x[0]
    T = np.asarray(T_0_x(x), dtype=float).reshape(-1)
    coeff = delta_t / dx**2
    A = np.diag([-2.0] * n_points) + np.diag([1.0] * (n_points - 1), k=1) + np.diag([1.0] * (n_points - 1), k=-1)
    # Equation in linear algebra form 
    # (I - Delta_t / Delta_x^2 * A) T^{n+1} = T^n + F 
    # F = delta_t / Delta_x^2 * (T_0, 0, 0, ..., T_1)^T
    F = np.zeros(n_points)
    F[0] = coeff * T_t_0
    F[-1] = coeff * T_t_1
    M = np.eye(n_points) - coeff * A
    time_evolution = [T.copy()]
    for _ in range(n_steps):
        T = np.linalg.solve(M, T + F)
        T[0] = T_t_0
        T[-1] = T_t_1
        time_evolution.append(T.copy())
    print("Implicit Euler completed")
    return x, T, time_evolution
    
if __name__ == "__main__":
    delta_t = 0.001
    n_steps = 200
    x, T_explicit, time_evolution_explicit = explicit_euler(n_points=15, delta_t=delta_t, n_steps=n_steps)
    x, T_implicit, time_evolution_implicit = implicit_euler(n_points=15, delta_t=delta_t, n_steps=n_steps)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Explicit Euler Time Evolution")
    for n in [0, 50, 100, 150, 200]:
        plt.plot(x, time_evolution_explicit[n], label=f't = {n * delta_t:.2f}')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature (T)')
    plt.legend()   
    plt.subplot(1, 2, 2)
    plt.title("Implicit Euler Time Evolution")
    for n in [0, 50, 100, 150, 200]:
        plt.plot(x, time_evolution_implicit[n], label=f't = {n * delta_t:.2f}')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature (T)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("euler_implicit_explicit_comparison.png")
    plt.close()

    # Lets now show that we can take larger timesteps in the implict scheme
    time_steps = [0.0001, 0.001, 0.002, 0.003, 0.004]
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title("Explicit Euler with Different Time Steps")
    for dt in time_steps:
        x, T_explicit, time_evolution_explicit = explicit_euler(n_points=15, delta_t=dt, n_steps=n_steps)
        plt.plot(x, T_explicit, label=f'delta_t = {dt}')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature (T)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Implicit Euler with Different Time Steps")
    for dt in time_steps:
        x, T_implicit, time_evolution_implicit = implicit_euler(n_points=15, delta_t=dt, n_steps=n_steps)
        plt.plot(x, T_implicit, label=f'delta_t = {dt}')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature (T)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("euler_implicit_explicit_time_step_comparison.png")
    plt.close()


