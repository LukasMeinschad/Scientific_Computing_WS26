""" 
In this exercise we implement the implicit Euler scheme using
the Thomas Algorithm

Tridiagonal System

A = [a_1, b_1, 0, ..., 0]
    [b_1 a_2, b_2, ..., 0]
    [0, b_2, a_3, ..., 0]
    ...
    [0, ..., b_{n-1}, a_n]

where a_i = 1 + 2 * delta_t / delta_x^2 and b_i = - delta_t / delta_x^2 
# Forward sweep
c_i = a_i - b_{i-1}^2 / c_{i-1},   c_1 = a_1, i>2
g_i = f_i - b_{i-1}/c_{i-1} * g_{i-1},   g_1 = f_1, i>2

# Backward sweep
u_i = (g_i - b_i * u_{i+1})/ c_i, u_n = g_n / c_n, i≤N-1
"""
import numpy as np
import matplotlib.pyplot as plt

def thomas_algorithm(a, b, f):
    """ 
    Solves the Tridiagonal system Ax = f using the Thomas algorithm

    Parameters
    ----------
    a: array_like
        The diagonal elements of the tridiagonal matrix A.
    b: array_like
        The sub-diagonal elements of the tridiagonal matrix A.
    f: array_like
        The right-hand side vector of the system Ax = f.
    """
    n = len(f)
    # initialize c, g and u
    c = [0] * n
    g = [0] * n
    u = [0] * n

    # Forward sweep
    c[0] = a[0]
    g[0] = f[0]
    for i in range(1, n):
        c[i] = a[i] - b[i-1]**2 / c[i-1]
        g[i] = f[i] - b[i-1]/c[i-1] * g[i-1]

    # Backward sweep
    u[n-1] = g[n-1] / c[n-1]
    for i in range(n-2, -1, -1):
        u[i] = (g[i] - b[i] * u[i+1]) / c[i]
    return u

def T_0_x(x):
    """ 
    Initial Value Equation
    """
    x = np.asarray(x, dtype=float)
    values = 350.0 - 100.0 * x + 200.0 * np.sin(np.pi * x)
    return values

def implicit_euler(n_points=15, delta_t=0.001, n_steps=50, T_t_0 = 350, T_t_1 = 250):
    """ 
    Solve the implicit Euler scheme for the one-dimensional heat equation using the Thomas algorithm.
    """
    x = np.linspace(0.0, 1.0, n_points)
    dx = x[1] - x[0]
    T = np.asarray(T_0_x(x), dtype=float).reshape(-1)

    coeff = delta_t / dx**2
    # Only the interior points are unknowns; the boundary values are fixed (Dirichlet)
    n_inner = n_points - 2
    a = [1 + 2 * coeff] * n_inner
    b = [-coeff] * (n_inner - 1)
    T[0] = T_t_0
    T[-1] = T_t_1
    time_evolution = [T.copy()]

    for _ in range(n_steps):
        f = T[1:-1].copy()
        # Move the known boundary values to the right-hand side
        f[0] += coeff * T_t_0
        f[-1] += coeff * T_t_1
        T[1:-1] = thomas_algorithm(a, b, f)
        time_evolution.append(T.copy())
    
    print("Implicit Euler completed")
    return x, T, time_evolution


if __name__ == "__main__":
    # Example usage and comparison with numpy.linalg.solve
    import numpy as np
    a = np.array([4, 4, 4, 4], dtype=float)
    b = np.array([-1, -1, -1], dtype=float)
    f = np.array([5, 5, 5, 5], dtype=float)
    u = thomas_algorithm(a, b, f)
    print("Solution using Thomas algorithm:", u)
    # Compare with numpy.linalg.solve
    A = np.diag(a) + np.diag(b, k=-1) + np.diag(b, k=1)
    u_np = np.linalg.solve(A, f)
    print("Solution using numpy.linalg.solve:", u_np)
    print("Difference:", np.linalg.norm(u - u_np))

    plt.figure(figsize=(10, 6))
    x, T_implicit, time_evolution_implicit = implicit_euler(n_points=15, delta_t=0.001, n_steps=200)

    plt.plot(x, T_implicit, label='Implicit Euler (Thomas Algorithm)', color='blue')
    plt.title("Implicit Euler Scheme using Thomas Algorithm")
    plt.xlabel("Position (x)")
    plt.ylabel("Temperature (T)")
    plt.legend()
    plt.grid()
    plt.savefig("implicit_euler_thomas_algorithm.png")

