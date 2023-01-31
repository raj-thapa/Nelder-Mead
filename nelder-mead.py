import numpy as np
import math

def nelder_mead(func, x0, alpha=1, beta=0.5, gamma=2, max_iter=1000, tol=1e-8):
    # Initialize the simplex
    n = len(x0)
    simplex = np.zeros((n+1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i+1] = x0 + (alpha * (i+1)) * np.eye(n)[i]
    
    # Iterate until the function reaches a local minimum or maximum
    for i in range(max_iter):
        # Sort the simplex by the function value
        f_values = np.array([func(x) for x in simplex])
        simplex = simplex[np.argsort(f_values)]
        
        # Compute the centroid of the simplex, excluding the worst point
        x_bar = np.mean(simplex[:-1], axis=0)
        
        # Compute the reflection point
        x_r = x_bar + alpha * (x_bar - simplex[-1])
        
        # Check if the reflection point is the best point
        if func(x_r) < f_values[0]:
            # Compute the expansion point
            x_e = x_bar + gamma * (x_r - x_bar)
            
            # Check if the expansion point is better than the reflection point
            if func(x_e) < func(x_r):
                simplex[-1] = x_e
            else:
                simplex[-1] = x_r
        else:
            # Check if the reflection point is better than the second worst point
            if func(x_r) < f_values[-2]:
                simplex[-1] = x_r
            else:
                # Check if the reflection point is worse than the worst point
                if func(x_r) < f_values[-1]:
                    # Compute the contraction point
                    x_c = x_bar + beta * (x_r - x_bar)
                    
                    # Check if the contraction point is better than the worst point
                    if func(x_c) < f_values[-1]:
                        simplex[-1] = x_c
                    else:
                        # Shrink the simplex
                        simplex[1:] = simplex[0] + (1/2) * (simplex[1:] - simplex[0])
                        
        # Check for convergence
        if np.max(np.abs(simplex[0] - simplex[-1])) < tol:
            return simplex[0]
    return simplex[0]


def function(x):
    return 2*math.pi*x[0]**2 + 2*math.pi*x[0]*x[1]

initial_guess = [0, 0]
result = nelder_mead(function, initial_guess)

print(result) 



























