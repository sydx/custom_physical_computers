import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the semiprime target
N = 221  # 13 * 17
alpha = 10  # weight for the integer bias

# Define the modified energy function incorporating entropy bias
def energy_prime_bias(xy):
    x, y = xy
    constraint_penalty = (x * y - N) ** 2
    integer_bias = (x - np.round(x)) ** 2 + (y - np.round(y)) ** 2
    entropy_bias = 1 / (x * y + 1e-6)  # Encourage simplicity (larger product = more composite)
    return constraint_penalty + alpha * integer_bias + 0.1 * entropy_bias

# Initial guesses
initial_guesses = [(10, 22), (12, 18), (14, 16), (15, 15)]
results = []

# Run minimization
for guess in initial_guesses:
    res = minimize(energy_prime_bias, guess, method='Nelder-Mead')
    results.append((res.x, res.fun))

# Sort by energy
results.sort(key=lambda x: x[1])
best_xy, best_energy = results[0]
best_xy_rounded = (round(best_xy[0]), round(best_xy[1]))

print(best_xy, best_xy_rounded, best_energy)
