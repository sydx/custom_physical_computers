from itertools import product
import numpy as np
import time

# Full mass list (19 items)
masses = [23, 43, 12, 54, 7, 3, 5, 10, 54, 55, 26, 9, 43, 54, 1, 8, 6, 38, 33]
num_buckets = 4

def variance(sums):
    mean = sum(sums) / len(sums)
    return sum((x - mean) ** 2 for x in sums) / len(sums)

def energy_heuristic(masses, num_buckets):
    bucket_sums = [0] * num_buckets
    for m in masses:
        idx = min(range(num_buckets), key=lambda i: (bucket_sums[i] + m) ** 2)
        bucket_sums[idx] += m
    return bucket_sums

def lpt_heuristic(masses, num_buckets):
    sorted_masses = sorted(masses, reverse=True)
    bucket_sums = [0] * num_buckets
    for m in sorted_masses:
        idx = min(range(num_buckets), key=lambda i: bucket_sums[i])
        bucket_sums[idx] += m
    return bucket_sums

def correct_qubo_brute_force(masses, num_buckets, penalty=1000):
    n = len(masses)
    min_energy = float('inf')
    best_sums = None
    best_assignment = None

    for assignment in product(range(num_buckets), repeat=n):
        x = [[1 if assignment[i] == k else 0 for k in range(num_buckets)] for i in range(n)]
        if not all(sum(x[i]) == 1 for i in range(n)):
            continue

        bucket_sums = [sum(masses[i] * x[i][k] for i in range(n)) for k in range(num_buckets)]
        energy = sum(s**2 for s in bucket_sums)

        if energy < min_energy:
            min_energy = energy
            best_sums = bucket_sums[:]
            best_assignment = assignment

    return best_sums, best_assignment, min_energy

# Run energy-based heuristic
start = time.time()
s1 = energy_heuristic(masses, num_buckets)
t1 = time.time() - start

# Run LPT heuristic
start = time.time()
s2 = lpt_heuristic(masses, num_buckets)
t2 = time.time() - start

# Run QUBO on first 10 masses only
masses_small = masses[:10]
start = time.time()
s3, a3, e3 = correct_qubo_brute_force(masses_small, num_buckets)
t3 = time.time() - start

# Display results
results = {
    "Energy-Based Heuristic": {
        "Bucket Sums": s1,
        "Variance": variance(s1),
        "Time (sec)": t1
    },
    "LPT Heuristic": {
        "Bucket Sums": s2,
        "Variance": variance(s2),
        "Time (sec)": t2
    },
    "QUBO Brute Force (10 masses)": {
        "Bucket Sums": s3,
        "Variance": variance(s3),
        "Assignment": a3,
        "Energy": e3,
        "Time (sec)": t3
    }
}

print(results)
