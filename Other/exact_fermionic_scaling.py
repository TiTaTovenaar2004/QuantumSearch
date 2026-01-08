import numpy as np
import matplotlib.pyplot as plt
import math
import random

def estimate_success_probability(N, M, time, gamma, r, precision=0.01, confidence=0.99):
    Omega = gamma * np.sqrt(M * (N - M))
    Delta = 0.5*(gamma*(N - 2*M) - 1)
    omega = np.sqrt(Omega**2 + Delta**2)
    w_prob = (Omega**2 / (Omega**2 + Delta**2)) * np.sin(omega * time)**2

    num_samples = math.ceil((np.log((1 - confidence) / 2)) / (-2 * precision**2))
    
    occupations = np.zeros((r, N), dtype=int)

    num_of_successes = 0

    for sample in range(num_samples):
        for round in range(r):
            if random.random() < w_prob:
                occupations[round, 0] = 1
                remaining_indices = np.random.choice(
                    np.arange(1, N),
                    size=M - 1,
                replace=False
            )
            else:
                remaining_indices = np.random.choice(
                    np.arange(1, N),
                    size=M,
                    replace=False
                )

            occupations[round, remaining_indices] = 1

        total_occupation = occupations.sum(axis=0)

        vertex_0_occupation = total_occupation[0]
        max_other_occupation = np.max(total_occupation[1:])

        if vertex_0_occupation > max_other_occupation:
            num_of_successes += 1
        
        occupations = np.zeros((r, N), dtype=int)
    
    p_hat = num_of_successes / num_samples

    return p_hat, p_hat - precision, p_hat + precision

def estimate_and_plot_success_probabilities(
    N,
    M,
    times,
    gamma,
    r_values,
    precision=0.01,
    confidence=0.99
):
    plt.figure(figsize=(8, 5))

    for r in r_values:
        probs = []
        lowers = []
        uppers = []

        for t in times:
            p, lo, hi = estimate_success_probability(
                N, M, t, gamma, r, precision, confidence
            )
            probs.append(p)
            lowers.append(lo)
            uppers.append(hi)

        probs = np.array(probs)
        lowers = np.array(lowers)
        uppers = np.array(uppers)

        plt.plot(times, probs, label=f"r = {r}")
        plt.fill_between(times, lowers, uppers, alpha=0.25)

    plt.xlabel("Time")
    plt.ylabel("Estimated success probability")
    plt.title("Success probability vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

estimate_and_plot_success_probabilities(
    N=6,
    M=2,
    times=np.linspace(0, 10, 10),
    r_values=[1, 2, 3, 4, 5, 6], 
    gamma=1/6
)

