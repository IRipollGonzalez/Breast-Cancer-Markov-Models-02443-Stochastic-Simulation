
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, anderson_ksamp
from scipy.linalg import expm
import seaborn as sns

np.random.seed(42)

# Define transition rate matrix Q
Q = np.array([
    [-0.0085, 0.005, 0.0025, 0, 0.001],
    [0, -0.014, 0.005, 0.004, 0.005],
    [0, 0, -0.008, 0.003, 0.005],
    [0, 0, 0, -0.009, 0.009],
    [0, 0, 0, 0, 0]
])

# Submatrix Qs (remove death state)
Qs = Q[:4, :4]
p0 = np.array([1, 0, 0, 0])
ones = np.ones((4, 1))

# Simulation parameters
n_patients = 1000
max_time = 30.5

lifetimes = []
distant_metastasis = []

# Simulate patient trajectories
for _ in range(n_patients):
    state = 0
    time = 0
    had_distant = False
    while state != 4:
        rate = -Q[state, state]
        if rate <= 0:
            break
        sojourn = np.random.exponential(scale=1 / rate)
        time += sojourn
        probs = Q[state] / -Q[state, state]
        probs[state] = 0
        probs = probs / probs.sum()
        state = np.random.choice(range(5), p=probs)
        if time <= max_time and state == 2:
            had_distant = True
    lifetimes.append(time)
    distant_metastasis.append(had_distant)

# Convert to arrays
lifetimes = np.array(lifetimes)
distant_metastasis = np.array(distant_metastasis)

# Summary statistics
mean_life = lifetimes.mean()
std_life = lifetimes.std(ddof=1)
ci_mean = norm.interval(0.95, loc=mean_life, scale=std_life / np.sqrt(n_patients))
ci_std = norm.interval(0.95, loc=std_life, scale=std_life / np.sqrt(2 * (n_patients - 1)))
distant_prop = np.mean(distant_metastasis)

# Theoretical CDF
times = np.linspace(0, lifetimes.max(), 300)
theoretical_cdf = [1 - (p0 @ expm(Qs * t) @ ones).item() for t in times]

# Save plots
plt.figure(figsize=(10, 6))
sns.histplot(lifetimes, bins=50, kde=True)
plt.title("Histogram of Simulated Lifetimes")
plt.xlabel("Time (months)")
plt.ylabel("Number of Women")
plt.grid(True)
plt.savefig("ctmc_lifetimes_histogram.png", bbox_inches='tight')
plt.show()
plt.close()

# Empirical CDF
sorted_lifetimes = np.sort(lifetimes)
empirical_cdf = np.arange(1, len(sorted_lifetimes)+1) / len(sorted_lifetimes)

# Simulated sample from theoretical CDF
simulated_theoretical_sample = np.interp(np.random.rand(n_patients), theoretical_cdf, times)

# Anderson-Darling k-sample test
from scipy.stats._stats_py import PermutationMethod
ad_stat, ad_crit, ad_significance = anderson_ksamp(
    [lifetimes, simulated_theoretical_sample],
    method=PermutationMethod()
)
# Print results
print("Mean lifetime: {:.2f} months (95% CI: {:.2f}, {:.2f})".format(mean_life, *ci_mean))
print("Standard deviation: {:.2f} months (95% CI: {:.2f}, {:.2f})".format(std_life, *ci_std))
print("Proportion with distant metastasis before 30.5 months: {:.3f}".format(distant_prop))
print("Anderson-Darling test statistic: {:.3f}".format(ad_stat))
print("p-value (approx): > {:.2f}".format(ad_significance))
print("Critical values:", ad_crit)

# Plot CDF comparison
plt.figure(figsize=(10, 6))
plt.step(sorted_lifetimes, empirical_cdf, label="Empirical CDF", color='blue')
plt.plot(times, theoretical_cdf, label="Theoretical CDF", color='red', linestyle='--')
plt.title("Comparison of Empirical and Theoretical CDFs")
plt.xlabel("Time (months)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid(True)
plt.savefig("ctmc_empirical_vs_theoretical_cdf.png", bbox_inches='tight')
plt.show()
plt.close()
