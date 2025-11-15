# Breast-Cancer-Markov-Models
**Stochastic Simulation – DTU 02443**  
**Group 15**

This repository contains the full implementation of a stochastic model for breast cancer progression based on discrete- and continuous-time Markov models.  
The project follows the official DTU assignment and implements all tasks (1–13), including simulation, estimation, likelihood construction, and the MCEM algorithm.

---

## 1. Project Context

**Course:** 02443 – Stochastic Simulation  
**Institution:** Technical University of Denmark  
**Academic Period:** 2025  
**Deliverables:** Code + Notebooks + Final Report  

The goal of the project is to model breast cancer progression using both:

1. **Discrete-Time Markov Chains (DTMC)**
2. **Continuous-Time Markov Chains (CTMC)**
3. **Likelihood-based inference**
4. **Monte Carlo EM (MCEM)**

Data for inference is **synthetic**, generated via simulation from the Markov models.

---

# 3. Overview of Tasks (1–13)

Below is a **detailed explanation** of each task and its implementation in the provided notebooks/scripts.

---

## **Tasks 1–6 — Discrete-Time Markov Chain (DTMC)**  
*Implemented in:*  
`notebooks/01_discrete_time_markov_chain.ipynb`

### **Task 1 – Define the State Space**
We model breast cancer using the standard clinical progression states:

1. Healthy  
2. Cancer Stage 1  
3. Cancer Stage 2  
4. Cancer Stage 3  
5. Dead (absorbing)  

These states define the Markov chain's domain.

---

### **Task 2 – Transition Probability Matrix**
The transition matrix **P** is specified using medical plausibility:

- Probability of progressing to a worse stage increases with severity  
- Probability of remission is low but nonzero  
- Death is absorbing  
- All rows sum to 1  

Example structure:
Healthy → Healthy, Stage1
Stage1 → Stage1, Stage2
…
Stage3 → Stage3, Dead
Dead → Dead

---

### **Task 3 – Simulating Patient Trajectories**
We simulate **longitudinal patient histories**:

- A starting state (usually Healthy)  
- T time steps    
- One-step transitions using P  

Multiple patients are simulated to approximate the distribution of disease progression.

---

### **Task 4 – Empirical vs. Theoretical State Distribution**
We compare:

- theoretical `π(t) = π(0) P^t`  
- empirical distribution from simulation  

This validates the DTMC implementation.

---

### **Task 5 – Long-Term Behavior**
Using properties of Markov chains:

- Identify absorbing states  
- Study absorption probabilities  
- Estimate long-run death probability  
- Compute expected number of steps until absorption  

---

### **Task 6 – Parameter Sensitivity Analysis**
We vary transition probabilities:

- progression rate  
- remission probability  
- mortality  
- time horizon  

Then measure the effect on:

- survival time  
- time spent in each state  
- distribution of cancer stages  

---

# **Tasks 7–8 — Continuous-Time Markov Chain (CTMC)**  
*Implemented in:*  
`src/ctmc_model.py`  
`notebooks/02_ctmc_simulation.ipynb`

### **Task 7 – Building the Generator Matrix Q**
A CTMC is defined via **Q**, where:

- Off-diagonal `q_ij` = transition rate  
- Diagonal entries `q_ii = - Σ_j≠i q_ij`  
- Rows sum to zero  
- All off-diagonals ≥ 0  

This model allows:

- exponentially distributed waiting times  
- realistic continuous progression  
- finer modeling than DTMCs  

---

### **Task 8 – Simulating CTMC Trajectories**
Simulation uses:

1. **Waiting time:** `Δt ~ Exponential(|q_ii|)`  
2. **Next state:** categorical distribution proportional to outgoing rates  
3. **Stop when absorbing**  

Outputs:

- full trajectory with timestamps  
- total time in each disease stage  
- time until death  

---

# **Tasks 9–13 — Likelihood, Inference, and MCEM**  
*Implemented in:*  
`notebooks/03_inference_MLEM_MCEM.ipynb`

---

## **Task 9 – Likelihood Function for DTMC/CTMC**
We derive:

### **DTMC likelihood**
For a trajectory with transitions `i→j`:
L(P | data) = Π (P_ij)^(N_ij)

### **CTMC likelihood**
For generator Q:
L(Q | data) = Π (q_ij)^(N_ij) * exp(-q_i * T_i)

Where:

- `N_ij` = number of transitions from i to j  
- `T_i` = time spent in state i  
- `q_i = -q_ii`  

---

## **Task 10 – Maximum Likelihood Estimation (MLE)**
Closed-form MLE:

- For DTMC:
P_ij_hat = N_ij / Σ_j N_ij

- For CTMC:
q_ij_hat = N_ij / T_i
q_ii_hat = - Σ_j≠i q_ij_hat

Estimates are computed directly from simulated trajectories.

---

## **Task 11 – Observed Data with Hidden Timings**
When time-in-state is unobserved, **likelihood is incomplete**.

We treat the insufficient data using a missing-data framework.

---

## **Task 12 – Monte Carlo E-Step**
Monte Carlo methods approximate missing expectations:
E[N_ij],  E[T_i]

by:

- simulating hidden paths  
- averaging over samples  

This gives a stochastic approximation of the complete-data statistics.

---

## **Task 13 – MCEM Algorithm**
The **Monte Carlo Expectation-Maximization** loop:

1. **E-step:** simulate hidden transitions and times  
2. **M-step:** update estimates with MLE formulas  
3. **Repeat until convergence**  

Outputs:

- estimated transition matrix for DTMC  
- estimated generator Q for CTMC  
- convergence plots  
- log-likelihood curves  

This section implements a *full working MCEM algorithm*.
