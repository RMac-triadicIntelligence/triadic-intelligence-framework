import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from src.state_memory_system import StateMemorySystem, sigmoid, logit

# Initialize model
model = StateMemorySystem()

# Initial conditions (low activity example)
z0 = np.array([
    logit(0.08),  # x1
    logit(0.22),  # x2
    logit(0.12),  # x3
    logit(0.05),  # closure (low initial)
    logit(0.10),  # fast memory
    logit(0.10),  # slow memory
    logit(0.20)   # dwelling
])

# Solve ODE
sol = solve_ivp(
    model.dynamics,
    (0, 50),
    z0,
    method="Radau",
    rtol=1e-7,
    atol=1e-9
)

# Transform back to bounded states
states = sigmoid(sol.y)
labels = ['x1', 'x2', 'x3', 'c', 'Mf', 'Ms', 'D']

# Plot trajectories
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    plt.plot(sol.t, states[i], label=label)
plt.xlabel('Time')
plt.ylabel('State Value')
plt.title('Triadic System Simulation (Low Initial Activity)')
plt.legend()
plt.grid(True)
plt.show()

print("Integration complete. Final states:", states[:, -1])
