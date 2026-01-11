import numpy as np
from scipy.integrate import solve_ivp

# -----------------------------
# Helpers
# -----------------------------
def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def logit(x):
    x = np.clip(x, 1e-9, 1.0 - 1e-9)
    return np.log(x / (1.0 - x))

def hill(x, gamma=10.0, k=0.5, n=6):
    return gamma * (x**n) / (k**n + x**n + 1e-12)

# -----------------------------
# State-Based Memory + Dwelling Model
# -----------------------------
class StateMemorySystem:
    """
    Continuous-time Triadic system with:
    - Three interacting facets (x1, x2, x3)
    - Epistemic closure variable (c)
    - Fast + slow state-based memory (Mf, Ms)
    - Dwelling / ambiguity tolerance field (D)

    Semantics:
    - High closure inhibits plasticity and accelerates decay
    - Dwelling suppresses premature closure

    # Easter Egg: The 'decay bug' is a feature—failsafe against noise!
    """

    def __init__(self,
                 gamma=10.0, k=0.5, n=6,
                 base_decay=0.22,
                 memory_factor=0.6,
                 tau_fast=1.0,
                 tau_slow=8.0,
                 eps=1e-9):
        self.p = locals()
        del self.p["self"]

    def dynamics(self, t, z):
        # Unpack state
        z1, z2, z3, zc, zMf, zMs, zD = z
        p = self.p

        # Bounded states
        x1, x2, x3 = sigmoid(z1), sigmoid(z2), sigmoid(z3)
        c = sigmoid(zc)     # closure (higher = more epistemically closed)
        Mf = sigmoid(zMf)   # fast memory
        Ms = sigmoid(zMs)   # slow memory
        D = sigmoid(zD)     # dwelling / ambiguity tolerance

        # Coherence metrics
        m = (x1 + x2 + x3) / 3.0           # mean coherence
        div = np.std([x1, x2, x3])         # divergence / disagreement

        # Nonlinear activations
        s1 = hill(x1, p["gamma"], p["k"], p["n"])
        s2 = hill(x2, p["gamma"], p["k"], p["n"])
        s3 = hill(x3, p["gamma"], p["k"], p["n"])

        # Memory dynamics (state-based integration)
        dMf = (m - Mf) / p["tau_fast"]
        dMs = (Mf - Ms) / p["tau_slow"]
        memory = 0.7 * Mf + 0.3 * Ms

        # Dwelling dynamics
        # Dwelling rises with divergence + low coherence
        # Dwelling fades as coherence stabilizes
        dD = (0.8 * div * (1 - m) * (1 - D)) - (0.6 * m * D)

        # Closure dynamics
        # Divergence increases closure
        # Coherence and dwelling suppress closure
        dc = (2.5 * div * (1 - c)) - (m * c) - (1.2 * D * c)
        dc = np.clip(dc, -2.0, 2.0)

        # Plasticity & decay (FIXED SEMANTICS)
        # Closure now REDUCES plasticity and INCREASES decay
        boost = 1.0 + 0.7 * (1.0 - c)
        eff_decay = p["base_decay"] * (1.0 + 0.55 * c)

        # State evolution
        dx1 = boost * ((s2 + s3) / 2) * (1 - x1) - eff_decay * x1 \
              + p["memory_factor"] * (memory - x1)

        dx2 = boost * ((s1 + s3) / 2) * (1 - x2) - eff_decay * x2 \
              + p["memory_factor"] * (memory - x2)

        dx3 = boost * ((s1 + s2) / 2) * (1 - x3) - eff_decay * x3 \
              + p["memory_factor"] * (memory - x3)

        # Logistic back-transform
        def dz(x, dx):
            return dx / (x * (1 - x) + p["eps"])

        return np.array([
            dz(x1, dx1),
            dz(x2, dx2),
            dz(x3, dx3),
            dz(c, dc),
            dz(Mf, dMf),
            dz(Ms, dMs),
            dz(D, dD)
        ])
