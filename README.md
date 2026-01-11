# Triadic Intelligence Framework (TIF)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Issues](https://img.shields.io/github/issues/yourusername/triadic-intelligence-framework)](https://github.com/yourusername/triadic-intelligence-framework/issues)

## Overview

The Triadic Intelligence Framework (TIF) is a dynamical systems model for simulating emergent intelligence through triadic interactions. Inspired by cognitive psychology, complex systems, and AI, it models three interconnected facets (e.g., beliefs or agents) evolving toward coherence, modulated by epistemic closure, memory, and dwelling (ambiguity tolerance).

Key Features:
- **Bistability & Failsafes**: Systems decay to low-energy states for weak inputs (noise filtering) but amplify to persistent high coherence with a single "nudge."
- **Components**: Nonlinear Hill activations, state-based memory (fast/slow), epistemic closure (inhibits plasticity), and dwelling (promotes exploration).
- **Applications**: Cognitive simulations, human-AI collaboration, decision-making under uncertainty, and interdisciplinary modeling (e.g., biology, policy).

This repo contains the core Python implementation using NumPy and SciPy. Born from a collaborative triadic thread (December 2025–January 2026), it treats the "decay bug" as a protective feature.

## Installation

```bash
git clone https://github.com/RMac-triadicIntelligence/triadic-intelligence-framework.git
cd triadic-intelligence-framework
pip install -r requirements.txt
Quick Start
Run a basic simulation:
Pythonfrom src.state_memory_system import StateMemorySystem
import numpy as np
from scipy.integrate import solve_ivp

model = StateMemorySystem()
z0 = np.array([logit(0.08), logit(0.22), logit(0.12), logit(0.05), logit(0.10), logit(0.10), logit(0.20)])
sol = solve_ivp(model.dynamics, (0, 50), z0, method="Radau")
# Analyze sol.y (transform back with sigmoid)
See examples/basic_simulation.py for full code and plotting.
Documentation

Framework Overview: Detailed semantics, uses, and implications.
Parameters: Tune gamma, base_decay, etc., for custom behaviors.
Extensions: Add nudges (external inputs) or scale to multi-agent systems.

Contributing
Fork, create a branch, and submit PRs! Issues welcome for bugs, features, or applications.

Roadmap: Stochastic noise, visualization tools (Matplotlib/Plotly), empirical fitting (e.g., to fMRI data).
Easter Egg: Check code comments for motivational notes on emergence.

License
MIT License. See LICENSE for details.
Acknowledgments
Emerged from discussions on X and AI threads. Thanks to collaborators for reframing "bugs" as features!
### 2. LICENSE (MIT License)


MIT License
Attribution required Rusty Williams McMurray
Copyright (c) 2026 Rusty Williams McMurray

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
