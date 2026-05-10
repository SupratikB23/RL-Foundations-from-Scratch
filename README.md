# RL Explorations

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
<!-- Add your GitHub Actions badge here once CI is set up -->
<!-- ![Status](https://img.shields.io/badge/Status-Active-brightgreen) -->

A hands-on, from-scratch journey through **Reinforcement Learning** — from the very basics of environments all the way to Deep Q-Networks and Policy Gradients. Every algorithm here is implemented manually, trained on real environments, and explained clearly so anyone can follow along.

> Built as a personal learning log, but structured so others can learn sequentially too.

---

## Learning Roadmap

Follow the chapters in order — each one builds on concepts from the previous.

| # | Chapter | Algorithm | Environment | Key Concept |
|---|---------|-----------|-------------|-------------|
| 00 | [Introduction](./00_introduction/) | — | LunarLander-v3 | Gymnasium basics, action/observation spaces |
| 01a | [Monte Carlo](./01_tabular_methods/monte_carlo/) | Monte Carlo Q-Learning | LunarLander-v3 | Episode-based learning, Q-tables |
| 01b | [Temporal Difference](./01_tabular_methods/temporal_difference/) | Q-Learning (TD-0) | LunarLander-v3 | Online learning, TD targets |
| 02 | [Deep Q-Network](./02_deep_rl/dqn/) | DQN | Atari Pong | Neural nets as Q-functions, experience replay |
| 03a | [REINFORCE](./03_policy_gradients/) | Vanilla Policy Gradient | CartPole-v1 | Policy-based learning, log-prob |
| 03b | [REINFORCE + Baseline](./03_policy_gradients/) | PG with Baseline | CartPole-v1 | Variance reduction, advantage |

---

## Concepts Covered

### Value-Based Methods
- Q-table construction and update rules
- State discretization for continuous observation spaces
- Monte Carlo vs Temporal Difference: offline vs online learning
- Deep Q-Network: replacing the Q-table with a CNN
- Experience replay buffer, target networks, frame stacking

### Policy-Based Methods
- Policy parameterization with a neural network
- REINFORCE algorithm: log-prob weighted by discounted return
- Baseline subtraction for variance reduction (advantage estimation)

### Exploration
- Epsilon-greedy strategy with decay schedules
- Greedy policy for evaluation

### Tooling
- [Gymnasium](https://gymnasium.farama.org/) for environments
- [PyTorch](https://pytorch.org/) for neural networks
- [TensorBoard](https://www.tensorflow.org/tensorboard) for training metrics
- Video recording of trained agents

---

## Repository Structure

```
rl-explorations/
├── 00_introduction/            # Gymnasium basics
├── 01_tabular_methods/
│   ├── monte_carlo/            # MC Q-Learning on LunarLander
│   └── temporal_difference/    # TD Q-Learning on LunarLander
├── 02_deep_rl/
│   └── dqn/                    # DQN on Atari Pong
├── 03_policy_gradients/        # REINFORCE on CartPole
└── assets/                     # Diagrams and images
```

---

## Prerequisites

- Python 3.10+
- Basic Python knowledge
- Familiarity with linear algebra and probability is helpful but not required

---

## Installation

```bash
# Clone the repo
git clone https://github.com/<SupratikB23>/rl-explorations.git
cd rl-explorations

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install gymnasium[box2d] torch torchvision tensorboard ale-py
```

For Atari (DQN chapter):
```bash
pip install gymnasium[atari] autorom
autorom --accept-license
```

---

## Running TensorBoard

All training scripts log metrics to TensorBoard. After running any script:

```bash
tensorboard --logdir runs/
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

---

## Progress

- [x] 00 — Environment basics
- [x] 01a — Monte Carlo Q-Learning
- [x] 01b — Temporal Difference Q-Learning
- [x] 02 — Deep Q-Network (DQN)
- [x] 03a — REINFORCE
- [x] 03b — REINFORCE with Baseline
- [ ] Actor-Critic (A2C) *(coming soon)*
- [ ] Proximal Policy Optimization (PPO) *(coming soon)*

---

## References & Resources

- [Reinforcement Learning: An Introduction — Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html) *(the RL bible)*
- [Deep Reinforcement Learning Hands-On — Maxim Lapan](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Playing Atari with Deep Reinforcement Learning — Mnih et al. (2013)](https://arxiv.org/abs/1312.5602)
- [Policy Gradient Methods — Sutton et al. (1999)](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

---

## License

MIT License — feel free to use, fork, and learn from this code.
