# Chapter 02 — Deep Reinforcement Learning

> When the state space is too large for a table, use a neural network.

---

## Why Deep RL?

Tabular methods store Q-values in a table with one row per (state, action) pair. This works fine for LunarLander with 65,536 discretized states. But consider Atari Pong:

- Input: 84×84 grayscale pixels
- Possible unique frames: 256^(84×84) ≈ 10^17,000

No table can store that. Instead, we train a **neural network** that takes the raw state as input and outputs Q-values for all actions:

```
State (pixels) → [Neural Network] → Q(s, do_nothing), Q(s, fire), ...
```

The network learns to generalize — similar states produce similar Q-value predictions.

---

## The Challenges of Neural Q-Learning

Naively replacing the Q-table with a neural network is unstable. Two key problems:

1. **Correlated samples** — consecutive frames are highly correlated, which breaks gradient descent assumptions.
2. **Moving target** — we're using the network to compute *both* the prediction and the target, which causes the loss to shift underneath us constantly.

DQN solves both with two innovations.

---

## Key DQN Innovations

### 1. Experience Replay
Store past transitions `(s, a, r, s')` in a replay buffer. During training, sample a **random mini-batch** from the buffer instead of using the latest transition. This breaks the correlation between consecutive samples.

### 2. Target Network
Keep two networks:
- **Main network** — updated every step
- **Target network** — a frozen copy, updated every N steps

Use the target network to compute the TD target, so it doesn't shift on every update.

---

## Sub-chapters

| Sub-chapter | Algorithm | Environment |
|-------------|-----------|-------------|
| [DQN](./dqn/) | Deep Q-Network | Atari Pong |

---

## Prerequisites

This chapter requires:
- Understanding of Q-learning (Chapter 01)
- Basic PyTorch (tensors, `nn.Module`, `optim`)
- Atari ROMs installed (`autorom --accept-license`)

---

## Next Step

DQN is still a value-based method — it learns Q-values and derives a policy from them. An alternative approach is to learn the policy directly.

→ [Chapter 03 — Policy Gradients](../03_policy_gradients/)
