# Chapter 03 — Policy Gradients

> Instead of learning what actions are worth, learn *directly* which actions to take.

---

## Value-Based vs Policy-Based

So far (MC, TD, DQN) we've learned a Q-function and derived a policy from it:

```
Q(s, a)  →  policy = argmax_a Q(s, a)
```

Policy gradient methods skip Q and parameterize the policy directly:

```
π_θ(a | s) = probability of taking action a in state s
```

The neural network outputs action probabilities. We update θ to make *good actions more probable*.

---

## Why Policy Gradients?

| Situation | Value-Based | Policy-Based |
|-----------|------------|--------------|
| Discrete actions | Works well | Works well |
| Continuous actions | Hard (argmax over ∞ actions) | Works naturally |
| Stochastic policy needed | Awkward | Natural |
| Convergence guarantees | None in general | Yes (for tabular case) |

---

## Environment: CartPole-v1

A pole balanced on a cart — keep it upright by pushing left or right.

- **State**: 4 values (cart position, cart velocity, pole angle, pole angular velocity)
- **Actions**: 2 (push left, push right)
- **Reward**: +1 for every step the pole stays up
- **Solved**: Mean reward ≥ 450 over 100 episodes

---

## Network Architecture

```python
class PGN(nn.Module):
    Input(4) → Linear(128) → ReLU → Linear(2) → logits
```

Action is sampled from `softmax(logits)` — not argmax. The policy is *stochastic*.

---

## The REINFORCE Algorithm

**Core idea:** Actions that led to high returns should be made more probable.

```
Loss = -Σ log π(a_t | s_t) · G_t
```

Where `G_t` is the discounted return from step t:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

Minimizing this loss = maximizing the log-probability of good actions.

---

## Files

| File | Algorithm | Key Difference |
|------|-----------|----------------|
| [reinforce.py](./reinforce.py) | Vanilla REINFORCE | Uses raw discounted returns as weights |
| [reinforce_with_baseline.py](./reinforce_with_baseline.py) | REINFORCE + Baseline | Subtracts mean return to reduce variance |

---

## Chapter 03a — Vanilla REINFORCE

**Algorithm:**
```
Collect 4 episodes
For each step t in each episode:
    G_t = discounted return from t
    loss += -log π(a_t | s_t) · G_t
Backpropagate and update
```

**Problem:** High variance. One lucky/unlucky episode can dominate the gradient.

---

## Chapter 03b — REINFORCE with Baseline

**The fix:** Subtract the mean return from each step's weight:

```python
advantages = [G_t - mean(G) for G_t in returns]
loss = -Σ log π(a_t | s_t) · advantage_t
```

This doesn't change the *direction* of the gradient (unbiased) but dramatically reduces its *variance*. Good actions still get reinforced, but the signal is cleaner.

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `GAMMA` | 0.99 | Discount factor |
| `LEARNING_RATE` | 0.01 | Adam optimizer LR |
| `EPISODES_TO_TRAIN` | 4 | Episodes collected before each gradient update |

---

## How to Run

```bash
cd 03_policy_gradients

# Vanilla REINFORCE
python reinforce.py

# REINFORCE with Baseline (converges faster)
python reinforce_with_baseline.py
```

**Expected output:**
```
1: Episode 0, Reward:  22.00, Mean-100:  22.00
142: Episode 1, Reward:  15.00, Mean-100:  18.50
...
Solved in XXXX steps and XXX episodes!
```

---

## Monitor with TensorBoard

```bash
tensorboard --logdir runs/
```

Metrics logged:
- `reward` — episode reward
- `reward_100` — 100-episode moving average
- `episodes` — episode count
- `qvals_variance` — variance of Q-values in the batch (watch this drop with baseline!)

---

## Variance Comparison

Run both scripts and compare `qvals_variance` in TensorBoard. The baseline version should show significantly lower variance, leading to more stable and faster convergence.

---

## Strengths and Weaknesses

**Strengths:**
- Works for continuous action spaces
- Directly optimizes the policy
- Simple and theoretically grounded

**Weaknesses:**
- High variance (mitigated by baseline, but not eliminated)
- Sample inefficient — each episode is discarded after one update
- Sensitive to learning rate

---

## What's Next?

The natural next step is **Actor-Critic** methods (A2C, A3C) which combine a policy network (actor) with a learned value network (critic) as a baseline — getting the best of both worlds.

Beyond that: **PPO** (Proximal Policy Optimization), the current go-to algorithm for most practical RL tasks.

→ Back to [Root README](../README.md) for the full roadmap
