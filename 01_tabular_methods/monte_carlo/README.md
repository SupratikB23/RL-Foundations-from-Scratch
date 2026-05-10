# Chapter 01a — Monte Carlo Q-Learning

> Learn by playing complete episodes, then looking back at what actually happened.

---

## Core Idea

Monte Carlo methods wait until the **end of an episode** before updating Q-values. The update is based on the actual discounted return `G` — the sum of all future rewards the agent actually received from that point forward.

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^T·r_T
```

The Q-value for (state, action) is the **running average** of all returns ever seen from that pair:

```
Q(s, a) = mean of all G values observed after taking action a in state s
```

No learning rate needed — it's just averaging.

---

## Algorithm

```
Initialize Q(s, a) = 0 for all (s, a)
Initialize returns_sum(s, a) = 0, returns_count(s, a) = 0

For each episode:
    1. Generate a full episode using ε-greedy policy
    2. Work backwards through the episode:
       For each step t from T to 0:
           G ← r_t + γ·G
           returns_sum(s_t, a_t) += G
           returns_count(s_t, a_t) += 1
           Q(s_t, a_t) = returns_sum / returns_count
    3. Decay epsilon
```

---

## State Discretization

LunarLander's 8 continuous dimensions are bucketed into discrete bins:

```python
NUM_BINS = (8, 8, 8, 8, 8, 8, 2, 2)
# Total states = 8^6 × 2^2 = 65,536
```

The 8 dimensions map to:
1. x position → 8 bins
2. y position → 8 bins
3. x velocity → 8 bins
4. y velocity → 8 bins
5. angle → 8 bins
6. angular velocity → 8 bins
7. left leg contact → 2 bins (binary)
8. right leg contact → 2 bins (binary)

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `GAMMA` | 0.99 | Discount factor — future rewards matter almost as much as immediate ones |
| `EPSILON` | 1.0 → 0.05 | Exploration rate, decays by 0.9995 per episode |
| `NUM_EPISODES` | 30,000 | Total training episodes |

---

## How to Run

```bash
cd 01_tabular_methods/monte_carlo
python monte_carlo.py
```

**Expected output:**
```
Starting LunarLander Monte Carlo training...
State space size: 65,536
Episode  1000/30000 - Avg Reward: -230.4 - States Visited: 4821
Episode  5000/30000 - Avg Reward: -150.2 - States Visited: 18432
...
=== MONTE CARLO RESULTS ===
Training Episodes: 30,000
Average Test Reward: XXX
Successful Landings: XX/50 (XX.X%)
```

Training takes ~15–30 minutes depending on your machine.

---

## Monitor with TensorBoard

```bash
tensorboard --logdir runs/lunarlander_mc
```

Metrics logged:
- `Train/EpisodeReward` — raw reward each episode
- `Train/MovingAverageReward` — 100-episode moving average
- `Train/Epsilon` — exploration rate over time
- `Test/AverageReward` — evaluation performance
- `Test/SuccessRate` — % of successful landings

---

## Results

Videos of 5 test episodes are saved to [results/](./results/) after training.

A "successful landing" is defined as achieving reward ≥ 200 in a test episode.

---

## Strengths and Weaknesses

**Strengths:**
- Simple — no learning rate to tune
- Unbiased — uses actual returns, not estimates
- Naturally handles long-horizon credit assignment

**Weaknesses:**
- Must wait for episode to end before learning (slow)
- High variance — a single unlucky episode can swing Q-values
- Not suitable for non-episodic (continuing) tasks

---

## Next Step

Monte Carlo is offline — it waits for whole episodes. What if we could learn *while the episode is running*?

→ [Chapter 01b — Temporal Difference](../temporal_difference/)
