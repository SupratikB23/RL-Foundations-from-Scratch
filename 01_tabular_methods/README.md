# Chapter 01 — Tabular Methods

> The simplest class of RL algorithms: we store Q-values in a **table** indexed by (state, action) pairs.

---

## What is a Q-Table?

A Q-table stores the expected future reward for taking action `a` in state `s`:

```
Q(s, a) = "How good is it to take action a when in state s?"
```

The agent's policy is simple: in any state, pick the action with the highest Q-value.

```python
best_action = argmax(Q[state])
```

Over many episodes, the agent updates these values to get closer to the true expected returns.

---

## The Problem: Continuous States

LunarLander has a continuous 8-dimensional state space — infinitely many possible states. A table can't store all of them.

**Solution: State Discretization**

We divide each dimension into bins and map the continuous state to a single integer index:

```python
NUM_BINS = (8, 8, 8, 8, 8, 8, 2, 2)   # 65,536 total states
# Each dimension gets 8 buckets (2 for binary legs)
```

This is a lossy compression — nearby states map to the same bin — but it's good enough for LunarLander.

---

## Exploration vs Exploitation

Both algorithms use **epsilon-greedy** exploration:

- With probability `ε` → take a **random** action (explore)
- With probability `1-ε` → take the **best known** action (exploit)

`ε` starts at 1.0 (fully random) and decays toward 0.05 (mostly greedy) over training.

---

## Chapters

| Sub-chapter | Algorithm | Key Idea |
|-------------|-----------|----------|
| [Monte Carlo](./monte_carlo/) | MC Q-Learning | Learn from **complete episodes** |
| [Temporal Difference](./temporal_difference/) | Q-Learning (TD-0) | Learn from **each step** |

---

## Monte Carlo vs Temporal Difference

| Property | Monte Carlo | Temporal Difference |
|----------|-------------|---------------------|
| When to update | End of episode | After every step |
| What it uses | Actual returns | Bootstrapped estimate |
| Requires learning rate? | No (averages returns) | Yes (ALPHA) |
| Variance | Higher | Lower |
| Bias | Zero | Some (bootstrapping) |
| Episodes needed | 30,000 | 50,000 |
| Sample efficiency | Lower | Higher |

Both converge to the same optimal policy — the difference is *how fast* and *how stable* the learning is.

---

## Next Step

After mastering tabular methods, the next challenge is environments where the state space is too large for any table (e.g., raw pixels from Atari games).

→ [Chapter 02 — Deep RL / DQN](../02_deep_rl/)
