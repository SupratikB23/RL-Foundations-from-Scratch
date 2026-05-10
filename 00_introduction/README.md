# Chapter 00 — Introduction to Reinforcement Learning

> **Start here.** Before writing any algorithm, you need to understand what an RL environment is and how to interact with it.

---

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent receives a **reward** signal after each action and learns to maximize the total reward over time.

The core loop looks like this:

```
Agent → takes action → Environment
       ← observes state + reward ←
```

Key terms:

| Term | Meaning |
|------|---------|
| **Agent** | The learner / decision maker |
| **Environment** | The world the agent lives in |
| **State (s)** | What the agent currently observes |
| **Action (a)** | What the agent chooses to do |
| **Reward (r)** | Feedback signal — how good the action was |
| **Episode** | One complete run from start to terminal state |
| **Policy (π)** | The agent's strategy — maps states to actions |

---

## Gymnasium

We use [Gymnasium](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym) to simulate environments. Every environment follows the same interface:

```python
env = gym.make("EnvironmentName")
state, info = env.reset()       # Start a new episode
action = env.action_space.sample()   # Random action
next_state, reward, done, truncated, info = env.step(action)
env.close()
```

### Action Space
What actions the agent can take. Can be:
- **Discrete** — a fixed number of choices (e.g., 0, 1, 2, 3 for LunarLander)
- **Continuous** — real-valued (e.g., steering angle)

### Observation Space
What the agent sees. LunarLander gives 8 numbers:
- x, y position
- x, y velocity
- angle, angular velocity
- left leg contact, right leg contact

---

## Environment: LunarLander-v3

The Lunar Lander must land safely between two flags using thrusters.

- **State**: 8 continuous values
- **Actions**: 4 discrete (do nothing, fire left, fire main, fire right)
- **Reward**: +100-140 for landing, -100 for crash, -0.3/frame for firing main engine
- **Solved**: Average reward ≥ 200 over 100 episodes

---

## Files

| File | Description |
|------|-------------|
| [env_basics.py](./env_basics.py) | Sample from action and observation spaces |
| [env_interaction.py](./env_interaction.py) | Step through an environment with random actions |

---

## How to Run

```bash
# See what actions and observations look like
python env_basics.py

# Watch a random agent interact with LunarLander (opens a window)
python env_interaction.py
```

> The random agent has no learning — it just takes random actions. This is your baseline before any RL algorithm.

---

## What to Notice

When you run `env_interaction.py`, watch how quickly the lander crashes without any learning. This motivates why we need algorithms — even for simple environments, random behaviour fails completely.

---

## Next Step

→ [Chapter 01 — Tabular Methods](../01_tabular_methods/)
