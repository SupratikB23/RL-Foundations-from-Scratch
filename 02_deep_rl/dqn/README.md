# Chapter 02a — Deep Q-Network (DQN)

> The algorithm that proved deep learning could master Atari games from raw pixels.

**Paper:** [Playing Atari with Deep Reinforcement Learning — Mnih et al. (2013)](https://arxiv.org/abs/1312.5602)

---

## Environment: Atari Pong

Pong is a two-player game — the agent controls a paddle and tries to beat the opponent.

- **Input**: Raw pixel frames (preprocessed to 84×84 grayscale)
- **Actions**: 6 discrete (NOOP, FIRE, UP, RIGHT, DOWN, LEFT — Pong uses 3 meaningfully)
- **Reward**: +1 for scoring, -1 for conceding
- **Solved**: Mean reward > 19 over last 100 episodes

---

## Architecture

### CNN (lib/dqn_model.py)

```
Input: (4, 84, 84) — 4 stacked frames, normalized to [0, 1]
  ↓
Conv2d(32, kernel=8, stride=4) + ReLU
  ↓
Conv2d(64, kernel=4, stride=2) + ReLU
  ↓
Conv2d(64, kernel=3, stride=1) + ReLU
  ↓
Flatten
  ↓
Linear(3136 → 512) + ReLU
  ↓
Linear(512 → n_actions)   ← Q-values for each action
```

### Why Frame Stacking?
A single frame gives no velocity information. Stacking 4 consecutive frames lets the network infer motion (where the ball is going).

---

## Key Components

### Experience Replay Buffer
```python
class ExperienceBuffer:
    buffer = deque(maxlen=10_000)
    # Stores: (state, action, reward, done, next_state)
    # Training samples a random batch of 32
```

Breaking correlation between consecutive samples stabilizes training.

### Target Network
```python
net      # main network — updated every step
tgt_net  # target network — synced from net every 1,000 frames
```

The TD target uses `tgt_net` so it doesn't shift on every gradient update.

### Epsilon Decay
```python
epsilon = max(EPSILON_FINAL, EPSILON_START - frame / EPSILON_DECAY_LAST_FRAME)
# Linear decay from 1.0 → 0.01 over 150,000 frames
```

---

## Hyperparameters

| Parameter | Value | Role |
|-----------|-------|------|
| `GAMMA` | 0.99 | Discount factor |
| `BATCH_SIZE` | 32 | Mini-batch size for training |
| `REPLAY_SIZE` | 10,000 | Experience buffer capacity |
| `LEARNING_RATE` | 1e-4 | Adam optimizer LR |
| `SYNC_TARGET_FRAMES` | 1,000 | How often to sync target network |
| `REPLAY_START_SIZE` | 10,000 | Frames before training starts |
| `EPSILON_START` | 1.0 | Initial exploration |
| `EPSILON_FINAL` | 0.01 | Minimum exploration |
| `EPSILON_DECAY_LAST_FRAME` | 150,000 | Frames over which epsilon decays |
| `MEAN_REWARD_BOUND` | 19 | Reward threshold to stop training |

---

## Loss Function

Mean Squared Error between predicted and target Q-values:

```
target   = r + γ · max_a' Q_target(s', a')    (if not terminal)
target   = r                                   (if terminal)

loss = MSE(Q_main(s, a), target)
```

---

## Environment Wrappers (lib/wrappers.py)

| Wrapper | What it does |
|---------|-------------|
| `AtariWrapper` | Clips rewards to [-1, 1], adds no-op at start, grayscale |
| `ImageToPyTorch` | Converts HWC → CHW format for PyTorch |
| `BufferWrapper` | Stacks last 4 frames into one observation |

---

## How to Run

```bash
cd 02_deep_rl/dqn

# CPU (slow but works)
python dqn_pong.py

# GPU (recommended — much faster)
python dqn_pong.py --dev cuda
```

**Expected output:**
```
1000: done 3 games, reward -20.900, eps 0.99, speed 142.33 f/s
5000: done 15 games, reward -20.600, eps 0.97, speed 156.22 f/s
...
Best reward updated -21.000 -> -19.500
...
Solved in 1050234 frames!
```

> Training takes **hours** on CPU and **~4-8 hours** on a mid-range GPU. The model saves checkpoints as `.dat` files when it achieves new best rewards.

---

## Monitor with TensorBoard

```bash
tensorboard --logdir runs/
```

Metrics logged:
- `reward` — episode reward
- `reward_100` — 100-episode moving average
- `epsilon` — exploration rate
- `speed` — frames per second

---

## Files

| File | Description |
|------|-------------|
| [dqn_pong.py](./dqn_pong.py) | Main training script |
| [lib/dqn_model.py](./lib/dqn_model.py) | CNN architecture |
| [lib/wrappers.py](./lib/wrappers.py) | Atari environment wrappers |

---

## Strengths and Weaknesses

**Strengths:**
- Handles raw high-dimensional inputs (pixels)
- Generalizes across similar states (unlike tables)
- Proven to work on 49 Atari games

**Weaknesses:**
- Still learns *values*, not a direct policy
- Overestimates Q-values (fixed in Double DQN)
- Long training time
- Memory-intensive (replay buffer)

---

## Next Step

DQN is value-based — it learns Q(s,a) and picks the greedy action. What if we parameterize the policy directly as a neural network and optimize it with gradient ascent?

→ [Chapter 03 — Policy Gradients](../../03_policy_gradients/)
