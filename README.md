# Splendor AI

An open-source playground for building and comparing game-playing AIs on the classic board game **Splendor**.  
The project starts with a complete rule implementation, a Monte-Carlo Tree Search (MCTS) agent that can play immediately, and a Reinforcement-Learning (RL) pipeline that lets the AI learn by self-play. Everything is written in **pure Python 3.9+** and runs on macOS or Windows with or without a GPU.

---

## 1. Project Goals

* Provide a clean, well-tested Python implementation of Splendor.
* Offer two complementary AI approaches:
  * **MCTS** – strong out-of-the-box play, no training required.
  * **RL (Proximal Policy Optimization)** – self-play training that improves over time.
* Serve as a template for adding other board / card games with minimal code changes.
* Be easy to install, run and extend on everyday hardware.

---

## 2. Splendor in 90 Seconds

* 2-4 players compete to reach **15 Prestige points** first.
* On your turn you may **take gems**, **reserve a development card**, or **purchase a card**.
* Development cards cost colored gems, grant permanent gem discounts, and award Prestige points.
* Acquire enough bonuses and nobles will visit for extra points.
* Perfect-information, no randomness after initial card draw – ideal for tree search and RL.

(For full rules see the `docs/` folder or the official rulebook.)

---

## 3. AI Approaches Implemented

| Approach | Strengths | When to Use |
|----------|-----------|-------------|
| **Pure MCTS** | Fast CPU-only, no training. Good baseline and explainable decisions. | Quick simulations, limited hardware. |
| **Self-play RL (PPO)** | Learns long-term strategy, scales to larger games, can combine with MCTS (AlphaZero style). | Research, stronger play, extending to complex games. |

Both agents share the same game engine and state representation, so you can switch or combine them with one flag.

---

## 4. Installation

### Prerequisites
* Python ≥ 3.9
* pip or Poetry
* Optional GPU: CUDA-enabled GPU & appropriate PyTorch wheel

### Quick Start (macOS & Windows)

```bash
# 1. Clone
git clone https://github.com/your-user/splendor-ai.git
cd splendor-ai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

PyTorch wheels in `requirements.txt` target the CPU.  
To use a GPU, uninstall the cpu wheel and follow the [official PyTorch install guide](https://pytorch.org) for your CUDA version.

---

## 5. Usage Examples

### Play MCTS vs. MCTS (headless)

```bash
python -m splendor_ai.play --agent1 mcts --agent2 mcts --games 100
```

### Train an RL Agent from Scratch

```bash
python -m splendor_ai.train \
    --algo ppo \
    --total-steps 2_000_000 \
    --save-dir runs/ppo_experiment
```

TensorBoard logs:

```bash
tensorboard --logdir runs/ppo_experiment
```

### Human vs. AI (CLI)

```bash
python -m splendor_ai.play --agent1 human --agent2 mcts
```

---

## 6. Project Structure

```
splendor-ai/
├─ splendor_ai/
│  ├─ core/          # Game engine & rules
│  ├─ mcts/          # Pure Monte-Carlo Tree Search agent
│  ├─ rl/            # PPO policy network, training loop
│  ├─ envs/          # Gym-style wrapper for RL
│  ├─ play.py        # Command-line playing script
│  └─ train.py       # Self-play training entry point
├─ tests/            # pytest unit & integration tests
├─ docs/             # Extended rule explanations
├─ requirements.txt
└─ README.md
```

---

## 7. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4-core 2 GHz | 6-core 3 GHz+ |
| RAM | 4 GB | 8 GB+ |
| GPU | *not required* | NVIDIA CUDA 11+ (for faster RL) |

*Splendor is lightweight – training on CPU is feasible albeit slower. The code automatically detects GPU and falls back to CPU.*

---

## 8. Roadmap & Future Games

1. **Performance tuning** – vectorised game engine, JIT compilation.
2. **AlphaZero-style agent** – MCTS guided by the neural network.
3. **GUI** – web-based visualisation & human play.
4. **Additional Games** – Carcassonne, Ticket to Ride, Catan. Each added game re-uses:
   * the `core/` turn system
   * common MCTS utilities
   * generic RL training loop

Contributions welcome – open an issue or pull request!

---

## 9. License

MIT License – see `LICENSE` file for details.
