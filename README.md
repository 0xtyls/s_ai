# Splendor AI — Comprehensive Guide
An open-source playground for building and comparing game-playing AIs on the classic board-game **Splendor**.

The repository ships with  
• a **rule-complete, deterministic game engine**  
• a **zero-training MCTS agent** (strong from the start)  
• a **self-play RL pipeline** (PPO by default, A2C also available)  
• colourful **CLI interfaces** for humans or AIs to play  
• batteries-included **test & demo scripts**.

Everything is pure Python 3.9 +, runs on Windows, macOS or Linux, CPU-only or with CUDA if available.

---

## Table of Contents
1. Quick Install  
2. Command-Line Cheat-Sheet  
3. Playing the Game  
   • Human vs AI   • AI vs AI  
4. Training Agents  
   • MCTS (no training)   • RL self-play  
5. Testing & Benchmarking  
6. Configuration Notes  
7. Troubleshooting  
8. Project Layout  
9. Contributing & License  

---

## 1. Quick Install
```bash
# Clone
git clone https://github.com/your-user/splendor-ai.git
cd splendor-ai

# Virtual-env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Dependencies (CPU PyTorch wheel)
pip install -r requirements.txt
```

# Optional: use GPU
# pip uninstall torch
# Follow https://pytorch.org for the matching CUDA wheel

# Having trouble building wheels on macOS?  
The default `pip install -r requirements.txt` occasionally fails when **pandas**
needs compilation (Clang error *“too many errors emitted”*).  
Use the **binary-only** fallback:
```bash
# 1. Core scientific stack – all wheels, no compilation
pip install numpy matplotlib tqdm rich pydantic

# 2. Pre-built pandas wheel
pip install pandas --only-binary=pandas

# 3. PyTorch CPU wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Optional TensorBoard
pip install tensorboard
```
After these steps the project imports fine and **all tests pass** (`python test_everything.py --engine --random`).

---

## 2. Command-Line Cheat-Sheet

| Task | One-liner |
|------|-----------|
| Quick sanity tests | `python test_everything.py --all` |
| Human vs MCTS | `python play_game.py --opponent mcts --mcts-iterations 500` |
| Human vs trained PPO | `python play_game.py --opponent rl --model models/my_agent.pt` |
| MCTS vs Random demo | `python demo_game.py --agent1 mcts --agent2 random --mcts-iterations 200` |
| PPO self-play training | `python simple_train.py --episodes 1000 --save-path models/ppo.pt` |
| Watch two PPO agents | `python demo_game.py --agent1 rl --agent2 rl --model models/ppo.pt` |
| Full system test suite | `pytest -q` |

---

## 3. Playing the Game

### 3.1 Human vs AI
```bash
# Play first against MCTS (500 simulations per move)
python play_game.py --opponent mcts --mcts-iterations 500 --first
```
Flags  
`--opponent` `random | mcts | rl | hybrid`  
`--model`     Path to saved RL weights (RL / hybrid)  
`--mcts-iterations` Number of MCTS rollouts.

### 3.2 AI vs AI (headless demo)
```bash
# MCTS vs Random
python demo_game.py --agent1 mcts --agent2 random --mcts-iterations 300

# PPO vs MCTS
python demo_game.py --agent1 rl --agent2 mcts --model models/ppo.pt --mcts-iterations 200
```
Use `--delay 0.1` to slow display or `--max-turns 200` to limit long games.

---

## 4. Training Agents

### 4.1 MCTS
MCTS is search-only: **no training required**. Adjust strength via `--mcts-iterations`.  
500–1 000 rollouts is plenty on a laptop.

### 4.2 RL (Self-Play PPO)
```bash
# 1 000 self-play episodes, batch 64, two-layer MLP 128-128
python simple_train.py \
    --episodes 1000 \
    --batch-size 64 \
    --hidden-size 128 \
    --lr 3e-4 \
    --save-path models/ppo.pt \
    --verbose
```
The script automatically spins up:
* game environment (2 players: PPO vs Random baseline)  
* experience collector (GAE λ 0.95, γ 0.99)  
* PPO update every episode  

Training metrics appear in stdout; extend the script for TensorBoard logging if desired.

#### Continue Training
```bash
python simple_train.py --episodes 500 --save-path models/ppo.pt --model models/ppo.pt
```

#### A2C
```bash
python simple_train.py --agent-type a2c --episodes 1000 --save-path models/a2c.pt
```

---

## 5. Testing & Benchmarking

### 5.1 Unit / Integration Tests
```bash
pytest                 # full suite (~30 s CPU)
python test_rl.py      # RL focussed quick checks
```

### 5.2 System Smoke Test
```bash
python test_everything.py --all  # runs engine, UI, agents, simulation
```

### 5.3 Benchmark MCTS strength
```bash
python demo_game.py --agent1 mcts --agent2 random \
    --mcts-iterations 1000 --max-turns 200 --delay 0.0
```

---

## 6. Configuration Notes

| File | Purpose |
|------|---------|
| `splendor_ai/mcts/config.py` | exploration weight, rollout count |
| `splendor_ai/rl/config.py`  | PPO/A2C hyper-parameters, network sizes |
| `simple_train.py`           | quick-start trainer, 50-dim state encoder |
| `play_game.py`              | CLI flags, colour codes |

Tune anything; everything is Python dataclasses.

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | State encoder mismatch. Ensure `state_dim` in `PPOAgent` equals 50 (from `encode_state`). |
| No valid actions error | Rare edge case when hand is full. Reset game or increase `max-turns`. |
| Slow MCTS | Reduce `--mcts-iterations` or run on PyPy. |
| GPU not used | Verify `torch.cuda.is_available()`; install the CUDA wheel. |
| Pandas wheel fails to compile (macOS clang) | Follow the binary-only fallback in *Quick Install* section. |

---

## 8. Project Layout
```
splendor-ai/
├─ splendor_ai/
│  ├─ core/         # Game engine, rules, constants
│  ├─ mcts/         # Node, search, config, agent
│  ├─ rl/           # Models, PPO/A2C agents, training util
│  ├─ play_game.py  # Human vs AI interface
│  ├─ demo_game.py  # AI vs AI showcase
│  └─ __init__.py
├─ simple_train.py  # Minimal PPO/A2C self-play trainer
├─ quick_train.py   # Random agents statistics demo
├─ test_everything.py # Comprehensive system test
├─ tests/           # pytest unit tests
├─ requirements.txt
└─ README.md
```

---

## 9. Contributing & License
Pull requests welcome! Please:
1. Fork & branch
2. Add unit tests
3. Run `pytest` & `python test_everything.py --all`
4. Submit PR with clear description

Licensed under **MIT** — see `LICENSE` for full text.

Happy splendor-coding! 🎉
