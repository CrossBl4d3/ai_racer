# AI Racer 🏎️

A top-down racing game written in Python with Pygame, and an AI that teaches itself to drive it,
using a neural network written **from scratch in NumPy** and trained with **Deep Q-Learning (DQN)**.

The AI starts out knowing nothing (not even that it should press the accelerator) and, after about
an hour of simulated driving, laps faster than a human and finishes races on tracks it has never seen.

Built by Humaam Ghazal, step by step, with Claude as a guide.

---

## Highlights

| | |
|---|---|
| Human best lap on the oval | 5.68 s |
| Best AI lap on the oval | **4.95 s** |
| Champion (`base1_pick`) on all 10 hand-made tracks | **197 / 200** races finished |
| Champion on the unseen exam track (Technical + mirror) | **40 / 40** |
| Fastest AI (`rand1_pick`), trained only on random tracks | 2.5 s faster than the champion on Technical, a track it never saw |
| Size of the neural network | 18,693 parameters |

---

## Quick start

You need **Python 3.11 or newer** (built with 3.13).

```bash
git clone https://github.com/CrossBl4d3/ai_racer.git
cd ai_racer
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
  (if it's blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again)
- **macOS / Linux:** `source .venv/bin/activate`

Then install the packages:

```bash
pip install -r requirements.txt
```

Always run commands from the `ai_racer` folder, with `(.venv)` showing in your terminal.

---

## Things to try

### Drive it yourself
```bash
python -m game.race                    # the oval
python -m game.race --track hairpin    # another track
python -m game.race --list             # every track
python -m game.race --track random_42  # a randomly generated track (any number works)
```
Controls: **arrows** to drive (hold Down to reverse) · **C** show checkpoints · **H** show hitbox · **R** restart · **Esc** quit

### Watch the AI drive
```bash
python watch.py --model models/base1_pick.npz --track technical
```
You'll see its 7 sensor rays and a bar chart of how good it thinks each action is (its Q-values).
**Space** pause · **F** fast-forward · **R** restart · **T** next track

### Race against the AI
```bash
python watch.py --race --model models/base1_pick.npz --track bean
```

### Watch two AIs race each other
```bash
python watch.py --duel models/base1_pick.npz models/rand1_pick.npz --track wiggle
```
**N** gives a new random start, and **F** fast-forwards.

---

## The tracks

| Track | Width | What it tests |
|---|---|---|
| `oval` | 110 px | The original: one long left turn |
| `bean` | 100 px | A dip: left, **right**, left |
| `hairpin` | 90 px | A long "tongue" ending in a very tight U-turn |
| `wiggle` | 100 px | Constant left-right-left waves |
| `technical` | 80 px | Narrow chicanes and S-bends. **The exam: never used for training** |

Every track also has a `_mirror` version, driven the other way round, and `random_<number>` gives
an endless supply of generated tracks. The same number always gives the same track.

```bash
python -m sandbox.preview_tracks          # picture of every hand-made track -> logs/tracks_preview.png
python -m sandbox.preview_random --stats 300   # 12 random tracks + statistics
```

---

## Train your own AI

```bash
python train.py --name mine --tracks train --backend numpy     # the 8 training tracks (~20-40 min)
python pick_best.py mine                                        # choose the best snapshot
python compare.py models/mine_pick.npz models/base1_pick.npz --tracks handmade --races 20
```

Other ways to train:

```bash
python train.py --name oval1                                   # the oval only (quick)
python train.py --name rand2 --tracks random --backend numpy   # random tracks only
python train.py --name mix2 --tracks mix --backend numpy       # training tracks + random tracks
python pick_best.py rand2 --tracks val                         # random runs choose on validation tracks
```

- `--backend numpy` uses the network written from scratch, and `--backend torch` uses PyTorch.
- `--seed 1` gives a different random start. **Results vary a lot between seeds**, so compare several.
- On Windows, running `$env:OMP_NUM_THREADS=1` first can make training noticeably faster.
- Training prints test drives every 10,000 steps and saves a snapshot each time.
  Afterwards, `pick_best.py` tests all of them and keeps the one that finishes the most races fastest,
  because the *last* brain is often not the best one.

---

## How it works

**1. The game** (`game/`): car physics, tracks built from smooth curves (Catmull-Rom splines),
walls, timing gates and collisions, all in plain Python and maths.

**2. The environment** (`env/racing_env.py`) turns the game into numbers for the AI:
- **What it senses (11 numbers):** 7 distance rays (−90° to +90°), its speed, and the direction to
  the next gate and to gates 4 and 8 ahead (its "look-ahead").
- **What it can do (5 actions):** accelerate, accelerate + left, accelerate + right, brake, coast.
  It decides every 4 frames.
- **Rewards:** +1 per gate, +10 per lap, −10 for hitting a wall, −0.01 per decision (hurry up!).

**3. The neural network** (`agent/numpy_network.py`): 11 inputs → 128 → 128 → 5 outputs, one value per
action. Forward pass, backpropagation and the Adam optimiser are all written by hand in NumPy, and a
test proves it matches an identical PyTorch network (`agent/network.py`).

**4. The learning algorithm, DQN** (`agent/dqn_agent.py`): the network learns to predict how much
future reward each action leads to (its *Q-value*), using:
- a **replay buffer** of 100,000 past experiences, studied in random batches of 64;
- **epsilon-greedy exploration**: random actions at first, fading from 100% to 5%;
- a **target network** and **Double DQN**, to keep learning stable.

**5. Tricks that made it generalise:**
- **Look-ahead inputs**, so it sees a sharp bend coming in time to set up for it.
- **Mirror memories:** every experience is also stored flipped left-to-right, so a left-hand
  hairpin teaches it the right-hand one too.
- **Many tracks and random tracks**, so it learns *driving* rather than memorising one layout.
- **Fair starts:** random start positions that can't be won are replaced.
- **Snapshots + `pick_best.py`:** the brain is chosen on tracks it is *not* examined on.

---

## Project layout

```
ai_racer/
├── config.py            every setting in one place
├── train.py             training
├── pick_best.py         choose the best snapshot after training
├── watch.py             watch / race / duel
├── compare.py           tournaments between models
├── game/                the racing game: car, tracks, race rules, drawing
├── env/                 the game as an RL environment (observations, actions, rewards)
├── agent/               the neural networks, replay buffer and DQN agent
├── sandbox/             tests and preview tools
├── assets/              images (Kenney Racing Pack)
├── models/              trained brains (.npz = NumPy, .pt = PyTorch)
└── logs/                training logs and charts
```

---

## Tests

```bash
python -m sandbox.test_network      # NumPy network matches PyTorch
python -m sandbox.test_agent        # DQN learns simple known answers
python -m sandbox.test_mirror       # the mirror trick is exactly right
python -m sandbox.test_fair_starts  # starts are winnable
python -m sandbox.test_env bench    # how fast the environment runs
```

---

## Lessons learned

- **An AI trained on one track only learns that track.** The oval-only champion finished 0 races on Bean.
- **Give the AI the information it needs.** Adding look-ahead took exam finishes from 2% to 93%.
- **Check experiments with an ablation:** remove one change at a time to see which one really works.
- **Seeds matter.** The same code can produce a great model or a mediocre one, so judge recipes on several runs.
- **The last brain isn't always the best.** DQN swings between better and worse as it trains,
  so keep snapshots and choose carefully.
- **Measure honestly.** Average lap time over finished races only hides every crash (survivorship bias).

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'game'`:** run from the `ai_racer` folder and use
  `python -m game.race`, not `python game/race.py`.
- **`Model not found`:** the model file isn't in `models/`. Train one, or pick an existing file.
- **`IndentationError`:** Python is strict about spaces. Use 4 spaces per level and never mix tabs.
- **Training seems slow:** close other programs, and try `$env:OMP_NUM_THREADS=1` (Windows) or
  `export OMP_NUM_THREADS=1` (macOS/Linux) before training.

---

## Credits

- Car, tree and grass graphics: [Kenney Racing Pack](https://kenney.nl/assets/racing-pack) (CC0).
- Built by **Humaam Ghazal** with guidance from Claude (Anthropic).

📖 **Full project write-up** (every phase, decision, bug and result, with diagrams): [read it here](https://claude.ai/artifact/7W4VtzHq6igG5FUSbyxSh6)
