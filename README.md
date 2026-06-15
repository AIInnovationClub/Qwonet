# Qwonet

A small, friendly starter kit for building and testing your own agents for
[ARC-AGI-3](https://three.arcprize.org) games.

You write one function — `Agent.act` — and `main.py` handles everything else:
loading the game, feeding your agent the board, applying its moves, and
reporting how it did.

---

## What's in here

| File | What it does |
|------|--------------|
| `agent.py` | Your agent. Edit `Agent.act` — this is the only file you need to change. |
| `main.py` | The runner. Loads a game, plays your agent, prints the result. |
| `requirements.txt` | Python dependencies. |
| `environment_files/` | Where games get downloaded and cached (git-ignored). |

---

## Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

(If you prefer to keep things isolated, a virtual environment works too — but
it's not required.)

## Run it

Just run it — the game downloads automatically the first time (you'll need an
internet connection) and is cached in `environment_files/` for next time. No
manual setup, no API key required.

`ls20` is a good first game:

```bash
python main.py ls20                     # play, no rendering
python main.py ls20 --render terminal   # watch it play in your terminal
python main.py ls20 --max-moves 200     # give it more moves
```

When it finishes you'll see a line like:

```
ls20: NOT_FINISHED | levels 0/7 | moves 100
```

- **state** — `WIN`, `GAME_OVER`, or `NOT_FINISHED` (ran out of moves).
- **levels** — levels solved / levels needed to win.
- **moves** — how many actions were played.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `game` (positional) | — | Game id, e.g. `ls20` or `ls20-9607627b`. |
| `--max-moves` | `100` | Stop after this many moves. |
| `--render` | off | `human`, `terminal`, or `terminal-fast`. |

---

## How your agent works

The whole job is to implement `Agent.act`. It's called once per turn:

```python
def act(self, states, mask):
    ...
    return action, x, y
```

### What you're given

- **`states`** — the full history of board states, oldest first. Each state is
  an unpadded **64×64 grid**: a list of 64 rows, each a list of 64 ints in
  **0–15** (the cell colors). The current board is `states[-1]`.
- **`mask`** — a list aligned to `ALL_ACTIONS`. `mask[i]` is `True` when
  `ALL_ACTIONS[i]` is allowed this turn, `False` when it's disabled. The action
  space is always the same size; only the mask changes, so a fixed-output model
  can just set the disabled logits to `-inf` before `argmax`.

### What you return

`(action, x, y)`:

- **`action`** — an id from `ALL_ACTIONS`.
- **`x, y`** — a click target in **0–1** (normalized) for the complex action,
  or `None` for every other action. The runner rescales 0–1 to grid pixels for
  you, so you never deal with raw coordinates.

### The actions

`ALL_ACTIONS = [1, 2, 3, 4, 5, 6, 7]`

Only two actions have a fixed, universal meaning:

| Action | Meaning |
|--------|---------|
| 0 (`RESET`) | Restarts the game/level. Handled by the runner — your agent never plays it. |
| 6 | **Complex action** — the only one that needs `x, y` coordinates. |

Actions **1–5 and 7 are simple actions whose effect varies from game to game**
(e.g. move, interact, rotate, undo, …). We deliberately don't assign them fixed
meanings: working out what each one does from the board is part of the puzzle.
The same action id can do completely different things in two different games.

---

## Writing your own agent

Replace the body of `act` with whatever you like. The simplest possible agent:

```python
import random

ALL_ACTIONS = [1, 2, 3, 4, 5, 6, 7]

COMPLEX_ACTION = 6           # the only action that needs coordinates
GRID_CELLS = 64 * 64         # each state is at most a 64x64 grid, flattened to 4096 inputs

class Agent:
    def act(self, states, mask):
        valid = [a for a, ok in zip(ALL_ACTIONS, mask) if ok]
        action = random.choice(valid)
        if action == COMPLEX_ACTION:
            # Complex action needs a target cell — pick one at random (0-1).
            return action, random.random(), random.random()
        return action, None, None
```

The version shipped in `agent.py` is a worked example using two tiny PyTorch
MLPs: one scores the actions, the other predicts the click coordinates for the
complex action. Use it as a starting point or throw it away — `main.py` only
cares that `act` returns `(action, x, y)`.
