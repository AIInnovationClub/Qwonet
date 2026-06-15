# Basic Agent Schema
#
# A starter template for building your own ARC-AGI-3 agent. The only thing you
# have to write is `Agent.act`. The version below uses two tiny MLPs as a
# worked example - feel free to replace them with anything you like.

import torch
from torch import nn

# RESET (action 0) is handled by the runner; the agent cannot play it.
ALL_ACTIONS = [1, 2, 3, 4, 5, 6, 7]

COMPLEX_ACTION = 6           # the only action that needs coordinates
GRID_CELLS = 64 * 64         # each state is at most a 64x64 grid, flattened to 4096 inputs


class ActionNet(nn.Module):
    """Tiny MLP that scores each action from a single flattened grid."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID_CELLS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, len(ALL_ACTIONS)),  # one logit per action
        )

    def forward(self, x):
        return self.net(x)


class CoordNet(nn.Module):
    """Tiny MLP that predicts an (x, y) click target for the complex action."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID_CELLS, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # x, y, squashed to 0-1 in `act`
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    """Plays a game from the history of its grid states.

    `act` is called once per turn and receives:
      * `states` - list of every grid state since the start. Each state is an
        unpadded 64x64 grid (a list of rows, cell values 0-15).
      * `mask`   - list aligned to ALL_ACTIONS: mask[i] is True when
        ALL_ACTIONS[i] is available this turn, False when it's disabled.

    `act` must return `(action, x, y)`:
      * `action` - an id from ALL_ACTIONS.
      * `x, y`   - normalized click target in 0-1 for the complex action, or
        None for every other action. The runner rescales 0-1 to grid pixels.

    Override `act` with your own logic - the body below is just an example.
    """

    def __init__(self):
        self.action_net = ActionNet()
        self.coord_net = CoordNet()

    def act(self, states, mask):
        # This simple example only looks at the most recent state.
        grid = torch.tensor(states[-1], dtype=torch.float32).flatten()

        # Score the actions, then forbid the disabled ones before choosing.
        logits = self.action_net(grid)
        forbidden = ~torch.tensor(mask, dtype=torch.bool)
        logits = logits.masked_fill(forbidden, float("-inf"))
        action = ALL_ACTIONS[int(logits.argmax())]

        # Only the complex action needs coordinates.
        if action == COMPLEX_ACTION:
            x, y = torch.sigmoid(self.coord_net(grid)).tolist()  # each in 0-1
        else:
            x, y = None, None

        return action, x, y
