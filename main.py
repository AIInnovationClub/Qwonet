# Runs your agent on an ARC-AGI-3 game so you can watch it play and test it.
#
# Usage:
#   python main.py ls20                       # play, no rendering
#   python main.py ls20 --render terminal     # watch it in the terminal
#   python main.py ls20 --max-moves 200       # give it more moves
import argparse

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from agent import ALL_ACTIONS, Agent


def to_grid(frame):
    """The game's top visible layer as an unpadded 64x64 list of rows (0-15)."""
    return frame.frame[-1].tolist()


def scale_coord(value, size):
    """Map a normalized 0-1 coordinate onto a 0..size-1 pixel index."""
    return max(0, min(size - 1, round(value * (size - 1))))


def build_data(action, x, y, grid):
    """Build the per-action payload. Only the complex action needs coordinates."""
    if not action.is_complex() or x is None or y is None:
        return {}
    height = len(grid)
    width = len(grid[0])
    return {"x": scale_coord(x, width), "y": scale_coord(y, height)}


def main():
    p = argparse.ArgumentParser(description="Run an agent on an ARC-AGI-3 game.")
    p.add_argument("game", help="Game id, e.g. 'ls20' or 'ls20-9607627b'.")
    p.add_argument("--max-moves", type=int, default=100, help="Max moves to play.")
    p.add_argument(
        "--render",
        choices=["human", "terminal", "terminal-fast"],
        default=None,
        help="How to display the game. Omit for no rendering.",
    )
    args = p.parse_args()

    # NORMAL mode auto-downloads the game on first use (and caches it in
    # environment_files/), then runs it locally. Already-downloaded games are
    # reused without a network call.
    arcade = Arcade(operation_mode=OperationMode.NORMAL)
    env = arcade.make(
        args.game,
        scorecard_id=arcade.create_scorecard(),
        render_mode=args.render,
    )
    if env is None:
        games = sorted({e.game_id for e in arcade.get_environments()})
        raise SystemExit(
            f"Could not load game '{args.game}'. Available games: {', '.join(games) or 'none'}"
        )

    agent = Agent()
    frame = env.reset()
    states = [to_grid(frame)]

    for _ in range(args.max_moves):
        if frame.state in (GameState.WIN, GameState.GAME_OVER):
            break

        # mask[i] is True when ALL_ACTIONS[i] is available this turn.
        mask = [aid in frame.available_actions for aid in ALL_ACTIONS]

        action_id, x, y = agent.act(states, mask)
        action = GameAction.from_id(action_id)
        data = build_data(action, x, y, states[-1])

        frame = env.step(action, data=data)
        states.append(to_grid(frame))

    print(
        f"{args.game}: {frame.state.value} | "
        f"levels {frame.levels_completed}/{frame.win_levels} | "
        f"moves {len(states) - 1}"
    )


if __name__ == "__main__":
    main()
