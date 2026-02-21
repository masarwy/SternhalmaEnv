# SternhalmaEnv

PettingZoo AEC environment for Sternhalma (Chinese Checkers) with configurable player count and board diagonal size.

## Features
- Multi-agent environment built on PettingZoo `AECEnv`
- Supported player counts: `2`, `3`, `4`, `6`
- Supported render modes: `None`, `"ansi"`, `"human"`, `"rgb_array"`
- Variable-length move actions (single-step and multi-hop paths)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start
```bash
python3 usage_example.py
```

Programmatic usage:
```python
import sternhalma_v0

env = sternhalma_v0.env(
    num_players=2,
    board_diagonal=5,
    render_mode=None,
)
env.reset()

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        break
    valid_moves = info.get("valid_moves", [])
    action = valid_moves[0] if valid_moves else None
    env.step(action)

env.close()
```

## Action And Observation
- Observation: board matrix encoded as `np.int8` where:
  - empty playable: `0`
  - player pieces: `1..6`
  - non-playable filler: `-1`
  - blank: `-2`
- Action: list of `(row, col)` tuples in zero-based env coordinates.
  - Length `2` means a simple move or one jump.
  - Length `>2` means a jump sequence.

## Current Rule Semantics
- A player wins when they have majority control of their target home triangle.
- Invalid actions are penalized (`-1.0`) and turn does not advance.
- If no move exists for the current player, a no-op/skip is used.

## Run Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

## Project Layout
- `sternhalma/env/sternhalma.py`: PettingZoo environment
- `sternhalma/utils/board.py`: move generation, validation, winner checks
- `sternhalma/utils/grid.py`: board geometry and triangle utilities
- `sternhalma/utils/types.py`: custom action space and no-op wrapper
- `tests/`: baseline behavior tests
