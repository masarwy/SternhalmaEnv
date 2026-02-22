# SternhalmaEnv

PettingZoo AEC environment for Sternhalma (Chinese Checkers) with configurable player count and board diagonal size.

## Features
- Multi-agent environment built on PettingZoo `AECEnv`
- Supported player counts: `2`, `3`, `4`, `6`
- Supported render modes: `None`, `"ansi"`, `"human"`, `"rgb_array"`
- Variable-length move actions (single-step and multi-hop paths)
- Markov-friendly observation includes both board and current turn owner
- Optional generic wrapper with fixed `Discrete(N)` actions and `action_mask`

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/masarwy/SternhalmaEnv.git
```

> Creating a virtual environment is optional but recommended.

---

## Development Setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/masarwy/SternhalmaEnv.git
cd SternhalmaEnv

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

This installs the package in editable mode so changes to the source code are reflected immediately.

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
    reward_mode="potential_shaped",  # "sparse" | "dense" | "potential_shaped"
    gamma=0.95,  # used when reward_mode="potential_shaped"
)
env.reset()

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        break
    valid_moves = info.get("valid_moves", [])
    action = valid_moves[0] if valid_moves else []  # explicit no-op
    env.step(action)

env.close()
```

If `reward_mode` is `"potential_shaped"`, you can set `gamma` in `[0, 1]` to control shaping strength (default: `1.0`).

## Action And Observation
- Observation (`observe(agent)`): dict with:
  - `board`: matrix encoded as `np.int8` where:
    - empty playable: `0`
    - player pieces: `1..6`
    - non-playable filler: `-1`
    - blank: `-2`
  - `current_player`: index of `agent_selection` in `self.agents`
- Action: list of `(row, col)` tuples in zero-based env coordinates.
  - Empty list `[]` is explicit no-op/skip.
  - Length `2` means a simple move or one jump.
  - Length `>2` means a jump sequence.
- Environment only accepts actions present in `info["valid_moves"]`.

## Discrete Action Wrapper
Use `sternhalma_v0.discrete_action_env(...)` for fixed-size discrete actions:

```python
import sternhalma_v0

env = sternhalma_v0.discrete_action_env(
    num_players=2,
    board_diagonal=5,
    render_mode=None,
    max_actions=256,
)
env.reset()
```

Wrapper behavior:
- Action space: `Discrete(max_actions)`
- Action meaning: action `i` maps to the `i`-th currently valid move
- Observation:
  - `observations`: base Sternhalma observation (`board`, `current_player`)
  - `action_mask`: `MultiBinary(max_actions)` marking valid indices for current agent

## Current Rule Semantics
- A player wins when they have majority control of their target home triangle.
- Invalid actions are penalized (`-1.0`) and turn does not advance.
- If no move exists for the current player, a no-op/skip is used automatically.
- Reward modes:
  - `sparse`: `+1` when a piece enters home triangle from outside, else `0`
  - `dense`: per-move distance progress toward home (`start_distance - final_distance`)
  - `potential_shaped`: `sparse + (gamma * phi(s') - phi(s))`, where `phi(s) = -distance_to_home`

## Run Tests
```bash
./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

## Project Layout
- `sternhalma/env/sternhalma.py`: PettingZoo environment
- `sternhalma/env/discrete_action_wrapper.py`: fixed discrete action wrapper + action masks
- `sternhalma/utils/board.py`: move generation, validation, winner checks
- `sternhalma/utils/grid.py`: board geometry and triangle utilities
- `sternhalma/utils/types.py`: custom action space and no-op wrapper
- `tests/`: board/env/wrapper behavior tests
