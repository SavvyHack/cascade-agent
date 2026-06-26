# Cascade — Adversarial Search Agent

A competitive game-playing agent for **Cascade**, a two-player, perfect-information
board game played on an 8×8 grid. The agent selects moves under a strict per-game
time budget using iterative-deepening alpha-beta search, and ships with a suite of
benchmark opponents for local evaluation.

> Full game mechanics are described in the accompanying rules document, and the
> internals of the search and evaluation are covered in [`report.pdf`](report.pdf).
> This README focuses on what the project *is*, how it is organised, and how to run it.

---

## Objective

The project goal is to build an agent that plays Cascade as strongly as possible
within the constraints imposed by the game's referee: a fixed CPU-time allowance
for the entire game and a per-process memory ceiling. A submission is a Python
package exposing an `Agent` class with three methods the referee drives:

- `__init__(color, **referee)` — set up internal state for the assigned side.
- `action(**referee)` — return the chosen action for the current turn.
- `update(color, action, **referee)` — observe an applied action (either player's)
  and keep the internal board in sync with the referee.

Success is measured by win rate against opponents of increasing strength while
never exceeding the time or memory limits — i.e. the engine must spend its budget
where it matters and degrade gracefully when time is short.

---

## Repository structure

```
.
├── cascade_engine/          # Primary agent (tournament entry point)
├── cascade_engine_v2/       # Tuned variant with additional search refinements
├── benchmarks/              # Opponents used for local testing
│   ├── random/              #   uniform-random legal move
│   ├── greedy/              #   1-ply material-maximising lookahead
│   ├── minimax_fixed_depth/ #   fixed-depth minimax with a tactical shortcut
│   ├── iddfs_tt/            #   iterative-deepening alpha-beta + transposition table
│   └── zobrist_engine/      #   alpha-beta with a Zobrist-hashed transposition table
├── referee/                 # Provided game driver (do not modify)
├── report.pdf               # Technical report: design, evaluation, results
└── team.py                  # Team / submission metadata
```

Each directory under `cascade_engine*` and `benchmarks/` is a self-contained Python
package exposing an `Agent` class, so any one can be pitted against any other.

### The engine

Two variants of the same core are provided. **`cascade_engine`** is the primary
agent and the intended entry point. **`cascade_engine_v2`** keeps the same
architecture and adds further search refinements; it is included so the two can be
played head-to-head to measure the marginal value of those additions. Both maintain
their own internal board representation rather than re-querying the referee, run a
positional heuristic during the placement phase, and switch to time-bounded search
during play. The algorithmic details, evaluation features, and tuning rationale are
documented in `report.pdf`.

### The benchmarks

The `benchmarks/` packages exist purely to provide a measurable ladder of opponents
during development, from a random baseline up to alternative search engines. They
are not part of the competitive submission. See the attribution note at the bottom.

---

## Requirements

- **Python 3.12+**
- No third-party packages are required to run local games — the engine and referee
  rely only on the standard library.
- The optional remote-play server mode additionally requires `websockets`:

  ```bash
  pip install websockets
  ```

A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

---

## Running a game

Games are run through the provided referee. The two positional arguments are
**package specifications** for the RED and BLUE players — the dotted import path of
a package containing an `Agent` class:

```bash
python -m referee <RED> <BLUE>
```

### Examples

Play the primary engine (RED) against the greedy benchmark (BLUE):

```bash
python -m referee cascade_engine benchmarks.greedy
```

Pit the two engine variants against each other:

```bash
python -m referee cascade_engine cascade_engine_v2
```

Sanity-check against the random baseline:

```bash
python -m referee cascade_engine benchmarks.random
```

If a package exposes its agent class under a different name, append it after a colon:

```bash
python -m referee cascade_engine benchmarks.iddfs_tt:Agent
```

### Useful options

Run `python -m referee --help` for the full list. The most relevant:

| Option | Effect |
|---|---|
| `-t <seconds>` | CPU-time limit per agent for the whole game. |
| `-s <MB>` | Memory limit per agent. |
| `-w <seconds>` | Wait time between turns (`-w 0` runs as fast as possible). |
| `-v <0–3>` | Verbosity; higher levels print board state and per-move detail. |
| `-l [file]` | Write a game log to disk (defaults to `game.log`). |

Example — a fast, silent batch-style game with an explicit time limit:

```bash
python -m referee -w 0 -v 1 -t 180 cascade_engine benchmarks.zobrist_engine
```

---

## Reproducing the evaluation

The win-rate experiments, opponent set, and runtime measurements that justify the
design decisions are reported in [`report.pdf`](report.pdf). To reproduce a single
matchup from that evaluation, run the corresponding `python -m referee` command with
the time limit set to the value documented there.

---

## Attribution

The `cascade_engine*` packages are the authors' own work. The `referee/` package is
the course-provided game driver and is unmodified. The opponents under `benchmarks/`
are used solely for local strength testing; where any originated from third parties,
they remain the property of their respective authors and are included here only as
sparring partners, not as part of the competitive submission.