# Cascade — Game-Playing Agent

A game-playing agent for **Cascade**, accompanied by a suite of opponent agents
used for local testing and strength evaluation.

The rules of the game are described in [`Cascade rules.pdf`](Cascade%20rules.pdf),
and the design, implementation, and experimental results of the primary agent are
documented in [`report.pdf`](report.pdf). This README does not restate either; it
covers what the repository contains, how it is organised, and how to run it.

---

## Project goal

The aim of the project is to produce an agent that plays Cascade as strongly as
possible while staying within the CPU-time and memory limits enforced by the
provided referee. The game's own objective and mechanics are covered in the rules
document; the strategy and algorithms used to pursue them are covered in the report.

Every agent in this repository is a self-contained Python package that exposes an
`Agent` class. The referee instantiates one `Agent` per side and drives the game by
requesting an action each turn and notifying each agent of actions as they are
applied, so any agent here can be played against any other.

---

## Repository structure

```
.
├── agent/                            # Primary agent — the intended entry point
├── agentv2/                          # Tuned variant of the primary agent
├── Baseline/                         # Simple reference opponents
│   ├── random_agent/                 #   chooses a legal move at random
│   └── greedy_agent/                 #   one-step lookahead opponent
├── Benchmarks/                       # Stronger search-based opponents
│   ├── iterative_deepening_agent/
│   ├── minimax_fixed_depth_agent/
│   └── zobrist_agent/
├── referee/                          # Provided game driver (unmodified)
├── report.pdf                        # Technical report: design, evaluation, results
├── Cascade rules.pdf                 # Game rules
└── team.py                           # Submission metadata
```

`agent` is the primary submission. `agentv2` shares its overall design with a set of
additional refinements, and is kept separate so the two can be played head-to-head to
measure their effect (see the report for details). The packages under `Baseline/` and
`Benchmarks/` are sparring opponents only — they form a ladder of increasing strength
for development testing and are not part of the competitive submission.

---

## Requirements

- **Python 3.12+**
- **`websockets`** — imported by the referee at startup (including for local games):

  ```bash
  pip install websockets
  ```

A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install websockets
```

---

## Running a game

Games are run through the referee from the repository root. The two positional
arguments are **package specifications** for the RED and BLUE players — the dotted
import path of a package containing an `Agent` class:

```bash
python -m referee <RED> <BLUE>
```

The nested packages are addressed with dotted paths (note the capitalised top-level
folder names). Run all commands from the repository root.

### Examples

Primary agent (RED) vs. the greedy baseline (BLUE):

```bash
python -m referee agent Baseline.greedy_agent
```

The two engine variants head-to-head:

```bash
python -m referee agent agentv2
```

Against a stronger search-based opponent:

```bash
python -m referee agent Benchmarks.zobrist_agent
```

Quick sanity check against the random baseline:

```bash
python -m referee agent Baseline.random_agent
```

If a package exposes its agent class under a name other than `Agent`, append it after
a colon, e.g. `Benchmarks.iterative_deepening_agent:Agent`.

### Useful options

Run `python -m referee --help` for the complete list. The most relevant:

| Option        | Effect                                                        |
|---------------|---------------------------------------------------------------|
| `-t <seconds>`| CPU-time limit per agent for the whole game.                  |
| `-s <MB>`     | Memory limit per agent.                                       |
| `-w <seconds>`| Wait time between turns; `-w 0` runs as fast as possible.     |
| `-v <0–3>`    | Verbosity; higher levels print board state and per-move info. |
| `-l [file]`   | Write a game log to disk (defaults to `game.log`).            |

Example — a fast, low-verbosity run suitable for batch testing:

```bash
python -m referee -w 0 -v 1 agent Benchmarks.minimax_fixed_depth_agent
```

---

## Further reading

- **Game rules** — [`Cascade rules.pdf`](Cascade%20rules.pdf)
- **Agent design, evaluation methodology, and results** — [`report.pdf`](report.pdf)

---

## Attribution

`agent` and `agentv2` are the authors' own work. The `referee/` package is the
course-provided game driver and is unmodified. The opponents under `Baseline/` and
`Benchmarks/` are included solely as local testing partners; where any originated
from third parties they remain the property of their respective authors.