# MABC: Multi-objective Imitation Learning

This repository contains the implementation of MA-BC for the paper "Split the Differences, Pool the Rest: Provably Efficient Multi-Objective Imitation"

## Repository Structure
- `mabc/`: Core package containing source code.
  - `src/dst_solver.py`: Implementation of the Deep Sea Treasure solver.
  - `src/rg_solver.py`: Implementation of the Resource Gathering solver.
- `notebooks/`: Comprehensive analysis and demonstration of the algorithms.
- `pyproject.toml`: Project configuration and dependency management (Poetry).

## Installation
This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mabc
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```

## Requirements
- Python >= 3.12
- mo-gymnasium
- numpy
- matplotlib
- torch
- stable-baselines3
- tqdm

(See `pyproject.toml` for full details.)
