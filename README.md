# MABC: Multi-objective Imitation Learning

This repository contains the implementation of MA-BC for the paper "Split the Differences, Pool the Rest: Provably Efficient Multi-Objective Imitation"

## Abstract
This work investigates multi-objective imitation learning: the problem of recovering policies that lie on the Pareto front given demonstrations from multiple Pareto-optimal experts in a Multi-Objective Markov Decision Process (MOMDP). Standard imitation approaches are ill-equipped for this regime, as naively aggregating conflicting expert trajectories can result in dominated policies. To address this, we introduce Multi-Output Augmented Behavioral Cloning (MA-BC), an algorithm that systematically partitions divergent expert data while pooling state-action pairs where no behavior conflict is observed. Theoretically, we prove that MA-BC converges to Pareto-optimal policies at a faster statistical rate than any learner that considers each expert dataset independently. Furthermore, we establish a novel lower bound for multi-objective imitation learning, demonstrating that MA-BC is minimax optimal. Finally, we empirically validate our algorithm across diverse discrete environments and, guided by our theoretical insights, extend and evaluate MA-BC on a continuous Linear Quadratic Regulator (LQR) control task. 

## Repository Structure
- `mabc/`: Core package containing source code.
  - `src/dst_solver.py`: Implementation of the Deep Sea Treasure solver.
  - `src/rg_solver.py`: Implementation of the Resource Gathering solver.
- `notebooks/`: Comprehensive analysis and demonstration of the algorithms across all environments.

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