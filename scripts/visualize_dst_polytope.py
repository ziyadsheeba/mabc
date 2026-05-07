import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

from mabc.src.dst_solver import DSTValueIteration, solve_weight_worker, is_pareto_efficient


def generate_circle_weights(num_samples):
    """Generates weight vectors in all directions (360 degrees)."""
    angles = np.linspace(0, 2 * np.pi, num_samples)
    weights = [[np.cos(a), np.sin(a)] for a in angles]
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", type=int, default=10000, help="Number of weight directions"
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

                                                  
    print("Building environment model...")
    full_solver = DSTValueIteration(gamma=args.gamma)

                              
    solver_data = (
        full_solver.transitions,
        full_solver.rewards,
        full_solver.terminals,
        full_solver.is_rock,
        full_solver.rows,
        full_solver.cols,
        full_solver.n_actions,
        full_solver.gamma,
        full_solver.theta,
    )

    weights = generate_circle_weights(args.samples)

                       
    print(f"Solving for {len(weights)} weight vectors in parallel...")
    policies = []
    with ProcessPoolExecutor() as executor:
                                                
        futures = [
            executor.submit(solve_weight_worker, solver_data, w) for w in weights
        ]
        for f in tqdm(futures, desc="Solving for weights"):
            policies.append(f.result())

                                                                                                  
    points = []
    seen_returns = set()
    print("Evaluating policies...")
    for policy in tqdm(policies, desc="Evaluating policies"):
        ret = full_solver.get_discounted_return(policy)
        ret_tuple = (round(ret[0], 4), round(ret[1], 4))
        if ret_tuple not in seen_returns:
            seen_returns.add(ret_tuple)
            points.append(ret)

    points = np.array(points)

                                       
    pareto_mask = is_pareto_efficient(points)
    pareto_points = points[pareto_mask]
    other_points = points[~pareto_mask]
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

    plt.figure(figsize=(12, 8))
    if len(points) >= 3:
        hull = ConvexHull(points)
        plt.fill(
            points[hull.vertices, 0],
            points[hull.vertices, 1],
            "skyblue",
            alpha=0.3,
            label="Achievable Returns",
        )
        for simplex in hull.simplices:
            plt.plot(
                points[simplex, 0], points[simplex, 1], "k--", linewidth=0.5, alpha=0.5
            )

    if len(other_points) > 0:
        plt.scatter(
            other_points[:, 0],
            other_points[:, 1],
            c="red",
            s=40,
            zorder=4,
            label="Dominated Vertices",
        )
    plt.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        c="green",
        s=60,
        zorder=5,
        label="Pareto Front Vertices",
    )
    plt.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        "g-",
        linewidth=2.5,
        zorder=6,
        label="Pareto Frontier",
    )

    plt.title(f"Full Return Polytope - Parallelized (gamma={args.gamma})")
    plt.xlabel("Discounted Treasure Value")
    plt.ylabel("Discounted Time Penalty")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left")

    save_path = "dst_full_polytope.png"
    plt.savefig(save_path)
    print(f"Full polytope visualization saved to {save_path}")
    print(f"Total vertices: {len(points)} ({len(pareto_points)} on Pareto front)")


if __name__ == "__main__":
    main()
