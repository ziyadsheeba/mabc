import argparse

from mabc.src.dst_solver import DSTValueIteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    weight = [args.w1, args.w2]
    solver = DSTValueIteration(gamma=args.gamma)
    policy = solver.solve(weight)

                     
    ret = solver.get_discounted_return(policy)
    print(f"Optimal Discounted Vector Return for weights {weight}:")
    print(f"Treasure: {ret[0]:.4f}, Time: {ret[1]:.4f}")


if __name__ == "__main__":
    main()
