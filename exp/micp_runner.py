import argparse
import os, sys, pickle
# add path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from exp.baselines import micp_cg_base, micp_ncg_linear_floor_embedding, micp_ncg_bezier_floor_embedding
from exp.env_generator import random_cg_env_2d, random_ncg_env_2d

from exp.exp_runner import BEZIER_CFG

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(description="Run MICP baseline for GCS-TSP")
    arg_parser.add_argument("problem", type=str, choices=["point", "linear", "bezier"])
    arg_parser.add_argument("--seed", type=int, default=0, help="Random seed for environment generation")
    arg_parser.add_argument("--num_sets", type=int, default=5, help="Number of sets in the environment")
    args = arg_parser.parse_args()

    if args.problem == "point":
        # env = random_cg_env_2d(
        #     seed=int(args.seed), N=int(args.grid_n), M=int(args.grid_m), num_sets=int(args.num_sets))
        with open(f"../data/envs/point/{args.seed}_{args.num_sets}.pkl", 'rb') as f:
            env = pickle.load(f)
        micp_cg_base(env)
    elif args.problem == "linear":
        # env = random_ncg_env_2d(seed=int(args.seed), num_edges=int(args.num_sets))
        with open(f"../data/envs/linear/{args.seed}_{args.num_sets}.pkl", 'rb') as f:
            env = pickle.load(f)
        micp_ncg_linear_floor_embedding(env)
    elif args.problem == "bezier":
        # env = random_ncg_env_2d(seed=int(args.seed), num_edges=int(args.num_sets))
        with open(f"../data/envs/linear/{args.seed}_{args.num_sets}.pkl", 'rb') as f:
            env = pickle.load(f)
        micp_ncg_bezier_floor_embedding(env, BEZIER_CFG)
    else:
        raise ValueError(f"Unknown experiment type: {args.problem}")
