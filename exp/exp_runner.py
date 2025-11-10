import argparse
import os, sys, pickle, time
import numpy as np

# add path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.gcs.base import BaseGCS
from src.gcs.linear import LinearGCS
from src.gcs.bezier import BezierGCS, Configuration
from src.lower_bound_graph import LowerBoundGraph
from src.search import GHOST, Recorder

from exp.baselines import EuclideanDistGraph, lbg_greedy
from exp.env_generator import random_cg_env_2d, random_ncg_env_2d

import logging
log_fp = f"{__file__}.log" # + f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
if os.path.exists(log_fp):
    os.remove(log_fp)
logging.basicConfig(filename=log_fp, level=logging.INFO)
logger = logging.getLogger(__name__)


BEZIER_CFG = Configuration(
    order=4,
    continuity=2,
    dt_min=1e-6,
    vmin=-1.0 * np.ones(2),
    vmax=1.0 * np.ones(2),
)


def exp_ghost(gcs_type, num_seeds, runtime_limit, epsilon):
    res = {}
    N_list = [int(n) for n in range(5, 26, 1)]

    for N in N_list[::-1]:
        for seed in range(num_seeds):
            print(f"Running GHOST with seed={seed}, N={N}...,")
            if gcs_type == "point":
                # env = random_cg_env_2d(seed, grid_n, grid_m, N)
                with open(f"../data/envs/point/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = BaseGCS.build_complete_graph(env)
            elif gcs_type == "linear":
                # env = random_ncg_env_2d(seed, N)
                with open(f"../data/envs/linear/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = LinearGCS.build(env)
            elif gcs_type == "bezier":
                # env = random_ncg_env_2d(seed, N)
                with open(f"../data/envs/linear/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = BezierGCS.build(env, BEZIER_CFG)
            else:
                raise ValueError("Unknown environment function")

            ts = time.perf_counter()
            lbg = LowerBoundGraph.build(gcs)
            t_lbg = time.perf_counter() - ts

            ts = time.perf_counter()
            tree = GHOST(gcs, lbg, epsilon=epsilon)
            opt, lb = tree.grow(runtime_limit=runtime_limit - t_lbg)
            t_osf = time.perf_counter() - ts

            res[(seed, N)] = {
                'lbg_time': t_lbg,
                'osf_time': t_osf,
                'opt_node': opt,
                'lb': lb,
                'M': lbg.G.number_of_edges()
            }
 
    if epsilon != 0.0:
        gcs_type += f"-eps{epsilon}"

    with open(f"../data/exp/{gcs_type}-ghost.pkl", 'wb') as f:
        pickle.dump(res, f)


def exp_ecg_ghost(gcs_type, num_seeds, runtime_limit):
    res = {}
    N_list = [int(n) for n in range(5, 26, 1)]

    for seed in range(num_seeds):
        for N in N_list:
            print(f"Running ECG+GHOST with seed={seed}, N={N}...")
            if gcs_type == "point":
                # env = random_cg_env_2d(seed, grid_n, grid_m, N)
                with open(f"../data/envs/point/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = BaseGCS.build_complete_graph(env)
            elif gcs_type == "linear":
                # env = random_ncg_env_2d(seed, N)
                with open(f"../data/envs/linear/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = LinearGCS.build(env)
            elif gcs_type == "bezier":
                # env = random_ncg_env_2d(seed, N)
                with open(f"../data/envs/linear/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = BezierGCS.build(env, BEZIER_CFG)
            else:
                raise ValueError("Unknown environment function")

            ts = time.perf_counter()
            ecg = EuclideanDistGraph.build(gcs)
            t_lbg = time.perf_counter() - ts

            ts = time.perf_counter()
            tree = GHOST(gcs, ecg, epsilon=0.0) # epsilon is menaingless for ECG
            opt, lb = tree.grow(runtime_limit=runtime_limit - t_lbg)
            t_osf = time.perf_counter() - ts
            res[(seed, N)] = {
                'lbg_time': t_lbg,
                'osf_time': t_osf,
                'opt_node': opt,
            }

    with open(f"../data/exp/{gcs_type}-ghost-ecg.pkl", 'wb') as f:
        pickle.dump(res, f)


def exp_greedy(gcs_type, num_seeds, runtime_limit):
    res = {}
    N_list = [int(n) for n in range(5, 26, 1)]

    for seed in range(num_seeds):
        for N in N_list:
            print(f"Running LBG+Greedy with seed={seed}, N={N}...")
            if gcs_type == "point":
                # env = random_cg_env_2d(seed, grid_n, grid_m, N)
                with open(f"../data/envs/point/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = BaseGCS.build_complete_graph(env)
            elif gcs_type == "linear":
                # env = random_ncg_env_2d(seed, N)
                with open(f"../data/envs/linear/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = LinearGCS.build(env)
            elif gcs_type == "bezier":
                # env = random_ncg_env_2d(seed, N)
                with open(f"../data/envs/linear/{seed}_{N}.pkl", 'rb') as f:
                    env = pickle.load(f)
                gcs = BezierGCS.build(env, BEZIER_CFG)
            else:
                raise ValueError("Unknown environment function")

            ts = time.perf_counter()
            lbg = LowerBoundGraph.build(gcs)
            t_lbg = time.perf_counter() - ts

            ts = time.perf_counter()
            _, best_cost = lbg_greedy(lbg, gcs, runtime_limit - t_lbg)
            t_alg = time.perf_counter() - ts

            res[(seed, N)] = {
                'lbg_time': t_lbg,
                'greedy_time': t_alg,
                'best_cost': best_cost,
            }

    with open(f"../data/exp/{gcs_type}-greedy.pkl", 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Experiment Runner")
    arg_parser.add_argument("problem", type=str, choices=["point", "linear", "bezier"])
    arg_parser.add_argument("--method", type=str, choices=["ghost", "ghost-ecg", "greedy"])
    arg_parser.add_argument("--timeout", type=int, default=100, help="Runtime limit in seconds")
    arg_parser.add_argument("--num_seeds", type=int, default=12, help="Number of random seeds to run")
    arg_parser.add_argument("--epsilon", type=float, default=0.0, help="Epsilon for GHOST algorithm")
    
    args = arg_parser.parse_args()

    if args.method == "ghost":
        exp_ghost(
            gcs_type=args.problem,
            num_seeds=int(args.num_seeds), 
            runtime_limit=float(args.timeout),
            epsilon=float(args.epsilon)
        )
    elif args.method == "ghost-ecg":
        exp_ecg_ghost(
            gcs_type=args.problem,
            num_seeds=int(args.num_seeds), 
            runtime_limit=float(args.timeout)
        )
    elif args.method == "greedy":
        exp_greedy(
            gcs_type=args.problem,
            num_seeds=int(args.num_seeds), 
            runtime_limit=float(args.timeout)
        )
    else:
        raise ValueError(f"Unknown experiment type: {args.problem}")
