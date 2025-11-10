#%%
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.gcs.base import BaseGCS
from src.gcs.linear import LinearGCS
from src.gcs.bezier import BezierGCS, Configuration

from src.env import COMPLETE2DV5
from src.lower_bound_graph import LowerBoundGraph
from src.search import GHOST, Recorder, gcs_convex_restriction

from exp.env_generator import random_ncg_env_2d, random_cg_env_2d
from exp.baselines import micp_ncg_linear_floor_embedding, micp_cg_base

import logging
log_fp = f"{__file__}.log" # + f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
if os.path.exists(log_fp):
    os.remove(log_fp)
logging.basicConfig(filename=log_fp, level=logging.INFO)
logger = logging.getLogger(__name__)


env = random_ncg_env_2d(2, 25)


def point_GCS():
    gcs = BaseGCS.build_complete_graph(env)
    
    lbg = LowerBoundGraph.build(gcs)
    tree = GHOST(gcs, lbg, epsilon=0.0)
    ts = time.perf_counter()
    opt, _ = tree.grow(runtime_limit=1e3)
    unfolded_path = opt.best_path
    print(f"GHOST found path in {time.perf_counter() - ts:.3f} seconds, cost={opt.ub}")

    # unfolded_path = [0, 1, 4, 5, 10, 9, 16, 17, 18, 19, 20, 21, 12, 6, 11, 13, 22, 14, 15, 7, 8, 3, 2, 0]

    # visualize the path
    print(f"unfolded_path={unfolded_path}")
    traj = gcs_convex_restriction(unfolded_path, gcs)
    
    ##### MICP
    # micp_cost = micp_cg_base(env).value
    # print('Optimal value:', micp_cost)

    return traj
    

def linear_GCS():
    gcs = LinearGCS.build(env)

    lbg = LowerBoundGraph.build(gcs)
    tree = GHOST(gcs, lbg, epsilon=0.0)
    ts = time.perf_counter()
    opt, _ = tree.grow(runtime_limit=1e3)
    unfolded_path = opt.best_path

    unfolded_path = [0, 1, 2, 3, 8, 3, 2, 7, 2, 5, 6, 13, 14, 15, 14, 13, 22, 13, 12, 21, 12, 11, 13, 6, 5, 10, 17, 18, 19, 20, 19, 18, 17, 10, 9, 16, 9, 4, 1, 0]

    # visualize the path
    print(f"unfolded_path={unfolded_path}")
    traj = gcs_convex_restriction(unfolded_path, gcs)

    # prob = micp_ncg_linear_floor_embedding(env)
    # print(f"prob={prob.value}")

    return traj


def bezier_GCS():
    cfg = Configuration(
        order = 4,
        continuity = 2,
        dt_min = 1e-6,
        derivative_regularization = (2, 1e-1, 1e-1),
        vmin = -1.0 * np.ones(env.dim),
        vmax =  1.0 * np.ones(env.dim),
    )
        
    gcs = BezierGCS.build(env, cfg)

    lbg = LowerBoundGraph.build(gcs)
    tree = GHOST(gcs, lbg, epsilon=0.0)
    ts = time.perf_counter()
    opt, _ = tree.grow(runtime_limit=1e3)
    unfolded_path = opt.best_path

    # unfolded_path = [0, 1, 2, 3, 8, 3, 2, 7, 2, 5, 6, 13, 12, 21, 12, 13, 22, 13, 14, 15, 14, 13, 11, 13, 6, 5, 10, 17, 18, 19, 20, 19, 18, 17, 10, 9, 16, 9, 4, 1, 0]

    # visualize the path
    print(f"unfolded_path={unfolded_path}")
    traj = gcs_convex_restriction(unfolded_path, gcs)

    return traj


fig, ax = plt.subplots()
ax.axis("equal")
env.draw_static(ax, draw_CSpace=True, v_text=False)

traj_a = point_GCS()
traj_b = linear_GCS()
traj_c = bezier_GCS()

for i in range(len(traj_a.points)-1):
    ax.plot(
        [traj_a.points[i][0], traj_a.points[i+1][0]],
        [traj_a.points[i][1], traj_a.points[i+1][1]],
        color='blue', marker='s', ms=4,
    )

for i in range(len(traj_b.points)-1):
    ax.plot(
        [traj_b.points[i][0], traj_b.points[i+1][0]],
        [traj_b.points[i][1], traj_b.points[i+1][1]],
        color='red', marker='o', ms=4,
    )

T = np.arange(traj_c.start_time(), traj_c.end_time() + 0.04, 0.04)
trajectory = [traj_c.value(t) for t in T]
ax.plot(
    [p[0] for p in trajectory],
    [p[1] for p in trajectory],
    '--k')
    

fig.savefig(f"{env.name}.png", bbox_inches='tight', dpi=500, transparent=True)