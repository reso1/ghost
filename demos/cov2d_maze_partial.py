#%%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.env import MazeEnv
from src.gcs.bezier import BezierGCS, Configuration
from src.lower_bound_graph import LowerBoundGraph
from src.search import GHOST, gcs_convex_restriction

import logging
log_fp = f"{__file__}.log" # + f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
if os.path.exists(log_fp):
    os.remove(log_fp)
logging.basicConfig(filename=log_fp, level=logging.INFO)
logger = logging.getLogger(__name__)


env = MazeEnv.make(seed=1000, num_rooms_to_visit=20)

# fig, ax = plt.subplots()
# ax.axis('equal')
# env.draw(ax)

cfg = Configuration(
    order = 4,
    continuity = 2,
    dt_min = 1e-6,
    derivative_regularization = (2, 1e-1, 1e-1),
    vmin = -5.0 * np.ones(env.dim),
    vmax =  5.0 * np.ones(env.dim),
)

gcs = BezierGCS(env._CSpace_hpoly, cfg)
for index, region in enumerate(env._CSpace_hpoly):
    gcs.add_vertex(region, index)

for u, v in env.edges:
    gcs.add_edge(u, v)
    gcs.add_edge(v, u)

print(f"|V|={gcs.nx_diG.number_of_nodes()}, |E|={gcs.nx_diG.number_of_edges()}")

lbg_fn = 'data/lbg/maze.lbg'
if not os.path.exists(lbg_fn):
    print("Building and saving LBG...")
    lbg = LowerBoundGraph.build(gcs)
    lbg.save(lbg_fn)
else:
    print("Loading LBG...")
    lbg = LowerBoundGraph.load(lbg_fn)

tree = GHOST(gcs, lbg, epsilon=0.0, shortest_unfolding=True, targets=env.targets)
opt, _ = tree.grow(runtime_limit=1e3)
unfolded_path = opt.best_path

# unfolded_path = [100, 91, 92, 93, 111, 93, 94, 84, 110, 84, 85, 75, 74, 73, 63, 64, 65, 55, 45, 44, 34, 35, 113, 35, 36, 37, 47, 46, 108, 46, 56, 57, 67, 68, 58, 116, 58, 68, 67, 77, 87, 97, 96, 86, 76, 66, 109, 66, 76, 86, 96, 95, 117, 95, 96, 97, 98, 99, 103, 99, 98, 97, 87, 77, 78, 88, 89, 104, 89, 88, 78, 79, 69, 105, 69, 59, 58, 48, 49, 39, 102, 39, 38, 28, 27, 17, 7, 6, 16, 15, 5, 4, 14, 119, 14, 13, 3, 2, 12, 101, 12, 22, 23, 33, 114, 33, 43, 53, 107, 53, 52, 62, 72, 71, 61, 60, 50, 51, 41, 31, 112, 31, 30, 20, 115, 20, 30, 31, 41, 51, 50, 106, 50, 60, 61, 71, 118, 71, 61, 60, 70, 80, 90, 91, 100]

print(f"unfolded_path={unfolded_path}")
traj = gcs_convex_restriction(unfolded_path, gcs)

# T = np.arange(traj.start_time(), traj.end_time() + 0.04, 0.04)
# trajectory = [traj.value(t) for t in T]

env.robot_radius = 0.075 # for visualization only

fig, ax = plt.subplots()
ax.axis('equal')
ax.axis('off')
env.draw(ax)

# colors = matplotlib.cm.get_cmap("gnuplot")
# for i in range(len(trajectory) - 1):
#     circle = Circle(
#         trajectory[i], 0.075, 
#         facecolor=colors(i / len(trajectory)), 
#         edgecolor='black',
#         linewidth=0.5,
#         alpha=0.5
#     )
#     ax.add_patch(circle)
    
# fig.savefig(f"maze.png", dpi=500, bbox_inches='tight')


env.draw_2d(ax, [traj], draw_CSpace=False)
env.animate_2d(ax, [traj], draw_CSpace=False, save_filename_prefix=env.name, dt=0.01)


