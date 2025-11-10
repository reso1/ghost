#%%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.env import GHOST_H
from src.gcs.bezier import BezierGCS, Configuration
from src.lower_bound_graph import LowerBoundGraph
from src.search import GHOST, gcs_convex_restriction

import logging
log_fp = f"{__file__}.log" # + f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
if os.path.exists(log_fp):
    os.remove(log_fp)
logging.basicConfig(filename=log_fp, level=logging.INFO)
logger = logging.getLogger(__name__)

env = GHOST_H
regions = env._CSpace_hpoly

cfg = Configuration(
    order = 4,
    continuity = 2,
    dt_min = 1e-6,
    derivative_regularization = (2, 1e-1, 1e-1),
    vmin = -1.0 * np.ones(env.dim),
    vmax =  1.0 * np.ones(env.dim),
)


#%%
gcs = BezierGCS.build(env, cfg)

lbg = LowerBoundGraph.build(gcs)
tree = GHOST(gcs, lbg, shortest_unfolding=True)
opt, _ = tree.grow(runtime_limit=1e3)
unfolded_path = opt.best_path

# unfolded_path = [0, 2, 7, 5, 4, 6, 4, 5, 7, 8, 9, 8, 12, 15, 17, 16, 17, 19, 20, 22, 23, 25, 26, 25, 22, 23, 21, 23, 24, 27, 23, 24, 17, 19, 18, 19, 20, 14, 13, 10, 11, 10, 7, 3, 1, 0]

print(f"unfolded_path={unfolded_path}")
traj = gcs_convex_restriction(unfolded_path, gcs)

fig, ax = plt.subplots()
ax.axis('equal')
ax.axis('off')
# for obs in env.O_Static:
#     obs.draw(ax, alpha=1.0)

# bounding_box_verts = np.array([
#     [env.lb[0] - env.robot_radius, env.lb[1] - env.robot_radius],
#     [env.ub[0] + env.robot_radius, env.lb[1] - env.robot_radius],
#     [env.ub[0] + env.robot_radius, env.ub[1] + env.robot_radius],
#     [env.lb[0] - env.robot_radius, env.ub[1] + env.robot_radius],
# ])
# for u, v in zip(bounding_box_verts, np.roll(bounding_box_verts, 1, axis=0)):
#     ax.plot([u[0], v[0]], [u[1], v[1]], '-k')

# for i, C in enumerate(env.C_Space):
#     ax.fill(C[:, 0], C[:, 1], alpha=0.8, fc='lightgrey', ec='none')
#     # draw the boundary of the C-Space
#     boundaries = np.vstack((C, C[0]))
#     ax.plot(boundaries[:, 0], boundaries[:, 1], '--k', lw=0.5)

# T = np.arange(traj.start_time(), traj.end_time() + 0.04, 0.04)
# trajectory = [traj.value(t) for t in T]
# colors = matplotlib.cm.get_cmap("gnuplot")
# for i in range(len(trajectory) - 1):
#     circle = Circle(
#         trajectory[i], env.robot_radius, 
#         facecolor=colors(i / len(trajectory)), 
#         edgecolor='black',
#         linewidth=0.5,
#         alpha=0.5
#     )
#     ax.add_patch(circle)
    

# fig.savefig(f"ghost.png", dpi=500, bbox_inches='tight')

env.draw_2d(ax, [traj], draw_CSpace=True)
env.animate_2d(ax, [traj], draw_CSpace=True, save_filename_prefix=env.name)

#%%
