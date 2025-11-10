#%%
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
log_fp = f"{__file__}.log" # + f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
if os.path.exists(log_fp):
    os.remove(log_fp)
logging.basicConfig(filename=log_fp, level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.env import BOXES3D

from src.gcs.bezier import BezierGCS, Configuration
from src.utils import draw_3d_set
from src.lower_bound_graph import LowerBoundGraph
from src.search import GHOST, Recorder, gcs_convex_restriction

cfg = Configuration(
    order = 4,
    continuity = 2,
    dt_min = 1e-6,
    derivative_regularization = (2, 1e-1, 1e-1),
    vmin = -1.0 * np.ones(3),
    vmax =  1.0 * np.ones(3),
)

env = BOXES3D
gcs = BezierGCS.build(env, cfg)

lbg = LowerBoundGraph.build(gcs)
tree = GHOST(gcs, lbg, epsilon=0.0, shortest_unfolding=True)
opt, _ = tree.grow(runtime_limit=1e3)
unfolded_path = opt.best_path

# unfolded_path = [0, 5, 10, 8, 11, 8, 12, 8, 7, 9, 7, 4, 3, 2, 5, 6, 5, 0, 1, 0]

print(f"unfolded_path={unfolded_path}")
traj = gcs_convex_restriction(unfolded_path, gcs)

T = np.arange(traj.start_time(), traj.end_time() + 0.1, 0.1)
trajectory = [traj.value(t) for t in T]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

for obs in env.O_Static:
    draw_3d_set(obs.hpoly, ax, fc='lightgray', alpha=0.3)

# ax.plot(
#     [p[0] for p in trajectory],
#     [p[1] for p in trajectory],
#     [p[2] for p in trajectory],
#     color='r', linewidth=2, label='trajectory')


point, = ax.plot([], [], [], 'ko', mfc='none', markersize=20)
tails, tail_length, tail_itvl  = [], 10, 2
for i in range(tail_length):
    alpha = 0.9 - i / tail_length
    trail, = ax.plot([], [], [], 'ko', mfc='none', alpha=alpha, markersize=20)
    tails.append(trail)

def animate(frame):
    p = trajectory[frame]
    point.set_data([p[0]], [p[1]])
    point.set_3d_properties([p[2]])

    for i, trail in enumerate(tails):
        prev_frame = max(0, frame - tail_itvl * i)
        x, y, z = trajectory[prev_frame]
        trail.set_data([x], [y])
        trail.set_3d_properties([z])

    return point, tails

# Create and run the animation
anim = animation.FuncAnimation(fig, animate, frames=len(trajectory), 
                             interval=50, blit=False, repeat=True)

# Display the animation
plt.tight_layout()
plt.show()

