from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List, Tuple
from copy import deepcopy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from pydrake.all import HPolyhedron, VPolytope, RandomGenerator, CompositeTrajectory

from src.utils import draw_3d_set, draw_2d_set, make_hpolytope


""" Environment """

class Env:

    def __init__(
        self, name, CSpace:List[np.ndarray], robot_radius:float, OStatic:List[StaticObstacle]=[]
    ) -> None:
        self.name = name
        self.C_Space = CSpace
        self.robot_radius = robot_radius
        self.O_Static = OStatic
        self.lb: np.ndarray = np.min(np.vstack(CSpace), axis=0)
        self.ub: np.ndarray = np.max(np.vstack(CSpace), axis=0)
        self.dim: int = len(self.lb)
        self._CSpace_hpoly: List[HPolyhedron] = [make_hpolytope(C) for C in CSpace]

    def copy(self) -> Env:
        return Env(self.name, deepcopy(self.C_Space), self.robot_radius, deepcopy(self.O_Static), deepcopy(self.O_Dynamic))
    
    def draw_2d(
        self, ax:Axes, concat_traj:List[CompositeTrajectory]=[], 
        dt:float=0.02, draw_CSpace=False, color:str='k'
    ) -> None:
        trajectory = []
        for pi in concat_traj:
            T = np.arange(pi.start_time(), pi.end_time() + dt, dt)
            trajectory.extend([pi.value(t) for t in T])

        trajectory = np.array(trajectory)
        self.draw_static(ax, draw_CSpace=draw_CSpace)
        ax.plot(trajectory[:, 0], trajectory[:, 1], f'{color}-')
        ax.plot(trajectory[0, 0], trajectory[0, 1], f'{color}o')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], f'{color}*')
        ax.axis('equal')

    def animate_2d(
        self, ax:Axes, concat_traj:List[CompositeTrajectory]=[], 
        dt:float=0.02, draw_CSpace=False, save_filename_prefix:str=None
    ) -> None:
        trajectory = []
        for pi in concat_traj:
            T = np.arange(pi.start_time(), pi.end_time() + dt, dt)
            trajectory.extend([pi.value(t) for t in T])

        ax.set_aspect('equal')
        self.draw_static(ax, draw_CSpace=draw_CSpace)
        anim = _animate_func_2d(ax, self.robot_radius, self.lb, self.ub, [np.array(trajectory)], dt)
        if save_filename_prefix is not None:
            anim.save(f"{save_filename_prefix}-{self.name}.mp4", writer='ffmpeg', fps=1/dt, dpi=500)
        plt.show()
    
    def draw_static(self, ax:Axes, alpha=0.8, draw_CSpace:bool=False, v_text:bool=True) -> None:
        for obs in self.O_Static:
            obs.draw(ax, alpha=alpha)
        
        bounding_box_verts = np.array([
            [self.lb[0] - self.robot_radius, self.lb[1] - self.robot_radius],
            [self.ub[0] + self.robot_radius, self.lb[1] - self.robot_radius],
            [self.ub[0] + self.robot_radius, self.ub[1] + self.robot_radius],
            [self.lb[0] - self.robot_radius, self.ub[1] + self.robot_radius],
        ])
        for u, v in zip(bounding_box_verts, np.roll(bounding_box_verts, 1, axis=0)):
            ax.plot([u[0], v[0]], [u[1], v[1]], '-k')
        
        if draw_CSpace:
            colors = matplotlib.cm.get_cmap("Pastel2")
            for i, C in enumerate(self.C_Space):
                ax.fill(C[:, 0], C[:, 1], alpha=alpha, fc=colors(i/len(self.C_Space)), ec='black')
                # if v_text:
                #     center = np.mean(C, axis=0)
                #     ax.text(center[0], center[1], f"v{i}", fontsize=12)

    def collision_checking_seg(self, p:np.ndarray, q:np.ndarray, tp:float, tq:float) -> bool:
        # collision checking w/ static obstacles
        for o in self.O_Static:
            if o.is_colliding_lineseg(p, q, self.robot_radius):
                return True
        
        if tp > tq:
            p, q, tp, tq = q, p, tq, tp

        return False

    def sample_CSpace(self, np_rng:np.random.RandomState, drake_rng:RandomGenerator) -> np.ndarray:
        idx = np_rng.randint(0, len(self.C_Space))
        return self._CSpace_hpoly[idx].UniformSample(drake_rng)
    
    def sample_bounding_box(self, np_rng:np.random.RandomState) -> np.ndarray:
        return np_rng.uniform(self.lb, self.ub)


def _animate_func_2d(
    ax: Axes,
    robot_radius: float,
    lb: np.ndarray,
    ub: np.ndarray,
    trajectories: List[np.ndarray],
    dt:float = 0.02,
    labels: List[str] = None,
    colors: List[str] = None,
    interval: int = 30,
    robot_tail_length: int = 10,
) -> FuncAnimation:
    
    k = len(trajectories)
    
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, k))
    
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(k)]
    
    x_min, y_min = lb
    x_max, y_max = ub

    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # draw robot trajectories
    footprint_itvl = int(interval / robot_tail_length)
    footprints, robots, texts = [None] * (robot_tail_length * k), [None] * k, [None] * k

    for i in range(k):
        for j in range(robot_tail_length):
            idx = i * robot_tail_length + j
            footprints[idx] = ax.add_patch(Circle(xy=trajectories[i][0], radius=robot_radius, color=colors[i], fill=False, alpha = 1 - (j / robot_tail_length)))
            # footprints[idx], = ax.plot([], [], '.', color=colors[i], markersize=robot_size, mfc='none', alpha = 1 - (j / robot_tail_length))
        
        robots[i] = ax.add_patch(Circle(xy=trajectories[i][0], radius=robot_radius, color=colors[i], fill=True))
        # robots[i], = ax.plot([], [], '.', color=colors[i], markersize=robot_size, mfc=colors[i])
        texts[i] = ax.text(0, 0, '', color='k', fontsize=12)    

    # animation function
    def animate(frame):

        for i, (trajectory, point, text) in enumerate(zip(trajectories, robots, texts)):
            if frame < len(trajectory):
                point.set_center(trajectory[frame])
                # point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
                text.set_position((trajectory[frame, 0], trajectory[frame, 1]))
                text.set_text(i)
                
                for j in range(robot_tail_length-1, -1, -1):
                    idx = i * robot_tail_length + j
                    prev_frame = max(0, frame - footprint_itvl * j)
                    # x, y = trajectory[prev_frame, 0], trajectory[prev_frame, 1]
                    # footprints[idx].set_data([x], [y])
                    footprints[idx].set_center(trajectory[prev_frame])
            
        return footprints + robots + texts
    
    fig = plt.gcf()
    if trajectories == []:
        n_frames = 1000
    else:
        n_frames = max(len(traj) for traj in trajectories)
    anim = FuncAnimation(
        fig, animate, frames=n_frames,
        interval=interval, blit=True
    )
    
    plt.grid(False)
    plt.tight_layout()
    
    return anim


""" Static Obstacles """

class StaticObstacle(ABC):
    
    @abstractmethod
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        raise NotImplementedError

    @abstractmethod
    def draw(self, ax:Axes, color='k') -> None:
        raise NotImplementedError


class StaticSphere(StaticObstacle):

    def __init__(self, pos:np.ndarray, radius:float,) -> None:
        self.pos, self.radius = pos, radius
    
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        return np.linalg.norm(self.pos - point) <= self.radius + robot_radius
    
    def is_colliding_lineseg(self, p:np.ndarray, q:np.ndarray, robot_radius:float) -> bool:
        if self.is_colliding(p, robot_radius) or self.is_colliding(q, robot_radius):
            return True
        
        p2m1 = q - p
        p1mc = p - self.pos

        dot = np.dot(p1mc, p2m1)
        squared_norm = np.linalg.norm(p2m1) ** 2
        t = -1 * (dot / squared_norm)
        if t < 0:
            t = 0
        elif t > 1:
            t = 1
        closest = p + p2m1 * t
        return np.linalg.norm(closest - self.pos) <= self.radius + robot_radius

    def draw(self, ax:Axes, color='k') -> None:
        circle = Circle(self.pos, self.radius, color=color)
        ax.add_artist(circle)
    

class StaticPolygon(StaticObstacle):

    def __init__(self, vertices:np.ndarray) -> None:
        self.vertices = vertices
        self.hpoly = HPolyhedron(VPolytope(vertices.T))
    
    def is_colliding(self, point:np.ndarray, robot_radius:float) -> bool:
        assert len(point) == 2
        for xc, yc in [[-1, -1], [1, -1], [1, 1], [-1, 1]]:
            if self.hpoly.PointInSet(point + robot_radius * np.array([xc, yc])):
                return True
        return False

    def draw(self, ax:Axes, alpha=0.8, color='k') -> None:
        ax.fill(self.vertices[:, 0], self.vertices[:, 1], alpha=alpha, fc=color, ec='black')

    def draw_with_time(self, ax:Axes3D, tmax:float, color='k', alpha:float=1.0) -> None:
        hpoly = self.hpoly.CartesianProduct(HPolyhedron.MakeBox([0], [tmax]))
        draw_3d_set(hpoly, ax, alpha=alpha, fc=color)


""" Example Envs """


COMPLETE2DV5 = Env(
    name="complete2dv5",
    CSpace = [
        np.array([[-4, 2], [4, 2], [4, 3], [-4, 3]]),
        np.array([[-4, -1], [-2, -1], [-2, 1], [-4, 1]]),
        np.array([[0.3, -0.7], [1.3, 0.3], [0.3, 1.3], [-0.7, 0.3]]),
        np.array([[0, -3], [0.71, -2.85], [1, -2.5], [0.71, -2.15], [0, -2], [-0.71, -2.15], [-1, -2.5], [-0.71, -2.85]]),
        np.array([[3, -1], [3.71, -0.71], [4, 0], [3.71, 0.71], [3, 1], [2.29, 0.71], [2, 0], [2.29, -0.71]]),
	],
    OStatic = [],
    robot_radius=0.05
)


COMPLEX2D = Env(
    name="complex2d",
    CSpace = [
        np.array([[0.4, 0.0], [0.4, 5.0], [0.0, 5.0], [0.0, 0.0]]),
		np.array([[0.4, 2.4], [1.0, 2.4], [1.0, 2.6], [0.4, 2.6]]),
		np.array([[1.4, 2.2], [1.4, 4.6], [1.0, 4.6], [1.0, 2.2]]),
		np.array([[1.4, 2.2], [2.4, 2.6], [2.4, 2.8], [1.4, 2.8]]),
		np.array([[2.2, 2.8], [2.4, 2.8], [2.4, 4.6], [2.2, 4.6]]),
		np.array([[1.4, 2.2], [1.0, 2.2], [1.0, 0.0], [3.7, 0.0], [3.7, 0.2]]),
		np.array([[3.7, 4.6], [3.7, 5.0], [1.0, 5.0], [1.0, 4.6]]),
		np.array([[5.0, 0.0], [5.0, 1.2], [4.8, 1.2], [3.7, 0.2], [3.7, 0.0]]),
		np.array([[3.4, 2.6], [4.8, 1.2], [5.0, 1.2], [5.0, 2.6]]),
		np.array([[3.4, 2.6], [3.7, 2.6], [3.7, 4.6], [3.4, 4.6]]),
		np.array([[3.7, 2.8], [4.4, 2.8], [4.4, 3.0], [3.7, 3.0]]),
		np.array([[5.0, 2.8], [5.0, 5.0], [4.4, 5.0], [4.4, 2.8]]),
	],
    OStatic = [
        StaticPolygon(np.array([[3.3793, 2.55], [2.4096, 2.55], [1.4964, 2.1847], [3.7969, 0.2676], [4.7293, 1.2]])),
		StaticPolygon(np.array([[3.35, 4.55], [2.45, 4.55], [2.45, 2.55], [3.35, 2.55]])),
		StaticPolygon(np.array([[1.45, 2.85], [2.15, 2.85], [2.15, 4.55], [1.45, 4.55]])),
		StaticPolygon(np.array([[0.95, 2.65], [0.95, 5.05], [0.45, 5.05], [0.45, 2.65]])),
		StaticPolygon(np.array([[0.95, 2.35], [0.95, -0.05], [0.45, -0.05], [0.45, 2.35]])),
		StaticPolygon(np.array([[3.85, 3.05], [3.85, 5.05], [4.35, 5.05], [4.35, 3.05]])),
		StaticPolygon(np.array([[3.85, 2.75], [3.85, 2.65], [5.05, 2.65], [5.05, 2.75]])),
	],
    robot_radius=0.05
)


GHOST_H = Env(
    name="ghost_h",
    CSpace = [
        np.array([[0.0, 0.5], [0.065, 0.5], [0.285, 0.875], [0.285, 1.0], [0.0, 1.0]]),
        np.array([[0.0, 0.5], [0.065, 0.5], [0.285, 0.125], [0.285, 0.0], [0.0, 0.0]]),
        np.array([[0.285, 0.875], [0.725, 0.875], [0.725, 1.0], [0.285, 1.0]]),
        np.array([[0.285, 0.125], [0.725, 0.125], [0.725, 0.0], [0.285, 0.0]]),
        np.array([[0.25, 0.5], [0.36, 0.725], [0.375, 0.575], [0.375, 0.425], [0.36, 0.275]]),
        np.array([[0.36, 0.725], [0.725, 0.725], [0.725, 0.575], [0.375, 0.575]]),
        np.array([[0.36, 0.275], [0.575, 0.275], [0.575, 0.425], [0.375, 0.425]]),
        np.array([[0.725, 0.0], [0.975, 0.0], [0.975, 1.0], [0.725, 1.0]]),
        np.array([[0.975, 0.875], [1.525, 0.875], [1.525, 1.0], [0.975, 1.0]]),
        np.array([[1.125, 0.875], [1.125, 0.575], [1.375, 0.575], [1.375, 0.875]]),
        np.array([[0.975, 0.125], [0.975, 0.0], [1.525, 0.0], [1.525, 0.125]]),
        np.array([[1.125, 0.125], [1.125, 0.425], [1.375, 0.425], [1.375, 0.125]]),
        np.array([[1.525, 0.5], [1.725, 0.5], [1.885, 0.775], [1.885, 1.0], [1.525, 1.0]]),
        np.array([[1.525, 0.5], [1.725, 0.5], [1.885, 0.225], [1.885, 0.0], [1.525, 0.0]]),
        np.array([[1.885, 0.225], [2.175, 0.225], [2.175, 0.0], [1.885, 0.0]]),
        np.array([[1.885, 0.775], [2.125, 0.775], [2.125, 1.0], [1.885, 1.0]]),
        np.array([[1.89, 0.5], [1.9675, 0.625], [2.12, 0.625], [2.175, 0.5], [2.085, 0.375], [1.97, 0.375]]),
        np.array([[2.125, 1.0], [2.1275, 0.6], [2.465, 0.7], [2.7, 0.88], [2.7, 1.0]]),
        np.array([[2.465, 0.7], [2.765, 0.4],  [2.7, 0.33], [2.6, 0.425]]),
        np.array([[2.24, 0.635], [2.465, 0.7], [2.6, 0.425], [2.47, 0.37], [2.325, 0.5]]),
        np.array([[2.325, 0.5], [2.7, 0.165], [2.7, 0.0], [2.175, 0.0], [2.175, 0.25]]),
        np.array([[2.645, 0.695], [2.7, 0.72], [2.85, 0.6], [2.93, 0.4]]),
        np.array([[2.7, 0.165], [2.7, 0.0], [3.325, 0.0], [3.325, 0.175], [2.93, 0.4]]),
        np.array([[2.93, 0.4], [3.325, 0.175], [3.325, 0.775], [3.125, 0.875], [2.93, 0.7], [2.85, 0.6]]),
        np.array([[2.93, 0.7], [3.125, 0.875], [3.2, 1.0], [2.7, 1.0], [2.7, 0.88]]),
        np.array([[3.325, 0.175], [3.7, 0.175], [3.7, 0.0], [3.325, 0.0]]),
        np.array([[3.475, 0.175], [3.7, 0.175], [3.7, 0.875], [3.68, 0.875], [3.475, 0.775]]),
        np.array([[3.125, 0.875], [3.7, 0.875], [3.7, 1.0], [3.2, 1.0]]),
    ],
    OStatic = [
        StaticPolygon(np.array(
        [[0.1, 0.5], [0.3, 0.85], [0.7, 0.85], [0.7, 0.75], 
         [0.35, 0.75], [0.225, 0.5], [0.35, 0.25], [0.6, 0.25], 
         [0.6, 0.45], [0.4, 0.45], [0.4, 0.55], [0.7, 0.55], [0.7, 0.15], [0.3, 0.15]])),
        StaticPolygon(np.array(
        [[1.0, 0.15], [1.0, 0.85], [1.1, 0.85], [1.1, 0.55], [1.4, 0.55], [1.4, 0.85],
         [1.5, 0.85], [1.5, 0.15], [1.4, 0.15], [1.4, 0.45], [1.1, 0.45], [1.1, 0.15]])),
        StaticPolygon(np.array(
        [[1.75, 0.5], [1.9, 0.75], [2.1, 0.75], [2.1, 0.65], [1.95, 0.65],
         [1.865, 0.5], [1.95, 0.35], [2.1, 0.35], [2.2, 0.5], [2.16, 0.59], 
         [2.23, 0.61], [2.3, 0.5], [2.15, 0.25], [1.9, 0.25]])),
        StaticPolygon(np.array(
        [[2.5, 0.7], [2.7, 0.85], [2.9, 0.7], [2.85, 0.63], [2.7, 0.75],
         [2.61, 0.7], [2.9, 0.4], [2.7, 0.2], [2.515, 0.36], [2.6, 0.4],
         [2.7, 0.3], [2.8, 0.4]])),
        StaticPolygon(np.array(
        [[3.225, 0.85], [3.575, 0.85], [3.45, 0.8], [3.45, 0.2], [3.35, 0.2], [3.35, 0.8]])),
    ],
    robot_radius=0.025
)


def make_3d_box(center:np.ndarray, halfsizes:np.ndarray=0.5*np.ones(3)) -> np.ndarray:
    assert len(center) == 3 and len(halfsizes) == 3
    vertices = np.array([
        [center[0] - halfsizes[0], center[1] - halfsizes[1], center[2] - halfsizes[2]],
        [center[0] + halfsizes[0], center[1] - halfsizes[1], center[2] - halfsizes[2]],
        [center[0] + halfsizes[0], center[1] + halfsizes[1], center[2] - halfsizes[2]],
        [center[0] - halfsizes[0], center[1] + halfsizes[1], center[2] - halfsizes[2]],
        [center[0] - halfsizes[0], center[1] - halfsizes[1], center[2] + halfsizes[2]],
        [center[0] + halfsizes[0], center[1] - halfsizes[1], center[2] + halfsizes[2]],
        [center[0] + halfsizes[0], center[1] + halfsizes[1], center[2] + halfsizes[2]],
        [center[0] - halfsizes[0], center[1] + halfsizes[1], center[2] + halfsizes[2]],
    ])
    return vertices


BOXES3D = Env(
    name = "boxes3d",
    CSpace = [
        make_3d_box(np.array([0.5, 1.75, 1.0]), np.array([0.5, 0.75, 1.0])),
        make_3d_box(np.array([0.4, 0.5, 1.6]), np.array([0.4, 0.5, 0.4])),
        make_3d_box(np.array([0.75, 3.5, 0.5]), np.array([0.75, 0.5, 0.5])),
        make_3d_box(np.array([1.25, 3.75, 1.25]), np.array([0.25, 0.25, 0.25])),
        make_3d_box(np.array([1.5, 3.0, 1.75]), np.array([0.5, 1.0, 0.25])),
        make_3d_box(np.array([1.5, 1.5, 0.5]), np.array([0.5, 1.5, 0.5])),
        make_3d_box(np.array([3.0, 3.5, 0.25]), np.array([1.0, 0.5, 0.25])),
        make_3d_box(np.array([3.0, 3.5, 1.5]), np.array([1.0, 0.5, 0.5])),
        make_3d_box(np.array([3.5, 2.0, 1.0]), np.array([0.5, 1.0, 1.0])),
        make_3d_box(np.array([3.75, 3.75, 0.75]), np.array([0.25, 0.25, 0.25])),
        make_3d_box(np.array([2.25, 1.5, 1.5]), np.array([0.75, 0.5, 0.5])),
        make_3d_box(np.array([2.5, 1.5, 0.25]), np.array([0.5, 0.5, 0.25])),
        make_3d_box(np.array([3.75, 0.5, 0.7]), np.array([0.25, 0.5, 0.3])),
    ],
    OStatic= [
        StaticPolygon(make_3d_box(np.array([0.5, 0.5, 0.6]), np.array([0.5, 0.5, 0.6]))),
        StaticPolygon(make_3d_box(np.array([2.5, 2.5, 1.0]), np.array([0.5, 0.5, 1.0]))),
        StaticPolygon(make_3d_box(np.array([3.0, 0.5, 0.2]), np.array([1.0, 0.5, 0.2]))),
        StaticPolygon(make_3d_box(np.array([0.5, 3.5, 1.5]), np.array([0.5, 0.5, 0.5]))),
        StaticPolygon(make_3d_box(np.array([2.5, 3.5, 0.75]), np.array([1.0, 0.5, 0.25]))),
        StaticPolygon(make_3d_box(np.array([1.75, 3.0, 1.25]), np.array([0.25, 1.0, 0.25]))),
        StaticPolygon(make_3d_box(np.array([2.5, 1.0, 0.7]), np.array([0.5, 1.0, 0.3]))),
        StaticPolygon(make_3d_box(np.array([2.4, 0.5, 1.5]), np.array([1.6, 0.5, 0.5]))),
        StaticPolygon(make_3d_box(np.array([3.25, 0.5, 0.7]), np.array([0.25, 0.5, 0.3]))),
        StaticPolygon(make_3d_box(np.array([1.25, 2.75, 1.25]), np.array([0.25, 0.75, 0.25]))),
        StaticPolygon(make_3d_box(np.array([1.25, 1.5, 1.5]), np.array([0.25, 0.5, 0.5]))),
        StaticPolygon(make_3d_box(np.array([1.75, 3.5, 0.25]), np.array([0.25, 0.5, 0.25]))),
        StaticPolygon(make_3d_box(np.array([3.75, 3.25, 0.75]), np.array([0.25, 0.25, 0.25]))),
    ],
    robot_radius = 0.05,
)


""" 2D Maze Environment (https://github.com/RobotLocomotion/gcs-science-robotics) """


class Cell:
    """A cell in the maze. A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.
    """

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def _knock_down_wall(self, wall):
        """Knock down the given wall."""

        self.walls[wall] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny):
        """Initialize the maze grid.
        The maze consists of nx x ny cells.

        """

        self.nx, self.ny = nx, ny
        self.map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.map[x][y]
    
    def plot(self, linewidth):
        plt.gca().axis('off')
        
        # Pad the maze all around by this amount.
        width = self.nx
        height = self.ny
        
        # Draw the South and West maze borders.
        for x in range(self.nx):
            for y in range(self.ny):
                if self.cell_at(x, y).walls['S'] and (x != 0 or y != 0):
                    plt.plot([x, x + 1], [y, y], c='k', linewidth=linewidth)
                if self.cell_at(x, y).walls['W']:
                    plt.plot([x, x], [y, y + 1], c='k', linewidth=linewidth)
                    
        # Draw the North and East maze border, which won't have been drawn
        # by the procedure above.
        plt.plot([0, width - 1], [height, height], c='k', linewidth=linewidth)
        plt.plot([width, width], [0, height], c='k', linewidth=linewidth)
        
    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, -1)),
                 ('N', (0, 1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self, rng):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(0, 0)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = rng.choice(neighbours)
            self.knock_down_wall(current_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1
            
    def knock_down_wall(self, cell, wall):
        cell._knock_down_wall(wall)
        increment = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}[wall]
        neighbor = self.cell_at(cell.x + increment[0], cell.y + increment[1])
        neighbor_wall = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[wall]
        neighbor._knock_down_wall(neighbor_wall)


class MazeEnv(Env):

    def __init__(
        self, name:str, CSpace:List[np.ndarray], robot_radius:float, 
        OStatic:List[StaticPolygon], targets:List[int], maze:Maze, edges:List[Tuple[int, int]]
    ) -> None:
        super().__init__(name, CSpace, robot_radius, OStatic)
        self.targets, self.maze, self.edges = targets, maze, edges

    def draw(self, ax) -> None:
        self.maze.plot(linewidth=1.0)
        for ind in self.targets:
            draw_2d_set(self._CSpace_hpoly[ind], ax, alpha=0.5, color='w')

    @staticmethod
    def make(seed:int=0, maze_size:int=10, knock_downs:int=10, num_rooms_to_visit:int=10) -> MazeEnv:
        rng = np.random.default_rng(seed)

        maze = Maze(maze_size, maze_size)
        maze.make_maze(rng)

        while knock_downs > 0:
            cell = maze.cell_at(rng.integers(1, maze_size - 1), rng.integers(1, maze_size - 1))
            walls = [w for w, up in cell.walls.items() if up]
            if len(walls) > 0:
                maze.knock_down_wall(cell, rng.choice(walls))
                knock_downs -= 1

        regions, edges = [], []
        for x in range(maze_size):
            for y in range(maze_size):
                bounding_verts = np.array([[x, y], [x+1., y], [x+1., y+1.], [x, y+1.]])
                regions.append(bounding_verts)
                C = y + x * maze.ny
                if not maze.map[x][y].walls['N']:
                    edges.append((C, C + 1))
                if not maze.map[x][y].walls['S']:
                    edges.append((C, C - 1))
                if not maze.map[x][y].walls['E']:
                    edges.append((C, C + maze.ny))
                if not maze.map[x][y].walls['W']:
                    edges.append((C, C - maze.ny))
        
        targets = []
        target_size = 0.1
        samples = rng.choice(list(range(len(regions))), size=num_rooms_to_visit, replace=False)
        for sample_idx in samples:
            center = np.mean(regions[sample_idx], axis=0)
            target_box = np.array([
                [center[0] - target_size, center[1] - target_size],
                [center[0] + target_size, center[1] - target_size],
                [center[0] + target_size, center[1] + target_size],
                [center[0] - target_size, center[1] + target_size]
            ])
            regions.append(target_box)
            box_idx = len(regions) - 1
            edges.extend([(sample_idx, box_idx), (box_idx, sample_idx)])
            targets.append(box_idx)
            

        return MazeEnv(
            name=f"maze_{maze_size}x{maze_size}",
            CSpace=regions,
            robot_radius=0.0,
            OStatic=[],
            targets=targets,
            maze=maze,
            edges=edges
        )

