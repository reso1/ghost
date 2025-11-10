from typing import List, Tuple
import time
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial import ConvexHull
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pydrake.all import (
    VPolytope, HPolyhedron, Point, CompositeTrajectory,
)



def timeit(func):

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Func {func.__name__} took {total_time:.3f} secs')
        return result

    return timeit_wrapper


""" convex set operations """

def make_hpolytope(V) -> HPolyhedron:
    dim = V.shape[-1]
    if dim == 1:
        return HPolyhedron.MakeBox([V[0]], [V[1]])

    ch = ConvexHull(V)
    return HPolyhedron(ch.equations[:, :-1], -ch.equations[:, -1])


def time_extruded(hpoly:HPolyhedron, t0:float, tf:float) -> HPolyhedron|None:
    if t0 < tf:
        return hpoly.CartesianProduct(HPolyhedron.MakeBox([t0], [tf]))
    
    return None


def collinear(points:np.ndarray, tol=1e-10):
    if len(points) <= 2:
        return True
        
    for i in range(len(points)-2):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        x3, y3 = points[i+2]
        
        area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2
        if area > tol:
            return False

    return True


def squash_multi_points(hpoly:HPolyhedron, dim:int) -> HPolyhedron:
    A, b = hpoly.A(), hpoly.b()
    num_pts = int(A.shape[-1] // dim)
    num_constraints = int(A.shape[0] // num_pts)
    return HPolyhedron(A[:num_constraints, :dim], b[:num_constraints])


""" visualization """

def draw_2d_set(obj:np.ndarray|HPolyhedron, ax:Axes, color='k', marker='o', linestyle='-', alpha:float=0.3, label=False) -> None:
    if isinstance(obj, HPolyhedron):
        obj = VPolytope(obj).vertices().T

    assert obj.shape[-1] == 2, "2D sets are supported"
    
    if label:
        for idx, v in enumerate(obj):
            ax.plot(v[0], v[1], f'{color}{marker}')
            ax.text(v[0], v[1], str(idx))

    if collinear(obj):
        min_x, min_y = np.min(obj, axis=0)
        max_x, max_y = np.max(obj, axis=0)
        plt.plot([min_x, max_x], [min_y, max_y], 'r-', alpha=alpha)
        ax.plot(obj[:, 0], obj[:, 1], f'{linestyle}{color}', alpha=alpha)
    else:
        hull = ConvexHull(obj)
        verts = obj[hull.vertices]
        ax.fill(verts[:, 0], verts[:, 1], f'{color}', ec='k', alpha=alpha)
            

def draw_3d_set(obj:np.ndarray|HPolyhedron, ax:Axes3D, 
    alpha:float=0.5, fc='lightgray', ec='k', time_scaler:float=1.0) -> None:
    if isinstance(obj, HPolyhedron):
        obj = VPolytope(obj).vertices().T

    assert obj.shape[-1] == 3, "3D sets are supported (2d space + time)"
    if len(obj) < 3:
        return
    
    obj[:, 2] *= time_scaler
    try:
        hull = ConvexHull(obj)
        hull_faces = []
        for simplex in hull.simplices:
            vertices = obj[simplex]
            hull_faces.append(vertices)

        hull_surface = Poly3DCollection(
            hull_faces, alpha=alpha, facecolor=fc, edgecolor=ec)
        hull_surface.set_edgecolor('none')
        
        ax.add_collection3d(hull_surface)

        for simplex in hull.simplices:
            ax.plot(obj[simplex, 0], obj[simplex, 1], obj[simplex, 2], 'k--', alpha=alpha)
    except:
        vertices = [order_points(obj)]
        plane = Poly3DCollection(vertices, alpha=0.5)
        plane.set_facecolor(fc)
        plane.set_edgecolor(ec)
        plane.set_alpha(alpha)
        ax.add_collection3d(plane)


def draw_cuboid(ax:Axes3D, xp:np.ndarray, xq:np.ndarray, halfsize:float=0.05, color='k', alpha=0.5) -> None:
    p_facet_verts = [[xp[0] - halfsize, xp[1] - halfsize, xp[2]], 
                     [xp[0] + halfsize, xp[1] - halfsize, xp[2]], 
                     [xp[0] + halfsize, xp[1] + halfsize, xp[2]], 
                     [xp[0] - halfsize, xp[1] + halfsize, xp[2]]]
    q_facet_verts = [[xq[0] - halfsize, xq[1] - halfsize, xq[2]],
                     [xq[0] + halfsize, xq[1] - halfsize, xq[2]],
                     [xq[0] + halfsize, xq[1] + halfsize, xq[2]],
                     [xq[0] - halfsize, xq[1] + halfsize, xq[2]]]
    
    # draw cuboid
    cuboid = np.array(p_facet_verts + q_facet_verts)
    draw_3d_set(cuboid, ax, alpha=alpha, fc=color)

    # draw xp --- xq
    vec = xq - xp
    offset = 0.0 # 0.1
    st = xp - offset * vec
    et = xq + offset * vec
    ax.plot([st[0], et[0]], [st[1], et[1]], [st[2], et[2]], 'x-k', lw=3, ms=10)


def draw_cylinder(ax:Axes3D, P:np.ndarray, Q:np.ndarray, r:float, alpha:float=0.5, num_points=20) -> None:
    P = np.array(P)
    Q = np.array(Q)
    
    v = Q - P
    length = np.linalg.norm(v)
    v = v / length
    
    if v[0] == 0 and v[1] == 0:
        n1 = np.array([1, 0, 0])
    else:
        n1 = np.array([-v[1], v[0], 0])
        n1 = n1 / np.linalg.norm(n1)
    
    n2 = np.cross(v, n1)
    
    theta = np.linspace(0, 2*np.pi, num_points)
    h = np.linspace(0, length, num_points)
    
    theta_grid, h_grid = np.meshgrid(theta, h)
    
    X = P[0] + h_grid*v[0] + r*np.cos(theta_grid)*n1[0] + r*np.sin(theta_grid)*n2[0]
    Y = P[1] + h_grid*v[1] + r*np.cos(theta_grid)*n1[1] + r*np.sin(theta_grid)*n2[1]
    Z = P[2] + h_grid*v[2] + r*np.cos(theta_grid)*n1[2] + r*np.sin(theta_grid)*n2[2]
        
    surf = ax.plot_surface(X, Y, Z, color='k', alpha=alpha)


def order_points(points:np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    centered_points = points - center
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[2]
    projection_matrix = np.eye(3) - normal[:, np.newaxis] * normal
    basis_1, basis_2 = vh[0], vh[1]
    points_2d = np.column_stack([
        np.dot(centered_points, basis_1),
        np.dot(centered_points, basis_2)
    ])
    
    hull = ConvexHull(points_2d)
    return points[hull.vertices]


def anim_rotating_camera(stgcs, sol, name="video") -> None:
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.axis("off")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_zticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_zlim(0, 2)

    ax.view_init(elev=20, azim=0)
    stgcs.draw(ax, set_labels=False)
    for wp in sol.trajectory:
        ax.plot([wp[0], wp[3]], [wp[1], wp[4]], [wp[2], wp[5]], '-ok')

    def update(frame):
        ax.view_init(elev=20, azim=frame)
        return ax,

    anim = animation.FuncAnimation(
        fig, update, np.linspace(0, 360, 120), interval=50,
        blit=True, repeat=True, cache_frame_data=False)

    # FFwriter=animation.FFMpegWriter(fps=60, extra_args=['-vcodec', 'libx264'])
    # fp = os.path.join(os.getcwd(), f'{name}.mp4')
    # anim.save(fp, writer=FFwriter, dpi=200,
    #             progress_callback=lambda i, n: print(f'saving frame {i}/{n}'))


def draw_parallelpiped(ax:Axes, xp:np.ndarray, xq:np.ndarray, halfsize:float):
    P = np.array([[xp[0] - halfsize, xp[1]], [xp[0] + halfsize, xp[1]],
                  [xq[0] + halfsize, xq[1]], [xq[0] - halfsize, xq[1]]])
    
    ax.fill(P[:, 0], P[:,1], color='k', alpha=0.5)


def draw_trajectory(ax:Axes, traj:CompositeTrajectory, color='k', linestyle='-') -> None:
    T = np.linspace(traj.start_time(), traj.end_time(), 100)
    X = np.array([traj.value(t) for t in T])
    ax.plot(X[:, 0], X[:, 1], f'{color}{linestyle}')
    ax.plot(X[0, 0], X[0, 1], 'X', color='r')
    ax.plot(X[-1, 0], X[-1, 1], 'X', color='g')

