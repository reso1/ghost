import pickle
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx

from scipy.spatial import ConvexHull

from src.env import Env
from src.utils import make_hpolytope


def random_cg_env_2d(seed:int, N:int, M:int, num_sets:int, num_pts_per_ply:int=6) -> Env:
    rng = np.random.default_rng(seed)
    # generate num_sets from NxN polygons in a NxN grid
    inds_1d = rng.choice(N*M, size=num_sets, replace=False)
    inds_2d = [(i // M, i % M) for i in inds_1d]
    polygons = []
    for i, j in inds_2d:
        # generate random points and get the convex hull
        points = 1.5 * rng.random((num_pts_per_ply, 2))
        hull = ConvexHull(points)
        vertices = points[hull.vertices] + np.array([i, j])
        polygons.append(vertices)

    return Env(f"rnd_cg_{seed}_sets{num_sets}_{N}x{M}", polygons, robot_radius=0.05)


def random_ncg_env_2d(seed:int, num_edges:int, num_pts_per_ply:int=6) -> Env:
    spiral_inds = []
    layer, N = 1, 100
    while len(spiral_inds) < N:
        if layer == 1:
            spiral_inds.append((1, 1))
        else:
            for col in range(1, layer + 1):
                if len(spiral_inds) >= N:
                    break
                spiral_inds.append((layer, col))
            for row in range(layer - 1, 0, -1):
                if len(spiral_inds) >= N:
                    break
                spiral_inds.append((row, layer))
        
        layer += 1
    
    rng = np.random.default_rng(seed)
    
    G = nx.Graph()
    polygons = []
    for i, j in spiral_inds:
        # generate random points and get the convex hull
        edge_found, num_resamples = False, 100
        while not edge_found and num_resamples > 0:
            num_resamples -= 1
            if (i, j) in G.nodes:
                G.remove_node((i, j))

            points = 1.5 * rng.random((num_pts_per_ply, 2))
            hull = ConvexHull(points)
            vertices = points[hull.vertices] + np.array([i, j])
            hpoly = make_hpolytope(vertices)
            
            G.add_node((i, j), hpoly=hpoly, polygon=vertices)
            if (i, j) == (1, 1):
                edge_found = True
            else:
                for node in G.nodes:
                    if node != (i, j) and hpoly.IntersectsWith(G.nodes[node]['hpoly']):
                        G.add_edge(node, (i, j))
                        edge_found = True
        
        if not edge_found:
            return random_ncg_env_2d(seed + 100, num_edges, num_pts_per_ply)
            
        if G.number_of_edges() >= num_edges:
            break

    polygons = [G.nodes[node]['polygon'] for node in G.nodes]
    return Env(f"rnd_ncg_{seed}_edges{num_edges}", polygons, robot_radius=0.05)


if __name__ == "__main__":
    N_list = [int(n) for n in range(5, 26, 1)]
    num_seeds = 12
    grid_n, grid_m = 5, 5

    for seed in range(num_seeds):
        for N in N_list:
            env = random_cg_env_2d(seed, grid_n, grid_m, N)
            with open(f"../data/envs/cg/{seed}_{N}.pkl", 'wb') as f:
                pickle.dump(env, f)
            env = random_ncg_env_2d(seed, N)
            with open(f"../data/envs/ncg/{seed}_{N}.pkl", 'wb') as f:
                pickle.dump(env, f)
