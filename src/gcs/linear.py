from __future__ import annotations
from typing import List, Tuple
import numpy as np
from itertools import combinations

from pydrake.all import L2NormCost, LinearEqualityConstraint

from src.gcs.base import BaseGCS, BaseTrajectory
from src.env import Env


class LinearGCS(BaseGCS):

    def __init__(self, env:Env):
        super(LinearGCS, self).__init__(env._CSpace_hpoly.copy())
        self.order = 1

        path_cost = L2NormCost(
            A = np.block([np.eye(self.dimension), -np.eye(self.dimension)]),
            b = np.zeros(self.dimension))
        self.vertex_costs.append(path_cost)

        cont_cstr = LinearEqualityConstraint(
            Aeq = np.block([np.zeros((self.dimension, self.dimension)), np.eye(self.dimension), 
                            -np.eye(self.dimension), np.zeros((self.dimension, self.dimension))]),
            beq = np.zeros(self.dimension))
        self.edge_constraints.append(cont_cstr)
        

    def add_vertex(self, region, index):
        comp_hpoly = region.CartesianPower(self.order + 1)
        return super()._add_vertex(comp_hpoly, region, index)
    
    @staticmethod
    def build(env:Env):
        gcs = LinearGCS(env)
        for index, region in enumerate(env._CSpace_hpoly):
            gcs.add_vertex(region, index)
        
        for u, v in combinations(gcs.nx_diG.nodes, 2):
            Xu, Xv = gcs.nx_diG.nodes[u]["set"], gcs.nx_diG.nodes[v]["set"]
            if u != v and Xu.IntersectsWith(Xv):
                gcs.add_edge(u, v)
                gcs.add_edge(v, u)
        
        return gcs

    def solve(
        self, x0_set_idx:int, xt_set_idx:int,
        rounding=False, verbose=False, preprocessing=False
    ) -> Tuple[List[int]|None, LinearTrajectory|None, float]:
        
        self.source = self.gcs.AddVertex(self.regions[x0_set_idx], "source")
        self.target = self.gcs.AddVertex(self.regions[xt_set_idx], "target")

        e_src = self.gcs.AddEdge(self.source, self.nx_diG.nodes[x0_set_idx]["vertex"], name='esource')
        e_tar = self.gcs.AddEdge(self.nx_diG.nodes[xt_set_idx]["vertex"], self.target, name='etarget')

        for jj in range(self.dimension):
            e_src.AddConstraint(e_src.xu()[jj] == e_src.xv()[jj])
        for jj in range(self.dimension):
            e_tar.AddConstraint(e_tar.xu()[-(self.dimension + self.order + 1) + jj] == e_tar.xv()[jj])

        best_path, best_result, results_dict = self.__solve_GCS__(rounding, preprocessing, verbose)

        if best_result is None:
            traj, vertex_path, cost = None, None, np.inf
        else:
            traj = LinearTrajectory([best_result.GetSolution(e.xv()) for e in best_path])
            vertex_path = [e.v().name() for e in best_path]
            vertex_path = [int(v[1:]) for v in vertex_path[:-1]]
            cost = traj.time_cost
        
        self.gcs.RemoveVertex(self.source)
        self.gcs.RemoveVertex(self.target)
        self.source = self.target = None

        return vertex_path, traj, cost

    def solve_convex_restriction(self, path:List[int]) -> LinearTrajectory:
        E = [self.nx_diG.edges[path[i], path[i+1]]["e"] for i in range(len(path)-1)]
        res = self.gcs.SolveConvexRestriction(E, self.options)
        traj = LinearTrajectory([res.GetSolution(e.xu()) for e in E] + [res.GetSolution(E[-1].xv())], self.dimension)
        return traj


class LinearTrajectory(BaseTrajectory):
    def __init__(self, points:List[np.ndarray], dim:int):
        super().__init__(points)
        self.dim = dim

    @property
    def size(self) -> int:
        return len(self.points)

    @property
    def time_cost(self) -> float:
        return sum([np.linalg.norm(pt[:self.dim] - pt[-self.dim:]) for pt in self.points])
