from __future__ import annotations
from typing import List, Tuple, Dict
from itertools import combinations

import numpy as np
import networkx as nx

from pydrake.all import (
    HPolyhedron, Point,
    Binding, Constraint, Cost, L2NormCost,
    GraphOfConvexSets as GCS, GraphOfConvexSetsOptions,
    CommonSolverOption, MosekSolver, SolverOptions,
)

from src.env import Env
from src.gcs.rounding import MipPathExtraction


class BaseGCS:

    def __init__(self, regions:List[HPolyhedron]) -> None:

        self.regions = regions
        self.dimension = self.regions[0].ambient_dimension()
        
        self.source, self.target = None, None
        self.nx_diG = nx.DiGraph()
        self.gcs = GCS()

        self.rounding_fn = []
        self.rounding_kwargs = {}

        self.options = GraphOfConvexSetsOptions()
        self.options.solver_options = SolverOptions()
        self.options.solver_options.SetOption(CommonSolverOption.kPrintToConsole, 0)
        self.options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        self.options.solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        self.options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        self.options.solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME", 3600.0)

        self.vertex_costs, self.vertex_constraints = [], []
        self.edge_costs, self.edge_constraints = [], []
    
    @staticmethod
    def build(env: Env) -> BaseGCS:
        gcs = BaseGCS(env._CSpace_hpoly.copy())

        gcs.edge_costs.append(
            L2NormCost(
                A = np.hstack((-np.eye(gcs.dimension, gcs.dimension), np.eye(gcs.dimension, gcs.dimension))),
                b = np.zeros(gcs.dimension))
        )

        for index, region in enumerate(env._CSpace_hpoly):
            gcs.add_vertex(region, region, index)
        
        for u, v in combinations(gcs.nx_diG.nodes, 2):
            Xu, Xv = gcs.nx_diG.nodes[u]["set"], gcs.nx_diG.nodes[v]["set"]
            if u != v and Xu.IntersectsWith(Xv):
                gcs.add_edge(u, v)
                gcs.add_edge(v, u)
        
        return gcs

    @staticmethod
    def build_complete_graph(env: Env) -> BaseGCS:
        gcs = BaseGCS(env._CSpace_hpoly.copy())

        gcs.edge_costs.append(
            L2NormCost(
                A = np.hstack((-np.eye(gcs.dimension, gcs.dimension), np.eye(gcs.dimension, gcs.dimension))),
                b = np.zeros(gcs.dimension))
        )

        for index, region in enumerate(env._CSpace_hpoly):
            gcs.add_vertex(region, index)

        for u, v in combinations(gcs.nx_diG.nodes, 2):
            if u != v:
                gcs.add_edge(u, v)
                gcs.add_edge(v, u)
        
        return gcs

    def __solve_GCS__(self, rounding, preprocessing, verbose):

        results_dict = {}
        self.options.convex_relaxation = rounding
        self.options.preprocessing = preprocessing
        self.options.max_rounded_paths = 0

        result = self.gcs.SolveShortestPath(self.source, self.target, self.options)

        if rounding:
            results_dict["relaxation_result"] = result
            results_dict["relaxation_solver_time"] = result.get_solver_details().optimizer_time
            results_dict["relaxation_cost"] = result.get_optimal_cost()
        else:
            results_dict["mip_result"] = result
            results_dict["mip_solver_time"] = result.get_solver_details().optimizer_time
            results_dict["mip_cost"] = result.get_optimal_cost()

        if not result.is_success():
            print("First solve failed")
            return None, None, results_dict

        if verbose:
            print("Solution\t",
                  "Success:", result.get_solution_result(),
                  "Cost:", result.get_optimal_cost(),
                  "Solver time:", result.get_solver_details().optimizer_time)

        # Solve with hard edge choices
        if rounding and len(self.rounding_fn) > 0:
            # Extract path
            active_edges = []
            found_path = False
            for fn in self.rounding_fn:
                rounded_edges = fn(self.gcs, result, self.source, self.target,
                                   **self.rounding_kwargs)
                if rounded_edges is None:
                    print(fn.__name__, "could not find a path.")
                    active_edges.append(rounded_edges)
                else:
                    found_path = True
                    active_edges.extend(rounded_edges)
            results_dict["rounded_paths"] = active_edges
            if not found_path:
                print("All rounding strategies failed to find a path.")
                return None, None, results_dict

            self.options.preprocessing = False
            rounded_results = []
            best_cost = np.inf
            best_path = None
            best_result = None
            max_rounded_solver_time = 0.0
            total_rounded_solver_time = 0.0
            for path_edges in active_edges:
                if path_edges is None:
                    rounded_results.append(None)
                    continue
                for edge in self.gcs.Edges():
                    if edge in path_edges:
                        edge.AddPhiConstraint(True)
                    else:
                        edge.AddPhiConstraint(False)
                rounded_results.append(self.gcs.SolveShortestPath(
                    self.source, self.target, self.options))
                solve_time = rounded_results[-1].get_solver_details().optimizer_time
                max_rounded_solver_time = np.maximum(solve_time, max_rounded_solver_time)
                total_rounded_solver_time += solve_time
                if (rounded_results[-1].is_success()
                    and rounded_results[-1].get_optimal_cost() < best_cost):
                    best_cost = rounded_results[-1].get_optimal_cost()
                    best_path = path_edges
                    best_result = rounded_results[-1]

            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = rounded_results
            results_dict["max_rounded_solver_time"] =  max_rounded_solver_time
            results_dict["total_rounded_solver_time"] = total_rounded_solver_time
            results_dict["rounded_cost"] = best_result.get_optimal_cost()

            if verbose:
                print("Rounded Solutions:")
                for r in rounded_results:
                    if r is None:
                        print("\t\tNo path to solve")
                        continue
                    print("\t\t",
                        "Success:", r.get_solution_result(),
                        "Cost:", r.get_optimal_cost(),
                        "Solver time:", r.get_solver_details().optimizer_time)

            if best_path is None:
                print("Second solve failed on all paths.")
                return best_path, best_result, results_dict
        elif rounding:
            self.options.max_rounded_paths = 10

            rounded_result = self.gcs.SolveShortestPath(self.source, self.target, self.options)
            best_path = MipPathExtraction(self.gcs, rounded_result, self.source, self.target)[0]
            best_result = rounded_result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = [rounded_result]
            results_dict["rounded_cost"] = best_result.get_optimal_cost()
        else:
            best_path = MipPathExtraction(self.gcs, result, self.source, self.target)[0]
            best_result = result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["mip_path"] = best_path

        if verbose:
            for edge in best_path:
                print("Added", edge.name(), "to path.")

        return best_path, best_result, results_dict

    def add_vertex(self, region:HPolyhedron, index:int) -> GCS.Vertex:
        return self._add_vertex(region, region, index)

    def _add_vertex(self, hpoly:HPolyhedron, region:HPolyhedron, index:int) -> GCS.Vertex:
        v = self.gcs.AddVertex(hpoly, name = f"v{index}")
        for cost in self.vertex_costs:
            v.AddCost(Binding[Cost](cost, v.x()))
        for cstr in self.vertex_constraints:
            v.AddConstraint(Binding[Constraint](cstr, v.x()))
        self.nx_diG.add_node(index, set=region, vertex=v)
        return v

    def remove_vertex(self, index:int) -> None:
        assert index in self.nx_diG.nodes
        self.gcs.RemoveVertex(self.nx_diG.nodes[index]["vertex"])
        self.nx_diG.remove_node(index)
    
    def add_edge(self, tail_index:int, head_index:int) -> GCS.Edge:
        tail: GCS.Vertex = self.nx_diG.nodes[tail_index]["vertex"]
        head: GCS.Vertex = self.nx_diG.nodes[head_index]["vertex"]
        e = self.gcs.AddEdge(tail, head, f"({tail}, {head})")
        for cost in self.edge_costs:
            e.AddCost(Binding[Cost](cost, np.append(tail.x(), head.x())))
        for cstr in self.edge_constraints:
            e.AddConstraint(Binding[Constraint](cstr, np.append(tail.x(), head.x())))
        self.nx_diG.add_edge(tail_index, head_index, e=e)
        return e

    def solve(
        self, x0_set_idx:int, xt_set_idx:int,
        rounding:bool=False, preprocessing:bool=False, verbose:bool=False,
    ) -> Tuple[List[int]|None, BaseTrajectory|None, float]:
        
        self.source = self.gcs.AddVertex(self.regions[x0_set_idx], "source")
        self.target = self.gcs.AddVertex(self.regions[xt_set_idx], "target")

        e_src = self.gcs.AddEdge(self.source, self.nx_diG.nodes[x0_set_idx]["vertex"], name='esource')
        e_tar = self.gcs.AddEdge(self.nx_diG.nodes[xt_set_idx]["vertex"], self.target, name='etarget')
        for jj in range(self.dimension):
            e_src.AddConstraint(e_src.xu()[jj] == e_src.xv()[jj])
            e_tar.AddConstraint(e_tar.xu()[jj] == e_tar.xv()[jj])
        
        best_path, best_result, results_dict = self.__solve_GCS__(rounding, preprocessing, verbose)

        if best_result is None:
            traj, vertex_path, cost = None, None, np.inf
        else:
            traj = BaseTrajectory([best_result.GetSolution(e.xv()) for e in best_path])
            vertex_path = [e.v().name() for e in best_path]
            vertex_path = [int(v[1:]) for v in vertex_path[:-1]]
            cost = traj.time_cost
        
        self.gcs.RemoveVertex(self.source)
        self.gcs.RemoveVertex(self.target)
        self.source = self.target = None

        return vertex_path, traj, cost
            
    def solve_convex_restriction(self, path:List[int]) -> BaseTrajectory:
        E = [self.nx_diG.edges[path[i], path[i+1]]["e"] for i in range(len(path)-1)]
        res = self.gcs.SolveConvexRestriction(E, self.options)
        traj = BaseTrajectory([res.GetSolution(e.xu()) for e in E] + [res.GetSolution(E[-1].xv())])
        return traj


class BaseTrajectory:
    
    def __init__(self, points: List[np.ndarray]) -> None:
        self.points = points

    @property
    def size(self) -> int:
        return len(self.points)
    
    @property
    def time_cost(self) -> float:
        return sum(np.linalg.norm(self.points[i+1] - self.points[i]) for i in range(len(self.points) - 1))
