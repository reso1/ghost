from __future__ import annotations
from typing import List, Dict, Generator, Tuple
from itertools import combinations

import time, pickle

from gcspy.graphs import GraphOfConvexSets
from pydrake.all import HPolyhedron, LinearEqualityConstraint

import numpy as np
import cvxpy as cp

from src.gcs.base import BaseGCS, BaseTrajectory
from src.gcs.bezier import BezierGCS, Configuration
from src.lower_bound_graph import CompleteGraph
from src.path_unfolding import PruningCost, gcs_convex_restriction, multi_label_A_star
from src.rtsp import rTSP, Tour
from src.env import Env

import heapq

import logging
logger = logging.getLogger(__name__)


class EuclideanDistGraph(CompleteGraph):

    def __init__(self) -> None:
        super().__init__()
        self.simple_paths: Dict[Tuple[int, int], List[List[int]]] = {}

    @staticmethod
    def build(gcs: BaseGCS) -> EuclideanDistGraph:
        ecg = EuclideanDistGraph()

        centers = []
        for hpoly in gcs.regions:
            centers.append(hpoly.ChebyshevCenter())
        
        ecg.cost_matrix = np.zeros((len(centers), len(centers))) 
        for c1, c2 in combinations(range(len(centers)), 2):
            ecg.cost_matrix[c1, c2] = np.linalg.norm(centers[c1] - centers[c2])
            ecg.cost_matrix[c2, c1] = ecg.cost_matrix[c1, c2]

        for u, v in gcs.nx_diG.edges():
            ecg.G.add_edge(u, v)
        
        ecg.simple_paths = CompleteGraph.compute_all_simple_paths(ecg.G)

        return ecg

    def save(self, filename:str) -> None:
        """ Save the graph to a file """
        with open(filename + '.ecg', 'wb') as f:
            pickle.dump({
                "cost_matrix": self.cost_matrix,
                "simple_paths": self.simple_paths,
                "G": self.G,
            }, f)

    @staticmethod
    def load(filename:str) -> EuclideanDistGraph:
        ecg = EuclideanDistGraph()
        with open(filename + '.ecg', 'rb') as f:
            data = pickle.load(f)
            ecg.cost_matrix = data["cost_matrix"]
            ecg.simple_paths = data["simple_paths"]
            ecg.G = data["G"]
        
        logger.info(f"Loaded EuclideanDistGraph from {filename}.ecg")
        return ecg


def greedy_tour_generator(lbg:CompleteGraph, gcs:BaseGCS) -> Generator[Tour, None, None]:
    stack = [([], list(gcs.nx_diG.nodes))]
    
    while stack:
        path, unvisited = stack.pop()
        if len(unvisited) == 0:
            yield Tour(path + [path[0]])
            continue
        
        heur = lambda v: lbg.cost_matrix[path[-2], path[-1], v] if len(path) >= 2 else 0.0
        for v in sorted(unvisited, key=heur, reverse=True):
            unvisited_cp = unvisited.copy()
            unvisited_cp.remove(v)
            stack.append((path + [v], unvisited_cp))


def lbg_greedy(lbg:CompleteGraph, gcs:BaseGCS, runtime_limit:float) -> Tuple[BaseTrajectory, float]:
    ts = time.perf_counter()
    best_traj = None
    best_cost = float('inf')
    gcs_cost_rec = {}
    pruning_cost = PruningCost(best_cost, 0.0)

    for tour in greedy_tour_generator(lbg, gcs):
        for unfolded, _ in multi_label_A_star(tour, lbg, pruning_cost):
            if time.perf_counter() - ts > runtime_limit:
                return best_traj, best_cost

            path_key = tuple(unfolded)
            if path_key in gcs_cost_rec:
                traj_cost = gcs_cost_rec[path_key]
            else:
                traj = gcs_convex_restriction(unfolded, gcs)
                if traj is None:
                    traj_cost = float('inf')
                else:
                    traj_cost = traj.time_cost
                    gcs_cost_rec[path_key] = traj_cost
            
            if traj_cost < best_cost:
                best_traj, best_cost = traj, traj_cost
                pruning_cost.update(best_cost)
        
        if time.perf_counter() - ts > runtime_limit:
            return best_traj, best_cost
            
    return best_traj, best_cost


def micp_cg_base(env:Env):
    # add MIPGap (Gurobi) for https://github.com/TobiaMarcucci/gcspy
    gcs = GraphOfConvexSets()

    for i, r in enumerate(env._CSpace_hpoly):
        v = gcs.add_vertex(f"v{i}")
        x = v.add_variable(env.dim)
        v.add_constraint(r.A() @ x <= r.b())

    for tail in gcs.vertices:
        for head in gcs.vertices:
            if tail != head:
                edge = gcs.add_edge(tail, head)
                edge.add_cost(cp.norm(tail.variables[0] - head.variables[0], 2))
    
    prob = gcs.solve_traveling_salesman()
    return prob


class Floor(GraphOfConvexSets):

    def __init__(self, sets:List[HPolyhedron], edges:List[Tuple[int, int]], name):
        super().__init__()
        self.name = name
        for i, r in enumerate(sets):
            v = self.add_vertex(f"{self.name}_{i}")
            x1 = v.add_variable(2)
            x2 = v.add_variable(2)
            v.add_constraint(r.A() @ x1 <= r.b())
            v.add_constraint(r.A() @ x2 <= r.b())
            v.add_cost(cp.norm(x2 - x1, 2))

        for tail_idx, head_idx in edges:
            tail = self.get_vertex(f"{self.name}_{tail_idx}")
            head = self.get_vertex(f"{self.name}_{head_idx}")
            e = self.add_edge(tail, head)
            e.add_constraint(tail.variables[1] == head.variables[0])


def micp_ncg_linear_floor_embedding(env:Env):
    edges = []
    for i, ri in enumerate(env._CSpace_hpoly):
        for j , rj in enumerate(env._CSpace_hpoly):
            if i != j and ri.IntersectsWith(rj):
                edges.append((i, j))

    gcs = GraphOfConvexSets()

    visit_rooms = list(range(len(env._CSpace_hpoly)))

    num_floors = len(visit_rooms)
    for floor in range(num_floors):
        gcs.add_disjoint_subgraph(Floor(env._CSpace_hpoly, edges, floor))

    def connect_floors(floor1, floor2, room):
        tail = gcs.get_vertex(f"{floor1}_{room}")
        head = gcs.get_vertex(f"{floor2}_{room}")
        edge = gcs.add_edge(tail, head)
        edge.add_constraint(tail.variables[1] == head.variables[0])

    first_room = visit_rooms[0]
    first_floor = 0
    last_floor = num_floors - 1
    connect_floors(last_floor, first_floor, first_room)
    for floor in range(last_floor):
        for room in visit_rooms[1:]:
            connect_floors(floor, floor + 1, room)

    ilp_constraints = []
    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    for i, v in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_edge_indices(v)
        out_edges = gcs.outgoing_edge_indices(v)
        ilp_constraints.append(yv[i] == sum(ye[inc_edges]))
        ilp_constraints.append(yv[i] == sum(ye[out_edges]))
        
    def get_flow(floor1, floor2, room):
        tail_name = f"{floor1}_{room}"
        head_name = f"{floor2}_{room}"
        edge = gcs.get_edge(tail_name, head_name)
        return ye[gcs.edge_index(edge)]

    ilp_constraints.append(get_flow(last_floor, first_floor, first_room) == 1)
    for room in visit_rooms[1:]:
        flow = sum(get_flow(floor, floor + 1, room) for floor in range(last_floor))
        ilp_constraints.append(flow == 1)
    
    prob = gcs.solve_from_ilp(ilp_constraints)
    return prob


class FloorBezier(GraphOfConvexSets):

    def __init__(self, sets:List[HPolyhedron], edges:List[Tuple[int, int]], name, bgcs:BezierGCS):
        super().__init__()
        self.name = name
        order = bgcs.order
        for i, r in enumerate(sets):
            v = self.add_vertex(f"{self.name}_{i}")
            x = v.add_variable(3 * (1 + bgcs.order))
            timed_hpoly = r.CartesianPower(order + 1).CartesianProduct(bgcs.time_scaling_set)
            v.add_constraint(timed_hpoly.A() @ x <= timed_hpoly.b())

            for c in bgcs.get_vlimit_constraints(bgcs.cfg.vmin, bgcs.cfg.vmax):
                v.add_constraint(c.get_sparse_A() @ x <= c.upper_bound())

            time_cost = bgcs.get_time_cost(bgcs.cfg.time_cost)
            v.add_cost(time_cost.a() @ x + time_cost.b())

        for tail_idx, head_idx in edges:
            tail = self.get_vertex(f"{self.name}_{tail_idx}")
            head = self.get_vertex(f"{self.name}_{head_idx}")
            e = self.add_edge(tail, head)
            xe = cp.concatenate([tail.variables[0], head.variables[0]])
            for c in bgcs.edge_constraints:
                assert type(c) == LinearEqualityConstraint
                e.add_constraint(c.get_sparse_A() @ xe == c.upper_bound())


def micp_ncg_bezier_floor_embedding(env:Env, cfg:Configuration):
    bgcs = BezierGCS.build(env, cfg)
    edges = []
    for i, ri in enumerate(env._CSpace_hpoly):
        for j , rj in enumerate(env._CSpace_hpoly):
            if i != j and ri.IntersectsWith(rj):
                edges.append((i, j))

    gcs = GraphOfConvexSets()
    visit_rooms = list(range(len(env._CSpace_hpoly)))

    num_floors = len(visit_rooms)
    for floor in range(num_floors):
        gcs.add_disjoint_subgraph(
            FloorBezier(env._CSpace_hpoly, edges, floor, bgcs))

    def connect_floors(floor1, floor2, room):
        tail = gcs.get_vertex(f"{floor1}_{room}")
        head = gcs.get_vertex(f"{floor2}_{room}")
        e = gcs.add_edge(tail, head)

        if floor1 < floor2:
            xe = cp.concatenate([tail.variables[0], head.variables[0]])
            for c in bgcs.edge_constraints:
                assert type(c) == LinearEqualityConstraint
                e.add_constraint(c.get_sparse_A() @ xe == c.upper_bound())
        else:
            tail_xy_last = tail.variables[0][2 * cfg.order : 2 * (1 + cfg.order)]
            head_xy_first = head.variables[0][:2]
            e.add_constraint(tail_xy_last == head_xy_first)

    first_room = visit_rooms[0]
    first_floor = 0
    last_floor = num_floors - 1
    connect_floors(last_floor, first_floor, first_room)
    for floor in range(last_floor):
        for room in visit_rooms[1:]:
            connect_floors(floor, floor + 1, room)

    ilp_constraints = []
    yv = gcs.vertex_binaries()
    ye = gcs.edge_binaries()

    for i, v in enumerate(gcs.vertices):
        inc_edges = gcs.incoming_edge_indices(v)
        out_edges = gcs.outgoing_edge_indices(v)
        ilp_constraints.append(yv[i] == sum(ye[inc_edges]))
        ilp_constraints.append(yv[i] == sum(ye[out_edges]))
        
    def get_flow(floor1, floor2, room):
        tail_name = f"{floor1}_{room}"
        head_name = f"{floor2}_{room}"
        edge = gcs.get_edge(tail_name, head_name)
        return ye[gcs.edge_index(edge)]

    ilp_constraints.append(get_flow(last_floor, first_floor, first_room) == 1)
    for room in visit_rooms[1:]:
        flow = sum(get_flow(floor, floor + 1, room) for floor in range(last_floor))
        ilp_constraints.append(flow == 1)
    
    prob = gcs.solve_from_ilp(ilp_constraints)
    return prob
