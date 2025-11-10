from __future__ import annotations
from typing import List, Tuple, Dict, Generator
from collections import defaultdict

from src.gcs.base import BaseGCS, BaseTrajectory
from src.lower_bound_graph import LowerBoundGraph
from src.rtsp import Tour
from src.gcs.linear import LinearGCS
from src.gcs.bezier import BezierGCS

from pydrake.all import LinearEqualityConstraint, Binding, Constraint, DecomposeLinearExpressions

import heapq

import numpy as np


# GCS Convex Resctriction with vertex duplication

def gcs_convex_restriction(path:List[int], gcs:BaseGCS) -> BaseTrajectory|None:
    if path[0] == path[-1]:
        # point in first set must equal point in last set
        return _gcr_closed_path(path, gcs)
    else:
        return _gcr_open_path(path, gcs)


def _gcr_closed_path(path:List[int], gcs:BaseGCS) -> BaseTrajectory|None:
    assert path[0] == path[-1], "path must be a closed loop (start and end vertices must be the same)"

    v_count:Dict[int, int] = defaultdict(int)
    v_dup_index = len(gcs.nx_diG.nodes)
    path_augmented, to_rmv, last_v_dup_index = [], [], None
    for vid in path[:-1]:
        v_count[vid] += 1
        if v_count[vid] == 1:
            path_augmented.append(vid)
            continue

        gcs.add_vertex(gcs.regions[vid], index=v_dup_index)
        for pred in gcs.nx_diG.predecessors(vid):
            gcs.add_edge(pred, v_dup_index)
        for succ in gcs.nx_diG.successors(vid):
            gcs.add_edge(v_dup_index, succ)
        if last_v_dup_index is not None:
            gcs.add_edge(last_v_dup_index, v_dup_index)

        path_augmented.append(v_dup_index)
        to_rmv.append(v_dup_index)
        last_v_dup_index = v_dup_index
        v_dup_index += 1
    
    if type(gcs) is BaseGCS:
        root = path_augmented[0]
        traj = gcs.solve_convex_restriction(path_augmented + [root])
    elif type(gcs) is LinearGCS:
        root, dummy_root = path_augmented[0], len(gcs.nx_diG.nodes)
        dummy_root_vert = gcs.add_vertex(gcs.regions[root], dummy_root)
        gcs.add_edge(path_augmented[-1], dummy_root)
        to_rmv.append(dummy_root)
        
        root_vert = gcs.nx_diG.nodes[root]['vertex']
        edge = gcs.gcs.AddEdge(dummy_root_vert, root_vert, "etar")
        gcs.nx_diG.add_edge(dummy_root, root, e=edge)

        for jj in range(gcs.dimension * (gcs.order + 1)):
            edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

        traj = gcs.solve_convex_restriction(path_augmented + [dummy_root, root])
    elif type(gcs) is BezierGCS:
        root, dummy_root = path_augmented[0], len(gcs.nx_diG.nodes)
        dummy_root_vert = gcs.add_vertex(gcs.regions[root], dummy_root)
        gcs.add_edge(path_augmented[-1], dummy_root)
        to_rmv.append(dummy_root)
        
        vert = gcs.nx_diG.nodes[path_augmented[1]]['vertex']
        edge = gcs.gcs.AddEdge(dummy_root_vert, vert, "etar")
        gcs.nx_diG.add_edge(dummy_root, path_augmented[1], e=edge)

        for jj in range(gcs.dimension):
            edge.AddConstraint(edge.xu()[jj + gcs.dimension * gcs.order] == edge.xv()[jj])

        u_path_control = gcs.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = gcs.u_h_trajectory.MakeDerivative(1).control_points()
        vt_error = np.squeeze(u_path_control[-1]) - 0 * np.squeeze(u_time_control[-1])
        vt_cstr = LinearEqualityConstraint(
            DecomposeLinearExpressions(vt_error, gcs.u_vars), np.zeros(gcs.dimension))
        edge.AddConstraint(Binding[Constraint](vt_cstr, edge.xu()))

        traj = gcs.solve_convex_restriction(path_augmented + [dummy_root, path_augmented[1]])
    else:
        raise TypeError(f"Unsupported GCS type: {type(gcs)}")

    for vid in to_rmv:
        gcs.remove_vertex(vid)
    
    return traj


def _gcr_open_path(path:List[int], gcs:BaseGCS) -> BaseTrajectory|None:
    v_count:Dict[int, int] = defaultdict(int)
    v_dup_index = len(gcs.nx_diG.nodes)
    path_augmented, to_rmv, last_v_dup_index = [], [], None
    for vid in path:
        v_count[vid] += 1
        if v_count[vid] == 1:
            path_augmented.append(vid)
            continue

        gcs.add_vertex(gcs.regions[vid], index=v_dup_index)
        for pred in gcs.nx_diG.predecessors(vid):
            gcs.add_edge(pred, v_dup_index)
        for succ in gcs.nx_diG.successors(vid):
            gcs.add_edge(v_dup_index, succ)
        if last_v_dup_index is not None:
            gcs.add_edge(last_v_dup_index, v_dup_index)

        path_augmented.append(v_dup_index)
        to_rmv.append(v_dup_index)
        last_v_dup_index = v_dup_index
        v_dup_index += 1
    
    if type(gcs) is BaseGCS:
        traj = gcs.solve_convex_restriction(path_augmented)
    elif type(gcs) is LinearGCS:
        traj = gcs.solve_convex_restriction(path_augmented)
    elif type(gcs) is BezierGCS:
        traj = gcs.solve_convex_restriction(path_augmented)
    else:
        raise TypeError(f"Unsupported GCS type: {type(gcs)}")

    for vid in to_rmv:
        gcs.remove_vertex(vid)
    
    return traj


# path unfolding 

class PruningCost:
    # declare a class for the pruning cost to use as a reference instead of a float
    def __init__(self, value:float, epsilon:float):
        self.epsilon = epsilon
        self.value = (1 - self.epsilon) * value
    
    def update(self, value:float) -> None:
        self.value = (1 - self.epsilon) * value


class MLNode:

    def __init__(self, label:int, pi:list, g:float, h:float, parent:MLNode=None) -> None:
        self.label, self.pi, self.g, self.h, self.parent = label, pi, g, h, parent
        if parent is None:
            self._vertex_set_cur_label = set(pi[-1]) if label == 0 else set()
        elif self.label != parent.label:
            self._vertex_set_cur_label = set([pi[-1][-1]])
        else:
            self._vertex_set_cur_label = parent._vertex_set_cur_label.union([pi[-1][-1]])

    def __lt__(self, other:MLNode) -> bool:
        return self.g + self.h < other.g + other.h
    
    def __repr__(self) -> str:
        return f"MLNode(label={self.label}, g={self.g:.2f}, h={self.h:.2f}, pi={[self.pi[0][0]] + [p[1] for p in self.pi] + [self.pi[-1][-1]]})"


def multi_label_A_star(
    tour:Tour, lbg:LowerBoundGraph, pruning_cost:PruningCost,
    simple_paths_only:bool=True
) -> Generator[Tuple[List[int],float], None, None]:

    def g_cost(pi):
        return sum([lbg.cost_matrix[p] for p in pi])
    
    def heur(p, label):
        if label >= len(tour) - 1:
            return 0.0
        return lbg.shortest_paths_G[(p[-1], tour[label+1])][1] + \
            sum([lbg.shortest_paths_G[(tour[l], tour[l+1])][1] for l in range(label+1, len(tour)-1)])

    OPEN: List[MLNode] = []

    for p in lbg.Hprime.nodes:
        if p[0] == tour[0]:
            label = 0
            if p[1] == tour[1]:
                label = 1
                if p[2] == tour[2]:
                    label = 2
            if p[2] == tour[1]:
                label = 1

            heapq.heappush(OPEN, MLNode(label, [p], g_cost([p]), heur(p, label)))

    while OPEN:
        n = heapq.heappop(OPEN)

        vertex_path = remove_2hop_loops([n.pi[0][0]] + [p[1] for p in n.pi] + [n.pi[-1][-1]])
        if (tour[0] != tour[-1] or simple_paths_only) and n.label == len(tour) - 1 and tour[-1] in n.pi[-1]:
            vertex_path = remove_2hop_loops([n.pi[0][0]] + [p[1] for p in n.pi] + [n.pi[-1][-1]])
            yield vertex_path, n.g + n.h
            continue

        if tour[0] == tour[-1] and n.label == len(tour) - 1 and n.pi[0] == n.pi[-1] and n.pi[0][1] == tour[-1]:
            vertex_path = remove_2hop_loops([n.pi[0][0]] + [p[1] for p in n.pi] + [n.pi[-1][-1]])
            yield vertex_path, n.g + n.h
            continue

        _, u, v = n.pi[-1]
        if simple_paths_only and (v, tour[n.label+1]) in lbg.G.edges:
            path_list = [(u, v, tour[n.label+1])]
        else:
            path_list = [(u, v, w) for w in lbg.G.nodes if (v, w) in lbg.G.edges]
        for u, v, w in path_list:
            if simple_paths_only and w in n._vertex_set_cur_label:
                continue

            label = (n.label + 1) if w == tour[n.label+1] else n.label
            pi = n.pi + [(u, v, w)]
            n_child = MLNode(label, pi, g_cost(pi), heur(pi[-1], label), n)
            
            if n_child.g + n_child.h <= pruning_cost.value:
                heapq.heappush(OPEN, n_child)
    
    return []


def remove_2hop_loops(path:List[int]) -> List[int]:
    # [..., u, v, u, v, ...] => [..., u, v, ...]
    if len(path) < 3:
        return path
    
    i = 0
    while i < len(path) - 4:
        a, b, c, d = path[i:i+4]
        if a == c and b == d:
            # remove the 2-hop loop
            path.pop(i+2)
            path.pop(i+2)
        else:
            i += 1

    return path


""" for ECG+GHOST """

class BFSNode:

    def __init__(self, depth:int, sp:list, g_cost:float, f_cost:float, parent:BFSNode):
        self.depth, self.sp, self.g_cost, self.f_cost, self.parent = \
            depth, sp, g_cost, f_cost, parent

    def __lt__(self, other:BFSNode):
        return self.f_cost < other.f_cost
    
    @property
    def name(self) -> str:
        return f"d{self.depth}"
    
    @property
    def reconstructed_path(self) -> List[int]:
        path, node = [], self
        while node is not None:
            path = node.sp[:-1] + path
            node = node.parent
        return path + [self.sp[-1]]

    def __repr__(self):
        return f"BFSNode({self.name}, g={self.g_cost:.3f}, f={self.f_cost:.3f}, parent={self.parent.name if self.parent else "None"})"


def ecg_unfold(tour:Tour, lbg, pruning_cost:PruningCost) -> Generator[Tuple[List[int],float], None, None]:
    path = tour[:-1]
    OPEN: List[BFSNode] = []

    def _branch(depth:int, parent:BFSNode=None):
        for sp in lbg.simple_paths[(path[depth], path[(depth+1)%len(path)])]:
            edge_cost = sum(lbg.cost_matrix[sp[i], sp[i+1]] for i in range(len(sp)-1))
            g_cost = parent.g_cost + edge_cost
            f_cost = g_cost
            if f_cost <= pruning_cost.value:
                heapq.heappush(OPEN, BFSNode(depth, sp, g_cost, f_cost, parent))

    _branch(0, BFSNode(-1, [], 0.0, 0.0, None)) # branch from the first goal vertex

    while OPEN:
        node = heapq.heappop(OPEN)
        if node.f_cost > pruning_cost.value:
            break # as the OPEN is a min-heap, all remaining nodes will have f_cost > pruning_cost

        if node.depth == len(path) - 1:
            yield node.reconstructed_path, 0.0
        else:
            _branch(node.depth + 1, parent=node)
