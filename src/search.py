from __future__ import annotations
from typing import List, Iterator, Dict, Tuple
from collections import defaultdict
from matplotlib.axes import Axes

import heapq, time, psutil
import numpy as np

from src.path_unfolding import gcs_convex_restriction
from src.lower_bound_graph import CompleteGraph, LowerBoundGraph
from src.path_unfolding import PruningCost, multi_label_A_star, ecg_unfold
from src.rtsp import rTSP, Tour
from src.gcs.base import BaseGCS

from exp.baselines import EuclideanDistGraph

import logging
logger = logging.getLogger(__name__)


class Node:

    def __init__(
        self, name:str, rtsp:rTSP, depth:int=0, 
        lb:float=-float('inf'), ub:float=float('inf'), 
        tour:Tour=None, best_path:List[int]=None
    ) -> None:
        self.name, self.rtsp, self.depth = name, rtsp, depth
        self.lb, self.ub, self.tour, self.best_path = lb, ub, tour, best_path

    def __hash__(self) -> int:
        assert self.tour is not None, "cannot hash Node with empty tour"
        return hash(self.tour)

    def __repr__(self) -> str:
        return f"Node(name={self.name}, lb={self.lb:.3f}, ub={self.ub:.3f})\n\t Node.rtsp={self.rtsp}"

    def __lt__(self, other:Node) -> bool:
        return self.lb < other.lb
    
    def split(self) -> Iterator[Node]:
        assert self.tour != [], "cannot split node with empty tour"
        E_list = self.tour.edge_list
        for i, e in enumerate(E_list):
            new_EI, new_EO = E_list[:i], [e]
            if e in self.rtsp.EI or self.rtsp.EO.intersection(new_EI) != set():
                logger.info(f"skipping child node as either e ∈ EI or π[:i] ∩ EO ≠ ∅")
                continue
            rtsp = self.rtsp.copy_and_update(new_EI, new_EO)
            yield Node(f"d{self.depth+1}i{i}", rtsp, self.depth+1)


class GHOST:
    
    def __init__(
        self, gcs:BaseGCS, lbg:CompleteGraph, 
        rtsp_solver_type=rTSP.SolverType.GUROBI, epsilon:float=0.0,
        shortest_unfolding:bool=False, targets:List[int]=None
    ) -> None:
        self.gcs, self.lbg = gcs, lbg
        self.rtsp_solver, self.epsilon = rtsp_solver_type, epsilon
        self.shortest_unfolding = shortest_unfolding
        self.targets = targets
        if self.targets is not None:
            self.cost_matrix = lbg.cost_matrix[np.ix_(targets, targets)]
        else:
            self.cost_matrix = lbg.cost_matrix

        self.__gcs_cost_rec: Dict[Tuple[int], float] = {}
        if type(lbg) is LowerBoundGraph:
            self.unfold_func = multi_label_A_star
        elif type(lbg) is EuclideanDistGraph:
            self.unfold_func = ecg_unfold
        else:
            raise TypeError(f"Unsupported LowerBoundGraph type: {type(lbg)}")

    def grow(self, runtime_limit:float=3600, recorder:Recorder=None) -> Tuple[Node, float]:
        """ find the next best tsp tour via partition method """
        self.ts = time.perf_counter()
        self.runtime_limit = runtime_limit
        root = self.__create_root()
        OPEN, CLOSED = [root], set()
        process = psutil.Process()
        while OPEN:
            node = heapq.heappop(OPEN)
            CLOSED.add(node)
            opt_gap = (self.OPT.ub - node.lb) / self.OPT.ub
            time_elapsed = time.perf_counter() - self.ts
            mem = process.memory_info().rss / 1024 / 1024

            if recorder is not None:
                recorder.lb.append(node.lb)
                recorder.ub.append(self.OPT.ub)
                recorder.gap.append(opt_gap)
                recorder.time.append(time_elapsed)
                recorder.mem.append(mem)
            
            # optimality pruning only applies to LBG
            if type(self.lbg) is LowerBoundGraph and (node.lb >= self.OPT.ub or opt_gap <= self.epsilon):
                logger.info(f"Reached opt. gap: {opt_gap:.3%}")
                break

            if time_elapsed >= self.runtime_limit:
                logger.info(f"Reached runtime limit: {time_elapsed:.3f}s")
                break
                
            print(f"|OPEN|={len(OPEN)}, |CLOSED|={len(CLOSED)}\t|\topt={self.OPT.ub:.3f}, lb={node.lb:.3f}, " + 
                  f"gap={opt_gap:.2%}\t|\truntime={time_elapsed:.3f}s, mem={mem:.0f}MB")
            logger.info(f"|OPEN|={len(OPEN)}; |CLOSED| = {len(CLOSED)}; OPT = (cost:{self.OPT.ub:.3f}, gap:{opt_gap:.2%})")
            logger.info(f"current node: {node}")

            self.__low_level_search(node)

            for child in node.split():
                tour, tour_cost = child.rtsp.solve(self.cost_matrix)
                child.lb, child.tour = tour_cost, tour
                if tour is None or child in CLOSED:
                    logger.warning(f"failed to solve rtsp for node {child.name}")
                else:
                    logger.info(f"TSP tour obtained: {tour}, cost={tour_cost:.3f}")
                    heapq.heappush(OPEN, child)

        return self.OPT, node.lb

    def __create_root(self) -> Node:
        rtsp = rTSP(len(self.cost_matrix))
        rtsp.set_solver(self.rtsp_solver)
        root = Node(name='root', rtsp=rtsp, depth=0)
        
        tour, tour_cost = root.rtsp.solve(self.cost_matrix)
        if tour is None:
            logger.warning(f"failed to solve rtsp for node {root.name}")
            return None

        remapped_tour = tour if self.targets is None else Tour([self.targets[i] for i in tour])
        logger.info(f"TSP tour obtained: {remapped_tour}, cost={tour_cost:.3f}")
        
        path, _ = next(self.unfold_func(remapped_tour, self.lbg, pruning_cost=PruningCost(float('inf'), self.epsilon)))
        traj_cost = gcs_convex_restriction(path, self.gcs).time_cost
        root.lb, root.tour = tour_cost, tour    # double-counting in the worst case
        root.ub, root.best_path = traj_cost, path
        self.OPT = root
        self.__low_level_search(root)
        return root

    def __low_level_search(self, node:Node) -> None:
        # inner-loop to find the best unfolded path of tour, that generates the best trajectory
        pruning_cost = PruningCost(self.OPT.ub, self.epsilon)
        remapped_tour = node.tour if self.targets is None else Tour([self.targets[i] for i in node.tour])
        for unfolded, cost in self.unfold_func(remapped_tour, self.lbg, pruning_cost):
            if type(self.lbg) is LowerBoundGraph and cost >= pruning_cost.value:
                break # unfold_func is a best-first-search, so we can stop early

            path_key = tuple(unfolded)
            if path_key in self.__gcs_cost_rec:
                traj_cost = self.__gcs_cost_rec[path_key]
            else:
                logger.info(f"GCS convex restriction w/ cost={cost:.3f}, pruning-cost={pruning_cost.value:.3f}, unfolded path={unfolded}")
                traj = gcs_convex_restriction(unfolded, self.gcs)
                if traj is None:
                    logger.warning(f"failed to generate trajectory for unfolded path: {unfolded}")
                    traj_cost = float('inf')
                else:
                    traj_cost = traj.time_cost
                    self.__gcs_cost_rec[path_key] = traj_cost
            if traj_cost < node.ub:
                node.ub, node.best_path = traj_cost, unfolded
                pruning_cost.update(self.OPT.ub)
                logger.info(f"found better unfolded path with cost: {traj_cost:.3f}")
                if traj_cost < self.OPT.ub:
                    self.OPT = node
                    logger.info(f"found better optimal solution: {self.OPT.ub:.3f}")
            
            if self.shortest_unfolding or time.perf_counter() - self.ts > self.runtime_limit:
                break


class Recorder:
    
    def __init__(self) -> None:
        self.lb, self.ub, self.gap = [], [], []
        self.time, self.mem = [], []
    
    def plot(self, ax:Axes):
        ax.set_ylim(0.0, 1.0)
        ax.plot(self.time, self.gap, label='Gap', color='b', linestyle=':')
        ax.legend(loc ='upper left')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gap")

        _ax = ax.twinx()
        _ax.plot(self.time, self.lb, label='LB', color='r', linestyle='-')
        _ax.plot(self.time, self.ub, label='UB', color='k', linestyle='--')
        _ax.set_ylim(min(self.lb) * 0.95, max(self.ub) * 1.05)
        _ax.set_ylabel("Cost")
        _ax.legend(loc='upper right')

    def save(self, fp:str) -> None:
        np.savez(fp, lb=self.lb, ub=self.ub, gap=self.gap, time=self.time, mem=self.mem)

    @staticmethod
    def load(fp:str) -> Recorder:
        data = np.load(fp)
        recorder = Recorder()
        recorder.lb, recorder.ub, recorder.gap = data['lb'], data['ub'], data['gap']
        recorder.time, recorder.mem = data['time'], data['mem']
        return recorder

