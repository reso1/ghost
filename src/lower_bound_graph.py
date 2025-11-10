from __future__ import annotations
from itertools import combinations, product
from typing import Dict, List, Tuple, Generator
from collections import defaultdict
from abc import ABC, abstractmethod

import heapq
import pickle
import numpy as np
import networkx as nx

from src.gcs.base import BaseGCS
from src.rtsp import rTSP, Tour

import logging
logger = logging.getLogger(__name__)


class CompleteGraph(ABC):

    def __init__(self):
        self.cost_matrix: np.ndarray = None
        self.G: nx.DiGraph = nx.DiGraph()
        self.simple_paths_G: Dict[Tuple[int, int], List[List[int]]] = {}
        self.shortest_paths_G = {}

    @staticmethod
    def compute_all_simple_paths(G: nx.DiGraph) -> Dict[Tuple[int, int], List[List[int]]]:
        simple_paths_G = {}
        for u, v in combinations(G.nodes, 2):
            if u == v:
                continue
            if G.has_edge(u, v):
                simple_paths_G[(u, v)] = [[u, v]]
                simple_paths_G[(v, u)] = [[v, u]]
                continue

            edges_to_rmv = []
            for pre in G.predecessors(u):
                edges_to_rmv.append((pre, u))
            for suc in G.successors(v):
                edges_to_rmv.append((v, suc))

            G.remove_edges_from(edges_to_rmv)
            simple_paths_G[(u, v)] = list(nx.all_simple_paths(G, u, v))
            simple_paths_G[(v, u)] = [sp[::-1] for sp in simple_paths_G[(u, v)]]
            G.add_edges_from(edges_to_rmv)
        
        return simple_paths_G
    
    @abstractmethod
    def build(gcs:BaseGCS) -> CompleteGraph:
        raise NotImplementedError("abstract method not implemented")

    @staticmethod
    def load(filename:str) -> CompleteGraph:
        raise NotImplementedError("abstract method not implemented")

    @abstractmethod
    def save(self, filename:str) -> None:
        raise NotImplementedError("abstract method not implemented")


class LowerBoundGraph(CompleteGraph):

    def __init__(self):
        super().__init__()
        self.Hprime = nx.DiGraph()

    @staticmethod
    def build(gcs:BaseGCS) -> LowerBoundGraph:
        lbg = LowerBoundGraph()

        for u, v in gcs.nx_diG.edges:
            lbg.G.add_edge(u, v)
        
        # build Hprime
        for u in gcs.nx_diG.nodes:
            for v in gcs.nx_diG.successors(u):
                for w in gcs.nx_diG.successors(v):
                    try:
                        lb_cost = gcs.solve_convex_restriction([u, v, w]).time_cost
                    except Exception as e:
                        if u == w:
                            lb_cost = 0.0
                        else:
                            raise e
                    lbg.Hprime.add_node((u, v, w), lb=lb_cost)
                    logger.info(f"Hprime: Setting lb[({u}, {v}, {w})] = {lb_cost}")

        for p, q in product(lbg.Hprime.nodes, repeat=2):
            if p[1:] == q[:-1]: # connects p=(·, u, v) and (u, v, ·)
                lbg.Hprime.add_edge(p, q)

        logger.info(f"Created Hprime: " + 
                    f"|V|={lbg.Hprime.number_of_nodes()}, " + 
                    f"|E|={lbg.Hprime.number_of_edges()}")
        
        # compute simple paths between all pairs of nodes in G
        lbg.simple_paths_G = CompleteGraph.compute_all_simple_paths(lbg.G)
        for key, val in lbg.simple_paths_G.items():
            lbg_cost = lambda pi: sum(lbg.Hprime.nodes[(pi[i], pi[i+1], pi[i+2])]['lb'] for i in range(len(pi)-2))
            shortest_path = min(val, key=lbg_cost)
            lbg.shortest_paths_G[key] = (shortest_path, lbg_cost(shortest_path))

        N = len(gcs.regions)
        lbg.cost_matrix = rTSP.INFEASIBLE_COST * np.ones((N, N, N), dtype=float)
        
        # compute the cost matrix for H
        for u, v in lbg.G.edges:
            lbg.cost_matrix[(u, v, u)] = lbg.cost_matrix[(v, u, v)] = 0.0

        for u, v, w in combinations(gcs.nx_diG.nodes, 3):
            lbg.__update_passage_lower_bound(u, v, w)
            lbg.__update_passage_lower_bound(u, w, v)
            lbg.__update_passage_lower_bound(v, u, w)

        return lbg
    
    def __update_passage_lower_bound(self, u:int, v:int, w:int) -> None:
        # calculate hat_lb
        if (u, v) in self.G.edges and (v, w) in self.G.edges:
            mid = (u, v, w)
        
        if (u, v) in self.G.edges and (v, w) not in self.G.edges:
            uvdot = [uvw for uvw in self.Hprime.nodes if uvw[0] == u and uvw[1] == v]
            mid = min(uvdot, key=lambda uvw: self.Hprime.nodes[uvw]['lb'])

        if (u, v) not in self.G.edges and (v, w) in self.G.edges:
            vwdot = [uvw for uvw in self.Hprime.nodes if uvw[1] == v and uvw[2] == w]
            mid = min(vwdot, key=lambda uvw: self.Hprime.nodes[uvw]['lb'])
        
        if (u, v) not in self.G.edges and (v, w) not in self.G.edges:
            dotvdot = [uvw for uvw in self.Hprime.nodes if uvw[1] == v]
            mid = min(dotvdot, key=lambda uvw: self.Hprime.nodes[uvw]['lb'])
        
        cost = 0.5 * self.shortest_paths_G[(u, v)][1] + \
               self.Hprime.nodes[mid]['lb'] + \
               0.5 * self.shortest_paths_G[(v, w)][1]
             
        # set cost matrix
        self.cost_matrix[u, v, w] = self.cost_matrix[w, v, u] = cost
        logger.info(f"H: Setting lb({u}, {v}, {w}) = lb({w}, {v}, {u})={cost:.3f} | pi={self.shortest_paths_G[(u, v)][0]}+{mid}+{self.shortest_paths_G[(v, w)][0]}")

    def save(self, filename:str) -> None:
        with open(filename + ".lbg", "wb") as f:
            pickle.dump({
                "cost_matrix": self.cost_matrix,
                "Hprime": self.Hprime,
                "G": self.G,
                "simple_paths_G": self.simple_paths_G,
                "shortest_paths_G": self.shortest_paths_G
            }, f)
            
        logger.info(f"LowerBoundGraph saved to {filename}")

    @staticmethod
    def load(filename:str) -> LowerBoundGraph:
        lbg = LowerBoundGraph()
        with open(filename + ".lbg", "rb") as f:
            data = pickle.load(f)
            lbg.cost_matrix = data["cost_matrix"]
            lbg.Hprime = data["Hprime"]
            lbg.G = data["G"]
            lbg.simple_paths_G = data["simple_paths_G"]
            lbg.shortest_paths_G = data["shortest_paths_G"]
        
        logger.info(f"loaded LowerBoundGraph from {filename}")
        return lbg

    def subgraph(self, selected:List[int]) -> LowerBoundGraph:
        """ Returns a subgraph of the lower bound graph with only the selected nodes """
        sub_lbg = LowerBoundGraph()
        sub_lbg.G = self.G.copy()
        sub_lbg.Hprime = self.Hprime.copy()
        sub_lbg.cost_matrix = self.cost_matrix[np.ix_(selected, selected, selected)]
        sub_lbg.simple_paths_G = self.simple_paths_G.copy()
        sub_lbg.shortest_paths_G = self.shortest_paths_G.copy()
        return sub_lbg


""" utility classes for unfolding the tour """

class PruningCost:
    # declare a class for the pruning cost to use as a reference instead of a float
    def __init__(self, value:float, epsilon:float):
        self.epsilon = epsilon
        self.value = (1 - self.epsilon) * value
    
    def update(self, value:float) -> None:
        self.value = (1 - self.epsilon) * value
