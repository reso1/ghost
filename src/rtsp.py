from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Set
from enum import IntEnum
from itertools import product

import numpy as np
import networkx as nx

from src.gcs.base import BaseGCS, BaseTrajectory

import logging
logger = logging.getLogger(__name__)

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise("Warning: gurobi is not installed. GUROBI solver will not be available.")


class Tour:

    def __init__(self, vertex_path:List[int] = []):
        self.__vertex_path = vertex_path
    
    def __hash__(self) -> int:
        return hash(tuple(self.__vertex_path))
    
    def __eq__(self, other:Tour) -> bool:
        if not isinstance(other, Tour):
            return False
        return self.__vertex_path == other.__vertex_path
    
    def __getitem__(self, index:int) -> int:
        return self.__vertex_path[index]
    
    def __len__(self) -> int:
        return len(self.__vertex_path)
    
    def __repr__(self) -> str:
        return f"[{'->'.join(map(str, self.__vertex_path))}]"
    
    @property
    def vertex_path(self) -> List[int]:
        return self.__vertex_path
    
    @property
    def edge_list(self) -> List[Tuple[int, int]]:
        return list(zip(self.__vertex_path[:-1], self.__vertex_path[1:]))

    @property
    def edge_set(self) -> Set[Tuple[int, int]]:
        return set(self.edge_list)


class rTSP:

    INFEASIBLE_COST = 1e9   # a large value for infeasible edge cost 

    class SolverType(IntEnum):
        GUROBI = 0   # optimal branch-and-cut solver

    def __init__(
        self, N:int, E_inclusive:Set[Tuple[int,int]]=set(), 
        E_exclusive:Set[Tuple[int,int]]=set()
    ) -> None:
        self.N, self.EI, self.EO = N, E_inclusive, E_exclusive
        self.solver = None
    
    def __repr__(self) -> str:
        return f"rTSP(EI={self.EI}, EO={self.EO})"

    def set_solver(self, solver_type:SolverType) -> None:
        if solver_type == self.SolverType.GUROBI:
            assert GRB is not None, "GUROBI solver is not available"
            self.solver = self.SolverType.GUROBI
        else:
            raise ValueError(f"Invalid solver type: {type}")
    
    def copy_and_update(
        self, new_EI:Iterable[Tuple[int, int]], new_EO:Iterable[Tuple[int, int]]
    ) -> rTSP:
        rtsp = rTSP(self.N, self.EI.union(new_EI), self.EO.union(new_EO))
        rtsp.solver = self.solver
        return rtsp
    
    def solve(self, cost:np.ndarray) -> Tuple[Tour|None, float]:
        if self.solver == self.SolverType.GUROBI:
            if len(cost.shape) == 2:
                return self.__solve_edge_lbg(cost)
            elif len(cost.shape) == 3:
                return self.__solve_passage_lbg(cost)
            else:
                raise ValueError(f"Invalid cost matrix shape: {cost.shape}")
        else:
            raise ValueError(f"Invalid solver type: {self.solver}")

    def __solve_edge_lbg(self, C:np.ndarray) -> Tuple[Tour|None, float]:
        V = list(range(self.N))
        LB, UB = np.zeros((self.N, self.N)), np.ones((self.N, self.N))
        for u, v in product(V, repeat=2):
            if u == v or (u, v) in self.EO:
                UB[u, v] = 0.0
            elif (u, v) in self.EI:
                LB[u, v] = 1.0

        def subtourelim(model, where):
            if where == GRB.Callback.MIPSOL:
                X = model.cbGetSolution(model._X)
                selected = gp.tuplelist((u,v) for u,v in model._X.keys() if X[u,v] > 0.5)
                tour_graph = nx.DiGraph()
                for u, v in selected:
                    tour_graph.add_edge(u, v)

                cycles = [Tour(cycle + [cycle[0]]) for cycle in nx.simple_cycles(tour_graph)]
                cycle = min(cycles, key=lambda c: len(c))
                if len(cycle) <= self.N:
                    model.cbLazy(gp.quicksum(model._X[cycle[i], cycle[i+1]] for i in range(len(cycle)-1)) <= len(cycle) - 2.0)

        model = gp.Model("rTSP_ELBG")
        model.Params.OutputFlag = 0
        model._X = model.addVars(product(V, repeat=2), vtype=GRB.BINARY, name="x", lb=LB, ub=UB)

        obj = gp.quicksum(C[u, v] * model._X[u, v] for u, v in product(V, repeat=2))
        model.setObjective(obj, GRB.MINIMIZE)
        
        for u in V:
            model.addConstr(gp.quicksum(model._X[u, v] for v in V) == 1.0, name=f"outflow_{u}")
            model.addConstr(gp.quicksum(model._X[v, u] for v in V) == 1.0, name=f"inflow_{u}")

        for u, v in product(V, repeat=2):
            model.addConstr(model._X[u, v] + model._X[v, u] <= 1.0, name=f"2hoploop{(u,v)}")
        
        model.Params.LazyConstraints = 1
        model.optimize(subtourelim)

        if model.Status != GRB.OPTIMAL:
            logger.warning(f"tsp infeasible from Gurobi rTSP solver")
            return None, float('inf')

        vals = model.getAttr('x', model._X)
        selected = gp.tuplelist((u,v) for u,v in model._X.keys() if vals[u,v] > 0.5)
        tour_graph = nx.DiGraph()
        for u, v in selected:
            tour_graph.add_edge(u, v)

        cycles = [Tour(cycle + [cycle[0]]) for cycle in nx.simple_cycles(tour_graph)]
        cycle = min(cycles, key=lambda c: len(c))
        
        assert len(cycle) == self.N + 1, "Gurobi rTSP solver should return exactly one cycle"
        return cycle, model.ObjVal

    def __solve_passage_lbg(self, C:np.ndarray) -> Tuple[Tour|None, float]:
        
        def subtourelim(model, where):
            if where == GRB.Callback.MIPSOL:
                X = model.cbGetSolution(model._X)
                selected = gp.tuplelist((u,v,w) for u,v,w in model._X.keys() if X[u,v,w] > 0.5)
                tour_graph = nx.DiGraph()
                for u, v, w in selected:
                    tour_graph.add_edge(u, v)
                    tour_graph.add_edge(v, w)

                cycles = [Tour(cycle + [cycle[0]]) for cycle in nx.simple_cycles(tour_graph)]
                cycle = min(cycles, key=lambda c: len(c))
                if len(cycle) <= self.N:
                    model.cbLazy(gp.quicksum(model._X[cycle[i], cycle[i+1], cycle[i+2]] for i in range(len(cycle)-2)) <= len(cycle) - 3.0)
    
        V = list(range(self.N))
        model = gp.Model("rTSP")
        model.Params.OutputFlag = 0

        LB, UB = np.zeros((self.N, self.N, self.N)), np.ones((self.N, self.N, self.N))
        for u, v, w in product(V, repeat=3):
            if u == w or u == v or v == w or (u, v) in self.EO or (v, w) in self.EO:
                UB[u, v, w] = 0.0
        
        model._X = model.addVars(product(V, repeat=3), vtype=GRB.BINARY, name="x", lb=LB, ub=UB)
        
        obj = gp.quicksum(C[u, v, w] * model._X[u, v, w] for u, v, w in product(V, repeat=3))
        model.setObjective(obj, GRB.MINIMIZE)

        # each vertex v is visited exactly once by some (·，v, ·)
        for v in V:
            # sum(x[:, v, :]) == 1.0
            model.addConstr(gp.quicksum(model._X[u, v, w] for u, w in product(V, repeat=2)) == 1.0, name=f"cover_{v}")
        
        # flow conservation for each edge (u, v)
        for u, v in product(V, repeat=2):
            if u == v:
                continue
            if (u, v) in self.EI:
                # sum(x[:, u, v]) == 1.0
                model.addConstr(gp.quicksum(model._X[z, u, v] for z in V) == 1.0, name=f"flow->{(u,v)}=1.0")
                # sum(x[u, v, :]) == 1.0
                model.addConstr(gp.quicksum(model._X[u, v, w] for w in V) == 1.0, name=f"flow<-{(u,v)}=1.0")
            else:
                # sum(x[:, u, v]) == sum(x[u, v, :])
                model.addConstr(
                    gp.quicksum(model._X[z, u, v] for z in V) == gp.quicksum(model._X[u, v, w] for w in V),
                    name=f"flow<->{(u,v)}"
                )

        model.Params.LazyConstraints = 1
        model.optimize(subtourelim)

        if model.Status != GRB.OPTIMAL:
            logger.warning(f"tsp infeasible from Gurobi rTSP solver")
            return None, float('inf')
        
        vals = model.getAttr('x', model._X)
        selected = gp.tuplelist((u,v,w) for u,v,w in model._X.keys() if vals[u,v,w] > 0.5)
        tour_graph = nx.DiGraph()
        for u, v, w in selected:
            tour_graph.add_edge(u, v)
            tour_graph.add_edge(v, w)

        cycles = [Tour(cycle + [cycle[0]]) for cycle in nx.simple_cycles(tour_graph)]
        cycle = min(cycles, key=lambda c: len(c))
        
        assert len(cycle) == self.N + 1, "Gurobi rTSP solver should return exactly one cycle"
        return cycle, model.ObjVal
