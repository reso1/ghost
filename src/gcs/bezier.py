# Code modified from https://github.com/RobotLocomotion/gcs-science-robotics

from __future__ import annotations
from typing import List, Tuple, Dict
from dataclasses import dataclass
from itertools import combinations
from scipy.optimize import root_scalar

import numpy as np

from pydrake.all import (
    HPolyhedron, Point,
    BsplineBasis, BsplineBasis_, KnotVectorType,
    Binding, Constraint, Cost, L2NormCost, 
    LinearConstraint, LinearCost, LinearEqualityConstraint, 
    QuadraticCost, PerspectiveQuadraticCost,
    DecomposeLinearExpressions, Expression,
    MakeMatrixContinuousVariable, MakeVectorContinuousVariable,
    BsplineTrajectory, BsplineTrajectory_, MathematicalProgramResult,
    GraphOfConvexSets as GCS
)

from src.gcs.base import BaseGCS, BaseTrajectory
from src.env import Env


@dataclass
class Configuration:
    order:int
    continuity:int
    dt_min:float
    time_cost: float = 1.0
    path_length_cost: float = 0.0
    path_integral_length_cost: float = 0.0
    energy_cost: float = 0.0
    derivative_regularization: Tuple[int, float, float] = None
    vmin: np.ndarray = None
    vmax: np.ndarray = None


class BezierGCS(BaseGCS):

    def __init__(self, regions:List[HPolyhedron], cfg:Configuration) -> None:

        super(BezierGCS, self).__init__(regions)

        assert cfg.continuity < cfg.order
        self.order, self.continuity, self.cfg = cfg.order, cfg.continuity, cfg
        
        """ utility variables """
        order, continuity = cfg.order, cfg.continuity
        # time scaling set for Bezier curves in each Vertex
        A_time = np.vstack((np.eye(order + 1), -np.eye(order + 1),
                            np.eye(order, order + 1) - np.eye(order, order + 1, 1)))
        b_time = np.concatenate((1e3*np.ones(order+1), np.zeros(order+1), -cfg.dt_min*np.ones(order)))
        self.time_scaling_set = HPolyhedron(A_time, b_time)
        # Bezier curves & variable shortcuts
        u_control = MakeMatrixContinuousVariable(self.dimension, order + 1, "xu")
        v_control = MakeMatrixContinuousVariable(self.dimension, order + 1, "xv")
        u_duration = MakeVectorContinuousVariable(order + 1, "Tu")
        v_duration = MakeVectorContinuousVariable(order + 1, "Tv")
        self.u_vars = np.concatenate((u_control.flatten("F"), u_duration))
        basis = lambda: BsplineBasis_[Expression](
            order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.)
        self.u_r_trajectory: BsplineTrajectory = BsplineTrajectory_[Expression](basis(),u_control)
        self.u_h_trajectory: BsplineTrajectory = BsplineTrajectory_[Expression](
            basis(), np.expand_dims(u_duration, 0))
        self.edge_vars = np.concatenate(
            (u_control.flatten("F"), u_duration, v_control.flatten("F"), v_duration))
        self.v_r_trajectory = BsplineTrajectory_[Expression](basis(), v_control)
        self.v_h_trajectory = BsplineTrajectory_[Expression](basis(), np.expand_dims(v_duration, 0))

        """ vertex costs & constraints """
        if self.cfg.time_cost > 0:  # time cost
            self.vertex_costs.append(self.get_time_cost(self.cfg.time_cost))
        if self.cfg.path_length_cost > 0:   # path length cost
            self.vertex_costs.extend(self.get_path_length_cost(self.cfg.path_length_cost))
        if self.cfg.path_integral_length_cost > 0:  # path integral cost
            self.get_path_length_integral_cost(self.cfg.path_integral_length_cost)
        if self.cfg.energy_cost > 0:    # path energy cost
            self.get_path_energy_cost(self.cfg.energy_cost)
        if self.cfg.derivative_regularization != None:  # regularization cost
            self.vertex_costs.extend(self.get_deriv_regularization(*self.cfg.derivative_regularization))
        if self.cfg.vmin is not None and self.cfg.vmax is not None: # velocity limit
            self.vertex_constraints.extend(self.get_vlimit_constraints(self.cfg.vmin, self.cfg.vmax))
        
        """ edge costs & constraints """
        # continuity constraints
        for deriv in range(continuity + 1):
            u_path_deriv = self.u_r_trajectory.MakeDerivative(deriv)
            v_path_deriv = self.v_r_trajectory.MakeDerivative(deriv)
            path_continuity_error = v_path_deriv.control_points()[0] - u_path_deriv.control_points()[-1]
            self.edge_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(path_continuity_error, self.edge_vars),
                np.zeros(self.dimension)))
            u_time_deriv = self.u_h_trajectory.MakeDerivative(deriv)
            v_time_deriv = self.v_h_trajectory.MakeDerivative(deriv)
            time_continuity_error = v_time_deriv.control_points()[0] - u_time_deriv.control_points()[-1]
            self.edge_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(time_continuity_error, self.edge_vars), 0.0))

    def add_vertex(self, region:HPolyhedron, index:int) -> GCS.Vertex:
        timed_hpoly = region.CartesianPower(self.order + 1).CartesianProduct(self.time_scaling_set)
        return super()._add_vertex(timed_hpoly, region, index)

    @staticmethod
    def build(env:Env, cfg:Configuration) -> BezierGCS:
        gcs = BezierGCS(env._CSpace_hpoly, cfg)
        for index, region in enumerate(env._CSpace_hpoly):
            gcs.add_vertex(region, index)
        
        for u, v in combinations(gcs.nx_diG.nodes, 2):
            Xu, Xv = gcs.nx_diG.nodes[u]["set"], gcs.nx_diG.nodes[v]["set"]
            if u != v and Xu.IntersectsWith(Xv):
                gcs.add_edge(u, v)
                gcs.add_edge(v, u)
        
        return gcs

    def get_time_cost(self, weight:float) -> LinearCost:
        u_time_control = self.u_h_trajectory.control_points()
        segment_time = u_time_control[-1] - u_time_control[0]
        return LinearCost(weight * DecomposeLinearExpressions(segment_time, self.u_vars)[0], 0.)

    def get_path_length_cost(self, weight:float) -> List[L2NormCost]:
        costs = []
        weight_matrix = weight * np.eye(self.dimension)
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_path_control)):
            H = DecomposeLinearExpressions(u_path_control[ii] / self.order, self.u_vars)
            path_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
            costs.append(path_cost)
        return costs

    def get_path_length_integral_cost(self, weight:float, integration_points=100) -> List[L2NormCost]:
        cost_list = []
        weight_matrix = weight * np.eye(self.dimension)
        s_points = np.linspace(0., 1., integration_points + 1)
        u_path_deriv = self.u_r_trajectory.MakeDerivative(1)

        if u_path_deriv.basis().order() == 1:
            for t in [0.0, 1.0]:
                q_ds = u_path_deriv.value(t)
                costs = []
                for ii in range(self.dimension):
                    costs.append(q_ds[ii])
                H = DecomposeLinearExpressions(costs, self.u_vars)
                integral_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
                cost_list.append(integral_cost)
        else:
            q_ds = u_path_deriv.vector_values(s_points)
            for ii in range(integration_points + 1):
                costs = []
                for jj in range(self.dimension):
                    if ii == 0 or ii == integration_points:
                        costs.append(0.5 * 1./integration_points * q_ds[jj, ii])
                    else:
                        costs.append(1./integration_points * q_ds[jj, ii])
                H = DecomposeLinearExpressions(costs, self.u_vars)
                integral_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
                cost_list.append(integral_cost)
        
        return cost_list

    def get_path_energy_cost(self, weight:float) -> List[PerspectiveQuadraticCost]:
        costs = []
        weight_matrix = weight * np.eye(self.dimension)
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_path_control)):
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            H = np.vstack(((self.order) * b_ctrl, np.matmul(np.sqrt(weight_matrix), A_ctrl)))
            energy_cost = PerspectiveQuadraticCost(H, np.zeros(H.shape[0]))
            costs.append(energy_cost)

        return costs

    def get_deriv_regularization(self, order:int, weight_r:float, weight_h:float) -> List[QuadraticCost]:
        regs =[]
        weights = [weight_r, weight_h]
        trajectories = [self.u_r_trajectory, self.u_h_trajectory]
        for traj, weight in zip(trajectories, weights):
            derivative_control = traj.MakeDerivative(order).control_points()
            for c in derivative_control:
                A_ctrl = DecomposeLinearExpressions(c, self.u_vars)
                H = A_ctrl.T.dot(A_ctrl) * 2 * weight / (1 + self.order - order)
                reg_cost = QuadraticCost(H, np.zeros(H.shape[0]), 0)
                regs.append(reg_cost)

        return regs

    def get_vlimit_constraints(self, lower_bound:np.ndarray, upper_bound:np.ndarray) -> List[LinearConstraint]:
        cstrs = []
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        lb = np.expand_dims(lower_bound, 1)
        ub = np.expand_dims(upper_bound, 1)

        for ii in range(len(u_path_control)):
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            A_constraint = np.vstack((A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl))
            velocity_con = LinearConstraint(
                A_constraint, -np.inf*np.ones(2*self.dimension), np.zeros(2*self.dimension))
            cstrs.append(velocity_con)

        return cstrs

    def solve(
        self, x0_set_idx:int, xt_set_idx:int,
        zero_deriv_constraints:int=None,
        rounding=False, verbose=False, preprocessing=False
    ) -> Tuple[List[int]|None, BezierTrajectory|None, float]:
        
        self.source = self.gcs.AddVertex(self.regions[x0_set_idx], "source")
        self.target = self.gcs.AddVertex(self.regions[xt_set_idx], "target")
        
        # add source edge
        e_src = self.gcs.AddEdge(self.source, self.nx_diG.nodes[x0_set_idx]["vertex"], name='esource')
        self._add_e_src_constraint(e_src, v0=None, zero_deriv_constraints=zero_deriv_constraints)

        # add target edge
        e_tar = self.gcs.AddEdge(self.nx_diG.nodes[xt_set_idx]["vertex"], self.target, name='etarget')
        self._add_e_tar_constraint(e_tar, vt=None, zero_deriv_constraints=zero_deriv_constraints)

        # solve GCS
        best_path, best_result, results_dict = self.__solve_GCS__(rounding, preprocessing, verbose)

        if best_result is None:
            traj, vertex_path, cost = None, None, np.inf
        else:
            traj = self._extract_trajectory(best_path, best_result)
            vertex_path = [e.v().name() for e in best_path]
            vertex_path = [int(v[1:]) for v in vertex_path[:-1]]
            cost = traj.time_cost

        # unset source and target
        self.gcs.RemoveVertex(self.source)
        self.gcs.RemoveVertex(self.target)
        self.source = self.target = None
        
        return vertex_path, traj, cost

    def solve_convex_restriction(self, path:List[int]) -> BezierTrajectory|None:   
        E = [self.nx_diG.edges[path[i], path[i+1]]["e"] for i in range(len(path)-1)]
        res = self.gcs.SolveConvexRestriction(E, self.options)
        traj = self._extract_trajectory(E, res)
        return traj

    def _add_e_src_constraint(self, edge:GCS.Edge, v0:np.ndarray, zero_deriv_constraints:int) -> None:
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        
        # continuity constraints
        for jj in range(self.dimension):
            edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])
        
        # velocity constraints
        if v0 is not None:
            v0_error = u_path_control[0].reshape(-1, 1) - (v0 * u_time_control[0]).reshape(-1, 1)
            v0_cstr = LinearEqualityConstraint(
                DecomposeLinearExpressions(v0_error, self.u_vars), np.zeros(self.dimension))
            edge.AddConstraint(Binding[Constraint](v0_cstr, edge.xv()))
        
        # zero derivative constraints
        if zero_deriv_constraints is not None:
            assert self.order > zero_deriv_constraints + 1
            for deriv in range(1, zero_deriv_constraints+1):
                u_path_control = self.u_r_trajectory.MakeDerivative(deriv).control_points()
                cstr = LinearEqualityConstraint(
                    DecomposeLinearExpressions(np.squeeze(u_path_control[0]), self.u_vars),
                    np.zeros(self.dimension))
                edge.AddConstraint(Binding[Constraint](cstr, edge.xv()))

    def _add_e_tar_constraint(self, edge:GCS.Edge, vt:np.ndarray, zero_deriv_constraints:int) -> None:
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()

        for jj in range(self.dimension):
            edge.AddConstraint(edge.xu()[-(self.dimension + self.order + 1) + jj] == edge.xv()[jj])
        
        if vt is not None:
            vt_error = np.squeeze(u_path_control[-1]) - vt * np.squeeze(u_time_control[-1])
            vt_cstr = LinearEqualityConstraint(
                DecomposeLinearExpressions(vt_error, self.u_vars), np.zeros(self.dimension))
            edge.AddConstraint(Binding[Constraint](vt_cstr, edge.xu()))
        
        if zero_deriv_constraints is not None:
            assert self.order > zero_deriv_constraints + 1
            for deriv in range(1, zero_deriv_constraints+1):
                u_path_control = self.u_r_trajectory.MakeDerivative(deriv).control_points()
                cstr = LinearEqualityConstraint(
                    DecomposeLinearExpressions(np.squeeze(u_path_control[-1]), self.u_vars),
                    np.zeros(self.dimension))
                edge.AddConstraint(Binding[Constraint](cstr, edge.xu()))

    def _extract_trajectory(
        self, path:List[GCS.Edge], res:MathematicalProgramResult
    ) -> BezierTrajectory|None:
        if not res.is_success():
            return None
        
        knots = np.zeros(self.order + 1)
        path_control_points = []
        time_control_points = []
        for idx, edge in enumerate(path):
            xv_value = res.GetSolution(edge.xv())
            if len(xv_value) == self.dimension:
                knots = np.concatenate((knots, [knots[-1]]))
                path_control_points.append(xv_value)
                time_control_points.append(np.array([res.GetSolution(edge.xu())[-1]]))
                break
            edge_time = knots[-1] + 1.
            knots = np.concatenate((knots, np.full(self.order, edge_time)))
            edge_path_points = np.reshape(xv_value[:-(self.order + 1)],
                                             (self.dimension, self.order + 1), "F")
            edge_time_points = xv_value[-(self.order + 1):]

            for ii in range(self.order):
                path_control_points.append(edge_path_points[:, ii])
                if idx == len(path) - 1: # pad the time control points of last edge
                    duration = edge_time_points[ii + 1] - edge_time_points[ii]
                    time_control_points.append(np.array([float(time_control_points[-1] + duration)]))
                else:
                    time_control_points.append(np.array([edge_time_points[ii]]))

        offset = time_control_points[0].copy()
        for ii in range(len(time_control_points)):
            time_control_points[ii] -= offset

        path_control_points = np.array(path_control_points).T
        time_control_points = np.array(time_control_points).T

        path = BsplineTrajectory(BsplineBasis(self.order + 1, knots), path_control_points)
        time_traj = BsplineTrajectory(BsplineBasis(self.order + 1, knots), time_control_points)
        return BezierTrajectory(path, time_traj)


class BezierTrajectory(BaseTrajectory):

    def __init__(self, path_traj, time_traj):
        assert path_traj.start_time() == time_traj.start_time()
        assert path_traj.end_time() == time_traj.end_time()
        self.path_traj = path_traj
        self.time_traj = time_traj
        self.start_s = path_traj.start_time()
        self.end_s = path_traj.end_time()
    
    def invert_time_traj(self, t):
        if t <= self.start_time():
            return self.start_s
        if t >= self.end_time():
            return self.end_s
        error = lambda s: self.time_traj.value(s)[0, 0] - t
        res = root_scalar(error, bracket=[self.start_s, self.end_s])
        return np.min([np.max([res.root, self.start_s]), self.end_s])

    def value(self, t):
        return self.path_traj.value(self.invert_time_traj(np.squeeze(t)))

    def vector_values(self, times):
        s = [self.invert_time_traj(t) for t in np.squeeze(times)]
        return self.path_traj.vector_values(s)

    def EvalDerivative(self, t, derivative_order=1):
        if derivative_order == 0:
            return self.value(t)
        elif derivative_order == 1:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            r_dot = self.path_traj.EvalDerivative(s, 1)
            return r_dot * s_dot
        elif derivative_order == 2:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            h_ddot = self.time_traj.EvalDerivative(s, 2)[0, 0]
            s_ddot = -h_ddot*(s_dot**3)
            r_dot = self.path_traj.EvalDerivative(s, 1)
            r_ddot = self.path_traj.EvalDerivative(s, 2)
            return r_ddot * s_dot * s_dot + r_dot * s_ddot
        else:
            raise ValueError()

    def start_time(self):
        return self.time_traj.value(self.start_s)[0, 0]

    def end_time(self):
        return self.time_traj.value(self.end_s)[0, 0]

    def rows(self):
        return self.path_traj.rows()

    def cols(self):
        return self.path_traj.cols()

    @property
    def time_cost(self) -> float:
        return self.end_time() - self.start_time()
