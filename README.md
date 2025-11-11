# GHOST
This repository implements the GHOST search framework for GCS-TSP from the following paper:

- Jingtao Tang and Hang Ma. "GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets." [[paper]](https://arxiv.org/abs/2511.06471), [[project]](https://sites.google.com/view/ghost-gcs-tsp)


## Installation
- Python dependencies can be installed via `pip install -r requirements.txt`
- [Mosek](https://www.mosek.com/) solver should be installed, which is used in the [Drake](https://drake.mit.edu/) library for the GCS program solving. 
- [Optional] You can also use Gurobi solver for Drake, however, building Drake from source is required. See detailed installation guidance [here](https://drake.mit.edu/installation.html).


## File Structure
- data/
  - lbg/: stores precomputed lower-bound graphs.
  - envs.zip: compressed environment files.
  - exp.zip: compressed experiment result files.
  - extract_micp_res.py: a simple helper to extract the MICP results from MOSEK solvers (given limited runtime).

- demos/: a collection of demonstrations using GHOST to solve GCS-TSP.

- exp/
  - baselines.py: implmentations of baseline algorithms.
  - env_generator.py: random environment generator.
  - exp_xxx.sh: experiment runner shell scripts.
  - exp_runner.py: main experiment runner for GHOST related algorithms.
  - micp_runner.py: experiment runner for MICP-based solver.

- gcspy/: A Python library for solving optimization problems over GCS. See official Github [repo](https://github.com/TobiaMarcucci/gcsopt) for details.

- src/
  - gcs/: GCS-related encoding & solving, e.g., Point-GCS, Linear-GCS, and Bezier-GCS. See original [repo](https://github.com/RobotLocomotion/gcs-science-robotics) for details.
  - env.py: Environment class for building GCS, GCS-TSP instances, etc.
  - lower_bound_graph.py: the lower-bound graph (LBG) class.
  - path_unfolding.py: functions related to the next-best abstract tour unfolding procedure.
  - rtsp.py: functions & classes related to the Restricted-TSP (RTSP) solving on GCS and its LBG.
  - search.py: the GHOST search framework class.
  - utils.py: miscellenous utility functions

- plot.py: plot functions for the experiments

## BibTex:
```
@misc{tang2025ghostsolvingtravelingsalesman,
      title={GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets}, 
      author={Jingtao Tang and Hang Ma},
      year={2025},
      eprint={2511.06471},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.06471}, 
}
```

## License
ST-GCS is released under the GPL version 3. See LICENSE.txt for further details.
