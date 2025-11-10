# Demo modified from https://deepnote.com/workspace/Manipulation-ac8201a1-470a-4c77-afd0-2cc45bc229ff/project/0762b167-402a-4362-9702-7d559f0e73bb/notebook/iris_builder-3c25c10bc29d4c9493e48eaced475d03

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gcs.bezier import BezierGCS, Configuration
from src.lower_bound_graph import LowerBoundGraph
from src.search import GHOST, gcs_convex_restriction
import os, time
from itertools import combinations


from src.gcs.bezier import BezierGCS, BezierTrajectory, Configuration
from src.search import GHOST
from src.lower_bound_graph import LowerBoundGraph

import logging
log_fp = f"{__file__}.log" # + f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
if os.path.exists(log_fp):
    os.remove(log_fp)
logging.basicConfig(filename=log_fp, level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cfg(vmin, vmax):
    return Configuration(
        order=3,
        continuity=1,
        dt_min=1e-6,
        derivative_regularization = (2, 1e-1, 1e-1),
        vmin=vmin,
        vmax=vmax,
    )


def get_gcs(regions, vmin, vmax) -> BezierGCS:
    cfg = get_cfg(vmin, vmax)

    gcs = BezierGCS(regions, cfg)
     
    for index, region in enumerate(regions):
        gcs.add_vertex(region, index)

    for u, v in combinations(gcs.nx_diG.nodes, 2):
        Xu, Xv = gcs.nx_diG.nodes[u]["set"], gcs.nx_diG.nodes[v]["set"]
        if u != v and Xu.IntersectsWith(Xv):
            gcs.add_edge(u, v)
            gcs.add_edge(v, u)

    return gcs


def run(vmin, vmax, regions, targets):
    cfg = get_cfg(vmin, vmax)
    gcs = get_gcs(regions, cfg)
    
    lbg_fn = 'data/lbg/iiwa.lbg'
    if not os.path.exists(lbg_fn):
        print("Building and saving LBG...")
        lbg = LowerBoundGraph.build(gcs)
        lbg.save(lbg_fn)
    else:
        print("Loading LBG...")
        lbg = LowerBoundGraph.load(lbg_fn)

    cur_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    lbg = LowerBoundGraph.load(os.path.join(cur_dir, "..",  "data/lbg/iiwa.pkl"))

    tree = GHOST(gcs, lbg, epsilon=0.0, shortest_unfolding=True, targets=targets)
    opt, _ = tree.grow(runtime_limit=1e3)
    unfolded_path = opt.best_path

    print(f"Unfolded path: {unfolded_path}")

    return unfolded_path, gcs

