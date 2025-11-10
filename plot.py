import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

legend_fontsize = 12
axis_fontsize = 12
title_fontsize = 12

def plot_point_res(num_seeds=12, axes=None):
    methods = ["ghost-ecg", "greedy", "micp", "ghost-eps0.5", "ghost"]
    colors = {"ghost": "k", "micp": "b", "ghost-ecg": "orange", "greedy": "gray", "ghost-eps0.5": "brown"}
    lines = {"ghost": "-", "micp": "-", "ghost-ecg": "-", "greedy": "-", "ghost-eps0.5": "-"}
    markers = {"ghost": ".", "micp": ".", "ghost-ecg": ".", "greedy": ".", "ghost-eps0.5": "."}
    labels = {"ghost": "GHOST", "micp": "MICP", "ghost-ecg": "ECG", "greedy": "Greedy", "ghost-eps0.5": "0.5-GHOST"}

    list_num_sets = [int(n) for n in range(5, 26, 1)]

    if axes is None:
        num_cols, nrows = 3, 1
        widths = [1.5] * num_cols
        heights = [1.5] * nrows
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axes = plt.subplots(ncols=num_cols, nrows=nrows, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2*num_cols, sum(heights)))
        ax_cost: Axes = axes[0]
        ax_gap: Axes = axes[1]
        ax_time: Axes = axes[2]
    else:
        fig = None
        ax_cost, ax_gap, ax_time = axes

    def _plot_func(costs, runtimes, opt_gaps, method):
        nonempty = [n for n in list_num_sets if n in costs]
        
        ax_cost.plot(nonempty, [costs[n] for n in nonempty], 
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method],
                    lw=2, ms=4, alpha=0.6)
        
        ax_time.plot(nonempty, [runtimes[n] for n in nonempty], 
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method], 
                    lw=2, ms=4, alpha=0.6)

        nonempty = [n for n in list_num_sets if n in opt_gaps]
        ax_gap.plot(nonempty, [opt_gaps[n] for n in nonempty],
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method],
                    lw=2, ms=4, alpha=0.6)

    with open(f"data/exp/point-micp.pkl", 'rb') as f:
        method, data = "micp", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total, seeds_in_total = 0, 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                seeds_in_total += 1
                gap_in_total += data[(seed, n)]['opt_gap']
                cost_in_total += data[(seed, n)]['cost']
                runtime_in_total += min(100, data[(seed, n)]['comp_time'] + data[(seed, n)]['solver_time'])

            if seeds_in_total != 0:
                costs[n] = cost_in_total / seeds_in_total
                runtimes[n] = runtime_in_total / seeds_in_total
                opt_gaps[n] = gap_in_total / seeds_in_total
        
        _plot_func(costs, runtimes, opt_gaps, method)
        
    with open("data/exp/point-ghost-ecg.pkl", 'rb') as f:
        method, data = "ghost-ecg", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total = 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time']

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/point-greedy.pkl", "rb") as f:
        method, data = "greedy", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total = 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                cost_in_total += data[(seed, n)]['best_cost']
                runtime_in_total += data[(seed, n)]['lbg_time'] + data[(seed, n)]['greedy_time']

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/point-ghost-eps0.5.pkl", 'rb') as f:
        method, data = "ghost-eps0.5", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total = 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue
                gap = (data[(seed, n)]['opt_node'].ub - data[(seed, n)]['lb']) / data[(seed, n)]['opt_node'].ub
                gap_in_total += max(gap, 0.0)
                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
            opt_gaps[n] = gap_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/point-ghost.pkl", 'rb') as f:
        method, data = "ghost", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total = 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue
                gap = (data[(seed, n)]['opt_node'].ub - data[(seed, n)]['lb']) / data[(seed, n)]['opt_node'].ub
                gap_in_total += max(gap, 0.0)
                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
            opt_gaps[n] = gap_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    ax_cost.grid(True, which='major', alpha=0.8)
    ax_cost.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_cost.set_xticks(np.linspace(5, 25, 5, endpoint=True))
    ax_cost.set_yticks(np.linspace(7, 19, 5, endpoint=True))
    ax_cost.set_title(f"Costs", fontsize=title_fontsize)

    ax_time.grid(True, which='major', alpha=0.8)
    ax_time.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_time.set_xticks(np.linspace(5, 25, 5, endpoint=True))
    ax_time.set_yticks([0, 25, 50, 75, 99])
    ax_time.set_title(f"Runtimes (secs.)", fontsize=title_fontsize)

    ax_gap.grid(True, which='major', alpha=0.8)
    ax_gap.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_gap.set_xticks(np.linspace(5, 25, 5, endpoint=True))
    ax_gap.set_yticks(np.linspace(0, 0.2, 3, endpoint=True))
    ax_gap.set_title(f"Optimality Gap", fontsize=title_fontsize)

    if fig is not None:
        handles, labels = ax_cost.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=legend_fontsize, 
                bbox_to_anchor=(0.5, 1.25), frameon=True, 
                columnspacing=0.5, ncols=len(methods), prop={'size': legend_fontsize})
        fig.savefig("exp_point.pdf", bbox_inches='tight', dpi=500)


def plot_linear_res(num_seeds=12, axes=None):
    methods = ["ghost-ecg", "greedy", "micp", "ghost-eps0.5", "ghost"]
    colors = {"ghost": "k", "micp": "b", "ghost-ecg": "orange", "greedy": "gray", "ghost-eps0.5": "brown"}
    lines = {"ghost": "-", "micp": "-", "ghost-ecg": "-", "greedy": "-", "ghost-eps0.5": "-"}
    markers = {"ghost": ".", "micp": ".", "ghost-ecg": ".", "greedy": ".", "ghost-eps0.5": "."}
    labels = {"ghost": "GHOST", "micp": "MICP", "ghost-ecg": "ECG", "greedy": "Greedy", "ghost-eps0.5": "0.5-GHOST"}

    if axes is None:
        num_cols, nrows = 3, 1
        widths = [1.5] * num_cols
        heights = [1.5] * nrows
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axes = plt.subplots(ncols=num_cols, nrows=nrows, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2*num_cols, sum(heights)))
        ax_cost: Axes = axes[0]
        ax_gap: Axes = axes[1]
        ax_time: Axes = axes[2]
    else:
        fig = None
        ax_cost, ax_gap, ax_time = axes
    
    list_num_sets = [int(n) for n in range(5, 26, 1)]

    def _plot_func(costs, runtimes, opt_gaps, method):
        
        nonempty = [n for n in list_num_sets if n in costs]
        
        # directed edges = 2 * undirected edges
        ax_cost.plot([2*n for n in nonempty], [costs[n] for n in nonempty], 
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method],
                    lw=2, ms=4, alpha=0.6)
        
        ax_time.plot([2*n for n in nonempty], [runtimes[n] for n in nonempty], 
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method], 
                    lw=2, ms=4, alpha=0.6)

        nonempty = [n for n in list_num_sets if n in opt_gaps]
        ax_gap.plot([2*n for n in nonempty], [opt_gaps[n] for n in nonempty],
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method],
                    lw=2, ms=4, alpha=0.6)

    with open(f"data/exp/linear-micp.pkl", 'rb') as f:
        method, data = "micp", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total, seeds_in_total = 0, 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                seeds_in_total += 1
                gap_in_total += data[(seed, n)]['opt_gap']
                cost_in_total += data[(seed, n)]['cost']
                runtime_in_total += min(100, data[(seed, n)]['comp_time'] + data[(seed, n)]['solver_time'])

            if seeds_in_total != 0:
                costs[n] = cost_in_total / seeds_in_total
                runtimes[n] = runtime_in_total / seeds_in_total
                opt_gaps[n] = gap_in_total / seeds_in_total
        
        _plot_func(costs, runtimes, opt_gaps, method)
        
    with open("data/exp/linear-ghost-ecg.pkl", 'rb') as f:
        method, data = "ghost-ecg", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total = 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/linear-greedy.pkl", "rb") as f:
        method, data = "greedy", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total = 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                cost_in_total += data[(seed, n)]['best_cost']
                runtime_in_total += data[(seed, n)]['lbg_time'] + data[(seed, n)]['greedy_time']

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/linear-eps0.5-ghost.pkl", 'rb') as f:
        method, data = "ghost-eps0.5", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total = 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue
                gap = (data[(seed, n)]['opt_node'].ub - data[(seed, n)]['lb']) / data[(seed, n)]['opt_node'].ub
                gap_in_total += max(gap, 0.0)
                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
            opt_gaps[n] = gap_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/linear-ghost.pkl", 'rb') as f:
        method, data = "ghost", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total = 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                if min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time']) < 90:
                    # means the solver searches all nodes
                    gap = 0.0
                else:
                    gap = (data[(seed, n)]['opt_node'].ub - data[(seed, n)]['lb']) / data[(seed, n)]['opt_node'].ub
                    
                gap_in_total += max(gap, 0.0)
                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
            opt_gaps[n] = gap_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    ax_cost.grid(True, which='major', alpha=0.8)
    ax_cost.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_cost.set_xticks(np.linspace(10, 50, 5, endpoint=True))
    ax_cost.set_yticks(np.linspace(2, 26, 5, endpoint=True))
    # ax_cost.set_title(f"Costs", fontsize=title_fontsize)

    ax_time.grid(True, which='major', alpha=0.8)
    ax_time.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_time.set_xticks(np.linspace(10, 50, 5, endpoint=True))
    ax_time.set_yticks([0, 25, 50, 75, 99])
    # ax_time.set_title(f"Runtimes (secs.)", fontsize=title_fontsize)

    ax_gap.grid(True, which='major', alpha=0.8)
    ax_gap.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_gap.set_xticks(np.linspace(10, 50, 5, endpoint=True))
    ax_gap.set_yticks([0.0, 0.3, 0.6, 0.9])
    # ax_gap.set_title(f"Optimality Gap", fontsize=title_fontsize)

    if fig is not None:
        handles, labels = ax_cost.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=legend_fontsize, 
                bbox_to_anchor=(0.5, 1.25), frameon=True, 
                columnspacing=1, ncols=len(methods), prop={'size': legend_fontsize})
        fig.savefig("exp_linear.pdf", bbox_inches='tight', dpi=500)


def plot_bezier_res(num_seeds=12, axes=None):
    methods = ["ghost-ecg", "greedy", "micp", "ghost-eps0.5", "ghost"]
    colors = {"ghost": "k", "micp": "b", "ghost-ecg": "orange", "greedy": "gray", "ghost-eps0.5": "brown"}
    lines = {"ghost": "-", "micp": "-", "ghost-ecg": "-", "greedy": "-", "ghost-eps0.5": "-"}
    markers = {"ghost": ".", "micp": ".", "ghost-ecg": ".", "greedy": ".", "ghost-eps0.5": "."}
    labels = {"ghost": "GHOST", "micp": "MICP", "ghost-ecg": "ECG", "greedy": "Greedy", "ghost-eps0.5": "0.5-GHOST"}

    if axes is None:
        num_cols, nrows = 3, 1
        widths = [1.5] * num_cols
        heights = [1.5] * nrows
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axes = plt.subplots(ncols=num_cols, nrows=nrows, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2*num_cols, sum(heights)))
        ax_cost: Axes = axes[0]
        ax_gap: Axes = axes[1]
        ax_time: Axes = axes[2]
    else:
        fig = None
        ax_cost, ax_gap, ax_time = axes
    
    list_num_sets = [int(n) for n in range(5, 26, 1)]

    def _plot_func(costs, runtimes, opt_gaps, method):
        
        nonempty = [n for n in list_num_sets if n in costs]
        
        # directed edges = 2 * undirected edges
        ax_cost.plot([2*n for n in nonempty], [costs[n] for n in nonempty], 
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method],
                    lw=2, ms=4, alpha=0.6)
        
        ax_time.plot([2*n for n in nonempty], [runtimes[n] for n in nonempty], 
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method], 
                    lw=2, ms=4, alpha=0.6)

        nonempty = [n for n in list_num_sets if n in opt_gaps]
        ax_gap.plot([2*n for n in nonempty], [opt_gaps[n] for n in nonempty],
                    color=colors[method], 
                    linestyle=lines[method], 
                    marker=markers[method], 
                    label=labels[method],
                    lw=2, ms=4, alpha=0.6)

    # with open(f"data/exp/bezier-micp.pkl", 'rb') as f:
    #     method, data = "micp", pickle.load(f)
    #     costs, runtimes, opt_gaps = {}, {}, {}
    #     for n in list_num_sets:
    #         cost_in_total, runtime_in_total, gap_in_total, seeds_in_total = 0, 0, 0, 0
    #         for seed in range(num_seeds):
    #             if (seed, n) not in data:
    #                 continue

    #             seeds_in_total += 1
    #             gap_in_total += data[(seed, n)]['opt_gap']
    #             cost_in_total += data[(seed, n)]['cost']
    #             runtime_in_total += min(100, data[(seed, n)]['comp_time'] + data[(seed, n)]['solver_time'])

    #         if seeds_in_total != 0:
    #             costs[n] = cost_in_total / seeds_in_total
    #             runtimes[n] = runtime_in_total / seeds_in_total
    #             opt_gaps[n] = gap_in_total / seeds_in_total
        
    #     _plot_func(costs, runtimes, opt_gaps, method)
        
    with open("data/exp/bezier-ghost-ecg.pkl", 'rb') as f:
        method, data = "ghost-ecg", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total = 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/bezier-greedy.pkl", "rb") as f:
        method, data = "greedy", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total = 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                cost_in_total += data[(seed, n)]['best_cost']
                runtime_in_total += data[(seed, n)]['lbg_time'] + data[(seed, n)]['greedy_time']

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/bezier-eps0.5-ghost.pkl", 'rb') as f:
        method, data = "ghost-eps0.5", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total = 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue
                gap = (data[(seed, n)]['opt_node'].ub - data[(seed, n)]['lb']) / data[(seed, n)]['opt_node'].ub
                gap_in_total += max(gap, 0.0)
                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
            opt_gaps[n] = gap_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    with open("data/exp/bezier-ghost.pkl", 'rb') as f:
        method, data = "ghost", pickle.load(f)
        costs, runtimes, opt_gaps = {}, {}, {}
        for n in list_num_sets:
            cost_in_total, runtime_in_total, gap_in_total = 0, 0, 0
            for seed in range(num_seeds):
                if (seed, n) not in data:
                    continue

                if min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time']) < 90:
                    # means the solver searches all nodes
                    gap = 0.0
                else:
                    gap = (data[(seed, n)]['opt_node'].ub - data[(seed, n)]['lb']) / data[(seed, n)]['opt_node'].ub
                    
                gap_in_total += max(gap, 0.0)
                cost_in_total += data[(seed, n)]['opt_node'].ub
                runtime_in_total += min(100, data[(seed, n)]['lbg_time'] + data[(seed, n)]['osf_time'])

            costs[n] = cost_in_total / num_seeds
            runtimes[n] = runtime_in_total / num_seeds
            opt_gaps[n] = gap_in_total / num_seeds
        
        _plot_func(costs, runtimes, opt_gaps, method)

    ax_cost.grid(True, which='major', alpha=0.8)
    ax_cost.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_cost.set_xticks(np.linspace(10, 50, 5, endpoint=True))
    ax_cost.set_yticks(np.linspace(2, 22, 5, endpoint=True))
    # ax_cost.set_title(f"Costs", fontsize=title_fontsize)

    ax_time.grid(True, which='major', alpha=0.8)
    ax_time.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_time.set_xticks(np.linspace(10, 50, 5, endpoint=True))
    ax_time.set_yticks([0, 25, 50, 75, 99])
    # ax_time.set_title(f"Runtimes (secs.)", fontsize=title_fontsize)

    ax_gap.grid(True, which='major', alpha=0.8)
    ax_gap.tick_params(axis='both', which='major', labelsize=axis_fontsize)
    ax_gap.set_xticks(np.linspace(10, 50, 5, endpoint=True))
    ax_gap.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    # ax_gap.set_title(f"Optimality Gap", fontsize=title_fontsize)

    if fig is not None:
        handles, labels = ax_cost.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=legend_fontsize, 
                bbox_to_anchor=(0.5, 1.25), frameon=True, 
                columnspacing=1, ncols=len(methods), prop={'size': legend_fontsize})
        fig.savefig("exp_bezier.pdf", bbox_inches='tight', dpi=500)


def plot():
    num_cols, nrows = 3, 3
    widths = [1.5] * num_cols
    heights = [1.5] * nrows
    gs_kw = dict(width_ratios=widths, height_ratios=heights)

    fig, axes = plt.subplots(ncols=num_cols, nrows=nrows, constrained_layout=True, gridspec_kw=gs_kw, figsize=(2*num_cols, sum(heights)))
    ax_r1 = [axes[0, 0], axes[0, 1], axes[0, 2]]
    ax_r2 = [axes[1, 0], axes[1, 1], axes[1, 2]]
    ax_r3 = [axes[2, 0], axes[2, 1], axes[2, 2]]

    plot_point_res(axes=ax_r1)
    plot_linear_res(axes=ax_r2)
    plot_bezier_res(axes=ax_r3)

    legend_fontsize = 12
    handles, labels = ax_r1[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", fontsize=legend_fontsize, 
            bbox_to_anchor=(0.5, 1.09), frameon=True, 
            columnspacing=0.75, ncols=5, prop={'size': legend_fontsize})
    fig.savefig("exp.pdf", bbox_inches='tight', dpi=500)


plot()
