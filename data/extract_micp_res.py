import os, sys, pickle
print()
import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Extract MICP results from log files.")
    argparser.add_argument('name', type=str, choices=['point', 'linear', 'bezier'])
    args = argparser.parse_args()
    
    res = {}
    directory = os.path.join(os.path.dirname(__file__), f'{args.name}-micp')

    for fn in os.listdir(directory):
        seed, num_sets = [int(val) for val in fn[:-4].split('-')]
        with open(os.path.join(directory, fn), 'r') as f:
            lines = f.readlines()
            found_summary = False
            for i, line in enumerate(lines):
                if "Summary" in line:
                    found_summary = True
                    cost = float(lines[i+3].split(':')[-1].strip())
                    comp_time = float(lines[i+4].split(' ')[-2].strip())
                    solver_time = float(lines[i+5].split(' ')[-2].strip())
                    res[(seed, num_sets)] = {
                        'cost': cost,
                        'opt_gap': 0.0,
                        'comp_time': comp_time,
                        'solver_time': solver_time
                    }

            if len(lines) != 0 and not found_summary:
                items = [val for val in lines[-1].strip().split(" ") if val != '']
                try:
                    res[(seed, num_sets)] = {
                        'cost': float(items[-4]),
                        'opt_gap': float(items[-2]) / 100,
                        'comp_time': 100 - float(items[-1]),
                        'solver_time': float(items[-1])
                    } 
                except Exception as e:
                    continue

    with open(f'{directory}.pkl', 'wb') as f:
        pickle.dump(res, f)
