import os
import sys
import re
import argparse
from itertools import product, chain, combinations
import subprocess
from pathlib import Path


def powerset(l): # list
	return list(chain(*[combinations(l, n) for n in range(len(l)+1)]))

def run_command(command, stdout_file, stderr_file=''):
    if not stderr_file:
        stderr_file = os.devnull
    with open(stdout_file, 'w') as stdout, open(stderr_file, 'w') as stderr:
        print(f"Run command: {' '.join(command)}")
        return subprocess.Popen(command , stdout=stdout, stderr=stderr)

def command_with_params(command, params_names, params_values, flags=[]):
    # command is a list
    params = ['='.join(n_v) for n_v in zip(params_names, params_values)]
    return command + params

def logfile_names(params, flags, logdir):
    base_name = '__'.join(params + flags)
    base_name = re.sub('\.|/', '_', base_name)
    base_name = re.sub('--', '', base_name)
    full_name = Path(logdir) / Path(base_name)
    return str(full_name) + '.log', str(full_name) + '.err' # TODO better


args_parser = argparse.ArgumentParser()
args_parser.add_argument(
    "--grid_dir",
    type=str,
    help="Path to a grid of parameters.")
args = args_parser.parse_args()

grid_file = Path(args.grid_dir) / Path('grid')
command_file = Path(args.grid_dir) / Path('command')
log_dir = Path(args.grid_dir) / Path('logs') # TODO date + time

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

with open(grid_file, 'r') as f:
    grid_lines = f.read().splitlines()

with open(command_file, 'r') as f:
    command = f.read().splitlines()[0].split(' ')
    print(command)

params_names = []
params_values = []
flags = []
for l in grid_lines:
    param_name, *param_values = l.split(' ')
    if param_values:
        params_names.append(param_name)
        params_values.append(param_values)
    else:
        flags.append(param_name)

params_combs = list(product(*params_values))
flags_combs = powerset(flags)

print(f"The size of the grid: {len(params_combs) * len(flags_combs)}.")
processes = []
for p in params_combs:
    for f in flags_combs:
        c_p = command_with_params(command, params_names, p, f)
        stdout, stderr = logfile_names(p, f, log_dir)
        processes.append(run_command(c_p, stdout, stderr))

for p in processes:
    p.wait()
