import argparse
import copy
import csv
import os
import os.path as osp
import random

import numpy as np
import yaml

import graphgym.contrib  # noqa
from graphgym.utils.comp_budget import match_baseline_cfg
from graphgym.utils.io import makedirs_rm_exist, string_to_python

random.seed(123)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        dest='config',
                        help='the base configuration file used for edit',
                        default=None,
                        type=str)
    parser.add_argument('--grid',
                        dest='grid',
                        help='configuration file for grid search',
                        required=True,
                        type=str)
    parser.add_argument('--sample',
                        dest='sample',
                        help='whether perform random sampling',
                        action='store_true',
                        required=False)
    parser.add_argument('--sample_alias',
                        dest='sample_alias',
                        help='configuration file for sample alias',
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument('--sample_num',
                        dest='sample_num',
                        help='Number of random samples in the space',
                        default=10,
                        type=int)
    parser.add_argument('--out_dir',
                        dest='out_dir',
                        help='output directory for generated config files',
                        default='configs',
                        type=str)
    parser.add_argument(
        '--config_budget',
        dest='config_budget',
        help='the base configuration file used for matching computation',
        default=None,
        type=str)
    return parser.parse_args()


def get_fname(string):
    if string is not None:
        return string.split('/')[-1].split('.')[0]
    else:
        return 'default'


def grid2list(grid):
    configs_in = [[]]
    for grid_temp in grid:
        configs_out = []
        for val in grid_temp:
            for list_temp in configs_in:
                configs_out.append(list_temp + [val])
        configs_in = configs_out
    return configs_in


def lists_distance(l1, l2):
    assert len(l1) == len(l2)
    dist = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            dist += 1
    return dist


def grid2list_sample(grid, sample=10):
    configs = []
    while len(configs) < sample:
        config = []
        for grid_temp in grid:
            config.append(random.choice(grid_temp))
        if config not in configs:
            configs.append(config)
    return configs


def load_config(fname):
    if fname is not None:
        with open(fname) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    else:
        return {}


def load_grid(fname):
    r'''Load a grid file for architecture search'''
    with open(fname, 'r') as f:
        rows = f.readlines()
        outs = []
        out = []
        for row in rows:
            if '#' in row:
                continue
            row = row.split()
            if len(row) > 0:
                row = row[:2] + [''.join(row[2:])]
                assert len(row) == 3, \
                    'File cannot be parsed'
                out.append(row)
            else:
                if len(out) > 0:
                    outs.append(out)
                out = []
        if len(out) > 0:
            outs.append(out)
    return outs


def merge_dict(a, b):
    r'''Merge b into a (2 nested dictionaries)'''
    if isinstance(a, dict):
        if isinstance(b, dict):
            for key in b:
                if key in a:
                    a[key] = merge_dict(a[key], b[key])
                else:
                    a[key] = b[key]
    else:
        a = b


def load_alias(fname):
    with open(fname, 'r') as f:
        file = csv.reader(f, delimiter=' ')
        for line in file:
            break
    return line


def exclude_list_id(list, id):
    return [list[i] for i in range(len(list)) if i != id]


def design_space_count(grid, sample_num):
    counts = []
    for out in grid:
        vars_grid = [string_to_python(row[2]) for row in out]
        count = 1
        for var in vars_grid:
            count *= len(var)
        counts.append(count)
    counts = np.array(counts)
    counts = (counts / np.sum(counts) * sample_num).astype(int)
    counts[0] += sample_num - np.sum(counts)
    print('Total sample size of each chunk of experiment space:', counts)
    return counts


def gen_grid(args, config, config_budget={}):
    r'''Generate a batch of experiment configs based on the grid file'''
    task_name = f'{get_fname(args.config)}_grid_{get_fname(args.grid)}'
    fname_base = get_fname(args.config)
    out_dir = osp.join(args.out_dir, task_name)
    makedirs_rm_exist(out_dir)
    config['out_dir'] = os.path.join(config['out_dir'], task_name)
    outs = load_grid(args.grid)

    if args.sample:
        counts = design_space_count(outs, args.sample_num)

    for i, out in enumerate(outs):
        vars_label = [row[0].split('.') for row in out]
        vars_alias = [row[1] for row in out]

        if args.sample:
            vars_value = grid2list_sample(
                [string_to_python(row[2]) for row in out], sample=counts[i])
        else:
            vars_value = grid2list([string_to_python(row[2]) for row in out])

        if i == 0:
            print(f'Variable label: {vars_label}')
            print(f'Variable alias: {vars_alias}')

        for vars in vars_value:
            config_out = copy.deepcopy(config)
            fname_out = fname_base
            for j, var_label in enumerate(vars_label):
                ptr = config_out
                for item in var_label[:-1]:
                    if item not in ptr:
                        ptr[item] = {}
                    ptr = ptr[item]
                ptr[var_label[-1]] = vars[j]
                var_repr = str(vars[j]).strip("[]").strip("''")
                fname_out += f'-{vars_alias[j]}={var_repr}'
            if len(config_budget) > 0:
                config_out = match_baseline_cfg(config_out, config_budget)
            with open(osp.join(out_dir, f'{fname_out[:250]}.yaml'), 'w') as f:
                yaml.dump(config_out, f, default_flow_style=False)
        print(f'{len(vars_value)} configurations saved to: {out_dir}')


def gen_grid_control_sample(args,
                            config,
                            config_budget={},
                            compare_alias_list=[]):
    task_name = f'{get_fname(args.config)}_grid_{get_fname(args.grid)}'
    fname_base = get_fname(args.config)
    out_dir = osp.join(args.out_dir, task_name)
    makedirs_rm_exist(out_dir)
    config['out_dir'] = os.path.join(config['out_dir'], task_name)
    outs = load_grid(args.grid)

    counts = design_space_count(outs, args.sample_num)

    for i, out in enumerate(outs):
        vars_label = [row[0].split('.') for row in out]
        vars_alias = [row[1] for row in out]
        if i == 0:
            print(f'Variable label: {vars_label}')
            print(f'Variable alias: {vars_alias}')
        vars_grid = [string_to_python(row[2]) for row in out]
        for alias in compare_alias_list:
            alias_id = vars_alias.index(alias)
            vars_grid_select = copy.deepcopy(vars_grid[alias_id])
            vars_grid[alias_id] = [vars_grid[alias_id][0]]
            vars_value = grid2list_sample(vars_grid, counts[i])

            vars_value_new = []
            for vars in vars_value:
                for grid in vars_grid_select:
                    vars[alias_id] = grid
                    vars_value_new.append(copy.deepcopy(vars))
            vars_value = vars_value_new

            vars_grid[alias_id] = vars_grid_select
            for vars in vars_value:
                config_out = config.copy()
                fname_out = fname_base + f'-sample={vars_alias[alias_id]}'
                for id, var in enumerate(vars):
                    if len(vars_label[id]) == 1:
                        config_out[vars_label[id][0]] = var
                    elif len(vars_label[id]) == 2:
                        if vars_label[id][0] in config_out:  # if key1 exist
                            config_out[vars_label[id][0]][vars_label[id]
                                                          [1]] = var
                        else:
                            config_out[vars_label[id][0]] = {
                                vars_label[id][1]: var
                            }
                    else:
                        raise ValueError(
                            'Only 2-level config files are supported')
                    var_repr = str(var).strip("[]").strip("''")
                    fname_out += f'-{vars_alias[id]}={var_repr}'
                if len(config_budget) > 0:
                    config_out = match_baseline_cfg(config_out,
                                                    config_budget,
                                                    verbose=False)
                with open(osp.join(out_dir, f'{fname_out[:250]}.yaml'),
                          'w') as f:
                    yaml.dump(config_out, f, default_flow_style=False)
            print(f'Chunk {i + 1}/{len(outs)}: '
                  f'Perturbing design dimension {alias}, '
                  f'{len(vars_value)} configurations saved to: {out_dir}')


args = parse_args()
config = load_config(args.config)
config_budget = load_config(args.config_budget)
if args.sample_alias is None:
    gen_grid(args, config, config_budget)
else:
    alias_list = load_alias(args.sample_alias)
    gen_grid_control_sample(args, config, config_budget, alias_list)
