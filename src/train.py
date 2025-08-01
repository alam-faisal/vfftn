import time
import pickle
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
from tqdm import tqdm

from environments import *
from gate_updates import *
from models2 import *

@dataclass
class TrainingConfig:
    num_qubits: int
    t: float
    num_w_layers: int
    num_trotter_steps: int = 100
    rz_angles: Optional[np.ndarray] = None
    w_input: Optional[np.ndarray] = None
    w_type: str = 'givens'
    min_sv_ratio: Optional[float] = None
    max_dim: Optional[int] = None
    max_iter: int = 10000
    max_time: int = 170000
    verbose: bool = True
    p_bar: bool = True
    left_envs: Optional[List[MPO]] = None
    right_envs: Optional[List[MPO]] = None

def sweep_up_down(ansatz, stack_idx, bottom_envs, top_envs, link_datas1, link_datas2):
    param_mpo = ansatz.param_mpo_stacks[stack_idx]
    num_envs = param_mpo.num_nodes if stack_idx == 0 else param_mpo.num_nodes // 2 - (stack_idx + 1) % 2
    cost_list = []
    update = rz_update if stack_idx == 0 else (givens_update if ansatz.w_type == 'givens' else kak_update)

    for direction in ['down', 'up']:
        range_func = range(num_envs - 1, 0, -1) if direction == 'down' else range(num_envs)
        for node_idx in range_func:
            n_env = full_env(link_datas1[node_idx], link_datas2[node_idx], top_envs[node_idx], bottom_envs[node_idx])
            new_cost, new_data = update(ansatz, stack_idx, node_idx, n_env)
            cost_list.append(new_cost)
            if direction == 'down' and node_idx > 0:
                bottom_envs[node_idx - 1] = build_bottom_env(bottom_envs[node_idx], new_data, link_datas1[node_idx], link_datas2[node_idx])
            elif direction == 'up' and node_idx < num_envs - 1:
                top_envs[node_idx + 1] = build_top_env(top_envs[node_idx], new_data, link_datas1[node_idx], link_datas2[node_idx])

    return cost_list

def sweep_right_left(ansatz, left_envs, right_envs, config):
    cost_list = []
    num_qubits = config.num_qubits

    for direction in ['right', 'left']:
        range_func = range(ansatz.num_stacks - 1) if direction == 'right' else range(ansatz.num_stacks - 1, 0, -1)
        
        for stack_idx in range_func:
            stack_env1, stack_env2 = right_envs[stack_idx], left_envs[stack_idx]
            param_mpo = ansatz.param_mpo_stacks[stack_idx]
            bottom_envs, top_envs = all_node_envs(param_mpo, stack_idx, stack_env1, stack_env2)

            link_datas1, link_datas2 = get_link_datas(stack_env1, stack_env2, num_qubits, stack_idx)
            cost_list += sweep_up_down(ansatz, stack_idx, bottom_envs, top_envs, link_datas1, link_datas2)
            
            if direction == 'right':
                left_envs[stack_idx + 1] = build_left_env(left_envs[stack_idx], param_mpo, config.min_sv_ratio, config.max_dim)
            else:
                right_envs[stack_idx - 1] = build_right_env(right_envs[stack_idx], param_mpo, config.min_sv_ratio, config.max_dim)

    return cost_list

def initialize_ansatz(config):
    rz_angles = np.zeros(config.num_qubits) if config.rz_angles is None else config.rz_angles
    if config.w_input is None:
        w_input = (np.random.rand(config.num_w_layers, config.num_qubits - 1) if config.w_type == 'givens' 
                   else np.random.rand(config.num_w_layers, config.num_qubits - 1, 15))
    else:
        w_input = config.w_input
    return Ansatz(config.t, rz_angles, w_input)

def check_termination_condition(config, i, cost_data, time_data):
    if time_data[-1] > config.max_time:
        return 'max time elapsed'
    if i > 10:
        if np.abs((cost_data[-10] - cost_data[-1])/cost_data[-10]) < 1e-4:
            return 'cost function plateaued'
        elif cost_data[-1] < 1e-8:
            return 'cost function below target'
    return None

def train(config):
    ansatz = initialize_ansatz(config)
    
    if config.left_envs is None or config.right_envs is None:
        target_mpo = xy_mpo(config.num_qubits, config.t, config.num_trotter_steps, order=2, min_sv_ratio=1e-9)
        left_envs, right_envs = all_stack_envs(ansatz, target_mpo, config.min_sv_ratio, config.max_dim)
        print("Stack environments have been generated")
    else:
        left_envs, right_envs = config.left_envs, config.right_envs
        print("Using provided stack environments")

    cost_data, time_data = [1 - ansatz.param_mpo_stacks[0].mult_and_trace(right_envs[0])[0,0,0,0].real], [0.0]
    snapshots = {10**(-i): None for i in range(1,9)}
    begin_time = time.time()
    term_cond = 'max iteration reached'
    
    iters = tqdm(range(config.max_iter)) if config.p_bar else range(config.max_iter)
    for i in iters:
        new_cost_list = sweep_right_left(ansatz, left_envs, right_envs, config)
        cost_data.append(new_cost_list[-1])
        time_data.append(time.time() - begin_time)

        for threshold in sorted(snapshots.keys()):
            if cost_data[-1] < threshold and snapshots[threshold] is None:
                snapshots[threshold] = (ansatz.rz_angles.copy(), ansatz.w_input.copy())
        
        if config.verbose and i % 10 == 0:
            print(cost_data[-1])
            snapshots[cost_data[-1]] = (ansatz.rz_angles.copy(), ansatz.w_input.copy())
        
        termination_reason = check_termination_condition(config, i, cost_data, time_data)
        if termination_reason:
            term_cond = termination_reason
            print(term_cond)
            print(ansatz.num_qubits)
            print(time_data[-1])
            break

    results = {
        'angles': ansatz.rz_angles,
        'w_input': ansatz.w_input,
        'num_w_layers': config.num_w_layers,
        'cost_data': cost_data,
        'time_data': time_data,
        'snapshots': snapshots,
        'term_cond': term_cond,
        'left_envs': left_envs,
        'right_envs': right_envs
    }
    return results

def layergrowth_train(config):
    results = train(config)
    while results['term_cond'] == 'cost function plateaued':
        new_layer = (np.random.rand(1, config.num_qubits - 1) * 0.01 if config.w_type == 'givens' 
                     else np.random.rand(1, config.num_qubits - 1, 15) * 0.01)
        config.w_input = np.concatenate((new_layer, results['w_input']))
        config.num_w_layers += 1
        
        new_results = train(config)
        results = merge_results(results, new_results)
    
    return results

def merge_results(old_results, new_results):
    merged = new_results.copy()
    merged['cost_data'] = old_results['cost_data'] + new_results['cost_data']
    merged['time_data'] = old_results['time_data'] + new_results['time_data']
    merged['snapshots'] = {**old_results['snapshots'], **new_results['snapshots']}
    return merged

def load_previous_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def run(args):
    if len(args) not in [6, 7]:
        print("Invalid number of arguments")
        return

    output_file, num_qubits, t, num_w_layers, max_dim, training_type, *input_file = args
    config = TrainingConfig(
        num_qubits=int(num_qubits),
        t=float(t),
        num_w_layers=int(num_w_layers),
        max_dim=int(max_dim),
        max_time=170000,
        verbose=True,
        p_bar=False
    )

    if input_file:
        old_results = load_previous_results(input_file[0])
        config.rz_angles = old_results['angles']
        config.w_input = old_results['w_input']
        config.num_w_layers = len(config.w_input)
        config.left_envs = old_results['left_envs']
        config.right_envs = old_results['right_envs']

    results = layergrowth_train(config) if training_type == 'l' else train(config)

    if input_file:
        results = merge_results(old_results, results)

    save_results(results, output_file)

if __name__ == "__main__":
    run(sys.argv[1:])