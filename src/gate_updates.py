from ansatz import * 
from scipy.optimize import minimize

def rz_update(ansatz, stack_idx, node_idx, n_env): 
    """ sol = theta*t/2, image of arctan is -pi/2 to pi/2, so this allows to keep D layer close to identity """  
    t = ansatz.t
    coeff1 = n_env[0,0].real + n_env[1,1].real
    coeff2 = n_env[0,0].imag - n_env[1,1].imag 
    sol = np.arctan(coeff2/coeff1)
    cost = 1 - (coeff1 * np.cos(sol)) - (coeff2 * np.sin(sol))
    
    ansatz.rz_angles[node_idx] = sol*2/t
    new_gate = rz_gate(t, sol*2/t)[np.newaxis,np.newaxis,:,:]
    param_mpo = ansatz.param_mpo_stacks[stack_idx]
    param_mpo.nodes[node_idx] = Node(new_gate)
    return cost, new_gate

def givens_cost(theta, coeffs):
    return 1 - (coeffs @ np.array([1, np.cos(theta/2), np.sin(theta/2), np.cos(theta/2)*np.sin(theta/2), np.cos(theta/2)**2, 
                                   np.sin(theta/2)**2])).real

def givens_coeffs_and_args(env): 
    coeff1 = env[0,0] + env[15,15] + env[0,15] + env[15,0]
    coeff2 = env[0,3] + env[3,0] + env[0,12] + env[12,0] + env[3,15] + env[15,3] + env[12,15] + env[15,12]
    coeff3 = env[0,6] + env[6,0] + env[15,6] + env[6,15] - (env[0,9] + env[9,0] + env[9,15] + env[15,9])
    coeff4 = env[3,6] + env[6,3] + env[6,12] + env[12,6] - (env[3,9] + env[9,3] + env[9,12] + env[12,9])
    coeff5 = env[3,3] + env[12,12] + env[3,12] + env[12,3]
    coeff6 = env[6,6] + env[9,9] - env[6,9] - env[9,6]
    coeffs = np.array([coeff1, coeff2, coeff3, coeff4, coeff5, coeff6])
    
    first = 4*coeff4**2 + 4*coeff5**2 - 8*coeff5*coeff6 + 4*coeff6**2
    second = 4*coeff4*coeff2 - 4*coeff3*coeff5 + 4*coeff3*coeff6
    third = coeff2**2 + coeff3**2 - 4*coeff4**2 - 4*coeff5**2 + 8*coeff5*coeff6 - 4*coeff6**2 
    fourth = - 2*coeff4*coeff2 + 4*coeff3*coeff5 - 4*coeff3*coeff6
    fifth = coeff4**2 - coeff3**2
    args = [first.real, second.real, third.real, fourth.real, fifth.real]
    
    return coeffs, args
    
def givens_optimize(env): 
    coeffs, args = givens_coeffs_and_args(env)    
    roots = np.roots(args)
    roots = roots[(np.abs(roots.imag) < 1e-2) * (np.abs(roots) <= 1.0)].real
    angles = list(2*np.arcsin(roots)) + [-np.pi, np.pi]
    cost_list = [givens_cost(angle, coeffs) for angle in angles]
    opt_idx = np.argmin(cost_list)
    return angles[opt_idx], cost_list[opt_idx]
  
def givens_update(ansatz, stack_idx, node_idx, n_env): 
    n = ansatz.num_qubits
    env = n_env.transpose(2, 7, 0, 5, 3, 6, 1, 4).reshape(2**4,2**4) 
    result = givens_optimize(env)
       
    ansatz.w_input[(stack_idx-1)//2][((stack_idx+1)%2)*(n//2) + node_idx] = result[0]
    new_gate = givens_gate(result[0])
    nodes = gate_to_nodes(new_gate)
    param_mpo = ansatz.param_mpo_stacks[stack_idx]
    param_mpo.nodes[2*node_idx+(stack_idx+1)%2] = nodes[0]
    param_mpo.nodes[2*node_idx+1+(stack_idx+1)%2] = nodes[1]
    
    return result[1], nodes_to_gate(*nodes)

def kak_cost(kak_angles, env): 
    gate = two_qubit_gate(kak_angles)
    env = np.einsum('ba,abcd->dc', gate.conj(), env)
    return 1 - np.trace(env@gate).real

def kak_update(ansatz, stack_idx, node_idx, n_env, randomized=False): 
    n = ansatz.num_qubits
    env = n_env.transpose(5,7,0,2,1,3,4,6).reshape(4,4,4,4)
    init_kak_angles = ansatz.w_input[(stack_idx-1)//2][((stack_idx+1)%2)*(n//2) + node_idx]
    angle_bounds = [(-np.pi, np.pi)] * 15
    
    result1 = minimize(kak_cost, np.pi*(2*np.random.rand(15)-1), method='Nelder-Mead', bounds=angle_bounds, args=(env))
    result2 = minimize(kak_cost, init_kak_angles, method='Nelder-Mead', bounds=angle_bounds, args=(env))
    result = result2 if result2.fun < result1.fun else result1
    
    ansatz.w_input[(stack_idx-1)//2][((stack_idx+1)%2)*(n//2) + node_idx] = result.x
    new_gate = two_qubit_gate(result.x)
    nodes = gate_to_nodes(new_gate)
    param_mpo = ansatz.param_mpo_stacks[stack_idx]
    param_mpo.nodes[2*node_idx+(stack_idx+1)%2] = nodes[0]
    param_mpo.nodes[2*node_idx+1+(stack_idx+1)%2] = nodes[1]
    return result.fun, nodes_to_gate(*nodes)