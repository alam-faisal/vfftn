from mpo import *
from functools import reduce

def kronecker_pad(matrix, num_qubits, starting_site):
    kron_list = [np.eye(2) for _ in range(num_qubits)]
    kron_list[starting_site] = matrix
    if matrix.shape[0] == 4:
        del kron_list[starting_site+1]

    if matrix.shape[0] == 8: 
        del kron_list[starting_site+1]
        del kron_list[starting_site+1]

    return reduce(np.kron, kron_list[::-1])

def xy_ham(num_qubits):
    return sum(kronecker_pad(pauli_tensor[i,i], num_qubits, j) 
               for j in range(num_qubits-1) for i in (1, 2))

def heisenberg_ham(num_qubits, jx=1, jy=1, jz=1, h=1):
    ''' full matrix representation of XY Hamiltonian '''
    terms = []
    for i in range(num_qubits-1): 
        terms.append(kronecker_pad(jx*pauli_tensor[1,1] + jy*pauli_tensor[2,2] + jz*pauli_tensor[3,3], num_qubits, i))
    for i in range(num_qubits):
        terms.append(kronecker_pad(h*pauli[3], num_qubits, i))
    return sum(terms)

def create_gate(t, operator):
    return expm(1.j * t * operator)

def xy_gate(t):
    return create_gate(t, pauli_tensor[1,1] + pauli_tensor[2,2])

def xxz_gate(t, jz=3):
    return create_gate(t, pauli_tensor[1,1] + pauli_tensor[2,2] + jz * pauli_tensor[3,3])

def create_layer_mpo(num_qubits, t, gate_func, even=True):
    node_list = []
    start = 0 if even else 1
    end = num_qubits if even else num_qubits - 1
    
    if not even:
        node_list.append(Node(np.eye(2)[np.newaxis,np.newaxis,:,:]))
    
    for i in range(start, end, 2):
        top_node, bottom_node = gate_to_nodes(gate_func(t))
        node_list.extend([top_node, bottom_node])
    
    if not even:
        node_list.append(Node(np.eye(2)[np.newaxis,np.newaxis,:,:]))
    
    return MPO(node_list)

def create_layer(num_qubits, t, gate_func, order=1, contracted=True):
    beta1, beta2, beta3 = 1/(2-np.cbrt(2)), -np.cbrt(2)/(2-np.cbrt(2)), 1/(2-np.cbrt(2))
    alpha1, alpha2, alpha3, alpha4 = beta1/2, (1-beta1)/2, (1-beta1)/2, beta1/2

    even_layer = lambda t: create_layer_mpo(num_qubits, t, gate_func, even=True)
    odd_layer = lambda t: create_layer_mpo(num_qubits, t, gate_func, even=False)

    if order == 1:
        layers = [even_layer(t), odd_layer(t)]
    elif order == 2:
        layers = [even_layer(t/2), odd_layer(t), even_layer(t/2)]
    elif order == 4:
        layers = [even_layer(t*a) if i % 2 == 0 else odd_layer(t*b) 
                  for i, (a, b) in enumerate(zip([alpha1, alpha2, alpha3, alpha4], 
                                                 [beta1, beta2, beta3]))]
    if contracted:
        mpo = layers[0]
        for layer in layers[1:]:
            mpo = layer @ mpo
        return mpo
    else:
        return layers

def create_mpo(num_qubits, t, gate_func, num_trotter_layers=1, order=1, min_sv_ratio=None, max_dim=None, contracted=True):
    single_step = create_layer(num_qubits, t/num_trotter_layers, gate_func, order=order, contracted=contracted)
    
    if not contracted:
        return single_step * num_trotter_layers
    
    single_step = single_step.compress(min_sv_ratio, max_dim)
    mpo = single_step
    for _ in range(1, num_trotter_layers):
        mpo = (single_step @ mpo).compress(min_sv_ratio, max_dim)
    return mpo

def xy_mpo(num_qubits, t, num_trotter_layers=1, order=1, min_sv_ratio=None, max_dim=None, contracted=True):
    return create_mpo(num_qubits, t, xy_gate, num_trotter_layers, order, min_sv_ratio, max_dim, contracted)

def xxz_mpo(num_qubits, t, num_trotter_layers=1, order=1, min_sv_ratio=None, max_dim=None, contracted=True):
    return create_mpo(num_qubits, t, xxz_gate, num_trotter_layers, order, min_sv_ratio, max_dim, contracted)