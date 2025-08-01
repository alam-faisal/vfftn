from mpo import *

def kronecker_pad(matrix, num_qubits, starting_site): 
    ''' takes a local gate described as a matrix and pads it with identity matrices to create a global operator '''
    kron_list = [np.eye(2) for i in range(num_qubits)]    
    kron_list[starting_site] = matrix
    if matrix.shape[0] == 4: 
        del kron_list[starting_site+1]
    padded_matrix = kron_list[0]
    for i in range(1, len(kron_list)):
        padded_matrix = np.kron(kron_list[i], padded_matrix)    
    return padded_matrix

def xy_ham(num_qubits): 
    terms = []        
    for i in range(num_qubits-1): 
        y_hop = kronecker_pad(pauli_tensor[2,2], num_qubits, i)
        terms.append(y_hop)
        x_hop = kronecker_pad(pauli_tensor[1,1], num_qubits, i)
        terms.append(x_hop)
    return sum(terms) 

def xy_gate(t):
    return expm(1.j * t * (pauli_tensor[1,1] + pauli_tensor[2,2]))    

def xy_even_layer_mpo(num_qubits, t):
    node_list = []
    for i in range(0, num_qubits, 2): 
        top_node, bottom_node = gate_to_nodes(xy_gate(t))
        node_list.append(top_node)       
        node_list.append(bottom_node)
    return MPO(node_list)

def xy_odd_layer_mpo(num_qubits, t):
    node_list = [Node(np.eye(2)[np.newaxis,np.newaxis,:,:])]
    for i in range(1, num_qubits-1, 2): 
        top_node, bottom_node = gate_to_nodes(xy_gate(t))
        node_list.append(top_node)
        node_list.append(bottom_node)
    node_list.append(Node(np.eye(2)[np.newaxis,np.newaxis,:,:]))
    return MPO(node_list)

beta1 = 1/(2-np.cbrt(2))
beta2 = -np.cbrt(2)*beta1
beta3 = beta1
alpha1 = beta1/2
alpha2 = (1-beta1)/2
alpha3 = (1-beta1)/2
alpha4 = beta1/2

def xy_layer(num_qubits, t, order=1, contracted=True):
    if contracted: 
        if order == 1:
            mpo = (xy_odd_layer_mpo(num_qubits, t) @ xy_even_layer_mpo(num_qubits,t))
        elif order == 2: 
            mpo = xy_even_layer_mpo(num_qubits, t/2) @ xy_odd_layer_mpo(num_qubits, t) @ xy_even_layer_mpo(num_qubits, t/2)
        elif order == 4: 
            mpo = xy_even_layer_mpo(num_qubits, t*alpha1)
            mpo = xy_odd_layer_mpo(num_qubits, t*beta1) @ mpo
            mpo = xy_even_layer_mpo(num_qubits, t*alpha2) @ mpo
            mpo = xy_odd_layer_mpo(num_qubits, t*beta2) @ mpo
            mpo = xy_even_layer_mpo(num_qubits, t*alpha3) @ mpo
            mpo = xy_odd_layer_mpo(num_qubits, t*beta3) @ mpo
            mpo = xy_even_layer_mpo(num_qubits, t*alpha4) @ mpo
        return mpo
    else: 
        if order == 1: 
            mpo_list = [xy_even_layer_mpo(num_qubits,t), xy_odd_layer_mpo(num_qubits, t)]
        elif order == 2: 
            mpo_list = [xy_even_layer_mpo(num_qubits, t/2), xy_odd_layer_mpo(num_qubits, t), xy_even_layer_mpo(num_qubits, t/2)]
        elif order == 4: 
            mpo_list = [xy_even_layer_mpo(num_qubits, t*alpha1), xy_odd_layer_mpo(num_qubits, t*beta1), 
                        xy_even_layer_mpo(num_qubits, t*alpha2), xy_odd_layer_mpo(num_qubits, t*beta2), 
                        xy_even_layer_mpo(num_qubits, t*alpha3), xy_odd_layer_mpo(num_qubits, t*beta3), 
                        xy_even_layer_mpo(num_qubits, t*alpha4)]
        return mpo_list

def xy_mpo(num_qubits, t, num_trotter_layers=1, order=1, min_sv_ratio=None, max_dim=None, contracted=True): 
    ''' breaks down t into num_trotter_layers, creates MPO for each step, contracts, truncates '''
    single_step = xy_layer(num_qubits, t/num_trotter_layers, order=order, contracted=contracted)
    if not contracted:
        return single_step * num_trotter_layers
    else: 
        single_step = single_step.compress(min_sv_ratio,max_dim)
        mpo = copy.deepcopy(single_step)
        for step in range(1,num_trotter_layers): 
            mpo = (single_step @ mpo).compress(min_sv_ratio, max_dim)
        return mpo
        
def xxz_gate(t, jz=3):
    return expm(1.j * t * (pauli_tensor[1,1] + pauli_tensor[2,2] + jz * pauli_tensor[3,3]))    

def xxz_even_layer_mpo(num_qubits, t):
    node_list = []
    for i in range(0, num_qubits, 2): 
        top_node, bottom_node = gate_to_nodes(xxz_gate(t))
        node_list.append(top_node)       
        node_list.append(bottom_node)
    return MPO(node_list)

def xxz_odd_layer_mpo(num_qubits, t):
    node_list = [Node(np.eye(2)[np.newaxis,np.newaxis,:,:])]
    for i in range(1, num_qubits-1, 2): 
        top_node, bottom_node = gate_to_nodes(xxz_gate(t))
        node_list.append(top_node)
        node_list.append(bottom_node)
    node_list.append(Node(np.eye(2)[np.newaxis,np.newaxis,:,:]))
    return MPO(node_list)

beta1 = 1/(2-np.cbrt(2))
beta2 = -np.cbrt(2)*beta1
beta3 = beta1
alpha1 = beta1/2
alpha2 = (1-beta1)/2
alpha3 = (1-beta1)/2
alpha4 = beta1/2

def xxz_layer(num_qubits, t, order=1, contracted=True):
    if contracted: 
        if order == 1:
            mpo = (xxz_odd_layer_mpo(num_qubits, t) @ xxz_even_layer_mpo(num_qubits,t))
        elif order == 2: 
            mpo = xxz_even_layer_mpo(num_qubits, t/2) @ xxz_odd_layer_mpo(num_qubits, t) @ xxz_even_layer_mpo(num_qubits, t/2)
        elif order == 4: 
            mpo = xxz_even_layer_mpo(num_qubits, t*alpha1)
            mpo = xxz_odd_layer_mpo(num_qubits, t*beta1) @ mpo
            mpo = xxz_even_layer_mpo(num_qubits, t*alpha2) @ mpo
            mpo = xxz_odd_layer_mpo(num_qubits, t*beta2) @ mpo
            mpo = xxz_even_layer_mpo(num_qubits, t*alpha3) @ mpo
            mpo = xxz_odd_layer_mpo(num_qubits, t*beta3) @ mpo
            mpo = xxz_even_layer_mpo(num_qubits, t*alpha4) @ mpo
        return mpo
    else: 
        if order == 1: 
            mpo_list = [xxz_even_layer_mpo(num_qubits,t), xxz_odd_layer_mpo(num_qubits, t)]
        elif order == 2: 
            mpo_list = [xxz_even_layer_mpo(num_qubits, t/2), xxz_odd_layer_mpo(num_qubits, t), xxz_even_layer_mpo(num_qubits, t/2)]
        elif order == 4: 
            mpo_list = [xxz_even_layer_mpo(num_qubits, t*alpha1), xxz_odd_layer_mpo(num_qubits, t*beta1), 
                        xxz_even_layer_mpo(num_qubits, t*alpha2), xxz_odd_layer_mpo(num_qubits, t*beta2), 
                        xxz_even_layer_mpo(num_qubits, t*alpha3), xxz_odd_layer_mpo(num_qubits, t*beta3), 
                        xxz_even_layer_mpo(num_qubits, t*alpha4)]
        return mpo_list

def xxz_mpo(num_qubits, t, num_trotter_layers=1, order=1, min_sv_ratio=None, max_dim=None, contracted=True): 
    ''' breaks down t into num_trotter_layers, creates MPO for each step, contracts, truncates '''
    single_step = xxz_layer(num_qubits, t/num_trotter_layers, order=order, contracted=contracted)
    if not contracted:
        return single_step * num_trotter_layers
    else: 
        single_step = single_step.compress(min_sv_ratio,max_dim)
        mpo = copy.deepcopy(single_step)
        for step in range(1,num_trotter_layers): 
            mpo = (single_step @ mpo).compress(min_sv_ratio, max_dim)
        return mpo
        
