from mpo import *

def rz_gate(t, theta): 
    """ rz rotation gate by angle t*theta """
    return np.array([[np.exp(-1.j*theta*t/2), 0], [0, np.exp(1.j*theta*t/2)]])

def givens_gate(theta): 
    return np.array([[1,0,0,0],
                     [0, np.cos(theta/2), -np.sin(theta/2), 0], 
                     [0, np.sin(theta/2), np.cos(theta/2), 0], 
                     [0,0,0,1]])

def one_qubit_gate(params): 
    """ general single qubit gate from three Euler angles """
    theta, lamb, phi = params
    return np.array([[np.cos(theta/2), -np.exp(1.j*lamb)*np.sin(theta/2)], 
                 [np.exp(1.j*phi)*np.sin(theta/2), np.exp(1.j*(lamb+phi))*np.cos(theta/2)]])

def two_qubit_gate(*args):
    """ 
    return general two qubit gate as a 4 by 4 matrix 
    if single arg is passed, it has shape (15,) and contains KAK parameters 
    if two args are passed they have shapes (1,b,2,2) and (b,1,2,2) respectively
    and represent two nodes of a two qubit gate within a brickwall circuit
    """ 
    if len(args) == 1:
        params = args[0]
        left_mat = np.kron(one_qubit_gate(params[0:3]), one_qubit_gate(params[3:6]))
        arg = sum([params[i]*pauli_tensor[(i-5,i-5)] for i in range(6,9)])
        center_mat = expm(1.j*arg)
        right_mat = np.kron(one_qubit_gate(params[9:12]), one_qubit_gate(params[12:15]))
        return (left_mat @ center_mat @ right_mat)#.real
    else:
        data1, data2 = args
        contracted = contract_bond(data1, data2)[0,0,:,:,:,:]
        return group_ind(group_ind(contracted, 0,1), 1,2)

class Rotation_MPO(MPO): 
    """
    single layer of Rz gates turned into nodes with bond_dim=2
    rz_angles must be np.array of shape (num_qubits,) 
    """
    def __init__(self, t, rz_angles):
        nodes = [Node(rz_gate(t, angle)[np.newaxis,np.newaxis,:,:]) for angle in rz_angles]
        super().__init__(nodes)
        
class Brickwall_MPO(MPO): 
    """
    returns MPOs of bond_dim=4, representing a long or short stack of two qubit gates in brickwall pattern
    w_stack_input can be a list of Givens parameters sent as an np.array of shape (num_qubits//2,) or (num_qubits//2 -1,)
    w_stack_input can be a list of KAK parameters sent as an np.array of shape (num_qubits//2,15) or (num_qubits//2 -1,15)
    w_stack_input can also be a list of two qubit matrices of size num_qubits//2 or num_qubits//2 -1
    short is True for short stacks and False for long stacks
    """ 
    def __init__(self, w_stack_input, short):
        if type(w_stack_input) == np.ndarray: 
            gate = givens_gate if len(w_stack_input.shape) == 1 else two_qubit_gate
            self.gate_list = [gate(params) for params in w_stack_input]
        else: 
            self.gate_list = w_stack_input
        
        nodes = [gate_to_nodes(gate) for gate in self.gate_list]
        nodes = list(sum(nodes, ()))
        
        if short: 
            eye_node = Node(np.eye(2,2)[np.newaxis,np.newaxis,:,:])
            nodes = [eye_node] + nodes + [eye_node] 
        
        super().__init__(nodes)
            
class Ansatz():
    """
    contains a collection of MPOs
    rz_angles is an np.array of shape (num_qubits,) 
    w_input can be an np.array of shape (num_w_layers, num_qubits-1,)
    w_input can be an np.array of shape (num_w_layers, num_qubits-1, 15)
    w_input can also be a list of num_w_layers lists each containing num_qubits-1 two qubit matrices
    """
    def __init__(self, t, rz_angles, w_input):
        self.t = t
        self.num_qubits = len(rz_angles)
        self.num_w_layers = len(w_input)
        self.num_stacks = 2*len(w_input)+1  
        self.rz_angles = rz_angles
        self.w_input = w_input
        if type(w_input) == np.ndarray: 
            self.w_type = 'givens' if len(w_input.shape) == 2 else 'kak_params'
        else: 
            self.w_type = 'kak_gates'
        
        stacks = [(Brickwall_MPO(w_input[i][:self.num_qubits//2], False), Brickwall_MPO(w_input[i][self.num_qubits//2:], True)) 
                                                                                     for i in range(self.num_w_layers)]
        self.param_mpo_stacks = [Rotation_MPO(t, rz_angles)] + list(sum(stacks, ()))       
        
    def mpo(self, min_sv_ratio=None, max_dim=None):
        mpo = self.param_mpo_stacks[0]
        for i in range(1,self.num_stacks): 
            mpo = (self.param_mpo_stacks[i] @ mpo @ self.param_mpo_stacks[i].conj()).compress(min_sv_ratio, max_dim)
        return mpo
    
    def overlap(self, target_mpo, min_sv_ratio=None, max_dim=None): 
        return target_mpo.conj().mult_and_trace(self.mpo(min_sv_ratio, max_dim))[0,0,0,0].real