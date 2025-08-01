import copy
import numpy as np
#from numpy.linalg import svd
from scipy.linalg import expm, svd
import pickle

default_min_sv_ratio = 1e-12

pauli = np.array([np.array([[1,0],[0,1]]), np.array([[0,1],[1,0]]), np.array([[0,-1.j],[1.j,0]]), np.array([[1,0],[0,-1]])])
pauli_tensor = np.array([[np.kron(pauli[i], pauli[j]) for i in range(4)] for j in range(4)])

########################
######### Node ########
########################

def group_ind(data, recipient_index, donor_index): 
    """ combines recipient_index and donor_index into a single index """
    data = np.moveaxis(data, donor_index, recipient_index+1)
    shape  = list(data.shape[:recipient_index]) + [data.shape[recipient_index] * data.shape[recipient_index+1]] \
            + list(data.shape[recipient_index+2:])
    return np.reshape(data, shape, order='F')

def ungroup_ind(data, index_to_ungroup, new_index_dim, new_index_location):
    """
    ungroups index_to_ungroup with dimension reduced_dim * new_index_dim into axes with dimensions reduced_dim and new_index_dim
    index with new_index_dim goes to new_index_location
    index with reduced_dim takes place of index_to_ungroup 
    """
    reduced_dim = data.shape[index_to_ungroup]//new_index_dim
    shape = list(data.shape[:index_to_ungroup]) + [reduced_dim, new_index_dim] + list(data.shape[index_to_ungroup+1:])
    ungrouped = np.reshape(data, shape, order='F')
    return np.moveaxis(ungrouped, index_to_ungroup+1, new_index_location)

def contract_bond(data1, data2): 
    """ 
    data1 and data2 have indices (l1,r1,t1,b1) and (l2,r2,t2,b2) respectively, 
    we contract r1 and l2 and return (l1,r2,t1,t2,b1,b2)
    """
    return np.tensordot(data1, data2, axes=([1],[0])).transpose(0,3,1,4,2,5)

class Node():  
    """ Tensor of order 4, shape format is (left_bond_dim, right_bond_dim, top_spin_dim, bottom_spin_dim) """
    def __init__(self, data): 
        self.data = data 
        self.shape = data.shape
    
    def svd(self, which='left', min_sv_ratio=None, max_dim=None):
        """
        does an asymmetric SVD, keeping both spin indices on 'which' side, truncates depending on either max_dim or min_sv_ratio
        """
        if which == 'left': 
            data = group_ind(group_ind(self.data, 0,2), 0,2)
        else: 
            data = group_ind(group_ind(self.data, 1,2), 1,2)
        
        u,s,vh = svd(data, full_matrices=False, lapack_driver='gesvd') # this safer version should remove svd convergence issues
                
        if min_sv_ratio is not None: 
            s = s[s>min_sv_ratio*s[0]]
        elif max_dim is not None:
            dim = min(max_dim, len(s[s>default_min_sv_ratio*s[0]]))
            s = s[:dim]
            
        u = u[:,:len(s)] 
        vh = vh[:len(s),:]
        
        if which == 'left': 
            left_tensor = Node(ungroup_ind(ungroup_ind(u, 0,2,2), 0,2,2))
            right_tensor = np.diag(s) @ vh    # in future versions, store and return the norm of s separately
        else: 
            left_tensor = u @ np.diag(s)   # in future versions, store and return the norm of s separately
            right_tensor = Node(ungroup_ind(ungroup_ind(vh, 1,2,2), 1,2,2))
        
        return left_tensor, right_tensor
    
    def conj(self): 
        """ swap spin indices """
        return Node(np.swapaxes(self.data.conj(), 2,3))  

def gate_to_nodes(gate): 
    """
    turns a two qubit gate matrix constructed from tensor product to two nodes connected by a bond
    the matrix has the form A_{ij}^{i'j'} where ij correspond to incoming wires and i'j' the outgoing wires of the two qubits 
    we have to first reshape this into A_{ii'}{jj'} before we can SVD
    we also delete any singular values below default_min_sv_ratio
    """
    gate = gate.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
    try:
        u, s, vh = svd(gate, lapack_driver='gesvd')  # lapack driver change here too
    except: 
        print(gate)
    s = s[s>default_min_sv_ratio*s[0]]     
    u = u[:,:len(s)]
    vh = vh[:len(s),:]
    return Node(u.T.reshape(len(s),2,2)[np.newaxis,:,:,:]), Node((np.diag(s) @ vh).reshape(len(s),2,2)[:,np.newaxis,:,:])

def nodes_to_gate(node1, node2): 
    """ 
    this takes two nodes and turns it into a two qubit gate with 4 physical indices 
    assuming gate has trivial bond dimension with other gates, 0 indexing is used to shave off the bond indices
    """
    return np.tensordot(node1.data, node2.data, axes=([1],[0]))[0,:,:,0,:,:].transpose(1,0,3,2)

def nodes_to_supernode(node1, node2): 
    """ this takes two nodes and turns it into a supernode with 4 physical indices and 2 bond indices """ 
    return np.tensordot(node1.data, node2.data, axes=([1],[0])).transpose(0,3,1,2,4,5)
    
######################
######## MPO #########
######################
    
class MPO: 
    def __init__(self, nodes): 
        self.nodes = nodes 
        self.num_nodes = len(nodes)
        self.skeleton = [node.shape for node in nodes]
        self.weights = [np.linalg.norm(np.ravel(node.data)) for node in nodes]
        self.max_dim = max(list(sum(self.skeleton, ())))
    
    def __matmul__(self, other):
        return MPO([Node(group_ind(group_ind(np.tensordot(self.nodes[i].data, other.nodes[i].data, 
                                axes=([3],[2])).transpose(0,3,1,4,2,5), 0,1), 1,2)) for i in range(self.num_nodes)])
    
    def compress(self, min_sv_ratio=None, max_dim=None): 
        nodes = copy.deepcopy(self.nodes)

        # first we orthogonalize
        for i in range(self.num_nodes-1, 0, -1):
            left_tensor, right_tensor = nodes[i].svd('right', None, None)
            nodes[i] = right_tensor
            nodes[i-1] = Node(np.tensordot(nodes[i-1].data, left_tensor, axes=([1],[0])).transpose(0,3,1,2))
        
        # then we sweep down and truncate
        for i in range(0, self.num_nodes-1):
            left_tensor, right_tensor = nodes[i].svd('left', min_sv_ratio, max_dim)
            nodes[i] = left_tensor
            nodes[i+1] = Node(np.tensordot(right_tensor, nodes[i+1].data, axes=([1],[0])))
        
        # then we sweep up and truncate
        for i in range(self.num_nodes-1, 0, -1):
            left_tensor, right_tensor = nodes[i].svd('right', min_sv_ratio, max_dim)
            nodes[i] = right_tensor
            nodes[i-1] = Node(np.tensordot(nodes[i-1].data, left_tensor, axes=([1],[0])).transpose(0,3,1,2))

        return MPO(nodes)
    
    def trace(self): 
        """ this returns Tr(MPO)/2^n, which matches the cost function and prevents blow up of numbers """
        trace = np.einsum('ijkk', self.nodes[0].data)/2
        for i in range(1, self.num_nodes): 
            trace = np.tensordot(trace, self.nodes[i].data, axes=([1],[0]))
            trace = np.einsum('ijkk', trace)/2
        return trace[0][0]
    
    def mult_and_trace(self, other, start='top'):
        """ this returns Tr(self @ other)/2^n """ 
        num_nodes = self.num_nodes
        if start == 'top':
            trace = np.tensordot(self.nodes[0].data, other.nodes[0].data, axes=([2,3],[3,2])).transpose(0,2,1,3)/2
            for i in range(1,num_nodes): 
                trace = np.tensordot(trace, self.nodes[i].data, axes=([2],[0])).transpose(0,1,3,2,4,5)
                trace = np.tensordot(trace, other.nodes[i].data, axes=([3,4,5],[0,3,2]))/2
            return trace
        else: 
            trace = np.tensordot(self.nodes[num_nodes-1].data, other.nodes[num_nodes-1].data, 
                                                     axes=([2,3],[3,2])).transpose(0,2,1,3)/2
            for i in range(num_nodes-2,-1,-1):
                trace = np.tensordot(trace, self.nodes[i].data, axes=([0],[1])).transpose(1,2,3,0,4,5)
                trace = np.tensordot(trace, other.nodes[i].data, axes=([3,4,5],[1,3,2])).transpose(2,3,0,1)/2
            return trace
    
    def conj(self): 
        nodes = [copy.deepcopy(node).conj() for node in self.nodes]
        return MPO(nodes)
    
    def to_matrix(self):
        """ for debugging purposes """
        matrix = self.nodes[0].data
        for i in range(1, self.num_nodes): 
            matrix = np.einsum('ijkl,jmno->imknlo', matrix, self.nodes[i].data)
            matrix = group_ind(group_ind(matrix, 4,5), 2,3) 
        return matrix[0][0] 
    
def random_mpo(num_qubits, bond_dim):
    return MPO([Node(np.random.rand(1,bond_dim,2,2) + 1.j*np.random.rand(1,bond_dim,2,2))] + [Node(
        np.random.rand(bond_dim,bond_dim,2,2) + 1.j*np.random.rand(bond_dim,bond_dim,2,2))for i in range(num_qubits-2)] + [Node(
        np.random.rand(bond_dim,1,2,2) + 1.j*np.random.rand(bond_dim,1,2,2))])

def id_mpo(num_qubits): 
    return MPO([Node(np.eye(2)[np.newaxis, np.newaxis,:,:])] * num_qubits)

def u1_proj_mpo(num_qubits, filling):
    Sm = np.array([[1, 0], [0, 0]])
    Sp = np.array([[0, 0], [0, 1]])
    sites = [None] * num_qubits

    for i in range(filling):
        sites[i] = np.zeros((i+1, i+2, 2, 2))
        for k in range(i+1):
            sites[i][k, k] = Sm
            sites[i][k, k+1] = Sp

    for i in range(filling, num_qubits-filling):
        sites[i] = np.zeros((filling+1, filling+1, 2, 2))
        for k in range(filling+1):
            sites[i][k, k] = Sm
        for k in range(filling):
            sites[i][k, k+1] = Sp

    for i in range(filling):
        sites[num_qubits-i-1] = np.zeros((i+2, i+1, 2, 2))
        for k in range(i+1):
            sites[num_qubits-i-1][k, k] = Sp
            sites[num_qubits-i-1][k+1, k] = Sm

    return MPO([Node(site) for site in sites])