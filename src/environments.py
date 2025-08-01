from ansatz import *

########################
#### Stack envs ########
########################

def stack_env(ansatz, target_mpo, stack_idx, min_sv_ratio=None, max_dim=None):
    """
    let T be target unitary and we have T^dWDW^d, 
    for stack_idx = 0, this function returns W^dT^dW, None
    for stack_idx > 0, this function returns A,B, where Tr(T^dWDW^d) = Tr(AMBM^d) and M,M^d are the MPOs at stack_idx
    contraction order and compression ensures that returned MPOs don't have large bond dim 
    example: 
    suppose we have Tr(T L4 L3 L2 L1 D L1^d L2^d L3^d L4^d)
    env1 of L2 is (L3^d L4^d T L3 L4)
    env2 of L2 is (L1 D L1^d)
    Tr(env1 L2 env2 L2^d)
    we call env1 the right environment and env2 the left environment (yes I understand that is weird and will fix it later) 
    """
    if stack_idx == 0:
        env = target_mpo.conj() 
        for i in range(ansatz.num_stacks-1,0,-1): 
            param_mpo = ansatz.param_mpo_stacks[i]
            env = (param_mpo.conj() @ env @ param_mpo).compress(min_sv_ratio, max_dim)
        return env, None
    
    else: 
        env1 = target_mpo.conj()  
        for i in range(ansatz.num_stacks-1,stack_idx,-1): 
            param_mpo = ansatz.param_mpo_stacks[i]
            env1 = (param_mpo.conj() @ env1 @ param_mpo).compress(min_sv_ratio, max_dim)

        env2 = ansatz.param_mpo_stacks[0]
        for i in range(1,stack_idx):
            param_mpo = ansatz.param_mpo_stacks[i]
            env2 = (param_mpo @ env2 @ param_mpo.conj()).compress(min_sv_ratio, max_dim)
        return env1, env2 
    
def build_right_env(cur_right_env, new_stack, min_sv_ratio=None, max_dim=None): 
    """
    Given the right environment at stack_idx + 1, this produces the right environment of stack_idx
    stack_idx can be 0,1,...,num_stacks-2
    example: 
    For stack_idx = 1, cur_right_env = L3^d T L3, new_stack = L2, 
    returns right environment of L1 = (L2^d L3^d T L3 L2)
    """
    return (new_stack.conj() @ cur_right_env @ new_stack).compress(min_sv_ratio, max_dim) 
    
def build_left_env(cur_left_env, new_stack, min_sv_ratio=None, max_dim=None): 
    """
    Given the left environment at stack_idx - 1, this produces the left environment of stack_idx
    stack_idx can be 1,2,...,num_stacks-1
    example: 
    For stack_idx = 3, cur_left_env = (L1 D L1^d), new_stack = L2, 
    returns left environment of L3 = (L2 L1 D L1^d L2^d) 
    """
    if cur_left_env is None: 
        return new_stack
    else: 
        return (new_stack @ cur_left_env @ new_stack.conj()).compress(min_sv_ratio, max_dim)
        
def all_stack_envs(ansatz, target_mpo, min_sv_ratio=None, max_dim=None): 
    all_envs = [stack_env(ansatz, target_mpo, i, min_sv_ratio, max_dim) for i in range(ansatz.num_stacks)]
    left_envs = [envs[1] for envs in all_envs]
    right_envs = [envs[0] for envs in all_envs]
    return left_envs, right_envs

def get_link_datas(stack_env1, stack_env2, num_qubits, stack_idx):
    if stack_env2 is None:
        link_datas1 = [stack_env1.nodes[i].data for i in range(num_qubits)]
        link_datas2 = [None] * num_qubits
    else:
        link_datas1 = [nodes_to_supernode(stack_env1.nodes[2*i+(stack_idx+1)%2].data, 
                                          stack_env1.nodes[2*i+1+(stack_idx+1)%2].data) 
                       for i in range(num_qubits // 2 - (stack_idx + 1) % 2)]
        link_datas2 = [nodes_to_supernode(stack_env2.nodes[2*i+(stack_idx+1)%2].data, 
                                          stack_env2.nodes[2*i+1+(stack_idx+1)%2].data) 
                       for i in range(num_qubits // 2 - (stack_idx + 1) % 2)]
    return link_datas1, link_datas2

######################
###### Node envs #####
######################

def build_bottom_env(cur_bottom_env, param_data, link_data1, link_data2=None):
    """ link_data are the nodes coming from stack_envs, param_data is the node coming from current stack """
    if link_data2 is None: 
        ''' rz stack '''
        if cur_bottom_env is not None: 
            a = np.tensordot(cur_bottom_env, link_data1, axes=([0],[1]))
            return np.tensordot(a, param_data, axes=([0,2,3],[1,3,2]))/2
        else: 
            return np.tensordot(link_data1, param_data, axes=([1,2,3],[1,3,2]))/2
    else: 
        ''' a general stack '''
        link_data1 = np.tensordot(link_data1, param_data.conj(), axes=([2,4],[1,3]))
        link_data2 = np.tensordot(link_data2, param_data, axes=([2,4],[0,2]))
        if cur_bottom_env is not None: 
            a = np.tensordot(link_data1, cur_bottom_env, axes=([1],[0]))
            return np.tensordot(a, link_data2, axes=([1,2,3,4,5],[4,5,2,3,1]))/4
        else: 
            return np.tensordot(link_data1, link_data2, axes=([2,3,4,5],[4,5,2,3]))[:,0,:,0]/4

def build_top_env(cur_top_env, param_data, link_data1, link_data2=None): 
    """ link_data are the nodes coming from stack_envs, param_data is the node coming from current stack """
    if link_data2 is None: 
        ''' rz stack '''
        if cur_top_env is not None: 
            a = np.tensordot(cur_top_env, link_data1, axes=([1],[0]))
            return np.tensordot(a, param_data, axes=([2,3],[3,2]))[:,:,0,0]/2
        else: 
            return np.tensordot(link_data1, param_data, axes=([0,2,3],[0,3,2])).T/2
    else: 
        ''' a general stack '''
        link_data1 = np.tensordot(link_data1, param_data.conj(), axes=([2,4],[1,3])) # added conj here
        link_data2 = np.tensordot(link_data2, param_data, axes=([2,4],[0,2])) # removed conj here
        
        if cur_top_env is not None: 
            a = np.tensordot(link_data1, cur_top_env, axes=([0],[0]))
            return np.tensordot(a, link_data2, axes=([1,2,3,4,5],[4,5,2,3,0]))/4
        else: 
            return np.tensordot(link_data1, link_data2, axes=([2,3,4,5],[4,5,2,3]))[0,:,0,:]/4
        
def node_semi_env(param_mpo, stack_idx, node_idx, stack_env1, stack_env2):
    """ Given stack_envs of param_mpo, this computes the top and bottom semi_envs of a particular node """ 
    n_q = param_mpo.num_nodes
    if stack_idx == 0: 
        ''' rz stack '''
        cur_bottom_env = None
        for i in range(n_q-1, node_idx, -1):
            cur_bottom_env = build_bottom_env(cur_bottom_env, param_mpo.nodes[i].data, stack_env1.nodes[i].data)
        cur_top_env = None 
        for i in range(node_idx): 
            cur_top_env = build_top_env(cur_top_env, param_mpo.nodes[i].data, stack_env1.nodes[i].data)
        return cur_top_env, cur_bottom_env
    
    else: 
        ''' general stack; here node_idx is really supernode_idx ''' 
        cur_bottom_env = None if stack_idx%2 else np.tensordot(stack_env1.nodes[n_q-1].data, stack_env2.nodes[n_q-1].data, axes=([1,2,3],[1,3,2]))/2
        for i in range(n_q//2 - 1, node_idx+(stack_idx+1)%2, -1):
            idx = 2*i-(stack_idx+1)%2
            param_data = nodes_to_gate(param_mpo.nodes[idx], param_mpo.nodes[idx+1])
            link_data1 = nodes_to_supernode(stack_env1.nodes[idx], stack_env1.nodes[idx+1])
            link_data2 = nodes_to_supernode(stack_env2.nodes[idx], stack_env2.nodes[idx+1])
            cur_bottom_env = build_bottom_env(cur_bottom_env, param_data, link_data1, link_data2)
            
        cur_top_env = None if stack_idx%2 else np.tensordot(stack_env1.nodes[0].data, stack_env2.nodes[0].data, axes=([0,2,3],[0,3,2]))/2
        for i in range(node_idx): 
            idx = 2*i+(stack_idx+1)%2
            param_data = nodes_to_gate(param_mpo.nodes[idx], param_mpo.nodes[idx+1])
            link_data1 = nodes_to_supernode(stack_env1.nodes[idx], stack_env1.nodes[idx+1])
            link_data2 = nodes_to_supernode(stack_env2.nodes[idx], stack_env2.nodes[idx+1])
            cur_top_env = build_top_env(cur_top_env, param_data, link_data1, link_data2)
        return cur_top_env, cur_bottom_env
    
def full_env(link_data1, link_data2=None, top_env=None, bottom_env=None): 
    """ link_data are the nodes of stack_envs at the current node_idx """ 
    if link_data2 is not None: 
        ''' we are in a general stack ''' 
        if top_env is not None and bottom_env is not None: 
            ''' we are in the bulk of the stack ''' 
            a = np.tensordot(top_env, link_data1, axes=([0],[0]))
            b = np.tensordot(a, link_data2, axes=([0],[0]))
            result = np.tensordot(b, bottom_env, axes=([0,5],[0,1]))/4  
        
        elif top_env is None: 
            ''' we are at the top boundary of the stack '''
            a = np.tensordot(bottom_env, link_data1, axes=([0],[1]))
            result = np.tensordot(a, link_data2, axes=([0],[1]))[0,:,:,:,:,0,:,:,:,:]/4
            
        else: 
            ''' we are at the bottom boundary of the stack '''
            a = np.tensordot(top_env, link_data1, axes=([0],[0])) 
            result = np.tensordot(a, link_data2, axes=([0],[0]))[0,:,:,:,:,0,:,:,:,:]/4 
        return result
    else: 
        ''' we are in the rz stack '''                           
        if top_env is not None and bottom_env is not None: 
            ''' we are in the bulk of the stack '''
            a = np.tensordot(top_env, link_data1, axes=([1],[0]))
            return np.tensordot(a, bottom_env, axes=([0,1],[1,0]))/2
        elif top_env is None: 
            ''' we are at the top boundary of the stack '''
            return np.tensordot(link_data1, bottom_env, axes=([0,1],[1,0]))/2
        else: 
            ''' we are at the bottom boundary of the stack '''
            return np.tensordot(top_env, link_data1, axes=([0,1],[1,0]))/2      
        
def node_env(param_mpo, stack_idx, node_idx, stack_env1, stack_env2=None):
    """ Given stack_envs of param_mpo, this computes the environment of a particular node inside param_mpo """ 
    top_env, bottom_env = node_semi_env(param_mpo, stack_idx, node_idx, stack_env1, stack_env2)
    if stack_env2 is None: 
        return full_env(stack_env1.nodes[node_idx].data, None, top_env, bottom_env)
    else: 
        idx = 2*node_idx+(stack_idx+1)%2
        link_data1 = nodes_to_supernode(stack_env1.nodes[idx], stack_env1.nodes[idx+1])
        link_data2 = nodes_to_supernode(stack_env2.nodes[idx], stack_env2.nodes[idx+1]) 
        return full_env(link_data1, link_data2, top_env, bottom_env)
    
def all_node_envs(param_mpo, stack_idx, stack_env1, stack_env2=None): 
    bottom_envs = [] 
    top_envs = []
    num_envs = param_mpo.num_nodes if stack_idx == 0 else param_mpo.num_nodes//2 - (stack_idx+1)%2
    for node_idx in range(num_envs): 
        top_env, bottom_env = node_semi_env(param_mpo, stack_idx, node_idx, stack_env1, stack_env2)
        bottom_envs.append(bottom_env)
        top_envs.append(top_env)
    return bottom_envs, top_envs

