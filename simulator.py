import dgl
from collections import OrderedDict
from random import randint
from random import sample

def get_fan(cur_layer):
    if (cur_layer == 1):
        return 10
    return 25

def simulate(khop, roots, graph, v2idx):
    nodes_set = [roots]
    total_edge_pair = 0 # total number of edges in the sample graph (including all layers)
    total_cross_edge = 0 # number of edges cross the partition
    total_memcpy = 0 # number of aggregated data transfer

    for cur_layer in range(1, khop+1):
        sample_nodes = nodes_set[cur_layer - 1]
        fan = get_fan(cur_layer)
        graph_sample = dgl.sampling.sample_neighbors(graph, sample_nodes, fanout=fan, edge_dir="out")
        v_set, e_set = graph_sample.edges()
        _next = list(OrderedDict.fromkeys(e_set)) # remove duplicates
        nodes_set.append(_next)
        size_v = len(v_set)
        prev_parent_id = -1
        first_leaf_partition = -1
        rand = 0
        p_set = set()
        for i in range(0, size_v):
            cur_parent_id = int(v_set[i])
            cur_leaf_id = int(e_set[i])
            
            parent_partition_id = int(v2idx[cur_parent_id])
            leaf_partition_id = int(v2idx[cur_leaf_id])
            
            # memcpy
            if prev_parent_id != cur_parent_id:
                total_memcpy += len(p_set)
                p_set.clear()
                rand = randint(0,3)
                first_leaf_partition = leaf_partition_id
                prev_parent_id = cur_parent_id
                
            # cross edge
            if parent_partition_id != leaf_partition_id:
                total_cross_edge += 1   
                
#             # memcpy startegy 1: transfer aggregated data to parents
#             if parent_partition_id != leaf_partition_id:
#                 p_set.add(leaf_partition_id)
                
#             # memcpy strategy 2: transfer aggregated data to a random gpu
#             if leaf_partition_id != rand:
#                 p_set.add(leaf_partition_id)

            # memcpy strategy 3: transfer aggregated data to the first leaf gpu
            if leaf_partition_id != first_leaf_partition:
                p_set.add(leaf_partition_id)
                
        total_memcpy += len(p_set)        
        total_edge_pair += size_v
    return total_edge_pair, total_cross_edge, total_memcpy



dataset = dgl.data.CoraGraphDataset()
graph = dataset[0] 
graph = dgl.remove_self_loop(graph)
graph = dgl.add_self_loop(graph)
print(f"e_num={graph.num_edges()}; v_num={graph.num_nodes()}")
partition_num = 4
v2idx = dgl.metis_partition_assignment(graph, partition_num)

v_num = graph.num_nodes()
v_list = [i for i in range(0, v_num)]

total_e_num = 0
total_c_num = 0
total_k_num = 0
khop = 4

for i in range(1, 10):
    roots = sample(v_list, 15)
    print(f"sample {i} batch")
    e_num, c_num, k_num = simulate(khop=khop, roots=roots, graph=graph, v2idx=v2idx)
    total_e_num += e_num
    total_c_num += c_num
    total_k_num += k_num

print("e_num=total edge pair, c_num=cross edge number, k_number = memory copy number")
print(f"e_num={total_e_num}; c_num={total_c_num}; k_num={total_k_num}")
if total_k_num > 0:
    print(f"cache miss:  {total_c_num * 100.0 / total_e_num }%")
    print(f"memory copy: {total_k_num * 100.0 / total_e_num }%")