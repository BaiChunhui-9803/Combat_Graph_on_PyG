from database_connector import DatabaseConnector
import torch
import json
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np
import os
import functional
from sklearn.cluster import SpectralClustering
import mark_action_label

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '5'

dbc = DatabaseConnector(
    host="localhost",
    user="root",
    password="19980311",
    database="pymarl"
)

action_dict = dbc.select_action_dict()
state_dict = dbc.select_state_dict()
im_state_dict = dbc.select_im_state_dict()
processed_action_dict = dbc.select_processed_action_dict()
processed_state_dict = dbc.select_processed_state_dict()

action_log = dbc.select_action_log()
state_log = dbc.select_state_log()
im_state_log = dbc.select_im_state_log()
result_log = dbc.select_result_log()
processed_action_log = dbc.select_processed_action_log()
processed_state_log = dbc.select_processed_state_log()

state_map = dbc.select_state_map()
processed_state_map = dbc.select_processed_state_map()

reversed_state_map = defaultdict(list)
for key, values in state_map.items():
    for value in values:
        reversed_state_map[value] = key

reversed_state_map = dict(reversed_state_map)
# =============================================================================
# 方法

def louvain(G):
    partition = community_louvain.best_partition(G)
    return partition

def spectral_clustering(G, num_clusters):
    adj_matrix = nx.adjacency_matrix(G).toarray()
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    labels = clustering.fit_predict(adj_matrix)
    return labels

# =============================================================================
# 原始数据 - graph
edge_weights = defaultdict(int)
for key in action_log.keys():
    actions = action_log[key][:-1]
    states = state_log[key]
    for i in range(len(actions)):
        src_state = states[i]
        dst_state = states[i + 1]
        edge = (src_state, dst_state)
        edge_weights[edge] += 1
edges = list(edge_weights.keys())
weights = list(edge_weights.values())
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(weights, dtype=torch.float)
num_nodes = max(max(state_log.values(), key=lambda x: max(x)))
x = torch.arange(1, num_nodes + 1, dtype=torch.float).view(-1, 1)
graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# =============================================================================
# 去除cooldown、去重的数据 - graph
# pro_edge_weights = {}
# for key in processed_action_log.keys():
#     states = processed_state_log[key]
#     actions = processed_action_log[key]
#     for i in range(len(states) - 1):
#         src_state = states[i]
#         dst_state = states[i + 1]
#         # origin_state = state_log[key][i]
#         edge = (src_state, dst_state)
#         action = actions[i]
#         if edge not in pro_edge_weights.keys():
#             pro_edge_weights[edge] = {
#                 'state-action': Counter([(src_state, action)]),
#                 'weight': 1
#             }
#         else:
#             pro_edge_weights[edge]['state-action'][(src_state, action)] += 1
#             pro_edge_weights[edge]['weight'] += 1
# pro_edges = list(pro_edge_weights.keys())
# pro_weights = [pro_edge_weights[edge]['weight'] for edge in pro_edges]
# pro_state_actions = [pro_edge_weights[edge]['state-action'] for edge in pro_edges]
#
# sorted_pro_edges_with_weights = sorted(zip(pro_edges, pro_weights))
# sorted_pro_edges, sorted_pro_weights = zip(*sorted_pro_edges_with_weights)
#
# pro_edge_index = torch.tensor(sorted_pro_edges, dtype=torch.long).t().contiguous()
# pro_edge_attr = torch.tensor(sorted_pro_edges, dtype=torch.float)
# pro_num_nodes = max(max(processed_state_log.values(), key=lambda x: max(x)))
# pro_x = torch.tensor([processed_state_dict[i]['hp_mean'] for i in range(1, pro_num_nodes + 1)], dtype=torch.float).view(-1, 1)
# processed_graph_data = Data(x=pro_x, edge_index=pro_edge_index, edge_attr=pro_edge_attr)
#
# G = to_networkx(processed_graph_data)
#
# functional.load_to_csv(pro_x, sorted_pro_edges, sorted_pro_weights, prefix='pro')

# -----------------------------------------------------------------------------
# 去除cooldown、去重的数据 - graph分割

# partition = partition_graph(G.to_undirected())
# functional.load_to_csv(pro_x, sorted_pro_edges, sorted_pro_weights, prefix='pro_partition', partition=partition)

# =============================================================================
# 势力图状态 - graph
im_edge_weights = {}
for key in im_state_log.keys():
    states = im_state_log[key]
    actions = action_log[key]
    for i in range(len(states) - 1):
        src_state = states[i] - 1
        dst_state = states[i + 1] - 1
        origin_state = state_log[key][i]
        edge = (src_state, dst_state)
        action = actions[i]
        if edge not in im_edge_weights.keys():
            im_edge_weights[edge] = {
                'state-action': Counter([(origin_state, action)]),
                'weight': 1
            }
        else:
            im_edge_weights[edge]['state-action'][(origin_state, action)] += 1
            im_edge_weights[edge]['weight'] += 1
# for edge, data in im_edge_weights.items():
    # print(f"Edge: {edge}, State-Action: {data['state-action']}, Weight: {data['weight']}")
im_edges = list(im_edge_weights.keys())
im_weights = [im_edge_weights[edge]['weight'] for edge in im_edges]
im_state_actions = [im_edge_weights[edge]['state-action'] for edge in im_edges]

mark_action_label.analyze_action(im_state_actions, action_dict, state_dict)

mark_action_label.mark_action_label(im_state_actions)

sorted_edges_with_weights = sorted(zip(im_edges, im_weights))
sorted_im_edges, sorted_im_weights = zip(*sorted_edges_with_weights)
im_edge_index = torch.tensor(sorted_im_edges, dtype=torch.long).t().contiguous()
im_edge_attr = torch.tensor(sorted_im_weights, dtype=torch.float)
im_num_nodes = max(max(im_state_log.values(), key=lambda x: max(x)))
im_x = torch.tensor([im_state_dict[i+1]['hp_mean'] for i in range(im_num_nodes)], dtype=torch.float).view(-1, 1)
im_graph_data = Data(x=im_x, edge_index=im_edge_index, edge_attr=im_edge_attr)
#
# G = to_networkx(im_graph_data)
#
# functional.load_to_csv(im_x, sorted_im_edges, sorted_im_weights, prefix='im')

# -----------------------------------------------------------------------------
# partitions = {}
# louvain_partition = louvain(G.to_undirected())
# spectral_clustering_partition = spectral_clustering(G.to_undirected(), 36)
#
# partitions['Partition_Louvain'] = louvain_partition
# partitions['Partition_SpectralClustering'] = spectral_clustering_partition
# functional.load_to_csv(im_x, sorted_im_edges, sorted_im_weights, prefix='im_partition', partitions=partitions)

# =============================================================================
# 势力图状态 - graph - 剪切保留正值状态
im_p_node = {}
im_p_edge_weights = {}
im_p_state_log = {}
for key in im_state_log.keys():
    temp = max([im_state_dict[s]['hp_mean'] for s in im_state_log[key]])
    if temp > 0:
        # 把全部的状态都保留
        states = im_state_log[key]
        actions = action_log[key]
        im_p_state_log[key] = [s_id - 1 for s_id in states]
        for i in range(len(states) - 1):
            src_state = states[i] - 1
            dst_state = states[i + 1] - 1
            origin_state = state_log[key][i]
            edge = (src_state, dst_state)
            action = actions[i]
            if src_state not in im_p_node.keys():
                im_p_node[src_state] = im_state_dict[states[i+1]]
            if dst_state not in im_p_node.keys():
                im_p_node[dst_state] = im_state_dict[states[i+1]]
            if edge not in im_p_edge_weights.keys():
                im_p_edge_weights[edge] = {
                    'state-action': Counter([(origin_state, action)]),
                    'weight': 1
                }
            else:
                im_p_edge_weights[edge]['state-action'][(origin_state, action)] += 1
                im_p_edge_weights[edge]['weight'] += 1
# for edge, data in im_edge_weights.items():
    # print(f"Edge: {edge}, State-Action: {data['state-action']}, Weight: {data['weight']}")
im_p_edges = list(im_p_edge_weights.keys())
im_p_weights = [im_p_edge_weights[edge]['weight'] for edge in im_p_edges]
im_p_state_actions = [im_p_edge_weights[edge]['state-action'] for edge in im_p_edges]

node_mapping = {node_id: idx for idx, node_id in enumerate(im_p_node.keys())}

mapped_im_p_node = {mapped_node_id: im_p_node[node_id] for node_id, mapped_node_id in node_mapping.items()}
mapped_im_p_edge_weights = {}
mapped_im_p_node_log = {key: [node_mapping[node_id] for node_id in value] for key, value in im_p_state_log.items()}

for edge, attr_dict in im_p_edge_weights.items():
    src_node, dst_node = edge
    src_node_mapped = node_mapping[src_node]
    dst_node_mapped = node_mapping[dst_node]
    mapped_edge = (src_node_mapped, dst_node_mapped)
    mapped_im_p_edge_weights[mapped_edge] = attr_dict


mapped_im_edges = list(mapped_im_p_edge_weights.keys())
mapped_im_weights = [mapped_im_p_edge_weights[edge]['weight'] for edge in mapped_im_edges]
mapped_im_actions = [mapped_im_p_edge_weights[edge]['state-action'] for edge in mapped_im_edges]

sorted_p_edges_with_weights = sorted(zip(mapped_im_edges, mapped_im_weights))
sorted_im_p_edges, sorted_im_p_weights = zip(*sorted_p_edges_with_weights)
im_p_edge_index = torch.tensor(sorted_im_p_edges, dtype=torch.long).t().contiguous()
im_p_edge_attr = torch.tensor(sorted_im_p_weights, dtype=torch.float)

# im_p_num_nodes = len(im_p_node)
im_p_x = torch.tensor([mapped_im_p_node[key]['hp_mean'] for key, value in mapped_im_p_node.items()], dtype=torch.float).view(-1, 1)
im_p_graph_data = Data(x=im_p_x, edge_index=im_p_edge_index, edge_attr=im_p_edge_attr)

im_p_data = {
    'node': mapped_im_p_node,
    'edge_index': im_p_edge_index,
    'edge_attr': im_p_edge_attr
}

G = to_networkx(im_p_graph_data)

partitions = {}
louvain_partition = louvain(G.to_undirected())
spectral_clustering_partition = spectral_clustering(G.to_undirected(), 6)

partitions['Partition_Louvain'] = louvain_partition
partitions['Partition_SpectralClustering'] = spectral_clustering_partition
functional.load_to_csv(im_p_x, sorted_im_p_edges, sorted_im_p_weights, prefix='im_p_partition', partitions=partitions)

im_p_log = {id: state_log[id] for id in [log for log in im_p_state_log.keys()]}
im_p_map = defaultdict(list)
for key in im_p_log.keys():
    if key in mapped_im_p_node_log.keys():
        for a, b in zip(im_p_log[key], mapped_im_p_node_log[key]):
            im_p_map[b].append(a)
im_p_map = dict(im_p_map)
im_p_attr_map = {key: {s: {
    'allies_xy': functional.ally_to_xy(state_dict[s]['allies']),
    'enemies_xy': functional.enemy_to_xy(state_dict[s]['enemies']),
    'hp_mean': state_dict[s]['hp_mean'],
    'hp_variance': state_dict[s]['hp_variance']} for s in value} for key, value in im_p_map.items()}















im_map = {key: state_map[value['hash_code']] for key, value in mapped_im_p_node.items()}
















im_map_attr_map = {key: {s: {
    'allies_xy': functional.ally_to_xy(state_dict[s]['allies']),
    'enemies_xy': functional.enemy_to_xy(state_dict[s]['enemies']),
    'hp_mean': state_dict[s]['hp_mean'],
    'hp_variance': state_dict[s]['hp_variance']} for s in value} for key, value
    in im_map.items()}


im_p_state_matrix = defaultdict(dict)
state_matrix_dict = {}
for im_p_state, im_p_state_map in im_map_attr_map.items():
    for state, state_dict in im_p_state_map.items():
        adm, edm = functional.get_state_distance_matrix(state_dict)
        state_dm_hash = functional.get_state_dict_hash({
            'ally_dm': tuple(adm.flatten()),
            'enemy_dm': tuple(edm.flatten())
        })
        if state_dm_hash not in state_matrix_dict:
            state_matrix_dict[state_dm_hash] = {
                'ally_dm': adm,
                'enemy_dm': edm
            }
        im_p_state_matrix[im_p_state][state] = state_dm_hash
        # print(adm, edm)
        # functional.plot_state(im_p_state, state, state_dict)

state_matrix_distance_dict = []

# 分类矩阵，并去重
ally_dm_by_shape = defaultdict(list)
ally_dm_hashes = defaultdict(set)  # 用于存储哈希值，去重

enemy_dm_by_shape = defaultdict(list)
enemy_dm_hashes = defaultdict(set)  # 用于存储哈希值，去重

for smd in state_matrix_dict.values():
    ally_shape = smd['ally_dm'].shape
    enemy_shape = smd['enemy_dm'].shape

    # 处理 ally_dm
    ally_hash = functional.get_state_dict_hash(smd['ally_dm'])
    if ally_hash not in ally_dm_hashes[ally_shape]:
        ally_dm_by_shape[ally_shape].append(smd['ally_dm'])
        ally_dm_hashes[ally_shape].add(ally_hash)

    # 处理 enemy_dm
    enemy_hash = functional.get_state_dict_hash(smd['enemy_dm'])
    if enemy_hash not in enemy_dm_hashes[enemy_shape]:
        enemy_dm_by_shape[enemy_shape].append(smd['enemy_dm'])
        enemy_dm_hashes[enemy_shape].add(enemy_hash)


# 比较矩阵
def compare_matrices(matrices):
    """比较矩阵列表中的所有矩阵对"""
    results = []
    for i, matrix1 in enumerate(matrices):
        for j, matrix2 in enumerate(matrices):
            if i < j:  # 避免重复比较
                distance = functional.compare_frobenius(matrix1, matrix2)
                results.append(distance)
    return results


# 比较 ally_dm 矩阵
for shape, matrices in ally_dm_by_shape.items():
    distances = compare_matrices(matrices)
    for adm_dis in distances:
        state_matrix_distance_dict.append((adm_dis, None))  # None 表示 enemy_dm 的比较结果

# 比较 enemy_dm 矩阵80
for shape, matrices in enemy_dm_by_shape.items():
    distances = compare_matrices(matrices)
    for edm_dis in distances:
        state_matrix_distance_dict.append((None, edm_dis))  # None 表示 ally_dm 的比较结果

# for im_p_state, im_p_state_map in im_p_sttr_map.items():
#     if im_p_state > 7:
#         for state, state_dict in im_p_state_map.items():
#             functional.plot_state(im_p_state, state, state_dict)
#             plt.close('all')
# for im_p_state, im_p_state_map in im_map_attr_map.items():
#     for state, state_dict in im_p_state_map.items():
#         functional.plot_state(im_p_state, state, state_dict)
#         plt.close('all')

save_data_set_path = "../dataset/"
functional.g_to_dataset(data=im_p_graph_data, save_path=save_data_set_path, pro_name="im_p_1")



pass


[[ 1.          0.         -0.08916558  0.02719971], [ 1.          0.         -0.06898103  0.02689941],
 [ 1.          0.         -0.06851761  0.00199951], [ 1.          0.         -0.01286938  0.02689941]]

[[ 1.         -0.06944444  0.05570068], [ 1.         -0.0481477   0.02629883],
 [ 1.          0.00564914  0.0275    ], [ 1.         -0.04925989  0.05530029]]


{
    'a': set{1, 2, 3},
    'b': set{4, 5, 6},
    'c': set{7, 1},
    'd': set{2, 8},
    'e': set{2},
    'f': set{3, 9},
}

{
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    6: 1,
    7: 0,
    8: 1,
    9: 0,
}











