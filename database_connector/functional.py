import pandas as pd
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt
import os

import community as community_louvain
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import json
import wandb
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import hashlib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def louvain(G):
    partition = community_louvain.best_partition(G)
    return partition

def spectral_clustering(G, num_clusters):
    adj_matrix = nx.adjacency_matrix(G).toarray()
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    labels = clustering.fit_predict(adj_matrix)
    return labels


def set_label(value):
    if value == 0:
        return 'neutral'
    elif value > 0:
        if value < 18:
            return 'advantage_2'
        else:
            return 'advantage_1'
    else:  # value < 0
        if value > -30:
            return 'disadvantage_4'
        elif value > -60:
            return 'disadvantage_3'
        elif value > -100:
            return 'disadvantage_2'
        else:
            return 'disadvantage_1'

def load_to_csv(x, edges, weights, prefix, partitions = None):
    nodes_df = pd.DataFrame({
        'Id': range(len(x)),
        'Value': x.numpy().flatten(),
        'Category': [set_label(value) for value in x.numpy().flatten()]
    })
    if partitions is not None:
        for key, value in partitions.items():
            nodes_df[key] = value
    nodes_df.to_csv(f'graph_data/{prefix}_nodes.csv', index=False)

    # 生成边文件
    edges_df = pd.DataFrame({
        'Source': [edge[0] for edge in edges],
        'Target': [edge[1] for edge in edges],
        'Type': ['Directed'] * len(edges),
        'Weight': weights
    })
    edges_df.to_csv(f'graph_data/{prefix}_edges.csv', index=False)

    pass

def units_to_xy(s_tensor):
    coordinates = []
    for element in s_tensor:
        if element[0] > 0:
            x = 64 + element[-2] * 108
            y = 64 + element[-1] * 100
            coordinates.append((x, y))
        else:
            coordinates.append(0)
    return Counter(coordinates)

# def enemy_to_xy(s_tensor):
#     coordinates = []
#     for element in s_tensor:
#         if element[0] > 0:
#             x = 64 + element[1] * 108
#             y = 64 + element[2] * 100
#             coordinates.append((x, y))
#         else:
#             coordinates.append(0)
#     return Counter(coordinates)


def plot_state(im_p_state, state, state_dict):
    allies_coords = [key for key, value in state_dict['allies_xy'].items() if key != 0]
    enemies_coords = [key for key, value in state_dict['enemies_xy'].items() if key != 0]

    plt.figure(figsize=(8, 6))
    # 分离 x 和 y 坐标
    if len(allies_coords) > 0:
        allies_x, allies_y = zip(*allies_coords)
        plt.scatter(allies_x, allies_y, color='red', label='Allies', zorder=5)

    if len(enemies_coords) > 0:
        enemies_x, enemies_y = zip(*enemies_coords)
        plt.scatter(enemies_x, enemies_y, color='blue', label='Enemies', zorder=5)

    plt.legend()

    plt.title("Allies and Enemies Coordinates")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # plt.show()
    hp_mean = state_dict['hp_mean']
    plt.savefig(f'state_plot/{im_p_state}-{state}-{hp_mean}.png')

def extract_points(counter):
    """从 Counter 对象中提取坐标点"""
    points = [point for point in counter.keys() if point != 0]
    return np.array(points)

def compute_distance_matrix(points):
    if len(points) == 0:
        return np.array([])
    distance_matrix = cdist(points, points)
    return distance_matrix

def get_state_dict_hash(state_dm_dict):
    return hashlib.md5(str(state_dm_dict).encode('utf-8')).hexdigest()
    # return hash(frozenset(state_dm_dict.items()))

def get_state_distance_matrix(origin_s):
    allies_points = extract_points(origin_s['allies_xy'])
    enemies_points = extract_points(origin_s['enemies_xy'])
    allies_distance_matrix = compute_distance_matrix(allies_points)
    enemies_distance_matrix = compute_distance_matrix(enemies_points)
    return allies_distance_matrix, enemies_distance_matrix

def compare_frobenius(matrix1, matrix2):
    difference = np.linalg.norm(matrix1 - matrix2, 'fro')
    return difference


def g_to_dataset(data, save_path, pro_name="default"):
    # use_wandb = True
    # wandb_project = "build_dataset" #@param {type:"string"}
    # wandb_run_name = f"{pro_name}_dataset_builder"  # @param {type:"string"}
    torch.save(data, f'{save_path}{pro_name}_graph_data.pt')

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

# def merge_sets_to_map(input_dict):
#     uf = UnionFind()
#     for value_set in input_dict.values():
#         elements = list(value_set)
#         for i in range(len(elements)):
#             for j in range(i + 1, len(elements)):
#                 uf.union(elements[i], elements[j])
#     result_map = {}
#     set_id = 0
#     id_map = {}
#
#     for element in uf.parent.keys():
#         root = uf.find(element)
#         if root not in id_map:
#             id_map[root] = set_id
#             set_id += 1
#         result_map[element] = id_map[root]
#
#     return result_map

def merge_sets_to_map(input_dict):
    from collections import defaultdict

    # 构建图的邻接表
    graph = defaultdict(set)
    for elements in input_dict.values():
        for element in elements:
            graph[element].update(elements)

    # 初始化访问标记和结果映射
    visited = set()
    result = {}
    group_id = 0

    # 深度优先搜索函数
    def dfs(node, group_id):
        visited.add(node)
        result[node] = group_id
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group_id)

    # 遍历所有节点，找到所有连通分量
    for node in graph:
        if node not in visited:
            dfs(node, group_id)
            group_id += 1

    return result

# def merge_sets_to_map(merged_map):
#     result_map = {}
#     set_id = 0
#
#     for value_set in merged_map.values():
#         for element in value_set:
#             if element not in result_map:
#                 result_map[element] = set_id
#         set_id += 1
#
#     return result_map

def get_merged_map(state_dict, state_map, refer_state_map):
    merged_map = {}
    merged_map_2 = {}
    for state_id, state in state_dict.items():
        combined_coordinates = (tuple(units_to_xy(state['allies'])), tuple(units_to_xy(state['enemies'])))
        for im_state in state_map[state_id]:
            if im_state not in merged_map_2:
                merged_map_2[im_state] = set()
            if combined_coordinates not in merged_map:
                merged_map[combined_coordinates] = set()
            # merged_map[im_state].add(combined_coordinates)
            merged_map[combined_coordinates].add(im_state)
            merged_map_2[im_state].add(state_id)

    merged_union_map = merge_sets_to_map(merged_map)

    reversed_merged_union_map = {}
    for element, set_id in merged_union_map.items():
        if set_id not in reversed_merged_union_map:
            reversed_merged_union_map[set_id] = set()
        reversed_merged_union_map[set_id].add(element)

    merged_map_3 = {}
    merged_map_4 = {}
    for state_id, im_state_list in state_map.items():
        for im_state in im_state_list:
            merged_map_3[state_id - 1] = merged_union_map[im_state]
            if merged_union_map[im_state] not in merged_map_4:
                merged_map_4[merged_union_map[im_state]] = set()
            merged_map_4[merged_union_map[im_state]].add(state_id - 1)

    sorted_keys = sorted(merged_map_3.keys())
    micro_graph_labels = [merged_map_3[key] for key in sorted_keys]

    return merged_union_map, reversed_merged_union_map, micro_graph_labels, merged_map_4


def build_bipartite_graph(state_dict, graph_label):
    allies = state_dict['allies'][state_dict['allies'][:, 0] != 0]
    enemies = state_dict['enemies'][state_dict['enemies'][:, 0] != 0]

    allies_xy = units_to_xy(allies)
    enemies_xy = units_to_xy(enemies)

    allies_xy_norm = allies[:, -2:]
    enemies_xy_norm = enemies[:, -2:]

    allies_xy_norm = torch.from_numpy(allies_xy_norm).float()
    enemies_xy_norm = torch.from_numpy(enemies_xy_norm).float()
    x = torch.cat([allies_xy_norm, enemies_xy_norm], dim=0).float()

    num_allies = allies.shape[0]
    num_enemies = enemies.shape[0]
    edge_index = torch.tensor([
        [i for i in range(num_allies) for _ in range(num_enemies)],
        [j + num_allies for _ in range(num_allies) for j in range(num_enemies)]
    ], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=graph_label)
    return data

def get_graphs(nodes, state_dict, state_map, node_labels):
    graph_data_list = []
    for node in nodes:
        state = state_dict[node + 1]
        im_state = state_map[node + 1]
        label = node_labels[node]
        graph_data = build_bipartite_graph(state, torch.tensor([label]))
        graph_data_list.append(graph_data)
    return graph_data_list


