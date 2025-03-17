from database_connector import DatabaseConnector
import torch
import json
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
import functional
import mark_action_label
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from MyGNNs import MyGCN, MyGAT, MyGraphSAGE
import train

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
        reversed_state_map[value].append(key)

reversed_state_map = dict(reversed_state_map)

# =============================================================================
# 原始数据 - graph
edge_weights = defaultdict(int)
for key in action_log.keys():
    actions = action_log[key][:-1]
    states = state_log[key]
    for i in range(len(actions)):
        src_state = states[i] - 1
        dst_state = states[i + 1] - 1
        edge = (src_state, dst_state)
        edge_weights[edge] += 1
edges = list(edge_weights.keys())
weights = list(edge_weights.values())
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
nodes = list(set([item for sublist in edges for item in sublist]))

im2merged_im_map, merged_im2im, micro_graph_labels, clustered_micro_graph = functional.get_merged_map(state_dict, reversed_state_map, state_map)

num_label = len(clustered_micro_graph)


state_graphs = functional.get_graphs(
    nodes=nodes,
    state_dict=state_dict,
    state_map=reversed_state_map,
    node_labels=micro_graph_labels
)

train_loader = DataLoader(state_graphs, batch_size=1024, shuffle=True)

train.execute_sweep(state_graphs, num_label)


pass

# model = MyGraphSAGE(in_channels=2, hidden_channels=128, out_channels=num_label)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# def train():
#     model.train()
#     total_loss = 0
#     for data in train_loader:
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch)  # 传递 batch 参数
#         loss = criterion(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs
#     return total_loss / len(train_loader.dataset)
#
# num_epochs = 50
# for epoch in range(num_epochs):
#     loss = train()
#     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

# edge_attr = torch.tensor(weights, dtype=torch.float)
# num_nodes = max(max(state_log.values(), key=lambda x: max(x)))
# x = torch.arange(1, num_nodes + 1, dtype=torch.float).view(-1, 1)
# graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# todo 1. label的独特码表示
# todo 2. 执行train_loader时候，获取相匹配的原始数据
# todo 3. 增加准确度，增加其他指标
# todo 4. 引入wandb进行参数搜索
# todo 5. 图对比学习？


# todo 6. 类分布不均衡的问题，只看准确率可能不行































