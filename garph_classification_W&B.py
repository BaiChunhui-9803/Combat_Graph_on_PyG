import os
import torch
torch_version = torch.__version__.split("+")
os.environ["TORCH"] = torch_version[0]
os.environ["CUDA"] = torch_version[1]

# General imports
import os
import json
import collections

# Data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import scipy.sparse as sp

# Import Weights & Biases for Experiment Tracking
import wandb

# Graph imports
import torch
from torch import Tensor
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx

import networkx as nx
from networkx.algorithms import community

from tqdm.auto import trange

from visualize import GraphVisualization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 设置W&B
use_wandb = True #@param {type:"boolean"}
wandb_project = "intro_to_pyg" #@param {type:"string"}
wandb_run_name = "upload_and_analyze_dataset" #@param {type:"string"}

if use_wandb:
    wandb.init(project=wandb_project, name=wandb_run_name)

# =============================================================================
# 加载数据集
from torch_geometric.datasets import TUDataset

dataset_path = "data/TUDataset"
dataset = TUDataset(root=dataset_path, name='MUTAG')

dataset.download()

data_details = {
    "num_data_set": len(dataset),
    "num_node_features": dataset.num_node_features,
    "num_edge_features": dataset.num_edge_features,
    "num_classes": dataset.num_classes,
    "num_node_labels": dataset.num_node_labels,
    "num_edge_labels": dataset.num_edge_labels
}

if use_wandb:
    # Log all the details about the data to W&B.
    wandb.log(data_details)
else:
    print(json.dumps(data_details, sort_keys=True, indent=4))

def create_graph(graph):
    g = to_networkx(graph)
    pos = nx.spring_layout(g)
    vis = GraphVisualization(
        g, pos, node_text_position='top left', node_size=20,
    )
    fig = vis.create_figure()
    return fig

# fig = create_graph(dataset[0])
# fig.show()

if use_wandb:
    # Log exploratory visualizations for each data point to W&B
    table = wandb.Table(columns=["Graph", "Number of Nodes", "Number of Edges", "Label"])
    for graph in dataset:
        print(graph)
        fig = create_graph(graph)
        n_nodes = graph.num_nodes
        n_edges = graph.num_edges
        label = graph.y.item()

        table.add_data(wandb.Html(plotly.io.to_html(fig)), n_nodes, n_edges, label)
    wandb.log({"data": table})

if use_wandb:
    # Log the dataset to W&B as an artifact.
    dataset_artifact = wandb.Artifact(name="MUTAG", type="dataset", metadata=data_details)
    dataset_artifact.add_dir(dataset_path)
    wandb.log_artifact(dataset_artifact)

    # End the W&B run
    wandb.finish()

# =============================================================================
# 加载数据集
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

# =============================================================================
# 图批处理
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

# =============================================================================
# 构建GCN
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
print(model)

# =============================================================================
# 训练模型
wandb_project = "intro_to_pyg" #@param {type:"string"}
wandb_run_name = "upload_and_analyze_dataset" #@param {type:"string"}

# Initialize W&B run for training
if use_wandb:
    wandb.init(project="intro_to_pyg")
    wandb.use_artifact('bch980311-/intro_to_pyg/MUTAG:v0', type='dataset')

from IPython.core.display_functions import display
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.


def test(loader, create_table=False):
    model.eval()
    table = wandb.Table(columns=['graph', 'ground truth', 'prediction']) if use_wandb else None
    correct = 0
    loss_ = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss_ += loss.item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        if create_table and use_wandb:
            table.add_data(wandb.Html(plotly.io.to_html(create_graph(data))), data.y.item(), pred.item())

        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset), loss_ / len(loader.dataset), table  # Derive ratio of correct predictions.

for epoch in trange(1, 171):
    train()
    train_acc, train_loss, _ = test(train_loader)
    test_acc, test_loss, test_table = test(test_loader, create_table=True)

    # Log metrics to W&B
    if use_wandb:
        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/acc": test_acc,
            "test/loss": test_loss,
            "test/table": test_table
        })

    torch.save(model, "graph_classification_model.pt")

    # Log model checkpoint as an artifact to W&B
    if use_wandb:
        artifact = wandb.Artifact(name="graph_classification_model", type="model")
        artifact.add_file("graph_classification_model.pt")
        wandb.log_artifact(artifact)

# Finish the W&B run
if use_wandb:
    wandb.finish()