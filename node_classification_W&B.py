enable_wandb = True
if enable_wandb:
    import wandb
    wandb.login(key='7f6d598bc0f0aca4dc78e698dabfcdb76e832a19')


import os
import pdb
import torch
import pandas

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def visualize_to_wandb(h, color, method):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    title = method + "/embedding_tSNE/trained"
    wandb.log({
        title: wandb.Plotly(plt.gcf())
    })
    # plt.show()

def embedding_to_wandb(h, color, key="embedding"):
    num_components = h.shape[-1]
    df = pandas.DataFrame(data=h.detach().cpu().numpy(),
                        columns=[f"c_{i}" for i in range(num_components)])
    df["target"] = color.detach().cpu().numpy().astype("str")
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]
    wandb.log({key: df})

# =============================================================================
# 加载数据集
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# =============================================================================
# 构建一个MLP
import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)

# 记录数据属性
if enable_wandb:
    wandb.init(project="node-classification")
    summary = dict()
    summary["data"] = dict()
    summary["data"]["num_features"] = dataset.num_features
    summary["data"]["num_classes"] = dataset.num_classes
    summary["data"]["num_nodes"] = data.num_nodes
    summary["data"]["num_edges"] = data.num_edges
    summary["data"]["has_isolated_nodes"] = data.has_isolated_nodes()
    summary["data"]["has_self_nodes"] = data.has_self_loops()
    summary["data"]["is_undirected"] = data.is_undirected()
    summary["data"]["num_training_nodes"] = data.train_mask.sum()
    wandb.summary = summary

# =============================================================================
# 训练一个MLP
from IPython.core.display_functions import display
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = MLP(hidden_channels=16)

with torch.no_grad():
  out = model(data.x)

if enable_wandb:
    embedding_to_wandb(out, color=data.y, key="mlp/embedding/init")
else:
    visualize(out, data.y)

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


for epoch in range(1, 201):
    loss = train()  # 假设train()函数返回当前epoch的损失
    test_acc = test()
    if enable_wandb:
        wandb.log({"mlp/loss": loss, "mlp/accuracy": test_acc})
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')

out = model(data.x)
if enable_wandb:
    embedding_to_wandb(out, color=data.y, key="mlp/embedding/trained")
    wandb.summary["mlp/accuracy"] = test_acc
    visualize_to_wandb(out, data.y, 'mlp')
else:
  visualize(out, data.y)

print(f'Test Accuracy: {test_acc:.4f}')
# =============================================================================
# 构建GCN
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)

model = GCN(hidden_channels=16)
model.eval()

out = model(data.x, data.edge_index)

if enable_wandb:
    embedding_to_wandb(out, color=data.y, key="gcn/embedding/init")
else:
    visualize(out, data.y)

# =============================================================================
# 训练GCN
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=16)
if enable_wandb:
    wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


for epoch in range(1, 101):
    loss = train()
    test_acc = test()
    if enable_wandb:
        wandb.log({"gcn/loss": loss, "gcn/accuracy": test_acc})
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

model.eval()

out = model(data.x, data.edge_index)

test_acc = test()
if enable_wandb:
    wandb.summary["gcn/accuracy"] = test_acc
    embedding_to_wandb(out, color=data.y, key="gcn/embedding/trained")
    visualize_to_wandb(out, data.y, 'gcn')
    wandb.finish()
else:
    visualize(out, data.y)

# =============================================================================
# 超参数搜索
import tqdm

def agent_fn():
    wandb.init()
    model = GCN(hidden_channels=wandb.config.hidden_channels)
    wandb.watch(model)

    with torch.no_grad():
      out = model(data.x, data.edge_index)
      embedding_to_wandb(out, color=data.y, key="gcn/embedding/init")

    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
          model.train()
          optimizer.zero_grad()  # Clear gradients.
          out = model(data.x, data.edge_index)  # Perform a single forward pass.
          loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          optimizer.step()  # Update parameters based on gradients.
          return loss

    def test():
          model.eval()
          out = model(data.x, data.edge_index)
          pred = out.argmax(dim=1)  # Use the class with highest probability.
          test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
          test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
          return test_acc


    for epoch in tqdm.tqdm(range(1, 101)):
        loss = train()
        wandb.log({"gcn/loss": loss})


    model.eval()

    out = model(data.x, data.edge_index)
    test_acc = test()
    wandb.summary["gcn/accuracy"] = test_acc
    wandb.log({"gcn/accuracy": test_acc})
    embedding_to_wandb(out, color=data.y, key="gcn/embedding/trained")
    wandb.finish()

sweep_config = {
    "name": "gcn-sweep-1",
    "method": "bayes",
    "metric": {
        "name": "gcn/accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "hidden_channels": {
            "values": [8, 16, 32]
        },
        "weight_decay": {
            "distribution": "normal",
            "mu": 5e-4,
            "sigma": 1e-5,
        },
        "lr": {
            "min": 0.001,
            "max": 0.5
        }
    }
}

# Register the Sweep with W&B
sweep_id = wandb.sweep(sweep_config, project="node-classification")

wandb.agent(sweep_id, project="node-classification", function=agent_fn, count=50)