import os

import torch
from torch_geometric.datasets import TUDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    pred_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
            pred = out.argmax(dim=-1).cpu().numpy()
            embeddings.append(out.cpu().numpy())
            labels.append(data.y.cpu().numpy())
            pred_labels.append(pred)
    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(pred_labels)

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# =============================================================================
# 数据集随机排序
torch.manual_seed(1234)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

# =============================================================================
# 图批处理
# batching multiple graphs into a single giant graph
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
        torch.manual_seed(1234)
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
# 训练GCN
from IPython.core.display_functions import display
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# =============================================================================
# 可视化GNN嵌入

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_embeddings, train_labels, train_pred_labels = get_embeddings(model, train_loader, device)
test_embeddings, test_labels, test_pred_labels = get_embeddings(model, test_loader, device)

# 执行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
train_embeddings_2d = tsne.fit_transform(train_embeddings)
test_embeddings_2d = tsne.fit_transform(test_embeddings)

plt.figure(figsize=(10, 10))

# 绘制训练集的嵌入结果 - 预测标签
plt.subplot(2, 2, 1)
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_pred_labels)
plt.title('Training Graph Embeddings (Predicted)')

# 绘制测试集的嵌入结果 - 预测标签
plt.subplot(2, 2, 2)
plt.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=test_pred_labels)
plt.title('Test Graph Embeddings (Predicted)')

# 绘制训练集的嵌入结果 - 真实标签
plt.subplot(2, 2, 3)
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_labels)
plt.title('Training Graph Embeddings (True Labels)')

# 绘制测试集的嵌入结果 - 真实标签
plt.subplot(2, 2, 4)
plt.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=test_labels)
plt.title('Test Graph Embeddings (True Labels)')

# 添加颜色条
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.colorbar()

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('plot/graph_classification/', 'GNN_graph_classification.png'))

# =============================================================================
# 改进GNN，将GCNConv替换为GraphConv
from torch_geometric.nn import GraphConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

model = GNN(hidden_channels=64)
print(model)

# =============================================================================
# 训练
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GNN(hidden_channels=64)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 201):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# =============================================================================
# 可视化GNN嵌入

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_embeddings, train_labels, train_pred_labels = get_embeddings(model, train_loader, device)
test_embeddings, test_labels, test_pred_labels = get_embeddings(model, test_loader, device)

# 执行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
train_embeddings_2d = tsne.fit_transform(train_embeddings)
test_embeddings_2d = tsne.fit_transform(test_embeddings)

plt.figure(figsize=(10, 10))

# 绘制训练集的嵌入结果 - 预测标签
plt.subplot(2, 2, 1)
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_pred_labels)
plt.title('Training Graph Embeddings (Predicted)')

# 绘制测试集的嵌入结果 - 预测标签
plt.subplot(2, 2, 2)
plt.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=test_pred_labels)
plt.title('Test Graph Embeddings (Predicted)')

# 绘制训练集的嵌入结果 - 真实标签
plt.subplot(2, 2, 3)
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_labels)
plt.title('Training Graph Embeddings (True Labels)')

# 绘制测试集的嵌入结果 - 真实标签
plt.subplot(2, 2, 4)
plt.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=test_labels)
plt.title('Test Graph Embeddings (True Labels)')

# 添加颜色条
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.colorbar()

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join('plot/graph_classification/', 'GraphConv_graph_classification.png'))