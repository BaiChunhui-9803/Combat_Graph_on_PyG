import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('==================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===============================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.3f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# =============================================================================
# 制作小批量
from torch_geometric.loader import ClusterData, ClusterLoader

torch.manual_seed(12345)
cluster_data = ClusterData(data, num_parts=128)  # 1. Create subgraphs.
train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)  # 2. Stochastic partioning scheme.

print()
total_num_nodes = 0
for step, sub_data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
    print(sub_data)
    print()
    total_num_nodes += sub_data.num_nodes

print(f'Iterated over {total_num_nodes} of {data.num_nodes} nodes!')

# =============================================================================
# 构建GCN
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
print(model)

# =============================================================================
# 批处理训练GCN
from IPython.core.display_functions import display
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for sub_data in train_loader:  # Iterate over each mini-batch.
        out = model(sub_data.x, sub_data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[sub_data.train_mask],
                         sub_data.y[sub_data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        accs.append(int(correct.sum()) / int(mask.sum()))  # Derive ratio of correct predictions.
    return accs

for epoch in range(1, 51):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# =============================================================================
# 可视化GNN嵌入
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h):
    embeddings = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    pred_labels = h.argmax(dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()

    plt.figure(figsize=(18, 12))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        mask = data.train_mask if i == 0 or i == 3 else data.val_mask if i == 1 or i == 4 else data.test_mask
        color = pred_labels if i < 3 else labels
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], c=color[mask], cmap="Set2")
        title_1 = 'Train' if i == 0 or i == 3 else 'Validation' if i == 1 or i == 4 else 'Test'
        title_2 = 'Predicted' if i < 3 else 'True'
        plt.title(f'{title_1} ({title_2})')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join('plot/scaling_GNNs/', 'scaling_node_classification.png'))

model.eval()
out = model(data.x, data.edge_index)
visualize(out)
