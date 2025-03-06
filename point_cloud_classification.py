import os

import numpy as np
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_mesh(id, data, pos, face):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=face.t(), antialiased=False)
    # plt.show()
    id_str = str(id).zfill(2)
    plt.savefig(os.path.join('plot/point_cloud_classification/GeometricShapes/', f'{id_str}_{data}_mesh.png'))

def visualize_mesh_rotate(id, data, pos, face):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=face.t(), antialiased=False)
    # plt.show()
    id_str = str(id).zfill(2)
    plt.savefig(os.path.join('plot/point_cloud_classification/GeometricShapesRotate/', f'{id_str}_{data}_mesh_rotate.png'))

def visualize_points(id, data, pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    # plt.axis('off')
    # plt.show()
    id_str = str(id).zfill(2)
    plt.savefig(os.path.join('plot/point_cloud_classification/GeometricShapesPoints/', f'{id_str}_{data}_point.png'))

def visualize_points_knn(id, data, pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    # plt.axis('off')
    # plt.show()
    id_str = str(id).zfill(2)
    plt.savefig(os.path.join('plot/point_cloud_classification/GeometricShapesPointsKNN/', f'{id_str}_{data.edge_index.shape}_point.png'))

# =============================================================================
# 加载数据集
from torch_geometric.datasets import GeometricShapes

dataset = GeometricShapes(root='data/GeometricShapes')
print(dataset)

# for i in range(len(dataset)):
#     data = dataset[i]
#     print(data)
#     visualize_mesh(i, data, data.pos, data.face)

# =============================================================================
# 点云采样
import torch
from torch_geometric.transforms import SamplePoints
torch.manual_seed(42)

dataset.transform = SamplePoints(num=256)

# for i in range(len(dataset)):
#     data = dataset[i]
#     print(data)
#     visualize_points(i, data, data.pos, data.face)

# =============================================================================
# 动态图生成 - knn
from torch_cluster import knn_graph

# for i in range(len(dataset)):
#     data = dataset[i]
#     data.edge_index = knn_graph(data.pos, k=6)
#     print(data.edge_index.shape)
#     visualize_points_knn(i, data, data.pos, edge_index=data.edge_index)


# =============================================================================
# 领域聚合
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, PPFConv


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.

# =============================================================================
# 网络定义
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, dataset.num_classes)

    def forward(self, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # 5. Classifier.
        return self.classifier(h)


model = PointNet()
print(model)

# =============================================================================
# 训练模型
from IPython.core.display_functions import display
from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

from torch_geometric.loader import DataLoader

train_dataset = GeometricShapes(root='data/GeometricShapes', train=True, transform=SamplePoints(128))
test_dataset = GeometricShapes(root='data/GeometricShapes', train=False, transform=SamplePoints(128))

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

model = PointNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos, data.batch)  # Forward pass.
        loss = criterion(logits, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(loader.dataset)

for epoch in range(1, 51):
    loss = train(model, optimizer, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')


# =============================================================================
# 可视化GNN嵌入
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    pred_labels = []
    with torch.no_grad():
        for data in loader:
            logits = model(data.pos, data.batch)
            pred = logits.argmax(dim=-1).cpu().numpy()
            embeddings.append(logits.cpu().numpy())
            labels.append(data.y.cpu().numpy())
            pred_labels.append(pred)
    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(pred_labels)

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
plt.savefig(os.path.join('plot/point_cloud_classification/result/', 'point_cloud_classification.png'))

# =============================================================================
# 模型旋转泛化验证
from torch_geometric.transforms import Compose, RandomRotate

torch.manual_seed(123)

random_rotate = Compose([
    RandomRotate(degrees=180, axis=0),
    RandomRotate(degrees=180, axis=1),
    RandomRotate(degrees=180, axis=2),
])

dataset = GeometricShapes(root='data/GeometricShapes', transform=random_rotate)

for i in range(len(dataset)):
    data = dataset[i]
    print(data)
    visualize_mesh_rotate(i, data, data.pos, data.face)

torch.manual_seed(42)

transform = Compose([
    random_rotate,
    SamplePoints(num=128),
])

test_dataset = GeometricShapes(root='data/GeometricShapes', train=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=10)

test_acc = test(model, test_loader)
print(f'Test Accuracy: {test_acc:.4f}')

# =============================================================================
# 旋转不变的PointNet层
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing

class PPFNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(12345)
        mlp1 = Sequential(Linear(4, 32),
                              ReLU(),
                              Linear(32, 32))
        self.conv1 = PPFConv(mlp1)
        mlp2 = Sequential(Linear(36, 32),
                              ReLU(),
                              Linear(32, 32))
        self.conv2 = PPFConv(mlp2)
        self.classifier = Linear(32, dataset.num_classes)

    def forward(self, pos, normal, batch):
        edge_index = knn_graph(pos, k=16, batch=batch, loop=False)

        x = self.conv1(x=None, pos=pos, normal=normal, edge_index=edge_index)
        x = x.relu()
        x = self.conv2(x=x, pos=pos, normal=normal, edge_index=edge_index)
        x = x.relu()

        x = global_max_pool(x, batch)  # [num_examples, hidden_channels]
        return self.classifier(x)

model = PPFNet()
print(model)

from IPython.display import Javascript  # Restrict height of output cell.
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

test_transform = Compose([
    random_rotate,
    SamplePoints(num=128, include_normals=True),
])

train_dataset = GeometricShapes(root='data/GeometricShapes', train=False,
                               transform=SamplePoints(128, include_normals=True))
test_dataset = GeometricShapes(root='data/GeometricShapes', train=False,
                               transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

model = PPFNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos, data.normal, data.batch)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        logits = model(data.pos, data.normal, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(loader.dataset)

for epoch in range(1, 101):
    loss = train(model, optimizer, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')

def get_embeddings_PPF(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    pred_labels = []
    with torch.no_grad():
        for data in loader:
            logits = model(data.pos, data.normal, data.batch)
            pred = logits.argmax(dim=-1).cpu().numpy()
            embeddings.append(logits.cpu().numpy())
            labels.append(data.y.cpu().numpy())
            pred_labels.append(pred)
    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(pred_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_embeddings, train_labels, train_pred_labels = get_embeddings_PPF(model, train_loader, device)
test_embeddings, test_labels, test_pred_labels = get_embeddings_PPF(model, test_loader, device)

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
plt.savefig(os.path.join('plot/point_cloud_classification/result_PPF/', 'point_cloud_classification_PPF.png'))

# =============================================================================
# fps池化/下采样
from torch_cluster import fps

dataset = GeometricShapes(root='data/GeometricShapes', transform=SamplePoints(128))

data = dataset[0]
index = fps(data.pos, ratio=0.25)

visualize_points(0, data, data.pos)
visualize_points(0, data, data.pos, index=index)